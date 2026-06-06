from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from load_rules import TrialRule
from rule_classifier import classify_rule


@dataclass(frozen=True)
class RuleSignature:
    rule_type: str
    missing_policy: str
    evidence_category: str
    domains: tuple[str, ...]
    complexity: str
    time_anchor: str
    has_numeric_condition: bool
    has_exception: bool

    @property
    def cluster_key(self) -> tuple[Any, ...]:
        risk_bucket = "complex" if self.complexity == "complex" else "normal"
        protected_anchor = self.time_anchor if risk_bucket == "complex" else "shared"
        return (
            self.rule_type,
            self.missing_policy,
            self.evidence_category,
            risk_bucket,
            protected_anchor,
        )


@dataclass(frozen=True)
class JudgmentUnit:
    canonical_rule_id: str
    representative: TrialRule
    members: tuple[TrialRule, ...]
    signature: RuleSignature
    merge_type: str


DOMAIN_PATTERNS = {
    "demographic": r"年龄|周岁|岁|性别|男性|女性",
    "reproductive": r"妊娠|怀孕|哺乳|生育|绝经|避孕|精子|卵子",
    "diagnosis_pathology": r"确诊|组织学|病理|细胞学|癌|肿瘤|分期|TNM|转移|复发|不可切除",
    "biomarker": r"基因|突变|表达|阳性|阴性|HER2|PD-L1|BRCA|EGFR|ALK|ROS1|KRAS|BRAF|免疫组化",
    "imaging_response": r"影像|病灶|RECIST|可测量|进展|缓解|CT|MRI|扫描|肿瘤负荷",
    "performance": r"ECOG|KPS|体能|活动状态|预期生存|预计生存|预期寿命",
    "hematology_lab": r"血红蛋白|中性粒|白细胞|血小板|骨髓|ANC",
    "liver_lab": r"胆红素|ALT|AST|转氨酶|白蛋白|肝功能",
    "renal_lab": r"肌酐|清除率|肾功能|尿蛋白|eGFR|CrCl",
    "other_lab": r"实验室|凝血|INR|APTT|QT|LVEF|射血分数|血清|电解质",
    "treatment": r"治疗|化疗|放疗|手术|免疫|靶向|用药|药物|既往接受|疗程|剂量|洗脱",
    "history_safety": r"病史|感染|乙肝|丙肝|HIV|心脏|心血管|肝病|肾病|脑|免疫缺陷|自身免疫|高血压|糖尿病",
    "allergy_intolerance": r"过敏|不耐受|禁忌",
    "concomitant_vaccine": r"合并用药|同时使用|疫苗|接种",
    "consent_compliance": r"知情同意|依从|遵守|签署|受试者愿意|研究者判断",
}

TIME_ANCHORS = (
    ("informed_consent", r"知情同意|签署.*同意"),
    ("screening", r"筛选|筛查"),
    ("randomization", r"随机"),
    ("first_dose", r"首次给药|研究给药|开始给药|首剂"),
    ("last_treatment", r"末次治疗|最后一次治疗|既往治疗后"),
    ("baseline", r"基线"),
    ("study_period", r"研究期间|试验期间"),
)

EXCEPTION_PATTERN = re.compile(r"除外|除非|但是|但不包括|例外|不适用于")
LOGIC_PATTERN = re.compile(r"且|并且|同时|以及|或|任一|至少.*项|全部")
NUMERIC_PATTERN = re.compile(r"(?:[<>≤≥]=?|不低于|不高于|至少|至多|不少于|不超过|超过|低于)\s*\d|\d+\s*(?:周|天|月|年|岁|分|%|×)")
CONTEXT_DEPENDENT_PATTERN = re.compile(
    r"本研究|本试验|研究药物|试验药物|研究治疗|试验治疗|研究方案|试验方案|"
    r"研究者|首次给药|首剂|研究给药|随机|筛选|筛查|基线|入组前|给药前"
)


def build_rule_signature(rule: TrialRule) -> RuleSignature:
    text = rule.rule_text
    domains = tuple(name for name, pattern in DOMAIN_PATTERNS.items() if re.search(pattern, text, re.I))
    if not domains:
        domains = ("general",)
    has_exception = bool(EXCEPTION_PATTERN.search(text))
    logic_count = len(LOGIC_PATTERN.findall(text))
    if len(text) > 180 or has_exception or logic_count >= 3 or len(domains) >= 4:
        complexity = "complex"
    elif len(text) > 80 or logic_count >= 1 or len(domains) >= 2:
        complexity = "medium"
    else:
        complexity = "simple"
    time_anchor = "none"
    matched_anchors = [name for name, pattern in TIME_ANCHORS if re.search(pattern, text, re.I)]
    if len(matched_anchors) == 1:
        time_anchor = matched_anchors[0]
    elif len(matched_anchors) > 1:
        time_anchor = "+".join(matched_anchors)
    profile = classify_rule(rule)
    return RuleSignature(
        rule_type=rule.rule_type,
        missing_policy=profile.missing_policy,
        evidence_category=profile.category,
        domains=domains,
        complexity=complexity,
        time_anchor=time_anchor,
        has_numeric_condition=bool(NUMERIC_PATTERN.search(text)),
        has_exception=has_exception,
    )


def build_judgment_units(
    rules: list[TrialRule],
    merge_exact_duplicates: bool = True,
    reviewed_groups_path: Path | None = None,
    use_reviewed_equivalent_groups: bool = True,
) -> list[JudgmentUnit]:
    reviewed_groups = (
        _load_reviewed_groups(reviewed_groups_path)
        if use_reviewed_equivalent_groups
        else []
    )
    by_key = {(rule.trial_id, rule.standard_no): rule for rule in rules}
    consumed: set[tuple[str, str]] = set()
    units: list[JudgmentUnit] = []

    for group in reviewed_groups:
        members = tuple(
            by_key[key]
            for key in group["members"]
            if key in by_key and key not in consumed
        )
        if len(members) < 2:
            continue
        _validate_reviewed_group(members)
        consumed.update((rule.trial_id, rule.standard_no) for rule in members)
        units.append(
            JudgmentUnit(
                canonical_rule_id=group["canonical_rule_id"],
                representative=members[0],
                members=members,
                signature=build_rule_signature(members[0]),
                merge_type="reviewed_equivalent",
            )
        )

    exact_groups: dict[tuple[str, str], list[TrialRule]] = {}
    for rule in rules:
        key = (rule.trial_id, rule.standard_no)
        if key in consumed:
            continue
        exact_groups.setdefault((rule.rule_type, rule.rule_text), []).append(rule)

    for group_rules in exact_groups.values():
        members = tuple(group_rules)
        merge_allowed = merge_exact_duplicates and not CONTEXT_DEPENDENT_PATTERN.search(members[0].rule_text)
        if not merge_allowed and len(members) > 1:
            for rule in members:
                units.append(_single_unit(rule))
            continue
        if len(members) == 1:
            units.append(_single_unit(members[0]))
            continue
        canonical_id = _canonical_id("exact", members[0].rule_type, members[0].rule_text)
        units.append(
            JudgmentUnit(
                canonical_rule_id=canonical_id,
                representative=members[0],
                members=members,
                signature=build_rule_signature(members[0]),
                merge_type="exact_duplicate",
            )
        )
    return units


def build_optimized_batches(
    units: list[JudgmentUnit],
    max_rules: int,
    max_rule_chars: int,
    max_complex_rules: int = 3,
    max_simple_rules: int | None = None,
    max_normal_rules: int | None = None,
    complexity_budget: int = 12,
) -> list[list[JudgmentUnit]]:
    max_rules = max(1, int(max_rules or 1))
    max_rule_chars = max(200, int(max_rule_chars or 4000))
    max_complex_rules = max(1, int(max_complex_rules or 3))
    max_simple_rules = max(1, min(max_rules, int(max_simple_rules or max_rules)))
    max_normal_rules = max(1, min(max_rules, int(max_normal_rules or max_rules)))
    complexity_budget = max(1, int(complexity_budget or 12))
    groups: dict[tuple[Any, ...], list[JudgmentUnit]] = {}
    for unit in units:
        groups.setdefault(unit.signature.cluster_key, []).append(unit)

    batches: list[list[JudgmentUnit]] = []
    for group in groups.values():
        group_max_rules = max_complex_rules if group[0].signature.complexity == "complex" else max_rules
        batches.extend(
            _similarity_aware_pack(
                group,
                max_rules=group_max_rules,
                max_simple_rules=max_simple_rules,
                max_normal_rules=max_normal_rules,
                max_rule_chars=max_rule_chars,
                complexity_budget=complexity_budget,
            )
        )
    return batches


def effective_batch_rule_limit(
    configured_max_rules: int,
    model_max_tokens: int,
    estimated_output_tokens_per_rule: int = 110,
    output_token_reserve: int = 128,
) -> int:
    configured_max_rules = max(1, int(configured_max_rules or 1))
    model_max_tokens = max(1, int(model_max_tokens or 1))
    estimated_output_tokens_per_rule = max(1, int(estimated_output_tokens_per_rule or 110))
    output_token_reserve = max(0, int(output_token_reserve or 0))
    usable_tokens = max(estimated_output_tokens_per_rule, model_max_tokens - output_token_reserve)
    safe_limit = max(1, usable_tokens // estimated_output_tokens_per_rule)
    return min(configured_max_rules, safe_limit)


def optimization_summary(units: list[JudgmentUnit], batches: list[list[JudgmentUnit]], original_rule_count: int) -> dict[str, Any]:
    merged_units = [unit for unit in units if len(unit.members) > 1]
    return {
        "original_rule_count": original_rule_count,
        "judgment_unit_count": len(units),
        "rules_saved_by_equivalence_merge": original_rule_count - len(units),
        "merged_group_count": len(merged_units),
        "exact_duplicate_group_count": sum(unit.merge_type == "exact_duplicate" for unit in merged_units),
        "reviewed_equivalent_group_count": sum(unit.merge_type == "reviewed_equivalent" for unit in merged_units),
        "optimized_batch_count": len(batches),
        "largest_batch_size": max((len(batch) for batch in batches), default=0),
        "average_batch_size": round(sum(map(len, batches)) / len(batches), 3) if batches else 0,
        "average_batch_similarity": round(_average_batch_similarity(batches), 6),
        "average_batch_complexity_cost": round(
            sum(sum(rule_complexity_cost(unit) for unit in batch) for batch in batches) / len(batches),
            3,
        ) if batches else 0,
        "largest_batch_complexity_cost": max(
            (sum(rule_complexity_cost(unit) for unit in batch) for batch in batches),
            default=0,
        ),
        "batch_size_distribution": {
            str(size): sum(len(batch) == size for batch in batches)
            for size in sorted({len(batch) for batch in batches})
        },
    }


def validate_optimization_plan(
    rules: list[TrialRule],
    units: list[JudgmentUnit],
    batches: list[list[JudgmentUnit]],
) -> None:
    original_keys = [(rule.trial_id, rule.standard_no) for rule in rules]
    if len(original_keys) != len(set(original_keys)):
        raise ValueError("Duplicate trial_id/standard_no keys exist in the source rules")

    member_keys = [
        (member.trial_id, member.standard_no)
        for unit in units
        for member in unit.members
    ]
    if len(member_keys) != len(set(member_keys)):
        raise ValueError("An original rule appears in more than one judgment unit")
    if set(member_keys) != set(original_keys):
        missing = sorted(set(original_keys) - set(member_keys))
        extra = sorted(set(member_keys) - set(original_keys))
        raise ValueError(f"Judgment units do not cover source rules: missing={missing[:5]} extra={extra[:5]}")

    unit_ids = [unit.canonical_rule_id for unit in units]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("Duplicate canonical_rule_id values exist")
    batched_ids = [unit.canonical_rule_id for batch in batches for unit in batch]
    if len(batched_ids) != len(set(batched_ids)):
        raise ValueError("A judgment unit appears in more than one batch")
    if set(batched_ids) != set(unit_ids):
        missing = sorted(set(unit_ids) - set(batched_ids))
        extra = sorted(set(batched_ids) - set(unit_ids))
        raise ValueError(f"Batches do not cover judgment units: missing={missing[:5]} extra={extra[:5]}")

    for batch in batches:
        if not batch:
            raise ValueError("The optimization plan contains an empty batch")
        if len({unit.signature.cluster_key for unit in batch}) != 1:
            raise ValueError("A batch crosses a protected clustering boundary")


def _single_unit(rule: TrialRule) -> JudgmentUnit:
    return JudgmentUnit(
        canonical_rule_id=_canonical_id("single", rule.trial_id, rule.standard_no),
        representative=rule,
        members=(rule,),
        signature=build_rule_signature(rule),
        merge_type="none",
    )


def _canonical_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _load_reviewed_groups(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    raw_groups = payload.get("groups", []) if isinstance(payload, dict) else []
    groups = []
    for raw in raw_groups:
        if not isinstance(raw, dict) or raw.get("review_status") != "approved":
            continue
        members = []
        for item in raw.get("members", []):
            if not isinstance(item, dict):
                continue
            trial_id = str(item.get("trial_id") or "").strip()
            standard_no = str(item.get("standard_no") or "").strip()
            if trial_id and standard_no:
                members.append((trial_id, standard_no))
        if len(members) >= 2:
            groups.append(
                {
                    "canonical_rule_id": str(raw.get("canonical_rule_id") or _canonical_id("reviewed", *sum(members, ()))),
                    "members": members,
                }
            )
    return groups


def _validate_reviewed_group(members: tuple[TrialRule, ...]) -> None:
    rule_types = {rule.rule_type for rule in members}
    if len(rule_types) != 1:
        raise ValueError("人工审核等价组中混入了不同规则标识")
    signatures = {build_rule_signature(rule) for rule in members}
    if len(signatures) != 1:
        raise ValueError("人工审核等价组的证据域、缺失策略或逻辑风险画像不一致")


def unit_similarity(left: JudgmentUnit, right: JudgmentUnit) -> float:
    left_domains = set(left.signature.domains)
    right_domains = set(right.signature.domains)
    domain_union = left_domains | right_domains
    domain_score = len(left_domains & right_domains) / len(domain_union) if domain_union else 1.0

    left_features = _text_features(left.representative.rule_text)
    right_features = _text_features(right.representative.rule_text)
    feature_union = left_features | right_features
    text_score = len(left_features & right_features) / len(feature_union) if feature_union else 1.0

    anchor_score = 1.0 if left.signature.time_anchor == right.signature.time_anchor else 0.0
    numeric_match = left.signature.has_numeric_condition == right.signature.has_numeric_condition
    numeric_score = 1.0 if numeric_match else -0.5
    exception_score = 1.0 if left.signature.has_exception == right.signature.has_exception else 0.0
    return (
        0.45 * domain_score
        + 0.35 * text_score
        + 0.08 * anchor_score
        + 0.08 * numeric_score
        + 0.04 * exception_score
    )


def _similarity_aware_pack(
    units: list[JudgmentUnit],
    max_rules: int,
    max_simple_rules: int,
    max_normal_rules: int,
    max_rule_chars: int,
    complexity_budget: int,
) -> list[list[JudgmentUnit]]:
    remaining = list(units)
    packed: list[list[JudgmentUnit]] = []
    while remaining:
        seed = max(
            remaining,
            key=lambda unit: (
                len(unit.signature.domains),
                len(unit.representative.rule_text),
                unit.canonical_rule_id,
            ),
        )
        remaining.remove(seed)
        batch = [seed]
        chars = len(seed.representative.rule_text)
        cost = rule_complexity_cost(seed)
        while remaining:
            batch_rule_limit = _batch_rule_limit(
                batch,
                max_rules=max_rules,
                max_simple_rules=max_simple_rules,
                max_normal_rules=max_normal_rules,
            )
            if len(batch) >= batch_rule_limit:
                break
            candidates = [
                unit
                for unit in remaining
                if chars + len(unit.representative.rule_text) <= max_rule_chars
                and cost + rule_complexity_cost(unit) <= complexity_budget
            ]
            if not candidates:
                break
            next_unit = max(
                candidates,
                key=lambda unit: (
                    sum(unit_similarity(unit, member) for member in batch) / len(batch),
                    -len(unit.representative.rule_text),
                    unit.canonical_rule_id,
                ),
            )
            remaining.remove(next_unit)
            batch.append(next_unit)
            chars += len(next_unit.representative.rule_text)
            cost += rule_complexity_cost(next_unit)
        packed.append(batch)
    return packed


def rule_complexity_cost(unit: JudgmentUnit) -> int:
    signature = unit.signature
    text = unit.representative.rule_text
    cost = 1
    if signature.has_numeric_condition:
        cost += 1
    if signature.time_anchor != "none":
        cost += 1
    cost += max(0, len(signature.domains) - 1)
    if signature.complexity == "medium":
        cost += 1
    elif signature.complexity == "complex":
        cost += 3
    if signature.has_exception:
        cost += 2
    if "研究者判断" in text or "临床意义" in text:
        cost += 2
    if len(text) > 300:
        cost += 4
    elif len(text) > 150:
        cost += 2
    return cost


def _batch_rule_limit(
    batch: list[JudgmentUnit],
    max_rules: int,
    max_simple_rules: int,
    max_normal_rules: int,
) -> int:
    if any(unit.signature.complexity == "complex" for unit in batch):
        return max_rules
    if any(unit.signature.complexity == "medium" for unit in batch):
        return min(max_rules, max_normal_rules)
    return min(max_rules, max_simple_rules)


@lru_cache(maxsize=4096)
def _text_features(text: str) -> frozenset[str]:
    normalized = re.sub(r"\s+", "", text.lower())
    normalized = re.sub(r"[，。；、：:“”‘’（）()【】\[\],.!！?？;:]", "", normalized)
    chinese_or_alnum = re.findall(r"[\u4e00-\u9fff]|[a-z]+|\d+(?:\.\d+)?", normalized)
    features = set(chinese_or_alnum)
    features.update(
        chinese_or_alnum[idx] + chinese_or_alnum[idx + 1]
        for idx in range(len(chinese_or_alnum) - 1)
    )
    return frozenset(features)


def _average_batch_similarity(batches: list[list[JudgmentUnit]]) -> float:
    pair_scores = []
    for batch in batches:
        for left_idx in range(len(batch)):
            for right_idx in range(left_idx + 1, len(batch)):
                pair_scores.append(unit_similarity(batch[left_idx], batch[right_idx]))
    return sum(pair_scores) / len(pair_scores) if pair_scores else 1.0

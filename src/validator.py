from __future__ import annotations

from typing import Any

from load_rules import TrialRule
from rule_classifier import RuleProfile

ALLOWED_LABELS = {"符合", "不符合", "未知"}
MISSING_KEYWORDS = (
    "未提供",
    "未提及",
    "未记录",
    "未检索",
    "未发现",
    "无相关",
    "无法确定",
    "无法判断",
    "证据不足",
    "没有明确",
    "无明确",
    "查不到",
)


def normalize_llm_result(data: dict[str, Any], rule: TrialRule) -> dict[str, Any]:
    items = data.get("items")
    item = items[0] if isinstance(items, list) and items else data
    return normalize_llm_item(item, rule)


def normalize_llm_item(item: dict[str, Any], rule: TrialRule) -> dict[str, Any]:
    label = str(item.get("label") or "").strip()
    if label not in ALLOWED_LABELS:
        label = "未知"
    return {
        "trial_register_id": item.get("trial_register_id") or rule.trial_register_id,
        "trial_id": item.get("trial_id") or rule.trial_id,
        "standard_no": item.get("standard_no") or rule.standard_no,
        "rule_type": item.get("rule_type") or rule.rule_type,
        "label": label,
        "explanation": str(item.get("explanation") or "").strip(),
        "evidence": str(item.get("evidence") or "").strip(),
        "confidence": float(item.get("confidence") or 0),
        "source": "llm",
    }


def normalize_llm_batch_result(data: dict[str, Any], rules: list[TrialRule]) -> list[dict[str, Any]]:
    raw_items = data.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("批量模型输出缺少 items 数组")

    item_map: dict[tuple[str, str], dict[str, Any]] = {}
    fallback_items: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        standard_no = str(item.get("standard_no") or "").strip()
        trial_id = str(item.get("trial_id") or "").strip()
        if standard_no or trial_id:
            item_map[(trial_id, standard_no)] = item
        fallback_items.append(item)

    normalized = []
    used_positions: set[int] = set()
    for index, rule in enumerate(rules):
        item = item_map.get((rule.trial_id, rule.standard_no))
        if item is None:
            item = item_map.get(("", rule.standard_no))
        if item is None:
            for pos, candidate in enumerate(fallback_items):
                if pos not in used_positions:
                    item = candidate
                    used_positions.add(pos)
                    break
        if item is None:
            raise ValueError(f"批量模型输出缺少规则结果：{rule.trial_id}/{rule.standard_no}")
        normalized.append(normalize_llm_item(item, rule))
    return normalized


def apply_missing_policy(result: dict[str, Any], rule: TrialRule, profile: RuleProfile) -> dict[str, Any]:
    text = f"{result.get('explanation', '')} {result.get('evidence', '')}"
    if not any(keyword in text for keyword in MISSING_KEYWORDS):
        result["policy_adjusted"] = result.get("policy_adjusted", "no")
        return result

    original_label = result["label"]
    new_label = original_label

    if "入选" in rule.rule_type:
        new_label = "未知"
    elif profile.category == "lab":
        new_label = "未知"
    elif profile.category in {"history", "treatment"}:
        new_label = "符合"

    if new_label != original_label:
        result = dict(result)
        result["label"] = new_label
        result["explanation"] = result.get("explanation", "") + f"（按缺失信息规则由“{original_label}”修正为“{new_label}”。）"
        result["policy_adjusted"] = "yes"
    else:
        result["policy_adjusted"] = "no"
    return result

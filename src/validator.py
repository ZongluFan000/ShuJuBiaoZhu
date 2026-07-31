from __future__ import annotations

from typing import Any

from load_rules import TrialRule
from rule_classifier import RuleProfile

ALLOWED_LABELS = {"满足", "不满足", "未知"}
LABEL_ALIASES = {
    "Y": "满足",
    "YES": "满足",
    "TRUE": "满足",
    "满足": "满足",
    "N": "不满足",
    "NO": "不满足",
    "FALSE": "不满足",
    "不满足": "不满足",
    "U": "未知",
    "UNK": "未知",
    "UNKNOWN": "未知",
    "未知": "未知",
}
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
    items = data.get("items") or data.get("i")
    item = items[0] if isinstance(items, list) and items else data
    return normalize_llm_item(item, rule)


def normalize_llm_item(item: dict[str, Any], rule: TrialRule) -> dict[str, Any]:
    label = _normalize_label(item.get("label", item.get("l", "")))
    if label not in ALLOWED_LABELS:
        label = "未知"
    return {
        "trial_register_id": item.get("trial_register_id") or rule.trial_register_id,
        "trial_id": item.get("trial_id") or rule.trial_id,
        "standard_no": item.get("standard_no") or rule.standard_no,
        "rule_type": item.get("rule_type") or rule.rule_type,
        "label": label,
        "explanation": str(item.get("explanation", item.get("x", "")) or "").strip(),
        "evidence": str(item.get("evidence", item.get("e", "")) or "").strip(),
        "confidence": _to_float(item.get("confidence", item.get("c", 0))),
        "source": "llm",
    }


def normalize_llm_batch_result(data: dict[str, Any], rules: list[TrialRule]) -> list[dict[str, Any]]:
    raw_items = data.get("items") or data.get("i")
    if not isinstance(raw_items, list):
        raise ValueError("批量模型输出缺少 items/i 数组")

    item_map: dict[tuple[str, str], dict[str, Any]] = {}
    fallback_items: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        standard_no = str(item.get("standard_no") or "").strip()
        trial_id = str(item.get("trial_id") or "").strip()
        short_no = _short_no(item)
        if short_no is not None and 1 <= short_no <= len(rules):
            rule = rules[short_no - 1]
            trial_id = rule.trial_id
            standard_no = rule.standard_no
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


def normalize_llm_batch_result_strict(data: dict[str, Any], rules: list[TrialRule]) -> list[dict[str, Any]]:
    raw_items = data.get("items") or data.get("i")
    if not isinstance(raw_items, list):
        raise ValueError("批量模型输出缺少 items/i 数组")
    if len(raw_items) != len(rules):
        raise ValueError(f"批量模型输出数量不一致：expected={len(rules)} actual={len(raw_items)}")

    item_map: dict[tuple[str, str], dict[str, Any]] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            raise ValueError("批量模型输出包含非对象元素")
        short_no = _short_no(item)
        if short_no is not None:
            if not (1 <= short_no <= len(rules)):
                raise ValueError(f"批量模型输出包含非法短编号：n={short_no}")
            rule = rules[short_no - 1]
            key = (rule.trial_id, rule.standard_no)
        else:
            key = (
                str(item.get("trial_id") or "").strip(),
                str(item.get("standard_no") or "").strip(),
            )
        if not all(key):
            raise ValueError("批量模型输出缺少 n 或 trial_id/standard_no")
        if key in item_map:
            raise ValueError(f"批量模型输出包含重复规则：{key[0]}/{key[1]}")
        label = _normalize_label(item.get("label", item.get("l", "")))
        if label not in ALLOWED_LABELS:
            raise ValueError(f"Invalid batch label: {key[0]}/{key[1]} label={label!r}")
        item_map[key] = item

    expected = {(rule.trial_id, rule.standard_no) for rule in rules}
    actual = set(item_map)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"批量模型规则ID不匹配：missing={missing[:5]} extra={extra[:5]}")
    return [normalize_llm_item(item_map[(rule.trial_id, rule.standard_no)], rule) for rule in rules]


def _normalize_label(value: Any) -> str:
    text = str(value or "").strip()
    return LABEL_ALIASES.get(text.upper(), LABEL_ALIASES.get(text, text))


def _to_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _short_no(item: dict[str, Any]) -> int | None:
    value = item.get("n", item.get("r", None))
    if value is None:
        return None
    text = str(value).strip()
    if text.upper().startswith("R"):
        text = text[1:].strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def apply_missing_policy(result: dict[str, Any], rule: TrialRule, profile: RuleProfile) -> dict[str, Any]:
    text = f"{result.get('explanation', '')} {result.get('evidence', '')}"
    if not any(keyword in text for keyword in MISSING_KEYWORDS):
        result["policy_adjusted"] = result.get("policy_adjusted", "no")
        return result

    original_label = result["label"]
    new_label = original_label

    # V4 scheme: no relevant evidence always means unknown, regardless of
    # inclusion/exclusion type or evidence category.
    if any(keyword in text for keyword in MISSING_KEYWORDS):
        new_label = "未知"

    if new_label != original_label:
        result = dict(result)
        result["label"] = new_label
        result["explanation"] = result.get("explanation", "") + f"（按缺失信息规则由“{original_label}”修正为“{new_label}”。）"
        result["policy_adjusted"] = "yes"
    else:
        result["policy_adjusted"] = "no"
    return result

from __future__ import annotations

import sys
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from load_rules import load_rules  # noqa: E402
from rule_classifier import classify_rule  # noqa: E402


TARGET_COUNTS = {
    "demographic": 6,
    "diagnosis": 10,
    "treatment": 10,
    "lab": 8,
    "exam": 5,
    "history": 8,
    "general": 3,
}


def main() -> int:
    source = ROOT / "data" / "input" / "rules.xlsx"
    dest = ROOT / "data" / "input" / "rules_representative_50.xlsx"
    rules = load_rules(source)
    selected = []
    seen = set()
    for category, count in TARGET_COUNTS.items():
        bucket = [r for r in rules if classify_rule(r).category == category]
        chosen = []
        for rule in bucket:
            key = (rule.trial_id, rule.standard_no, rule.rule_text)
            if key in seen:
                continue
            chosen.append(rule)
            seen.add(key)
            if len(chosen) >= count:
                break
        selected.extend(chosen)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "representative_rules_50"
    ws.append(["试验标识", "试验注册号", "试验方案编号", "山东大学齐鲁医院", "癌症类型", "癌症分类", "试验分期", "试验分期", "伦理", "标准编号", "规则", "规则标识"])
    for rule in selected:
        ws.append([rule.trial_id, rule.trial_register_id, rule.protocol_no, "", rule.cancer_type, rule.cancer_category, "", "", "", rule.standard_no, rule.rule_text, rule.rule_type])
    wb.save(dest)
    print(f"wrote {len(selected)} rules to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

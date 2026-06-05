from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openpyxl


@dataclass(frozen=True)
class TrialRule:
    trial_id: str
    trial_register_id: str
    protocol_no: str
    cancer_type: str
    cancer_category: str
    standard_no: str
    rule_text: str
    rule_type: str


HEADER_ALIASES = {
    "trial_id": ["试验标识"],
    "trial_register_id": ["试验注册号"],
    "protocol_no": ["试验方案编号"],
    "cancer_type": ["癌症类型"],
    "cancer_category": ["癌症分类"],
    "standard_no": ["标准编号"],
    "rule_text": ["规则"],
    "rule_type": ["规则标识"],
}


def _header_map(headers: list[Any]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for idx, header in enumerate(headers):
        if header is None:
            continue
        name = str(header).strip()
        for field, aliases in HEADER_ALIASES.items():
            if name in aliases and field not in mapping:
                mapping[field] = idx
    missing = [field for field in HEADER_ALIASES if field not in mapping]
    if missing:
        raise ValueError(f"Rules file missing columns: {missing}")
    return mapping


def load_rules(path: Path, limit: int | None = None) -> list[TrialRule]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    mapping = _header_map(headers)
    rules: list[TrialRule] = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not any(row):
            continue
        rule_text = str(row[mapping["rule_text"]] or "").strip()
        if not rule_text:
            continue
        rules.append(
            TrialRule(
                trial_id=str(row[mapping["trial_id"]] or "").strip(),
                trial_register_id=str(row[mapping["trial_register_id"]] or "").strip(),
                protocol_no=str(row[mapping["protocol_no"]] or "").strip(),
                cancer_type=str(row[mapping["cancer_type"]] or "").strip(),
                cancer_category=str(row[mapping["cancer_category"]] or "").strip(),
                standard_no=str(row[mapping["standard_no"]] or "").strip(),
                rule_text=rule_text,
                rule_type=str(row[mapping["rule_type"]] or "").strip(),
            )
        )
        if limit and len(rules) >= limit:
            break
    return rules

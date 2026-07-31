from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from load_patient import PatientRecord
from load_rules import TrialRule


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text[:19], fmt).date()
        except ValueError:
            pass
    return None


def _age(birth_date: date, today: date | None = None) -> int:
    today = today or date.today()
    return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))


def try_structured_judge(patient: PatientRecord, rule: TrialRule, reference_date: date | None = None) -> dict[str, Any] | None:
    text = rule.rule_text
    info = (patient.sheets.get("raw_patient_info") or [{}])[0]

    gender = str(info.get("gender_name") or "")
    if text.strip() in {"男性", "女性"}:
        wanted = text.strip()
        if gender:
            ok = wanted in gender
            return _result(rule, "满足" if ok else "不满足", f"患者性别为{gender}。", "raw_patient_info.gender_name", 0.95)
        return _result(rule, "未知", "未检索到患者性别。", "", 0.8)

    age_match = re.search(r"年龄.*?≥\s*(\d+)|年龄.*?>=\s*(\d+)|(\d+)\s*岁以上|成人", text)
    if age_match:
        threshold = next((int(x) for x in age_match.groups() if x), 18)
        birth = _parse_date(info.get("birth_date"))
        if birth:
            age = _age(birth, reference_date)
            ok = age >= threshold
            return _result(rule, "满足" if ok else "不满足", f"出生日期为{birth}，按时间锚点估算年龄{age}岁，阈值为{threshold}岁。", "raw_patient_info.birth_date", 0.9)
        return _result(rule, "未知", "未检索到出生日期，无法判断年龄。", "", 0.75)

    return None


def _result(rule: TrialRule, label: str, explanation: str, evidence: str, confidence: float) -> dict[str, Any]:
    return {
        "trial_register_id": rule.trial_register_id,
        "trial_id": rule.trial_id,
        "standard_no": rule.standard_no,
        "rule_type": rule.rule_type,
        "label": label,
        "explanation": explanation,
        "evidence": evidence,
        "confidence": confidence,
        "source": "structured",
    }

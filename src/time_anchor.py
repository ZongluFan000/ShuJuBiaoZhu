from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any


# Only clinical-event fields participate.  `create_time` and derived sheets are
# intentionally excluded because they describe data import/export, not care.
EVENT_DATE_FIELDS = {
    "visit_date",
    "admission_date",
    "discharge_date",
    "test_time",
    "sampling_time",
    "exam_date",
    "diagnosis_date",
    "prescribed_time",
}


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    text = str(value or "").strip()
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d",
    ):
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            continue
    return None


def build_time_anchor(sheets: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    candidates: list[tuple[datetime, str, str]] = []
    for sheet, records in sheets.items():
        if sheet == "latest_3_records":
            continue
        for record in records:
            for field in EVENT_DATE_FIELDS:
                value = _parse_datetime(record.get(field))
                if value is not None:
                    candidates.append((value, sheet, field))
    if not candidates:
        return {
            "latest_record_time": "",
            "latest_record_sheet": "",
            "latest_record_field": "",
            "screening_time": "",
            "randomization_enrollment_time": "",
            "treatment_start_time": "",
        }

    latest, sheet, field = max(candidates)
    return {
        "latest_record_time": latest.isoformat(sep=" ", timespec="seconds"),
        "latest_record_sheet": sheet,
        "latest_record_field": field,
        "screening_time": latest.isoformat(sep=" ", timespec="seconds"),
        "randomization_enrollment_time": (latest + timedelta(days=14)).isoformat(sep=" ", timespec="seconds"),
        "treatment_start_time": (latest + timedelta(days=16)).isoformat(sep=" ", timespec="seconds"),
    }


def time_anchor_prompt(anchor: dict[str, str]) -> str:
    latest = anchor.get("latest_record_time") or "未能从原始临床事件字段解析"
    source = ""
    if anchor.get("latest_record_sheet"):
        source = f"（来源：{anchor['latest_record_sheet']}.{anchor['latest_record_field']}）"
    return (
        "本患者动态时间锚点：\n"
        f"- 最新病历记录时间/筛选/招募时间：{latest}{source}\n"
        f"- 随机分组、入组、入选时间：{anchor.get('randomization_enrollment_time') or '未知'}\n"
        f"- 启用研究治疗、研究开始、首次给药、首次用药、C1D1：{anchor.get('treatment_start_time') or '未知'}\n"
        "- 标准未说明参考时间时，按筛选时间判断。若所需时间锚点未知，时间相关标准标为未知。"
    )

from __future__ import annotations

from collections import defaultdict
from typing import Any


def build_trial_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("患者编号", "")), str(row.get("试验注册号", "")), str(row.get("试验标识", "")))
        grouped[key].append(row)

    summaries = []
    for (patient_sn, register_id, trial_id), items in sorted(grouped.items()):
        failed = []
        unknown = []
        for item in items:
            label = str(item.get("标注结果", ""))
            rule_type = str(item.get("规则标识", ""))
            is_inclusion = "入选" in rule_type
            is_exclusion = "排除" in rule_type
            if label == "未知":
                unknown.append(str(item.get("标准编号", "")))
            if (is_inclusion and label == "不满足") or (is_exclusion and label == "满足"):
                failed.append(str(item.get("标准编号", "")))
        summaries.append(
            {
                "患者编号": patient_sn,
                "试验注册号": register_id,
                "试验标识": trial_id,
                "试验结论": "符合试验" if not failed else "不符合试验",
                "是否含未知标准": "是" if unknown else "否",
                "未知标准数": len(unknown),
                "未知标准编号": ";".join(unknown),
                "触发不符合的标准编号": ";".join(failed),
                "单条标准数": len(items),
            }
        )
    return summaries

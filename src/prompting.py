from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from evidence_builder import EvidencePackage
from load_rules import TrialRule


@lru_cache(maxsize=8)
def _read_template(template_path: str) -> str:
    return Path(template_path).read_text(encoding="utf-8")


def build_prompt(template_path: Path, patient_sn: str, rule: TrialRule, evidence: EvidencePackage) -> str:
    template = _read_template(str(template_path))
    return (
        template
        + "\n\n患者编号：\n"
        + patient_sn
        + "\n\n待判断标准：\n"
        + f"试验注册号：{rule.trial_register_id}\n"
        + f"试验标识：{rule.trial_id}\n"
        + f"标准编号：{rule.standard_no}\n"
        + f"规则标识：{rule.rule_type}\n"
        + f"标准内容：{rule.rule_text}\n"
        + "\n患者证据：\n"
        + (evidence.text or "未检索到相关证据")
    )


def build_batch_prompt(template_path: Path, patient_sn: str, items: list[tuple[TrialRule, EvidencePackage]]) -> str:
    template = _read_template(str(template_path))
    shared_evidence = ""
    if items:
        first_evidence = items[0][1].text
        if all(evidence.text == first_evidence for _, evidence in items):
            shared_evidence = first_evidence or "未检索到相关证据"
    blocks = []
    for idx, (rule, evidence) in enumerate(items, start=1):
        lines = [
            f"### 规则 {idx}",
            f"试验注册号：{rule.trial_register_id}",
            f"试验标识：{rule.trial_id}",
            f"标准编号：{rule.standard_no}",
            f"规则标识：{rule.rule_type}",
            f"标准内容：{rule.rule_text}",
        ]
        if not shared_evidence:
            lines.extend(["患者证据：", evidence.text or "未检索到相关证据"])
        blocks.append("\n".join(lines))
    evidence_block = ""
    if shared_evidence:
        evidence_block = "\n\n患者共用证据：\n" + shared_evidence
    return (
        template
        + "\n\n患者编号：\n"
        + patient_sn
        + evidence_block
        + "\n\n下面有多条待判断标准。请分别判断每一条，并在 items 数组中返回同样数量的结果。"
        + "\n每个结果必须保留对应的 trial_register_id、trial_id、standard_no、rule_type。"
        + "\n不要合并、不要省略任何标准。"
        + "\n\n待判断标准列表：\n"
        + "\n\n".join(blocks)
    )

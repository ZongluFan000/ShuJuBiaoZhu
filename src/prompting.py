from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

from evidence_builder import EvidencePackage
from load_rules import TrialRule
from time_anchor import time_anchor_prompt


@lru_cache(maxsize=8)
def _read_template(template_path: str) -> str:
    return Path(template_path).read_text(encoding="utf-8")


def _compact_text(value: str) -> str:
    """Only remove layout noise; do not delete medical conditions."""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _rule_line(idx: int, rule: TrialRule) -> str:
    rule_text = _compact_text(rule.rule_text)
    rule_type = _compact_text(rule.rule_type)
    trial_id = _compact_text(rule.trial_id)
    standard_no = _compact_text(rule.standard_no)
    return f"R{idx}|T={trial_id}|S={standard_no}|K={rule_type}|C={rule_text}"


def _evidence_line(idx: int, evidence: EvidencePackage) -> str:
    text = _compact_text(evidence.text or "未检索到相关证据")
    return f"R{idx}证据：{text}"


def _deduplicated_evidence_block(evidences: list[tuple[int, EvidencePackage]]) -> str:
    evidence_ids: dict[str, int] = {}
    evidence_lines: list[str] = []
    refs: list[str] = []
    for idx, evidence in evidences:
        text = _compact_text(evidence.text or "未检索到相关证据")
        evidence_id = evidence_ids.get(text)
        if evidence_id is None:
            evidence_id = len(evidence_ids) + 1
            evidence_ids[text] = evidence_id
            evidence_lines.append(f"E{evidence_id}：{text}")
        refs.append(f"R{idx}->E{evidence_id}")
    return "证据引用：" + "；".join(refs) + "\n证据正文：\n" + "\n".join(evidence_lines)


def build_prompt(template_path: Path, patient_sn: str, rule: TrialRule, evidence: EvidencePackage, time_anchor: dict[str, str] | None = None) -> str:
    template = _read_template(str(template_path))
    return (
        template
        + "\n\n患者编号：\n"
        + patient_sn
        + "\n\n"
        + time_anchor_prompt(time_anchor or {})
        + "\n\n待判断标准：\n"
        + f"试验注册号：{rule.trial_register_id}\n"
        + f"试验标识：{rule.trial_id}\n"
        + f"标准编号：{rule.standard_no}\n"
        + f"规则标识：{rule.rule_type}\n"
        + f"标准内容：{rule.rule_text}\n"
        + "\n患者证据：\n"
        + (evidence.text or "未检索到相关证据")
    )


def build_batch_prompt(template_path: Path, patient_sn: str, items: list[tuple[TrialRule, EvidencePackage]], time_anchor: dict[str, str] | None = None) -> str:
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


def build_optimized_batch_prompt(
    template_path: Path,
    patient_sn: str,
    items: list[tuple[TrialRule, EvidencePackage]],
    use_compact_template: bool = True,
    time_anchor: dict[str, str] | None = None,
) -> str:
    compact_template_path = template_path.with_name(template_path.stem + "_compact.md")
    if not compact_template_path.exists():
        # Keep non-V4 utility callers working; V4 has its own paired compact
        # template and therefore never reaches this compatibility fallback.
        compact_template_path = template_path.with_name("label_rules_compact.md")
    selected_template_path = (
        compact_template_path
        if use_compact_template and compact_template_path.exists()
        else template_path
    )
    template = _read_template(str(selected_template_path))
    template = template.split("\n输出 JSON 格式：", 1)[0].rstrip()
    shared_evidence = ""
    if items:
        first_evidence = items[0][1].text
        if all(evidence.text == first_evidence for _, evidence in items):
            shared_evidence = first_evidence or "未检索到相关证据"
    rule_lines = []
    evidence_items: list[tuple[int, EvidencePackage]] = []
    for idx, (rule, evidence) in enumerate(items, start=1):
        rule_lines.append(_rule_line(idx, rule))
        if not shared_evidence:
            evidence_items.append((idx, evidence))
    if shared_evidence:
        evidence_block = "\n\n患者共用证据：\n" + _compact_text(shared_evidence)
    else:
        evidence_block = "\n\n患者分规则证据：\n" + _deduplicated_evidence_block(evidence_items)
    rule_block = "\n\n待判断标准列表，R号为本批短编号：\n" + "\n".join(rule_lines)
    return (
        template
        + "\n\n严格独立判断要求："
        + "\n1. 每条规则都是独立判断任务，不得沿用相邻规则的阈值、时间窗、例外条件或结论。"
        + "\n2. 即使规则文本相似，也必须逐条核对标准内容。"
        + "\n3. 返回数量必须与输入规则数量完全一致，不得漏掉任何R号。"
        + "\n4. 只能用R号对齐结果，禁止按位置猜测或省略。"
        + "\n5. 只能返回一个合法JSON对象，不得输出Markdown、JSONL或额外说明。"
        + "\n6. 短JSON格式固定为："
        + '\n{"i":[{"n":1,"l":"Y/N/U","e":"证据<=20字","x":"解释<=40字","c":0.0}]}'
        + "\n7. l取值：Y=满足，N=不满足，U=未知。n必须是输入R号里的数字。"
        + "\n\n患者编号：\n"
        + patient_sn
        + "\n\n"
        + time_anchor_prompt(time_anchor or {})
        + evidence_block
        # Keep the patient evidence before batch-specific rules so API prefix caching
        # can reuse the shared context across multiple batches for one patient.
        + rule_block
    )

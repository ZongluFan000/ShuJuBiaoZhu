from __future__ import annotations

import re
from dataclasses import dataclass

from load_rules import TrialRule


@dataclass(frozen=True)
class RuleProfile:
    category: str
    needs_llm: bool
    missing_policy: str


PATTERNS = {
    "demographic": r"年龄|岁|男性|女性|性别|妊娠|哺乳|避孕",
    "lab": r"血红蛋白|中性粒|白细胞|血小板|胆红素|ALT|AST|肌酐|清除率|实验室|器官|骨髓|尿蛋白|INR|QT|LVEF|血清",
    "diagnosis": r"确诊|组织学|病理|分期|TNM|转移|复发|晚期|不可切除|癌",
    "treatment": r"治疗|用药|化疗|放疗|手术|免疫|靶向|药物|既往",
    "exam": r"影像|病灶|RECIST|可测量|进展|CT|MRI",
    "history": r"病史|疾病史|感染|乙肝|丙肝|HIV|心|肝|肾|肺|脑|免疫缺陷|自身免疫|高血压|糖尿病|过敏|不耐受",
}


def classify_rule(rule: TrialRule) -> RuleProfile:
    text = rule.rule_text
    matched = [name for name, pat in PATTERNS.items() if re.search(pat, text, re.I)]
    if "入选" in rule.rule_type:
        missing_policy = "inclusion_default"
    elif "lab" in matched:
        missing_policy = "exclusion_lab_default"
    elif "history" in matched or "treatment" in matched:
        missing_policy = "exclusion_history_default"
    else:
        missing_policy = "exclusion_general_default"

    if matched == ["demographic"]:
        return RuleProfile(category="demographic", needs_llm=False, missing_policy=missing_policy)
    if "lab" in matched and len(matched) == 1:
        return RuleProfile(category="lab", needs_llm=True, missing_policy=missing_policy)
    category = matched[0] if matched else "general"
    return RuleProfile(category=category, needs_llm=True, missing_policy=missing_policy)

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from load_patient import PatientRecord
from load_rules import TrialRule
from rule_classifier import RuleProfile


@dataclass(frozen=True)
class EvidencePackage:
    patient_sn: str
    text: str
    sources: list[str]


SECTION_FIELDS = {
    "raw_patient_info": ["gender_name", "birth_date", "create_time"],
    "diagnosis_all": ["diagnosis_date", "diagnosis_type_name", "disease_name", "diagnosis_code"],
    "pathology_record": ["exam_date", "pathology_item_chinese_name", "imaging_conclusion", "immunohistochemistry_result", "gross_finding"],
    "examination_record": ["exam_date", "exam_item_chinese_name", "imaging_finding", "imaging_conclusion"],
    "lab_test_item": ["test_time", "item_name_cn", "test_item_chinese_name", "item_value", "item_unit", "qualitative_results", "normal_ref_value_text", "test_item_abnormal_status"],
    "drug_all": ["prescribed_time", "administration_start_date", "administration_end_date", "order_name", "common_name", "class_name", "order_status"],
    "emr_document": ["visit_date", "category_name", "document_name", "body_text"],
    "outpatient_visit_summary": ["visit_date", "visit_dept_name", "visit_type", "register_status"],
}


PROFILE_SHEETS = {
    "demographic": ["raw_patient_info", "diagnosis_all", "emr_document"],
    "lab": ["lab_test_item", "emr_document"],
    "diagnosis": ["diagnosis_all", "pathology_record", "emr_document", "examination_record"],
    "treatment": ["drug_all", "emr_document", "diagnosis_all"],
    "exam": ["examination_record", "pathology_record", "emr_document"],
    "history": ["emr_document", "diagnosis_all", "drug_all", "examination_record"],
    "general": ["raw_patient_info", "diagnosis_all", "pathology_record", "examination_record", "lab_test_item", "drug_all", "emr_document"],
}


def _format_record(sheet: str, record: dict, fields: Iterable[str]) -> str:
    parts = []
    for field in fields:
        value = record.get(field)
        if value is None or str(value).strip() == "":
            continue
        parts.append(f"{field}={str(value).strip()}")
    return f"[{sheet}] " + "；".join(parts)


def build_evidence(
    patient: PatientRecord,
    rule: TrialRule,
    profile: RuleProfile,
    max_section_chars: int = 3500,
    max_total_chars: int = 16000,
) -> EvidencePackage:
    selected_sheets = PROFILE_SHEETS.get(profile.category, PROFILE_SHEETS["general"])
    sections: list[str] = []
    sources: list[str] = []
    for sheet in selected_sheets:
        records = patient.sheets.get(sheet) or []
        fields = SECTION_FIELDS.get(sheet, [])
        lines: list[str] = []
        for record in records:
            line = _format_record(sheet, record, fields)
            if line.strip() != f"[{sheet}]":
                lines.append(line)
            if sum(len(x) for x in lines) >= max_section_chars:
                break
        if lines:
            body = "\n".join(lines)
            sections.append(f"## {sheet}\n{body[:max_section_chars]}")
            sources.append(sheet)
        if sum(len(x) for x in sections) >= max_total_chars:
            break
    text = "\n\n".join(sections)[:max_total_chars]
    return EvidencePackage(patient_sn=patient.patient_sn, text=text, sources=sources)

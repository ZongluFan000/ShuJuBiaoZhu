from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openpyxl


@dataclass
class PatientRecord:
    patient_sn: str
    source_file: str
    sheets: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def _safe_text(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value.replace("\r\n", "\n").strip()
    return value


def load_patient(path: Path) -> PatientRecord:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheets: dict[str, list[dict[str, Any]]] = {}
    patient_sn = path.stem
    for ws in wb.worksheets:
        rows = ws.iter_rows(values_only=True)
        try:
            headers_raw = next(rows)
        except StopIteration:
            sheets[ws.title] = []
            continue
        headers = [str(h).strip() if h is not None else f"col_{i+1}" for i, h in enumerate(headers_raw)]
        records: list[dict[str, Any]] = []
        for row in rows:
            if not any(cell is not None and str(cell).strip() for cell in row):
                continue
            item = {headers[i]: _safe_text(row[i]) if i < len(row) else None for i in range(len(headers))}
            records.append(item)
            if not patient_sn:
                patient_sn = str(item.get("patient_sn") or path.stem)
        sheets[ws.title] = records
    info = sheets.get("raw_patient_info") or []
    if info and info[0].get("patient_sn"):
        patient_sn = str(info[0]["patient_sn"])
    return PatientRecord(patient_sn=patient_sn, source_file=str(path), sheets=sheets)


def list_patient_files(patient_dir: Path, limit: int | None = None) -> list[Path]:
    files = sorted(path for path in patient_dir.glob("*.xlsx") if not path.name.startswith("~$"))
    if limit:
        return files[:limit]
    return files

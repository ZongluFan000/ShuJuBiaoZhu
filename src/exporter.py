from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import openpyxl


def export_outputs(
    output_dir: Path,
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    summary: dict[str, Any],
    write_xlsx: bool = True,
    patient_rows: list[dict[str, Any]] | None = None,
    failed_task_rows: list[dict[str, Any]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = ""
    try:
        _write_csv(output_dir / "annotation_results.csv", results)
    except PermissionError:
        suffix = "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        _write_csv(output_dir / f"annotation_results{suffix}.csv", results)
    _write_csv(_safe_path(output_dir / "failed_tasks.csv", suffix), failures)
    if patient_rows is not None:
        _write_csv(_safe_path(output_dir / "patient_status.csv", suffix), patient_rows)
        failed_patients = [row for row in patient_rows if str(row.get("status")) in {"failed", "partial"} or int(row.get("failed_rules") or 0) > 0]
        _write_csv(_safe_path(output_dir / "failed_patients.csv", suffix), failed_patients)
    if failed_task_rows is not None:
        _write_csv(_safe_path(output_dir / "failed_tasks_from_checkpoint.csv", suffix), failed_task_rows)
    trace_cols = ["患者编号", "试验注册号", "试验标识", "标准编号", "证据来源", "参考原始病历信息"]
    traces = [{k: row.get(k, "") for k in trace_cols} for row in results]
    _write_csv(_safe_path(output_dir / "evidence_trace.csv", suffix), traces)
    if write_xlsx:
        sheets = {
            "annotation_results": results,
            "failed_tasks": failures,
        }
        if patient_rows is not None:
            sheets["patient_status"] = patient_rows
        try:
            _write_xlsx(_safe_path(output_dir / "annotation_results.xlsx", suffix), sheets)
        except PermissionError:
            alt_suffix = suffix or "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            _write_xlsx(_safe_path(output_dir / "annotation_results.xlsx", alt_suffix), sheets)
    _safe_path(output_dir / "run_summary.json", suffix).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


class LiveCsvWriter:
    def __init__(self, path: Path, fields: list[str] | None = None):
        self.path = path
        self.fields: list[str] | None = fields
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = None

    def set_lock(self, lock) -> None:
        self._lock = lock

    def append(self, row: dict[str, Any]) -> None:
        if self._lock:
            with self._lock:
                self._append_unlocked(row)
        else:
            self._append_unlocked(row)

    def _append_unlocked(self, row: dict[str, Any]) -> None:
        if self.fields is None:
            self.fields = list(row.keys())
        exists = self.path.exists() and self.path.stat().st_size > 0
        with self.path.open("a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    writer = LiveCsvWriter(path)
    writer.append(row)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = _fields(rows)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_xlsx(path: Path, sheets: dict[str, list[dict[str, Any]]]) -> None:
    wb = openpyxl.Workbook()
    first = True
    for name, rows in sheets.items():
        ws = wb.active if first else wb.create_sheet()
        first = False
        ws.title = name[:31]
        fields = _fields(rows)
        ws.append(fields)
        for row in rows:
            ws.append([row.get(field, "") for field in fields])
    wb.save(path)


def _fields(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields or ["empty"]


def _safe_path(path: Path, suffix: str) -> Path:
    if not suffix:
        return path
    return path.with_name(path.stem + suffix + path.suffix)

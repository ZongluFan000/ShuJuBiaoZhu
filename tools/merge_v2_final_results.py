from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SOURCES = [
    ROOT / "data/output_v2_deepseek_one_patient/runs/20260707_171703/annotation_results_live.csv",
    ROOT / "data/output_v2_one_patient_eval/runs/20260707_214909/annotation_results_live.csv",
    ROOT / "data/output_v2_patients_3_10_eval/runs/20260707_220044/annotation_results_live.csv",
    ROOT / "data/output_v2_patients_11_120_run/runs/20260707_223344/annotation_results_live.csv",
]

OUT_DIR = ROOT / "data/final_v2_120_patients"
OUT_CSV = OUT_DIR / "annotation_results_v2_120_patients_final.csv"
OUT_XLSX = OUT_DIR / "annotation_results_v2_120_patients_final.xlsx"
OUT_SUMMARY = OUT_DIR / "annotation_results_v2_120_patients_summary.json"

PATIENT_COL = "患者编号"
TRIAL_COL = "试验标识"
STANDARD_COL = "标准编号"
EXPECTED_RULES_PER_PATIENT = 445
EXPECTED_PATIENTS = 120


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in rows:
        row["_source_file"] = str(path.relative_to(ROOT))
    return rows


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row.get(PATIENT_COL, "").strip(),
        row.get(TRIAL_COL, "").strip(),
        row.get(STANDARD_COL, "").strip(),
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, str]] = []
    source_stats: list[dict[str, object]] = []
    for source in SOURCES:
        rows = read_rows(source)
        all_rows.extend(rows)
        per_patient = Counter(row.get(PATIENT_COL, "").strip() for row in rows)
        source_stats.append(
            {
                "source": str(source.relative_to(ROOT)),
                "rows": len(rows),
                "patients": len(per_patient),
                "complete_445_patients": sum(1 for count in per_patient.values() if count == EXPECTED_RULES_PER_PATIENT),
                "partial_patients": {
                    patient: count
                    for patient, count in sorted(per_patient.items())
                    if count != EXPECTED_RULES_PER_PATIENT
                },
            }
        )

    # Later sources are preferred if a patient/rule appears in more than one file.
    merged_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    duplicate_keys: Counter[tuple[str, str, str]] = Counter()
    for row in all_rows:
        key = row_key(row)
        if not all(key):
            continue
        if key in merged_by_key:
            duplicate_keys[key] += 1
        merged_by_key[key] = row

    merged_rows = list(merged_by_key.values())
    merged_rows.sort(
        key=lambda row: (
            row.get(PATIENT_COL, ""),
            row.get(TRIAL_COL, ""),
            row.get(STANDARD_COL, ""),
            row.get("规则标识", ""),
        )
    )

    per_patient_keys: dict[str, set[tuple[str, str]]] = defaultdict(set)
    label_counts: Counter[str] = Counter()
    for row in merged_rows:
        patient = row.get(PATIENT_COL, "").strip()
        per_patient_keys[patient].add((row.get(TRIAL_COL, "").strip(), row.get(STANDARD_COL, "").strip()))
        label_counts[row.get("标注结果", "").strip()] += 1

    patient_counts = {patient: len(keys) for patient, keys in sorted(per_patient_keys.items())}
    incomplete_patients = {
        patient: count
        for patient, count in patient_counts.items()
        if count != EXPECTED_RULES_PER_PATIENT
    }

    fieldnames = [name for name in all_rows[0].keys() if name != "_source_file"] + ["_source_file"]
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged_rows)

    summary = {
        "output_csv": str(OUT_CSV),
        "expected_patients": EXPECTED_PATIENTS,
        "expected_rules_per_patient": EXPECTED_RULES_PER_PATIENT,
        "expected_rows": EXPECTED_PATIENTS * EXPECTED_RULES_PER_PATIENT,
        "actual_patients": len(patient_counts),
        "actual_rows": len(merged_rows),
        "incomplete_patients": incomplete_patients,
        "duplicate_patient_trial_standard_keys_replaced": sum(duplicate_keys.values()),
        "label_counts": dict(label_counts),
        "source_stats": source_stats,
    }
    with OUT_SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    try:
        import pandas as pd

        results_df = pd.DataFrame(merged_rows, columns=fieldnames)
        summary_rows = [
            {"指标": "预期患者数", "值": EXPECTED_PATIENTS},
            {"指标": "实际患者数", "值": len(patient_counts)},
            {"指标": "每患者预期规则数", "值": EXPECTED_RULES_PER_PATIENT},
            {"指标": "预期总行数", "值": EXPECTED_PATIENTS * EXPECTED_RULES_PER_PATIENT},
            {"指标": "实际总行数", "值": len(merged_rows)},
            {"指标": "不完整患者数", "值": len(incomplete_patients)},
            {"指标": "去重替换记录数", "值": sum(duplicate_keys.values())},
        ]
        for label, count in sorted(label_counts.items()):
            summary_rows.append({"指标": f"标注结果={label}", "值": count})
        summary_df = pd.DataFrame(summary_rows)

        with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
            results_df.to_excel(writer, index=False, sheet_name="results")
            summary_df.to_excel(writer, index=False, sheet_name="summary")

            workbook = writer.book
            for sheet_name in ("results", "summary"):
                worksheet = workbook[sheet_name]
                worksheet.freeze_panes = "A2"
                for cell in worksheet[1]:
                    cell.font = cell.font.copy(bold=True)
                for column_cells in worksheet.columns:
                    header = str(column_cells[0].value or "")
                    width = min(max(len(header) + 4, 12), 42)
                    worksheet.column_dimensions[column_cells[0].column_letter].width = width

        summary["output_xlsx"] = str(OUT_XLSX)
        with OUT_SUMMARY.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        summary["output_xlsx_error"] = str(exc)
        with OUT_SUMMARY.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from load_rules import load_rules
from main import RESULT_FIELDS
from rule_optimizer import build_judgment_units


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "data" / "ab_experiment"
KEY_FIELDS = ("患者编号", "试验标识", "标准编号")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-file", default="100026214540.xlsx")
    parser.add_argument("--model-config", default="config/model_api.yaml")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-optimized", action="store_true")
    parser.add_argument("--gold-csv", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    patient_file = Path(args.patient_file).name
    if not (ROOT / "data" / "input" / "patient" / patient_file).exists():
        raise FileNotFoundError(f"患者文件不存在：{patient_file}")

    if not args.skip_baseline:
        reset_variant("baseline")
        run_variant("baseline", "config/project_ab_baseline.yaml", args.model_config, patient_file)
    if not args.skip_optimized:
        reset_variant("optimized")
        run_variant("optimized", "config/project_ab_optimized.yaml", args.model_config, patient_file)

    report = compare_variants(patient_file, Path(args.gold_csv) if args.gold_csv else None)
    report_dir = EXPERIMENT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"ab_report_{timestamp}.json"
    mismatch_rows = report.pop("_mismatch_rows", [])
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    mismatch_path = report_dir / f"ab_mismatches_{timestamp}.csv"
    write_csv(mismatch_path, mismatch_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report={report_path}")
    print(f"mismatches={mismatch_path}")
    return 0 if report["validation"]["optimized_output_valid"] else 2


def reset_variant(name: str) -> None:
    targets = [
        EXPERIMENT_ROOT / name,
        EXPERIMENT_ROOT / f"{name}_logs",
        EXPERIMENT_ROOT / f"{name}_checkpoint",
    ]
    root = EXPERIMENT_ROOT.resolve()
    for target in targets:
        resolved = target.resolve()
        if root not in resolved.parents:
            raise RuntimeError(f"拒绝删除试验目录之外的路径：{resolved}")
        if target.exists():
            shutil.rmtree(target)


def run_variant(name: str, project_config: str, model_config: str, patient_file: str) -> None:
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "src" / "main.py"),
        "--mode",
        "full",
        "--config",
        project_config,
        "--model-config",
        model_config,
        "--patient-file",
        patient_file,
    ]
    print(f"[{name}] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} 运行失败，returncode={proc.returncode}")


def compare_variants(patient_file: str, gold_csv: Path | None) -> dict[str, Any]:
    rules = load_rules(ROOT / "data" / "input" / "rules.xlsx")
    units = build_judgment_units(
        rules,
        merge_exact_duplicates=True,
        reviewed_groups_path=ROOT / "config" / "reviewed_equivalent_rules.json",
    )
    merge_type_by_rule = {
        (member.trial_id, member.standard_no): unit.merge_type
        for unit in units
        for member in unit.members
    }
    expected_keys = {(Path(patient_file).stem, rule.trial_id, rule.standard_no) for rule in rules}
    baseline_rows = read_csv(EXPERIMENT_ROOT / "baseline" / "annotation_results.csv")
    optimized_rows = read_csv(EXPERIMENT_ROOT / "optimized" / "annotation_results.csv")
    baseline_summary = read_json(EXPERIMENT_ROOT / "baseline" / "run_summary.json")
    optimized_summary = read_json(EXPERIMENT_ROOT / "optimized" / "run_summary.json")

    baseline_index, baseline_duplicates = index_rows(baseline_rows)
    optimized_index, optimized_duplicates = index_rows(optimized_rows)
    common_keys = sorted(set(baseline_index) & set(optimized_index))
    mismatch_rows = []
    agreement_count = 0
    confusion = Counter()
    stratified = {}
    for key in common_keys:
        baseline_label = baseline_index[key].get("标注结果", "")
        optimized_label = optimized_index[key].get("标注结果", "")
        confusion[(baseline_label, optimized_label)] += 1
        if baseline_label == optimized_label:
            agreement_count += 1
        else:
            mismatch_rows.append(
                {
                    "患者编号": key[0],
                    "试验标识": key[1],
                    "标准编号": key[2],
                    "规则标识": optimized_index[key].get("规则标识", ""),
                    "标准内容": optimized_index[key].get("标准内容", ""),
                    "基线标签": baseline_label,
                    "优化标签": optimized_label,
                    "基线解释": baseline_index[key].get("匹配解释", ""),
                    "优化解释": optimized_index[key].get("匹配解释", ""),
                }
            )
        add_stratified_result(
            stratified,
            "rule_type",
            optimized_index[key].get("规则标识", ""),
            baseline_label,
            optimized_label,
        )
        add_stratified_result(
            stratified,
            "rule_category",
            optimized_index[key].get("规则分类", ""),
            baseline_label,
            optimized_label,
        )
        add_stratified_result(
            stratified,
            "merge_type",
            merge_type_by_rule.get((key[1], key[2]), "none"),
            baseline_label,
            optimized_label,
        )

    baseline_elapsed = float(baseline_summary.get("elapsed_seconds") or 0)
    optimized_elapsed = float(optimized_summary.get("elapsed_seconds") or 0)
    speedup = baseline_elapsed / optimized_elapsed if optimized_elapsed > 0 else None
    baseline_calls = baseline_summary.get("llm_calls")
    optimized_calls = optimized_summary.get("llm_calls")
    call_reduction_rate = None
    try:
        call_reduction_rate = (int(baseline_calls) - int(optimized_calls)) / int(baseline_calls)
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    timing_valid = baseline_summary.get("model_provider") != "mock" and optimized_summary.get("model_provider") != "mock"
    baseline_format = validate_output(baseline_rows, baseline_index, baseline_duplicates, expected_keys)
    optimized_format = validate_output(optimized_rows, optimized_index, optimized_duplicates, expected_keys)
    report: dict[str, Any] = {
        "patient_file": patient_file,
        "expected_rule_count": len(expected_keys),
        "baseline": {
            "row_count": len(baseline_rows),
            "elapsed_seconds": baseline_elapsed,
            "llm_calls": baseline_summary.get("llm_calls"),
            "format_validation": baseline_format,
            "label_distribution": dict(Counter(row.get("标注结果", "") for row in baseline_rows)),
        },
        "optimized": {
            "row_count": len(optimized_rows),
            "elapsed_seconds": optimized_elapsed,
            "llm_calls": optimized_summary.get("llm_calls"),
            "optimization_plan": optimized_summary.get("optimization_plan", {}),
            "format_validation": optimized_format,
            "label_distribution": dict(Counter(row.get("标注结果", "") for row in optimized_rows)),
            "batch_failures": optimized_summary.get("optimized_batch_failures", 0),
            "fallback_splits": optimized_summary.get("optimized_fallback_splits", 0),
            "unit_fallbacks": optimized_summary.get("optimized_unit_fallbacks", 0),
        },
        "comparison": {
            "common_rule_count": len(common_keys),
            "label_agreement_count": agreement_count,
            "label_agreement_rate": round(agreement_count / len(common_keys), 6) if common_keys else None,
            "label_mismatch_count": len(mismatch_rows),
            "speedup_ratio": round(speedup, 4) if speedup is not None else None,
            "speed_measurement_valid": timing_valid,
            "speed_measurement_note": (
                "真实模型计时，可用于评估加速。"
                if timing_valid
                else "Mock不包含网络和模型推理时间，耗时与加速比仅用于程序开销观察。"
            ),
            "time_saved_seconds": round(baseline_elapsed - optimized_elapsed, 2),
            "llm_calls_saved": _subtract(baseline_calls, optimized_calls),
            "llm_call_reduction_rate": round(call_reduction_rate, 6) if call_reduction_rate is not None else None,
            "call_count_speedup_upper_bound": (
                round(int(baseline_calls) / int(optimized_calls), 4)
                if baseline_calls and optimized_calls
                else None
            ),
            "confusion": {
                f"{left}->{right}": count
                for (left, right), count in sorted(confusion.items())
            },
            "stratified_agreement": finalize_stratified(stratified),
            "headers_identical": list(baseline_rows[0]) == list(optimized_rows[0]) if baseline_rows and optimized_rows else False,
        },
        "validation": {
            "baseline_output_valid": baseline_format["valid"],
            "optimized_output_valid": optimized_format["valid"],
            "note": "没有人工金标准时，label_agreement_rate 是与原逻辑的一致率，不等同于临床准确率。",
        },
        "_mismatch_rows": mismatch_rows,
    }
    if gold_csv:
        report["gold_standard"] = compare_with_gold(baseline_index, optimized_index, gold_csv)
    return report


def validate_output(
    rows: list[dict[str, str]],
    index: dict[tuple[str, str, str], dict[str, str]],
    duplicates: list[tuple[str, str, str]],
    expected_keys: set[tuple[str, str, str]],
) -> dict[str, Any]:
    headers = set(rows[0]) if rows else set()
    missing_columns = [field for field in RESULT_FIELDS if field not in headers]
    actual_keys = set(index)
    missing_keys = sorted(expected_keys - actual_keys)
    extra_keys = sorted(actual_keys - expected_keys)
    invalid_labels = sorted({row.get("标注结果", "") for row in rows} - {"符合", "不符合", "未知"})
    required_value_fields = ("患者编号", "试验注册号", "试验标识", "标准编号", "规则标识", "标准内容", "标注结果")
    blank_required_values = sum(
        1
        for row in rows
        if any(not str(row.get(field) or "").strip() for field in required_value_fields)
    )
    valid = (
        not missing_columns
        and not duplicates
        and not missing_keys
        and not extra_keys
        and not invalid_labels
        and blank_required_values == 0
        and len(rows) == len(expected_keys)
    )
    return {
        "valid": valid,
        "required_columns_present": not missing_columns,
        "missing_columns": missing_columns,
        "duplicate_key_count": len(duplicates),
        "missing_rule_count": len(missing_keys),
        "extra_rule_count": len(extra_keys),
        "invalid_labels": invalid_labels,
        "rows_with_blank_required_values": blank_required_values,
        "sample_missing_keys": missing_keys[:10],
        "sample_extra_keys": extra_keys[:10],
    }


def compare_with_gold(
    baseline_index: dict[tuple[str, str, str], dict[str, str]],
    optimized_index: dict[tuple[str, str, str], dict[str, str]],
    gold_csv: Path,
) -> dict[str, Any]:
    gold_rows = read_csv(gold_csv)
    gold_index, _ = index_rows(gold_rows)
    return {
        "baseline": label_accuracy(baseline_index, gold_index),
        "optimized": label_accuracy(optimized_index, gold_index),
    }


def label_accuracy(
    candidate: dict[tuple[str, str, str], dict[str, str]],
    gold: dict[tuple[str, str, str], dict[str, str]],
) -> dict[str, Any]:
    keys = set(candidate) & set(gold)
    labels = ("符合", "不符合", "未知")
    correct = sum(candidate[key].get("标注结果") == gold[key].get("标注结果") for key in keys)
    per_label = {}
    f1_values = []
    for label in labels:
        true_positive = sum(
            candidate[key].get("标注结果") == label and gold[key].get("标注结果") == label
            for key in keys
        )
        predicted_positive = sum(candidate[key].get("标注结果") == label for key in keys)
        actual_positive = sum(gold[key].get("标注结果") == label for key in keys)
        precision = true_positive / predicted_positive if predicted_positive else None
        recall = true_positive / actual_positive if actual_positive else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else None
        )
        if f1 is not None:
            f1_values.append(f1)
        per_label[label] = {
            "support": actual_positive,
            "precision": round(precision, 6) if precision is not None else None,
            "recall": round(recall, 6) if recall is not None else None,
            "f1": round(f1, 6) if f1 is not None else None,
        }
    return {
        "compared": len(keys),
        "correct": correct,
        "accuracy": round(correct / len(keys), 6) if keys else None,
        "macro_f1": round(sum(f1_values) / len(f1_values), 6) if f1_values else None,
        "per_label": per_label,
    }


def index_rows(rows: list[dict[str, str]]) -> tuple[dict[tuple[str, str, str], dict[str, str]], list[tuple[str, str, str]]]:
    index = {}
    duplicates = []
    for row in rows:
        key = tuple(str(row.get(field) or "").strip() for field in KEY_FIELDS)
        if key in index:
            duplicates.append(key)
        index[key] = row
    return index, duplicates


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _subtract(left: Any, right: Any) -> int | None:
    try:
        return int(left) - int(right)
    except (TypeError, ValueError):
        return None


def add_stratified_result(
    target: dict[str, dict[str, dict[str, int]]],
    dimension: str,
    value: str,
    baseline_label: str,
    optimized_label: str,
) -> None:
    bucket = target.setdefault(dimension, {}).setdefault(value or "<empty>", {"total": 0, "agree": 0})
    bucket["total"] += 1
    bucket["agree"] += int(baseline_label == optimized_label)


def finalize_stratified(target: dict[str, dict[str, dict[str, int]]]) -> dict[str, Any]:
    return {
        dimension: {
            value: {
                **counts,
                "agreement_rate": round(counts["agree"] / counts["total"], 6) if counts["total"] else None,
            }
            for value, counts in values.items()
        }
        for dimension, values in target.items()
    }


if __name__ == "__main__":
    raise SystemExit(main())

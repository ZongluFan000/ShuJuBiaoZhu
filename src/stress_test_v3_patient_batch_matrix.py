from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from config_loader import load_yaml, resolve_path
from evidence_builder import build_evidence
from llm_client import LLMClient, LLMConfig
from load_patient import list_patient_files, load_patient
from load_rules import TrialRule, load_rules
from prompting import build_optimized_batch_prompt
from rule_classifier import classify_rule
from rule_optimizer import JudgmentUnit, build_judgment_units
from stress_test_v3 import build_llm_config, percentile


SCENARIOS = (
    ("p12_b3", 12, 3),
    ("p6_b6", 6, 6),
    ("p4_b9", 4, 9),
    ("p3_b12", 3, 12),
    ("p2_b18", 2, 18),
    ("p1_b36", 1, 36),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare patient-concurrency and batch-size trade-offs with V3 prompts.")
    parser.add_argument("--config", default="config/project_v3_stress.yaml")
    parser.add_argument("--requests-per-scenario", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    project = load_yaml(resolve_path(root, args.config))
    paths = project["paths"]
    model_settings = project["model"]
    output_dir = resolve_path(root, "data/output_v3_patient_batch_matrix")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = resolve_path(root, model_settings["config_file"])
    cfg = build_llm_config(load_yaml(model_path))
    cfg = replace(
        cfg,
        timeout_seconds=int(model_settings.get("timeout_seconds", cfg.timeout_seconds)),
        max_retries=int(model_settings.get("max_retries", cfg.max_retries)),
        enable_thinking=bool(model_settings.get("enable_thinking", False)),
    )
    rules = load_rules(resolve_path(root, paths["rules_file"]))
    units = build_judgment_units(rules, merge_exact_duplicates=False, use_reviewed_equivalent_groups=False)
    max_batch = max(batch_size for _, _, batch_size in SCENARIOS)
    source_units, source_key = select_homogeneous_units(units, max_batch)
    patient_files = list_patient_files(resolve_path(root, paths["patient_dir"]))[: args.requests_per_scenario]
    if len(patient_files) < args.requests_per_scenario:
        raise ValueError(f"Need {args.requests_per_scenario} patient files, found {len(patient_files)}")
    patients = [load_patient(path) for path in patient_files]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["LLM_USAGE_LOG_FILE"] = str(output_dir / f"matrix_usage_{run_id}.jsonl")
    os.environ["LLM_RUN_ID"] = run_id

    details: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for name, patient_concurrency, batch_size in SCENARIOS:
        scenario_cfg = replace(cfg, max_tokens=max(int(cfg.max_tokens), 110 * batch_size + 256))
        prompts = build_prompts(root, patients, source_units[:batch_size])
        print(f"scenario={name} patient_concurrency={patient_concurrency} batch_rules={batch_size} max_tokens={scenario_cfg.max_tokens}")
        if args.dry_run:
            continue
        started = time.perf_counter()
        rows = run_scenario(scenario_cfg, prompts, name, patient_concurrency)
        elapsed = time.perf_counter() - started
        details.extend(rows)
        summary = summarize(name, patient_concurrency, batch_size, rows, elapsed)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))

    files = write_outputs(output_dir, run_id, details, summaries, source_units, patient_files, args.dry_run)
    print(json.dumps(files, ensure_ascii=False, indent=2))
    return 0


def select_homogeneous_units(units: list[JudgmentUnit], required: int) -> tuple[list[JudgmentUnit], tuple[Any, ...]]:
    groups: dict[tuple[Any, ...], list[JudgmentUnit]] = {}
    for unit in units:
        key = (unit.signature.rule_type, unit.signature.evidence_category, unit.signature.complexity)
        groups.setdefault(key, []).append(unit)
    candidates = [(key, group) for key, group in groups.items() if len(group) >= required]
    if not candidates:
        raise ValueError(f"No homogeneous V3 rule group contains {required} rules")
    key, group = max(candidates, key=lambda item: (len(item[1]), sum(len(unit.representative.rule_text) for unit in item[1])))
    return sorted(group, key=lambda unit: len(unit.representative.rule_text), reverse=True), key


def build_prompts(root: Path, patients: list[Any], units: list[JudgmentUnit]) -> list[tuple[str, int, str]]:
    template = root / "config" / "prompt_templates" / "label_rules_v3_stress.md"
    prompts = []
    for patient in patients:
        evidence_cache: dict[str, Any] = {}
        prepared = []
        for unit in units:
            rule = unit.representative
            profile = classify_rule(rule)
            evidence = evidence_cache.get(profile.category)
            if evidence is None:
                evidence = build_evidence(patient, rule, profile, max_section_chars=600, max_total_chars=2200)
                evidence_cache[profile.category] = evidence
            prepared.append((rule, evidence))
        prompts.append((build_optimized_batch_prompt(template, patient.patient_sn, prepared, use_compact_template=True), len(units), patient.patient_sn))
    return prompts


def run_scenario(cfg: LLMConfig, prompts: list[tuple[str, int, str]], name: str, concurrency: int) -> list[dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(call_once, cfg, prompt, name, concurrency, index + 1) for index, prompt in enumerate(prompts)]
        rows = [future.result() for future in as_completed(futures)]
    return sorted(rows, key=lambda row: int(row["request_id"]))


def call_once(cfg: LLMConfig, prompt_info: tuple[str, int, str], name: str, concurrency: int, request_id: int) -> dict[str, Any]:
    prompt, expected_count, patient_sn = prompt_info
    started = time.perf_counter()
    try:
        data = LLMClient(cfg).label(prompt)
        items = data.get("i") if isinstance(data, dict) else None
        items = items if isinstance(items, list) else (data.get("items") if isinstance(data, dict) else None)
        if not isinstance(items, list) or len(items) != expected_count:
            raise ValueError(f"item_count_mismatch expected={expected_count} actual={len(items) if isinstance(items, list) else 'non-list'}")
        if data.get("i") is not None and {int(item.get("n", -1)) for item in items if isinstance(item, dict)} != set(range(1, expected_count + 1)):
            raise ValueError("r_number_mismatch")
        status, error_type, error = "success", "", ""
    except Exception as exc:
        status, error = "failed", str(exc)[:1000]
        error_type = classify_error(error)
    return {"scenario": name, "patient_concurrency": concurrency, "request_id": request_id, "patient_sn": patient_sn, "batch_rules": expected_count, "status": status, "error_type": error_type, "latency_seconds": round(time.perf_counter() - started, 4), "error": error}


def classify_error(error: str) -> str:
    lowered = error.lower()
    if "item_count_mismatch" in lowered:
        return "item_count_mismatch"
    if "r_number_mismatch" in lowered:
        return "r_number_mismatch"
    if "non-json" in lowered or "json" in lowered:
        return "json_format"
    if "timeout" in lowered or "502" in lowered or "503" in lowered or "504" in lowered:
        return "transport_or_timeout"
    return "api_or_other"


def summarize(name: str, patient_concurrency: int, batch_size: int, rows: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    successes = [row for row in rows if row["status"] == "success"]
    latencies = sorted(float(row["latency_seconds"]) for row in successes)
    errors: dict[str, int] = {}
    for row in rows:
        if row["status"] != "success":
            errors[row["error_type"]] = errors.get(row["error_type"], 0) + 1
    completed_rules = sum(int(row["batch_rules"]) for row in successes)
    return {"scenario": name, "patient_concurrency": patient_concurrency, "rule_concurrency": 1, "batch_rules": batch_size, "requests": len(rows), "success": len(successes), "failed": len(rows) - len(successes), "failure_rate": round((len(rows) - len(successes)) / len(rows), 4), "completed_rules": completed_rules, "elapsed_seconds": round(elapsed, 3), "rules_per_minute": round(completed_rules / elapsed * 60, 3), "avg_latency_seconds": round(statistics.mean(latencies), 3) if latencies else None, "p95_latency_seconds": percentile(latencies, 95), "failure_reasons": json.dumps(errors, ensure_ascii=False)}


def write_outputs(output_dir: Path, run_id: str, details: list[dict[str, Any]], summaries: list[dict[str, Any]], units: list[JudgmentUnit], patient_files: list[Path], dry_run: bool) -> dict[str, str]:
    detail_path = output_dir / f"matrix_detail_{run_id}.csv"
    summary_path = output_dir / f"matrix_summary_{run_id}.csv"
    json_path = output_dir / f"matrix_summary_{run_id}.json"
    for path, rows in ((detail_path, details), (summary_path, summaries)):
        fields = list(rows[0]) if rows else ["scenario"]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    json_path.write_text(json.dumps({"run_id": run_id, "dry_run": dry_run, "source_rule_group": {"rule_type": units[0].signature.rule_type, "evidence_category": units[0].signature.evidence_category, "complexity": units[0].signature.complexity}, "source_rule_keys": [f"{unit.representative.trial_id}/{unit.representative.standard_no}" for unit in units], "patient_files": [path.name for path in patient_files], "summaries": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"detail_csv": str(detail_path), "summary_csv": str(summary_path), "summary_json": str(json_path)}


if __name__ == "__main__":
    sys.exit(main())

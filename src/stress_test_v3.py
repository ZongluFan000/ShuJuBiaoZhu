from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import MISSING, fields, replace
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
from rule_optimizer import JudgmentUnit, build_judgment_units, build_optimized_batches, rule_complexity_cost


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V3 standard-layer DeepSeek stress test; no annotation output is written.")
    parser.add_argument("--config", default="config/project_v3_stress.yaml")
    parser.add_argument("--model-config", default=None, help="Override model config; API key is never copied into this project.")
    parser.add_argument("--strategy", choices=["conservative", "balanced", "upper_bound", "all"], default="all")
    parser.add_argument("--requests-per-level", type=int, default=None)
    parser.add_argument("--levels", default=None, help="Override comma-separated concurrency levels for one stress run.")
    parser.add_argument("--continue-after-failed-level", action="store_true", help="Record all requested levels even when a guardrail is exceeded.")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts and manifests without calling the model API.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    project = load_yaml(resolve_path(root, args.config))
    paths = project["paths"]
    sampling = project.get("sampling", {})
    guards = project.get("guardrails", {})
    model_settings = project.get("model", {})
    patient_dir = resolve_path(root, paths["patient_dir"])
    rules_file = resolve_path(root, paths["rules_file"])
    out_dir = resolve_path(root, paths["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = resolve_path(root, args.model_config or model_settings["config_file"])
    cfg = build_llm_config(load_yaml(model_path))
    cfg = replace(cfg, timeout_seconds=int(model_settings.get("timeout_seconds", cfg.timeout_seconds)), max_retries=int(model_settings.get("max_retries", cfg.max_retries)), max_tokens=int(model_settings.get("max_tokens", cfg.max_tokens)), enable_thinking=bool(model_settings.get("enable_thinking", False)))

    rules = load_rules(rules_file)
    units = build_judgment_units(rules, merge_exact_duplicates=False, use_reviewed_equivalent_groups=False)
    patient_files = list_patient_files(patient_dir)[: max(1, int(sampling.get("patient_count", 1)))]
    if not patient_files:
        raise FileNotFoundError(f"No patient files found: {patient_dir}")
    patients = [load_patient(path) for path in patient_files]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["LLM_USAGE_LOG_FILE"] = str(out_dir / f"stress_usage_{run_id}.jsonl")
    os.environ["LLM_RUN_ID"] = run_id

    names = [args.strategy] if args.strategy != "all" else ["conservative", "balanced", "upper_bound"]
    all_details: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    stopped = False
    for name in names:
        strategy = project["strategies"][name]
        prompts, sample_rows, plan = build_v3_prompts(root, patients, units, strategy, int(sampling.get("batch_count", 12)))
        manifest.extend({"strategy": name, **row} for row in sample_rows)
        print(f"strategy={name} rules={len(rules)} plan_batches={len(plan)} sampled_prompts={len(prompts)} chars={min(len(p[0]) for p in prompts)}-{max(len(p[0]) for p in prompts)}")
        if args.dry_run:
            continue
        levels = parse_levels(args.levels or str(strategy["concurrency_levels"]))
        for level in levels:
            warmup = int(sampling.get("warmup_requests", 1))
            for index in range(warmup):
                row = call_once(cfg, prompts[index % len(prompts)], name, level, f"warmup-{index + 1}")
                all_details.append(row)
                print(format_row(row))
                if row["status"] != "success":
                    stopped = True
                    break
            if stopped:
                break
            requests = args.requests_per_level or int(sampling.get("requests_per_level", 12))
            started = time.perf_counter()
            rows = run_level(cfg, prompts, name, level, requests)
            elapsed = time.perf_counter() - started
            all_details.extend(rows)
            summary = summarize(name, level, rows, elapsed)
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False))
            exceeded = summary["failure_rate"] > float(guards.get("failure_threshold", 0.10)) or (summary["p95_latency_seconds"] is not None and summary["p95_latency_seconds"] > float(guards.get("max_p95_latency_seconds", 45))) or summary["timeout_like_failures"] >= int(guards.get("max_timeout_like_failures", 1))
            if exceeded and not args.continue_after_failed_level:
                print("STOP: V3 guardrail exceeded; higher pressure levels are skipped.")
                stopped = True
                break
            if exceeded:
                print("WARN: V3 guardrail exceeded; continuing because --continue-after-failed-level is enabled.")
            time.sleep(float(sampling.get("cooldown_seconds", 0)))
        if stopped:
            break
    output = write_outputs(out_dir, run_id, all_details, summaries, manifest, rules, patient_files, args.dry_run)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 1 if stopped else 0


def build_llm_config(raw: dict[str, Any]) -> LLMConfig:
    values: dict[str, Any] = {}
    for field in fields(LLMConfig):
        values[field.name] = raw.get(field.name) if raw.get(field.name) is not None else (field.default if field.default is not MISSING else "")
    return LLMConfig(**values)


def build_v3_prompts(root: Path, patients: list[Any], units: list[JudgmentUnit], strategy: dict[str, Any], sample_count: int) -> tuple[list[tuple[str, int, str]], list[dict[str, Any]], list[list[JudgmentUnit]]]:
    plan = build_optimized_batches(units, max_rules=int(strategy["max_batch_rules"]), max_rule_chars=int(strategy["max_batch_rule_chars"]), max_complex_rules=int(strategy["max_complex_batch_rules"]), max_simple_rules=int(strategy["max_simple_batch_rules"]), max_normal_rules=int(strategy["max_normal_batch_rules"]), complexity_budget=int(strategy["batch_complexity_budget"]))
    selected = select_representative_batches(plan, sample_count)
    template = root / "config" / "prompt_templates" / "label_rules_v3_stress.md"
    prompts: list[tuple[str, int, str]] = []
    manifest: list[dict[str, Any]] = []
    for patient_index, patient in enumerate(patients):
        for batch_index, batch in selected:
            prepared = []
            for unit in batch:
                rule = unit.representative
                profile = classify_rule(rule)
                evidence = build_evidence(patient, rule, profile, max_section_chars=600, max_total_chars=2200)
                prepared.append((rule, evidence))
            prompt = build_optimized_batch_prompt(template, patient.patient_sn, prepared, use_compact_template=True)
            prompts.append((prompt, len(batch), f"P{patient_index + 1}-B{batch_index}"))
            manifest.append({"sample_id": f"P{patient_index + 1}-B{batch_index}", "patient_sn": patient.patient_sn, "batch_index": batch_index, "batch_rules": len(batch), "rule_chars": sum(len(unit.representative.rule_text) for unit in batch), "complexity_cost": sum(rule_complexity_cost(unit) for unit in batch), "complexities": ",".join(unit.signature.complexity for unit in batch), "time_anchors": ",".join(unit.signature.time_anchor for unit in batch), "prompt_chars": len(prompt), "rule_keys": ";".join(f"{unit.representative.trial_id}/{unit.representative.standard_no}" for unit in batch)})
    return prompts, manifest, plan


def select_representative_batches(plan: list[list[JudgmentUnit]], count: int) -> list[tuple[int, list[JudgmentUnit]]]:
    ordered = sorted(enumerate(plan, start=1), key=lambda item: (len(item[1]), sum(rule_complexity_cost(unit) for unit in item[1]), sum(len(unit.representative.rule_text) for unit in item[1])), reverse=True)
    chosen: list[tuple[int, list[JudgmentUnit]]] = []
    seen: set[tuple[int, int, str]] = set()
    for index, batch in ordered:
        key = (len(batch), sum(rule_complexity_cost(unit) for unit in batch), batch[0].signature.complexity)
        if key not in seen or len(chosen) < max(3, count // 2):
            chosen.append((index, batch)); seen.add(key)
        if len(chosen) >= count:
            return chosen
    return chosen or [(1, plan[0])]


def parse_levels(value: str) -> list[int]:
    return [max(1, int(item.strip())) for item in value.split(",") if item.strip()]


def run_level(cfg: LLMConfig, prompts: list[tuple[str, int, str]], strategy: str, level: int, requests: int) -> list[dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=level) as pool:
        futures = [pool.submit(call_once, cfg, prompts[index % len(prompts)], strategy, level, index + 1) for index in range(requests)]
        rows = [future.result() for future in as_completed(futures)]
    return sorted(rows, key=lambda row: str(row["request_id"]))


def call_once(cfg: LLMConfig, prompt_info: tuple[str, int, str], strategy: str, level: int, request_id: int | str) -> dict[str, Any]:
    prompt, expected_count, sample_id = prompt_info
    started = time.perf_counter()
    try:
        data = LLMClient(cfg).label(prompt)
        items = data.get("i") if isinstance(data, dict) else None
        items = items if isinstance(items, list) else (data.get("items") if isinstance(data, dict) else None)
        if not isinstance(items, list) or len(items) != expected_count:
            raise ValueError(f"output item count mismatch: expected={expected_count} actual={len(items) if isinstance(items, list) else 'non-list'}")
        if data.get("i") is not None and {int(item.get("n", -1)) for item in items if isinstance(item, dict)} != set(range(1, expected_count + 1)):
            raise ValueError("compact output R numbers do not exactly match the batch")
        return {"strategy": strategy, "concurrency": level, "request_id": request_id, "sample_id": sample_id, "batch_rules": expected_count, "status": "success", "latency_seconds": round(time.perf_counter() - started, 4), "error": "", "finished_at": datetime.now().isoformat(timespec="seconds")}
    except Exception as exc:
        return {"strategy": strategy, "concurrency": level, "request_id": request_id, "sample_id": sample_id, "batch_rules": expected_count, "status": "failed", "latency_seconds": round(time.perf_counter() - started, 4), "error": str(exc)[:1000], "finished_at": datetime.now().isoformat(timespec="seconds")}


def summarize(strategy: str, level: int, rows: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    ok = [row for row in rows if row["status"] == "success"]
    latencies = sorted(float(row["latency_seconds"]) for row in ok)
    timeouts = sum("timeout" in str(row["error"]).lower() or "502" in str(row["error"]) or "503" in str(row["error"]) or "504" in str(row["error"]) for row in rows if row["status"] != "success")
    return {"strategy": strategy, "concurrency": level, "requests": len(rows), "success": len(ok), "failed": len(rows) - len(ok), "failure_rate": round((len(rows) - len(ok)) / max(len(rows), 1), 4), "successful_rules": sum(int(row["batch_rules"]) for row in ok), "elapsed_seconds": round(elapsed, 3), "requests_per_minute": round(len(ok) / elapsed * 60, 3) if elapsed else 0, "rules_per_minute": round(sum(int(row["batch_rules"]) for row in ok) / elapsed * 60, 3) if elapsed else 0, "avg_latency_seconds": round(statistics.mean(latencies), 3) if latencies else None, "p95_latency_seconds": percentile(latencies, 95), "timeout_like_failures": timeouts}


def percentile(values: list[float], p: int) -> float | None:
    if not values: return None
    pos = (len(values) - 1) * p / 100; lo = int(pos); hi = min(lo + 1, len(values) - 1)
    return round(values[lo] * (hi - pos) + values[hi] * (pos - lo), 3)


def format_row(row: dict[str, Any]) -> str:
    return f"strategy={row['strategy']} concurrency={row['concurrency']} sample={row['sample_id']} {row['status']} {row['latency_seconds']}s {row['error'][:160]}"


def write_outputs(out_dir: Path, run_id: str, details: list[dict[str, Any]], summaries: list[dict[str, Any]], manifest: list[dict[str, Any]], rules: list[TrialRule], patient_files: list[Path], dry_run: bool) -> dict[str, str]:
    files = {"detail_csv": out_dir / f"v3_stress_detail_{run_id}.csv", "summary_csv": out_dir / f"v3_stress_summary_{run_id}.csv", "manifest_csv": out_dir / f"v3_stress_manifest_{run_id}.csv", "summary_json": out_dir / f"v3_stress_summary_{run_id}.json"}
    for key, rows in (("detail_csv", details), ("summary_csv", summaries), ("manifest_csv", manifest)):
        fields_out = list(rows[0]) if rows else (["strategy", "concurrency", "requests", "success", "failed"] if key == "summary_csv" else ["sample_id"])
        with files[key].open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields_out); writer.writeheader(); writer.writerows(rows)
    files["summary_json"].write_text(json.dumps({"run_id": run_id, "dry_run": dry_run, "rules_file": "rules_v3_standard.xlsx", "rule_count": len(rules), "patient_files": [path.name for path in patient_files], "summaries": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {key: str(path) for key, path in files.items()}


if __name__ == "__main__":
    sys.exit(main())

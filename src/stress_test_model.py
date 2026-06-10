from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import MISSING, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from config_loader import load_yaml, resolve_path
from evidence_builder import build_evidence
from load_patient import list_patient_files, load_patient
from load_rules import TrialRule, load_rules
from llm_client import LLMClient, LLMConfig
from prompting import build_batch_prompt, build_optimized_batch_prompt
from rule_classifier import classify_rule


SHORT_PROMPT = """请只输出合法 JSON，不要输出 Markdown，不要解释：
{"items":[{"label":"未知","explanation":"并发压测","evidence":"stress_test","confidence":0.1}]}
"""


ANNOTATION_PROMPT = """你是临床试验入排标准标注助手。请根据患者证据判断标准。
标注规则：入选标准满足=符合，不满足=不符合，证据不足=未知；排除标准满足排除条件=不符合，不满足=符合，证据不足=未知。
只输出合法 JSON，不要输出 Markdown，不要输出额外解释。

患者证据：
患者，女，56 岁。诊断：宫颈癌。病理：鳞状细胞癌。ECOG 评分 1。血常规、肝肾功能未见明确异常记录。未见既往免疫治疗记录。

待判断标准：
1. 入选标准：年龄 18-75 岁。
2. 入选标准：ECOG 评分 0-1。
3. 排除标准：既往接受过免疫检查点抑制剂治疗。
4. 入选标准：组织学确诊为宫颈癌。

输出格式：
{"items":[{"standard_no":"1","label":"符合/不符合/未知","explanation":"80字以内","evidence":"40字以内","confidence":0.0}]}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stepped concurrency stress test for an OpenAI-compatible model API.")
    parser.add_argument("--model-config", default="config/model_api.yaml", help="Model YAML config path.")
    parser.add_argument("--project-config", default="config/project_full_ai_only.yaml", help="Project YAML used to build production-like prompts.")
    parser.add_argument("--levels", default="16,20,24", help="Comma-separated concurrency levels, e.g. 16,20,24.")
    parser.add_argument("--requests-per-level", type=int, default=24, help="Total requests for each concurrency level.")
    parser.add_argument("--warmup", type=int, default=1, help="Sequential warmup requests before each level.")
    parser.add_argument("--cooldown-seconds", type=float, default=5.0, help="Sleep seconds between levels.")
    parser.add_argument("--prompt-mode", choices=["production", "short", "annotation"], default="production")
    parser.add_argument("--sample-patients", type=int, default=2, help="Patients used to build production-like prompt samples.")
    parser.add_argument("--sample-rule-batches", type=int, default=8, help="Rule batches used to build production-like prompt samples.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override production batch size; default reads project labeling.llm_batch_size.")
    parser.add_argument("--timeout-seconds", type=int, default=None, help="Override model timeout for stress test.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max_tokens for stress test.")
    parser.add_argument("--stream", choices=["config", "true", "false"], default="config", help="Override stream setting.")
    parser.add_argument("--failure-threshold", type=float, default=0.2, help="Stop after a level if failure rate is above this value.")
    parser.add_argument("--max-avg-latency-seconds", type=float, default=60.0, help="Stop before higher levels if average latency exceeds this value.")
    parser.add_argument("--max-p95-latency-seconds", type=float, default=90.0, help="Stop before higher levels if P95 latency exceeds this value.")
    parser.add_argument("--max-timeout-like-failures", type=int, default=2, help="Stop before higher levels after this many timeout/connection failures in one level.")
    parser.add_argument("--max-initial-submit", type=int, default=8, help="Initial requests submitted per level before rolling refill.")
    parser.add_argument("--output-dir", default="data/output_model_stress")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = build_config(load_yaml(resolve_arg_path(root, args.model_config)))
    cfg = apply_overrides(cfg, args)
    levels = parse_levels(args.levels)
    prompts = build_prompt_samples(root, args)

    out_dir = resolve_arg_path(root, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"base_url={cfg.base_url}")
    print(f"model_name={cfg.model_name}")
    print(f"levels={levels}")
    print(f"requests_per_level={args.requests_per_level} prompt_mode={args.prompt_mode} prompt_samples={len(prompts)}")

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    stopped_early = False

    for level in levels:
        if args.warmup > 0:
            print(f"\n[warmup] level={level} requests={args.warmup}")
            warmup_failed = False
            for index in range(1, args.warmup + 1):
                row = call_once(cfg, prompts[(index - 1) % len(prompts)], level, f"warmup-{index}")
                print(format_row(row))
                if row.get("status") != "success":
                    warmup_failed = True
            if warmup_failed:
                summaries.append({
                    "run_id": run_id,
                    "base_url": cfg.base_url,
                    "model_name": cfg.model_name,
                    "prompt_mode": args.prompt_mode,
                    "concurrency": level,
                    "requests": 0,
                    "success": 0,
                    "failed": 0,
                    "success_rate": 0.0,
                    "failure_rate": 0.0,
                    "total_elapsed_seconds": 0,
                    "requests_per_second": 0,
                    "requests_per_minute": 0,
                    "avg_latency_seconds": None,
                    "p50_latency_seconds": None,
                    "p90_latency_seconds": None,
                    "p95_latency_seconds": None,
                    "p99_latency_seconds": None,
                    "timeout_like_failures": 0,
                    "error_counts": {},
                    "stop_reason": "warmup failed; stopped before submitting concurrency requests",
                })
                print("STOP: warmup failed; stopped before submitting concurrency requests")
                stopped_early = True
                break

        print(f"\n[level] concurrency={level} requests={args.requests_per_level}")
        started = time.perf_counter()
        rows = run_level(cfg, prompts, level, max(1, args.requests_per_level), args)
        elapsed = time.perf_counter() - started
        all_rows.extend(rows)
        summary = summarize_level(cfg, args, run_id, level, rows, elapsed)
        summaries.append(summary)

        print(json.dumps(summary, ensure_ascii=False, indent=2))
        should_stop, stop_reason = should_stop_after_level(summary, args)
        if should_stop:
            summary["stop_reason"] = stop_reason
            print(f"STOP: {stop_reason}")
            stopped_early = True
            break
        if args.cooldown_seconds > 0 and level != levels[-1]:
            time.sleep(args.cooldown_seconds)

    paths = write_outputs(out_dir, run_id, all_rows, summaries, cfg, args, stopped_early)
    print("\nresult_files:")
    for key, value in paths.items():
        print(f"{key}={value}")
    return 1 if stopped_early or any(item["failed"] for item in summaries) else 0


def build_config(raw: dict[str, Any]) -> LLMConfig:
    values = {}
    for field in fields(LLMConfig):
        if field.name in raw and raw.get(field.name) is not None:
            values[field.name] = raw.get(field.name)
        elif field.default is not MISSING:
            values[field.name] = field.default
        else:
            values[field.name] = ""
    return LLMConfig(**values)


def apply_overrides(cfg: LLMConfig, args: argparse.Namespace) -> LLMConfig:
    updates: dict[str, Any] = {}
    if args.timeout_seconds is not None:
        updates["timeout_seconds"] = args.timeout_seconds
    if args.max_tokens is not None:
        updates["max_tokens"] = args.max_tokens
    if args.stream == "true":
        updates["stream"] = True
    elif args.stream == "false":
        updates["stream"] = False
    return replace(cfg, **updates) if updates else cfg


def build_prompt_samples(root: Path, args: argparse.Namespace) -> list[str]:
    if args.prompt_mode == "short":
        return [SHORT_PROMPT]
    if args.prompt_mode == "annotation":
        return [ANNOTATION_PROMPT]

    project_path = resolve_arg_path(root, args.project_config)
    project = load_yaml(project_path)
    paths = project.get("paths", {})
    labeling = project.get("labeling", {})
    patient_dir = resolve_path(root, paths.get("patient_dir", "data/input/patient"))
    rules_file = resolve_path(root, paths.get("rules_file", "data/input/rules.xlsx"))
    template_path = root / "config" / "prompt_templates" / "label_rules.md"

    patients = list_patient_files(patient_dir, limit=max(1, args.sample_patients))
    if not patients:
        raise FileNotFoundError(f"no patient xlsx files found in {patient_dir}")
    rules = load_rules(rules_file)
    if not rules:
        raise ValueError(f"no rules found in {rules_file}")

    batch_size = args.batch_size or int(labeling.get("llm_batch_size", 10) or 10)
    batch_size = max(1, min(batch_size, 50))
    max_section_chars = int(labeling.get("max_evidence_chars_per_section", 600) or 600)
    max_total_chars = int(labeling.get("max_total_evidence_chars", 2200) or 2200)
    use_optimized = bool(project.get("optimization", {}).get("enabled", False))
    use_compact = bool(labeling.get("use_compact_prompt", True))

    prompts: list[str] = []
    batch_count = max(1, args.sample_rule_batches)
    for patient_file in patients:
        patient = load_patient(patient_file)
        for offset in range(0, min(len(rules), batch_count * batch_size), batch_size):
            batch_rules = rules[offset : offset + batch_size]
            prepared: list[tuple[TrialRule, Any]] = []
            for rule in batch_rules:
                profile = classify_rule(rule)
                evidence = build_evidence(
                    patient,
                    rule,
                    profile,
                    max_section_chars=max_section_chars,
                    max_total_chars=max_total_chars,
                )
                prepared.append((rule, evidence))
            if use_optimized:
                prompt = build_optimized_batch_prompt(template_path, patient.patient_sn, prepared, use_compact_template=use_compact)
            else:
                prompt = build_batch_prompt(template_path, patient.patient_sn, prepared)
            prompts.append(prompt)
            if len(prompts) >= batch_count:
                break
        if len(prompts) >= batch_count:
            break

    if not prompts:
        raise ValueError("failed to build production prompt samples")
    avg_chars = round(sum(len(prompt) for prompt in prompts) / len(prompts))
    print(f"production_prompt_samples={len(prompts)} avg_chars={avg_chars} batch_size={batch_size}")
    return prompts


def parse_levels(text: str) -> list[int]:
    levels = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 1:
            raise ValueError("concurrency level must be >= 1")
        levels.append(value)
    if not levels:
        raise ValueError("at least one concurrency level is required")
    return levels


def resolve_arg_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def run_level(cfg: LLMConfig, prompts: list[str], level: int, requests: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=level) as pool:
        futures = {}
        next_index = 1
        initial_submit = max(1, min(level, requests, int(args.max_initial_submit or level)))
        while next_index <= initial_submit:
            futures[pool.submit(call_once, cfg, prompts[(next_index - 1) % len(prompts)], level, next_index)] = next_index
            next_index += 1

        while futures:
            future = next(as_completed(futures))
            futures.pop(future, None)
            row = future.result()
            rows.append(row)
            print(format_row(row))
            if should_stop_mid_level(rows, args):
                print("STOP current level: protective breaker triggered; no more requests will be submitted for this level.")
                for pending in futures:
                    pending.cancel()
                break
            if next_index <= requests:
                futures[pool.submit(call_once, cfg, prompts[(next_index - 1) % len(prompts)], level, next_index)] = next_index
                next_index += 1
    return sorted(rows, key=lambda item: int(item["request_id"]))


def call_once(cfg: LLMConfig, prompt: str, level: int, request_id: int | str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        data = LLMClient(cfg).label(prompt)
        elapsed = time.perf_counter() - started
        items = data.get("items") if isinstance(data, dict) else None
        ok = isinstance(items, list) and bool(items)
        return {
            "concurrency": level,
            "request_id": request_id,
            "status": "success" if ok else "failed",
            "latency_seconds": round(elapsed, 4),
            "error": "" if ok else "response JSON missing non-empty items",
            "response_preview": json.dumps(data, ensure_ascii=False)[:500],
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return {
            "concurrency": level,
            "request_id": request_id,
            "status": "failed",
            "latency_seconds": round(elapsed, 4),
            "error": str(exc)[:1000],
            "response_preview": "",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }


def format_row(row: dict[str, Any]) -> str:
    error = f" error={row['error'][:120]}" if row.get("error") else ""
    return f"concurrency={row['concurrency']} request={row['request_id']} {row['status']} {row['latency_seconds']}s{error}"


def summarize_level(
    cfg: LLMConfig,
    args: argparse.Namespace,
    run_id: str,
    level: int,
    rows: list[dict[str, Any]],
    elapsed: float,
) -> dict[str, Any]:
    success = [row for row in rows if row["status"] == "success"]
    failed = [row for row in rows if row["status"] != "success"]
    latencies = sorted(float(row["latency_seconds"]) for row in success)
    error_counts: dict[str, int] = {}
    timeout_like_failures = 0
    for row in failed:
        error_text = str(row.get("error") or "unknown")
        key = error_text[:180]
        error_counts[key] = error_counts.get(key, 0) + 1
        if is_timeout_like_error(error_text):
            timeout_like_failures += 1
    return {
        "run_id": run_id,
        "base_url": cfg.base_url,
        "model_name": cfg.model_name,
        "prompt_mode": args.prompt_mode,
        "concurrency": level,
        "requests": len(rows),
        "success": len(success),
        "failed": len(failed),
        "success_rate": round(len(success) / max(len(rows), 1), 4),
        "failure_rate": round(len(failed) / max(len(rows), 1), 4),
        "total_elapsed_seconds": round(elapsed, 3),
        "requests_per_second": round(len(success) / elapsed, 3) if elapsed > 0 else 0,
        "requests_per_minute": round(len(success) / elapsed * 60, 3) if elapsed > 0 else 0,
        "avg_latency_seconds": round(statistics.mean(latencies), 3) if latencies else None,
        "min_latency_seconds": round(min(latencies), 3) if latencies else None,
        "max_latency_seconds": round(max(latencies), 3) if latencies else None,
        "p50_latency_seconds": percentile(latencies, 50),
        "p90_latency_seconds": percentile(latencies, 90),
        "p95_latency_seconds": percentile(latencies, 95),
        "p99_latency_seconds": percentile(latencies, 99),
        "timeout_like_failures": timeout_like_failures,
        "error_counts": error_counts,
    }


def should_stop_after_level(summary: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    failure_rate = float(summary.get("failure_rate") or 0)
    if failure_rate > args.failure_threshold:
        return True, f"failure_rate {failure_rate} > threshold {args.failure_threshold}"

    avg_latency = summary.get("avg_latency_seconds")
    if avg_latency is not None and float(avg_latency) > args.max_avg_latency_seconds:
        return True, f"avg_latency_seconds {avg_latency} > threshold {args.max_avg_latency_seconds}"

    p95_latency = summary.get("p95_latency_seconds")
    if p95_latency is not None and float(p95_latency) > args.max_p95_latency_seconds:
        return True, f"p95_latency_seconds {p95_latency} > threshold {args.max_p95_latency_seconds}"

    timeout_like_failures = int(summary.get("timeout_like_failures") or 0)
    if timeout_like_failures >= args.max_timeout_like_failures:
        return True, f"timeout_like_failures {timeout_like_failures} >= threshold {args.max_timeout_like_failures}"

    return False, ""


def should_stop_mid_level(rows: list[dict[str, Any]], args: argparse.Namespace) -> bool:
    if len(rows) < 3:
        return False
    failed = [row for row in rows if row.get("status") != "success"]
    failure_rate = len(failed) / max(len(rows), 1)
    if failure_rate > args.failure_threshold:
        return True
    timeout_like_failures = sum(1 for row in failed if is_timeout_like_error(str(row.get("error") or "")))
    if timeout_like_failures >= args.max_timeout_like_failures:
        return True
    success_latencies = sorted(float(row["latency_seconds"]) for row in rows if row.get("status") == "success")
    if len(success_latencies) >= 3:
        avg_latency = statistics.mean(success_latencies)
        if avg_latency > args.max_avg_latency_seconds:
            return True
        p95_latency = percentile(success_latencies, 95)
        if p95_latency is not None and p95_latency > args.max_p95_latency_seconds:
            return True
    return False


def is_timeout_like_error(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
        "connection refused",
        "remote host",
        "bad gateway",
        "502",
        "503",
        "504",
        "winerror 10060",
        "winerror 10065",
        "10060",
        "10065",
        "无法连接",
        "没有正确答复",
        "主机没有反应",
        "连接尝试失败",
        "远程主机",
        "强迫关闭",
    ]
    return any(marker in lowered for marker in markers)


def percentile(values: list[float], p: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 3)
    pos = (len(values) - 1) * p / 100
    lower = int(pos)
    upper = min(lower + 1, len(values) - 1)
    weight = pos - lower
    return round(values[lower] * (1 - weight) + values[upper] * weight, 3)


def write_outputs(
    out_dir: Path,
    run_id: str,
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    cfg: LLMConfig,
    args: argparse.Namespace,
    stopped_early: bool,
) -> dict[str, str]:
    detail_path = out_dir / f"stress_detail_{run_id}.csv"
    summary_csv_path = out_dir / f"stress_summary_{run_id}.csv"
    summary_json_path = out_dir / f"stress_summary_{run_id}.json"

    detail_fields = ["concurrency", "request_id", "status", "latency_seconds", "error", "response_preview", "finished_at"]
    with detail_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_fields = [
        "concurrency",
        "requests",
        "success",
        "failed",
        "success_rate",
        "failure_rate",
        "total_elapsed_seconds",
        "requests_per_second",
        "requests_per_minute",
        "avg_latency_seconds",
        "p50_latency_seconds",
        "p90_latency_seconds",
        "p95_latency_seconds",
        "p99_latency_seconds",
    ]
    with summary_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows([{key: item.get(key) for key in summary_fields} for item in summaries])

    payload = {
        "run_id": run_id,
        "base_url": cfg.base_url,
        "model_name": cfg.model_name,
        "prompt_mode": args.prompt_mode,
        "levels": [item["concurrency"] for item in summaries],
        "requests_per_level": args.requests_per_level,
        "warmup": args.warmup,
        "cooldown_seconds": args.cooldown_seconds,
        "failure_threshold": args.failure_threshold,
        "max_avg_latency_seconds": args.max_avg_latency_seconds,
        "max_p95_latency_seconds": args.max_p95_latency_seconds,
        "max_timeout_like_failures": args.max_timeout_like_failures,
        "max_initial_submit": args.max_initial_submit,
        "stopped_early": stopped_early,
        "summaries": summaries,
    }
    summary_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "detail_csv": detail_path.as_posix(),
        "summary_csv": summary_csv_path.as_posix(),
        "summary_json": summary_json_path.as_posix(),
    }


if __name__ == "__main__":
    raise SystemExit(main())

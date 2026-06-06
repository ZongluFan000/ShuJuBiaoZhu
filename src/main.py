from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import MISSING, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from checkpoint import Checkpoint
from config_loader import load_yaml, resolve_path
from evidence_builder import build_evidence
from exporter import LiveCsvWriter, export_outputs
from llm_client import LLMClient, LLMConfig
from load_patient import PatientRecord, list_patient_files, load_patient
from load_rules import TrialRule, load_rules
from prompting import build_batch_prompt, build_optimized_batch_prompt, build_prompt
from rule_classifier import classify_rule
from rule_optimizer import (
    JudgmentUnit,
    build_judgment_units,
    build_optimized_batches,
    effective_batch_rule_limit,
    optimization_summary,
    validate_optimization_plan,
)
from structured_judge import try_structured_judge
from validator import (
    apply_missing_policy,
    normalize_llm_batch_result,
    normalize_llm_batch_result_strict,
    normalize_llm_result,
)


RESULT_FIELDS = [
    "试验注册号",
    "试验标识",
    "标准编号",
    "规则标识",
    "标准内容",
    "患者编号",
    "标注结果",
    "匹配解释",
    "参考原始病历信息",
    "证据来源",
    "置信度",
    "标注来源",
    "规则后处理",
    "规则分类",
    "运行批次号",
    "标注时间",
]

FAILURE_FIELDS = ["patient_sn", "patient_file", "trial_id", "standard_no", "rule_type", "error", "run_id"]
PATIENT_STATUS_FIELDS = ["patient_sn", "source_file", "status", "total_rules", "done_rules", "failed_rules", "last_error", "run_id", "updated_at"]
MODEL_HEALTH_PROMPT = """请只输出一个合法 JSON 对象，不要输出 Markdown：
{"items":[{"label":"未知","explanation":"连通性测试","evidence":"测试","confidence":0.1}]}
"""


class CircuitOpenError(RuntimeError):
    pass


class FailureCircuitBreaker:
    def __init__(self, max_consecutive_failures: int):
        self.max_consecutive_failures = max(0, int(max_consecutive_failures or 0))
        self.consecutive_failures = 0
        self.reason = ""
        self.lock = threading.Lock()

    def is_open(self) -> bool:
        with self.lock:
            return bool(self.reason)

    def before_call(self) -> None:
        with self.lock:
            if self.reason:
                raise CircuitOpenError(self.reason)

    def record_success(self) -> None:
        with self.lock:
            self.consecutive_failures = 0

    def record_failure(self, message: str) -> None:
        if self.max_consecutive_failures <= 0:
            return
        with self.lock:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_consecutive_failures and not self.reason:
                self.reason = (
                    f"模型连续失败 {self.consecutive_failures} 次，已自动熔断暂停；"
                    f"最后错误：{message[:300]}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["check", "test", "benchmark", "full", "retry_failed"], default="test")
    parser.add_argument("--config", default="config/project_full_ai_only.yaml")
    parser.add_argument("--model-config", default="config/model_api.yaml")
    parser.add_argument("--patient-limit", type=int, default=None)
    parser.add_argument("--rule-limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--patient-file", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    project = load_yaml(root / args.config)
    model_raw = load_yaml(root / args.model_config)

    paths = project["paths"]
    patient_dir = resolve_path(root, paths["patient_dir"])
    rules_file = resolve_path(root, paths["rules_file"])
    output_dir = resolve_path(root, paths["output_dir"])
    log_dir = resolve_path(root, paths.get("log_dir", "data/logs"))
    checkpoint_db = resolve_path(root, paths["checkpoint_db"])
    template_path = root / "config" / "prompt_templates" / "label_rules.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "check":
        return check_environment(patient_dir, rules_file, model_raw)

    run_cfg = project.get("run", {})
    labeling_cfg = project.get("labeling", {})
    optimization_cfg = project.get("optimization", {})
    model_config = build_llm_config(model_raw)
    patient_limit = args.patient_limit if args.patient_limit is not None else run_cfg.get("patient_limit")
    rule_limit = args.rule_limit if args.rule_limit is not None else run_cfg.get("rule_limit")
    concurrency = int(args.concurrency if args.concurrency is not None else run_cfg.get("concurrency", model_raw.get("concurrency", 1)) or 1)
    concurrency = max(1, concurrency)
    rule_concurrency = max(1, int(run_cfg.get("rule_concurrency", 1) or 1))
    if model_config.provider == "local_transformers" and (concurrency > 1 or rule_concurrency > 1):
        print("local_transformers 模式不适合多线程加载模型，已将 concurrency 强制设为 1。")
        concurrency = 1
        rule_concurrency = 1
    if bool(run_cfg.get("model_preflight", True)):
        try:
            check_model_ready(model_config)
        except Exception as exc:
            print(f"模型连通性检查失败，已停止启动，避免继续写入大量失败 checkpoint: {exc}")
            return 3
    if args.mode == "full":
        patient_limit = None
        rule_limit = None
    elif args.mode == "benchmark":
        patient_limit = patient_limit or 100

    rules = load_rules(rules_file, limit=rule_limit)
    optimized_plan_units: list[JudgmentUnit] | None = None
    optimized_plan_batches: list[list[JudgmentUnit]] | None = None
    optimized_plan_summary: dict[str, Any] | None = None
    if bool(optimization_cfg.get("enabled", False)):
        reviewed_path = resolve_path(root, optimization_cfg.get("reviewed_groups_file", "config/reviewed_equivalent_rules.json"))
        optimized_plan_units = build_judgment_units(
            rules,
            merge_exact_duplicates=bool(optimization_cfg.get("merge_exact_duplicates", True)),
            reviewed_groups_path=reviewed_path,
            use_reviewed_equivalent_groups=bool(
                optimization_cfg.get("use_reviewed_equivalent_groups", False)
            ),
        )
        configured_max_rules = int(optimization_cfg.get("max_batch_rules", labeling_cfg.get("llm_batch_size", 8)) or 8)
        effective_max_rules = effective_batch_rule_limit(
            configured_max_rules=configured_max_rules,
            model_max_tokens=model_config.max_tokens,
            estimated_output_tokens_per_rule=int(optimization_cfg.get("estimated_output_tokens_per_rule", 110) or 110),
            output_token_reserve=int(optimization_cfg.get("output_token_reserve", 128) or 128),
        )
        optimized_plan_batches = build_optimized_batches(
            optimized_plan_units,
            max_rules=effective_max_rules,
            max_rule_chars=int(optimization_cfg.get("max_batch_rule_chars", 3000) or 3000),
            max_complex_rules=int(optimization_cfg.get("max_complex_batch_rules", 3) or 3),
            max_simple_rules=int(optimization_cfg.get("max_simple_batch_rules", effective_max_rules) or effective_max_rules),
            max_normal_rules=int(optimization_cfg.get("max_normal_batch_rules", effective_max_rules) or effective_max_rules),
            complexity_budget=int(optimization_cfg.get("batch_complexity_budget", 12) or 12),
        )
        validate_optimization_plan(rules, optimized_plan_units, optimized_plan_batches)
        optimized_plan_summary = optimization_summary(optimized_plan_units, optimized_plan_batches, len(rules))
        optimized_plan_summary["configured_max_batch_rules"] = configured_max_rules
        optimized_plan_summary["effective_max_batch_rules"] = effective_max_rules
        optimized_plan_summary["model_max_tokens"] = model_config.max_tokens
        optimized_plan_summary["compact_prompt_enabled"] = bool(
            labeling_cfg.get("use_compact_prompt", True)
        )
        optimized_plan_summary["exact_duplicate_merge_enabled"] = bool(
            optimization_cfg.get("merge_exact_duplicates", False)
        )
        optimized_plan_summary["reviewed_equivalent_groups_enabled"] = bool(
            optimization_cfg.get("use_reviewed_equivalent_groups", False)
        )
    patient_files = list_patient_files(patient_dir)
    if args.patient_file:
        requested_name = Path(args.patient_file).name
        patient_files = [path for path in patient_files if path.name == requested_name]
        if not patient_files:
            raise FileNotFoundError(f"指定患者文件不存在：{requested_name}")
    else:
        patient_files = apply_patient_selection(root, patient_dir, patient_files)
    if patient_limit:
        patient_files = patient_files[:patient_limit]
    checkpoint = Checkpoint(checkpoint_db)
    if args.mode == "retry_failed":
        failed_files = checkpoint.failed_patient_files()
        if failed_files:
            patient_files = [Path(path) for path in failed_files.values() if Path(path).exists()]
        else:
            failed_ids = checkpoint.failed_patient_ids()
            patient_files = [p for p in patient_files if p.stem in failed_ids]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_writer = LiveCsvWriter(output_dir / "annotation_results_live.csv", RESULT_FIELDS)
    failure_writer = LiveCsvWriter(output_dir / "failed_tasks_live.csv", FAILURE_FIELDS)
    patient_writer = LiveCsvWriter(output_dir / "patient_status_live.csv", PATIENT_STATUS_FIELDS)
    write_lock = threading.Lock()
    result_writer.set_lock(write_lock)
    failure_writer.set_lock(write_lock)
    patient_writer.set_lock(write_lock)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    result_lock = threading.Lock()
    circuit = FailureCircuitBreaker(int(run_cfg.get("max_consecutive_llm_failures", 5) or 0))
    run_metrics = {
        "llm_calls": 0,
        "optimized_batch_failures": 0,
        "optimized_fallback_splits": 0,
        "optimized_unit_fallbacks": 0,
    }
    metrics_lock = threading.Lock()
    started = time.time()

    def worker(patient_file: Path) -> dict[str, Any]:
        client = LLMClient(model_config)
        process = process_patient_optimized if bool(optimization_cfg.get("enabled", False)) else process_patient
        return process(
            patient_file=patient_file,
            rules=rules,
            client=client,
            template_path=template_path,
            labeling_cfg=labeling_cfg,
            run_cfg=run_cfg,
            checkpoint=checkpoint,
            run_id=run_id,
            result_writer=result_writer,
            failure_writer=failure_writer,
            patient_writer=patient_writer,
            results=results,
            failures=failures,
            result_lock=result_lock,
            rule_concurrency=rule_concurrency,
            circuit=circuit,
            run_metrics=run_metrics,
            metrics_lock=metrics_lock,
            optimization_cfg=optimization_cfg,
            root=root,
            prebuilt_optimized_batches=optimized_plan_batches,
        )

    patient_summaries: list[dict[str, Any]] = []
    if concurrency == 1:
        for patient_file in patient_files:
            patient_summaries.append(worker(patient_file))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_map = {pool.submit(worker, patient_file): patient_file for patient_file in patient_files}
            for future in as_completed(future_map):
                patient_file = future_map[future]
                try:
                    patient_summaries.append(future.result())
                except Exception as exc:
                    failure = {"patient_sn": patient_file.stem, "patient_file": str(patient_file), "trial_id": "", "standard_no": "", "rule_type": "", "error": f"worker_crash: {exc}", "run_id": run_id}
                    with result_lock:
                        failures.append(failure)
                    failure_writer.append(failure)

    elapsed = time.time() - started
    task_counts = checkpoint.task_status_counts()
    patient_rows = checkpoint.patient_rows()
    failed_task_rows = checkpoint.failed_task_rows()
    summary = {
        "mode": args.mode,
        "run_id": run_id,
        "concurrency": concurrency,
        "rule_concurrency": rule_concurrency,
        "group_rules_for_llm": bool(labeling_cfg.get("group_rules_for_llm", False)),
        "llm_batch_size": int(labeling_cfg.get("llm_batch_size", 1) or 1),
        "llm_calls": int(run_metrics["llm_calls"]),
        "optimized_batch_failures": int(run_metrics["optimized_batch_failures"]),
        "optimized_fallback_splits": int(run_metrics["optimized_fallback_splits"]),
        "optimized_unit_fallbacks": int(run_metrics["optimized_unit_fallbacks"]),
        "optimization_enabled": bool(optimization_cfg.get("enabled", False)),
        "max_consecutive_llm_failures": circuit.max_consecutive_failures,
        "circuit_open_reason": circuit.reason,
        "patients_seen": len(patient_files),
        "rules_seen": len(rules),
        "results_written_this_run": len(results),
        "failures_this_run": len(failures),
        "checkpoint_task_counts": task_counts,
        "failed_patients": len([r for r in patient_rows if str(r.get("status")) in {"failed", "partial"} or int(r.get("failed_rules") or 0) > 0]),
        "elapsed_seconds": round(elapsed, 2),
        "estimated_40000_patients_days": estimate_days(elapsed, len(patient_files), 40000),
        "model_provider": model_raw.get("provider"),
        "model_name": model_raw.get("model_name"),
    }
    if optimized_plan_summary is not None:
        summary["optimization_plan"] = optimized_plan_summary
    export_outputs(
        output_dir,
        results,
        failures,
        summary,
        write_xlsx=bool(run_cfg.get("write_xlsx", False)),
        patient_rows=patient_rows,
        failed_task_rows=failed_task_rows,
    )
    (log_dir / f"run_{run_id}.log").write_text(
        "\n".join(
            [
                f"mode={args.mode}",
                f"run_id={run_id}",
                f"concurrency={concurrency}",
                f"rule_concurrency={rule_concurrency}",
                f"group_rules_for_llm={bool(labeling_cfg.get('group_rules_for_llm', False))}",
                f"llm_batch_size={int(labeling_cfg.get('llm_batch_size', 1) or 1)}",
                f"llm_calls={int(run_metrics['llm_calls'])}",
                f"optimization_enabled={bool(optimization_cfg.get('enabled', False))}",
                f"max_consecutive_llm_failures={circuit.max_consecutive_failures}",
                f"circuit_open_reason={circuit.reason}",
                f"patients_seen={len(patient_files)}",
                f"rules_seen={len(rules)}",
                f"results_written_this_run={len(results)}",
                f"failures_this_run={len(failures)}",
                f"elapsed_seconds={round(elapsed, 2)}",
            ]
        ),
        encoding="utf-8",
    )
    print(f"完成: results={len(results)} failures={len(failures)} output={output_dir}")
    return 0


def process_patient(
    patient_file: Path,
    rules: list[TrialRule],
    client: LLMClient,
    template_path: Path,
    labeling_cfg: dict[str, Any],
    run_cfg: dict[str, Any],
    checkpoint: Checkpoint,
    run_id: str,
    result_writer: LiveCsvWriter,
    failure_writer: LiveCsvWriter,
    patient_writer: LiveCsvWriter,
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    result_lock: threading.Lock,
    rule_concurrency: int,
    circuit: FailureCircuitBreaker,
    run_metrics: dict[str, int] | None = None,
    metrics_lock: threading.Lock | None = None,
    optimization_cfg: dict[str, Any] | None = None,
    root: Path | None = None,
    prebuilt_optimized_batches: list[list[JudgmentUnit]] | None = None,
) -> dict[str, Any]:
    try:
        patient = load_patient(patient_file)
    except Exception as exc:
        patient_sn = patient_file.stem
        failure = {"patient_sn": patient_sn, "patient_file": str(patient_file), "trial_id": "", "standard_no": "", "rule_type": "", "error": f"load_patient: {exc}", "run_id": run_id}
        checkpoint.mark_patient(patient_sn, str(patient_file), "failed", len(rules), 0, len(rules), str(exc))
        with result_lock:
            failures.append(failure)
        failure_writer.append(failure)
        patient_writer.append(_patient_status_row(patient_sn, str(patient_file), "failed", len(rules), 0, len(rules), str(exc), run_id))
        return failure

    checkpoint.mark_patient(patient.patient_sn, patient.source_file, "running", len(rules), 0, 0)
    done_count = 0
    failed_count = 0
    last_error = ""

    def run_rule(rule: TrialRule) -> dict[str, Any]:
        circuit.before_call()
        _record_llm_call(run_metrics, metrics_lock)
        try:
            row = label_one(client, template_path, patient, rule, labeling_cfg, run_id)
        except Exception as exc:
            circuit.record_failure(str(exc))
            raise
        circuit.record_success()
        return row

    def run_batch(batch_rules: list[TrialRule]) -> list[dict[str, Any]]:
        circuit.before_call()
        _record_llm_call(run_metrics, metrics_lock)
        try:
            rows = label_batch(client, template_path, patient, batch_rules, labeling_cfg, run_id)
        except Exception as exc:
            circuit.record_failure(str(exc))
            raise
        circuit.record_success()
        return rows

    def write_success(rule: TrialRule, row: dict[str, Any]) -> None:
        nonlocal done_count
        with result_lock:
            results.append(row)
        result_writer.append(row)
        checkpoint.mark_task(patient.patient_sn, rule.trial_id, rule.standard_no, "done")
        done_count += 1

    def write_failure(rule: TrialRule, exc: Exception) -> None:
        nonlocal failed_count, last_error
        failed_count += 1
        last_error = str(exc)
        checkpoint.mark_task(patient.patient_sn, rule.trial_id, rule.standard_no, "failed", last_error)
        failure = {
            "patient_sn": patient.patient_sn,
            "patient_file": patient.source_file,
            "trial_id": rule.trial_id,
            "standard_no": rule.standard_no,
            "rule_type": rule.rule_type,
            "error": last_error,
            "run_id": run_id,
        }
        with result_lock:
            failures.append(failure)
        failure_writer.append(failure)

    group_rules = bool(labeling_cfg.get("group_rules_for_llm", False))
    llm_batch_size = max(1, int(labeling_cfg.get("llm_batch_size", 1) or 1))
    if group_rules and llm_batch_size > 1:
        pending_rules = []
        for rule in rules:
            if run_cfg.get("resume", True) and checkpoint.done(patient.patient_sn, rule.trial_id, rule.standard_no):
                done_count += 1
            else:
                pending_rules.append(rule)
        batches = group_rules_for_batches(pending_rules, llm_batch_size)
        batch_iter = iter(batches)
        futures = {}

        def run_single_fallback(batch_rules: list[TrialRule], error: Exception) -> None:
            nonlocal last_error
            if len(batch_rules) <= 1 or circuit.is_open():
                for fallback_rule in batch_rules:
                    write_failure(fallback_rule, error)
                return
            for fallback_rule in batch_rules:
                if circuit.is_open():
                    last_error = circuit.reason
                    return
                try:
                    write_success(fallback_rule, run_rule(fallback_rule))
                except CircuitOpenError as exc:
                    last_error = str(exc)
                    return
                except Exception as exc:
                    write_failure(fallback_rule, exc)

        def submit_more_batches(pool: ThreadPoolExecutor) -> None:
            while len(futures) < rule_concurrency and not circuit.is_open():
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    return
                futures[pool.submit(run_batch, batch)] = batch

        if rule_concurrency <= 1:
            for batch in batches:
                if circuit.is_open():
                    last_error = circuit.reason
                    break
                try:
                    for batch_rule, row in zip(batch, run_batch(batch)):
                        write_success(batch_rule, row)
                except CircuitOpenError as exc:
                    last_error = str(exc)
                    break
                except Exception as exc:
                    run_single_fallback(batch, exc)
        else:
            with ThreadPoolExecutor(max_workers=rule_concurrency) as pool:
                submit_more_batches(pool)
                while futures:
                    for future in as_completed(list(futures.keys())):
                        batch = futures.pop(future)
                        try:
                            for batch_rule, row in zip(batch, future.result()):
                                write_success(batch_rule, row)
                        except CircuitOpenError as exc:
                            last_error = str(exc)
                        except Exception as exc:
                            run_single_fallback(batch, exc)
                        break
                    submit_more_batches(pool)
            if circuit.is_open() and not last_error:
                last_error = circuit.reason
    elif rule_concurrency <= 1:
        for rule in rules:
            if run_cfg.get("resume", True) and checkpoint.done(patient.patient_sn, rule.trial_id, rule.standard_no):
                done_count += 1
                continue
            if circuit.is_open():
                last_error = circuit.reason
                break
            try:
                write_success(rule, run_rule(rule))
            except CircuitOpenError as exc:
                last_error = str(exc)
                break
            except Exception as exc:
                write_failure(rule, exc)
    else:
        rule_iter = iter(rules)
        futures = {}

        def submit_more(pool: ThreadPoolExecutor) -> None:
            nonlocal done_count
            while len(futures) < rule_concurrency and not circuit.is_open():
                try:
                    rule = next(rule_iter)
                except StopIteration:
                    return
                if run_cfg.get("resume", True) and checkpoint.done(patient.patient_sn, rule.trial_id, rule.standard_no):
                    done_count += 1
                    continue
                futures[pool.submit(run_rule, rule)] = rule

        with ThreadPoolExecutor(max_workers=rule_concurrency) as pool:
            submit_more(pool)
            while futures:
                for future in as_completed(list(futures.keys())):
                    rule = futures.pop(future)
                    try:
                        write_success(rule, future.result())
                    except CircuitOpenError as exc:
                        last_error = str(exc)
                    except Exception as exc:
                        write_failure(rule, exc)
                    break
                submit_more(pool)
        if circuit.is_open() and not last_error:
            last_error = circuit.reason

    if failed_count == 0 and done_count >= len(rules):
        status = "done"
    elif done_count > 0 or failed_count > 0:
        status = "partial"
    else:
        status = "failed"
    checkpoint.mark_patient(patient.patient_sn, patient.source_file, status, len(rules), done_count, failed_count, last_error)
    status_row = _patient_status_row(patient.patient_sn, patient.source_file, status, len(rules), done_count, failed_count, last_error, run_id)
    patient_writer.append(status_row)
    return status_row


def process_patient_optimized(
    patient_file: Path,
    rules: list[TrialRule],
    client: LLMClient,
    template_path: Path,
    labeling_cfg: dict[str, Any],
    run_cfg: dict[str, Any],
    checkpoint: Checkpoint,
    run_id: str,
    result_writer: LiveCsvWriter,
    failure_writer: LiveCsvWriter,
    patient_writer: LiveCsvWriter,
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    result_lock: threading.Lock,
    rule_concurrency: int,
    circuit: FailureCircuitBreaker,
    run_metrics: dict[str, int] | None = None,
    metrics_lock: threading.Lock | None = None,
    optimization_cfg: dict[str, Any] | None = None,
    root: Path | None = None,
    prebuilt_optimized_batches: list[list[JudgmentUnit]] | None = None,
) -> dict[str, Any]:
    optimization_cfg = optimization_cfg or {}
    root = root or Path(__file__).resolve().parents[1]
    try:
        patient = load_patient(patient_file)
    except Exception as exc:
        patient_sn = patient_file.stem
        failure = {"patient_sn": patient_sn, "patient_file": str(patient_file), "trial_id": "", "standard_no": "", "rule_type": "", "error": f"load_patient: {exc}", "run_id": run_id}
        checkpoint.mark_patient(patient_sn, str(patient_file), "failed", len(rules), 0, len(rules), str(exc))
        with result_lock:
            failures.append(failure)
        failure_writer.append(failure)
        patient_writer.append(_patient_status_row(patient_sn, str(patient_file), "failed", len(rules), 0, len(rules), str(exc), run_id))
        return failure

    checkpoint.mark_patient(patient.patient_sn, patient.source_file, "running", len(rules), 0, 0)
    done_count = 0
    failed_count = 0
    last_error = ""
    if prebuilt_optimized_batches is None:
        reviewed_path = resolve_path(root, optimization_cfg.get("reviewed_groups_file", "config/reviewed_equivalent_rules.json"))
        units = build_judgment_units(
            rules,
            merge_exact_duplicates=bool(optimization_cfg.get("merge_exact_duplicates", True)),
            reviewed_groups_path=reviewed_path,
            use_reviewed_equivalent_groups=bool(
                optimization_cfg.get("use_reviewed_equivalent_groups", False)
            ),
        )
        effective_max_rules = effective_batch_rule_limit(
            configured_max_rules=int(optimization_cfg.get("max_batch_rules", labeling_cfg.get("llm_batch_size", 8)) or 8),
            model_max_tokens=client.config.max_tokens,
            estimated_output_tokens_per_rule=int(optimization_cfg.get("estimated_output_tokens_per_rule", 110) or 110),
            output_token_reserve=int(optimization_cfg.get("output_token_reserve", 128) or 128),
        )
        prebuilt_optimized_batches = build_optimized_batches(
            units,
            max_rules=effective_max_rules,
            max_rule_chars=int(optimization_cfg.get("max_batch_rule_chars", 3000) or 3000),
            max_complex_rules=int(optimization_cfg.get("max_complex_batch_rules", 3) or 3),
            max_simple_rules=int(optimization_cfg.get("max_simple_batch_rules", effective_max_rules) or effective_max_rules),
            max_normal_rules=int(optimization_cfg.get("max_normal_batch_rules", effective_max_rules) or effective_max_rules),
            complexity_budget=int(optimization_cfg.get("batch_complexity_budget", 12) or 12),
        )
        validate_optimization_plan(rules, units, prebuilt_optimized_batches)
    batches = _filter_pending_optimized_batches(
        prebuilt_optimized_batches,
        patient.patient_sn,
        checkpoint,
        resume=bool(run_cfg.get("resume", True)),
    )
    done_count = len(rules) - sum(len(unit.members) for batch in batches for unit in batch)

    def run_unit(unit: JudgmentUnit) -> list[tuple[TrialRule, dict[str, Any]]]:
        circuit.before_call()
        _record_llm_call(run_metrics, metrics_lock)
        try:
            representative_row = label_one(client, template_path, patient, unit.representative, labeling_cfg, run_id)
        except Exception as exc:
            circuit.record_failure(str(exc))
            raise
        circuit.record_success()
        return [(member, clone_result_for_rule(representative_row, member)) for member in unit.members]

    def run_batch(batch_units: list[JudgmentUnit]) -> list[tuple[TrialRule, dict[str, Any]]]:
        circuit.before_call()
        _record_llm_call(run_metrics, metrics_lock)
        representatives = [unit.representative for unit in batch_units]
        try:
            representative_rows = label_batch_optimized(client, template_path, patient, representatives, labeling_cfg, run_id)
        except Exception as exc:
            _increment_metric(run_metrics, metrics_lock, "optimized_batch_failures")
            circuit.record_failure(str(exc))
            raise
        circuit.record_success()
        expanded: list[tuple[TrialRule, dict[str, Any]]] = []
        for unit, representative_row in zip(batch_units, representative_rows):
            expanded.extend((member, clone_result_for_rule(representative_row, member)) for member in unit.members)
        return expanded

    def write_success(rule: TrialRule, row: dict[str, Any]) -> None:
        nonlocal done_count
        with result_lock:
            results.append(row)
        result_writer.append(row)
        checkpoint.mark_task(patient.patient_sn, rule.trial_id, rule.standard_no, "done")
        done_count += 1

    def write_failure(rule: TrialRule, exc: Exception) -> None:
        nonlocal failed_count, last_error
        failed_count += 1
        last_error = str(exc)
        checkpoint.mark_task(patient.patient_sn, rule.trial_id, rule.standard_no, "failed", last_error)
        failure = {
            "patient_sn": patient.patient_sn,
            "patient_file": patient.source_file,
            "trial_id": rule.trial_id,
            "standard_no": rule.standard_no,
            "rule_type": rule.rule_type,
            "error": last_error,
            "run_id": run_id,
        }
        with result_lock:
            failures.append(failure)
        failure_writer.append(failure)

    def fallback_batch(batch_units: list[JudgmentUnit], batch_error: Exception) -> None:
        nonlocal last_error
        if circuit.is_open():
            for unit in batch_units:
                for member in unit.members:
                    write_failure(member, batch_error)
            return
        if len(batch_units) == 1:
            _increment_metric(run_metrics, metrics_lock, "optimized_unit_fallbacks")
            unit = batch_units[0]
            try:
                for member, row in run_unit(unit):
                    write_success(member, row)
            except Exception as exc:
                for member in unit.members:
                    write_failure(member, exc)
            return
        midpoint = len(batch_units) // 2
        _increment_metric(run_metrics, metrics_lock, "optimized_fallback_splits")
        for half in (batch_units[:midpoint], batch_units[midpoint:]):
            if not half:
                continue
            try:
                for member, row in run_batch(half):
                    write_success(member, row)
            except Exception as exc:
                fallback_batch(half, exc)
            if circuit.is_open():
                last_error = circuit.reason
                return

    batch_iter = iter(batches)
    futures: dict[Any, list[JudgmentUnit]] = {}

    def submit_more(pool: ThreadPoolExecutor) -> None:
        while len(futures) < rule_concurrency and not circuit.is_open():
            try:
                batch = next(batch_iter)
            except StopIteration:
                return
            futures[pool.submit(run_batch, batch)] = batch

    if rule_concurrency <= 1:
        for batch in batches:
            if circuit.is_open():
                last_error = circuit.reason
                break
            try:
                for member, row in run_batch(batch):
                    write_success(member, row)
            except Exception as exc:
                fallback_batch(batch, exc)
    else:
        with ThreadPoolExecutor(max_workers=rule_concurrency) as pool:
            submit_more(pool)
            while futures:
                for future in as_completed(list(futures.keys())):
                    batch = futures.pop(future)
                    try:
                        for member, row in future.result():
                            write_success(member, row)
                    except Exception as exc:
                        fallback_batch(batch, exc)
                    break
                submit_more(pool)
        if circuit.is_open() and not last_error:
            last_error = circuit.reason

    if failed_count == 0 and done_count >= len(rules):
        status = "done"
    elif done_count > 0 or failed_count > 0:
        status = "partial"
    else:
        status = "failed"
    checkpoint.mark_patient(patient.patient_sn, patient.source_file, status, len(rules), done_count, failed_count, last_error)
    status_row = _patient_status_row(patient.patient_sn, patient.source_file, status, len(rules), done_count, failed_count, last_error, run_id)
    patient_writer.append(status_row)
    return status_row


def build_llm_config(raw: dict[str, Any]) -> LLMConfig:
    values: dict[str, Any] = {}
    for field in fields(LLMConfig):
        if field.name in raw and raw.get(field.name) is not None:
            values[field.name] = raw.get(field.name)
        elif field.default is not MISSING:
            values[field.name] = field.default
        else:
            values[field.name] = ""
    return LLMConfig(**values)


def check_model_ready(config: LLMConfig) -> None:
    if config.provider in {"mock", "local_transformers"}:
        return
    check_config = replace(
        config,
        timeout_seconds=max(5, min(int(config.timeout_seconds or 180), 20)),
        max_retries=1,
        max_tokens=max(64, min(int(config.max_tokens or 1024), 128)),
    )
    LLMClient(check_config).label(MODEL_HEALTH_PROMPT)


def label_one(client: LLMClient, template_path: Path, patient: PatientRecord, rule: TrialRule, labeling_cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    profile = classify_rule(rule)
    structured = None
    use_structured_judge = bool(labeling_cfg.get("use_structured_judge", True))
    if use_structured_judge and profile.needs_llm is False:
        structured = try_structured_judge(patient, rule)
    evidence = build_evidence(
        patient,
        rule,
        profile,
        max_section_chars=int(labeling_cfg.get("max_evidence_chars_per_section", 3500)),
        max_total_chars=int(labeling_cfg.get("max_total_evidence_chars", 16000)),
    )
    if structured:
        normalized = structured
    else:
        prompt = build_prompt(template_path, patient.patient_sn, rule, evidence)
        normalized = normalize_llm_result(client.label(prompt), rule)
        if bool(labeling_cfg.get("apply_missing_policy", True)):
            normalized = apply_missing_policy(normalized, rule, profile)

    return build_result_row(patient, rule, profile.category, evidence, normalized, run_id)


def label_batch(client: LLMClient, template_path: Path, patient: PatientRecord, rules: list[TrialRule], labeling_cfg: dict[str, Any], run_id: str) -> list[dict[str, Any]]:
    prepared = []
    profiles = {}
    evidences = {}
    evidence_cache = {}
    for rule in rules:
        profile = classify_rule(rule)
        evidence = evidence_cache.get(profile.category)
        if evidence is None:
            evidence = build_evidence(
                patient,
                rule,
                profile,
                max_section_chars=int(labeling_cfg.get("max_evidence_chars_per_section", 3500)),
                max_total_chars=int(labeling_cfg.get("max_total_evidence_chars", 16000)),
            )
            evidence_cache[profile.category] = evidence
        prepared.append((rule, evidence))
        profiles[(rule.trial_id, rule.standard_no)] = profile
        evidences[(rule.trial_id, rule.standard_no)] = evidence

    prompt = build_batch_prompt(template_path, patient.patient_sn, prepared)
    normalized_items = normalize_llm_batch_result(client.label(prompt), rules)
    rows = []
    for rule, normalized in zip(rules, normalized_items):
        profile = profiles[(rule.trial_id, rule.standard_no)]
        evidence = evidences[(rule.trial_id, rule.standard_no)]
        if bool(labeling_cfg.get("apply_missing_policy", True)):
            normalized = apply_missing_policy(normalized, rule, profile)
        rows.append(build_result_row(patient, rule, profile.category, evidence, normalized, run_id))
    if len(rows) != len(rules):
        raise ValueError(f"批量模型输出数量不一致：expected={len(rules)} actual={len(rows)}")
    return rows


def label_batch_optimized(
    client: LLMClient,
    template_path: Path,
    patient: PatientRecord,
    rules: list[TrialRule],
    labeling_cfg: dict[str, Any],
    run_id: str,
) -> list[dict[str, Any]]:
    prepared = []
    profiles = {}
    evidences = {}
    for rule in rules:
        profile = classify_rule(rule)
        evidence = build_evidence(
            patient,
            rule,
            profile,
            max_section_chars=int(labeling_cfg.get("max_evidence_chars_per_section", 3500)),
            max_total_chars=int(labeling_cfg.get("max_total_evidence_chars", 16000)),
        )
        prepared.append((rule, evidence))
        profiles[(rule.trial_id, rule.standard_no)] = profile
        evidences[(rule.trial_id, rule.standard_no)] = evidence

    prompt = build_optimized_batch_prompt(
        template_path,
        patient.patient_sn,
        prepared,
        use_compact_template=bool(labeling_cfg.get("use_compact_prompt", True)),
    )
    normalized_items = normalize_llm_batch_result_strict(client.label(prompt), rules)
    rows = []
    for rule, normalized in zip(rules, normalized_items):
        profile = profiles[(rule.trial_id, rule.standard_no)]
        evidence = evidences[(rule.trial_id, rule.standard_no)]
        if bool(labeling_cfg.get("apply_missing_policy", True)):
            normalized = apply_missing_policy(normalized, rule, profile)
        rows.append(build_result_row(patient, rule, profile.category, evidence, normalized, run_id))
    return rows


def build_result_row(
    patient: PatientRecord,
    rule: TrialRule,
    category: str,
    evidence: Any,
    normalized: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    return {
        "试验注册号": rule.trial_register_id,
        "试验标识": rule.trial_id,
        "标准编号": rule.standard_no,
        "规则标识": rule.rule_type,
        "标准内容": rule.rule_text,
        "患者编号": patient.patient_sn,
        "标注结果": normalized["label"],
        "匹配解释": normalized["explanation"],
        "参考原始病历信息": normalized["evidence"] or evidence.text[:1200],
        "证据来源": ",".join(evidence.sources),
        "置信度": normalized["confidence"],
        "标注来源": normalized["source"],
        "规则后处理": normalized.get("policy_adjusted", "no"),
        "规则分类": category,
        "运行批次号": run_id,
        "标注时间": datetime.now().isoformat(timespec="seconds"),
    }


def clone_result_for_rule(row: dict[str, Any], rule: TrialRule) -> dict[str, Any]:
    cloned = dict(row)
    cloned.update(
        {
            "试验注册号": rule.trial_register_id,
            "试验标识": rule.trial_id,
            "标准编号": rule.standard_no,
            "规则标识": rule.rule_type,
            "标准内容": rule.rule_text,
        }
    )
    return cloned


def chunked(items: list[TrialRule], size: int) -> list[list[TrialRule]]:
    size = max(1, int(size or 1))
    return [items[idx:idx + size] for idx in range(0, len(items), size)]


def group_rules_for_batches(rules: list[TrialRule], size: int) -> list[list[TrialRule]]:
    groups: dict[str, list[TrialRule]] = {}
    for rule in rules:
        category = classify_rule(rule).category
        groups.setdefault(category, []).append(rule)
    batches: list[list[TrialRule]] = []
    for grouped_rules in groups.values():
        batches.extend(chunked(grouped_rules, size))
    return batches


def _filter_pending_optimized_batches(
    batches: list[list[JudgmentUnit]],
    patient_sn: str,
    checkpoint: Checkpoint,
    resume: bool,
) -> list[list[JudgmentUnit]]:
    if not resume:
        return batches
    pending_batches: list[list[JudgmentUnit]] = []
    for batch in batches:
        pending_units: list[JudgmentUnit] = []
        for unit in batch:
            pending_members = tuple(
                member
                for member in unit.members
                if not checkpoint.done(patient_sn, member.trial_id, member.standard_no)
            )
            if not pending_members:
                continue
            if len(pending_members) == len(unit.members):
                pending_units.append(unit)
            else:
                pending_units.append(
                    JudgmentUnit(
                        canonical_rule_id=unit.canonical_rule_id,
                        representative=pending_members[0],
                        members=pending_members,
                        signature=unit.signature,
                        merge_type=unit.merge_type,
                    )
                )
        if pending_units:
            pending_batches.append(pending_units)
    return pending_batches


def _patient_status_row(patient_sn: str, source_file: str, status: str, total_rules: int, done_rules: int, failed_rules: int, last_error: str, run_id: str) -> dict[str, Any]:
    return {
        "patient_sn": patient_sn,
        "source_file": source_file,
        "status": status,
        "total_rules": total_rules,
        "done_rules": done_rules,
        "failed_rules": failed_rules,
        "last_error": last_error,
        "run_id": run_id,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def apply_patient_selection(root: Path, patient_dir: Path, patient_files: list[Path]) -> list[Path]:
    selection_path = root / "data" / "selected_patients.json"
    if not selection_path.exists():
        return patient_files
    try:
        payload = json.loads(selection_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        print(f"患者选择清单读取失败，将运行全部患者: {exc}")
        return patient_files

    if not bool(payload.get("enabled", False)):
        return patient_files

    selected_names = {Path(str(name)).name for name in payload.get("files", []) if str(name).strip()}
    if not selected_names:
        print("患者选择清单已启用但未选择任何患者，本次不会运行患者文件。")
        return []

    filtered = [path for path in patient_files if path.name in selected_names]
    missing = sorted(selected_names - {path.name for path in filtered})
    if missing:
        print(f"患者选择清单中有 {len(missing)} 个文件不在当前患者目录 {patient_dir} 中，已忽略。")
    print(f"患者选择清单已启用: selected={len(filtered)} total_available={len(patient_files)}")
    return filtered


def check_environment(patient_dir: Path, rules_file: Path, model_raw: dict[str, Any]) -> int:
    print(f"patient_dir={patient_dir} exists={patient_dir.exists()}")
    print(f"rules_file={rules_file} exists={rules_file.exists()}")
    provider = str(model_raw.get("provider") or "").strip()
    base_url = str(model_raw.get("base_url") or "").strip()
    model_name = str(model_raw.get("model_name") or "").strip()
    api_key = str(model_raw.get("api_key") or "").strip()
    print(f"model_provider={provider} base_url={base_url} model={model_name} api_key={_mask_secret(api_key)}")
    if not rules_file.exists():
        print("规则文件不存在。")
        return 2
    rules = load_rules(rules_file, limit=3)
    print(f"规则读取成功，示例数量={len(rules)}")
    files = list_patient_files(patient_dir, limit=3) if patient_dir.exists() else []
    print(f"患者文件示例数量={len(files)}")
    model_path = str(model_raw.get("model_path") or "").strip()
    config_errors = validate_model_config(provider, base_url, model_name, api_key, model_path)
    if config_errors:
        print("模型配置尚不能进行真实测试：")
        for error in config_errors:
            print(f"- {error}")
        if provider == "mock":
            print("当前是 mock 模式：可测试流程，但不会调用真实模型。")
    elif provider == "mock":
        print("当前是 mock 模式：可测试流程，但不会调用真实模型。")
    else:
        print("模型配置格式检查通过。")
    return 0


def estimate_days(elapsed: float, processed_patients: int, target_patients: int) -> float | None:
    if processed_patients <= 0:
        return None
    return round((elapsed / processed_patients * target_patients) / 86400, 2)


def validate_model_config(provider: str, base_url: str, model_name: str, api_key: str, model_path: str = "") -> list[str]:
    errors: list[str] = []
    placeholder_markers = ["替换", "服务器C地址", "API key", "实际模型名", "your", "xxx"]
    if provider not in {"mock", "openai_compatible", "local_transformers"}:
        errors.append("provider 必须是 mock、openai_compatible 或 local_transformers。")
    if provider == "openai_compatible":
        if not base_url or any(marker in base_url for marker in placeholder_markers):
            errors.append("base_url 还没有填写真实模型服务地址。")
        if not model_name or any(marker in model_name for marker in placeholder_markers):
            errors.append("model_name 还没有填写真实模型名称。")
        if not api_key or any(marker in api_key for marker in placeholder_markers):
            errors.append("api_key 还没有填写真实 API key。")
    if provider == "local_transformers":
        if not model_path:
            errors.append("model_path 还没有填写本地模型路径。")
        elif not Path(model_path).exists():
            errors.append(f"model_path 不存在：{model_path}")
        for module in ["torch", "transformers", "accelerate", "safetensors"]:
            try:
                __import__(module)
            except Exception:
                errors.append(f"沙盒 Python 缺少依赖：{module}")
    return errors


def _mask_secret(value: str) -> str:
    if not value:
        return "<empty>"
    if any(marker in value for marker in ["替换", "API key", "your", "xxx"]):
        return "<placeholder>"
    if len(value) <= 8:
        return "***"
    return value[:4] + "***" + value[-4:]


def _record_llm_call(run_metrics: dict[str, int] | None, lock: threading.Lock | None) -> None:
    _increment_metric(run_metrics, lock, "llm_calls")


def _increment_metric(
    run_metrics: dict[str, int] | None,
    lock: threading.Lock | None,
    name: str,
) -> None:
    if run_metrics is None:
        return
    if lock is None:
        run_metrics[name] = int(run_metrics.get(name, 0)) + 1
        return
    with lock:
        run_metrics[name] = int(run_metrics.get(name, 0)) + 1


if __name__ == "__main__":
    sys.exit(main())

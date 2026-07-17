from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import json
import mimetypes
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
from dataclasses import MISSING, fields, replace
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from config_loader import load_yaml, resolve_path
from load_rules import load_rules
from llm_client import LLMClient, LLMConfig


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
ALLOWED_ROOTS = [ROOT / "config", ROOT / "data", ROOT / "src", ROOT / "tools"]
TEXT_SUFFIXES = {".txt", ".csv", ".json", ".yaml", ".yml", ".log", ".md", ".py", ".bat", ".ps1"}
SELECTION_FILE = ROOT / "data" / "selected_patients.json"
DEFAULT_PROJECT_CONFIG = "project_full_ai_only.yaml"
DEFAULT_MODEL_CONFIG = "model_api.yaml"
EDITABLE_PROJECT_CONFIGS = [
    "project_full_ai_only.yaml",
    "project_full_ai_only_model1.yaml",
    "project_full_ai_only_model2.yaml",
]
EDITABLE_MODEL_CONFIGS = [
    "model_api.yaml",
    "model_api.example.yaml",
    "model_api_1.yaml",
    "model_api_2.yaml",
]
RUN_DIR = ROOT / "data" / "dashboard_runs"
TASK_REGISTRY_FILE = ROOT / "data" / "task_registry.json"
RULE_COUNT_CACHE: dict[str, tuple[float, int, int]] = {}
PROGRESS_CACHE: dict[str, tuple[float, float, float, dict]] = {}
ACTIVE_PROCESS_CACHE: tuple[float, dict[int, dict]] | None = None
RUN_CONTROL_LOCK = threading.Lock()
MODEL_HEALTH_PROMPT = """请只输出一个合法 JSON 对象，不要输出 Markdown：
{"items":[{"label":"未知","explanation":"连通性测试","evidence":"测试","confidence":0.1}]}
"""


def list_project_config_names() -> list[str]:
    configured = [name for name in EDITABLE_PROJECT_CONFIGS if (ROOT / "config" / name).exists()]
    discovered = sorted(p.name for p in (ROOT / "config").glob("project_*.yaml"))
    return sorted(set(configured + discovered))


def list_model_config_names() -> list[str]:
    configured = [name for name in EDITABLE_MODEL_CONFIGS if (ROOT / "config" / name).exists()]
    discovered = sorted(
        p.name
        for p in (ROOT / "config").glob("model*.yaml")
        if p.name != "model_internal_health.example.yaml"
    )
    return sorted(set(configured + discovered))


def default_model_config_name() -> str:
    configs = list_model_config_names()
    if DEFAULT_MODEL_CONFIG in configs:
        return DEFAULT_MODEL_CONFIG
    return configs[0] if configs else DEFAULT_MODEL_CONFIG


def count_rules_cached(rules_file: Path) -> int:
    path = rules_file.resolve()
    stat = path.stat()
    key = str(path)
    cached = RULE_COUNT_CACHE.get(key)
    if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
        return cached[2]
    count = len(load_rules(path))
    RULE_COUNT_CACHE[key] = (stat.st_mtime, stat.st_size, count)
    return count


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._serve_file(WEB / "index.html")
            elif parsed.path.startswith("/static/"):
                self._serve_file(WEB / parsed.path.lstrip("/"))
            elif parsed.path == "/api/configs":
                project_configs = list_project_config_names()
                model_configs = list_model_config_names()
                self._json({
                    "configs": project_configs,
                    "project_configs": project_configs,
                    "model_configs": model_configs,
                })
            elif parsed.path == "/api/config":
                name = parse_qs(parsed.query).get("name", [""])[0]
                path = self._safe_config(name)
                self._json({"name": name, "content": path.read_text(encoding="utf-8")})
            elif parsed.path == "/api/stats":
                self._json(build_stats())
            elif parsed.path == "/api/patients":
                config_name = parse_qs(parsed.query).get("config", [DEFAULT_PROJECT_CONFIG])[0]
                self._json(list_patients(config_name))
            elif parsed.path == "/api/concurrency":
                model_config = parse_qs(parsed.query).get("model_config", [""])[0]
                self._json(get_concurrency_state(model_config))
            elif parsed.path == "/api/projects":
                self._json(get_project_state())
            elif parsed.path == "/api/progress":
                mode = parse_qs(parsed.query).get("mode", ["all"])[0]
                self._json(get_progress_state(mode))
            elif parsed.path == "/api/tasks":
                self._json(get_task_state())
            elif parsed.path == "/api/checkpoints":
                self._json(get_checkpoint_state())
            elif parsed.path == "/api/checkpoint-archives":
                self._json(list_checkpoint_archives())
            elif parsed.path == "/api/runs":
                self._json(list_dashboard_runs())
            elif parsed.path == "/api/server":
                self._json(get_server_state(int(self.server.server_address[1])))
            elif parsed.path == "/api/files":
                directory = parse_qs(parsed.query).get("dir", ["."])[0]
                self._json(list_files(directory))
            elif parsed.path == "/api/file":
                rel = parse_qs(parsed.query).get("path", [""])[0]
                self._json(read_preview(rel))
            elif parsed.path == "/download":
                rel = parse_qs(parsed.query).get("path", [""])[0]
                self._serve_file(safe_path(rel), as_attachment=True)
            else:
                self.send_error(404)
        except Exception as exc:
            self._error(exc)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            if parsed.path == "/api/config":
                name = parse_qs(parsed.query).get("name", [""])[0]
                path = self._safe_config(name)
                path.write_text(str(payload.get("content", "")), encoding="utf-8")
                self._json({"ok": True, "name": name})
                return
            if parsed.path == "/api/patients":
                files = [Path(str(item)).name for item in payload.get("files", []) if str(item).strip()]
                SELECTION_FILE.parent.mkdir(parents=True, exist_ok=True)
                SELECTION_FILE.write_text(
                    json.dumps({"enabled": bool(payload.get("enabled", True)), "files": files}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                self._json({"ok": True, "selected": len(files), "path": str(SELECTION_FILE)})
                return
            if parsed.path == "/api/patient-dir":
                self._json(save_patient_dir(payload))
                return
            if parsed.path == "/api/concurrency":
                action = str(payload.get("action", "set"))
                if action == "benchmark":
                    self._json(run_concurrency_benchmark(payload))
                else:
                    self._json(save_concurrency(payload))
                return
            if parsed.path == "/api/runs":
                self._json(start_dashboard_runs(payload))
                return
            if parsed.path == "/api/new-task":
                self._json(create_new_task(payload))
                return
            if parsed.path == "/api/restore-checkpoint":
                self._json(restore_checkpoint_archive(payload))
                return
            if parsed.path == "/api/delete-checkpoint-archive":
                self._json(delete_checkpoint_archive(payload))
                return
            if parsed.path == "/api/preflight":
                self._json(preflight_dashboard_runs(payload))
                return
            if parsed.path == "/api/stop":
                self._json(stop_dashboard_runs(payload))
                return
            self.send_error(404)
        except Exception as exc:
            self._error(exc)

    def _safe_config(self, name: str) -> Path:
        if not name.endswith((".yaml", ".yml")) or "/" in name or "\\" in name:
            raise ValueError("invalid config name")
        path = (ROOT / "config" / name).resolve()
        if path.parent != (ROOT / "config").resolve():
            raise ValueError("invalid config path")
        return path

    def _json(self, data: dict) -> None:
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: Path, as_attachment: bool = False) -> None:
        path = path.resolve()
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        body = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if as_attachment:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        self.wfile.write(body)

    def _error(self, exc: Exception) -> None:
        body = str(exc).encode("utf-8", errors="replace")
        self.send_response(400)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return


def safe_path(rel: str) -> Path:
    rel = unquote(rel).replace("\\", "/").strip("/")
    path = (ROOT / rel).resolve()
    allowed = path == ROOT or any(path == base.resolve() or base.resolve() in path.parents for base in ALLOWED_ROOTS)
    if not allowed:
        raise ValueError("path not allowed")
    return path


def safe_config_name(name: str, default: str) -> str:
    clean = Path(str(name or default)).name
    if not clean.endswith((".yaml", ".yml")):
        clean = default
    path = ROOT / "config" / clean
    return clean if path.exists() else default


def list_files(directory: str) -> dict:
    path = safe_path(directory)
    if not path.exists() or not path.is_dir():
        path = ROOT
    items = []
    for child in sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        if child.name == "__pycache__":
            continue
        rel = child.relative_to(ROOT).as_posix()
        items.append({
            "name": child.name,
            "path": rel,
            "type": "dir" if child.is_dir() else "file",
            "size": child.stat().st_size if child.is_file() else "",
        })
    return {"dir": path.relative_to(ROOT).as_posix() if path != ROOT else ".", "items": items}


def read_preview(rel: str) -> dict:
    path = safe_path(rel)
    if not path.is_file():
        raise ValueError("not a file")
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return {"path": rel, "preview": f"{path.name} is not a text file. Use download to view it."}
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    return {"path": rel, "preview": text[:20000]}


def build_stats() -> dict:
    patient_info = list_patients(DEFAULT_PROJECT_CONFIG)
    try:
        rule_count = count_rules_cached(ROOT / "data" / "input" / "rules.xlsx")
    except Exception:
        rule_count = 0
    output_dirs = sorted((ROOT / "data").glob("output*"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    run_dirs = []
    for output_base in output_dirs:
        run_parent = output_base / "runs"
        if run_parent.exists():
            run_dirs.extend(path for path in run_parent.iterdir() if path.is_dir())
    summary_dirs = sorted(
        [*run_dirs, *output_dirs],
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    checkpoint_dirs = list((ROOT / "data").glob("checkpoint*"))
    runs = []
    latest_results = 0
    latest_failures = 0
    for directory in summary_dirs[:8]:
        summary_path = directory / "run_summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        summary["dir"] = directory.relative_to(ROOT).as_posix()
        summary["output_updated_at"] = datetime.fromtimestamp(directory.stat().st_mtime).isoformat(timespec="seconds")
        summary["summary_updated_at"] = datetime.fromtimestamp(summary_path.stat().st_mtime).isoformat(timespec="seconds")
        runs.append(summary)
    if runs:
        latest_results = runs[0].get("results_written_this_run", runs[0].get("results_written", 0))
        latest_failures = runs[0].get("failures_this_run", runs[0].get("failures", 0))
    return {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "patient_files": patient_info["total"],
        "selected_patient_files": patient_info["selected"],
        "rule_count": rule_count,
        "output_dirs": len(output_dirs) + len(run_dirs),
        "checkpoint_dirs": len(checkpoint_dirs),
        "latest_results": latest_results,
        "latest_failures": latest_failures,
        "runs": runs,
    }


def list_patients(config_name: str) -> dict:
    config_name = safe_config_name(config_name, DEFAULT_PROJECT_CONFIG)
    project = load_yaml(ROOT / "config" / config_name)
    patient_dir = resolve_path(ROOT, project.get("paths", {}).get("patient_dir", "data/input/patient")).resolve()
    files = list_patient_workbooks(patient_dir)
    selection = load_selection()
    selected_names = set(selection.get("files", [])) if selection.get("enabled") else set()
    items = []
    for path in files:
        items.append({
            "name": path.name,
            "path": str(path),
            "size": path.stat().st_size,
            "selected": path.name in selected_names,
        })
    return {
        "config": config_name,
        "patient_dir": str(patient_dir),
        "selection_enabled": bool(selection.get("enabled", False)),
        "total": len(items),
        "selected": len([item for item in items if item["selected"]]),
        "items": items,
    }


def list_patient_workbooks(patient_dir: Path) -> list[Path]:
    if not patient_dir.exists():
        return []
    return sorted(path for path in patient_dir.glob("*.xlsx") if not path.name.startswith("~$"))


def save_patient_dir(payload: dict) -> dict:
    project_config = safe_config_name(payload.get("project_config", DEFAULT_PROJECT_CONFIG), DEFAULT_PROJECT_CONFIG)
    raw_dir = str(payload.get("patient_dir") or "").strip().strip('"')
    if not raw_dir:
        raise ValueError("患者目录不能为空")
    path = resolve_path(ROOT, raw_dir).resolve()
    if not path.exists() or not path.is_dir():
        raise ValueError(f"患者目录不存在：{path}")
    files = list_patient_workbooks(path)
    if not files:
        raise ValueError(f"患者目录中没有 xlsx 文件：{path}")
    value = path.as_posix()
    try:
        rel = path.relative_to(ROOT).as_posix()
        value = rel
    except ValueError:
        pass
    update_yaml_scalar(ROOT / "config" / project_config, ["paths", "patient_dir"], value)
    return {"ok": True, "project_config": project_config, "patient_dir": str(path), "total": len(files)}


def load_selection() -> dict:
    if not SELECTION_FILE.exists():
        return {"enabled": False, "files": []}
    try:
        payload = json.loads(SELECTION_FILE.read_text(encoding="utf-8-sig"))
    except Exception:
        return {"enabled": False, "files": []}
    files = [Path(str(item)).name for item in payload.get("files", []) if str(item).strip()]
    return {"enabled": bool(payload.get("enabled", False)), "files": files}


def get_concurrency_state(model_config_name: str = "") -> dict:
    project = load_yaml(ROOT / "config" / DEFAULT_PROJECT_CONFIG)
    model_config = safe_config_name(model_config_name, default_model_config_name())
    model = load_yaml(ROOT / "config" / model_config)
    run_cfg = project.get("run", {})
    labeling_cfg = project.get("labeling", {})
    return {
        "project_config": DEFAULT_PROJECT_CONFIG,
        "model_config": model_config,
        "run_concurrency": int(run_cfg.get("concurrency", 1) or 1),
        "rule_concurrency": int(run_cfg.get("rule_concurrency", 1) or 1),
        "max_consecutive_llm_failures": int(run_cfg.get("max_consecutive_llm_failures", 5) or 0),
        "write_xlsx": bool(run_cfg.get("write_xlsx", False)),
        "group_rules_for_llm": bool(labeling_cfg.get("group_rules_for_llm", False)),
        "llm_batch_size": int(labeling_cfg.get("llm_batch_size", 1) or 1),
        "model_concurrency": int(model.get("concurrency", 1) or 1),
        "model_max_tokens": int(model.get("max_tokens", 1024) or 1024),
        "model_max_retries": int(model.get("max_retries", 2) or 2),
        "latest_benchmark": latest_benchmark_summary(),
    }


def save_concurrency(payload: dict) -> dict:
    value = max(1, min(int(payload.get("concurrency", 1) or 1), 64))
    rule_value = max(1, min(int(payload.get("rule_concurrency", 1) or 1), 64))
    failure_limit = max(0, min(int(payload.get("max_consecutive_llm_failures", 5) or 0), 100))
    max_tokens = max(128, min(int(payload.get("max_tokens", 1024) or 1024), 4096))
    write_xlsx = bool(payload.get("write_xlsx", False))
    group_rules = bool(payload.get("group_rules_for_llm", False))
    batch_size = max(1, min(int(payload.get("llm_batch_size", 5) or 1), 50))
    normal_batch_size = max(1, min(batch_size, 5))
    complex_batch_size = max(1, min(batch_size, 3))
    complexity_budget = max(8, min(48, normal_batch_size * 6 - 2))
    project_config = safe_config_name(payload.get("project_config", DEFAULT_PROJECT_CONFIG), DEFAULT_PROJECT_CONFIG)
    model_config = safe_config_name(payload.get("model_config", default_model_config_name()), default_model_config_name())
    update_yaml_scalar(ROOT / "config" / project_config, ["run", "concurrency"], value)
    update_yaml_scalar(ROOT / "config" / project_config, ["run", "rule_concurrency"], rule_value)
    update_yaml_scalar(ROOT / "config" / project_config, ["run", "max_consecutive_llm_failures"], failure_limit)
    update_yaml_scalar(ROOT / "config" / project_config, ["run", "write_xlsx"], write_xlsx)
    update_yaml_scalar(ROOT / "config" / project_config, ["labeling", "group_rules_for_llm"], group_rules)
    update_yaml_scalar(ROOT / "config" / project_config, ["labeling", "llm_batch_size"], batch_size)
    update_yaml_scalar(ROOT / "config" / project_config, ["optimization", "max_batch_rules"], batch_size)
    update_yaml_scalar(ROOT / "config" / project_config, ["optimization", "max_simple_batch_rules"], batch_size)
    update_yaml_scalar(ROOT / "config" / project_config, ["optimization", "max_normal_batch_rules"], normal_batch_size)
    update_yaml_scalar(ROOT / "config" / project_config, ["optimization", "max_complex_batch_rules"], complex_batch_size)
    update_yaml_scalar(ROOT / "config" / project_config, ["optimization", "batch_complexity_budget"], complexity_budget)
    update_yaml_scalar(ROOT / "config" / model_config, ["concurrency"], rule_value)
    update_yaml_scalar(ROOT / "config" / model_config, ["max_tokens"], max_tokens)
    return {
        "ok": True,
        "concurrency": value,
        "rule_concurrency": rule_value,
        "max_consecutive_llm_failures": failure_limit,
        "write_xlsx": write_xlsx,
        "group_rules_for_llm": group_rules,
        "llm_batch_size": batch_size,
        "max_simple_batch_rules": batch_size,
        "max_normal_batch_rules": normal_batch_size,
        "max_complex_batch_rules": complex_batch_size,
        "batch_complexity_budget": complexity_budget,
        "max_tokens": max_tokens,
        "project_config": project_config,
        "model_config": model_config,
    }


def run_concurrency_benchmark(payload: dict) -> dict:
    concurrency = max(1, min(int(payload.get("concurrency", 1) or 1), 64))
    requests = max(1, min(int(payload.get("requests", 4) or 4), 200))
    prompt_mode = str(payload.get("prompt_mode", "short"))
    if prompt_mode not in {"short", "realish"}:
        prompt_mode = "short"
    model_config = safe_config_name(payload.get("model_config", default_model_config_name()), default_model_config_name())
    cmd = [
        sys.executable,
        str(ROOT / "src" / "benchmark_concurrency.py"),
        "--model-config",
        str(ROOT / "config" / model_config),
        "--concurrency",
        str(concurrency),
        "--requests",
        str(requests),
        "--prompt-mode",
        prompt_mode,
    ]
    started = datetime.now().isoformat(timespec="seconds")
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(120, requests * 240),
    )
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "started_at": started,
        "stdout": proc.stdout[-5000:],
        "stderr": proc.stderr[-3000:],
        "latest_benchmark": latest_benchmark_summary(),
    }


def latest_benchmark_summary() -> dict | None:
    out_dir = ROOT / "data" / "output_concurrency_benchmark"
    if not out_dir.exists():
        return None
    files = sorted(out_dir.glob("concurrency_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["file"] = path.relative_to(ROOT).as_posix()
            return data
        except Exception:
            continue
    return None


def get_project_state() -> dict:
    project_configs = list_project_config_names()
    model_configs = list_model_config_names()
    default_model_config = default_model_config_name()
    single_pairs = [{
        "name": "单模型",
        "project_config": DEFAULT_PROJECT_CONFIG,
        "model_config": default_model_config,
        "enabled": True,
        "mode": "single",
        "ready": model_config_ready(default_model_config),
        "note": "默认正式任务",
    }]
    dual_pairs = []
    for idx in (1, 2):
        project_name = f"project_full_ai_only_model{idx}.yaml"
        model_name = f"model_api_{idx}.yaml"
        if (ROOT / "config" / model_name).exists():
            dual_pairs.append({
                "name": f"模型{idx}",
                "project_config": project_name if (ROOT / "config" / project_name).exists() else DEFAULT_PROJECT_CONFIG,
                "model_config": model_name,
                "enabled": True,
                "mode": "dual",
                "ready": model_config_ready(model_name),
                "note": "双模型任务",
            })
    return {
        "project_configs": project_configs,
        "model_configs": model_configs,
        "pairs": single_pairs,
        "single_pairs": single_pairs,
        "dual_pairs": dual_pairs,
        "runs": list_dashboard_runs()["runs"],
    }


def get_server_state(port: int) -> dict:
    processes = find_dashboard_processes(port)
    return {
        "pid": os.getpid(),
        "root": str(ROOT),
        "package": ROOT.name,
        "port": port,
        "dashboard_process_count": len(processes),
        "other_dashboard_process_count": len([item for item in processes if item.get("pid") != os.getpid()]),
        "dashboard_processes": processes[:12],
    }


def find_dashboard_processes(port: int) -> list[dict]:
    return [{"pid": os.getpid(), "package": ROOT.name, "current": True}]


def infer_package_from_command(command: str) -> str:
    for name in ("annotation_sandbox_api_only", "annotation_sandbox_full_local", "annotation_sandbox"):
        if name in command:
            return name
    return "unknown"


def model_config_ready(name: str) -> bool:
    try:
        data = load_yaml(ROOT / "config" / name)
    except Exception:
        return False
    text = " ".join(str(data.get(key, "")) for key in ("base_url", "api_key", "model_name"))
    placeholders = ("SERVER_C_IP", "REPLACE_WITH_API_KEY", "model_1", "model_2")
    return not any(marker in text for marker in placeholders)


def get_progress_state(mode: str = "all") -> dict:
    mode = mode if mode in {"single", "dual", "all"} else "all"
    project_state = get_project_state()
    if mode == "single":
        pairs = project_state.get("single_pairs", [])
    elif mode == "dual":
        pairs = project_state.get("dual_pairs", [])
    else:
        pairs = project_state.get("single_pairs", []) + project_state.get("dual_pairs", [])
    projects = []
    for pair in pairs:
        project_config = pair.get("project_config")
        if project_config not in projects:
            projects.append(project_config)
    rows = [project_progress(project_config) for project_config in projects]
    return {"updated_at": datetime.now().isoformat(timespec="seconds"), "items": rows}


def get_checkpoint_state() -> dict:
    items = []
    project_state = get_project_state()
    projects = []
    for pair in project_state.get("single_pairs", []) + project_state.get("dual_pairs", []):
        project_config = pair.get("project_config")
        if project_config and project_config not in projects:
            projects.append(project_config)
    for project_config in projects:
        item = project_progress(project_config)
        checkpoint_db = resolve_path(ROOT, load_yaml(ROOT / "config" / project_config).get("paths", {}).get("checkpoint_db", ""))
        item["failed_patient_rows"] = []
        item["failed_task_rows"] = []
        if checkpoint_db.exists():
            try:
                conn = sqlite3.connect(f"file:{checkpoint_db}?mode=ro", uri=True)
                item["failed_patient_rows"] = [
                    {
                        "patient_sn": str(row[0]),
                        "source_file": str(row[1] or ""),
                        "status": str(row[2]),
                        "done_rules": int(row[3] or 0),
                        "failed_rules": int(row[4] or 0),
                        "last_error": str(row[5] or ""),
                        "updated_at": str(row[6] or ""),
                    }
                    for row in conn.execute(
                        """
                        select patient_sn, source_file, status, done_rules, failed_rules, last_error, updated_at
                        from patient_status
                        where status in ('failed', 'partial') or failed_rules > 0
                        order by updated_at desc
                        limit 100
                        """
                    ).fetchall()
                ]
                item["failed_task_rows"] = [
                    {
                        "patient_sn": str(row[0]),
                        "trial_id": str(row[1]),
                        "standard_no": str(row[2]),
                        "error": str(row[3] or ""),
                        "updated_at": str(row[4] or ""),
                    }
                    for row in conn.execute(
                        """
                        select patient_sn, trial_id, standard_no, error, updated_at
                        from progress
                        where status='failed'
                        order by updated_at desc
                        limit 100
                        """
                    ).fetchall()
                ]
                conn.close()
            except Exception as exc:
                item["checkpoint_error"] = str(exc)
        items.append(item)
    return {"updated_at": datetime.now().isoformat(timespec="seconds"), "items": items}


def load_task_registry() -> dict:
    if not TASK_REGISTRY_FILE.exists():
        return {"current": {}, "archives": {}}
    try:
        data = json.loads(TASK_REGISTRY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"current": {}, "archives": {}}
    if not isinstance(data, dict):
        return {"current": {}, "archives": {}}
    data.setdefault("current", {})
    data.setdefault("archives", {})
    return data


def save_task_registry(data: dict) -> None:
    TASK_REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    TASK_REGISTRY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def chinese_number(value: int) -> str:
    digits = "零一二三四五六七八九"
    if value <= 0:
        return str(value)
    if value < 10:
        return digits[value]
    if value == 10:
        return "十"
    if value < 20:
        return "十" + digits[value % 10]
    if value < 100:
        ones = value % 10
        return digits[value // 10] + "十" + (digits[ones] if ones else "")
    return str(value)


def default_task_name(index: int) -> str:
    return f"任务{chinese_number(index)}"


def used_task_names(registry: dict | None = None) -> set[str]:
    registry = registry or load_task_registry()
    names = set()
    for group in ("current", "archives"):
        values = registry.get(group, {})
        if isinstance(values, dict):
            names.update(str(value) for value in values.values() if str(value).strip())
    archive_root = ROOT / "data" / "checkpoint_archives"
    if archive_root.exists():
        for manifest_path in archive_root.glob("*/*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            task_name = str(manifest.get("task_name") or "").strip()
            if task_name:
                names.add(task_name)
    return names


def next_default_task_name(registry: dict | None = None) -> str:
    names = used_task_names(registry)
    index = 1
    while default_task_name(index) in names:
        index += 1
    return default_task_name(index)


def current_task_name(project_config: str, create: bool = True) -> str:
    safe_project = safe_config_name(project_config, DEFAULT_PROJECT_CONFIG)
    registry = load_task_registry()
    current = registry.setdefault("current", {})
    task_name = str(current.get(safe_project) or "").strip()
    if not task_name and create:
        task_name = next_default_task_name(registry)
        current[safe_project] = task_name
        save_task_registry(registry)
    return task_name


def set_current_task_name(project_config: str, task_name: str) -> None:
    safe_project = safe_config_name(project_config, DEFAULT_PROJECT_CONFIG)
    registry = load_task_registry()
    registry.setdefault("current", {})[safe_project] = task_name.strip() or next_default_task_name(registry)
    save_task_registry(registry)


def clear_current_task_name(project_config: str) -> None:
    safe_project = safe_config_name(project_config, DEFAULT_PROJECT_CONFIG)
    registry = load_task_registry()
    current = registry.setdefault("current", {})
    if safe_project in current:
        current.pop(safe_project, None)
        save_task_registry(registry)


def archive_task_name(archive_rel: str, manifest: dict | None = None) -> str:
    registry = load_task_registry()
    archives = registry.setdefault("archives", {})
    task_name = ""
    if manifest:
        task_name = str(manifest.get("task_name") or "").strip()
    task_name = task_name or str(archives.get(archive_rel) or "").strip()
    if not task_name:
        task_name = next_default_task_name(registry)
    archives[archive_rel] = task_name
    save_task_registry(registry)
    return task_name


def current_display_task_name(project_config: str, progress: dict, archive_rows: list[dict]) -> str:
    existing = current_task_name(project_config, create=False)
    processed = int(progress.get("processed_tasks") or 0)
    total = int(progress.get("total_tasks") or 0)
    if processed > 0:
        for row in archive_rows:
            if row.get("project_config") != project_config:
                continue
            if int(row.get("processed_tasks") or 0) == processed and int(row.get("total_tasks") or 0) == total:
                task_name = str(row.get("task_name") or "").strip()
                if task_name:
                    set_current_task_name(project_config, task_name)
                    return task_name
    if existing:
        return existing
    if processed > 0 or bool(progress.get("active")):
        return current_task_name(project_config)
    return ""


def get_task_state() -> dict:
    items: list[dict] = []
    archive_rows = list_checkpoint_archives().get("items", [])
    seen_projects: set[str] = set()
    current_signatures: set[tuple[str, int, int]] = set()
    archive_signatures: set[tuple[str, str, int, str]] = set()
    project_state = get_project_state()
    for pair in project_state.get("single_pairs", []) + project_state.get("dual_pairs", []):
        project_config = safe_config_name(pair.get("project_config"), DEFAULT_PROJECT_CONFIG)
        if project_config in seen_projects:
            continue
        seen_projects.add(project_config)
        progress = project_progress(project_config)
        task_name = current_display_task_name(project_config, progress, archive_rows)
        if not task_name and not progress.get("active") and not progress.get("processed_tasks"):
            clear_current_task_name(project_config)
            continue
        current_signatures.add((
            project_config,
            int(progress.get("processed_tasks") or 0),
            int(progress.get("total_tasks") or 0),
        ))
        items.append({
            "kind": "current",
            "task_name": task_name,
            "project_config": project_config,
            "active": bool(progress.get("active")),
            **progress,
        })
    for row in archive_rows:
        signature = (
            str(row.get("project_config") or ""),
            int(row.get("processed_tasks") or 0),
            int(row.get("total_tasks") or 0),
        )
        if signature in current_signatures:
            continue
        archive_signature = (
            str(row.get("project_config") or ""),
            str(row.get("task_name") or ""),
            int(row.get("processed_tasks") or 0),
            str(row.get("output_dir") or ""),
        )
        if archive_signature in archive_signatures:
            continue
        archive_signatures.add(archive_signature)
        items.append({
            "kind": "archive",
            "task_name": str(row.get("task_name") or archive_task_name(str(row.get("path") or ""), row)),
            "active": False,
            **row,
        })
    return {"updated_at": datetime.now().isoformat(timespec="seconds"), "items": items}


def checkpoint_paths_for_project(project_config: str) -> list[Path]:
    project_config = safe_config_name(project_config, DEFAULT_PROJECT_CONFIG)
    project = load_yaml(ROOT / "config" / project_config)
    checkpoint_db = resolve_path(ROOT, project.get("paths", {}).get("checkpoint_db", "data/checkpoint/progress.sqlite"))
    candidates = [checkpoint_db, Path(str(checkpoint_db) + "-wal"), Path(str(checkpoint_db) + "-shm")]
    return [path.resolve() for path in candidates]


def archive_checkpoints_for_projects(project_configs: list[str], reason: str = "new_task") -> list[dict]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archived: list[dict] = []
    for project_config in sorted(set(project_configs)):
        safe_project = safe_config_name(project_config, DEFAULT_PROJECT_CONFIG)
        task_name = current_task_name(safe_project, create=False)
        archive_dir = ROOT / "data" / "checkpoint_archives" / Path(safe_project).stem / timestamp
        moved: list[str] = []
        missing: list[str] = []
        archive_dir.mkdir(parents=True, exist_ok=True)
        for path in checkpoint_paths_for_project(safe_project):
            try:
                path.relative_to(ROOT)
            except ValueError:
                raise ValueError(f"checkpoint path is outside sandbox: {path}")
            if path.exists():
                target = archive_dir / path.name
                shutil.move(str(path), str(target))
                moved.append(target.relative_to(ROOT).as_posix())
            else:
                missing.append(path.relative_to(ROOT).as_posix())
        if not moved:
            try:
                archive_dir.rmdir()
            except OSError:
                pass
            archived.append({
                "project_config": safe_project,
                "task_name": task_name,
                "archived_at": datetime.now().isoformat(timespec="seconds"),
                "reason": reason,
                "moved": moved,
                "missing": missing,
                "skipped": "no_checkpoint_files",
            })
            continue
        if not task_name:
            task_name = next_default_task_name()
            set_current_task_name(safe_project, task_name)
        manifest = {
            "project_config": safe_project,
            "task_name": task_name,
            "archived_at": datetime.now().isoformat(timespec="seconds"),
            "reason": reason,
            "moved": moved,
            "missing": missing,
        }
        (archive_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        archive_task_name(archive_dir.relative_to(ROOT).as_posix(), manifest)
        archived.append(manifest)
    return archived


def list_checkpoint_archives() -> dict:
    archive_root = ROOT / "data" / "checkpoint_archives"
    rows: list[dict] = []
    if archive_root.exists():
        for manifest_path in archive_root.glob("*/*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            archive_dir = manifest_path.parent
            project_config = str(manifest.get("project_config") or archive_dir.parent.name)
            progress = checkpoint_archive_progress(project_config, archive_dir / "progress.sqlite")
            archive_rel = archive_dir.relative_to(ROOT).as_posix()
            task_name = archive_task_name(archive_rel, manifest)
            rows.append({
                "project_config": project_config,
                "task_name": task_name,
                "archived_at": str(manifest.get("archived_at") or archive_dir.name),
                "reason": str(manifest.get("reason") or ""),
                "path": archive_rel,
                "moved": manifest.get("moved", []),
                "file_count": len([path for path in archive_dir.iterdir() if path.is_file() and path.name != "manifest.json"]),
                **progress,
            })
    rows.sort(key=lambda item: str(item.get("archived_at") or ""), reverse=True)
    return {"updated_at": datetime.now().isoformat(timespec="seconds"), "items": rows}


def checkpoint_archive_progress(project_config: str, db_path: Path) -> dict:
    if not db_path.exists():
        return {"done_tasks": 0, "failed_tasks": 0, "processed_tasks": 0, "total_tasks": 0, "percent": 0}
    counts: dict[str, int] = {}
    meta: dict[str, str] = {}
    patient_rows = 0
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        if conn.execute("select name from sqlite_master where type='table' and name='progress'").fetchone():
            counts = {str(status): int(count) for status, count in conn.execute("select status, count(*) from progress group by status").fetchall()}
        if conn.execute("select name from sqlite_master where type='table' and name='patient_status'").fetchone():
            patient_rows = int(conn.execute("select count(*) from patient_status").fetchone()[0] or 0)
        if conn.execute("select name from sqlite_master where type='table' and name='run_meta'").fetchone():
            meta = {str(key): str(value) for key, value in conn.execute("select key, value from run_meta").fetchall()}
        conn.close()
    except Exception as exc:
        return {"done_tasks": 0, "failed_tasks": 0, "processed_tasks": 0, "total_tasks": 0, "percent": 0, "archive_error": str(exc)}

    total_tasks = 0
    try:
        project = load_yaml(ROOT / "config" / project_config)
        paths = project.get("paths", {})
        rules_file = resolve_path(ROOT, paths.get("rules_file", "data/input/rules.xlsx"))
        patient_dir = resolve_path(ROOT, paths.get("patient_dir", "data/input/patient"))
        patient_count = len(list_patient_workbooks(patient_dir))
        if not patient_count and patient_rows:
            patient_count = patient_rows
        total_tasks = count_rules_cached(rules_file) * patient_count
    except Exception:
        total_tasks = 0
    done = int(counts.get("done", 0))
    failed = int(counts.get("failed", 0))
    processed = done + failed
    percent = round(processed / total_tasks * 100, 2) if total_tasks else 0
    output_dir = meta.get("active_output_dir", "")
    try:
        output_dir_display = Path(output_dir).relative_to(ROOT).as_posix() if output_dir else ""
    except ValueError:
        output_dir_display = output_dir
    return {
        "run_id": meta.get("active_run_id", ""),
        "output_dir": output_dir_display,
        "done_tasks": done,
        "failed_tasks": failed,
        "processed_tasks": processed,
        "total_tasks": total_tasks,
        "percent": percent,
        "patient_status_rows": patient_rows,
        "task_counts": counts,
    }


def restore_checkpoint_archive(payload: dict) -> dict:
    archive_rel = str(payload.get("archive_path") or "").strip()
    if not archive_rel:
        raise ValueError("archive_path is required")
    archive_dir = safe_path(archive_rel)
    archive_root = (ROOT / "data" / "checkpoint_archives").resolve()
    if not archive_dir.is_dir() or not (archive_dir == archive_root or archive_root in archive_dir.parents):
        raise ValueError(f"invalid checkpoint archive path: {archive_rel}")
    manifest_path = archive_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"checkpoint archive manifest not found: {archive_rel}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    project_config = safe_config_name(manifest.get("project_config"), DEFAULT_PROJECT_CONFIG)
    if not (archive_dir / "progress.sqlite").exists():
        raise ValueError(f"checkpoint archive has no progress.sqlite: {archive_rel}")

    active_processes = active_process_map()
    active_runs = [
        run for run in load_run_records()
        if run.get("project_config") == project_config and run_is_active(run, active_processes)
    ]
    if active_runs:
        raise ValueError(f"cannot restore while project is running: {project_config}")

    current_backup = archive_checkpoints_for_projects([project_config], reason="restore_current_before_history")
    restored_task_name = archive_task_name(archive_dir.relative_to(ROOT).as_posix(), manifest)
    restored: list[str] = []
    missing: list[str] = []
    for dest in checkpoint_paths_for_project(project_config):
        source = archive_dir / dest.name
        try:
            dest.relative_to(ROOT)
        except ValueError:
            raise ValueError(f"checkpoint path is outside sandbox: {dest}")
        if source.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            restored.append(dest.relative_to(ROOT).as_posix())
        else:
            missing.append(source.relative_to(ROOT).as_posix())
    set_current_task_name(project_config, restored_task_name)
    return {
        "ok": True,
        "project_config": project_config,
        "task_name": restored_task_name,
        "archive_path": archive_dir.relative_to(ROOT).as_posix(),
        "restored": restored,
        "missing": missing,
        "current_backup": current_backup,
        "restored_at": datetime.now().isoformat(timespec="seconds"),
    }


def delete_checkpoint_archive(payload: dict) -> dict:
    archive_rel = str(payload.get("archive_path") or "").strip()
    if not archive_rel:
        raise ValueError("archive_path is required")
    archive_dir = safe_path(archive_rel)
    archive_root = (ROOT / "data" / "checkpoint_archives").resolve()
    if not archive_dir.is_dir() or archive_root not in archive_dir.parents:
        raise ValueError(f"invalid checkpoint archive path: {archive_rel}")
    manifest_path = archive_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"checkpoint archive manifest not found: {archive_rel}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    project_config = safe_config_name(manifest.get("project_config"), DEFAULT_PROJECT_CONFIG)

    active_processes = active_process_map()
    active_runs = [
        run for run in load_run_records()
        if run.get("project_config") == project_config and run_is_active(run, active_processes)
    ]
    if active_runs:
        raise ValueError(f"cannot delete archive while project is running: {project_config}")

    archive_relative = archive_dir.relative_to(ROOT).as_posix()
    task_name = str(manifest.get("task_name") or "").strip()
    shutil.rmtree(archive_dir)
    try:
        archive_dir.parent.rmdir()
    except OSError:
        pass

    registry = load_task_registry()
    archives = registry.setdefault("archives", {})
    archives.pop(archive_relative, None)
    save_task_registry(registry)
    return {
        "ok": True,
        "project_config": project_config,
        "task_name": task_name,
        "deleted_archive": archive_relative,
        "deleted_at": datetime.now().isoformat(timespec="seconds"),
    }


def normalize_project_names_from_pairs(payload: dict) -> list[str]:
    requested = payload.get("pairs", [])
    if not isinstance(requested, list) or not requested:
        requested = get_project_state()["pairs"]
    project_names: list[str] = []
    for pair in requested[:4]:
        project_config = safe_config_name(pair.get("project_config"), DEFAULT_PROJECT_CONFIG)
        if project_config not in project_names:
            project_names.append(project_config)
    return project_names


def create_new_task(payload: dict) -> dict:
    project_names = normalize_project_names_from_pairs(payload)
    if not project_names:
        raise ValueError("no project configs selected")
    existing_runs = list_dashboard_runs().get("runs", [])
    active_projects = [
        run.get("project_config")
        for run in existing_runs
        if run.get("active") and run.get("project_config") in project_names
    ]
    if active_projects:
        raise ValueError("cannot create a new task while selected projects are running: " + ", ".join(sorted(set(active_projects))))
    archived = archive_checkpoints_for_projects(project_names, reason="new_task")
    task_name = str(payload.get("task_name") or "").strip()
    registry = load_task_registry()
    if not task_name:
        task_name = next_default_task_name(registry)
    for project_name in project_names:
        set_current_task_name(project_name, task_name)
    return {
        "ok": True,
        "task_name": task_name,
        "project_configs": project_names,
        "archived_checkpoints": archived,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def active_output_dir_from_checkpoint(checkpoint_db: Path, fallback: Path) -> Path:
    if not checkpoint_db.exists():
        return fallback
    try:
        conn = sqlite3.connect(f"file:{checkpoint_db}?mode=ro", uri=True)
        table = conn.execute(
            "select name from sqlite_master where type='table' and name='run_meta'"
        ).fetchone()
        if not table:
            conn.close()
            return fallback
        row = conn.execute("select value from run_meta where key='active_output_dir'").fetchone()
        conn.close()
        if row and row[0]:
            return Path(str(row[0]))
    except Exception:
        return fallback
    return fallback


def project_progress(project_config: str) -> dict:
    project_config = safe_config_name(project_config, DEFAULT_PROJECT_CONFIG)
    project = load_yaml(ROOT / "config" / project_config)
    paths = project.get("paths", {})
    checkpoint_db = resolve_path(ROOT, paths.get("checkpoint_db", "data/checkpoint/progress.sqlite"))
    checkpoint_mtime = checkpoint_db.stat().st_mtime if checkpoint_db.exists() else 0.0
    selection_mtime = SELECTION_FILE.stat().st_mtime if SELECTION_FILE.exists() else 0.0
    now_ts = datetime.now().timestamp()
    cached = PROGRESS_CACHE.get(project_config)
    if cached and now_ts - cached[0] <= 10 and cached[1] == checkpoint_mtime and cached[2] == selection_mtime:
        return copy.deepcopy(cached[3])
    patient_info = list_patients(project_config)
    selected_enabled = bool(patient_info.get("selection_enabled"))
    total_patients = int(patient_info.get("selected") if selected_enabled else patient_info.get("total") or 0)
    rules_file = resolve_path(ROOT, paths.get("rules_file", "data/input/rules.xlsx"))
    try:
        total_rules = count_rules_cached(rules_file)
    except Exception:
        total_rules = 0
    total_tasks = total_patients * total_rules
    base_output_dir = resolve_path(ROOT, paths.get("output_dir", "data/output"))
    output_dir = active_output_dir_from_checkpoint(checkpoint_db, base_output_dir)
    task_counts: dict[str, int] = {}
    patient_counts: dict[str, int] = {}
    latest_updated = ""
    allowed_files = [
        item["name"]
        for item in patient_info.get("items", [])
        if item.get("selected") or not selected_enabled
    ]
    allowed_stems = {Path(name).stem for name in allowed_files}
    if checkpoint_db.exists():
        try:
            conn = sqlite3.connect(f"file:{checkpoint_db}?mode=ro", uri=True)
            status_columns = {row[1] for row in conn.execute("pragma table_info(patient_status)").fetchall()}
            elapsed_expr = "elapsed_seconds" if "elapsed_seconds" in status_columns else "0 as elapsed_seconds"
            started_expr = "started_at" if "started_at" in status_columns else "'' as started_at"
            patient_rows = conn.execute(
                f"select patient_sn, source_file, status, total_rules, done_rules, failed_rules, last_error, updated_at, {elapsed_expr}, {started_expr} from patient_status"
            ).fetchall()
            if allowed_files or allowed_stems:
                scoped_patient_rows = [
                    row for row in patient_rows
                    if str(row[0]) in allowed_stems or Path(str(row[1] or "")).name in allowed_files
                ]
            else:
                scoped_patient_rows = list(patient_rows)
                total_patients = len(scoped_patient_rows)
                total_tasks = total_patients * total_rules
            allowed_patient_sns = sorted({str(row[0]) for row in scoped_patient_rows} | allowed_stems)
            if allowed_patient_sns:
                placeholders = ",".join("?" for _ in allowed_patient_sns)
                task_counts = {
                    str(status): int(count)
                    for status, count in conn.execute(
                        f"select status, count(*) from progress where patient_sn in ({placeholders}) group by status",
                        allowed_patient_sns,
                    ).fetchall()
                }
                row = conn.execute(
                    f"""
                    select max(updated_at) from (
                      select updated_at from progress where patient_sn in ({placeholders})
                      union all
                      select updated_at from patient_status where patient_sn in ({placeholders})
                    )
                    """,
                    allowed_patient_sns + allowed_patient_sns,
                ).fetchone()
            else:
                row = None
            for row_item in scoped_patient_rows:
                status = str(row_item[2])
                patient_counts[status] = patient_counts.get(status, 0) + 1
            completed_elapsed = [
                float(row_item[8] or 0)
                for row_item in scoped_patient_rows
                if str(row_item[2]) == "done" and float(row_item[8] or 0) > 0
            ]
            avg_patient_elapsed = round(sum(completed_elapsed) / len(completed_elapsed), 2) if completed_elapsed else 0
            latest_updated = str(row[0] or "") if row else ""
            recent_patients = [
                {
                    "patient_sn": str(row[0]),
                    "source_file": str(row[1]),
                    "status": str(row[2]),
                    "total_rules": int(row[3] or 0),
                    "done_rules": int(row[4] or 0),
                    "failed_rules": int(row[5] or 0),
                    "last_error": str(row[6] or ""),
                    "updated_at": str(row[7] or ""),
                    "elapsed_seconds": round(float(row[8] or 0), 2),
                    "started_at": str(row[9] or ""),
                }
                for row in sorted(scoped_patient_rows, key=lambda item: str(item[7] or ""), reverse=True)[:20]
            ]
            conn.close()
        except Exception:
            task_counts = {}
            patient_counts = {}
            recent_patients = []
            avg_patient_elapsed = 0
    else:
        recent_patients = []
        avg_patient_elapsed = 0
    done = int(task_counts.get("done", 0))
    failed = int(task_counts.get("failed", 0))
    processed = min(done + failed, total_tasks) if total_tasks else done + failed
    percent = round(processed / total_tasks * 100, 2) if total_tasks else 0
    summary_path = output_dir / "run_summary.json"
    has_summary = summary_path.exists()
    run_summary: dict = {}
    if has_summary:
        try:
            run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            run_summary = {}
    patient_timing = run_summary.get("patient_timing") if isinstance(run_summary.get("patient_timing"), dict) else {}
    server_stability = run_summary.get("server_stability") if isinstance(run_summary.get("server_stability"), dict) else {}
    avg_wall_seconds_per_patient = float(
        patient_timing.get("avg_wall_seconds_per_patient")
        or run_summary.get("avg_wall_seconds_per_patient")
        or 0
    )
    observed_patient_count = int(
        patient_timing.get("observed_patient_count")
        or patient_timing.get("done_patient_count")
        or patient_counts.get("done", 0) + patient_counts.get("partial", 0)
        or 0
    )
    total_wall_seconds = float(patient_timing.get("total_wall_seconds") or run_summary.get("elapsed_seconds") or 0)
    if total_wall_seconds and observed_patient_count > 0:
        avg_wall_seconds_per_patient = total_wall_seconds / observed_patient_count
    avg_pure_patient_elapsed = float(
        patient_timing.get("avg_pure_patient_seconds_observed")
        or run_summary.get("avg_pure_patient_elapsed_seconds")
        or avg_patient_elapsed
        or 0
    )
    dashboard_runs = list_dashboard_runs().get("runs", [])
    active_run = next((run for run in dashboard_runs if run.get("active") and run.get("project_config") == project_config), None)
    active = active_run is not None
    if active_run and total_patients and not avg_wall_seconds_per_patient:
        try:
            started_at = datetime.fromisoformat(str(active_run.get("started_at") or ""))
            avg_wall_seconds_per_patient = (datetime.now() - started_at).total_seconds() / total_patients
        except Exception:
            pass
    state = "not_started"
    if total_tasks and processed >= total_tasks:
        state = "complete" if failed == 0 else "complete_with_failures"
    elif active:
        state = "running"
    elif processed > 0 or has_summary:
        state = "paused_or_stopped"
    result = {
        "project_config": project_config,
        "checkpoint_db": checkpoint_db.relative_to(ROOT).as_posix() if checkpoint_db.exists() else checkpoint_db.as_posix(),
        "output_dir": output_dir.relative_to(ROOT).as_posix() if output_dir.exists() else output_dir.as_posix(),
        "state": state,
        "active": active,
        "patient_dir": patient_info.get("patient_dir", ""),
        "selection_enabled": selected_enabled,
        "selected_patient_files": [item["name"] for item in patient_info.get("items", []) if item.get("selected")][:20],
        "total_patients": total_patients,
        "total_rules": total_rules,
        "total_tasks": total_tasks,
        "processed_tasks": processed,
        "done_tasks": done,
        "failed_tasks": failed,
        "percent": percent,
        "patient_counts": patient_counts,
        "avg_patient_elapsed_seconds": avg_patient_elapsed,
        "avg_wall_seconds_per_patient": round(avg_wall_seconds_per_patient, 2),
        "avg_pure_patient_elapsed_seconds": round(avg_pure_patient_elapsed, 2),
        "patient_timing": patient_timing,
        "server_stability": server_stability,
        "task_counts": task_counts,
        "recent_patients": recent_patients,
        "latest_updated": latest_updated,
    }
    PROGRESS_CACHE[project_config] = (now_ts, checkpoint_mtime, selection_mtime, copy.deepcopy(result))
    return result


def active_process_map() -> dict[int, dict]:
    global ACTIVE_PROCESS_CACHE
    now_ts = datetime.now().timestamp()
    if ACTIVE_PROCESS_CACHE and now_ts - ACTIVE_PROCESS_CACHE[0] <= 10:
        return ACTIVE_PROCESS_CACHE[1]
    try:
        proc = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process | "
                "Select-Object ProcessId,Name,ExecutablePath,CommandLine,CreationDate | "
                "ConvertTo-Json -Compress",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        if proc.returncode != 0:
            return {}
        raw = json.loads(proc.stdout or "[]")
        items = raw if isinstance(raw, list) else [raw]
        processes: dict[int, dict] = {}
        for item in items:
            try:
                pid = int(item.get("ProcessId") or item.get("Id") or 0)
            except Exception:
                continue
            if pid > 0:
                processes[pid] = item
        ACTIVE_PROCESS_CACHE = (now_ts, processes)
        return processes
    except Exception:
        return {}


def active_pid_set() -> set[int]:
    return set(active_process_map())


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except Exception:
            return False


def run_is_active(run: dict, active_processes: dict[int, dict] | set[int] | None = None) -> bool:
    try:
        pid = int(run.get("pid") or 0)
    except Exception:
        return False
    if pid <= 0:
        return False
    if active_processes is None:
        active_processes = active_process_map()
    if isinstance(active_processes, set):
        return pid in active_processes
    process = active_processes.get(pid)
    if not process:
        return False
    command_line = str(process.get("CommandLine") or "").lower().replace("/", "\\")
    if not command_line:
        return False
    root_text = str(ROOT).lower().replace("/", "\\")
    project_config = str(run.get("project_config") or "").lower()
    return (
        ("src\\main.py" in command_line or "\\main.py" in command_line)
        and root_text in command_line
        and (not project_config or project_config in command_line)
    )


def mark_inactive_runs_stopped(rows: list[dict], active_processes: dict[int, dict] | set[int] | None) -> bool:
    changed = False
    now = datetime.now().isoformat(timespec="seconds")
    for run in rows:
        if run.get("stopped_at") or run.get("stop_status"):
            continue
        try:
            pid = int(run.get("pid") or 0)
        except Exception:
            pid = 0
        try:
            started_at = datetime.fromisoformat(str(run.get("started_at") or ""))
            if (datetime.now() - started_at).total_seconds() < 15:
                continue
        except Exception:
            pass
        if pid > 0 and not run_is_active(run, active_processes):
            run["stopped_at"] = now
            run["stop_status"] = "not_active"
            changed = True
    return changed


def start_dashboard_runs(payload: dict) -> dict:
    with RUN_CONTROL_LOCK:
        return _start_dashboard_runs_locked(payload)


def _start_dashboard_runs_locked(payload: dict) -> dict:
    global ACTIVE_PROCESS_CACHE
    ACTIVE_PROCESS_CACHE = None
    requested = payload.get("pairs", [])
    if not isinstance(requested, list) or not requested:
        requested = get_project_state()["pairs"]
    preflight = preflight_dashboard_runs({"pairs": requested})
    if not preflight.get("ok"):
        raise ValueError("启动前检查未通过：" + "；".join(preflight.get("errors", [])))
    started = []
    normalized = []
    for pair in requested[:4]:
        project_config = safe_config_name(pair.get("project_config"), DEFAULT_PROJECT_CONFIG)
        model_config = safe_config_name(pair.get("model_config"), default_model_config_name())
        name = str(pair.get("name") or Path(model_config).stem)
        normalized.append((name, project_config, model_config))
    project_names = [item[1] for item in normalized]
    if len(project_names) != len(set(project_names)):
        raise ValueError("parallel runs must use different project configs to avoid output/checkpoint collisions")
    existing_runs = list_dashboard_runs().get("runs", [])
    skipped = []
    for name, project_config, model_config in normalized:
        active = next((run for run in existing_runs if run.get("active") and run.get("project_config") == project_config), None)
        if active:
            skipped.append({
                "name": name,
                "project_config": project_config,
                "model_config": model_config,
                "reason": "already_running",
                "pid": active.get("pid"),
            })
            continue
        started.append(start_one_run(name, project_config, model_config))
    return {"ok": True, "started": started, "skipped": skipped}


def preflight_dashboard_runs(payload: dict) -> dict:
    requested = payload.get("pairs", [])
    if not isinstance(requested, list) or not requested:
        requested = get_project_state()["pairs"]
    errors: list[str] = []
    warnings: list[str] = []
    checked = []
    checked_projects: list[str] = []
    selection = load_selection()
    for pair in requested[:4]:
        project_config = safe_config_name(pair.get("project_config"), DEFAULT_PROJECT_CONFIG)
        model_config = safe_config_name(pair.get("model_config"), default_model_config_name())
        checked_projects.append(project_config)
        project_path = ROOT / "config" / project_config
        model_path = ROOT / "config" / model_config
        project = load_yaml(project_path)
        model = load_yaml(model_path)
        paths = project.get("paths", {})
        patient_dir = resolve_path(ROOT, paths.get("patient_dir", "data/input/patient"))
        rules_file = resolve_path(ROOT, paths.get("rules_file", "data/input/rules.xlsx"))
        output_dir = resolve_path(ROOT, paths.get("output_dir", "data/output"))
        checkpoint_db = resolve_path(ROOT, paths.get("checkpoint_db", "data/checkpoint/progress.sqlite"))
        patient_files = list_patient_workbooks(patient_dir)
        valid_patient_names = {path.name for path in patient_files}
        selected_files = [name for name in selection.get("files", []) if name in valid_patient_names]
        run_patient_count = len(selected_files) if selection.get("enabled") else len(patient_files)
        try:
            rule_count = count_rules_cached(rules_file) if rules_file.exists() else 0
        except Exception as exc:
            rule_count = 0
            errors.append(f"{project_config} 规则文件读取失败：{exc}")
        for label, path in (("患者目录", patient_dir), ("规则文件", rules_file)):
            if not path.exists():
                errors.append(f"{project_config} {label}不存在：{path}")
        if patient_dir.exists() and not patient_files:
            errors.append(f"{project_config} 患者目录中没有 xlsx 文件：{patient_dir}")
        if selection.get("enabled") and not selected_files:
            errors.append(f"{project_config} 已启用患者筛选，但没有选中任何有效患者文件。")
        provider = str(model.get("provider") or "").strip()
        base_url = str(model.get("base_url") or "").strip()
        model_name = str(model.get("model_name") or "").strip()
        placeholder_text = " ".join(str(model.get(key, "")) for key in ("base_url", "api_key", "model_name"))
        if provider != "openai_compatible":
            warnings.append(f"{model_config} provider={provider or '-'}，当前前端主要按 API 模型运行设计。")
        if not base_url or any(marker in base_url for marker in ("SERVER_C_IP", "REPLACE_WITH")):
            errors.append(f"{model_config} base_url 未配置真实地址。")
        if not model_name or any(marker in model_name for marker in ("model_1", "model_2", "REPLACE_WITH")):
            errors.append(f"{model_config} model_name 未配置真实模型名。")
        if "REPLACE_WITH_API_KEY" in placeholder_text:
            errors.append(f"{model_config} api_key 仍是占位值。")
        if not str(model.get("api_key") or "").strip():
            warnings.append(f"{model_config} api_key 为空；如果服务不需要 key 可以忽略。")
        if bool(project.get("run", {}).get("model_preflight", True)) and provider == "openai_compatible":
            if base_url and model_name and "REPLACE_WITH" not in placeholder_text and "SERVER_C_IP" not in placeholder_text:
                try:
                    check_model_ready(build_llm_config(model))
                except Exception as exc:
                    errors.append(f"{model_config} 模型连通性检查失败：{exc}")
        checked.append({
            "project_config": project_config,
            "model_config": model_config,
            "patient_dir": patient_dir.as_posix(),
            "patient_files": len(patient_files),
            "selection_enabled": bool(selection.get("enabled")),
            "selected_patients": len(selected_files),
            "run_patient_count": run_patient_count,
            "rule_count": rule_count,
            "estimated_tasks": run_patient_count * rule_count,
            "output_dir": output_dir.as_posix(),
            "checkpoint_db": checkpoint_db.as_posix(),
            "model_base_url": base_url,
            "model_name": model_name,
            "run_concurrency": int(project.get("run", {}).get("concurrency", 1) or 1),
            "rule_concurrency": int(project.get("run", {}).get("rule_concurrency", 1) or 1),
            "write_xlsx": bool(project.get("run", {}).get("write_xlsx", False)),
            "max_consecutive_llm_failures": int(project.get("run", {}).get("max_consecutive_llm_failures", 5) or 0),
            "group_rules_for_llm": bool(project.get("labeling", {}).get("group_rules_for_llm", False)),
            "llm_batch_size": int(project.get("labeling", {}).get("llm_batch_size", 1) or 1),
        })
    if len(checked_projects) != len(set(checked_projects)):
        errors.append("本次勾选了重复的项目配置，会导致输出和 checkpoint 互相覆盖，请为每个并行模型选择不同项目配置。")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "checked": checked}


def build_llm_config(raw: dict) -> LLMConfig:
    values = {}
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


def stop_dashboard_runs(payload: dict) -> dict:
    global ACTIVE_PROCESS_CACHE
    with RUN_CONTROL_LOCK:
        mode = str(payload.get("mode", "active"))
        project_config = str(payload.get("project_config") or "")
        pid = int(payload.get("pid") or 0)
        rows = load_run_records()
        ACTIVE_PROCESS_CACHE = None
        active_processes = active_process_map()
        candidates: dict[int, dict] = {}
        for run in rows:
            run_pid = int(run.get("pid") or 0)
            if run_pid > 0:
                candidates[run_pid] = run
        for process_pid, process in active_processes.items():
            command_line = str(process.get("CommandLine") or "")
            lowered = command_line.lower().replace("/", "\\")
            if str(ROOT).lower().replace("/", "\\") not in lowered or "src\\main.py" not in lowered:
                continue
            detected_project = next((name for name in list_project_config_names() if name.lower() in lowered), "")
            detected_model = next((name for name in list_model_config_names() if name.lower() in lowered), "")
            candidates.setdefault(process_pid, {
                "pid": process_pid,
                "project_config": detected_project,
                "model_config": detected_model,
                "log": "",
            })
        stopped = []
        skipped = []
        for run_pid, run in candidates.items():
            if pid and run_pid != pid:
                continue
            if project_config and run.get("project_config") != project_config:
                continue
            if mode == "active" and not run_is_active(run, active_processes):
                continue
            result = stop_one_run(run)
            if result.get("ok"):
                now = datetime.now().isoformat(timespec="seconds")
                for recorded in rows:
                    if int(recorded.get("pid") or 0) == run_pid:
                        recorded["stopped_at"] = now
                        recorded["stop_status"] = "stopped"
                stopped.append(result)
            else:
                skipped.append(result)
        if stopped:
            save_run_records(rows)
            ACTIVE_PROCESS_CACHE = None
            PROGRESS_CACHE.clear()
        return {"ok": True, "stopped": stopped, "skipped": skipped}


def stop_one_run(run: dict) -> dict:
    try:
        pid = int(run.get("pid") or 0)
    except Exception:
        return {"ok": False, "reason": "invalid_pid", "run": run}
    if pid <= 0:
        return {"ok": False, "reason": "invalid_pid", "run": run}
    if not run_is_active(run):
        return {"ok": False, "reason": "not_active", "pid": pid, "run": run}
    try:
        proc = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        ok = proc.returncode == 0
        append_to_run_log(run, f"\n[dashboard] stop_requested_at={datetime.now().isoformat(timespec='seconds')} pid={pid}\n{proc.stdout}{proc.stderr}\n")
        return {
            "ok": ok,
            "pid": pid,
            "project_config": run.get("project_config"),
            "model_config": run.get("model_config"),
            "stdout": proc.stdout[-1000:],
            "stderr": proc.stderr[-1000:],
        }
    except Exception as exc:
        return {"ok": False, "reason": str(exc), "pid": pid, "run": run}


def append_to_run_log(run: dict, text: str) -> None:
    try:
        log_path = safe_path(str(run.get("log") or ""))
        with log_path.open("a", encoding="utf-8", errors="replace") as f:
            f.write(text)
    except Exception:
        return


def start_one_run(name: str, project_config: str, model_config: str) -> dict:
    global ACTIVE_PROCESS_CACHE
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)[:40] or "run"
    log_path = RUN_DIR / f"{run_id}_{safe_name}.log"
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "src" / "main.py"),
        "--mode",
        "full",
        "--config",
        str(ROOT / "config" / project_config),
        "--model-config",
        str(ROOT / "config" / model_config),
    ]
    log_file = log_path.open("a", encoding="utf-8", errors="replace")
    log_file.write(
        f"[dashboard] started_at={datetime.now().isoformat(timespec='seconds')}\n"
        f"[dashboard] project_config={project_config} model_config={model_config}\n"
        f"[dashboard] command={' '.join(cmd)}\n\n"
    )
    log_file.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    ACTIVE_PROCESS_CACHE = None
    record = {
        "run_id": run_id,
        "name": safe_name,
        "pid": proc.pid,
        "project_config": project_config,
        "model_config": model_config,
        "log": log_path.relative_to(ROOT).as_posix(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    append_run_record(record)
    return record


def append_run_record(record: dict) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_run_records()
    rows.insert(0, record)
    save_run_records(rows[:100])


def load_run_records() -> list[dict]:
    registry = RUN_DIR / "runs.json"
    if not registry.exists():
        return []
    try:
        rows = json.loads(registry.read_text(encoding="utf-8"))
    except Exception:
        rows = []
    return rows if isinstance(rows, list) else []


def save_run_records(rows: list[dict]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "runs.json").write_text(json.dumps(rows[:100], ensure_ascii=False, indent=2), encoding="utf-8")


def list_dashboard_runs() -> dict:
    all_rows = load_run_records()
    if mark_inactive_runs_stopped(all_rows, None):
        save_run_records(all_rows)
    rows = all_rows[:20]
    for row in rows:
        log_path = ROOT / str(row.get("log") or "")
        if log_path.exists():
            row["log_updated_at"] = datetime.fromtimestamp(log_path.stat().st_mtime).isoformat(timespec="seconds")
        row["active"] = run_is_active(row)
    return {"runs": rows}


def update_yaml_scalar(path: Path, key_path: list[str], value: int | str | bool) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    parent_key = key_path[0] if len(key_path) > 1 else None
    target_key = key_path[-1]
    in_parent = parent_key is None
    parent_indent = -1
    replaced = False
    rendered_value = json.dumps(value, ensure_ascii=False) if isinstance(value, str) else str(value).lower() if isinstance(value, bool) else str(value)

    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        key = stripped.split(":", 1)[0].strip()
        if parent_key and key == parent_key:
            in_parent = True
            parent_indent = indent
            continue
        if parent_key and in_parent and indent <= parent_indent:
            in_parent = False
        if in_parent and key == target_key:
            lines[idx] = f"{' ' * indent}{target_key}: {rendered_value}"
            replaced = True
            break

    if not replaced:
        if parent_key:
            lines.extend([f"{parent_key}:", f"  {target_key}: {rendered_value}"])
        else:
            lines.append(f"{target_key}: {rendered_value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    global WEB, DEFAULT_PROJECT_CONFIG, DEFAULT_MODEL_CONFIG
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--web-root", default="web", help="Frontend directory relative to the project root.")
    parser.add_argument("--default-project-config", default=DEFAULT_PROJECT_CONFIG)
    parser.add_argument("--default-model-config", default=DEFAULT_MODEL_CONFIG)
    args = parser.parse_args()
    web_root = (ROOT / args.web_root).resolve()
    if web_root.parent != ROOT.resolve() or not (web_root / "index.html").exists():
        raise ValueError("--web-root must name a project-root frontend directory containing index.html")
    WEB = web_root
    DEFAULT_PROJECT_CONFIG = safe_config_name(args.default_project_config, DEFAULT_PROJECT_CONFIG)
    DEFAULT_MODEL_CONFIG = safe_config_name(args.default_model_config, DEFAULT_MODEL_CONFIG)
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard: http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

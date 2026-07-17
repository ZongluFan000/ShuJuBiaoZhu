from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import MISSING, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from config_loader import load_yaml
from llm_client import LLMConfig, LLMClient


JSON_PROMPT = """请只输出一个合法 JSON 对象，不要输出 Markdown，不要输出解释：
{"items":[{"label":"未知","explanation":"模型连通性测试","evidence":"health_check","confidence":0.1}]}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Health check an OpenAI-compatible model API.")
    parser.add_argument("--model-config", default="config/model_internal_health.yaml")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--output-dir", default="data/output_model_health")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = build_config(load_yaml(root / args.model_config))
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    checks: list[dict[str, Any]] = []
    checks.append(check_models_endpoint(cfg))
    checks.append(check_chat(cfg, stream=False))
    checks.append(check_chat(cfg, stream=True))
    bench = benchmark(cfg, max(1, args.concurrency), max(1, args.requests))
    checks.extend(bench["rows"])

    ok = all(row["ok"] for row in checks)
    summary = {
        "run_id": run_id,
        "ok": ok,
        "base_url": cfg.base_url,
        "model_name": cfg.model_name,
        "checks": checks,
        "benchmark": bench["summary"],
    }
    path = out_dir / f"model_health_{run_id}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"base_url={cfg.base_url}")
    print(f"model_name={cfg.model_name}")
    for row in checks:
        status = "OK" if row["ok"] else "FAIL"
        print(f"{status} {row['name']} {row.get('elapsed_seconds', '-') }s {row.get('message', '')}")
    print(json.dumps(bench["summary"], ensure_ascii=False, indent=2))
    print(f"result_file={path}")
    return 0 if ok else 1


def build_config(raw: dict[str, Any]) -> LLMConfig:
    values = {}
    for field in LLMConfig.__dataclass_fields__.values():
        if field.name in raw and raw.get(field.name) is not None:
            values[field.name] = raw.get(field.name)
        elif field.default is not MISSING:
            values[field.name] = field.default
        else:
            values[field.name] = ""
    return LLMConfig(**values)


def check_models_endpoint(cfg: LLMConfig) -> dict[str, Any]:
    url = cfg.base_url.rstrip("/") + "/models"
    headers = {}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    started = time.perf_counter()
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with direct_urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        elapsed = round(time.perf_counter() - started, 3)
        ids = [item.get("id") for item in data.get("data", []) if isinstance(item, dict)]
        model_seen = cfg.model_name in ids if ids else None
        return {
            "name": "GET /models",
            "ok": bool(ids) and model_seen is not False,
            "elapsed_seconds": elapsed,
            "message": f"models={ids[:5]} model_seen={model_seen}",
        }
    except Exception as exc:
        return fail("GET /models", started, exc)


def check_chat(cfg: LLMConfig, stream: bool) -> dict[str, Any]:
    test_cfg = replace(cfg, stream=stream, max_tokens=min(int(cfg.max_tokens or 512), 512))
    started = time.perf_counter()
    try:
        data = LLMClient(test_cfg).label(JSON_PROMPT)
        elapsed = round(time.perf_counter() - started, 3)
        items = data.get("items") if isinstance(data, dict) else None
        ok = isinstance(items, list) and bool(items)
        return {
            "name": f"POST /chat/completions stream={str(stream).lower()}",
            "ok": ok,
            "elapsed_seconds": elapsed,
            "message": json.dumps(data, ensure_ascii=False)[:300],
        }
    except Exception as exc:
        return fail(f"POST /chat/completions stream={str(stream).lower()}", started, exc)


def benchmark(cfg: LLMConfig, concurrency: int, requests: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    def one_call(index: int) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            data = LLMClient(cfg).label(JSON_PROMPT)
            ok = isinstance(data.get("items"), list)
            return {
                "name": f"benchmark request {index}",
                "ok": ok,
                "elapsed_seconds": round(time.perf_counter() - t0, 3),
                "message": "success" if ok else "response JSON missing items",
            }
        except Exception as exc:
            return fail(f"benchmark request {index}", t0, exc)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one_call, i) for i in range(1, requests + 1)]
        for future in as_completed(futures):
            rows.append(future.result())

    latencies = [float(row["elapsed_seconds"]) for row in rows if row["ok"]]
    total = time.perf_counter() - started
    summary = {
        "concurrency": concurrency,
        "requests": requests,
        "success": sum(1 for row in rows if row["ok"]),
        "failed": sum(1 for row in rows if not row["ok"]),
        "total_elapsed_seconds": round(total, 3),
        "requests_per_minute": round(len(latencies) / total * 60, 3) if total > 0 else 0,
        "avg_latency_seconds": round(statistics.mean(latencies), 3) if latencies else None,
        "max_latency_seconds": round(max(latencies), 3) if latencies else None,
    }
    return {"rows": sorted(rows, key=lambda row: row["name"]), "summary": summary}


def fail(name: str, started: float, exc: Exception) -> dict[str, Any]:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            detail = str(exc)
        message = f"HTTP {exc.code}: {detail}"
    else:
        message = str(exc)
    return {
        "name": name,
        "ok": False,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "message": message,
    }


def direct_urlopen(req: urllib.request.Request, timeout: int):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(req, timeout=timeout)


if __name__ == "__main__":
    raise SystemExit(main())

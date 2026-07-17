from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from config_loader import load_yaml
from llm_client import LLMConfig, LLMClient


SHORT_PROMPT = """请只输出一个合法 JSON 对象，不要输出 Markdown：
{"label":"符合","explanation":"并发测试","evidence":"测试","confidence":0.9}
"""

REALISH_PROMPT = """你是临床试验入排标准标注助手。请根据患者证据判断标准。
标注规则：入选标准满足=符合，不满足=不符合，证据不足=未知。排除标准满足排除条件=不符合，不满足=符合。
只输出合法 JSON，不要输出 Markdown。

患者证据：
患者，女，1985-08-03。诊断：宫颈癌。病理：宫颈鳞状细胞癌。未见前列腺癌相关记录。

待判断标准：
试验注册号：CTR_TEST
试验标识：试验_TEST
标准编号：1
规则标识：入选标准
标准内容：组织学上确诊为前列腺癌。

输出格式：
{"items":[{"trial_register_id":"CTR_TEST","trial_id":"试验_TEST","standard_no":"1","rule_type":"入选标准","label":"符合/不符合/未知","explanation":"120字以内","evidence":"30字以内","confidence":0.0}]}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark model API concurrency.")
    parser.add_argument("--model-config", default="config/model_api.example.yaml")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--prompt-mode", choices=["short", "realish"], default="short")
    parser.add_argument("--output-dir", default="data/output_concurrency_benchmark")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg_raw = load_yaml(root / args.model_config)
    cfg = LLMConfig(**{k: cfg_raw.get(k) for k in LLMConfig.__annotations__})
    prompt = SHORT_PROMPT if args.prompt_mode == "short" else REALISH_PROMPT

    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    def one_call(i: int) -> dict[str, Any]:
        client = LLMClient(cfg)
        t0 = time.perf_counter()
        try:
            data = client.label(prompt)
            elapsed = time.perf_counter() - t0
            return {
                "request_id": i,
                "status": "success",
                "latency_seconds": round(elapsed, 4),
                "error": "",
                "response_preview": json.dumps(data, ensure_ascii=False)[:300],
            }
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            return {
                "request_id": i,
                "status": "failed",
                "latency_seconds": round(elapsed, 4),
                "error": str(exc)[:500],
                "response_preview": "",
            }

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one_call, i) for i in range(1, args.requests + 1)]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"{row['request_id']}/{args.requests} {row['status']} {row['latency_seconds']}s")

    total_elapsed = time.perf_counter() - started
    success_rows = [r for r in rows if r["status"] == "success"]
    failed_rows = [r for r in rows if r["status"] == "failed"]
    latencies = sorted(float(r["latency_seconds"]) for r in success_rows)
    summary = {
        "run_id": run_id,
        "provider": cfg.provider,
        "model_name": cfg.model_name,
        "base_url": cfg.base_url,
        "prompt_mode": args.prompt_mode,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "success": len(success_rows),
        "failed": len(failed_rows),
        "failure_rate": round(len(failed_rows) / max(args.requests, 1), 4),
        "total_elapsed_seconds": round(total_elapsed, 3),
        "requests_per_minute": round(len(success_rows) / total_elapsed * 60, 3) if total_elapsed > 0 else 0,
        "avg_latency_seconds": round(statistics.mean(latencies), 3) if latencies else None,
        "p50_latency_seconds": percentile(latencies, 50),
        "p95_latency_seconds": percentile(latencies, 95),
    }

    csv_path = out_dir / f"concurrency_{args.concurrency}_{args.prompt_mode}_{run_id}.csv"
    json_path = out_dir / f"concurrency_{args.concurrency}_{args.prompt_mode}_{run_id}.json"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["request_id", "status", "latency_seconds", "error", "response_preview"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: int(r["request_id"])))
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failed_rows else 1


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


if __name__ == "__main__":
    raise SystemExit(main())

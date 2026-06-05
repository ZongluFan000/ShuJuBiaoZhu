# 并发测试说明

用于测试服务器 C 或公网 API 的并发承载能力。测试不会读取患者数据，只向模型发送短 prompt 或接近真实标注的 prompt。

## 前置配置

先编辑：

```text
config/model_api.yaml
```

填写：

```yaml
provider: "openai_compatible"
base_url: "http://服务器C地址:端口/v1"
api_key: "实际 API key"
model_name: "实际模型名"
```

## 快速测试

```bat
runtime\python\python.exe src\benchmark_concurrency.py --model-config config\model_api.yaml --concurrency 1 --requests 10 --prompt-mode short
```

## 阶梯压测

建议依次跑：

```bat
runtime\python\python.exe src\benchmark_concurrency.py --model-config config\model_api.yaml --concurrency 1 --requests 20 --prompt-mode short
runtime\python\python.exe src\benchmark_concurrency.py --model-config config\model_api.yaml --concurrency 2 --requests 20 --prompt-mode short
runtime\python\python.exe src\benchmark_concurrency.py --model-config config\model_api.yaml --concurrency 4 --requests 20 --prompt-mode short
runtime\python\python.exe src\benchmark_concurrency.py --model-config config\model_api.yaml --concurrency 8 --requests 20 --prompt-mode short
```

再跑真实一点的 prompt：

```bat
runtime\python\python.exe src\benchmark_concurrency.py --model-config config\model_api.yaml --concurrency 4 --requests 20 --prompt-mode realish
```

## 看什么指标

输出 JSON 中重点看：

```text
success
failed
failure_rate
total_elapsed_seconds
requests_per_minute
avg_latency_seconds
p50_latency_seconds
p95_latency_seconds
```

选择并发数的标准：

```text
failure_rate < 1%
p95 不超过平均耗时的 2-3 倍
requests_per_minute 相比更低并发仍明显增加
```

如果并发升高后吞吐不再增加、P95 大幅升高或失败率上升，就说明已经超过服务器可承受范围。

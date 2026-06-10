@echo off
setlocal
cd /d %~dp0

runtime\python\python.exe src\stress_test_model.py ^
  --model-config config\model_api.yaml ^
  --project-config config\project_full_ai_only.yaml ^
  --prompt-mode production ^
  --levels 16,20,24 ^
  --requests-per-level 24 ^
  --warmup 1 ^
  --cooldown-seconds 5 ^
  --failure-threshold 0.1 ^
  --max-avg-latency-seconds 60 ^
  --max-p95-latency-seconds 90 ^
  --max-timeout-like-failures 2 ^
  --max-initial-submit 8

pause

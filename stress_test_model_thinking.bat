@echo off
setlocal
cd /d %~dp0

runtime\python\python.exe src\stress_test_model.py ^
  --model-config config\model_api.yaml ^
  --project-config config\project_full_ai_only.yaml ^
  --prompt-mode production ^
  --thinking true ^
  --stream false ^
  --levels 2,4,6,8 ^
  --requests-per-level 12 ^
  --warmup 1 ^
  --cooldown-seconds 10 ^
  --failure-threshold 0.1 ^
  --max-avg-latency-seconds 120 ^
  --max-p95-latency-seconds 180 ^
  --max-timeout-like-failures 1 ^
  --max-initial-submit 4 ^
  --timeout-seconds 300

pause

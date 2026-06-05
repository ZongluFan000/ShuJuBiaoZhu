@echo off
setlocal
cd /d "%~dp0"
call tools\select_python.bat
echo Concurrency benchmark: default concurrency=1, requests=10, prompt=short.
%SANDBOX_PYTHON% src\benchmark_concurrency.py --model-config config\model_api.yaml --concurrency 1 --requests 10 --prompt-mode short
pause

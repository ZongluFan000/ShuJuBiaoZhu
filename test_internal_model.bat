@echo off
setlocal
cd /d "%~dp0"
call tools\select_python.bat
%SANDBOX_PYTHON% src\model_health_check.py --model-config config\model_internal_health.yaml --concurrency 2 --requests 4
pause

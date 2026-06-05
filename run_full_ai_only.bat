@echo off
setlocal
cd /d "%~dp0"
call tools\select_python.bat
%SANDBOX_PYTHON% src\main.py --mode full --config config\project_full_ai_only.yaml --model-config config\model_api.yaml
pause

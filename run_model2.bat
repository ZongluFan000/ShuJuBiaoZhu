@echo off
setlocal
cd /d "%~dp0"
call tools\select_python.bat
%SANDBOX_PYTHON% src\main.py --mode full --config config\project_full_ai_only_model2.yaml --model-config config\model_api_2.yaml
pause

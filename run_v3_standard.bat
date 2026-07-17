@echo off
setlocal
cd /d %~dp0

call tools\select_python.bat
if errorlevel 1 exit /b 1
%SANDBOX_PYTHON% src\main.py --mode full --config config\project_v3_standard.yaml --model-config config\model_api_v3.yaml
pause

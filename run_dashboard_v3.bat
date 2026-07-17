@echo off
setlocal
cd /d "%~dp0"

echo V3 Dashboard: http://127.0.0.1:7861/
call tools\select_python.bat
if errorlevel 1 exit /b 1
%SANDBOX_PYTHON% src\dashboard_server.py --host 127.0.0.1 --port 7861 --web-root web_v3 --default-project-config project_v3_standard.yaml --default-model-config model_api_v3.yaml
pause

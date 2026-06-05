@echo off
setlocal
cd /d "%~dp0"
call tools\select_python.bat
echo Cleaning old dashboard services on port 7860...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*dashboard_server.py*' -and $_.CommandLine -like '*--port 7860*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>nul
echo Dashboard: http://127.0.0.1:7860/
%SANDBOX_PYTHON% src\dashboard_server.py --host 127.0.0.1 --port 7860
pause

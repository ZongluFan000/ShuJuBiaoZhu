@echo off
cd /d "%~dp0"
echo Starting local LLM proxy...
if exist "%~dp0..\annotation_sandbox\runtime\python\python.exe" (
  start "" "http://127.0.0.1:8765"
  "%~dp0..\annotation_sandbox\runtime\python\python.exe" local_proxy.py
  goto :end
)

if exist "%~dp0..\annotation_sandbox_api_only\runtime\python\python.exe" (
  start "" "http://127.0.0.1:8765"
  "%~dp0..\annotation_sandbox_api_only\runtime\python\python.exe" local_proxy.py
  goto :end
)

where py >nul 2>nul
if %errorlevel%==0 (
  start "" "http://127.0.0.1:8765"
  py local_proxy.py
  goto :end
)

where python >nul 2>nul
if %errorlevel%==0 (
  start "" "http://127.0.0.1:8765"
  python local_proxy.py
  goto :end
)

where powershell >nul 2>nul
if %errorlevel%==0 (
  powershell -ExecutionPolicy Bypass -File "%~dp0local_proxy.ps1"
  goto :end
)

echo Python and PowerShell were not found on this computer.
echo Please install Python 3, or run this folder on a computer with Python or PowerShell available.
pause

:end

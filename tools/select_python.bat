@echo off
if exist "%~dp0..\runtime\python\python.exe" (
  set "SANDBOX_PYTHON=%~dp0..\runtime\python\python.exe"
  goto :eof
)
where python >nul 2>nul
if %errorlevel%==0 (
  set "SANDBOX_PYTHON=python"
  goto :eof
)
where py >nul 2>nul
if %errorlevel%==0 (
  set "SANDBOX_PYTHON=py"
  goto :eof
)
echo Python not found. Put portable Python at runtime\python\python.exe.
exit /b 1

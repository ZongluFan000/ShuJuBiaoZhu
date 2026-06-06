@echo off
setlocal
cd /d "%~dp0"
call tools\select_python.bat
%SANDBOX_PYTHON% src\ab_experiment.py --patient-file 100026214540.xlsx --model-config config\model_api.yaml
pause

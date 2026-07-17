@echo off
setlocal
cd /d %~dp0

runtime\python\python.exe src\main.py --mode full --config config\project_v3_standard.yaml --model-config config\model_api_v3.yaml
pause

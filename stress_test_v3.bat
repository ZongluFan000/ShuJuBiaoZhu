@echo off
setlocal
cd /d %~dp0

runtime\python\python.exe src\stress_test_v3.py --config config\project_v3_stress.yaml
pause

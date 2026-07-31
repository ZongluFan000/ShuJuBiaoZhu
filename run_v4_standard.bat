@echo off
call tools\select_python.bat
%SANDBOX_PYTHON% src\main.py --config config\project_v4_standard.yaml --model-config config\model_api_v4.yaml

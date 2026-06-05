@echo off
setlocal
cd /d "%~dp0"
call tools\select_python.bat
start "model1" cmd /c "%SANDBOX_PYTHON% src\main.py --mode full --config config\project_full_ai_only_model1.yaml --model-config config\model_api_1.yaml"
start "model2" cmd /c "%SANDBOX_PYTHON% src\main.py --mode full --config config\project_full_ai_only_model2.yaml --model-config config\model_api_2.yaml"
echo Started model1 and model2. Outputs are separated under data\output_full_ai_only_model1 and data\output_full_ai_only_model2.
pause

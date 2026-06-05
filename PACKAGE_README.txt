annotation_sandbox_api_only

Purpose:
- Production package for intranet Windows machines.
- Uses remote OpenAI-compatible model APIs only.
- Does not include local GPU model dependencies such as torch or transformers.

Start:
- run_dashboard.bat
- Open http://127.0.0.1:7860

Required configuration:
- config/model_api.yaml for single-model API mode
- config/model_api_1.yaml and config/model_api_2.yaml for two-model API mode
- Put patient Excel files under data/input/patient/
- Rules file is data/input/rules.xlsx

Main run scripts:
- run_full_ai_only.bat
- run_model1.bat
- run_model2.bat
- run_both_models.bat
- run_retry_failed.bat
- run_test.bat

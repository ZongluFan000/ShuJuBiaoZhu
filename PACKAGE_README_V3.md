# 数据标注脚本 V3

此包包含 V3 标准层标注代码、V3 前端、229 条标准层规则和 V3 标注方案。

运行前：

1. 安装 Python 依赖：`pip install -r requirements.txt`。
2. 将 `config/model_api_v3.example.yaml` 复制为 `config/model_api_v3.yaml`，填写 API key。
3. 将患者 xlsx 文件放在项目同级的 `patientTestDeepseek` 目录，或修改项目配置中的 `paths.patient_dir`。

正式运行：`run_v3_standard.bat`。

V3 前端：`run_dashboard_v3.bat`，访问 `http://127.0.0.1:7861/`。

发布包不包含患者数据、运行结果、checkpoint、Python runtime 或真实 API key。

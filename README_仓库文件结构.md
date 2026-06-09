# 仓库文件结构说明

这个仓库只提交代码、规则、模板和必要的沙箱运行脚本。

不会提交到 GitHub 的本地内容：

- `config/model_api.yaml`、`config/model_api_1.yaml`、`config/model_api_2.yaml`、`config/model_internal_health.yaml`
  - 这些文件包含本地模型地址或 API key。
  - 新环境请复制 `config/model_api.example.yaml` 后再填写真实配置。
- `data/input/patient/`、`data/input/patient100/`
  - 患者数据只保留在本机或内网机器。
- `data/output*/`、`data/checkpoint*/`、`data/logs*/`、`data/dashboard_runs/`
  - 运行结果、断点、日志只保留在本机。
- `__pycache__/`、`*.pyc`
  - Python 运行缓存不提交。

新增模型配置时，优先新增 `.example.yaml` 模板，不要提交真实 API key。

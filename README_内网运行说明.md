# 临床试验数据 AI 标注沙箱

这是 Windows 内网运行版沙箱。内网电脑不需要预装 Python、IDE、Node.js 或前端依赖，直接使用沙箱内置的 `runtime/python/python.exe`。

## 推荐入口

```text
run_dashboard.bat
```

打开后访问：

```text
http://127.0.0.1:7860
```

前端支持：

- 患者文件选择
- 单模型运行
- 双模型运行
- 启动前检查
- 中断运行中任务
- Checkpoint 断点查看
- 失败患者和失败任务查看
- 日志查看
- 文件查看
- 并发压测

## 保留脚本

```text
check_env.bat                    检查环境和配置
run_dashboard.bat                启动前端控制台
run_full_ai_only.bat             单模型正式运行
run_both_models.bat              双模型正式运行
run_model1.bat                   只运行模型1
run_model2.bat                   只运行模型2
run_retry_failed.bat             重跑失败患者
run_concurrency_benchmark.bat    API 并发压测
```

正式使用建议优先从前端启动，不要直接双击多个运行脚本，避免重复启动。

## 数据放置

患者文件：

```text
data/input/patient/
```

规则文件：

```text
data/input/rules.xlsx
```

## 模型配置

单模型运行使用：

```text
config/model_api.yaml
config/project_full_ai_only.yaml
```

双模型运行使用：

```text
config/model_api_1.yaml
config/project_full_ai_only_model1.yaml

config/model_api_2.yaml
config/project_full_ai_only_model2.yaml
```

如果双模型中某个模型配置还是占位值，前端启动前检查会拦截，不会误启动。

## 断点续跑

程序会写入 checkpoint：

```text
data/checkpoint_full_ai_only/
data/checkpoint_full_ai_only_model1/
data/checkpoint_full_ai_only_model2/
```

中断任务不会删除 checkpoint。再次启动时会跳过已完成任务，继续运行剩余任务。

## 输出目录

```text
data/output_full_ai_only/
data/output_full_ai_only_model1/
data/output_full_ai_only_model2/
```

常见输出文件：

```text
annotation_results_live.csv
annotation_results.csv
failed_tasks.csv
failed_patients.csv
patient_status.csv
failed_tasks_from_checkpoint.csv
run_summary.json
```

## 内网使用流程

1. 解压沙箱。
2. 把患者 Excel 放到 `data/input/patient/`。
3. 确认规则文件为 `data/input/rules.xlsx`。
4. 填写 `config/model_api.yaml`，如果要双模型运行，也填写 `model_api_1.yaml` 和 `model_api_2.yaml`。
5. 双击 `run_dashboard.bat`。
6. 在前端选择患者、选择单模型或双模型。
7. 点击启动，前端会先做启动前检查。
8. 运行中可查看进度、日志、checkpoint，也可以中断任务。

# 规则聚类与合并 A/B 对照试验

本试验不会修改原正式配置和正式输出目录。

## 两个试验组

- 基线组：使用原有按规则分类组批逻辑，每批 5 条。
- 优化组：按规则类型、缺失策略、证据主类和复杂度聚类；普通规则每批最多 10 条，复杂规则每批最多 3 条。
- 优化组使用紧凑 JSON 输出，重复的试验注册号和规则标识由后端从原始规则回填，最终结果字段不变。
- 普通批次大小会根据模型 `max_tokens` 自动收缩，避免批量输出被截断。
- 优化组使用 `label_rules_compact.md`，只压缩固定说明，不删除原始规则全文和规则相关患者证据。
- 批次同时受条数、复杂度总分、规则文本长度和预计输出 token 限制。

优化组只自动合并满足以下条件的规则：

- 原始规则文本完全相同。
- 规则标识相同。
- 不包含研究药物、本研究、首次给药、随机、筛选、基线等试验特定指代。

语义相似但文本不同的规则不会自动合并。只有在
`config/reviewed_equivalent_rules.json` 中人工审核为 `approved` 的等价组才允许共享一次判断。

无论内部是否共享判断，最终仍会按原始规则展开，每个患者输出全部 1714 条记录。

## 运行

```text
run_ab_experiment.bat
```

或：

```bat
runtime\python\python.exe src\ab_experiment.py --patient-file 100026214540.xlsx --model-config config\model_api.yaml
```

## 输出

```text
data/ab_experiment/baseline/
data/ab_experiment/optimized/
data/ab_experiment/reports/
```

报告检查：

- 输出行数是否等于规则数。
- 必需字段是否完整。
- 是否存在重复、遗漏或额外规则。
- 标签一致率和不一致明细。
- 两组模型调用次数和耗时。
- 优化组相对基线组的加速比。

没有人工金标准时，报告中的 `label_agreement_rate` 仅表示优化组与原逻辑的一致率，
不能解释为临床准确率。提供人工金标准 CSV 时可使用：

```bat
runtime\python\python.exe src\ab_experiment.py --gold-csv data\gold.csv
```

# 可填写的21日配置模板

这些模板不附带Data，也不是开箱即用案例。必须填写全部`<...>`，使用实际冻结日历、历史成员和来源证据；不要编造Snapshot、交易日期或摘要。

CLI直接读取`options.json`，再按其中路径读取`experiment.yaml`、`dataset.yaml`、`label.yaml`、`model.yaml`及登记了实际摘要的完整axes JSON。subset同理读取其三个subset文件，并复用主`feature-result.json`和`batch.json`。

以下文件是装配片段，CLI不会自动读取；仅修改片段不会改变运行配置：

- `time.yaml` → `dataset.yaml.periodic_schedule`及subset对应字段。
- `universe.json` + `features.json` → 完整axes JSON → dataset的`axes_input.path/file_digest`和options的`feature_spec`。
- `evaluation.json` → `options.json.evaluation`及subset对应字段。
- `strategies.json` → Engine的`stock_strategy_spec(**variant)`；Research CLI不运行账户。

复制模板到本地目录，填完路径及冻结Data字段后，按照统一说明的配置拼合示例写出axes、计算真实文件摘要、更新并保存两个dataset/options，再运行validate。不要保留`<ASSEMBLED_AXES_JSON>`或旧文件摘要。主训练完成后，subset的`feature_result`和`reuse_raw_from_batch`须指向主产物。

完整装配代码、运行命令、PIT/价格口径与复用限制只在[axiom-docs统一说明](https://github.com/sinnergarden/axiom-docs/blob/codex/docs-m2-one-year-case-20261011/docs/design/m2-one-year-executable-case.md#可运行入口)维护。执行入口为`examples/a01_one_year.py`。

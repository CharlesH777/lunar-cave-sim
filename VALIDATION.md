# 导入验证记录

2026-11-12 将 2027 上的 V16b 完整包原样导入本仓库。以下结果区分文件搬运、离线功能检查和仿真运行。

| 检查 | 结果 |
| --- | --- |
| 原包 SHA256 / 文件清单 | 9,968 个清单条目全部通过，缺失、差异、额外文件均为 0 |
| Git 跟踪范围 | 原包连同清单共 9,969 个文件全部纳入 |
| 凭证特征扫描 | 检查 2,981 个文本文件；常见 API/GitHub/AWS 密钥、私钥头和带凭证 URL 特征未命中 |
| 原始采集生产者 CPU 自检 | 10 tests 通过 |
| 标注消费者 CPU 自检 | 17 tests 通过 |
| 稀疏训练视图 / 语义可见性 CPU 回归 | 均通过 |
| 真实示例稀疏视图 | 172,119 checks 通过 |
| 真实示例查询划分 | 13,638 checks 通过，3,632 train / 908 heldout 索引行 |
| 真实密集示例严格重放 | 54,051 checks 中有 28 项失败，详见下文 |
| Isaac GPU 渲染、训练、付费 API | 本次未执行 |

CPU 功能复核使用 macOS、Python 3.14.7、NumPy 2.4.2；这不是项目要求的 Isaac Sim 5.1 / Python 3.11 环境。

执行过的 CPU 入口如下，工作目录为 `simulation/`，均设置 `PYTHONDONTWRITEBYTECODE=1`；自检临时目录和 dense 报告位于包外：

```bash
python selftest_lunar_raw_batch_v16_cpu.py
python selftest_lunar_annotation_queue_v16_cpu.py
python selftest_lunar_training_stride_v14_cpu.py
python selftest_lunar_semantic_visibility_v14_cpu.py
python validate_lunar_dataset_v11.py --dataset examples/dataset_smoke/dataset --require-cot --output /absolute/path/outside-package/dense-validation.json
python -m cave.lunar_query_indexes_v14 --output examples/dataset_smoke/query_partitions --validate
python -c "from pathlib import Path; from cave.lunar_training_stride_v14 import validate_training_view; print(validate_training_view(Path('examples/dataset_smoke/training_stride5')))"
```

## 密集示例的严格重放差异

完整报告保存在 [validation/dense-validation-macos.json](validation/dense-validation-macos.json)。28 项中，11 项涉及地标证据，17 项涉及语义识别/历史证据。

首个失败发生在 step 23：保存的 `bearing_deg=6.241914347415048`，本机重新计算为 `6.241914347415049`，差约 `8.88e-16`。原校验器对证据字典使用严格相等比较。其余 26 项尚未逐个定位，不能将全部失败都归因于浮点环境。

原数据和原校验器均未修改，包内原始验收记录也保持原样。本次不将该密集示例标记为重新完整验收通过。

## 复核文件搬运

在下载全部 Git LFS 内容后执行：

```bash
python3 simulation/scripts/verify_delivery_v11.py simulation
```

`simulation/` 包内应保持只有原始文件；将运行输出和 Python 缓存写入该目录会被完整性校验器识别为额外文件。CPU 检查应使用 `PYTHONDONTWRITEBYTECODE=1`，并将临时输出放在包外。

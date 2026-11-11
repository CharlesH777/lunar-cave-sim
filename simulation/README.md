# Lunar cave dataset pipeline V16

完整入口与环境说明见 [docs/PIPELINE_V16.md](docs/PIPELINE_V16.md)。

阶段：地形与障碍生成 → 随机四目标布局 → 连续 RGB-D 轨迹 → 真实路线 query 编译 / 当前图像短 CoT → 数据校验 → 独立队列发布。

提供 templates/lunar_fork_v11 和 examples/dataset_smoke。示例是完整单元，密集基底位于其 dataset/，稀疏视图和 query_partitions 与其并列。数据域为 isaac_continuous；不是现有 129 训练器已接入证明。

运行前用 Python 3.11 执行 scripts/check_pipeline_runtime_v11.py；服务器系统默认 Python 3.6 仅可运行下面的完整性校验器。

```bash
python3 scripts/verify_delivery_v11.py .
```

# 月洞平滑采集与第一人称关键帧

本版沿用 V11 地形和随机目标布局。先在真实地形和障碍范围内生成平滑路径，再按 0.2 秒积分车体运动。相机跟随实际车体朝向。轨迹的速度上限为 0.4 m/s，角速度上限为 0.32 rad/s，角加速度上限为 0.24 rad/s²。采集前复核完整扫掠轨迹，目标接近点和真实可见性共同决定末端 STOP。

环境与数据契约见 [V11 流水线说明](PIPELINE_V11.md)。配置 `configs/lunar_dataset_pipeline_v12.json` 使用 `motion_controller=smooth` 和 `annotation.cot_style=first_person`。每条轨迹最多送 8 个关键帧，逐帧 RGB-D 和控制均保留。首次试运行可用 `--max-episodes 1`，复用包内地形模板。

```bash
"$ISAAC_PYTHON" lunar_dataset_pipeline_v11.py \
  --config configs/lunar_dataset_pipeline_v12.json \
  --output ../lunar_smooth_run \
  --isaac-python "$ISAAC_PYTHON" \
  --ppio-env /secure/path/ppio.env
```

关键帧短句存于 `cot_text`。当车在洞口继续前进时，可写“我看到了洞口，我应该进去。” 当当前深度和标定确认两条通道可见，并且这条轨迹确实选择左侧通道时，可写“我看到了分叉，我应该走左边的通道。” 当车到达接近点、当前帧看见红旗并执行 STOP 时，可写“我看到了红旗，我应该停下来。” 其他帧按实际转向或前进动作生成说明。`cot_valid` 区分通过核验的文本和只有动作监督的帧。

地形是 Isaac Sim 5.1 的合成场景，车辆数据是碰撞检查后的运动学教师示范。现有 129 训练器接入需验证其连续控制和文字监督接口。2027 的运行环境情况见交付预检报告。

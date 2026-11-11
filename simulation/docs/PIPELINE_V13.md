# 导航路线、墙距和配套语言标注

配置入口是 `configs/lunar_dataset_pipeline_v13.json`，运行环境和基础字段见 [流水线说明](PIPELINE_V11.md)。车辆沿经过碰撞检查的连续路径行进，在接近过程中完成方向调整并减速停车。这一版要求车身外缘离侧墙至少一米，出发和到达均不允许原地转向。V11 文档中的分段原地转向、精确到原始接近点和末端对齐描述仅适用于旧控制器；V13 的行为以本页为准。

```bash
"$ISAAC_PYTHON" lunar_dataset_pipeline_v11.py \
  --config configs/lunar_dataset_pipeline_v13.json \
  --output ../lunar_route_run \
  --isaac-python "$ISAAC_PYTHON" \
  --ppio-env /secure/path/ppio.env
```

`motion_controller=smooth_v13` 使用全局可见路径和局部转弯曲线。`annotation.instruction_style=route_grounded` 生成配套路线提示，`annotation.cot_style=first_person` 生成第一人称关键帧短句。首次可加 `--max-episodes 1`。

墙距由实际洞壁隐式场核验。以车身外接半径 0.65 米加 1.05 米余量，在距地板 0.6、1.0、1.4 米三个高度采样扩张车体侧面；沿执行路径的扫掠步长不超过 2.5 厘米，周向间隔不超过 2 度。`body_wall_clearance.passed=true` 表示此离散侧墙检查通过。栅格墙距另含一个完整栅格对角线的保守扣除，仅是寻路诊断，不能当作连续表面精确最小距离。碎石、物品、四轮地面支撑、坡度、车腹和车身空气包络仍分别按实际几何检查。`runtime_collision_mode=actual_geometry` 显式区分执行阶段的实际几何门槛与寻路栅格的额外离散膨胀。

终点保留原始 `approach_xy_m`，在其一米范围内选择安全停车点；只在实际积分到所选点 0.05 米内、当前目标实例遮罩至少 12 像素时接受唯一零速 STOP。相机在行进中完成朝向调整，之后直接停车。固定 `dt=0.2s`，速度上限 0.4 米/秒，角速度上限 0.32 弧度/秒，线加速度上限 0.25 米/秒²、角加速度上限 0.24 弧度/秒²。验收拒收任何原地转向帧。1 米停车标签约定及 3 米后续评测距离不会由这组教师轨迹替代实现。

原始 RGB-D、实际控制和历史帧引用保留在 `trajectory`。CPU 后处理根据完整轨迹中的真实洞口、分叉、物品遮罩和停车事件生成 `route_contract.json`，再在 `route_trajectory` 中统一绑定固定指令。提示中的地标和方向按实际顺序排列，三条 query 改写保留同一语义。原始文件哈希和派生过程记录在 provenance 中。

洞口事件沿用入口跨越检查；地标来自同帧实例遮罩。V13 分叉证据使用同帧深度、相机视野、两侧开口和中间岩石遮挡核验，并依据实际选定通道判断左右；不能因规划路线经过分叉就直接标成“看到了分叉”。旧 pilot 的分叉记录仅在派生副本中由原始 RGB-D 重算，原始控制、位姿和图像不改写。

迁移后校验或重新标注已打包的派生数据，可直接使用相对路径的 manifest。若需要从历史 raw 重新生成路线视图，应显式传入当前包内的 `--layout`；工具先按 raw 的 `source_sha256` 校验场景文件，metadata 中的原机器路径仅作来源记录。不要修改原始 metadata 来迎合新路径。例如在交付包根目录运行：

```bash
"$ISAAC_PYTHON" -m cave.lunar_route_language_v13 \
  --source examples/dataset_smoke/raw_episodes/seed_20261118_chinese_flag \
  --layout examples/dataset_smoke/layouts/seed_20261118 \
  --fork-evidence-revision v13 \
  --output runs/rebound_flag
```

历史 integrity binding 的原报告可通过 `--raw-validation` 重新定位；原路径无法确认时工具会重新核对全部文件哈希，不以文件尺寸代替原始内容校验。此优化不影响普通的完整哈希路径。

例如某条路线实际先看见月球车、再遇到左侧分叉，提示可以写为“找到红旗，它在洞穴深处。进入洞穴，看到月球车之后继续前进，在第一个分叉向左转，最后靠近红旗并停下来。” 对应关键帧分别输出“我看到了月球车，按照提示我应该继续前进。”“我看到了第一个分叉，提示让我走左边。”“我看到了红旗，我应该停下来。” 文本、当前可见性、实际动作及路线合同共同参与校验。

完整索引使用派生轨迹与配套语言目录。策略输入保持固定任务提示、当前 RGB-D 和同轨迹的过去观测。实例遮罩、路线事件和位姿属于监督审计字段。`dataset_validation.json` 记录每条样本的检查结果，接入具体训练器时仍需验证其读取接口。

当前十条样例按布局划分：20261118 为 train（红旗、宇航员、月球车），20261119 为 val（月球车、宇航员、基地），20261120 为 test（四类目标）。一条轨迹对应三条语言改写，最多八个第一人称关键帧说明；其余帧保持 `cot_valid=false`。PPIO 主模型为 `qwen/qwen3.8-27b`，全部样例共享两元估算预算和持久请求缓存。预览、俯瞰地图及标注面板不进入策略输入。采集是碰撞检查的运动学教师，不代表车辆轮土动力学或学习策略成功率。

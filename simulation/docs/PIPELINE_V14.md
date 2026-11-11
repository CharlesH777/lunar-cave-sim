# 可复用的复杂指令导航数据流水线

V14 在固定洞穴几何中随机放置宇航员、月球车、基地和红旗，使用同一套配置规划、采集、标注、验证和发布。当前地图的尺寸与拓扑来源、Isaac Sim 5.1 环境及运动学边界见 [基础说明](PIPELINE_V11.md)；平滑行进、侧墙间距和直接停车规则见 [V13 运动说明](PIPELINE_V13.md)。本版增加经过地标的可见路线、靠近后触发的语义说明、复杂 query 家族隔离、每隔五帧的训练视图和远端原子发布。

## 一条可移动的数据单元

单元入口是 `lunar_dataset_pipeline_v14.py`，配置为 `configs/lunar_dataset_pipeline_v14.json`。它依次执行随机布局、完整原始 RGB-D、原始完整性检查、语义路线派生、PPIO 标注、标准数据验证、稀疏训练视图及 query 分区索引。阶段以代码、配置、输入和全部产物哈希绑定；只有完整且一致的阶段可 `--resume`，不会自动重新采集失败的 GPU 阶段或重付语言请求。

```bash
"$ISAAC_PYTHON" lunar_dataset_pipeline_v14.py \
  --config configs/lunar_dataset_pipeline_v14.json \
  --output /path/to/new_unit \
  --isaac-python "$ISAAC_PYTHON" \
  --ppio-env /secure/path/ppio.env \
  --provider-cache /path/to/shared_provider_cache
```

配置中的 `terrain_template` 在交付包内指向 `templates/lunar_fork_v11`；工作区试运行可显式改为实际已有的地形目录。`--layout-source` 可采用一个已完成且 seed 匹配的布局，`--raw-source` 可采用该布局对应的完整原始轨迹，两者仅允许单 episode 配置。采用前后验证文件哈希与场景绑定，保留来源记录；不改写原始数据。`--skip-llm` 用于不调用供应商 API 的完整性验收，不能把它标成已完成 CoT。

输出单元使用互不嵌套的阶段目录：`dataset/` 为完整密集基底，`training_stride5/` 为稀疏视图，`query_partitions/` 为语言分区，`review/` 为实际预览。`TRAINING_INPUTS.json` 和最终 `status.json` 位于单元根目录。根目录的 `REQUEST_STOP` 会协作转发到当前阶段；阶段缓存、日志和完整验证保存在 `dataset/` 内。这样后续生成索引不会改变已完成视图的阶段哈希。

## 走近后再说明

`collect_lunar_trajectory_v14_isaac.py --via-landmark ID` 使用通用的 `LandmarkPassConfig`，依据实际目标、地标、障碍和相机参数生成候选经过路线。它预测持续移动中的近距离观看机会，再由真实渲染确认，不把几何视线当成物品可辨认证明。显式地标采用 `required` 策略；`auto` 配合 `prefer` 可在有界搜索不可行时记录拒绝原因，使用不承诺该地标的自然路线。两种策略都保留真实几何、侧墙、原地转向和最终 STOP 的检查。

`configs/lunar_semantic_visibility_v14.json` 按物品类别规定当前实例面积、bbox 尺寸、边缘间距、方位及有效深度门槛。原始 48 像素实例检测记录保留；派生语义事件只在图像清楚且符合五帧网格时触发。例如月球车要求 RGB224 实际实例面积至少 700 像素、bbox 至少 32×24、距画面边缘至少 8 像素、方位不超过 30°。这是一套可配置的操作门槛，不是通用视觉识别定理。

基地使用 800 像素、40×28、边缘至少 8、方位至多 30° 的类别门槛。大物体近景允许一个严格限定的裁切 STOP 例外：真实终止帧、目标此前已在规则网格上严格清晰可见、当前至少 2500 像素且 bbox 至少 64×48、深度和方位仍通过，且失败原因只有边缘。这不会放宽远处小物体的首次触发条件。

`cave.lunar_route_materializer_v14` 根据实际可辨认地标、当前分叉深度证据、洞口跨越及终点事件建立固定路线提示，保留不可变原始图像和控制。CoT 只在对应真实帧说明任务相关事件；不会给小到难以辨认的目标补写“我看到了”。没有合格地标时，不会把该地标加入提示词。

## 复杂 query 与单场景评测

新配置使用 `configs/lunar_query_protocol_v15.json`，默认一次请求生成二十条有家族标识的 query。每条由匿名路线段和逻辑段组成：路线段不泄露具体目标或停车答案，逻辑段绑定真正目标、靠近后停止和途经点角色等条件，两段合起来保持完整真实路线。段落、算子及家族元数据不进入策略输入，策略只看到拼接后的自然语言。旧版三条 query 与 V14 表达模板接口仍保留。

物理布局的 train/val/test 与 query 家族的 train/heldout 分开记录。八种单逻辑算子各有两种训练句法；四个留出家族组合训练中单独见过的算子，并由结构和真值守卫核对。`query_eval` 仅使用物理训练集中的已知路线和画面，评估未参与训练的表达组合；物理 val/test 另外索引。专用索引不重复计入 canonical 别名，也不把 heldout 指令加入默认训练索引。

## 五帧采样保留真实动作

`training_stride5` 取原始索引 0、5、10……并额外保留唯一末帧 STOP。观测通常间隔一秒，动作仍保留原始 0.2 秒时间步、future8 和到下一观测之间的所有控制；不会平均速度、跳掉中间转弯或把视频播放速度当作车辆速度。历史只取同一语言轨迹的过去采样帧。CoT 必须对应当前原始图像，不能把被跳过帧的文字贴到新帧。

`TRAINING_INPUTS.json` 指定默认训练索引和独立评测索引。稀疏视图以相对路径和哈希引用同一单元中的完整基底，整单元移动后可重新校验。它仍是独立数据契约；现有129训练器的适配与文本输出/loss 接口未因此自动完成。

## 上传和恢复

`upload_lunar_dataset_v14_cpu.py` 对完整单元建立逐文件 SHA 清单，上传到远端暂存，使用仅依赖 Python 标准库的检查器核验，再原子发布并保存收据。同一状态只能复用相同源内容和目标；缺失、损坏或源内容变化会拒收。发布器不删除本地文件。批次调度仅在完整单元已远端验证后才允许回收自己的大体积帧文件，保留检查、索引、来源和收据；未上传的数据不会删除。

当前批次目标是本地生成一百条，上传到 a100 的独立目录 `/data/zhenghengcao/lunar_cave_query100_20261111_v14`。远端只接收与校验文件，不运行 Isaac、模型训练或 GPU 评分。凭证、SDK、模型权重不在交付包中。

批次入口为 `run_lunar_dataset_batch_v14.py --config configs/lunar_dataset_batch_v14.json`。默认仅准备计划；部署时显式指定 `--output`、已有 `--isaac-python`、外部 `--ppio-env`、共享 `--provider-cache`，并加 `--run` 执行。默认二十五个随机布局各四个目标，共一百条；二十个布局用于 train、三个 val、两个 test。工作区与交付包的地形路径不同，运行配置必须指向实际模板；本地 experience、extension cache 和运行库路径也在部署配置中提供。采集最多双路，资源不足时等待，未成功发布的单元不会被清理。标准库发布器不依赖 Isaac 环境。

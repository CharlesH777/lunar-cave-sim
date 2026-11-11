# 月洞导航数据生成流水线 V11

这份交接包括地形与材质、障碍与随机目标布局、真实连续 RGB-D 轨迹、语言 query 与逐步短决策说明、校验和可移植交付。数据域固定为 `isaac_continuous`。通过数据校验只说明观测和教师标签一致；`training_ready=false`，现有129训练器的独立数据适配与逐步文本输出/loss接口仍需实现。

## 交付内容与运行边界

| 内容 | 入口/目录 | 说明 |
|---|---|---|
| 地形与障碍模板 | `templates/lunar_fork_v11/`（交付包） | 不含四个搜索目标的最终地形；含真实 USD、mesh、碎石、碰撞报告及材质引用 |
| CPU 地形工具 | `make_lunar_fork_v11_configs.py`、`build_lunar_cave_v4_cpu.py`、`build_lunar_surface_detail_v5_cpu.py`、`build_lunar_obstacle_scene_v7_cpu.py` | 从参数生成地形、表面细节和障碍；具体整合命令以 `terrain_build.command` 为准 |
| 随机四目标布局 | `build_lunar_search_demo_v8_cpu.py --asset-revision 10` | 宇航员、小型月球车、小型月球基地、中国国旗；布局 seed 决定位置，导航终点使用各自 `approach_xy_m` |
| 连续采集 | `collect_lunar_trajectory_v11_isaac.py`、`cave/lunar_trajectory_v11.py` | 实际顺序积分的运动学教师、真实 Isaac 当前 RGB-D/instance、严格因果历史 |
| query/短 CoT | `annotate_lunar_trajectory_v11_ppio.py`、`cave/ppio_client_v11.py` | 单轨迹指令固定；语言改写引用同一物理轨迹；逐步公开决策摘要，不保存供应商内部推理 |
| 编排/校验 | `lunar_dataset_pipeline_v11.py`、`validate_lunar_dataset_v11.py` | 顺序执行、按源和产物 hash 复用已完成阶段、拒绝静默重启失败阶段 |
| 完整性与环境 | `scripts/verify_delivery_v11.py`、`scripts/check_pipeline_runtime_v11.py` | 前者校验交付文件 SHA256；后者只读检查运行环境，不启动 GPU/Isaac、不安装包 |
| 示例与视觉审查 | `examples/dataset_smoke/layouts/seed_20261118/`、`examples/dataset_smoke/`、`review/` | layout复用数据示例内资产；以各自报告判断是否完整；俯瞰图不能放入策略输入 |

月洞长度、分叉拓扑、粗糙度及目标摆放属于受论文约束的合成假设，不是实测月球地下三维重建。论文依据和不确定性见交付包 `docs/evidence/`。地面连续起伏和局部岩石障碍保留；没有从地表驾车下降竖井的任务，车辆从坑底 `task_spawn` 开始。

## 环境检查：先选解释器

流水线 Python 工具要求 Python ≥3.10；**Isaac Sim 5.1 的二进制 Python 环境要求准确的 Python 3.11 ABI**。官方5.1文档还规定 Linux pip wheel 需要 GLIBC≥2.35；不要把系统 Python 3.6、普通 Python 3.10/3.12 或独立 USD wheel当成已配置的 Isaac 运行环境。[NVIDIA Python 环境说明](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)

NVIDIA 的5.1 x86_64表列出 Ubuntu22.04/24.04、Windows10/11，32GB RAM、50GB SSD，GPU参考最低规格为RTX4080/16GB；测试驱动为 Linux **580.65.06**、Windows **580.88**。官方特别说明无RT Core的A100/H100不支持。驱动表给的是测试版本，不能推导“所有旧驱动必失败”或“任意新驱动必成功”；本项目仍须做实际传感器 smoke 验收。[NVIDIA Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)

在项目/解包根目录，先设置部署方已有的解释器与可选 experience。下面路径是占位符，需要对应实际已配置环境；这些命令不安装或升级主机。

```bash
export ISAAC_PYTHON=/path/to/isaacsim/python.sh
export ISAAC_EXPERIENCE=/path/to/isaaclab/apps/isaaclab.python.headless.rendering.kit
export ISAAC_EXT_FOLDER=/path/to/optional/extension/cache
"$ISAAC_PYTHON" scripts/check_pipeline_runtime_v11.py --mode all --output runtime_preflight.json
```

没有额外 experience/extension cache 时不要设置后两个变量，允许 `SimulationApp` 使用自身默认 experience。也可给 collector 显式传 `--experience` 和可重复的 `--ext-folder`。不需要把作者机器的 `/home/charles/...` 路径复制到新服务器。已有 Isaac Lab/Conda 环境应保留SDK配套的 numpy/pxr；`requirements-pipeline-v11.txt` 是CPU依赖清单，不能据此盲目替换SDK依赖。

预检报告包含选定解释器的版本、NumPy/SciPy/trimesh/Pillow/PXR/scikit-image/matplotlib 的CPU导入结果、OS/GLIBC、Isaac包可发现性、experience路径检查、只读 `nvidia-smi` 清单，以及凭证是否存在的布尔值。它不导入 `isaacsim`、不创建CUDA/RTX上下文、不调用API，不输出凭证值。`--mode cpu --no-nvidia-smi` 可只检查CPU环境。脚本本身兼容Python3.6语法，旧解释器可运行诊断，但不能据此运行数据流水线。退出0表示选定模式预检通过，退出1表示报告中仍有阻塞；`render_compatibility_verified` 始终为false，实际渲染还需下一步验收。

### 2027的已观察状态

2026-11-10的只读检查记录见交付包 `docs/evidence/REMOTE_2027_PREFLIGHT.json`：服务 `115.156.156.22:2027`，hostname=`user`；Ubuntu18.04.3LTS/kernel5.4.0-150-generic；`/usr/bin/python3` 为3.6.9；四张RTX3090，驱动550.144.03；`/data/charles` 当时可用41,254,182,912字节。Docker、Conda、Python3.11不在该次PATH；在检查的常见路径中未找到Isaac。搜索有明确范围，不能据此宣称整机绝对没有任何环境；GLIBC当次未查询，不能凭OS推断为已测值。

因此“完整文件已上传”与“2027能生成真实RGB-D”是两个独立状态：旧Python不满足工具要求，OS/驱动不在本次5.1验收基线，常见路径也没有找到可用SDK。**不能把上传和SHA256通过写成远端GPU运行通过。** 这里不要求升级宿主、修改驱动或触碰现有训练。可在兼容的独立RTX运行环境生成并验证数据，再把完整数据和流水线放到2027；或先由部署方提供合适环境，再重跑预检与smoke。2027现有Python3.6可先运行交付完整性检查：

```bash
python3 scripts/verify_delivery_v11.py .
python3 scripts/check_pipeline_runtime_v11.py --mode cpu --no-nvidia-smi --output runtime_system_python.json
```

第二条在旧环境退出1是准确诊断，不是数据包损坏；不要为通过检查而删改阻塞字段。

## 运行顺序

所有输出使用新目录；不要覆盖旧图、旧训练或旧交付。默认配置 `configs/lunar_dataset_pipeline_v11.json` 的 `terrain_template` 必须指向实际模板：工作区可用 `outputs/lunar_fork_v11/terrain_final`，交付包用 `templates/lunar_fork_v11`。布局和目标随机种子在配置中，不能把已有目标的layout当成空模板。

1. **CPU与短传感器验收。** 先运行纯CPU回归和目标路线检查，再采3步RGB-D。短smoke应以 `max_steps` 正常收尾为partial，不能要求它伪造成功STOP。

```bash
"$ISAAC_PYTHON" collect_lunar_trajectory_v11_isaac.py --self-test
"$ISAAC_PYTHON" collect_lunar_trajectory_v11_isaac.py --source examples/dataset_smoke/layouts/seed_20261118 --output runs/cpu_flag_check --target chinese_flag --cpu-validate
"$ISAAC_PYTHON" collect_lunar_trajectory_v11_isaac.py --source examples/dataset_smoke/layouts/seed_20261118 --output runs/sensor_smoke --target chinese_flag --max-steps 3 --capture-stride 1
```

检查 `status.json`、RGB、depth、K、instance与N/N+1对齐；同时检查 `renderer_settings.json` 和实际日志。采集器请求FXAA、关闭DLSS/DLSSG/运动模糊、缩放比1，在warmup后及每次采集前重申，并保存前后readback；若设置被覆盖会拒收。256×256数组本身不证明RTX内部以该分辨率着色，报告不会伪称已直接测量内部buffer分辨率。本地第三版3帧smoke已确认设置前后readback一致，日志未再出现DLSS低分辨率警告；这项短验收仍不等于整条轨迹成功。

2. **先做一个完整episode，无API。** 编排器生成seed对应四目标layout，再采选定完整轨迹，生成无文字侧车并校验。

```bash
"$ISAAC_PYTHON" lunar_dataset_pipeline_v11.py --config configs/lunar_dataset_pipeline_v11.json --output runs/one_episode_no_api --isaac-python "$ISAAC_PYTHON" --max-episodes 1 --skip-llm
```

`--skip-llm`明确不调用供应商API，`cot_valid=false/output=null`，不得把模板句当成已生成CoT。成功应有 `status.json phase=complete`、`dataset_manifest.json`、`reader_index.jsonl`、`validation/report.json passed=true`。即使这些通过，数据仍是独立 `isaac_continuous` 契约。

3. **完整布局/目标与语言标注。** 默认配置是seed20261118/19/20分别归train/val/test，每seed四个目标，共12条物理轨迹。先在本地实际验收配置，再运行所需规模。凭证文件放仓库之外，支持 `PPIO_API_KEY` 或既有 `ANTHROPIC_API_KEY`；不要把密钥加入包、日志或manifest。

```bash
"$ISAAC_PYTHON" lunar_dataset_pipeline_v11.py --config configs/lunar_dataset_pipeline_v11.json --output runs/dataset_with_text --isaac-python "$ISAAC_PYTHON" --ppio-env /secure/path/ppio.env
"$ISAAC_PYTHON" validate_lunar_dataset_v11.py --dataset runs/dataset_with_text --require-cot --output runs/dataset_with_text/dataset_validation.json
```

编排器默认会调用API。配置中的主模型 `qwen/qwen3.8-27b` 生成query与短说明，仅被事实/schema guard拒收的公开响应再交 `qwen/qwen3.8-max`；每episode默认最多16个关键帧。`annotation.budget_cny=2` 是共享provider cache对应的代码估算总预算上限，并非每条轨迹另给2元；价格估算表以客户端当前配置为准，账单以供应商为准。可使用 `--all-frames`，但必须先调整并审核预算；不在同一次实验里悄悄扩大调用量。失败请求不会自动重试，`.pending.json` 表示请求可能已计费，需要保留并显式核查。

4. **按需单独标注或校验。** 独立执行时必须沿用真实轨迹，不得从四张展示图拼时序。

```bash
"$ISAAC_PYTHON" annotate_lunar_trajectory_v11_ppio.py --source runs/one_episode_no_api/episodes/seed_20261118_astronaut/trajectory --output runs/standalone_language --split train --ppio-env /secure/path/ppio.env --budget-cny 2 --max-llm-frames 16
"$ISAAC_PYTHON" validate_lunar_dataset_v11.py --episode runs/one_episode_no_api/episodes/seed_20261118_astronaut/trajectory --annotations runs/standalone_language --layout runs/one_episode_no_api/layouts/seed_20261118 --require-cot --output runs/standalone_language/validation.json
```

从配置首个目标或实际manifest核对episode目录，不能假定任意目录都有上述名字。地形重新生成可通过编排器 `--build-terrain`，前提是配置的 `terrain_build.command` 指向交付中实际存在且验收的整合脚本；默认直接使用已打包模板，避免依赖作者工作区隐含中间产物。

## 轨迹与标签契约

设执行动作数为N。教师从原始 `task_spawn` 出发，在保留障碍的A*路线中进行有限速原地转向/直行，按 `dt=0.2s` 精确unicycle积分；`0≤v≤0.4m/s`、`|omega|≤0.8rad/s`。没有将路径节点瞬移成轨迹，没有轮土动力学或学习策略成功率的结论。终点为目标的原始 `approach_xy_m`，并在末端朝向物体；只有距该点≤0.05m、当前真实目标mask≥12像素时才接受最终零速度STOP。接近目标≤1m的独立STOP专家训练正例、<3m闭环通过标准均是后续训练/评测约定，不由此终止信号冒充生成。

| 文件 | 核心字段/形状 |
|---|---|
| `metadata.json` | domain、source SHA256、layout seed、终止成功/end_reason、N/N+1/帧数、碰撞与相机范围、`training_ready=false` |
| `supervision.npz` | `vw_si float32[N,2]`、`future_vw_si float32[N,8,2]`、`horizon_valid bool[N,8]`、`stop uint8[N]`、`dt_s float32[N]`、`observation_time_s float64[N]`、phase |
| `teacher_only.npz` | `executed_positions float64[N+1,3]`、`executed_yaw_rad float64[N+1]`、`timestamp_s[N+1]`、原始approach与teacher路径；仅监督/核查 |
| `transitions.jsonl` | 每个已执行control的step、前后时间、当前观测ID、实际 `[v,w]`/STOP、前后真实积分pose、碰撞通过状态；仅监督/核查 |
| `frame_records.jsonl` | 每张真实当前帧的身份、固定instruction、`inputs={instruction,observation,history}`、machine evidence、对应已执行控制 |
| `frames/*_rgb256.png` / `*_rgb224.png` | 原始RGB256 uint8；显式Pillow bilinear缩到224，不把224图当原始相机标定 |
| `frames/*_sensors.npz` | depth_m/depth_valid/instance_id、K256/K224、world_from_camera、当前pose/time；读取策略观测只提取允许字段 |
| `language/queries.json` / `annotations.jsonl` | query变体与物理轨迹/split绑定；逐帧 `cot_valid`、可空output、不可变输入/证据、公开响应来源 |
| `dataset_manifest.json` / `reader_index.jsonl` | layout→split、路径/哈希、按帧索引；policy inputs与privileged监督引用分离 |

未来8步标签来自**实际执行序列**，不足8步补零且valid=false；末端STOP为唯一最终静止控制。控制i对应pose i→i+1和当前观测i，不能错用执行后的帧。编排器显式传 `--episode-id seed_TARGET`，防止各目录都叫trajectory时物理轨迹ID碰撞。history最多默认8张严格早于当前、同episode的真实过去观测，不含当前帧、未来帧或其他layout。默认 `capture_stride=1` 为完整训练候选；大于1时缺失当前帧被明确记录，密集数据校验拒收，不能复用上一张图假装当前。`validate_lunar_dataset_v11.load_policy_inputs(dataset, reader_row)` 按allow-list只加载instruction、当前RGB/depth和真实过去history，不能把传感器NPZ内的instance、pose、K或teacher字段整个透传给模型。

相机高度为实际floor+1.5m、水平、90°HFOV；depth为256方图、沿光轴Z的米，K256 fx=fy=cx=cy=128。RGB224单独保存K224缩放。坐标为world Z-up，OpenCV相机右/下/前、列向量变换。控制时间是step×0.2s，与为固定pose渲染而执行的Kit update/subframe时间分开。灯光固定为24000主灯+两盏8000工作灯，归一化、exposure6，dome45、sun3500，不按目标位置额外打光。

### 碰撞与短CoT证据

每步检查实际地形三角网四轮支撑/车腹净空/车身air envelope，执行段扫掠所有floor-level宏石、源 `obstacle_report.json` 中全部≥25cm碎石circle及四个目标circle；原地转向也检查车身。<25cm碎石为保留的视觉细节，没有轮胎碾压/轮土响应模拟。碰撞circle和离散包络是保守几何筛查，不是完整动力学认证，不能写成车辆越障性能结论。

可见性来自同帧instance mask，方向来自当前可见bbox与相机K；不可见时bearing=unknown，禁止目标世界坐标伪造“看到了红旗”。当前洞口状态使用作者定义连接面x=0附近的当前几何触发区；“进入洞穴”只允许前一步到当前真实积分pose实际跨入口，而不是看见某张相似图片就声明发生事件。LLM输入只有固定任务、当前图、可选上一张真实过去图，以及受限核验事实；不发送目标绝对坐标、完整路线、未来动作或精确剩余距离。训练策略输入仍仅允许instruction、当前RGB-D和因果视觉历史，machine evidence不进入策略输入。

短CoT是可公开的四字段外部说明：`progress`、`observation`、`decision_reason`、`action_intent`。观察≤60字符、理由≤80字符；动作意图必须与该步真实控制一致。关键帧优先入口、首次可见、转向变化和末端STOP。未核验局部障碍时不能编“某块石头挡路”；收到STOP指令只能写准备停止，不能声称此前已停车。Schema、当前/此前、可见性、左右、入口事件、意图或因果性任一guard失败即拒收，不能改写机器证据迎合文本。无效/未选中的帧保持 `cot_valid=false, output=null`。

布局seed先划分train/val/test，同一物理轨迹的所有帧与query改写继承同一split，不能按句子随机拆分。现有129协议的Native离散标签不由这些连续速度量化冒充生成。

## 收尾、续跑与交付

在run目录写 `REQUEST_STOP`，编排器会将它传给正在执行阶段；collector保留已完成的RGB-D/controls/N+1 pose并写partial状态，禁止用SIGTERM截断。单独运行collector或语言阶段时在它的output目录写同名文件。已停止/失败阶段不会自动重试；先保留日志查原因，修复后用新output显式执行。`--resume`只复用完整且配置/代码/输入/全部产物hash一致的阶段，不增加训练步数、不默认重采partial。

只有本地真实渲染审查、完整episode、语言/动作/时序校验通过后，才用 `package_lunar_pipeline_v11_cpu.py` 打包全部代码、configs、assets与许可、可重建地形模板、随机layout示例、完整数据示例、原生渲染审查、论文依据和runtime记录。包内不含密钥、模型权重或Isaac SDK。`MANIFEST.json`记录文件SHA256，`DELIVERY.json`记录交付范围；在2027的 `/data/charles` 下上传到新目录并复核manifest，保留上传/验证记录。远端环境仍需独立通过预检和实际GPU smoke，交付文档不得提前标成已通过。

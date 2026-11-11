# Lunar Cave Sim

Isaac Sim 月球熔岩管导航、找物与 RGB-D 数据生成项目。这里保存从 2027 服务器导出的 **V16b 完整包**，包括源码、月洞 USD 场景、程序化物品模型、贴图、示例轨迹、预览、校验器和历史来源记录。

## 获取完整项目

场景、图像、深度数据、视频和轨迹索引使用 Git LFS。克隆后必须下载 LFS 内容：

```bash
git lfs install
git clone https://github.com/CharlesH777/lunar-cave-sim.git
cd lunar-cave-sim
git lfs pull
python3 simulation/scripts/verify_delivery_v11.py simulation
```

原始封包位于 `simulation/`，文件内容保持不变。完整性校验覆盖原包的 9,968 个清单条目；连同清单本身共 9,969 个文件，约 1.87 GB。服务器传输清单和验收记录单独保存在 `provenance/server2027/`。

## 目录

| 路径 | 内容 |
| --- | --- |
| `simulation/cave/`、根目录 Python 脚本 | 地形、布局、导航、渲染、采集、语言标注、发布与校验 |
| `simulation/templates/lunar_fork_v11/` | 月洞场景、几何与依赖贴图 |
| `simulation/assets/` | 宇航员、月球车、基地、红旗模型及素材来源 |
| `simulation/examples/dataset_smoke/` | 一条完整的 1,130 帧宇航员路线、RGB-D、动作、语言与查询划分 |
| `simulation/review/` | 原生 Isaac 渲染的资产和场景预览 |
| `simulation/provenance/frozen_sources/` | 历史冻结源码与生成证据，当前入口在 `simulation/` 根目录 |

## 环境与运行

项目要求 **Isaac Sim 5.1 / Python 3.11**。SDK、模型权重和 API 凭证需另外配置，仓库不包含这些内容。CPU 依赖见 [requirements-pipeline-v11.txt](simulation/requirements-pipeline-v11.txt)，不要用通用依赖覆盖 Isaac SDK 自带环境。

在正确的运行环境中先执行只读预检：

```bash
cd simulation
python scripts/check_pipeline_runtime_v11.py
```

新机器应复制 `configs/portable_raw_batch_v16.json`、`configs/portable_annotation_queue_v16.json` 和 `configs/portable_annotation_unit_v16.json` 到自己的配置目录，填写本机 Isaac Python、扩展、输入输出目录、SSH 发布目标和外部凭证文件路径。模板仍含原来的发布主机和目录示例，使用前需修改。凭证放在仓库之外。

主要入口：

- `run_lunar_raw_batch_v16.py`：布局与路线检查、Isaac RGB-D/动作采集。
- `run_lunar_annotation_queue_v16.py`：独立消费原始队列，生成语言、训练视图与查询划分。
- `lunar_dataset_pipeline_v14.py`：单条完整流水线入口。
- `run_demo.py` / `run_demo.sh`：较早的交互/录像 Demo；脚本默认环境为历史路径，需设置 `ISAAC_CAVE_ENV`。

详细用法见 [V16 说明](simulation/docs/PIPELINE_V16.md)、[V16b 修正](simulation/docs/DELIVERY_V16B_FIX.md) 和 [来源说明](simulation/docs/DELIVERY_V16_PROVENANCE.md)。

## 当前范围

包内已验收的是一条完整示例；100 条轨迹、25 个布局属于生产配置计划。移动代理采用带碰撞检查的运动学控制，月球车物品为程序化静态碰撞模型，尚未实现车轮接触动力学或月壤沉陷。现有 129 导航训练器尚需数据适配，包中的 `training_ready=false` 保留原始含义。

本次导入完成了文件完整性与离线检查；没有在 2027 或本机重新启动 Isaac 渲染，也没有运行训练或付费标注。

本机 Python 3.14 的密集示例严格重放有 28 项语义/地标证据差异，尚未全部定位；首项是浮点尾数差异。其余 CPU 自检和真实稀疏视图、查询划分校验通过。详细范围见 [本次验证记录](VALIDATION.md)，这不替代指定 Python 3.11 / Isaac 环境的运行验收。

## 素材与许可

第三方贴图来源、作者、许可与哈希保存在 `simulation/assets/ASSET_SOURCES.json`、`simulation/assets/GRAVEL_V5_SOURCES.json` 和 `simulation/templates/lunar_fork_v11/ASSET_LICENSES.json`。其中 Poly Haven 素材标注为 CC0；请同时保留各来源文件。原项目未附整个项目的统一开源许可证，本仓库未额外授予开源许可。

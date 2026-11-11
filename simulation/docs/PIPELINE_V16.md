# 固定月洞、随机物品、复杂语言导航数据 V16

V16把物理采集和语言标注拆为两个独立进程。洞穴几何保持不变，每个布局随机放置宇航员、月球车、月球基地和中国红旗。生产计划25个布局、每布局4个目标，共100条轨迹；每条20个复杂query。物理划分为20/3/2个train/val/test布局，同一布局的四条轨迹保持相同划分。

## 入口

1. `run_lunar_raw_batch_v16.py --config RAW_CONFIG --output RAW_ROOT --run`：CPU先检查布局和可通行轨迹，再用本地Isaac Sim采集。最多两路渲染，启动前检查主存、显存和磁盘余量。完整原始数据通过RGB-D、动作、碰撞和目标到达检查，上传a100并校验SHA后，才生成`queue/ready/<id>.json`。
2. `run_lunar_annotation_queue_v16.py --config ANNOTATION_CONFIG --run`：CPU独立消费ready队列，复核来源哈希、真实动作和布局划分，生成语言、稀疏训练视图、query划分及预览，校验后发布`units/<id>`。全部100条通过后发布`catalog_v16`。
3. `lunar_dataset_pipeline_v14.py --config UNIT_CONFIG --output UNIT --layout-source LAYOUT --raw-source RAW --provider-cache CACHE --ppio-env ENV`：单条端到端入口，可复用已完成轨迹。

轨迹元数据中的`seed`记录采集随机数种子；`source_layout_seed`记录物理布局种子，后者应与`targets.json`、轨迹身份和实际场景SHA一致。两种种子允许不同，不能据此误判场景不匹配。

配置模板见`configs/lunar_raw_batch_v16.json`、`configs/lunar_annotation_queue_v16.json`。使用主机实际路径配置Isaac Python、SDK扩展、固定地形模板、PPIO密钥文件和共享账本；密钥不在资产包内。原始生产队列与标注输出必须为两个独立目录。

## 语言与时间约定

query采用`configs/lunar_query_protocol_v15.json`的20个类型化family，其中16个用于训练，4个保留组合用于query泛化评估。`cave/lunar_query_compiler_v15.py`按当前轨迹实际事件顺序填充`configs/lunar_query_bank_v15.json`：路线主句使用匿名终点，条件句绑定真实目标、地标和接近后停止条件。模板来源包括经Max校验的表达及明确标注的程序语法；生成记录保留模板槽位证据，不伪称每条均为新API输出。

CoT使用`qwen/qwen3.8-27b`读取当前真实图像，关键帧通过语义可见性、面积、框尺寸和视角检查。先看清物品才触发“看到了”的短句；禁止未来图像、坐标或教师路径进入模型观测。共享PPIO账本预算上限为12元，已有响应不重置。

保留原始0.2秒动作。训练观测每5帧采样一次，动作仍为原始控制序列；每个训练锚点提供8步未来动作与因果历史。终点STOP唯一保留，不以平均动作或加快车速替代。`training_stride5`和`query_partitions`与密集`dataset`为同级目录。

## 失败、恢复及存储

物理GPU失败会保存现场并请求协作停止，不自动重试。语言失败标记`needs_review`，继续处理其他完整轨迹，禁止触发生产者STOP或重复付费重试。原始队列最多积压8条（包括在途预留），达到上限等待消费者。

每个发布单元均为完整文件树，经SHA清单、远端完整校验和原子改名后产生receipt。消费者可用已校验原始发布预置相同SHA、大小、权限和时间戳的文件；最终完整校验仍必需。所有远端最终单元独立可读。

本地空间不足时，仅在原始和最终远端receipt同时绑定后回收当前队列自己拥有的帧文件；保留前4条审核单元、CoT当前/前帧、元数据、清单和预览。未上传或校验失败的数据不删除。旧版本源包、失败轨迹和停止标记保留。恢复必须显式`--resume`且代码、配置和输入哈希一致。

## 使用范围

这是`isaac_continuous`域的生成流水线，包含地形/物品资产、代码、示例和校验器；SDK、模型权重和凭证需单独配置。现有129导航训练器仍需接入该数据域及语言监督，`training_ready=false`是当前接口状态。预览和物理代理检查不能作为导航模型训练成绩。

# T010.04 本地32帧执行入口

日期：2026-10-16。采用冻结 Q010 SPEC 的 M1/M2 和 L1。当前范围是 episode16 原帧50—81，单段32帧。配置、依赖锁、生成代码及本地writer须先进入真实本地commit。模型与视频在仓外，不进入Git。正式来源URL、远端访问、全30和评分未由此入口完成。

Python：`/Users/zhuanz/.venvs/openwebui/bin/python`。源码：`/Volumes/My PSSD/NIVIDA/DD/生产/track1/source/`。输出：`/Volumes/My PSSD/NIVIDA/DD/实验/T010.04_连续32/`。mesh依赖：`/Volumes/My PSSD/NIVIDA/运行环境/track1-pack/site-packages/`。运行时读取实际版本；OpenCV的5.0发行元数据不代表实际4.13导入模块。

在源码目录执行，命令中的引号须保留：

```sh
/Users/zhuanz/.venvs/openwebui/bin/python check_contracts.py --output '/Volumes/My PSSD/NIVIDA/DD/实验/T010.04_连续32/preparation/synthetic_contracts_new_date.json'
/Users/zhuanz/.venvs/openwebui/bin/python run.py --stage prepare --episode 16 --start 50 --stop 81 --output '/Volumes/My PSSD/NIVIDA/DD/实验/T010.04_连续32'
/Users/zhuanz/.venvs/openwebui/bin/python run.py --stage A --output '/Volumes/My PSSD/NIVIDA/DD/实验/T010.04_连续32'
/Users/zhuanz/.venvs/openwebui/bin/python run.py --stage B --output '/Volumes/My PSSD/NIVIDA/DD/实验/T010.04_连续32'
/Users/zhuanz/.venvs/openwebui/bin/python run.py --stage C --output '/Volumes/My PSSD/NIVIDA/DD/实验/T010.04_连续32'
```

prepare先核源码干净、真实commit/tree和全部输入hash，写run-manifest。A创建一个DNN、一个ORT Session，无warmup或重试；每原帧一次检测和一次HMR。B由本次原始输出计算固定68/45，再跑32对原root/适配root完整rig。C写原号dense数组、NaN/false未处理位置、米制OBJ，保留官方pack_track1组装，只在进程内替换末端write。worker的Python文件读白名单排除旧手标及派生预测，禁止网络和子进程；该审计不是OS syscall追踪。

每阶段使用永久单次启动锁。失败先看`logs/*_receipt`、`*_supervision`、事件和stderr。锁不是可以随意删除的缓存：当前入口不提供模型重试或中段精确续跑。A失败需要总控给出剩余调用和纠正授权；B/C错误只处理受影响依赖，不自动重复模型。原失败日志须保留。单帧HMR异常按本段最近完整有效五输出回填，平分取早；原始文件保持原件。持框不刷新最后命中时间。

`source/final_tracker_state`保留本次完整关联状态和hash。后续82—289扩展若关联输入/状态全部一致，可另实现经批准的精确恢复；本版本未实现240续跑CLI，不能直接把该状态视为已完成恢复。CLI具有episode/start/stop参数，当前config只授权16/50/81。扩大范围需新任务卡、更新资产与预算配置、审阅stage内32/64约束、先提交新commit，再运行到新目录。其余29条的下载器、全30生产和五榜发送均未在此实现或运行。

原始五输出在`hmr/raw_*`。自身身份和填补来源在`identity/`；rig完整网格/关节与残差在`rig/`；head70代理和固定尺寸在`proxy/`；官方packer输入在`episodes/`；局部sample、Parquet和分列验收在`pack/`。预选叠图为50/60/70/81，不按图调整规则。未知真实接触、尺度、表面及head70/rig实体差异属于质量状态。

局部产物`track1_local32_not_for_submission.parquet`严格五列，`code_commit_url=local-git:<本次真实P>`，sidecar标`submission_eligible=false`。local writer核P、manifest、repo一致，保留one_to_one、finite及sample顺序。官方磁盘源码与regex不变。局部数值/索引/mesh通过不代表official writer、正式URL、远端可访问、全30或Kaggle评分通过。

经另行授权发布同一P并完成全量输入后，用已固定官方packer及真实URL重新pack。若序列化代码改变，先新commit C，再实际重pack并比较逐行xyz，保留预测生产P。当前没有上传动作或GitHub远端仓库。

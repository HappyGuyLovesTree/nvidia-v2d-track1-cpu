# T011.06 tracker API

日期：2026-10-17。采纳 R011 SPEC a2 的 R11-TRK01—04。范围仅关联模块、六张量检查和最小 worker 接入。全流水线由 T011.07 实施。

## 调用

`tracking.LowScoreTracker(identity=identity, numeric_config_hash=hash)` 不读写文件、不加载模型。identity 必须包含 video_sha256、detector_sha256、postprocess_sha256、runtime_sha256，值为实际64位小写SHA256；可附加 wholePTS、预处理、代码producer等身份，恢复时完整映射严格相等。numeric_config_hash 是调用者冻结的数值合同 hash；模块另外核自己的 RULES_HASH。调度范围/输出路径不应混入数值合同。hash不是替代实际输入核验的凭据。

`step(fid, candidates, w, h, *, episode, source_segment, pts, time_base, input_parents) -> decision`。fid/pts/w/h 为 Python int，episode/source_segment 为 int 或 str，time_base 为可被 Fraction 读取的正有理数。input_parents 是非空的实际父SHA256映射；生产应含本帧 raw/RGB/blob 身份，T07还须在帧事务中绑定wholePTS及完整输入。candidates沿 common.candidate 的字典格式：index/xyxy/score/class_id/valid，index唯一且为非负int。框先裁剪，面积/排序统一float64；标valid=True却shape/finite/面积非法时明确失败，标valid=False的可序列化拒绝记录保留供审计。非法输入不改变tracker状态，也不修改传入候选。

同episode要求尺寸一致及PTS严格递增；episode/source切换或原帧不加1时清关联状态。source_segment由T07按清单传入真实段号，不能拿track_segment代替。新episode可从新的PTS起点开始。track_segment在本episode内逐次初始化/重启递增；新episode从0开始。

`export_state() -> {'state': complete_state, 'state_sha256': hash}` 返回独立JSON副本。state包括schema/method/numeric_config_hash/rules_hash/完整identity、episode/source_segment、previous_frame/previous_pts/time_base/width/height、last_bbox/last_observed_frame/lost_count/track_segment、last_high_frame/consecutive_low_matches和last_input_parents。两个低分审计字段没有隐藏链长上限。

`LowScoreTracker.restore_state(checkpoint, *, identity, numeric_config_hash, expected_state_sha256, expected_input_parents, expected_context) -> tracker`。expected_context必须恰含episode/source_segment/previous_frame/previous_pts/time_base/width/height。expected_state_sha256、context及parents须来自已核验的帧提交记录，不可从未信任的待恢复文件直接照抄。仅核自带hash不能证明来源。旧__dict__/旧method、schema、数值/身份/父链/上下文不符及状态结构不合法均拒绝。父hash改变须重新验证受影响层，不能改expected迎合文件。

## decision与worker

兼容字段：frame_index/source_segment/source_discontinuity/track_segment/branch/bbox/score/selected_index/lost_count/last_observed_bbox/last_observed_frame/true_gap/reason/candidates。新增：method/episode/selected_tier/selected_iou、last_high_frame/consecutive_low_matches、input_parents/tracker_before_sha256/tracker_after_sha256。每个有效候选保存tier/iou/initial_key/association_key。branch沿旧名，低分真实命中为associated_low_detection；high仍为associated_detection。held的score=null且不刷新命中帧。

worker.stageA按CFG.method选择旧common.Tracker/.35或新LowScoreTracker/.10，NMS保持.60；原config默认旧方法，T07新配置须显式选择single_iou_low_v1。配对decoder、topk、argmax、跨类NMS、逆变换及原模型文件均保持。检测后处理前调用validate_raw，严格要求六个真实float32/finite张量及原shape，dict须恰为output_0..5。

新方法的每帧decision附tracker_after完整导出；最终文件的外层state值为导出envelope，不能把它当旧__dict__。T07应直接将export_state绑定原子帧终态；HMR已知失败也可保留该检测后的合法状态。tracker不保存HMR结果、不管理进程锁/attempt/事务。当前worker仍为旧32帧/单source管线，最小接入暂传source_segment=0，未实现运行恢复或全量调度；T07须替换真实source段映射并绑定实际数值身份/稳定commit及输入校验。

## 局部检查

Python `${PYTHON_EXECUTABLE}`，cwd `${LOCAL_PROJECT_ROOT}/DD/生产/track1/source`：

```sh
PYTHONDONTWRITEBYTECODE=1 ${PYTHON_EXECUTABLE} check_tracking.py --root '${LOCAL_PROJECT_ROOT}' --output '${LOCAL_PROJECT_ROOT}/DD/证据/T011.06/local_checks_20261017a2.json'
```

输出采用exclusive-create，已有文件不可覆盖，应按日期序号递增。检查13组合成行为及多个断点、冻结32raw逐帧decision/基线/16变框及5个真实回放断点；测试identity是独立合成值，raw和旧decision按Q011.02冻结hash核验。NanoDet.__new__仅构建后处理元数据，绝不调用构造器/readNet/forward或图像视频解码。

通过仅说明机制和接口一致；不证明选对人、HMR改善、3D质量、全量生产恢复或240/全30通过。

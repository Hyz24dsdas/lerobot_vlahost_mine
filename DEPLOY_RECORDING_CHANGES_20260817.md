# 部署同步录制改造记录（2026-08-17）

## 目标

保持原部署命令不变：

```bash
cd /home/zzx23457/lerobot_vlahost-main
conda activate lerobot_vlahost
python workflows/robot_interaction/deploy.py
```

每次部署自动同步录制状态、实际控制命令、夹爪命令和相机；按 Ctrl+C 后继续录制机器人回标准 Home 的过程。只有 Home 反馈校验成功才询问是否保存。直接关闭终端、异常退出、Home 失败或选择不保存时，停止录制并删除该次临时数据。

## 修改文件

1. `workflows/robot_interaction/deploy.py`
   - 为每轮部署生成唯一的 LeRobot 暂存数据集目录。
   - 强制 `episodic` 使用本地保存、单 episode、实时 H.264 编码，不上传 Hub。
   - Ctrl+C 时由 `episodic` 先停止推理，再保持 episode 打开并记录回 Home。
   - 增加交互式“是否保存本次评测录制的数据集？”询问。
   - 保存前校验 Home 成功标记、LeRobot v3 版本和必需 feature。
   - 保存前把 `robot_type` 和 16 维 feature 名称规范化为参考数据集的 `marvin` / `Joint*_L/R` 契约。
   - 不保存、直接关闭终端、异常退出或 Home 失败时，只删除本轮精确匹配的临时目录。
   - 增加独立 watchdog，外层脚本即使被强制关闭也会停止录制并丢弃未完成评测。

2. `src/lerobot/rollout/configs.py`
   - 为 `EpisodicStrategyConfig` 增加“退出时继续录制 Home”的可选参数；默认关闭，不影响其他用法。

3. `src/lerobot/rollout/context.py`
   - 仅在本评测模式启用时，为直接录制的数据集声明 16 维 `observation.velocity` feature。
   - 保存当前 chunk 的动作序列与起始时间，供 30 Hz 录制循环进行 action 对齐。

4. `src/lerobot/rollout/strategies/core.py`
   - chunk 成功发给 VLAHost 后，记录机器人实际接受（含安全裁剪）的完整 action 序列及时间基准。

5. `src/lerobot/rollout/strategies/episodic.py`
   - 退出策略循环后先停止推理引擎。
   - 支持 chunk 推理：每个控制周期记录当前计划 waypoint，而不是调用无效的单步接口。
   - 以 30 Hz 下发标准 Home，同时将观测和 Home action 继续加入当前 LeRobot episode。
   - 用相邻状态帧与真实时间差计算 16 维 `observation.velocity`。
   - 连续多帧满足关节误差阈值后才保存 episode、finalize 并写入 Home 成功标记。

6. `workflows/robot_interaction/deploy_config_chunk.yaml`
   - 将 `strategy` 设为 `episodic`，启用 `direct_lerobot` 模式。
   - 配置唯一 LeRobot 输出目录、30 FPS、H.264、CRF 20、GOP 2。
   - 配置 Home 校验容差与超时时间。

## 数据路径和格式

- 每轮唯一暂存/保存路径：`/home/zzx23457/record_files/lerobot_datasets/deployment_evaluations/rollout_eval_*`
- 输出：LeRobot v3.0，16 维 `observation.state`、`observation.velocity`、`action`，以及 `top`、`wrist_L`、`wrist_R` 三路 640×480 H.264 视频。

录制实现是运行中直接调用 `LeRobotDataset.add_frame()` 写 LeRobot，不经过 rosbag。为满足 Home 生命周期，`episodic` 在停止推理后继续向同一 episode 写入 Home 动作和观测，Home 校验后才 `save_episode()` 和 `finalize()`。

## 仓库内直接 LeRobot 方法的调查

- `src/lerobot/rollout/strategies/sentry.py`：策略循环中直接调用 `dataset.add_frame()`，退出时 `save_episode()` 和 `finalize()`。
- `src/lerobot/rollout/context.py`：用机器人/策略 feature 创建 `LeRobotDataset`。
- `sentry`、`highlight`、`dagger` 和原始 `episodic` 都在硬件回位前执行 `save_episode/finalize`；`base` 不录制。因此选择对最接近需求的 `episodic` 做可选扩展。

## 恢复与备份

写回服务器前，对六个被替换文件建立带时间戳的 `.codex-backup-20260817-eval-record` 备份。恢复时将相应备份复制回原文件即可。

## 实机验证

- 原命令成功进入 `episodic + chunk` 推理，连续下发 100 帧 action chunk。
- Ctrl+C 后停止推理，保持 episode 打开并完成 Home；末帧 14 个手臂关节最大 Home 误差为 0.00255 rad。
- 验证数据集 `rollout_eval_20260817_164837`：1 episode、2264 帧；三路视频均为 640×480、H.264、yuv420p、30 FPS、2264 帧。

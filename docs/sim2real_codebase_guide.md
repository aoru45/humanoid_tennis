# Sim2Real Codebase 详细解读

本文面向你当前这个仓库的 `sim2real/` 目录，目标是帮助你快速建立以下认知：

- 代码模块怎么分层
- `sim2sim` 和 `sim2real` 两种模式分别怎么跑
- 策略输入输出如何在运行时被构造和下发
- `UDP`/`VR` 两种 motion source 的实现差异
- 配置文件各字段在代码里如何生效
- 常见问题和排查路径

---

## 1. 目录结构与职责

`sim2real/` 关键结构如下：

- `src/deploy.py`：高层控制主入口（真实机器人与 sim2sim 共用）
- `src/sim2sim.py`：MuJoCo 低层仿真与 DDS 桥接
- `src/policy.py`：ONNX 策略封装、obs 拼接、动作映射
- `src/observation.py`：部署侧观测模块实现
- `src/motion_sources.py`：参考动作源（`udp` / `vr`）
- `src/motion_select.py`：UDP 交互式动作切换工具
- `src/common/`：通用工具（关节映射、remote 解码、timer、命令模板、数学工具）
- `config/controller.yaml`：控制器硬件与关节映射配置
- `config/tracking.yaml`：策略路径、obs 配置、motion source 配置
- `assets/ckpts/*`：部署模型（`policy.onnx + policy.json`）

---

## 2. 整体运行架构

## 2.1 两类进程

- 低层状态/执行侧
  - `sim2sim` 模式：`src/sim2sim.py`（MuJoCo 模拟硬件）
  - `sim2real` 模式：真实 Unitree 机体（仓库外硬件）
- 高层策略侧
  - 统一是 `src/deploy.py`
  - 订阅 `lowstate`，发布 `lowcmd`
  - 内部运行 ONNX policy + obs 构造 + motion source 管理

## 2.2 数据流（高层）

1. `deploy.py` 从 `lowstate` 读到机器人状态（关节、IMU、遥控器）
2. 把 real joint 顺序映射到训练时的 Isaac 顺序（`qj_isaac`, `dqj_isaac`, `tau_isaac`）
3. `policy.update_obs()` 在 `observation.py` 中拼接单步 `policy` 向量
4. ONNX 推理得到 `action`（Isaac 动作空间）
5. 乘 `action_scale` 并映射到 real joint 顺序
6. `deploy.py` 发送 `q/qd/kp/kd/tau` 到低层（`tau=0`，位置目标 + PD）

---

## 3. 入口文件逐个看

## 3.1 `src/deploy.py`（高层主控）

核心职责：

- 初始化 DDS 通道（`lowcmd_topic`, `lowstate_topic`）
- 管理遥控器状态机（零力矩 -> 站立初始位 -> tracking）
- 读取低层状态并做关节映射
- 驱动策略更新与动作下发

关键流程：

- `Controller.zero_torque_state()`
  - 等待 `start` 键，期间持续发零命令
- `Controller.move_to_default_qpos()`
  - 2 秒插值移动到初始姿态，使用 `kps_real/kds_real`
- `Controller.default_qpos_state()`
  - 等待 `A` 键进入 tracking policy
- `Controller.run()`
  - 控制循环（`control_freq`）
  - `process_state()` -> `current_policy.update_obs()` -> `compute_action()` -> `_apply_action_real()`

动作下发关键点：

- `_apply_action_real()` 里将 `action_real_delta` 加到 `default_qpos_real`
- 最终每个关节发送：
  - `q = default_qpos_real + delta`
  - `qd = 0`
  - `kp = kps_real[i]`
  - `kd = kds_real[i]`
  - `tau = 0`

这说明部署是标准位置目标 + 低层 PD 执行，而不是直接力矩控制。

## 3.2 `src/sim2sim.py`（仿真硬件替身）

核心职责：

- 用 MuJoCo 加载机器人模型
- 订阅高层下发的 `lowcmd`
- 在模拟器内按 `kp*(q_tgt-q)+kd*(0-dq)` 计算控制量并步进
- 发布 `lowstate` 给 `deploy.py`

它等效于“把真实机器人硬件接口替换成 MuJoCo 实现”，用于安全联调高层策略链路。

关键实现点：

- `cmd_sub_handler()` 收到命令后缓存 `q/kp/kd`
- `simulate_control()` 循环里计算 ctrl 并 clip 到 actuator range
- `state_pub_handler()` 从 `qpos/qvel/sensor` 组装 `lowstate`（含 IMU）
- 键盘映射模拟遥控器（`s/a/x` 等）

## 3.3 `src/policy.py`（策略运行时封装）

分两层：

- `ONNXModule`
  - 加载 `policy.onnx`
  - 读取同名 `policy.json`，拿到 `in_keys/out_keys`
  - 用 onnxruntime CPU 执行
- `Policy` / `TrackingPolicyRaw`
  - 构建 obs modules
  - 聚合为 `policy_input["policy"]`
  - 前向得到 `action`
  - `clip + scale + joint mapper` 输出 real 空间动作增量

部署输入假设：

- 默认输入是 `policy`（加 `is_init`）
- 若模型有 `adapt_hx` 则会走状态缓存逻辑（代码中已兼容）

`TrackingPolicyRaw` 额外管理：

- 引用动作缓冲（`ref_joint_pos/root_pos/root_quat`）
- `future_steps` 索引逻辑
- 与 motion source（UDP/VR）的联动

---

## 4. 观测构造：`src/observation.py`

该文件是部署侧“手工复刻训练 obs”的核心。

主要模块：

- `TrackingCommandObsRaw`
  - 未来参考轨迹的相对位姿特征（位置差 + 旋转6D）
- `TargetJointPosObs`
  - 目标关节角和目标-当前差
- `TargetRootZObs`
  - 未来 root z 序列
- `TargetProjectedGravityBObs`
  - 未来参考姿态下的重力方向
- `RootAngVelBHistory` / `ProjectedGravityBHistory`
  - IMU 历史序列
- `JointPos` / `JointVel`
  - 关节状态历史
- `PrevActions`
  - 历史动作
- `BootIndicator`, `ComplianceFlagObs`
  - 辅助标记量

最终由 `Policy.update_obs()` 依序拼接成一个一维向量，写入 `policy_input["policy"]`。

---

## 5. Motion Source：`src/motion_sources.py`

## 5.1 公共基类 `MotionSourceBase`

负责：

- 从 `tracking.yaml` 的 `motions` / `motion_clips` 加载参考数据
- 按 joint name 做重排映射（dataset -> obs joint order）
- 动作切换时做“尾部对齐 + 过渡段插值 + append”

不是硬切换，而是 append 进 reference buffer，减少突变。

## 5.2 `UDPMotionSource`

特点：

- 使用 `MotionUDPServer`（`common/utils.py`）收 motion name
- 通过 `motion_select.py` 交互式发送命令
- 有切换门控逻辑（非 default 动作不能随意跳）

适合离线动作回放和安全验证。

## 5.3 `VRMotionSource`

特点：

- 通过 ZMQ 与 teleop 桥通信（req/rep/control 三路）
- 使用 low/high watermark 维持 future horizon 缓冲
- 有 start/stop 控制、inflight 请求控制、延迟回复丢弃保护
- 首包会做 yaw 对齐与过渡，降低参考切入冲击

适合在线遥操作流。

---

## 6. 配置文件怎么驱动代码

## 6.1 `config/controller.yaml`

主要决定：

- 控制频率 `control_freq`
- DDS topic 名
- joint name 列表（`isaac_joint_names_state`, `real_joint_names`, `mujoco_joint_names`）
- 默认姿态、初始化姿态
- 默认 `kps_real` / `kds_real`

这些参数被 `deploy.py` 和 `sim2sim.py` 共同使用。

## 6.2 `config/tracking.yaml`

主要决定：

- 使用哪个模型：`policy_path`
- obs 维度相关超参：`future_steps`, 历史 step 列表, `prev_action_steps`
- 动作后处理：`action_clip`, `action_scale`, `action_joint_names`
- motion source 选择：`motion_source: udp|vr`
- VR 通讯地址与水位策略：`vr_*`
- 离线动作库列表：`motions`, `motion_clips`
- policy 级默认 `kps_real/kds_real`（进入策略后会覆盖 controller 默认）

---

## 7. 关节映射机制（很关键）

文件：`src/common/joint_mapper.py`

- `JointMapper` 通过 joint name 交集自动建立索引映射
- 支持：
  - action 从 source -> target
  - state 从 target -> source
- 如果某侧有多余关节，未映射部分自动留默认值/0

代码中会打印 mapped/unmapped 统计，强烈建议启动时检查。

---

## 8. Remote / 命令模板 / 计时器

- `src/common/remote_controller.py`
  - 解析遥控器按键位，维护 `button[16]`
- `src/common/command_helper.py`
  - `create_zero_cmd`, `create_damping_cmd`, `init_cmd_hg` 等安全命令模板
- `src/common/utils.py`
  - `Timer` 基于 linuxfd timerfd，保证循环频率稳定
  - `MotionUDPServer` 提供轻量 UDP 命令队列

---

## 9. 运行模式对比

| 维度 | sim2sim | sim2real |
|---|---|---|
| 低层对象 | MuJoCo 模拟器 (`sim2sim.py`) | 真实 G1 硬件 |
| 高层对象 | `deploy.py` | `deploy.py` |
| 网络参数 `--net` | 一般 `lo` | 真实网卡名 |
| 遥控器输入 | 键盘模拟 | Unitree 遥控器 |
| 风险 | 低 | 高（需严格安全流程） |

---

## 10. 一次完整的推理链（从状态到电机命令）

1. 低层发布 `lowstate`（关节+IMU+遥控）
2. `deploy.py` 读取并映射到 `isaac` 顺序
3. `policy.py` 调 `observation.py` 各模块拼 obs
4. ONNX 推理输出 `action`（Isaac 空间）
5. `action * action_scale` 后映射到 real joint 顺序
6. `deploy.py` 组合为 `q_target`，附 `kp/kd`
7. 发布 `lowcmd` 到低层执行

---

## 11. 常见坑与排查建议

## 11.1 观测维度不匹配

现象：

- 启动时报 `Observation dim mismatch`

排查：

- 检查 `tracking.yaml` 的 obs 配置是否与导出 ONNX 时一致
- 检查 `policy.json` 的 `in_shapes`/`in_keys`

## 11.2 joint 映射错位

现象：

- 动作方向奇怪、局部关节不动或抖动

排查：

- 看启动日志里 mapped/unmapped joints
- 检查 `controller.yaml` 三组 joint name 顺序与命名一致性

## 11.3 VR 数据卡顿/突变

排查：

- 调整 `vr_low_watermark`, `vr_high_watermark`, `vr_chunk_frames`
- 看 `VRMotionSource` 请求/回复日志是否频繁 drop

## 11.4 真机过冲或振荡

排查与处理：

- 先降 `kps_real/kds_real` 或减小 `action_scale`
- 确认 `default_qpos_real` 与机器人当前姿态一致
- 先在 `sim2sim` 完整验证动作与切换逻辑

---

## 12. 快速阅读顺序（建议）

为了最快建立全局理解，建议按这个顺序读源码：

1. `sim2real/README.md`
2. `src/deploy.py`
3. `src/policy.py`
4. `src/observation.py`
5. `src/motion_sources.py`
6. `config/controller.yaml` + `config/tracking.yaml`
7. `src/sim2sim.py`
8. `src/common/joint_mapper.py`

---

## 13. 与训练侧的连接关系（简版）

- 训练时导出 ONNX（通常 student deploy 路径）后拷到 `sim2real/assets/ckpts/...`
- `tracking.yaml` 的 `policy_path` 指向该 ONNX
- 部署侧只负责 runtime 推理和控制，不包含训练逻辑

这也解释了为什么 `sim2real` 工程依赖集中在：ONNXRuntime + DDS + MuJoCo(仅 sim2sim) + 通讯桥接。


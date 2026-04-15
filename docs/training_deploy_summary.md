# Motion Tracking 训练与部署总览（分阶段/分网络）

> 目标：整理本项目的  
> 1) 每个网络输入输出与各阶段 obs 使用方式  
> 2) 控制方式（仿真 vs 真机）  
> 3) 全部 randomization  
> 4) 全部 reward 及计算公式  

---

## 1. 网络与阶段总览

### 1.1 核心网络（PPOPolicy 内部）

| 网络 | 作用 | 输入 | 输出 | 备注 |
|---|---|---|---|---|
| `encoder_priv` | 将特权观测编码成潜变量（teacher 分支） | `priv` | `priv_feature` | 用于 Stage1 teacher actor；并作为 Stage2 蒸馏标签 |
| `adapt_module` | 学生侧状态估计器 | `policy` | `priv_pred` | 部署侧关键模块；仅依赖可部署观测 |
| `actor_teacher` | teacher 策略 | `concat(policy, priv_feature)` | `loc, scale -> action` | Stage1 rollout 用 |
| `actor_student` | student 策略 | `concat(policy, priv_pred)` | `loc, scale -> action` | Stage3 rollout 用；部署导出一般走这条链 |
| `critic` | 价值函数 | `concat(policy, priv, priv_critic)` | `state_value` | 训练用，不参与部署推理 |

说明：`actor` 的动作分布是 `IndependentNormal(loc, scale)`，采样得到 `action`，并计算 `action_log_prob`。

### 1.2 三阶段训练时各网络更新关系

| 阶段 | 配置 | rollout 路径 | 主要优化目标 | 更新哪些网络 |
|---|---|---|---|---|
| Stage1 `train` | `+exp=train` / `ppo_train` | `encoder_priv + actor_teacher` | PPO +（train 阶段内的特征回归项） | `actor_teacher`, `critic`, `encoder_priv`, `adapt_module`（经 `train_estimator`） |
| Stage2 `adapt` | `+exp=adapt` / `ppo_adapt` | 训练数据来自环境 rollout，但 `train_op` 仅做估计器训练 | `MSE(priv_pred, priv_feature)` | `adapt_module` |
| Stage3 `finetune` | `+exp=finetune` / `ppo_finetune` | `adapt_module + actor_student` | student 路径下 PPO 微调 | `actor_student`, `critic`（`adapt_module`在 PPO 更新里不反传） |

补充：Stage3 前 2.5% 进度代码中会先偏向只更新 critic，然后再更新 student actor。

---

## 2. 每阶段网络输入输出与 Obs 来源

### 2.1 观测组定义（环境层）

| 观测组 | 组成 | 用途 |
|---|---|---|
| `policy` | 可部署观测（命令、IMU/重力历史、关节历史、前序动作等） | student 可直接使用 |
| `priv` | 特权观测（更多目标/当前 keypoint 与动力学信息） | teacher 与训练 critic 使用 |
| `priv_critic` | 额外 critic 特权项（当前配置仅 `cum_error`） | critic 使用 |

### 2.2 分阶段网络 I/O（含 obs）

| 阶段 | 网络链路 | 网络输入 | 网络输出 | 实际 obs 依赖 |
|---|---|---|---|---|
| Stage1 `train` | `priv -> encoder_priv -> priv_feature` | `priv` | `priv_feature` | 依赖 `priv` |
| Stage1 `train` | `actor_teacher` | `concat(policy, priv_feature)` | `action` | 同时依赖 `policy + priv`（经编码） |
| Stage1 `train` | `critic` | `concat(policy, priv, priv_critic)` | `state_value` | 依赖 `policy + priv + priv_critic` |
| Stage2 `adapt` | `adapt_module` | `policy` | `priv_pred` | 仅 `policy` |
| Stage2 `adapt` | 监督目标（非独立网络） | `priv` 经 `encoder_priv` | `priv_feature` 标签 | teacher 特权标签 |
| Stage3 `finetune` | `adapt_module` | `policy` | `priv_pred` | 仅 `policy` |
| Stage3 `finetune` | `actor_student` | `concat(policy, priv_pred)` | `action` | 实际部署路径 |
| Stage3 `finetune` | `critic` | `concat(policy, priv, priv_critic)` | `state_value` | 训练时仍用特权 critic |

### 2.3 `policy` / `priv` / `priv_critic` 具体条目

| 组 | 条目 |
|---|---|
| `policy` | `boot_indicator_state_obs`, `command_obs`, `compliance_flag_obs`, `target_joint_pos_obs`, `target_root_z_obs`, `target_projected_gravity_b_obs`, `root_angvel_b_history`, `projected_gravity_history`, `joint_pos_history`, `joint_vel_history`, `prev_actions` |
| `priv` | `target_joint_pos_obs`, `target_pos_b_obs`, `target_rot_b_obs`, `target_linvel_b_obs`, `target_angvel_b_obs`, `current_keypoint_pos_b_obs`, `current_keypoint_rot_b_obs`, `current_keypoint_linvel_b_obs`, `current_keypoint_angvel_b_obs`, `target_keypoints_pos_b_obs`, `target_keypoints_rot_b_obs`, `root_linvel_b_history`, `root_angvel_b_history`, `projected_gravity_history`, `joint_pos_history`, `joint_vel_history`, `applied_action`, `applied_torque`, `body_height`, `feet_contact_state`, `target_feet_contact_state_obs` |
| `priv_critic` | `cum_error` |

---

## 3. 控制方式（你关心的 KP/KD）

### 3.1 仿真训练侧（MJLab/MuJoCo）

| 项 | 说明 |
|---|---|
| 策略输出 | `action`（高层动作） |
| 动作落地 | `JointPosition`：`pos_tgt = default_joint_pos + offset + action * action_scaling` |
| 低层执行 | 仿真内部 actuator（隐式 PD 语义）执行位置目标 |
| KP/KD 来源 | 来自模型 actuator 参数；`motor_params_implicit` 可随机化 stiffness/damping（以及 armature） |

### 3.2 真机部署侧（sim2real）

| 项 | 说明 |
|---|---|
| 策略输出 | `action_real_delta`（映射到真机关节空间后的目标增量） |
| 动作落地 | `q = default_qpos_real + action_real_delta`, `qd = 0`, `tau = 0` |
| 低层执行 | 真机低层 PD 控制 |
| KP/KD 来源 | `sim2real/config/tracking.yaml` 中的 `kps_real`, `kds_real`；运行时下发到每个电机命令 |

结论：训练与部署都属于“位置目标 + 低层 PD”，但训练侧是仿真 actuator，部署侧是机器人控制器的真实 PD。

---

## 4. 全部 Randomization（当前 G1_tracking 配置）

> 调用时机：环境初始化时 `startup`、reset 时 `reset`、部分项在每步 `update/step` 生效。  
> 以下按 `cfg/task/G1/G1_tracking.yaml` 的 `randomization` 区域。

| 名称 | 参数（配置） | 作用对象 | 采样/计算方式 | 典型时机 |
|---|---|---|---|---|
| `perturb_body_com` | `body_names`, `com_range` | 指定刚体 COM | `body_ipos = default + U(com_range)` | `startup` |
| `perturb_body_materials` | `static_friction_range`, `solref_time_constant_range`, `solref_dampratio_range`, `homogeneous` | 接触几何材质 | 摩擦/solref 按区间采样（含 log-uniform dampratio） | `startup` |
| `motor_params_implicit` | `stiffness_range`, `damping_range`, `armature_range` | actuator 增益与关节 armature | 按模式匹配后 log-uniform 采样比例，乘默认参数写回 | `startup`(armature), `reset`(PD gain) |
| `random_joint_offset` | `.*: [-0.01, 0.01]` | 动作偏置 `action_manager.offset` | 每个关节采样 `U(low, high)` | `reset` |
| `motion_tracking_target_joint_pos_bias` | 按关节模式给 `noise_std` | tracking 目标关节位置偏置 | `bias = U(-1,1) * std` | `reset` |
| `motion_tracking_root_drift_vel` | `xy_max`, `z_max` | tracking 参考根速度漂移 | `xy` 在圆盘内采样，`z` 均匀采样 | `reset` |
| `motion_tracking_root_z_offset` | `z_offset_range` | tracking 参考根高偏移 | `z_offset ~ U(low, high)` | `reset` |
| `perturb_root_vel` | `min_s/max_s` + xyz/rpy范围 | 机器人根速度（线速度+角速度） | 按时间间隔触发，给根速度加随机增量 | `update`（周期触发） |
| `perturb_gravity` | `mean`, `std` | 每环境重力向量 | `gravity = mean + spherical_noise(std)` | `startup` |

---

## 5. Reward 总表与公式

## 5.1 奖励汇总结构

| 组 | 子项（当前启用配置） |
|---|---|
| `tracking` | `root_pos_tracking`, `root_rot_tracking`, `root_vel_tracking`, `root_ang_vel_tracking`, `keypoint_tracking`, `keypoint_vel_tracking`, `keypoint_rot_tracking`, `keypoint_angvel_tracking`, `lower_keypoint_tracking`, `upper_keypoint_tracking`, `joint_pos_tracking`, `joint_vel_tracking` |
| `loco` | `survival`, `joint_vel_l2`, `action_rate_l2`, `feet_air_time_ref`, `feet_air_time_ref_dense`, `joint_pos_limits`, `joint_torque_limits`（`impact_force_l2/feet_slip/feet_contact_count` 在该配置里未启用） |

环境最终 reward 张量为各组拼接；PPO 侧会对组和再做 `clamp_min(0)` 后参与 GAE。

## 5.2 通用记号

- 误差指数核：  
  `R_exp(error; sigma_list) = mean_{sigma in sigma_list}(exp(-error / sigma))`
- 大部分 tracking 奖励形如 `w * R_exp(error, sigma)`
- `w` 为配置里的 `weight`

## 5.3 Tracking 组公式

| 子项 | 误差定义（按代码语义） | 奖励 |
|---|---|---|
| `root_pos_tracking` | `error = ||p_ref - p_cur||_2` | `w * R_exp(error, sigma)` |
| `root_rot_tracking` | `error = ||axis_angle(q_ref * conj(q_cur))||_2` | `w * R_exp(error, sigma)` |
| `root_vel_tracking` | 在 body frame 比较线速度：`error = ||v_ref^b - v_cur^b||_2` | `w * R_exp(error, sigma)` |
| `root_ang_vel_tracking` | 在 body frame 比较角速度：`error = ||w_ref^b - w_cur^b||_2` | `w * R_exp(error, sigma)` |
| `keypoint_tracking` | `error = mean_k ||x_ref_k - x_cur_k||_2` | `w * R_exp(error, sigma)` |
| `lower_keypoint_tracking` | 同上（下肢 keypoint 子集） | `w * R_exp(error, sigma)` |
| `upper_keypoint_tracking` | 同上（上肢 keypoint 子集） | `w * R_exp(error, sigma)` |
| `keypoint_vel_tracking` | `error = mean_k ||v_ref_k^b - v_cur_k^b||_2` | `w * R_exp(error, sigma)` |
| `keypoint_rot_tracking` | `error = mean_k ||axis_angle(q_ref_k^b * conj(q_cur_k^b))||_2` | `w * R_exp(error, sigma)` |
| `keypoint_angvel_tracking` | `error = mean_k ||omega_ref_k^b - omega_cur_k^b||_2` | `w * R_exp(error, sigma)` |
| `joint_pos_tracking` | `error = mean_j |q_ref_j - q_cur_j|` | `w * R_exp(error, sigma)` |
| `joint_vel_tracking` | `error = mean_j |dq_ref_j - dq_cur_j|` | `w * R_exp(error, sigma)` |

## 5.4 Loco 组公式

| 子项 | 公式（代码语义） | 符号含义 |
|---|---|---|
| `survival` | `w * 1` | 常数生存奖励 |
| `joint_vel_l2` | `w * ( - sum_j dq_j^2 )` | 关节速度正则 |
| `action_rate_l2` | `w * ( - sum_j (a_t - a_{t-1})_j^2 )` | 动作平滑正则 |
| `feet_air_time_ref` | 按接触/离地时序累计 `reward_time`，落地瞬间给 `(reward_time - thres).clamp_max(0)` | 与参考脚步接触模式一致性 |
| `feet_air_time_ref_dense` | 接触状态不匹配给 `-1`；匹配时按足高映射到 `[-1,0]` 连续惩罚 | 稠密步态一致性 |
| `joint_pos_limits` | `w * ( -sum_j(violation_min_j + violation_max_j)/(1-soft_factor) )` | 超 soft 关节限位惩罚 |
| `joint_torque_limits` | `w * ( -sum_j(violation_high_j + violation_low_j) )` | 超软力矩限位惩罚 |

---

## 6. 关键实现备注（避免误解）

| 主题 | 备注 |
|---|---|
| Stage2 是否 DAgger | 不是标准 DAgger；更准确是 privileged feature distillation（`priv_pred` 拟合 `priv_feature`） |
| 部署是否需要 critic | 不需要；部署只用策略链路（通常是 student 路径） |
| 第一阶段能否直接部署 | 一般不建议，且通常不满足部署输入假设；至少应经过 adapt/finetune 形成 student 可部署路径 |
| KP/KD 是否完全免调 | 真机侧通常仍需工程校准；训练随机化用于提升鲁棒性与降低调参敏感性 |


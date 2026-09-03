# InstinctLab Parkour 对话记忆压缩与服务器交接

> 更新时间：2026-09-03（Asia/Shanghai）
>
> 用途：在另一台服务器或新的对话中继续本项目。后续助手应先阅读本文件，再检查实际工作区的 `git status` 和当前代码；不要覆盖用户在新服务器上的未提交修改。

## 1. 项目目标

用户正在训练 Unitree G1 完成以下任务：

- 平地行走。
- 平地站立。
- 稳定上楼梯。
- 稳定下楼梯，尤其希望逐级、交替、一步一级地下楼。

用户将 parkour 环境缩减为平地/站立地形和楼梯地形，主要配置为：

```text
source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg_onlystairs.py
```

注册的训练任务为：

```text
Instinct-Parkour-Only-Stairs-G1-v0
```

## 2. GitHub 权威状态

私人仓库：

```text
git@github.com:huxiaomo1/project_instinct_beiche.git
```

当前远端只保留 `main` 分支。OnlyStairs 功能分支已经合并并删除。

截至本文件创建前，远端 `main` 的关键提交为：

```text
152fa4f19c0f129896b440285c3a0fd9b41003d1
feat(parkour): penalize unsupported foot contact

022b0c91a134321f811df91de6c93bc7f76f38c2
feat(parkour): add only-stairs safety rewards
```

本交接文件将在后续单独提交，因此新服务器拉取后应以更新后的 `origin/main` 为准。

## 3. 新服务器同步方式

全新克隆：

```bash
git clone git@github.com:huxiaomo1/project_instinct_beiche.git InstinctLab
cd InstinctLab
git switch main
git pull --ff-only origin main
```

已有 Git 工作区：

```bash
cd /path/to/InstinctLab
git status
git remote -v
git fetch origin
git switch main
git pull --rebase origin main
```

如果已有未提交修改，应先提交到临时分支或执行 `git stash push -u`，不要直接覆盖。

如果服务器无法连接 GitHub SSH 22 端口，可在 `~/.ssh/config` 使用：

```sshconfig
Host github.com
    HostName ssh.github.com
    User git
    Port 443
```

## 4. OnlyStairs 已提交文件

主要文件：

```text
run_train_Parkour_OnlyStairs_script.sh
source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg_onlystairs.py
source/instinctlab/instinctlab/tasks/parkour/config/g1/g1_parkour_target_amp_cfg_only_stairs.py
source/instinctlab/instinctlab/tasks/parkour/config/g1/__init__.py
source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py
source/instinctlab/instinctlab/tasks/parkour/mdp/rewards.py
source/instinctlab/instinctlab/envs/mdp/rewards/volume_points.py
source/instinctlab/instinctlab/tasks/parkour/STAIR_REWARD_CHANGELOG.md
source/instinctlab/instinctlab/terrains/height_field/hf_terrains.py
source/instinctlab/instinctlab/terrains/height_field/hf_terrains_cfg.py
source/instinctlab/instinctlab/terrains/height_field/__init__.py
```

OnlyStairs 使用独立 runner：

```text
G1OnlyStairsPPORunnerCfg
experiment_name = "g1_parkour_onlystairs"
```

还加入了用于可视化/测试的 `DualPyramidStairsTerrainCfg`。

## 5. 动作数据结论

用户目前不准备修改动作数据，希望直接使用已有动作先训练平地行走、站立和上下楼梯。

当前 G1 OnlyStairs AMP 配置：

```text
source/instinctlab/instinctlab/tasks/parkour/config/g1/g1_parkour_target_amp_cfg_only_stairs.py
```

关键路径：

```python
path = os.path.expanduser("~/Datasets")
filtered_motion_selection_filepath = os.path.expanduser("~/Datasets/parkour_motion_without_run.yaml")
```

AMP 判别器输入包含 10 帧历史的：

- projected gravity。
- joint position。
- joint velocity。
- base linear velocity。
- base angular velocity。

判别器奖励系数：

```text
discriminator_reward_coef = 0.25
```

分析结论：

- 不修改动作数据并不妨碍先训练，但动作先验会影响步态风格。
- 如果数据缺少真实上下楼动作，AMP 可能对高抬腿、屈膝缓冲和下楼动作产生约束冲突。
- 楼梯能力主要仍由地形感知、任务速度奖励、接触安全奖励和课程难度建立。
- 后续应同时观察任务奖励与 discriminator reward，判断动作先验是否压制楼梯动作。

## 6. 动作 NPZ 数据格式

原始数据位置（原服务器）：

```text
/home/dhy/Datasets/parkour_motion_without_run_retargetted.npz
```

格式：

```text
framerate    shape=()          float64，值为 50.0
joint_names  shape=(29,)       Unicode 字符串
joint_pos    shape=(18982, 29) float32
base_pos_w   shape=(18982, 3)  float32
base_quat_w  shape=(18982, 4)  float32，顺序 [w, x, y, z]
```

29 个关节顺序：

```text
left_hip_pitch_joint
left_hip_roll_joint
left_hip_yaw_joint
left_knee_joint
left_ankle_pitch_joint
left_ankle_roll_joint
right_hip_pitch_joint
right_hip_roll_joint
right_hip_yaw_joint
right_knee_joint
right_ankle_pitch_joint
right_ankle_roll_joint
waist_yaw_joint
waist_roll_joint
waist_pitch_joint
left_shoulder_pitch_joint
left_shoulder_roll_joint
left_shoulder_yaw_joint
left_elbow_joint
left_wrist_roll_joint
left_wrist_pitch_joint
left_wrist_yaw_joint
right_shoulder_pitch_joint
right_shoulder_roll_joint
right_shoulder_yaw_joint
right_elbow_joint
right_wrist_roll_joint
right_wrist_pitch_joint
right_wrist_yaw_joint
```

## 7. GMR 渲染脚本修改

原服务器本地存在：

```text
/data4/dhy/InstinctLab/render_g1_npz.py
```

该文件目前是未跟踪文件，没有进入 GitHub，需要在新服务器另行传输或提交。

它已按上述 NPZ 格式修改：

- 使用 `framerate`，而不是旧字段 `fps`。
- 读取 `joint_names`、`joint_pos`、`base_pos_w`、`base_quat_w`。
- 不再要求 `joint_vel`、`body_pos_w`、`body_quat_w`。
- 校验帧数、29 个唯一关节名和四元数范数。
- 通过 MuJoCo 关节名称查找 qpos 地址，不再假设 XML qpos 顺序与 NPZ 完全一致。
- 查找唯一 free joint，并写入 7 维根姿态。

在另一台 GMR 服务器运行时曾遇到：

```text
ValueError: Expected a scalar joint for left_hip_pitch_joint, but got type 3
```

MuJoCo 中 type `3` 实际是 hinge。原因是服务器上的枚举对象与 NumPy 整数比较兼容性问题，已改为显式整数比较：

```python
joint_type = int(model.jnt_type[joint_id])
scalar_joint_types = {
    int(mujoco.mjtJoint.mjJNT_HINGE),
    int(mujoco.mjtJoint.mjJNT_SLIDE),
}
```

该脚本通过 `py_compile`，但原服务器缺少可直接运行的 MuJoCo/GMR 环境，未完成实际渲染验证。

## 8. 已讲解的基础奖励

### `feet_slide`

配置使用 `mdp.contact_slide`，权重 `-0.4`。

```text
raw = sum(contact_i * norm(foot_velocity_xy_i))
```

只在脚接触时惩罚世界 XY 平面滑动。

### `joint_deviation_hip`

配置使用 `mdp.joint_deviation_square`，权重 `-0.5`，约束左右髋关节 yaw 和 roll，不约束 pitch。

```text
raw = sum((q - q_default)^2)
```

用于防止双腿过度侧张、内外旋和交叉，同时保留前后抬腿能力。

### `feet_flat_ori`

配置权重 `-0.4`，实现为 `feet_orientation_contact`。

```text
raw = contact * sqrt(projected_gravity_x^2 + projected_gravity_y^2)
```

它鼓励接触脚局部 Z 轴与世界竖直方向一致，即鞋底平行于世界水平面。当前楼梯踏面为水平面，因此适用；它不是与任意局部地面法向比较。

### `feet_at_plane`

配置权重 `-0.1`。接触时惩罚脚部中心高于扫描地面加鞋底偏移的部分。基础偏移 `0.035 m`，G1 鞋模型覆盖为 `0.058 m`。

原扫描器很稀疏，只沿前后方向采样，不等同于完整鞋底支撑检测。

### `feet_close_xy`

函数实际只检查机器人朝向坐标系中的左右脚 Y 距离：

```text
raw = exp(-clamp(0.12 - distance_y, min=0) / std^2) - 1
std^2 = 0.05
```

原始值位于负数到零，配置权重为 `+0.4`，所以最终仍是双脚过近惩罚。距离不小于 `0.12 m` 时为零。

## 9. RewardManager 数值规则

当前环境：

```text
sim.dt = 0.005 s
decimation = 4
env.step_dt = 0.02 s
```

对于 sum 奖励组：

```text
单步总奖励贡献 = raw_function_value * weight * 0.02
```

`_termwise_reward_buf` 记录的是乘权重但未乘 `dt` 的值；episode 累计值会乘 `dt`。

## 10. `feet_stumble` 已实现方案

位置：

```text
source/instinctlab/instinctlab/tasks/parkour/mdp/rewards.py
```

初始配置：

```text
weight = -0.5
horizontal_vertical_ratio = 3.0
horizontal_force_threshold = 20.0 N
```

判断：

```text
norm(F_xy) > 3 * abs(F_z)
AND
norm(F_xy) > 20 N
```

接触力形状为：

```text
(num_envs, history_length=3, num_feet=2, xyz=3)
```

实现会在历史维度执行 `any`，然后左右脚求和。每个策略步的原始输出为 `0`、`1` 或 `2`。

用途与限制：

- 主要检测脚尖/鞋前部水平撞击台阶立面。
- 不能判断鞋底支撑面积，也不能判断是否跨级。
- 正常斜向落脚可能误触发。
- 若误触发多，可提高比例到 `4.0` 或阈值到 `30-50 N`。
- 若漏检，可降低比例到 `2.0-2.5` 或阈值到 `10-15 N`。

## 11. 静态楼梯边缘惩罚的历史与当前状态

曾新增 `feet_edge_overlap`，使用虚拟边缘圆柱与脚部体积点，按重合点比例和归一化侵入深度惩罚接触脚。

后来用户认为“靠近边缘”不等于“踩空”，该约束可能对窄台阶过强，因此当前 OnlyStairs 配置中：

```text
feet_edge_overlap.weight = 0.0
```

函数实现仍保留，便于消融对照，但训练中被 RewardManager 跳过。

注意：

- `STAIR_EDGE_REGION_RADIUS = 0.05 m` 仍存在。
- `volume_points_penetration` 仍以 `-4.0` 权重使用虚拟边缘，负责动态碰撞和清障。
- 如果用户将来要求完全取消所有基于边缘的信号，还需讨论是否同时关闭 `volume_points_penetration`；当前未关闭它，因为脚尖撞台阶仍需要约束。

## 12. 当前鞋底踩空惩罚：`feet_support_deficit`

最新实现已进入提交 `152fa4f`。

### 独立鞋底扫描器

新增：

```text
left_support_scanner
right_support_scanner
```

配置：

```text
offset = (0.0475, 0.0, 20.0)
ray_alignment = "yaw"
size = [0.145, 0.06]
resolution = 0.02
update_period = 0.02
```

IsaacLab `GridPatternCfg` 会生成约 `8 x 4 = 32` 个竖直向下射线，覆盖约：

```text
foot-local X: -0.025 到 0.115 m
foot-local Y: -0.03 到 0.03 m
```

扫描器只随脚的 yaw 旋转，网格保持水平，不随 roll/pitch 倾斜。

使用独立扫描器是为了不改变旧 `feet_at_plane` 稀疏扫描器的数值尺度。

### 函数定义

位置：

```text
source/instinctlab/instinctlab/tasks/parkour/mdp/rewards.py
```

参数：

```text
weight = -1.0
height_offset = 0.058 m（G1 鞋模型覆盖值）
support_tolerance = 0.02 m
min_support_ratio = 0.7
contact_force_threshold = 1.0 N
```

计算公式：

```text
sole_height = ankle_roll_link_z - height_offset
height_error_j = abs(sole_height - ground_hit_height_j)
point_support_j = exp(-(height_error_j / support_tolerance)^2)
support_ratio = mean(point_support_j)
support_deficit = clamp((min_support_ratio - support_ratio) / min_support_ratio, 0, 1)
foot_penalty = is_contact * support_deficit^2
raw = left_foot_penalty + right_foot_penalty
```

未命中地形或非有限的射线支撑分数为 `0`。只有接触脚受罚，摆动脚不会因为悬空而受罚。原始输出范围 `[0, 2]`。

意义：

- 完整鞋底处在同一踏面高度时，支撑率接近 `1`，惩罚为零。
- 部分鞋底伸出台阶、下方射线落到较低踏面时，支撑率下降并产生惩罚。
- 不依赖显式边缘距离，更贴近“脚底实际踩空”的目标。

已验证：

- 三个 Python 文件通过 `py_compile`。
- 纯 Torch 张量测试：完整支撑为 `0`；半脚支撑产生正惩罚；未接触脚为 `0`。
- 普通 Python 导入 IsaacLab 时因缺少 Isaac Sim 的 `pxr` 失败，因此尚未执行完整环境运行测试。

运行时必须重点验证：

- `height_offset=0.058` 是否与实际鞋底碰撞几何准确对应。
- 正常落脚时 `support_ratio` 是否接近 `1`。
- 脚 roll/pitch 较大时，固定水平鞋底平面是否造成误罚。
- RayCaster 更新和实际地形命中是否存在 `inf`。

调参方向：

```text
正常落脚误罚：support_tolerance -> 0.025-0.03，或 min_support_ratio -> 0.6-0.65
半脚踩空漏罚：min_support_ratio -> 0.75-0.8，或 weight -> -1.5
机器人不敢落脚：weight -> -0.3 到 -0.7
```

训练日志：

```text
Episode_Reward/rewards_feet_support_deficit/max_episode_len_s
Episode_Reward/rewards_feet_support_deficit/sum
Episode_Reward/rewards_feet_support_deficit/timestep
```

## 13. 楼梯逐级行走分析结论

当前仍未实现“明确保证一步一级”的状态奖励。

重要发现：当前 `feet_air_time` 没有上限。它在恰好单脚支撑时返回支撑时间与摆动时间的较小值，并随单脚阶段持续增长。配合楼梯速度命令 `0.45-0.8 m/s`，可能鼓励长单支撑、大步长甚至跨级。

推荐后续优先级：

1. 将 `feet_air_time` 改成有目标值的落脚事件奖励，例如目标摆动时间 `0.35-0.50 s`，而不是时间越长越好。
2. 对超过约 `0.55 s` 的摆动时间增加惩罚。
3. 下楼阶段降低速度范围，例如先使用 `0.25-0.55 m/s`。
4. 增加双脚同时离地惩罚，允许短暂双支撑，不鼓励跳跃下楼。
5. 用首次接触事件记录相邻两次落脚的地面高度和水平位置，直接检测跨级。
6. 增加落地冲击惩罚，鼓励屈膝缓冲；不要简单强罚所有 base Z 速度，因为正常上下楼需要竖直运动。

一步一级的理想状态奖励应在首次接触时检查：

```text
左右脚是否合理交替
本次落脚地面高度与上次落脚高度是否只变化一级
本次落脚水平距离是否接近一个踏面宽度
```

对于随机台阶，尽量从射线地面高度推断相邻踏面，不要把台阶高度写死。

## 14. 当前稳定性奖励组合

OnlyStairs 当前与楼梯稳定性直接相关的主要项：

```text
volume_points_penetration  weight=-4.0  动态边缘/立面穿透
feet_support_deficit       weight=-1.0  接触脚鞋底支撑不足
feet_stumble              weight=-0.5  水平撞击台阶
feet_slide                weight=-0.4  支撑脚水平滑动
feet_flat_ori             weight=-0.4  接触脚保持水平
feet_at_plane             weight=-0.1  接触脚相对局部地面高度
feet_edge_overlap         weight= 0.0  已关闭
flat_orientation_l2       weight=-3.0  基座保持直立
pelvis_orientation_l2     weight=-3.0  骨盆保持直立
ang_vel_xy_l2             weight=-0.05 抑制 roll/pitch 角速度
undesired_contacts        weight=-1.0  非脚部接触
```

## 15. 中文奖励修改日志

详细奖励公式、参数和调参记录位于：

```text
source/instinctlab/instinctlab/tasks/parkour/STAIR_REWARD_CHANGELOG.md
```

后续每次修改奖励时，应继续按日期追加记录，不要覆盖历史条目。

## 16. 原服务器未提交内容

原服务器工作区还有大量与本次提交无关的修改和未跟踪文件，包括：

- `render_g1_npz.py`。
- ONNX 模型目录 `model/`。
- 多个 `output_*.log` 训练输出。
- MuJoCo sim2sim 脚本。
- 其他 parkour 实验配置和可视化脚本。
- G1 资源、通用 parkour 配置、setup 等未提交修改。

这些内容没有随 OnlyStairs 奖励提交上传。新服务器仅执行 `git pull` 不会获得它们。不要因为 GitHub 中不存在这些文件而认为它们已丢失；它们仍保留在原服务器工作区。

## 17. 建议新对话的第一条指令

可将下面内容发给新服务器上的助手：

```text
请先阅读仓库根目录 PARKOUR_CHAT_HANDOFF.md，并检查 git status、当前 main 提交和 OnlyStairs 配置。不要覆盖未提交修改。我的目标是继续训练 G1 完成平地行走/站立以及稳定逐级上下楼梯。当前最新工作是 feet_support_deficit 鞋底网格踩空惩罚，请先验证它在 Isaac Sim 中的实际支撑率分布，再决定调参或继续设计一步一级奖励。
```

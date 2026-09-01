# 楼梯安全奖励修改日志

## 2026-09-01：楼梯边缘安全与绊倒惩罚

### 适用范围

- 任务：`Instinct-Parkour-Only-Stairs-G1-v0`
- 环境配置：`config/parkour_env_cfg_onlystairs.py`
- 策略控制周期：`0.02 s`（`sim.dt=0.005`，`decimation=4`）
- RewardManager 中每个策略步的奖励贡献：`奖励函数原始值 * weight * 0.02`

### 修改概述

本次修改组合了以下三个互补的楼梯安全奖励项：

1. `volume_points_penetration`：已有的动态边缘穿透惩罚。
2. `feet_edge_overlap`：新增的支撑脚与边缘危险区重合惩罚。
3. `feet_stumble`：新增的脚部水平撞击惩罚。

本次没有修改 `volume_points_penetration` 的函数实现。由于它和
`feet_edge_overlap` 共用虚拟边缘与脚部体积点数据，因此将其一并记录，便于理解三项奖励之间的关系。

## 1. `volume_points_penetration`

### 修改状态

保留已有奖励及其原始配置：

```python
weight = -4.0
```

### 奖励定义

对于进入虚拟障碍物的每个脚部体积采样点：

```text
惩罚值 = sum(是否位于障碍物内 * (采样点速度 + 1e-6) * 穿透深度)
```

### 主要作用

- 惩罚摆动脚在运动过程中进入楼梯边缘。
- 惩罚脚尖或鞋体穿入台阶立面。
- 鼓励机器人保持足够的摆动脚离地高度。

### 当前局限

当脚部停止运动时，该惩罚会变得非常小。因此，它不能充分惩罚已经踩在楼梯边缘并保持静止的支撑脚。

## 2. `feet_edge_overlap`

### 修改状态

新增奖励函数，实现在 `envs/mdp/rewards/volume_points.py`，并在
`config/parkour_env_cfg_onlystairs.py` 中启用。

### 初始配置

```python
STAIR_EDGE_REGION_RADIUS = 0.05
weight = -1.0
contact_force_threshold = 1.0
penetration_tolerance = 1e-4
depth_weight = 1.0
```

`STAIR_EDGE_REGION_RADIUS` 同时用于生成虚拟边缘圆柱和归一化奖励中的穿透深度，确保危险区几何尺寸与奖励计算保持一致。

### 奖励定义

对于每只脚：

```text
重合比例 = 位于边缘区内的脚部采样点数 / 脚部采样点总数
平均侵入深度 = mean(clamp(穿透深度 / 边缘区半径, 0, 1))
单脚惩罚 = 是否接触 * (重合比例 + depth_weight * 平均侵入深度)
总惩罚 = 左右脚惩罚之和
```

奖励函数原始输出范围约为 `[0, 4]`：每只脚的重合比例最多贡献 `1`，归一化穿透深度最多贡献 `1`。

### 主要作用

- 惩罚脚掌只有一部分踩在楼梯边缘的情况。
- 惩罚支撑脚静止在楼梯边缘危险区内的情况。
- 同时利用重合范围和侵入深度，提供连续的危险程度信号。

### 与 `volume_points_penetration` 的关系

- `volume_points_penetration` 负责运动中的脚部碰撞和摆动脚清障。
- `feet_edge_overlap` 负责接触后的静态支撑与边缘踩踏。

这两个奖励约束步态的不同阶段，初始训练时建议同时启用。

## 3. `feet_stumble`

### 修改状态

新增奖励函数，实现在 `tasks/parkour/mdp/rewards.py`，并在
`config/parkour_env_cfg_onlystairs.py` 中启用。

### 初始配置

```python
weight = -0.5
horizontal_vertical_ratio = 3.0
horizontal_force_threshold = 20.0
```

### 奖励定义

只有同时满足以下两个条件时，才将该脚判定为发生绊碰：

```text
norm(F_xy) > horizontal_vertical_ratio * abs(F_z)
norm(F_xy) > horizontal_force_threshold
```

即使一次绊碰出现在接触传感器历史记录的多个帧中，每只脚在一个策略步内最多只计数一次。原始输出为 `0`、`1` 或 `2`，分别表示没有脚、一只脚或两只脚发生绊碰。

### 主要作用

- 检测脚尖或鞋前部撞击台阶立面。
- 惩罚以水平撞击力为主、而不是以竖直支撑力为主的接触。
- 提高上下楼梯时的摆动脚清障能力。

## 训练日志指标

RewardManager 会自动记录所有已配置的奖励项，不需要额外修改训练循环。预期生成的指标前缀为：

```text
Episode_Reward/rewards_volume_points_penetration/
Episode_Reward/rewards_feet_edge_overlap/
Episode_Reward/rewards_feet_stumble/
```

每个指标前缀通常包含：

```text
max_episode_len_s
sum
timestep
```

分析训练效果时，建议同时对比以下指标：

- Episode Length。
- 地形课程等级。
- 速度跟踪奖励。
- 非期望接触惩罚。
- 视频中实际观察到的楼梯通过率与落脚质量。

某个惩罚项变得更加负，并不一定说明策略变差。它既可能表示不安全事件增加，也可能表示该奖励的检测阈值过于敏感，需要结合视频和其他指标判断。

## 调参建议

| 训练现象 | 建议调整 |
|---|---|
| 正常落脚也频繁触发 `feet_stumble` | 将力比例提高到 `4.0`，或将水平力阈值提高到 `30-50 N` |
| 视频中能够看到脚尖撞击，但 `feet_stumble` 接近零 | 将力比例降低到 `2.0-2.5`，或将水平力阈值降低到 `10-15 N` |
| 机器人不愿接近楼梯或不愿落脚 | 将边缘区半径降低到 `0.03-0.04 m`，或将边缘重合权重降低到 `-0.3` 至 `-0.7` |
| 仍然频繁出现半脚踩在边缘的情况 | 将边缘重合权重提高到 `-1.5` 至 `-2.0` |
| 摆动脚撞击台阶的情况仍然较多 | 逐渐将 stumble 权重提高到 `-0.8` 至 `-1.0` |
| 步态变得过于谨慎或机器人不敢迈步 | 优先降低两个新增奖励的权重，再考虑修改任务奖励 |

调整边缘区宽度时，只修改 `STAIR_EDGE_REGION_RADIUS`，以保证虚拟边缘几何尺寸和奖励归一化参数始终一致。

## 验证记录

- 两个奖励函数模块和 only-stairs 环境配置均已通过 Python 语法编译检查。
- 当前终端环境中没有安装 `ruff`，因此没有执行格式化和 lint 检查。
- 当前终端中没有运行 Isaac Sim 训练，也没有完成奖励分布与实际触发频率验证。

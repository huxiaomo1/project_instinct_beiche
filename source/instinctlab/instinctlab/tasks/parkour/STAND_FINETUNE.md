# Parkour 零速站立后训练

新增任务 `Instinct-Parkour-Stand-Finetune-G1-v0`，从普通 Parkour checkpoint
继续训练。原任务 ID 的训练行为不变。Actor、Critic、AMP 判别器的网络维度不变，
继续使用旧权重和旧策略的动作标准差。

## 启动

在安装了 Isaac Lab、Instinct-RL 的环境中运行，替换下面的 checkpoint 路径：

外部 `instinct_rl/runners/on_policy_runner.py` 必须支持 `模块:类名` 的算法加载：
将名称按 `:` 拆分后，通过 `getattr(importlib.import_module(module_name), class_name)`
获取算法类。本项目直接传入配置，不再注册算法别名或提供加载兼容层。
请确认训练环境实际导入的那份 Instinct-RL 已修复此入口。

```bash
python scripts/instinct_rl/train.py \
  --task Instinct-Parkour-Stand-Finetune-G1-v0 \
  --resume --load_run /absolute/path/to/old_run \
  --checkpoint model_30000.pt \
  --run_name stop_walk_stairs --max_iterations 5000 --headless
```

这里的 `--load_run` 是包含 checkpoint 的目录。新日志单独写入
`logs/instinct_rl/g1_parkour_stand_finetune/`，不会覆盖原 checkpoint。
本任务禁止不带 `--resume` 从头启动。

入口会读取 Actor/Critic 深度编码器第一层全连接权重，自动匹配原模型的
32×18 裁剪输入或 64×36 完整输入，不裁切、插值或丢弃网络权重。
不支持的编码器尺寸会报错；其余权重仍严格加载。
`auto_depth_from_checkpoint=False` 可关闭此功能，但此时必须自行匹配输入。

## 奖励和停车控制

```text
总奖励 = 原环境奖励 × dt
       + 0.25 × [(1 − ηm) × AMP + ηm × 替代奖励]
替代奖励 = (1 − b) × 减速奖励 + b × 站立奖励
```

AMP、减速、站立奖励均在 [0,1]；它们在算法辅助奖励路径中混合，统一乘
`discriminator_reward_coef`，不再乘 dt。相同范围不代表均值或方差相同。
没有通过无条件常数补偿来强行保持总奖励不变。

- `m` 只由处理后的线速度/偏航命令决定：零速为 1，正常前进或转向为 0。
  漂移不会重新打开运动 AMP。
- `b` 随命令进一步接近零而提高。双脚接触、机身较安静可以提前加强站立；
  命令归零时 b 必须为 1，不能靠一直不落脚无限逃避静止约束。
- 第一版不引入独立的停车计时器。缓冲时间受命令斜率约束：例如
  0.6 m/s 在 1.5 m/s² 下约 0.4 s 到零。它是有限的落脚宽限，并不保证
  每种步态都恰好在命令归零前落脚，需要以仿真停车测试调整。
- 减速奖励：0.7 × 速度追踪 + 0.3 × 姿态稳定。
- 站立奖励：姿态稳定系数 × (0.25 × 默认关节姿态 + 0.20 × 低关节速度
  + 0.40 × 低机身速度 + 0.15 × 低机身角速度)。
- 用运行时 `default_joint_pos` 作姿态目标。双脚高度不同时减弱腿部姿态要求，
  避免在跨台阶时强迫双腿回到平地对称姿态。
- 原环境的速度追踪、滑动、接触安全等奖励保留；原 `stand_still` 权重置零，
  避免其在低速阶段重复、过早地要求默认姿态。

这些特征在环境奖励计算时采集，即命令重采样和自动重置之前，按当前
动作对应的命令与状态计算。它们通过 extras 送入算法，不增加策略或价值网络输入。

## 后训练默认参数

配置位于 `config/g1/agents/instinct_rl_amp_cfg.py` 的
`StandFineTuneAlgoCfg`，环境参数位于 `g1_parkour_stand_finetune_cfg.py`。

- 学习率 5e-5，固定调度；PPO clip 0.1，熵系数 0.001。
- 前 100 次 PPO 更新 η=0，之后 2000 次更新线性增加到 1。
  这是本次后训练的更新次数，不使用旧 checkpoint 的迭代号。
- 从普通 checkpoint 首次加载时保留网络权重，清除旧优化器动量；再次加载
  本功能生成的 checkpoint 时恢复优化器和渐变进度。加载后会重设本次配置的学习率，
  避免优化器状态把旧的大学习率带回来。
- 每 3–5 秒重采样命令，35% 概率为站立；同一个环境会反复经历走、停、再走。
- 平地采用随机速度命令，楼梯保留地形目标方向；地形比例为平地 35%、普通
  上下楼梯 50%、其余原地形 15%。

可以通过 Hydra 参数调整，例如：

```bash
# 追加到上述训练命令末尾
agent.algorithm.stand_warmup_updates=100 \
agent.algorithm.stand_blend_updates=3000 \
agent.algorithm.learning_rate=0.00003
```

## 评估和部署

```bash
python scripts/instinct_rl/play.py \
  --task Instinct-Parkour-Stand-Finetune-G1-Play-v0 \
  --load_run /absolute/path/to/finetuned_run --checkpoint model_35000.pt
```

训练记录 `stand/amp_raw`、`stand/brake`、`stand/stand`、`stand/replacement`、
`stand/style_contribution`、`stand/stop_mix`、`stand/settle_mix`、
`stand/double_support`、`stand/linear_speed` 和 `stand/progress`。
另记录 `stand/stop_speed`、`stand/stop_amp_mean`、`stand/stop_replacement_mean`，
仅统计近零速命令样本；无此类样本时为零，结合 `stand/stop_fraction` 判断。
原 PPO 日志仍包含 value loss。上游 runner 的回报日志只统计环境回报，
评估风格奖励实际贡献应查看 `stand/style_contribution`。

固定检查平地静止 10 秒位移、行走后停车时间/距离、再次起步、楼梯成功率。
默认参数是初始方案，不代表已经完成机器人仿真或实机验证。

部署控制器必须使用与训练相同的命令限速。复用 `stand_control.py` 的
`limit_velocity_command(previous, raw_target, actual_dt, (1.5, 1.5, 3.0))`：
先处理手柄死区，再限速；用限速后的三维命令填入原观测及观测历史。
控制器激活/重置时清零 previous，使用真实策略控制周期。
本次没有修改外部 `instinct_onboard` 仓库。重新给出运动指令也经过同一限速器，
不直接切换动作输出或把关节锁定。

## 不需要 Isaac Sim 的检查

```bash
python tests/test_parkour_stand_finetune.py
```

这些 CPU 检查覆盖奖励范围、平滑混合、命令斜率、落脚宽限、跨台阶姿态放宽、
旧模型尺寸匹配和 checkpoint 恢复逻辑。算法生命周期单元测试使用小型 Wasabi
替身；完整训练仍需在 Isaac Lab 环境验证。

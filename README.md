# 2D Car World-Model Lab: Stage 1

这一阶段实现一个完全可控的二维“小世界”。它的作用不是训练模型，而是提供之后世界模型需要学习的真实状态转移：

```text
current state + action
          ↓ CarEnv.step()
      next state
```

## 状态与动作

状态为：

```text
[x, y, heading, velocity]
```

动作为：

```text
[steering, acceleration]
```

环境使用简化运动学自行车模型：

```text
x_next     = x + velocity * cos(heading) * dt
y_next     = y + velocity * sin(heading) * dt
heading    = heading + velocity / wheelbase * tan(steering) * dt
velocity   = clip(velocity + acceleration * dt, 0, max_speed)
```

它还负责判断目标、障碍物、地图边界和最大步数。环境是确定性的：同一个状态和动作总会产生同一个下一状态，便于后续测量学习型 World Model 的预测误差。

## 项目结构

```text
world_model_lab/
├── .venv/                  # 只属于本实验的 Python 环境
├── pyproject.toml          # 项目元数据与依赖
├── src/world_model_lab/    # 可导入的 Python 包
└── tests/                  # 独立测试
```

虚拟环境放在项目根目录，而不是 `src/world_model_lab/` 包目录内。这样运行环境、项目配置、源码和测试各自有清晰边界。

以下命令都从独立项目根目录执行：

```bash
cd /Users/andyzhao/Workspace/world_model_lab
```

## 安装

首次使用时创建虚拟环境并以 editable 模式安装项目：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

`-e` 表示 editable install：修改 `src/world_model_lab/` 后无需重新安装。

## 运行

运行测试：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

打开轨迹演示：

```bash
.venv/bin/python -m world_model_lab.demo_random_drive
```

也可以使用安装时生成的命令：

```bash
.venv/bin/world-model-car-demo
```

在没有桌面窗口的环境中保存图片：

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python -m world_model_lab.demo_random_drive \
  --steps 80 --save /tmp/world-model-car-demo.png
```

## 文件分工

- `src/world_model_lab/car_env.py`：真实动力学、碰撞、奖励和 episode 状态。
- `src/world_model_lab/visualize.py`：把环境公开状态画出来，不计算或修改动力学。
- `src/world_model_lab/demo_random_drive.py`：生成一条可复现的演示轨迹。
- `tests/`：验证状态转移、终止条件、复制安全和无界面渲染。

## 下一阶段怎样使用它

数据采集器会运行大量 episode，并保存：

```text
(state_t, action_t, state_t+1)
```

随后训练一个神经网络近似 `CarEnv.step()` 的状态转移。到那时，`CarEnv` 仍代表 ground truth，而神经网络才是 learned world model。

## 采集 World Model 训练数据

默认采集最多 250 个 episode × 200 步，并保存为压缩 NumPy 文件：

```bash
.venv/bin/python -m world_model_lab.collect_data \
  --episodes 250 \
  --max-steps 200 \
  --action-hold-steps 5 \
  --seed 7 \
  --output data/transitions.npz
```

安装项目后也可以使用：

```bash
.venv/bin/world-model-collect-data
```

`.npz` 中包含：

| 数组 | 形状 | 含义 |
|---|---|---|
| `states` | `[N, 4]` | 执行动作前的 `[x, y, heading, velocity]` |
| `actions` | `[N, 2]` | 环境实际执行的 `[steering, acceleration]` |
| `next_states` | `[N, 4]` | 执行动作后的真实状态 |
| `rewards` | `[N]` | 环境奖励 |
| `dones` | `[N]` | 当前 transition 是否结束 episode |
| `episode_ids` | `[N]` | episode 编号 |
| `step_ids` | `[N]` | episode 内的步编号 |
| `terminal_reasons` | `[N]` | `goal`、`collision`、`out_of_bounds` 或 `time_limit` |

检查数据：

```bash
.venv/bin/python -c "import numpy as np; d=np.load('data/transitions.npz'); print({k: d[k].shape for k in d.files})"
```

固定相同的 `--seed` 和采集参数会得到完全相同的数据，便于后续训练与实验复现。`data/` 是生成产物目录，默认不会进入 Git。

## 训练第一个 Learned World Model

安装新增的 PyTorch 依赖：

```bash
.venv/bin/python -m pip install -e .
```

训练一个预测单步状态变化量的 MLP：

```bash
.venv/bin/python -m world_model_lab.train_world_model \
  --data data/transitions.npz \
  --epochs 100 \
  --output artifacts/world_model.pt
```

也可以使用安装时生成的命令：

```bash
.venv/bin/world-model-train --epochs 100
```

模型不会直接预测绝对的下一状态。它接收：

```text
[x, y, sin(heading), cos(heading), velocity, steering, acceleration]
```

并预测：

```text
[delta_x, delta_y, wrapped_delta_heading, delta_velocity]
```

数据会按 episode ID 以 80% / 10% / 10% 划分，避免相邻 transition
同时进入训练集和验证集。归一化统计量只由训练集计算。命令输出验证集和测试集的
位置、角度、速度单步 MAE，并在 checkpoint 中保存模型权重、归一化统计量、
episode 切分和训练参数。

训练期间会在每个 epoch 记录训练集和验证集的 normalized MSE。返回的模型是
验证 loss 最低 epoch 的权重；测试集只在训练结束后评估一次。checkpoint 还会
保存最佳 epoch、两条 loss 曲线和最终测试指标。

训练完成后可以直接从 checkpoint 生成曲线，不需要重新训练：

```bash
.venv/bin/python -m world_model_lab.plot_training \
  --checkpoint artifacts/world_model.pt \
  --output artifacts/training_loss.png
```

重新执行 editable install 后，也可以使用：

```bash
.venv/bin/world-model-plot-training
```

## 多步训练实验

默认的 `--rollout-horizon 1` 保留原来的单步训练路径，只优化真实
`state_t` 输入下的下一状态变化量误差：

```bash
# 单步基线
.venv/bin/python -m world_model_lab.train_world_model \
  --data data/transitions.npz \
  --output artifacts/world_model_h1.pt \
  --rollout-horizon 1 \
  --epochs 100 \
  --seed 0
```

多步模式在单步损失之外加入可微分 Free Rollout 损失。模型只接收序列的真实
初始状态；之后将自己的预测状态递归作为下一步输入，并与记录的真实状态序列比较：

```bash
# 单步损失 + 10 步 Free Rollout 损失
.venv/bin/python -m world_model_lab.train_world_model \
  --data data/transitions.npz \
  --output artifacts/world_model_h10.pt \
  --rollout-horizon 10 \
  --rollout-loss-weight 1.0 \
  --epochs 100 \
  --seed 0
```

总训练目标为：

```text
total_loss = one_step_loss + rollout_loss_weight * rollout_loss
```

两项误差都用训练集 target 标准差缩放。rollout 期间仍使用数据集中记录的动作，
不预测动作、奖励或终止条件。horizon 1 不要求数据包含 `step_ids`；horizon 大于 1
时，`step_ids` 用于验证窗口始终位于同一个 episode 内且步数连续。

checkpoint 使用格式版本 3，同时保存 `train_losses`、`validation_losses` 两条总
损失历史，以及 one-step、rollout 的四条分项历史；加载器仍兼容格式版本 1 和 2。
最佳 epoch 按总验证损失选择，测试集不参与选择。可以对两个 checkpoint 使用下方
完全相同的诊断参数，公平比较 horizon 1、5、10、20、50 的 Free Rollout 误差。

## 多步 Rollout 评估

单步评估每次都使用真实状态。rollout 评估只使用 episode 的真实初始状态，之后
将模型自己的预测状态递归作为下一步输入，从而测量误差累积：

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python -m world_model_lab.evaluate_rollout \
  --data data/transitions.npz \
  --checkpoint artifacts/world_model.pt \
  --horizons 1 5 10 20 50 \
  --plot artifacts/rollout_evaluation.png
```

重新执行 editable install 后，也可以使用：

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/world-model-evaluate-rollout
```

命令只评估 checkpoint 中记录的测试 episode，并报告不同 horizon 的平均位置、
heading 和速度误差。默认选择最长的测试 episode 绘制真实/预测 XY 轨迹以及三类
误差随 rollout 步数的变化。这里使用数据集中记录的动作，不让模型预测动作或终止
条件。

## 模型诊断实验室

普通 rollout 图适合查看某一条轨迹；模型诊断命令使用固定评估协议回答三个问题：

1. 测试数据覆盖了哪些状态和动作区域？
2. 模型在哪些区域单步误差较大？
3. 递归使用预测状态时，误差会怎样随 horizon 累积？

运行完整诊断：

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python -m world_model_lab.diagnose_model \
  --data data/transitions.npz \
  --checkpoint artifacts/world_model.pt \
  --output-dir artifacts/diagnostics/baseline \
  --horizons 1 5 10 20 50 \
  --windows-per-episode 8 \
  --xy-bins 12 \
  --feature-bins 8 \
  --min-bin-count 5
```

重新执行 editable install 后，也可以使用：

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/world-model-diagnose
```

命令生成一个可复现的 benchmark bundle：

| 文件 | 内容 |
|---|---|
| `metrics.json` | schema v2：单步误差、状态/动作分箱、稀疏 horizon 汇总，以及每一步的物理误差和归一化 MSE 分量 |
| `manifest.json` | 数据集和 checkpoint 的 SHA-256、测试 episode 和诊断参数 |
| `overview.png` | 训练/测试 XY 覆盖、XY 误差、速度和动作误差切片 |
| `rollout_errors.png` | Teacher Forcing 与 Free Rollout 从第 1 步到最大 horizon 的稠密物理误差曲线 |
| `rollout_loss_components.png` | `x`、`y`、heading、velocity 四个归一化 MSE 分量的 2×2 对比图 |

误差指标只使用 checkpoint 记录的测试 episode。覆盖图同时显示训练集和测试集，
用于识别分布差异，但训练 transition 不会参与误差计算。

所有 horizon 使用同一组最大长度窗口。每个 episode 最多选择固定数量、均匀分布的
窗口；聚合时先在 episode 内平均，再对 episode 求平均，避免长 episode 因为窗口多
而获得更大权重。

- **Teacher Forcing**：每一步都输入数据集中记录的真实状态，反映局部单步误差。
- **Free Rollout**：只给初始真实状态，后续递归输入模型预测，反映误差累积。

`--horizons` 仍定义稀疏 benchmark 点，其中最大值同时决定稠密曲线长度。例如
`--horizons 1 5 10 20 50` 会保留五个带分布统计的 horizon，同时在
`metrics.json` 和两张 rollout 图中记录第 1 到第 50 步。归一化分量使用 checkpoint
保存的 target-delta 标准差，与多步训练目标处于同一尺度；`total` 是四个分量的算术
平均值。所有曲线先在同一 episode 内平均窗口，再对 episode 等权平均。

当两条曲线随 horizon 分离时，差值主要来自 compounding error。样本数小于
`min_bin_count` 的空间或特征区间会在误差图中被遮罩；其样本数量仍保留在 JSON
和覆盖图中，不能把低覆盖区域误判成模型表现良好。

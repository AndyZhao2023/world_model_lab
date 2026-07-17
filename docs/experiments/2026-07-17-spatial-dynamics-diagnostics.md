# Spatial latent dynamics diagnostics

## Question

Does the trained spatial dynamics CNN predict motion beyond reusing the last
encoded frame, and do its one-step predictions depend on the aligned actions?

## Diagnostic baselines

The existing spatial checkpoint is frozen. No model is retrained or mutated.

- `Oracle`: decode the encoded true target frame.
- `World`: predict the next latent from four latent frames and four actions.
- `Decoded last latent`: decode the last context latent without dynamics.
- `Mean action`: replace every action with the training-set action mean, which
  becomes zero after the checkpoint's action normalization.
- `Shuffled action`: move each complete three-history-plus-current action row
  to another test window while keeping its internal temporal order.
- `Raw copy-last`: compare the original last RGB frame directly with the
  target; this bypasses both autoencoder and dynamics.

## Detailed method definitions

下面把六种方法放在同一个符号系统里说明。对一个测试窗口，定义：

```text
context frames:  x[t-3], x[t-2], x[t-1], x[t]
aligned actions: a[t-3], a[t-2], a[t-1], a[t]
target frame:    x[t+1]

encoder:         E(image) -> latent
decoder:         D(latent) -> image
dynamics:        F(context latents, aligned actions) -> next latent
```

其中 `z[i] = E(x[i])`。所有方法使用完全相同的测试窗口和目标帧，区别只在于
预测路径中允许使用哪些信息。

### 1. Oracle

```text
prediction = D(E(x[t+1]))
```

Oracle 直接编码并重建真实目标帧，不使用历史帧、动作或 dynamics。因为它看到了
`x[t+1]`，所以它不是可以部署的预测方法，而是 autoencoder 的诊断下界：

- 如果 Oracle 误差很大，说明视觉表示本身没有保存目标细节；
- 如果 World 明显差于 Oracle，差值主要来自 dynamics；
- World 不应被期望稳定优于 Oracle，因为 Oracle 已经拿到了真实目标 latent。

本实验中 Oracle 的变化区 MAE 为 `0.266102`，代表当前视觉表示能够达到的参考水平。

### 2. World with recorded actions

```text
context latents = [E(x[t-3]), E(x[t-2]), E(x[t-1]), E(x[t])]
predicted latent = F(context latents, [a[t-3], a[t-2], a[t-1], a[t]])
prediction = D(predicted latent)
```

这是实际的一步 world-model 路径，也是唯一同时使用真实视觉历史、正确动作对齐和
dynamics 的方法。它不能看到目标帧 `x[t+1]`。

它回答的问题是：给定过去四帧和期间实际执行的动作，模型能否预测下一帧？本实验
中它的变化区 MAE 为 `0.314072`。

### 3. Mean-action ablation

```text
replacement action = training-set action mean mu[a]
normalized replacement action = 0

prediction = D(F(context latents, [mu[a], mu[a], mu[a], mu[a]]))
```

该方法保留正确的四帧视觉历史，只移除每个测试窗口特有的动作信息。这里使用的是
训练集动作均值，而不是物理意义上的 `[steering=0, acceleration=0]`。原因是模型
接收归一化动作；用训练均值替换后，动作通道恰好全部为零，不会引入额外的分布外
数值。

它回答的问题是：在视觉历史不变时，模型是否需要“这个窗口实际执行了什么动作”？
如果它明显差于 World，说明具体动作信息有帮助；如果二者接近，只能说明动作对
当前一步指标的增量较小，不能证明动作在更长 rollout 中无用。

本实验中 Mean-action 的变化区 MAE 为 `0.312239`，略好于 World，但 latent MSE
变差 `0.40%`，说明 latent 目标和小车像素质量并不完全对齐。

### 4. Shuffled-action ablation

对测试窗口生成一个固定随机置换 `pi`：

```text
original sample i:
    context[i] + actions[i]

shuffled sample i:
    context[i] + actions[pi(i)]
```

来自另一个窗口的三条历史动作和一条当前动作会作为一个完整四动作序列一起移动。
序列内部的时间顺序保持不变，只破坏“这段视觉历史和这段动作是否属于同一窗口”的
对应关系。我们不会分别打乱每一个时间步，因为那会同时破坏动作内部的时序结构，
无法单独检验跨模态对齐。

该方法回答的是：模型是否依赖动作与视觉历史之间的正确配对？实验使用 seed
`0` 到 `9` 重复置换，以避免结论依赖某一次随机排列。十次置换的变化区 MAE 为：

```text
0.314896 ± 0.000209
```

它比 World 平均差 `0.26%`，而 normalized latent MSE 平均差 `2.36%`。因此模型
能感知动作对齐，但这种影响在当前一步解码图像中很弱。

### 5. Decoded-last-latent

```text
prediction = D(E(x[t]))
```

该方法把最后一张 context frame 编码后立即解码，不使用 dynamics，也不使用动作。
它仍然经过与 World 完全相同的 autoencoder，因此是
`representation-matched no-dynamics baseline`。

World 与它的差值可以较干净地回答“dynamics 是否学到了下一步变化”，因为两者都
承担相同的编码和解码误差。本实验中：

```text
Decoded-last-latent changed MAE = 0.359824
World changed MAE               = 0.314072
```

World 改善 `12.72%`，说明 dynamics 不是简单返回最后一个 latent。

### 6. Raw copy-last

```text
prediction = x[t]
```

Raw copy-last 直接把最后一张原始 RGB context frame 当作下一帧。它不经过 encoder、
decoder 或 dynamics，也不使用动作。这是最简单的“世界不会变化”基线。

由于画面绝大多数像素是静态背景，它的全图 MSE 很低；但在真实变化区域内，它会把
旧位置的小车像素原样复制，因此变化区 MAE 最高：

```text
full-frame MSE   = 0.00086896
changed-pixel MAE = 0.601716
```

所以 Raw copy-last 适合揭示全图指标的背景偏置，但不适合单独判断模型是否学会了
运动。

### Information and component comparison

| Method | Sees target frame | Uses encoder/decoder | Uses dynamics | Action input | Main diagnostic purpose |
|---|---:|---:|---:|---|---|
| Oracle | Yes | Yes | No | None | Isolate representation/reconstruction limit |
| World | No | Yes | Yes | Correct aligned actions | Measure deployable one-step prediction |
| Mean-action | No | Yes | Yes | Training mean | Test incremental value of sample-specific actions |
| Shuffled-action | No | Yes | Yes | Wrong window, internally ordered | Test action/vision alignment sensitivity |
| Decoded-last-latent | No | Yes | No | None | Test whether dynamics improves on latent reuse |
| Raw copy-last | No | No | No | None | Expose static-background shortcut |

### Shared changed-pixel metric

所有方法使用同一个变化区域 mask：

```text
changed[p] = any_rgb_channel(x[t+1][p] != x[t][p])
```

`Changed-pixel MAE` 只在 `changed[p] = true` 的像素上，对三个 RGB 通道计算绝对
误差均值。它关注小车移动或转向实际改变的区域，数值越低越好。因为 mask 来自真实
的 `x[t]` 和 `x[t+1]`，它只用于评估，不会成为模型输入。

## Protocol

- Dataset: `data/visual_episodes.npz`
- Dataset SHA-256:
  `2f260fb55c1e90f5d4e836050c9ddfd1e91fa6dab78bb189e2d6bd8736207aba`
- Checkpoint: `artifacts/visual_latent_spatial8.pt`
- Checkpoint SHA-256:
  `1b6d1b2ace6d9da013fbc18dff82de0fa7afb6480f6e7a538aacbacbe17c8792`
- Test windows: `920`
- Evaluation batch size: `256`
- Primary action shuffle seed: `0`
- Robustness check: shuffle seeds `0` through `9`

The evaluator rebuilds latent windows from the checkpoint's recorded test
episode IDs and train-only normalizers. It does not use physical states,
rewards, dones, or labels.

## Primary results

| Evaluation | Normalized latent MSE | Full-frame MSE | Changed-pixel MAE |
|---|---:|---:|---:|
| Oracle | n/a | **0.00151472** | **0.266102** |
| World, recorded actions | **0.0308590** | 0.00158799 | 0.314072 |
| Decoded last latent | n/a | 0.00163793 | 0.359824 |
| Mean-action ablation | 0.0309824 | 0.00158819 | 0.312239 |
| Shuffled actions, seed 0 | 0.0318060 | 0.00158999 | 0.314928 |
| Raw copy-last RGB | n/a | **0.00086896** | 0.601716 |

Raw copy-last still wins whole-frame MSE because most pixels are static, but it
is the worst method on changed pixels. Decoded-last-latent is the appropriate
no-dynamics comparison because it crosses the same encoder and decoder as the
world prediction.

## Does dynamics help?

Yes. Relative to decoded-last-latent, the trained world model reduces
changed-pixel MAE from `0.359824` to `0.314072`, an improvement of `12.72%`.

Using Oracle as the representation-limited lower endpoint:

```text
decoded last -> world -> Oracle
0.359824         0.314072   0.266102    changed-pixel MAE
```

The dynamics model closes approximately `48.8%` of the decoded-last-to-Oracle
changed-pixel gap and `40.5%` of the corresponding full-frame MSE gap. It is
therefore learning useful temporal change rather than merely returning the
last latent.

## Does action alignment help?

The answer is weakly in latent space and barely in decoded pixels:

- Replacing actions with their training mean worsens normalized latent MSE by
  `0.40%`, while changed-pixel MAE improves by `0.58%`.
- Shuffling actions across ten seeds worsens normalized latent MSE by `2.36%`
  on average.
- The same shuffles worsen changed-pixel MAE by only `0.26%` and full-frame MSE
  by `0.09%` on average.

Ten-shuffle robustness summary:

| Metric | Mean | Sample std | Min | Max |
|---|---:|---:|---:|---:|
| Normalized latent MSE | 0.0315880 | 0.0001900 | 0.0312583 | 0.0318390 |
| Full-frame MSE | 0.00158939 | 0.000000576 | 0.00158874 | 0.00159045 |
| Changed-pixel MAE | 0.314896 | 0.000209 | 0.314527 | 0.315156 |

The consistent latent degradation shows that the model detects action
alignment. The tiny and non-monotonic decoded-pixel changes show that current
one-step image quality is driven mainly by visual history, and that lower
latent MSE is not perfectly aligned with small-object pixel quality. This does
not establish that actions are generally unnecessary: one-step rendered
motion may expose little immediate action effect, and longer rollouts can
amplify it.

## Decision

Keep the existing spatial dynamics model as a real learned baseline; it beats
the representation-matched no-dynamics baseline. Do not increase network size
yet. The next diagnostic should ablate temporal history by repeating the last
latent across all four context positions and by perturbing history order. That
will test whether the useful gain comes from observed motion history before a
ConvGRU or temporal Transformer is introduced.

That history diagnostic has now been completed. Repeat-last and
reverse-history changed-pixel MAE degrade to `0.357479` and `0.356096`,
respectively, versus `0.314072` with recorded ordered history. See
`docs/experiments/2026-07-17-spatial-history-diagnostics.md`.

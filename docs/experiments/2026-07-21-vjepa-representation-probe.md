# Frozen V-JEPA 2 representation probe

## Status

Completed. The frozen representation passed three of four preregistered
feasibility gates. The current global pooling candidate is rejected because
held-out centre error remained above `3 px`; no action-conditioned predictor
was trained.

## Question

Before building a new JEPA-style world model, does an official frozen V-JEPA 2
encoder already expose enough state and short-horizon motion information for
this visual environment?

This experiment is intentionally smaller than a world model. It tests only
whether a deterministic linear readout can recover
`[x, y, sin(theta), cos(theta), velocity]` from four rendered frames, and
whether the recovered velocity changes when frame order is destroyed.

## Locked data and encoder

```text
data
  data/visual_episodes.npz
  SHA-256 2f260fb55c1e90f5d4e836050c9ddfd1e91fa6dab78bb189e2d6bd8736207aba
  renderer pillow-raster-v1
  schema 1

encoder
  facebook/vjepa2-vitl-fpc64-256
  revision b3c1679b7c34d3255ef3547f27c7b226aefab26f
  parameters 325,971,328
  frozen true
```

The model revision is an immutable Hugging Face commit. Only the official
configuration, video processor configuration, and safetensors checkpoint were
downloaded. The source model is published by Meta at
[facebook/vjepa2-vitl-fpc64-256](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256),
with reference code in the
[V-JEPA 2 repository](https://github.com/facebookresearch/vjepa2).

## Fixed representation path

```text
four RGB frames [B, 4, 3, 256, 256]
  -> frozen V-JEPA 2 ViT-L encoder
  -> 2 temporal tubelets x 16 x 16 spatial tokens
  -> encoder tokens [B, 512, 1024]

first 256-token tubelet -> spatial mean -> first [B, 1024]
last  256-token tubelet -> spatial mean -> last  [B, 1024]

feature = concat(last, last - first) -> [B, 2048]
  -> deterministic ridge probe, coefficient 1e-3
  -> [x, y, sin(theta), cos(theta), velocity]
```

There is no target encoder, predictor, EMA update, decoder, action input,
fine-tuning, or pixel-generation loss in this probe. The ridge fit uses the
dual solution because the capped training sample count is smaller than the
2048-value feature dimension.

## Protocol

The registered pilot uses split seed `42` and evenly caps the available
four-frame windows at:

```text
train 128
val    32
test   32
batch   1
```

The same fitted probe is evaluated on three test inputs:

1. `recorded`: frames in their real temporal order;
2. `reversed`: the same frames in reverse order;
3. `repeat-last`: the final frame repeated four times.

The state target is unchanged across these variants. Centre and heading test
state readability. Velocity additionally tests whether the representation uses
temporal order instead of only recognizing the final image.

## Preregistered gates

| Gate | Limit |
|---|---:|
| recorded mean centre error | `<= 3 px` |
| recorded mean heading error | `< 45 deg` |
| recorded velocity MAE versus reversed | at least `5%` lower |
| recorded velocity MAE versus repeat-last | at least `5%` lower |

All four gates were required before starting an action-conditioned predictor.

## Environment and runtime

```text
host                  macOS arm64
Python                3.12.13
torch                 2.13.0
torchvision           0.28.0
transformers          5.14.1
device                CPU
CUDA / MPS            false / false
```

The first real smoke test failed before model loading because
`AutoVideoProcessor` requires Torchvision. A dry dependency resolution showed
that `torchvision==0.28.0` matches the existing `torch==2.13.0`; the V-JEPA
optional dependency now declares Torchvision explicitly.

Measured runs:

| Run | Runtime | Approx. encoder time per clip | Peak RSS | Swap |
|---|---:|---:|---:|---:|
| 8 train / 2 val / 2 test smoke | - | `0.29 s` | `1.754 GB` | `0` |
| 128 train / 32 val / 32 test pilot | `78.64 s` | `0.30 s` | `1.772 GB` | `0` |

## Results

| Test input | Centre error (px) | Heading error (deg) | Velocity MAE |
|---|---:|---:|---:|
| mean-target baseline | 18.979349 | 88.042072 | 0.762322 |
| recorded | 4.758912 | 25.384753 | 0.324048 |
| reversed | 4.501779 | 92.370681 | 0.393570 |
| repeat-last | 4.768330 | 58.533563 | 0.982455 |

Relative to the mean-target baseline, the recorded probe improves centre,
heading, and velocity error by `74.93%`, `71.17%`, and `57.49%`. Recorded
velocity MAE is `17.66%` below reversed and `67.02%` below repeat-last.

| Gate | Result |
|---|---:|
| centre `<= 3 px` | **FAIL** (`4.758912 px`) |
| heading `< 45 deg` | PASS (`25.384753 deg`) |
| velocity beats reversed by 5% | PASS (`17.66%`) |
| velocity beats repeat-last by 5% | PASS (`67.02%`) |

## Interpretation

The frozen encoder contains useful state and motion information: heading is
linearly readable, and velocity degrades substantially when temporal order is
reversed or removed. The failure is narrower than “V-JEPA does not work.”

The likely bottleneck is the registered pooling operation. V-JEPA returns a
`16 x 16` spatial token grid for each tubelet, but this candidate averages all
256 positions before fitting the probe. That retains global appearance and
motion signals while discarding the token location needed to localize a small
car precisely. This is an inference from the failure pattern, not yet a
verified cause.

Per the preregistered boundary, the current representation candidate is
rejected and the action-conditioned predictor remains blocked. The threshold
is not relaxed after seeing the result.

## Next decision

The next bounded experiment is **P1b: a spatial-token probe**:

1. keep the same frozen checkpoint, data split, four-frame input, and state
   targets;
2. preserve the `16 x 16` token grid instead of taking a global spatial mean;
3. use a small position-aware readout and keep the same centre, heading, and
   temporal-sensitivity gates;
4. start the action-conditioned predictor only if all four gates pass.

This isolates whether the failed centre gate comes from pooling. It does not
yet train a Micro-JEPA, generate pixels, or change the encoder.

## Limitations

- This is a capped feasibility pilot, not a full benchmark.
- Metrics are sample-level means from one deterministic split, not
  episode-equal aggregates or multi-seed confidence intervals.
- Only a linear ridge readout and one registered pooling function were tested.
- The frozen encoder was not fine-tuned and received no action input.
- The spatial-pooling explanation remains a hypothesis until P1b tests it.

## Registered artifacts

```text
artifacts/vjepa2_probe_features.npz
artifacts/vjepa2_probe_result.json
```

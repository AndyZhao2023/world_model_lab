# Visual object-slot global affine locator

## Status

Completed. The candidate passed three of six representation gates. Per the
registered boundary, no H5 dynamics candidate was trained.

## Question

If the failed spatial-attention locator is replaced by the already-feasible
full-latent affine centre readout, can the existing 11x11 local writer improve
the car while preserving the frozen scene?

## Why this experiment

The first local object-slot candidate established two separate facts:

1. the support mechanism works mechanically: alpha is exactly zero outside at
   most 121 pixels and mean background alpha fell from `0.0255` to `0.0059`;
2. its 1x1 spatial-attention locator failed: held-out centre error was
   `15.7508 px`, despite a full-latent ridge probe reading the same frozen
   representation at `1.6964 px`.

The next controlled change keeps the local writer and replaces only the failed
state-reading path. It does not add a Transformer and does not alter the source
encoder, base decoder, or dynamics.

## Locked source and data

```text
data
  data/visual_episodes.npz
  SHA-256 2f260fb55c1e90f5d4e836050c9ddfd1e91fa6dab78bb189e2d6bd8736207aba

source
  artifacts/visual_latent_spatial8_objective_w01.pt
  SHA-256 5022fe5f11c099550f6971f6f92362505126beb70883b415f1a2ad338e803369
```

The source encoder, base decoder, dynamics tensors, normalizers, split IDs,
renderer version, and four-frame action alignment remain exact.

## Locator feasibility

A deterministic ridge map with coefficient `1e-3` is fit from normalized
source latents to
`[image_cx, image_cy, sin(theta), cos(theta)]` using training frames only.

For centre rows, the normalized-latent affine map

```text
y = Wn * ((z - mean) / std) + bn
```

is converted exactly into a raw-latent layer:

```text
Wr = Wn / std
br = bn - Wn * (mean / std)
y  = Wr * z + br
```

The pre-implementation check over 1020 held-out frames produced:

```text
maximum converted prediction delta  1.8896e-13
mean centre error                    1.6963808911 px
p95 centre error                     5.6991804073 px
```

The locator-only feasibility gate is therefore passed before candidate
training.

## Fixed architecture

```text
image
  -> frozen spatial encoder
  -> latent grid [B, 8, 8, 8]
  -> flatten [B, 512]

flattened latent
  -> frozen affine centre head 512 -> 2
  -> [cx, cy]

flattened latent
  -> trainable Linear 512 -> 64
  -> ReLU
  -> trainable Linear 64 -> 2
  -> L2 normalize
  -> [sin(theta), cos(theta)]

slot = [cx, cy, sin(theta), cos(theta)]

slot
  -> Linear 4 -> 64
  -> ReLU
  -> Linear 64 -> 4 * 11 * 11
  -> local foreground RGB + local alpha
  -> differentiable placement at [cx, cy]
  -> hard 11x11 support clamp

composite = frozen_base * (1 - placed_alpha)
          + placed_foreground * placed_alpha
```

The centre affine parameters are initialized from the train-only ridge fit and
remain frozen. Only the heading MLP and local patch decoder may train.
Renderer state and masks are privileged training supervision; inference uses
only the image latent.

## Fixed objective

The loss remains identical to the first local-slot experiment:

```text
total =
    full-frame MSE
    + 1.0 * foreground object MSE
    + 0.01 * balanced alpha BCE
    + 1.0 * centre MSE
    + 0.1 * heading unit-vector MSE
```

The centre term is reported in the objective but cannot move the frozen centre
head. This keeps validation values comparable while guaranteeing that patch
training cannot trade away the feasible centre.

Fixed training protocol:

```text
locator               global_affine
centre probe ridge     0.001
centre trainable       false
heading hidden size    64
patch size             11
patch hidden size      64
initial alpha          0.01
epochs                 20
batch size             128
learning rate          0.001
optimizer              Adam
seed                   0
split seed             42, inherited from source
```

No seed, architecture, weight, epoch, or hidden-size sweep is allowed.

## Representation gates

The candidate passes only if all six gates pass:

| Gate | Limit |
|---|---:|
| held-out object MSE | `< 0.1580440933868` |
| held-out full-frame MSE | `<= 0.0016618691395` |
| held-out background MSE | `<= 0.0006603598343` |
| H5 oracle cumulative changed-pixel MAE | `< 0.1996767445825` |
| held-out mean centre error stability | `<= 1.7811998513 px` |
| held-out mean heading error | `< 81.6836050288 deg` |

The centre limit is 105% of the source affine probe's
`1.6963808107 px`; it is a stability gate because the same frozen readout is
being embedded, not a claim that this experiment improves representation
position information.

The four image gates use the existing matched diagnostic over 18 eligible
held-out episodes and 136 deterministic windows. State metrics use all 1020
held-out frames.

## Conditional H5 stage

Any failed representation gate stops the experiment. Only if all six pass:

1. freeze the enhanced autoencoder;
2. train one fresh H5 recursive dynamics candidate;
3. run the existing ten-seed matched simulator counterfactual diagnostic;
4. require every preregistered H5 dynamics gate before promotion or MPC work.

## Registered outputs

```text
artifacts/visual_latent_spatial8_object_slot_affine.pt
artifacts/visual_latent_spatial8_object_slot_affine_predictions.png
artifacts/diagnostics/visual-object-slot-affine/
```

Conditional:

```text
artifacts/visual_latent_spatial8_object_slot_affine_h5.pt
artifacts/diagnostics/visual-object-slot-affine-h5-counterfactual/
```

## Decision boundary

This experiment tests whether a known-good centre tied to a structurally local
decoder closes the prior failure. Passing centre stability alone is expected
and insufficient. The candidate must also improve heading, object rendering,
the H5 oracle moving-pixel floor, and preserve both full-frame and background
quality.

## Results

The single registered candidate trained for 20 epochs. Validation selected
epoch `14`:

```text
initial train objective       0.2382596408
final train objective         0.1710256562
best validation objective     0.2534706716
```

### Locator and heading

The normalized-to-raw affine conversion embedded the train-only probe with a
maximum float32 prediction delta of `4.0184e-07`.

| Held-out metric | source affine probe | candidate | relative change |
|---|---:|---:|---:|
| mean centre error | 1.6963808107 px | 1.6963242026 px | -0.0033% |
| mean heading error | 81.6836050288 deg | 84.3681286382 deg | +3.29% |

The centre stability gate passes. The previous `15.75 px` failure was caused
by the restricted 1x1 spatial-attention reader, not by missing position
information in the source latent. A Transformer is therefore not required to
recover the centre in this lab.

The trainable global heading MLP does not improve held-out heading and fails
its strict gate.

### Reconstruction

The matched diagnostic used the same 18 eligible held-out episodes and 136
deterministic windows:

| Held-out metric | source | candidate | relative change |
|---|---:|---:|---:|
| object-region MSE | 0.1580440934 | 0.0776697559 | -50.86% |
| full-frame MSE | 0.0015107901 | 0.0029748690 | +96.91% |
| background MSE | 0.0006003271 | 0.0025404125 | +323.17% |
| H5 oracle cumulative changed-pixel MAE | 0.1996767446 | 0.1899594067 | -4.87% |

Correct centre placement materially improves both the car and the H5
moving-pixel oracle floor. It still damages the scene around the car.

### Local support diagnostics

| Mechanistic metric | candidate |
|---|---:|
| maximum support pixels | 121 |
| object mask IoU at alpha 0.5 | 0.3777407099 |
| precision | 0.4061902105 |
| recall | 0.8435844371 |
| mean object alpha | 0.8290054018 |
| mean background alpha | 0.0098826936 |

The exact renderer object occupies only 13 to 24 pixels per frame, while the
square support permits up to 121. With the centre now correct, alpha is high
over the car but also overwrites non-object pixels inside the local square.
Those pixels count as background even though alpha remains exactly zero across
the rest of the image. The preview shows a yellow/blue halo around the car,
matching the full-frame and background failures.

This failure is narrower than the dense residual's global leak: the branch now
writes in the right neighborhood, but its support shape and foreground
composition are not precise enough inside that neighborhood.

### Gates

| Gate | limit | candidate | result |
|---|---:|---:|---|
| held-out object MSE strictly improves | 0.1580440934 | 0.0776697559 | PASS |
| held-out full-frame MSE <= 110% source | 0.0016618691 | 0.0029748690 | **FAIL** |
| held-out background MSE <= 110% source | 0.0006603598 | 0.0025404125 | **FAIL** |
| H5 oracle cumulative changed-pixel MAE strictly improves | 0.1996767446 | 0.1899594067 | PASS |
| held-out centre error <= 105% source | 1.7811998513 px | 1.6963242026 px | PASS |
| held-out heading error strictly improves | 81.6836050288 deg | 84.3681286382 deg | **FAIL** |

This is three of six gates. No H5 dynamics checkpoint or matched H5
counterfactual diagnostic was produced.

## Artifacts

Ignored local outputs:

```text
artifacts/visual_latent_spatial8_object_slot_affine.pt
artifacts/visual_latent_spatial8_object_slot_affine_predictions.png
artifacts/diagnostics/visual-object-slot-affine/
```

SHA-256:

```text
5a7a4012f700f332501e1ed121811387e78b2fd7049a2fd4e44c13332b2306c6  visual_latent_spatial8_object_slot_affine.pt
31f06af71a1c278898ce25355301f99f92ba90695a9c74f5f2e367c4c70ed9fa  visual_latent_spatial8_object_slot_affine_predictions.png
ec3578007daec57b96325b9f2758e6e831ae97003106e76590a118a08023c9ad  manifest.json
f55415cc19944db6d6b2f5fd81a416169f5499fe18b238adc2d9f778a3e9cc33  metrics.json
213fbac09f567e96e4ddd4350d222594c0e3d95aa89767de90506fe371d8d065  visual_autoencoder_comparison.png
```

## Decision

Reject the candidate and retain
`artifacts/visual_latent_spatial8_objective_w01.pt` as the current default.
Do not train conditional H5 dynamics and do not connect this candidate to MPC.

The experiment resolves the locator question:

1. a simple frozen global affine map is sufficient for the centre;
2. correct centre placement is useful, because object MSE and the H5 oracle
   moving-pixel floor both improve;
3. the remaining failures are heading generalization and false-positive
   support inside the local square.

A justified follow-up should preserve the frozen affine centre and replace the
square RGBA overwrite with a shape-constrained object mask or a
base-preserving correction inside the window. It should not add a Transformer
for centre localization and should not tune the same mask-loss scalar against
the already-used held-out benchmark.

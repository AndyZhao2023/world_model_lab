# Visual object residual decoder

## Status

Completed. Candidate rejected after passing 2 of 4 representation gates; the
conditional H5 stage was not run.

## Question

Can a learned object-only foreground/alpha head improve car reconstruction
without modifying the source encoder or the already-good background decoder?

## Motivation

The equal-region autoencoder objective reduced held-out object MSE by `70.15%`
but made full-frame MSE `5.22x` worse and background MSE `12.76x` worse. It
proved that direct object supervision can recover the car, but applying that
pressure to the entire decoder destroys the static scene.

This experiment moves the object correction across a narrower boundary:

```text
source latent
  ├── frozen base decoder ───────────────> base RGB
  └── learned object head ──> foreground RGB + alpha

composite = base * (1 - alpha) + foreground * alpha
```

At inference the head receives only the latent. The renderer-derived mask is
a training target, not an input.

## Locked source and invariants

```text
data
  data/visual_episodes.npz
  SHA-256 2f260fb55c1e90f5d4e836050c9ddfd1e91fa6dab78bb189e2d6bd8736207aba

source
  artifacts/visual_latent_spatial8_objective_w01.pt
  SHA-256 5022fe5f11c099550f6971f6f92362505126beb70883b415f1a2ad338e803369
```

The following must remain exact:

- `[B, 8, 8, 8]` source encoder and all encoder tensors;
- source base RGB decoder and all decoder tensors;
- source one-step dynamics tensors;
- latent/action normalizers;
- train/validation/test episode IDs;
- data digest and renderer version.

Only the new object head may train.

## Fixed architecture

From the `[B, 8, 8, 8]` latent grid:

```text
ConvTranspose 8 -> 32,  8x8 -> 16x16
ReLU
ConvTranspose 32 -> 16, 16x16 -> 32x32
ReLU
ConvTranspose 16 -> 4,  32x32 -> 64x64

channels 0:3 -> sigmoid -> foreground RGB
channel 3    -> alpha logit -> sigmoid -> alpha
```

The final alpha bias is initialized to `log(0.01 / 0.99)`, so the initial
candidate is close to the source decoder and the new branch must earn the
right to overwrite pixels.

## Fixed objective

For exact renderer object mask `m`:

```text
full loss =
    mean((composite - target)^2)

foreground object loss =
    sum(m * (foreground - target)^2)
    / number of masked RGB values

balanced mask BCE =
    mean BCE on object pixels
    + mean BCE on background pixels

total =
    full loss
    + 1.0 * foreground object loss
    + 0.01 * balanced mask BCE
```

The full-frame term protects the existing scene. The foreground term teaches
object colour. The mask term teaches where the branch may write and penalizes
false-positive alpha across the background.

Fixed training protocol:

```text
head hidden channels 16
initial alpha         0.01
epochs                20
batch size            128
learning rate         0.001
optimizer             Adam
seed                  0
split seed            42, inherited from source
```

No weight, channel, seed, or epoch sweep is allowed.

## Representation gates

Use the existing matched autoencoder diagnostic over 18 test episodes and 136
deterministic windows:

| Gate | Limit |
|---|---:|
| held-out object MSE | `< 0.1580440933868` |
| held-out full-frame MSE | `<= 0.0016618691395` |
| held-out background MSE | `<= 0.0006603598343` |
| H5 oracle cumulative changed-pixel MAE | `< 0.1996767445825` |

Mask IoU, precision, recall, object alpha, and background alpha are reported
as mechanistic diagnostics but are not post-hoc promotion gates.

Any gate failure stops the experiment before H5.

## Conditional H5 stage

If the representation passes, freeze the entire enhanced autoencoder and
train fresh H5 dynamics:

```text
one-step loss =
    normalized latent MSE + 0.1 * decoded changed-pixel MAE

H5 loss =
    mean of the same objective over five self-fed steps

total = one-step loss + 1.0 * H5 loss
```

Compare residual H1 against residual H5 under the existing ten-seed matched
simulator protocol. Because the encoder is unchanged, latent coordinates and
normalizers remain directly comparable; because both sides use the same
enhanced decoder, decoded comparisons remain controlled.

## Registered outputs

```text
artifacts/visual_latent_spatial8_object_residual.pt
artifacts/visual_latent_spatial8_object_residual_predictions.png
artifacts/diagnostics/visual-object-residual/
```

Conditional:

```text
artifacts/visual_latent_spatial8_object_residual_h5.pt
artifacts/visual_latent_spatial8_object_residual_h5_predictions.png
artifacts/diagnostics/visual-object-residual-h5-counterfactual/
```

## Decision boundary

Object improvement alone is insufficient. The candidate must preserve the
source scene, improve the H5 oracle moving-pixel floor, and then pass the
separate H5 dynamics gates before replacing the current default or entering
MPC work.

## Results

The single registered candidate trained for 20 epochs. Validation selected
epoch `19`:

```text
initial train objective       0.1104824110
final train objective         0.0474003545
best validation objective     0.0573087752
```

The diagnostic used the same 18 held-out episodes and 136 deterministic
windows as the source:

| Held-out metric | source | candidate | relative change |
|---|---:|---:|---:|
| object-region MSE | 0.1580440934 | 0.0695477366 | -55.99% |
| full-frame MSE | 0.0015107901 | 0.0024444846 | +61.80% |
| background MSE | 0.0006003271 | 0.0020541841 | +242.18% |
| H5 oracle cumulative changed-pixel MAE | 0.1996767446 | 0.1756834692 | -12.02% |

The residual branch materially improved the car and the H5 moving-pixel
oracle floor. It also preserved the source encoder, base decoder, dynamics,
normalizers, and episode splits exactly. However, blending the learned branch
still damaged the composite background even though the base decoder itself
was frozen.

### Alpha-mask diagnostics

| Held-out metric | value |
|---|---:|
| object mask IoU at alpha 0.5 | 0.4143886773 |
| precision | 0.4301886061 |
| recall | 0.9185844371 |
| mean object alpha | 0.8860656738 |
| mean background alpha | 0.0255067018 |

The mechanism is high-recall but low-precision. A background alpha of only
`2.55%` is still consequential because the held-out background contains
about `172x` as many pixels as the object region. The preview confirms weak
residual texture across the scene and colour halos around static objects, not
only a localized correction over the car.

### Gates

| Gate | limit | candidate | result |
|---|---:|---:|---|
| held-out object MSE strictly improves | 0.1580440934 | 0.0695477366 | PASS |
| held-out full-frame MSE <= 110% source | 0.0016618691 | 0.0024444846 | **FAIL** |
| held-out background MSE <= 110% source | 0.0006603598 | 0.0020541841 | **FAIL** |
| H5 oracle cumulative changed-pixel MAE strictly improves | 0.1996767446 | 0.1756834692 | PASS |

This is 2 of 4 gates. Per the registered boundary, no residual H5 dynamics
checkpoint or matched H5 counterfactual diagnostic was produced.

## Artifacts

Ignored local outputs:

```text
artifacts/visual_latent_spatial8_object_residual.pt
artifacts/visual_latent_spatial8_object_residual_predictions.png
artifacts/diagnostics/visual-object-residual/
```

SHA-256:

```text
c2a96ef33e30d79a034ae9e42d68c481f79c49d6f4376e67a83bf5bc62fe260f  visual_latent_spatial8_object_residual.pt
4b35394c45c7a17b64acf7c5213d533786c7467417663598e14e93953c395c97  visual_latent_spatial8_object_residual_predictions.png
f64b2a5f9509cb6a038692b755a9fdd33536bfcc057fb38371ece7d89e2eeb65  manifest.json
7abd6671abe20410646767b0b739e8112ce9fce4ccb6665cff21527b77c9f1b0  metrics.json
e1ef1ef320ad720417ba4904e20be0db61da36deb8fd392afae6cb3496e8c2ee  visual_autoencoder_comparison.png
```

## Decision

Reject the residual candidate and retain the current spatial representation.
Do not train its conditional H5 dynamics model and do not connect it to MPC.

Compared with the previous equal-region candidate, the residual mechanism is
directionally better: it passes the H5 oracle gate and reduces background
damage substantially. The remaining failure is now localized to alpha
precision rather than destruction of the frozen base path. A future
experiment should change the support mechanism itself, for example a
strictly sparse or renderer-independent learned object slot, rather than
performing a scalar sweep on the same held-out benchmark.

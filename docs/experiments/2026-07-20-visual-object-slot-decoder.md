# Visual object-slot local-write decoder

## Status

Completed. The candidate passed one of six representation gates. Per the
registered boundary, no H5 dynamics candidate was trained.

## Question

Can an explicit object slot improve car reconstruction while making it
structurally impossible for the learned object branch to write across the
whole background?

## Why this experiment

The dense residual decoder improved held-out object MSE by `55.99%` and H5
oracle cumulative changed-pixel MAE by `12.02%`, but failed both scene
stability gates:

```text
full-frame MSE  +61.80%
background MSE +242.18%
```

Its mean background alpha was only `0.0255`, but background pixels outnumber
object pixels by about `172:1`. A small dense leak was therefore enough to
dominate the aggregate error. This experiment changes the support mechanism,
not the weight on the same dense alpha objective.

## Locked source and feasibility measurements

```text
data
  data/visual_episodes.npz
  SHA-256 2f260fb55c1e90f5d4e836050c9ddfd1e91fa6dab78bb189e2d6bd8736207aba

source
  artifacts/visual_latent_spatial8_objective_w01.pt
  SHA-256 5022fe5f11c099550f6971f6f92362505126beb70883b415f1a2ad338e803369
```

The renderer-derived object mask was measured over all `9018` aligned frames:

```text
empty object masks                 0
maximum object bounding box        8 x 8 pixels
maximum offset from rounded centre +/- 5 pixels
edge-touching object masks         430
object pixels per frame            13 to 24
```

The local write window is therefore fixed at `11 x 11`. It covers every
observed car/heading mask relative to its rounded centre, including clipped
boundary cases, while limiting the branch to at most about `2.95%` of the
image.

A deterministic ridge probe with coefficient `1e-3` was fit from normalized
source latents on training frames only. Its fixed held-out baselines are:

| Metric | source |
|---|---:|
| mean centre error | `1.6963808730 px` |
| mean heading error | `81.6836092677 deg` |

The heading target is `(sin(theta), cos(theta))`, so the objective has no angle
wrap discontinuity.

## Fixed architecture

The source encoder, base decoder, dynamics model, normalizers, and episode
splits remain exact. Only the new object-slot branch may train.

```text
image
  -> frozen spatial encoder
  -> latent grid [B, 8, 8, 8]
     -> learned 1x1 attention logits
     -> spatial softmax over 8x8
     -> expected image centre [cx, cy]
     -> attention-weighted 8-D object feature
     -> linear heading head
     -> normalized [sin(theta), cos(theta)]

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

The placement uses normalized image coordinates and differentiable bilinear
sampling. The final support clamp makes placed alpha exactly zero outside the
11x11 window. Renderer masks and physical state are training supervision only;
inference receives only the image latent.

The local alpha bias is initialized to `log(0.01 / 0.99)`, keeping the initial
candidate close to the source decoder.

## Fixed objective

For target image `x`, renderer object mask `m`, target centre `c`, and target
heading unit vector `h`:

```text
full MSE =
    mean((composite - x)^2)

foreground object MSE =
    sum(m * (placed_foreground - x)^2)
    / number of masked RGB values

balanced alpha BCE =
    mean BCE(alpha, m) on object pixels
    + mean BCE(alpha, m) on background pixels

centre MSE =
    mean((predicted_centre - c)^2)

heading MSE =
    mean((predicted_heading - h)^2)

total =
    full MSE
    + 1.0 * foreground object MSE
    + 0.01 * balanced alpha BCE
    + 1.0 * centre MSE
    + 0.1 * heading MSE
```

Fixed training protocol:

```text
patch size             11
patch hidden size      64
initial alpha          0.01
epochs                 20
batch size             128
learning rate          0.001
optimizer              Adam
seed                   0
split seed             42, inherited from source
source probe ridge     0.001
```

No architecture, loss-weight, epoch, seed, or patch-size sweep is allowed.
Best epoch is selected by the complete validation objective.

## Representation gates

The candidate passes only if all six gates pass:

| Gate | Limit |
|---|---:|
| held-out object MSE | `< 0.1580440933868` |
| held-out full-frame MSE | `<= 0.0016618691395` |
| held-out background MSE | `<= 0.0006603598343` |
| H5 oracle cumulative changed-pixel MAE | `< 0.1996767445825` |
| held-out mean centre error | `< 1.6963808730 px` |
| held-out mean heading error | `< 81.6836092677 deg` |

The first four gates use the existing matched autoencoder diagnostic over 18
eligible test episodes and 136 deterministic windows. The state metrics use
all 1020 held-out test frames. Alpha IoU, precision, recall, mean object alpha,
mean background alpha, and support size are mechanistic diagnostics, not
post-hoc promotion gates.

## Conditional H5 stage

If any representation gate fails, stop and retain the current source model.

Only if all six pass:

1. freeze the entire enhanced autoencoder;
2. train one fresh H5 recursive dynamics candidate with the existing
   one-step plus H5 decoded changed-pixel objective;
3. run the existing ten-seed matched simulator counterfactual protocol;
4. require all preregistered H5 dynamics gates before promotion.

No MPC work starts from a representation that fails these gates.

## Registered outputs

```text
artifacts/visual_latent_spatial8_object_slot.pt
artifacts/visual_latent_spatial8_object_slot_predictions.png
artifacts/diagnostics/visual-object-slot/
```

Conditional:

```text
artifacts/visual_latent_spatial8_object_slot_h5.pt
artifacts/diagnostics/visual-object-slot-h5-counterfactual/
```

## Decision boundary

This experiment is successful only if the object slot is more physically
readable, improves moving-object rendering, and preserves the existing scene.
A centre/head metric win or an object-MSE win alone is insufficient.

## Results

The single registered candidate trained for 20 epochs. Validation selected
epoch `20`:

```text
initial train objective       1.8302296930
final train objective         1.3090778940
best validation objective     1.3789342753
```

### State readout

| Held-out metric | source ridge probe | candidate slot | relative change |
|---|---:|---:|---:|
| mean centre error | 1.6963808107 px | 15.7507923799 px | +828.50% |
| mean heading error | 81.6836050288 deg | 84.9939860026 deg | +4.05% |

The frozen latent does contain position information: the full 512-dimensional
ridge probe still reads the centre at `1.70 px`. The candidate's much narrower
mechanism—a shared 1x1 score over each 8-channel cell followed by spatial
softmax—does not find the car. A spatial latent grid preserves image topology,
but that does not guarantee one channel-local object saliency map already
exists.

### Reconstruction

The matched diagnostic used the same 18 eligible held-out episodes and 136
deterministic windows:

| Held-out metric | source | candidate | relative change |
|---|---:|---:|---:|
| object-region MSE | 0.1580440934 | 0.1504808405 | -4.79% |
| full-frame MSE | 0.0015107901 | 0.0018729984 | +23.97% |
| background MSE | 0.0006003271 | 0.0010086332 | +68.01% |
| H5 oracle cumulative changed-pixel MAE | 0.1996767446 | 0.2023588409 | +1.34% |

The support constraint worked mechanically:

| Mechanistic metric | candidate |
|---|---:|
| maximum support pixels | 121 |
| object mask IoU at alpha 0.5 | 0.0249190939 |
| precision | 0.1595671195 |
| recall | 0.0286837748 |
| mean object alpha | 0.0484393369 |
| mean background alpha | 0.0058572674 |

Mean background alpha fell from the dense residual candidate's `0.0255` to
`0.0059`, and alpha is exactly zero outside the local support. However, the
predicted centre is usually wrong, so the permitted window edits unrelated
background pixels and rarely overlaps the true car. Structural locality
prevents a full-image leak; it cannot compensate for a failed object locator.

### Gates

| Gate | limit | candidate | result |
|---|---:|---:|---|
| held-out object MSE strictly improves | 0.1580440934 | 0.1504808405 | PASS |
| held-out full-frame MSE <= 110% source | 0.0016618691 | 0.0018729984 | **FAIL** |
| held-out background MSE <= 110% source | 0.0006603598 | 0.0010086332 | **FAIL** |
| H5 oracle cumulative changed-pixel MAE strictly improves | 0.1996767446 | 0.2023588409 | **FAIL** |
| held-out centre error strictly improves | 1.6963808107 px | 15.7507923799 px | **FAIL** |
| held-out heading error strictly improves | 81.6836050288 deg | 84.9939860026 deg | **FAIL** |

This is one of six gates. No H5 dynamics checkpoint or matched H5
counterfactual diagnostic was produced.

## Artifacts

Ignored local outputs:

```text
artifacts/visual_latent_spatial8_object_slot.pt
artifacts/visual_latent_spatial8_object_slot_predictions.png
artifacts/diagnostics/visual-object-slot/
```

SHA-256:

```text
a39d3c78d6d6d0d5e2c9b70bbf01aac36c84bb91eb05fca089f12817eea55079  visual_latent_spatial8_object_slot.pt
27e8f8568941e948c4f1671528a6ce1b0fa0e0fac682543121420fb1553b61c6  visual_latent_spatial8_object_slot_predictions.png
2c19fb32a0559158e8db93bd5366cf58cd479bbf62d63ace9c226fc41795fd85  manifest.json
fc0799c89b3ed922c2756fd5df526b17d96155e7f15babc497b2b09fa30d3500  metrics.json
7ad5ea613db78abc46824b9e7070af83fc506911994db829df144bc7a5bb8783  visual_autoencoder_comparison.png
```

## Decision

Reject the candidate and retain
`artifacts/visual_latent_spatial8_objective_w01.pt` as the current default.
Do not train conditional H5 dynamics and do not connect this candidate to MPC.

The experiment separates two questions:

1. **Local support:** successful mechanically. It makes global alpha leakage
   impossible and lowers mean background alpha by about fourfold relative to
   the dense residual.
2. **Object localization:** unsuccessful. The lightweight spatial-attention
   head discards position information that a full-latent linear probe can
   recover.

A justified follow-up should retain the 11x11 local writer but replace only
the failed locator with the already-demonstrated full-latent affine centre
readout. That would test whether a correctly tied centre-to-pixel path can
preserve the background without repeating the earlier dynamics-only
position-loss experiment.

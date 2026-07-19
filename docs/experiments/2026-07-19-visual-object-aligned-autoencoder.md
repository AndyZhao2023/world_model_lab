# Visual object-aligned autoencoder

## Status

Completed. The representation candidate passed one of four pre-registered
gates, so training stopped before the conditional H5 dynamics stage.

## Question

Can a renderer-derived object-region reconstruction objective make the spatial
autoencoder preserve the small car in both its representation and RGB decoder,
before we ask latent dynamics to predict longer trajectories?

## Why this experiment

The H5 object-position candidate reduced matched normalized position MSE from
`0.1468615915` to `0.0234369108`, but H5 cumulative changed-pixel MAE worsened
from `0.2764975452` to `0.3062392396`. The frozen position probe could read the
car centre from latent space, while the frozen decoder did not render the
corresponding moving-car pixels.

That isolates the next broken boundary:

```text
physical target -> latent position: improved
latent position -> rendered moving object: still poor
```

The next controlled change therefore belongs in the autoencoder, not in
another dynamics position-loss sweep.

## Locked source and data

```text
data
  data/visual_episodes.npz
  SHA-256 2f260fb55c1e90f5d4e836050c9ddfd1e91fa6dab78bb189e2d6bd8736207aba

source representation
  artifacts/visual_latent_spatial8_objective_w01.pt
  [B, 8, 8, 8] latent
  base channels 16
  autoencoder epochs 20
  autoencoder batch size 128
  learning rate 0.001
  training seed 0
  split seed 42
```

Source held-out reconstruction:

| Metric | Value |
|---|---:|
| full-frame MSE | 0.0015107901268 |
| full-frame MAE | 0.0151237188601 |
| object-region MSE | 0.1580440933868 |
| object-region MAE | 0.3283802360888 |
| background MSE | 0.0006003271221 |

Exact renderer object pixels account for `213,797` pixels over the full
dataset, or `0.57880387%`. A plain full-frame average is therefore dominated
by the static scene.

## Controlled change

For every current frame, define:

```text
object_mask =
    pixel RGB equals CAR_COLOR
    OR pixel RGB equals HEADING_COLOR
```

For positive object weight:

```text
object MSE =
    sum squared RGB error inside object mask
    / number of masked RGB values

background MSE =
    sum squared RGB error outside object mask
    / number of background RGB values

autoencoder loss =
    background MSE + 1.0 * object MSE
```

Each region is normalized independently. Thus the small object receives one
region-level vote and the background receives one region-level vote. The
background term also penalizes false car pixels outside the true object.

This differs from the rejected global-latent motion-weight experiments:

- the latent remains spatial `[8, 8, 8]`;
- the target is the current rendered object, not only frame-to-frame change;
- no scalar weight `100/500` multiplies a tiny subset inside one global
  denominator;
- only one registered weight, `1.0`, is trained.

Architecture, split, optimizer, epoch budget, and seed remain fixed. Positive
motion and object weights are prohibited together.

## Pre-registered representation gates

The candidate passes only if all four gates pass:

| Gate | Limit |
|---|---:|
| held-out object-region MSE | `< 0.1580440933868` |
| held-out full-frame MSE | `<= 0.0016618691395` |
| held-out background MSE | `<= 0.0006603598343` |
| H5 oracle cumulative changed-pixel MAE | `< 0.1996767445825` |

The full/background limits permit at most 10% regression. The H5 oracle gate
uses the same factual test episodes and target-frame indices for both
autoencoders. Each model reconstructs true future frames through its own
encoder and decoder; no learned dynamics participates.

Source-versus-source diagnostics locked the exact H1/H5/H10 oracle values
before candidate creation:

| Horizon | full-frame MSE | object MSE | background MSE | cumulative changed-pixel MAE |
|---:|---:|---:|---:|---:|
| 1 | 0.0014161407 | 0.1424530673 | 0.0005917136 | 0.2142595117 |
| 5 | 0.0014455051 | 0.1495510282 | 0.0005817449 | 0.1996767446 |
| 10 | 0.0014866989 | 0.1647175693 | 0.0005570519 | 0.1928875371 |

The source-versus-source run passed both 10% stability gates and failed the
two strict-improvement gates, as required. It used 18 eligible test episodes
and 136 deterministic windows.

Baseline bundle digests:

```text
d75917f13d3c4c2b9a9408c69ced88c6b8aea4b72efe47981f9c8da69bc85668  manifest.json
78b66cf34bdf8bc122d1c4d8199b638ab7b7fc861b6ff3151a67f53561fbc7ab  metrics.json
a9aa5369b0c12062f7bec081c654aeddc476e7b9d238cf880fcac5c534443b72  visual_autoencoder_comparison.png
```

If any representation gate fails, the experiment stops. H5 dynamics will not
be trained.

## Conditional H5 stage

Only an accepted representation is frozen and passed to recursive dynamics:

```text
one-step objective =
    normalized latent MSE
    + 0.1 * decoded changed-pixel MAE

H5 objective =
    mean over five self-fed predicted steps of the same objective

total dynamics loss =
    one-step objective + 1.0 * H5 objective
```

The new representation has a different latent coordinate system. Therefore
raw normalized latent MSE is model-local diagnostic evidence, not a valid
cross-model promotion metric. Final comparison must use common pixel metrics
and physical counterfactual effects under identical simulator branches.

## Registered outputs

```text
artifacts/visual_latent_spatial8_object_aligned_w1.pt
artifacts/visual_latent_spatial8_object_aligned_w1_predictions.png
artifacts/diagnostics/visual-object-autoencoder-w1/
```

Conditional outputs:

```text
artifacts/visual_latent_spatial8_object_aligned_w1_h5.pt
artifacts/visual_latent_spatial8_object_aligned_w1_h5_predictions.png
```

## Decision boundary

Do not promote based only on a lower object-region training loss. Promotion
first requires held-out object improvement, stable scene reconstruction, and
a lower H5 oracle moving-pixel floor. Only then is another H5 dynamics run
informative.

## Results

The single registered candidate trained for 20 autoencoder epochs. Validation
selected epoch `19`:

```text
initial train objective       0.2481097955
final train objective         0.0482332884
best validation objective     0.0548882786
epoch 20 validation objective 0.0551444567
```

The objective successfully concentrated capacity on the rendered car:

| Held-out metric | source | candidate | relative change |
|---|---:|---:|---:|
| object-region MSE | 0.1580440934 | 0.0471738988 | -70.15% |
| object-region MAE | 0.3283802361 | 0.1438693520 | -56.19% |
| full-frame MSE | 0.0015107901 | 0.0078870368 | +422.05% |
| background MSE | 0.0006003271 | 0.0076585282 | +1175.73% |
| full-frame PSNR | 28.2079586201 | 21.0308613192 | -7.18 dB |

Thus independently normalizing the two regions overcorrected the original
imbalance. The object region contributes about half of the configured loss
despite occupying only `0.5788%` of pixels. The decoder renders the car more
strongly, but also produces broad colour halos around the car, obstacle, goal,
and otherwise static background.

### Oracle rollout reconstruction

Both models reconstructed the exact same true target frames through their own
encoder and decoder. No learned dynamics participated.

| Horizon | Metric | source | candidate | relative change |
|---:|---|---:|---:|---:|
| 1 | object MSE | 0.1424530673 | 0.0455512600 | -68.02% |
| 1 | full-frame MSE | 0.0014161407 | 0.0081872489 | +478.14% |
| 1 | cumulative changed-pixel MAE | 0.2142595117 | 0.2500475739 | +16.70% |
| 5 | object MSE | 0.1495510282 | 0.0460997526 | -69.17% |
| 5 | full-frame MSE | 0.0014455051 | 0.0080948725 | +460.00% |
| 5 | cumulative changed-pixel MAE | 0.1996767446 | 0.2144154992 | +7.38% |
| 10 | object MSE | 0.1647175693 | 0.0484080883 | -70.61% |
| 10 | full-frame MSE | 0.0014866989 | 0.0076193150 | +412.50% |
| 10 | cumulative changed-pixel MAE | 0.1928875371 | 0.1607435380 | -16.66% |

The H10 changed-pixel metric is secondary and does not reverse the H5
decision. At longer displacement, the cumulative mask covers a larger region;
the candidate's stronger car-coloured reconstruction can lower the average
inside that broad mask while the full image remains substantially corrupted.
The registered decision horizon is H5, where the candidate is worse.

### Gates

| Gate | limit | candidate | result |
|---|---:|---:|---|
| held-out object MSE strictly improves | 0.1580440934 | 0.0471738988 | PASS |
| held-out full-frame MSE <= 110% source | 0.0016618691 | 0.0078870368 | **FAIL** |
| held-out background MSE <= 110% source | 0.0006603598 | 0.0076585282 | **FAIL** |
| H5 oracle cumulative changed-pixel MAE strictly improves | 0.1996767446 | 0.2144154992 | **FAIL** |

The diagnostic used 18 eligible episodes and 136 windows, unchanged from the
locked source run.

## Artifacts

Ignored local outputs:

```text
artifacts/visual_latent_spatial8_object_aligned_w1.pt
artifacts/visual_latent_spatial8_object_aligned_w1_predictions.png
artifacts/diagnostics/visual-object-autoencoder-w1/
```

SHA-256:

```text
87dce81dcd31e1e7847b9fd855bacc374195bdc27b7402bacaee96772d03e7ba  visual_latent_spatial8_object_aligned_w1.pt
3e29e6a3bbb2df2d7542cf9b19d8b23f74326149574b0dd597fba8b622e96ccd  visual_latent_spatial8_object_aligned_w1_predictions.png
6149976c7bdf32e17ce13b6e4b1e4af02fd83075ef831f65320dc8bc73d19dc0  manifest.json
0835e225a75d413cce4246f655566771120aa004d520bdaa130cfa347b3fe647  metrics.json
719b009434658f059b939f76973be67f2f8f9ad359049dc9cec00f775b0aa6b5  visual_autoencoder_comparison.png
```

## Decision

Reject the candidate and retain the current spatial representation. Do not
train the conditional H5 dynamics checkpoint and do not connect this
autoencoder to MPC.

The experiment confirms that the decoder can be pushed toward the car, but an
equal region-level objective is too aggressive: it trades a 70.15% object-MSE
gain for a 5.22x full-frame MSE and 12.76x background MSE. The next
representation experiment should change the mechanism rather than sweep this
scalar weight against the same held-out benchmark. A narrower direction is
object-aware latent routing or a residual/object decoder head that leaves the
already-good background path intact, with the same four gates applied before
any dynamics training.

import VersoManual

open Verso.Genre Manual

#doc (Manual) "Three End-to-End Case Studies" =>
%%%
tag := "model-examples-deep-dive"
%%%

This chapter follows three case studies far enough to inspect the objects that move through
TorchLean. The first is a character-level Transformer, where sequence length, masking, and
generation matter. The second compares two runs, residual and patch-token vision models, on the same
prepared dataset. The third is a Fourier neural operator, where the important boundary is the
spectral kernel rather than an image or token representation.

The commands below were run against the current checkout. The displayed losses are deterministic
for the stated seeds and one-example datasets, but they are not performance benchmarks. Their
purpose is to make the data path, model shape, runtime selection, and generated artifacts concrete.

# Case Study One: CharGPT

The Tiny Shakespeare experiment is the clearest sequence-model application because it begins with
a text file and ends with both a trained parameter file and generated text.

Prepare the corpus:

```
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
```

Then run the two-update smoke configuration:

```
lake -R -K cuda=true exe torchlean chargpt --device cuda \
  --tiny-shakespeare --preset smoke \
  --save-params /tmp/chargpt-params.json \
  --log /tmp/chargpt-trainlog.json
```

The current run reports:

```
torchlean chargpt: char-level GPT training
  trainable_parameters=30017
  step 0: val loss=4.207507
  step 1: val loss=4.195092
  step 2: val loss=4.174880
  wrote params: /tmp/chargpt-params.json
  vocab=65 (unique chars)
  architecture=width 32, heads 4, layers 2, dropout 0.000000
  sampled="First Citizen:ITJ?P,bduOc$Eaf'yjhYXGLHkR3vQq;V"
  wrote TrainLog JSON: /tmp/chargpt-trainlog.json
torchlean chargpt: ok
```

The sample is mostly noise because two optimizer updates are not language training. The useful
facts are elsewhere in the trace:

- the corpus produced a 65-character vocabulary;
- the model has 30,017 trainable scalar parameters;
- validation is computed on a disjoint ten-percent corpus suffix;
- the loss decreased at both reported updates;
- parameter and training-log artifacts were written explicitly.

## The Architecture

The command constructs a `CausalTransformerConfig` from the runtime options:

$$`D=32,\qquad H=4,\qquad D_h=D/H=8,\qquad L=2.`

For a batch of token windows $`I\in\operatorname{Fin}(V)`, the model first gathers embeddings

$$`E[I]\in\mathbb{R}^{B\times T\times D},`

adds a learned positional table, and applies two pre-normalized Transformer blocks. In one block,

$$`\begin{aligned}
Z_1&=X+\operatorname{Dropout}
  \left(\operatorname{MHA}(\operatorname{LN}(X))\right),\\
Z_2&=Z_1+\operatorname{Dropout}
  \left(W_2\,\rho(W_1\operatorname{LN}(Z_1)+b_1)+b_2\right).
\end{aligned}`

The final layer normalization and linear projection produce logits

$$`\operatorname{logits}\in\mathbb{R}^{B\times T\times V}.`

The training target is the same token window shifted by one position. Cross entropy at location
$`t` therefore uses the target character $`x_{t+1}`.

The mask is semantic, not merely a convenient floating-point bias:

$$`j>i\quad\Longrightarrow\quad
\operatorname{attentionWeight}_{i,j}=0.`

The source of the shared architecture is
[NN/API/Models/Gpt2.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/Gpt2.lean).
The corpus split, configuration presets, evaluation loop, generation, and checkpoint handling are
in
[NN/Examples/Models/Sequence/CharGpt.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Sequence/CharGpt.lean).

## Make The Run Your Own

Every structural parameter in the smoke preset can be overridden:

```
lake -R -K cuda=true exe torchlean chargpt --device cuda \
  --tiny-shakespeare --preset smoke \
  --width 64 --heads 4 --layers 3 --seq-len 64 \
  --batch 8 --steps 20 --eval-every 5 --eval-iters 4
```

Two constraints are worth testing deliberately.

First, `--heads` must divide `--width`, because the model width is split into equal head dimensions.
Try `--width 30 --heads 4`; the command rejects the configuration before training.

Second, increasing `seq-len` increases attention work quadratically:

$$`\operatorname{cost}_{\mathrm{attention}}
=O(BHT^2D_h).`

Increasing width mostly changes matrix multiplication; increasing context changes both the score
matrix and its stored autograd state. This is why a “larger Transformer” is not described by
parameter count alone.

The `karpathy` preset records the well-known lecture configuration: batch 64, context 256, width
384, six heads, six blocks, dropout 0.2, and 5,000 updates. It is a real long-running experiment,
not a quick check:

```
lake -R -K cuda=true exe torchlean chargpt --device cuda \
  --tiny-shakespeare --preset karpathy
```

TorchLean follows the architecture and hyperparameter lineage of Karpathy's lecture model, but the
runtime and implementation are TorchLean's. It does not claim bit-for-bit identity with the Python
program or with a pretrained GPT-2 checkpoint.

# Case Study Two: Two Views Of CIFAR

ResNet and ViT consume the same prepared CIFAR arrays but impose different structure on them. The
examples make that difference visible while sharing the same trainer, optimizer, loss, and runtime
options.

```
python3 scripts/datasets/download_example_data.py --cifar10

lake exe torchlean resnet --device cpu --n-total 1 --steps 1 \
  --log /tmp/resnet-trainlog.json

lake exe torchlean vit --device cpu --n-total 1 --steps 1 \
  --log /tmp/vit-trainlog.json
```

Observed summaries:

```
torchlean resnet: ResNet CIFAR training (device=cpu)
dataset size = 1
mean_loss(before) = 2.333221
mean_loss(after) = 2.327496
steps=1 loss0=2.333221 loss1=2.327496
torchlean resnet: ok
```

```
torchlean vit: ViT CIFAR training (device=cpu)
dataset size = 1
mean_loss(before) = 2.302585
mean_loss(after) = 2.300389
steps=1 loss0=2.302585 loss1=2.300389
torchlean vit: ok
```

Both are ten-class models, but they organize computation differently.

## Residual Geometry

The current
[`ResNet application`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Vision/ResNet.lean)
crops each image to $`3\times8\times8`, lifts it to four hidden channels, applies two shape-preserving
residual blocks, globally averages the spatial axes, and emits ten logits.

At every residual join, both branches have shape

$$`1\times4\times8\times8.`

That equality is part of model construction. The optimizer never sees a malformed residual block.
It may still see a numerically poor model, a bad label, or an incorrect unproved kernel; shape
typing solves one problem rather than pretending to solve all of them.

## Patch Geometry

The
[`ViT application`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Vision/Vit.lean)
uses a convolution to create patch embeddings, reshapes the patch grid into a token sequence, and
applies one Transformer encoder block. If the patch output grid is $`H'\times W'`, the token count is

$$`N=H'W'.`

The conversion

$$`B\times D\times H'\times W'
\longrightarrow B\times N\times D`

is an explicit layer in the reusable ViT model. It is not a hidden `view` whose correctness depends on
remembering which axis currently stores channels.

## A Useful Comparison

Try ten updates on both commands:

```
lake exe torchlean resnet --device cpu --n-total 8 --steps 10
lake exe torchlean vit --device cpu --n-total 8 --steps 10
```

Do not compare the final losses as if this were a controlled architecture benchmark. The examples
are intentionally compact and their initialization and capacity differ. Instead, compare:

- model summaries and parameter shapes;
- the residual join versus the spatial-to-token conversion;
- the backend capsules printed by adding `--show-backend`;
- the JSON metadata written by `--log`.

That comparison teaches more about TorchLean's architecture than a single accuracy number.

# Case Study Three: A Burgers Neural Operator

The Fourier neural operator example learns the terminal-time solution map for the viscous Burgers
equation

$$`\partial_t u+u\,\partial_xu=\nu\,\partial_{xx}u,
\qquad x\in[0,1],`

from sampled initial conditions. Each training pair is

$$`u_0(x_i)\longmapsto u_T(x_i),\qquad i=0,\ldots,31.`

Prepare the dataset:

```
python3 NN/Examples/Data/prepare_fno1d_burgers.py \
  --download --grid 32 --ntrain 128 --ntest 32
```

A one-update CUDA run over four training fields and two held-out fields is:

```
lake -R -K cuda=true exe torchlean fno1d_burgers --device cuda \
  --steps 1 --lr 0.003 \
  --train-rows 4 --test-rows 2 --eval-rows 2 \
  --log /tmp/fno-trainlog.json \
  --plot-csv /tmp/fno-predictions.csv
```

The current output identifies the numerical path before reporting the loss:

```
torchlean fno1d_burgers: native real-split FNO1D Burgers
  device=cuda backend=eager
  grid=32 width=8 modes=8 blocks=1
  rows train=4 test=2 eval_prefix=2
  spectral path=fused cuFFT RFFT autograd op
  before: train_mse=0.482041 test_mse=0.486850
  step 1: train_mse=0.481925 test_mse=0.486735
  after: train_mse=0.481925 test_mse=0.486735
  wrote prediction CSV: /tmp/fno-predictions.csv
  wrote TrainLog JSON: /tmp/fno-trainlog.json
torchlean fno1d_burgers: ok
```

Plot the prediction artifact with:

```
python3 NN/Examples/Data/plot_fno1d_burgers.py \
  --csv /tmp/fno-predictions.csv
```

## What The Spectral Layer Computes

Let $`v\in\mathbb R^{N\times C}` be a field with $`C` latent channels. The spectral branch computes
a discrete Fourier transform, retains $`m` modes at each end of the real spectrum, multiplies those modes by
learned complex weights, and transforms back:

$$`\widehat v_k
=\sum_{j=0}^{N-1}v_j e^{-2\pi i jk/N},`

$$`\widehat y_k
=R_{\theta,k}\widehat v_k
\quad\text{for retained }k,\qquad
y=\mathcal F^{-1}(\widehat y).`

A pointwise linear branch is added before the activation. The command uses grid $`N=32`, width
$`8`, eight retained modes on each side, and one spectral residual block.

The reusable
[`FNO constructor`](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/FNO.lean)
states the grid and mode constraints independently of the backend. The
[`Burgers application`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Operators/Fno1dBurgers.lean)
chooses between:

- a portable dense multidimensional DFT path;
- a fused CUDA real-FFT autograd operation backed by cuFFT.

The output line `spectral path=fused cuFFT RFFT autograd op` is therefore part of the scientific
record. It says which external numerical provider produced the transform. It does not turn cuFFT
into a Lean-proved implementation.

## Change The Evidence, Not Only The Runtime

Run the same small dataset on CPU:

```
lake exe torchlean fno1d_burgers --device cpu \
  --steps 1 --train-rows 4 --test-rows 2 --eval-rows 2
```

The command reports `spectral path=portable dense multidimensional DFT`. The model contract and
dataset remain the same, but the numerical provider changes. This is precisely the kind of
comparison for which backend capsules are useful: provider, reduction policy, layout, and evidence
can change without silently changing the model's mathematical interface.

# Reading The Case Studies

The four application runs stress different parts of TorchLean:

:::table +header
*
  * Application
  * Structural pressure
  * External boundary
  * Primary artifact
*
  * CharGPT
  * causal windows, heads, depth, token IDs
  * CUDA kernels and corpus file
  * checkpoint, validation log, generated text
*
  * ResNet / ViT
  * residual joins or patch-token layout
  * CIFAR arrays and selected runtime
  * classification loss log
*
  * FNO
  * field shape, retained Fourier modes
  * dataset preparation and cuFFT on CUDA
  * train/test loss and prediction CSV
:::

The next two chapters add stochastic state. In generative modeling and reinforcement learning, the
network is only one part of the run, so schedules, samplers, environments, and rollout data matter
as much as the architecture.

Checkpoint and dataset files in these runs are runtime artifacts, not proof objects. Loading one
checks its declared schema and dimensions where the command implements those checks; it does not
establish provenance, reproduce the optimizer history, or prove that two files with the same shape
encode the same model or dataset.

# References

- Karpathy,
  [*Let's build GPT: from scratch, in code, spelled out*](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py),
  2023.
- He et al., [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385),
  2015/2016.
- Dosovitskiy et al.,
  [*An Image Is Worth 16x16 Words*](https://arxiv.org/abs/2010.11929), 2020/2021.
- Li et al.,
  [*Fourier Neural Operator for Parametric Partial Differential Equations*](https://arxiv.org/abs/2010.08895),
  2020/2021.

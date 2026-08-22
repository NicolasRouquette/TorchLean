import VersoManual

open Verso.Genre Manual

#doc (Manual) "Modern Models" =>
%%%
tag := "modern-models"
%%%

A two-layer perceptron is enough to introduce typed tensors and reverse-mode differentiation. It is
not enough to test whether an ML library has found the right abstractions. Modern architectures add
branches, state, masks, patch grids, spectral transforms, latent variables, and interaction with an
environment. Each addition creates a new place where an informal convention can become a bug.

Each architecture makes a different convention explicit. A residual network must align two
branches. A vision Transformer must turn a patch grid into a token sequence. A recurrent model must
expose state, a causal model must expose its mask, and an FNO must say which Fourier modes it retains.

Through all of them, the outer discipline stays the same. Inputs, outputs, and parameter layout are
typed; the forward program names its runtime; specifications and proofs refer to identifiable
objects rather than to an opaque training log; and a dataset, environment, or external kernel stays
visible where it enters the run.

The reusable constructors live under
[`NN/API/Models`](https://github.com/lean-dojo/TorchLean/tree/main/NN/API/Models).
The runnable applications live under
[`NN/Examples/Models`](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Models).
A constructor such as `nn.models.resnet` is reusable and
rank-polymorphic; the `resnet` command chooses a deliberately small CIFAR configuration so that a
reader can run the whole data and training path locally.

# From A Formula To A Typed Model

Consider an ordinary one-hidden-layer network

$$`f_\theta(x)=W_2\,\rho(W_1x+b_1)+b_2.`

For a batch of $`B` vectors, the shapes are

$$`X\in\mathbb{R}^{B\times d_{\mathrm{in}}},\qquad
W_1\in\mathbb{R}^{d_{\mathrm{in}}\times d_h},\qquad
W_2\in\mathbb{R}^{d_h\times d_{\mathrm{out}}}.`

The TorchLean constructor records the outer contract as

```
nn.Sequential
  (shape![batch, inDim])
  (shape![batch, outDim])
```

and composes layers only when their intermediate shapes agree. The runtime still performs ordinary
matrix multiplication, bias addition, and activation evaluation. The type checker removes one
class of invalid programs before that runtime is reached.

The same pattern survives when the architecture becomes less linear. What changes is the shape
equation that must be made explicit.

# Residual Networks

A residual block computes

$$`y=F_\theta(x)+S_\theta(x),`

where $`S` is either the identity or a projection. The addition is defined only if the two branches
have the same output shape. In
[`NN.API.Models.ResNet`](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/ResNet.lean),
the reusable configuration is indexed by the number $`d` of spatial axes:

```
structure ResNetConfig (d : Nat) where
  inChannels     : Nat
  spatial        : Vector Nat d
  spatialNonzero : ∀ i, spatial.get i ≠ 0
  hiddenChannels : Nat
  numClasses     : Nat
```

The constructor accepts a separate leading shape $`L`. Its input and output are

$$`\operatorname{input}
=L\mathbin{+\!+}(C_{\mathrm{in}},n_1,\ldots,n_d),`

$$`\operatorname{output}=L\mathbin{+\!+}(C_{\mathrm{class}}).`

The constructor uses a convolutional stem, two residual blocks, global average pooling over all
$`d` spatial axes, and a linear classifier. The pooling operation is parameterized by the spatial
vector. The CIFAR example instantiates $`d=2` and $`L=(B)`; the model API itself is not tied to
images, two dimensions, or one batch axis.

The next chapter runs this compact CIFAR application and inspects the resulting loss log. Here we
stay with the architectural question: where must the two residual branches agree?

## Try A Shape Change

Open
[`NN/Examples/Models/Vision/ResNet.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Vision/ResNet.lean)
and inspect `cfg`. The command uses an $`8\times8` crop and four hidden channels. Changing the hidden
width changes both residual branches and the classifier input. Removing the projection that keeps
the branch shapes aligned is not a late runtime error: the residual composition no longer
elaborates.

# Vision Transformers

A vision Transformer first turns a spatial field into a token sequence. For an input with spatial
extent $`n_1\times\cdots\times n_d`, patch kernel $`k`, stride $`s`, and padding $`p`, each output
extent is the usual convolution expression

$$`n'_i
=\left\lfloor\frac{n_i+2p_i-k_i}{s_i}\right\rfloor+1.`

If the patch convolution emits $`D` channels, then the patch grid becomes

$$`B\times D\times n'_1\times\cdots\times n'_d
\;\longrightarrow\;
B\times N\times D,\qquad
N=\prod_i n'_i.`

[`NN.API.Models.Vit`](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/Vit.lean)
defines this conversion as `spatialToTokens`. The implementation reshapes the patch grid and moves
the channel axis to the end. The following Transformer block therefore receives the conventional
`batch × sequence × embedding` layout.

For $`H` attention heads of width $`D_h`, the model width is

$$`D=H D_h,`

and scaled dot-product attention is

$$`\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{D_h}}+M\right)V.`

The current ViT application uses one encoder block and flattens all patch tokens before the final
classifier. It is a compact architecture check, not an implementation of a particular pretrained
ViT checkpoint.
The next chapter runs it beside the compact ResNet so their different spatial conventions can be
compared on the same prepared data.

# Recurrence, Attention, And Causality

Sequence models add state or a causal dependency.

For a vanilla recurrent network,

$$`h_{t+1}=\phi(W_xx_t+W_hh_t+b),\qquad
y_t=W_yh_t+b_y.`

An LSTM replaces the single update by input, forget, output, and candidate gates:

$$`\begin{aligned}
i_t&=\sigma(W_ix_t+U_ih_{t-1}+b_i),\\
f_t&=\sigma(W_fx_t+U_fh_{t-1}+b_f),\\
o_t&=\sigma(W_ox_t+U_oh_{t-1}+b_o),\\
\tilde c_t&=\tanh(W_cx_t+U_ch_{t-1}+b_c),\\
c_t&=f_t\odot c_{t-1}+i_t\odot\tilde c_t,\\
h_t&=o_t\odot\tanh(c_t).
\end{aligned}`

The hidden and cell states are explicit tensors whose dimensions must remain stable across the
unrolled sequence. The runnable `rnn`, `lstm`, and `lstm_regression` commands use that typed state
path. There is currently no GRU training subcommand. The LiRPA verifier has a GRU-gate certificate
fixture, but that is a different artifact and should not be presented as a trainable GRU model.

Transformers remove recurrent state but introduce a mask. A causal language model factors

$$`p_\theta(x_0,\ldots,x_T)
=\prod_{t=0}^{T}p_\theta(x_t\mid x_0,\ldots,x_{t-1}).`

Consequently, attention row $`i` must assign exactly zero probability to every key $`j>i`. TorchLean
models this as a hard mask in the semantic layer. It does not use a finite additive constant such
as $`-1000` as a mathematical substitute for negative infinity.

The shared GPT-family constructor in
[`NN.API.Models.CausalTransformer`](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/CausalTransformer.lean)
has the architecture

```
token embedding
  -> learned positional embedding
  -> masked Transformer blocks
  -> layer normalization
  -> vocabulary projection
```

Its configuration records sequence length, vocabulary size, head count, head width, feed-forward
width, depth, activation, dropout, normalization order, and initialization. A model constructor
then receives the leading shape used by a particular run, so batch size is not part of the
Transformer architecture. The name “GPT-2-style” describes this architecture lineage. It does not
mean that the command loads OpenAI GPT-2 weights.

# State-Space Models

Mamba-style models return to explicit state but allow input-dependent scan parameters. A simplified
state-space recurrence is

$$`h_{t+1}=A_t h_t+B_t x_t,\qquad
y_t=C_t h_t+D_t x_t.`

[`NN.API.Models.Mamba`](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/Mamba.lean)
contains two related objects:

- `Mamba.textLM`, the trainable recurrent model used by the text command;
- `Mamba.Reference.selective`, a deterministic full selective block used for reference evaluation.

The trainable path uses TorchLean autograd operations on CPU or CUDA. The repository also has a
selective-scan CUDA operation for supported float execution. That runtime kernel is not thereby
proved equivalent to every equation in the high-level Mamba specification; the kernel boundary is
reported separately.

# Neural Operators

An FNO learns a map between functions sampled on a grid rather than a map between fixed feature
vectors. One spectral block has the schematic form

$$`v_{\ell+1}(x)
=\sigma\!\left(W_\ell v_\ell(x)
+\mathcal F^{-1}\!\left(R_\ell\cdot\mathcal F(v_\ell)\right)(x)\right).`

The configuration in
[`NN.API.Models.FNO`](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/FNO.lean)
is again parameterized by spatial rank. For every axis it requires

$$`2m_i\le n_i,`

so the retained low- and high-frequency bands do not overlap. The portable implementation uses a
dense multidimensional DFT. The CUDA Burgers application selects a fused real-FFT path backed by
cuFFT. Both paths implement the same typed field-to-field interface, but their numerical and trust
boundaries are recorded separately.

# From Architectures To Runs

We have seen how the model families are assembled. The next chapter turns them into three case
studies spanning four runs: a character Transformer, ResNet and ViT on the same vision data, and a
Fourier neural operator. It prepares their data, launches training, and inspects the artifacts they
leave behind.

# References

- He et al., [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385),
  2015/2016.
- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), 2017.
- Dosovitskiy et al.,
  [*An Image Is Worth 16x16 Words*](https://arxiv.org/abs/2010.11929), 2020/2021.
- Gu and Dao,
  [*Mamba: Linear-Time Sequence Modeling with Selective State Spaces*](https://arxiv.org/abs/2312.00752),
  2023/2024.
- Li et al.,
  [*Fourier Neural Operator for Parametric Partial Differential Equations*](https://arxiv.org/abs/2010.08895),
  2020/2021.

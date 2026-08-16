# `NN.Spec.Layers`

This directory contains TorchLean's layer definitions: common neural-network building blocks written
as pure functions on spec tensors and parameter records.

The emphasis is a clear reference definition. Runtime code, graph lowering, CUDA kernels, and
verification passes should be able to point back here when they need the mathematical meaning of an
operation. For layers used by reverse-mode training, the spec layer also records derivative or VJP
rules so the backward meaning is visible.

Most models in `NN/Spec/Models/*` are built by composing these layer specs, sometimes through the
`NN/Spec/Module/*` wrappers that provide the `NNModuleSpec` interface for `SpecChain`.

Files:

- `Activation.lean`: scalar activation formulas under `Activation.Math`, tensor activation wrappers,
  real last-axis softmax/log-softmax, and VJP specs.
- `Linear.lean`: fully connected layer spec ($y=Wx+b$) and gradients.
- `Attention.lean`: scaled dot product attention and multihead attention with hard-mask semantics;
  blocked weights are exactly zero, and a fully blocked row evaluates to the zero vector
  semantics and VJPs.
- `FlashAttention.lean`: FlashAttention style tiling metadata/specs tied back to the same attention
  semantics.
- `Conv.lean`: 1D/2D convolution and transposed convolution specs plus explicit backward rules.
- `Pooling.lean`: max/avg pooling, padded pooling, adaptive pooling, and smooth max pooling
  surrogates, including backward/JVP rules.
- `Pooling/`: two-dimensional, padded, and N-dimensional helpers used by the main pooling
  umbrella.
- `GlobalPooling.lean`: global avg/max pooling and backward rules.
- `Normalization.lean`: LayerNorm and BatchNorm style utilities with explicit backward specs.
- `Embedding.lean`: one-hot embeddings (`oneHot @ W`) and the corresponding VJP.
- `PositionalEncoding.lean`: learnable/sinusoidal positional encodings and RoPE style rotations.
- `Dropout.lean`: deterministic inference and mask-driven training dropout specs.
- `Loss.lean`: common scalar losses and their derivatives.
- `Gnn.lean`: a compact GCN-style graph layer and backward rules.
- `Rnn.lean`, `Lstm.lean`, `Gru.lean`: recurrent layers and BPTT-style backwards.
- `SelectiveScan.lean`: affine scan primitives used by S4/Mamba style state space models.
- `Utils.lean`: shared indexing and sliding-window geometry used by convolution and pooling layers.

The underlying tensor primitives (maps, matmul, reshape, broadcasting) live under `NN/Spec/Core/*`.

This folder defines meanings, not runtime performance. A typed graph node, fused CUDA kernel, or
ATen provider should be documented against the spec operation it implements or approximates.

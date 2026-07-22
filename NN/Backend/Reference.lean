/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Capsule

/-!
# Reference and Portable Backend Capsules

Portable capsules for paths that do not require CUDA or LibTorch.

These are not meant to win benchmarks. They keep TorchLean runnable on CPU-only machines and give
the planner a clear fallback vocabulary for reference/spec-aligned execution.
-/

@[expose] public section

namespace NN
namespace Backend
namespace Reference

/-- Build a checked portable CPU capsule with explicit value, VJP, shape, and layout contracts. -/
def referenceCapsule
    (name : String) (op : BackendOp) (specName valueSummary vjpSummary : String)
    (vjpMode : VJPMode := .torchLeanTape) : KernelCapsule :=
  { name
    op
    provider := .reference
    device := .cpu
    trustLevel := .checked
    supportsForward := true
    vjpMode
    shapeContract :=
      { claim := .shapeSafety op
        summary := "Shapes are checked in Lean before entering the portable runtime path."
        evidence := .runtimeGuard "portable runtime shape checks" }
    layoutContract :=
      { claim := .layoutCompatibility op .canonicalTensor
        summary := "Portable paths use TorchLean's canonical tensor representation."
        evidence := .runtimeGuard "typed tensor layout" }
    valueContract :=
      { claim := .valueRefinement op specName
        summary := valueSummary
        evidence := .testSuite "NN.Tests.Runtime.Floats.Suite" }
    vjpContract :=
      { claim := .vjpRefinement op specName vjpMode
        summary := vjpSummary
        evidence := .testSuite "NN.Tests.Runtime.Floats.Suite" }
    numericalPolicy :=
      { rounding := .scalarContext
        subnormals := .implementationDefined
        contraction := .notApplicable
        reduction := .notApplicable }
    notes := "Reference/portable capsules are the cross-platform fallback, not the scaling path." }

/-- Build the standard portable capsule for a pointwise operation. -/
def referencePointwiseCapsule (op : BackendOp) : KernelCapsule :=
  referenceCapsule
    s!"reference.{op.name}"
    op
    s!"Spec.{op.name}"
    s!"Reference `{op.name}` follows the pointwise tensor contract."
    s!"TorchLean tape supplies the `{op.name}` VJP where differentiable."

/-- Build a portable reduction capsule with deterministic left-to-right accumulation. -/
def referenceReductionCapsule (op : BackendOp) : KernelCapsule :=
  { referenceCapsule
    s!"reference.{op.name}"
    op
    s!"IR.{op.name} / Spec reduction contract"
    s!"Reference `{op.name}` follows the explicit reduction shape contract."
    s!"TorchLean tape supplies the `{op.name}` adjoint where differentiable." with
    numericalPolicy.reduction := .fixedLeft }

/-- Reference kernel whose scalar result contains an explicit left-to-right accumulation.

Matrix products, affine layers, convolutions, and averaging operations all reduce several products
or samples into one output entry. Keeping this constructor separate from pointwise kernels prevents
the numerical audit from incorrectly reporting that reduction order is irrelevant. -/
def referenceAccumulationCapsule (name : String) (op : BackendOp) (specName valueSummary
    vjpSummary : String) : KernelCapsule :=
  { referenceCapsule name op specName valueSummary vjpSummary with
    numericalPolicy.reduction := .fixedLeft }

/-- Build the standard portable capsule for a shape or layout transformation. -/
def referenceViewCapsule (op : BackendOp) : KernelCapsule :=
  referenceCapsule
    s!"reference.{op.name}"
    op
    s!"IR.{op.name} shape/layout contract"
    s!"Reference `{op.name}` follows the explicit shape/layout contract."
    s!"TorchLean tape supplies the `{op.name}` adjoint where differentiable."

/-- Build a portable forward-only capsule with no registered reverse derivative. -/
def referenceForwardOnlyCapsule (op : BackendOp) (specName valueSummary : String) : KernelCapsule :=
  referenceCapsule
    s!"reference.{op.name}"
    op
    specName
    valueSummary
    s!"Reference `{op.name}` is a forward-only capsule with no registered VJP."
    .none

/-- Build a portable capsule for channel-first convolution or pooling. -/
def referenceConvPoolCapsule (op : BackendOp) : KernelCapsule :=
  referenceCapsule
    s!"reference.{op.name}"
    op
    s!"Spec.{op.name} / channel-first convolution-pooling contract"
    s!"Reference `{op.name}` follows the channel-first runtime contract."
    s!"TorchLean tape supplies the `{op.name}` VJP where differentiable."

/-- Reference window selection with deterministic traversal and tie handling. -/
def referenceSelectionCapsule (op : BackendOp) : KernelCapsule :=
  { referenceConvPoolCapsule op with
    numericalPolicy.reduction := .fixedLeft }

/-- Reference ReLU activation. -/
def relu : KernelCapsule :=
  referenceCapsule
    "reference.relu"
    .relu
    "Spec.relu / pointwise max(x, 0)"
    "Reference ReLU follows pointwise tensor semantics."
    "TorchLean tape supplies the VJP."

/-- Reference GELU activation. -/
def gelu : KernelCapsule :=
  referenceCapsule
    "reference.gelu"
    .gelu
    "Spec.gelu"
    "Reference GELU follows the documented runtime approximation contract."
    "TorchLean tape supplies the VJP."

/-- Reference pointwise addition. -/
def add : KernelCapsule := referencePointwiseCapsule .add
/-- Reference pointwise subtraction. -/
def sub : KernelCapsule := referencePointwiseCapsule .sub
/-- Reference pointwise multiplication. -/
def mul : KernelCapsule := referencePointwiseCapsule .mul
/-- Reference scalar multiplication. -/
def scale : KernelCapsule := referencePointwiseCapsule .scale
/-- Reference pointwise absolute value. -/
def abs : KernelCapsule := referencePointwiseCapsule .abs
/-- Reference pointwise square root. -/
def sqrt : KernelCapsule := referencePointwiseCapsule .sqrt
/-- Reference pointwise interval clamp. -/
def clamp : KernelCapsule := referencePointwiseCapsule .clamp
/-- Reference pointwise maximum. -/
def max : KernelCapsule := referencePointwiseCapsule .max
/-- Reference pointwise minimum. -/
def min : KernelCapsule := referencePointwiseCapsule .min
/-- Reference pointwise sigmoid. -/
def sigmoid : KernelCapsule := referencePointwiseCapsule .sigmoid
/-- Reference pointwise hyperbolic tangent. -/
def tanh : KernelCapsule := referencePointwiseCapsule .tanh
/-- Reference pointwise softplus. -/
def softplus : KernelCapsule := referencePointwiseCapsule .softplus
/-- Reference pointwise exponential. -/
def exp : KernelCapsule := referencePointwiseCapsule .exp
/-- Reference pointwise natural logarithm. -/
def log : KernelCapsule := referencePointwiseCapsule .log
/-- Reference pointwise sine. -/
def sin : KernelCapsule := referencePointwiseCapsule .sin
/-- Reference pointwise cosine. -/
def cos : KernelCapsule := referencePointwiseCapsule .cos
/-- Reference pointwise reciprocal. -/
def inv : KernelCapsule := referencePointwiseCapsule .inv
/-- Reference guarded logarithm used by numerically defensive programs. -/
def safeLog : KernelCapsule := referencePointwiseCapsule .safeLog
/-- Reference log-softmax reduction and normalization. -/
def logSoftmax : KernelCapsule :=
  referenceAccumulationCapsule
    "reference.log_softmax"
    .logSoftmax
    "Spec.logSoftmax"
    "Reference log-softmax follows the stable row/axis normalization contract."
    "TorchLean tape supplies the VJP."

/-- Reference softmax path. -/
def softmax : KernelCapsule :=
  referenceAccumulationCapsule
    "reference.softmax"
    .softmax
    "Spec.softmax"
    "Reference softmax follows the row/axis normalization contract."
    "TorchLean tape supplies the VJP."

/-- Reference sum reduction. -/
def reduceSum : KernelCapsule := referenceReductionCapsule .reduceSum
/-- Reference arithmetic-mean reduction. -/
def reduceMean : KernelCapsule := referenceReductionCapsule .reduceMean

/-- Reference shape-preserving reshape view. -/
def reshape : KernelCapsule := referenceViewCapsule .reshape
/-- Reference axis permutation. -/
def permute : KernelCapsule := referenceViewCapsule .permute
/-- Reference tensor broadcasting. -/
def broadcast : KernelCapsule := referenceViewCapsule .broadcast
/-- Reference tensor concatenation. -/
def concat : KernelCapsule := referenceViewCapsule .concat
/-- Reference contiguous tensor slice. -/
def slice : KernelCapsule := referenceViewCapsule .slice
/-- Reference indexed gather. -/
def gather : KernelCapsule := referenceViewCapsule .gather
/-- Reference indexed scatter-add. -/
def scatterAdd : KernelCapsule := referenceViewCapsule .scatterAdd

/-- Reference seeded uniform-random tensor generation. -/
def randUniform : KernelCapsule :=
  referenceForwardOnlyCapsule
    .randUniform
    "IR.randUniform"
    "Reference deterministic random-uniform tensors follow the seeded spec/runtime contract."

/-- Reference seeded Bernoulli-mask generation. -/
def bernoulliMask : KernelCapsule :=
  referenceForwardOnlyCapsule
    .bernoulliMask
    "IR.bernoulliMask"
    "Reference deterministic Bernoulli masks follow the seeded spec/runtime contract."

/-- Reference matmul path. -/
def matmul : KernelCapsule :=
  referenceAccumulationCapsule
    "reference.matmul"
    .matmul
    "Spec.matmul / IR.matmul"
    "Portable matmul follows the spec-level matrix product contract."
    "TorchLean tape supplies the VJP."

/-- Reference linear layer. -/
def linear : KernelCapsule :=
  referenceAccumulationCapsule
    "reference.linear"
    .linear
    "Spec.linear"
    "Reference linear follows the matvec/matmul plus bias contract."
    "TorchLean tape supplies the VJP."

/-- Reference mean-squared-error loss. -/
def mseLoss : KernelCapsule :=
  referenceAccumulationCapsule
    "reference.mse_loss"
    .mseLoss
    "Spec.mseLoss"
    "Reference MSE follows the mean squared residual contract."
    "TorchLean tape supplies the VJP."

/-- Reference layer normalization. -/
def layerNorm : KernelCapsule :=
  referenceAccumulationCapsule
    "reference.layer_norm"
    .layerNorm
    "Spec.layerNorm"
    "Reference LayerNorm follows the per-row normalization contract."
    "TorchLean tape supplies the VJP."

/-- Reference batch normalization. -/
def batchNorm : KernelCapsule :=
  referenceAccumulationCapsule
    "reference.batch_norm"
    .batchNorm
    "Spec.batchNorm"
    "Reference BatchNorm follows the channel-first normalization contract."
    "TorchLean tape supplies the VJP."

/-- Reference generic channel-first convolution. -/
def conv : KernelCapsule :=
  referenceAccumulationCapsule
    "reference.conv"
    .conv
    "Spec.conv"
    "Reference convolution follows the generic channel-first contract."
    "TorchLean tape supplies the VJP."

/-- Reference generic channel-first transpose convolution. -/
def convTranspose : KernelCapsule :=
  referenceAccumulationCapsule
    "reference.conv_transpose"
    .convTranspose
    "Spec.convTranspose"
    "Reference transpose convolution follows the generic channel-first contract."
    "TorchLean tape supplies the VJP."

/-- Reference max pooling. -/
def maxPool : KernelCapsule :=
  referenceSelectionCapsule .maxPool

/-- Reference smooth max pooling. -/
def smoothMaxPool : KernelCapsule :=
  referenceAccumulationCapsule
    "reference.smooth_max_pool"
    .smoothMaxPool
    "Spec.smoothMaxPool"
    "Reference smooth max pooling follows the generic window contract."
    "TorchLean tape supplies the VJP."

/-- Reference average pooling. -/
def avgPool : KernelCapsule :=
  referenceAccumulationCapsule
    "reference.avg_pool"
    .avgPool
    "Spec.avgPool"
    "Reference average-pooling follows the channel-first window contract."
    "TorchLean tape supplies the VJP."

/-- Reference attention path using the composed TorchLean expression. -/
def attention : KernelCapsule :=
  { name := "reference.attention"
    op := .scaledDotProductAttention
    provider := .reference
    device := .cpu
    trustLevel := .checked
    supportsForward := true
    vjpMode := .torchLeanTape
    shapeContract := ContractDescriptor.guarded (.shapeSafety .scaledDotProductAttention)
      "Q/K/V and mask shapes are checked by the typed tensor layer."
      "typed attention shapes"
    layoutContract := ContractDescriptor.guarded
      (.layoutCompatibility .scaledDotProductAttention .canonicalTensor)
      "Reference attention uses TorchLean tensor semantics rather than a foreign layout."
      "typed tensor layout"
    valueContract := ContractDescriptor.tested
      (.valueRefinement .scaledDotProductAttention "Spec.scaledDotProductAttention")
      "Composed reference attention uses hard-mask zero-numerator semantics."
      "NN.Tests.Runtime.Floats.Suite"
    vjpContract := ContractDescriptor.tested
      (.vjpRefinement .scaledDotProductAttention "Spec.scaledDotProductAttentionBackward"
        .torchLeanTape)
      "TorchLean tape supplies the composed VJP."
      "NN.Tests.Runtime.Floats.Suite"
    numericalPolicy :=
      { rounding := .scalarContext
        subnormals := .implementationDefined
        contraction := .notApplicable
        reduction := .fixedLeft }
    notes := "This is the CPU/reference attention contract." }

/-- Cross-platform reference capsules. -/
def capsules : List KernelCapsule :=
  [ matmul
  , linear
  , mseLoss
  , relu
  , gelu
  , add
  , sub
  , mul
  , scale
  , abs
  , sqrt
  , clamp
  , max
  , min
  , sigmoid
  , tanh
  , softplus
  , exp
  , log
  , sin
  , cos
  , inv
  , safeLog
  , logSoftmax
  , softmax
  , reduceSum
  , reduceMean
  , reshape
  , permute
  , broadcast
  , concat
  , slice
  , gather
  , scatterAdd
  , randUniform
  , bernoulliMask
  , layerNorm
  , batchNorm
  , conv
  , convTranspose
  , maxPool
  , smoothMaxPool
  , avgPool
  , attention
  ]

end Reference
end Backend
end NN

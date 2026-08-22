import VersoManual

open Verso.Genre Manual

#doc (Manual) "Building A Model" =>
%%%
tag := "building-models"
%%%

A TorchLean model begins as a map between tensor shapes. Initialization, loss functions, optimizer
state, and execution devices are attached later. This separation is useful even before proving a
theorem: the architecture can be inspected without allocating parameters, and the same model can
be initialized twice with different seeds or interpreted by different runtimes.

We will build three complete architectures:

1. a $`2\to8\to1` regression MLP;
2. a small convolutional classifier;
3. a transformer encoder block.

They use the same layer-composition mechanism. The differences are in their shapes and operations,
not in a new model universe for each application. We then use masking and a low-rank adapter to show
how the same tensor contracts extend those architectures.

# A Layer Is A Checked Shape Map

The layer type is:

```
nn.Layer inputShape outputShape
```

A sequential model has:

```
nn.Sequential inputShape outputShape
```

The input and output shapes are indices of the type, so composition requires equality at the
boundary. If:

$$`f:s_0\to s_1`

and

$$`g:s_1\to s_2`,

then $`g` may follow $`f`. A layer expecting $`s_3` cannot be inserted there merely because the two
shapes contain the same number of values.

Model builders use:

```
nn.Builder A
```

which is a seeded construction of `A`. It allocates deterministic seeds for parameterized layers
but does not run a forward pass or optimizer update.

# What A Layer Definition Contains

When I add a layer, I check more than its input and output shape. `nn.Layer σ τ` stores the
pieces that have to remain aligned as the layer moves from model construction to execution:

:::table +header
*
  * Field
  * Meaning
*
  * `kind`
  * the name printed in model summaries
*
  * `paramShapes`
  * the ordered shapes of parameters and persistent buffers
*
  * `initParams`
  * reproducible Float-authored initial values for that shape list
*
  * `runtimeInit`
  * an optional storage-first initializer for executable Float runtimes
*
  * `paramRequiresGrad`
  * one gradient flag for each entry of `paramShapes`
*
  * `updateBuffers`
  * an optional train/eval-dependent update for state such as running statistics
*
  * `forward`
  * the execution-polymorphic tensor program from `σ` to `τ`
:::

The forward program receives the parameters followed by the layer input. Its input shape list is
`paramShapes ++ [σ]`, and its result has shape `τ`. A new layer cannot provide an initializer in one
order and quietly read parameters in another; both uses share the same dependent shape list.

Sequential composition is correspondingly small. The internal type is `Seq`, exported to
application code as `nn.Sequential`:

```
inductive Seq : Shape → Shape → Type
  | id (s : Shape) : Seq s s
  | cons : nn.Layer σ τ → Seq τ υ → Seq σ υ
```

The middle shape `τ` appears on both sides of `cons`. This is the check performed by
`nn.Sequential!`; the macro saves syntax but does not weaken the type.

# A Model Definition And Its Live State

`nn.Sequential` is an immutable model definition. It contains the checked layer composition,
parameter layout, initializers, and forward program; it does not hide mutable weights inside the
value used by proofs and graph lowering.

Manual runtime code can instantiate that definition as a live module:

```
nn.withModel model fun checked => do
  let module ← nn.Module.instantiate checked { device := .cpu }
  let trainingOutput ← module.run input
  module.eval
  let evaluationOutput ← module.run input
```

The module owns parameters, persistent buffers, and its train/eval mode. It begins in training
mode. `run` executes without a backward tape and respects the current mode, whereas `predict`
temporarily selects eval mode. The ordinary instantiator uses native `Float32`; `instantiateAs`
exists for code that deliberately chooses a different scalar interpretation. The immutable
`checked` value is still the object passed to `nn.lowerToTypedGraph` or a proof-facing
interpretation.

This distinction is intentional. Mutating an `nn.Sequential` or `nn.Layer` would make the meaning
of a lowered graph depend on hidden state. Mutation belongs to the instantiated `nn.Module`; the
definition passed to lowering and proofs remains an ordinary Lean value.

Token models use the same split without turning indices into floating-point tensors:

```
let table := nn.build 2026 <| nn.embedding vocab embedDim
let module ← nn.IndexedModule.instantiate (table.model tokenShape) { device := .cpu }
let vectors ← module.predict tokenIds
```

The input is `Tensor Nat tokenShape`, and the output shape is
`tokenShape.appendDim embedDim`. Repeated IDs select the same row and therefore accumulate into the
same weight gradient. An index outside the vocabulary is rejected before the backend gather runs.
Fresh tables use independent standard-normal entries, as `torch.nn.Embedding` does. Architectures
may replace that default, and GPT-2 uses its smaller initialization scale.

To start from an existing table, use the tensor itself:

```
let table := nn.Embedding.ofWeight weight
let frozenTable := nn.Embedding.ofWeight weight (freeze := true)
```

The type of `weight` fixes the vocabulary and embedding dimensions. The first table participates in
reverse mode; the second remains part of module state but receives no parameter gradient. Current
embedding modules use dense row lookup and scatter-add gradients. Options that change PyTorch's
forward or backward semantics, such as `padding_idx`, `max_norm`, frequency-scaled gradients, and
sparse gradients, require their own checked definitions and are not accepted as ignored flags.

# The Running MLP

Here is the model used throughout the introduction:

```
import NN.API
open TorchLean

def inDim : Nat := 2
def hidden : Nat := 8
def outDim : Nat := 1

def model :
    nn.Builder (nn.Sequential (shape![inDim]) (shape![outDim])) :=
  nn.Sequential![
    nn.linear inDim hidden,
    nn.relu,
    nn.linear hidden outDim
  ]
```

Read the macro from top to bottom:

```
input [2]
  -> linear 2 8
hidden [8]
  -> relu
hidden [8]
  -> linear 8 1
output [1]
```

`ReLU` is shape-preserving. The two linear layers each change the final feature axis. The macro
checks that the chain is composable and returns one `Sequential [2] [1]`.

The parameter layout follows PyTorch's linear-layer convention:

$$`
W:\operatorname{Tensor}\;\alpha\;[\mathrm{out},\mathrm{in}],
\qquad
b:\operatorname{Tensor}\;\alpha\;[\mathrm{out}].
`

For this MLP the parameter shapes, in order, are:

```
[8, 2]   first weight
[8]      first bias
[1, 8]   second weight
[1]      second bias
```

The total parameter count is:

$$`8\cdot2+8+1\cdot8+1=33`.

Parameter order is part of the typed model interface. A pack with the two bias tensors exchanged
does not match the expected dependent list.

The dependent pack records a list of shapes and follows the homogeneous-scalar rule developed in
*Tensors And Shapes*. Executing the model at another `α` reuses the architecture without changing
the ordered parameter layout.

# Make A Shape Error On Purpose

Change the final layer to:

```
nn.linear 7 1
```

The preceding ReLU produces shape `[8]`, while this layer accepts `[7]`. Lean rejects the model at
definition time. No data or parameters need to be loaded to expose the mismatch.

A subtler experiment is:

```
nn.Sequential![
  nn.linear 2 8,
  nn.linear 8 8,
  nn.relu,
  nn.linear 8 1
]
```

This version compiles because the shapes compose. It is a different architecture with another
weight matrix and bias. Shape safety prevents malformed composition; it does not declare two
well-shaped networks equivalent.

# Initialization Is Reproducible State

The model declaration describes how parameters should be created. The trainer supplies the seed:

```
def trainer (seed : Nat) :=
  Trainer.new model
    { task := .regression
      optimizer := optim.adam { lr := 0.03 }
      seed := seed }
```

Constructing this value records the training configuration:

- the model builder;
- the task and loss convention;
- the optimizer configuration;
- the initialization seed;
- runtime options such as scalar type, backend, and device.

Using the same seed and configuration should produce the same initial TorchLean parameter values.
Changing only the seed is therefore a controlled experiment rather than an accidental mutation of
a global generator.

# Inspect The Architecture Before Training

The running-example chapter trained this MLP. Here we are interested in the object that existed
*before* the trainer was created:

```
def initialized := nn.build 2026 model

#eval IO.println (nn.info initialized)
```

The summary is:

```
Sequential: [2] -> [1], layers=3, params=33
  [0] Linear(2, 8): [2] -> [8] params=24 [[8, 2], [8]]
  [1] ReLU: [8] -> [8] params=0 []
  [2] Linear(8, 1): [8] -> [1] params=9 [[1, 8], [1]]
```

This is a useful design loop. Change `hidden` to `2`, `16`, and `64`; predict the parameter count
before asking Lean. Then insert another hidden `linear` and `relu` pair. The external input and
output stay `[2] → [1]`, while the internal shape chain and parameter payload grow.

# Prefix Shapes Give Batches For Free

Linear layers act on the final dimension and preserve the prefix. The model can therefore be lifted
to a fixed batch:

```
def batchedModel {batch : Nat} :
    nn.Builder
      (nn.Sequential
        (shape![batch, 2])
        (shape![batch, 1])) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]
```

Nothing in `nn.linear` calls the prefix “batch.” It could equally be `[time]`, `[batch,time]`, or a
higher-rank collection. The operation's contract is:

$$`
[\ldots,\mathrm{inFeatures}]
\longrightarrow
[\ldots,\mathrm{outFeatures}].
`

This is the same broad behavior users expect from PyTorch, with the full map recorded in the Lean
type. `Data.batchDataset` later collates per-sample tensors into a model whose prefix begins with
the chosen batch size.

# A Rank-Generic Convolutional Model

TorchLean does not need separate tensor types for signals, images, and volumes. A convolution is
parameterized by a vector of spatial sizes. The length of that vector determines the spatial rank.

For a two-dimensional input, the conventional shape is:

$$`
[\mathrm{batch},\mathrm{channels},\mathrm{height},\mathrm{width}].
`

For one-dimensional signals it is:

$$`
[\mathrm{batch},\mathrm{channels},\mathrm{length}].
`

The `nn.conv` constructor handles both. Pooling is rank-generic for the same reason. A helper
such as `nn.models.cnn` composes convolution, activation, pooling, flattening, and classification,
but the component layers remain ordinary checked maps.

Run the small convolutional example:

```
lake exe torchlean quickstart_cnn \
  --device cpu --batch 2 --steps 3 --seed 2026
```

The current run gives:

```
dataset size = 3
mean_loss(before) = 0.689656
mean_loss(after) = 0.683643
steps=3 loss0=0.689656 loss1=0.683643
vertical-1 expected=0 =
  [[0.160828, -0.082255], [0.160828, -0.082255]]
```

The leading dimension in the final output is two because we requested batch size two. The two rows
are not created by a special image container; they are the preserved tensor prefix.

The relevant source files are:

- [`SimpleCnnTrain.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleCnnTrain.lean);
- [`Cnn.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/Cnn.lean);
- [`ResNet.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/ResNet.lean).

Residual blocks require their main and skip paths to return the same shape before addition. A
projection shortcut is therefore an explicit layer, not a runtime broadcasting guess.

# A Transformer Encoder

A transformer encoder normally consumes:

$$`
[\mathrm{batch},\mathrm{sequenceLength},d_{\mathrm{model}}].
`

The block constructor is:

```
nn.transformerEncoderBlock
  (.dim batch .scalar)
  (n := sequenceLength)
  (dModel := dModel)
  { numHeads := 2
    headDim := dModel / 2
    ffnHidden := 4 * dModel }
```

Its architecture contains:

1. multi-head self-attention;
2. a residual connection and layer normalization;
3. a position-wise feed-forward network;
4. another residual connection and normalization.

The head configuration must agree with `dModel`; dimensions that must be nonzero appear as Lean
obligations. Boolean masks are explicit and use hard-mask semantics: blocked positions have zero
softmax numerator. TorchLean does not silently replace that mathematical operation with a finite
additive constant such as `-1000`.

Run one optimizer step:

```
lake exe torchlean transformer \
  --device cpu --steps 1 --log false
```

The current example reports:

```
[TorchLean] scalar: Float32 (native binary32 runtime)
[TorchLean] execution: Runtime.Autograd.Torch.ExecutionMode.eager
[TorchLean] device: cpu
dataset size = 1
mean_loss(before) = 2.499999
mean_loss(after) = 2.498999
torchlean transformer: ok
```

The log identifies the scalar type, execution mode, and device chosen for this run. The next runtime
chapters explain how those choices are made.

# Model Families Are Constructors, Not New Frameworks

KANs, GPT-style language models, vision transformers, recurrent models, neural operators,
autoencoders, diffusion models, and reinforcement-learning policies all build from the same shape,
parameter, and runtime interfaces.

For example, `nn.models.KanConfig` records input/output dimensions, hidden widths, and an edge basis
family. The basis is explicit because a KAN edge performs a learned scalar function rather than an
ordinary affine weight. It still returns a seeded model builder that can be trained through the
same trainer boundary.

A Fourier neural operator uses spectral transforms and mode truncation, but its input and output
remain general tensors. The Burgers example later in the guide shows how a PDE trajectory dataset,
FNO model, exported prediction, and Lean-checkable residual artifact fit together.

The practical rule is: a model helper should remove repetitive construction, not invent a private
tensor type or execution engine.

# A Masked Model Still Needs An Honest Target

Self-supervised learning changes the training problem more than it changes the tensor foundation.
Imagine eight consecutive sensor readings. We want the model to see alternating two-value blocks
and reconstruct the readings that were hidden:

```
import NN.API
open TorchLean

def signal : Tensor Float (shape![8]) :=
  tensor! [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

def maskedSignal : Tensor Float (shape![8]) :=
  ssl.BlockMask.apply #v[8] #v[some 2] 2 0 signal

#eval Tensor.pretty maskedSignal
```

The result is:

```
"[0.000000, 0.000000, 3.000000, 4.000000, 0.000000, 0.000000, 7.000000, 8.000000]"
```

`some 2` divides the participating axis into blocks of width two. The period `2` and offset `0`
hide block indices congruent to zero modulo two, hence the first and third blocks. An axis marked
`none` is left out of the block index; an image policy such as
`#v[none, some 4, some 4]` therefore repeats the same 4-by-4 spatial mask across channels.

For training, `ssl.BlockMAE.sample` pairs a masked batch with a row-major prefix of the original,
unmasked batch. The reconstruction width appears in the target shape, and Lean requires a proof
that the requested prefix fits. This makes the masking convention and reconstruction target part
of the sample rather than an agreement hidden between a data loader and a decoder.

The mask is deterministic. In the typed `BlockMask.apply` call, the policy and tensor ranks already
agree by construction. A zero period or zero block width hides nothing; the lower-level coordinate
helpers likewise treat a rank mismatch or out-of-bounds coordinate as visible rather than sampling
a fallback mask. Randomized mask selection should choose the period/offset or another explicit mask
from recorded generator state. The coordinate theorems
`TorchLean.ssl.BlockMask.hidden_scalar_eq_zero` and
`TorchLean.ssl.BlockMask.visible_scalar_eq_input` then describe exactly what the executable
transformation did.

# Add A Low-Rank Linear Adapter Explicitly

Masks change which data reaches the model. A low-rank adapter instead changes a parameterized
projection without replacing its base weight.

`NN.API` also exports the LoRA tensor helpers as `TorchLean.Adapters.LoRA`. An input whose final
axis has size `inDim` multiplies a base weight of shape `[inDim,outDim]` on the right. Every leading
axis is preserved, so the same definition handles a vector, a batch, or several leading axes. An
adapter stores

$$`
A:\operatorname{Tensor}\;\alpha\;[\mathrm{inDim},\mathrm{rank}],
\qquad
B:\operatorname{Tensor}\;\alpha\;[\mathrm{rank},\mathrm{outDim}],
`

and contributes $`\mathrm{scale}\cdot(AB)` to the base weight. For example, this function applies a
rank-two adapter to a batch of four eight-dimensional inputs:

```
import NN.API
open TorchLean

def adaptedProjection
    (x : Spec.Tensor Float (.dim 4 (.dim 8 .scalar)))
    (base : Spec.Tensor Float (.dim 8 (.dim 16 .scalar)))
    (A : Spec.Tensor Float (.dim 8 (.dim 2 .scalar)))
    (B : Spec.Tensor Float (.dim 2 (.dim 16 .scalar))) :
    Spec.Tensor Float (.dim 4 (.dim 16 .scalar)) :=
  let adapter : Adapters.LoRA.Params Float 8 2 16 := { A := A, B := B }
  Adapters.LoRA.linear x base adapter (0.5 : Float)
```

Use `Adapters.LoRA.delta adapter scale` to inspect only the scaled low-rank update, or
`Adapters.LoRA.effectiveWeight base adapter scale` to construct the combined weight. Keeping
`scale` explicit supports the usual `alpha / rank` choice as well as scheduled or experimental
scales.

These are pure tensor definitions, not yet a trainer-integrated LoRA workflow. They do not insert
an adapter into an `nn.Sequential`, initialize its factors, freeze the base weight, or tell an
optimizer to update only `A` and `B`. A training experiment must currently wire those choices into
its parameter pack and forward/loss program explicitly.

# The Task Belongs To The Trainer

Architecture determines the output tensor, not what that tensor means. A `[classes]` output can be
used as logits for cross entropy, scores for a margin loss, or values passed to a custom objective.

TorchLean therefore writes:

```
Trainer.new model { task := .regression }
Trainer.new model { task := .oneHotCrossEntropy 0 }
Trainer.new model { task := .custom lossProgram }
```

Regression uses mean-squared error by default. Classification variants specify their target
convention and the zero-based class dimension. For example, `[batch, classes]` uses axis `1`, while
`[batch, time, vocabulary]` uses axis `2`. A custom task supplies a checked scalar loss program.
This makes the loss visible in the training configuration rather than baking it into the model
architecture.

# What We Carry Forward

Every model family above produces the same kind of object: a checked map between shapes with an
ordered parameter layout and a forward program. The next chapters add data, a loss, and mutable
training state to that object. Graph lowering later reads the same shapes and payload order.

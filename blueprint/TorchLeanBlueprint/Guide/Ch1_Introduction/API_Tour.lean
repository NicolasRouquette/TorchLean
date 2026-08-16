import VersoManual

open Verso.Genre Manual

#doc (Manual) "A First Walk Through The API" =>
%%%
tag := "api_tour"
%%%

The shortest useful TorchLean import is:

```
import NN.API

open TorchLean
```

It gives application code several main places to begin:

- `Tensor` for shape-indexed tensor values and constructors;
- `nn` for layers, blocks, model families, and seeded initialization;
- `Data` for in-memory datasets, loaders, checkpoints, and text helpers;
- `optim` for optimizer configurations;
- `Trainer` for prediction, training, summaries, and the verification bridge;
- `autograd` for explicit function and model derivatives;
- `Verification` for lowering a checked model to IBP/CROWN inputs;
- `classical` for statistical and non-neural model families.

This tour starts with tensors, models, data, and training, then briefly points to autograd and
verification. Later chapters develop the remaining namespaces. Every command below runs from the
repository root.

# First Contact: Print A Few Tensors

Run:

```
lake exe torchlean quickstart_tensors
```

The checked-in example prints:

```
== Quickstart: tensor basics ==
[Float] [0.100000, 0.200000, 0.300000, 0.400000]
[ℚ] [1/10, 1/5, 3/10, 2/5]
[Int] [1, 2, 3, 4]
[Float32] [0.100000, 0.200000, 0.300000, 0.400000]
[IEEE32Exec] [0.100000, 0.200000, 0.300000, 0.400000]
[Float] [[[1.000000, 2.000000], [3.000000, 4.000000]],
         [[5.000000, 6.000000], [7.000000, 8.000000]]]
Expected failure printing Tensor ℝ: Refusing to print `Tensor ℝ` ...
```

The same tensor structure can carry several scalar types. `Float32` is Lean's native binary32 type;
`IEEE32Exec` is TorchLean's independent bit-level binary32 reference. `Float` is Lean's binary64
host type, and `ℚ` supplies exact rational arithmetic. `ℝ` is useful in specifications and proofs,
but arbitrary real values do not have a general executable printer.

Open
[`NN/Examples/Quickstart/TensorBasics.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/TensorBasics.lean)
and find the definitions of `xF`, `xQ`, `x32`, and `x32Ref`. The values look alike when printed,
but their types select different arithmetic.

One tensor still has one element type, and one model or IR evaluation chooses one numeric scalar
`α`. The tensor chapter develops this rule and shows how a runtime records the chosen arithmetic.

# Build A Small File Of Your Own

Create `Tour.lean` at the repository root:

```
import NN.API

open TorchLean

def point : Tensor Float (shape![2]) :=
  tensorOfList! [2] [0.25, -0.75]

def model : nn.Builder (nn.Sequential (.dim 2 .scalar) (.dim 1 .scalar)) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]

def initialized :=
  nn.build 2026 model

#eval Tensor.pretty point
#eval IO.println (nn.info initialized)
```

Run it:

```
lake env lean Tour.lean
```

The expected output is:

```
"[0.250000, -0.750000]"
Sequential: [2] -> [1], layers=3, params=33
  [0] Linear(2, 8): [2] -> [8] params=24 [[8, 2], [8]]
  [1] ReLU: [8] -> [8] params=0 []
  [2] Linear(8, 1): [8] -> [1] params=9 [[1, 8], [1]]
```

The summary accounts for every scalar parameter:

$$`8\cdot2+8+1\cdot8+1=33`.

It also exposes the ordered payload shapes. A later lowering pass or checkpoint adapter must provide
weights and biases in this order and with these shapes.

As a quick experiment, change the hidden width from `8` to `12` in both linear layers. Before
running Lean, predict the parameter count:

$$`12\cdot2+12+1\cdot12+1=49`.

The model summary should confirm `params=49`.

# Tensor Constructors

TorchLean offers two useful styles for fixed data. `tensorOfList!` takes dimensions and a flat
row-major list:

```
def matrix : Tensor Float (shape![2, 3]) :=
  tensorOfList! [2, 3] [
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0
  ]
```

The nested `tensor!` syntax mirrors the visible dimensions:

```
def sameMatrix : Tensor Float (shape![2, 3]) :=
  tensor! [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
  ]
```

Both constructors check their dimensions. `Tensor.vector` is convenient when the length should be
inferred from a list:

```
def inferred := Tensor.vector (α := Float) [1.0, 2.0, 3.0]

#check inferred
-- inferred : Tensor Float (shape![3])
```

Use an explicit type when the shape belongs to an interface. Use inference for local data whose
length is already clear from the value.

# Turn Rows Into A Dataset

A supervised dataset pairs one input tensor with one target tensor. Add to `Tour.lean`:

```
def xs : Tensor Float (shape![4, 2]) :=
  tensorOfList! [4, 2] [
    0.0, 0.0,
    0.0, 1.0,
    1.0, 0.0,
    1.0, 1.0
  ]

def ys : Tensor Float (shape![4, 1]) :=
  tensorOfList! [4, 1] [0.2, 1.0, 1.0, 1.8]

def dataset : Trainer.DataSource (.dim 2 .scalar) (.dim 1 .scalar) :=
  Data.tensorDataset xs ys

#check dataset
```

The outer dimension `4` counts examples. `Data.tensorDataset` removes that common batch axis from
the sample types, so each input has shape `[2]` and each target has shape `[1]`.

Try changing the declared type of `ys` to `shape![4]` while leaving `dataset` unchanged. Lean
rejects the dataset because its target samples would be scalars rather than length-one vectors. This
is the same distinction the model output type makes.

For large data, application code does not need to place every value in a source literal. `Data`
also exposes CSV, NPY, image, token-stream, batching, shuffling, and checkpoint helpers. The shape
conversion still occurs at a named boundary: runtime dimensions are checked before values enter a
statically shaped dataset.

# Configure A Trainer

Add:

```
def trainer : Trainer (.dim 2 .scalar) (.dim 1 .scalar) :=
  Trainer.new model
    { task := .regression
      optimizer := optim.adam { lr := 0.03 }
      seed := 2026 }

#check trainer
```

`Trainer.new` materializes the seeded builder and returns a `Trainer` value containing:

- the checked model;
- the loss task;
- optimizer and runtime settings;
- the initialization seed.

The default execution profile is checked CPU, using the eager runtime and native `Float32`. A
trainer configuration may instead select typed graph execution or the bit-level `IEEE32Exec`
reference. The generic scalar dispatcher also has an executable complex-binary32 scalar, but the
high-level trainer does not yet expose complex prediction readback. Device and provider selection
are runtime concerns; they do not change the model's input and output shapes.

Training options belong to the call:

```
def options : Trainer.TrainOptions :=
  { steps := 200
    batchSize := 4
    logEvery := 25 }
```

Here `steps` counts optimizer updates. Each update differentiates four rows at the current
parameter values, averages their gradients, and applies the optimizer once.

In an `IO` definition, the trainer lifecycle is:

```
def trainTour : IO Unit := do
  let before ← trainer.predict point
  IO.println s!"before = {Tensor.pretty before}"

  let trained ← trainer.train dataset options
  trained.printSummary
  trained.printPrediction "after" point
```

`trainer.train` returns a `TrainResult`. It does not mutate the immutable `trainer` definition in
the source file. The result retains the runtime state containing the updated parameters.

# Hand The Model To The Trainer

At this point `Tour.lean` has a model, a dataset, and a trainer. The checked-in version of the same
workflow is ready to run:

```
lake exe torchlean quickstart_mlp \
  --device cpu \
  --steps 200 \
  --seed 2026
```

Its summary ends with:

```
dataset size = 25
mean_loss(before) = 0.761530
mean_loss(after) = 0.003234
heldout x=(0.25,-0.75), target=0.2, prediction(after)=[0.210239]
```

Later, the training chapter will open this run up and follow one step through the model, loss,
tape, optimizer, and parameter update. The useful distinction here is simpler: `Trainer.new`
creates the initial `Trainer`, and `train` returns a `TrainResult` containing the trained runtime
state.
The next two chapters explain why TorchLean keeps those dependencies explicit and teach just enough
Lean to read their types without guesswork.

Command-specific help shows the available runtime options:

```
lake exe torchlean quickstart_mlp --help
```

At the top level,

```
lake exe torchlean --help
```

lists runnable model families and the common `--device`, `--scalar`, `--execution`, `--seed`, and
`--show-backend` flags.

# Take A Quick Look At Autograd

Training uses reverse-mode automatic differentiation. There is a small command for inspecting it
without the rest of the trainer:

```
lake exe torchlean quickstart_autograd
```

It prints gradients, VJPs, Jacobian rows and columns, and a Hessian-vector product. One particularly
useful pair compares a loss with the same loss after `detach`: the forward values agree, while the
detached gradient is zero. We will derive that behavior in the autograd walkthrough instead of
reproducing the full log here.

# Move From A Run To A Region

Executable verification tools use a separate runner:

```
lake exe verify -- list
```

For the small MLP workflow:

```
lake exe verify -- torchlean-ibp
```

This lowers a model to the graph IR, places an input box at its input node, and propagates intervals
to the output. At this level the useful objects are `NN.IR.Graph`, the
parameter payload, the input region, and the IBP state. The graph and verification chapters develop
those objects carefully.

The shorter application names are also available for a manual run:

```
def lowered :=
  Verification.lowerForwardToIR initialized (nn.initState initialized)
```

The result is an `Except`: unsupported operations or malformed lowering stop explicitly before a
bound pass is run.

Application code that needs these lower layers should use focused imports:

```
import NN.IR
import NN.Verification
```

`import NN` loads the complete library for files that combine models, proofs, floating-point
semantics, backend contracts, and verification. Ordinary training code can use `NN.API`, which
keeps its dependencies and generated documentation smaller.

# Looking Up Exact Names

The generated API page is the declaration index. In a Lean file, `#check` shows the elaborated type
of a name, while editor hover reveals its documentation and source. The guide stays focused on the
design and the worked programs; the API page answers exact-name questions without repeating the
module tree here.

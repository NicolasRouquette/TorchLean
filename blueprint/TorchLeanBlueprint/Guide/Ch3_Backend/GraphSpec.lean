import VersoManual

open Verso.Genre Manual

#doc (Manual) "GraphSpec: One Architecture, Several Meanings" =>
%%%
tag := "graphspec"
%%%

The pure MLP formula from the previous chapter is excellent for a theorem:

$$`x\mapsto W_2\operatorname{ReLU}(W_1x+b_1)+b_2.`

It is less convenient for a tool that wants to enumerate layers, export parameter shapes, replace
one operation, or lower the same architecture to several targets. For those tasks TorchLean uses
GraphSpec, a small typed language in which the architecture itself is data.

GraphSpec sits between two other layers:

```
application model builders
        ↓
GraphSpec architecture and parameter ABI
        ↓
pure interpretation / TorchLean program / sequential model / DAG tools
        ↓
canonical NN.IR.Graph and runtime-specific lowering
```

GraphSpec preserves model structure and makes every input, output, and parameter shape explicit.
The tensor runtime sits above it, while the low-level backend IR sits below it.

# Write The Running MLP As A Graph

The complete architecture is:

```
import NN.GraphSpec.Models.Mlp

open NN
open NN.GraphSpec
open NN.GraphSpec.Models
open Spec

def mlpGraph (input hidden output : Nat) :
    Graph
      [ shape![hidden, input], shape![hidden],
        shape![output, hidden], shape![output] ]
      (shape![input])
      (shape![output]) :=
  Graph.linear input hidden >>>
  Graph.relu (shape![hidden]) >>>
  Graph.linear hidden output
```

Read the type from right to left:

- the graph consumes one tensor of shape `[input]`;
- it produces one tensor of shape `[output]`;
- its parameter environment contains exactly four tensors;
- the order is $`W_1`, $`b_1`, $`W_2`, $`b_2`.

For the $`2\to3\to1` model used by the GraphSpec tutorial, the parameter shapes are

```
W₁ : [3, 2]   six values
b₁ : [3]      three values
W₂ : [1, 3]   three values
b₂ : [1]      one value
```

There are thirteen trainable scalars. That count is not recovered from strings such as
`"layer1.weight"`. It follows from the graph's type.

# Composition Computes The ABI

The composition operator `>>>` does more than connect two arrows. If

```
g₁ : Graph ps₁ σ τ
g₂ : Graph ps₂ τ υ
```

then

```
g₁ >>> g₂ : Graph (ps₁ ++ ps₂) σ υ.
```

The intermediate shape must be the same $`\tau`, and the parameter lists are concatenated in
construction order. Replacing the first linear layer by `Graph.linear input 5` changes its output
shape to `[5]`; the existing ReLU can still consume it, but the second linear layer must now accept
five inputs. Lean reports the mismatch at the architecture definition.

This is the first hands-on experiment:

1. open
   [NN/GraphSpec/Models/Mlp.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/GraphSpec/Models/Mlp.lean);
2. change the hidden dimension at only one of the two linear nodes;
3. ask Lean to elaborate the file.

The failure occurs before initialization, data loading, or runtime execution because the broken
edge has no well-typed composition.

# A Primitive Has Two Interpretations

A sequential GraphSpec primitive stores:

```
specFwd      : pure shape-indexed tensor function
torchProgram : executable TorchLean frontend program
```

For a linear primitive, `specFwd` calls the mathematical `linearSpec`; `torchProgram` constructs the
runtime operation with the same input and parameter shapes. Composition builds both interpretations
in parallel.

The record does not prove that the two fields agree merely by storing them together. The important
difference is:

- *construction* keeps the intended programs adjacent;
- *a theorem* establishes that their meanings coincide.

TorchLean proves such relationships at the primitive or model level where they are available.
This avoids a global axiom saying that every future GraphSpec operation is correct by construction.

# Current Primitive Adapters

The minimal sequential vocabulary in `NN.GraphSpec.Core` is `linear`, `relu`, and `softmax`.
`NN.GraphSpec.Primitives.Vision` adds CHW `conv2d`, `maxPool2d`, `flatten`, and channel-first
BatchNorm. Each of these adapters supplies the same three pieces of information:

- the exact parameter-shape list;
- a pure `specFwd` meaning;
- an executable `torchProgram` meaning.

The MLP and two-convolution CNN compose the sequential adapters. `residualLinear` demonstrates the
DAG language and an explicit skip connection. These are implemented examples, not a claim that
every layer under `NN.Spec` already has a GraphSpec adapter. `GraphSpec.ToTorchLean.toSeq` is also
deliberately partial: it succeeds only when every primitive provides a corresponding layer
constructor.

# Run The Complete Lowering

The repository contains an executable GraphSpec tutorial:

```
lake exe torchlean graphspec --device cpu --backend eager
```

The current checkout prints:

```
== GraphSpec tutorial ==
GraphSpec architecture ladder:
  1. MLP: sequential layer stack; lowers to nn.Sequential and trains below.
  2. CNN: sequential vision graph with checked conv/pool shape arithmetic.
  3. residualLinear: minimal DAG-native skip connection.

model:
Sequential: [2] -> [1], layers=3, params=13
  [0] Linear(2, 3): [2] -> [3] params=9 [[3, 2], [3]]
  [1] ReLU: [3] -> [3] params=0 []
  [2] Linear(3, 1): [3] -> [1] params=4 [[1, 3], [1]]
dataset size = 1
mean_loss(before) = 1.239197
mean_loss(after) = 0.247518
steps=3 loss0=1.239197 loss1=0.247518
```

The execution path is:

```
Models.mlp
   │ Graph ps [2] [1]
   ▼
GraphSpec.ToTorchLean.toSeq
   │ Except String (nn.Sequential [2] [1])
   ▼
Trainer.new
   ▼
eager runtime and autograd tape
```

`toSeq` is intentionally partial. A sequential linear/ReLU stack has an `nn` layer counterpart, so
the conversion succeeds. An arbitrary custom primitive may have a pure and program interpretation
without having an `nn.LayerDef` constructor; in that case the conversion returns an error rather
than inventing a layer.

Try the compiled frontend as a second run:

```
lake exe torchlean graphspec --device cpu --backend compiled
```

The architecture and parameter ABI are unchanged. Only the execution path selected after lowering
changes.

# Pure Interpretation

GraphSpec can be evaluated without the trainer:

```
Interp.spec mlpGraph params x
```

Here `params` is a heterogeneous tensor list whose shape index is exactly

```
[[hidden, input], [hidden], [output, hidden], [output]].
```

Pattern matching on that list reveals $`W_1`, $`b_1`, $`W_2`, and $`b_2` in ABI order. The
interpreter then computes

$$`\operatorname{linearSpec}
  (W_2,b_2)
  \left(\operatorname{ReLU}
    \left(\operatorname{linearSpec}(W_1,b_1,x)\right)\right).`

The theorem
`NN.GraphSpec.Models.mlp_interp_eq_spec_mlp_forward` proves that this interpretation equals
TorchLean's hand-written MLP specification for every scalar $`\alpha` satisfying the required `Context`,
every well-shaped parameter list, and every input.

That is stronger than checking a few Float examples: it is a universally quantified equality of
the two pure definitions. It is also narrower than runtime correctness: neither CUDA buffers nor
an eager tape occur in the theorem.

# Why A Second, DAG-Shaped Syntax Exists

A chain can be written with `>>>`. A residual block cannot:

$$`r(x)=\operatorname{ReLU}(Wx+b+x).`

The input $`x` is used twice. Hiding this in a special `ResidualLinear` primitive would make one
example work, but every new sharing pattern would demand another special primitive. GraphSpec's DAG
language instead represents sharing directly.

A term has type

```
DAG.Term Γ τ
```

meaning that, given a typed environment $`\Gamma`, it computes a tensor of shape $`\tau`. Its essential
constructors are:

- `var`, which reads an existing value;
- `op`, which applies an arbitrary-arity primitive;
- `let1`, which computes an intermediate once and extends the environment.

The residual computation is conceptually:

```
let y = linear(W, b, x)
let z = add(y, x)
relu(z)
```

The same $`x` variable appears in the linear and add arguments, while $`y` is bound once. Because the
language has no recursion and only extends the environment with prior values, its terms denote
acyclic graphs by construction.

Run the tutorial command again and notice the third architecture in its printed ladder:
`residualLinear` is checked as a DAG model even though the short tutorial trains only the sequential
MLP.

# Sequential Graph Versus DAG Model

The two GraphSpec syntaxes are related but not identical:

:::table +header
*
  * Form
  * Best use
  * Sharing
  * Parameter representation
*
  * `Graph ps σ τ`
  * readable layer chains
  * no explicit fan-out
  * type-indexed list `ps`
*
  * `DAG.Model ps ins τ`
  * residual and multi-input models
  * explicit variables and `let1`
  * typed model environment
:::

A sequential graph can be lowered structurally into a DAG model. This does not require a numerical
theorem because the conversion also comes with a theorem relating the relevant pure interpretation.
The reverse direction is not generally possible: a DAG with fan-out has no faithful representation
as a plain chain without adding duplication or a special combinator.

# Initialization Is Separate From Meaning

GraphSpec knows the parameter shapes and order, but an architecture does not mathematically require
one initializer. The repository supports zero initialization for simple structural examples and
deterministic seeded initialization for executable models.

For the MLP, the theorem
`mlp_detInitParams_eq_torchlean_linear_inits` checks that GraphSpec's deterministic traversal
produces parameters in the same ABI order as the corresponding TorchLean linear-layer
initializers. It does not say that those random-looking values are optimal, and it says nothing
about an external checkpoint.

An imported checkpoint needs a different argument:

1. parse the external names and arrays;
2. check each concrete shape and finite-value condition;
3. map values into the GraphSpec ABI;
4. state or assume that this mapping matches the source framework's layout.

Architecture correctness cannot authenticate training provenance.

# GraphSpec Is Not The Backend IR

It is tempting to call every graph representation “the graph,” but GraphSpec and `NN.IR.Graph`
have different jobs.

GraphSpec is intrinsically typed. A model that connects incompatible shapes does not elaborate.
Its primitives carry pure and TorchLean-program interpretations.

`NN.IR.Graph` is a serializable op-tagged DAG. Nodes carry numeric IDs, parent IDs, output shapes,
and attributes; tensors and parameters live in an external payload. This form is better for
importers, validators, generic passes, verification, and backend planning.

There is not currently a universal compiler from every GraphSpec model to canonical IR. Selected
frontend and model paths lower to IR, and selected semantic theorems cover those paths. The next
chapter builds and inspects that lower-level representation directly.

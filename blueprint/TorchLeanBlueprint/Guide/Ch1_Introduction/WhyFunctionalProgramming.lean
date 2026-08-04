import VersoManual

open Verso.Genre Manual

#doc (Manual) "Why The Model Is Written As A Function" =>
%%%
tag := "why_functional"
%%%

A neural network is usually introduced as a function

$$`f_\theta : X\to Y`.

The notation tells us something important: the output depends on an input $`x` and parameters
$`\theta`.
Real training code has more dependencies. Batch normalization reads and updates running statistics.
Dropout depends on a mode and a random mask. An optimizer carries momentum or moment estimates. A
checkpoint loader reads bytes from a file. A GPU execution owns buffers whose lifetime matters.

TorchLean does not pretend that these effects disappear. It begins with a functional description so
that each dependency can be named before an efficient runtime decides how to store it.

# Start With An Ordinary Function

Here is the smallest possible affine model:

```
structure Affine where
  weight : Float
  bias : Float
deriving Repr

def Affine.forward (p : Affine) (x : Float) : Float :=
  p.weight * x + p.bias

#eval Affine.forward { weight := 2.0, bias := 0.5 } 3.0
```

Running the file with

```
lake env lean Affine.lean
```

prints:

```
6.500000
```

There is no hidden parameter lookup in `Affine.forward`. If we want another model, we pass another
`Affine` value. If we want another input, we pass another `Float`. The same definition can be
evaluated by the compiler or mentioned in a proposition:

```
def HasExpectedOutput : Prop :=
  Affine.forward { weight := 2.0, bias := 0.5 } 3.0 = 6.5
```

`HasExpectedOutput` is a statement, not a proof. Proving claims about executable floating-point
operations takes a little more machinery, which we will build later. Notice the useful
split between the model family `Affine.forward` and the concrete model
`{ weight := 2.0, bias := 0.5 }`.

# Parameters Are Inputs To The Program

TorchLean's layer representation follows the same idea at tensor scale. A layer records an ordered
list of parameter shapes, initial values for those tensors, gradient flags, and a forward program.
When that program runs, it receives the current parameter references and the input. Training may
replace the live parameter values many times without changing the architecture.

For a two-layer MLP:

```
import NN.API

open TorchLean

def model : nn.M (nn.Sequential (.dim 2 .scalar) (.dim 1 .scalar)) :=
  nn.Sequential![
    nn.linear 2 4,
    nn.relu,
    nn.linear 4 1
  ]

def initialized :=
  nn.run 2026 model

#eval (nn.paramShapes initialized).map Shape.toList
```

Lean prints:

```
[[4, 2], [4], [1, 4], [1]]
```

These are the first weight matrix, first bias, second weight matrix, and second bias. Their order is
part of the forward program's type. `nn.initParams initialized` returns the corresponding initial
payload; a trained runtime owns a later payload with the same shape list.

The model is polymorphic in its scalar type, while one execution uses a single `α` for the parameter
pack and activations. The tensor chapter gives the full homogeneous-scalar contract.

Try changing the hidden width from `4` to `6` in both linear layers. The printed parameter shapes
become:

```
[[6, 2], [6], [1, 6], [1]]
```

Change only the second layer to `nn.linear 6 1`. The model no longer elaborates: the preceding
ReLU produces a length-four vector, while the final layer requires length six. The functional
composition and the dependent shape type catch the disagreement at the model boundary.

# Initialization Is A Pure State Computation

The linear layers need random initial weights. Rather than reading an unnamed global generator,
layer constructors return `nn.M`, a deterministic state computation over a seed stream.

```
def firstBuild := nn.run 2026 model
def secondBuild := nn.run 2026 model
def anotherBuild := nn.run 7 model
```

`firstBuild` and `secondBuild` consume the same sequence of initialization seeds. `anotherBuild`
uses the same architecture and parameter shapes but a different initialization stream.

This distinction matters when reproducing a run. The architecture alone does not determine the
initial parameter values. TorchLean can therefore state separately:

- which seeded builder describes the architecture;
- which seed initialized it;
- which parameter payload is currently used;
- whether a theorem concerns the initial or trained payload.

The trainer accepts either the builder or an already initialized model:

```
def fromBuilder :=
  Trainer.new model { task := .regression, seed := 2026 }

def fromValue :=
  Trainer.new initialized { task := .regression, seed := 999 }
```

In the first definition, `Trainer.new` uses `2026` to run the builder. In the second, the model is
already built, so the seed does not reinitialize it. This behavior is implemented by the
`Trainer.ToModel` instances rather than by inspecting the value at runtime.

# An Optimizer Is A State Transition

Plain SGD can be written as

$$`\theta_{t+1}=\theta_t-\eta g_t`.

Momentum adds another state variable:

$$`
\begin{aligned}
v_{t+1} &= \mu v_t+g_t,\\
\theta_{t+1} &= \theta_t-\eta v_{t+1}.
\end{aligned}
`

A direct Lean version makes both outputs explicit:

```
structure StepState where
  weight : Float
  velocity : Float
deriving Repr

def momentumStep
    (learningRate momentum gradient : Float)
    (state : StepState) : StepState :=
  let velocity := momentum * state.velocity + gradient
  { weight := state.weight - learningRate * velocity
    velocity }

#eval momentumStep 0.1 0.9 0.25
  { weight := 2.0, velocity := 0.0 }
```

The result is:

```
{ weight := 1.975000, velocity := 0.250000 }
```

Nothing in this definition mutates `state`; it returns the next state. Adam and AdamW carry more
fields, but the semantic picture is the same. This makes it possible to state a theorem about one
exact update convention, including epsilon placement, bias correction, and weight decay.

The performance runtime need not allocate a fresh high-level tree for every update. It can reuse
uniquely owned Lean arrays, mutate references inside `IO`, or update native device buffers. The
functional rule says what the update means. The runtime implementation decides how to realize it.

The parameter, gradient, and optimizer-state tensors follow the scalar choice already made for the
run; optimizer state does not introduce a second hidden dtype policy.

# Mode Is An Argument, Not Background Knowledge

Some layers denote different functions during training and evaluation. Dropout samples a mask in
training and is the identity in evaluation. Batch normalization uses batch statistics and updates
running buffers during training; evaluation reads those saved statistics.

A small sketch shows the relevant interface:

```
inductive Mode where
  | train
  | eval

structure RunningMean where
  value : Float

def normalizeSketch
    (mode : Mode) (state : RunningMean) (x : Float) :
    Float × RunningMean :=
  match mode with
  | .eval =>
      (x - state.value, state)
  | .train =>
      let next := { value := 0.9 * state.value + 0.1 * x }
      (x - next.value, next)
```

The example shows the dependency shape rather than TorchLean's BatchNorm formula. The actual
`LayerDef.forward` receives a `Mode`, and `LayerDef.updateBuffers` optionally returns updated
parameter or buffer tensors. `nn.programWithMode` and `nn.updateBuffers` compose that behavior
through a sequential model.

Explicit state lets a graph export or theorem say whether it describes `.train` or `.eval`.

# Effects Still Have A Home

Reading a dataset and launching a kernel are effects, so their result types mention `IO`:

```
def readManifest (path : System.FilePath) : IO String := do
  IO.FS.readFile path

def announce (message : String) : IO Unit := do
  IO.println message
```

A pure parser can inspect the returned string. A pure checker can inspect a parsed certificate. A
theorem can describe the checker. The filesystem and the launched kernel remain effects, while the
objects we reason about after those calls can still be ordinary Lean values.

TorchLean uses the same separation for training:

- model construction and initialization are pure seed-state computations;
- the model's mathematical operations have pure interpretations;
- training is an `IO` session with mutable parameters, optimizer state, tapes, and buffers;
- checkers consume stable Lean data;
- theorems state what accepted data implies.

# Why This Can Still Be Fast

Lean 4 uses deterministic reference counting. When an immutable value has a unique owner, compiled
code can often reuse its storage. TorchLean also uses explicitly mutable runtime objects and foreign
buffers where the workload calls for them.

It would therefore be misleading to equate “functional interface” with “copy every tensor after
every operation.” The source-level interface controls dependencies. Storage ownership and mutation
belong to the execution strategy.

The same separation lets several backends implement one model. A CPU evaluator, native CUDA
kernel, or external provider receives the same explicit inputs, so changing device does not require
rewriting the model around a different collection of hidden fields.

# A Useful Reading Test

When you encounter a TorchLean definition, ask three questions:

1. Which values determine the result?
2. Is the definition pure, a deterministic state computation such as `nn.M`, or an `IO` action?
3. If state changes, where is the old state and where is the new state named?

These questions are more useful than trying to label the whole repository “functional.” Training
and native execution are stateful jobs; specifications and many model transformations are pure.
The types make the difference visible.

# Further Reading

- Ullrich and de Moura, ["Counting Immutable Beans: Reference Counting Optimized for Purely
  Functional Programming"](https://arxiv.org/abs/1908.05647), IFL 2019.
- Lean 4 language reference:
  [Functions](https://lean-lang.org/doc/reference/latest/Terms/Functions/) and
  [Do notation](https://lean-lang.org/doc/reference/latest/Terms/do--Notation/).
- TorchLean's
  [`LayerDef`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/TorchLean/NN/Core.lean)
  and
  [`Seq`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/TorchLean/NN/Seq.lean)
  definitions.

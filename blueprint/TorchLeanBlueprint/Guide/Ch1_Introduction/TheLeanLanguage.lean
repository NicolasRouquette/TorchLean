import VersoManual

open Verso.Genre Manual

#doc (Manual) "Enough Lean To Begin" =>
%%%
tag := "lean_ecosystem"
%%%

TorchLean is ordinary Lean code. Models are definitions, shapes are values that appear in types,
training entry points are `IO` programs, and proofs are terms checked by Lean's kernel. You do not
need to learn all of Lean before running a model, but a few language ideas make the rest of this
guide much easier to read.

You can run every snippet below from a file named `Primer.lean` at the TorchLean repository
root:

```
lake env lean Primer.lean
```

`lake env` selects the Lean version and dependencies declared by this checkout. When the file
contains only successful commands, Lean exits silently except for output requested by `#check` and
`#eval`.

# Imports And Namespaces

Begin the file with:

```
import NN.API

open TorchLean
```

`import NN.API` loads the focused application interface: tensors, model builders, datasets,
optimizers, and the high-level trainer. It does not import every proof or backend-internal module.

Most application-facing names live in the `TorchLean` namespace. `open TorchLean` lets us write `Tensor.T`,
`nn.linear`, and `Trainer.new` instead of prefixing each name with `TorchLean.`. Lower-case `nn` and
`optim` are namespaces, not Python objects.

Lean's editor can show the type of any known name. Add:

```
#check Tensor.T
#check nn.linear 2 4
#check Trainer.new
```

The relevant parts of the output are:

```
TorchLean.Tensor.T (α : Type) : Spec.Shape → Type
nn.linear 2 4 : nn.M (nn.Sequential ...2... ...4...)
Trainer.new ... : Trainer.Handle σ τ
```

The full output contains qualified internal names and implicit arguments. That detail is useful when
debugging, but the second line already gives the useful information: a `2 → 4` linear layer is a seeded model
builder whose input and output shapes occur in its type.

# Definitions Compute

`def` gives a name to a value or function:

```
def square (n : Nat) : Nat :=
  n * n

#eval square 12
```

The output is:

```
144
```

The first `Nat` is the input type and the second is the result type. Lean can infer many result
types, but explicit annotations are helpful at API boundaries.

Local values are introduced with `let`:

```
def centeredSquare (x center : Int) : Int :=
  let displacement := x - center
  displacement * displacement

#eval centeredSquare 7 3
```

This prints `16`. The name `displacement` does not mutate; it abbreviates the value `x - center`
inside the remaining expression.

# Structures Name The Pieces

A `structure` groups fields:

```
structure Affine where
  weight : Float
  bias : Float
deriving Repr

def Affine.forward (p : Affine) (x : Float) : Float :=
  p.weight * x + p.bias

#eval Affine.forward { weight := 1.5, bias := -0.5 } 2.0
```

Lean prints:

```
2.500000
```

TorchLean uses structures for configuration records, model summaries, backend contracts, graph
nodes, and certificate data. A value such as

```
{ task := .regression
  optimizer := optim.adam { lr := 0.03 }
  seed := 2026 }
```

is a structure literal. Expected types tell Lean which structure and constructors are intended.

# Inductive Types Describe The Alternatives

An `inductive` type lists the possible forms of a value:

```
inductive Reduction where
  | mean
  | sum

def reductionName : Reduction → String
  | .mean => "mean"
  | .sum => "sum"

#eval reductionName .mean
```

The output is `"mean"`. Pattern matching covers the two constructors explicitly.

TorchLean uses inductive types for execution modes, runtime dtypes, devices, backend providers,
graph operations, optimizer choices, and model syntax. When a new constructor is added, pattern
matches that need the new case stop compiling until they are updated. This is one way a growing
codebase keeps option sets synchronized.

# Tensors Are Indexed By Their Shapes

The type of a TorchLean tensor has two arguments:

```
Tensor.T α s
```

`α` is the scalar type and `s` is the shape. Because the result type depends on the *value* `s`,
this is a dependent type.

Add a vector to `Primer.lean`:

```
def point : Tensor.T Float (shape![2]) :=
  tensorOfList! [2] [0.25, -0.75]

#check point
#eval Tensor.pretty point
```

The output is:

```
point : Tensor.T Float (NN.Tensor.shapeOfDims [2])
"[0.250000, -0.750000]"
```

The macro `shape![2]` expands to TorchLean's recursive shape value. The constructor
`tensorOfList! [2]` checks that two scalar entries were supplied and builds a tensor with that shape.
The quotation marks in the second output appear because `Tensor.pretty` returns a `String`.

Shapes themselves are ordinary values:

```
#eval Shape.toList (shape![2, 3])
#eval Shape.size (shape![2, 3])
```

Lean prints:

```
[2, 3]
6
```

The shape indexes the tensor's type as well as its runtime array, so a function can require that two
tensors share exactly the same shape.

# Shape Errors Appear While The Model Is Built

Add a valid model:

```
def model : nn.M (nn.Sequential (.dim 2 .scalar) (.dim 1 .scalar)) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]

#check model
#check nn.run 2026 model
```

The first linear layer maps a length-two vector to length eight. ReLU preserves that shape. The
second linear layer consumes length eight and returns length one. Lean checks each composition while
elaborating `nn.Sequential!`.

For a deliberate experiment, change only the final layer:

```
-- nn.linear 7 1
```

After uncommenting that replacement, `lake env lean Primer.lean` exits with an application type
mismatch. The useful part says that a sequence beginning with shape `7` was supplied where shape `8`
was expected.

Lean does not insert a reshape or guess that one width is a typo. If the shape change is intended,
the source must contain an operation that explains it. Restore `nn.linear 8 1` before continuing.

# Implicit Arguments Carry Repeated Information

Curly braces mark an implicit argument:

```
def requireSameShape {s : Shape}
    (_left _right : Tensor.T Float s) : Unit :=
  ()
```

Callers normally omit `s`; Lean infers it from the tensors. The single implicit variable ensures
that both arguments have the same shape. This pattern appears throughout tensor operations: sizes
and shapes are inferred from typed inputs rather than repeated as unchecked integers.

Square brackets introduce type-class arguments:

```
def showTwice {α : Type} [ToString α] (x : α) : String :=
  toString x ++ ", " ++ toString x

#eval showTwice 7
```

Lean finds the `ToString Nat` implementation automatically and prints `"7, 7"`. TorchLean uses type
classes to request scalar operations, runtime conversion support, decidable shape equality, and
printing behavior without fixing every program to one scalar type.

# One Type Per Evaluation

A declaration such as

```
def preserveShape {α : Type} (x : Tensor.T α (shape![2])) := x
```

can be instantiated once with `Float`, once with `Rat`, and once with
`TorchLean.Floats.IEEE32Exec`. Within any one instantiation, both the input and output use the same
`α`. *Tensors And Shapes* develops the consequence for a whole run, while *TorchLean And PyTorch*
discusses mixed precision directly.

# Programs And Propositions Are Different Values

A definition that returns data can compute:

```
def outputWidth : Nat :=
  Shape.size (shape![1])
```

A definition that returns `Prop` states a claim:

```
def HasOneOutput : Prop :=
  outputWidth = 1
```

A theorem supplies a proof:

```
theorem hasOneOutput : HasOneOutput := by
  rfl
```

`#eval outputWidth` prints `1`. `HasOneOutput` does not print a model result; it is a proposition.
Here `rfl` works because the two definitions reduce to the same natural number. When Lean accepts
`hasOneOutput`, the kernel has checked a proof of exactly `outputWidth = 1`.

The same separation appears in verification. An interval pass computes a candidate output box. A
predicate states that the box contains every semantic output, and a soundness theorem proves that
statement under explicit assumptions. Running the pass produces data; the theorem explains what
accepting that data establishes.

# Effects Appear In The Result Type

A pure function returns its value directly:

```
def increment (n : Nat) : Nat :=
  n + 1
```

Printing has an effect, so the result is wrapped in `IO`:

```
def printIncrement (n : Nat) : IO Unit := do
  IO.println s!"next = {increment n}"
```

The `do` block sequences effectful actions. Training sessions, checkpoint access, subprocesses, and
native runtime calls use `IO`. Pure shape calculations and mathematical specifications do not.

You can run an `IO` definition at compile time while experimenting:

```
#eval printIncrement 4
```

It prints:

```
next = 5
```

In executable modules, an entry point such as `main : List String → IO UInt32` or
`main : List String → IO Unit` receives command-line arguments and performs the run.

# Read Types From Right To Left

Long TorchLean types become easier when read from the result backward. For example:

```
nn.linear 2 4 :
  nn.M (nn.Sequential (.dim 2 .scalar) (.dim 4 .scalar))
```

Read it as:

1. after initialization, the result is an `nn.Sequential`;
2. that sequence maps a length-two vector to a length-four vector;
3. constructing it lives in `nn.M` because initialization consumes deterministic seeds.

Similarly:

```
Trainer.new model ... : Trainer.Handle σ τ
```

says that construction returns a trainer whose model input is `σ` and output is `τ`. Later method
calls preserve those shape indices.

# When Lean Reports An Error

Lean's full elaboration messages can be noisy because they expose generated metavariables and
qualified implementation names. Start by finding:

- *has type*: the value that was actually supplied;
- *but is expected to have type*: the interface it must satisfy;
- the first shape, scalar type, or namespace where they differ.

For model composition, work from left to right and compare each layer's output with the next layer's
input. For tensors, compare both the scalar type and the recursive shape. For training, check that
the dataset target shape agrees with the model output and task.

The editor's hover information and `#check` are often faster than guessing. Reduce a large
expression to a named `def`, ask Lean for its type, and then compose it with the next piece.

# Further Reading

- [Functional Programming in Lean](https://lean-lang.org/functional_programming_in_lean/) is a
  friendly introduction to Lean as a programming language.
- The [Lean language reference](https://lean-lang.org/doc/reference/latest/) gives the precise
  syntax and semantics.
- de Moura and Ullrich, ["The Lean 4 Theorem Prover and Programming
  Language"](https://lean-lang.org/papers/lean4.pdf), CADE 2021.
- [Mathematics in Lean](https://leanprover-community.github.io/mathematics_in_lean/) introduces
  theorem proving with Mathlib.

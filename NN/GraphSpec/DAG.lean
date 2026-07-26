/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.DAG.Core
public import NN.GraphSpec.Primitives
import Mathlib.Algebra.Order.Algebra

/-!
# GraphSpec DAG Entry Point

Umbrella import for the canonical **general DAG-shaped** GraphSpec surface.

If you are building:

- a plain pipeline like `Linear >>> ReLU >>> Linear`, author it with `NN.GraphSpec.Core` and lower
  it to DAG when needed;
- anything with skip connections, shared subexpressions, or true multi-input nodes, author it
  directly with `NN.GraphSpec.DAG`.

This import gives you the DAG term language plus the standard DAG-side primitive pack. The
GraphSpec-specific example architectures live under:

- `NN.GraphSpec.Models`

So the intended reading order is:

1. `NN.GraphSpec.Core` for the small sequential surface,
2. `NN.GraphSpec.DAG.Core` when you need explicit sharing,
3. `NN.GraphSpec.Models` for concrete examples.

## Where are the DAG primitives?

They live in this entrypoint, not in a separate `NN.GraphSpec.DAG.Primitives` module. That keeps the
DAG surface compact and keeps primitive constructors next to the public DAG import.

The key dependency reason is that `DAG.Core` defines the term calculus and is imported by the
sequential `GraphSpec.Core` lowering code. The DAG primitive pack, however, is mostly derived from
sequential primitives such as `Primitive.linear`, `Primitive.relu`, and `Primitive.conv2d`. Putting
those derived definitions into `DAG.Core` would create an import cycle:

`DAG.Core → GraphSpec.Core → DAG.Core`.

So the split is:

- `NN.GraphSpec.DAG.Core`: calculus only (`PrimOp`, `Term`, `Args`, `Model`, eval/compile).
- `NN.GraphSpec.DAG`: public DAG entrypoint plus the derived primitive constructors.
-/

@[expose] public section


namespace NN
namespace GraphSpec
namespace DAG

open Spec
open Spec.Tensor
open NN.Tensor

namespace PrimOp

/-! ## Basic DAG primitives -/

/--
Dense linear layer in DAG form.

Inputs are ordered as `[W, b, x]`:

- `W : Mat outDim inDim`,
- `b : Vec outDim`,
- `x : Vec inDim`.

The output is `Vec outDim`. This is the DAG embedding of `Primitive.linear`, so the DAG and
sequential authoring surfaces share the same Spec semantics and TorchLean compiler path.
-/
def linear (inDim outDim : Nat) :
    PrimOp [.dim outDim (.dim inDim .scalar), .dim outDim .scalar, .dim inDim .scalar] (.dim outDim .scalar) :=
  (LowerToDAG.Primitive.toDAGPrimOp (Primitive.linear inDim outDim) : PrimOp _ _)

/--
Flatten a tensor to a one-dimensional vector in DAG form.

Input: `[x : Spec.Tensor s]`.
Output: `Vec (Spec.Shape.size s)`.

This is the DAG embedding of `Primitive.flatten`, so it has exactly the same row-major view
semantics as the sequential primitive.
-/
def flatten (s : Shape) : PrimOp [s] (.dim (Spec.Shape.size s) .scalar) :=
  (LowerToDAG.Primitive.toDAGPrimOp (Primitive.flatten s) : PrimOp _ _)

/-! ## Vision / residual DAG primitives -/

/--
ReLU activation in DAG form.

Input: `[x : s]`, output: `s`.

Semantics: elementwise $\max(x,0)$. This is parameter-free and derived from `Primitive.relu`.

Reference: Nair and Hinton (2010), "Rectified Linear Units Improve Restricted Boltzmann Machines".
-/
def relu (s : Shape) : PrimOp [s] s :=
  (LowerToDAG.Primitive.toDAGPrimOp (Primitive.relu s) : PrimOp _ _)

/--
Add two tensors of the same shape.

Input shapes: `[s, s]`, output shape: `s`.

This is the primitive used for residual/skip connections:
$\mathrm{out}=\operatorname{main}(x)+x$. It is defined
directly because the sequential surface is unary, while residual addition is genuinely multi-input.
-/
def add (s : Shape) : PrimOp [s, s] s :=
  { name := "add"
    specFwd := fun {α} _ctx xs =>
      match xs with
      | .cons a (.cons b .nil) => _root_.Spec.Tensor.addSpec (α := α) a b
    torchProgram := fun {α} _ctx _deq =>
      fun {m} _ _ =>
        fun a b => Runtime.Autograd.TorchLean.add (m := m) (α := α) (s := s) a b
  }

/--
2D convolution in DAG form, using channel-first `CHW` tensors without an explicit batch dimension.

Inputs are ordered as `[kernel, bias, x]`:

- `kernel : OIHW outC inC kH kW`,
- `bias   : Vec outC`,
- `x      : CHW inC inH inW`.

The output shape uses the standard convolution formula:

`outH = Spec.Shape.slidingWindowOutDim inH kH stride padding`

and similarly for `outW`. This is derived from the sequential `Primitive.conv2d`.
-/
def conv2d
    (inC outC kH kW stride padding inH inW : Nat)
    {h_inC : inC ≠ 0} {h_kH : kH ≠ 0} {h_kW : kW ≠ 0} {hStride : stride ≠ 0} :
    PrimOp
      [ .dim outC (.dim inC (.dim kH (.dim kW .scalar))), .dim outC .scalar, .dim inC (.dim inH (.dim inW .scalar)) ]
      (.dim outC (.dim (Spec.Shape.slidingWindowOutDim inH kH stride padding) (.dim (Spec.Shape.slidingWindowOutDim inW kW stride padding) .scalar))) :=
  (LowerToDAG.Primitive.toDAGPrimOp
      (Primitive.conv2d (inC := inC) (outC := outC) (kH := kH) (kW := kW)
        (stride := stride) (padding := padding) (inH := inH) (inW := inW)
        (h_inC := h_inC) (h_kH := h_kH) (h_kW := h_kW) (hStride := hStride)) : PrimOp _ _)

/--
Max pooling in DAG form for channel-first `CHW` tensors.

Input: `[x : CHW inC inH inW]`.
Output shape uses the standard pooling formula:

`outH = Spec.Shape.slidingWindowOutDim inH kH stride 0`

and similarly for `outW`. This is derived from the sequential `Primitive.maxPool2d`.
-/
def maxPool2d
    (kH kW inH inW inC stride : Nat)
    {h_kH : kH ≠ 0} {h_kW : kW ≠ 0} {hStride : stride ≠ 0} :
    PrimOp [.dim inC (.dim inH (.dim inW .scalar))] (.dim inC (.dim (Spec.Shape.slidingWindowOutDim inH kH stride 0) (.dim (Spec.Shape.slidingWindowOutDim inW kW stride 0) .scalar))) :=
  (LowerToDAG.Primitive.toDAGPrimOp
      (Primitive.maxPool2d (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride :=
        stride)
        (h_kH := h_kH) (h_kW := h_kW) (hStride := hStride)) : PrimOp _ _)

/--
Batch normalization on `CHW` tensors in DAG form.

Inputs are `[gamma, beta, x]` where `gamma,beta : Vec channels` and
`x : CHW channels height width`.

This version models the learnable affine parameters but does not carry running mean/variance state
in the graph; stateful training statistics belong in an explicit runtime/state model.

Reference: Ioffe and Szegedy (2015), "Batch Normalization: Accelerating Deep Network Training...".
-/
def batchnormChw
    (channels height width : Nat)
    (h_c : channels > 0) (h_h : height > 0) (h_w : width > 0) :
    PrimOp
      [.dim channels .scalar, .dim channels .scalar, .dim channels (.dim height (.dim width .scalar))]
      (.dim channels (.dim height (.dim width .scalar))) :=
  (LowerToDAG.Primitive.toDAGPrimOp
      (Primitive.batchnormChw (channels := channels) (height := height) (width := width)
        (h_c := h_c) (h_h := h_h) (h_w := h_w)) : PrimOp _ _)

end PrimOp

end DAG
end GraphSpec
end NN

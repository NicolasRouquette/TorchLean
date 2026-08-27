/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape

/-!
# Graph neural network layers (spec layer)

We provide a couple of small, standard GNN building blocks that show up in lots of papers and
PyTorch GNN libraries:

- a basic "message passing / neighbor aggregation" primitive, and
- a GCN-style graph convolution layer.

## Message passing (the common core idea)

Most GNN layers have the same shape of computation:

- aggregate neighbor features using the graph structure, then
- optionally apply a learnable transformation and a nonlinearity.

In this file the aggregation step is written with a matrix `A : (n×n)`:

`Agg(A, H) = A · H`.

This captures many common conventions:

- if `A` is the raw adjacency, you are summing neighbors,
- if `A` is normalized (e.g. `D^{-1/2} (A + I) D^{-1/2}`), you are doing the "GCN normalization"
  flavor,
- if `A` includes edge weights, you are doing a weighted sum.

## GCN layer (one very common choice)

We model a GCN-style layer as:

  `H' = A · H · W + b`

where:
- `A : (n×n)` is an adjacency-like matrix (often normalized, and often with self-loops),
- `H : (n×inDim)` are node features,
- `W : (inDim×outDim)` and `b : outDim` are trainable parameters.

PyTorch mental picture:

- This is the algebraic core of what libraries like PyTorch Geometric call `GCNConv` once you pick
  a concrete choice of `A` (raw adjacency, `D^{-1/2} (A + I) D^{-1/2}`, etc.) and batch conventions.

Why this file defines only these two:

- GCN + plain aggregation are enough to cover a lot of examples and give us something we can
  reason about cleanly.
- We do plan to add other families (GraphSAGE, GAT, generic MPNNs). Those require more choices
  (per-edge features, masking/batching conventions, and tie-ins to attention-style ops), so we want
  to introduce them carefully instead of piling on half-finished variants.
-/

@[expose] public section


namespace Spec

open Tensor
open Shape

variable {α : Type} [Context α]

/-- Neighbor aggregation / message passing via a graph matrix: `Agg(A, X) = A · X`.

This is the reusable "mix neighbors" step. The semantics are entirely determined by `A`
(raw adjacency, normalized adjacency, weighted adjacency, etc.). -/
def messagePassingSpec {n inDim : Nat}
  (A : Tensor α [n, n])
  (x : Tensor α [n, inDim]) :
  Tensor α [n, inDim] :=
  matMulSpec A x

/-- Backward/VJP for `message_passing_spec`: returns `(dA, dX)`. -/
def messagePassingBackwardSpec {n inDim : Nat}
  (A : Tensor α [n, n])
  (x : Tensor α [n, inDim])
  (dY : Tensor α [n, inDim]) :
  (Tensor α [n, n] × Tensor α [n, inDim]) :=
  matmulBackwardSpec (α := α) (m := n) (n := n) (p := inDim)
    (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar) A x dY

/-- Parameters/data for a single GCN-style layer.

We bundle `A` with the layer because many code paths treat `A` as a fixed input per graph, while
others treat it as a parameter (e.g. learned normalization). Keeping it in the record makes both
uses explicit.
-/
structure GCNLayerSpec (n inDim outDim : Nat) (α : Type) where
  /-- A. -/
  A : Tensor α [n, n]
  /-- W. -/
  W : Tensor α [inDim, outDim]
  /-- b. -/
  b : Tensor α [outDim]

/-- Forward spec for a GCN-style layer: `Y = A · X · W + b`.

Notes:

- The bias `b` is broadcast across the `n` nodes (row-wise add).
- Any normalization/self-loop convention belongs in the choice of `A` supplied to the layer.
-/
def gcnLayerSpec {n inDim outDim : Nat}
  (layer : GCNLayerSpec n inDim outDim α)
  (x : Tensor α [n, inDim]) :
  Tensor α [n, outDim] :=
  let ax : Tensor α [n, inDim] :=
    messagePassingSpec (α := α) (n := n) (inDim := inDim) layer.A x
  let axw : Tensor α [n, outDim] :=
    matMulSpec ax layer.W
  let hB : Shape.CanBroadcastTo (.dim outDim .scalar) (.dim n (.dim outDim .scalar)) := by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    exact Shape.CanBroadcastTo.scalar
  addSpec axw (broadcastTo hB layer.b)

/-!
## Gradients

For the simple GCN-style layer

`Y = A · X · W + b`

the reverse-mode derivatives are the standard matrix calculus ones:

- `dW = (A·X)ᵀ · dY`
- `db = Σᵢ dYᵢ` (sum across the node axis)
- `dX = Aᵀ · (dY · Wᵀ)`
- `dA = (dY · Wᵀ) · Xᵀ`

We include `dA` because in some setups the adjacency/normalization is also:

- treated as an input you want sensitivities for, or
- treated as a parameter (e.g. learned edge weights / learned normalization).
-/

/-- Backward/VJP spec for `gcn_layer_spec`.

Returns `(dA, dW, db, dX)` in that order. -/
def gcnLayerBackwardSpec {n inDim outDim : Nat}
  (layer : GCNLayerSpec n inDim outDim α)
  (x : Tensor α [n, inDim])
  (grad_output : Tensor α [n, outDim])
  (h_n : n ≠ 0) :
  (Tensor α [n, n] ×               -- ∂L/∂A
   Tensor α [inDim, outDim] ×      -- ∂L/∂W
   Tensor α [outDim] ×                   -- ∂L/∂b
   Tensor α [n, inDim]) :=         -- ∂L/∂x

  let ax : Tensor α [n, inDim] := matMulSpec layer.A x

  -- Backprop through the second matmul: (A·X) · W
  let (dAx, dW) :=
    matmulBackwardSpec (Shape.CanBroadcastTo.refl .scalar)
      (Shape.CanBroadcastTo.refl .scalar) ax layer.W grad_output

  -- Bias gradient: sum across the node axis.
  let db := reduceSum (α := α) (s := Shape.dim n (Shape.dim outDim Shape.scalar)) 0
    grad_output (Shape.hasNonemptyAxisZeroOfNe h_n).proof

  -- Backprop through the first matmul: A · X
  let (dA, dX) :=
    matmulBackwardSpec (Shape.CanBroadcastTo.refl .scalar)
      (Shape.CanBroadcastTo.refl .scalar) layer.A x dAx
  (dA, dW, db, dX)

end Spec

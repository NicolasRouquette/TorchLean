/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Gnn
public import NN.Spec.Module.Core

/-!
# Graph layers as `Spec.Module`s

`NN/Spec/Layers/Gnn.lean` defines a small GCN-style layer spec:

`H' = A · H · W + b`

This file wraps that forward spec as an `Spec.Module` so it can be composed in `Spec.Module.Chain`
pipelines and carry simple export metadata.
-/

@[expose] public section


namespace Spec.Module

open Tensor

variable {α : Type} [Context α]

/-- GCN layer wrapper: `(n, inDim) -> (n, outDim)`. -/
def gcn {n inDim outDim : Nat}
  (layer : GCNLayerSpec n inDim outDim α) :
  Spec.Module α ([n, inDim]) ([n, outDim]) :=
{ forward := fun x => gcnLayerSpec (α := α) layer x
  kind := "GCN"
  pythonExpr := s!"GCNLayer(n={n}, in={inDim}, out={outDim})" }

end Spec.Module

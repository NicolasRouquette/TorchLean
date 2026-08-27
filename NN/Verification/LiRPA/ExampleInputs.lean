/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Cert.IBPCert

/-!
# LiRPA Example Inputs

Deterministic centers and input boxes shared by the LiRPA examples.
-/

@[expose] public section

namespace NN.Verification.LiRPA.ExampleInputs

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open _root_.Spec
open _root_.Spec.Tensor

/-- Center vector `[1, 2, ..., dim]`, used by the small deterministic LiRPA examples. -/
def naturalCenter (dim : Nat) : Tensor Float [dim] :=
  Tensor.dim (fun i => Tensor.scalar (Float.ofNat (i.val + 1)))

/-- Insert an $L^\infty$ input box around `center` into a graph parameter store. -/
def seedInputBox (inputId dim : Nat)
    (center : Tensor Float [dim]) (eps : Float)
    (ps : ParamStore Float) : ParamStore Float :=
  ps.seedLInfBall inputId center eps

/-- Insert an $L^\infty$ input box around the center vector $[1,2,\ldots,\mathrm{dim}]$. -/
def seedNaturalInputBox (inputId dim : Nat) (eps : Float)
    (ps : ParamStore Float) : ParamStore Float :=
  seedInputBox inputId dim (naturalCenter dim) eps ps

end NN.Verification.LiRPA.ExampleInputs

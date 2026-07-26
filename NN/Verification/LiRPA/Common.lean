/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Cert.IBPCert

/-!
# Shared LiRPA Certificate Helpers

Small utilities used by the LiRPA certificate examples.  The model graphs stay in their own modules;
this file only contains the repeated artifact-checking and input-box plumbing.
-/

@[expose] public section

namespace NN.Verification.LiRPA

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open _root_.Spec
open _root_.Spec.Tensor

/-- Center vector `[1, 2, ..., dim]`, used by the small deterministic LiRPA examples. -/
def naturalCenter (dim : Nat) : Tensor Float (.dim dim .scalar) :=
  Tensor.dim (fun i => Tensor.scalar (Float.ofNat (i.val + 1)))

/-- Insert an $L^\infty$ input box around `center` into a graph parameter store. -/
def seedVectorInputBox (inputId dim : Nat)
    (center : Tensor Float (.dim dim .scalar)) (eps : Float)
    (ps : ParamStore Float) : ParamStore Float :=
  ps.seedLInfBall inputId center eps

/-- Insert an $L^\infty$ input box around the center vector $[1,2,\ldots,\mathrm{dim}]$. -/
def seedNaturalInputBox (inputId dim : Nat) (eps : Float)
    (ps : ParamStore Float) : ParamStore Float :=
  seedVectorInputBox inputId dim (naturalCenter dim) eps ps

/-- Recompute Lean IBP bounds and compare them against a JSON certificate. -/
def checkIBPCert (g : Graph) (ps : ParamStore Float) (outId : Nat)
    (path : String) : IO Unit :=
  NN.Verification.IBPCert.checkOrThrow g ps outId path

end NN.Verification.LiRPA

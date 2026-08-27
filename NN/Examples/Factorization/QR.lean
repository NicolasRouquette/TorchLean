/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common
meta import NN.Examples.Factorization.Common

/-!
# QR Factorization

`qrSpec A` returns `(Q, R)` with $A=QR$, where $Q$ has orthonormal columns and $R$ is
upper-triangular. We check both $A=QR$ and $Q^\mathsf{T}Q=I$.
-/

@[expose] public section

namespace NN.Examples.Factorization.QR

/-- A standard $3\times3$ QR example. -/
def A : Spec.Tensor Float [3, 3] :=
  mkMat #[#[12, -51, 4], #[6, 167, -68], #[-4, 24, -41]]

/-- Orthonormal `Q` factor. -/
def Q : Spec.Tensor Float [3, 3] := Spec.qrQSpec A

/-- Upper-triangular `R` factor. -/
def R : Spec.Tensor Float [3, 3] := Spec.qrRSpec A

/-- Reconstruction error $\lVert A-QR\rVert_{\max}$. -/
def reconErr : Float := maxMatErr A (matmul Q R)

/-- Orthonormality error $\lVert Q^\mathsf{T}Q-I\rVert_{\max}$. -/
def orthoErr : Float := maxMatErr (matmul (tr Q) Q) (Spec.identityTensorSpec 3)

#guard_msgs (drop info) in
#eval assertLt "QR A = Q·R" reconErr

#guard_msgs (drop info) in
#eval assertLt "QR Qᵀ·Q = I" orthoErr

/-! ## Negative Control

The orthonormality theorem requires full column rank. The following matrix has one dependent
column. Gram-Schmidt still reconstructs it, but the corresponding column of `Q` vanishes and
$Q^\mathsf{T}Q\ne I$.
-/

/-- A matrix whose second column is twice its first. -/
def Adef : Spec.Tensor Float [3, 3] :=
  mkMat #[#[1, 2, 0], #[2, 4, 1], #[1, 2, 0]]

def Qdef : Spec.Tensor Float [3, 3] := Spec.qrQSpec Adef

def Rdef : Spec.Tensor Float [3, 3] := Spec.qrRSpec Adef

/-- Reconstruction still holds without full rank. -/
def reconErrDef : Float := maxMatErr Adef (matmul Qdef Rdef)

/-- Orthonormality fails because `Q` has a zero column. -/
def orthoErrDef : Float := maxMatErr (matmul (tr Qdef) Qdef) (Spec.identityTensorSpec 3)

#guard_msgs (drop info) in
#eval assertLt "QR(rank-deficient) A = Q·R still reconstructs" reconErrDef

#guard_msgs (drop info) in
#eval assertGe "QR(rank-deficient) Qᵀ·Q = I correctly fails (needs full column rank)" orthoErrDef

end NN.Examples.Factorization.QR

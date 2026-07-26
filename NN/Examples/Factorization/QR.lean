/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common
meta import NN.Examples.Factorization.Common

/-!
# Example: QR factorization

`qrSpec A` returns `(Q, R)` with $A=QR$, where $Q$ has orthonormal columns and $R$ is
upper-triangular (classical Gram–Schmidt). We check both $A=QR$ and $Q^\mathsf{T}Q=I$.
-/

@[expose] public section


namespace NN.Examples.Factorization.QR

/-- A 3×3 test matrix (the classic Householder/QR example). -/
def A : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) :=
  mkMat [[12, -51, 4],
         [6, 167, -68],
         [-4, 24, -41]]

/-- Orthonormal `Q` factor. -/
def Q : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) := Spec.qrQSpec A
/-- Upper-triangular `R` factor. -/
def R : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) := Spec.qrRSpec A

/-- Reconstruction error $\lVert A-QR\rVert_{\max}$. -/
def reconErr : Float := maxMatErr A (mm Q R)
/-- Orthonormality error $\lVert Q^\mathsf{T}Q-I\rVert_{\max}$. -/
def orthoErr : Float := maxMatErr (mm (tr Q) Q) (Spec.identityTensorSpec 3)

-- Compiled assertions (fail the build otherwise).
#guard_msgs (drop info) in
#eval assertLt "QR A = Q·R" reconErr
#guard_msgs (drop info) in
#eval assertLt "QR Qᵀ·Q = I" orthoErr

/-! ## Negative control: full column rank is necessary for orthonormality

`qrSpec_orthonormal` ($Q^\mathsf{T}Q=I$) requires full column rank — positive $R$-pivots
($0<R_{jj}$). The matrix below has a dependent column ($\mathrm{col}_2=2\,\mathrm{col}_1$), so
Gram–Schmidt produces a **zero** $Q$ column where the pivot vanishes: $A=QR$ still holds, but
$Q^\mathsf{T}Q$ has a zero on the
diagonal, so orthonormality fails. This separates the two guarantees and shows the rank hypothesis
genuinely bites. -/

/-- A rank-2 matrix ($\mathrm{col}_2=2\,\mathrm{col}_1$): reconstructs, but $Q$ cannot be
orthonormal. -/
def Adef : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) :=
  mkMat [[1, 2, 0],
         [2, 4, 1],
         [1, 2, 0]]

def Qdef : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) := Spec.qrQSpec Adef
def Rdef : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) := Spec.qrRSpec Adef

/-- Reconstruction still holds even without full rank. -/
def reconErrDef : Float := maxMatErr Adef (mm Qdef Rdef)
/-- Orthonormality fails: $Q^\mathsf{T}Q$ has a zero diagonal entry, so it is far from $I$. -/
def orthoErrDef : Float := maxMatErr (mm (tr Qdef) Qdef) (Spec.identityTensorSpec 3)

#guard_msgs (drop info) in
#eval assertLt "QR(rank-deficient) A = Q·R still reconstructs" reconErrDef
#guard_msgs (drop info) in
#eval assertGe "QR(rank-deficient) Qᵀ·Q = I correctly fails (needs full column rank)" orthoErrDef

end NN.Examples.Factorization.QR

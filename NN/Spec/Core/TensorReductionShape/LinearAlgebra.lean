/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape.Reductions

@[expose] public section


namespace Spec
namespace Tensor

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-!
# Linear Algebra Helpers

Matrix transpose, 3D transposes, matmul/bmm backward specs, and shape matching.
-/

/-- Transpose a matrix `(m×n)` into `(n×m)`.

PyTorch analogy: `A.transpose(0, 1)` or `A.T` for 2D tensors. -/
def matrixTransposeSpec
  {α : Type} {m n : Nat}
  (t : Tensor α (.dim m (.dim n .scalar))) :
  Tensor α (.dim n (.dim m .scalar)) :=
  match t with
  | Tensor.dim rows =>
    Tensor.dim (fun j : Fin n =>
      Tensor.dim (fun i : Fin m =>
        match rows i with
        | Tensor.dim cols =>
          match cols j with
          | Tensor.scalar value => Tensor.scalar value))

-- Advanced Transpose Operations
/-- Permute a 3D tensor from `(a,b,c)` to `(b,c,a)`. -/
def transpose3DFirstToLastSpec {α : Type} {a b c : Nat}
  (t : Tensor α (.dim a (.dim b (.dim c .scalar)))) :
  Tensor α (.dim b (.dim c (.dim a .scalar))) :=
  match t with
  | .dim f =>
    .dim fun j =>
      .dim fun k =>
        .dim fun i =>
          match f i with
          | .dim g =>
            match g j with
            | .dim h => .scalar (match h k with | .scalar x => x)

/-- Permute a 3D tensor from `(a,b,c)` to `(c,a,b)`. -/
def transpose3DLastToFirstSpec {α : Type} {a b c : Nat}
  (t : Tensor α (.dim a (.dim b (.dim c .scalar)))) :
  Tensor α (.dim c (.dim a (.dim b .scalar))) :=
  match t with
  | .dim f =>
    .dim fun k =>
      .dim fun i =>
        .dim fun j =>
          match f i with
          | .dim g =>
            match g j with
            | .dim h => .scalar (match h k with | .scalar x => x)

/-- Swap the last two axes of a 3D tensor: `(a,b,c)` to `(a,c,b)`. -/
def transpose3DLastTwoSpec {α : Type} {a b c : Nat}
  (t : Tensor α (.dim a (.dim b (.dim c .scalar)))) :
  Tensor α (.dim a (.dim c (.dim b .scalar))) :=
  match t with
  | .dim f =>
    .dim fun i =>
      match f i with
      | .dim g =>
        .dim fun k =>
          .dim fun j =>
            match g j with
            | .dim h => .scalar (match h k with | .scalar x => x)

/-- Swap the first two dimensions of a tensor `(m,n,...)` to `(n,m,...)`. -/
def swapFirstTwoSpec {α : Type} {m n : Nat} {s : Shape}
  (t : Tensor α (.dim m (.dim n s))) :
  Tensor α (.dim n (.dim m s)) :=
  match t with
  | .dim f =>
    .dim fun j =>
      .dim fun i =>
        match f i with
        | .dim g => g j

/-- Swap adjacent tensor axes at `depth` and `depth + 1`. -/
def swapAdjacentAxes {β : Type} {shape : Shape} (tensor : Tensor β shape) (d : Nat) :
      Tensor β (shape.swapAdjacentAtDepth d) :=
      match d, shape, tensor with
      | 0, .dim m (.dim k rest), .dim g =>
        -- Swap axes 0 and 1 at this level.
        .dim fun j =>
          .dim fun i =>
            match g i with
            | .dim h => h j
      | d + 1, .dim m rest, .dim g =>
        -- Preserve this axis and recurse into the remaining shape.
        .dim fun i => swapAdjacentAxes (g i) d
      | _, .scalar, .scalar x =>
        -- A scalar has no axes to swap.
        by simp [Shape.swapAdjacentAtDepth]; exact .scalar x
      | 0, .dim _ .scalar, .dim g =>
        -- A rank-one tensor has no adjacent pair at this level.
        .dim g

/-- Apply adjacent-axis swaps while retaining the resulting shape in the return type. -/
def permuteByAdjacentSwaps {β : Type} {s : Shape} (tensor : Tensor β s) :
    (depths : List Nat) → Tensor β (s.applyAdjacentSwaps depths)
  | [] => tensor
  | depth :: depths =>
      permuteByAdjacentSwaps (swapAdjacentAxes tensor depth) depths

/-- Swapping at depth zero exchanges the two leading axes. -/
@[simp] theorem swapAdjacentAxes_zero {β : Type} {m n : Nat} {s : Shape}
    (tensor : Tensor β (.dim m (.dim n s))) :
    swapAdjacentAxes tensor 0 =
      .dim (fun j => .dim (fun i => _root_.Spec.get (_root_.Spec.get tensor i) j)) := by
  cases tensor
  rfl

/-- Swap adjacent dimensions at a given depth inside a leading batch dimension. -/
def swapAdjacentAxesWithinLeading {α : Type} {n : Nat} {s : Shape}
  (t : Tensor α (.dim n s)) (depth : Nat) :
  Tensor α (.dim n (s.swapAdjacentAtDepth depth)) :=
  match t with
  | .dim f =>
    .dim fun i => swapAdjacentAxes (f i) depth

-- Backward pass for matrix multiplication
/-- Backward pass for matrix multiplication: returns `(dA, dB)` given `dC`.

PyTorch analogy: if $C=AB$, then:

- $dA=dC\,B^\mathsf{T}$
- $dB=A^\mathsf{T}dC$ -/
def matMulBackwardSpec
  {m n p : Nat}
  (A : Tensor α (.dim m (.dim n .scalar)))
  (B : Tensor α (.dim n (.dim p .scalar)))
  (dC : Tensor α (.dim m (.dim p .scalar))) :
  (Tensor α (.dim m (.dim n .scalar))) × (Tensor α (.dim n (.dim p .scalar))) :=
  let dA := matMulSpec dC (matrixTransposeSpec B) -- dA = dC * Bᵀ
  let dB := matMulSpec (matrixTransposeSpec A) dC -- dB = Aᵀ * dC
  (dA, dB)

/-- Batched matrix multiplication: `[batch,m,n] × [batch,n,p] → [batch,m,p]`. -/
def bmmSpec {α : Type} [Add α] [Mul α] [Zero α]
  {batch m n p : Nat}
  (A : Tensor α (.dim batch (.dim m (.dim n .scalar))))
  (B : Tensor α (.dim batch (.dim n (.dim p .scalar)))) :
  Tensor α (.dim batch (.dim m (.dim p .scalar))) :=
  match A, B with
  | .dim fA, .dim fB =>
      .dim fun i => matMulSpec (match fA i with | t => t) (match fB i with | t => t)

/-- Backward pass for batched matrix multiplication. -/
def bmmBackwardSpec {α : Type} [Add α] [Mul α] [Zero α]
  {batch m n p : Nat}
  (A : Tensor α (.dim batch (.dim m (.dim n .scalar))))
  (B : Tensor α (.dim batch (.dim n (.dim p .scalar))))
  (dC : Tensor α (.dim batch (.dim m (.dim p .scalar)))) :
  (Tensor α (.dim batch (.dim m (.dim n .scalar)))) × (Tensor α (.dim batch (.dim n (.dim p
    .scalar)))) :=
  let dA :=
    bmmSpec (α := α) (batch := batch) (m := m) (n := p) (p := n) dC
      (transpose3DLastTwoSpec (α := α) (a := batch) (b := n) (c := p) B)
  let dB :=
    bmmSpec (α := α) (batch := batch) (m := n) (n := m) (p := p)
      (transpose3DLastTwoSpec (α := α) (a := batch) (b := m) (c := n) A) dC
  (dA, dB)
end Tensor
end Spec

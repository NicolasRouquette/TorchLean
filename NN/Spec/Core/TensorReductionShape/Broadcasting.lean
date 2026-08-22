/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape.ShapeChange

@[expose] public section


namespace Spec
namespace Tensor

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-!
# Broadcasting

Spec-level broadcasting and broadcasted binary maps.
-/

/-
Broadcasting in the spec layer is defined in terms of `Shape.CanBroadcastTo`:

- you pick an explicit target shape `t`,
- and provide evidence that each operand can broadcast to `t`.

TorchLean standardizes on this explicit target style throughout core.

We intentionally do *not* provide a second "implicit" broadcasting API that tries to infer a common
output shape from two operands, because that would split the codebase into two parallel styles.
Instead, core code names the output shape and carries broadcast evidence explicitly (often inferred
by typeclass search via `Shape.BroadcastTo`).

This choice also matches the backward pass semantics: broadcasting duplicates values, so the adjoint is
a sum-reduction along the broadcasted axes (see `reduceFromBroadcastTo` below).
-/
/-- Broadcast a tensor along a `Shape.CanBroadcastTo` proof (spec-level analogue of
  `torch.broadcast_to`). -/
def broadcastTo {α : Type} [Inhabited α] :
  {s₁ s₂ : Shape} → Shape.CanBroadcastTo s₁ s₂ → Tensor α s₁ → Tensor α s₂
| _, _, Shape.CanBroadcastTo.scalar, t => t
| _, _, Shape.CanBroadcastTo.dim_eq tail, Tensor.dim xs =>
    Tensor.dim (fun i => broadcastTo tail (xs i))
| _, _, Shape.CanBroadcastTo.dim_1_to_n tail, Tensor.dim xs =>
    Tensor.dim (fun _ => broadcastTo tail (xs 0))
| _, _, Shape.CanBroadcastTo.expand_dims tail, t =>
    Tensor.dim (fun _ => broadcastTo tail t)

/-! ## Broadcasted maps -/

/--
Broadcast a scalar tensor to match a template tensor's shape.

This is the "like" broadcasting form used by specs that want to follow a template shape without
spelling out the `Shape.CanBroadcastTo` evidence.
-/
def broadcastLike {α : Type} [Inhabited α]
  {s : Shape} (_template : Tensor α s) (t : Tensor α .scalar) : Tensor α s :=
  replicate t

/-- Helper: map a scalar on the left over any tensor shape. -/
def mapScalarLeft {α : Type} (f : α → α → α) (x : α) :
  ∀ {s : Shape}, Tensor α s → Tensor α s
| _, Tensor.scalar y => Tensor.scalar (f x y)
| _, Tensor.dim g => Tensor.dim (fun i => mapScalarLeft f x (g i))

/-- Helper: map a scalar on the right over any tensor shape. -/
def mapScalarRight {α : Type} (f : α → α → α) (y : α) :
  ∀ {s : Shape}, Tensor α s → Tensor α s
| _, Tensor.scalar x => Tensor.scalar (f x y)
| _, Tensor.dim g => Tensor.dim (fun i => mapScalarRight f y (g i))

/--
Binary element-wise operation with broadcasting to an explicit target shape.

This is the helper you typically want in spec code:
- pick the output shape `t`,
- broadcast each operand to `t`,
- then `map2_spec` the pointwise operation.

PyTorch analogy: `f(x, y)` where `x` and/or `y` are broadcastable to a common shape.
The common shape is explicit rather than discovered at runtime, which makes the result type
predictable and fixes the intended output shape at the call site.
-/
def broadcastMapTo {α} [Inhabited α] (f : α → α → α)
    {s₁ s₂ t : Shape} (cbx : Shape.CanBroadcastTo s₁ t) (cby : Shape.CanBroadcastTo s₂ t) :
    Tensor α s₁ → Tensor α s₂ → Tensor α t :=
  fun x y => map2Spec f (broadcastTo cbx x) (broadcastTo cby y)
end Tensor
end Spec

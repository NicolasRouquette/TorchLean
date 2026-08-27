/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor.Constructors

/-!
# Tensor Printing

Human-facing dtype labels and checked tensor printing. Proof-only real and rounding-model tensors
deliberately reject printing instead of pretending to provide executable scalar output.
-/

@[expose] public section

namespace TorchLean.Tensor

/-- A compact dtype name used by `TorchLean.Tensor.print`. -/
class DTypeName (α : Type) where
  /-- Human-facing scalar type name. -/
  name : String

instance : DTypeName Float where name := "Float"
instance : DTypeName Float32 where name := "Float32"
instance : DTypeName ℚ where name := "ℚ"
instance : DTypeName Int where name := "Int"
instance : DTypeName TorchLean.Floats.FP32 where name := "FP32"
instance : DTypeName TorchLean.Floats.IEEE32Exec where name := "IEEE32Exec"
instance : DTypeName ℝ where name := "ℝ"

/-- Display name for TorchLean complex scalars. -/
instance {α : Type} [DTypeName α] : DTypeName (TorchLean.Complex α) where
  name := s!"Complex[{DTypeName.name (α := α)}]"

/-- Pretty-printing that can reject proof-only scalar types. -/
class TensorPrintable (α : Type) where
  /-- Render one tensor or explain why the scalar type is not printable. -/
  pretty : {s : Spec.Shape} → Spec.Tensor α s → Except String String

/-- Default printing for scalar types that support `ToString`. -/
instance (priority := 10) {α : Type} [ToString α] : TensorPrintable α where
  pretty := fun {_s} tensor => .ok (Spec.pretty tensor)

/-- Printing is disabled for proof-level real tensors. -/
instance (priority := 100) : TensorPrintable ℝ where
  pretty := fun {_s} _ =>
    .error
      "Refusing to print `Tensor ℝ` (proof-level); cast to `Float`/`Float32`/`IEEE32Exec`/`ℚ` to display."

/-- Printing is disabled for the proof-only `FP32` rounding model. -/
instance (priority := 100) : TensorPrintable TorchLean.Floats.FP32 where
  pretty := fun {_s} _ =>
    .error
      ("Refusing to print `Tensor FP32` (proof-only rounding model); " ++
        "use `IEEE32Exec`/`Float` for runtime printing.")

/-- Print a tensor with a dtype tag, or throw an `IO.userError` when printing is unsupported. -/
def print {α : Type} [DTypeName α] [TensorPrintable α] {s : Spec.Shape}
    (tensor : Spec.Tensor α s) : IO Unit := do
  match TensorPrintable.pretty (α := α) tensor with
  | .ok rendered => IO.println s!"[{DTypeName.name (α := α)}] {rendered}"
  | .error message => throw <| IO.userError message

end TorchLean.Tensor

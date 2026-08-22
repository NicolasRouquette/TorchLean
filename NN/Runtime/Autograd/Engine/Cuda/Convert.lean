/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

CUDA helpers: row-major conversions between spec tensors and `FloatArray`.

Motivation:
- CUDA buffers (`Runtime.Autograd.Cuda.Buffer`) are contiguous float32 arrays.
- Many CUDA kernels interpret buffers in row-major order for a given `Spec.Shape`.
- `Spec.Tensor` is a functional/nested representation that does not commit to a layout.

This module fixes a single layout convention for CUDA interop:
outermost-first recursion, where the last axis varies fastest (row-major / C-order).

We provide conversions for:
- `Spec.Tensor Float s` ↔ `FloatArray`
- `Spec.Tensor Bool s` ↔ `FloatArray` (Bool masks encoded as `1.0` for `true`, `0.0` for `false`)
-/

module

public import NN.Spec.Core.Tensor.Core

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

open Spec

namespace Convert

namespace Internal

/-!
### Flatten (`Spec.Tensor → FloatArray`)

Row-major order with outermost axis first and innermost axis last.
-/

def flattenFloat : {s : Shape} → Tensor Float s → FloatArray → FloatArray
  | .scalar, .scalar x, out => out.push x
  | .dim n s, .dim f, out =>
      (List.finRange n).foldl (fun acc i => flattenFloat (s := s) (f i) acc) out

end Internal

/-- Flatten a `Spec.Tensor Float s` into a row-major `FloatArray` (CUDA-compatible). -/
def flattenFloat {s : Shape} (t : Tensor Float s) : FloatArray :=
  Internal.flattenFloat (s := s) t (FloatArray.emptyWithCapacity (Spec.Shape.size s))

namespace Internal

def flattenBoolMask : {s : Shape} → Tensor Bool s → FloatArray → FloatArray
  | .scalar, .scalar b, out => out.push (if b then 1.0 else 0.0)
  | .dim n s, .dim f, out =>
      (List.finRange n).foldl (fun acc i => flattenBoolMask (s := s) (f i) acc) out

end Internal

/-- Flatten a `Spec.Tensor Bool s` mask to `FloatArray` as `0.0/1.0` values in row-major order. -/
def flattenBoolMask {s : Shape} (mask : Tensor Bool s) : FloatArray :=
  Internal.flattenBoolMask (s := s) mask (FloatArray.emptyWithCapacity (Spec.Shape.size s))

/-!
### Unflatten (`FloatArray → Spec.Tensor`)

These functions assume row-major order. The public operations check the expected length and return
`none` on mismatch. Runtime code that already owns the buffer-size invariant calls the internal
workers directly.
-/

namespace Internal

def unflattenFloat : {s : Shape} → FloatArray → (offset : Nat) → Tensor Float s
  | .scalar, a, offset =>
      Tensor.scalar (a.get! offset)
  | .dim n s, a, offset =>
      Tensor.dim (fun i : Fin n =>
        unflattenFloat (s := s) a (offset + i.val * Spec.Shape.size s))

end Internal

/-- Unflatten a row-major `FloatArray` into a `Spec.Tensor Float s` when `a.size` matches. -/
def unflattenFloat? {s : Shape} (a : FloatArray) : Option (Tensor Float s) :=
  if a.size = Spec.Shape.size s then
    some (Internal.unflattenFloat (s := s) a 0)
  else
    none

namespace Internal

def unflattenBoolMask : {s : Shape} → FloatArray → (offset : Nat) → Tensor Bool s
  | .scalar, a, offset =>
      Tensor.scalar (a.get! offset != 0.0)
  | .dim n s, a, offset =>
      Tensor.dim (fun i : Fin n =>
        unflattenBoolMask (s := s) a (offset + i.val * Spec.Shape.size s))

end Internal

/-- Unflatten a `FloatArray` into a `Spec.Tensor Bool s` mask when `a.size` matches. -/
def unflattenBoolMask? {s : Shape} (a : FloatArray) : Option (Tensor Bool s) :=
  if a.size = Spec.Shape.size s then
    some (Internal.unflattenBoolMask (s := s) a 0)
  else
    none

end Convert

end Cuda
end Autograd
end Runtime

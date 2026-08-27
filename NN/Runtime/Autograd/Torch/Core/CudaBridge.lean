/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Convert
public import NN.Runtime.Autograd.Engine.Cuda.Tape
public import NN.Runtime.Autograd.Torch.Core.TensorTransfer

/-!
# CUDA Tensor Storage Bridge

Adapt the public `TensorTransfer` capability to the float32 row-major storage owned by the eager
CUDA tape.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec

namespace Internal
namespace CudaBridge

/-- Upload a tensor to CUDA float32 storage through its runtime transfer representation. -/
def toAnyBuffer {α : Type} [TensorTransfer α] {s : Shape}
    (tensor : Tensor α s) : IO Runtime.Autograd.Cuda.AnyBuffer := do
  let host ← TensorTransfer.toFloatTensor tensor
  let values := Runtime.Autograd.Cuda.Convert.flattenFloat host
  let buffer ← Runtime.Autograd.Cuda.Buffer.ofFloatArrayIO values
  pure { s := s, buf := buffer }

/-- Download CUDA float32 storage through the scalar type's runtime transfer representation. -/
def ofAnyBuffer {α : Type} [TensorTransfer α]
    (stored : Runtime.Autograd.Cuda.AnyBuffer) : IO (Spec.SomeTensor α) := do
  let values := Runtime.Autograd.Cuda.Buffer.toFloatArray stored.buf
  match Runtime.Autograd.Cuda.Convert.unflattenFloat? (s := stored.s) values with
  | some tensor =>
      let decoded ← TensorTransfer.ofFloatTensor tensor
      pure { shape := stored.s, tensor := decoded }
  | none =>
      throw <| IO.userError <|
        s!"torch: cuda: bad buffer length (expected {Spec.Shape.size stored.s}, " ++
          s!"got {values.size})"

end CudaBridge
end Internal
end Torch
end Autograd
end Runtime

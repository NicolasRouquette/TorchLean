/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Runtime.Training.Trainer

import Mathlib.Algebra.Order.Algebra
import NN.Spec.Autograd.AutogradSpec

@[expose] public section

namespace NN
namespace API
namespace TorchLean

/-
The exports below expose the imperative session interface.

Most code should start from `Trainer.new` and `trainer.train`. Use these exports when a file needs:
- interactive/debug workflows that want mutable tape control, or
- advanced tooling that needs the low-level session primitives.
-/

namespace ScalarTrainer
/-!
Re-export of the low-level imperative scalar trainer interface.

This exposes `forwardT`/`backwardT`/`stepT` from `Runtime.Autograd.TorchLean.ScalarTrainer`.
Use the higher-level `TorchLean.Trainer` API unless a file needs these lower-level training hooks.
-/
export _root_.Runtime.Autograd.TorchLean.ScalarTrainer (forwardT backwardT stepT)
end ScalarTrainer

namespace Session
/-!
Imperative session API: a tape-backed interface that can run in eager or compiled mode.

This is approximately analogous to using PyTorch "eager tensors", except TorchLean makes the tape/session
explicit. The `Session` surface is useful for:
- interactive experiments in `IO`,
- debugging (inspect intermediate values),
- building higher-level runners.
-/
export _root_.Runtime.Autograd.TorchLean.Session
  (new resetTape param use input inputNat getNat setNat inputNatVec getNatVec setNatVec const
    getValue
   withFreshTape sgdStepScalarGraph
   add sub mul scale abs sqrt clamp max min
   broadcastTo reshape transpose2d transpose3dFirstToLast transpose3dLastToFirst
     transpose3dLastTwo
   swapAdjacentAtDepth
   reduceSum reduceMean
   gatherScalar gatherRow gatherScalarRef gatherRowRef gatherVecRef gatherRowsRef
   gatherScalarNat gatherVecNat gatherRowsNat scatterAddVec scatterAddRow
   matmul bmm concatVectors concatLeadingAxis sliceLeadingAxisRange maxPool2d smoothMaxPool2d avgPool2d
   relu sigmoid tanh softmax softplus exp log safeLog sum flatten
   linear mseLoss layerNorm conv2d multiHeadAttention
   backwardDenseAll backwardScalarDenseAll grad sgdStepAll)
end Session

end TorchLean
end API
end NN

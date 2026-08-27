/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Buffer
public import NN.Runtime.Autograd.Engine.Cuda.Kernels
public import NN.Runtime.Autograd.Engine.Cuda.ConvPool
public import NN.Tests.Runtime.Cuda.Utils
public import Std

/-!
# CUDA Deterministic Reductions Mode

These tests exercise TorchLean's opt-in "deterministic reductions" mode.

Goal: when deterministic mode is enabled, kernels that would otherwise accumulate using `atomicAdd`
must become bit-stable across runs (same input, same output, exact float equality).

The tests are written so they run:
- with CUDA enabled (`lake test -K cuda=true`) using real device buffers, and
- with CUDA disabled (default) via the CPU stub implementations.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace DeterministicReductions

open Runtime.Autograd.Cuda

def assertFloatArrayEq (msg : String) (a b : FloatArray) : IO Unit := do
  if a.size != b.size then
    throw <| IO.userError s!"{msg}: size mismatch ({a.size} vs {b.size})"
  for i in [:a.size] do
    let x := a.get! i
    let y := b.get! i
    if x != y then
      throw <| IO.userError s!"{msg}[{i}]: got {x}, expected {y}"

def runScatterAddTwice : IO Unit := do
  IO.println "== deterministic scatter_add: exact repeatability =="

  -- Enable deterministic mode via Lean side API.
  let enabled := Buffer.setDeterministicReductionsChecked true
  if !enabled then
    throw <| IO.userError "deterministic mode: expected flag to be enabled"

  -- Construct a values buffer with a large leading element + many ones, and scatter-add all
  -- updates into a single output index. This is a worst-case accumulator for non-associativity.
  let k : UInt32 := 4096
  let one : UInt32 := 1
  let n : UInt32 := 1

  let x := Buffer.zeros n
  let big := Buffer.full one 1.0e8
  let ones := Buffer.full (k - one) 1.0
  let values := Buffer.concatBuffers big ones one (k - one)

  let idx : Array Nat := Array.replicate k.toNat 0
  let y1 := Buffer.scatterAdd x values n idx k
  let y2 := Buffer.scatterAdd x values n idx k

  assertFloatArrayEq "scatterAdd deterministic run1 vs run2"
    (Buffer.toFloatArray y1) (Buffer.toFloatArray y2)

def outDim (inDim k stride padding : Nat) : Nat :=
  Spec.Shape.slidingWindowOutDim inDim k stride padding

def runAvgPoolBwdTwice : IO Unit := do
  IO.println "== deterministic avg_pool backward: exact repeatability =="

  let enabled := Buffer.setDeterministicReductionsChecked true
  if !enabled then
    throw <| IO.userError "deterministic mode: expected flag to be enabled"

  -- A small overlapping-window case (stride=1) so the backward pass needs accumulation.
  let inC : UInt32 := 1
  let inSpatial : Array Nat := #[17, 17]
  let kernel : Array Nat := #[3, 3]
  let stride : Array Nat := #[1, 1]
  let padding : Array Nat := #[1, 1]

  let outH : Nat := outDim 17 3 1 1
  let outW : Nat := outDim 17 3 1 1
  let outElems : UInt32 := UInt32.ofNat (inC.toNat * outH * outW)

  let gradOutput := Buffer.randUniform outElems 12345
  let y1 := torchleanAvgPoolBwdCuda gradOutput inSpatial kernel stride padding inC
  let y2 := torchleanAvgPoolBwdCuda gradOutput inSpatial kernel stride padding inC

  assertFloatArrayEq "avg_pool backward deterministic run1 vs run2"
    (Buffer.toFloatArray y1) (Buffer.toFloatArray y2)

/-- Entry point called by the CUDA runtime suite. -/
def run : IO Unit := do
  IO.println "== CUDA deterministic reductions =="
  runScatterAddTwice
  runAvgPoolBwdTwice
  let _ := Buffer.setDeterministicReductionsChecked false
  IO.println "== CUDA deterministic reductions: OK =="

end DeterministicReductions
end Cuda
end Tests

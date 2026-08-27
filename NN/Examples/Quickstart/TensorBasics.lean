/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Scalar
public import NN.Tensor

/-!
# Quickstart: Tensor Basics

This is the first stop in the TorchLean examples. It does **not** use sessions, CUDA, or autograd.
It is just about building typed tensors in Lean with a convenient constructor layer.

What it covers:
- arbitrary-rank constructors from literals or runtime arrays (`tensor!`, `Tensor.ofArray`),
- the fact that the element type `α` selects the tensor's scalar semantics,
- conversion of host `Float` literals to native `Float32` and reference `IEEE32Exec`,
- why we generally do not try to `print` tensors over `ℝ` (noncomputable / too large).

Run:
  `lake exe torchlean quickstart_tensors`
-/

@[expose] public section


namespace NN.Examples.Quickstart.TensorBasics

open TorchLean

/-- Command-line help for the tensor-basics quickstart. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean tensor basics quickstart"
    , ""
    , "Usage:"
    , "  lake exe torchlean quickstart_tensors"
    , ""
    , "This demo has no tutorial-specific flags."
    ]

def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  CLI.requireNoArgs "quickstart_tensors" args
  IO.println "== Quickstart: tensor basics =="

  -- Each tensor has one scalar type `α`; this is more static than a PyTorch runtime dtype.
  let xF : Tensor Float [4] := tensor! [0.1, 0.2, 0.3, 0.4]
  let xQ : Tensor ℚ [4] := tensor! (ty := ℚ) [0.1, 0.2, 0.3, 0.4]
  let xI : Tensor Int [4] := tensor! (ty := Int) [1, 2, 3, 4]

  Tensor.print xF
  Tensor.print xQ
  Tensor.print xI

  -- Native binary32 and the independent raw-bit reference have deliberately different names.
  let x32 ← CLI.orThrowIO <|
    Tensor.fromFloatArray Float.toFloat32 [4] #[0.1, 0.2, 0.3, 0.4]
  let x32Ref ← CLI.orThrowIO <|
    Tensor.fromFloatArray TorchLean.Floats.IEEE754.IEEE32Exec.ofFloat [4]
      #[0.1, 0.2, 0.3, 0.4]
  Tensor.print x32
  Tensor.print x32Ref

  -- N-D tensor using "nested brackets" (like nested Python lists in PyTorch).
  -- This is often the clearest way to see where each element goes.
  let x3 : Tensor Float [2, 2, 2] :=
    tensor! [
      [ [1, 2], [3, 4] ],
      [ [5, 6], [7, 8] ]
    ]
  Tensor.print x3

  -- Runtime values use `Tensor.ofArray`: provide dimensions and flat row-major storage.
  -- Row-major means the last dimension changes fastest:
  -- the above `x3` is the same as `Tensor.ofArray [2,2,2] #[1,2,3,4,5,6,7,8]`.

  -- Types may depend on runtime values, so dynamic dimensions still produce one `Tensor`.
  let dims := #[2, 2]
  let dynamic ← CLI.orThrowIO <|
    Tensor.ofArray dims.toList #[1.0, 2.0, 3.0, 4.0]
  if dynamic.toArray != #[1.0, 2.0, 3.0, 4.0] then
    throw <| IO.userError "dynamic tensor changed its row-major payload"

  -- Showing the intentional “Real tensors refuse to print” behavior.
  let xR : Tensor ℝ [4] := tensor! (ty := ℝ) [0.1, 0.2, 0.3, 0.4]
  try
    Tensor.print xR
  catch e =>
    IO.println s!"Expected failure printing Tensor ℝ: {e}"

end NN.Examples.Quickstart.TensorBasics

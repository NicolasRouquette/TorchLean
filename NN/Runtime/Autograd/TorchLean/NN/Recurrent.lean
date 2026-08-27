/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.NN.Seq

/-!
# TorchLean NN: Linear and Recurrent Layers
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor
open Proofs.Autograd.Algebra

namespace NN

/-! ## Convenience constructors (layers) -/

namespace Internal

/-- Write one leading-axis slice through the general `scatterAdd` operation. -/
def writeLeading {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {rows : Nat} {tail : Shape}
    (base : Ref (m := m) (α := α) (tail.prependDim rows))
    (value : Ref (m := m) (α := α) tail) (index : Fin rows) :
    m (Ref (m := m) (α := α) (tail.prependDim rows)) := do
  let source ← TorchLean.reshape (m := m) (α := α)
    (s₁ := tail) (s₂ := tail.prependDim 1) value (by simp [Shape.size])
  let indices : Tensor (Fin rows) [1] := Tensor.ofFn fun _ => index
  TorchLean.scatterAdd (m := m) (α := α) (s := tail.prependDim rows) 0 1
    base source (_root_.Runtime.Autograd.Torch.dataConst (m := m) (α := α) indices)

end Internal

/--
Fully-connected affine layer on vectors: $y=Wx+b$.

Parameters:
- `W : (outDim × inDim)` initialized with Xavier initialization,
- `b : (outDim)` initialized to zeros.

PyTorch analogy: `torch.nn.Linear(inDim, outDim)`.
-/
def linear (inDim outDim : Nat) (seedW seedB : Nat := 0) :
    Layer ([inDim]) ([outDim]) :=
  let WShape : Shape := [outDim, inDim]
  let bShape : Shape := [outDim]
  let w0 : Tensor Float WShape := Torch.Init.xavierW (outDim := outDim) (inDim := inDim) (seed :=
    seedW)
  let b0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed := seedB)
  { kind := s!"Linear({inDim}, {outDim})"
    stateShapes := [WShape, bShape]
    initState := .cons w0 (.cons b0 .nil)
    runtimeInit := some (.cons (.xavierUniform inDim outDim seedW) (.cons .zeros .nil))
    requiresGrad := #[true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun w b x =>
          _root_.Runtime.Autograd.Torch.linear
            (m := m) (α := α) (inDim := inDim) (outDim := outDim) w b x
  }

/--
Vanilla RNN layer (time-major sequence, no batch axis).

Semantics:
$$
h_t=\tanh\!\left(W[x_t;h_{t-1}]+b\right),
\qquad h_{-1}=0.
$$

This is implemented by unrolling a fixed number of steps (`seqLen`) using existing TorchLean ops,
so it works on both CPU and CUDA backends.

PyTorch analogy: `torch.nn.RNN(inputSize, hiddenSize, nonlinearity="tanh")` with
`batch_first=false`, specialized to a single batch element.
Docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html
-/
def rnn (seqLen inputSize hiddenSize : Nat) (seedW seedB : Nat := 0) :
    Layer ([seqLen, inputSize]) ([seqLen, hiddenSize]) :=
  let WShape : Shape := [hiddenSize, inputSize + hiddenSize]
  let bShape : Shape := [hiddenSize]
  let w0 : Tensor Float WShape := Torch.Init.xavierW (outDim := hiddenSize) (inDim := inputSize +
    hiddenSize) (seed := seedW)
  let b0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed := seedB)
  { kind := s!"RNN({inputSize}, {hiddenSize})"
    stateShapes := [WShape, bShape]
    initState := .cons w0 (.cons b0 .nil)
    runtimeInit := some (.cons (.xavierUniform (inputSize + hiddenSize) hiddenSize seedW)
      (.cons .zeros .nil))
    requiresGrad := #[true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun w b xs => show m (Ref ([seqLen, hiddenSize])) from do
          let h0T : Tensor α [hiddenSize] :=
            Spec.fill (α := α) (0 : α) ([hiddenSize])
          let out0T : Tensor α [seqLen, hiddenSize] :=
            Spec.fill (α := α) (0 : α) ([seqLen, hiddenSize])
          let h0 ← TorchLean.const (m := m) (α := α) (s := [hiddenSize]) h0T
          let out0 ← TorchLean.const (m := m) (α := α) (s := [seqLen, hiddenSize])
            out0T
          let (_, out) ← (List.finRange seqLen).foldlM (init := (h0, out0)) (fun st t => do
            let (hPrev, outPrev) := st
            let x_t ← TorchLean.select (m := m) (α := α)
              (s := [seqLen, inputSize]) 0 xs t
            let concat ← TorchLean.concatLeadingAxis (m := m) (α := α) (s := .scalar)
              (nDim := inputSize) (mDim := hiddenSize) x_t hPrev
            let pre ← _root_.Runtime.Autograd.Torch.linear (m := m) (α := α)
              (inDim := inputSize + hiddenSize) (outDim := hiddenSize)
              w b concat
            let h_t ← TorchLean.tanh (m := m) (α := α) (s := [hiddenSize]) pre
            let outNext ← Internal.writeLeading (m := m) (α := α) outPrev h_t t
            pure (h_t, outNext))
          pure out
  }

/--
GRU layer (time-major sequence, no batch axis).

This is an unrolled, time-major Cho-style GRU with $h_{-1}=0$. The reset gate is applied before the
candidate's hidden-state linear map. That differs from PyTorch's reset-after parameterization, so
PyTorch GRU weights require an explicit conversion rather than direct loading.
-/
def gru (seqLen inputSize hiddenSize : Nat) (seedW seedB : Nat := 0) :
    Layer ([seqLen, inputSize]) ([seqLen, hiddenSize]) :=
  let WShape : Shape := [hiddenSize, inputSize + hiddenSize]
  let bShape : Shape := [hiddenSize]
  let wReset0 : Tensor Float WShape := Torch.Init.xavierW (outDim := hiddenSize) (inDim := inputSize +
    hiddenSize) (seed := seedW + 0)
  let bReset0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed :=
    seedB + 0)
  let wUpdate0 : Tensor Float WShape := Torch.Init.xavierW (outDim := hiddenSize) (inDim := inputSize +
    hiddenSize) (seed := seedW + 1)
  let bUpdate0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed :=
    seedB + 1)
  let wNew0 : Tensor Float WShape := Torch.Init.xavierW (outDim := hiddenSize) (inDim := inputSize +
    hiddenSize) (seed := seedW + 2)
  let bNew0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed :=
    seedB + 2)
  { kind := s!"ChoGRU({inputSize}, {hiddenSize})"
    stateShapes := [WShape, bShape, WShape, bShape, WShape, bShape]
    initState := .cons wReset0 (.cons bReset0 (.cons wUpdate0 (.cons bUpdate0 (.cons wNew0 (.cons bNew0 .nil)))))
    runtimeInit := some <| .cons (.xavierUniform (inputSize + hiddenSize) hiddenSize (seedW + 0)) <|
      .cons .zeros <| .cons (.xavierUniform (inputSize + hiddenSize) hiddenSize (seedW + 1)) <|
      .cons .zeros <| .cons (.xavierUniform (inputSize + hiddenSize) hiddenSize (seedW + 2)) <|
      .cons .zeros .nil
    requiresGrad := #[true, true, true, true, true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun wReset bReset wUpdate bUpdate wNew bNew xs =>
          show m (Ref ([seqLen, hiddenSize])) from do
          let h0T : Tensor α [hiddenSize] :=
            Spec.fill (α := α) (0 : α) ([hiddenSize])
          let out0T : Tensor α [seqLen, hiddenSize] :=
            Spec.fill (α := α) (0 : α) ([seqLen, hiddenSize])
          let onesT : Tensor α [hiddenSize] :=
            Spec.fill (α := α) (1 : α) ([hiddenSize])
          let h0 ← TorchLean.const (m := m) (α := α) (s := [hiddenSize]) h0T
          let out0 ← TorchLean.const (m := m) (α := α) (s := [seqLen, hiddenSize])
            out0T
          let ones ← TorchLean.const (m := m) (α := α) (s := [hiddenSize]) onesT
          let (_, out) ← (List.finRange seqLen).foldlM (init := (h0, out0)) (fun st t => do
            let (hPrev, outPrev) := st
            let x_t ← TorchLean.select (m := m) (α := α)
              (s := [seqLen, inputSize]) 0 xs t
            let concat ← TorchLean.concatLeadingAxis (m := m) (α := α) (s := .scalar)
              (nDim := inputSize) (mDim := hiddenSize) x_t hPrev
            let r_pre ← _root_.Runtime.Autograd.Torch.linear (m := m) (α := α)
              (inDim := inputSize + hiddenSize) (outDim := hiddenSize)
              wReset bReset concat
            let r ← TorchLean.sigmoid (m := m) (α := α) (s := [hiddenSize]) r_pre
            let z_pre ← _root_.Runtime.Autograd.Torch.linear (m := m) (α := α)
              (inDim := inputSize + hiddenSize) (outDim := hiddenSize)
              wUpdate bUpdate concat
            let z ← TorchLean.sigmoid (m := m) (α := α) (s := [hiddenSize]) z_pre
            let r_hPrev ← TorchLean.mul (m := m) (α := α) (s := [hiddenSize]) r hPrev
            let concat2 ← TorchLean.concatLeadingAxis (m := m) (α := α) (s := .scalar)
              (nDim := inputSize) (mDim := hiddenSize) x_t r_hPrev
            let n_pre ← _root_.Runtime.Autograd.Torch.linear (m := m) (α := α)
              (inDim := inputSize + hiddenSize) (outDim := hiddenSize)
              wNew bNew concat2
            let n ← TorchLean.tanh (m := m) (α := α) (s := [hiddenSize]) n_pre
            let oneMinusZ ← TorchLean.sub (m := m) (α := α) (s := [hiddenSize]) ones z
            let newContrib ← TorchLean.mul (m := m) (α := α) (s := [hiddenSize]) oneMinusZ n
            let hiddenContrib ← TorchLean.mul (m := m) (α := α) (s := [hiddenSize]) z hPrev
            let h_t ← TorchLean.add (m := m) (α := α) (s := [hiddenSize]) newContrib hiddenContrib
            let outNext ← Internal.writeLeading (m := m) (α := α) outPrev h_t t
            pure (h_t, outNext))
          pure out
  }

/--
Mamba-style gated diagonal state-space layer (time-major sequence, no batch axis).

This is the trainable recurrent core used by the runnable Mamba text example.  At each time step it
learns an input candidate, a token/state-dependent retention gate, and an output gate:

$$
\begin{aligned}
u_t &= \operatorname{SiLU}(W_u x_t+b_u),\\
\delta_t &= \operatorname{sigmoid}\!\left(W_\delta[x_t;h_{t-1}]+b_\delta\right),\\
h_t &= \delta_t\odot h_{t-1}+(1-\delta_t)\odot u_t,\\
y_t &= h_t\odot\operatorname{SiLU}(W_zx_t+b_z).
\end{aligned}
$$

The recurrence is unrolled with ordinary TorchLean differentiable ops, so the same definition trains
on the CPU backend and on the CUDA backend.  The lower-level selective-scan CUDA kernels are still
available for forward experiments, but this layer is built from autograd-covered ops so
all projections and gates train correctly.
-/
def mamba (seqLen inputSize hiddenSize : Nat) (seedW seedB : Nat := 0) :
    Layer ([seqLen, inputSize]) ([seqLen, hiddenSize]) :=
  let WInShape : Shape := [hiddenSize, inputSize]
  let WDeltaShape : Shape := [hiddenSize, inputSize + hiddenSize]
  let bShape : Shape := [hiddenSize]
  let wIn0 : Tensor Float WInShape := Torch.Init.xavierW
    (outDim := hiddenSize) (inDim := inputSize) (seed := seedW + 0)
  let bIn0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros)
    (seed := seedB + 0)
  let wDelta0 : Tensor Float WDeltaShape := Torch.Init.xavierW
    (outDim := hiddenSize) (inDim := inputSize + hiddenSize) (seed := seedW + 1)
  let bDelta0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros)
    (seed := seedB + 1)
  let wGate0 : Tensor Float WInShape := Torch.Init.xavierW
    (outDim := hiddenSize) (inDim := inputSize) (seed := seedW + 2)
  let bGate0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros)
    (seed := seedB + 2)
  { kind := s!"Mamba({inputSize}, {hiddenSize})"
    stateShapes := [WInShape, bShape, WDeltaShape, bShape, WInShape, bShape]
    initState := .cons wIn0 (.cons bIn0 (.cons wDelta0 (.cons bDelta0
      (.cons wGate0 (.cons bGate0 .nil)))))
    runtimeInit := some <| .cons (.xavierUniform inputSize hiddenSize (seedW + 0)) <|
      .cons .zeros <| .cons (.xavierUniform (inputSize + hiddenSize) hiddenSize (seedW + 1)) <|
      .cons .zeros <| .cons (.xavierUniform inputSize hiddenSize (seedW + 2)) <|
      .cons .zeros .nil
    requiresGrad := #[true, true, true, true, true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun wIn bIn wDelta bDelta wGate bGate xs =>
          show m (Ref ([seqLen, hiddenSize])) from do
          let h0T : Tensor α [hiddenSize] :=
            Spec.fill (α := α) (0 : α) ([hiddenSize])
          let out0T : Tensor α [seqLen, hiddenSize] :=
            Spec.fill (α := α) (0 : α) ([seqLen, hiddenSize])
          let onesT : Tensor α [hiddenSize] :=
            Spec.fill (α := α) (1 : α) ([hiddenSize])
          let h0 ← TorchLean.const (m := m) (α := α) (s := [hiddenSize]) h0T
          let out0 ← TorchLean.const (m := m) (α := α) (s := [seqLen, hiddenSize])
            out0T
          let ones ← TorchLean.const (m := m) (α := α) (s := [hiddenSize]) onesT
          let (_, out) ← (List.finRange seqLen).foldlM (init := (h0, out0)) (fun st t => do
            let (hPrev, outPrev) := st
            let x_t ← TorchLean.select (m := m) (α := α)
              (s := [seqLen, inputSize]) 0 xs t
            let uPre ← _root_.Runtime.Autograd.Torch.linear (m := m) (α := α)
              (inDim := inputSize) (outDim := hiddenSize) wIn bIn x_t
            let u ← _root_.Runtime.Autograd.Torch.silu
              (m := m) (α := α) (s := [hiddenSize]) uPre
            let concat ← TorchLean.concatLeadingAxis (m := m) (α := α) (s := .scalar)
              (nDim := inputSize) (mDim := hiddenSize) x_t hPrev
            let deltaPre ← _root_.Runtime.Autograd.Torch.linear (m := m) (α := α)
              (inDim := inputSize + hiddenSize) (outDim := hiddenSize)
              wDelta bDelta concat
            let delta ← TorchLean.sigmoid (m := m) (α := α) (s := [hiddenSize]) deltaPre
            let oneMinusDelta ← TorchLean.sub (m := m) (α := α) (s := [hiddenSize])
              ones delta
            let keep ← TorchLean.mul (m := m) (α := α) (s := [hiddenSize])
              delta hPrev
            let write ← TorchLean.mul (m := m) (α := α) (s := [hiddenSize])
              oneMinusDelta u
            let h_t ← TorchLean.add (m := m) (α := α) (s := [hiddenSize])
              keep write
            let gatePre ← _root_.Runtime.Autograd.Torch.linear (m := m) (α := α)
              (inDim := inputSize) (outDim := hiddenSize) wGate bGate x_t
            let gate ← _root_.Runtime.Autograd.Torch.silu
              (m := m) (α := α) (s := [hiddenSize]) gatePre
            let y_t ← TorchLean.mul (m := m) (α := α) (s := [hiddenSize])
              h_t gate
            let outNext ← Internal.writeLeading (m := m) (α := α) outPrev y_t t
            pure (h_t, outNext))
          pure out
  }

/--
LSTM layer (time-major sequence, no batch axis).

This is an unrolled LSTM using the standard four gates, with
$(h_{-1},c_{-1})=(0,0)$.

PyTorch analogy: `torch.nn.LSTM(inputSize, hiddenSize)` with `batch_first=false`, specialized to a
single batch element.
Docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html
-/
def lstm (seqLen inputSize hiddenSize : Nat) (seedW seedB : Nat := 0) :
    Layer ([seqLen, inputSize]) ([seqLen, hiddenSize]) :=
  let WShape : Shape := [hiddenSize, inputSize + hiddenSize]
  let bShape : Shape := [hiddenSize]
  let wF0 : Tensor Float WShape := Torch.Init.xavierW (outDim := hiddenSize) (inDim := inputSize +
    hiddenSize) (seed := seedW + 0)
  let bF0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed := seedB + 0)
  let wI0 : Tensor Float WShape := Torch.Init.xavierW (outDim := hiddenSize) (inDim := inputSize +
    hiddenSize) (seed := seedW + 1)
  let bI0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed := seedB + 1)
  let wC0 : Tensor Float WShape := Torch.Init.xavierW (outDim := hiddenSize) (inDim := inputSize +
    hiddenSize) (seed := seedW + 2)
  let bC0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed := seedB + 2)
  let wO0 : Tensor Float WShape := Torch.Init.xavierW (outDim := hiddenSize) (inDim := inputSize +
    hiddenSize) (seed := seedW + 3)
  let bO0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed := seedB + 3)
  { kind := s!"LSTM({inputSize}, {hiddenSize})"
    stateShapes := [WShape, bShape, WShape, bShape, WShape, bShape, WShape, bShape]
    initState :=
      .cons wF0 (.cons bF0 (.cons wI0 (.cons bI0 (.cons wC0 (.cons bC0 (.cons wO0 (.cons bO0 .nil)))))))
    runtimeInit := some <| .cons (.xavierUniform (inputSize + hiddenSize) hiddenSize (seedW + 0)) <|
      .cons .zeros <| .cons (.xavierUniform (inputSize + hiddenSize) hiddenSize (seedW + 1)) <|
      .cons .zeros <| .cons (.xavierUniform (inputSize + hiddenSize) hiddenSize (seedW + 2)) <|
      .cons .zeros <| .cons (.xavierUniform (inputSize + hiddenSize) hiddenSize (seedW + 3)) <|
      .cons .zeros .nil
    requiresGrad := #[true, true, true, true, true, true, true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun wF bF wI bI wC bC wO bO xs =>
          show m (Ref ([seqLen, hiddenSize])) from do
          let h0T : Tensor α [hiddenSize] :=
            Spec.fill (α := α) (0 : α) ([hiddenSize])
          let out0T : Tensor α [seqLen, hiddenSize] :=
            Spec.fill (α := α) (0 : α) ([seqLen, hiddenSize])
          let h0 ← TorchLean.const (m := m) (α := α) (s := [hiddenSize]) h0T
          let c0 ← TorchLean.const (m := m) (α := α) (s := [hiddenSize]) h0T
          let out0 ← TorchLean.const (m := m) (α := α) (s := [seqLen, hiddenSize])
            out0T
          let (_, _, out) ← (List.finRange seqLen).foldlM (init := (h0, c0, out0)) (fun st t => do
            let (hPrev, cPrev, outPrev) := st
            let x_t ← TorchLean.select (m := m) (α := α)
              (s := [seqLen, inputSize]) 0 xs t
            let concat ← TorchLean.concatLeadingAxis (m := m) (α := α) (s := .scalar)
              (nDim := inputSize) (mDim := hiddenSize) x_t hPrev
            let f_pre ← _root_.Runtime.Autograd.Torch.linear (m := m) (α := α)
              (inDim := inputSize + hiddenSize) (outDim := hiddenSize)
              wF bF concat
            let f ← TorchLean.sigmoid (m := m) (α := α) (s := [hiddenSize]) f_pre
            let i_pre ← _root_.Runtime.Autograd.Torch.linear (m := m) (α := α)
              (inDim := inputSize + hiddenSize) (outDim := hiddenSize)
              wI bI concat
            let i ← TorchLean.sigmoid (m := m) (α := α) (s := [hiddenSize]) i_pre
            let g_pre ← _root_.Runtime.Autograd.Torch.linear (m := m) (α := α)
              (inDim := inputSize + hiddenSize) (outDim := hiddenSize)
              wC bC concat
            let g ← TorchLean.tanh (m := m) (α := α) (s := [hiddenSize]) g_pre
            let o_pre ← _root_.Runtime.Autograd.Torch.linear (m := m) (α := α)
              (inDim := inputSize + hiddenSize) (outDim := hiddenSize)
              wO bO concat
            let o ← TorchLean.sigmoid (m := m) (α := α) (s := [hiddenSize]) o_pre
            let fc ← TorchLean.mul (m := m) (α := α) (s := [hiddenSize]) f cPrev
            let ig ← TorchLean.mul (m := m) (α := α) (s := [hiddenSize]) i g
            let c_t ← TorchLean.add (m := m) (α := α) (s := [hiddenSize]) fc ig
            let tanhC ← TorchLean.tanh (m := m) (α := α) (s := [hiddenSize]) c_t
            let h_t ← TorchLean.mul (m := m) (α := α) (s := [hiddenSize]) o tanhC
            let outNext ← Internal.writeLeading (m := m) (α := α) outPrev h_t t
            pure (h_t, c_t, outNext))
          pure out
  }
end NN

end TorchLean
end Autograd
end Runtime

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Linalg
public import NN.Spec.Dynamics.StateSpace
public import NN.Spec.Layers.Activation

/-!
# Mamba-style selective state-space blocks

Mamba replaces quadratic attention with a linear-time selective state-space recurrence.  In full
models, the token controls discretization and input/output state parameters.

This file exposes two layers:

- `MambaBlockSpec`: a compact theorem-friendly diagonal SSM block for scan laws and kernel
  validation.
- `SelectiveMambaBlockSpec`: a fuller Mamba-style block with input/gate projections, causal
  depthwise convolution, SiLU, token-dependent `Delta/B/C`, diagonal selective scan, gated output,
  and output projection.

The compact block is intentionally retained: it is the smallest reusable core for proving scan
algebra and for validating CUDA kernels.  The full block builds the paper-style Mamba dataflow on
top of the same affine-scan idea.

- recurrent selective scan (`h ← A ⊙ h + B ⊙ x_state`),
- a gated state readout,
- tokenwise input/output projections.

References:
- Gu, Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", COLM 2024.
- Dao, Gu. "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured
  State Space Duality" (Mamba-2), ICML 2024.
-/

@[expose] public section

namespace Models

open Spec
open Tensor
open NN.Spec.Dynamics

/-- Parameters for a compact diagonal Mamba-style block. -/
structure MambaBlockSpec (α : Type) (inputDim stateDim outputDim : Nat) where
  /-- Input projection into SSM state channels. -/
  inProj : Tensor α [inputDim, stateDim]
  /-- Gate projection. The gate is `sigmoid(x @ gateProj)`. -/
  gateProj : Tensor α [inputDim, stateDim]
  /-- Output projection from gated state channels. -/
  outProj : Tensor α [stateDim, outputDim]
  /-- Diagonal state-space core. -/
  ssm : DiagonalSSM α stateDim

namespace MambaBlockSpec

variable {α : Type} [Context α]
variable {inputDim stateDim outputDim : Nat}

/-- Input-to-state projection. -/
def projectInput (m : MambaBlockSpec α inputDim stateDim outputDim)
    (x : Tensor α [inputDim]) : Tensor α [stateDim] :=
  vecMatMulSpec x m.inProj

/-- Token-dependent sigmoid gate. -/
def gate (m : MambaBlockSpec α inputDim stateDim outputDim)
    (x : Tensor α [inputDim]) : Tensor α [stateDim] :=
  Tensor.mapSpec Activation.Math.sigmoidSpec (vecMatMulSpec x m.gateProj)

/-- One Mamba-style token step, returning `(new_state, output)`. -/
def step (m : MambaBlockSpec α inputDim stateDim outputDim)
    (h : Tensor α [stateDim])
    (x : Tensor α [inputDim]) :
    Tensor α [stateDim] × Tensor α [outputDim] :=
  let xState := m.projectInput x
  let h' := m.ssm.step h xState
  let yState := m.ssm.readout h' xState
  let gated := yState * m.gate x
  (h', vecMatMulSpec gated m.outProj)

/-- Run an array of tokens through the recurrent block. -/
def runArray (m : MambaBlockSpec α inputDim stateDim outputDim)
    (h0 : Tensor α [stateDim])
    (xs : Array (Tensor α [inputDim])) :
    Tensor α [stateDim] × Array (Tensor α [outputDim]) :=
  Spec.scanArray m.step h0 xs

@[simp] theorem runArray_empty (m : MambaBlockSpec α inputDim stateDim outputDim)
    (h0 : Tensor α [stateDim]) :
    m.runArray h0 #[] = (h0, #[]) := by
  rfl

/-- A Mamba recurrent pass emits one output token per input token. -/
@[simp] theorem runArray_outputs_size (m : MambaBlockSpec α inputDim stateDim outputDim)
    (h0 : Tensor α [stateDim])
    (xs : Array (Tensor α [inputDim])) :
    (m.runArray h0 xs).2.size = xs.size := by
  exact Spec.scanArray_outputs_size m.step h0 xs

end MambaBlockSpec

/-- Parameters for a fuller Mamba-style selective SSM block.

Shape conventions:
- `inputDim`: token/input feature width,
- `innerDim`: expanded channel width used by Mamba's convolution and SSM path,
- `stateDim`: per-channel diagonal SSM state size,
- `outputDim`: output feature width,
- `convWidth`: causal depthwise-convolution width.

The recurrence state has shape `[innerDim, stateDim]`.  This mirrors the common implementation
view of Mamba where each expanded channel carries a small diagonal state vector.
-/
structure SelectiveMambaBlockSpec
    (α : Type) (inputDim innerDim stateDim outputDim convWidth : Nat) where
  /-- Content/input projection `x -> x_path`. -/
  xProj : Tensor α [inputDim, innerDim]
  /-- Gate projection `x -> z_path`. -/
  zProj : Tensor α [inputDim, innerDim]
  /-- Causal depthwise-convolution kernel, indexed by `(tap, channel)`. -/
  convKernel : Tensor α [convWidth, innerDim]
  /-- Causal depthwise-convolution bias. -/
  convBias : Tensor α [innerDim]
  /-- Projection from activated convolution features to per-channel time steps `Delta`. -/
  dtProj : Tensor α [innerDim, innerDim]
  /-- Bias before the `softplus` time-step nonlinearity. -/
  dtBias : Tensor α [innerDim]
  /-- Positive diagonal state rates `A[d,n]` used as `exp(-Delta[d] * A[d,n])`. -/
  A : Tensor α [innerDim, stateDim]
  /-- Token-dependent input-state projection `B_t = u_t @ bProj`. -/
  bProj : Tensor α [innerDim, stateDim]
  /-- Token-dependent state-output projection `C_t = u_t @ cProj`. -/
  cProj : Tensor α [innerDim, stateDim]
  /-- Per-channel residual/skip coefficient. -/
  dSkip : Tensor α [innerDim]
  /-- Output projection from expanded channels to output features. -/
  outProj : Tensor α [innerDim, outputDim]

namespace SelectiveMambaBlockSpec

variable {α : Type} [Context α]
variable {inputDim innerDim stateDim outputDim convWidth : Nat}

/-- Projection feeding the content path before convolution and selective state updates. -/
def projectX (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (x : Tensor α [inputDim]) : Tensor α [innerDim] :=
  vecMatMulSpec x m.xProj

/-- Projection feeding the multiplicative gate path in the selective state-space block. -/
def projectZ (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (x : Tensor α [inputDim]) : Tensor α [innerDim] :=
  vecMatMulSpec x m.zProj

/-- SiLU/Swish applied channelwise. -/
def siluVec (x : Tensor α [innerDim]) : Tensor α [innerDim] :=
  Tensor.mapSpec Activation.Math.swishSpec x

/--
Causal depthwise convolution from a newest-first history of projected tokens.

`history[0]` is the current projected token, `history[1]` is the previous token, etc.  Missing
history entries are treated as zero padding.
-/
def causalDepthwiseConv
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (history : Array (Tensor α [innerDim])) :
    Tensor α [innerDim] :=
  Tensor.dim (fun c : Fin innerDim =>
    Tensor.scalar <|
      (List.finRange convWidth).foldl
        (fun acc tap =>
          let zeroInner : Tensor α [innerDim] :=
            Tensor.dim (fun _ => Tensor.scalar 0)
          let xTap : α := Tensor.getScalar (history[tap.val]?.getD zeroInner) c
          acc + xTap * get2 m.convKernel tap c)
        (Tensor.getScalar m.convBias c))

/-- Token-dependent positive time steps `Delta = softplus(u @ dtProj + dtBias)`. -/
def delta
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (u : Tensor α [innerDim]) : Tensor α [innerDim] :=
  Tensor.mapSpec Activation.Math.softplusSpec (vecMatMulSpec u m.dtProj + m.dtBias)

/-- Token-dependent input-state vector `B_t`. -/
def bToken
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (u : Tensor α [innerDim]) : Tensor α [stateDim] :=
  vecMatMulSpec u m.bProj

/-- Token-dependent state-output vector `C_t`. -/
def cToken
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (u : Tensor α [innerDim]) : Tensor α [stateDim] :=
  vecMatMulSpec u m.cProj

/--
One selective diagonal SSM update:

`h'[d,n] = exp(-Delta[d] * A[d,n]) * h[d,n] + (Delta[d] * B_t[n]) * u[d]`.
-/
def selectiveStateStep
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (h : Tensor α [innerDim, stateDim])
    (u : Tensor α [innerDim]) :
    Tensor α [innerDim, stateDim] :=
  let Δ := m.delta u
  let B := m.bToken u
  Tensor.dim (fun d : Fin innerDim =>
    Tensor.dim (fun n : Fin stateDim =>
      let deltaD := Tensor.getScalar Δ d
      let aBar := MathFunctions.exp (-(deltaD * get2 m.A d n))
      let bBar := deltaD * Tensor.getScalar B n
      Tensor.scalar (aBar * get2 h d n + bBar * Tensor.getScalar u d)))

/-- Read out expanded channels from the updated state using `C_t`, plus the Mamba skip path. -/
def stateReadout
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (h : Tensor α [innerDim, stateDim])
    (u : Tensor α [innerDim]) :
    Tensor α [innerDim] :=
  let C := m.cToken u
  Tensor.dim (fun d : Fin innerDim =>
    Tensor.scalar <|
      (List.finRange stateDim).foldl
        (fun acc n => acc + get2 h d n * Tensor.getScalar C n)
        (Tensor.getScalar m.dSkip d * Tensor.getScalar u d))

/--
One full Mamba token step from an already-updated convolution history.

The `history` argument is newest-first and must include the current projected content token.
-/
def stepWithHistory
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (h : Tensor α [innerDim, stateDim])
    (history : Array (Tensor α [innerDim]))
    (z : Tensor α [innerDim]) :
    Tensor α [innerDim, stateDim] × Tensor α [outputDim] :=
  let u := siluVec (m.causalDepthwiseConv history)
  let h' := m.selectiveStateStep h u
  let y := m.stateReadout h' u
  let gated := y * siluVec z
  (h', vecMatMulSpec gated m.outProj)

/-- One recurrent step while carrying the newest-first convolution history. -/
def stepWithConvolutionHistory
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (state : Tensor α [innerDim, stateDim] ×
      Array (Tensor α [innerDim]))
    (x : Tensor α [inputDim]) :
    (Tensor α [innerDim, stateDim] ×
      Array (Tensor α [innerDim])) ×
      Tensor α [outputDim] :=
  let xPath := m.projectX x
  let zPath := m.projectZ x
  let history := #[xPath] ++ state.2
  let (nextHidden, output) := m.stepWithHistory state.1 history zPath
  ((nextHidden, history), output)

/-- Recurrent runner from an existing state and newest-first convolution history. -/
def runArrayWithHistory
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (h0 : Tensor α [innerDim, stateDim])
    (history : Array (Tensor α [innerDim]))
    (xs : Array (Tensor α [inputDim])) :
    Tensor α [innerDim, stateDim] ×
      Array (Tensor α [outputDim]) :=
  let result := Spec.scanArray m.stepWithConvolutionHistory (h0, history) xs
  (result.1.1, result.2)

/-- Run a sequence through the full selective Mamba block. -/
def runArray
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (h0 : Tensor α [innerDim, stateDim])
    (xs : Array (Tensor α [inputDim])) :
    Tensor α [innerDim, stateDim] ×
      Array (Tensor α [outputDim]) :=
  m.runArrayWithHistory h0 #[] xs

@[simp] theorem runArrayWithHistory_empty
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (h0 : Tensor α [innerDim, stateDim])
    (history : Array (Tensor α [innerDim])) :
    m.runArrayWithHistory h0 history #[] = (h0, #[]) := by
  rfl

@[simp] theorem runArray_empty
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (h0 : Tensor α [innerDim, stateDim]) :
    m.runArray h0 #[] = (h0, #[]) := by
  rfl

/-- The full Mamba recurrent pass emits one output token per input token. -/
@[simp] theorem runArrayWithHistory_outputs_size
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (h0 : Tensor α [innerDim, stateDim])
    (history : Array (Tensor α [innerDim]))
    (xs : Array (Tensor α [inputDim])) :
    (m.runArrayWithHistory h0 history xs).2.size = xs.size := by
  exact Spec.scanArray_outputs_size _ (h0, history) xs

/-- The public full Mamba runner emits one output token per input token. -/
@[simp] theorem runArray_outputs_size
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (h0 : Tensor α [innerDim, stateDim])
    (xs : Array (Tensor α [inputDim])) :
    (m.runArray h0 xs).2.size = xs.size := by
  unfold runArray
  exact m.runArrayWithHistory_outputs_size h0 #[] xs

end SelectiveMambaBlockSpec

end Models

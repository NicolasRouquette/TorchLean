/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Rnn

/-!
# LSTM (spec layer)

TorchLean provides a small LSTM specification that is:

- explicit about shapes (so common dimension mistakes are caught early),
- explicit about the gate math (so gradients are inspectable and proofs can refer to the equations),
- close in spirit to the way PyTorch documents `nn.LSTMCell` / `nn.LSTM`.

## References (math + PyTorch behavior)

- Hochreiter, Schmidhuber, "Long Short-Term Memory" (Neural Computation, 1997).
  Free PDF: http://www.bioinf.jku.at/publications/older/2604.pdf
- PyTorch `LSTMCell` equations:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html
- PyTorch `LSTM` equations: https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html

## Notes on parameterization

Many libraries expose two matrices per gate (`W_ih` and `W_hh`) and add them.
In this spec we use a single matrix applied to a concatenated vector `[x_t; h_{t-1}]`.
It's the same computation, just packaged to reuse TorchLean's tensor building blocks.
-/

@[expose] public section


namespace Spec

open Tensor
open Activation

variable {α : Type} [Context α]

/-- Parameters for an LSTM cell, with one `(hiddenSize × (inputSize + hiddenSize))` matrix per gate.

This corresponds to the usual `(W_ih, W_hh)` parameterization in libraries like PyTorch, but we
package it as a single matrix applied to `[x_t; h_{t-1}]` to reuse TorchLean's tensor building
blocks.
-/
structure LSTMSpec (α : Type) (inputSize hiddenSize : Nat) where
  /-- Forget-gate weights for `f_t = sigmoid(W_f [x_t; h_{t-1}] + b_f)`. -/
  forgetWeight : Tensor α [hiddenSize, inputSize + hiddenSize]
  /-- Forget-gate bias. -/
  forgetBias : Tensor α [hiddenSize]
  /-- Input-gate weights for `i_t = sigmoid(W_i [x_t; h_{t-1}] + b_i)`. -/
  inputWeight : Tensor α [hiddenSize, inputSize + hiddenSize]
  /-- Input-gate bias. -/
  inputBias : Tensor α [hiddenSize]
  /-- Candidate/cell-proposal weights for `g_t = tanh(W_g [x_t; h_{t-1}] + b_g)`. -/
  candidateWeight : Tensor α [hiddenSize, inputSize + hiddenSize]
  /-- Candidate/cell-proposal bias. -/
  candidateBias : Tensor α [hiddenSize]
  /-- Output-gate weights for `o_t = sigmoid(W_o [x_t; h_{t-1}] + b_o)`. -/
  outputWeight : Tensor α [hiddenSize, inputSize + hiddenSize]
  /-- Output-gate bias. -/
  outputBias : Tensor α [hiddenSize]

/-- LSTM recurrent state: hidden vector `h_t` and cell vector `c_t`. -/
structure LSTMState (α : Type) (hiddenSize : Nat) where
  /-- Exposed hidden state `h_t`. -/
  hidden : Tensor α [hiddenSize]  -- h_t
  /-- Internal memory/cell state `c_t`. -/
  cell   : Tensor α [hiddenSize]  -- c_t

/-- One LSTM cell step: update `(h_{t-1}, c_{t-1})` given `x_t` and parameters. -/
def lstmCellSpec {inputSize hiddenSize : Nat}
  (lstm : LSTMSpec α inputSize hiddenSize)
  (input : Tensor α [inputSize])
  (prev_state : LSTMState α hiddenSize) :
  LSTMState α hiddenSize :=
  -- We follow the standard LSTM equations (same layout as in the PyTorch docs):
  --
  --   f_t = sigmoid(W_f [x_t; h_{t-1}] + b_f)      (forget gate)
  --   i_t = sigmoid(W_i [x_t; h_{t-1}] + b_i)      (input gate)
  --   g_t = tanh   (W_g [x_t; h_{t-1}] + b_g)      (candidate / cell proposal)
  --   o_t = sigmoid(W_o [x_t; h_{t-1}] + b_o)      (output gate)
  --   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t              (cell state update)
  --   h_t = o_t ⊙ tanh(c_t)                        (exposed hidden state)
  --
  -- The `cell` component is what lets information persist over long ranges.
  let concat := concatAxisSpec .scalar input prev_state.hidden

  -- Forget gate: f_t = σ(W_f @ [x_t; h_{t-1}] + b_f)
  let forget_gate := sigmoidSpec (addSpec (matVecMulSpec lstm.forgetWeight concat)
    lstm.forgetBias)

  -- Input gate: i_t = σ(W_i @ [x_t; h_{t-1}] + b_i)
  let input_gate := sigmoidSpec (addSpec (matVecMulSpec lstm.inputWeight concat)
    lstm.inputBias)

  -- Candidate values: ĉ_t = tanh(W_c @ [x_t; h_{t-1}] + b_c)
  let candidate := tanhSpec (addSpec (matVecMulSpec lstm.candidateWeight concat)
    lstm.candidateBias)

  -- Output gate: o_t = σ(W_o @ [x_t; h_{t-1}] + b_o)
  let output_gate := sigmoidSpec (addSpec (matVecMulSpec lstm.outputWeight concat)
    lstm.outputBias)

  -- Cell state: c_t = f_t ⊙ c_{t-1} + i_t ⊙ ĉ_t
  let new_cell := addSpec (mulSpec forget_gate prev_state.cell) (mulSpec input_gate candidate)

  -- Hidden state: h_t = o_t ⊙ tanh(c_t)
  let new_hidden := mulSpec output_gate (tanhSpec new_cell)

  ⟨new_hidden, new_cell⟩

/-- Run an LSTM cell over a length-`seqLen` input sequence, returning outputs and final state. -/
def lstmSequenceSpec {seqLen inputSize hiddenSize : Nat}
  (lstm : LSTMSpec α inputSize hiddenSize)
  (inputs : Tensor α [seqLen, inputSize])
  (initial_state : LSTMState α hiddenSize) :
  (Tensor α [seqLen, hiddenSize] × LSTMState α hiddenSize) :=
  let (finalState, outputs) := Sequence.mapAccum seqLen initial_state fun i previous =>
    let state := lstmCellSpec lstm (get inputs i) previous
    (state, state.hidden)
  (Tensor.dim outputs.getScalar, finalState)

/-- Batched wrapper around `lstmSequenceSpec` (runs one sequence per batch element). -/
def lstmBatchedSpec {batchSize seqLen inputSize hiddenSize : Nat}
  (lstm : LSTMSpec α inputSize hiddenSize)
  (inputs : Tensor α [batchSize, seqLen, inputSize])
  (initial_hiddens : Tensor α [batchSize, hiddenSize]) :
  (Tensor α [batchSize, seqLen, hiddenSize] × Tensor α [batchSize, hiddenSize]) :=
  match inputs, initial_hiddens with
  | .dim batch_inputs, .dim batch_hidden =>
    -- In PyTorch you typically pass both `h_0` and `c_0`. Here we take only `h_0` and set `c_0 =
    -- 0`.
    let batch_cell := Tensor.dim (fun _ => fill 0 (.dim hiddenSize .scalar))

    -- compute per-batch results
    let outputs := Tensor.dim (fun b =>
      let initial_state : LSTMState α hiddenSize :=
        { hidden := batch_hidden b, cell := get batch_cell b }
      (lstmSequenceSpec lstm (batch_inputs b) initial_state).1)

    let final_hiddens := Tensor.dim (fun b =>
      let initial_state : LSTMState α hiddenSize :=
        { hidden := batch_hidden b, cell := get batch_cell b }
      (lstmSequenceSpec lstm (batch_inputs b) initial_state).2.hidden)

    (outputs, final_hiddens)

-- ============================================================================
-- Backpropagation (BPTT)
-- ============================================================================

/--
Forward pass for one LSTM cell that also returns the gate activations.

This is the spec analogue of the "saved tensors" that a runtime will keep for backward.
-/
def lstmCellSpecWithIntermediates {inputSize hiddenSize : Nat}
  (lstm : LSTMSpec α inputSize hiddenSize)
  (input : Tensor α [inputSize])
  (prev_state : LSTMState α hiddenSize) :
  (LSTMState α hiddenSize ×                -- new state (h_t, c_t)
   Tensor α [hiddenSize] ×             -- forget gate f_t
   Tensor α [hiddenSize] ×             -- input gate i_t
   Tensor α [hiddenSize] ×             -- candidate g_t
   Tensor α [hiddenSize]) :=           -- output gate o_t
  let concat := concatAxisSpec .scalar input prev_state.hidden
  let f := sigmoidSpec (addSpec (matVecMulSpec lstm.forgetWeight concat) lstm.forgetBias)
  let i := sigmoidSpec (addSpec (matVecMulSpec lstm.inputWeight concat) lstm.inputBias)
  let g := tanhSpec (addSpec (matVecMulSpec lstm.candidateWeight concat) lstm.candidateBias)
  let o := sigmoidSpec (addSpec (matVecMulSpec lstm.outputWeight concat) lstm.outputBias)
  let c := addSpec (mulSpec f prev_state.cell) (mulSpec i g)
  let h := mulSpec o (tanhSpec c)
  (⟨h, c⟩, f, i, g, o)

-- Single LSTM cell backward pass.
-- Returns:
--   dX_t, dPrevState, (dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo)
/--
Backward pass (VJP) for a single LSTM cell.

Inputs:
- parameters `lstm`,
- inputs `x_t`, previous state `(h_{t-1}, c_{t-1})`, and current state `(h_t, c_t)`,
- the gate activations from the forward pass,
- upstream gradients for both `h_t` and `c_t`.

Outputs:
- gradients w.r.t. `x_t` and the previous state,
- plus gradients for each parameter tensor.

This is the quantity computed by PyTorch autograd for an `nn.LSTMCell` unrolled in time.
-/
def lstmCellBackwardSpec {inputSize hiddenSize : Nat}
  (lstm : LSTMSpec α inputSize hiddenSize)
  (input : Tensor α [inputSize])
  (prev_state : LSTMState α hiddenSize)
  (state : LSTMState α hiddenSize)
  (forget_gate : Tensor α [hiddenSize])
  (input_gate : Tensor α [hiddenSize])
  (candidate : Tensor α [hiddenSize])
  (output_gate : Tensor α [hiddenSize])
  (grad_hidden : Tensor α [hiddenSize])
  (grad_cell : Tensor α [hiddenSize]) :
  ( Tensor α [inputSize] × LSTMState α hiddenSize ×
    Tensor α [hiddenSize, inputSize + hiddenSize] × Tensor α [hiddenSize] ×
    Tensor α [hiddenSize, inputSize + hiddenSize] × Tensor α [hiddenSize] ×
    Tensor α [hiddenSize, inputSize + hiddenSize] × Tensor α [hiddenSize] ×
    Tensor α [hiddenSize, inputSize + hiddenSize] × Tensor α [hiddenSize] ) :=

  let concat := concatAxisSpec .scalar input prev_state.hidden

  let tanh_c := tanhSpec state.cell
  let tanh_c_deriv := subSpec (fill 1 (.dim hiddenSize .scalar)) (mulSpec tanh_c tanh_c)

  -- h = o ⊙ tanh(c)
  let dO := mulSpec grad_hidden tanh_c
  let dC_from_h := mulSpec (mulSpec grad_hidden output_gate) tanh_c_deriv
  let dC := addSpec grad_cell dC_from_h

  -- c = f ⊙ c_prev + i ⊙ g
  let dF := mulSpec dC prev_state.cell
  let dI := mulSpec dC candidate
  let dG := mulSpec dC input_gate
  let dC_prev := mulSpec dC forget_gate

  -- preactivation gradients
  let dF_pre := mulSpec dF (Activation.sigmoidOutputDerivSpec forget_gate)
  let dI_pre := mulSpec dI (Activation.sigmoidOutputDerivSpec input_gate)
  let dO_pre := mulSpec dO (Activation.sigmoidOutputDerivSpec output_gate)
  let dG_pre :=
    let tanh_deriv := subSpec (fill 1 (.dim hiddenSize .scalar)) (mulSpec candidate candidate)
    mulSpec dG tanh_deriv

  let dWf := outerProductSpec dF_pre concat
  let dbf := dF_pre
  let dWi := outerProductSpec dI_pre concat
  let dbi := dI_pre
  let dWc := outerProductSpec dG_pre concat
  let dbc := dG_pre
  let dWo := outerProductSpec dO_pre concat
  let dbo := dO_pre

  let dConcat_f := vecMatMulSpec dF_pre lstm.forgetWeight
  let dConcat_i := vecMatMulSpec dI_pre lstm.inputWeight
  let dConcat_c := vecMatMulSpec dG_pre lstm.candidateWeight
  let dConcat_o := vecMatMulSpec dO_pre lstm.outputWeight
  let dConcat := addSpec (addSpec dConcat_f dConcat_i) (addSpec dConcat_c dConcat_o)

  let dInput := sliceRangeSpec dConcat 0 inputSize (by simp)
  let dPrevHidden := sliceRangeSpec dConcat inputSize hiddenSize (by simp)

  ( dInput, ⟨dPrevHidden, dC_prev⟩
  , dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo )

-- Full BPTT backward pass through an LSTM sequence.
-- Recomputes intermediate gates/states internally to avoid requiring a "tape" argument.
/--
Backprop through time (BPTT) for the whole sequence.

This function recomputes and stores the forward intermediates (gates and states) internally, then
walks time backward accumulating parameter gradients and input gradients. This matches the usual
PyTorch training structure, with the save-vs-recompute choice made explicit.
-/
def lstmSequenceBackwardSpec {seqLen inputSize hiddenSize : Nat}
  (lstm : LSTMSpec α inputSize hiddenSize)
  (inputs : Tensor α [seqLen, inputSize])
  (initial_state : LSTMState α hiddenSize)
  (grad_hiddens : Tensor α [seqLen, hiddenSize]) :
  ( Tensor α [hiddenSize, inputSize + hiddenSize] × Tensor α [hiddenSize] ×  -- dWf, dbf
    Tensor α [hiddenSize, inputSize + hiddenSize] × Tensor α [hiddenSize] ×  -- dWi, dbi
    Tensor α [hiddenSize, inputSize + hiddenSize] × Tensor α [hiddenSize] ×  -- dWc, dbc
    Tensor α [hiddenSize, inputSize + hiddenSize] × Tensor α [hiddenSize] ×  -- dWo, dbo
    Tensor α [seqLen, inputSize] ×                                                    -- dInputs
    LSTMState α hiddenSize ) :=
    -- dInitialState

  let (_, saved) := Sequence.mapAccum seqLen initial_state fun index state =>
    let (next, forget, input, candidate, output) :=
      lstmCellSpecWithIntermediates lstm (get inputs index) state
    (next, (next, forget, input, candidate, output))

  let zeroHidden := fill 0 (.dim hiddenSize .scalar)
  let zeroWeights := fill 0 (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar))
  let initial :=
    (⟨zeroHidden, zeroHidden⟩, zeroWeights, zeroHidden, zeroWeights, zeroHidden, zeroWeights,
      zeroHidden, zeroWeights, zeroHidden)
  let (result, dInputs) := Sequence.mapAccumRight seqLen initial fun index state =>
    let (dNext, forgetWeights, forgetBias, inputWeights, inputBias, candidateWeights,
        candidateBias, outputWeights, outputBias) := state
    let (current, forget, inputGate, candidate, output) := saved.getScalar index
    let previous :=
      if h : index.val > 0 then
        have hp : index.val - 1 < seqLen := by grind
        let (previous, _, _, _, _) := saved.getScalar ⟨index.val - 1, hp⟩
        previous
      else
        initial_state
    let totalHidden := addSpec (get grad_hiddens index) dNext.hidden
    let (dInput, dPrevious, dForgetWeights, dForgetBias, dInputWeights, dInputBias,
        dCandidateWeights, dCandidateBias, dOutputWeights, dOutputBias) :=
      lstmCellBackwardSpec lstm (get inputs index) previous current forget inputGate candidate
        output totalHidden dNext.cell
    ((dPrevious, addSpec forgetWeights dForgetWeights, addSpec forgetBias dForgetBias,
      addSpec inputWeights dInputWeights, addSpec inputBias dInputBias,
      addSpec candidateWeights dCandidateWeights, addSpec candidateBias dCandidateBias,
      addSpec outputWeights dOutputWeights, addSpec outputBias dOutputBias), dInput)
  let (dInitialState, dForgetWeights, dForgetBias, dInputWeights, dInputBias, dCandidateWeights,
      dCandidateBias, dOutputWeights, dOutputBias) := result
  (dForgetWeights, dForgetBias, dInputWeights, dInputBias, dCandidateWeights, dCandidateBias,
    dOutputWeights, dOutputBias, Tensor.dim dInputs.getScalar, dInitialState)

end Spec

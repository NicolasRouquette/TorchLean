/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Rnn

/-!
# GRU (spec layer)

TorchLean provides a small GRU specification that is:

- explicit about shapes (so dimension mistakes are caught early),
- explicit about the math (so we can reason about it and differentiate it),
- explicit about using the original Cho et al. candidate equation.

## References (math + PyTorch behavior)

- Cho et al.,
  "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"
  (EMNLP 2014): https://aclanthology.org/D14-1179/ (PDF: https://aclanthology.org/D14-1179.pdf)
- Chung et al., "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling"
  (2014):
  https://arxiv.org/abs/1412.3555
- PyTorch `GRUCell` equations: https://docs.pytorch.org/docs/stable/generated/torch.nn.GRUCell.html
- PyTorch `GRU` equations:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.rnn.GRU.html

## Notes on parameterization

The GRU equations are often written with separate matrices $W_\bullet$ for the input and
$U_\bullet$ for the hidden state. In this spec we use a single matrix per gate applied to a
concatenated vector $[x_t;h_{t-1}]$ (or $[x_t;r_t\odot h_{t-1}]$ for the candidate). This is the
same idea, just packaged in a way that reuses the tensor building blocks already present in the
spec layer.

One important place where libraries differ is the candidate equation. This file applies the reset
gate before the hidden-state linear map, as in Cho et al. PyTorch applies it after the hidden-state
linear map and has separate input/hidden candidate biases. Consequently, a PyTorch GRU checkpoint
cannot be loaded into this parameterization without an explicit conversion or a matching custom
module.
-/

@[expose] public section


namespace Spec

open Tensor
open Activation

variable {α : Type} [Context α]

-- GRU cell specification: separate weights for reset, update, and new gates
-- Each gate has weights [hidden_size, input_size + hidden_size] and bias [hidden_size]
/--
Parameters for a single GRU cell.

This is the original concatenated GRU parameterization, using $[x_t;h_{t-1}]$ (shape
`inputSize + hiddenSize`) for the reset/update gates and
$[x_t;r_t\odot h_{t-1}]$ for the candidate gate.

Shapes:

- each gate weight is `[hiddenSize, inputSize + hiddenSize]`,
- each gate bias is `[hiddenSize]`.
-/
structure GRUSpec (α : Type) (inputSize hiddenSize : Nat) where
  /-- Reset-gate weights for
  $r_t=\operatorname{sigmoid}(W_r[x_t;h_{t-1}]+b_r)$. -/
  resetWeight : Tensor α [hiddenSize, inputSize + hiddenSize]
  /-- Reset-gate bias. -/
  resetBias : Tensor α [hiddenSize]
  /-- Update-gate weights for
  $z_t=\operatorname{sigmoid}(W_z[x_t;h_{t-1}]+b_z)$. -/
  updateWeight : Tensor α [hiddenSize, inputSize + hiddenSize]
  /-- Update-gate bias. -/
  updateBias : Tensor α [hiddenSize]
  /-- Candidate-state weights for
  $n_t=\tanh(W_n[x_t;r_t\odot h_{t-1}]+b_n)$. -/
  candidateWeight : Tensor α [hiddenSize, inputSize + hiddenSize]
  /-- Candidate-state bias. -/
  candidateBias : Tensor α [hiddenSize]

/--
Forward pass for a single GRU cell.

Given input $x_t$ and previous hidden state $h_{t-1}$, compute the next hidden state $h_t$ using
the standard GRU equations.

This is not PyTorch's reset-after candidate parameterization; see the module note above.
-/
def gruCellSpec {inputSize hiddenSize : Nat}
  (gru : GRUSpec α inputSize hiddenSize)
  (input : Tensor α [inputSize])
  (prev_hidden : Tensor α [hiddenSize]) :
  Tensor α [hiddenSize] :=
  -- We follow the textbook GRU layout:
  --
  --   r_t = sigmoid(W_r [x_t; h_{t-1}] + b_r)
  --   z_t = sigmoid(W_z [x_t; h_{t-1}] + b_z)
  --   n_t = tanh   (W_n [x_t; r_t ⊙ h_{t-1}] + b_n)
  --   h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
  --
  -- PyTorch instead resets the hidden affine contribution after its matrix multiplication.
  let concat := concatAxisSpec .scalar input prev_hidden

  -- Reset gate.
  let reset_gate := sigmoidSpec (addSpec (matVecMulSpec gru.resetWeight concat)
    gru.resetBias)

  -- Update gate.
  let update_gate := sigmoidSpec (addSpec (matVecMulSpec gru.updateWeight concat)
    gru.updateBias)

  -- The reset gate decides what portion of the previous state is used in the candidate update.
  let reset_hidden := mulSpec reset_gate prev_hidden

  -- Candidate uses `[x_t; r_t ⊙ h_{t-1}]`.
  let reset_concat := concatAxisSpec .scalar input reset_hidden

  -- Candidate (sometimes called `n_t` or `h~_t` in the literature).
  let new_candidate := tanhSpec (addSpec (matVecMulSpec gru.candidateWeight reset_concat)
    gru.candidateBias)

  -- Final hidden state:
  --   h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}.
  let one_minus_update := subSpec (fill 1 (.dim hiddenSize .scalar)) update_gate
  let new_contribution := mulSpec one_minus_update new_candidate
  let hidden_contribution := mulSpec update_gate prev_hidden
  addSpec new_contribution hidden_contribution

-- GRU sequence forward pass: processes a sequence of inputs
/--
Unroll a GRU over `seqLen` timesteps (time-major).

This returns the sequence of hidden states $[h_0,\ldots,h_{\mathtt{seqLen}-1}]$. It is a pure
spec-level
definition of semantics; an efficient runtime is free to implement the same behavior with loops and
caching.

The input is time-major and the result contains every hidden state. The candidate semantics remain
the Cho-style equations of `gruCellSpec`.
-/
def gruSequenceSpec {seqLen inputSize hiddenSize : Nat}
  (gru : GRUSpec α inputSize hiddenSize)
  (inputs : Tensor α [seqLen, inputSize])
  (initial_hidden : Tensor α [hiddenSize]) :
  Tensor α [seqLen, hiddenSize] :=
  let (_, outputs) := Sequence.mapAccum seqLen initial_hidden fun i previous =>
    let hidden := gruCellSpec gru (get inputs i) previous
    (hidden, hidden)
  Tensor.dim outputs.getScalar

-- GRU cell forward pass that returns all intermediate values for BPTT
/--
GRU cell forward pass that also returns cached intermediates for BPTT.

This computes the same next hidden state as `gruCellSpec`, but additionally returns:

- `reset_gate` ($r_t$),
- `update_gate` ($z_t$),
- `new_candidate` ($n_t$), and
- `reset_hidden` ($r_t\odot h_{t-1}$).

These are exactly the quantities commonly saved by a reverse-mode implementation (PyTorch-style
autograd) to compute gradients efficiently in the backward pass.
-/
def gruCellSpecWithIntermediates {inputSize hiddenSize : Nat}
  (gru : GRUSpec α inputSize hiddenSize)
  (input : Tensor α [inputSize])
  (prev_hidden : Tensor α [hiddenSize]) :
  (Tensor α [hiddenSize] ×  -- new_hidden
   Tensor α [hiddenSize] ×  -- reset_gate
   Tensor α [hiddenSize] ×  -- update_gate
   Tensor α [hiddenSize] ×  -- new_candidate
   Tensor α [hiddenSize]) := -- reset_hidden
  -- Same computation as `gru_cell_spec`, but we also return the gate activations and the candidate.
  -- Those values are what a "tape" would store for a standard BPTT implementation.
  let concat := concatAxisSpec .scalar input prev_hidden

  -- Reset gate: r_t = σ(W_r @ [x_t; h_{t-1}] + b_r)
  let reset_gate := sigmoidSpec (addSpec (matVecMulSpec gru.resetWeight concat)
    gru.resetBias)

  -- Update gate: z_t = σ(W_z @ [x_t; h_{t-1}] + b_z)
  let update_gate := sigmoidSpec (addSpec (matVecMulSpec gru.updateWeight concat)
    gru.updateBias)

  -- Reset hidden state: h_reset = r_t ⊙ h_{t-1}
  let reset_hidden := mulSpec reset_gate prev_hidden

  -- Concatenate input with reset hidden state
  let reset_concat := concatAxisSpec .scalar input reset_hidden

  -- New hidden state candidate: ĥ_t = tanh(W_h @ [x_t; r_t ⊙ h_{t-1}] + b_h)
  let new_candidate := tanhSpec (addSpec (matVecMulSpec gru.candidateWeight reset_concat)
    gru.candidateBias)

  -- Final hidden state follows the same convention as `gru_cell_spec`.
  let one_minus_update := subSpec (fill 1 (.dim hiddenSize .scalar)) update_gate
  let new_contribution := mulSpec one_minus_update new_candidate
  let hidden_contribution := mulSpec update_gate prev_hidden
  let new_hidden := addSpec new_contribution hidden_contribution

  (new_hidden, reset_gate, update_gate, new_candidate, reset_hidden)

/--
Run a GRU forward pass while collecting the per-timestep intermediates needed for BPTT.

This is the "spec-level" analogue of what frameworks do internally:

- the forward pass produces $h_t$,
- and it also saves gate activations $r_t$, $z_t$, and candidate $n_t$ for the backward pass.

The returned tensors are all time-major (`seqLen` first) to match the rest of the spec layer.
-/
def gruExtractIntermediateValues {seqLen inputSize hiddenSize : Nat}
  (gru : GRUSpec α inputSize hiddenSize)
  (inputs : Tensor α [seqLen, inputSize])
  (initial_hidden : Tensor α [hiddenSize]) :
  (Tensor α [seqLen, hiddenSize] ×  -- hidden_states
   Tensor α [seqLen, hiddenSize] ×  -- reset_gates
   Tensor α [seqLen, hiddenSize] ×  -- update_gates
   Tensor α [seqLen, hiddenSize] ×  -- new_candidates
   Tensor α [seqLen, hiddenSize]) := -- reset_hiddens
  let (_, saved) := Sequence.mapAccum seqLen initial_hidden fun i previous =>
    let (hidden, reset, update, candidate, resetHidden) :=
      gruCellSpecWithIntermediates gru (get inputs i) previous
    (hidden, (hidden, reset, update, candidate, resetHidden))

  let hidden_states := Tensor.dim fun i =>
    let (hidden, _, _, _, _) := saved.getScalar i
    hidden
  let reset_gates := Tensor.dim fun i =>
    let (_, reset, _, _, _) := saved.getScalar i
    reset
  let update_gates := Tensor.dim fun i =>
    let (_, _, update, _, _) := saved.getScalar i
    update
  let new_candidates := Tensor.dim fun i =>
    let (_, _, _, candidate, _) := saved.getScalar i
    candidate
  let reset_hiddens := Tensor.dim fun i =>
    let (_, _, _, _, resetHidden) := saved.getScalar i
    resetHidden

  (hidden_states, reset_gates, update_gates, new_candidates, reset_hiddens)

-- Batched GRU sequence forward pass
/--
Batched GRU forward pass (map `gruSequenceSpec` over the batch dimension).

This is a simple spec-level definition for semantics, not an optimized kernel. It maps the same
Cho-style cell over a batch; it is not a `torch.nn.GRU` checkpoint format.
-/
def gruBatchedSpec {batchSize seqLen inputSize hiddenSize : Nat}
  (gru : GRUSpec α inputSize hiddenSize)
  (inputs : Tensor α [batchSize, seqLen, inputSize])
  (initial_hidden : Tensor α [batchSize, hiddenSize]) :
  Tensor α [batchSize, seqLen, hiddenSize] :=
  -- This is a simple "map over the batch dimension".
  -- It matches the semantics of a batched GRU, but it is not an optimized runtime kernel.
  match inputs, initial_hidden with
  | Tensor.dim batch_inputs, Tensor.dim batch_hidden =>
    Tensor.dim (fun b =>
      gruSequenceSpec gru (batch_inputs b) (batch_hidden b))

-- Gradient computations for GRU

-- Gradient w.r.t. reset gate weights
/--
Reference gradient for reset-gate weights via the generic RNN weight-gradient helper.

This uses `rnnWeightsDerivSpec` on the concatenated inputs/hidden states. It is a convenient
building block, but the more explicit BPTT helpers below show the time-unrolled
accumulation form.
-/
def gruResetWeightsDerivSpec {seqLen inputSize hiddenSize : Nat}
  (inputs : Tensor α [seqLen, inputSize])
  (hiddens : Tensor α [seqLen, hiddenSize])
  (grad_reset : Tensor α [seqLen, hiddenSize]) :
  Tensor α [hiddenSize, inputSize + hiddenSize] :=
  rnnWeightsDerivSpec inputs hiddens grad_reset

-- Gradient w.r.t. update gate weights
/-- Reference gradient for update-gate weights (via `rnnWeightsDerivSpec`). -/
def gruUpdateWeightsDerivSpec {seqLen inputSize hiddenSize : Nat}
  (inputs : Tensor α [seqLen, inputSize])
  (hiddens : Tensor α [seqLen, hiddenSize])
  (grad_update : Tensor α [seqLen, hiddenSize]) :
  Tensor α [hiddenSize, inputSize + hiddenSize] :=
  rnnWeightsDerivSpec inputs hiddens grad_update

-- Gradient w.r.t. new gate weights
/--
Reference gradient for candidate ("new") gate weights (via `rnnWeightsDerivSpec`).

The second sequence argument satisfies
$\mathtt{reset\_hiddens}_t=r_t\odot h_{t-1}$.
-/
def gruNewWeightsDerivSpec {seqLen inputSize hiddenSize : Nat}
  (inputs : Tensor α [seqLen, inputSize])
  (reset_hiddens : Tensor α [seqLen, hiddenSize]) -- r_t ⊙ h_{t-1}
  (grad_new : Tensor α [seqLen, hiddenSize]) :
  Tensor α [hiddenSize, inputSize + hiddenSize] :=
  rnnWeightsDerivSpec inputs reset_hiddens grad_new

-- Gradient w.r.t. biases (sum over sequence length)
/--
Bias gradient by summing per-timestep gradients over the time axis.

This is the spec-level analogue of the common "sum across batch/time" reduction used for bias
gradients. The `seqLen ≠ 0` hypothesis is exactly what makes axis `0` a valid reduction axis.
-/
def gruBiasDerivSpec {seqLen hiddenSize : Nat}
  (grad_outputs : Tensor α [seqLen, hiddenSize])
  (h : seqLen ≠ 0) :
  Tensor α [hiddenSize] :=
  reduceSum 0 grad_outputs (Shape.hasNonemptyAxisZeroOfNe h).proof

-- Gradient w.r.t. reset gate weights with proper BPTT
/--
Reset-gate weight gradient by explicit time-unrolled accumulation (BPTT-style).

This computes
$$
\sum_t \frac{\partial L}{\partial r_t}\otimes[x_t;h_{t-1}],
$$
where $\otimes$ is an outer product.
-/
def gruResetWeightsDerivBpttSpec {seqLen inputSize hiddenSize : Nat}
  (inputs : Tensor α [seqLen, inputSize])
  (hiddens : Tensor α [seqLen, hiddenSize])
  (_reset_gates : Tensor α [seqLen, hiddenSize])
  (grad_reset_gates : Tensor α [seqLen, hiddenSize]) :
  Tensor α [hiddenSize, inputSize + hiddenSize] :=
  -- Accumulate gradients over time steps
  let rec accumulate_grads (t : Nat) (acc : Tensor α [hiddenSize, inputSize + hiddenSize]) :
    Tensor α [hiddenSize, inputSize + hiddenSize] :=
    if h : t < seqLen then
      let input_t := get inputs ⟨t, h⟩
      let hidden_prev :=
        if ht : t > 0 then
          have ht0 : t ≠ 0 := Nat.ne_of_gt ht
          have htPred : t - 1 < t := by
            simpa [Nat.pred_eq_sub_one] using Nat.pred_lt ht0
          have htPrev : t - 1 < seqLen := lt_trans htPred h
          get hiddens ⟨t - 1, htPrev⟩
        else
          fill 0 (.dim hiddenSize .scalar)
      let concat_t := concatAxisSpec .scalar input_t hidden_prev
      let grad_reset_t := get grad_reset_gates ⟨t, h⟩
      let grad_w_t := outerProductSpec grad_reset_t concat_t
      accumulate_grads (t + 1) (addSpec acc grad_w_t)
    else acc
  accumulate_grads 0 (fill 0 (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)))

-- Gradient w.r.t. update gate weights with proper BPTT
/--
Update-gate weight gradient by explicit time-unrolled accumulation (BPTT-style).

This computes
$$
\sum_t \frac{\partial L}{\partial z_t}\otimes[x_t;h_{t-1}].
$$
-/
def gruUpdateWeightsDerivBpttSpec {seqLen inputSize hiddenSize : Nat}
  (inputs : Tensor α [seqLen, inputSize])
  (hiddens : Tensor α [seqLen, hiddenSize])
  (grad_update_gates : Tensor α [seqLen, hiddenSize]) :
  Tensor α [hiddenSize, inputSize + hiddenSize] :=
  -- Accumulate gradients over time steps
  let rec accumulate_grads (t : Nat) (acc : Tensor α [hiddenSize, inputSize + hiddenSize]) :
    Tensor α [hiddenSize, inputSize + hiddenSize] :=
    if h : t < seqLen then
      let input_t := get inputs ⟨t, h⟩
      let hidden_prev :=
        if ht : t > 0 then
          have ht0 : t ≠ 0 := Nat.ne_of_gt ht
          have htPred : t - 1 < t := by
            simpa [Nat.pred_eq_sub_one] using Nat.pred_lt ht0
          have htPrev : t - 1 < seqLen := lt_trans htPred h
          get hiddens ⟨t - 1, htPrev⟩
        else
          fill 0 (.dim hiddenSize .scalar)
      let concat_t := concatAxisSpec .scalar input_t hidden_prev
      let grad_update_t := get grad_update_gates ⟨t, h⟩
      let grad_w_t := outerProductSpec grad_update_t concat_t
      accumulate_grads (t + 1) (addSpec acc grad_w_t)
    else acc
  accumulate_grads 0 (fill 0 (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)))

-- Gradient w.r.t. new gate weights with proper BPTT
/--
Candidate-gate weight gradient by explicit time-unrolled accumulation (BPTT-style).

This computes
$$
\sum_t \frac{\partial L}{\partial n_t}\otimes[x_t;r_t\odot h_{t-1}].
$$
-/
def gruNewWeightsDerivBpttSpec {seqLen inputSize hiddenSize : Nat}
  (inputs : Tensor α [seqLen, inputSize])
  (reset_hiddens : Tensor α [seqLen, hiddenSize]) -- r_t ⊙ h_{t-1}
  (grad_new_candidates : Tensor α [seqLen, hiddenSize]) :
  Tensor α [hiddenSize, inputSize + hiddenSize] :=
  -- Accumulate gradients over time steps
  let rec accumulate_grads (t : Nat) (acc : Tensor α [hiddenSize, inputSize + hiddenSize]) :
    Tensor α [hiddenSize, inputSize + hiddenSize] :=
    if h : t < seqLen then
      let input_t := get inputs ⟨t, h⟩
      let reset_hidden_t := get reset_hiddens ⟨t, h⟩
      let concat_t := concatAxisSpec .scalar input_t reset_hidden_t
      let grad_new_t := get grad_new_candidates ⟨t, h⟩
      let grad_w_t := outerProductSpec grad_new_t concat_t
      accumulate_grads (t + 1) (addSpec acc grad_w_t)
    else acc
  accumulate_grads 0 (fill 0 (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)))

/--
Backward (VJP) for a single GRU cell.

Inputs:

- the cell parameters `gru`,
- the current input $x_t$,
- the previous hidden state $h_{t-1}$,
- an upstream gradient $\partial L/\partial h_t$,
- and the forward intermediates $r_t$, $z_t$, and $n_t$ that a typical BPTT implementation would
  cache.

Outputs:

- gradients w.r.t. the input and previous hidden state,
- plus gradients for each parameter tensor (weights and biases).

This is written to match the forward equations in `gruCellSpec`. It is not an optimized kernel;
it is a precise spec for what gradients *should* be.
-/
def gruCellBackwardFullSpec {inputSize hiddenSize : Nat}
  (gru : GRUSpec α inputSize hiddenSize)
  (input : Tensor α [inputSize])
  (prev_hidden : Tensor α [hiddenSize])
  (grad_output : Tensor α [hiddenSize])
  (reset_gate : Tensor α [hiddenSize])
  (update_gate : Tensor α [hiddenSize])
  (new_candidate : Tensor α [hiddenSize]) :
  ( Tensor α [inputSize] ×                     -- dInput
    Tensor α [hiddenSize] ×                    -- dPrevHidden
    Tensor α [hiddenSize, inputSize + hiddenSize] ×  -- dResetW
    Tensor α [hiddenSize] ×                    -- dResetB
    Tensor α [hiddenSize, inputSize + hiddenSize] ×  -- dUpdateW
    Tensor α [hiddenSize] ×                    -- dUpdateB
    Tensor α [hiddenSize, inputSize + hiddenSize] ×  -- dNewW
    Tensor α [hiddenSize]                      -- dNewB
  ) :=
  let concat := concatAxisSpec .scalar input prev_hidden

  -- Start from the output equation:
  --   h = (1 - z) ⊙ n + z ⊙ h_prev
  --
  -- This yields three immediate partials:
  --   d n      = d h ⊙ (1 - z)
  --   d z      = d h ⊙ (h_prev - n)
  --   d h_prev (direct) = d h ⊙ z
  let one_minus_z := subSpec (fill 1 (.dim hiddenSize .scalar)) update_gate
  let dHtilde := mulSpec grad_output one_minus_z
  let dZ := mulSpec grad_output (subSpec prev_hidden new_candidate)
  let dPrev_direct := mulSpec grad_output update_gate

  -- tanh preactivation derivative using output new_candidate = tanh(pre_h)
  let dPre_h := mulSpec dHtilde (subSpec (fill 1 (.dim hiddenSize .scalar)) (mulSpec
    new_candidate new_candidate))

  -- h_reset = r ⊙ h_prev, reset_concat = [x; h_reset]
  let reset_hidden := mulSpec reset_gate prev_hidden
  let reset_concat := concatAxisSpec .scalar input reset_hidden

  -- New gate grads.
  let dNewW := outerProductSpec dPre_h reset_concat
  let dNewB := dPre_h
  let dResetConcat := vecMatMulSpec dPre_h gru.candidateWeight
  let dX_from_h := sliceRangeSpec dResetConcat 0 inputSize (by
    simp)
  let dHreset := sliceRangeSpec dResetConcat inputSize hiddenSize (by
    simp)

  -- Backprop through reset_hidden = r ⊙ h_prev.
  let dR_from_reset := mulSpec dHreset prev_hidden
  let dPrev_from_reset := mulSpec dHreset reset_gate

  -- Reset gate grads: r = sigmoid(pre_r)
  let dPre_r := mulSpec dR_from_reset (Activation.sigmoidOutputDerivSpec reset_gate)
  let dResetW := outerProductSpec dPre_r concat
  let dResetB := dPre_r
  let dConcat_from_r := vecMatMulSpec dPre_r gru.resetWeight
  let dX_from_r := sliceRangeSpec dConcat_from_r 0 inputSize (by
    simp)
  let dPrev_from_r := sliceRangeSpec dConcat_from_r inputSize hiddenSize (by
    simp)

  -- Update gate grads: z = sigmoid(pre_z)
  let dPre_z := mulSpec dZ (Activation.sigmoidOutputDerivSpec update_gate)
  let dUpdateW := outerProductSpec dPre_z concat
  let dUpdateB := dPre_z
  let dConcat_from_z := vecMatMulSpec dPre_z gru.updateWeight
  let dX_from_z := sliceRangeSpec dConcat_from_z 0 inputSize (by
    simp)
  let dPrev_from_z := sliceRangeSpec dConcat_from_z inputSize hiddenSize (by
    simp)

  let dInput := addSpec (addSpec dX_from_h dX_from_r) dX_from_z
  let dPrevHidden := addSpec (addSpec (addSpec dPrev_direct dPrev_from_reset) dPrev_from_r)
    dPrev_from_z

  (dInput, dPrevHidden, dResetW, dResetB, dUpdateW, dUpdateB, dNewW, dNewB)

/--
Reverse-mode backprop through an unrolled GRU over `seqLen` steps (BPTT).

This function consumes the same intermediates produced by `gruExtractIntermediateValues`:
per-timestep gate activations and candidates. The backward pass walks time in reverse and
accumulates gradients for the Cho-style forward equation.
-/
def gruSequenceBackwardFullSpec {seqLen inputSize hiddenSize : Nat}
  (gru : GRUSpec α inputSize hiddenSize)
  (inputs : Tensor α [seqLen, inputSize])
  (hiddens : Tensor α [seqLen, hiddenSize])
  (grad_outputs : Tensor α [seqLen, hiddenSize])
  (reset_gates : Tensor α [seqLen, hiddenSize])
  (update_gates : Tensor α [seqLen, hiddenSize])
  (new_candidates : Tensor α [seqLen, hiddenSize])
  (initial_hidden : Tensor α [hiddenSize] := fill 0 [hiddenSize]) :
  ( Tensor α [hiddenSize, inputSize + hiddenSize] ×  -- dResetW
    Tensor α [hiddenSize] ×                                  -- dResetB
    Tensor α [hiddenSize, inputSize + hiddenSize] ×  -- dUpdateW
    Tensor α [hiddenSize] ×                                  -- dUpdateB
    Tensor α [hiddenSize, inputSize + hiddenSize] ×  -- dNewW
    Tensor α [hiddenSize] ×                                  -- dNewB
    Tensor α [seqLen, inputSize] ×                     -- dInputs
    Tensor α [hiddenSize]                                    -- dInitialHidden
  ) :=

  let zeroHidden := fill 0 (.dim hiddenSize .scalar)
  let zeroWeights := fill 0 (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar))
  let initial :=
    (zeroHidden, zeroWeights, zeroHidden, zeroWeights, zeroHidden, zeroWeights, zeroHidden)
  let (result, dInputs) := Sequence.mapAccumRight seqLen initial fun index state =>
    let (dHiddenNext, resetWeights, resetBias, updateWeights, updateBias, newWeights, newBias) :=
      state
    let input := get inputs index
    let previous :=
      if h : index.val > 0 then
        have hp : index.val - 1 < seqLen := by grind
        get hiddens ⟨index.val - 1, hp⟩
      else
        initial_hidden
    let totalGradient := addSpec (get grad_outputs index) dHiddenNext
    let (dInput, dHidden, dResetWeights, dResetBias, dUpdateWeights, dUpdateBias,
        dNewWeights, dNewBias) :=
      gruCellBackwardFullSpec gru input previous totalGradient (get reset_gates index)
        (get update_gates index) (get new_candidates index)
    ((dHidden, addSpec resetWeights dResetWeights, addSpec resetBias dResetBias,
      addSpec updateWeights dUpdateWeights, addSpec updateBias dUpdateBias,
      addSpec newWeights dNewWeights, addSpec newBias dNewBias), dInput)
  let (dInitialHidden, dResetWeights, dResetBias, dUpdateWeights, dUpdateBias, dNewWeights,
      dNewBias) := result
  (dResetWeights, dResetBias, dUpdateWeights, dUpdateBias, dNewWeights, dNewBias,
    Tensor.dim dInputs.getScalar, dInitialHidden)

/--
Return the input-sequence and initial-hidden gradients from `gruSequenceBackwardFullSpec`.

The full backward pass also returns parameter gradients. This projection records the common contract
used by callers that only propagate gradients to the preceding recurrent computation.
-/
def gruSequenceBackwardSpec {seqLen inputSize hiddenSize : Nat}
  (gru : GRUSpec α inputSize hiddenSize)
  (inputs : Tensor α [seqLen, inputSize])
  (hiddens : Tensor α [seqLen, hiddenSize])
  (grad_outputs : Tensor α [seqLen, hiddenSize])
  (reset_gates : Tensor α [seqLen, hiddenSize])
  (update_gates : Tensor α [seqLen, hiddenSize])
  (new_candidates : Tensor α [seqLen, hiddenSize])
  (initial_hidden : Tensor α [hiddenSize] := fill 0 [hiddenSize]) :
  (Tensor α [seqLen, inputSize] × Tensor α [hiddenSize]) :=
  let (_, _, _, _, _, _, dInputs, dInitialHidden) :=
    gruSequenceBackwardFullSpec gru inputs hiddens grad_outputs reset_gates update_gates
      new_candidates initial_hidden
  (dInputs, dInitialHidden)

end Spec

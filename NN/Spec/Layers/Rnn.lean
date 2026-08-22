/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Sequence
public import NN.Spec.Layers.Activation

/-!
# RNN (spec layer)

Defines a vanilla RNN cell and sequence semantics, along with BPTT-style gradients.

This is the recurrent core that TorchLean builds on:

- a single-step cell (`rnnCellSpec`),
- an explicit unrolling over time (`rnnSequenceSpec`),
- and a reverse-time VJP (`rnnSequenceBackwardSpec`).

PyTorch analogy:

- `rnnCellSpec` corresponds to `torch.nn.RNNCell` with `nonlinearity="tanh"`.
- `rnnSequenceSpec` corresponds to `torch.nn.RNN` unrolled over `seqLen`.

## References

- Elman, "Finding Structure in Time" (1990): https://crl.ucsd.edu/~elman/Papers/fsit.pdf
- PyTorch `RNNCell`: https://docs.pytorch.org/docs/stable/generated/torch.nn.RNNCell.html
- PyTorch `RNN`: https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html
-/

@[expose] public section


namespace Spec

open Tensor
open Activation

variable {α : Type} [Context α]

/-!
## Recurrent tensor shapes

These names keep recurrent signatures close to the standard mathematical notation without adding
new tensor representations:

- `Recurrent.InputVector α inputSize` is a length-`inputSize` vector,
- `Recurrent.HiddenVector α hiddenSize` is a length-`hiddenSize` vector,
- `Recurrent.WeightMatrix α hiddenSize inputSize` is a `(hiddenSize × inputSize)` matrix,
- `Recurrent.SequenceTensor α seqLen s` is a time-major sequence of length `seqLen`.

PyTorch note: `torch.nn.RNN` can be configured as batch-first or time-first. In the spec layer we
standardize on time-major (`seqLen` outermost) because it matches recursive definitions and proofs.
-/

namespace Recurrent

/-- Length-`inputSize` input vector. -/
abbrev InputVector (α : Type) (inputSize : Nat) := Tensor α (.dim inputSize .scalar)

/-- Length-`hiddenSize` hidden-state vector. -/
abbrev HiddenVector (α : Type) (hiddenSize : Nat) := Tensor α (.dim hiddenSize .scalar)

/-- `(hiddenSize × inputSize)` dense weight matrix. -/
abbrev WeightMatrix (α : Type) (hiddenSize inputSize : Nat) :=
  Tensor α (.dim hiddenSize (.dim inputSize .scalar))

/-- Time-major sequence of length `seqLen` with per-step shape `shape`. -/
abbrev SequenceTensor (α : Type) (seqLen : Nat) (shape : Shape) := Tensor α (.dim seqLen shape)

/-- Batch of `batchSize` tensors with per-item shape `shape`. -/
abbrev BatchedTensor (α : Type) (batchSize : Nat) (shape : Shape) := Tensor α (.dim batchSize shape)

end Recurrent

open Recurrent

/--
RNN cell parameters.

We use a single weight matrix applied to a concatenated vector `[x_t; h_{t-1}]`:

`h_t = tanh(W [x_t; h_{t-1}] + b)`.

This is equivalent to the common split-parameter form:

`h_t = tanh(W_ih x_t + W_hh h_{t-1} + b)`,

just packaged to reuse the same tensor primitives elsewhere in TorchLean.
-/
structure RNNSpec (α : Type) (inputSize hiddenSize : Nat) where
  /-- Combined input-to-hidden and hidden-to-hidden weight matrix. -/
  weights : WeightMatrix α hiddenSize (inputSize + hiddenSize)
  /-- Hidden-state bias vector. -/
  bias    : HiddenVector α hiddenSize

/--
Single RNN cell forward pass.

Math:
`h_t = tanh(W [x_t; h_{t-1}] + b)`.

PyTorch analogy: `RNNCell(input, hidden)` with `tanh` nonlinearity.
-/
def rnnCellSpec {inputSize hiddenSize : Nat}
  (rnn : RNNSpec α inputSize hiddenSize)
  (input : InputVector α inputSize)
  (hidden : HiddenVector α hiddenSize) :
  HiddenVector α hiddenSize :=
  -- Concatenate input and hidden state
  let concat := concatLeadingAxisSpec input hidden
  -- Apply linear transformation: Wx + b
  let linear_out := addSpec (matVecMulSpec rnn.weights concat) rnn.bias
  -- Apply tanh activation
  tanhSpec linear_out

-- ============================================================================
-- Backpropagation (BPTT)
-- ============================================================================

-- Single RNN cell backward pass.
-- Forward: h_t = tanh(W @ [x_t; h_{t-1}] + b)
-- Backward returns:
--   dX_t, dH_{t-1}, dW, db
/--
Backward/VJP for a single RNN cell.

Inputs:
- `x_t`, `h_{t-1}`,
- the cached forward output `h_t` (so we can write `tanh'` in terms of `h_t`),
- an upstream gradient `dL/dh_t`.

Outputs:
- `dL/dx_t`, `dL/dh_{t-1}`, and parameter gradients `(dL/dW, dL/db)`.
-/
def rnnCellBackwardSpec {inputSize hiddenSize : Nat}
  (rnn : RNNSpec α inputSize hiddenSize)
  (input : InputVector α inputSize)
  (prev_hidden : HiddenVector α hiddenSize)
  (hidden : HiddenVector α hiddenSize)
  (grad_hidden : HiddenVector α hiddenSize) :
  (InputVector α inputSize × HiddenVector α hiddenSize ×
    WeightMatrix α hiddenSize (inputSize + hiddenSize) × HiddenVector α hiddenSize) :=
  let concat := concatLeadingAxisSpec input prev_hidden

  -- tanh'(z) = 1 - tanh(z)^2, and tanh(z) = hidden
  let tanh_deriv := subSpec (fill 1 (.dim hiddenSize .scalar)) (mulSpec hidden hidden)
  let grad_preact := mulSpec grad_hidden tanh_deriv

  let grad_weights := outerProductSpec grad_preact concat
  let grad_bias := grad_preact

  -- dConcat = grad_preactᵀ * W  (shape: inputSize + hiddenSize)
  let grad_concat := vecMatMulSpec grad_preact rnn.weights
  let grad_input := sliceRangeSpec grad_concat 0 inputSize (by simp)
  let grad_prev_hidden := sliceRangeSpec grad_concat inputSize hiddenSize (by simp)

  (grad_input, grad_prev_hidden, grad_weights, grad_bias)

-- RNN sequence forward pass: processes a sequence of inputs
/--
Unroll an RNN over `seqLen` steps (time-major).

Returns the sequence of hidden states `[h_0, ..., h_{seqLen-1}]`.
-/
def rnnSequenceSpec {seqLen inputSize hiddenSize : Nat}
  (rnn : RNNSpec α inputSize hiddenSize)
  (inputs : SequenceTensor α seqLen (.dim inputSize .scalar))
  (initial_hidden : HiddenVector α hiddenSize) :
  SequenceTensor α seqLen (.dim hiddenSize .scalar) :=
  let (_, outputs) := Sequence.mapAccum seqLen initial_hidden fun i previous =>
    let hidden := rnnCellSpec rnn (getAtSpec inputs i) previous
    (hidden, hidden)
  Tensor.dim outputs.get

/-- Batched RNN forward pass (maps `rnnSequenceSpec` over the batch dimension). -/
def rnnBatchedSpec {batchSize seqLen inputSize hiddenSize : Nat}
  (rnn : RNNSpec α inputSize hiddenSize)
  (inputs : BatchedTensor α batchSize (.dim seqLen (.dim inputSize .scalar)))
  (initial_hidden : BatchedTensor α batchSize (.dim hiddenSize .scalar)) :
  BatchedTensor α batchSize (.dim seqLen (.dim hiddenSize .scalar)) :=
  match inputs, initial_hidden with
  | Tensor.dim batch_inputs, Tensor.dim batch_hidden =>
    Tensor.dim (fun b =>
      rnnSequenceSpec rnn (batch_inputs b) (batch_hidden b))

/--
Gradient w.r.t. weights from a full unroll, given per-step preactivation gradients.

This helper is for analyses that already have preactivation gradients. It assumes:
- the initial hidden state is `0`, and
- `grad_outputs[t]` is already `dL/dz_t` (preactivation gradient).

For end-to-end BPTT from `dL/dh_t`, prefer `rnnSequenceBackwardSpec`.
-/
def rnnWeightsDerivSpec {seqLen inputSize hiddenSize : Nat}
  (inputs : SequenceTensor α seqLen (.dim inputSize .scalar))
  (hiddens : SequenceTensor α seqLen (.dim hiddenSize .scalar))
  (grad_outputs : SequenceTensor α seqLen (.dim hiddenSize .scalar)) :
  WeightMatrix α hiddenSize (inputSize + hiddenSize) :=
  -- Assumes initial hidden state is 0 (matches the default module wrappers).
  -- Assumes `grad_outputs` is the preactivation gradient at each timestep.
  -- For full BPTT from post-activation gradients, use `rnnSequenceBackwardSpec`.
  let rec accumulate_grads (t : Nat) (acc : WeightMatrix α hiddenSize (inputSize + hiddenSize)) :
      WeightMatrix α hiddenSize (inputSize + hiddenSize) :=
    if h : t < seqLen then
      let input_t := getAtSpec inputs ⟨t, h⟩
      let hidden_prev :=
        if ht : t > 0 then
          have h_pred : t - 1 < t := by
            simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt ht)
          have h_t' : t - 1 < seqLen := lt_trans h_pred h
          getAtSpec hiddens ⟨t - 1, h_t'⟩
        else
          fill 0 (.dim hiddenSize .scalar)
      let grad_preact_t := getAtSpec grad_outputs ⟨t, h⟩
      let concat_t := concatLeadingAxisSpec input_t hidden_prev
      let grad_w_t := outerProductSpec grad_preact_t concat_t
      accumulate_grads (t + 1) (addSpec acc grad_w_t)
    else
      acc
  accumulate_grads 0 (fill 0 (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)))

/--
Gradient w.r.t. bias from per-step preactivation gradients.

This is `sum_t dL/dz_t` over the sequence dimension.
-/
def rnnBiasDerivSpec {seqLen hiddenSize : Nat}
  (grad_outputs : SequenceTensor α seqLen (.dim hiddenSize .scalar))
  (h : seqLen ≠ 0) :
  HiddenVector α hiddenSize :=
  -- Assumes `grad_outputs` is already the preactivation gradient.
  -- For full RNN backprop, prefer `rnnSequenceBackwardSpec`.
  reduceSum 0 grad_outputs (Shape.hasNonemptyAxisZeroOfNe h).proof

/--
Full BPTT backward pass through an RNN sequence.

This is the spec-level version of what PyTorch autograd computes for `nn.RNN` when unrolled:

- we walk time in reverse,
- accumulate parameter gradients,
- and compute gradients for each input step plus the initial hidden state.

### Diagram: forward unroll + BPTT (vanilla RNN)

One step (forward):

```
x_t        h_{t-1}
 |            |
 +---- concat ----+
                 |
             z_t = W · [x_t; h_{t-1}] + b
                 |
             h_t = tanh(z_t)
```

Unrolled over time (forward):

```
h_-1 = h0

x0 -> [cell] -> h0 -> [cell] -> h1 -> ... -> [cell] -> h_{T-1}
        ^          ^                       ^
      uses h_-1  uses h0                 uses h_{T-2}
```

Backprop through time (reverse):

At each time step we combine two sources of gradient for `h_t`:

- the gradient coming from the loss that touches `h_t` directly (`grad_hiddens[t]`),
- plus the gradient flowing "from the future" through the recurrence (`dHidden_next`).

Then we push `total_grad` through the single-step VJP (`rnn_cell_backward_spec`), producing:

- `dInput_t` and `dHidden_prev`,
- and parameter gradients `dW_t`, `db_t` which are accumulated across time.
-/

def rnnSequenceBackwardSpec {seqLen inputSize hiddenSize : Nat}
  (rnn : RNNSpec α inputSize hiddenSize)
  (inputs : SequenceTensor α seqLen (.dim inputSize .scalar))
  (initial_hidden : HiddenVector α hiddenSize)
  (hiddens : SequenceTensor α seqLen (.dim hiddenSize .scalar))
  (grad_hiddens : SequenceTensor α seqLen (.dim hiddenSize .scalar)) :
  ( WeightMatrix α hiddenSize (inputSize + hiddenSize) ×  -- dW
    HiddenVector α hiddenSize ×                            -- db
    SequenceTensor α seqLen (.dim inputSize .scalar) ×     -- dInputs
    HiddenVector α hiddenSize ) :=                          -- dInitialHidden

  let initial :=
    (fill 0 (.dim hiddenSize .scalar),
      fill 0 (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)),
      fill 0 (.dim hiddenSize .scalar))
  let (result, dInputs) := Sequence.mapAccumRight seqLen initial fun index state =>
    let (dHiddenNext, accumulatedWeights, accumulatedBias) := state
    let input := getAtSpec inputs index
    let hidden := getAtSpec hiddens index
    let previous :=
      if h : index.val > 0 then
        have hp : index.val - 1 < seqLen := by grind
        getAtSpec hiddens ⟨index.val - 1, hp⟩
      else
        initial_hidden
    let totalGradient := addSpec (getAtSpec grad_hiddens index) dHiddenNext
    let (dInput, dHidden, dWeights, dBias) :=
      rnnCellBackwardSpec rnn input previous hidden totalGradient
    ((dHidden, addSpec accumulatedWeights dWeights, addSpec accumulatedBias dBias), dInput)
  let (dInitialHidden, dWeights, dBias) := result
  (dWeights, dBias, Tensor.dim dInputs.get, dInitialHidden)

end Spec

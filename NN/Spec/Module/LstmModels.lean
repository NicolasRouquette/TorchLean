/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Dropout
public import NN.Spec.Layers.Loss
public import NN.Spec.Module.Linear
public import NN.Spec.Module.Rnn

/-!
# Long Short-Term Memory Models

Higher‑level LSTM architectures built from module specs (`Spec.Module.Chain`), including:

- sequence‑to‑sequence outputs,
- classifier heads (many‑to‑one),
- multi‑layer compositions.

Cell equations are in `NN/Spec/Layers/Lstm.lean`; this file focuses on composing modules.

References (math + PyTorch behavior):

- Hochreiter and Schmidhuber (1997), "Long Short-Term Memory" (original LSTM):
  https://www.bioinf.jku.at/publications/older/2604.pdf
- PyTorch `nn.LSTM` docs:
  https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- PyTorch `nn.LSTMCell` docs:
  https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html
-/

@[expose] public section

namespace Spec

open Tensor
open Spec.Module

variable {α : Type} [Context α]

namespace Lstm

/-!
## Model and gradient types

The LSTM model layer exposes first-class model objects with:

- a forward pass,
- a standard training objective, and
- an explicit reverse-mode / BPTT backward pass producing parameter gradients.

The backward functions reuse the gate-aware implementation in `NN.Spec.Layers.Lstm`.
-/

/-! ### Gradient records -/

/--
Gradients for a linear layer `y = W x + b`.

This is the natural gradient bundle for `Spec.LinearSpec` (PyTorch analogue: `torch.nn.Linear`),
with `dW` matching the weight shape `[outDim, inDim]` and `db` matching `[outDim]`.
-/
structure LinearGrads (α : Type) (inDim outDim : Nat) where
  /-- Gradient of the weight matrix. -/
  weight : Tensor α [outDim, inDim]
  /-- Gradient of the bias. -/
  bias : Tensor α [outDim]

/--
Gate-wise gradients for an LSTM cell.

This matches the parameterization used by `Spec.LSTMSpec` (see `NN/Spec/Layers/Lstm.lean`): each
gate has a weight matrix of shape `[hiddenSize, inputSize + hiddenSize]` applied to a concatenated
vector, plus a bias of shape `[hiddenSize]`.
-/
structure CellGrads (α : Type) (inputSize hiddenSize : Nat) where
  /-- Gradient of the forget-gate weight matrix. -/
  forgetWeight : Tensor α [hiddenSize, inputSize + hiddenSize]
  /-- Gradient of the forget-gate bias. -/
  forgetBias : Tensor α [hiddenSize]
  /-- Gradient of the input-gate weight matrix. -/
  inputWeight : Tensor α [hiddenSize, inputSize + hiddenSize]
  /-- Gradient of the input-gate bias. -/
  inputBias : Tensor α [hiddenSize]
  /-- Gradient of the candidate-state weight matrix. -/
  candidateWeight : Tensor α [hiddenSize, inputSize + hiddenSize]
  /-- Gradient of the candidate-state bias. -/
  candidateBias : Tensor α [hiddenSize]
  /-- Gradient of the output-gate weight matrix. -/
  outputWeight : Tensor α [hiddenSize, inputSize + hiddenSize]
  /-- Gradient of the output-gate bias. -/
  outputBias : Tensor α [hiddenSize]

/--
Parameter gradients for `Lstm.Model`.

This bundles the LSTM cell gradients and the time-distributed linear head gradients.
-/
structure Grads (α : Type) (inputSize hiddenSize outputSize : Nat) where
  /-- Gradients of the recurrent cell. -/
  cell : CellGrads α inputSize hiddenSize
  /-- Gradients of the output projection. -/
  output : LinearGrads α hiddenSize outputSize

/-- Parameter gradients for an LSTM classifier. -/
structure ClassifierGrads (α : Type) (inputSize hiddenSize numClasses : Nat) where
  /-- Gradients of the recurrent cell. -/
  cell : CellGrads α inputSize hiddenSize
  /-- Gradients of the classifier head. -/
  classifier : LinearGrads α hiddenSize numClasses

/--
Sequence-to-sequence LSTM model as a `Spec.Module.Chain`: LSTM over time, then a per-timestep linear head.

PyTorch analogue: `nn.LSTM` producing an output sequence, followed by `nn.linear` applied at each
time step.
-/
def sequence
  {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen inputSize hiddenSize outputSize : Nat}
  (lstmSpec : LSTMSpec α inputSize hiddenSize)
  (linearSpec : LinearSpec α hiddenSize outputSize) :
  Spec.Module.Chain α ([seqLen, inputSize]) ([seqLen, outputSize]) :=
  let lstmModule := Spec.Module.lstm lstmSpec
  let linearModule := Spec.Module.mapEach (Spec.Module.linear linearSpec)
  Spec.Module.Chain.single lstmModule
    |>.append linearModule

/--
Many-to-one LSTM classifier as a `Spec.Module.Chain`.

This runs an LSTM over the sequence and applies a linear classifier head to the final hidden state.
PyTorch analogue: `nn.LSTM` + `nn.linear`, taking the last output/hidden.
-/
def classifier
  {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen inputSize hiddenSize numClasses : Nat}
  (lstmSpec : LSTMSpec α inputSize hiddenSize)
  (classifierHead : LinearSpec α hiddenSize numClasses)
  (h : seqLen ≠ 0) :
  Spec.Module.Chain α ([seqLen, inputSize]) ([numClasses]) :=
  let lstmModule := Spec.Module.lstm lstmSpec
  let lastOutput := Spec.Module.select (shape := [seqLen, hiddenSize]) 0
    (⟨Nat.pred seqLen, Nat.pred_lt h⟩)
  let classifierModule := Spec.Module.linear classifierHead
  Spec.Module.Chain.single lstmModule
    |>.append lastOutput
    |>.append classifierModule

/--
Two-layer LSTM stack (sequence-to-sequence), followed by a per-timestep linear head.

The second LSTM consumes the hidden stream produced by the first.
-/
def stacked
  {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen inputSize hiddenSize outputSize : Nat}
  (firstSpec : LSTMSpec α inputSize hiddenSize)
  (secondSpec : LSTMSpec α hiddenSize hiddenSize)
  (linearSpec : LinearSpec α hiddenSize outputSize) :
  Spec.Module.Chain α ([seqLen, inputSize]) ([seqLen, outputSize]) :=
  let firstModule := Spec.Module.lstm firstSpec
  let secondModule := Spec.Module.lstm secondSpec
  let linearModule := Spec.Module.mapEach (Spec.Module.linear linearSpec)
  Spec.Module.Chain.single firstModule
    |>.append secondModule
    |>.append linearModule

/--
Simple LSTM language-model pipeline as a `Spec.Module.Chain`: embedding, LSTM core, and output projection.

In this spec layer we represent the embedding/projection as `LinearSpec`s (often used with one-hot
token vectors). PyTorch analogue: `nn.Embedding` (conceptually) + `nn.LSTM` + `nn.linear`.
-/
def languageModel
  {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen vocabSize hiddenSize : Nat}
  (embeddingSpec : LinearSpec α vocabSize hiddenSize)
  (lstmSpec : LSTMSpec α hiddenSize hiddenSize)
  (outputSpec : LinearSpec α hiddenSize vocabSize) :
  Spec.Module.Chain α ([seqLen, vocabSize]) ([seqLen, vocabSize]) :=
  let embeddingModule := Spec.Module.mapEach (Spec.Module.linear embeddingSpec)
  let lstmModule := Spec.Module.lstm lstmSpec
  let outputModule := Spec.Module.mapEach (Spec.Module.linear outputSpec)
  Spec.Module.Chain.single embeddingModule
    |>.append lstmModule
    |>.append outputModule

/-- Bidirectional LSTM followed by a classifier on the final concatenated state. -/
def bidirectionalClassifier
  {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen inputSize hiddenSize numClasses : Nat}
  (forwardSpec backwardSpec : LSTMSpec α inputSize hiddenSize)
  (classifierHead : LinearSpec α (hiddenSize + hiddenSize) numClasses)
  (h : seqLen ≠ 0) :
  Spec.Module.Chain α ([seqLen, inputSize]) ([numClasses]) :=
  Spec.Module.Chain.single (Spec.Module.bidirectionalLstm forwardSpec backwardSpec)
    |>.append (Spec.Module.select (shape := [seqLen, hiddenSize + hiddenSize]) 0
      (⟨Nat.pred seqLen, Nat.pred_lt h⟩))
    |>.append (Spec.Module.linear classifierHead)

-- Basic LSTM model with single layer
/--
Bundle of parameters for a single-layer LSTM model with a linear output head.

This is a direct record representation (as opposed to the `Spec.Module.Chain` representation above).
-/
structure Model (α : Type) (inputSize hiddenSize outputSize : Nat) where
  /-- Recurrent cell parameters. -/
  lstm : LSTMSpec α inputSize hiddenSize
  /-- Linear output projection. -/
  outputLayer : LinearSpec α hiddenSize outputSize

-- Multi-layer LSTM model
/--
Bundle of parameters for a multi-layer LSTM model with a linear output head.

The first layer consumes `inputSize`, and all subsequent layers consume `hiddenSize`.
-/
structure StackedModel (α : Type) (inputSize hiddenSize outputSize numLayers : Nat) where
  /-- First recurrent layer, whose input may differ from the hidden width. -/
  firstLayer : LSTMSpec α inputSize hiddenSize
  /-- Remaining recurrent layers. -/
  hiddenLayers : Fin (numLayers - 1) → LSTMSpec α hiddenSize hiddenSize
  /-- Linear output projection. -/
  outputLayer : LinearSpec α hiddenSize outputSize

-- LSTM model for classification (many-to-one)
/--
Bundle of parameters for a many-to-one LSTM classifier.

The classifier head is applied to the final hidden state.
-/
structure Classifier (α : Type) (inputSize hiddenSize numClasses : Nat) where
  /-- Recurrent cell parameters. -/
  lstm : LSTMSpec α inputSize hiddenSize
  /-- Linear classifier head. -/
  classifier : LinearSpec α hiddenSize numClasses

-- LSTM model for sequence generation (many-to-many)
/--
Bundle of parameters for a many-to-many LSTM generator (language-model style).

This includes an (embedding) linear map, recurrent core, and output projection back to vocabulary.
-/
structure Generator (α : Type) (vocabSize hiddenSize : Nat) where
  /-- Token projection used by this one-hot specification. -/
  embedding : LinearSpec α vocabSize hiddenSize
  /-- Recurrent cell parameters. -/
  lstm : LSTMSpec α hiddenSize hiddenSize
  /-- Projection from hidden states to vocabulary logits. -/
  outputProjection : LinearSpec α hiddenSize vocabSize

/--
Bundle of parameters for a bidirectional LSTM model with an output head.

The head consumes the concatenation of forward and backward hidden states.
PyTorch analogue: `nn.LSTM(..., bidirectional=true)` plus a linear projection.
-/
structure BidirectionalModel (α : Type) (inputSize hiddenSize outputSize : Nat) where
  /-- Recurrent cell for the original sequence order. -/
  forwardLstm : LSTMSpec α inputSize hiddenSize
  /-- Recurrent cell for the reversed sequence order. -/
  backwardLstm : LSTMSpec α inputSize hiddenSize
  /-- Projection from concatenated forward and backward states. -/
  outputLayer : LinearSpec α (hiddenSize + hiddenSize) outputSize

/--
Bundle of parameters for a stacked LSTM language model with deterministic dropout.

This model uses an array of LSTM layers (all with `hiddenSize` input/output) and applies an
evaluation-mode dropout step between the recurrent stack and the output projection.
-/
structure LanguageModel (α : Type) (vocabSize hiddenSize : Nat) where
  /-- Token projection used by this one-hot specification. -/
  embedding : LinearSpec α vocabSize hiddenSize
  /-- Recurrent layers, ordered from input to output. -/
  layers : Array (LSTMSpec α hiddenSize hiddenSize)
  /-- Projection from hidden states to vocabulary logits. -/
  outputProjection : LinearSpec α hiddenSize vocabSize
  /-- Dropout probability used between the recurrent stack and output projection. -/
  dropoutRate : α

-- Forward pass for simple LSTM model
/--
One-step forward pass for `Lstm.Model`.

Given an input vector and the previous LSTM state `(hidden, cell)`, compute `(output, new_state)`.
PyTorch analogue: `nn.LSTMCell` step followed by a `nn.linear` head.
-/
def Model.forward {inputSize hiddenSize outputSize : Nat}
  (model : Model α inputSize hiddenSize outputSize)
  (input : Tensor α [inputSize])
  (state : LSTMState α hiddenSize) :
  (Tensor α [outputSize] × LSTMState α hiddenSize) :=
  let nextState := lstmCellSpec model.lstm input state
  let output := linearSpec model.outputLayer nextState.hidden
  (output, nextState)

-- Forward pass for simple LSTM model on sequences
/--
Sequence forward pass for `Lstm.Model`.

Runs the LSTM over all timesteps (time-major), applies the output head to each hidden state, and
returns `(outputs, final_state)`.
-/
def Model.forwardSequence {seqLen inputSize hiddenSize outputSize : Nat}
  (model : Model α inputSize hiddenSize outputSize)
  (inputs : Tensor α [seqLen, inputSize])
  (initialState : LSTMState α hiddenSize) :
  (Tensor α [seqLen, outputSize] × LSTMState α hiddenSize) :=
  let (hiddenStates, finalState) := lstmSequenceSpec model.lstm inputs initialState
  let outputs := Tensor.mapEach ([seqLen])
    (linearSpec model.outputLayer) hiddenStates
  (outputs, finalState)

/-!
### Backward pass (BPTT) for the simple LSTM sequence model

This is the model-level analogue of `Spec.lstmSequenceBackwardSpec`. The only extra work we do
here is to backprop through the per-timestep output projection and feed its gradient into the LSTM
sequence backward pass.
-/

namespace Internal

/-- Backprop through the time-distributed linear head and produce hidden-state gradients. -/
def timeDistributedLinearBackward
  {seqLen hiddenSize outputSize : Nat}
  (layer : LinearSpec α hiddenSize outputSize)
  (hiddens : Tensor α [seqLen, hiddenSize])
  (outputGrad : Tensor α [seqLen, outputSize]) :
  (LinearGrads α hiddenSize outputSize × Tensor α [seqLen, hiddenSize]) :=
  let step (i : Fin seqLen) (acc : LinearGrads α hiddenSize outputSize) :=
    let hi := get hiddens i
    let outputGradI := get outputGrad i
    let (weightGrad, biasGrad, hiddenGrad) := linearBackwardSpec layer hi outputGradI
    ({ weight := addSpec acc.weight weightGrad
       bias := addSpec acc.bias biasGrad }, hiddenGrad)
  let init : LinearGrads α hiddenSize outputSize := {
    weight := fill 0 ([outputSize, hiddenSize]),
    bias := fill 0 ([outputSize])
  }
  let (linearGrads, hiddenGrads) := Sequence.mapAccum seqLen init step
  (linearGrads, Tensor.dim hiddenGrads.getScalar)

end Internal

open Internal

/--
Backward pass for `Lstm.Model.forwardSequence`.

Returns:
- parameter gradients (`Lstm.Grads`)
- gradient w.r.t. input sequence (`dInputs`)
- gradient w.r.t. initial recurrent state (`dInitialState`)
-/
def Model.backward
  {seqLen inputSize hiddenSize outputSize : Nat}
  (model : Model α inputSize hiddenSize outputSize)
  (inputs : Tensor α [seqLen, inputSize])
  (initialState : LSTMState α hiddenSize)
  (outputGrad : Tensor α [seqLen, outputSize]) :
  (Grads α inputSize hiddenSize outputSize ×
    Tensor α [seqLen, inputSize] ×
    LSTMState α hiddenSize) :=
  let (hiddens, _) := lstmSequenceSpec model.lstm inputs initialState
  let (headGrads, hiddenGrad) := timeDistributedLinearBackward (α := α) (seqLen := seqLen)
    (hiddenSize := hiddenSize) (outputSize := outputSize) model.outputLayer hiddens outputGrad
  let (forgetWeight, forgetBias, inputWeight, inputBias, candidateWeight, candidateBias,
       outputWeight, outputBias, inputGrad, initialStateGrad) :=
    lstmSequenceBackwardSpec model.lstm inputs initialState hiddenGrad
  let cell : CellGrads α inputSize hiddenSize :=
    { forgetWeight, forgetBias, inputWeight, inputBias, candidateWeight, candidateBias,
      outputWeight, outputBias }
  ({ cell, output := headGrads }, inputGrad, initialStateGrad)

/--
MSE loss for the simple LSTM sequence model.

This runs `Lstm.Model.forwardSequence` and compares the predicted output sequence against
`targets` using `mseSpec`.
-/
def Model.mseLoss
  {seqLen inputSize hiddenSize outputSize : Nat}
  (model : Model α inputSize hiddenSize outputSize)
  (inputs : Tensor α [seqLen, inputSize])
  (targets : Tensor α [seqLen, outputSize])
  (initialState : LSTMState α hiddenSize) : α :=
  let (prediction, _) := model.forwardSequence inputs initialState
  mseSpec prediction targets

/--
Compute `(loss, grads)` for the simple LSTM sequence model under MSE.

This is the “full training API” building block: an optimizer (SGD/Adam) can consume these grads.
-/
def Model.mseGrad
  {seqLen inputSize hiddenSize outputSize : Nat}
  (model : Model α inputSize hiddenSize outputSize)
  (inputs : Tensor α [seqLen, inputSize])
  (targets : Tensor α [seqLen, outputSize])
  (initialState : LSTMState α hiddenSize) :
  (α × Grads α inputSize hiddenSize outputSize) :=
  let (prediction, _) := model.forwardSequence inputs initialState
  let loss := mseSpec prediction targets
  let predictionGrad := mseDerivSpec prediction targets
  let (grads, _, _) := model.backward inputs initialState predictionGrad
  (loss, grads)

-- Forward pass for LSTM classifier (many-to-one)
/--
Forward pass for an `Lstm.Classifier` (many-to-one).

This uses the final hidden state of the LSTM sequence as the classifier input.
-/
def Classifier.forward {seqLen inputSize hiddenSize numClasses : Nat}
  (model : Classifier α inputSize hiddenSize numClasses)
  (inputs : Tensor α [seqLen, inputSize])
  (initialState : LSTMState α hiddenSize) :
  Tensor α [numClasses] :=
  let (_, finalState) := lstmSequenceSpec model.lstm inputs initialState
  linearSpec model.classifier finalState.hidden

/-!
### Backward for the classifier head (many-to-one)

The classifier only consumes the final hidden state. We express that by feeding a gradient sequence
that is zero everywhere except the last timestep.
-/

/--
Backward pass for an `Lstm.Classifier` (many-to-one).

This backprops through the classifier head, then runs an LSTM sequence backward pass where the
hidden-state gradient is zero at all timesteps except the last.
-/
def Classifier.backward
  {seqLen inputSize hiddenSize numClasses : Nat}
  (model : Classifier α inputSize hiddenSize numClasses)
  (inputs : Tensor α [seqLen, inputSize])
  (initialState : LSTMState α hiddenSize)
  (logitGrad : Tensor α [numClasses]) :
  (ClassifierGrads α inputSize hiddenSize numClasses ×
    Tensor α [seqLen, inputSize] × LSTMState α hiddenSize) :=
  let (hiddens, _) := lstmSequenceSpec model.lstm inputs initialState
  let finalHidden :=
    if h0 : seqLen = 0 then
      fill 0 ([hiddenSize])
    else
      get hiddens ⟨seqLen - 1, by
        have : seqLen - 1 < seqLen := Nat.sub_lt (Nat.pos_of_ne_zero h0) (by decide : 0 < 1)
        simpa using this⟩
  let (classifierWeight, classifierBias, finalHiddenGrad) :=
    linearBackwardSpec model.classifier finalHidden logitGrad
  let classifierGrads : LinearGrads α hiddenSize numClasses :=
    { weight := classifierWeight, bias := classifierBias }
  let hiddenGrad :=
    if h0 : seqLen = 0 then
      fill 0 ([seqLen, hiddenSize])
    else
      Tensor.dim (fun i =>
        if _ : i.val = seqLen - 1 then finalHiddenGrad else fill 0 ([hiddenSize]))
  let (forgetWeight, forgetBias, inputWeight, inputBias, candidateWeight, candidateBias,
       outputWeight, outputBias, inputGrad, initialStateGrad) :=
    lstmSequenceBackwardSpec model.lstm inputs initialState hiddenGrad
  let cell : CellGrads α inputSize hiddenSize :=
    { forgetWeight, forgetBias, inputWeight, inputBias, candidateWeight, candidateBias,
      outputWeight, outputBias }
  ({ cell, classifier := classifierGrads }, inputGrad, initialStateGrad)

-- Forward pass for LSTM generator (many-to-many)
/--
Forward pass for an `Lstm.Generator` (many-to-many).

This applies an embedding linear map to each token vector, runs the LSTM, and projects each hidden
state back into vocabulary space.
-/
def Generator.forward {seqLen vocabSize hiddenSize : Nat}
  (model : Generator α vocabSize hiddenSize)
  (inputTokens : Tensor α [seqLen, vocabSize])
  (initialState : LSTMState α hiddenSize) :
  (Tensor α [seqLen, vocabSize] × LSTMState α hiddenSize) :=
  let embedded := Tensor.mapEach ([seqLen]) (linearSpec model.embedding) inputTokens
  let (hiddenStates, finalState) := lstmSequenceSpec model.lstm embedded initialState
  let outputs := Tensor.mapEach ([seqLen])
    (linearSpec model.outputProjection) hiddenStates
  (outputs, finalState)

-- Forward pass for bidirectional LSTM
/--
Forward pass for a bidirectional LSTM model (time-major).

This runs a forward LSTM on the sequence, a backward LSTM on the reversed sequence, concatenates
the two hidden streams per timestep, and applies an output head.
-/
def BidirectionalModel.forward {seqLen inputSize hiddenSize outputSize : Nat}
  (model : BidirectionalModel α inputSize hiddenSize outputSize)
  (inputs : Tensor α [seqLen, inputSize])
  (forwardState : LSTMState α hiddenSize)
  (backwardState : LSTMState α hiddenSize) :
  Tensor α [seqLen, outputSize] :=
  let (forwardStates, _) := lstmSequenceSpec model.forwardLstm inputs forwardState
  let reversedInputs := Tensor.reverseAxis 0 inputs
  let (reversedBackwardStates, _) :=
    lstmSequenceSpec model.backwardLstm reversedInputs backwardState
  let backwardStates := Tensor.reverseAxis 0 reversedBackwardStates
  let combinedStates := Tensor.zipEach ([seqLen])
    ([(hiddenSize + hiddenSize)])
    (Tensor.concatAxisSpec .scalar) forwardStates backwardStates
  Tensor.mapEach ([seqLen]) (linearSpec model.outputLayer) combinedStates

-- Multi-layer LSTM forward pass (stack multiple LSTM layers)
/--
Forward pass for a `Lstm.StackedModel`.

This runs the first layer on the input sequence, then threads the resulting hidden stream through
each additional hidden layer, and finally applies the output head per timestep.
-/
def StackedModel.forward {seqLen inputSize hiddenSize outputSize numLayers : Nat}
  (model : StackedModel α inputSize hiddenSize outputSize numLayers)
  (inputs : Tensor α [seqLen, inputSize])
  (initialStates : Fin numLayers → LSTMState α hiddenSize) (hLayers : 0 < numLayers) :
  (Tensor α [seqLen, outputSize] × (Fin numLayers → LSTMState α hiddenSize)) :=
  let rec processHiddenLayers (layer : Nat)
    (layerInput : Tensor α [seqLen, hiddenSize])
    (states : Fin numLayers → LSTMState α hiddenSize) :
    (Tensor α [seqLen, hiddenSize] × (Fin numLayers → LSTMState α hiddenSize)) :=
    if hLayer : layer < numLayers - 1 then
      let layerIndex : Fin (numLayers - 1) := ⟨layer, hLayer⟩
      have hState : layer + 1 < numLayers := by
        have hState' : layer + 1 ≤ numLayers - 1 := Nat.succ_le_of_lt hLayer
        exact lt_of_le_of_lt hState' (Nat.sub_one_lt (Nat.ne_of_gt hLayers))
      let stateIndex : Fin numLayers := ⟨layer + 1, hState⟩
      let (layerOutput, nextState) :=
        lstmSequenceSpec (model.hiddenLayers layerIndex) layerInput (states stateIndex)
      let updatedStates := Function.update states stateIndex nextState
      processHiddenLayers (layer + 1) layerOutput updatedStates
    else
      (layerInput, states)

  let firstLayerIndex : Fin numLayers := ⟨0, hLayers⟩
  let (firstOutput, firstState) :=
    lstmSequenceSpec model.firstLayer inputs (initialStates firstLayerIndex)
  let updatedInitialStates := Function.update initialStates firstLayerIndex firstState

  let (finalHidden, finalStates) := processHiddenLayers 0 firstOutput updatedInitialStates
  let outputs := Tensor.mapEach ([seqLen])
    (linearSpec model.outputLayer) finalHidden
  (outputs, finalStates)

-- LSTM Language Model forward pass with teacher forcing
/--
Forward pass for `Lstm.LanguageModel` (teacher forcing, time-major).

This runs the embedding, then a stack of LSTM layers with provided initial states, applies
evaluation-mode dropout (`dropoutInferenceSpec`), and projects to vocabulary logits.
-/
def LanguageModel.forward {seqLen vocabSize hiddenSize : Nat}
  (model : LanguageModel α vocabSize hiddenSize)
  (inputTokens : Tensor α [seqLen, vocabSize])
  (initialStates : Array (LSTMState α hiddenSize)) :
  Option (Tensor α [seqLen, vocabSize] × Array (LSTMState α hiddenSize)) := do
  let embedded :=
    Tensor.mapEach ([seqLen]) (linearSpec model.embedding) inputTokens
  let rec processLayers (layers : List (LSTMSpec α hiddenSize hiddenSize))
    (states : List (LSTMState α hiddenSize))
    (layerInput : Tensor α [seqLen, hiddenSize]) :
    Option (Tensor α [seqLen, hiddenSize] × List (LSTMState α hiddenSize)) :=
    match layers, states with
    | [], [] => some (layerInput, [])
    | layer :: remainingLayers, state :: remainingStates => do
      let (layerOutput, nextState) := lstmSequenceSpec layer layerInput state
      let (finalOutput, finalStates) ←
        processLayers remainingLayers remainingStates layerOutput
      pure (finalOutput, nextState :: finalStates)
    | _, _ => none
  let (lstmOutput, finalStates) ←
    processLayers model.layers.toList initialStates.toList embedded
  let droppedOutput := dropoutInferenceSpec (p := model.dropoutRate) lstmOutput
  let logits := Tensor.mapEach ([seqLen])
    (linearSpec model.outputProjection) droppedOutput
  pure (logits, finalStates.toArray)

/--
Package `Lstm.Model` as a shape-indexed module.

The Python expression records the intended runtime analogue; `forward` remains the mathematical
meaning of the module.
-/
def Model.toModule {seqLen inputSize hiddenSize outputSize : Nat}
  (model : Model α inputSize hiddenSize outputSize) :
  Spec.Module α ([seqLen, inputSize]) ([seqLen, outputSize]) :=
{
  forward := fun inputs =>
    let initialState : LSTMState α hiddenSize := {
      hidden := fill 0 ([hiddenSize]),
      cell := fill 0 ([hiddenSize])
    }
    (model.forwardSequence inputs initialState).1,
  kind := "SimpleLSTM",
  pythonExpr := s!"SimpleLSTM(input_size={inputSize}, hidden_size={hiddenSize}, output_size={outputSize})"
}

/--
Package `Lstm.Classifier` as an `Spec.Module`.

PyTorch analogue: `nn.LSTM` feeding a `nn.linear` classifier head.
-/
def Classifier.toModule {seqLen inputSize hiddenSize numClasses : Nat}
  (model : Classifier α inputSize hiddenSize numClasses) :
  Spec.Module α ([seqLen, inputSize]) ([numClasses]) :=
{
  forward := fun inputs =>
    let initialState : LSTMState α hiddenSize := {
      hidden := fill 0 ([hiddenSize]),
      cell := fill 0 ([hiddenSize])
    }
    model.forward inputs initialState,
  kind := "LSTMClassifier",
  pythonExpr := s!"LSTMClassifier(input_size={inputSize}, hidden_size={hiddenSize}, " ++
        s!"num_classes={numClasses})"
}

/--
Package `Lstm.BidirectionalModel` as an `Spec.Module`.

PyTorch analogue: `nn.LSTM(..., bidirectional=true)` feeding a per-timestep linear head.
-/
def BidirectionalModel.toModule {seqLen inputSize hiddenSize outputSize : Nat}
  (model : BidirectionalModel α inputSize hiddenSize outputSize) :
  Spec.Module α ([seqLen, inputSize]) ([seqLen, outputSize]) :=
{
  forward := fun inputs =>
    let initialState : LSTMState α hiddenSize := {
      hidden := fill 0 ([hiddenSize]),
      cell := fill 0 ([hiddenSize])
    }
    model.forward inputs initialState initialState,
  kind := "BiLSTM",
  pythonExpr := s!"SimpleLSTM(input_size={inputSize}, hidden_size={hiddenSize}, output_size={outputSize}, " ++
        s!"bidirectional=True)"
}

end Lstm

end Spec

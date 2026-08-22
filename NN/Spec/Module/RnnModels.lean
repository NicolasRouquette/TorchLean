/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Loss
public import NN.Spec.Module.Linear
public import NN.Spec.Module.Rnn

/-!
# Recurrent Models

This file builds higher-level recurrent architectures by composing module specifications:

- sequence‑to‑sequence: RNN over inputs + per‑step linear projection,
- many‑to‑one classification: RNN + classifier head on the final hidden state,
- bidirectional variants (where supported by module specs).

The cell dynamics live in `NN.Spec.Layers.Rnn`; this module defines model-level compositions and
reference forward and backward functions.
-/

@[expose] public section


namespace Spec

open Tensor
open Spec.Module

variable {α : Type} [Context α]

namespace Rnn

/--
A simple sequence-to-sequence RNN "model wiring" expressed as a `Spec.Module.Chain`.

This composes `Spec.Module.rnn` with a linear projection applied independently at each timestep,
so the overall model maps:

- input shape:  `[seqLen, inputSize]`
- output shape: `[seqLen, outputSize]`

PyTorch analogue: applying `nn.RNN` (or a custom recurrent cell) followed by a `nn.linear` at each
time step.
-/
def sequence
  {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen inputSize hiddenSize outputSize : Nat}
  (rnnSpec : RNNSpec α inputSize hiddenSize)
  (linearSpec : LinearSpec α hiddenSize outputSize) :
  Spec.Module.Chain α (.dim seqLen (.dim inputSize .scalar)) (.dim seqLen (.dim outputSize .scalar)) :=
  let rnnModule := Spec.Module.rnn rnnSpec
  let linearModule := Spec.Module.mapLeading (Spec.Module.linear linearSpec)
  Spec.Module.Chain.single rnnModule
    |>.append linearModule

/--
A many-to-one RNN classifier expressed as a `Spec.Module.Chain`.

This runs an RNN over the input sequence and then applies a linear classifier head to the final
hidden state.

PyTorch analogue: `nn.RNN` (or `nn.GRU`/`nn.LSTM`) feeding a `nn.linear` head, taking the last
hidden/output.
-/
def classifier
  {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen inputSize hiddenSize numClasses : Nat}
  (rnnSpec : RNNSpec α inputSize hiddenSize)
  (classifierHead : LinearSpec α hiddenSize numClasses)
  (h : seqLen ≠ 0) :
  Spec.Module.Chain α (.dim seqLen (.dim inputSize .scalar)) (.dim numClasses .scalar) :=
  let rnnModule := Spec.Module.rnn rnnSpec
  let lastOutput := Spec.Module.selectLeading (⟨Nat.pred seqLen, Nat.pred_lt h⟩)
  let classifierModule := Spec.Module.linear classifierHead
  Spec.Module.Chain.single rnnModule
    |>.append lastOutput
    |>.append classifierModule

/--
A two-layer RNN encoder with a per-step linear projection, expressed as a `Spec.Module.Chain`.

The second recurrent layer consumes the hidden stream of the first.
-/
def stacked
  {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen inputSize hiddenSize outputSize : Nat}
  (firstSpec : RNNSpec α inputSize hiddenSize)
  (secondSpec : RNNSpec α hiddenSize hiddenSize)
  (linearSpec : LinearSpec α hiddenSize outputSize) :
  Spec.Module.Chain α (.dim seqLen (.dim inputSize .scalar)) (.dim seqLen (.dim outputSize .scalar)) :=
  let firstModule := Spec.Module.rnn firstSpec
  let secondModule := Spec.Module.rnn secondSpec
  let linearModule := Spec.Module.mapLeading (Spec.Module.linear linearSpec)
  Spec.Module.Chain.single firstModule
    |>.append secondModule
    |>.append linearModule

/--
A simple RNN language model spec: "embedding" linear map, RNN core, and output projection.

This file treats embedding/projection as `LinearSpec`s. A common spec-level usage is that tokens
are one-hot vectors of length `vocabSize`, so the embedding is just a matrix multiply.

PyTorch analogue: `nn.Embedding` (conceptually) + `nn.RNN` + `nn.linear` vocabulary projection.
-/
def languageModel
  {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen vocabSize hiddenSize : Nat}
  (embeddingSpec : LinearSpec α vocabSize hiddenSize)
  (rnnSpec : RNNSpec α hiddenSize hiddenSize)
  (outputSpec : LinearSpec α hiddenSize vocabSize) :
  Spec.Module.Chain α (.dim seqLen (.dim vocabSize .scalar)) (.dim seqLen (.dim vocabSize .scalar)) :=
  let embeddingModule := Spec.Module.mapLeading (Spec.Module.linear embeddingSpec)
  let rnnModule := Spec.Module.rnn rnnSpec
  let outputModule := Spec.Module.mapLeading (Spec.Module.linear outputSpec)
  Spec.Module.Chain.single embeddingModule
    |>.append rnnModule
    |>.append outputModule

/--
Bundle of parameters for a simple single-layer RNN model with a linear output head.

This is a "record of specs" representation, as opposed to the `Spec.Module.Chain` representation used
above.
-/
structure Model (α : Type) (inputSize hiddenSize outputSize : Nat) where
  /-- Recurrent cell parameters. -/
  rnn : RNNSpec α inputSize hiddenSize
  /-- Linear output projection. -/
  outputLayer : LinearSpec α hiddenSize outputSize

/-- Parameter gradients for a recurrent model and its linear output head. -/
structure Grads (α : Type) (inputSize hiddenSize outputSize : Nat) where
  /-- Gradient of the recurrent weight matrix. -/
  rnnWeight : Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar))
  /-- Gradient of the recurrent bias. -/
  rnnBias : Tensor α (.dim hiddenSize .scalar)
  /-- Gradient of the output projection weight. -/
  outputWeight : Tensor α (.dim outputSize (.dim hiddenSize .scalar))
  /-- Gradient of the output projection bias. -/
  outputBias : Tensor α (.dim outputSize .scalar)

/--
Bundle of parameters for a multi-layer RNN model.

`layers` is indexed by `Fin numLayers` and selects the appropriate input size for the first layer
versus subsequent layers.
-/
structure StackedModel (α : Type) (inputSize hiddenSize outputSize numLayers : Nat) where
  /-- Layer stack. -/
  layers :
    (i : Fin numLayers) →
      RNNSpec α (if i.val = 0 then inputSize else hiddenSize) hiddenSize
  /-- Linear output projection. -/
  outputLayer : LinearSpec α hiddenSize outputSize

/--
Bundle of parameters for a many-to-one RNN classifier.

The classifier head is a linear layer applied to the final hidden state.
-/
structure Classifier (α : Type) (inputSize hiddenSize numClasses : Nat) where
  /-- Recurrent cell parameters. -/
  rnn : RNNSpec α inputSize hiddenSize
  /-- Linear classifier head. -/
  classifier : LinearSpec α hiddenSize numClasses

/--
Bundle of parameters for a many-to-many RNN generator (language-model style).

This includes a (linear) embedding, recurrent core, and output projection back to vocabulary.
-/
structure Generator (α : Type) (vocabSize hiddenSize : Nat) where
  /-- Token projection used by this one-hot specification. -/
  embedding : LinearSpec α vocabSize hiddenSize
  /-- Recurrent cell parameters. -/
  rnn : RNNSpec α hiddenSize hiddenSize
  /-- Projection from hidden states to vocabulary logits. -/
  outputProjection : LinearSpec α hiddenSize vocabSize

/--
Bundle of parameters for a bidirectional RNN model with an output head.

The output head consumes the concatenation of forward and backward hidden states.
-/
structure BidirectionalModel (α : Type) (inputSize hiddenSize outputSize : Nat) where
  /-- Recurrent cell for the original sequence order. -/
  forwardRnn : RNNSpec α inputSize hiddenSize
  /-- Recurrent cell for the reversed sequence order. -/
  backwardRnn : RNNSpec α inputSize hiddenSize
  /-- Projection from concatenated forward and backward states. -/
  outputLayer : LinearSpec α (hiddenSize + hiddenSize) outputSize

/--
One-step forward pass for `Rnn.Model`.

Given an input vector and current hidden state, compute the output and next state using the RNN cell
and the linear head.
-/
def Model.forward {inputSize hiddenSize outputSize : Nat}
  (model : Model α inputSize hiddenSize outputSize)
  (input : Tensor α (.dim inputSize .scalar))
  (hidden : Tensor α (.dim hiddenSize .scalar)) :
  (Tensor α (.dim outputSize .scalar) × Tensor α (.dim hiddenSize .scalar)) :=
  let nextHidden := rnnCellSpec model.rnn input hidden
  let output := linearSpec model.outputLayer nextHidden
  (output, nextHidden)

/--
Sequence forward pass for `Rnn.Model`.

Runs the recurrent cell over the full sequence, applies the output layer at each time step, and
returns both the per-step outputs and the final hidden state.
-/
def Model.forwardSequence {seqLen inputSize hiddenSize outputSize : Nat}
  (model : Model α inputSize hiddenSize outputSize)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (initialHidden : Tensor α (.dim hiddenSize .scalar)) (h : 0 < seqLen) :
  (Tensor α (.dim seqLen (.dim outputSize .scalar)) × Tensor α (.dim hiddenSize .scalar))  :=
  let hiddenStates := rnnSequenceSpec model.rnn inputs initialHidden
  let outputs := Tensor.mapLeading (.dim seqLen .scalar)
    (linearSpec model.outputLayer) hiddenStates
  have hLast : seqLen - 1 < seqLen := by
    simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt h)
  let finalHidden := getAtSpec hiddenStates ⟨seqLen - 1, hLast⟩
  (outputs, finalHidden)

/--
Forward pass for an `Rnn.Classifier` (many-to-one).

This runs the recurrent core over the input sequence and feeds the last hidden state to the
classifier head.
-/
def Classifier.forward {seqLen inputSize hiddenSize numClasses : Nat}
  (model : Classifier α inputSize hiddenSize numClasses)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (initialHidden : Tensor α (.dim hiddenSize .scalar)) (h : 0 < seqLen) :
  Tensor α (.dim numClasses .scalar) :=
  let hiddenStates := rnnSequenceSpec model.rnn inputs initialHidden
  have hLast : seqLen - 1 < seqLen := by
    simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt h)
  let finalHidden := getAtSpec hiddenStates ⟨seqLen - 1, hLast⟩
  linearSpec model.classifier finalHidden

/--
Forward pass for an `Rnn.Generator` (many-to-many).

This applies an "embedding" linear map to each token, runs the RNN, and projects each hidden state
back into vocabulary space.
-/
def Generator.forward {seqLen vocabSize hiddenSize : Nat}
  (model : Generator α vocabSize hiddenSize)
  (inputTokens : Tensor α (.dim seqLen (.dim vocabSize .scalar)))
  (initialHidden : Tensor α (.dim hiddenSize .scalar)) (h : 0 < seqLen) :
  (Tensor α (.dim seqLen (.dim vocabSize .scalar)) × Tensor α (.dim hiddenSize .scalar)) :=
  let embedded := Tensor.mapLeading (.dim seqLen .scalar) (linearSpec model.embedding) inputTokens
  let hiddenStates := rnnSequenceSpec model.rnn embedded initialHidden
  let outputs := Tensor.mapLeading (.dim seqLen .scalar)
    (linearSpec model.outputProjection) hiddenStates
  have hLast : seqLen - 1 < seqLen := by
    simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt h)
  let finalHidden := getAtSpec hiddenStates ⟨seqLen - 1, hLast⟩
  (outputs, finalHidden)

/--
Forward pass for a bidirectional RNN model.

This runs a forward RNN on the sequence, a backward RNN on the reversed sequence, concatenates the
two state streams per time step, and applies the output head.
-/
def BidirectionalModel.forward {seqLen inputSize hiddenSize outputSize : Nat}
  (model : BidirectionalModel α inputSize hiddenSize outputSize)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (forwardHidden : Tensor α (.dim hiddenSize .scalar))
  (backwardHidden : Tensor α (.dim hiddenSize .scalar)) :
  Tensor α (.dim seqLen (.dim outputSize .scalar)) :=
  let forwardStates := rnnSequenceSpec model.forwardRnn inputs forwardHidden
  let reversedInputs := Tensor.reverseLeadingAxis inputs
  let reversedBackwardStates := rnnSequenceSpec model.backwardRnn reversedInputs backwardHidden
  let backwardStates := Tensor.reverseLeadingAxis reversedBackwardStates
  let combinedStates := Tensor.zipWithLeading (.dim seqLen .scalar)
    (.dim (hiddenSize + hiddenSize) .scalar)
    Tensor.concatLeadingAxisSpec forwardStates backwardStates
  Tensor.mapLeading (.dim seqLen .scalar) (linearSpec model.outputLayer) combinedStates

-- One-step helper used by some compact examples (single cell update + output projection).
/--
One-step helper: run a single RNN cell update and apply an output projection.

This is used by some compact examples that do not build a full `Spec.Module.Chain` or multi-layer bundle.
-/
def cellWithHead {inputSize hiddenSize outputSize : Nat}
  (cell : RNNSpec α inputSize hiddenSize)
  (outputLayer : LinearSpec α hiddenSize outputSize)
  (inputs : Tensor α (.dim inputSize .scalar))
  (hidden : Tensor α (.dim hiddenSize .scalar)) :
  (Tensor α (.dim outputSize .scalar) × Tensor α (.dim hiddenSize .scalar)) :=
  let nextHidden := rnnCellSpec cell inputs hidden
  let output := linearSpec outputLayer nextHidden
  (output, nextHidden)

-- Backward pass for simple RNN model
/--
Backward pass for `Rnn.Model` over a full sequence.

Returns parameter gradients together with the gradient of the input sequence.

This is a spec-level reference implementation; performance is not a goal here.
-/
def Model.backward {seqLen inputSize hiddenSize outputSize : Nat}
  (model : Model α inputSize hiddenSize outputSize)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (hiddenStates : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
  (gradOutputs : Tensor α (.dim seqLen (.dim outputSize .scalar))) (h : 0 < seqLen) :
  Grads α inputSize hiddenSize outputSize ×
    Tensor α (.dim seqLen (.dim inputSize .scalar)) :=

  let gradHiddenFromOutput := Tensor.mapLeading (.dim seqLen .scalar)
    (fun gradOutput => linearInputDerivSpec model.outputLayer.weights gradOutput) gradOutputs
  let gradOutputWeights := Tensor.sumLeadingAxis
    (Tensor.zipWithLeading (.dim seqLen .scalar)
      (Shape.dim outputSize (Shape.dim hiddenSize Shape.scalar))
      linearWeightsDerivSpec hiddenStates gradOutputs) h.ne'
  let gradOutputBias := Tensor.sumLeadingAxis gradOutputs h.ne'

  let initialHidden := fill 0 (.dim hiddenSize .scalar)
  let (gradRnnWeights, gradRnnBias, gradInputs, _gradInitialHidden) :=
    rnnSequenceBackwardSpec model.rnn inputs initialHidden hiddenStates gradHiddenFromOutput

  ( { rnnWeight := gradRnnWeights
      rnnBias := gradRnnBias
      outputWeight := gradOutputWeights
      outputBias := gradOutputBias },
    gradInputs )

namespace Internal

/--
Map a scalar-valued function over two aligned sequences, producing a sequence of scalars.

This helper lifts a scalar comparison over aligned sequence elements.
-/
def mapSequence2 {seqLen dim1 dim2 : Nat}
  (f : Tensor α (.dim dim1 .scalar) → Tensor α (.dim dim2 .scalar) → α)
  (leftSeq : Tensor α (.dim seqLen (.dim dim1 .scalar)))
  (rightSeq : Tensor α (.dim seqLen (.dim dim2 .scalar))) :
  Tensor α (.dim seqLen .scalar) :=
  match leftSeq, rightSeq with
  | Tensor.dim func, Tensor.dim rightFn =>
    Tensor.dim (fun i => Tensor.scalar (f (func i) (rightFn i)))

end Internal

-- Loss function for sequence classification
/--
Mean cross-entropy loss over a sequence of class-probability predictions.

This is the spec-level analogue of a per-time-step classification loss, averaged across steps.
PyTorch analogue: `torch.nn.CrossEntropyLoss` applied per step and then averaged.
-/
def classificationLoss {seqLen numClasses : Nat}
  [Shape.HasNonemptyAxis 0 (.dim numClasses .scalar)]
  (predictions : Tensor α (.dim seqLen (.dim numClasses .scalar)))
  (targets : Tensor α (.dim seqLen (.dim numClasses .scalar))) :
  α :=
  let losses := Internal.mapSequence2 (crossEntropySpec 0) predictions targets
  meanSpec losses

/--
Package an `Rnn.Model` as a shape-indexed module.

The Python expression records the intended runtime analogue; `forward` remains the mathematical
meaning of the module.
-/
def Model.toModule {seqLen inputSize hiddenSize outputSize : Nat}
  (model : Model α inputSize hiddenSize outputSize) (h : 0 < seqLen) :
  Spec.Module α (.dim seqLen (.dim inputSize .scalar)) (.dim seqLen (.dim outputSize .scalar)) :=
{
  forward := fun inputs =>
    let initialHidden := fill 0 (.dim hiddenSize .scalar)
    (model.forwardSequence inputs initialHidden h).1,
  kind := "SimpleRNN",
  pythonExpr := s!"SimpleRNN(input_size={inputSize}, hidden_size={hiddenSize}, output_size={outputSize})"
}

/--
Package an `Rnn.Classifier` as an `Spec.Module`.

This plugs the classifier into the common module pipeline and records a PyTorch-oriented summary.
-/
def Classifier.toModule {seqLen inputSize hiddenSize numClasses : Nat}
  (model : Classifier α inputSize hiddenSize numClasses) (h : 0 < seqLen) :
  Spec.Module α (.dim seqLen (.dim inputSize .scalar)) (.dim numClasses .scalar) :=
{
  forward := fun inputs =>
    let initialHidden := fill 0 (.dim hiddenSize .scalar)
    model.forward inputs initialHidden h,
  kind := "RNNClassifier",
  pythonExpr := s!"RNNClassifier(input_size={inputSize}, hidden_size={hiddenSize}, num_classes={numClasses})"
}

end Rnn

end Spec

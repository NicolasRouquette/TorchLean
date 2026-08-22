/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Dropout
public import NN.Spec.Module.Linear
public import NN.Spec.Module.Rnn

/-!
# Gated Recurrent Models

TorchLean provides GRU layers/cells in `NN.Spec.Layers.Gru`. This file builds *models* on top of
that layer API: common compositions, heads, and a couple of end-to-end forward/backward routines.

Higher‑level GRU architectures built from module specs (`Spec.Module.Chain`):

- sequence‑to‑sequence outputs,
- classifier heads (many‑to‑one),
- multi‑layer compositions.

GRU cell equations are in `NN/Spec/Layers/Gru.lean`; this file is primarily “wiring”.

References:

- Cho et al. (2014), "Learning Phrase Representations using RNN Encoder–Decoder for Statistical
  Machine Translation" (introduces GRU): https://arxiv.org/abs/1406.1078
- Chung et al. (2014), "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence
  Modeling" (GRU variants/ablation): https://arxiv.org/abs/1412.3555
- PyTorch `nn.GRUCell` docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.GRUCell.html
- PyTorch `nn.GRU` docs: https://pytorch.org/docs/stable/generated/torch.nn.GRU.html

PyTorch analogy: this corresponds to wiring `torch.nn.GRU` with linear heads and pooling over time
(e.g. last hidden state for classification). This is an architectural comparison only. The
recurrent core is the original Cho reset-before GRU from `NN.Spec.Layers.Gru`; PyTorch uses a
reset-after candidate-state equation, so its checkpoints are not equation-compatible with these
models without an explicit conversion.
-/

@[expose] public section

namespace Spec

open Tensor
open Spec.Module

variable {α : Type} [Context α]

namespace Gru

/-- A sequence-to-sequence GRU model, written as a `Spec.Module.Chain`.

Pipeline:
`GRU(seqLen, inputSize → hiddenSize)` then `Linear` applied at each timestep.

PyTorch analogy: `nn.GRU(..., batch_first=False)` followed by an `nn.linear` on the output sequence.
-/
def sequence
  [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen inputSize hiddenSize outputSize : Nat}
  (gruSpec : GRUSpec α inputSize hiddenSize)
  (linearSpec : LinearSpec α hiddenSize outputSize) :
  Spec.Module.Chain α (.dim seqLen (.dim inputSize .scalar)) (.dim seqLen (.dim outputSize .scalar)) :=
  let gruModule := Spec.Module.gru gruSpec
  let linearModule := Spec.Module.mapLeading (Spec.Module.linear linearSpec)
  Spec.Module.Chain.single gruModule
    |>.append linearModule

/-- A many-to-one GRU classifier (use the last hidden state, then a linear head).

PyTorch analogy: run `nn.GRU` over the sequence and feed the last output/hidden state into
`nn.Linear(hiddenSize, numClasses)`.
-/
def classifier
  [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen inputSize hiddenSize numClasses : Nat}
  (gruSpec : GRUSpec α inputSize hiddenSize)
  (classifierHead : LinearSpec α hiddenSize numClasses)
  (h : seqLen ≠ 0) :
  Spec.Module.Chain α (.dim seqLen (.dim inputSize .scalar)) (.dim numClasses .scalar) :=
  let gruModule := Spec.Module.gru gruSpec
  let lastOutput := Spec.Module.selectLeading (⟨Nat.pred seqLen, Nat.pred_lt h⟩)
  let classifierModule := Spec.Module.linear classifierHead
  Spec.Module.Chain.single gruModule
    |>.append lastOutput
    |>.append classifierModule

/-- A 2-layer GRU stack (sequence-to-sequence), followed by a per-timestep linear head. -/
def stacked
  [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen inputSize hiddenSize outputSize : Nat}
  (firstSpec : GRUSpec α inputSize hiddenSize)
  (secondSpec : GRUSpec α hiddenSize hiddenSize)
  (linearSpec : LinearSpec α hiddenSize outputSize) :
  Spec.Module.Chain α (.dim seqLen (.dim inputSize .scalar)) (.dim seqLen (.dim outputSize .scalar)) :=
  let firstModule := Spec.Module.gru firstSpec
  let secondModule := Spec.Module.gru secondSpec
  let linearModule := Spec.Module.mapLeading (Spec.Module.linear linearSpec)
  Spec.Module.Chain.single firstModule
    |>.append secondModule
    |>.append linearModule

/-- A simple GRU language-model style pipeline:

`Linear` as the embedding/projection map, then GRU, then a per-timestep projection back to
`vocabSize`.

PyTorch analogy: embedding (often `nn.Embedding`), `nn.GRU`, and `nn.Linear(hiddenSize, vocabSize)`.
We use `LinearSpec` here as a spec-friendly stand-in for a one-hot embedding matrix.
-/
def languageModel
  [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen vocabSize hiddenSize : Nat}
  (embeddingSpec : LinearSpec α vocabSize hiddenSize)
  (gruSpec : GRUSpec α hiddenSize hiddenSize)
  (outputSpec : LinearSpec α hiddenSize vocabSize) :
  Spec.Module.Chain α (.dim seqLen (.dim vocabSize .scalar)) (.dim seqLen (.dim vocabSize .scalar)) :=
  let embeddingModule := Spec.Module.mapLeading (Spec.Module.linear embeddingSpec)
  let gruModule := Spec.Module.gru gruSpec
  let outputModule := Spec.Module.mapLeading (Spec.Module.linear outputSpec)
  Spec.Module.Chain.single embeddingModule
    |>.append gruModule
    |>.append outputModule

/-!
## Record-style model specs

The `Spec.Module.Chain` builders above are the most uniform way to assemble models in TorchLean.

The declarations below use small record types with explicit forward functions. This is useful when
you want to talk about a particular architecture directly (e.g. encoder-decoder), or when you need
to carry extra per-model parameters (e.g. a dropout rate) without building a full module stack.
-/

-- Basic GRU model with a single GRU cell + a linear output head.
/--
Bundle of parameters for a single-layer GRU model with a linear output head.

This is a direct record representation (as opposed to the `Spec.Module.Chain` representation above).
-/
structure Model (α : Type) (inputSize hiddenSize outputSize : Nat) where
  /-- Recurrent cell parameters. -/
  gru : GRUSpec α inputSize hiddenSize
  /-- Linear output projection. -/
  outputLayer : LinearSpec α hiddenSize outputSize

/-- Gradients of the three GRU gates. -/
structure CellGrads (α : Type) (inputSize hiddenSize : Nat) where
  /-- Gradient of the reset-gate weight matrix. -/
  resetWeight : Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar))
  /-- Gradient of the reset-gate bias. -/
  resetBias : Tensor α (.dim hiddenSize .scalar)
  /-- Gradient of the update-gate weight matrix. -/
  updateWeight : Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar))
  /-- Gradient of the update-gate bias. -/
  updateBias : Tensor α (.dim hiddenSize .scalar)
  /-- Gradient of the candidate-state weight matrix. -/
  candidateWeight : Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar))
  /-- Gradient of the candidate-state bias. -/
  candidateBias : Tensor α (.dim hiddenSize .scalar)

/-- Parameter gradients for a GRU model and its linear output head. -/
structure Grads (α : Type) (inputSize hiddenSize outputSize : Nat) where
  /-- Gradients of the recurrent cell. -/
  cell : CellGrads α inputSize hiddenSize
  /-- Gradient of the output projection weight. -/
  outputWeight : Tensor α (.dim outputSize (.dim hiddenSize .scalar))
  /-- Gradient of the output projection bias. -/
  outputBias : Tensor α (.dim outputSize .scalar)

-- Multi-layer GRU model
/--
Bundle of parameters for a multi-layer GRU model.

The first layer consumes `inputSize`, and all subsequent layers consume `hiddenSize`.
-/
structure StackedModel (α : Type) (inputSize hiddenSize outputSize numLayers : Nat) where
  /-- First recurrent layer, whose input may differ from the hidden width. -/
  firstLayer : GRUSpec α inputSize hiddenSize
  /-- Remaining recurrent layers. -/
  hiddenLayers : Fin (numLayers - 1) → GRUSpec α hiddenSize hiddenSize
  /-- Linear output projection. -/
  outputLayer : LinearSpec α hiddenSize outputSize

-- GRU model for classification (many-to-one)
/--
Bundle of parameters for a many-to-one GRU classifier.

The classifier head is applied to the final hidden state.
-/
structure Classifier (α : Type) (inputSize hiddenSize numClasses : Nat) where
  /-- Recurrent cell parameters. -/
  gru : GRUSpec α inputSize hiddenSize
  /-- Linear classifier head. -/
  classifier : LinearSpec α hiddenSize numClasses

-- GRU model for sequence generation (many-to-many)
/--
Bundle of parameters for a many-to-many GRU generator (language-model style).

This includes an (embedding) linear map, recurrent core, and output projection back to vocabulary.
-/
structure Generator (α : Type) (vocabSize hiddenSize : Nat) where
  /-- Token projection used by this one-hot specification. -/
  embedding : LinearSpec α vocabSize hiddenSize
  /-- Recurrent cell parameters. -/
  gru : GRUSpec α hiddenSize hiddenSize
  /-- Projection from hidden states to vocabulary logits. -/
  outputProjection : LinearSpec α hiddenSize vocabSize

/--
Bundle of parameters for a bidirectional GRU model with an output head.

The head consumes the concatenation of forward and backward hidden states.
PyTorch analogue: `nn.GRU(..., bidirectional=true)` plus a linear projection.
-/
structure BidirectionalModel (α : Type) (inputSize hiddenSize outputSize : Nat) where
  /-- Recurrent cell for the original sequence order. -/
  forwardGru : GRUSpec α inputSize hiddenSize
  /-- Recurrent cell for the reversed sequence order. -/
  backwardGru : GRUSpec α inputSize hiddenSize
  /-- Projection from concatenated forward and backward states. -/
  outputLayer : LinearSpec α (hiddenSize + hiddenSize) outputSize

/--
Bundle of parameters for a stacked GRU language model with deterministic dropout.

This model uses a list of GRU layers (all with `hiddenSize` input/output) and applies
evaluation-mode dropout between the GRU stack and the output projection.
-/
structure LanguageModel (α : Type) (vocabSize hiddenSize : Nat) where
  /-- Token projection used by this one-hot specification. -/
  embedding : LinearSpec α vocabSize hiddenSize
  /-- Recurrent layers, ordered from input to output. -/
  layers : List (GRUSpec α hiddenSize hiddenSize)
  /-- Projection from hidden states to vocabulary logits. -/
  outputProjection : LinearSpec α hiddenSize vocabSize
  /-- Dropout probability used between the recurrent stack and output projection. -/
  dropoutRate : α

-- GRU Encoder-Decoder Model
/--
Bundle of parameters for a GRU encoder-decoder model (seq2seq).

This uses separate embeddings and GRU cores for encoder and decoder, plus an output projection.
PyTorch analogue: an encoder `nn.GRU` and a decoder `nn.GRU` with teacher forcing.
-/
structure EncoderDecoder (α : Type) (inputVocabSize hiddenSize outputVocabSize : Nat) where
  /-- Source-token projection. -/
  encoderEmbedding : LinearSpec α inputVocabSize hiddenSize
  /-- Encoder recurrent cell. -/
  encoderGru : GRUSpec α hiddenSize hiddenSize
  /-- Target-token projection. -/
  decoderEmbedding : LinearSpec α outputVocabSize hiddenSize
  /-- Decoder recurrent cell. -/
  decoderGru : GRUSpec α hiddenSize hiddenSize
  /-- Projection from decoder states to target-vocabulary logits. -/
  outputProjection : LinearSpec α hiddenSize outputVocabSize

/-- One-step forward for `Gru.Model`.

Input: `(x_t, h_{t-1})`. Output: `(y_t, h_t)`.
-/
def Model.forward {inputSize hiddenSize outputSize : Nat}
  (model : Model α inputSize hiddenSize outputSize)
  (input : Tensor α (.dim inputSize .scalar))
  (hidden : Tensor α (.dim hiddenSize .scalar)) :
  (Tensor α (.dim outputSize .scalar) × Tensor α (.dim hiddenSize .scalar)) :=
  let nextHidden := gruCellSpec model.gru input hidden
  let output := linearSpec model.outputLayer nextHidden
  (output, nextHidden)

/-- Sequence forward for `Gru.Model` (time-major).

Returns `(outputs, final_hidden)`.

PyTorch analogy: run `nn.GRU` over the sequence, then apply `nn.linear` at each timestep.
-/
def Model.forwardSequence {seqLen inputSize hiddenSize outputSize : Nat}
  (model : Model α inputSize hiddenSize outputSize)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (initialHidden : Tensor α (.dim hiddenSize .scalar)) (h : 0 < seqLen) :
  (Tensor α (.dim seqLen (.dim outputSize .scalar)) × Tensor α (.dim hiddenSize .scalar)) :=
  let hiddenStates := gruSequenceSpec model.gru inputs initialHidden
  let outputs := Tensor.mapLeading (.dim seqLen .scalar)
    (linearSpec model.outputLayer) hiddenStates
  have hLast : seqLen - 1 < seqLen := by
    simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt h)
  let finalHidden := getAtSpec hiddenStates ⟨seqLen - 1, hLast⟩
  (outputs, finalHidden)

-- Forward pass for GRU classifier (many-to-one)
/--
Forward pass for a `Gru.Classifier` (many-to-one).

This runs the GRU over the input sequence and applies the classifier head to the final hidden
state.
-/
def Classifier.forward {seqLen inputSize hiddenSize numClasses : Nat}
  (model : Classifier α inputSize hiddenSize numClasses)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (initialHidden : Tensor α (.dim hiddenSize .scalar)) (h : 0 < seqLen) :
  Tensor α (.dim numClasses .scalar) :=
  let hiddenStates := gruSequenceSpec model.gru inputs initialHidden
  have hLast : seqLen - 1 < seqLen := by
    simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt h)
  let finalHidden := getAtSpec hiddenStates ⟨seqLen - 1, hLast⟩
  linearSpec model.classifier finalHidden

-- Forward pass for GRU generator (many-to-many)
/--
Forward pass for a `Gru.Generator` (many-to-many).

This applies an embedding linear map to each token vector, runs the GRU, and projects each hidden
state back into vocabulary space.
-/
def Generator.forward {seqLen vocabSize hiddenSize : Nat}
  (model : Generator α vocabSize hiddenSize)
  (inputTokens : Tensor α (.dim seqLen (.dim vocabSize .scalar)))
  (initialHidden : Tensor α (.dim hiddenSize .scalar)) (h : 0 < seqLen) :
  (Tensor α (.dim seqLen (.dim vocabSize .scalar)) × Tensor α (.dim hiddenSize .scalar)) :=
  let embedded := Tensor.mapLeading (.dim seqLen .scalar) (linearSpec model.embedding) inputTokens
  let hiddenStates := gruSequenceSpec model.gru embedded initialHidden
  let outputs := Tensor.mapLeading (.dim seqLen .scalar)
    (linearSpec model.outputProjection) hiddenStates
  have hLast : seqLen - 1 < seqLen := by
    simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt h)
  let finalHidden := getAtSpec hiddenStates ⟨seqLen - 1, hLast⟩
  (outputs, finalHidden)

/--
Forward pass for a bidirectional GRU model (time-major).

This runs a forward GRU on the sequence, a backward GRU on the reversed sequence, concatenates the
two hidden streams per timestep, and applies an output head.
-/
def BidirectionalModel.forward {seqLen inputSize hiddenSize outputSize : Nat}
  (model : BidirectionalModel α inputSize hiddenSize outputSize)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (forwardHidden : Tensor α (.dim hiddenSize .scalar))
  (backwardHidden : Tensor α (.dim hiddenSize .scalar)) :
  Tensor α (.dim seqLen (.dim outputSize .scalar)) :=
  let forwardStates := gruSequenceSpec model.forwardGru inputs forwardHidden
  let reversedInputs := Tensor.reverseLeadingAxis inputs
  let reversedBackwardStates :=
    gruSequenceSpec model.backwardGru reversedInputs backwardHidden
  let backwardStates := Tensor.reverseLeadingAxis reversedBackwardStates
  let combinedStates := Tensor.zipWithLeading (.dim seqLen .scalar)
    (.dim (hiddenSize + hiddenSize) .scalar)
    Tensor.concatLeadingAxisSpec forwardStates backwardStates
  Tensor.mapLeading (.dim seqLen .scalar) (linearSpec model.outputLayer) combinedStates

-- Multi-layer GRU forward pass (stack multiple GRU layers)
/--
Forward pass for a `Gru.StackedModel`.

This runs the first layer on the input sequence, then threads the resulting hidden stream through
each additional hidden layer, and finally applies the output head per timestep.
-/
def StackedModel.forward {seqLen inputSize hiddenSize outputSize numLayers : Nat}
  (model : StackedModel α inputSize hiddenSize outputSize numLayers)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (initialHiddens : Fin numLayers → Tensor α (.dim hiddenSize .scalar))
  (hLayers : 0 < numLayers) (hSeq : 0 < seqLen) :
  (Tensor α (.dim seqLen (.dim outputSize .scalar)) × (Fin numLayers → Tensor α (.dim hiddenSize
    .scalar))) :=
  let rec processHiddenLayers (layer : Nat)
    (layerInput : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
    (hiddens : Fin numLayers → Tensor α (.dim hiddenSize .scalar)) :
    (Tensor α (.dim seqLen (.dim hiddenSize .scalar)) × (Fin numLayers → Tensor α (.dim hiddenSize
      .scalar))) :=
    if hLayer : layer < numLayers - 1 then
      let layerIndex : Fin (numLayers - 1) := ⟨layer, hLayer⟩
      have hState : layer + 1 < numLayers := by
        have hState' : layer + 1 ≤ numLayers - 1 := Nat.succ_le_of_lt hLayer
        exact lt_of_le_of_lt hState' (Nat.sub_one_lt (Nat.ne_of_gt hLayers))
      let stateIndex : Fin numLayers := ⟨layer + 1, hState⟩
      let layerHidden := hiddens stateIndex
      let layerOutput :=
        gruSequenceSpec (model.hiddenLayers layerIndex) layerInput layerHidden
      have hLast : seqLen - 1 < seqLen := by
        simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt hSeq)
      let finalLayerHidden := getAtSpec layerOutput ⟨seqLen - 1, hLast⟩
      let updatedHiddens := Function.update hiddens stateIndex finalLayerHidden
      processHiddenLayers (layer + 1) layerOutput updatedHiddens
    else
      (layerInput, hiddens)

  let firstLayerIndex : Fin numLayers := ⟨0, hLayers⟩
  let firstHidden := initialHiddens firstLayerIndex
  let firstOutput := gruSequenceSpec model.firstLayer inputs firstHidden
  have hLast : seqLen - 1 < seqLen := by
    simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt hSeq)
  let firstFinalHidden := getAtSpec firstOutput ⟨seqLen - 1, hLast⟩
  let updatedInitialHiddens :=
    Function.update initialHiddens firstLayerIndex firstFinalHidden

  let (finalHiddenStates, finalHiddens) :=
    processHiddenLayers 0 firstOutput updatedInitialHiddens
  let outputs := Tensor.mapLeading (.dim seqLen .scalar)
    (linearSpec model.outputLayer) finalHiddenStates
  (outputs, finalHiddens)

-- GRU Language Model forward pass
/--
Forward pass for `Gru.LanguageModel` (teacher forcing, time-major).

This runs the embedding, then a stack of GRU layers with provided initial hiddens, applies
evaluation-mode dropout (`dropoutInferenceSpec`), and projects to vocabulary logits.
-/
def LanguageModel.forward {seqLen vocabSize hiddenSize : Nat}
  (model : LanguageModel α vocabSize hiddenSize)
  (inputTokens : Tensor α (.dim seqLen (.dim vocabSize .scalar)))
  (initialHiddens : List (Tensor α (.dim hiddenSize .scalar))) (h : 0 < seqLen) :
  Option
    (Tensor α (.dim seqLen (.dim vocabSize .scalar)) ×
      List (Tensor α (.dim hiddenSize .scalar))) := do
  let embedded :=
    Tensor.mapLeading (.dim seqLen .scalar) (linearSpec model.embedding) inputTokens
  let rec processLayers (layers : List (GRUSpec α hiddenSize hiddenSize))
    (hiddens : List (Tensor α (.dim hiddenSize .scalar)))
    (layerInput : Tensor α (.dim seqLen (.dim hiddenSize .scalar))) :
    Option
      (Tensor α (.dim seqLen (.dim hiddenSize .scalar)) ×
        List (Tensor α (.dim hiddenSize .scalar))) :=
    match layers, hiddens with
    | [], [] => some (layerInput, [])
    | layer :: remainingLayers, hidden :: remainingHiddens => do
      let layerOutput := gruSequenceSpec layer layerInput hidden
      have hLast : seqLen - 1 < seqLen := by
        simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt h)
      let finalHidden := getAtSpec layerOutput ⟨seqLen - 1, hLast⟩
      let (finalOutput, finalHiddens) ←
        processLayers remainingLayers remainingHiddens layerOutput
      pure (finalOutput, finalHidden :: finalHiddens)
    | _, _ => none
  let (gruOutput, finalHiddens) ← processLayers model.layers initialHiddens embedded
  let droppedOutput := dropoutInferenceSpec (p := model.dropoutRate) gruOutput
  let logits := Tensor.mapLeading (.dim seqLen .scalar)
    (linearSpec model.outputProjection) droppedOutput
  pure (logits, finalHiddens)

-- GRU Encoder-Decoder forward pass
/-- Encoder-decoder forward pass (GRU encoder + GRU decoder).

This is a small reference architecture:

- encode `sourceTokens` into a final hidden state,
- decode `targetTokens` starting from that hidden state (teacher forcing),
- project decoder states into output-vocabulary logits.

PyTorch analogy: `nn.GRU` encoder + `nn.GRU` decoder with a linear output projection.
-/
def EncoderDecoder.forward {srcSeqLen tgtSeqLen inputVocabSize hiddenSize outputVocabSize :
  Nat}
  (model : EncoderDecoder α inputVocabSize hiddenSize outputVocabSize)
  (sourceTokens : Tensor α (.dim srcSeqLen (.dim inputVocabSize .scalar)))
  (targetTokens : Tensor α (.dim tgtSeqLen (.dim outputVocabSize .scalar)))
  (encoderHidden : Tensor α (.dim hiddenSize .scalar))
  (hSource : 0 < srcSeqLen) (hTarget : 0 < tgtSeqLen) :
  (Tensor α (.dim tgtSeqLen (.dim outputVocabSize .scalar)) ×
   Tensor α (.dim hiddenSize .scalar) × Tensor α (.dim hiddenSize .scalar)) :=
  let sourceEmbedded := Tensor.mapLeading (.dim srcSeqLen .scalar)
    (linearSpec model.encoderEmbedding) sourceTokens
  let encoderStates := gruSequenceSpec model.encoderGru sourceEmbedded encoderHidden
  have hSourceLast : srcSeqLen - 1 < srcSeqLen := by
    simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt hSource)
  let encoderFinal := getAtSpec encoderStates ⟨srcSeqLen - 1, hSourceLast⟩
  let targetEmbedded := Tensor.mapLeading (.dim tgtSeqLen .scalar)
    (linearSpec model.decoderEmbedding) targetTokens
  let decoderStates := gruSequenceSpec model.decoderGru targetEmbedded encoderFinal
  have hTargetLast : tgtSeqLen - 1 < tgtSeqLen := by
    simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt hTarget)
  let decoderFinal := getAtSpec decoderStates ⟨tgtSeqLen - 1, hTargetLast⟩
  let outputs := Tensor.mapLeading (.dim tgtSeqLen .scalar)
    (linearSpec model.outputProjection) decoderStates
  (outputs, encoderFinal, decoderFinal)

-- Backward pass for simple GRU model with full BPTT
/-- Backward pass for `Gru.Model` using full backpropagation through time.

This assumes you already ran a forward pass that saved:
- `hidden_states`,
- the GRU intermediates (`reset_gates`, `update_gates`, `new_candidates`, `reset_hiddens`).

Those intermediates can be produced using `Spec.gruExtractIntermediateValues` from
`NN.Spec.Layers.Gru`.
-/
def Model.backward {seqLen inputSize hiddenSize outputSize : Nat}
  (model : Model α inputSize hiddenSize outputSize)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (hiddenStates : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
  (outputGrad : Tensor α (.dim seqLen (.dim outputSize .scalar)))
  (resetGates : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
  (updateGates : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
  (candidates : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
  (h : seqLen ≠ 0) :
  Grads α inputSize hiddenSize outputSize ×
    Tensor α (.dim seqLen (.dim inputSize .scalar)) :=
  let hiddenGrad := Tensor.mapLeading (.dim seqLen .scalar)
    (fun grad => linearInputDerivSpec model.outputLayer.weights grad) outputGrad
  let outputWeightGrad := Tensor.sumLeadingAxis
    (Tensor.zipWithLeading (.dim seqLen .scalar)
      (Shape.dim outputSize (Shape.dim hiddenSize Shape.scalar))
      linearWeightsDerivSpec hiddenStates outputGrad) h
  let outputBiasGrad := Tensor.sumLeadingAxis outputGrad h
  let initialHidden := fill 0 (.dim hiddenSize .scalar)
  let (resetWeight, resetBias, updateWeight, updateBias,
       candidateWeight, candidateBias, inputGrad, _) :=
    gruSequenceBackwardFullSpec model.gru inputs hiddenStates hiddenGrad
      resetGates updateGates candidates initialHidden
  ({ cell := { resetWeight, resetBias, updateWeight, updateBias, candidateWeight, candidateBias }
     outputWeight := outputWeightGrad
     outputBias := outputBiasGrad }, inputGrad)

/--
Bundle of parameters for a residual GRU model.

This includes a projection from input space to hidden space so the input can be added as a residual
to the GRU hidden stream.
-/
structure ResidualModel (α : Type) (inputSize hiddenSize outputSize : Nat) where
  /-- Recurrent cell parameters. -/
  gru : GRUSpec α inputSize hiddenSize
  /-- Projection that gives the input stream the recurrent hidden width. -/
  residualProjection : LinearSpec α inputSize hiddenSize
  /-- Linear output projection. -/
  outputLayer : LinearSpec α hiddenSize outputSize

/--
Forward pass for `Gru.ResidualModel`.

This runs the GRU, adds a projected version of the input as a residual connection, and applies the
output head per timestep.
-/
def ResidualModel.forward {seqLen inputSize hiddenSize outputSize : Nat}
  (model : ResidualModel α inputSize hiddenSize outputSize)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (initialHidden : Tensor α (.dim hiddenSize .scalar)) (h : 0 < seqLen) :
  (Tensor α (.dim seqLen (.dim outputSize .scalar)) × Tensor α (.dim hiddenSize .scalar)) :=
  let projectedInputs := Tensor.mapLeading (.dim seqLen .scalar)
    (linearSpec model.residualProjection) inputs
  let hiddenStates := gruSequenceSpec model.gru inputs initialHidden
  let residualStates := Tensor.zipWithLeading (.dim seqLen .scalar) (.dim hiddenSize .scalar)
    addSpec hiddenStates projectedInputs
  let outputs := Tensor.mapLeading (.dim seqLen .scalar)
    (linearSpec model.outputLayer) residualStates
  have hLast : seqLen - 1 < seqLen := by
    simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt h)
  let finalHidden := getAtSpec residualStates ⟨seqLen - 1, hLast⟩
  (outputs, finalHidden)

/--
Package `Gru.Model` as a shape-indexed module.

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
  kind := "SimpleGRU",
  pythonExpr := s!"SimpleGRU(input_size={inputSize}, hidden_size={hiddenSize}, output_size={outputSize})"
}

/--
Package `Gru.Classifier` as an `Spec.Module`.

PyTorch analogue: `nn.GRU` feeding a `nn.linear` classifier head.
-/
def Classifier.toModule {seqLen inputSize hiddenSize numClasses : Nat}
  (model : Classifier α inputSize hiddenSize numClasses) (h : 0 < seqLen) :
  Spec.Module α (.dim seqLen (.dim inputSize .scalar)) (.dim numClasses .scalar) :=
{
  forward := fun inputs =>
    let initialHidden := fill 0 (.dim hiddenSize .scalar)
    model.forward inputs initialHidden h,
  kind := "GRUClassifier",
  pythonExpr := s!"GRUClassifier(input_size={inputSize}, hidden_size={hiddenSize}, num_classes={numClasses})"
}

/--
Package `Gru.BidirectionalModel` as an `Spec.Module`.

PyTorch analogue: `nn.GRU(..., bidirectional=true)` feeding a per-timestep linear head.
-/
def BidirectionalModel.toModule {seqLen inputSize hiddenSize outputSize : Nat}
  (model : BidirectionalModel α inputSize hiddenSize outputSize) :
  Spec.Module α (.dim seqLen (.dim inputSize .scalar)) (.dim seqLen (.dim outputSize .scalar)) :=
{
  forward := fun inputs =>
    let initialHidden := fill 0 (.dim hiddenSize .scalar)
    model.forward inputs initialHidden initialHidden,
  kind := "BiGRU",
  pythonExpr := s!"SimpleGRU(input_size={inputSize}, hidden_size={hiddenSize}, output_size={outputSize}, " ++
        s!"bidirectional=True)"
}

/--
Package `Gru.Generator` as an `Spec.Module`.

PyTorch analogue: GRU language model (`nn.GRU` + vocabulary projection) producing a sequence of
logits.
-/
def Generator.toModule {seqLen vocabSize hiddenSize : Nat}
  (model : Generator α vocabSize hiddenSize) (h : 0 < seqLen) :
  Spec.Module α (.dim seqLen (.dim vocabSize .scalar)) (.dim seqLen (.dim vocabSize .scalar)) :=
{
  forward := fun inputs =>
    let initialHidden := fill 0 (.dim hiddenSize .scalar)
    (model.forward inputs initialHidden h).1,
  kind := "GRUGenerator",
  pythonExpr := s!"GRULanguageModel(vocab_size={vocabSize}, hidden_size={hiddenSize})"
}

end Gru

end Spec

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Linear
public import NN.Spec.Layers.Loss
public import NN.Spec.Layers.Lstm
public import NN.Spec.Models.Transformer

/-!
# Seq2Seq (spec model)

Encoder-decoder models for sequence generation.

This file supports both bounded token indices and differentiable token distributions. A bounded
index has type `Fin vocabSize`, so embedding lookup cannot silently substitute a value for an
invalid token. A token distribution uses a vector of length `vocabSize` and realizes embedding as a
matrix multiplication.

PyTorch analogue:

- encoder: `nn.RNN` / `nn.LSTM` (or `nn.TransformerEncoder`) over source token embeddings
- decoder: `nn.RNN` over target embeddings (teacher forcing in training), then a final `nn.linear`
  to vocabulary logits

Scope of this baseline:

- the optional attention in `Seq2SeqDecoderSpec` is *self-attention over the decoder inputs* (a
  small variant you can toggle on/off); this file does not model encoder-decoder cross-attention
  in the main baseline.
- for cross-attention style mechanisms, we include a small additive/Bahdanau-style attention at the
  bottom of the file (`compute_attention_weights_spec` / `apply_attention_spec`).

The transformer encoder blocks used by the transformer variant come from
`NN/Spec/Models/Transformer.lean`.

References:
- Sutskever et al., "Sequence to Sequence Learning with Neural Networks" (NeurIPS 2014).
- Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2015).
- Hochreiter and Schmidhuber, "Long Short-Term Memory" (1997).
- Cho et al.,
  "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"
  (2014).
- Vaswani et al., "Attention Is All You Need" (2017) for the transformer encoder variant.

PyTorch docs (for API intuition, not semantics):
- `torch.nn.Embedding`: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
- `torch.nn.RNN`: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
- `torch.nn.LSTM`: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- `torch.nn.Linear`: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
- `torch.nn.MultiheadAttention`:
  https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
- `torch.nn.TransformerEncoderLayer`:
  https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
-/

@[expose] public section


namespace Spec

open Tensor
open Recurrent

variable {α : Type} [Context α]

/-!
## Training + gradients (one-hot inputs)

Most of this file focuses on *architecture variants* and *forward passes* (teacher-forcing,
inference-time decoding, optional self-attention in the decoder, etc.).

To make Seq2Seq usable as a first-class baseline, we also provide an explicit training objective
and reverse-mode gradients for the differentiable path:

- inputs are **one-hot / token distributions** (so embedding lookup is a matrix multiply),
- teacher forcing is used in the decoder,
- the loss is per-timestep cross-entropy between `softmax(logits)` and the target distribution,
- gradients flow through embeddings, encoder RNN, decoder RNN, output projection, and (optionally)
  the decoder self-attention block.

Bounded token indices are intentionally treated as non-differentiable.
-/

/-! ### Small gradient records -/

/--
Gradients for a time-distributed affine map `y = x·Wᵀ + b`.

This mirrors the parameters in `LinearSpec` and is used for the decoder output projection.
PyTorch analogue: the gradient pair for `nn.linear`.
-/
structure Seq2SeqLinearGrads (α : Type) (inDim outDim : Nat) where
  /-- Gradient of the weight matrix `W`. -/
  weight : Tensor α (.dim outDim (.dim inDim .scalar))
  /-- Gradient of the bias vector `b`. -/
  bias : Tensor α (.dim outDim .scalar)

/--
Gradients for an `RNNSpec` cell.

PyTorch analogue: the gradients for `nn.RNN` parameters (weight and bias).
-/
structure Seq2SeqRNNGrads (α : Type) (inputSize hiddenSize : Nat) where
  /-- Gradient of the concatenated input+hidden weight matrix. -/
  weight : WeightMatrix α hiddenSize (inputSize + hiddenSize)
  /-- Gradient of the bias term. -/
  bias : HiddenVector α hiddenSize

/--
Gradients for a token embedding table `E : (vocabSize × embedDim)`.

PyTorch analogue: `nn.Embedding.weight.grad`.
-/
structure Seq2SeqEmbeddingGrads (α : Type) (vocabSize embedDim : Nat) where
  /-- Gradient of the embedding matrix. -/
  embedding : Tensor α (.dim vocabSize (.dim embedDim .scalar))

/--
End-to-end gradient record for the differentiable Seq2Seq baseline.

This bundles gradients for:
- source/target embeddings,
- encoder RNN,
- decoder RNN,
- decoder output projection,
- optional decoder self-attention (if enabled in the decoder spec).
-/
structure Seq2SeqGrads (α : Type)
    (srcVocabSize tgtVocabSize embedDim hiddenDim : Nat) where
  /-- Gradients for the source embedding table. -/
  sourceEmbedding : Seq2SeqEmbeddingGrads α srcVocabSize embedDim
  /-- Gradients for the target embedding table. -/
  targetEmbedding : Seq2SeqEmbeddingGrads α tgtVocabSize embedDim
  /-- Gradients for the encoder RNN parameters. -/
  encoder : Seq2SeqRNNGrads α embedDim hiddenDim
  /-- Gradients for the decoder RNN parameters. -/
  decoderRnn : Seq2SeqRNNGrads α embedDim hiddenDim
  /-- Gradients for the decoder output projection (`hiddenDim -> tgtVocabSize`). -/
  outputProjection : Seq2SeqLinearGrads α hiddenDim tgtVocabSize
  /-- Gradients for optional decoder self-attention parameters. -/
  decoderAttention :
    Option (Σ numHeads : Nat, MultiHeadAttentionGrads numHeads embedDim (embedDim / numHeads) α) :=
    none

/--
Seq2Seq token embedding specification.

Parameters:
- `embedding`: a lookup table `E : (vocabSize × embedDim)`.

PyTorch analogue: `nn.Embedding(vocabSize, embedDim)`.
-/
structure Seq2SeqEmbeddingSpec (α : Type) [Numbers α] (vocabSize embedDim : Nat) where
  /-- Embedding table `E : (vocabSize × embedDim)`. -/
  embedding : Tensor α (.dim vocabSize (.dim embedDim .scalar))

/--
Embedding forward pass for discrete token ids.

Inputs:
- `tokenIds : (seqLen)`, a tensor of indices bounded by the vocabulary size.

Output:
- `y : (seqLen × embedDim)`, where each timestep selects a row of the embedding table.

PyTorch analogue: `nn.Embedding` on an integer tensor. The `Fin vocabSize` element type expresses
the lookup precondition directly, rather than assigning an arbitrary meaning to an invalid token.
-/
def Seq2SeqEmbeddingSpec.forward {vocabSize embedDim seqLen : Nat}
  (embedding : Seq2SeqEmbeddingSpec α vocabSize embedDim)
  (tokenIds : Tensor (Fin vocabSize) (.dim seqLen .scalar)):
  Tensor α (.dim seqLen (.dim embedDim .scalar)) :=
  Tensor.dim (fun i =>
    match get tokenIds i with
    | Tensor.scalar tokenId => get embedding.embedding tokenId)

/-- Seq2Seq embedding forward pass for one-hot / token distributions.

This is the usual "embedding lookup as a matrix multiply":

- if `E : (vocabSize × embedDim)` is the embedding table,
- and `x_t : (vocabSize)` is a one-hot / probability vector for time step `t`,
- then the embedded vector is `y_t = x_tᵀ · E : (embedDim)`.

PyTorch analogy: `y = x @ E` where `x` is one-hot / a distribution; this matches `nn.Embedding`
when the input is exactly one-hot.
-/
def Seq2SeqEmbeddingSpec.forwardOneHot {vocabSize embedDim seqLen : Nat}
  (embedding : Seq2SeqEmbeddingSpec α vocabSize embedDim)
  (tokenOneHot : Tensor α (.dim seqLen (.dim vocabSize .scalar))) :
  Tensor α (.dim seqLen (.dim embedDim .scalar)) :=
  match tokenOneHot with
  | Tensor.dim f =>
      Tensor.dim (fun i => vecMatMulSpec (f i) embedding.embedding)

/--
Backward pass for `Seq2SeqEmbeddingSpec.forwardOneHot`.

This is just a time-distributed linear layer:

`y_t = token_tᵀ · E`

So:
- `dE = Σ_t token_t ⊗ dY_t`
- `dToken_t = E · dY_t` (not usually needed, but included for completeness)
-/
def Seq2SeqEmbeddingSpec.backwardOneHot {vocabSize embedDim seqLen : Nat}
  (embedding : Seq2SeqEmbeddingSpec α vocabSize embedDim)
  (tokenOneHot : Tensor α (.dim seqLen (.dim vocabSize .scalar)))
  (grad_output : Tensor α (.dim seqLen (.dim embedDim .scalar))) :
  (Seq2SeqEmbeddingGrads α vocabSize embedDim × Tensor α (.dim seqLen (.dim vocabSize .scalar))) :=
  let step (i : Fin seqLen) (acc : Seq2SeqEmbeddingGrads α vocabSize embedDim) :=
    let token_t := get tokenOneHot i
    let dY_t := get grad_output i
    let dE_t := outerProductSpec token_t dY_t
    let dToken_t := matVecMulSpec embedding.embedding dY_t
    ({ embedding := addSpec acc.embedding dE_t }, dToken_t)
  let init : Seq2SeqEmbeddingGrads α vocabSize embedDim :=
    { embedding := fill 0 (.dim vocabSize (.dim embedDim .scalar)) }
  let (dE, dX) := Sequence.mapAccum seqLen init step
  (dE, Tensor.dim dX.get)

/--
RNN-based encoder specification for Seq2Seq.

This models an `nn.RNN`-style encoder over embedded tokens:
- input is a sequence of embeddings `(seqLen × embedDim)`,
- output is the full hidden-state sequence plus the final hidden state.

PyTorch analogue: `nn.RNN(..., batch_first=True)` (ignoring the batch axis), returning `(output,
  h_n)`.
-/
structure Seq2SeqRNNEncoderSpec (α : Type) [Numbers α] (embedDim hiddenDim : Nat) where
  /-- RNN cell parameters. -/
  rnn : RNNSpec α embedDim hiddenDim

/--
Forward pass for `Seq2SeqRNNEncoderSpec`.

Inputs:
- `x : (seqLen × embedDim)`, embedded source tokens,
- `h0`, optional initial hidden state (`hiddenDim`).

Returns:
- `(outputs, final_h)` where `outputs : (seqLen × hiddenDim)` is the per-timestep hidden sequence.
-/
def Seq2SeqRNNEncoderSpec.forward {α : Type} [Context α] {embedDim hiddenDim seqLen : Nat}
  (encoder : Seq2SeqRNNEncoderSpec α embedDim hiddenDim)
  (x : Tensor α (.dim seqLen (.dim embedDim .scalar)))
  (h0 : Option (Tensor α (.dim hiddenDim .scalar))):
  (Tensor α (.dim seqLen (.dim hiddenDim .scalar)) × Tensor α (.dim hiddenDim .scalar)) :=
  let initialHidden := match h0 with
    | some h => h
    | none => fill 0 (.dim hiddenDim .scalar)
  let (finalHidden, outputs) := Sequence.mapAccum seqLen initialHidden fun i previous =>
    let hidden := rnnCellSpec encoder.rnn (get x i) previous
    (hidden, hidden)
  (Tensor.dim outputs.get, finalHidden)


/--
LSTM-based encoder specification for Seq2Seq.

This models an `nn.LSTM`-style encoder over embedded tokens, returning the full hidden sequence,
final hidden state, and final cell state.

PyTorch analogue: `nn.LSTM(..., batch_first=True)` (ignoring the batch axis), returning
`(output, (h_n, c_n))`.
-/
structure Seq2SeqLSTMEncoderSpec (α : Type) [Numbers α] (embedDim hiddenDim : Nat) where
  /-- LSTM cell parameters. -/
  lstm : LSTMSpec α embedDim hiddenDim

/--
Forward pass for `Seq2SeqLSTMEncoderSpec`.

Inputs:
- `x : (seqLen × embedDim)`, embedded source tokens,
- `h0`, optional initial hidden state (`hiddenDim`),
- `c0`, optional initial cell state (`hiddenDim`).

Returns:
- `(outputs, final_h, final_c)` where `outputs : (seqLen × hiddenDim)` is the per-timestep hidden
  sequence.
-/
def Seq2SeqLSTMEncoderSpec.forward {embedDim hiddenDim seqLen : Nat}
  (encoder : Seq2SeqLSTMEncoderSpec α embedDim hiddenDim)
  (x : Tensor α (.dim seqLen (.dim embedDim .scalar)))
  (h0 : Option (Tensor α (.dim hiddenDim .scalar)))
  (c0 : Option (Tensor α (.dim hiddenDim .scalar))):
  (Tensor α (.dim seqLen (.dim hiddenDim .scalar)) ×
   Tensor α (.dim hiddenDim .scalar) ×
   Tensor α (.dim hiddenDim .scalar)) :=
  let initialHidden := match h0 with
  | some h => h
  | none => fill 0 (.dim hiddenDim .scalar)
  let initialCell := match c0 with
  | some c => c
  | none => fill 0 (.dim hiddenDim .scalar)
  let (outputs, finalState) :=
    lstmSequenceSpec encoder.lstm x { hidden := initialHidden, cell := initialCell }
  (outputs, finalState.hidden, finalState.cell)

/--
Transformer-based encoder specification for Seq2Seq.

This wrapper applies exactly `numLayers` `TransformerEncoderLayer`s from
`NN.Spec.Models.Transformer` as a left fold.

PyTorch analogue: `nn.TransformerEncoder(nn.TransformerEncoderLayer(...), num_layers=...)`
(ignoring dropout and most configuration knobs).
-/
structure Seq2SeqTransformerEncoderSpec (α : Type) [Context α] [Numbers α] (embedDim numHeads
  numLayers : Nat) where
  /-- Encoder layer stack. Its length is part of the type. -/
  layers : Vector (TransformerEncoderLayer numHeads embedDim (embedDim * 4) α) numLayers

/--
Forward pass for `Seq2SeqTransformerEncoderSpec`.

Input/output shape: `(seqLen × embedDim)`.

This uses post-norm transformer layers from `NN.Spec.Models.Transformer` and does not model
dropout; it is meant as a clean semantic reference rather than a full training-ready implementation.
-/
def Seq2SeqTransformerEncoderSpec.forward {embedDim numHeads numLayers seqLen : Nat}
  (encoder : Seq2SeqTransformerEncoderSpec α embedDim numHeads numLayers)
  (x : Tensor α (.dim seqLen (.dim embedDim .scalar)))
  (h1 : seqLen > 0) (h2 : embedDim > 0) :
  Tensor α (.dim seqLen (.dim embedDim .scalar)) :=
  encoder.layers.toArray.foldl (fun acc layer => TransformerEncoderLayer.forward layer acc h1 h2) x

/--
RNN decoder specification for Seq2Seq.

This decoder consumes a sequence of target-side embeddings and produces vocabulary logits:
- an `RNNSpec` cell updates the hidden state per timestep,
- a time-distributed `LinearSpec` maps hidden states to logits,
- optionally, a self-attention block can be applied over the *decoder input embeddings* before the
  RNN.

PyTorch analogue: a hand-rolled decoder using `nn.RNN` and `nn.linear`, optionally preceded by
`nn.MultiheadAttention` over the target embeddings (note: this is not encoder-decoder
  cross-attention).
-/
structure Seq2SeqDecoderSpec (α : Type) [Numbers α] (embedDim hiddenDim vocabSize : Nat) where
  /-- Decoder RNN cell parameters. -/
  rnn : RNNSpec α embedDim hiddenDim
  /-- Optional self-attention block over decoder input embeddings. -/
  attention :
    Option (Σ numHeads : Nat, MultiHeadAttention α numHeads embedDim (embedDim / numHeads)) := none
  /-- Output projection (`hiddenDim -> vocabSize`) producing per-timestep logits. -/
  outputProjection : LinearSpec α hiddenDim vocabSize

/--
Seq2Seq decoder forward pass (teacher forcing)

- `target_embeddings` : Tensor of shape (tgtSeqLen × embedDim)
- `h0` : initial hidden state (hiddenDim)
- Returns: Tensor of shape (tgtSeqLen × vocabSize)
- If `decoder.attention` is `some`, this runs self-attention over `target_embeddings` and feeds the
  attended embedding at each timestep.
- Note: this spec does not model cross-attention over encoder outputs.
-/
def Seq2SeqDecoderSpec.forwardTeacherForcing {embedDim hiddenDim vocabSize tgtSeqLen : Nat}
  (decoder : Seq2SeqDecoderSpec α embedDim hiddenDim vocabSize)
  (target_embeddings : Tensor α (.dim tgtSeqLen (.dim embedDim .scalar)))
  (h0 : Tensor α (.dim hiddenDim .scalar))
  (h_len_nonzero : tgtSeqLen ≠ 0) :
  Tensor α (.dim tgtSeqLen (.dim vocabSize .scalar)) :=

  -- Optional self-attention over the full target embedding sequence.
  let attendedEmbeddings :=
    match decoder.attention with
    | some ⟨_numHeads, attn⟩ =>
        MultiHeadAttention.forward tgtSeqLen h_len_nonzero attn target_embeddings none
    | none => target_embeddings
  let hiddens := rnnSequenceSpec decoder.rnn attendedEmbeddings h0
  Tensor.dim (fun i => linearSpec decoder.outputProjection (get hiddens i))

/-!
### Decoder backward (teacher forcing)

The decoder is: (optional self-attention) → RNN → time-distributed linear projection.

We compute gradients by:
1) recomputing the attended embeddings (if any),
2) recomputing the decoder hidden sequence,
3) backpropagating through the output projection per timestep,
4) backpropagating through the RNN sequence,
5) optionally backpropagating through self-attention.
-/

/--
Backward pass for a time-distributed `LinearSpec`.

Given a hidden-state sequence `hiddens : (tgtSeqLen × hiddenDim)` and upstream gradients
`grad_logits : (tgtSeqLen × vocabSize)`, computes:
- accumulated parameter gradients for the shared `LinearSpec`,
- gradients w.r.t. each hidden state `(tgtSeqLen × hiddenDim)`.

PyTorch analogue: backprop through `nn.linear` applied at each timestep.
-/
def timeDistributedLinearBackward
  {tgtSeqLen hiddenDim vocabSize : Nat}
  (layer : LinearSpec α hiddenDim vocabSize)
  (hiddens : Tensor α (.dim tgtSeqLen (.dim hiddenDim .scalar)))
  (grad_logits : Tensor α (.dim tgtSeqLen (.dim vocabSize .scalar))) :
  (Seq2SeqLinearGrads α hiddenDim vocabSize × Tensor α (.dim tgtSeqLen (.dim hiddenDim .scalar))) :=
  let step (i : Fin tgtSeqLen) (acc : Seq2SeqLinearGrads α hiddenDim vocabSize) :=
    let hi := get hiddens i
    let dYi := get grad_logits i
    let (dW, db, dH) := linearBackwardSpec layer hi dYi
    ({ weight := addSpec acc.weight dW, bias := addSpec acc.bias db }, dH)
  let init : Seq2SeqLinearGrads α hiddenDim vocabSize := {
    weight := fill 0 (.dim vocabSize (.dim hiddenDim .scalar)),
    bias := fill 0 (.dim vocabSize .scalar)
  }
  let (linearGrads, dH) := Sequence.mapAccum tgtSeqLen init step
  (linearGrads, Tensor.dim dH.get)

/--
Backward pass for `Seq2SeqDecoderSpec.forwardTeacherForcing`.

Returns:
- RNN parameter gradients,
- output projection gradients,
- optional self-attention parameter gradients,
- gradient w.r.t. the target embeddings sequence,
- gradient w.r.t. the initial hidden state `h0`.

Implementation note: this spec recomputes the attended embeddings and hidden sequence to keep the
backward pass self-contained (no mutable tape).
-/
def Seq2SeqDecoderSpec.backwardTeacherForcing
  {embedDim hiddenDim vocabSize tgtSeqLen : Nat}
  (decoder : Seq2SeqDecoderSpec α embedDim hiddenDim vocabSize)
  (target_embeddings : Tensor α (.dim tgtSeqLen (.dim embedDim .scalar)))
  (h0 : Tensor α (.dim hiddenDim .scalar))
  (h_len_nonzero : tgtSeqLen ≠ 0)
  (grad_logits : Tensor α (.dim tgtSeqLen (.dim vocabSize .scalar))) :
  (Seq2SeqRNNGrads α embedDim hiddenDim ×
    Seq2SeqLinearGrads α hiddenDim vocabSize ×
    Option (Σ numHeads : Nat, MultiHeadAttentionGrads numHeads embedDim (embedDim / numHeads) α) ×
    Tensor α (.dim tgtSeqLen (.dim embedDim .scalar)) ×
    Tensor α (.dim hiddenDim .scalar)) :=

  let attendedEmbeddings :=
    match decoder.attention with
    | some ⟨_numHeads, attn⟩ =>
        MultiHeadAttention.forward tgtSeqLen h_len_nonzero attn target_embeddings none
    | none => target_embeddings
  let hiddens := rnnSequenceSpec decoder.rnn attendedEmbeddings h0
  let (projGrads, dH) := timeDistributedLinearBackward (α := α)
    (tgtSeqLen := tgtSeqLen) (hiddenDim := hiddenDim) (vocabSize := vocabSize)
    decoder.outputProjection hiddens grad_logits

  let (dW_rnn, db_rnn, dAttended0, dH0) :=
    rnnSequenceBackwardSpec decoder.rnn attendedEmbeddings h0 hiddens dH

  let rnnGrads : Seq2SeqRNNGrads α embedDim hiddenDim :=
    { weight := dW_rnn, bias := db_rnn }

  match decoder.attention with
  | none =>
      (rnnGrads, projGrads, none, dAttended0, dH0)
  | some ⟨numHeads, attn⟩ =>
      let (dTargetEmb, queryWeight, keyWeight, valueWeight, outputWeight) :=
        multiHeadAttentionBackward (α := α) (n := tgtSeqLen) (dModel := embedDim)
          h_len_nonzero attn target_embeddings none dAttended0
      let attnGrads : MultiHeadAttentionGrads numHeads embedDim (embedDim / numHeads) α :=
        { queryWeight, keyWeight, valueWeight, outputWeight }
      (rnnGrads, projGrads, some ⟨numHeads, attnGrads⟩, dTargetEmb, dH0)

/--
Seq2Seq decoder forward pass (inference-time autoregressive decoding).

This runs a greedy decoding loop for `maxLen` steps, starting from:
- an initial hidden state `h0`,
- a bounded starting token index,
- and a target embedding table used to embed each predicted token.

Returns:
- the per-step logits `(maxLen × vocabSize)`,
- the greedy-decoded token ids `(maxLen)`.

PyTorch analogue: a manual decoding loop using `nn.RNNCell`/`nn.RNN` + `nn.linear`, with
`argmax` sampling and embedding lookup each step.

Note: `decoder.attention` is only modeled in the teacher-forcing forward/backward in this file; the
greedy decoding loop below does not implement autoregressive self-attention.
-/
def Seq2SeqDecoderSpec.forwardInference {embedDim hiddenDim vocabSize : Nat}
  (decoder : Seq2SeqDecoderSpec α embedDim hiddenDim vocabSize)
  (h0 : Tensor α (.dim hiddenDim .scalar))
  (targetEmbedding : Tensor α (.dim vocabSize (.dim embedDim .scalar)))
  (startToken : Fin vocabSize) (maxLen : Nat) :
  (Tensor α (.dim maxLen (.dim vocabSize .scalar)) ×
    Tensor (Fin vocabSize) (.dim maxLen .scalar)) :=
  let initialInput := get targetEmbedding startToken
  let hVocab : 0 < vocabSize := lt_of_le_of_lt (Nat.zero_le startToken.val) startToken.isLt
  let (_, results) := Sequence.mapAccum maxLen (h0, initialInput) fun _ state =>
    let (hidden, input) := state
    -- The optional self-attention block is defined over a complete teacher-forcing sequence. An
    -- autoregressive attention decoder needs a causal prefix or cache and is therefore a separate
    -- model, rather than an implicit reinterpretation of this RNN loop.
    let nextHidden := rnnCellSpec decoder.rnn input hidden
    let logits := linearSpec decoder.outputProjection nextHidden
    let token := argmaxVector hVocab logits
    let nextInput := get targetEmbedding token
    ((nextHidden, nextInput), (logits, token))
  (Tensor.dim (fun i => (results.get i).1),
    Tensor.dim (fun i => Tensor.scalar (results.get i).2))

/--
Complete Seq2Seq model specification (baseline).

This bundles:
- source and target embedding tables,
- an RNN encoder,
- an RNN decoder with output projection (and optional decoder self-attention).

PyTorch analogue: a small encoder-decoder model built from `nn.Embedding`, `nn.RNN`, and
  `nn.linear`.
-/
structure Seq2SeqSpec (α : Type) [Numbers α] (srcVocabSize tgtVocabSize embedDim hiddenDim : Nat)
  where
  /-- Source embedding table. -/
  sourceEmbedding : Seq2SeqEmbeddingSpec α srcVocabSize embedDim
  /-- Target embedding table. -/
  targetEmbedding : Seq2SeqEmbeddingSpec α tgtVocabSize embedDim
  /-- Encoder RNN parameters. -/
  encoder : Seq2SeqRNNEncoderSpec α embedDim hiddenDim
  /-- Decoder parameters (RNN + output projection + optional self-attention). -/
  decoder : Seq2SeqDecoderSpec α embedDim hiddenDim tgtVocabSize

/--
Seq2Seq forward pass for training (teacher forcing) using discrete token ids.

Inputs:
- `src_tokens : (srcSeqLen)` and `tgt_tokens : (tgtSeqLen)` are token id tensors.

Output:
- logits of shape `(tgtSeqLen × tgtVocabSize)`.

This path is for token-id inputs. The lookup is treated as a discrete operation, so gradients are
not assigned to the token ids themselves.
-/
def Seq2SeqSpec.forwardTraining {srcVocabSize tgtVocabSize embedDim hiddenDim srcSeqLen tgtSeqLen :
  Nat}
  (model : Seq2SeqSpec α srcVocabSize tgtVocabSize embedDim hiddenDim)
  (sourceTokens : Tensor (Fin srcVocabSize) (.dim srcSeqLen .scalar))
  (targetTokens : Tensor (Fin tgtVocabSize) (.dim tgtSeqLen .scalar))
  (hTarget : tgtSeqLen ≠ 0) :
  Tensor α (.dim tgtSeqLen (.dim tgtVocabSize .scalar)) :=
  let sourceEmbeddings := Seq2SeqEmbeddingSpec.forward model.sourceEmbedding sourceTokens
  let (_encoderOutputs, encoderHidden) :=
    Seq2SeqRNNEncoderSpec.forward model.encoder sourceEmbeddings
    none
  let targetEmbeddings := Seq2SeqEmbeddingSpec.forward model.targetEmbedding targetTokens
  Seq2SeqDecoderSpec.forwardTeacherForcing model.decoder targetEmbeddings encoderHidden hTarget

/--
Seq2Seq forward pass for inference-time decoding using discrete token ids.

This embeds the source token ids, encodes them to get an initial decoder hidden state, then runs
greedy decoding for `maxTgtLen` steps starting from the given `start_token`.

Returns:
- logits `(maxTgtLen × tgtVocabSize)`,
- greedy-decoded token ids `(maxTgtLen)`.
-/
def Seq2SeqSpec.forwardInference {srcVocabSize tgtVocabSize embedDim hiddenDim srcSeqLen : Nat}
  (maxTgtLen : Nat)
  (model : Seq2SeqSpec α srcVocabSize tgtVocabSize embedDim hiddenDim)
  (sourceTokens : Tensor (Fin srcVocabSize) (.dim srcSeqLen .scalar))
  (startToken : Fin tgtVocabSize) :
  (Tensor α (.dim maxTgtLen (.dim tgtVocabSize .scalar)) ×
    Tensor (Fin tgtVocabSize) (.dim maxTgtLen .scalar)) :=
  let sourceEmbeddings := Seq2SeqEmbeddingSpec.forward model.sourceEmbedding sourceTokens
  let (_encoderOutputs, encoderHidden) :=
    Seq2SeqRNNEncoderSpec.forward model.encoder sourceEmbeddings none
  Seq2SeqDecoderSpec.forwardInference model.decoder encoderHidden model.targetEmbedding.embedding
    startToken maxTgtLen

/-!
### Differentiable training + backward (one-hot inputs)

This is the “full” training interface for the Seq2Seq baseline.
-/

/--
Differentiable forward pass for training (teacher forcing) using one-hot/token-distribution inputs.

This is the same computation as `Seq2SeqSpec.forwardTraining`, except that embedding lookup is
expressed as a matrix multiplication (`forwardOneHot`), so gradients can flow into the embedding
tables and back into upstream token distributions (if desired).
-/
def Seq2SeqSpec.forwardTrainingOneHot
  {srcVocabSize tgtVocabSize embedDim hiddenDim srcSeqLen tgtSeqLen : Nat}
  (model : Seq2SeqSpec α srcVocabSize tgtVocabSize embedDim hiddenDim)
  (srcOneHot : Tensor α (.dim srcSeqLen (.dim srcVocabSize .scalar)))
  (tgtOneHot : Tensor α (.dim tgtSeqLen (.dim tgtVocabSize .scalar)))
  (hTgt : tgtSeqLen ≠ 0) :
  Tensor α (.dim tgtSeqLen (.dim tgtVocabSize .scalar)) :=
  let src_embeds := Seq2SeqEmbeddingSpec.forwardOneHot model.sourceEmbedding srcOneHot
  let (_encOut, encHidden) := Seq2SeqRNNEncoderSpec.forward model.encoder src_embeds none
  let tgt_embeds := Seq2SeqEmbeddingSpec.forwardOneHot model.targetEmbedding tgtOneHot
  Seq2SeqDecoderSpec.forwardTeacherForcing model.decoder tgt_embeds encHidden hTgt

/--
Per-timestep cross-entropy loss for the differentiable Seq2Seq baseline.

Computes:
1. logits via `Seq2SeqSpec.forwardTrainingOneHot`,
2. probabilities via `softmax`,
3. cross-entropy against the target token distribution at each timestep.

PyTorch analogue: `nn.CrossEntropyLoss` applied per timestep (with probabilities represented as
  one-hot).
-/
def Seq2SeqSpec.crossEntropyLossOneHot
  {srcVocabSize tgtVocabSize embedDim hiddenDim srcSeqLen tgtSeqLen : Nat}
  [Shape.HasNonemptyAxis 1 (.dim tgtSeqLen (.dim tgtVocabSize .scalar))]
  (model : Seq2SeqSpec α srcVocabSize tgtVocabSize embedDim hiddenDim)
  (srcOneHot : Tensor α (.dim srcSeqLen (.dim srcVocabSize .scalar)))
  (tgtOneHot : Tensor α (.dim tgtSeqLen (.dim tgtVocabSize .scalar)))
  (hTgt : tgtSeqLen ≠ 0) : α :=
  let logits := Seq2SeqSpec.forwardTrainingOneHot (α := α) model srcOneHot tgtOneHot hTgt
  let probs := Activation.softmaxLastSpec logits
  crossEntropySpec 1 probs tgtOneHot

/--
Compute `(loss, grads)` for the Seq2Seq baseline under per-timestep cross-entropy.

This returns gradients for:
- both embedding tables,
- the encoder RNN,
- the decoder RNN,
- the decoder output projection,
- and decoder self-attention (if present).
-/
def Seq2SeqSpec.crossEntropyGradOneHot
  {srcVocabSize tgtVocabSize embedDim hiddenDim srcSeqLen tgtSeqLen : Nat}
  [Shape.HasNonemptyAxis 1 (.dim tgtSeqLen (.dim tgtVocabSize .scalar))]
  (model : Seq2SeqSpec α srcVocabSize tgtVocabSize embedDim hiddenDim)
  (srcOneHot : Tensor α (.dim srcSeqLen (.dim srcVocabSize .scalar)))
  (tgtOneHot : Tensor α (.dim tgtSeqLen (.dim tgtVocabSize .scalar)))
  (hTgt : tgtSeqLen ≠ 0) :
  (α × Seq2SeqGrads α srcVocabSize tgtVocabSize embedDim hiddenDim) :=

  let src_embeds := Seq2SeqEmbeddingSpec.forwardOneHot model.sourceEmbedding srcOneHot
  let (encHiddens, encHidden) := Seq2SeqRNNEncoderSpec.forward model.encoder src_embeds none
  let tgt_embeds := Seq2SeqEmbeddingSpec.forwardOneHot model.targetEmbedding tgtOneHot

  let logits := Seq2SeqDecoderSpec.forwardTeacherForcing model.decoder tgt_embeds encHidden hTgt
  let probs := Activation.softmaxLastSpec logits
  let loss := crossEntropySpec 1 probs tgtOneHot

  let dProbs := crossEntropyDerivSpec 1 probs tgtOneHot
  let dLogits := Activation.softmaxLastBackwardSpec logits dProbs

  let (decRnnGrads, outProjGrads, attnGradsOpt, dTgtEmbeds, dEncHidden) :=
    Seq2SeqDecoderSpec.backwardTeacherForcing (α := α)
      (embedDim := embedDim) (hiddenDim := hiddenDim) (vocabSize := tgtVocabSize) (tgtSeqLen :=
        tgtSeqLen)
      model.decoder tgt_embeds encHidden hTgt dLogits

  let (dTgtEmbTable, _dTgtOneHot) :=
    Seq2SeqEmbeddingSpec.backwardOneHot (α := α)
      (vocabSize := tgtVocabSize) (embedDim := embedDim) (seqLen := tgtSeqLen)
      model.targetEmbedding tgtOneHot dTgtEmbeds

  -- Encoder only feeds the decoder through the final hidden state.
  let dEncHiddens :=
    if _h0 : srcSeqLen = 0 then
      fill 0 (.dim srcSeqLen (.dim hiddenDim .scalar))
    else
      Tensor.dim (fun i =>
        if _ : i.val = srcSeqLen - 1 then dEncHidden else fill 0 (.dim hiddenDim .scalar))

  let (dW_enc, db_enc, dSrcEmbeds, _dH0_enc) :=
    rnnSequenceBackwardSpec model.encoder.rnn src_embeds (fill 0 (.dim hiddenDim .scalar))
      encHiddens dEncHiddens
  let encGrads : Seq2SeqRNNGrads α embedDim hiddenDim :=
    { weight := dW_enc, bias := db_enc }

  let (dSrcEmbTable, _dSrcOneHot) :=
    Seq2SeqEmbeddingSpec.backwardOneHot (α := α)
      (vocabSize := srcVocabSize) (embedDim := embedDim) (seqLen := srcSeqLen)
      model.sourceEmbedding srcOneHot dSrcEmbeds

  let grads : Seq2SeqGrads α srcVocabSize tgtVocabSize embedDim hiddenDim :=
    { sourceEmbedding := dSrcEmbTable
      targetEmbedding := dTgtEmbTable
      encoder := encGrads
      decoderRnn := decRnnGrads
      outputProjection := outProjGrads
      decoderAttention := attnGradsOpt }

  (loss, grads)

/--
Attention-augmented Seq2Seq specification (simple encoder-output attention).

This record extends the baseline with an additional projection matrix used by the helper
attention functions below (`compute_attention_weights_spec` / `apply_attention_spec`).

Note: this file includes these attention helpers as a building block; the main baseline forward
passes above do not integrate encoder-decoder cross-attention by default.
-/
structure AttentionSeq2SeqSpec (α : Type) [Numbers α] (srcVocabSize tgtVocabSize embedDim hiddenDim
  : Nat) where
  /-- Source embedding table. -/
  sourceEmbedding : Seq2SeqEmbeddingSpec α srcVocabSize embedDim
  /-- Target embedding table. -/
  targetEmbedding : Seq2SeqEmbeddingSpec α tgtVocabSize embedDim
  /-- Encoder RNN parameters. -/
  encoder : Seq2SeqRNNEncoderSpec α embedDim hiddenDim
  /-- Decoder parameters (RNN + output projection + optional self-attention). -/
  decoder : Seq2SeqDecoderSpec α embedDim hiddenDim tgtVocabSize
  /-- Attention projection matrix used to score encoder outputs against the decoder hidden state. -/
  attentionWeights : Tensor α (.dim hiddenDim (.dim hiddenDim .scalar))

/--
Compute attention weights over encoder outputs for a single decoder hidden state.

This is a simple dot-product style attention:
1. project the decoder hidden state (`attention_weights · decoder_hidden`),
2. score each encoder hidden vector by an elementwise product + sum,
3. normalize scores with `softmax` over the sequence axis.

It is inspired by classic encoder-decoder attention mechanisms (Bahdanau-style), and this spec keeps
the scoring rule compact.
-/
def computeAttentionWeightsSpec {α : Type} [Context α] {hiddenDim seqLen : Nat}
  (attentionWeights : Tensor α (.dim hiddenDim (.dim hiddenDim .scalar)))
  (decoderHidden : Tensor α (.dim hiddenDim .scalar))
  (encoderOutputs : Tensor α (.dim seqLen (.dim hiddenDim .scalar)))
  (h1 : hiddenDim ≠ 0) (_h2 : seqLen ≠ 0) :
  Tensor α (.dim seqLen .scalar) :=
  -- Compute attention scores
  let projected_hidden := matVecMulSpec attentionWeights decoderHidden
  let scores := Tensor.dim (fun i =>
    match get encoderOutputs i with
    | Tensor.dim encoder_hidden =>
      let encoder_vec := Tensor.dim encoder_hidden
      let mul_vec := mulSpec projected_hidden encoder_vec
      reduceSum 0 mul_vec (Shape.hasNonemptyAxisZeroOfNe h1).proof
  )
  -- Apply softmax to get attention weights
  Activation.softmaxLastSpec scores

/--
Apply attention weights to encoder outputs (weighted sum / context vector).

Given attention weights `a : (seqLen)` and encoder outputs `H : (seqLen × hiddenDim)`, returns the
context vector `c = Σ_i a_i · H_i : (hiddenDim)`.
-/
def applyAttentionSpec {hiddenDim seqLen : Nat}
  (attentionWeights : Tensor α (.dim seqLen .scalar))
  (encoderOutputs : Tensor α (.dim seqLen (.dim hiddenDim .scalar)))
  (h1 : seqLen ≠ 0) (_h2 : hiddenDim ≠ 0) :
  Tensor α (.dim hiddenDim .scalar) :=
  -- Weighted sum of encoder outputs
  let weighted_outputs := Tensor.dim (fun i =>
    match get attentionWeights i, get encoderOutputs i with
    | Tensor.scalar weight, Tensor.dim encoder_hidden =>
      let encoder_vec := Tensor.dim encoder_hidden
      scaleSpec encoder_vec weight
  )
  -- Sum across sequence dimension
  reduceSum 0 weighted_outputs (Shape.hasNonemptyAxisZeroOfNe h1).proof

end Spec

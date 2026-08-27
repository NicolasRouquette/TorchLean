/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor

/-!
# BugZoo: KV-cache contracts

LLM inference-engine bug reports include cache-shift bugs, RoPE/position mismatches, shape mistakes, and
resource/configuration errors. TorchLean does not verify a full serving engine, paged
attention allocator, or multi-GPU scheduler. The useful first step is still precise: represent the
cache update as a typed tensor operation and prove the append invariant we rely on.

Reference:
- Liu et al., "A First Look at Bugs in LLM Inference Engines", 2025.

This file proves the cache append boundary. A stronger future theorem should connect cached decode
to full-sequence attention:

$$
\operatorname{decodeWithCache}(\mathit{prefix},\mathit{newToken})
=\operatorname{fullAttention}(\mathit{prefix}\mathbin{+\!\!+}[\mathit{newToken}]).
$$

under the same mask, RoPE/position encoding, and numeric semantics.
-/

@[expose] public section

namespace NN.Examples.BugZoo.KVCache

open TorchLean

/-- A key/value cache with an explicit sequence length and head dimension. -/
structure Cache (α : Type) (seqLen headDim : Nat) where
  /-- Cached key vectors, indexed by time. -/
  keys : Tensor α [seqLen, headDim]
  /-- Cached value vectors, indexed by time. -/
  values : Tensor α [seqLen, headDim]

/-- View one token vector as a length-one sequence. -/
def singletonToken {α : Type} {headDim : Nat}
    (x : Tensor α [headDim]) :
    Tensor α [1, headDim] := by
  change Spec.Tensor α (([headDim] : Spec.Shape).insertAxis 0 1)
  exact TorchLean.Tensor.stack 0 fun _ => x

@[simp] theorem singletonToken_get {α : Type} {headDim : Nat}
    (x : Tensor α [headDim]) (i : Fin 1) :
    Spec.get (singletonToken x) i = x := by
  change Spec.get (TorchLean.Tensor.stack 0 fun _ => x) i = x
  rfl

/-- Append one token vector to a sequence cache along the time axis. -/
def appendToken {α : Type} {seqLen headDim : Nat}
    (past : Tensor α [seqLen, headDim])
    (newToken : Tensor α [headDim]) :
    Tensor α [seqLen + 1, headDim] :=
  Spec.Tensor.concatAxisSpec .scalar past (singletonToken newToken)

@[simp] theorem appendToken_last {α : Type} {seqLen headDim : Nat}
    (past : Tensor α [seqLen, headDim])
    (newToken : Tensor α [headDim]) :
    Spec.get (appendToken past newToken) ⟨seqLen, Nat.lt_succ_self seqLen⟩ = newToken := by
  cases past with
  | dim rows =>
      cases hSingleton : singletonToken newToken with
      | dim singletonRows =>
          simpa [appendToken, hSingleton, Spec.Tensor.concatAxisSpec, Spec.get]
            using singletonToken_get newToken ⟨0, Nat.zero_lt_succ 0⟩

/-- Append both key and value vectors to the KV cache. -/
def appendKV {α : Type} {seqLen headDim : Nat}
    (cache : Cache α seqLen headDim)
    (newKey newValue : Tensor α [headDim]) :
    Cache α (seqLen + 1) headDim where
  keys := appendToken cache.keys newKey
  values := appendToken cache.values newValue

/-- The newly appended key is exactly the final key in the updated cache. -/
theorem appendKV_last_key {α : Type} {seqLen headDim : Nat}
    (cache : Cache α seqLen headDim)
    (newKey newValue : Tensor α [headDim]) :
    Spec.get (appendKV cache newKey newValue).keys
        ⟨seqLen, Nat.lt_succ_self seqLen⟩ = newKey := by
  exact appendToken_last cache.keys newKey

/-- The newly appended value is exactly the final value in the updated cache. -/
theorem appendKV_last_value {α : Type} {seqLen headDim : Nat}
    (cache : Cache α seqLen headDim)
    (newKey newValue : Tensor α [headDim]) :
    Spec.get (appendKV cache newKey newValue).values
        ⟨seqLen, Nat.lt_succ_self seqLen⟩ = newValue := by
  exact appendToken_last cache.values newValue

end NN.Examples.BugZoo.KVCache

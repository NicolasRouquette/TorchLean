/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural.Builders
public import NN.Spec.Layers.PositionalEncoding

import Mathlib.Algebra.Order.Algebra

@[expose] public section

namespace TorchLean

/-!
# Positional Encodings

Trainable learned positions, fixed sinusoidal encodings, and rotary positional embeddings for
shape-typed sequential models.
-/

namespace nn
namespace Internal

/--
Learned positional embedding configuration.

This is a trainable parameter tensor of shape `(seqLen × embedDim)` that is broadcast across any
leading dimensions and added to the input.
-/
structure LearnedPositionalEmbedding where
  /-- Seed for deterministic initialization. -/
  seedPos : Nat := 0
  /-- Initialization scheme for the positional embedding table. -/
  posInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.02) 0.02

/--
Add learned positional embeddings to the `(seqLen × embedDim)` suffix of a tensor.

PyTorch analogue: `x + pos[:seqLen]` where `pos` is a parameter table.
-/
def learnedPositionalEmbedding (leading : List Nat := []) {seqLen embedDim : Nat}
    (cfg : LearnedPositionalEmbedding := {}) :
    Sequential
      (leading ++ [seqLen, embedDim])
      (leading ++ [seqLen, embedDim]) :=
  let leadingShape := Spec.Shape.ofList leading
  let posShape : Spec.Shape := [seqLen, embedDim]
  let xShape : Spec.Shape := leading ++ [seqLen, embedDim]
  let pos0 : Tensor Float posShape :=
    _root_.Runtime.Autograd.Torch.Init.tensor (s := posShape) (sch := cfg.posInit) (seed := cfg.seedPos)
  letI : Spec.Shape.BroadcastTo posShape xShape := by
    simpa [xShape, leadingShape, Spec.Shape.ofList_append] using
      (show Spec.Shape.BroadcastTo posShape (leadingShape.concat posShape) from
        ⟨Spec.Shape.CanBroadcastTo.prependTarget leadingShape posShape⟩)
  letI : Spec.Shape.BroadcastTo xShape xShape :=
    ⟨Spec.Shape.CanBroadcastTo.refl xShape⟩
  of
    { kind := "LearnedPositionalEmbedding"
      stateShapes := [posShape]
      initState := _root_.TorchLean.TensorPack! pos0
      runtimeInit := some (.cons
        (_root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.FloatInit.ofScheme
          cfg.posInit cfg.seedPos) .nil)
      requiresGrad := #[true]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun pos x =>
            -- Broadcast the positional table across every leading axis.
            (_root_.Runtime.Autograd.TorchLean.F.addB (m := m) (α := α)
              (s₁ := posShape) (s₂ := xShape) (t := xShape) pos x)
    }

/--
Sinusoidal positional encoding configuration.

Classic non-trainable Transformer sinusoidal encoding, added to token embeddings. `startPos` is an
absolute-position offset for KV-cache decoding.
-/
structure SinusoidalPositionalEncoding where
  /-- Absolute position offset for the first row of the encoding table. -/
  startPos : Nat := 0

/--
Add sinusoidal positional encodings to the `(seqLen × embedDim)` suffix of a tensor.

Implementation:
- precompute `PE : (seqLen × embedDim)` at initialization time (stored as a non-trainable buffer),
- broadcast it across every leading axis and add it to the input.
-/
def sinusoidalPositionalEncoding (leading : List Nat := []) {seqLen embedDim : Nat}
    (cfg : SinusoidalPositionalEncoding := {}) :
    Sequential
      (leading ++ [seqLen, embedDim])
      (leading ++ [seqLen, embedDim]) :=
  let leadingShape := Spec.Shape.ofList leading
  let peShape : Spec.Shape := [seqLen, embedDim]
  let xShape : Spec.Shape := leading ++ [seqLen, embedDim]
  let pe0 : Tensor Float peShape :=
    Spec.sinusoidalPositionalEncodingSpec (α := Float) seqLen embedDim cfg.startPos
  letI : Spec.Shape.BroadcastTo peShape xShape := by
    simpa [xShape, leadingShape, Spec.Shape.ofList_append] using
      (show Spec.Shape.BroadcastTo peShape (leadingShape.concat peShape) from
        ⟨Spec.Shape.CanBroadcastTo.prependTarget leadingShape peShape⟩)
  letI : Spec.Shape.BroadcastTo xShape xShape :=
    ⟨Spec.Shape.CanBroadcastTo.refl xShape⟩
  of
    { kind := "SinusoidalPositionalEncoding"
      stateShapes := [peShape]
      initState := _root_.TorchLean.TensorPack! pe0
      requiresGrad := #[false]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun pe x =>
            -- Broadcast `PE : (seqLen × embedDim)` across every leading axis.
            (_root_.Runtime.Autograd.TorchLean.F.addB (m := m) (α := α)
              (s₁ := peShape) (s₂ := xShape) (t := xShape) pe x)
    }

/--
Rotary positional embedding (RoPE) configuration.

`startPos` is an absolute-position offset for KV-cache decoding.
-/
structure RotaryEmbeddingConfig where
  /-- Absolute position offset for the first row of RoPE angles. -/
  startPos : Nat := 0

/--
Apply RoPE to the `(seqLen × headDim)` suffix of a tensor.

This matches the standard identity:

$$
\operatorname{rope}(x)
  = x \odot \cos + \operatorname{rotatePairs}(x) \odot \sin
$$

where `cos` and `sin` depend only on `(pos, dim)` and broadcast across every leading axis.

Notes:
- This layer is *differentiable* (gradients flow through the rotation), but it has no trainable
  parameters; the precomputed `cos`/`sin` tables are stored as non-trainable buffers.
- The pure spec version is in `NN.Spec.Layers.PositionalEncoding` (`Spec.rope_apply_heads_spec`).
-/
def rope (leading : List Nat := []) {seqLen headDim : Nat}
    (cfg : RotaryEmbeddingConfig := {}) :
    Sequential
      (leading ++ [seqLen, headDim])
      (leading ++ [seqLen, headDim]) :=
  let leadingShape := Spec.Shape.ofList leading
  let xShape : Spec.Shape := leading ++ [seqLen, headDim]
  let csShape : Spec.Shape := [seqLen, headDim]

  -- Precompute cos/sin tables (as Float buffers). These depend only on `(seqLen, headDim, startPos)`.
  let cos0 : Tensor Float csShape :=
    Spec.Tensor.dim (fun (pos : Fin seqLen) =>
      Spec.ropeCosVectorSpec (α := Float) (cfg.startPos + pos.val) headDim)
  let sin0 : Tensor Float csShape :=
    Spec.Tensor.dim (fun (pos : Fin seqLen) =>
      Spec.ropeSinVectorSpec (α := Float) (cfg.startPos + pos.val) headDim)

  -- Column permutation indices implementing pairwise swap `(0↔1, 2↔3, ...)`.
  -- When `headDim` is odd, the last index is left unchanged.
  let permIdx : Tensor (Fin headDim) [headDim] :=
    Spec.Tensor.dim (fun (j : Fin headDim) =>
      let idx := j.val
      let out : Fin headDim :=
        if h : idx % 2 = 0 ∧ idx + 1 < headDim then
          ⟨idx + 1, h.2⟩
        else if idx % 2 = 0 then
          j
        else
          ⟨idx - 1, Nat.lt_of_le_of_lt (Nat.sub_le idx 1) j.isLt⟩
      Spec.Tensor.scalar out)

  letI : Spec.Shape.BroadcastTo csShape xShape := by
    simpa [xShape, leadingShape, Spec.Shape.ofList_append] using
      (show Spec.Shape.BroadcastTo csShape (leadingShape.concat csShape) from
        ⟨Spec.Shape.CanBroadcastTo.prependTarget leadingShape csShape⟩)
  letI : Spec.Shape.BroadcastTo xShape xShape :=
    ⟨Spec.Shape.CanBroadcastTo.refl xShape⟩

  of
    { kind := "RoPE"
      stateShapes := [csShape, csShape]
      initState := _root_.TorchLean.TensorPack! cos0, sin0
      requiresGrad := #[false, false]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun cos sin x =>
            ((do
            -- Rotate last-dim pairs by a fixed 2D permutation/sign pattern.
            let rowsFold : Nat := Spec.Shape.size leadingShape * seqLen
            let flatShape : Spec.Shape := [rowsFold, headDim]

            let x2d ←
              _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
                (s₁ := xShape) (s₂ := flatShape)
                x (by
                  simp [xShape, flatShape, rowsFold, leadingShape, Spec.Shape.size_concat,
                    Spec.Shape.size_ofList, Spec.Shape.size, Nat.mul_assoc])

            let xT ←
              _root_.Runtime.Autograd.Torch.swapAdjacentAtDepth (m := m) (α := α)
                (s := flatShape) 0 x2d

            let xPerm ←
              _root_.Runtime.Autograd.Torch.indexSelect (m := m) (α := α)
                (s := [headDim, rowsFold]) 0 headDim xT
                (_root_.Runtime.Autograd.Torch.dataConst (m := m) (α := α) permIdx)

            let xBack ←
              _root_.Runtime.Autograd.Torch.swapAdjacentAtDepth (m := m) (α := α)
                (s := [headDim, rowsFold]) 0 xPerm

            -- Sign pattern for `rotatePairs`: even outputs get a negation (except the final unpaired entry).
            let signT : Tensor α [headDim] :=
              Spec.Tensor.dim (fun (j : Fin headDim) =>
                let idx := j.val
                let v : α :=
                  if idx % 2 = 0 ∧ idx + 1 < headDim then (-1 : α) else (1 : α)
                Spec.Tensor.scalar v)
            let sign ← _root_.Runtime.Autograd.Torch.const (m := m) (α := α) (s := [headDim]) signT

            let xRot2d ←
              _root_.Runtime.Autograd.TorchLean.F.mulB (m := m) (α := α)
                (s₁ := flatShape) (s₂ := [headDim]) (t := flatShape)
                xBack sign

            let xRot ←
              _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
                (s₁ := flatShape) (s₂ := xShape)
                xRot2d (by
                  simp [xShape, flatShape, rowsFold, leadingShape, Spec.Shape.size_concat,
                    Spec.Shape.size_ofList, Spec.Shape.size, Nat.mul_assoc])

            -- Apply the RoPE formula with broadcasting of `cos/sin : (seqLen × headDim)`.
            let xCos ←
              _root_.Runtime.Autograd.TorchLean.F.mulB (m := m) (α := α)
                (s₁ := xShape) (s₂ := csShape) (t := xShape)
                x cos
            let rotSin ←
              _root_.Runtime.Autograd.TorchLean.F.mulB (m := m) (α := α)
                (s₁ := xShape) (s₂ := csShape) (t := xShape)
                xRot sin
            _root_.Runtime.Autograd.Torch.add (m := m) (α := α) (s := xShape) xCos rotSin
            ) : m (_root_.TorchLean.Runtime.ValueRef (m := m) (α := α) xShape))
    }

end Internal
end nn
end TorchLean

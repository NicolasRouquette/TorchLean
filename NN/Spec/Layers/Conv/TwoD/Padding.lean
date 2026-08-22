/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorOps

/-!
# Two-Dimensional Convolution Padding

Materialized symmetric zero-padding for channels-first tensors of shape `[C, H, W]`.

The generic convolution specification reads padded coordinates without materializing a larger
tensor. This module exists for proofs that compare that implementation with the traditional
"pad, then convolve" presentation of two-dimensional convolution.
-/

@[expose] public section

namespace Spec

open Tensor

/--
Pad both spatial axes of a channels-first tensor with zeros.

The input has shape `[C, H, W]`; the output has shape `[C, H + 2p, W + 2p]`. The channel axis is
unchanged.
-/
def padChannelsFirst2d {α : Type} [Context α] {channels height width : Nat}
    (input : Tensor α (.dim channels (.dim height (.dim width .scalar)))) (padding : Nat) :
    Tensor α
      (.dim channels (.dim (height + 2 * padding) (.dim (width + 2 * padding) .scalar))) :=
  Tensor.dim fun channel =>
    Tensor.dim fun row =>
      Tensor.dim fun col =>
        if row.val < padding ∨ col.val < padding then
          Tensor.scalar 0
        else
          Tensor.scalar <| getAtOrZero input
            [channel.val, row.val - padding, col.val - padding]

/-- Reading materialized padding agrees with an offset read from the input tensor. -/
theorem getAtOrZero_padChannelsFirst2d
    {α : Type} [Context α] {channels height width padding : Nat}
    (input : Tensor α (.dim channels (.dim height (.dim width .scalar))))
    (channel : Fin channels) (row col : Nat) :
    getAtOrZero
        (padChannelsFirst2d (channels := channels) (height := height) (width := width)
          input padding)
        [channel.val, row, col] =
      if row < padding ∨ col < padding then
        0
      else
        getAtOrZero input [channel.val, row - padding, col - padding] := by
  cases input with
  | dim values =>
      by_cases hRow : row < height + 2 * padding
      · by_cases hCol : col < width + 2 * padding
        · by_cases hPadding : row < padding ∨ col < padding
          · simp [padChannelsFirst2d, channel.isLt, hRow, hCol, hPadding]
          · simp [padChannelsFirst2d, channel.isLt, hRow, hCol, hPadding]
        · have hColGe : width ≤ col - padding := by
            have hTwoPadding : width + 2 * padding ≤ col := Nat.le_of_not_gt hCol
            have hOnePadding : width + padding ≤ col - padding := by
              have : (width + padding) + padding ≤ col := by
                simpa [two_mul, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using hTwoPadding
              exact Nat.le_sub_of_add_le this
            exact le_trans (Nat.le_add_right width padding) hOnePadding
          have hColOut : ¬ col - padding < width := Nat.not_lt_of_ge hColGe
          by_cases hPadding : row < padding ∨ col < padding
          · simp [padChannelsFirst2d, channel.isLt, hRow, hCol, hPadding]
          · by_cases hRowIn : row - padding < height
            · cases hChannel : values channel with
              | dim rows =>
                  cases hSelectedRow : rows ⟨row - padding, hRowIn⟩ with
                  | dim cols =>
                      simp [padChannelsFirst2d, channel.isLt, hRow, hCol, hPadding, hChannel,
                        hRowIn, hSelectedRow, hColOut]
            · cases hChannel : values channel with
              | dim rows =>
                  simp [padChannelsFirst2d, channel.isLt, hRow, hCol, hPadding, hChannel,
                    hRowIn]
      · have hRowGe : height ≤ row - padding := by
          have hTwoPadding : height + 2 * padding ≤ row := Nat.le_of_not_gt hRow
          have hOnePadding : height + padding ≤ row - padding := by
            have : (height + padding) + padding ≤ row := by
              simpa [two_mul, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using hTwoPadding
            exact Nat.le_sub_of_add_le this
          exact le_trans (Nat.le_add_right height padding) hOnePadding
        have hRowOut : ¬ row - padding < height := Nat.not_lt_of_ge hRowGe
        by_cases hPadding : row < padding ∨ col < padding
        · simp [padChannelsFirst2d, channel.isLt, hRow, hPadding]
        · cases hChannel : values channel with
          | dim rows =>
              simp [padChannelsFirst2d, channel.isLt, hRow, hPadding, hChannel, hRowOut]

/-- An in-bounds input coordinate is preserved after shifting it by the padding width. -/
theorem getAtOrZero_padChannelsFirst2d_shift
    {α : Type} [Context α] {channels height width padding : Nat}
    (input : Tensor α (.dim channels (.dim height (.dim width .scalar))))
    (channel : Fin channels) (row : Fin height) (col : Fin width) :
    getAtOrZero
        (padChannelsFirst2d (channels := channels) (height := height) (width := width)
          input padding)
        [channel.val, row.val + padding, col.val + padding] =
      getAtOrZero input [channel.val, row.val, col.val] := by
  rw [getAtOrZero_padChannelsFirst2d]
  simp

end Spec

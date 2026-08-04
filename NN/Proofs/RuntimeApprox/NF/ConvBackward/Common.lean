/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.BackwardOps
public import NN.Proofs.RuntimeApprox.NF.ConvForward
public import NN.Proofs.RuntimeApprox.NF.Utils
public import NN.Spec.Layers.Utils

/-!
# Conv2D Backward Approximation

NF (rounded) backend: Conv2D backward (VJP) runtime→spec approximation.

This file proves soundness of explicit bounds for the three Conv2D gradients computed by
`Spec.conv2d_backward_spec`:
- kernel gradient
- bias gradient
- input gradient

Each gradient has a different nested-indexing pattern, so the proof keeps the three bound families
visible rather than hiding them behind one large opaque lemma.
The important public objects are the tensor-level bounds (`conv2d*BoundTensor`), the approximation
theorems (`approxT_conv2d_*_deriv_spec`), and `conv2dRevNode`, which packages Conv2D as a `RevNode`
so it composes via `RevGraph.backprop_approx`.

PyTorch analogue: these are the VJP/gradient computations produced by Autograd for Conv2D.
https://pytorch.org/docs/stable/autograd.html
https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html

## Map of this file
- Shared padding/read lemmas (to relate the padded-input branches to the original `approxT`
  hypothesis).
- Bias gradient bounds: `conv2dBiasPointBound`, `approx_conv2d_bias_point`, and tensor-lifted
  bound.
- Kernel gradient bounds: `conv2dKernelPointBound`, `approx_conv2d_kernel_point`, and
  tensor-lifted bound.
- Input gradient bounds: `conv2dInputPointBound`, `approx_conv2d_input_point`, and tensor-lifted
  bound.
- `conv2dRevNode`: packaging as a `RevNode` so the bound composes inside larger graphs.

## References
- Dumoulin & Visin, *A guide to convolution arithmetic for deep learning* (indexing/stride/padding
  conventions).
- Goodfellow, Bengio, Courville, *Deep Learning* (MIT Press, 2016), convolution/backprop background.
- Baydin et al., *Automatic Differentiation in Machine Learning: a Survey* (JMLR 2018) (VJP
  framing).
-/

@[expose] public section


namespace Proofs
namespace RuntimeApprox

open Spec
open Tensor
open NN.MLTheory.Robustness.Spec

noncomputable section

namespace NFBackend

open TorchLean.Floats

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]
variable {rnd : ℝ → ℤ} [NeuralValidRndToNearest rnd]

local notation "R" => TorchLean.Floats.NF β fexp rnd


-- ---------------------------------------------------------------------------
-- Fold helper: turn nested zero-start folds into threaded accumulator folds.
-- ---------------------------------------------------------------------------

lemma specFold5_eq_threadFold5
    {α : Type} [AddMonoid α]
    {outC out_h out_w kH kW : Nat}
    (term : Fin outC → Fin out_h → Fin out_w → Fin kH → Fin kW → α) :
    let specFold : α :=
      (List.finRange outC).foldl (fun acc out_ch =>
          acc +
            (List.finRange out_h).foldl (fun acc out_i =>
                acc +
                  (List.finRange out_w).foldl (fun acc out_j =>
                      acc +
                        (List.finRange kH).foldl (fun acc di =>
                            acc +
                              (List.finRange kW).foldl (fun acc dj =>
                                  acc + term out_ch out_i out_j di dj)
                                0)
                          0)
                    0)
              0)
        0
    let threadFold : α :=
      (List.finRange outC).foldl (fun accC out_ch =>
          (List.finRange out_h).foldl (fun accH out_i =>
              (List.finRange out_w).foldl (fun accW out_j =>
                  (List.finRange kH).foldl (fun accKH di =>
                      (List.finRange kW).foldl (fun accKW dj =>
                          accKW + term out_ch out_i out_j di dj)
                        accKH)
                    accW)
                accH)
            accC)
        0
    specFold = threadFold := by
  intro specFold threadFold
  classical
  -- Auxiliary "spec-sum" functions at each nested level.
  let sumKW0 : Fin outC → Fin out_h → Fin out_w → Fin kH → α :=
    fun out_ch out_i out_j di =>
      (List.finRange kW).foldl (fun acc dj => acc + term out_ch out_i out_j di dj) 0
  let sumKH0 : Fin outC → Fin out_h → Fin out_w → α :=
    fun out_ch out_i out_j =>
      (List.finRange kH).foldl (fun acc di => acc + sumKW0 out_ch out_i out_j di) 0
  let sumW0 : Fin outC → Fin out_h → α :=
    fun out_ch out_i =>
      (List.finRange out_w).foldl (fun acc out_j => acc + sumKH0 out_ch out_i out_j) 0
  let sumH0 : Fin outC → α :=
    fun out_ch =>
      (List.finRange out_h).foldl (fun acc out_i => acc + sumW0 out_ch out_i) 0

  -- Rewrite the spec fold into the threaded fold by repeatedly pushing the accumulator inward.
  have hC :
      (List.finRange outC).foldl (fun acc out_ch => acc + sumH0 out_ch) 0 =
        (List.finRange outC).foldl (fun accC out_ch =>
            (List.finRange out_h).foldl (fun accH out_i =>
                (List.finRange out_w).foldl (fun accW out_j =>
                    (List.finRange kH).foldl (fun accKH di =>
                        (List.finRange kW).foldl (fun accKW dj =>
                            accKW + term out_ch out_i out_j di dj)
                          accKH)
                      accW)
                  accH)
              accC)
          0 := by
    refine foldl_congr (l := List.finRange outC)
      (f := fun acc out_ch => acc + sumH0 out_ch)
      (g := fun accC out_ch =>
        (List.finRange out_h).foldl (fun accH out_i =>
            (List.finRange out_w).foldl (fun accW out_j =>
                (List.finRange kH).foldl (fun accKH di =>
                    (List.finRange kW).foldl (fun accKW dj =>
                        accKW + term out_ch out_i out_j di dj)
                      accKH)
                  accW)
              accH)
          accC)
      (init := (0 : α)) ?_
    intro accC out_ch
    -- Push `accC` into the out_h fold, then convert the body similarly at deeper levels.
    have hPushH :
        accC + sumH0 out_ch =
          (List.finRange out_h).foldl (fun accH out_i => accH + sumW0 out_ch out_i) accC := by
      simpa [sumH0, add_assoc] using
        (Spec.add_finRange_foldl_add_zero
          (n := out_h) (f := fun out_i => sumW0 out_ch out_i) (acc := accC))
    -- Now rewrite the out_h fold body to the threaded out_w fold.
    have hH :
        (List.finRange out_h).foldl (fun accH out_i => accH + sumW0 out_ch out_i) accC =
          (List.finRange out_h).foldl (fun accH out_i =>
              (List.finRange out_w).foldl (fun accW out_j =>
                  (List.finRange kH).foldl (fun accKH di =>
                      (List.finRange kW).foldl (fun accKW dj =>
                          accKW + term out_ch out_i out_j di dj)
                        accKH)
                    accW)
                accH)
            accC := by
      refine foldl_congr (l := List.finRange out_h)
        (f := fun accH out_i => accH + sumW0 out_ch out_i)
        (g := fun accH out_i =>
          (List.finRange out_w).foldl (fun accW out_j =>
              (List.finRange kH).foldl (fun accKH di =>
                  (List.finRange kW).foldl (fun accKW dj =>
                      accKW + term out_ch out_i out_j di dj)
                    accKH)
                accW)
            accH)
        (init := accC) ?_
      intro accH out_i
      have hPushW :
          accH + sumW0 out_ch out_i =
            (List.finRange out_w).foldl (fun accW out_j => accW + sumKH0 out_ch out_i out_j) accH :=
              by
        simpa [sumW0, add_assoc] using
          (Spec.add_finRange_foldl_add_zero
            (n := out_w) (f := fun out_j => sumKH0 out_ch out_i out_j) (acc := accH))
      -- Convert the out_w fold body to the threaded kH/kW fold.
      have hW :
          (List.finRange out_w).foldl (fun accW out_j => accW + sumKH0 out_ch out_i out_j) accH =
            (List.finRange out_w).foldl (fun accW out_j =>
                (List.finRange kH).foldl (fun accKH di =>
                    (List.finRange kW).foldl (fun accKW dj =>
                        accKW + term out_ch out_i out_j di dj)
                      accKH)
                  accW)
              accH := by
        refine foldl_congr (l := List.finRange out_w)
          (f := fun accW out_j => accW + sumKH0 out_ch out_i out_j)
          (g := fun accW out_j =>
            (List.finRange kH).foldl (fun accKH di =>
                (List.finRange kW).foldl (fun accKW dj =>
                    accKW + term out_ch out_i out_j di dj)
                  accKH)
              accW)
          (init := accH) ?_
        intro accW out_j
        have hPushKH :
            accW + sumKH0 out_ch out_i out_j =
              (List.finRange kH).foldl (fun accKH di => accKH + sumKW0 out_ch out_i out_j di) accW
                := by
          simpa [sumKH0, add_assoc] using
            (Spec.add_finRange_foldl_add_zero
              (n := kH) (f := fun di => sumKW0 out_ch out_i out_j di) (acc := accW))
        have hKH :
            (List.finRange kH).foldl (fun accKH di => accKH + sumKW0 out_ch out_i out_j di) accW =
              (List.finRange kH).foldl (fun accKH di =>
                  (List.finRange kW).foldl (fun accKW dj =>
                      accKW + term out_ch out_i out_j di dj)
                    accKH)
                accW := by
          refine foldl_congr (l := List.finRange kH)
            (f := fun accKH di => accKH + sumKW0 out_ch out_i out_j di)
            (g := fun accKH di =>
              (List.finRange kW).foldl (fun accKW dj => accKW + term out_ch out_i out_j di dj)
                accKH)
            (init := accW) ?_
          intro accKH di
          simpa [sumKW0, add_assoc] using
            (Spec.add_finRange_foldl_add_zero
              (n := kW) (f := fun dj => term out_ch out_i out_j di dj) (acc := accKH))
        exact hPushKH.trans hKH
      exact hPushW.trans hW
    exact hPushH.trans hH

  simpa [specFold, threadFold, sumH0, sumW0, sumKH0, sumKW0, add_assoc] using hC

-- ---------------------------------------------------------------------------
-- Component selection lemmas (match-based indexing ↔ `get_at_or_zero`)
-- ---------------------------------------------------------------------------

@[simp] lemma get_at_or_zero_tensor_cast {α : Type} [Zero α] {s t : Shape} (h : s = t)
    (x : Tensor α s) (idx : List Nat) :
    getAtOrZero (Tensor.castShape (t := x) h) idx = getAtOrZero x idx := by
  cases h
  rfl

/-- Padded-input helper used by the Conv2D spec: cast when `padding = 0`, otherwise `padMultiChannel`. -/
def paddedInput {α : Type} [Context α] {inC inH inW padding : Nat}
    (img : Spec.Tensor α (.dim inC (.dim inH (.dim inW .scalar)))) :
    Spec.Tensor α (.dim inC (.dim (inH + 2 * padding) (.dim (inW + 2 * padding) .scalar))) :=
  if h4 : padding = 0 then
    tensorCast
      (Shape.dim inC
        (Shape.dim (inH + 2 * padding) (Shape.dim (inW + 2 * padding) Shape.scalar)))
      (by simp; rw [h4])
      img
  else
    Spec.padMultiChannel img padding

lemma get_at_or_zero_paddedInput
    {α : Type} [Context α] {inC inH inW padding : Nat}
    (img : Spec.Tensor α (.dim inC (.dim inH (.dim inW .scalar)))) (c : Fin inC) (p q : Nat) :
    getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) img)
        [c.val, p, q]
      =
    (if _h : p < padding ∨ q < padding then
        (0 : α)
      else
        getAtOrZero img [c.val, p - padding, q - padding]) := by
  classical
  by_cases h0 : padding = 0
  · subst h0
    simp [paddedInput]
  · simpa [paddedInput, h0] using
      (Spec.get_at_or_zero_pad_multi_channel (α := α) (img := img) (c := c) (p := p) (q := q)
        (padding := padding))

lemma mkInputIdx_match_eq_paddedInput
    {α : Type} [Context α] {inC inH inW stride padding : Nat}
    (img : Spec.Tensor α (.dim inC (.dim inH (.dim inW .scalar)))) (c : Fin inC)
    (oi di oj dj : Nat) :
    (match Spec.Private.mkInputIdx? [oi, oj] [di, dj] [stride, stride] [padding, padding] with
      | none => (0 : α)
      | some inIdx => getAtOrZero img (c.val :: inIdx))
      =
    getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) img)
        [c.val, oi * stride + di, oj * stride + dj] := by
  classical
  -- Compare both sides via the explicit `paddedInput` read formula.
  rw [get_at_or_zero_paddedInput (img := img) (c := c) (p := oi * stride + di) (q := oj * stride + dj)]
  by_cases h0 : oi * stride + di < padding
  · simp [Spec.Private.mkInputIdx?, h0]
  · by_cases h1 : oj * stride + dj < padding
    · simp [Spec.Private.mkInputIdx?, h0, h1]
    · simp [Spec.Private.mkInputIdx?, h0, h1]

lemma conv2dKernelFoldRead_eq_paddedFold
    {α : Type} [Context α] {inC outC inH inW outH outW stride padding : Nat}
    (input : Spec.Tensor α (.dim inC (.dim inH (.dim inW .scalar))))
    (grad : Spec.Tensor α (.dim outC (.dim outH (.dim outW .scalar))))
    (out_ch : Fin outC) (in_ch : Fin inC) (di dj : Nat) :
    (List.finRange outH).foldl (fun acc i =>
        (List.finRange outW).foldl (fun acc j =>
          acc +
            (match Spec.Private.mkInputIdx? [i.val, j.val] [di, dj] [stride, stride]
                [padding, padding] with
              | none => 0
              | some inIdx => getAtOrZero input (in_ch.val :: inIdx)) *
              getAtOrZero grad [out_ch.val, i.val, j.val]) acc) 0 =
      (List.finRange outH).foldl (fun acc i =>
        (List.finRange outW).foldl (fun acc j =>
          acc +
            getAtOrZero
                (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding)
                  input)
                [in_ch.val, i.val * stride + di, j.val * stride + dj] *
              getAtOrZero grad [out_ch.val, i.val, j.val]) acc) 0 := by
  refine foldl_congr (l := List.finRange outH)
    (f := fun acc i =>
      (List.finRange outW).foldl (fun acc j =>
        acc +
          (match Spec.Private.mkInputIdx? [i.val, j.val] [di, dj] [stride, stride]
              [padding, padding] with
            | none => 0
            | some inIdx => getAtOrZero input (in_ch.val :: inIdx)) *
            getAtOrZero grad [out_ch.val, i.val, j.val]) acc)
    (g := fun acc i =>
      (List.finRange outW).foldl (fun acc j =>
        acc +
          getAtOrZero
              (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding)
                input)
              [in_ch.val, i.val * stride + di, j.val * stride + dj] *
            getAtOrZero grad [out_ch.val, i.val, j.val]) acc)
    (init := (0 : α)) ?_
  intro acc i
  refine foldl_congr (l := List.finRange outW)
    (f := fun acc j =>
      acc +
        (match Spec.Private.mkInputIdx? [i.val, j.val] [di, dj] [stride, stride]
            [padding, padding] with
          | none => 0
          | some inIdx => getAtOrZero input (in_ch.val :: inIdx)) *
          getAtOrZero grad [out_ch.val, i.val, j.val])
    (g := fun acc j =>
      acc +
        getAtOrZero
            (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding)
              input)
            [in_ch.val, i.val * stride + di, j.val * stride + dj] *
          getAtOrZero grad [out_ch.val, i.val, j.val])
    (init := acc) ?_
  intro acc j
  have hRead :=
    mkInputIdx_match_eq_paddedInput (stride := stride) (padding := padding)
      (img := input) (c := in_ch) (oi := i.val) (di := di) (oj := j.val) (dj := dj)
  simpa using congrArg (fun x => acc + x * getAtOrZero grad [out_ch.val, i.val, j.val]) hRead

lemma entry_eq_scalar_get_at_or_zero1
    {α : Type} [Zero α] {n : Nat}
    (t : Tensor α (.dim n .scalar)) (i : Fin n) :
    (match t with
    | .dim f => f i) = Tensor.scalar (getAtOrZero t [i.val]) := by
  cases t with
  | dim f =>
      have hi : i.val < n := i.isLt
      cases h1 : f i with
      | scalar v =>
          simp [hi, h1]

lemma entry_eq_scalar_get_at_or_zero4
    {α : Type} [Zero α] {n1 n2 n3 n4 : Nat}
    (t : Tensor α (.dim n1 (.dim n2 (.dim n3 (.dim n4 .scalar)))))
    (i1 : Fin n1) (i2 : Fin n2) (i3 : Fin n3) (i4 : Fin n4) :
    (match
      match
        match
          match t with
          | .dim f => f i1 with
        | .dim g => g i2 with
      | .dim h => h i3 with
    | .dim k => k i4) =
      Tensor.scalar (getAtOrZero t [i1.val, i2.val, i3.val, i4.val]) := by
  cases t with
  | dim f =>
      have hi1 : i1.val < n1 := i1.isLt
      cases h1 : f i1 with
      | dim g =>
          have hi2 : i2.val < n2 := i2.isLt
          cases h2 : g i2 with
          | dim h =>
              have hi3 : i3.val < n3 := i3.isLt
              cases h3 : h i3 with
              | dim k =>
                  have hi4 : i4.val < n4 := i4.isLt
                  cases h4 : k i4 with
                  | scalar v =>
                      simp [hi1, hi2, hi3, hi4, h1, h2, h3, h4]

-- ---------------------------------------------------------------------------
-- Conv2D bias gradient: pointwise bound
-- ---------------------------------------------------------------------------


end NFBackend

end
end RuntimeApprox
end Proofs

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Core.FDeriv
public import NN.Proofs.Autograd.Tape.Ops.Conv.Index
public import Mathlib.Analysis.Calculus.FDeriv.Add
public import Mathlib.Analysis.Calculus.FDeriv.Bilinear

/-!
# Derivative of General Convolution

This file proves the derivative and reverse-mode formulas for channels-first convolution at an
arbitrary spatial rank.  All sums use bounded multi-indices; no axis count is fixed in a theorem.
-/

@[expose] public section

namespace Proofs.Autograd.Conv

open scoped BigOperators
open Spec
open Spec.Tensor
open Spec.Conv.Internal
open Proofs.TensorAlgebra

noncomputable section

/-- Read a channel and bounded spatial coordinate from a channels-first tensor. -/
def channelGet {α : Type} {channels : Nat} {dims : List Nat}
    (x : Tensor α (Shape.ofList (channels :: dims)))
    (channel : Fin channels) (i : MultiIndex dims) : α :=
  MultiIndex.get x (channel, i)

/-- Read two leading channels followed by a bounded spatial coordinate. -/
def channelPairGet {α : Type} {outer inner : Nat} {dims : List Nat}
    (x : Tensor α (Shape.ofList (outer :: inner :: dims)))
    (i : Fin outer) (j : Fin inner) (k : MultiIndex dims) : α :=
  MultiIndex.get x (i, (j, k))

/-- Split a channels-first tensor dot product into channel and spatial sums. -/
theorem dot_eq_sum_channel {α : Type} [CommSemiring α]
    (channels : Nat) (dims : List Nat)
    (x y : Tensor α (Shape.ofList (channels :: dims))) :
    TensorAlgebra.dot x y =
      ∑ channel : Fin channels, ∑ i : MultiIndex dims,
        channelGet x channel i * channelGet y channel i := by
  rw [dot_eq_sum_get, MultiIndex.sum_cons]
  rfl

/-- Expand a channels-first lookup into the unique bounded spatial coordinate that it names. -/
theorem getAtOrZero_channel_eq_sum_indicator {α : Type} [AddCommMonoid α]
    {channels : Nat} {dims : List Nat}
    (x : Tensor α (Shape.ofList (channels :: dims)))
    (channel : Fin channels) (indices : List Nat) :
    getAtOrZero x (channel.val :: indices) =
      ∑ i : MultiIndex dims,
        if indices = i.toList then channelGet x channel i else 0 := by
  cases x with
  | dim values =>
      rw [getAtOrZero_dim_cons]
      simp only [channel.isLt, ↓reduceDIte]
      rw [getAtOrZero_eq_sum_indicator]
      rfl

/-- The scalar coefficient connecting one input coordinate to one output coordinate. -/
def convCoefficient
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Spec.Tensor Nat [d]}
    (weights : Tensor ℝ (Shape.ofList (outC :: inC :: kernel.toList)))
    (outCh : Fin outC)
    (outIdx : MultiIndex (convOutSpatial inSpatial kernel stride padding).toList)
    (inCh : Fin inC) (inIdx : MultiIndex inSpatial.toList) : ℝ :=
  ∑ kIdx : MultiIndex kernel.toList,
    if mkInputIdx? outIdx.toList kIdx.toList stride.toList padding.toList =
        some inIdx.toList then
      channelPairGet weights outCh inCh kIdx
    else
      0

/-- One coordinate of the generic convolution contraction, written as finite sums. -/
theorem channelGet_convCoreSpec
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Spec.Tensor Nat [d]}
    (weights : Tensor ℝ (Shape.ofList (outC :: inC :: kernel.toList)))
    (input : Tensor ℝ (Shape.ofList (inC :: inSpatial.toList)))
    (outCh : Fin outC)
    (outIdx : MultiIndex (convOutSpatial inSpatial kernel stride padding).toList) :
    channelGet (convCoreSpec weights input) outCh outIdx =
      ∑ inCh : Fin inC, ∑ kIdx : MultiIndex kernel.toList,
        (match mkInputIdx? outIdx.toList kIdx.toList stride.toList padding.toList with
          | none => 0
          | some inIdx => getAtOrZero input (inCh.val :: inIdx)) *
        channelPairGet weights outCh inCh kIdx := by
  simp only [channelGet, convCoreSpec]
  change MultiIndex.get
      (Spec.Tensor.generate (convOutSpatial inSpatial kernel stride padding).toList
        (fun outIdx =>
          (List.finRange inC).foldl (fun acc inCh =>
            foldlIndices kernel.toList acc (fun acc kIdx =>
              acc +
                (match mkInputIdx? outIdx kIdx stride.toList padding.toList with
                  | none => 0
                  | some inIdx => getAtOrZero input (inCh.val :: inIdx)) *
                getAtOrZero weights (outCh.val :: inCh.val :: kIdx))) 0)) outIdx = _
  rw [MultiIndex.get_generate]
  simp_rw [foldlIndices_add]
  rw [List.finRange_foldl_add_eq_finset_sum]
  apply Finset.sum_congr rfl
  intro inCh _
  apply Finset.sum_congr rfl
  intro kIdx _
  have hWeight := getAtOrZero_toList
    (dims := outC :: inC :: kernel.toList) weights (outCh, (inCh, kIdx))
  have hWeight' :
      getAtOrZero weights (outCh.val :: inCh.val :: kIdx.toList) =
        MultiIndex.get weights (outCh, (inCh, kIdx)) := by
    simpa only [MultiIndex.toList] using hWeight
  change
    (match mkInputIdx? outIdx.toList kIdx.toList stride.toList padding.toList with
      | none => 0
      | some inIdx => getAtOrZero input (inCh.val :: inIdx)) *
        getAtOrZero weights (outCh.val :: inCh.val :: kIdx.toList) =
      (match mkInputIdx? outIdx.toList kIdx.toList stride.toList padding.toList with
        | none => 0
        | some inIdx => getAtOrZero input (inCh.val :: inIdx)) *
        MultiIndex.get weights (outCh, (inCh, kIdx))
  rw [hWeight']

/-- Convolution is the matrix represented by `convCoefficient` at every spatial rank. -/
theorem channelGet_convCoreSpec_eq_coefficients
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Spec.Tensor Nat [d]}
    (weights : Tensor ℝ (Shape.ofList (outC :: inC :: kernel.toList)))
    (input : Tensor ℝ (Shape.ofList (inC :: inSpatial.toList)))
    (outCh : Fin outC)
    (outIdx : MultiIndex (convOutSpatial inSpatial kernel stride padding).toList) :
    channelGet (convCoreSpec weights input) outCh outIdx =
      ∑ inCh : Fin inC, ∑ inIdx : MultiIndex inSpatial.toList,
        channelGet input inCh inIdx *
          convCoefficient weights outCh outIdx inCh inIdx := by
  rw [channelGet_convCoreSpec]
  apply Finset.sum_congr rfl
  intro inCh _
  unfold convCoefficient
  calc
    (∑ kIdx : MultiIndex kernel.toList,
        (match mkInputIdx? outIdx.toList kIdx.toList stride.toList padding.toList with
          | none => 0
          | some inputIdx => getAtOrZero input (inCh.val :: inputIdx)) *
          channelPairGet weights outCh inCh kIdx) =
      ∑ kIdx : MultiIndex kernel.toList,
        ∑ inIdx : MultiIndex inSpatial.toList,
          channelGet input inCh inIdx *
            (if mkInputIdx? outIdx.toList kIdx.toList stride.toList padding.toList =
                some inIdx.toList then
              channelPairGet weights outCh inCh kIdx
            else 0) := by
        apply Finset.sum_congr rfl
        intro kIdx _
        cases hInput : mkInputIdx? outIdx.toList kIdx.toList stride.toList padding.toList with
        | none => simp
        | some inputIdx =>
            change getAtOrZero input (inCh.val :: inputIdx) *
                channelPairGet weights outCh inCh kIdx =
              ∑ inIdx : MultiIndex inSpatial.toList,
                channelGet input inCh inIdx *
                  (if some inputIdx = some inIdx.toList then
                    channelPairGet weights outCh inCh kIdx
                  else 0)
            rw [getAtOrZero_channel_eq_sum_indicator input inCh inputIdx]
            rw [Finset.sum_mul]
            apply Finset.sum_congr rfl
            intro inIdx _
            by_cases hEq : inputIdx = inIdx.toList <;> simp [hEq]
    _ =
      ∑ inIdx : MultiIndex inSpatial.toList,
        ∑ kIdx : MultiIndex kernel.toList,
          channelGet input inCh inIdx *
            (if mkInputIdx? outIdx.toList kIdx.toList stride.toList padding.toList =
                some inIdx.toList then
              channelPairGet weights outCh inCh kIdx
            else 0) := by
        rw [Finset.sum_comm]
    _ =
      ∑ inIdx : MultiIndex inSpatial.toList,
        channelGet input inCh inIdx *
          ∑ kIdx : MultiIndex kernel.toList,
            if mkInputIdx? outIdx.toList kIdx.toList stride.toList padding.toList =
                some inIdx.toList then
              channelPairGet weights outCh inCh kIdx
            else 0 := by
        apply Finset.sum_congr rfl
        intro inIdx _
        rw [Finset.mul_sum]

/-- One kernel-gradient coordinate is the contraction of input and output cotangent. -/
theorem channelPairGet_convKernelDerivSpec
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Spec.Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding ℝ)
    (input : Tensor ℝ (Shape.ofList (inC :: inSpatial.toList)))
    (gradOutput : Tensor ℝ
      (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).toList)))
    (outCh : Fin outC) (inCh : Fin inC) (kIdx : MultiIndex kernel.toList) :
    channelPairGet (convKernelDerivSpec layer input gradOutput) outCh inCh kIdx =
      ∑ outIdx : MultiIndex (convOutSpatial inSpatial kernel stride padding).toList,
        (match mkInputIdx? outIdx.toList kIdx.toList stride.toList padding.toList with
          | none => 0
          | some inIdx => getAtOrZero input (inCh.val :: inIdx)) *
        channelGet gradOutput outCh outIdx := by
  simp only [channelPairGet, convKernelDerivSpec]
  change MultiIndex.get
      (Spec.Tensor.generate kernel.toList (fun kIdx =>
        foldlIndices (convOutSpatial inSpatial kernel stride padding).toList 0
          (fun acc outIdx =>
            acc +
              (match mkInputIdx? outIdx kIdx stride.toList padding.toList with
                | none => 0
                | some inputIdx => getAtOrZero input (inCh.val :: inputIdx)) *
              getAtOrZero gradOutput (outCh.val :: outIdx)))) kIdx = _
  rw [MultiIndex.get_generate, foldlIndices_add]
  simp only [zero_add]
  apply Finset.sum_congr rfl
  intro outIdx _
  have hGrad := getAtOrZero_toList
    (dims := outC :: (convOutSpatial inSpatial kernel stride padding).toList)
    gradOutput (outCh, outIdx)
  have hGrad' :
      getAtOrZero gradOutput (outCh.val :: outIdx.toList) =
        MultiIndex.get gradOutput (outCh, outIdx) := by
    simpa only [MultiIndex.toList] using hGrad
  change
    (match mkInputIdx? outIdx.toList kIdx.toList stride.toList padding.toList with
      | none => 0
      | some inputIdx => getAtOrZero input (inCh.val :: inputIdx)) *
        getAtOrZero gradOutput (outCh.val :: outIdx.toList) =
      (match mkInputIdx? outIdx.toList kIdx.toList stride.toList padding.toList with
        | none => 0
        | some inputIdx => getAtOrZero input (inCh.val :: inputIdx)) *
        MultiIndex.get gradOutput (outCh, outIdx)
  rw [hGrad']

/-- One bias-gradient coordinate is the spatial sum of the output cotangent. -/
theorem channelGet_convBiasDerivSpec
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Spec.Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding ℝ)
    (input : Tensor ℝ (Shape.ofList (inC :: inSpatial.toList)))
    (gradOutput : Tensor ℝ
      (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).toList)))
    (outCh : Fin outC) :
    channelGet (dims := []) (convBiasDerivSpec layer input gradOutput) outCh PUnit.unit =
      ∑ outIdx : MultiIndex (convOutSpatial inSpatial kernel stride padding).toList,
        channelGet gradOutput outCh outIdx := by
  simp only [channelGet, convBiasDerivSpec]
  change
    foldlIndices (convOutSpatial inSpatial kernel stride padding).toList 0
        (fun acc outIdx => acc + getAtOrZero gradOutput (outCh.val :: outIdx)) = _
  rw [foldlIndices_add]
  simp only [zero_add]
  apply Finset.sum_congr rfl
  intro outIdx _
  have hGrad := getAtOrZero_toList
    (dims := outC :: (convOutSpatial inSpatial kernel stride padding).toList)
    gradOutput (outCh, outIdx)
  change getAtOrZero gradOutput (outCh.val :: outIdx.toList) =
    MultiIndex.get gradOutput (outCh, outIdx)
  simpa only [MultiIndex.toList] using hGrad

/-- A bias-gradient coordinate in the ordinary rank-one tensor view. -/
theorem getScalar_convBiasDerivSpec
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Spec.Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding ℝ)
    (input : Tensor ℝ (Shape.ofList (inC :: inSpatial.toList)))
    (gradOutput : Tensor ℝ
      (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).toList)))
    (outCh : Fin outC) :
    Spec.Tensor.getScalar (convBiasDerivSpec layer input gradOutput) outCh =
      ∑ outIdx : MultiIndex (convOutSpatial inSpatial kernel stride padding).toList,
        channelGet gradOutput outCh outIdx := by
  simp only [convBiasDerivSpec, Spec.Tensor.getScalar_dim]
  rw [foldlIndices_add]
  simp only [zero_add]
  apply Finset.sum_congr rfl
  intro outIdx _
  change getAtOrZero gradOutput (outCh.val :: outIdx.toList) =
    MultiIndex.get (dims := outC ::
      (convOutSpatial inSpatial kernel stride padding).toList)
      gradOutput (outCh, outIdx)
  have hGrad := getAtOrZero_toList
    (dims := outC :: (convOutSpatial inSpatial kernel stride padding).toList)
    gradOutput (outCh, outIdx)
  convert hGrad using 1
  rfl

/-- Broadcasting a bias reads the same channel value at every spatial coordinate. -/
theorem channelGet_convBiasBroadcastSpec
    {d outC : Nat}
    {kernel stride padding inSpatial : Spec.Tensor Nat [d]}
    (bias : Tensor ℝ [outC])
    (outCh : Fin outC)
    (outIdx : MultiIndex (convOutSpatial inSpatial kernel stride padding).toList) :
    channelGet (convBiasBroadcastSpec (kernel := kernel) (stride := stride)
      (padding := padding) (inSpatial := inSpatial) bias) outCh outIdx =
        Spec.Tensor.getScalar bias outCh := by
  simp only [channelGet, convBiasBroadcastSpec]
  change MultiIndex.get (dims := outC ::
      (convOutSpatial inSpatial kernel stride padding).toList)
    (Tensor.dim fun outCh =>
      Spec.Tensor.generate (convOutSpatial inSpatial kernel stride padding).toList
        fun _ => getAtOrZero bias [outCh.val]) (outCh, outIdx) = _
  rw [MultiIndex.get_dim, MultiIndex.get_generate]
  cases bias with
  | dim values =>
      cases h : values outCh with
      | scalar value =>
          simp [Spec.Tensor.getScalar, h, outCh.isLt]

/-- One input-gradient coordinate is the transpose-index convolution used by the runtime. -/
theorem channelGet_convInputDerivSpec
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Spec.Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding ℝ)
    (input : Tensor ℝ (Shape.ofList (inC :: inSpatial.toList)))
    (gradOutput : Tensor ℝ
      (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).toList)))
    (inCh : Fin inC) (inIdx : MultiIndex inSpatial.toList) :
    channelGet (convInputDerivSpec layer input gradOutput) inCh inIdx =
      ∑ outCh : Fin outC, ∑ kIdx : MultiIndex kernel.toList,
        (match mkTransposeInputIdx? inIdx.toList kIdx.toList stride.toList padding.toList with
          | none => 0
          | some outIdx =>
              getAtOrZero gradOutput (outCh.val :: outIdx) *
                channelPairGet layer.kernel outCh inCh kIdx) := by
  simp only [channelGet, convInputDerivSpec]
  change MultiIndex.get
      (Spec.Tensor.generate inSpatial.toList (fun inIdx =>
        (List.finRange outC).foldl (fun acc outCh =>
          foldlIndices kernel.toList acc (fun acc kIdx =>
            acc +
              (match mkTransposeInputIdx? inIdx kIdx stride.toList padding.toList with
                | none => 0
                | some outIdx =>
                    getAtOrZero gradOutput (outCh.val :: outIdx) *
                      getAtOrZero layer.kernel (outCh.val :: inCh.val :: kIdx)))) 0)) inIdx = _
  rw [MultiIndex.get_generate]
  simp_rw [foldlIndices_add]
  rw [List.finRange_foldl_add_eq_finset_sum]
  apply Finset.sum_congr rfl
  intro outCh _
  apply Finset.sum_congr rfl
  intro kIdx _
  have hWeight := getAtOrZero_toList
    (dims := outC :: inC :: kernel.toList) layer.kernel (outCh, (inCh, kIdx))
  have hWeight' :
      getAtOrZero layer.kernel (outCh.val :: inCh.val :: kIdx.toList) =
        MultiIndex.get layer.kernel (outCh, (inCh, kIdx)) := by
    simpa only [MultiIndex.toList] using hWeight
  split <;> simp_all [channelPairGet]

/-- The implemented input gradient is multiplication by the transposed coefficient matrix. -/
theorem channelGet_convInputDerivSpec_eq_coefficients
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Spec.Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding ℝ)
    (input : Tensor ℝ (Shape.ofList (inC :: inSpatial.toList)))
    (gradOutput : Tensor ℝ
      (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).toList)))
    (hStride : PositiveStrides stride.toList)
    (inCh : Fin inC) (inIdx : MultiIndex inSpatial.toList) :
    channelGet (convInputDerivSpec layer input gradOutput) inCh inIdx =
      ∑ outCh : Fin outC,
        ∑ outIdx : MultiIndex (convOutSpatial inSpatial kernel stride padding).toList,
          convCoefficient layer.kernel outCh outIdx inCh inIdx *
            channelGet gradOutput outCh outIdx := by
  rw [channelGet_convInputDerivSpec]
  apply Finset.sum_congr rfl
  intro outCh _
  unfold convCoefficient
  simp_rw [Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro kIdx _
  cases hTranspose : mkTransposeInputIdx? inIdx.toList kIdx.toList stride.toList
      padding.toList with
  | none =>
      simp only
      symm
      apply Finset.sum_eq_zero
      intro outIdx _
      have hForward :
          mkInputIdx? outIdx.toList kIdx.toList stride.toList padding.toList ≠
            some inIdx.toList := by
        intro h
        have := mkTransposeInputIdx?_of_mkInputIdx?_eq_some hStride h
        rw [hTranspose] at this
        contradiction
      simp [hForward]
  | some outputIdx =>
      change getAtOrZero gradOutput (outCh.val :: outputIdx) *
          channelPairGet layer.kernel outCh inCh kIdx =
        ∑ outIdx : MultiIndex
            (convOutSpatial inSpatial kernel stride padding).toList,
          (if mkInputIdx? outIdx.toList kIdx.toList stride.toList padding.toList =
              some inIdx.toList then
            channelPairGet layer.kernel outCh inCh kIdx
          else 0) * channelGet gradOutput outCh outIdx
      rw [getAtOrZero_channel_eq_sum_indicator gradOutput outCh outputIdx]
      rw [Finset.sum_mul]
      apply Finset.sum_congr rfl
      intro outIdx _
      have hRelation := mkInputIdx?_eq_some_iff
        (outIdx := outIdx.toList) (kIdx := kIdx.toList)
        (stride := stride.toList) (padding := padding.toList)
        (inIdx := inIdx.toList) hStride
      have hEq :
          mkInputIdx? outIdx.toList kIdx.toList stride.toList padding.toList =
              some inIdx.toList ↔ outputIdx = outIdx.toList := by
        rw [hRelation, hTranspose]
        simp
      by_cases hOutput : outputIdx = outIdx.toList
      · simp [hOutput, hEq.mpr hOutput]
        ring
      · have hForward := fun h => hOutput (hEq.mp h)
        rw [if_neg hForward]
        simp [hOutput]

/-! ## Adjoint identities -/

/-- The forward input map and implemented input gradient are adjoint at every spatial rank. -/
theorem convCoreSpec_convInputDerivSpec_adjoint
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Spec.Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding ℝ)
    (input : Tensor ℝ (Shape.ofList (inC :: inSpatial.toList)))
    (deltaInput : Tensor ℝ (Shape.ofList (inC :: inSpatial.toList)))
    (gradOutput : Tensor ℝ
      (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).toList)))
    (hStride : PositiveStrides stride.toList) :
    TensorAlgebra.dot (convCoreSpec layer.kernel deltaInput) gradOutput =
      TensorAlgebra.dot deltaInput (convInputDerivSpec layer input gradOutput) := by
  classical
  rw [dot_eq_sum_channel, dot_eq_sum_channel]
  simp_rw [channelGet_convCoreSpec_eq_coefficients]
  simp_rw [channelGet_convInputDerivSpec_eq_coefficients layer input gradOutput hStride]
  simp_rw [Finset.sum_mul, Finset.mul_sum]
  calc
    (∑ outCh,
        ∑ outIdx,
          ∑ inCh,
            ∑ inIdx,
              channelGet deltaInput inCh inIdx *
                convCoefficient layer.kernel outCh outIdx inCh inIdx *
                channelGet gradOutput outCh outIdx) =
      ∑ outCh,
        ∑ inCh,
          ∑ outIdx,
            ∑ inIdx,
              channelGet deltaInput inCh inIdx *
                convCoefficient layer.kernel outCh outIdx inCh inIdx *
                channelGet gradOutput outCh outIdx := by
        apply Finset.sum_congr rfl
        intro outCh _
        rw [Finset.sum_comm]
    _ =
      ∑ inCh,
        ∑ outCh,
          ∑ outIdx,
            ∑ inIdx,
              channelGet deltaInput inCh inIdx *
                convCoefficient layer.kernel outCh outIdx inCh inIdx *
                channelGet gradOutput outCh outIdx := by
        rw [Finset.sum_comm]
    _ =
      ∑ inCh,
        ∑ inIdx,
          ∑ outCh,
            ∑ outIdx,
              channelGet deltaInput inCh inIdx *
                (convCoefficient layer.kernel outCh outIdx inCh inIdx *
                  channelGet gradOutput outCh outIdx) := by
        apply Finset.sum_congr rfl
        intro inCh _
        calc
          (∑ outCh, ∑ outIdx, ∑ inIdx,
              channelGet deltaInput inCh inIdx *
                convCoefficient layer.kernel outCh outIdx inCh inIdx *
                channelGet gradOutput outCh outIdx) =
            ∑ outCh, ∑ inIdx, ∑ outIdx,
              channelGet deltaInput inCh inIdx *
                convCoefficient layer.kernel outCh outIdx inCh inIdx *
                channelGet gradOutput outCh outIdx := by
              apply Finset.sum_congr rfl
              intro outCh _
              rw [Finset.sum_comm]
          _ =
            ∑ inIdx, ∑ outCh, ∑ outIdx,
              channelGet deltaInput inCh inIdx *
                convCoefficient layer.kernel outCh outIdx inCh inIdx *
                channelGet gradOutput outCh outIdx := by
              rw [Finset.sum_comm]
          _ = _ := by
              apply Finset.sum_congr rfl
              intro inIdx _
              apply Finset.sum_congr rfl
              intro outCh _
              apply Finset.sum_congr rfl
              intro outIdx _
              ring

/-- The kernel contraction and implemented kernel gradient are adjoint. -/
theorem convCoreSpec_convKernelDerivSpec_adjoint
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Spec.Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding ℝ)
    (input : Tensor ℝ (Shape.ofList (inC :: inSpatial.toList)))
    (deltaKernel : Tensor ℝ (Shape.ofList (outC :: inC :: kernel.toList)))
    (gradOutput : Tensor ℝ
      (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).toList))) :
    TensorAlgebra.dot (convCoreSpec deltaKernel input) gradOutput =
      TensorAlgebra.dot deltaKernel (convKernelDerivSpec layer input gradOutput) := by
  classical
  rw [dot_eq_sum_get, dot_eq_sum_get]
  rw [MultiIndex.sum_cons outC
    (convOutSpatial inSpatial kernel stride padding).toList]
  rw [MultiIndex.sum_cons outC (inC :: kernel.toList)]
  simp_rw [MultiIndex.sum_cons inC kernel.toList]
  change
    (∑ outCh : Fin outC,
      ∑ outIdx : MultiIndex (convOutSpatial inSpatial kernel stride padding).toList,
        channelGet (convCoreSpec deltaKernel input) outCh outIdx *
          channelGet gradOutput outCh outIdx) =
    ∑ outCh : Fin outC, ∑ inCh : Fin inC, ∑ kIdx : MultiIndex kernel.toList,
      channelPairGet deltaKernel outCh inCh kIdx *
        channelPairGet (convKernelDerivSpec layer input gradOutput) outCh inCh kIdx
  simp_rw [channelGet_convCoreSpec, channelPairGet_convKernelDerivSpec]
  apply Finset.sum_congr rfl
  intro outCh _
  simp_rw [Finset.sum_mul]
  calc
    (∑ outIdx,
        ∑ inCh,
          ∑ kIdx,
            (match mkInputIdx? outIdx.toList kIdx.toList stride.toList padding.toList with
              | none => 0
              | some inputIdx => getAtOrZero input (inCh.val :: inputIdx)) *
              channelPairGet deltaKernel outCh inCh kIdx *
              channelGet gradOutput outCh outIdx) =
      ∑ inCh,
        ∑ outIdx,
          ∑ kIdx,
            (match mkInputIdx? outIdx.toList kIdx.toList stride.toList padding.toList with
              | none => 0
              | some inputIdx => getAtOrZero input (inCh.val :: inputIdx)) *
              channelPairGet deltaKernel outCh inCh kIdx *
              channelGet gradOutput outCh outIdx := by
        rw [Finset.sum_comm]
    _ =
      ∑ inCh,
        ∑ kIdx,
          ∑ outIdx,
            (match mkInputIdx? outIdx.toList kIdx.toList stride.toList padding.toList with
              | none => 0
              | some inputIdx => getAtOrZero input (inCh.val :: inputIdx)) *
              channelPairGet deltaKernel outCh inCh kIdx *
              channelGet gradOutput outCh outIdx := by
        apply Finset.sum_congr rfl
        intro inCh _
        rw [Finset.sum_comm]
    _ =
      ∑ inCh,
        ∑ kIdx,
          channelPairGet deltaKernel outCh inCh kIdx *
            ∑ outIdx,
              (match mkInputIdx? outIdx.toList kIdx.toList stride.toList padding.toList with
                | none => 0
                | some inputIdx => getAtOrZero input (inCh.val :: inputIdx)) *
                channelGet gradOutput outCh outIdx := by
        apply Finset.sum_congr rfl
        intro inCh _
        apply Finset.sum_congr rfl
        intro kIdx _
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro outIdx _
        ring

/-- Bias broadcasting and spatial reduction are adjoint. -/
theorem convBiasBroadcastSpec_convBiasDerivSpec_adjoint
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Spec.Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding ℝ)
    (input : Tensor ℝ (Shape.ofList (inC :: inSpatial.toList)))
    (deltaBias : Tensor ℝ [outC])
    (gradOutput : Tensor ℝ
      (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).toList))) :
    TensorAlgebra.dot (convBiasBroadcastSpec (kernel := kernel) (stride := stride) (padding := padding)
        (inSpatial := inSpatial) deltaBias) gradOutput =
      TensorAlgebra.dot deltaBias (convBiasDerivSpec layer input gradOutput) := by
  classical
  rw [dot_eq_sum_channel]
  rw [TensorAlgebra.dot_vec_eq_sum]
  apply Finset.sum_congr rfl
  intro outCh _
  rw [getScalar_convBiasDerivSpec, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro outIdx _
  rw [channelGet_convBiasBroadcastSpec]

/-! ## Fréchet derivative -/

/-- Euclidean coordinates indexed by a tensor shape list. -/
abbrev CoordVec (dims : List Nat) := EuclideanSpace ℝ (MultiIndex dims)

/-- Build shape-indexed Euclidean coordinates from a coordinate function. -/
def coordVecOfFun {dims : List Nat} (f : MultiIndex dims → ℝ) : CoordVec dims :=
  (EuclideanSpace.equiv (𝕜 := ℝ) (ι := MultiIndex dims)).symm f

@[simp]
theorem coordVecOfFun_apply {dims : List Nat} (f : MultiIndex dims → ℝ)
    (i : MultiIndex dims) : coordVecOfFun f i = f i := by
  simp [coordVecOfFun, EuclideanSpace.equiv]

/-- Vectorize a tensor using bounded multi-indices rather than flattened natural indices. -/
def tensorToCoordVec {dims : List Nat} (x : Tensor ℝ (Shape.ofList dims)) : CoordVec dims :=
  coordVecOfFun fun i ↦ i.get x

@[simp]
theorem tensorToCoordVec_apply {dims : List Nat} (x : Tensor ℝ (Shape.ofList dims))
    (i : MultiIndex dims) : tensorToCoordVec x i = i.get x := by
  simp [tensorToCoordVec]

/-- The kernel/input contraction in Euclidean coordinates. -/
def convCoreVec
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Spec.Tensor Nat [d]}
    (weights : CoordVec (outC :: inC :: kernel.toList))
    (input : CoordVec (inC :: inSpatial.toList)) :
    CoordVec (outC :: (convOutSpatial inSpatial kernel stride padding).toList) :=
  coordVecOfFun fun outCoord ↦
    ∑ inCh : Fin inC, ∑ inIdx : MultiIndex inSpatial.toList,
      input (inCh, inIdx) *
        ∑ kIdx : MultiIndex kernel.toList,
          if mkInputIdx? outCoord.2.toList kIdx.toList stride.toList padding.toList =
              some inIdx.toList then
            weights (outCoord.1, (inCh, kIdx))
          else
            0

@[simp]
theorem convCoreVec_apply
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Spec.Tensor Nat [d]}
    (weights : CoordVec (outC :: inC :: kernel.toList))
    (input : CoordVec (inC :: inSpatial.toList))
    (outCoord : MultiIndex
      (outC :: (convOutSpatial inSpatial kernel stride padding).toList)) :
    convCoreVec (kernel := kernel) (stride := stride) (padding := padding) weights input outCoord =
      ∑ inCh : Fin inC, ∑ inIdx : MultiIndex inSpatial.toList,
        input (inCh, inIdx) *
          ∑ kIdx : MultiIndex kernel.toList,
            if mkInputIdx? outCoord.2.toList kIdx.toList stride.toList padding.toList =
                some inIdx.toList then
              weights (outCoord.1, (inCh, kIdx))
            else
              0 := by
  simp [convCoreVec]

/-- Tensor convolution and its Euclidean-coordinate contraction agree at every spatial rank. -/
theorem tensorToCoordVec_convCoreSpec
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Spec.Tensor Nat [d]}
    (weights : Tensor ℝ (Shape.ofList (outC :: inC :: kernel.toList)))
    (input : Tensor ℝ (Shape.ofList (inC :: inSpatial.toList))) :
    tensorToCoordVec (convCoreSpec weights input) =
      convCoreVec (kernel := kernel) (stride := stride) (padding := padding)
        (tensorToCoordVec weights) (tensorToCoordVec input) := by
  ext outCoord
  rcases outCoord with ⟨outCh, outIdx⟩
  rw [tensorToCoordVec_apply]
  change channelGet (convCoreSpec weights input) outCh outIdx = _
  rw [channelGet_convCoreSpec_eq_coefficients]
  simp only [convCoreVec_apply, tensorToCoordVec_apply, channelGet, channelPairGet,
    convCoefficient]

/-- Continuous bilinear form of rank-general convolution. -/
def convCoreBilin
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Spec.Tensor Nat [d]} :
    CoordVec (outC :: inC :: kernel.toList) →L[ℝ]
      CoordVec (inC :: inSpatial.toList) →L[ℝ]
        CoordVec (outC :: (convOutSpatial inSpatial kernel stride padding).toList) := by
  classical
  let inner : CoordVec (outC :: inC :: kernel.toList) →
      CoordVec (inC :: inSpatial.toList) →ₗ[ℝ]
        CoordVec (outC :: (convOutSpatial inSpatial kernel stride padding).toList) :=
    fun weights ↦
      { toFun := fun input ↦
          convCoreVec (kernel := kernel) (stride := stride) (padding := padding) weights input
        map_add' := by
          intro x y
          ext outCoord
          simp [convCoreVec, Finset.sum_add_distrib, add_mul]
        map_smul' := by
          intro c x
          ext outCoord
          simp [convCoreVec, smul_eq_mul, Finset.mul_sum, mul_assoc] }
  let innerContinuous (weights : CoordVec (outC :: inC :: kernel.toList)) :
      CoordVec (inC :: inSpatial.toList) →L[ℝ]
        CoordVec (outC :: (convOutSpatial inSpatial kernel stride padding).toList) :=
    ⟨inner weights, LinearMap.continuous_of_finiteDimensional (f := inner weights)⟩
  let outer : CoordVec (outC :: inC :: kernel.toList) →ₗ[ℝ]
      CoordVec (inC :: inSpatial.toList) →L[ℝ]
        CoordVec (outC :: (convOutSpatial inSpatial kernel stride padding).toList) :=
    { toFun := innerContinuous
      map_add' := by
        intro w z
        ext x outCoord
        change convCoreVec (kernel := kernel) (stride := stride) (padding := padding)
            (w + z) x outCoord =
          convCoreVec (kernel := kernel) (stride := stride) (padding := padding) w x outCoord +
            convCoreVec (kernel := kernel) (stride := stride) (padding := padding) z x outCoord
        simp only [convCoreVec_apply, PiLp.add_apply]
        rw [← Finset.sum_add_distrib]
        apply Finset.sum_congr rfl
        intro inCh _
        rw [← Finset.sum_add_distrib]
        apply Finset.sum_congr rfl
        intro inIdx _
        rw [← mul_add, ← Finset.sum_add_distrib]
        apply congrArg (x (inCh, inIdx) * ·)
        apply Finset.sum_congr rfl
        intro kIdx _
        by_cases h : mkInputIdx? outCoord.2.toList kIdx.toList stride.toList
            padding.toList = some inIdx.toList <;> simp [h]
      map_smul' := by
        intro c w
        ext x outCoord
        simp [innerContinuous, inner, convCoreVec, smul_eq_mul, Finset.mul_sum,
          mul_left_comm] }
  exact ⟨outer, LinearMap.continuous_of_finiteDimensional (f := outer)⟩

@[simp]
theorem convCoreBilin_apply
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Spec.Tensor Nat [d]}
    (weights : CoordVec (outC :: inC :: kernel.toList))
    (input : CoordVec (inC :: inSpatial.toList)) :
    convCoreBilin (kernel := kernel) (stride := stride) (padding := padding) weights input =
      convCoreVec (kernel := kernel) (stride := stride) (padding := padding) weights input := by
  rfl

/-- Continuous linear bias broadcast in Euclidean coordinates. -/
def convBiasBroadcastCLM
    {d outC : Nat}
    {kernel stride padding inSpatial : Spec.Tensor Nat [d]} :
    CoordVec [outC] →L[ℝ]
      CoordVec (outC :: (convOutSpatial inSpatial kernel stride padding).toList) := by
  let linear : CoordVec [outC] →ₗ[ℝ]
      CoordVec (outC :: (convOutSpatial inSpatial kernel stride padding).toList) :=
    { toFun := fun bias ↦ coordVecOfFun fun outCoord ↦ bias (outCoord.1, PUnit.unit)
      map_add' := by intro x y; ext i; simp
      map_smul' := by intro c x; ext i; simp [smul_eq_mul] }
  exact ⟨linear, LinearMap.continuous_of_finiteDimensional (f := linear)⟩

@[simp]
theorem convBiasBroadcastCLM_apply
    {d outC : Nat}
    {kernel stride padding inSpatial : Spec.Tensor Nat [d]}
    (bias : CoordVec [outC])
    (outCoord : MultiIndex
      (outC :: (convOutSpatial inSpatial kernel stride padding).toList)) :
    convBiasBroadcastCLM (kernel := kernel) (stride := stride) (padding := padding) bias outCoord =
      bias (outCoord.1, PUnit.unit) := by
  simp [convBiasBroadcastCLM]

/-- Coordinate conversion preserves pointwise tensor addition. -/
theorem tensorToCoordVec_addSpec
    {dims : List Nat} (x y : Tensor ℝ (Shape.ofList dims)) :
    tensorToCoordVec (addSpec x y) = tensorToCoordVec x + tensorToCoordVec y := by
  ext i
  rw [tensorToCoordVec_apply, MultiIndex.get_addSpec]
  rfl

/-- Tensor bias broadcasting agrees with the corresponding coordinate map. -/
theorem tensorToCoordVec_convBiasBroadcastSpec
    {d outC : Nat}
    {kernel stride padding inSpatial : Spec.Tensor Nat [d]}
    (bias : Tensor ℝ [outC]) :
    tensorToCoordVec (convBiasBroadcastSpec
        (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial) bias) =
      convBiasBroadcastCLM (kernel := kernel) (stride := stride) (padding := padding)
        (tensorToCoordVec bias) := by
  ext outCoord
  rcases outCoord with ⟨outCh, outIdx⟩
  rw [tensorToCoordVec_apply]
  change channelGet (convBiasBroadcastSpec bias) outCh outIdx = _
  rw [channelGet_convBiasBroadcastSpec]
  rw [convBiasBroadcastCLM_apply]
  rw [tensorToCoordVec_apply]
  exact (MultiIndex.get_vector_eq_getScalar bias outCh).symm

/-- Coordinate form of a complete convolution layer, including bias. -/
def convForwardVec
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Spec.Tensor Nat [d]}
    (state : (CoordVec (outC :: inC :: kernel.toList) × CoordVec [outC]) ×
      CoordVec (inC :: inSpatial.toList)) :
    CoordVec (outC :: (convOutSpatial inSpatial kernel stride padding).toList) :=
  convCoreBilin (kernel := kernel) (stride := stride) (padding := padding)
      state.1.1 state.2 +
    convBiasBroadcastCLM (kernel := kernel) (stride := stride) (padding := padding) state.1.2

/-- Tensor-level `convSpec` agrees with `convForwardVec`. -/
theorem tensorToCoordVec_convSpec
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Spec.Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding ℝ)
    (input : Tensor ℝ (Shape.ofList (inC :: inSpatial.toList))) :
    tensorToCoordVec (convSpec layer input) =
      convForwardVec (kernel := kernel) (stride := stride) (padding := padding)
        ((tensorToCoordVec layer.kernel, tensorToCoordVec layer.bias), tensorToCoordVec input) := by
  rw [show convSpec layer input =
    addSpec (convCoreSpec layer.kernel input) (convBiasBroadcastSpec layer.bias) by rfl]
  rw [tensorToCoordVec_addSpec, tensorToCoordVec_convCoreSpec,
    tensorToCoordVec_convBiasBroadcastSpec]
  rfl

/-- Euclidean state space of one rank-general convolution application. -/
abbrev ConvState
    {d : Nat} (inC outC : Nat) (kernel inSpatial : Spec.Tensor Nat [d]) :=
  (CoordVec (outC :: inC :: kernel.toList) × CoordVec [outC]) ×
    CoordVec (inC :: inSpatial.toList)

/-- Projection of the kernel coordinates from a convolution state. -/
def convWeightProjection
    {d inC outC : Nat} {kernel inSpatial : Spec.Tensor Nat [d]} :
    ConvState inC outC kernel inSpatial →L[ℝ]
        CoordVec (outC :: inC :: kernel.toList) :=
  (ContinuousLinearMap.fst ℝ _ _).comp (ContinuousLinearMap.fst ℝ _ _)

/-- Projection of the bias coordinates from a convolution state. -/
def convBiasProjection
    {d inC outC : Nat} {kernel inSpatial : Spec.Tensor Nat [d]} :
    ConvState inC outC kernel inSpatial →L[ℝ] CoordVec [outC] :=
  (ContinuousLinearMap.snd ℝ _ _).comp (ContinuousLinearMap.fst ℝ _ _)

/-- Projection of the input coordinates from a convolution state. -/
def convInputProjection
    {d inC outC : Nat} {kernel inSpatial : Spec.Tensor Nat [d]} :
    ConvState inC outC kernel inSpatial →L[ℝ]
        CoordVec (inC :: inSpatial.toList) :=
  ContinuousLinearMap.snd ℝ _ _

/-- Product-rule derivative of a complete convolution state. -/
def convDerivative
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Spec.Tensor Nat [d]}
    (state : ConvState inC outC kernel inSpatial) :
    ConvState inC outC kernel inSpatial →L[ℝ]
        CoordVec (outC :: (convOutSpatial inSpatial kernel stride padding).toList) :=
  (convCoreBilin (kernel := kernel) (stride := stride) (padding := padding)).precompR
      (ConvState inC outC kernel inSpatial) state.1.1
      (convInputProjection (inC := inC) (outC := outC) (kernel := kernel)
        (inSpatial := inSpatial))
    + (convCoreBilin (kernel := kernel) (stride := stride) (padding := padding)).precompL
      (ConvState inC outC kernel inSpatial)
      (convWeightProjection (inC := inC) (outC := outC) (kernel := kernel)
        (inSpatial := inSpatial)) state.2
    + (convBiasBroadcastCLM (kernel := kernel) (stride := stride) (padding := padding)).comp
      (convBiasProjection (inC := inC) (outC := outC) (kernel := kernel)
        (inSpatial := inSpatial))

/-- The exact derivative of a rank-general convolution is its kernel/input product rule plus bias. -/
theorem hasFDerivAt_convForwardVec
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Spec.Tensor Nat [d]}
    (state : ConvState inC outC kernel inSpatial) :
    HasFDerivAt
      (convForwardVec (kernel := kernel) (stride := stride) (padding := padding))
      (convDerivative (kernel := kernel) (stride := stride) (padding := padding) state)
      state := by
  let weightProjection :=
    convWeightProjection (inC := inC) (outC := outC) (kernel := kernel)
      (inSpatial := inSpatial)
  let inputProjection :=
    convInputProjection (inC := inC) (outC := outC) (kernel := kernel)
      (inSpatial := inSpatial)
  let biasProjection :=
    convBiasProjection (inC := inC) (outC := outC) (kernel := kernel)
      (inSpatial := inSpatial)
  let core := convCoreBilin (inC := inC) (outC := outC)
    (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
  have hCore : HasFDerivAt
      (fun s : ConvState inC outC kernel inSpatial ↦ core (weightProjection s) (inputProjection s))
      (core.precompR (ConvState inC outC kernel inSpatial) state.1.1 inputProjection +
        core.precompL (ConvState inC outC kernel inSpatial) weightProjection state.2)
      state :=
    core.hasFDerivAt_of_bilinear weightProjection.hasFDerivAt inputProjection.hasFDerivAt
  let biasMap : ConvState inC outC kernel inSpatial →L[ℝ]
      CoordVec (outC :: (convOutSpatial inSpatial kernel stride padding).toList) :=
    (convBiasBroadcastCLM (kernel := kernel) (stride := stride) (padding := padding)
      (inSpatial := inSpatial)).comp biasProjection
  have hBias : HasFDerivAt (fun s : ConvState inC outC kernel inSpatial ↦ biasMap s)
      biasMap state := biasMap.hasFDerivAt
  have hSum := hCore.add hBias
  change HasFDerivAt
    ((fun s : ConvState inC outC kernel inSpatial ↦
        core (weightProjection s) (inputProjection s)) + fun s ↦ biasMap s)
    (core.precompR (ConvState inC outC kernel inSpatial) state.1.1 inputProjection +
      core.precompL (ConvState inC outC kernel inSpatial) weightProjection state.2 + biasMap)
    state
  exact hSum

/-- Applying the analytic derivative gives the tensor-level convolution JVP. -/
theorem tensorToCoordVec_convJvpSpec
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Spec.Tensor Nat [d]}
    (layer tangentLayer : ConvSpec d inC outC kernel stride padding ℝ)
    (input tangentInput : Tensor ℝ (Shape.ofList (inC :: inSpatial.toList))) :
    tensorToCoordVec (convJvpSpec layer tangentLayer input tangentInput) =
      convDerivative (kernel := kernel) (stride := stride) (padding := padding)
        ((tensorToCoordVec layer.kernel, tensorToCoordVec layer.bias), tensorToCoordVec input)
        ((tensorToCoordVec tangentLayer.kernel, tensorToCoordVec tangentLayer.bias),
          tensorToCoordVec tangentInput) := by
  rw [show convJvpSpec layer tangentLayer input tangentInput =
    addSpec
      (addSpec (convCoreSpec tangentLayer.kernel input)
        (convCoreSpec layer.kernel tangentInput))
      (convBiasBroadcastSpec tangentLayer.bias) by rfl]
  rw [tensorToCoordVec_addSpec, tensorToCoordVec_addSpec]
  rw [tensorToCoordVec_convCoreSpec, tensorToCoordVec_convCoreSpec,
    tensorToCoordVec_convBiasBroadcastSpec]
  simp [convDerivative, convWeightProjection, convInputProjection, convBiasProjection]
  rw [add_comm]
  rw [convCoreBilin_apply, convCoreBilin_apply]

/-- The implemented convolution backward pass is the adjoint of the exact JVP. -/
theorem convJvpSpec_convBackwardSpec_adjoint
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Spec.Tensor Nat [d]}
    (layer tangentLayer : ConvSpec d inC outC kernel stride padding ℝ)
    (input tangentInput : Tensor ℝ (Shape.ofList (inC :: inSpatial.toList)))
    (gradOutput : Tensor ℝ
      (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).toList)))
    (hStride : PositiveStrides stride.toList) :
    TensorAlgebra.dot (convJvpSpec layer tangentLayer input tangentInput) gradOutput =
      TensorAlgebra.dot tangentLayer.kernel (convBackwardSpec layer input gradOutput).1 +
      TensorAlgebra.dot tangentLayer.bias (convBackwardSpec layer input gradOutput).2.1 +
      TensorAlgebra.dot tangentInput (convBackwardSpec layer input gradOutput).2.2 := by
  rw [show convJvpSpec layer tangentLayer input tangentInput =
    addSpec
      (addSpec (convCoreSpec tangentLayer.kernel input)
        (convCoreSpec layer.kernel tangentInput))
      (convBiasBroadcastSpec tangentLayer.bias) by rfl]
  rw [TensorAlgebra.dot_add_left, TensorAlgebra.dot_add_left]
  rw [convCoreSpec_convKernelDerivSpec_adjoint layer input tangentLayer.kernel gradOutput]
  rw [convCoreSpec_convInputDerivSpec_adjoint layer input tangentInput gradOutput hStride]
  rw [convBiasBroadcastSpec_convBiasDerivSpec_adjoint layer input tangentLayer.bias gradOutput]
  simp only [convBackwardSpec]
  ring

end
end Proofs.Autograd.Conv

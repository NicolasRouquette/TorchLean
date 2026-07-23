/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Profile
public import NN.Floats.Interval.IEEEExec32Soundness
public import NN.IR.Graph
public import NN.IR.Semantics
public import NN.Proofs.Analysis.Softmax
public import NN.Spec.Core.FloatInstances
public import NN.Spec.Core.TensorOps

/-!
# Numerical certificates for TorchLean graphs

This module joins three existing parts of TorchLean without introducing another graph or another
floating-point representation:

* `NN.IR.Graph` remains the program being analysed;
* `IEEE32Exec.Interval32` supplies executable, outward-rounded binary32 intervals;
* `NN.Backend.ExecutionAudit` records the kernel capsules selected by backend planning.

A raw certificate is proof-free data that an application may construct or decode using its own
artifact format; this module does not prescribe a JSON schema. `check` does not trust its node
ranges. It reconstructs the canonical range trace from the graph and source
assumptions, checks every interval for finite ordered endpoints, replans the graph under the named
backend profile, and compares the result with the raw artifact. Successful checking returns a
`CheckedCertificate`, whose node ranges carry finite-endpoint and ordering proofs. This executable
check does not by itself prove enclosure of the exact-real graph denotation; that evidence is the
separate `CheckedRealExecution` value used by `CheckedExecution.errorTrace`.

The range trace deliberately starts with operations whose enclosure is already provided by the
sound `Interval32` core. Unsupported operations fail with the node id and operation name. They are
not assigned `[-inf,+inf]`, because that would turn a missing numerical theorem into an apparently
successful certificate.

The numerical conventions follow IEEE Std 754-2019. Outward-rounded interval propagation follows
IEEE Std 1788-2015 and the standard inclusion principle for interval arithmetic. For the error
model that composes local bounds across forward and reverse graphs, see `ForwardApprox.lean` and
`BackwardApprox.lean`; the organization follows the local-error/global-error distinction in
N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., 2002.
-/

@[expose] public section

namespace Proofs
namespace RuntimeApprox
namespace NumericalCertificate

open NN
open NN.Backend
open NN.IR
open Spec
open TorchLean.Floats.IEEE754

/-! ## Raw and checked source assumptions -/

/-- A binary32 range supplied for an input, constant, or explicit random source node. -/
structure SourceRange where
  nodeId : Nat
  enclosure : IEEE32Exec.Interval32
  deriving Repr

/-- A source range after the checker has established finite, ordered endpoints. -/
structure CheckedSourceRange extends SourceRange where
  valid : enclosure.Valid

instance : Repr CheckedSourceRange where
  reprPrec r _ := repr r.toSourceRange

/-- Bitwise equality for executable binary32 intervals.

Bitwise equality is intentional: it distinguishes signed zero and preserves the exact endpoints
written in a certificate. NaNs are rejected separately by `Interval32.Valid`.
-/
def sameIntervalBits (a b : IEEE32Exec.Interval32) : Bool :=
  a.lo.bits == b.lo.bits && a.hi.bits == b.hi.bits

/-- Executable counterpart of `Interval32.Valid`. -/
def validInterval (interval : IEEE32Exec.Interval32) : Bool :=
  IEEE32Exec.isFinite interval.lo &&
    (IEEE32Exec.isFinite interval.hi && IEEE32Exec.Interval32.leB interval.lo interval.hi)

/-- `Interval32.leB` decides the proposition-level IEEE non-strict order. -/
theorem leB_eq_true_iff (x y : IEEE32Exec) :
    IEEE32Exec.Interval32.leB x y = true <-> IEEE32Exec.le x y := by
  unfold IEEE32Exec.Interval32.leB IEEE32Exec.le
  cases h : IEEE32Exec.compare x y with
  | none => simp
  | some order =>
      cases order <;> simp

/-- IEEE comparison between finite values implies the corresponding order on their real
interpretations. This lemma is intentionally finite: IEEE comparisons involving NaN are unordered,
and `toReal` is not the semantic interface for infinities. -/
theorem toReal_le_toReal_of_le {x y : IEEE32Exec}
    (hx : IEEE32Exec.isFinite x = true) (hy : IEEE32Exec.isFinite y = true)
    (hxy : IEEE32Exec.le x y) : IEEE32Exec.toReal x <= IEEE32Exec.toReal y := by
  unfold IEEE32Exec.le at hxy
  cases hcompare : IEEE32Exec.compare x y with
  | none => simp [hcompare] at hxy
  | some order =>
      cases order with
      | lt =>
          exact le_of_lt <|
            (IEEE32Exec.compare_eq_some_lt_iff_toReal_lt_of_isFinite x y hx hy).mp hcompare
      | eq =>
          exact le_of_eq <|
            (IEEE32Exec.compare_eq_some_eq_iff_toReal_eq_of_isFinite x y hx hy).mp hcompare
      | gt => simp [hcompare] at hxy

/-- Negation of a finite executable binary32 value decodes to real negation. -/
theorem toReal_neg_of_isFinite {x : IEEE32Exec} (hx : IEEE32Exec.isFinite x = true) :
    IEEE32Exec.toReal (IEEE32Exec.neg x) = -IEEE32Exec.toReal x := by
  obtain ⟨dx, hdx⟩ := IEEE32Exec.exists_toDyadic?_of_isFinite hx
  exact IEEE32Exec.toReal_neg_eq_neg x hdx

/-- Flipping the sign bit preserves finiteness. -/
theorem isFinite_neg_of_isFinite {x : IEEE32Exec} (hx : IEEE32Exec.isFinite x = true) :
    IEEE32Exec.isFinite (IEEE32Exec.neg x) = true := by
  obtain ⟨dx, hdx⟩ := IEEE32Exec.exists_toDyadic?_of_isFinite hx
  have hdxNeg := IEEE32Exec.toDyadic?_neg_of_toDyadic?_some x hdx
  have hnan := IEEE32Exec.isNaN_eq_false_of_toDyadic?_some hdxNeg
  have hinf := IEEE32Exec.isInf_eq_false_of_toDyadic?_some hdxNeg
  exact IEEE32Exec.isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false
    (IEEE32Exec.neg x) hnan hinf

/-- The executable validity test accepts exactly finite, ordered intervals. -/
theorem validInterval_eq_true_iff (interval : IEEE32Exec.Interval32) :
    validInterval interval = true <-> interval.Valid := by
  simp [validInterval, IEEE32Exec.Interval32.Valid, leB_eq_true_iff]

/-! ## Real semantics of the arithmetic transfers

The executable checker propagates binary32 endpoints, while the graph specification is normally
read over real scalars. `RealEncloses` is the bridge between those views. Runtime rounding error is
then composed separately by `FwdGraph.eval_approx` and `RevGraph.backprop_approx`; keeping these two
claims separate prevents an interval enclosure from silently standing in for a floating-point
error theorem.
-/

/-- A real scalar lies between the real interpretations of an executable interval's endpoints. -/
def RealEncloses (interval : IEEE32Exec.Interval32) (value : Real) : Prop :=
  value ∈ Set.Icc (IEEE32Exec.toReal interval.lo) (IEEE32Exec.toReal interval.hi)

/-- Convert the extended-real endpoint form used by the interval soundness library into an
ordinary real interval when the output endpoints are finite. -/
theorem realEncloses_of_eReal_bounds {interval : IEEE32Exec.Interval32} {value : Real}
    (valid : interval.Valid)
    (bounds : IEEE32Exec.toEReal interval.lo <= (value : EReal) ∧
      (value : EReal) <= IEEE32Exec.toEReal interval.hi) :
    RealEncloses interval value := by
  have hlo := IEEE32Exec.toEReal_eq_coe_toReal_of_isFinite (x := interval.lo) valid.1
  have hhi := IEEE32Exec.toEReal_eq_coe_toReal_of_isFinite (x := interval.hi) valid.2.1
  constructor
  · rw [hlo] at bounds
    exact EReal.coe_le_coe_iff.mp bounds.1
  · rw [hhi] at bounds
    exact EReal.coe_le_coe_iff.mp bounds.2

/-- Sound real enclosure for the canonical addition transfer. -/
theorem add_realEncloses {a b : IEEE32Exec.Interval32} {x y : Real}
    (ha : a.Valid) (hb : b.Valid) (hout : (a.add b).Valid)
    (hx : RealEncloses a x) (hy : RealEncloses b y) :
    RealEncloses (a.add b) (x + y) :=
  realEncloses_of_eReal_bounds hout (a.add_sound b ha hb hx hy)

/-- Sound real enclosure for the canonical subtraction transfer. -/
theorem sub_realEncloses {a b : IEEE32Exec.Interval32} {x y : Real}
    (ha : a.Valid) (hb : b.Valid) (hout : (a.sub b).Valid)
    (hx : RealEncloses a x) (hy : RealEncloses b y) :
    RealEncloses (a.sub b) (x - y) :=
  realEncloses_of_eReal_bounds hout (a.sub_sound b ha hb hx hy)

/-- Sound real enclosure for the canonical multiplication transfer. -/
theorem mul_realEncloses {a b : IEEE32Exec.Interval32} {x y : Real}
    (ha : a.Valid) (hb : b.Valid) (hout : (a.mul b).Valid)
    (hx : RealEncloses a x) (hy : RealEncloses b y) :
    RealEncloses (a.mul b) (x * y) :=
  realEncloses_of_eReal_bounds hout (a.mul_sound b ha.1 ha.2.1 hb.1 hb.2.1 hx hy)

/-- Sound real enclosure for the canonical reciprocal transfer. -/
theorem inv_realEncloses {a : IEEE32Exec.Interval32} {x : Real}
    (ha : a.Valid) (hout : a.inv.Valid) (hx : RealEncloses a x) :
    RealEncloses a.inv x⁻¹ := by
  simpa [one_div] using
    (realEncloses_of_eReal_bounds hout (a.inv_sound ha hx))

/-- Every scalar entry of a shape-indexed real tensor lies in one interval. -/
def TensorEnclosed (interval : IEEE32Exec.Interval32) :
    {shape : Shape} -> Tensor Real shape -> Prop
  | .scalar, .scalar value => RealEncloses interval value
  | .dim _ _, .dim values => ∀ i, TensorEnclosed interval (values i)

/-- The exact executable interval `[0,1]`. -/
def unitInterval : IEEE32Exec.Interval32 :=
  { lo := IEEE32Exec.posZero, hi := IEEE32Exec.posOne }

/-- The exact executable interval `[-1,1]`. -/
def signedUnitInterval : IEEE32Exec.Interval32 :=
  { lo := IEEE32Exec.negOne, hi := IEEE32Exec.posOne }

/-- Executable test for a finite endpoint's nonnegative IEEE sign. Both signed zeros are accepted;
all other accepted values have a clear sign bit. Finiteness is supplied by interval validity. -/
def nonnegativeEndpoint (x : IEEE32Exec) : Bool :=
  IEEE32Exec.isZero x || !IEEE32Exec.signBit x

/-- The stable real vector softmax is enclosed by the certificate transfer `[0,1]`. -/
theorem softmaxVec_tensor_enclosed {n : Nat}
    (input : Tensor Real (.dim (Nat.succ n) .scalar)) :
    TensorEnclosed unitInterval (Activation.softmaxVecSpec input) := by
  cases hsoft : Activation.softmaxVecSpec input with
  | dim values =>
      intro i
      cases hvalue : values i with
      | scalar value =>
          have h := Proofs.softmax_vec_spec_mem_unitInterval input i
          rw [hsoft] at h
          simpa [TensorEnclosed, RealEncloses, unitInterval, Spec.toVec, hvalue,
            IEEE32Exec.Interval32.toReal_posOne] using h

/-- Sound real enclosure for the canonical ReLU interval transfer. -/
theorem relu_realEncloses {a : IEEE32Exec.Interval32} {x : Real}
    (ha : a.Valid) (hx : RealEncloses a x) :
    RealEncloses a.relu (max x 0) := by
  have hzero : IEEE32Exec.isFinite IEEE32Exec.posZero = true := by decide
  have hlo := IEEE32Exec.toReal_maximum_eq_max_of_isFinite a.lo IEEE32Exec.posZero ha.1 hzero
  have hhi := IEEE32Exec.toReal_maximum_eq_max_of_isFinite a.hi IEEE32Exec.posZero ha.2.1 hzero
  constructor
  · simpa [IEEE32Exec.Interval32.relu, hlo] using max_le_max hx.1 (le_refl (0 : Real))
  · simpa [IEEE32Exec.Interval32.relu, hhi] using max_le_max hx.2 (le_refl (0 : Real))

/-- Sound real enclosure for the canonical absolute-value interval transfer. -/
theorem abs_realEncloses {a : IEEE32Exec.Interval32} {x : Real}
    (ha : a.Valid) (hx : RealEncloses a x) :
    RealEncloses a.abs |x| := by
  by_cases hneg : IEEE32Exec.Interval32.leB a.hi IEEE32Exec.negZero = true
  · have hhiNonpos : IEEE32Exec.toReal a.hi <= 0 := by
      have hle := (leB_eq_true_iff a.hi IEEE32Exec.negZero).mp hneg
      simpa using toReal_le_toReal_of_le ha.2.1 (by decide) hle
    have hxNonpos : x <= 0 := hx.2.trans hhiNonpos
    have hnegLo := toReal_neg_of_isFinite ha.1
    have hnegHi := toReal_neg_of_isFinite ha.2.1
    rw [abs_of_nonpos hxNonpos]
    constructor <;>
      simp only [IEEE32Exec.Interval32.abs, hneg, if_pos, IEEE32Exec.Interval32.neg] <;>
      simp only [hnegLo, hnegHi] <;> linarith [hx.1, hx.2]
  · by_cases hpos : IEEE32Exec.Interval32.leB IEEE32Exec.posZero a.lo = true
    · have hloNonneg : 0 <= IEEE32Exec.toReal a.lo := by
        have hle := (leB_eq_true_iff IEEE32Exec.posZero a.lo).mp hpos
        simpa using toReal_le_toReal_of_le (by decide) ha.1 hle
      have hxNonneg : 0 <= x := hloNonneg.trans hx.1
      simpa [IEEE32Exec.Interval32.abs, hneg, hpos, abs_of_nonneg hxNonneg] using hx
    · have hzero : IEEE32Exec.isFinite IEEE32Exec.posZero = true := by decide
      have hnegLo := toReal_neg_of_isFinite ha.1
      have hmax := IEEE32Exec.toReal_maximum_eq_max_of_isFinite
        (IEEE32Exec.neg a.lo) a.hi
        (isFinite_neg_of_isFinite ha.1) ha.2.1
      have hupper : |x| <= max (-IEEE32Exec.toReal a.lo) (IEEE32Exec.toReal a.hi) := by
        apply (abs_le).2
        constructor
        · have := le_max_left (-IEEE32Exec.toReal a.lo) (IEEE32Exec.toReal a.hi)
          linarith [hx.1]
        · exact hx.2.trans (le_max_right _ _)
      constructor
      · simp [IEEE32Exec.Interval32.abs, hneg, hpos, abs_nonneg]
      · simpa [IEEE32Exec.Interval32.abs, hneg, hpos, hnegLo, hmax] using hupper

/-- A directed lower square-root endpoint lies below the exact real square root. Signed zero is
handled separately because IEEE preserves its sign, while the general directed-rounding theorem is
stated for sign-bit-false inputs. -/
theorem toReal_sqrtDown_le {x : IEEE32Exec}
    (hfin : IEEE32Exec.isFinite x = true) (hdomain : nonnegativeEndpoint x = true)
    (hout : IEEE32Exec.isFinite (IEEE32Exec.sqrtDown x) = true) :
    IEEE32Exec.toReal (IEEE32Exec.sqrtDown x) <= Real.sqrt (IEEE32Exec.toReal x) := by
  by_cases hzero : IEEE32Exec.isZero x = true
  · obtain ⟨dx, hdx⟩ := IEEE32Exec.exists_toDyadic?_of_isFinite hfin
    have hnan := IEEE32Exec.isNaN_eq_false_of_toDyadic?_some hdx
    have hinf := IEEE32Exec.isInf_eq_false_of_toDyadic?_some hdx
    have hchoose : IEEE32Exec.chooseNaN1 x = none := by simp [IEEE32Exec.chooseNaN1, hnan]
    have hsqrt : IEEE32Exec.sqrtDown x = x := by
      simp [IEEE32Exec.sqrtDown, hchoose, hinf, hzero]
    have hreal := IEEE32Exec.toReal_eq_zero_of_isZero x hdx hzero
    simp [hsqrt, hreal]
  · have hsign : IEEE32Exec.signBit x = false := by
      simp [nonnegativeEndpoint, hzero] at hdomain
      exact hdomain
    have h := IEEE32Exec.toEReal_sqrtDown_le x hfin hsign
    rw [IEEE32Exec.toEReal_eq_coe_toReal_of_isFinite (IEEE32Exec.sqrtDown x) hout] at h
    exact EReal.coe_le_coe_iff.mp h

/-- Upper counterpart of `toReal_sqrtDown_le`. -/
theorem toReal_sqrtUp_ge {x : IEEE32Exec}
    (hfin : IEEE32Exec.isFinite x = true) (hdomain : nonnegativeEndpoint x = true)
    (hout : IEEE32Exec.isFinite (IEEE32Exec.sqrtUp x) = true) :
    Real.sqrt (IEEE32Exec.toReal x) <= IEEE32Exec.toReal (IEEE32Exec.sqrtUp x) := by
  by_cases hzero : IEEE32Exec.isZero x = true
  · obtain ⟨dx, hdx⟩ := IEEE32Exec.exists_toDyadic?_of_isFinite hfin
    have hnan := IEEE32Exec.isNaN_eq_false_of_toDyadic?_some hdx
    have hinf := IEEE32Exec.isInf_eq_false_of_toDyadic?_some hdx
    have hchoose : IEEE32Exec.chooseNaN1 x = none := by simp [IEEE32Exec.chooseNaN1, hnan]
    have hsqrt : IEEE32Exec.sqrtUp x = x := by
      simp [IEEE32Exec.sqrtUp, hchoose, hinf, hzero]
    have hreal := IEEE32Exec.toReal_eq_zero_of_isZero x hdx hzero
    simp [hsqrt, hreal]
  · have hsign : IEEE32Exec.signBit x = false := by
      simp [nonnegativeEndpoint, hzero] at hdomain
      exact hdomain
    have h := IEEE32Exec.toEReal_sqrtUp_ge x hfin hsign
    rw [IEEE32Exec.toEReal_eq_coe_toReal_of_isFinite (IEEE32Exec.sqrtUp x) hout] at h
    exact EReal.coe_le_coe_iff.mp h

/-- Sound real enclosure for directed interval square root. -/
theorem sqrt_realEncloses {a : IEEE32Exec.Interval32} {x : Real}
    (ha : a.Valid) (hlo : nonnegativeEndpoint a.lo = true)
    (hhi : nonnegativeEndpoint a.hi = true) (hout : a.sqrt.Valid)
    (hx : RealEncloses a x) : RealEncloses a.sqrt (Real.sqrt x) := by
  constructor
  · exact (toReal_sqrtDown_le ha.1 hlo hout.1).trans (Real.sqrt_le_sqrt hx.1)
  · exact (Real.sqrt_le_sqrt hx.2).trans (toReal_sqrtUp_ge ha.2.1 hhi hout.2.1)

/-- Lift a sound unary scalar transfer to tensors of arbitrary rank. -/
theorem tensor_map_enclosed
    (op : Real -> Real) (input output : IEEE32Exec.Interval32)
    (sound : ∀ {x}, RealEncloses input x -> RealEncloses output (op x)) :
    ∀ {shape : Shape} {x : Tensor Real shape},
      TensorEnclosed input x -> TensorEnclosed output (Tensor.mapSpec op x) := by
  intro shape
  induction shape with
  | scalar =>
      intro x hx
      cases x with
      | scalar value => exact sound hx
  | dim n shape ih =>
      intro x hx
      cases x with
      | dim values =>
          intro i
          exact ih (x := values i) (hx i)

/-- Tensor-level soundness of the ReLU interval transfer. -/
theorem tensor_relu_enclosed {shape : Shape} {x : Tensor Real shape}
    {a : IEEE32Exec.Interval32} (ha : a.Valid) (hx : TensorEnclosed a x) :
    TensorEnclosed a.relu (Tensor.mapSpec (fun value => max value 0) x) :=
  tensor_map_enclosed (fun value => max value 0) a a.relu
    (fun hx' => relu_realEncloses ha hx') hx

/-- Tensor-level soundness of the absolute-value interval transfer. -/
theorem tensor_abs_enclosed {shape : Shape} {x : Tensor Real shape}
    {a : IEEE32Exec.Interval32} (ha : a.Valid) (hx : TensorEnclosed a x) :
    TensorEnclosed a.abs (Tensor.mapSpec abs x) :=
  tensor_map_enclosed abs a a.abs (fun hx' => abs_realEncloses ha hx') hx

/-- Tensor-level soundness of directed interval square root. -/
theorem tensor_sqrt_enclosed {shape : Shape} {x : Tensor Real shape}
    {a : IEEE32Exec.Interval32} (ha : a.Valid)
    (hlo : nonnegativeEndpoint a.lo = true) (hhi : nonnegativeEndpoint a.hi = true)
    (hout : a.sqrt.Valid) (hx : TensorEnclosed a x) :
    TensorEnclosed a.sqrt (Tensor.mapSpec Real.sqrt x) :=
  tensor_map_enclosed Real.sqrt a a.sqrt
    (fun hx' => sqrt_realEncloses ha hlo hhi hout hx') hx

/-- Lift a sound binary scalar transfer to tensors of arbitrary rank. -/
theorem tensor_map2_enclosed
    (op : Real -> Real -> Real) (a b out : IEEE32Exec.Interval32)
    (sound : ∀ {x y}, RealEncloses a x -> RealEncloses b y -> RealEncloses out (op x y)) :
    ∀ {shape : Shape} {x y : Tensor Real shape},
      TensorEnclosed a x -> TensorEnclosed b y ->
        TensorEnclosed out (Tensor.map2Spec op x y) := by
  intro shape
  induction shape with
  | scalar =>
      intro x y hx hy
      cases x with
      | scalar value =>
          cases y with
          | scalar other => exact sound hx hy
  | dim n shape ih =>
      intro x y hx hy
      cases x with
      | dim values =>
          cases y with
          | dim others =>
              intro i
              exact ih (x := values i) (y := others i) (hx i) (hy i)

/-- Tensor-level soundness of outward-rounded interval addition. -/
theorem tensor_add_enclosed {shape : Shape} {x y : Tensor Real shape}
    {a b : IEEE32Exec.Interval32}
    (ha : a.Valid) (hb : b.Valid) (hout : (a.add b).Valid)
    (hx : TensorEnclosed a x) (hy : TensorEnclosed b y) :
    TensorEnclosed (a.add b) (Tensor.addSpec x y) :=
  tensor_map2_enclosed (fun u v => u + v) a b (a.add b)
    (fun hx' hy' => add_realEncloses ha hb hout hx' hy') hx hy

/-- Tensor-level soundness of outward-rounded interval subtraction. -/
theorem tensor_sub_enclosed {shape : Shape} {x y : Tensor Real shape}
    {a b : IEEE32Exec.Interval32}
    (ha : a.Valid) (hb : b.Valid) (hout : (a.sub b).Valid)
    (hx : TensorEnclosed a x) (hy : TensorEnclosed b y) :
    TensorEnclosed (a.sub b) (Tensor.subSpec x y) :=
  tensor_map2_enclosed (fun u v => u - v) a b (a.sub b)
    (fun hx' hy' => sub_realEncloses ha hb hout hx' hy') hx hy

/-- Tensor-level soundness of outward-rounded interval multiplication. -/
theorem tensor_mul_enclosed {shape : Shape} {x y : Tensor Real shape}
    {a b : IEEE32Exec.Interval32}
    (ha : a.Valid) (hb : b.Valid) (hout : (a.mul b).Valid)
    (hx : TensorEnclosed a x) (hy : TensorEnclosed b y) :
    TensorEnclosed (a.mul b) (Tensor.mulSpec x y) :=
  tensor_map2_enclosed (fun u v => u * v) a b (a.mul b)
    (fun hx' hy' => mul_realEncloses ha hb hout hx' hy') hx hy

/-! ## Replay against bit-level graph execution -/

/-- Executable check that every binary32 tensor entry lies in an interval. -/
def tensorWithinRange (interval : IEEE32Exec.Interval32) :
    {shape : Shape} -> Tensor IEEE32Exec shape -> Bool
  | .scalar, .scalar value =>
      IEEE32Exec.isFinite value &&
        (IEEE32Exec.Interval32.leB interval.lo value &&
          IEEE32Exec.Interval32.leB value interval.hi)
  | .dim n _, .dim values =>
      (List.finRange n).all (fun i => tensorWithinRange interval (values i))

/-- Proposition expressed by `tensorWithinRange`. -/
def IEEETensorEnclosed (interval : IEEE32Exec.Interval32) :
    {shape : Shape} -> Tensor IEEE32Exec shape -> Prop
  | .scalar, .scalar value =>
      IEEE32Exec.isFinite value = true ∧
        IEEE32Exec.le interval.lo value ∧ IEEE32Exec.le value interval.hi
  | .dim _ _, .dim values => ∀ i, IEEETensorEnclosed interval (values i)

/-- The executable tensor range check is exact for the IEEE comparison semantics. -/
theorem tensorWithinRange_eq_true_iff (interval : IEEE32Exec.Interval32)
    {shape : Shape} (tensor : Tensor IEEE32Exec shape) :
    tensorWithinRange interval tensor = true <-> IEEETensorEnclosed interval tensor := by
  induction shape with
  | scalar =>
      cases tensor with
      | scalar value =>
          simp [tensorWithinRange, IEEETensorEnclosed, leB_eq_true_iff]
  | dim n shape ih =>
      cases tensor with
      | dim values =>
          simp [tensorWithinRange, IEEETensorEnclosed, List.all_eq_true, ih]

/-! ## From checked ranges to explicit error bounds -/

/-- Decode an executable tensor entrywise and state that the resulting real tensor lies in an
interval. Unlike `IEEETensorEnclosed`, this predicate talks directly about the real values used by
the approximation layer. -/
def DecodedTensorEnclosed (interval : IEEE32Exec.Interval32) :
    {shape : Shape} -> Tensor IEEE32Exec shape -> Prop
  | .scalar, .scalar value => RealEncloses interval (IEEE32Exec.toReal value)
  | .dim _ _, .dim values => ∀ i, DecodedTensorEnclosed interval (values i)

/-- A successful IEEE range check decodes to an ordinary real enclosure. Finiteness is an explicit
part of `IEEETensorEnclosed`, so this theorem never assigns a real meaning to NaN or infinity. -/
theorem decodedTensorEnclosed_of_ieee {interval : IEEE32Exec.Interval32}
    (valid : interval.Valid) :
    ∀ {shape : Shape} {tensor : Tensor IEEE32Exec shape},
      IEEETensorEnclosed interval tensor -> DecodedTensorEnclosed interval tensor := by
  intro shape
  induction shape with
  | scalar =>
      intro tensor htensor
      cases tensor with
      | scalar value =>
          exact ⟨toReal_le_toReal_of_le valid.1 htensor.1 htensor.2.1,
            toReal_le_toReal_of_le htensor.1 valid.2.1 htensor.2.2⟩
  | dim n shape ih =>
      intro tensor htensor
      cases tensor with
      | dim values =>
          intro i
          exact ih (htensor i)

/-- Pointwise absolute error between a real specification tensor and an executable binary32
tensor. The shape index is shared, so no runtime shape cast is hidden in the relation. -/
def TensorErrorLe (eps : Real) :
    {shape : Shape} -> Tensor Real shape -> Tensor IEEE32Exec shape -> Prop
  | .scalar, .scalar exact, .scalar computed =>
      |IEEE32Exec.toReal computed - exact| <= eps
  | .dim _ _, .dim exact, .dim computed =>
      ∀ i, TensorErrorLe eps (exact i) (computed i)

/-- Width of a finite executable interval, interpreted in the reals. -/
noncomputable def intervalWidth (interval : IEEE32Exec.Interval32) : Real :=
  IEEE32Exec.toReal interval.hi - IEEE32Exec.toReal interval.lo

/-- A valid interval has nonnegative real width. -/
theorem intervalWidth_nonneg {interval : IEEE32Exec.Interval32} (valid : interval.Valid) :
    0 <= intervalWidth interval := by
  have hle := toReal_le_toReal_of_le valid.1 valid.2.1 valid.2.2
  simp only [intervalWidth]
  linarith

/-- Two tensors enclosed by the same interval differ entrywise by at most its width.

This is the elementary bridge from range analysis to approximation analysis. It is deliberately
pointwise; a later norm theorem can package the same statement as an `L∞` bound without changing
the checker or its certificate format. -/
theorem tensor_error_le_width_of_enclosed {interval : IEEE32Exec.Interval32} :
    ∀ {shape : Shape} {exact : Tensor Real shape} {computed : Tensor IEEE32Exec shape},
      TensorEnclosed interval exact ->
      DecodedTensorEnclosed interval computed ->
      TensorErrorLe (intervalWidth interval) exact computed := by
  intro shape
  induction shape with
  | scalar =>
      intro exact computed hexact hcomputed
      cases exact with
      | scalar x =>
          cases computed with
          | scalar y =>
              simp only [TensorErrorLe, intervalWidth]
              apply (abs_le).2
              constructor <;> linarith [hexact.1, hexact.2, hcomputed.1, hcomputed.2]
  | dim n shape ih =>
      intro exact computed hexact hcomputed
      cases exact with
      | dim exactValues =>
          cases computed with
          | dim computedValues =>
              intro i
              exact ih (hexact i) (hcomputed i)

/-- A successful executable range check and a real enclosure proof yield a concrete pointwise
error bound. This theorem is the tensor-level core used by graph-wide numerical certificates. -/
theorem tensor_error_le_width_of_check {interval : IEEE32Exec.Interval32}
    (valid : interval.Valid) {shape : Shape} {exact : Tensor Real shape}
    {computed : Tensor IEEE32Exec shape}
    (hexact : TensorEnclosed interval exact)
    (hcheck : tensorWithinRange interval computed = true) :
    TensorErrorLe (intervalWidth interval) exact computed :=
  tensor_error_le_width_of_enclosed hexact <|
    decodedTensorEnclosed_of_ieee valid <|
      (tensorWithinRange_eq_true_iff interval computed).mp hcheck

/-- Check source ranges once, rejecting malformed intervals and duplicate node ids. -/
def checkSources (sources : Array SourceRange) : Except String (Array CheckedSourceRange) := do
  let mut checked : Array CheckedSourceRange := #[]
  let mut seen : List Nat := []
  for source in sources do
    if seen.contains source.nodeId then
      throw s!"numerical certificate: duplicate source range for node {source.nodeId}"
    if h : validInterval source.enclosure then
      checked := checked.push
        { source with valid := (validInterval_eq_true_iff source.enclosure).mp h }
      seen := source.nodeId :: seen
    else
      throw s!"numerical certificate: source range for node {source.nodeId} is not finite and ordered"
  pure checked

/-- Find the checked assumption for a source node. -/
def findSource (sources : Array CheckedSourceRange) (nodeId : Nat) :
    Except String CheckedSourceRange :=
  match sources.find? (fun source => source.nodeId == nodeId) with
  | some source => pure source
  | none => throw s!"numerical certificate: missing source range for node {nodeId}"

/-- Whether a graph node obtains its enclosure directly from a certificate source assumption. -/
def opUsesSourceRange : OpKind -> Bool
  | .input | .const _ | .randUniform _ | .bernoulliMask _ => true
  | _ => false

/-- Reject source assumptions that do not name a source-like node in the checked graph.

Unused assumptions do not make interval propagation unsound, but they make artifacts ambiguous:
an exporter may have attached a valid range to the wrong node id without noticing. Requiring every
row to be consumed gives source arrays one canonical interpretation and catches that error before
range propagation begins.
-/
def checkSourceOwnership (graph : Graph) (sources : Array CheckedSourceRange) : Except String Unit :=
  for source in sources do
    match graph.nodes[source.nodeId]? with
    | none =>
        throw s!"numerical certificate: source range names missing node {source.nodeId}"
    | some node =>
        if opUsesSourceRange node.kind then
          pure ()
        else
          throw s!"numerical certificate: node {source.nodeId} ({node.kind.describe}) does not consume a source range"

/-! ## Canonical local transfer rules -/

/-- How a node enclosure was obtained from source assumptions or earlier nodes.

The rule is recorded to make certificate diagnostics useful. It is not accepted on faith:
`check` reconstructs the rule and endpoints from the graph.
-/
inductive RangeRule where
  | source
  | preserve (parent : Nat)
  | add (left right : Nat)
  | sub (left right : Nat)
  | mul (left right : Nat)
  | inv (parent : Nat)
  | hull (parents : List Nat)
  | hullZero (parent : Nat)
  | sumLeft (parent count : Nat)
  | meanLeft (parent count : Nat)
  | matmulLeft (left right innerDim : Nat)
  | averageWindowLeft (parent windowSize : Nat) (includesPadding : Bool)
  | mseLeft (prediction target count : Nat)
  | layerNormLeft (parent axis normalizedSize : Nat)
  | softmaxUnit (parent axis : Nat)
  | relu (parent : Nat)
  | abs (parent : Nat)
  | sqrtNonnegative (parent : Nat)
  | unitBound (parent : Nat)
  | signedUnitBound (parent : Nat)
  deriving DecidableEq, Repr

/-- Proof-free data for one graph node's numerical range. -/
structure NodeRange where
  nodeId : Nat
  outShape : Shape
  rule : RangeRule
  enclosure : IEEE32Exec.Interval32
  deriving Repr

/-- A node range whose executable interval has passed the finite/order check. -/
structure CheckedNodeRange extends NodeRange where
  valid : enclosure.Valid

instance : Repr CheckedNodeRange where
  reprPrec r _ := repr r.toNodeRange

/-- Check a dynamic graph value against the declared shape and interval of one certificate row. -/
def dvalWithinRange (range : CheckedNodeRange) (value : NN.IR.DVal IEEE32Exec) : Bool :=
  if h : value.shape = range.outShape then
    tensorWithinRange range.enclosure (h ▸ value.tensor)
  else
    false

/-- A real dynamic graph value has the shape declared by a certificate row and is enclosed by its
interval. The equality witness makes the dependent tensor cast explicit. -/
def RealDValEnclosed (range : CheckedNodeRange) (value : NN.IR.DVal Real) : Prop :=
  ∃ h : value.shape = range.outShape,
    TensorEnclosed range.enclosure (h ▸ value.tensor)

/-- Pointwise approximation relation for real and IEEE dynamic graph values at one certificate
row. -/
def DValErrorLe (range : CheckedNodeRange) (exact : NN.IR.DVal Real)
    (computed : NN.IR.DVal IEEE32Exec) : Prop :=
  ∃ hexact : exact.shape = range.outShape,
    ∃ hcomputed : computed.shape = range.outShape,
      TensorErrorLe (intervalWidth range.enclosure)
        (hexact ▸ exact.tensor) (hcomputed ▸ computed.tensor)

/-- One successful dynamic replay row yields a pointwise error bound whenever the corresponding
real graph value has the proved enclosure. -/
theorem dval_error_le_of_range_check {range : CheckedNodeRange}
    {exact : NN.IR.DVal Real} {computed : NN.IR.DVal IEEE32Exec}
    (hexact : RealDValEnclosed range exact)
    (hcomputed : dvalWithinRange range computed = true) :
    DValErrorLe range exact computed := by
  rcases hexact with ⟨hexactShape, hexactRange⟩
  unfold dvalWithinRange at hcomputed
  split at hcomputed
  next hcomputedShape =>
    exact ⟨hexactShape, hcomputedShape,
      tensor_error_le_width_of_check range.valid hexactRange hcomputed⟩
  next _ => simp at hcomputed

/-- List-level replay check. Its structural recursion is also the proof interface for composing
per-node numerical guarantees over a complete execution trace. -/
def executionWithinRangesList : List CheckedNodeRange -> List (NN.IR.DVal IEEE32Exec) -> Bool
  | [], [] => true
  | range :: ranges, value :: values =>
      dvalWithinRange range value && executionWithinRangesList ranges values
  | _, _ => false

/-- Check every value produced by `IR.Graph.denoteAll` against the corresponding certificate row. -/
def executionWithinRanges (ranges : Array CheckedNodeRange)
    (values : Array (NN.IR.DVal IEEE32Exec)) : Bool :=
  executionWithinRangesList ranges.toList values.toList

/-- Proof-level meaning of a complete successful IEEE replay trace. -/
theorem executionWithinRangesList_eq_true_iff
    (ranges : List CheckedNodeRange) (values : List (NN.IR.DVal IEEE32Exec)) :
    executionWithinRangesList ranges values = true <->
      List.Forall₂ (fun range value => dvalWithinRange range value = true) ranges values := by
  induction ranges generalizing values with
  | nil => cases values <;> simp [executionWithinRangesList]
  | cons range ranges ih =>
      cases values with
      | nil => simp [executionWithinRangesList]
      | cons value values => simp [executionWithinRangesList, ih]

/-- Graph-wide pointwise approximation evidence, one row per intermediate value. -/
inductive ExecutionErrorTrace :
    List CheckedNodeRange -> List (NN.IR.DVal Real) -> List (NN.IR.DVal IEEE32Exec) -> Prop
  | nil : ExecutionErrorTrace [] [] []
  | cons {range ranges exact exacts computed computeds} :
      DValErrorLe range exact computed ->
      ExecutionErrorTrace ranges exacts computeds ->
      ExecutionErrorTrace (range :: ranges) (exact :: exacts) (computed :: computeds)

/-- Compose real enclosure proofs and successful IEEE replay checks into an error trace. -/
theorem executionErrorTrace_of_enclosed
    {ranges : List CheckedNodeRange} {exact : List (NN.IR.DVal Real)}
    {computed : List (NN.IR.DVal IEEE32Exec)}
    (hexact : List.Forall₂ RealDValEnclosed ranges exact)
    (hcomputed : List.Forall₂
      (fun range value => dvalWithinRange range value = true) ranges computed) :
    ExecutionErrorTrace ranges exact computed := by
  induction hexact generalizing computed with
  | nil =>
      cases hcomputed
      exact .nil
  | cons hexactHead hexactTail ih =>
      cases hcomputed with
      | cons hcomputedHead hcomputedTail =>
          exact .cons (dval_error_le_of_range_check hexactHead hcomputedHead)
            (ih hcomputedTail)

/-- Array-facing whole-trace theorem used by checked graph executions. -/
theorem execution_error_trace_of_check
    {ranges : Array CheckedNodeRange} {exact : Array (NN.IR.DVal Real)}
    {computed : Array (NN.IR.DVal IEEE32Exec)}
    (hexact : List.Forall₂ RealDValEnclosed ranges.toList exact.toList)
    (hcomputed : executionWithinRanges ranges computed = true) :
    ExecutionErrorTrace ranges.toList exact.toList computed.toList :=
  executionErrorTrace_of_enclosed hexact <|
    (executionWithinRangesList_eq_true_iff ranges.toList computed.toList).mp hcomputed

/-- Compare a checked canonical row with untrusted raw certificate data. -/
def sameNodeRange (checked : CheckedNodeRange) (raw : NodeRange) : Bool :=
  decide (checked.nodeId = raw.nodeId) &&
    decide (checked.outShape = raw.outShape) &&
    decide (checked.rule = raw.rule) &&
    sameIntervalBits checked.enclosure raw.enclosure

/-- Read a previously checked parent enclosure. Graph well-formedness guarantees that successful
lookups refer only to earlier rows; the explicit error still protects this API when called alone. -/
def parentRange (ranges : Array CheckedNodeRange) (nodeId : Nat) :
    Except String IEEE32Exec.Interval32 :=
  match ranges[nodeId]? with
  | some range => pure range.enclosure
  | none => throw s!"numerical certificate: missing range for parent node {nodeId}"

/-- Read the complete checked row for a parent node. -/
def parentNodeRange (ranges : Array CheckedNodeRange) (nodeId : Nat) :
    Except String CheckedNodeRange :=
  match ranges[nodeId]? with
  | some range => pure range
  | none => throw s!"numerical certificate: missing range for parent node {nodeId}"

/-- Outward-rounded left-fold range for a sum of `count` values from one enclosure. The initial
point interval at positive zero matches `Tensor.sumSpec`. -/
def sumLeftRange (count : Nat) (range : IEEE32Exec.Interval32) : IEEE32Exec.Interval32 :=
  (List.range count).foldl
    (fun acc _ => IEEE32Exec.Interval32.add acc range)
    (IEEE32Exec.Interval32.point IEEE32Exec.posZero)

/-- Left-fold mean range, using the same binary32 conversion of the divisor as the tensor context. -/
def meanLeftRange (count : Nat) (range : IEEE32Exec.Interval32) : IEEE32Exec.Interval32 :=
  IEEE32Exec.Interval32.div (sumLeftRange count range)
    (IEEE32Exec.Interval32.point (count : IEEE32Exec))

/-- Numerical policy selected for a runtime-relevant graph node. -/
def nodeNumericalPolicy (plan : AcceptedGraphPlan) (nodeId : Nat) : Option NumericalPolicy :=
  (plan.graphPlan.kernels.find? (fun kernel => kernel.nodeId == nodeId)).map
    (fun kernel => kernel.capsule.numericalPolicy)

/-- Reductions are propagated only when the selected capsule promises the same fixed left fold as
the canonical tensor semantics. Other schedules need the order-independent reduction bound from
`NN.Floats.IEEEExec.Reductions` and are rejected here rather than mislabeled as deterministic. -/
def requireFixedLeftReduction (plan : AcceptedGraphPlan) (node : Node) : Except String Unit :=
  match nodeNumericalPolicy plan node.id with
  | some policy =>
      if policy.reduction = .fixedLeft then
        pure ()
      else
        throw s!"numerical certificate: node {node.id} ({node.kind.describe}) uses reduction policy {repr policy.reduction}; fixedLeft is required by this transfer"
  | none =>
      throw s!"numerical certificate: node {node.id} ({node.kind.describe}) has no backend numerical policy"

/-- Inner accumulation length for the rank-2 and batched rank-3 matrix products implemented by
`IR.Graph.denoteAll`. The graph shape checker has already validated matching dimensions; retaining
the checks here gives callers of `deriveNodeRange` a precise error instead of relying on that
ambient invariant. -/
def matmulInnerDim (left right : Shape) : Except String Nat :=
  match left, right with
  | .dim _ (.dim n .scalar), .dim n' (.dim _ .scalar) =>
      if n = n' then pure n else throw s!"matmul inner dimensions differ: {n} vs {n'}"
  | .dim batch (.dim _ (.dim n .scalar)),
      .dim batch' (.dim n' (.dim _ .scalar)) =>
      if batch = batch' then
        if n = n' then pure n else throw s!"batched matmul inner dimensions differ: {n} vs {n'}"
      else
        throw s!"batched matmul batch dimensions differ: {batch} vs {batch'}"
  | _, _ => throw s!"unsupported matmul shapes: {repr left} and {repr right}"

/-- Hull of a nonempty list of parent ranges. -/
def hullParents (ranges : Array CheckedNodeRange) :
    List Nat -> Except String IEEE32Exec.Interval32
  | [] => throw "numerical certificate: an interval hull requires at least one parent"
  | parent :: parents => do
      let first <- parentRange ranges parent
      parents.foldlM (fun acc id => do
        let next <- parentRange ranges id
        pure (IEEE32Exec.Interval32.hull acc next)) first

/-! ## Graph range contracts

Architectures do not participate in range propagation directly. They lower to `NN.IR.Graph`, and
each graph node is handled by a reusable operation contract. This keeps MLPs, convolutional
networks, transformers, and future model families on one checker path: adding a model requires no
new certificate traversal, while adding a genuinely new primitive requires one local contract.

The registry is an explicit value rather than global mutable state. Certificate generation and
checking therefore use the same inspectable rule set, and downstream projects may extend it
without changing TorchLean's graph walker.
-/

/-- Stable key for a numerical range contract.

Input-like nodes share the `source` contract, `detach` uses the structural identity contract, and
runtime operations use the same `BackendOp` vocabulary as kernel capsules and execution plans.
-/
inductive NumericalOpKey where
  | source
  | structural
  | wholeSum
  | maxPool
  | maxPoolPad
  | averagePool
  | averagePoolPad
  | backend (op : BackendOp)
  | unclassified
  deriving DecidableEq, Repr

/-- Classify an IR operation for numerical-contract lookup. -/
def numericalOpKey : OpKind -> NumericalOpKey
  | .input | .const _ | .randUniform _ | .bernoulliMask _ => .source
  | .detach => .structural
  | .sum => .wholeSum
  | .maxPool2d .. => .maxPool
  | .maxPool2dPad .. => .maxPoolPad
  | .avgPool2d .. => .averagePool
  | .avgPool2dPad .. => .averagePoolPad
  | kind =>
      match NN.Backend.IR.op? kind with
      | some op => .backend op
      | none => .unclassified

/-- Read-only state supplied to one local range transfer. -/
structure NumericalRangeContext where
  sources : Array CheckedSourceRange
  plan : AcceptedGraphPlan
  ranges : Array CheckedNodeRange

/-- Result computed by one numerical operation contract. -/
abbrev RangeTransferResult := Prod RangeRule IEEE32Exec.Interval32

/-- Executable range transfer for one operation family.

The proof-facing meaning of the resulting row remains `RealDValEnclosed`; local soundness lemmas
for interval arithmetic and NF approximation are kept in their mathematical modules. The contract
contains only executable dispatch and a stable key, so serializable certificates cannot inject
proof evidence.
-/
structure GraphRangeContract where
  key : NumericalOpKey
  name : String
  derive : NumericalRangeContext -> Node -> Except String RangeTransferResult

/-- Deterministic registry used by graph certificate generation and replay. -/
structure GraphRangeRegistry where
  name : String
  contracts : List GraphRangeContract

namespace GraphRangeRegistry

/-- Empty named registry for downstream composition. -/
def empty (name : String) : GraphRangeRegistry := ⟨name, []⟩

/-- Find the unique contract associated with a numerical operation key. -/
def find? (registry : GraphRangeRegistry) (key : NumericalOpKey) :
    Option GraphRangeContract :=
  registry.contracts.find? (fun contract => decide (contract.key = key))

/-- Add one contract, rejecting duplicate keys so dispatch never depends on list order. -/
def register (registry : GraphRangeRegistry) (contract : GraphRangeContract) :
    Except String GraphRangeRegistry :=
  match registry.find? contract.key with
  | some previous =>
      throw s!"numerical contract: duplicate key {repr contract.key} ({previous.name}, {contract.name})"
  | none => pure { registry with contracts := registry.contracts ++ [contract] }

/-- Build a registry while checking key uniqueness. -/
def ofList (name : String) (contracts : List GraphRangeContract) :
    Except String GraphRangeRegistry :=
  contracts.foldlM register (empty name)

end GraphRangeRegistry

/-- One graph node for which a numerical registry has no local transfer. -/
structure MissingNumericalContract where
  nodeId : Nat
  operation : String
  key : NumericalOpKey
  deriving Repr

/-- Architecture-independent coverage report obtained after lowering a model to `NN.IR.Graph`. -/
structure NumericalCoverageReport where
  registryName : String
  nodeCount : Nat
  coveredCount : Nat
  missing : List MissingNumericalContract
  deriving Repr

/-- Inspect contract coverage without attempting interval propagation. -/
def numericalCoverage (registry : GraphRangeRegistry) (graph : Graph) :
    NumericalCoverageReport :=
  let missing := graph.nodes.toList.filterMap fun node =>
    let key := numericalOpKey node.kind
    if registry.find? key |>.isSome then none
    else some { nodeId := node.id, operation := node.kind.describe, key }
  { registryName := registry.name
    nodeCount := graph.nodes.size
    coveredCount := graph.nodes.size - missing.length
    missing }

/-- Reject a graph before propagation when any primitive lacks a numerical contract. -/
def requireNumericalCoverage (registry : GraphRangeRegistry) (graph : Graph) :
    Except String NumericalCoverageReport := do
  let report := numericalCoverage registry graph
  if report.missing.isEmpty then
    pure report
  else
    throw s!"numerical certificate: registry {registry.name} does not cover graph nodes {repr report.missing}"

/-- Standard diagnostic for a contract whose graph arity does not match its operation. -/
def arityError (contractName : String) (node : Node) (expected : String) : String :=
  s!"numerical contract {contractName}: node {node.id} ({node.kind.describe}) expected {expected}, got {node.parents.length} parent(s)"

/-- Shared source-node contract. The source interval remains an explicit certificate assumption. -/
def sourceContract : GraphRangeContract where
  key := .source
  name := "source"
  derive := fun context node => do
    let assumption <- findSource context.sources node.id
    pure (.source, assumption.enclosure)

/-- Structural identity used by `detach`. -/
def structuralContract : GraphRangeContract where
  key := .structural
  name := "structural identity"
  derive := fun context node =>
    match node.parents with
    | [parent] => do
        let enclosure <- parentRange context.ranges parent
        pure (.preserve parent, enclosure)
    | _ => throw (arityError "structural identity" node "one parent")

/-- Reusable contract constructor for value-preserving graph operations. -/
def preserveContract (op : BackendOp) : GraphRangeContract where
  key := .backend op
  name := s!"{op.name} value preservation"
  derive := fun context node =>
    match node.parents with
    | [parent] => do
        let enclosure <- parentRange context.ranges parent
        pure (.preserve parent, enclosure)
    | _ => throw (arityError op.name node "one parent")

/-- Reusable contract constructor for pointwise binary interval operations. -/
def binaryContract (op : BackendOp) (rule : Nat -> Nat -> RangeRule)
    (transfer : IEEE32Exec.Interval32 -> IEEE32Exec.Interval32 -> IEEE32Exec.Interval32) :
    GraphRangeContract where
  key := .backend op
  name := s!"{op.name} binary transfer"
  derive := fun context node =>
    match node.parents with
    | [left, right] => do
        let a <- parentRange context.ranges left
        let b <- parentRange context.ranges right
        pure (rule left right, transfer a b)
    | _ => throw (arityError op.name node "two parents")

/-- Reusable contract for operations whose output is enclosed by the hull of their parents. -/
def hullContract (op : BackendOp) : GraphRangeContract where
  key := .backend op
  name := s!"{op.name} hull"
  derive := fun context node => do
    let enclosure <- hullParents context.ranges node.parents
    pure (.hull node.parents, enclosure)


/-- Max pooling without padding selects existing values and therefore preserves the input hull. -/
def maxPoolContract : GraphRangeContract where
  key := .maxPool
  name := "max-pool value preservation"
  derive := fun context node =>
    match node.parents with
    | [parent] => do
        let enclosure <- parentRange context.ranges parent
        pure (.preserve parent, enclosure)
    | _ => throw (arityError "max pool" node "one parent")

/-- Padded max pooling may additionally select the padding value zero. -/
def maxPoolPadContract : GraphRangeContract where
  key := .maxPoolPad
  name := "padded max-pool hull"
  derive := fun context node =>
    match node.parents with
    | [parent] => do
        let input <- parentRange context.ranges parent
        let enclosure := IEEE32Exec.Interval32.hull input
          (IEEE32Exec.Interval32.point IEEE32Exec.posZero)
        pure (.hullZero parent, enclosure)
    | _ => throw (arityError "padded max pool" node "one parent")

/-- Shared average-pooling contract constructor. -/
def averagePoolContract (padded : Bool) : GraphRangeContract where
  key := if padded then .averagePoolPad else .averagePool
  name := if padded then "padded average-pool fixed-left transfer"
    else "average-pool fixed-left transfer"
  derive := fun context node =>
    match node.kind, node.parents with
    | .avgPool2d kH kW _, [parent] => do
        requireFixedLeftReduction context.plan node
        let input <- parentRange context.ranges parent
        let count := kH * kW
        if count = 0 then
          throw s!"numerical certificate: node {node.id} has an empty average-pooling window"
        pure (.averageWindowLeft parent count false, meanLeftRange count input)
    | .avgPool2dPad kH kW _ _, [parent] => do
        requireFixedLeftReduction context.plan node
        let input <- parentRange context.ranges parent
        let count := kH * kW
        if count = 0 then
          throw s!"numerical certificate: node {node.id} has an empty average-pooling window"
        let paddedInput := IEEE32Exec.Interval32.hull input
          (IEEE32Exec.Interval32.point IEEE32Exec.posZero)
        pure (.averageWindowLeft parent count true, meanLeftRange count paddedInput)
    | _, _ => throw (arityError "average pool" node "one parent")

/-- Reciprocal contract with an explicit nonzero-domain check. -/
def inverseContract : GraphRangeContract where
  key := .backend .inv
  name := "reciprocal"
  derive := fun context node =>
    match node.parents with
    | [parent] => do
        let enclosure <- parentRange context.ranges parent
        if enclosure.containsZero then
          throw s!"numerical certificate: node {node.id} reciprocal range contains zero"
        pure (.inv parent, IEEE32Exec.Interval32.inv enclosure)
    | _ => throw (arityError "reciprocal" node "one parent")

/-- Whole-tensor fixed-left sum contract. -/
def sumContract : GraphRangeContract where
  key := .wholeSum
  name := "fixed-left sum"
  derive := fun context node =>
    match node.parents with
    | [parent] => do
        requireFixedLeftReduction context.plan node
        let input <- parentNodeRange context.ranges parent
        let count := input.outShape.size
        pure (.sumLeft parent count, sumLeftRange count input.enclosure)
    | _ => throw (arityError "sum" node "one parent")

/-- Axis reduction contract shared by sum and mean. -/
def axisReductionContract (mean : Bool) : GraphRangeContract where
  key := .backend (if mean then .reduceMean else .reduceSum)
  name := if mean then "fixed-left axis mean" else "fixed-left axis sum"
  derive := fun context node =>
    match node.kind, node.parents with
    | .reduceSum axis, [parent] => do
        requireFixedLeftReduction context.plan node
        let input <- parentNodeRange context.ranges parent
        let count <- match input.outShape.getDim axis with
          | some count => pure count
          | none => throw s!"numerical certificate: node {node.id} has invalid reduction axis {axis}"
        pure (.sumLeft parent count, sumLeftRange count input.enclosure)
    | .reduceMean axis, [parent] => do
        requireFixedLeftReduction context.plan node
        let input <- parentNodeRange context.ranges parent
        let count <- match input.outShape.getDim axis with
          | some count => pure count
          | none => throw s!"numerical certificate: node {node.id} has invalid reduction axis {axis}"
        if count = 0 then
          throw s!"numerical certificate: node {node.id} cannot certify a mean over an empty axis"
        pure (.meanLeft parent count, meanLeftRange count input.enclosure)
    | _, _ => throw (arityError "axis reduction" node "one parent")

/-- Matrix multiplication contract using the selected fixed-left accumulation schedule. -/
def matmulContract : GraphRangeContract where
  key := .backend .matmul
  name := "fixed-left matrix multiplication"
  derive := fun context node =>
    match node.parents with
    | [left, right] => do
        requireFixedLeftReduction context.plan node
        let a <- parentNodeRange context.ranges left
        let b <- parentNodeRange context.ranges right
        let innerDim <- matmulInnerDim a.outShape b.outShape
        let product := IEEE32Exec.Interval32.mul a.enclosure b.enclosure
        pure (.matmulLeft left right innerDim, sumLeftRange innerDim product)
    | _ => throw (arityError "matrix multiplication" node "two parents")

/-- Mean-squared-error contract with nonnegativity restored after dependent squaring. -/
def mseContract : GraphRangeContract where
  key := .backend .mseLoss
  name := "mean squared error"
  derive := fun context node =>
    match node.parents with
    | [prediction, target] => do
        requireFixedLeftReduction context.plan node
        let y <- parentNodeRange context.ranges prediction
        let t <- parentNodeRange context.ranges target
        if y.outShape != t.outShape then
          throw s!"numerical certificate: node {node.id} MSE parents have different shapes"
        let residual := IEEE32Exec.Interval32.sub y.enclosure t.enclosure
        -- The two residual occurrences are dependent. Generic interval multiplication forgets
        -- that dependency; the proved ReLU transfer restores nonnegativity of the square.
        let squared := (IEEE32Exec.Interval32.mul residual residual).relu
        let count := y.outShape.size
        let enclosure :=
          if count = 0 then IEEE32Exec.Interval32.point IEEE32Exec.posZero
          else meanLeftRange count squared
        pure (.mseLeft prediction target count, enclosure)
    | _ => throw (arityError "mean squared error" node "two parents")

/-- Pure LayerNorm contract over an arbitrary normalized suffix. -/
def layerNormContract : GraphRangeContract where
  key := .backend .layerNorm
  name := "layer normalization"
  derive := fun context node =>
    match node.kind, node.parents with
    | .layernorm axis, [parent] => do
        requireFixedLeftReduction context.plan node
        let input <- parentNodeRange context.ranges parent
        let (_, normalizedSize) <-
          match OpContracts.layerNorm2DParams axis input.outShape with
          | .ok dims => pure dims
          | .error message =>
              throw s!"numerical certificate: node {node.id} layernorm: {message}"
        if normalizedSize = 0 then
          throw s!"numerical certificate: node {node.id} cannot normalize an empty suffix"
        let mean := meanLeftRange normalizedSize input.enclosure
        let centered := IEEE32Exec.Interval32.sub input.enclosure mean
        let squared := (IEEE32Exec.Interval32.mul centered centered).relu
        let variance := meanLeftRange normalizedSize squared
        let epsilon : IEEE32Exec := Numbers.epsilon
        let stabilized := IEEE32Exec.Interval32.add variance
          (IEEE32Exec.Interval32.point epsilon)
        if nonnegativeEndpoint stabilized.lo && nonnegativeEndpoint stabilized.hi then
          let denominator := stabilized.sqrt
          if denominator.containsZero then
            throw s!"numerical certificate: node {node.id} layernorm denominator may be zero"
          pure (.layerNormLeft parent axis normalizedSize,
            IEEE32Exec.Interval32.div centered denominator)
        else
          throw s!"numerical certificate: node {node.id} layernorm variance range became negative"
    | _, _ => throw (arityError "layer normalization" node "one parent")

/-- Softmax contract: exact-real outputs lie in the unit interval on every nonempty axis. -/
def softmaxContract : GraphRangeContract where
  key := .backend .softmax
  name := "softmax unit interval"
  derive := fun context node =>
    match node.kind, node.parents with
    | .softmax axis, [parent] => do
        let input <- parentNodeRange context.ranges parent
        let count <- match input.outShape.getDim axis with
          | some count => pure count
          | none => throw s!"numerical certificate: node {node.id} has invalid softmax axis {axis}"
        if count = 0 then
          throw s!"numerical certificate: node {node.id} cannot certify softmax on an empty axis"
        pure (.softmaxUnit parent axis, unitInterval)
    | _, _ => throw (arityError "softmax" node "one parent")

/-- ReLU interval contract. -/
def reluContract : GraphRangeContract where
  key := .backend .relu
  name := "ReLU"
  derive := fun context node =>
    match node.parents with
    | [parent] => do
        let input <- parentRange context.ranges parent
        pure (.relu parent, input.relu)
    | _ => throw (arityError "ReLU" node "one parent")

/-- Absolute-value interval contract. -/
def absContract : GraphRangeContract where
  key := .backend .abs
  name := "absolute value"
  derive := fun context node =>
    match node.parents with
    | [parent] => do
        let input <- parentRange context.ranges parent
        pure (.abs parent, input.abs)
    | _ => throw (arityError "absolute value" node "one parent")

/-- Square-root contract with a checked nonnegative domain. -/
def sqrtContract : GraphRangeContract where
  key := .backend .sqrt
  name := "square root"
  derive := fun context node =>
    match node.parents with
    | [parent] => do
        let input <- parentRange context.ranges parent
        if nonnegativeEndpoint input.lo && nonnegativeEndpoint input.hi then
          pure (.sqrtNonnegative parent, input.sqrt)
        else
          throw s!"numerical certificate: node {node.id} square-root range contains negative values"
    | _ => throw (arityError "square root" node "one parent")

/-- Constructor for bounded transcendental contracts. -/
def boundedUnaryContract (op : BackendOp) (rule : Nat -> RangeRule)
    (enclosure : IEEE32Exec.Interval32) : GraphRangeContract where
  key := .backend op
  name := s!"{op.name} codomain"
  derive := fun _ node =>
    match node.parents with
    | [parent] => pure (rule parent, enclosure)
    | _ => throw (arityError op.name node "one parent")

/-- Built-in numerical contracts. Grouping is by operation semantics, never by architecture. -/
def defaultContracts : List GraphRangeContract :=
  [ sourceContract
  , structuralContract
  , preserveContract .permute
  , preserveContract .broadcast
  , preserveContract .reshape
  , maxPoolContract
  , maxPoolPadContract
  , averagePoolContract false
  , averagePoolContract true
  , binaryContract .add .add IEEE32Exec.Interval32.add
  , binaryContract .sub .sub IEEE32Exec.Interval32.sub
  , binaryContract .mul .mul IEEE32Exec.Interval32.mul
  , inverseContract
  , hullContract .max
  , hullContract .min
  , hullContract .concat
  , sumContract
  , axisReductionContract false
  , axisReductionContract true
  , matmulContract
  , mseContract
  , layerNormContract
  , softmaxContract
  , reluContract
  , absContract
  , sqrtContract
  , boundedUnaryContract .sigmoid .unitBound unitInterval
  , boundedUnaryContract .tanh .signedUnitBound signedUnitInterval
  , boundedUnaryContract .sin .signedUnitBound signedUnitInterval
  , boundedUnaryContract .cos .signedUnitBound signedUnitInterval
  ]

/-- TorchLean's built-in numerical registry. Construction is checked once at use sites so a future
duplicate produces an explicit configuration failure. -/
def defaultRegistry : Except String GraphRangeRegistry :=
  GraphRangeRegistry.ofList "torchlean.graph-numerics.v1" defaultContracts

/-- Compute one node range using an explicit numerical contract registry. -/
def deriveNodeRangeWith (registry : GraphRangeRegistry)
    (sources : Array CheckedSourceRange) (plan : AcceptedGraphPlan)
    (ranges : Array CheckedNodeRange) (node : Node) : Except String RangeTransferResult := do
  let key := numericalOpKey node.kind
  let contract <- match registry.find? key with
    | some contract => pure contract
    | none =>
        throw s!"numerical certificate: node {node.id} ({node.kind.describe}) has no registered numerical contract"
  contract.derive { sources, plan, ranges } node

/-- Compute one node range using TorchLean's built-in registry. -/
def deriveNodeRange (sources : Array CheckedSourceRange) (plan : AcceptedGraphPlan)
    (ranges : Array CheckedNodeRange) (node : Node) : Except String RangeTransferResult := do
  let registry <- defaultRegistry
  deriveNodeRangeWith registry sources plan ranges node

/-- Construct and validate the canonical range trace using an explicit contract registry. -/
def buildRangeTraceWith (registry : GraphRangeRegistry)
    (graph : Graph) (sources : Array CheckedSourceRange)
    (plan : AcceptedGraphPlan) :
    Except String (Array CheckedNodeRange) := do
  graph.checkWellFormed
  let _ <- requireNumericalCoverage registry graph
  checkSourceOwnership graph sources
  let mut ranges : Array CheckedNodeRange := #[]
  for node in graph.nodes do
    let (rule, enclosure) <- deriveNodeRangeWith registry sources plan ranges node
    if h : validInterval enclosure then
      ranges := ranges.push
        { nodeId := node.id
          outShape := node.outShape
          rule
          enclosure
          valid := (validInterval_eq_true_iff enclosure).mp h }
    else
      throw s!"numerical certificate: node {node.id} ({node.kind.describe}) produced a non-finite or unordered enclosure"
  pure ranges

/-- Construct and validate the canonical range trace using TorchLean's built-in contracts. -/
def buildRangeTrace (graph : Graph) (sources : Array CheckedSourceRange)
    (plan : AcceptedGraphPlan) : Except String (Array CheckedNodeRange) := do
  let registry <- defaultRegistry
  buildRangeTraceWith registry graph sources plan

/-- Erase validity proofs from a checked trace. -/
def eraseRangeTrace (ranges : Array CheckedNodeRange) : Array NodeRange :=
  ranges.map (fun range => range.toNodeRange)

/-- Compare a canonical checked trace with untrusted raw rows. -/
def sameRangeTrace (checked : Array CheckedNodeRange) (raw : Array NodeRange) : Bool :=
  checked.size == raw.size &&
    (List.finRange checked.size).all (fun i =>
      match checked[i]?, raw[i]? with
      | some expected, some claimed => sameNodeRange expected claimed
      | _, _ => false)

/-! ## Backend-linked graph certificates -/

/-- Untrusted certificate data.

The audit field is a data-only snapshot: theorem and checker proof terms remain in TorchLean and
are reconstructed by replanning the graph under `profileName`. This separation makes the artifact
portable without allowing it to manufacture backend evidence.
-/
structure GraphNumericalCertificate where
  profileName : String
  registryName : String
  sources : Array SourceRange
  ranges : Array NodeRange
  audit : ExecutionAuditSnapshot

instance : Repr GraphNumericalCertificate where
  reprPrec certificate _ := Std.Format.text <|
    s!"GraphNumericalCertificate(profile={certificate.profileName}, " ++
      s!"registry={certificate.registryName}, " ++
      s!"sources={certificate.sources.size}, ranges={certificate.ranges.size}, " ++
      s!"capsules={repr certificate.audit.capsuleNames})"

/-- Proof-carrying result returned by `check`. Raw endpoint data has been replaced by the canonical
trace reconstructed from the graph, and `backendPlan` contains the acceptance-gate proof. -/
structure CheckedCertificate where
  /-- The exact graph whose ranges and backend plan were reconstructed by the checker. -/
  graph : Graph
  raw : GraphNumericalCertificate
  sources : Array CheckedSourceRange
  ranges : Array CheckedNodeRange
  backendPlan : AcceptedGraphPlan
  rangesMatch : sameRangeTrace ranges raw.ranges = true
  auditMatch : backendPlan.audit.snapshot = raw.audit

instance : Repr CheckedCertificate where
  reprPrec certificate _ := repr certificate.raw

/-- Result of executing the canonical IR with bit-level binary32 semantics and replaying every
intermediate value against a checked numerical certificate. -/
structure CheckedExecution where
  certificate : CheckedCertificate
  values : Array (NN.IR.DVal IEEE32Exec)
  withinRanges : executionWithinRanges certificate.ranges values = true

/-- Convert an accepted backend plan and checked range trace into raw certificate data. -/
def toRaw (profile : BackendProfile) (registry : GraphRangeRegistry)
    (sources : Array SourceRange)
    (ranges : Array CheckedNodeRange) (plan : AcceptedGraphPlan) : GraphNumericalCertificate :=
  { profileName := profile.name
    registryName := registry.name
    sources
    ranges := eraseRangeTrace ranges
    audit := plan.audit.snapshot }

/-- Obtain an accepted backend plan or report the acceptance-gate failures. -/
def acceptedPlan (profile : BackendProfile) (graph : Graph) : Except String AcceptedGraphPlan := do
  match <- profile.acceptGraph graph with
  | .accepted plan => pure plan
  | .rejected _ failures =>
      throw s!"numerical certificate: backend profile {profile.name} rejected the graph: {repr failures}"

/-- Generate a canonical certificate using an explicit numerical operation registry. -/
def generateWith (registry : GraphRangeRegistry) (profile : BackendProfile)
    (graph : Graph) (sources : Array SourceRange) :
    Except String GraphNumericalCertificate := do
  let checkedSources <- checkSources sources
  let plan <- acceptedPlan profile graph
  let ranges <- buildRangeTraceWith registry graph checkedSources plan
  pure (toRaw profile registry sources ranges plan)

/-- Generate a canonical certificate using TorchLean's built-in numerical contracts. -/
def generate (profile : BackendProfile) (graph : Graph) (sources : Array SourceRange) :
    Except String GraphNumericalCertificate := do
  let registry <- defaultRegistry
  generateWith registry profile graph sources

/-- Check an untrusted certificate with an explicit numerical operation registry. -/
def checkWith (registry : GraphRangeRegistry) (profile : BackendProfile)
    (graph : Graph) (raw : GraphNumericalCertificate) :
    Except String CheckedCertificate := do
  if raw.profileName != profile.name then
    throw s!"numerical certificate: profile mismatch; artifact names {raw.profileName}, checker uses {profile.name}"
  if raw.registryName != registry.name then
    throw s!"numerical certificate: registry mismatch; artifact names {raw.registryName}, checker uses {registry.name}"
  let sources <- checkSources raw.sources
  let plan <- acceptedPlan profile graph
  let ranges <- buildRangeTraceWith registry graph sources plan
  if hRanges : sameRangeTrace ranges raw.ranges then
    if hAudit : plan.audit.snapshot = raw.audit then
      pure
        { graph
          raw
          sources
          ranges
          backendPlan := plan
          rangesMatch := hRanges
          auditMatch := hAudit }
    else
      throw "numerical certificate: backend audit differs from the plan selected by the checker"
  else
    throw "numerical certificate: node ranges differ from canonical outward-rounded propagation"

/-- Check an untrusted certificate using TorchLean's built-in numerical contracts. -/
def check (profile : BackendProfile) (graph : Graph) (raw : GraphNumericalCertificate) :
    Except String CheckedCertificate := do
  let registry <- defaultRegistry
  checkWith registry profile graph raw

/-- Generate and immediately check a certificate. This is convenient for in-process callers and
ensures examples exercise exactly the same checker used for imported artifacts. -/
def generateChecked (profile : BackendProfile) (graph : Graph) (sources : Array SourceRange) :
    Except String CheckedCertificate := do
  let raw <- generate profile graph sources
  check profile graph raw

/-- Generate and immediately check with one explicit registry. -/
def generateCheckedWith (registry : GraphRangeRegistry) (profile : BackendProfile)
    (graph : Graph) (sources : Array SourceRange) : Except String CheckedCertificate := do
  let raw <- generateWith registry profile graph sources
  checkWith registry profile graph raw

/-- Execute a graph under `IEEE32Exec` and check all intermediate tensors against the certificate.

This reference replay path gives imported runtime artifacts a bit-level oracle. The backend audit
records the capsules and numerical policies
selected when the graph is replanned. The audit is not runtime provenance and does not prove that
those kernels produced the imported values.
-/
def executeIEEE32 (payload : NN.IR.Payload IEEE32Exec)
    (input : NN.IR.DVal IEEE32Exec) (certificate : CheckedCertificate) :
    Except String CheckedExecution := do
  let values <- certificate.graph.denoteAll payload input
  if h : executionWithinRanges certificate.ranges values then
    pure { certificate, values, withinRanges := h }
  else
    throw "numerical certificate: IEEE32 graph replay produced a value outside the certified range"

/-- Exact-real execution evidence for the graph stored in a checked certificate.

The numerical checker reconstructs interval transfers, while a semantic proof establishes that the
real graph trace lies in those intervals. Keeping this proof separate prevents successful endpoint
replay from being mistaken for a theorem about an unsupported real operation. -/
structure CheckedRealExecution (certificate : CheckedCertificate) where
  payload : NN.IR.Payload Real
  input : NN.IR.DVal Real
  values : Array (NN.IR.DVal Real)
  denotation : certificate.graph.denoteAll payload input = .ok values
  enclosed : List.Forall₂ RealDValEnclosed certificate.ranges.toList values.toList

namespace CheckedExecution

/-- Pair a checked IEEE replay with a proved real enclosure trace to obtain a graph-wide,
pointwise error trace. Each node's error budget is the width of its checked outward interval. -/
theorem errorTrace (execution : CheckedExecution)
    (exact : CheckedRealExecution execution.certificate) :
    ExecutionErrorTrace execution.certificate.ranges.toList exact.values.toList
      execution.values.toList :=
  execution_error_trace_of_check exact.enclosed execution.withinRanges

end CheckedExecution

end NumericalCertificate
end RuntimeApprox
end Proofs

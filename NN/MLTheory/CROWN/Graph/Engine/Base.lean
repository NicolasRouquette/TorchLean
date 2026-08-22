/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Core
public import NN.IR.Payload
public import NN.Spec.Core.Shape
public import NN.Spec.Core.Tensor.Packed

/-!
Shared definitions for the graph CROWN engine.

This file contains the flat vector representation, parameter stores, interval boxes, shape
permutation helpers, and tensor casts used by the IBP, derivative, affine, CROWN, and backward
objective passes.
-/

public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN
open NN.IR

variable {α : Type} [Context α]
variable [BoundOps α]

open BoundOps

/--
Flat vector pack: a tensor paired with its flattened dimension.

This is used for constant payloads and objective coefficient vectors in the flat LiRPA engine.
-/
structure FlatVec (α : Type) [Context α] where
  /-- Vector dimension. -/
  n : Nat
  /-- Vector payload (shape `.dim n .scalar`). -/
  v : Tensor α (.dim n .scalar)

-- The flat-vector engine is the canonical executable path for the current graph verifier.

/--
Parameters for a linear layer `y = W*x + b` in flattened form.

`m` is the output dimension and `n` is the input dimension.
-/
structure LinParams (α : Type) [Context α] where
  /-- Output dimension. -/
  m : Nat
  /-- Input dimension. -/
  n : Nat
  /-- Weight matrix `W` (shape `m × n`). -/
  w : Tensor α (.dim m (.dim n .scalar))
  /-- Bias vector `b` (shape `m`). -/
  b : Tensor α (.dim m .scalar)

/-- Matrix parameters for bias-free matmul: y = W x. -/
structure MatParams (α : Type) [Context α] where
  /-- Output dimension. -/
  m : Nat
  /-- Input dimension. -/
  n : Nat
  /-- Weight matrix `W` (shape `m × n`). -/
  w : Tensor α (.dim m (.dim n .scalar))

/-- Channel index for a flattened `N×C×H×W` tensor in row-major order. -/
def nchwChannelOfFlat (c h w idx : Nat) : Nat :=
  if h * w = 0 then
    0
  else
    (idx / (h * w)) % c

/-- Eval BatchNorm scale for one channel. -/
def batchNorm2dNchwEvalScale (cfg : NN.IR.BatchNorm2dNchwEvalParams α) (ci : Fin cfg.c) : α :=
  match getAtSpec cfg.gamma ci, getAtSpec cfg.var ci with
  | .scalar gamma, .scalar var =>
      gamma / MathFunctions.sqrt (max var Numbers.zero + cfg.eps)

/-- Eval BatchNorm bias for one channel after folding running statistics into an affine map. -/
def batchNorm2dNchwEvalBias (cfg : NN.IR.BatchNorm2dNchwEvalParams α) (ci : Fin cfg.c) : α :=
  match getAtSpec cfg.beta ci, getAtSpec cfg.mean ci with
  | .scalar beta, .scalar mean =>
      beta - mean * batchNorm2dNchwEvalScale (α := α) cfg ci

/--
Build the exact diagonal affine form for eval-mode BatchNorm2d over an `N×C×H×W` tensor.

The IR stores the channel parameters in the node payload. The spatial dimensions come from the
checked parent shape, so malformed shapes produce no verifier transfer rule.
-/
def batchNorm2dNchwEvalLinear? (parentShape : Shape)
    (cfg : NN.IR.BatchNorm2dNchwEvalParams α) : Option (LinParams α) :=
  match parentShape with
  | .dim _n (.dim c (.dim h (.dim w .scalar))) =>
      if hcfg : cfg.c = 0 then
        none
      else if c = cfg.c then
        haveI : NeZero cfg.c := ⟨hcfg⟩
        let outDim := parentShape.size
        let weight : Tensor α (.dim outDim (.dim outDim .scalar)) :=
          Tensor.dim (fun oi =>
            Tensor.dim (fun ii =>
              let ch := nchwChannelOfFlat cfg.c h w oi.val
              let scale := batchNorm2dNchwEvalScale (α := α) cfg (Fin.ofNat cfg.c ch)
              Tensor.scalar (if decide (oi.val = ii.val) then scale else Numbers.zero)))
        let bias : Tensor α (.dim outDim .scalar) :=
          Tensor.dim (fun oi =>
            let ch := nchwChannelOfFlat cfg.c h w oi.val
            Tensor.scalar (batchNorm2dNchwEvalBias (α := α) cfg (Fin.ofNat cfg.c ch)))
        some { m := outDim, n := outDim, w := weight, b := bias }
      else
        none
  | _ => none

/--
Parameters keyed by node id (weights, biases, constants, and seeded input boxes).

This is kept compact: it is the graph interpreter used to run IBP/CROWN on a pure `Graph`
without pulling in a heavyweight runtime.
-/
structure ParamStore (α : Type) [Context α] where
  /-- Seed boxes for designated input nodes (`id -> FlatBox`). -/
  inputBoxes : Std.HashMap Nat (FlatBox α) := Std.HashMap.emptyWithCapacity
  /-- Constants (`id -> FlatVec`). -/
  constVals  : Std.HashMap Nat (FlatVec α) := Std.HashMap.emptyWithCapacity
  /-- Linear layer params (`id -> (W,b)`). -/
  linearWB   : Std.HashMap Nat (LinParams α) := Std.HashMap.emptyWithCapacity
  /-- Matmul params (`id -> W`) for bias-free multiplication. -/
  matmulW    : Std.HashMap Nat (MatParams α) := Std.HashMap.emptyWithCapacity
  /-- Conv2d specs (`id -> conv configuration`). -/
  conv2dCfg  : Std.HashMap Nat (NN.IR.Conv2dParams α) := Std.HashMap.emptyWithCapacity
  /-- Eval-mode BatchNorm2d parameters (`id -> gamma/beta/running stats`). -/
  batchNorm2dNchwEval : Std.HashMap Nat (NN.IR.BatchNorm2dNchwEvalParams α) :=
    Std.HashMap.emptyWithCapacity

namespace ParamStore

/-- Insert an input interval box for a graph node. -/
def seedInputBox {α : Type} [Context α]
    (ps : ParamStore α) (inputId : Nat) (xB : FlatBox α) : ParamStore α :=
  { ps with inputBoxes := ps.inputBoxes.insert inputId xB }

/-- Seed a graph input with a uniform `ℓ∞` box around a shaped tensor. -/
def seedLInfBall {α : Type} [Context α] {s : Shape}
    (ps : ParamStore α) (inputId : Nat) (center : Tensor α s) (eps : α) : ParamStore α :=
  ps.seedInputBox inputId <| FlatBox.lInfBall (α := α) center eps

end ParamStore

/-- Read a node's interval box from an IBP-style result array. -/
def outputBox? {α : Type} [Context α]
    (boxes : Array (Option (FlatBox α))) (outId : Nat) : Except String (FlatBox α) := do
  match boxes[outId]? with
  | some (some outB) => pure outB
  | some none => throw s!"output box missing at node {outId}"
  | none => throw s!"output node {outId} is out of bounds for {boxes.size} boxes"

/-- Default inhabitant for `FlatBox` (a 0-dimensional box at `0`). -/
instance : Inhabited (FlatBox α) where
  default := { dim := 0, lo := Spec.fill (α:=α) 0 (.dim 0 .scalar), hi := Spec.fill (α:=α) 0 (.dim 0
    .scalar) }

/-- Elementwise product of two FlatBoxes (interval product per component). Requires equal dims. -/
@[expose] public def boxMulElem (B1 B2 : FlatBox α) : Option (FlatBox α) :=
  match B1, B2 with
  | ⟨n1, l1, u1⟩, ⟨n2, l2, u2⟩ =>
    if h : n1 = n2 then
      by
        cases h
        let lo :=
          match l1, u1, l2, u2 with
          | .dim l1, .dim u1, .dim l2, .dim u2 =>
            Tensor.dim (fun i =>
              match l1 i, u1 i, l2 i, u2 i with
              | .scalar lx, .scalar ux, .scalar ly, .scalar uy =>
                let p1 := BoundOps.mulDown lx ly; let p2 := BoundOps.mulDown lx uy
                let p3 := BoundOps.mulDown ux ly; let p4 := BoundOps.mulDown ux uy
                let m1 := min2 p1 p2
                let m2 := min2 p3 p4
                Tensor.scalar (min2 m1 m2))
        let hi :=
          match l1, u1, l2, u2 with
          | .dim l1, .dim u1, .dim l2, .dim u2 =>
            Tensor.dim (fun i =>
              match l1 i, u1 i, l2 i, u2 i with
              | .scalar lx, .scalar ux, .scalar ly, .scalar uy =>
                let p1 := BoundOps.mulUp lx ly; let p2 := BoundOps.mulUp lx uy
                let p3 := BoundOps.mulUp ux ly; let p4 := BoundOps.mulUp ux uy
                let m1 := max2 p1 p2
                let m2 := max2 p3 p4
                Tensor.scalar (max2 m1 m2))
        exact some { dim := n1, lo := lo, hi := hi }
    else none

/-- Chain-rule multiplication for derivative intervals. Returns `none` on dimension mismatch. -/
def chainMul (dZ dF : FlatBox α) : Option (FlatBox α) :=
  boxMulElem (α:=α) dZ dF

/-- Convert a dependent `Box` of shape `.dim n .scalar` into a `FlatBox` with `dim := n`. -/
@[expose]
def toFlatBox (n : Nat) (B : Box α (.dim n .scalar)) : FlatBox α :=
  { dim := n, lo := B.lo, hi := B.hi }

/-- Convert a `FlatBox` to a dependent `Box` at shape `.dim B.dim .scalar`. -/
@[expose]
public def ofFlatBox (B : FlatBox α) : Box α (.dim B.dim .scalar) :=
  { lo := B.lo, hi := B.hi }

/-- Add two flat interval boxes coordinatewise; dimension mismatches preserve the left box. -/
@[expose]
public def boxAdd (B1 B2 : FlatBox α) : FlatBox α :=
  match B1 with
  | ⟨n1, lo1, hi1⟩ =>
    match B2 with
    | ⟨n2, lo2, hi2⟩ =>
      if h : n1 = n2 then
        by
          cases h
          exact
            { dim := n1
              lo := Tensor.map2Spec BoundOps.addDown lo1 lo2
              hi := Tensor.map2Spec BoundOps.addUp hi1 hi2 }
      else
        { dim := n1, lo := lo1, hi := hi1 }

/-- Interval subtraction on `FlatBox` endpoints (sound enclosure). -/
@[expose]
public def boxSub (B1 B2 : FlatBox α) : FlatBox α :=
  match B1 with
  | ⟨n1, lo1, hi1⟩ =>
    match B2 with
    | ⟨n2, lo2, hi2⟩ =>
      if h : n1 = n2 then
        by
          cases h
          -- Sound interval subtraction: [l1,u1] - [l2,u2] = [l1 - u2, u1 - l2]
          exact
            { dim := n1
              lo := Tensor.map2Spec BoundOps.subDown lo1 hi2
              hi := Tensor.map2Spec BoundOps.subUp hi1 lo2 }
      else
        { dim := n1, lo := lo1, hi := hi1 }

/-- Directed lower and upper sums of all coordinates in a flat box. -/
private def boxSumEndpoints (B : FlatBox α) : α × α :=
  let loValues := match B.lo with | .dim values => values
  let hiValues := match B.hi with | .dim values => values
  let lo := (List.finRange B.dim).foldl (fun acc i =>
    match loValues i with
    | .scalar x => BoundOps.addDown acc x) Numbers.zero
  let hi := (List.finRange B.dim).foldl (fun acc i =>
    match hiValues i with
    | .scalar x => BoundOps.addUp acc x) Numbers.zero
  (lo, hi)

/-- Sum all coordinates of a flat box with directed accumulation. -/
def boxSum (B : FlatBox α) : FlatBox α :=
  let (lo, hi) := boxSumEndpoints (α := α) B
  { dim := 1
    lo := Spec.fill (α := α) lo (.dim 1 .scalar)
    hi := Spec.fill (α := α) hi (.dim 1 .scalar) }

/-- Average all coordinates of a nonempty flat box with directed division. -/
def boxMean? [NonlinearBoundOps α] (B : FlatBox α) : Option (FlatBox α) := do
  if B.dim = 0 then
    none
  else
    let (lo, hi) := boxSumEndpoints (α := α) B
    let n : α := B.dim
    let (meanLo, meanHi) ← NonlinearBoundOps.divBounds lo hi n n
    pure
      { dim := 1
        lo := Spec.fill (α := α) meanLo (.dim 1 .scalar)
        hi := Spec.fill (α := α) meanHi (.dim 1 .scalar) }

/-- Apply ReLU to both endpoints of a `FlatBox` (monotone activation, so endpoints suffice). -/
@[expose]
public def boxRelu (B : FlatBox α) : FlatBox α :=
  { dim := B.dim
    lo := Tensor.mapSpec (fun x => Activation.Math.reluSpec (α := α) x) B.lo
    hi := Tensor.mapSpec (fun x => Activation.Math.reluSpec (α := α) x) B.hi }

/-- Componentwise absolute value bounds. Soundly encloses `abs` over each interval component. -/
def boxAbs (B : FlatBox α) : FlatBox α :=
  match B.lo, B.hi with
  | .dim lo, .dim hi =>
      let lo' :=
        Tensor.dim (fun i =>
          match lo i, hi i with
          | .scalar l, .scalar u =>
              let al := MathFunctions.abs l
              let au := MathFunctions.abs u
              let minAbs :=
                if l < Numbers.zero then
                  if Numbers.zero < u then Numbers.zero else (if al < au then al else au)
                else
                  if al < au then al else au
              Tensor.scalar minAbs)
      let hi' :=
        Tensor.dim (fun i =>
          match lo i, hi i with
          | .scalar l, .scalar u =>
              let al := MathFunctions.abs l
              let au := MathFunctions.abs u
              let maxAbs := if al > au then al else au
              Tensor.scalar maxAbs)
      { dim := B.dim, lo := lo', hi := hi' }

namespace boxUnaryEnclosure

/-- Traverse a finite family without converting its index to an untyped list. -/
private def traverseFin {β : Type} {n : Nat} (f : Fin n → Option β) : Option (Fin n → β) :=
  if h : ∀ i, (f i).isSome then
    some fun i => (f i).get (h i)
  else
    none

private theorem traverseFin_eq_some_iff {β : Type} {n : Nat}
    {f : Fin n → Option β} {g : Fin n → β} :
    traverseFin f = some g ↔ ∀ i, f i = some (g i) := by
  unfold traverseFin
  split_ifs with h
  · constructor
    · intro hfg i
      have : (fun i => (f i).get (h i)) = g := Option.some.inj hfg
      rw [← this]
      exact (Option.some_get (h i)).symm
    · intro hfg
      congr
      funext i
      obtain ⟨_, hi⟩ := Option.eq_some_iff_get_eq.mp (hfg i)
      exact hi
  · constructor
    · simp
    · intro hfg
      exfalso
      apply h
      intro i
      simp [hfg i]

end boxUnaryEnclosure

/-- Apply a scalar interval enclosure coordinatewise to a flat box. -/
def boxUnaryEnclosure? [NonlinearBoundOps α]
    (enclose : α → α → Option (α × α)) (B : FlatBox α) : Option (FlatBox α) := do
  let lo := match B.lo with | .dim values => values
  let hi := match B.hi with | .dim values => values
  let bounds ← boxUnaryEnclosure.traverseFin fun i =>
    match lo i, hi i with
    | .scalar l, .scalar u => enclose l u
  let lower : Tensor α (.dim B.dim .scalar) :=
    Tensor.dim fun i => Tensor.scalar (bounds i).1
  let upper : Tensor α (.dim B.dim .scalar) :=
    Tensor.dim fun i => Tensor.scalar (bounds i).2
  pure { dim := B.dim, lo := lower, hi := upper }

/--
The real value represented by each coordinate of `x` lies between the interpreted endpoints of
`B`. This is the semantic relation used to connect executable endpoint arithmetic to the real graph
semantics.
-/
def EnclosesReal [LawfulBoundOps α]
    (B : FlatBox α) (x : Tensor ℝ (.dim B.dim .scalar)) : Prop :=
  ∀ i,
    LawfulBoundOps.toReal (FlatBox.getScalar B.lo i) ≤ FlatBox.getScalar x i ∧
      FlatBox.getScalar x i ≤ LawfulBoundOps.toReal (FlatBox.getScalar B.hi i)

/-- Dimension-aware enclosure of a real vector by a backend box. -/
def EnclosesRealValue [LawfulBoundOps α] {n : Nat}
    (B : FlatBox α) (x : Tensor ℝ (.dim n .scalar)) : Prop :=
  ∃ h : B.dim = n, EnclosesReal B (h.symm ▸ x)

/-- A lawful scalar transfer remains sound when applied coordinatewise to a flat graph box. -/
theorem boxUnaryEnclosure?_enclosesReal [LawfulBoundOps α] [NonlinearBoundOps α]
    (f : ℝ → ℝ) (enclose : α → α → Option (α × α))
    (henclose : UnaryEnclosure (α := α) f enclose) (B : FlatBox α)
    (x : Tensor ℝ (.dim B.dim .scalar)) (hx : EnclosesReal B x)
    {out : FlatBox α} (hout : boxUnaryEnclosure? (α := α) enclose B = some out) :
    EnclosesRealValue out (Tensor.mapSpec f x) := by
  cases hlo : B.lo with
  | dim lo =>
    cases hhi : B.hi with
    | dim hi =>
      cases hxv : x with
      | dim xv =>
        simp only [boxUnaryEnclosure?, hlo, hhi] at hout
        obtain ⟨bounds, hbounds, hout⟩ := Option.bind_eq_some_iff.mp hout
        have hpoint := boxUnaryEnclosure.traverseFin_eq_some_iff.mp hbounds
        have houtEq :
            out =
              { dim := B.dim
                lo := Tensor.dim fun i => Tensor.scalar (bounds i).1
                hi := Tensor.dim fun i => Tensor.scalar (bounds i).2 } := by
          exact (Option.some.inj hout).symm
        subst out
        refine ⟨rfl, ?_⟩
        intro i
        cases hloi : lo i with
        | scalar l =>
          cases hhii : hi i with
          | scalar u =>
            cases hxvi : xv i with
            | scalar v =>
              have hx_i := hx i
              have htransfer : enclose l u = some (bounds i) := by
                simpa [hloi, hhii] using hpoint i
              have hscalar := henclose htransfer
                (by simpa [EnclosesReal, FlatBox.getScalar, hlo, hhi, hxv, hloi, hhii, hxvi]
                  using hx_i.1)
                (by simpa [EnclosesReal, FlatBox.getScalar, hlo, hhi, hxv, hloi, hhii, hxvi]
                  using hx_i.2)
              simpa [EnclosesReal, FlatBox.getScalar, Tensor.mapSpec, hxv, hxvi] using hscalar

/-- Componentwise square-root enclosure supplied by the scalar backend. -/
def boxSqrt? [NonlinearBoundOps α] (B : FlatBox α) : Option (FlatBox α) :=
  boxUnaryEnclosure? (α := α) NonlinearBoundOps.sqrtBounds B

/-- Componentwise reciprocal bounds, failing when an input coordinate interval crosses zero. -/
@[expose]
def boxInv? [NonlinearBoundOps α] (B : FlatBox α) : Option (FlatBox α) :=
  boxUnaryEnclosure? (α := α)
    (fun lo hi => NonlinearBoundOps.divBounds Numbers.one Numbers.one lo hi) B

/-- Derivative range for `exp`; `exp' = exp`. -/
def derivBoxExp? [NonlinearBoundOps α] (zB : FlatBox α) : Option (FlatBox α) :=
  boxUnaryEnclosure? (α := α) NonlinearBoundOps.expBounds zB

/-- Derivative range for `log`; `log' x = 1/x` on a strictly positive interval. -/
def derivBoxLog? [NonlinearBoundOps α] (zB : FlatBox α) : Option (FlatBox α) :=
  boxUnaryEnclosure? (α := α)
    (fun lo hi =>
      if lo > Numbers.zero then
        NonlinearBoundOps.divBounds Numbers.one Numbers.one lo hi
      else
        none) zB

/-- Second-derivative range for `log`; `log'' x = -1/x²` on a positive interval. -/
def secondDerivBoxLog? [NonlinearBoundOps α] (zB : FlatBox α) : Option (FlatBox α) :=
  boxUnaryEnclosure? (α := α)
    (fun lo hi =>
      if lo > Numbers.zero then do
        let squareLo := BoundOps.mulDown lo lo
        let squareHi := BoundOps.mulUp hi hi
        let reciprocal ←
          NonlinearBoundOps.divBounds Numbers.one Numbers.one squareLo squareHi
        pure (-reciprocal.2, -reciprocal.1)
      else
        none) zB

/-- Negate an interval box by swapping and negating its endpoints. -/
def boxNeg (B : FlatBox α) : FlatBox α :=
  { dim := B.dim
    lo := Tensor.mapSpec (fun x => -x) B.hi
    hi := Tensor.mapSpec (fun x => -x) B.lo }

/-- Decompose an axis permutation into adjacent swaps, rejecting invalid permutations. -/
def swapDepthsForPerm? (perm : List Nat) (r : Nat) : Option (List Nat) :=
  let rec bubbleLeft (cur : List Nat) (swapsRev : List Nat) (i j : Nat) : List Nat × List Nat :=
    if j ≤ i then
      (cur, swapsRev)
    else
      bubbleLeft (Spec.Shape.swapAdjacentAxes cur (j - 1)) ((j - 1) :: swapsRev) i (j - 1)
  if perm.length = r && perm.all (fun d => d < r) then
    let rec go (i : Nat) (targets : List Nat) (cur : List Nat) (swapsRev : List Nat) :
        Option (List Nat) :=
      match targets with
      | [] => some swapsRev.reverse
      | target :: targets' =>
          match cur.findIdx? (· == target) with
          | none => none
          | some j =>
              let (cur', swapsRev') := bubbleLeft cur swapsRev i j
              go (i + 1) targets' cur' swapsRev'
    go 0 perm (List.range r) []
  else
    none

/-- Apply a full axis permutation to a shape-tagged tensor when the permutation is valid. -/
def permutePackedTensor? {α : Type} [Context α]
    (v : Spec.PackedTensor α) (perm : List Nat) : Option (Spec.PackedTensor α) :=
  let sIn := v.shape
  match Spec.Shape.permute? sIn perm with
  | none => none
  | some _ =>
      match swapDepthsForPerm? perm (Spec.Shape.rank sIn) with
      | none => none
      | some swaps =>
          some <| swaps.foldl (fun acc d => Spec.PackedTensor.swapAdjacentAtDepth acc d) v

/-- Componentwise max bounds: `max(x,y)` over interval boxes. -/
def boxMaxElem (B1 B2 : FlatBox α) : FlatBox α :=
  match B1, B2 with
  | ⟨n1, lo1, hi1⟩, ⟨n2, lo2, hi2⟩ =>
      if h : n1 = n2 then
        by
          cases h
          exact { dim := n1
                  lo := Tensor.maxSpec (α := α) lo1 lo2
                  hi := Tensor.maxSpec (α := α) hi1 hi2 }
      else
        { dim := n1, lo := lo1, hi := hi1 }

/-- Componentwise min bounds: `min(x,y)` over interval boxes. -/
def boxMinElem (B1 B2 : FlatBox α) : FlatBox α :=
  match B1, B2 with
  | ⟨n1, lo1, hi1⟩, ⟨n2, lo2, hi2⟩ =>
      if h : n1 = n2 then
        by
          cases h
          exact { dim := n1
                  lo := Tensor.minSpec (α := α) lo1 lo2
                  hi := Tensor.minSpec (α := α) hi1 hi2 }
      else
        { dim := n1, lo := lo1, hi := hi1 }

/--
Componentwise square of an interval box: for each component `[l,u]` produce `[min (l^2,u^2), max
  (l^2,u^2)]`, with `0` as the minimum when the interval crosses `0`.

The body is exposed because the proof layer theorem module unfolds this executable rule when
proving dimension preservation and pointwise enclosure.
-/
@[expose] def boxSquare (B : FlatBox α) : FlatBox α :=
  let loF : Fin B.dim → Tensor α .scalar :=
    match B.lo with
    | .dim f => f
  let hiF : Fin B.dim → Tensor α .scalar :=
    match B.hi with
    | .dim f => f
  let lo' :=
    Tensor.dim (fun i =>
      match loF i, hiF i with
      | .scalar l, .scalar u =>
        let l2 := l * l
        let u2 := u * u
        let minSq :=
          if l < Numbers.zero then
            if Numbers.zero < u then Numbers.zero else (if l2 < u2 then l2 else u2)
          else (if l2 < u2 then l2 else u2)
        Tensor.scalar minSq)
  let hi' :=
    Tensor.dim (fun i =>
      match loF i, hiF i with
      | .scalar l, .scalar u =>
        let l2 := l * l
        let u2 := u * u
        let maxSq := if l2 > u2 then l2 else u2
        Tensor.scalar maxSq)
  { dim := B.dim, lo := lo', hi := hi' }

/-- Interval multiplication for scalar endpoints: given `[aLo,aHi]` and `[bLo,bHi]`, return bounds
  on the product. -/
def intervalMul (aLo aHi bLo bHi : α) : α × α :=
  let p1 := aLo * bLo
  let p2 := aLo * bHi
  let p3 := aHi * bLo
  let p4 := aHi * bHi
  let lo1 := if p1 < p2 then p1 else p2
  let lo2 := if p3 < p4 then p3 else p4
  let lo  := if lo1 < lo2 then lo1 else lo2
  let hi1 := if p1 > p2 then p1 else p2
  let hi2 := if p3 > p4 then p3 else p4
  let hi  := if hi1 > hi2 then hi1 else hi2
  (lo, hi)

/-- Length of the last axis of a shape; scalars are treated as length one. -/
def lastDimLen : Shape → Nat
  | .scalar => 1
  | .dim n .scalar => n
  | .dim _ rest => lastDimLen rest

/-- Runtime witness that one shape can broadcast to another. -/
def mkCanBroadcastTo? : (s₁ s₂ : Shape) → Option (Shape.CanBroadcastTo s₁ s₂)
  | s₁, s₂ =>
    if hlt : Spec.Shape.rank s₁ < Spec.Shape.rank s₂ then
      match s₂ with
      | .scalar => none
      | .dim n₂ t₂ =>
        (mkCanBroadcastTo? s₁ t₂).map (fun tail =>
          Shape.CanBroadcastTo.expand_dims (n := n₂) (s₁ := s₁) (s₂ := t₂) tail)
    else if hgt : Spec.Shape.rank s₂ < Spec.Shape.rank s₁ then
      none
    else
      match s₁, s₂ with
      | .scalar, .scalar => some .scalar
      | .dim n₁ t₁, .dim n₂ t₂ =>
          letI : Shape.SameRank t₁ t₂ := ⟨by
            apply Nat.le_antisymm
            · exact Nat.le_of_not_gt (by simpa [Spec.Shape.rank] using hgt)
            · exact Nat.le_of_not_gt (by simpa [Spec.Shape.rank] using hlt)⟩
          if hEq : n₁ = n₂ then
            (mkCanBroadcastTo? t₁ t₂).map (fun tail =>
              hEq ▸ Shape.CanBroadcastTo.dim_eq (n := n₁) (s₁ := t₁) (s₂ := t₂) tail)
          else if h1 : n₁ = 1 then
            (mkCanBroadcastTo? t₁ t₂).map (fun tail =>
              h1 ▸ Shape.CanBroadcastTo.dim_1_to_n (n := n₂) (s₁ := t₁) (s₂ := t₂) tail)
          else
            none
      | _, _ => none

/-- Reinterpret a flattened tensor as shape `s` when the element counts agree. -/
def ibpUnflatten {s : Shape} (dim : Nat) (t : Tensor α (.dim dim .scalar)) (h : dim =
  Spec.Shape.size s) :
    Tensor α s :=
  let t' : Tensor α (.dim (Spec.Shape.size s) .scalar) := by
    simpa [h] using t
  Tensor.unflattenSpec (α := α) s t'

/-- IBP rule for broadcasting a flattened input box to a target shape. -/
def ibpBroadcastTo (s₁ s₂ : Shape) (Xin : FlatBox α) : Option (FlatBox α) :=
  if h : Xin.dim = Spec.Shape.size s₁ then
    match mkCanBroadcastTo? s₁ s₂ with
    | none => none
    | some cb =>
        let xLo : Tensor α s₁ := ibpUnflatten (α := α) (s := s₁) Xin.dim Xin.lo h
        let xHi : Tensor α s₁ := ibpUnflatten (α := α) (s := s₁) Xin.dim Xin.hi h
        let yLo : Tensor α s₂ := Tensor.broadcastTo (α := α) (s₁ := s₁) (s₂ := s₂) cb xLo
        let yHi : Tensor α s₂ := Tensor.broadcastTo (α := α) (s₁ := s₁) (s₂ := s₂) cb xHi
        let flatLo := Tensor.flattenSpec (α := α) yLo
        let flatHi := Tensor.flattenSpec (α := α) yHi
        some { dim := Spec.Shape.size s₂, lo := flatLo, hi := flatHi }
  else
    none

/-- IBP rule for reducing a shaped box by summing along one axis. -/
def ibpReduceSumAxis (axis : Nat) (Xin : FlatBox α) (s : Shape) : Option (FlatBox α) :=
  if h : Xin.dim = Spec.Shape.size s then
    match Spec.Shape.nonemptyAxis? (axis := axis) s with
    | none => none
    | some hAxis =>
        let hRed := hAxis.down
        let xLo : Tensor α s := ibpUnflatten (α := α) (s := s) Xin.dim Xin.lo h
        let xHi : Tensor α s := ibpUnflatten (α := α) (s := s) Xin.dim Xin.hi h
        let yLo := Tensor.reduceSum (α := α) (s := s) axis xLo hRed
        let yHi := Tensor.reduceSum (α := α) (s := s) axis xHi hRed
        let outS := Tensor.shapeAfterSum s axis
        let flatLo := Tensor.flattenSpec (α := α) yLo
        let flatHi := Tensor.flattenSpec (α := α) yHi
        some { dim := Spec.Shape.size outS, lo := flatLo, hi := flatHi }
  else
    none

/-- IBP rule for reducing a shaped box by averaging along one axis. -/
def ibpReduceMeanAxis (axis : Nat) (Xin : FlatBox α) (s : Shape) : Option (FlatBox α) :=
  if h : Xin.dim = Spec.Shape.size s then
    match Spec.Shape.nonemptyAxis? (axis := axis) s with
    | none => none
    | some hAxis =>
        let hRed := hAxis.down
        let xLo : Tensor α s := ibpUnflatten (α := α) (s := s) Xin.dim Xin.lo h
        let xHi : Tensor α s := ibpUnflatten (α := α) (s := s) Xin.dim Xin.hi h
        let yLo := Tensor.reduceMean (α := α) (s := s) axis xLo hRed
        let yHi := Tensor.reduceMean (α := α) (s := s) axis xHi hRed
        let outS := Tensor.shapeAfterSum s axis
        let flatLo := Tensor.flattenSpec (α := α) yLo
        let flatHi := Tensor.flattenSpec (α := α) yHi
        some { dim := Spec.Shape.size outS, lo := flatLo, hi := flatHi }
  else
    none

/--
Format-independent softmax enclosure on a flattened tensor.

A singleton row is exactly one. Every coordinate of a longer row lies in `[0,1]`. This deliberately
forgoes the tighter exponential formula above so executable checking does not assume a directed
transcendental implementation that its scalar backend has not supplied.
-/
def ibpSoftmaxRange (s : Shape) (dim : Nat) : FlatBox α :=
  let rowLength := lastDimLen s
  if rowLength = 1 then
    let ones := Spec.fill (α := α) Numbers.one (.dim dim .scalar)
    { dim := dim, lo := ones, hi := ones }
  else
    { dim := dim
      lo := Spec.fill (α := α) Numbers.zero (.dim dim .scalar)
      hi := Spec.fill (α := α) Numbers.one (.dim dim .scalar) }

/-!
## Hard-masked softmax IBP (last axis)

Blocked coordinates have weight zero. An allowed coordinate lies in `[0,1]`, and it has weight one
when it is the only allowed coordinate in its row. These bounds do not evaluate `exp` or division,
so they remain valid for executable endpoint types whose `BoundOps` instance covers only directed
arithmetic. A tighter transfer rule requires separately certified directed bounds for
transcendental operations.
-/

/-- Conservative interval bounds for hard-masked softmax along the last tensor axis. -/
def ibpHardMaskedSoftmaxLastTensor : {s : Shape} →
    Tensor α s → Tensor α s → Tensor Bool s → (Tensor α s × Tensor α s)
  | .scalar, _lo, _hi, Tensor.scalar allowed =>
      let value := if allowed then Numbers.one else Numbers.zero
      (Tensor.scalar value, Tensor.scalar value)
  | .dim n .scalar, Tensor.dim _lo, Tensor.dim _hi, Tensor.dim allowed =>
      let lower := Tensor.dim fun i =>
        match allowed i with
        | Tensor.scalar false => Tensor.scalar Numbers.zero
        | Tensor.scalar true =>
            let hasOtherAllowed := (List.finRange n).any fun j =>
              i != j && match allowed j with
                | Tensor.scalar a => a
            Tensor.scalar (if hasOtherAllowed then Numbers.zero else Numbers.one)
      let upper := Tensor.dim fun i =>
        match allowed i with
        | Tensor.scalar true => Tensor.scalar Numbers.one
        | Tensor.scalar false => Tensor.scalar Numbers.zero
      (lower, upper)
  | .dim n inner, Tensor.dim lo, Tensor.dim hi, Tensor.dim allowed =>
      let lower := Tensor.dim fun i : Fin n =>
        (ibpHardMaskedSoftmaxLastTensor (s := inner) (lo i) (hi i) (allowed i)).1
      let upper := Tensor.dim fun i : Fin n =>
        (ibpHardMaskedSoftmaxLastTensor (s := inner) (lo i) (hi i) (allowed i)).2
      (lower, upper)

/-!
## LayerNorm IBP (last axis)

Layer normalization (Ba et al.) computes, per vector, something like:

`y = (x - mean(x)) / sqrt(var(x) + eps)`.

We implement a conservative enclosure by:
1. Bounding mean using sums of endpoints.
2. Bounding variance using a max-deviation upper bound.
3. Bounding the per-component ratio by checking endpoint combinations against a positive denominator
   interval.

This is intended as a simple checker-side transfer rule. It is conservative and is not an
optimized relaxation.

References:
- Ba, Kiros, Hinton, "Layer Normalization", 2016: https://arxiv.org/abs/1607.06450
- Bound propagation context: Xu et al., 2020 (auto_LiRPA): https://arxiv.org/abs/2002.12920
-/

/--
Ideal-arithmetic upper bound on the variance term used by analytic LayerNorm rules.

Given endpoint bounds for a vector and bounds on its mean, each coordinate is at most
`max |x_i - μ|` away from the bounded mean interval. Squaring and summing those coordinate radii
gives a conservative variance upper bound. The implementation uses ordinary scalar arithmetic and
is called only from exact-arithmetic branches; executable endpoint propagation uses
`ibpLayerNormRange?` instead.
-/
def idealLayerNormVarianceUpper {n : Nat}
    (lo hi : Tensor α (.dim n .scalar)) (muLo muHi : α) : α :=
  if _h : n > 0 then
    match lo, hi with
    | .dim flo, .dim fhi =>
        let sumAbsSq : α := (List.finRange n).foldl (fun acc (i : Fin n) =>
          match flo i, fhi i with
          | .scalar l, .scalar u =>
            let dl := MathFunctions.abs (l - muHi)
            let du := MathFunctions.abs (u - muLo)
            let a := if dl > du then dl else du
            acc + (a * a)) 0
        sumAbsSq / (n : Nat)
  else
    Numbers.zero

/-- Ideal-arithmetic mean bounds for a nonempty vector with bounded coordinates.

For `n = 0`, the mathematical mean is undefined; this total helper returns `(0,0)` so callers do
not accidentally divide by zero while they reject or totalize the empty case.
-/
def idealLayerNormMeanBounds {n : Nat}
    (lo hi : Tensor α (.dim n .scalar)) : α × α :=
  if _h : n > 0 then
    let nA : α := (n : Nat)
    (Spec.Tensor.sumSpec lo / nA, Spec.Tensor.sumSpec hi / nA)
  else
    (Numbers.zero, Numbers.zero)

/--
Ideal-arithmetic bounds for `x - μ` when `x` and `μ` are bounded by intervals.

LayerNorm transfer rules repeatedly need this centered interval for the input, first derivative,
and second derivative streams. Keeping it here avoids duplicating the same endpoint arithmetic in
IBP and derivative propagation.
-/
def idealLayerNormCenteredBounds {n : Nat}
    (lo hi : Tensor α (.dim n .scalar)) (muLo muHi : α) :
    Tensor α (.dim n .scalar) × Tensor α (.dim n .scalar) :=
  let flo := match lo with | .dim f => f
  let fhi := match hi with | .dim f => f
  let loOut :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let dl := l - muHi
        let du := u - muLo
        Tensor.scalar (if dl < du then dl else du))
  let hiOut :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let dl := l - muHi
        let du := u - muLo
        Tensor.scalar (if dl > du then dl else du))
  (loOut, hiOut)

/-- Ideal-arithmetic reciprocal-denominator bounds from an upper variance bound. -/
def idealLayerNormInvStdBounds (varHi : α) : α × α :=
  let sLo := MathFunctions.sqrt Numbers.epsilon
  let sHi := MathFunctions.sqrt (varHi + Numbers.epsilon)
  (Numbers.one / (if sHi > Numbers.epsilon then sHi else Numbers.epsilon),
   Numbers.one / (if sLo > Numbers.epsilon then sLo else Numbers.epsilon))

/-- Analytic real-arithmetic LayerNorm bounds on the last axis, lifted over leading dimensions. -/
def idealLayerNormLastTensor : {s : Shape} → Tensor α s → Tensor α s → (Tensor α s × Tensor α s)
  | .scalar, lo, hi => (lo, hi)
  | .dim n .scalar, lo, hi =>
      if n > 0 then
        let nA : α := (n : Nat)
        let sum_lo := Spec.Tensor.sumSpec lo
        let sum_hi := Spec.Tensor.sumSpec hi
        let mu_lo := sum_lo / nA
        let mu_hi := sum_hi / nA
        let flo := match lo with | .dim f => f
        let fhi := match hi with | .dim f => f
        let var_hi := idealLayerNormVarianceUpper (α := α) lo hi mu_lo mu_hi
        let den_lo := MathFunctions.sqrt Numbers.epsilon
        let den_hi := MathFunctions.sqrt (var_hi + Numbers.epsilon)
        let outLo :=
          Tensor.dim (fun i =>
            match flo i, fhi i with
            | .scalar l, .scalar u =>
              let dl := l - mu_hi
              let du := u - mu_lo
              -- For positive denom interval [den_lo, den_hi], bound (x/denom) by checking all
              -- endpoint ratios.
              let c1 := dl / den_lo
              let c2 := dl / den_hi
              let c3 := du / den_lo
              let c4 := du / den_hi
              let mn12 := if c1 < c2 then c1 else c2
              let mn34 := if c3 < c4 then c3 else c4
              let mn := if mn12 < mn34 then mn12 else mn34
              Tensor.scalar mn)
        let outHi :=
          Tensor.dim (fun i =>
            match flo i, fhi i with
            | .scalar l, .scalar u =>
              let dl := l - mu_hi
              let du := u - mu_lo
              let c1 := dl / den_lo
              let c2 := dl / den_hi
              let c3 := du / den_lo
              let c4 := du / den_hi
              let mx12 := if c1 > c2 then c1 else c2
              let mx34 := if c3 > c4 then c3 else c4
              let mx := if mx12 > mx34 then mx12 else mx34
              Tensor.scalar mx)
        (outLo, outHi)
      else
        -- Degenerate n=0: pass through
        (lo, hi)
  | .dim n inner, Tensor.dim loF, Tensor.dim hiF =>
      let outLo := Tensor.dim (fun i : Fin n => (idealLayerNormLastTensor (s := inner) (loF i) (hiF
        i)).1)
      let outHi := Tensor.dim (fun i : Fin n => (idealLayerNormLastTensor (s := inner) (loF i) (hiF
        i)).2)
      (outLo, outHi)

/--
Uniform finite enclosure for last-axis layer normalization.

For a row of length `n`, exact LayerNorm without affine scale or bias satisfies
`|yᵢ| ≤ sqrt n`; a singleton row is identically zero. Backends provide the outward-rounded bound
through `NonlinearBoundOps.layerNormAbsBound`. Returning `none` is preferable to evaluating the
normalization with unqualified host division and square root.
-/
def ibpLayerNormRange? [NonlinearBoundOps α]
    (s : Shape) (dim : Nat) : Option (FlatBox α) :=
  let rowLength := lastDimLen s
  if rowLength = 0 then
    some
      { dim := dim
        lo := Spec.fill (α := α) Numbers.zero (.dim dim .scalar)
        hi := Spec.fill (α := α) Numbers.zero (.dim dim .scalar) }
  else if rowLength = 1 then
    some
      { dim := dim
        lo := Spec.fill (α := α) Numbers.zero (.dim dim .scalar)
        hi := Spec.fill (α := α) Numbers.zero (.dim dim .scalar) }
  else do
    let radius ← NonlinearBoundOps.layerNormAbsBound (α := α) rowLength
    pure
      { dim := dim
        lo := Spec.fill (α := α) (-radius) (.dim dim .scalar)
        hi := Spec.fill (α := α) radius (.dim dim .scalar) }

/-- For tensors known to have shape `.dim n .scalar`, extract the underlying function. -/
@[expose] public def getDimScalarFn {n : Nat} (t : Tensor α (.dim n .scalar)) : (Fin n → Tensor α
  .scalar) :=
  match t with
  | .dim f => f

-- Casting helpers for dependent shapes
/-- Cast a 1D `Box` along an equality of dimensions. -/
@[expose]
public def castBoxDim {n n' : Nat}
  (h : n = n')
  (B : Box α (.dim n .scalar)) : Box α (.dim n' .scalar) := by
  simpa [h] using B

/-- Cast a ReLU relaxation vector across a proven-equal hidden dimension. -/
def castRelax {n n' : Nat}
  (h : n = n')
  (r : Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α) (.dim n .scalar)) :
  Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α) (.dim n' .scalar) := by
  simpa [h] using r

/-- Cast the input dimension of an affine map across a proven equality. -/
def castAffineIn {n n' m : Nat}
  (h : n = n') (a : AffineVec α n m) : AffineVec α n' m := by
  simpa [h] using a

/-- Cast the output dimension of an affine map across a proven equality. -/
@[expose]
def castAffineOut {n m m' : Nat}
  (h : m = m') (a : AffineVec α n m) : AffineVec α n m' := by
  simpa [h] using a

/--
Cast a dim-scalar tensor across an equality of dimensions.

We keep this as an `abbrev` so it unfolds aggressively in simp-based soundness proofs.
-/
abbrev castDimScalar {n n' : Nat}
    (h : n = n') (t : Tensor α (.dim n .scalar)) : Tensor α (.dim n' .scalar) :=
  Tensor.castShape t (congrArg (fun k => Shape.dim k Shape.scalar) h)

omit [Context α] [BoundOps α] in
/-- Casting a flat vector tensor along an equality from a dimension to itself changes no data. -/
@[simp] theorem castDimScalar_self {n : Nat}
    (h : n = n) (t : Tensor α (.dim n .scalar)) :
    castDimScalar (α := α) h t = t := by
  exact Tensor.cast_shape_self t _

/-- IBP propagation through explicit linear parameters. -/
@[expose]
public def ibpLinearParams (p : LinParams α) (Xin : FlatBox α) : Option (FlatBox α) :=
  if h : Xin.dim = p.n then
    let xB   : Box α (.dim p.n .scalar) := castBoxDim (α:=α) h (ofFlatBox Xin)
    let bBox : Box α (.dim p.m .scalar) := Box.point (α:=α) p.b
    let yB   := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB bBox
    -- Materialize to avoid deep closure chains in multi-layer verifier runs.
    let yB' : Box α (.dim p.m .scalar) :=
      { lo := Tensor.materialize yB.lo
        hi := Tensor.materialize yB.hi }
    some (toFlatBox p.m yB')
  else none

/-- IBP propagation for a `.linear` node using `ParamStore.linearWB`. -/
@[expose]
public def ibpLinear (id : Nat) (ps : ParamStore α) (Xin : FlatBox α) : Option (FlatBox α) :=
  match ps.linearWB[id]? with
  | none => none
  | some p => ibpLinearParams (α := α) p Xin

/-- IBP propagation for a `.matmul` node (bias-free) using `ParamStore.matmulW`. -/
@[expose]
public def ibpMatmul (id : Nat) (ps : ParamStore α) (Xin : FlatBox α) : Option (FlatBox α) :=
  match ps.matmulW[id]? with
  | none => none
  | some p =>
    if h : Xin.dim = p.n then
      let xB   : Box α (.dim p.n .scalar) := castBoxDim (α:=α) h (ofFlatBox Xin)
      let zeroB : Box α (.dim p.m .scalar) :=
        let z := Spec.fill (α:=α) 0 (.dim p.m .scalar)
        Box.point (α:=α) z
      let yB := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB zeroB
      -- Materialize to avoid deep closure chains (runtime performance).
      let yB' : Box α (.dim p.m .scalar) :=
        { lo := Tensor.materialize yB.lo
          hi := Tensor.materialize yB.hi }
      some (toFlatBox p.m yB')
    else none

/-- IBP transfer for a convolution node whose parameters are stored in `ParamStore.conv2dCfg`. -/
def ibpConv2dNode (id : Nat) (ps : ParamStore α) (Xin : FlatBox α) : Option (FlatBox α) :=
  match ps.conv2dCfg[id]? with
  | none => none
  | some cfg =>
    let expected := cfg.inC * cfg.inH * cfg.inW
    if _hs : cfg.stride = 0 then
      none
    else if hdim : Xin.dim = expected then
      let sFlat := Shape.dim Xin.dim Shape.scalar
      let sIn := Shape.dim cfg.inC (Shape.dim cfg.inH (Shape.dim cfg.inW Shape.scalar))
      have hsize : sFlat.size = sIn.size := by
        simp [Spec.Shape.size, sFlat, sIn, hdim, expected, Nat.mul_assoc]
      let xLo := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.lo hsize
      let xHi := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.hi hsize
      let xBox : Box α sIn := { lo := xLo, hi := xHi }
      let yBox := NN.MLTheory.CROWN.ibpConv2d (α:=α)
        (layer:=cfg.spec) (xB:=xBox)
      let outH := Spec.Shape.slidingWindowOutDim cfg.inH cfg.kH cfg.stride cfg.padding
      let outW := Spec.Shape.slidingWindowOutDim cfg.inW cfg.kW cfg.stride cfg.padding
      let outShape := Shape.dim cfg.outC (Shape.dim outH (Shape.dim outW Shape.scalar))
      let flatLo := Tensor.flattenSpec (α:=α) yBox.lo
      let flatHi := Tensor.flattenSpec (α:=α) yBox.hi
      some { dim := outShape.size, lo := flatLo, hi := flatHi }
    else none


end NN.MLTheory.CROWN.Graph

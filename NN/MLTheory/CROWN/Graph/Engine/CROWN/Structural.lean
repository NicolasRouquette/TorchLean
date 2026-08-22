/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.CROWN.Activations

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN
open NN.IR

variable {α : Type} [Context α]
variable [BoundOps α]

open BoundOps

/-!
# CROWN Structural Operators

Affine propagation through permutations, normalization, softmax, and elementwise products.
-/

/-- Permute the output coordinates of an affine bound when the output shape permutation is valid. -/
def permuteAffineOut {inDim outDim : Nat}
  (perm : Fin outDim → Fin outDim) (aff : AffineVec α inDim outDim) : AffineVec α inDim outDim :=
  match aff.A, aff.c with
  | .dim rows, .dim cvec =>
    { A := Tensor.dim (fun i => rows (perm i))
      c := Tensor.dim (fun i => cvec (perm i)) }

/-- Conservative CROWN-style affine bounds for softmax along the last tensor axis. -/
def propagateSoftmaxBoundsLastAxis
  (s : Shape) (preB : FlatBox α) (xB : FlatAffineBounds α) (hout : xB.outDim = preB.dim) :
  FlatAffineBounds α :=
  let xLo : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α := α) (n := xB.inDim) (m := xB.outDim) (m' := preB.dim) hout xB.loAff
  let xHi : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α := α) (n := xB.inDim) (m := xB.outDim) (m' := preB.dim) hout xB.hiAff
  let m := lastDimLen s
  if m = 0 then
    { inDim := xB.inDim, outDim := preB.dim, loAff := xLo, hiAff := xHi }
  else if m = 1 then
    -- Each last-axis slice has length 1, so softmax is identically 1.
    let ones : Tensor α (.dim preB.dim .scalar) := Spec.fill (α := α) Numbers.one (.dim preB.dim
      .scalar)
    boundsConst (α := α) (inputDim := xB.inDim) (outDim := preB.dim) ones ones
  else
    let dim := preB.dim
    if dim % m = 0 then
      let expLo : Tensor α (.dim dim .scalar) := Tensor.expSpec preB.lo
      let expHi : Tensor α (.dim dim .scalar) := Tensor.expSpec preB.hi
      let groups : Nat := dim / m
      let totalExpLo : Tensor α (.dim groups .scalar) :=
        Tensor.dim (fun g =>
          let base := g.val * m
          let sum : α := (List.range m).foldl (fun acc j => acc + getAtOrZero expLo [base + j]) 0
          Tensor.scalar sum)
      let totalExpHi : Tensor α (.dim groups .scalar) :=
        Tensor.dim (fun g =>
          let base := g.val * m
          let sum : α := (List.range m).foldl (fun acc j => acc + getAtOrZero expHi [base + j]) 0
          Tensor.scalar sum)
      let flo := getDimScalarFn (α := α) preB.lo
      let fhi := getDimScalarFn (α := α) preB.hi
      -- Upper bound via logistic with C = Σ_{j≠i} exp(lo_j)
      let slopes_hi : Tensor α (.dim dim .scalar) :=
        Tensor.dim (fun i =>
          let g := i.val / m
          match flo i, fhi i with
          | .scalar l, .scalar u =>
            let tot := getAtOrZero totalExpLo [g]
            let eLi := getAtOrZero expLo [i.val]
            let c := tot - eLi
            if c > Numbers.epsilon then
              let logC := MathFunctions.log c
              let (_aLo, _bLo, aHi, _bHi) := sigmoidLineBounds (α := α) (l - logC) (u - logC)
              Tensor.scalar aHi
            else
              Tensor.scalar Numbers.zero)
      let bias_hi : Tensor α (.dim dim .scalar) :=
        Tensor.dim (fun i =>
          let g := i.val / m
          match flo i, fhi i with
          | .scalar l, .scalar u =>
            let tot := getAtOrZero totalExpLo [g]
            let eLi := getAtOrZero expLo [i.val]
            let c := tot - eLi
            if c > Numbers.epsilon then
              let logC := MathFunctions.log c
              let (_aLo, _bLo, aHi, bHi) := sigmoidLineBounds (α := α) (l - logC) (u - logC)
              Tensor.scalar (bHi - aHi * logC)
            else
              Tensor.scalar Numbers.one)
      -- Lower bound via logistic with C = Σ_{j≠i} exp(hi_j)
      let slopes_lo : Tensor α (.dim dim .scalar) :=
        Tensor.dim (fun i =>
          let g := i.val / m
          match flo i, fhi i with
          | .scalar l, .scalar u =>
            let tot := getAtOrZero totalExpHi [g]
            let eUi := getAtOrZero expHi [i.val]
            let c := tot - eUi
            if c > Numbers.epsilon then
              let logC := MathFunctions.log c
              let (aLo, _bLo, _aHi, _bHi) := sigmoidLineBounds (α := α) (l - logC) (u - logC)
              Tensor.scalar aLo
            else
              Tensor.scalar Numbers.zero)
      let bias_lo : Tensor α (.dim dim .scalar) :=
        Tensor.dim (fun i =>
          let g := i.val / m
          match flo i, fhi i with
          | .scalar l, .scalar u =>
            let tot := getAtOrZero totalExpHi [g]
            let eUi := getAtOrZero expHi [i.val]
            let c := tot - eUi
            if c > Numbers.epsilon then
              let logC := MathFunctions.log c
              let (aLo, bLo, _aHi, _bHi) := sigmoidLineBounds (α := α) (l - logC) (u - logC)
              Tensor.scalar (bLo - aLo * logC)
            else
              Tensor.scalar Numbers.zero)
      let loAff := affApplyDiag (α := α) (inDim := xB.inDim) (outDim := dim) slopes_lo bias_lo xLo
      let hiAff := affApplyDiag (α := α) (inDim := xB.inDim) (outDim := dim) slopes_hi bias_hi xHi
      { inDim := xB.inDim, outDim := dim, loAff := loAff, hiAff := hiAff }
    else
      -- Shape mismatch: fall back to trivial [0,1] bounds.
      let zeros : Tensor α (.dim dim .scalar) := Spec.fill (α := α) Numbers.zero (.dim dim .scalar)
      let ones : Tensor α (.dim dim .scalar) := Spec.fill (α := α) Numbers.one (.dim dim .scalar)
      boundsConst (α := α) (inputDim := xB.inDim) (outDim := dim) zeros ones

/-- Conservative affine bounds for layer normalization over the last tensor axis. -/
def propagateLayernormBoundsLastAxis
  (s : Shape) (preB : FlatBox α) (xB : FlatAffineBounds α) (hout : xB.outDim = preB.dim) :
  FlatAffineBounds α :=
  let xLo : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α := α) (n := xB.inDim) (m := xB.outDim) (m' := preB.dim) hout xB.loAff
  let xHi : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α := α) (n := xB.inDim) (m := xB.outDim) (m' := preB.dim) hout xB.hiAff
  let m := lastDimLen s
  if m = 0 then
    { inDim := xB.inDim, outDim := preB.dim, loAff := xLo, hiAff := xHi }
  else if m = 1 then
    -- Each slice has length 1: (x - mean)/sqrt(var+eps) = 0.
    let zeros : Tensor α (.dim preB.dim .scalar) := Spec.fill (α := α) Numbers.zero (.dim preB.dim
      .scalar)
    boundsConst (α := α) (inputDim := xB.inDim) (outDim := preB.dim) zeros zeros
  else
    let dim := preB.dim
    if dim % m = 0 then
      let groups : Nat := dim / m
      let mA : α := (m : Nat)
      let denLo : α := MathFunctions.sqrt Numbers.epsilon
      let muLoG : Tensor α (.dim groups .scalar) :=
        Tensor.dim (fun g =>
          let base := g.val * m
          let sumLo : α := (List.range m).foldl (fun acc j => acc + getAtOrZero preB.lo [base +
            j]) 0
          Tensor.scalar (sumLo / mA))
      let muHiG : Tensor α (.dim groups .scalar) :=
        Tensor.dim (fun g =>
          let base := g.val * m
          let sumHi : α := (List.range m).foldl (fun acc j => acc + getAtOrZero preB.hi [base +
            j]) 0
          Tensor.scalar (sumHi / mA))
      let denHiG : Tensor α (.dim groups .scalar) :=
        Tensor.dim (fun g =>
          let base := g.val * m
          let muLo := getAtOrZero muLoG [g.val]
          let muHi := getAtOrZero muHiG [g.val]
          let loSlice : Tensor α (.dim m .scalar) :=
            Tensor.dim (fun j => Tensor.scalar (getAtOrZero preB.lo [base + j.val]))
          let hiSlice : Tensor α (.dim m .scalar) :=
            Tensor.dim (fun j => Tensor.scalar (getAtOrZero preB.hi [base + j.val]))
          let varHi := idealLayerNormVarianceUpper (α := α) loSlice hiSlice muLo muHi
          Tensor.scalar (MathFunctions.sqrt (varHi + Numbers.epsilon)))
      let flo := getDimScalarFn (α := α) preB.lo
      let fhi := getDimScalarFn (α := α) preB.hi
      let slopes_hi : Tensor α (.dim dim .scalar) :=
        Tensor.dim (fun i =>
          let g := i.val / m
          let muLo := getAtOrZero muLoG [g]
          let denHi := getAtOrZero denHiG [g]
          match flo i, fhi i with
          | .scalar l, .scalar u =>
            let uL :=
              let num := l - muLo
              let den := if decide (l > muLo) then denLo else denHi
              num / den
            let uU :=
              let num := u - muLo
              let den := if decide (u > muLo) then denLo else denHi
              num / den
            let denx := u - l
            let a := if denx > Numbers.epsilon then (uU - uL) / denx else Numbers.zero
            Tensor.scalar a)
      let bias_hi : Tensor α (.dim dim .scalar) :=
        Tensor.dim (fun i =>
          let g := i.val / m
          let muLo := getAtOrZero muLoG [g]
          let denHi := getAtOrZero denHiG [g]
          match flo i, fhi i with
          | .scalar l, .scalar u =>
            let uL :=
              let num := l - muLo
              let den := if decide (l > muLo) then denLo else denHi
              num / den
            let uU :=
              let num := u - muLo
              let den := if decide (u > muLo) then denLo else denHi
              num / den
            let denx := u - l
            if denx > Numbers.epsilon then
              let a := (uU - uL) / denx
              Tensor.scalar (uL - a * l)
            else
              Tensor.scalar (if uL > uU then uL else uU))
      let slopes_lo : Tensor α (.dim dim .scalar) :=
        Tensor.dim (fun i =>
          let g := i.val / m
          let muHi := getAtOrZero muHiG [g]
          let denHi := getAtOrZero denHiG [g]
          match flo i, fhi i with
          | .scalar l, .scalar u =>
            let lL :=
              let num := l - muHi
              let den := if decide (l > muHi) then denHi else denLo
              num / den
            let lU :=
              let num := u - muHi
              let den := if decide (u > muHi) then denHi else denLo
              num / den
            let denx := u - l
            let a := if denx > Numbers.epsilon then (lU - lL) / denx else Numbers.zero
            Tensor.scalar a)
      let bias_lo : Tensor α (.dim dim .scalar) :=
        Tensor.dim (fun i =>
          let g := i.val / m
          let muHi := getAtOrZero muHiG [g]
          let denHi := getAtOrZero denHiG [g]
          match flo i, fhi i with
          | .scalar l, .scalar u =>
            let lL :=
              let num := l - muHi
              let den := if decide (l > muHi) then denHi else denLo
              num / den
            let lU :=
              let num := u - muHi
              let den := if decide (u > muHi) then denHi else denLo
              num / den
            let denx := u - l
            if denx > Numbers.epsilon then
              let a := (lU - lL) / denx
              Tensor.scalar (lL - a * l)
            else
              Tensor.scalar (if lL < lU then lL else lU))
      let loAff := affApplyDiag (α := α) (inDim := xB.inDim) (outDim := dim) slopes_lo bias_lo xLo
      let hiAff := affApplyDiag (α := α) (inDim := xB.inDim) (outDim := dim) slopes_hi bias_hi xHi
      { inDim := xB.inDim, outDim := dim, loAff := loAff, hiAff := hiAff }
    else
      -- Shape mismatch: conservative constant bounds from IBP on this op.
      let (flatLo, flatHi) :=
        idealLayerNormLastTensor (α := α) (s := .dim dim .scalar) preB.lo preB.hi
      boundsConst (α := α) (inputDim := xB.inDim) (outDim := dim) flatLo flatHi

namespace Internal

/-- Matmul affine propagation using McCormick-style product planes. -/
def propagateMatmulBounds
  (sA sB : Shape) (Bx By : FlatBox α)
  (aB bB : FlatAffineBounds α) :
  Option (FlatAffineBounds α) :=
  if hin : aB.inDim = bB.inDim then
    let inDim := aB.inDim
    let bLo : AffineVec α inDim bB.outDim :=
      castAffineIn (α:=α) (n:=bB.inDim) (n':=inDim) (m:=bB.outDim) hin.symm bB.loAff
    let bHi : AffineVec α inDim bB.outDim :=
      castAffineIn (α:=α) (n:=bB.inDim) (n':=inDim) (m:=bB.outDim) hin.symm bB.hiAff
    let split (a : α) : α × α :=
      if a > Numbers.zero then (a, Numbers.zero) else (Numbers.zero, a)
    let dims? : Option (Nat × Nat × Nat × Nat) :=
      match sA, sB with
      | .dim m (.dim k .scalar), .dim k' (.dim n .scalar) =>
        if k = k' then
          some (1, m, k, n)
        else
          none
      | .dim b (.dim m (.dim k .scalar)), .dim b' (.dim k' (.dim n .scalar)) =>
        if hb : b = b' then
          match hb with
          | rfl =>
            if k = k' then
              some (b, m, k, n)
            else
              none
        else
          none
      | _, _ => none
    match dims? with
    | none => none
    | some (batch, m, k, n) =>
      let dimA := batch * m * k
      let dimB := batch * k * n
      let outDim := batch * m * n
      if Bx.dim = dimA ∧ aB.outDim = dimA then
        if By.dim = dimB ∧ bB.outDim = dimB then
          let block : Nat := m * n
          let strideA : Nat := m * k
          let strideB : Nat := k * n

          let termUpperCoeff (aIdx bIdx inJ : Nat) : α :=
            let lx := getAtOrZero Bx.lo [aIdx]
            let ux := getAtOrZero Bx.hi [aIdx]
            let ly := getAtOrZero By.lo [bIdx]
            let uy := getAtOrZero By.hi [bIdx]
            let cx := (lx + ux) * Numbers.half
            let cy := (ly + uy) * Numbers.half
            let u1 := ux * cy + ly * cx - ux * ly
            let u2 := lx * cy + uy * cx - lx * uy
            let aX := if u1 < u2 then ly else uy
            let aY := if u1 < u2 then ux else lx
            let (aXpos, aXneg) := split aX
            let (aYpos, aYneg) := split aY
            let xU := getAtOrZero aB.hiAff.A [aIdx, inJ]
            let xL := getAtOrZero aB.loAff.A [aIdx, inJ]
            let yU := getAtOrZero bHi.A [bIdx, inJ]
            let yL := getAtOrZero bLo.A [bIdx, inJ]
            aXpos * xU + aXneg * xL + aYpos * yU + aYneg * yL

          let termUpperConst (aIdx bIdx : Nat) : α :=
            let lx := getAtOrZero Bx.lo [aIdx]
            let ux := getAtOrZero Bx.hi [aIdx]
            let ly := getAtOrZero By.lo [bIdx]
            let uy := getAtOrZero By.hi [bIdx]
            let cx := (lx + ux) * Numbers.half
            let cy := (ly + uy) * Numbers.half
            let u1 := ux * cy + ly * cx - ux * ly
            let u2 := lx * cy + uy * cx - lx * uy
            let aX := if u1 < u2 then ly else uy
            let aY := if u1 < u2 then ux else lx
            let off := if u1 < u2 then (-(ux * ly)) else (-(lx * uy))
            let (aXpos, aXneg) := split aX
            let (aYpos, aYneg) := split aY
            let xU := getAtOrZero aB.hiAff.c [aIdx]
            let xL := getAtOrZero aB.loAff.c [aIdx]
            let yU := getAtOrZero bHi.c [bIdx]
            let yL := getAtOrZero bLo.c [bIdx]
            aXpos * xU + aXneg * xL + aYpos * yU + aYneg * yL + off

          let termLowerCoeff (aIdx bIdx inJ : Nat) : α :=
            let lx := getAtOrZero Bx.lo [aIdx]
            let ux := getAtOrZero Bx.hi [aIdx]
            let ly := getAtOrZero By.lo [bIdx]
            let uy := getAtOrZero By.hi [bIdx]
            let cx := (lx + ux) * Numbers.half
            let cy := (ly + uy) * Numbers.half
            let l1 := lx * cy + ly * cx - lx * ly
            let l2 := ux * cy + uy * cx - ux * uy
            let aX := if l1 > l2 then ly else uy
            let aY := if l1 > l2 then lx else ux
            let (aXpos, aXneg) := split aX
            let (aYpos, aYneg) := split aY
            let xL := getAtOrZero aB.loAff.A [aIdx, inJ]
            let xU := getAtOrZero aB.hiAff.A [aIdx, inJ]
            let yL := getAtOrZero bLo.A [bIdx, inJ]
            let yU := getAtOrZero bHi.A [bIdx, inJ]
            aXpos * xL + aXneg * xU + aYpos * yL + aYneg * yU

          let termLowerConst (aIdx bIdx : Nat) : α :=
            let lx := getAtOrZero Bx.lo [aIdx]
            let ux := getAtOrZero Bx.hi [aIdx]
            let ly := getAtOrZero By.lo [bIdx]
            let uy := getAtOrZero By.hi [bIdx]
            let cx := (lx + ux) * Numbers.half
            let cy := (ly + uy) * Numbers.half
            let l1 := lx * cy + ly * cx - lx * ly
            let l2 := ux * cy + uy * cx - ux * uy
            let aX := if l1 > l2 then ly else uy
            let aY := if l1 > l2 then lx else ux
            let off := if l1 > l2 then (-(lx * ly)) else (-(ux * uy))
            let (aXpos, aXneg) := split aX
            let (aYpos, aYneg) := split aY
            let xL := getAtOrZero aB.loAff.c [aIdx]
            let xU := getAtOrZero aB.hiAff.c [aIdx]
            let yL := getAtOrZero bLo.c [bIdx]
            let yU := getAtOrZero bHi.c [bIdx]
            aXpos * xL + aXneg * xU + aYpos * yL + aYneg * yU + off

          let A_hi : Tensor α (.dim outDim (.dim inDim .scalar)) :=
            Tensor.dim (fun outI =>
              let t := outI.val
              let bi := t / block
              let rem := t % block
              let i := rem / n
              let j := rem % n
              let baseA := bi * strideA
              let baseB := bi * strideB
              Tensor.dim (fun inJ =>
                let coeff :=
                  (List.range k).foldl (fun acc kk =>
                    acc + termUpperCoeff (baseA + i * k + kk) (baseB + kk * n + j) inJ.val
                  ) 0
                Tensor.scalar coeff))
          let c_hi : Tensor α (.dim outDim .scalar) :=
            Tensor.dim (fun outI =>
              let t := outI.val
              let bi := t / block
              let rem := t % block
              let i := rem / n
              let j := rem % n
              let baseA := bi * strideA
              let baseB := bi * strideB
              let coeff :=
                (List.range k).foldl (fun acc kk =>
                  acc + termUpperConst (baseA + i * k + kk) (baseB + kk * n + j)
                ) 0
              Tensor.scalar coeff)

          let A_lo : Tensor α (.dim outDim (.dim inDim .scalar)) :=
            Tensor.dim (fun outI =>
              let t := outI.val
              let bi := t / block
              let rem := t % block
              let i := rem / n
              let j := rem % n
              let baseA := bi * strideA
              let baseB := bi * strideB
              Tensor.dim (fun inJ =>
                let coeff :=
                  (List.range k).foldl (fun acc kk =>
                    acc + termLowerCoeff (baseA + i * k + kk) (baseB + kk * n + j) inJ.val
                  ) 0
                Tensor.scalar coeff))
          let c_lo : Tensor α (.dim outDim .scalar) :=
            Tensor.dim (fun outI =>
              let t := outI.val
              let bi := t / block
              let rem := t % block
              let i := rem / n
              let j := rem % n
              let baseA := bi * strideA
              let baseB := bi * strideB
              let coeff :=
                (List.range k).foldl (fun acc kk =>
                  acc + termLowerConst (baseA + i * k + kk) (baseB + kk * n + j)
                ) 0
              Tensor.scalar coeff)

          some
            { inDim := inDim
              outDim := outDim
              loAff := { A := A_lo, c := c_lo }
              hiAff := { A := A_hi, c := c_hi } }
        else
          none
      else
        none
  else
    none

end Internal

/-- Propagate affine bounds through componentwise multiplication using per-coordinate product planes. -/
def propagateMulElemBounds
  (Bx By : FlatBox α)
  (xB yB : FlatAffineBounds α)
  (houtX : xB.outDim = Bx.dim) (houtY : yB.outDim = By.dim) :
  Option (FlatAffineBounds α) :=
  -- Require equal vector lengths and equal input widths.
  if hdim : Bx.dim = By.dim then
    if hin : xB.inDim = yB.inDim then
      let n := Bx.dim
      let hyo : yB.outDim = n := Eq.trans houtY (Eq.symm hdim)
      let hBy : By.dim = n := by simpa [n] using (Eq.symm hdim)
      let ByLo : Tensor α (.dim n .scalar) := castDimScalar (α:=α) (n:=By.dim) (n':=n) hBy By.lo
      let ByHi : Tensor α (.dim n .scalar) := castDimScalar (α:=α) (n:=By.dim) (n':=n) hBy By.hi

      let xLo : AffineVec α xB.inDim n :=
        castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=n) (by simpa [n] using houtX)
          xB.loAff
      let xHi : AffineVec α xB.inDim n :=
        castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=n) (by simpa [n] using houtX)
          xB.hiAff
      let yLo0 : AffineVec α yB.inDim n :=
        castAffineOut (α:=α) (n:=yB.inDim) (m:=yB.outDim) (m':=n) hyo yB.loAff
      let yHi0 : AffineVec α yB.inDim n :=
        castAffineOut (α:=α) (n:=yB.inDim) (m:=yB.outDim) (m':=n) hyo yB.hiAff
        let yLo : AffineVec α xB.inDim n :=
          castAffineIn (α:=α) (n:=yB.inDim) (n':=xB.inDim) (m:=n) hin.symm yLo0
        let yHi : AffineVec α xB.inDim n :=
          castAffineIn (α:=α) (n:=yB.inDim) (n':=xB.inDim) (m:=n) hin.symm yHi0

        -- Helper to split a scalar coefficient into (pos, neg).
        let split (a : α) : α × α := if a > Numbers.zero then (a, Numbers.zero) else (Numbers.zero,
          a)

        -- Build row-wise A/c for upper and lower using a single selected McCormick plane per
        -- component.
        let A_hi :=
          match xLo.A, xHi.A, yLo.A, yHi.A, Bx.lo, Bx.hi, ByLo, ByHi with
          | .dim AxL, .dim AxU, .dim AyL, .dim AyU, .dim lxF, .dim uxF, .dim lyF, .dim uyF =>
            Tensor.dim (fun i =>
              match lxF i, uxF i, lyF i, uyF i, AxL i, AxU i, AyL i, AyU i with
              | .scalar lx, .scalar ux, .scalar ly, .scalar uy,
                .dim rowXL, .dim rowXU, .dim rowYL, .dim rowYU =>
                -- Choose min of two upper planes at the interval center.
                let cx := (lx + ux) * Numbers.half
                let cy := (ly + uy) * Numbers.half
                let u1 := ux * cy + ly * cx - ux * ly
                let u2 := lx * cy + uy * cx - lx * uy
                let aX := if u1 < u2 then ly else uy     -- coeff for x
                let aY := if u1 < u2 then ux else lx     -- coeff for y
                let (aXpos, aXneg) := split aX
                let (aYpos, aYneg) := split aY
                Tensor.dim (fun j =>
                  match rowXL j, rowXU j, rowYL j, rowYU j with
                  | .scalar xl, .scalar xu, .scalar yl, .scalar yu =>
                    Tensor.scalar (aXpos * xu + aXneg * xl + aYpos * yu + aYneg * yl)))
        let c_hi :=
          match xLo.c, xHi.c, yLo.c, yHi.c, Bx.lo, Bx.hi, ByLo, ByHi with
          | .dim cxL, .dim cxU, .dim cyL, .dim cyU, .dim lxF, .dim uxF, .dim lyF, .dim uyF =>
            Tensor.dim (fun i =>
              match lxF i, uxF i, lyF i, uyF i, cxL i, cxU i, cyL i, cyU i with
              | .scalar lx, .scalar ux, .scalar ly, .scalar uy,
                .scalar cxl, .scalar cxu, .scalar cyl, .scalar cyu =>
                let cx := (lx + ux) * Numbers.half
                let cy := (ly + uy) * Numbers.half
                let u1 := ux * cy + ly * cx - ux * ly
                let u2 := lx * cy + uy * cx - lx * uy
                let aX := if u1 < u2 then ly else uy
                let aY := if u1 < u2 then ux else lx
                let off := if u1 < u2 then (-(ux * ly)) else (-(lx * uy))
                let (aXpos, aXneg) := split aX
                let (aYpos, aYneg) := split aY
                Tensor.scalar (aXpos * cxu + aXneg * cxl + aYpos * cyu + aYneg * cyl + off))
        let A_lo :=
          match xLo.A, xHi.A, yLo.A, yHi.A, Bx.lo, Bx.hi, ByLo, ByHi with
          | .dim AxL, .dim AxU, .dim AyL, .dim AyU, .dim lxF, .dim uxF, .dim lyF, .dim uyF =>
            Tensor.dim (fun i =>
              match lxF i, uxF i, lyF i, uyF i, AxL i, AxU i, AyL i, AyU i with
              | .scalar lx, .scalar ux, .scalar ly, .scalar uy,
                .dim rowXL, .dim rowXU, .dim rowYL, .dim rowYU =>
                -- Choose max of two lower planes at the interval center.
                let cx := (lx + ux) * Numbers.half
                let cy := (ly + uy) * Numbers.half
                let l1 := ux * cy + uy * cx - ux * uy
                let l2 := lx * cy + ly * cx - lx * ly
                let aX := if l1 > l2 then uy else ly     -- coeff for x
                let aY := if l1 > l2 then ux else lx     -- coeff for y
                let (aXpos, aXneg) := split aX
                let (aYpos, aYneg) := split aY
                Tensor.dim (fun j =>
                  match rowXL j, rowXU j, rowYL j, rowYU j with
                  | .scalar xl, .scalar xu, .scalar yl, .scalar yu =>
                    -- For lower bound, negative coeffs use the *upper* input bound.
                    Tensor.scalar (aXpos * xl + aXneg * xu + aYpos * yl + aYneg * yu)))
        let c_lo :=
          match xLo.c, xHi.c, yLo.c, yHi.c, Bx.lo, Bx.hi, ByLo, ByHi with
          | .dim cxL, .dim cxU, .dim cyL, .dim cyU, .dim lxF, .dim uxF, .dim lyF, .dim uyF =>
            Tensor.dim (fun i =>
              match lxF i, uxF i, lyF i, uyF i, cxL i, cxU i, cyL i, cyU i with
              | .scalar lx, .scalar ux, .scalar ly, .scalar uy,
                .scalar cxl, .scalar cxu, .scalar cyl, .scalar cyu =>
                let cx := (lx + ux) * Numbers.half
                let cy := (ly + uy) * Numbers.half
                let l1 := ux * cy + uy * cx - ux * uy
                let l2 := lx * cy + ly * cx - lx * ly
                let aX := if l1 > l2 then uy else ly
                let aY := if l1 > l2 then ux else lx
                let off := if l1 > l2 then (-(ux * uy)) else (-(lx * ly))
                let (aXpos, aXneg) := split aX
                let (aYpos, aYneg) := split aY
                Tensor.scalar (aXpos * cxl + aXneg * cxu + aYpos * cyl + aYneg * cyu + off))

        some
          { inDim := xB.inDim
            outDim := n
            loAff := { A := A_lo, c := c_lo }
            hiAff := { A := A_hi, c := c_hi } }
    else
      none
  else
    none


end NN.MLTheory.CROWN.Graph

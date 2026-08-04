/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.IBP

/-!
# Affine Propagation

This module contains the one-sided affine pass for the flat graph engine. It builds an upper affine
form for each node with respect to a chosen input node. Linear and structural operations preserve
affine dependence. A nonlinear operation uses the upper endpoint of its checked IBP box as a
constant affine bound unless a directed affine rule is available.
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
Context for affine (CROWN/DeepPoly) propagation.

Affine bounds are computed with respect to a single designated *input* node, whose flattened
dimension is `inputDim`.
-/
structure AffineCtx where
  /-- Node id treated as the input variable for affine bounds. -/
  inputId  : Nat
  /-- Flattened input dimension. -/
  inputDim : Nat

/-- Identity affine map on a flattened vector of length `n`. -/
@[expose]
def affIdentity (n : Nat) : AffineVec α n n :=
  let A :=
    Tensor.dim (fun i =>
      Tensor.dim (fun j => Tensor.scalar (if decide (i.val = j.val) then 1 else 0)))
  let c := Spec.fill (α:=α) 0 (.dim n .scalar)
  { A := A, c := c }

/-- Pointwise addition of two affine maps with the same input and output dimensions. -/
def affAdd {n m : Nat} (a1 a2 : AffineVec α n m) : AffineVec α n m :=
  { A := Tensor.addSpec a1.A a2.A, c := Tensor.addSpec a1.c a2.c }

/-- Pointwise subtraction of two affine maps with the same input and output dimensions. -/
def affSub {n m : Nat} (a1 a2 : AffineVec α n m) : AffineVec α n m :=
  { A := Tensor.subSpec a1.A a2.A, c := Tensor.subSpec a1.c a2.c }

-- Affine helpers for linear/matmul are handled by the explicit transfer rules below.

private def affOfLinear (p : LinParams α) : AffineVec α p.n p.m :=
  AffineVec.ofLinear (α:=α) (inDim:=p.n) (outDim:=p.m) p.w p.b

private def affOfMatmul (p : MatParams α) : AffineVec α p.n p.m :=
  let zb := Spec.fill (α:=α) 0 (.dim p.m .scalar)
  AffineVec.ofLinear (α:=α) (inDim:=p.n) (outDim:=p.m) p.w zb

/-- Regard the upper endpoint of a checked box as a constant upper affine bound. -/
private def upperConstAffine (inputDim : Nat) (B : FlatBox α) : FlatAffine α :=
  { inDim := inputDim
    outDim := B.dim
    aff :=
      { A := Spec.fill (α := α) Numbers.zero (.dim B.dim (.dim inputDim .scalar))
        c := B.hi } }

/--
Flatten a typed convolution into the affine map it denotes.

The CROWN pass uses this when a convolution is linear in the selected input. Keeping the conversion
here lets convolution share the same affine machinery as linear and matmul nodes.

Precondition: `cfg.stride ≠ 0`. Engine call sites check this before calling the converter.
-/
def affOfConv2d (cfg : Conv2DParams α) :
  let outH := Spec.Shape.slidingWindowOutDim cfg.inH cfg.kH cfg.stride cfg.padding
  let outW := Spec.Shape.slidingWindowOutDim cfg.inW cfg.kW cfg.stride cfg.padding
  AffineVec α (cfg.inC * cfg.inH * cfg.inW) (cfg.outC * outH * outW) :=
  let outH := Spec.Shape.slidingWindowOutDim cfg.inH cfg.kH cfg.stride cfg.padding
  let outW := Spec.Shape.slidingWindowOutDim cfg.inW cfg.kW cfg.stride cfg.padding
  let inShape := Shape.dim cfg.inC (Shape.dim cfg.inH (Shape.dim cfg.inW Shape.scalar))
  let outShape := Shape.dim cfg.outC (Shape.dim outH (Shape.dim outW Shape.scalar))
  let nIn := inShape.size
  let nOut := outShape.size
  have hIn' : nIn = cfg.inC * (cfg.inH * cfg.inW) := by
    simp [nIn, inShape, Spec.Shape.size]
  have hIn : nIn = cfg.inC * cfg.inH * cfg.inW := by
    simpa [Nat.mul_assoc] using hIn'
  have hOut' : nOut = cfg.outC * (outH * outW) := by
    simp [nOut, outShape, Spec.Shape.size, outH, outW]
  have hOut : nOut = cfg.outC * outH * outW := by
    simpa [Nat.mul_assoc] using hOut'
  let Wraw := NN.MLTheory.CROWN.conv2dLinearMatrix (α:=α)
    (inC:=cfg.inC) (outC:=cfg.outC) (kH:=cfg.kH) (kW:=cfg.kW)
    (stride:=cfg.stride) (padding:=cfg.padding)
    (inH:=cfg.inH) (inW:=cfg.inW) cfg.spec
  let bRaw := NN.MLTheory.CROWN.conv2dBiasBroadcast (α:=α)
    (outC:=cfg.outC) (inH:=cfg.inH) (inW:=cfg.inW)
    (kH:=cfg.kH) (kW:=cfg.kW)
    (stride:=cfg.stride) (padding:=cfg.padding) cfg.spec.bias
  have hShapeW : Shape.dim nOut (Shape.dim nIn Shape.scalar) =
      Shape.dim (cfg.outC * outH * outW) (Shape.dim (cfg.inC * cfg.inH * cfg.inW) Shape.scalar) :=
        by
    simp [hIn, hOut]
  have hShapeB : Shape.dim nOut Shape.scalar = Shape.dim (cfg.outC * outH * outW) Shape.scalar := by
    simp [hOut]
  let W := Spec.tensorCast
    (Shape.dim (cfg.outC * outH * outW) (Shape.dim (cfg.inC * cfg.inH * cfg.inW) Shape.scalar))
    hShapeW Wraw
  let b := Spec.tensorCast
    (Shape.dim (cfg.outC * outH * outW) Shape.scalar)
    hShapeB bRaw
  AffineVec.ofLinear (α:=α)
    (inDim:=cfg.inC * cfg.inH * cfg.inW)
    (outDim:=cfg.outC * outH * outW)
    W b

/--
Propagate a single node’s affine form (CROWN/DeepPoly style) given parent affine forms.

This updates the `affs` array at index `id` when the node kind admits an affine transfer rule.
For non-affine nodes (or missing parents/params), the array is left unchanged so downstream code
can fall back to IBP boxes.
-/
def propagateAffineNode
  (nodes : Array Node) (ps : ParamStore α)
  (ibp : Array (Option (FlatBox α)))
  (affs : Array (Option (FlatAffine α)))
  (ctx : AffineCtx) (id : Nat) : Array (Option (FlatAffine α)) :=
  let node := nodes[id]!
  let getAff (pid : Nat) := (affs[pid]!)
  match node.kind with
  | .input =>
    if node.id = ctx.inputId then
      let aff := affIdentity (α:=α) ctx.inputDim
      affs.set! id (some { inDim := ctx.inputDim, outDim := ctx.inputDim, aff := aff })
    else affs
  | .const _ =>
    -- Lift constant to an affine with zero A and constant c; use ctx.inputDim for input width
    match ps.constVals[id]? with
    | some v =>
      let zA := Spec.fill (α:=α) 0 (.dim v.n (.dim ctx.inputDim .scalar))
      let aff : AffineVec α ctx.inputDim v.n := { A := zA, c := v.v }
      affs.set! id (some { inDim := ctx.inputDim, outDim := v.n, aff := aff })
    | none => affs
  | .detach =>
    match node.parents with
    | p1 :: _ =>
      match getAff p1 with
      | some a => affs.set! id (some a)
      | none => affs
    | _ => affs
  | .randUniform _ | .bernoulliMask _ =>
    -- Stochastic nodes are treated as non-affine; downstream passes can fall back to IBP boxes.
    affs
  | .maxPool2d .. | .avgPool2d .. | .maxPool2dPad .. | .avgPool2dPad .. =>
    -- Pooling is non-affine; downstream passes can fall back to IBP boxes.
    affs
  | .add =>
    match node.parents with
    | p1 :: p2 :: _ =>
      match getAff p1, getAff p2 with
      | some a1, some a2 =>
        if hout : a1.outDim = a2.outDim then
          if hin : a1.inDim = a2.inDim then
            let a2' := castAffineIn (α:=α) (n:=a2.inDim) (n':=a1.inDim) (m:=a2.outDim) hin.symm
              a2.aff
            let a2'' := castAffineOut (α:=α) (n:=a1.inDim) (m:=a2.outDim) (m':=a1.outDim) hout.symm
              a2'
            let outAff := affAdd (α:=α) (n:=a1.inDim) (m:=a1.outDim) a1.aff a2''
            affs.set! id (some { inDim := a1.inDim, outDim := a1.outDim, aff := outAff })
          else affs
        else affs
      | _, _ => affs
    | _ => affs
  | .sub =>
    match node.parents with
    | p1 :: p2 :: _ =>
      match getAff p1, getAff p2 with
      | some a1, some a2 =>
        if hout : a1.outDim = a2.outDim then
          if hin : a1.inDim = a2.inDim then
            let a2' := castAffineIn (α:=α) (n:=a2.inDim) (n':=a1.inDim) (m:=a2.outDim) hin.symm
              a2.aff
            let a2'' := castAffineOut (α:=α) (n:=a1.inDim) (m:=a2.outDim) (m':=a1.outDim) hout.symm
              a2'
            let outAff := affSub (α:=α) (n:=a1.inDim) (m:=a1.outDim) a1.aff a2''
            affs.set! id (some { inDim := a1.inDim, outDim := a1.outDim, aff := outAff })
          else affs
        else affs
      | _, _ => affs
    | _ => affs
  | .relu =>
    match ibp[id]! with
    | some B => affs.set! id (some (upperConstAffine (α := α) ctx.inputDim B))
    | none => affs
  | .linear =>
    match node.parents with
    | p1 :: _ =>
      match getAff p1, ps.linearWB[id]? with
      | some paff, some p =>
        if hdim : paff.outDim = p.n then
          let wbaff0 := affOfLinear (α:=α) p
          let wbaff  := castAffineIn (α:=α) (n:=p.n) (n':=paff.outDim) (m:=p.m) hdim.symm wbaff0
          let composed := AffineVec.compose (α:=α) (n:=paff.inDim) (h:=paff.outDim) (m:=p.m) wbaff
            paff.aff
          affs.set! id (some { inDim := paff.inDim, outDim := p.m, aff := composed })
        else
          affs
      | _, _ => affs
    | _ => affs
  | .matmul =>
    match node.parents with
    | p1 :: _ =>
      match getAff p1, ps.matmulW[id]? with
      | some paff, some p =>
        if hdim : paff.outDim = p.n then
          let waff0 := affOfMatmul (α:=α) p
          let waff  := castAffineIn (α:=α) (n:=p.n) (n':=paff.outDim) (m:=p.m) hdim.symm waff0
          let composed := AffineVec.compose (α:=α) (n:=paff.inDim) (h:=paff.outDim) (m:=p.m) waff
            paff.aff
          affs.set! id (some { inDim := paff.inDim, outDim := p.m, aff := composed })
        else
          affs
      | _, _ => affs
    | _ => affs
  | .sum =>
    match node.parents with
    | p1 :: _ =>
      match getAff p1 with
      | some paff =>
        let onesRow : Tensor α (.dim 1 (.dim paff.outDim .scalar)) :=
          Spec.fill (α := α) Numbers.one (.dim 1 (.dim paff.outDim .scalar))
        let outAff : AffineVec α paff.inDim 1 :=
          { A := Spec.matMulSpec onesRow paff.aff.A
            c := Spec.matVecMulSpec onesRow paff.aff.c }
        affs.set! id (some { inDim := paff.inDim, outDim := 1, aff := outAff })
      | none => affs
    | _ => affs
  | .reshape _ _ => affs
  | .flatten _ => affs
  | .swap_first_two => affs
  | .transpose3dLastTwo => affs
  | .permute _ => affs
  | .mseLoss => affs
  | .mul_elem =>
    match node.parents with
    | p1 :: p2 :: _ =>
      match getAff p1, getAff p2, ibp[p1]!, ibp[p2]! with
      | some ax, some ay, some Bx, some By =>
        -- Require matching output dims and input dims; otherwise skip
        if hout : ax.outDim = ay.outDim then
          if hin : ax.inDim = ay.inDim then
            if hbx : Bx.dim = ax.outDim then
              if hby : By.dim = ay.outDim then
                let ayOut := castAffineOut (α:=α) (n:=ay.inDim) (m:=ay.outDim) (m':=ax.outDim)
                  (h:=hout.symm) ay.aff
                let ayAligned := castAffineIn (α:=α) (n:=ay.inDim) (n':=ax.inDim) (m:=ax.outDim)
                  (h:=hin.symm) ayOut
                let bxBox := castBoxDim (α:=α) (n:=Bx.dim) (n':=ax.outDim) (h:=hbx) (ofFlatBox Bx)
                -- align By box dim to ax.outDim via ay.outDim using hout
                let hby2 : By.dim = ax.outDim := Eq.trans hby hout.symm
                let byBox := castBoxDim (α:=α) (n:=By.dim) (n':=ax.outDim) (h:=hby2) (ofFlatBox By)
                -- McCormick upper affine envelope per component i
                let A' :=
                  match ax.aff.A, ayAligned.A, bxBox.lo, bxBox.hi, byBox.lo, byBox.hi with
                  | .dim rowsX, .dim rowsY, .dim lox, .dim hix, .dim loy, .dim hiy =>
                    Tensor.dim (fun i =>
                      let rowX := rowsX i
                      let rowY := rowsY i
                      match rowX, rowY, lox i, hix i, loy i, hiy i with
                      | .dim colsX, .dim colsY,
                        .scalar lx, .scalar ux,
                        .scalar ly, .scalar uy =>
                        let cx := (lx + ux) * Numbers.pointfive
                        let cy := (ly + uy) * Numbers.pointfive
                        let u1_center := ux * cy + ly * cx - ux * ly
                        let u2_center := lx * cy + uy * cx - lx * uy
                        let sX := if u1_center < u2_center then ly else uy
                        let sY := if u1_center < u2_center then ux else lx
                        Tensor.dim (fun j =>
                          match colsX j, colsY j with
                          | .scalar aijx, .scalar aijy => Tensor.scalar (sX * aijx + sY * aijy)))
                let c' :=
                  match ax.aff.c, ayAligned.c, bxBox.lo, bxBox.hi, byBox.lo, byBox.hi with
                  | .dim cxv, .dim cyv, .dim lox, .dim hix, .dim loy, .dim hiy =>
                    Tensor.dim (fun i =>
                      match cxv i, cyv i, lox i, hix i, loy i, hiy i with
                      | .scalar cxi, .scalar cyi,
                        .scalar lx, .scalar ux,
                        .scalar ly, .scalar uy =>
                        let cx := (lx + ux) * Numbers.pointfive
                        let cy := (ly + uy) * Numbers.pointfive
                        let u1_center := ux * cy + ly * cx - ux * ly
                        let u2_center := lx * cy + uy * cx - lx * uy
                        let sX := if u1_center < u2_center then ly else uy
                        let sY := if u1_center < u2_center then ux else lx
                        let off := if u1_center < u2_center then (-(ux * ly)) else (-(lx * uy))
                        Tensor.scalar (sX * cxi + sY * cyi + off))
                let outAff : AffineVec α ax.inDim ax.outDim := { A := A', c := c' }
                affs.set! id (some { inDim := ax.inDim, outDim := ax.outDim, aff := outAff })
              else affs
            else affs
          else affs
        else affs
      | _, _, _, _ => affs
    | _ => affs
  | .conv2d .. =>
    match node.parents with
    | p1 :: _ =>
      match getAff p1, ps.conv2dCfg[id]? with
      | some paff, some cfg =>
        let convIn := cfg.inC * cfg.inH * cfg.inW
        if _hs : cfg.stride = 0 then
          affs
        else if hdim : paff.outDim = convIn then
          let outH := Spec.Shape.slidingWindowOutDim cfg.inH cfg.kH cfg.stride cfg.padding
          let outW := Spec.Shape.slidingWindowOutDim cfg.inW cfg.kW cfg.stride cfg.padding
          let convAff0 := affOfConv2d (α:=α) cfg
          let convAff := castAffineIn (α:=α)
            (n:=convIn) (n':=paff.outDim) (m:=cfg.outC * outH * outW)
            hdim.symm convAff0
          let composed := AffineVec.compose (α:=α)
            (n:=paff.inDim) (h:=paff.outDim) (m:=cfg.outC * outH * outW)
            convAff paff.aff
          affs.set! id (some { inDim := paff.inDim, outDim := cfg.outC * outH * outW, aff :=
            composed })
        else affs
      | some paff, none =>
        match ps.linearWB[id]? with
        | some p =>
          if hdim : paff.outDim = p.n then
            let wbaff0 := affOfLinear (α:=α) p
            let wbaff := castAffineIn (α:=α) (n:=p.n) (n':=paff.outDim) (m:=p.m) hdim.symm wbaff0
            let composed := AffineVec.compose (α:=α) (n:=paff.inDim) (h:=paff.outDim) (m:=p.m) wbaff
              paff.aff
            affs.set! id (some { inDim := paff.inDim, outDim := p.m, aff := composed })
          else affs
        | none => affs
      | _, _ => affs
    | _ => affs
  | .batchNorm2dNchwEval .. =>
    match node.parents with
    | p1 :: _ =>
      match getAff p1, ps.batchNorm2dNchwEval[id]? with
      | some paff, some cfg =>
        match batchNorm2dNchwEvalLinear? (α := α) nodes[p1]!.outShape cfg with
        | some p =>
          if hdim : paff.outDim = p.n then
            let bnAff0 := affOfLinear (α := α) p
            let bnAff := castAffineIn (α := α) (n := p.n) (n' := paff.outDim) (m := p.m)
              hdim.symm bnAff0
            let composed := AffineVec.compose (α := α)
              (n := paff.inDim) (h := paff.outDim) (m := p.m) bnAff paff.aff
            affs.set! id (some { inDim := paff.inDim, outDim := p.m, aff := composed })
          else
            affs
        | none => affs
      | _, _ => affs
    | _ => affs
  | .exp | .log | .softmax _ | .hardMaskedSoftmax _ =>
    match ibp[id]! with
    | some B => affs.set! id (some (upperConstAffine (α := α) ctx.inputDim B))
    | none => affs
  | .layernorm _ =>
    match ibp[id]! with
    | some B => affs.set! id (some (upperConstAffine (α := α) ctx.inputDim B))
    | none => affs
  | .concat axis =>
    -- Concatenation on axis zero stacks the flattened output rows.
    -- For other axes/shapes, this requires stride-aware flatten/reshape bookkeeping.
    if axis != 0 then affs
    else
      let rec collect (ps : List Nat) (acc : List (FlatAffine α)) : Option (List (FlatAffine α)) :=
        match ps with
        | [] => some acc.reverse
        | p :: ps =>
          match getAff p with
          | some a => collect ps (a :: acc)
          | none => none
      match collect node.parents [] with
      | none => affs
      | some parentsAff =>
        match parentsAff with
        | [] => affs
        | first :: rest =>
          let inDim := first.inDim
          if rest.all (fun a => a.inDim == inDim) then
            let totalOut := parentsAff.foldl (fun acc a => acc + a.outDim) 0
            if Spec.Shape.size node.outShape = totalOut then
              let rec pick (k : Nat) (l : List (FlatAffine α)) : FlatAffine α × Nat :=
                match l with
                | [] => (first, 0)
                | a :: tl =>
                  if k < a.outDim then (a, k) else pick (k - a.outDim) tl
              let A' : Tensor α (.dim totalOut (.dim inDim .scalar)) :=
                Tensor.dim (fun i =>
                  let (a, k) := pick i.val parentsAff
                  Tensor.dim (fun j => Tensor.scalar (getAtOrZero a.aff.A [k, j.val])))
              let c' : Tensor α (.dim totalOut .scalar) :=
                Tensor.dim (fun i =>
                  let (a, k) := pick i.val parentsAff
                  Tensor.scalar (getAtOrZero a.aff.c [k]))
              let outAff : AffineVec α inDim totalOut := { A := A', c := c' }
              affs.set! id (some { inDim := inDim, outDim := totalOut, aff := outAff })
            else affs
          else affs
  | .abs | .sqrt | .inv | .maxElem | .minElem | .broadcastTo .. | .reduceSum .. | .reduceMean ..
  | .tanh | .sin | .cos | .sigmoid =>
    match ibp[id]! with
    | some B => affs.set! id (some (upperConstAffine (α := α) ctx.inputDim B))
    | none => affs

/--
Run the one-sided affine pass.

Linear nodes keep their affine dependence on the selected input. Nonlinear nodes use their checked
IBP upper endpoint as a constant affine bound unless this pass has a separately justified rule.
-/
def runAffine (g : Graph) (ps : ParamStore α) (ctx : AffineCtx) (ibp : Array (Option (FlatBox α))) :
  Array (Option (FlatAffine α)) :=
  let init := Array.replicate g.nodes.size none
  (List.finRange g.nodes.size).foldl (fun acc i => propagateAffineNode (α:=α) g.nodes ps ibp acc ctx
    i) init

end NN.MLTheory.CROWN.Graph

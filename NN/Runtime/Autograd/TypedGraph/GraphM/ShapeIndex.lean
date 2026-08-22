/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TypedGraph.GraphM.Pooling

/-!
# GraphM Shape And Indexing Ops

Reshape, transpose, broadcast, reduction, gather, and scatter builders for typed graphs.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace TypedGraph
namespace GraphM

open Spec
open Tensor
open Proofs.Autograd.Algebra
open Runtime.Autograd.TorchLean

/--
Flatten a tensor to a 1D vector (preserving total size).

PyTorch comparison: `torch.flatten(x)` (for a single tensor value).
-/
def flatten {α : Type} {Δ : Type} [Inhabited α] [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var (.dim (Spec.Shape.size s) .scalar)) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let outS : Shape := .dim (Spec.Shape.size s) .scalar
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d => flattenSpec (α := α) (s := s) (getIdx (α := α) (xs := ctx) ix)
      jvp := fun _ctx dctx _d =>
        flattenSpec (α := α) (s := s) (getIdx (α := α) (xs := dctx) ix)
      vjp := fun _ctx _d δ =>
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix (unflattenSpec (α := α) s δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
Reshape a tensor, given a proof that the total sizes match.

PyTorch comparison: `torch.reshape(x, new_shape)`.
-/
def reshape {α : Type} {Δ : Type} [Inhabited α] [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {s₁ s₂ : Shape} (x : Var s₁) (h : Spec.Shape.size s₁ = Spec.Shape.size s₂) :
    MWith α Δ Γ (Var s₂) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s₂ :=
    { forward := fun ctx _d =>
        Spec.Tensor.reshapeSpec (α := α) (s₁ := s₁) (s₂ := s₂) (getIdx (α := α) (xs := ctx) ix) h
      jvp := fun _ctx dctx _d =>
        Spec.Tensor.reshapeSpec (α := α) (s₁ := s₁) (s₂ := s₂) (getIdx (α := α) (xs := dctx) ix) h
      vjp := fun _ctx _d δ =>
        TList.single (α := α) (Γ := Γ ++ ss) (s := s₁) ix
          (Spec.Tensor.reshapeSpec (α := α) (s₁ := s₂) (s₂ := s₁) δ h.symm) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s₂) g node

/-- Transpose a 2D matrix. PyTorch comparison: `x.transpose(0, 1)` / `x.T` for matrices. -/
def transpose2d {α : Type} {Δ : Type} [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {m n : Nat} (x : Var (.dim m (.dim n .scalar))) :
    MWith α Δ Γ (Var (.dim n (.dim m .scalar))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let outS : Shape := .dim n (.dim m .scalar)
  let inS : Shape := .dim m (.dim n .scalar)
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        matrixTransposeSpec (α := α) (getIdx (α := α) (xs := ctx) ix)
      jvp := fun _ctx dctx _d =>
        matrixTransposeSpec (α := α) (getIdx (α := α) (xs := dctx) ix)
      vjp := fun _ctx _d δ =>
        TList.single (α := α) (Γ := Γ ++ ss) (s := inS) ix (matrixTransposeSpec (α := α) δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
Transpose a rank-3 tensor by moving the first axis to the last (`(a,b,c) → (b,c,a)`).

PyTorch comparison: `x.permute(1, 2, 0)`.
-/
def transpose3dFirstToLast {α : Type} {Δ : Type} [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {a b c : Nat} (x : Var (.dim a (.dim b (.dim c .scalar)))) :
    MWith α Δ Γ (Var (.dim b (.dim c (.dim a .scalar)))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let inS : Shape := .dim a (.dim b (.dim c .scalar))
  let outS : Shape := .dim b (.dim c (.dim a .scalar))
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        Spec.Tensor.transpose3DFirstToLastSpec (α := α) (a := a) (b := b) (c := c)
          (getIdx (α := α) (xs := ctx) ix)
      jvp := fun _ctx dctx _d =>
        Spec.Tensor.transpose3DFirstToLastSpec (α := α) (a := a) (b := b) (c := c)
          (getIdx (α := α) (xs := dctx) ix)
      vjp := fun _ctx _d δ =>
        TList.single (α := α) (Γ := Γ ++ ss) (s := inS) ix
          (Spec.Tensor.transpose3DLastToFirstSpec (α := α) (a := b) (b := c) (c := a) δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
Transpose a rank-3 tensor by moving the last axis to the first (`(a,b,c) → (c,a,b)`).

PyTorch comparison: `x.permute(2, 0, 1)`.
-/
def transpose3dLastToFirst {α : Type} {Δ : Type} [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {a b c : Nat} (x : Var (.dim a (.dim b (.dim c .scalar)))) :
    MWith α Δ Γ (Var (.dim c (.dim a (.dim b .scalar)))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let inS : Shape := .dim a (.dim b (.dim c .scalar))
  let outS : Shape := .dim c (.dim a (.dim b .scalar))
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        Spec.Tensor.transpose3DLastToFirstSpec (α := α) (a := a) (b := b) (c := c)
          (getIdx (α := α) (xs := ctx) ix)
      jvp := fun _ctx dctx _d =>
        Spec.Tensor.transpose3DLastToFirstSpec (α := α) (a := a) (b := b) (c := c)
          (getIdx (α := α) (xs := dctx) ix)
      vjp := fun _ctx _d δ =>
        TList.single (α := α) (Γ := Γ ++ ss) (s := inS) ix
          (Spec.Tensor.transpose3DFirstToLastSpec (α := α) (a := c) (b := a) (c := b) δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
Swap the last two axes of a rank-3 tensor (`(a,b,c) → (a,c,b)`).

PyTorch comparison: `x.transpose(1, 2)` for a 3D tensor.
-/
def transpose3dLastTwo {α : Type} {Δ : Type} [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {a b c : Nat} (x : Var (.dim a (.dim b (.dim c .scalar)))) :
    MWith α Δ Γ (Var (.dim a (.dim c (.dim b .scalar)))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let inS : Shape := .dim a (.dim b (.dim c .scalar))
  let outS : Shape := .dim a (.dim c (.dim b .scalar))
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        Spec.Tensor.transpose3DLastTwoSpec (α := α) (a := a) (b := b) (c := c)
          (getIdx (α := α) (xs := ctx) ix)
      jvp := fun _ctx dctx _d =>
        Spec.Tensor.transpose3DLastTwoSpec (α := α) (a := a) (b := b) (c := c)
          (getIdx (α := α) (xs := dctx) ix)
      vjp := fun _ctx _d δ =>
        TList.single (α := α) (Γ := Γ ++ ss) (s := inS) ix
          (Spec.Tensor.transpose3DLastTwoSpec (α := α) (a := a) (b := c) (c := b) δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
  Swap two adjacent axes at a given nesting `depth`.

  This is the typed-graph analogue of the eager `Tape.swapAdjacentAtDepth`.
  PyTorch comparison: a `permute` that swaps two neighboring dimensions.
  -/
  def swapAdjacentAtDepth {α : Type} {Δ : Type} [Zero α] [DecidableEq Shape]
    {Γ : List Shape} {s : Shape} (depth : Nat) (x : Var s) :
      MWith α Δ Γ (Var (s.swapAdjacentAtDepth depth)) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let outS : Shape := s.swapAdjacentAtDepth depth
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        Spec.Tensor.swapAdjacentAxes (tensor := getIdx (α := α) (xs := ctx) ix) depth
      jvp := fun _ctx dctx _d =>
        let dx := getIdx (α := α) (xs := dctx) ix
        Spec.Tensor.swapAdjacentAxes (tensor := dx) depth
      vjp := fun _ctx _d δ =>
        let dx' := Spec.Tensor.swapAdjacentAxes (tensor := δ) depth
        let dx : Tensor α s :=
          Tensor.castShape dx' (by simpa [outS] using (Spec.Shape.swapAdjacentAtDepth_involutive s
            depth))
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix dx }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
Broadcast `x : s₁` to a larger shape `s₂` (given a `CanBroadcastTo` witness).

PyTorch comparison: `x.expand(...)` / broadcasting semantics in elementwise ops.
-/
def broadcastTo {α : Type} {Δ : Type} [Inhabited α] [Add α] [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {s₁ s₂ : Shape} (cb : Shape.CanBroadcastTo s₁ s₂) (x : Var s₁) :
  MWith α Δ Γ (Var s₂) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s₂ :=
    { forward := fun ctx _d =>
        Spec.Tensor.broadcastTo (α := α) cb (getIdx (α := α) (xs := ctx) ix)
      jvp := fun _ctx dctx _d =>
        Spec.Tensor.broadcastTo (α := α) cb (getIdx (α := α) (xs := dctx) ix)
      vjp := fun _ctx _d δ =>
        TList.single (α := α) (Γ := Γ ++ ss) (s := s₁) ix
          (Spec.Tensor.reduceFromBroadcastTo (α := α) cb δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s₂) g node

/--
Reduce-sum along a given `axis`.

PyTorch comparison: `torch.sum(x, dim=axis)`.
-/
def reduceSum {α : Type} {Δ : Type} [Add α] [Zero α] [Inhabited α] [DecidableEq Shape]
  {Γ : List Shape} {s : Shape} (axis : Nat)
  [_valid : Shape.HasNonemptyAxis axis s] [_wf : Shape.WellFormed s]
  (x : Var s) : MWith α Δ Γ (Var (shapeAfterSum s axis)) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let outS : Shape := shapeAfterSum s axis
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        Spec.Tensor.reduceSum (α := α) (s := s) axis
          (getIdx (α := α) (xs := ctx) ix) _valid.proof
      jvp := fun _ctx dctx _d =>
        Spec.Tensor.reduceSum (α := α) (s := s) axis
          (getIdx (α := α) (xs := dctx) ix) _valid.proof
      vjp := fun _ctx _d δ =>
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix
          (Spec.Tensor.broadcastAfterSum s axis δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
Reduce-mean along a given `axis`.

PyTorch comparison: `torch.mean(x, dim=axis)`.
-/
def reduceMean {α : Type} {Δ : Type} [Context α] [DecidableEq Shape]
  {Γ : List Shape} {s : Shape} (axis : Nat)
  [valid : Shape.HasNonemptyAxis axis s] [_wf : Shape.WellFormed s]
  (x : Var s) : MWith α Δ Γ (Var (shapeAfterSum s axis)) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let outS : Shape := shapeAfterSum s axis
  letI : Shape.AxisInBounds axis s := valid.proof.toAxisInBounds
  let denomNat := Shape.axisSize s axis
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        let xv := getIdx (α := α) (xs := ctx) ix
        let h := valid.proof
        Spec.Tensor.reduceMean (α := α) (s := s) axis xv h
      jvp := fun _ctx dctx _d =>
        let dx := getIdx (α := α) (xs := dctx) ix
        let h := valid.proof
        Spec.Tensor.reduceMean (α := α) (s := s) axis dx h
      vjp := fun _ctx _d δ =>
        let dLdx := Spec.Tensor.broadcastAfterSum s axis δ
        let dLdx' := scaleSpec (α := α) (s := s) dLdx (1 / (denomNat : α))
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix dLdx' }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
  Gather a single scalar from a vector at a known-in-bounds index.

  PyTorch comparison: `x[i]` for a 1D tensor.
  -/
  def gatherScalar {α : Type} {Δ : Type} [Zero α] [DecidableEq Shape]
    {Γ : List Shape} {n : Nat} (x : Var (.dim n .scalar)) (i : Fin n) : MWith α Δ Γ (Var
      Shape.scalar) := do
    let ⟨ss, g⟩ ← get
    let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
    let node : NodeData α Δ (Γ ++ ss) Shape.scalar :=
      { forward := fun ctx _d =>
          getAtSpec (getIdx (α := α) (xs := ctx) ix) i
        jvp := fun _ctx dctx _d =>
          getAtSpec (getIdx (α := α) (xs := dctx) ix) i
        vjp := fun _ctx _d δ =>
          let gVal : α := Tensor.item δ
          let dx : Tensor α (.dim n .scalar) :=
            Tensor.dim (fun j => Tensor.scalar (if decide (j = i) then gVal else 0))
          TList.single (α := α) (Γ := Γ ++ ss) (s := .dim n .scalar) ix dx }
    push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := Shape.scalar) g node

/--
  Gather a row from a matrix at a known-in-bounds row index.

  PyTorch comparison: `x[i, :]` for a 2D tensor.
  -/
def gatherRow {α : Type} {Δ : Type} [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {rows cols : Nat} (x : Var (.dim rows (.dim cols .scalar))) (i : Fin rows) :
  MWith α Δ Γ (Var (.dim cols .scalar)) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let outS : Shape := .dim cols .scalar
  let inS : Shape := .dim rows (.dim cols .scalar)
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        getAtSpec (getIdx (α := α) (xs := ctx) ix) i
      jvp := fun _ctx dctx _d =>
        getAtSpec (getIdx (α := α) (xs := dctx) ix) i
      vjp := fun _ctx _d δ =>
        let dx : Tensor α inS :=
          Tensor.dim (fun r =>
            if decide (r = i) then
              δ
            else
              fill (0 : α) outS)
        TList.single (α := α) (Γ := Γ ++ ss) (s := inS) ix dx }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
  Gather a scalar from a vector at a runtime `Nat` index.

  If `i` is out of bounds we return `0` and propagate no gradient (matching the forward choice).
  -/
  def gatherScalarNatOrZero {α : Type} {Δ : Type} [Zero α] [DecidableEq Shape]
    {Γ : List Shape} {n : Nat} (x : Var (.dim n .scalar)) (i : Nat) :
    MWith α Δ Γ (Var Shape.scalar) := do
    let ⟨ss, g⟩ ← get
    let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
    let node : NodeData α Δ (Γ ++ ss) Shape.scalar :=
      { forward := fun ctx _d =>
          let xv := getIdx (α := α) (xs := ctx) ix
          if h : i < n then
            getAtSpec xv ⟨i, h⟩
          else
            Tensor.scalar 0
        jvp := fun _ctx dctx _d =>
          let dx := getIdx (α := α) (xs := dctx) ix
          if h : i < n then
            getAtSpec dx ⟨i, h⟩
          else
            Tensor.scalar 0
        vjp := fun _ctx _d δ =>
          let gVal : α := Tensor.item δ
          let dx : Tensor α (.dim n .scalar) :=
            Tensor.dim (fun j =>
            if _hi : i < n then
              Tensor.scalar (if decide (j.val = i) then gVal else 0)
            else
              Tensor.scalar 0)
          TList.single (α := α) (Γ := Γ ++ ss) (s := .dim n .scalar) ix dx }
    push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := Shape.scalar) g node

/--
  Gather a vector of length `k` from a length-`n` vector using an index tensor of `Nat`s.

  Out-of-bounds indices yield `0` at the corresponding output position.

  PyTorch comparison: `torch.gather` for 1D inputs, with explicit bounds handling.
  -/
def gatherVecNatOrZero {α : Type} {Δ : Type} [Add α] [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {n k : Nat} (x : Var (.dim n .scalar))
  (idx : Δ → Tensor Nat (.dim k .scalar)) :
  MWith α Δ Γ (Var (.dim k .scalar)) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let outS : Shape := .dim k .scalar
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx d =>
        let xv := getIdx (α := α) (xs := ctx) ix
        match idx d with
        | Tensor.dim f =>
            Tensor.dim (fun j =>
              match f j with
              | Tensor.scalar ij =>
                  if h : ij < n then
                    getAtSpec xv ⟨ij, h⟩
                  else
                    Tensor.scalar 0)
      jvp := fun _ctx dctx d =>
        let dx := getIdx (α := α) (xs := dctx) ix
        match idx d with
        | Tensor.dim f =>
            Tensor.dim (fun j =>
              match f j with
              | Tensor.scalar ij =>
                  if h : ij < n then
                    getAtSpec dx ⟨ij, h⟩
                  else
                    Tensor.scalar 0)
      vjp := fun _ctx d δ =>
        let dx : Tensor α (.dim n .scalar) :=
          Tensor.dim (fun iFin =>
            let sum : α :=
              (List.finRange k).foldl (fun acc j =>
                let ij :=
                  match idx d with
                  | Tensor.dim f =>
                      match f j with
                      | Tensor.scalar v => v
                if _hij : ij < n then
                  if decide (ij = iFin.val) then
                    let gj : α :=
                      match getAtSpec δ j with
                      | Tensor.scalar v => v
                    acc + gj
                  else acc
                else acc
              ) 0
            Tensor.scalar sum)
        TList.single (α := α) (Γ := Γ ++ ss) (s := .dim n .scalar) ix dx }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
  Gather `k` rows from a `(rows×cols)` matrix using an index vector of `Nat`s.

  Out-of-bounds indices yield a zero row.

  PyTorch comparison: `torch.index_select(x, dim=0, index=idx)` with explicit bounds handling.
  -/
def gatherRowsNatOrZero {α : Type} {Δ : Type} [Add α] [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {rows cols k : Nat} (x : Var (.dim rows (.dim cols .scalar)))
  (idx : Δ → Tensor Nat (.dim k .scalar)) :
  MWith α Δ Γ (Var (.dim k (.dim cols .scalar))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let outS : Shape := .dim k (.dim cols .scalar)
  let inS : Shape := .dim rows (.dim cols .scalar)
  let rowS : Shape := .dim cols .scalar
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx d =>
        let xv := getIdx (α := α) (xs := ctx) ix
        match idx d with
        | Tensor.dim f =>
            Tensor.dim (fun j =>
              match f j with
              | Tensor.scalar ij =>
                  if h : ij < rows then
                    getAtSpec xv ⟨ij, h⟩
                  else
                    fill (0 : α) rowS)
      jvp := fun _ctx dctx d =>
        let dx0 := getIdx (α := α) (xs := dctx) ix
        match idx d with
        | Tensor.dim f =>
            Tensor.dim (fun j =>
              match f j with
              | Tensor.scalar ij =>
                  if h : ij < rows then
                    getAtSpec dx0 ⟨ij, h⟩
                  else
                    fill (0 : α) rowS)
      vjp := fun _ctx d δ =>
        let dx : Tensor α inS :=
          Tensor.dim (fun rFin =>
            (List.finRange k).foldl (fun acc j =>
              let ij :=
                match idx d with
                | Tensor.dim f =>
                    match f j with
                    | Tensor.scalar v => v
              if _hij : ij < rows then
                if decide (ij = rFin.val) then
                  addSpec acc (getAtSpec δ j)
                else
                  acc
              else
                acc
            ) (fill (0 : α) rowS))
        TList.single (α := α) (Γ := Γ ++ ss) (s := inS) ix dx }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
Scatter-add into a vector at a single in-bounds index.

`scatter_add_vec x v i` adds the scalar `v` into `x[i]`.

PyTorch comparison: `x.index_add_(dim=0, index=[i], source=[v])` (conceptually).
-/
def scatterAddVec {α : Type} {Δ : Type} [Add α] [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {n : Nat} (x : Var (.dim n .scalar)) (v : Var Shape.scalar) (i : Fin n) :
  MWith α Δ Γ (Var (.dim n .scalar)) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let iv ← liftM (mkIdx (_α := α) (Γ := Γ) ss v)
  let outS : Shape := .dim n .scalar
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        let xv := getIdx (α := α) (xs := ctx) ix
        let vv : α := Tensor.item (getIdx (α := α) (xs := ctx) iv)
        let xi : α := Tensor.item (getAtSpec xv i)
        updateSpec xv [i.val] (xi + vv)
      jvp := fun _ctx dctx _d =>
        let dx := getIdx (α := α) (xs := dctx) ix
        let dv : α := Tensor.item (getIdx (α := α) (xs := dctx) iv)
        let dxi : α := Tensor.item (getAtSpec dx i)
        updateSpec dx [i.val] (dxi + dv)
      vjp := fun _ctx _d δ =>
        let dv : Tensor α Shape.scalar := getAtSpec δ i
        TList.add (α := α) (ss := Γ ++ ss)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := outS) ix δ)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := Shape.scalar) iv dv) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
Scatter-add into a matrix at a single in-bounds row index.

`scatter_add_row x v i` adds the row vector `v` into `x[i, :]`.

PyTorch comparison: `x.index_add_(dim=0, index=[i], source=v.unsqueeze(0))` (conceptually).
-/
def scatterAddRow {α : Type} {Δ : Type} [Add α] [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {rows cols : Nat}
  (x : Var (.dim rows (.dim cols .scalar))) (v : Var (.dim cols .scalar)) (i : Fin rows) :
  MWith α Δ Γ (Var (.dim rows (.dim cols .scalar))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let iv ← liftM (mkIdx (_α := α) (Γ := Γ) ss v)
  let outS : Shape := .dim rows (.dim cols .scalar)
  let rowS : Shape := .dim cols .scalar
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        let xv := getIdx (α := α) (xs := ctx) ix
        let vv := getIdx (α := α) (xs := ctx) iv
        Tensor.dim (fun r =>
          if decide (r = i) then
            addSpec (getAtSpec xv r) vv
          else
            getAtSpec xv r)
      jvp := fun _ctx dctx _d =>
        let dx := getIdx (α := α) (xs := dctx) ix
        let dv := getIdx (α := α) (xs := dctx) iv
        Tensor.dim (fun r =>
          if decide (r = i) then
            addSpec (getAtSpec dx r) dv
          else
            getAtSpec dx r)
      vjp := fun _ctx _d δ =>
        let dv : Tensor α rowS := getAtSpec δ i
        TList.add (α := α) (ss := Γ ++ ss)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := outS) ix δ)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := rowS) iv dv) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

end GraphM
end TypedGraph
end Autograd
end Runtime

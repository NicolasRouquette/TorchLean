/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Check
public import NN.IR.HardMask
public import NN.MLTheory.CROWN.Graph
public import NN.Runtime.Autograd.TorchLean.Program

/-!
# Verifier IR Lowering

TorchLean → verifier bridge.

This module lets you take a TorchLean `Program` (written once over the `TorchLean.Ops` interface)
and lower it into the op-tagged IR (`NN.IR.Graph`) plus a CROWN/LiRPA-style `ParamStore`.

Why this exists:
- TorchLean programs can use eager or typed graph execution for training, while the same computation
  graph
  is also useful as a *verification artifact*.
- By compiling to an explicit DAG IR, we can run bound propagation (IBP, CROWN/DeepPoly variants)
  and certificate checkers on the exact same model definition.

Scope:
- targets the forward-only TorchLean fragment used by bound-propagation checkers;
- supports a curated operator set (arithmetic, shape ops, common nonlinearities, pooling, `linear`,
  `conv2d`) plus a few composite ops lowered to IR subgraphs (e.g. `layer_norm`,
  `multi_head_attention`);
- ops outside the verifier fragment throw with an explicit error (notably: general Nat-indexed gather/scatter, and
  training-style BatchNorm).

Related tooling:
- PyTorch interop lives under `NN.Runtime.PyTorch.Import.*` and is used by many examples to import
  weights or certificates produced by Python scripts.

References (informal):
- IBP: Gowal et al. (2018).
- CROWN / DeepPoly-style linear relaxations: Zhang et al. (2018).
- LiRPA unification viewpoint: Xu et al. (2020).
-/

@[expose] public section


namespace NN.Verification.TorchLean

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

/-! ## IR builder -/

/--
Reference produced while lowering a TorchLean program.

A value is either already materialized as an IR node, or it is still a compile-time tensor constant
that can be inserted into the verifier `ParamStore` if a later operation needs a node parent.
-/
inductive Ref (α : Type) : Shape → Type where
  | node  {s : Shape} (id : Nat) : Ref α s
  | const {s : Shape} (t : Tensor α s) : Ref α s

/-- Mutable builder state for translating a TorchLean program into verifier IR. -/
structure BuildState (α : Type) [Context α] where
  /-- IR nodes emitted so far, in construction/topological order. -/
  nodes : Array Node := #[]
  /-- Parameter payload accumulated for constant tensors and layer weights. -/
  ps    : NN.MLTheory.CROWN.Graph.ParamStore α := {}

/-- Builder monad used by TorchLean-to-IR lowering. -/
abbrev BuildM (α : Type) [Context α] : Type → Type :=
  StateT (BuildState α) (Except String)

/-- Raise a lowering error inside `BuildM`. -/
def fail {α : Type} [Context α] {β : Type} (msg : String) : BuildM α β :=
  throw msg

/-- Run a shared IR shape contract and surface its error from lowering. -/
def requireContract {α : Type} [Context α] {β : Type} (r : Except String β) : BuildM α Unit := do
  match r with
  | .ok _ => pure ()
  | .error msg => fail (α := α) msg

/-- Append a freshly constructed IR node to the builder state. -/
def pushNode {α : Type} [Context α] (n : Node) : BuildM α Unit := do
  modify fun st => { st with nodes := st.nodes.push n }

/-- Return the next node identifier, which is the current node-array size. -/
def freshId {α : Type} [Context α] : BuildM α Nat := do
  pure (←get).nodes.size

/--
Ensure a `Ref` is represented by an IR node.

Compile-time constants are materialized as `.const` nodes and recorded in the verifier
`ParamStore`; existing graph nodes are returned unchanged.
-/
def ensureNode {α : Type} [Context α]
    {s : Shape} (r : Ref α s) : BuildM α Nat := do
  match r with
  | .node id => pure id
  | .const t =>
      let id ← freshId (α := α)
      let flat := Tensor.flattenSpec (α := α) t
      let fv : NN.MLTheory.CROWN.Graph.FlatVec α := { n := Spec.Shape.size s, v := flat }
      let node : Node := { id := id, parents := [], kind := .const s, outShape := s }
      modify fun st => { st with ps := { st.ps with constVals := st.ps.constVals.insert id fv } }
      pushNode (α := α) node
      pure id

/-- Emit a unary IR operation with one parent node. -/
def emitUnary {α : Type} [Context α]
    {s t : Shape} (kind : OpKind) (x : Ref α s) (outShape : Shape := t) : BuildM α (Ref α t) := do
  let pid ← ensureNode (α := α) x
  let id ← freshId (α := α)
  let node : Node := { id := id, parents := [pid], kind := kind, outShape := outShape }
  pushNode (α := α) node
  pure (.node id)

/-- Emit a binary IR operation whose operands have the same shape. -/
def emitBinary {α : Type} [Context α]
    {s : Shape} (kind : OpKind) (a b : Ref α s) : BuildM α (Ref α s) := do
  let pa ← ensureNode (α := α) a
  let pb ← ensureNode (α := α) b
  let id ← freshId (α := α)
  let node : Node := { id := id, parents := [pa, pb], kind := kind, outShape := s }
  pushNode (α := α) node
  pure (.node id)

/-- Emit a matrix-multiplication IR node. -/
def emitMatmul {α : Type} [Context α]
    {sA sB sOut : Shape} (a : Ref α sA) (b : Ref α sB) (outShape : Shape := sOut) :
    BuildM α (Ref α sOut) := do
  let pa ← ensureNode (α := α) a
  let pb ← ensureNode (α := α) b
  let id ← freshId (α := α)
  let node : Node := { id := id, parents := [pa, pb], kind := .matmul, outShape := outShape }
  pushNode (α := α) node
  pure (.node id)

/-- Emit the designated verifier input node.  We keep this at id `0` for bound seeding. -/
def emitInput {α : Type} [Context α] {s : Shape} : BuildM α (Ref α s) := do
  let id ← freshId (α := α)
  if id ≠ 0 then
    -- keep the designated input id stable for verifier seeding
    fail (α := α) "TorchLean IR lowering: internal error (input node must be id 0)"
  let node : Node := { id := id, parents := [], kind := .input, outShape := s }
  pushNode (α := α) node
  pure (.node id)

/-- Read a compile-time constant tensor, failing if the value already depends on graph input. -/
def getConst {α : Type} [Context α] {s : Shape} (r : Ref α s) : BuildM α (Tensor α s) :=
  match r with
  | .const t => pure t
  | .node _ => fail (α := α)
    "TorchLean IR lowering: expected a compile-time constant tensor (got a graph node)"

/-- Lower one sample of multi-head attention into the verifier IR. -/
def emitMultiHeadAttention {α : Type} [Context α]
    {n numHeads dModel headDim : Nat}
    (wq : Ref α (.dim dModel (.dim (numHeads * headDim) .scalar)))
    (wk : Ref α (.dim dModel (.dim (numHeads * headDim) .scalar)))
    (wv : Ref α (.dim dModel (.dim (numHeads * headDim) .scalar)))
    (wo : Ref α (.dim (numHeads * headDim) (.dim dModel .scalar)))
    (x : Ref α (.dim n (.dim dModel .scalar)))
    (mask : Option (Tensor Bool (.dim n (.dim n .scalar)))) :
    BuildM α (Ref α (.dim n (.dim dModel .scalar))) := do
  let sX : Shape := .dim n (.dim dModel .scalar)
  let sBig : Shape := .dim n (.dim (numHeads * headDim) .scalar)
  let Q : Ref α sBig ← emitMatmul (α := α) (a := x) (b := wq) (sOut := sBig) (outShape := sBig)
  let K : Ref α sBig ← emitMatmul (α := α) (a := x) (b := wk) (sOut := sBig) (outShape := sBig)
  let V : Ref α sBig ← emitMatmul (α := α) (a := x) (b := wv) (sOut := sBig) (outShape := sBig)

  -- The projection coordinate is `(head, coordinate-within-head)`. Preserve that row-major
  -- interpretation by reshaping each token first, then moving the head axis outward. Directly
  -- reshaping `(n, numHeads * headDim)` to `(numHeads, n, headDim)` changes which projected
  -- features belong to each head.
  let sProjected : Shape := .dim n (.dim numHeads (.dim headDim .scalar))
  let sHeads : Shape := .dim numHeads (.dim n (.dim headDim .scalar))
  let QProjected : Ref α sProjected ←
    emitUnary (α := α) (kind := .reshape sBig sProjected) (x := Q) (t := sProjected)
      (outShape := sProjected)
  let KProjected : Ref α sProjected ←
    emitUnary (α := α) (kind := .reshape sBig sProjected) (x := K) (t := sProjected)
      (outShape := sProjected)
  let VProjected : Ref α sProjected ←
    emitUnary (α := α) (kind := .reshape sBig sProjected) (x := V) (t := sProjected)
      (outShape := sProjected)
  let Qh : Ref α sHeads ←
    emitUnary (α := α) (kind := .swap_first_two) (x := QProjected) (t := sHeads)
      (outShape := sHeads)
  let Kh : Ref α sHeads ←
    emitUnary (α := α) (kind := .swap_first_two) (x := KProjected) (t := sHeads)
      (outShape := sHeads)
  let Vh : Ref α sHeads ←
    emitUnary (α := α) (kind := .swap_first_two) (x := VProjected) (t := sHeads)
      (outShape := sHeads)

  let sKt : Shape := .dim numHeads (.dim headDim (.dim n .scalar))
  let Kt : Ref α sKt ←
    emitUnary (α := α) (kind := .transpose3dLastTwo) (x := Kh) (t := sKt)
      (outShape := sKt)
  let sScores : Shape := .dim numHeads (.dim n (.dim n .scalar))
  let scores : Ref α sScores ←
    emitMatmul (α := α) (a := Qh) (b := Kt) (sOut := sScores) (outShape := sScores)

  let invScale : α := Numbers.one / Spec.attentionScaleDenom (α := α) headDim
  let scaleTensor : Tensor α sScores := Spec.fill (α := α) invScale sScores
  let scaledScores ←
    emitBinary (α := α) (kind := .mul_elem) (a := scores) (b := .const scaleTensor)

  -- A blocked mask entry has exactly zero numerator. No finite sentinel is introduced here.
  let attn : Ref α sScores ←
    match mask with
    | none =>
        emitUnary (α := α) (kind := .softmax (axis := 2)) (x := scaledScores) (t := sScores)
          (outShape := sScores)
    | some m => do
        let mask3D : Tensor Bool sScores := Tensor.dim (fun _ => m)
        emitUnary (α := α)
          (kind := .hardMaskedSoftmax (NN.IR.HardMask.ofTensor mask3D))
          (x := scaledScores) (t := sScores) (outShape := sScores)

  let outHeads : Ref α sHeads ←
    emitMatmul (α := α) (a := attn) (b := Vh) (sOut := sHeads) (outShape := sHeads)
  let sSwap : Shape := .dim n (.dim numHeads (.dim headDim .scalar))
  let swapped : Ref α sSwap ←
    emitUnary (α := α) (kind := .swap_first_two) (x := outHeads) (t := sSwap)
      (outShape := sSwap)
  let concat : Ref α sBig ←
    emitUnary (α := α) (kind := .reshape sSwap sBig) (x := swapped) (t := sBig)
      (outShape := sBig)
  emitMatmul (α := α) (a := concat) (b := wo) (sOut := sX) (outShape := sX)

/-- Exact leading-axis slice expressed through the verifier's affine matrix fragment. -/
def emitLeadingSlice {α : Type} [Context α]
    {n len : Nat} {s : Shape} (start : Nat) (_h : start + len ≤ n)
    (x : Ref α (.dim n s)) : BuildM α (Ref α (.dim len s)) := do
  let block : Nat := Spec.Shape.size s
  let xMat : Ref α (.dim n (.dim block .scalar)) ←
    emitUnary (α := α)
      (kind := .reshape (.dim n s) (.dim n (.dim block .scalar)))
      (x := x) (t := .dim n (.dim block .scalar))
      (outShape := .dim n (.dim block .scalar))
  let selector : Tensor α (.dim len (.dim n .scalar)) :=
    Tensor.dim (fun row =>
      Tensor.dim (fun col =>
        Tensor.scalar (if col.val = start + row.val then (1 : α) else (0 : α))))
  let yMat : Ref α (.dim len (.dim block .scalar)) ←
    emitMatmul (α := α) (a := .const selector) (b := xMat)
      (sOut := .dim len (.dim block .scalar))
      (outShape := .dim len (.dim block .scalar))
  emitUnary (α := α)
    (kind := .reshape (.dim len (.dim block .scalar)) (.dim len s))
    (x := yMat) (t := .dim len s) (outShape := .dim len s)

/-- Emit a verifier-IR concatenation along the leading axis. -/
def emitLeadingConcat {α : Type} [Context α]
    {n m : Nat} {s : Shape} (a : Ref α (.dim n s)) (b : Ref α (.dim m s)) :
    BuildM α (Ref α (.dim (n + m) s)) := do
  let pa ← ensureNode (α := α) a
  let pb ← ensureNode (α := α) b
  let id ← freshId (α := α)
  let node : Node :=
    { id := id, parents := [pa, pb], kind := .concat 0, outShape := .dim (n + m) s }
  pushNode (α := α) node
  pure (.node id)

/--
Lower batched attention as the leading-axis map of the single-sample verifier graph.

This is intentionally a semantic lowering rather than a claim that the verifier understands a new
opaque fused kernel.
-/
def emitBatchedMultiHeadAttention {α : Type} [Context α]
    {batch n numHeads dModel headDim : Nat}
    (wq : Ref α (.dim dModel (.dim (numHeads * headDim) .scalar)))
    (wk : Ref α (.dim dModel (.dim (numHeads * headDim) .scalar)))
    (wv : Ref α (.dim dModel (.dim (numHeads * headDim) .scalar)))
    (wo : Ref α (.dim (numHeads * headDim) (.dim dModel .scalar)))
    (x : Ref α (.dim batch (.dim n (.dim dModel .scalar))))
    (mask : Option (Tensor Bool (.dim n (.dim n .scalar)))) :
    BuildM α (Ref α (.dim batch (.dim n (.dim dModel .scalar)))) :=
  match batch with
  | 0 =>
      pure <| .const <| Tensor.dim (fun i : Fin 0 => Fin.elim0 i)
  | batch + 1 => do
      let head1 ← emitLeadingSlice (α := α) (n := batch + 1) (len := 1) 0 (by simp) x
      let head : Ref α (.dim n (.dim dModel .scalar)) ←
        emitUnary (α := α)
          (kind := .reshape
            (.dim 1 (.dim n (.dim dModel .scalar)))
            (.dim n (.dim dModel .scalar)))
          (x := head1) (t := .dim n (.dim dModel .scalar))
          (outShape := .dim n (.dim dModel .scalar))
      let yHead ← emitMultiHeadAttention (α := α) wq wk wv wo head mask
      let yHead1 : Ref α (.dim 1 (.dim n (.dim dModel .scalar))) ←
        emitUnary (α := α)
          (kind := .reshape
            (.dim n (.dim dModel .scalar))
            (.dim 1 (.dim n (.dim dModel .scalar))))
          (x := yHead) (t := .dim 1 (.dim n (.dim dModel .scalar)))
          (outShape := .dim 1 (.dim n (.dim dModel .scalar)))
      let tail ← emitLeadingSlice (α := α) (n := batch + 1) (len := batch) 1
        (by simp [Nat.add_comm]) x
      let yTail ← emitBatchedMultiHeadAttention (α := α) wq wk wv wo tail mask
      let y ← emitLeadingConcat (α := α)
        (n := 1) (m := batch) (s := .dim n (.dim dModel .scalar)) yHead1 yTail
      return (by simpa [Nat.one_add] using y)

private theorem vector2_toList {α : Type} (v : Vector α 2) :
    v.toList = [v.get ⟨0, by decide⟩, v.get ⟨1, by decide⟩] := by
  -- Reduce to the underlying array.
  simp [Vector.toList, Vector.get]
  apply List.ext_getElem
  · simp
  · intro i hi
    have hi2 : i < 2 := by
      simpa using hi
    cases i with
    | zero =>
        simp [List.getElem_cons, Array.getElem_toList]
    | succ i =>
        cases i with
        | zero =>
            simp [List.getElem_cons, Array.getElem_toList]
        | succ i =>
            have : 2 ≤ Nat.succ (Nat.succ i) :=
              Nat.succ_le_succ (Nat.succ_le_succ (Nat.zero_le i))
            exact (False.elim ((Nat.not_lt_of_ge this) hi2))

instance {α : Type} [Context α] [DecidableEq Shape] :
    Runtime.Autograd.Torch.Ops (m := BuildM α) (α := α) where
  Ref := Ref α
  NatTensorRef := fun s => Tensor Nat s

  const := fun {_s} t => pure (.const t)
  natTensorConst := fun t => t
  mapNatTensor := fun f t => f t

  add := fun {_s} a b => emitBinary (α := α) (kind := .add) (a := a) (b := b)
  sub := fun {_s} a b => emitBinary (α := α) (kind := .sub) (a := a) (b := b)
  mul := fun {_s} a b => emitBinary (α := α) (kind := .mul_elem) (a := a) (b := b)

  scale := fun {s} x c => do
    -- IR has no dedicated `scale`; encode as elementwise mul with a constant tensor.
    let cT : Tensor α s := Spec.fill c s
    emitBinary (α := α) (kind := .mul_elem) (a := x) (b := .const cT)

  abs := fun {s} _x =>
    emitUnary (α := α) (kind := .abs) (x := _x) (t := s) (outShape := s)
  sqrt := fun {s} x =>
    emitUnary (α := α) (kind := .sqrt) (x := x) (t := s) (outShape := s)
  clamp := fun {s} x lo hi => do
    -- Lower to `min(max(x, lo), hi)` using const-filled tensors.
    let loT : Tensor α s := Spec.fill (α := α) lo s
    let hiT : Tensor α s := Spec.fill (α := α) hi s
    let y ← emitBinary (α := α) (kind := .maxElem) (a := x) (b := .const loT)
    emitBinary (α := α) (kind := .minElem) (a := y) (b := .const hiT)
  max := fun {_s} a b =>
    emitBinary (α := α) (kind := .maxElem) (a := a) (b := b)
  min := fun {_s} a b =>
    emitBinary (α := α) (kind := .minElem) (a := a) (b := b)

  broadcastTo := fun {s₁ s₂} _cb x => do
    emitUnary (α := α) (kind := .broadcastTo s₁ s₂) (x := x) (t := s₂) (outShape := s₂)

  reshape := fun {s₁ s₂} x _h => do
    let out : Shape := s₂
    emitUnary (α := α) (kind := .reshape s₁ s₂) (x := x) (t := s₂) (outShape := out)

  transpose2d := fun {mDim nDim} x => do
    let out : Shape := .dim nDim (.dim mDim .scalar)
    emitUnary (α := α) (kind := .swap_first_two) (x := x) (t := out) (outShape := out)

  transpose3dFirstToLast := fun {a b c} x => do
    -- (a,b,c) -> (b,c,a) = swap_first_two then transpose3d_last_two
    let s1 : Shape := .dim b (.dim a (.dim c .scalar))
    let tmp : Ref α s1 ←
      emitUnary (α := α) (kind := .swap_first_two) (x := x) (t := s1) (outShape := s1)
    let out : Shape := .dim b (.dim c (.dim a .scalar))
    emitUnary (α := α) (kind := .transpose3dLastTwo) (x := tmp) (t := out) (outShape := out)
  transpose3dLastToFirst := fun {a b c} x => do
    -- (a,b,c) -> (c,a,b) = transpose3d_last_two then swap_first_two
    let s1 : Shape := .dim a (.dim c (.dim b .scalar))
    let tmp : Ref α s1 ←
      emitUnary (α := α) (kind := .transpose3dLastTwo) (x := x) (t := s1) (outShape := s1)
    let out : Shape := .dim c (.dim a (.dim b .scalar))
    emitUnary (α := α) (kind := .swap_first_two) (x := tmp) (t := out) (outShape := out)
  transpose3dLastTwo := fun {a b c} x => do
    let out : Shape := .dim a (.dim c (.dim b .scalar))
    emitUnary (α := α) (kind := .transpose3dLastTwo) (x := x) (t := out) (outShape := out)

  swapAdjacentAtDepth := fun {s} depth x => do
    let out := s.swapAdjacentAtDepth depth
    if h : out = s then
      pure (Eq.mp (congrArg (fun sh => Ref α sh) h.symm) x)
    else
      let r := Spec.Shape.rank s
      if depth + 1 < r then
        let perm : List Nat :=
          (List.range r).map (fun i =>
            if i = depth then depth + 1 else if i = depth + 1 then depth else i)
        let kind :=
          match depth, s with
          | 0, .dim _ (.dim _ _) => OpKind.swap_first_two
          | 1, .dim _ (.dim _ (.dim _ .scalar)) => OpKind.transpose3dLastTwo
          | _, _ => OpKind.permute perm
        emitUnary (α := α) (kind := kind) (x := x) (t := out) (outShape := out)
      else
        -- This should be unreachable because an invalid adjacent swap leaves the shape unchanged.
        fail (α := α)
          s!"TorchLean→IR: swapAdjacentAtDepth (depth={depth}) invalid for shape {repr s}"

  reduceSum := fun {s} axis _valid _wf x => do
    let out : Shape := Spec.Tensor.shapeAfterSum s axis
    emitUnary (α := α) (kind := .reduceSum axis) (x := x) (t := out) (outShape := out)
  reduceMean := fun {s} axis _valid _wf x => do
    let out : Shape := Spec.Tensor.shapeAfterSum s axis
    emitUnary (α := α) (kind := .reduceMean axis) (x := x) (t := out) (outShape := out)

  gatherScalar := fun {n} x i => do
    -- Lower `gather_scalar` to IR ops:
    --   (1×n) one-hot row  @  reshape(x, n×1)   → (1×1) → scalar
    let sel : Tensor α (.dim 1 (.dim n .scalar)) :=
      Tensor.dim (fun _ =>
        Tensor.dim (fun j => Tensor.scalar (if j = i then (1 : α) else (0 : α))))
    let xCol : Ref α (.dim n (.dim 1 .scalar)) ←
      emitUnary (α := α)
        (kind := .reshape (.dim n .scalar) (.dim n (.dim 1 .scalar)))
        (x := x) (t := .dim n (.dim 1 .scalar)) (outShape := .dim n (.dim 1 .scalar))
    let y11 : Ref α (.dim 1 (.dim 1 .scalar)) ←
      emitMatmul (α := α) (a := .const sel) (b := xCol)
        (sOut := .dim 1 (.dim 1 .scalar)) (outShape := .dim 1 (.dim 1 .scalar))
    emitUnary (α := α)
      (kind := .reshape (.dim 1 (.dim 1 .scalar)) .scalar)
      (x := y11) (t := Shape.scalar) (outShape := Shape.scalar)

  gatherRow := fun {rows cols} x i => do
    -- Lower `gather_row` to IR ops:
    --   (1×rows) one-hot row  @  x(rows×cols)  → (1×cols) → (cols)
    let sel : Tensor α (.dim 1 (.dim rows .scalar)) :=
      Tensor.dim (fun _ =>
        Tensor.dim (fun j => Tensor.scalar (if j = i then (1 : α) else (0 : α))))
    let y1c : Ref α (.dim 1 (.dim cols .scalar)) ←
      emitMatmul (α := α) (a := .const sel) (b := x)
        (sOut := .dim 1 (.dim cols .scalar)) (outShape := .dim 1 (.dim cols .scalar))
    emitUnary (α := α)
      (kind := .reshape (.dim 1 (.dim cols .scalar)) (.dim cols .scalar))
      (x := y1c) (t := .dim cols .scalar) (outShape := .dim cols .scalar)

  gatherScalarNatOrZero := fun {_n} _x _i =>
    fail (α := α) "TorchLean→IR: gather is outside the verifier IR fragment"
  gatherVecNatOrZero := fun {_n _k} _x _idx =>
    fail (α := α) "TorchLean→IR: gather is outside the verifier IR fragment"
  gatherRowsNatOrZero := fun {_rows _cols _k} _x _idx =>
    fail (α := α) "TorchLean→IR: gather is outside the verifier IR fragment"
  scatterAddVec := fun {_n} _x _val _i =>
    fail (α := α) "TorchLean→IR: scatter is outside the verifier IR fragment"
  scatterAddRow := fun {_rows _cols} _x _row _i =>
    fail (α := α) "TorchLean→IR: scatter is outside the verifier IR fragment"

  matmul := fun {mDim _nDim pDim} a b => do
    let out : Shape := .dim mDim (.dim pDim .scalar)
    emitMatmul (α := α) (a := a) (b := b) (sOut := out) (outShape := out)

  bmm := fun {batch mDim _nDim pDim} a b => do
    let out : Shape := .dim batch (.dim mDim (.dim pDim .scalar))
    emitMatmul (α := α) (a := a) (b := b) (sOut := out) (outShape := out)

  concatLeadingAxis := fun {_nDim _mDim} {_s} a b =>
    emitLeadingConcat (α := α) a b

  sliceLeadingAxisRange := fun {_nDim} {_s} start _len h x =>
    emitLeadingSlice (α := α) start h x

  -- ---------------------------------------------------------------------------
  -- ND pooling/conv wrappers (verifier IR supports CHW 2D ops)
  -- ---------------------------------------------------------------------------

  maxPool := fun {d C} {inSpatial kernel stride padding} {_hKernel} x => do
    -- The verifier IR only supports CHW max_pool2d(_pad) with symmetric stride/padding.
    if hd : d = 2 then
        let kernel2 : Vector Nat 2 := hd ▸ kernel
        let stride2 : Vector Nat 2 := hd ▸ stride
        let padding2 : Vector Nat 2 := hd ▸ padding
        let inSpatial2 : Vector Nat 2 := hd ▸ inSpatial
        let kH : Nat := kernel2.get ⟨0, by decide⟩
        let kW : Nat := kernel2.get ⟨1, by decide⟩
        let sH : Nat := stride2.get ⟨0, by decide⟩
        let sW : Nat := stride2.get ⟨1, by decide⟩
        let pH : Nat := padding2.get ⟨0, by decide⟩
        let pW : Nat := padding2.get ⟨1, by decide⟩
        let inH : Nat := inSpatial2.get ⟨0, by decide⟩
        let inW : Nat := inSpatial2.get ⟨1, by decide⟩

        -- Cast input to explicit CHW so the verifier IR checker can pattern-match on it.
        have hinSpatial : inSpatial.toList = [inH, inW] := by
          subst d
          simpa [inH, inW, inSpatial2] using (vector2_toList (v := inSpatial))
        have hx :
            Shape.ofList (C :: inSpatial.toList) = .dim C (.dim inH (.dim inW .scalar)) := by
          simp [Shape.ofList, hinSpatial]
        let xCHW : Ref α (.dim C (.dim inH (.dim inW .scalar))) :=
          Eq.mp (congrArg (fun sh => Ref α sh) hx) x

        if hs : sH = sW then
          if hp : pH = pW then
            requireContract (α := α) <|
              NN.IR.OpContracts.inferPool2dOutShape "max_pool2d_pad" kH kW sH pH
                (.dim C (.dim inH (.dim inW .scalar)))
            let xId ← ensureNode (α := α) xCHW
            let id ← freshId (α := α)
            let outShape : Shape :=
              Shape.ofList
                (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)
            let node : Node :=
              { id := id
                parents := [xId]
                kind := .maxPool2dPad kH kW sH pH
                outShape := outShape }
            pushNode (α := α) node
            pure (.node id)
          else
            fail (α := α) "TorchLean→IR: max_pool: verifier IR requires uniform padding"
        else
          fail (α := α) "TorchLean→IR: max_pool: verifier IR requires uniform stride"
    else
      fail (α := α) "TorchLean→IR: max_pool: verifier IR accepts d=2"

  avgPool := fun {d C} {inSpatial kernel stride padding} _hKernel x => do
    -- The verifier IR only supports CHW avg_pool2d(_pad) with symmetric stride/padding.
    if hd : d = 2 then
        let kernel2 : Vector Nat 2 := hd ▸ kernel
        let stride2 : Vector Nat 2 := hd ▸ stride
        let padding2 : Vector Nat 2 := hd ▸ padding
        let inSpatial2 : Vector Nat 2 := hd ▸ inSpatial
        let kH : Nat := kernel2.get ⟨0, by decide⟩
        let kW : Nat := kernel2.get ⟨1, by decide⟩
        let sH : Nat := stride2.get ⟨0, by decide⟩
        let sW : Nat := stride2.get ⟨1, by decide⟩
        let pH : Nat := padding2.get ⟨0, by decide⟩
        let pW : Nat := padding2.get ⟨1, by decide⟩
        let inH : Nat := inSpatial2.get ⟨0, by decide⟩
        let inW : Nat := inSpatial2.get ⟨1, by decide⟩

        -- Cast input to explicit CHW so the verifier IR checker can pattern-match on it.
        have hinSpatial : inSpatial.toList = [inH, inW] := by
          subst d
          simpa [inH, inW, inSpatial2] using (vector2_toList (v := inSpatial))
        have hx :
            Shape.ofList (C :: inSpatial.toList) = .dim C (.dim inH (.dim inW .scalar)) := by
          simp [Shape.ofList, hinSpatial]
        let xCHW : Ref α (.dim C (.dim inH (.dim inW .scalar))) :=
          Eq.mp (congrArg (fun sh => Ref α sh) hx) x

        if hs : sH = sW then
          if hp : pH = pW then
            requireContract (α := α) <|
              NN.IR.OpContracts.inferPool2dOutShape "avg_pool2d_pad" kH kW sH pH
                (.dim C (.dim inH (.dim inW .scalar)))
            let xId ← ensureNode (α := α) xCHW
            let id ← freshId (α := α)
            let outShape : Shape :=
              Shape.ofList
                (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)
            let node : Node :=
              { id := id
                parents := [xId]
                kind := .avgPool2dPad kH kW sH pH
                outShape := outShape }
            pushNode (α := α) node
            pure (.node id)
          else
            fail (α := α) "TorchLean→IR: avg_pool: verifier IR requires uniform padding"
        else
          fail (α := α) "TorchLean→IR: avg_pool: verifier IR requires uniform stride"
    else
      fail (α := α) "TorchLean→IR: avg_pool: the verifier IR supports only d=2"

  smoothMaxPool := fun {_d _C} {_inSpatial _kernel _stride _padding} {_hKernel} _x _temp =>
    fail (α := α) "TorchLean→IR: smooth_max_pool is outside the verifier IR fragment"

  maxPool2d := fun {kH kW inH inW inC stride} {_h1 : kH ≠ 0} {_h2 : kW ≠ 0} x => do
    requireContract (α := α) <|
      NN.IR.OpContracts.inferPool2dOutShape "max_pool2d" kH kW stride 0
        (.dim inC (.dim inH (.dim inW .scalar)))
    let xId ← ensureNode (α := α) (s := .dim inC (.dim inH (.dim inW .scalar))) x
    let id ← freshId (α := α)
    let outShape : Shape := Spec.pool2dMultiOutShape inC inH inW kH kW stride
    let node : Node :=
      { id := id, parents := [xId], kind := .maxPool2d kH kW stride, outShape := outShape }
    pushNode (α := α) node
    pure (.node id)
  maxPool2dPad := fun {kH kW inH inW inC stride padding} {_h1 : kH ≠ 0} {_h2 : kW ≠ 0} x => do
    requireContract (α := α) <|
      NN.IR.OpContracts.inferPool2dOutShape "max_pool2d_pad" kH kW stride padding
        (.dim inC (.dim inH (.dim inW .scalar)))
    let xId ← ensureNode (α := α) (s := .dim inC (.dim inH (.dim inW .scalar))) x
    let id ← freshId (α := α)
    let outShape : Shape := Spec.pool2dMultiOutShapePad inC inH inW kH kW stride padding
    let node : Node :=
      { id := id
        parents := [xId]
        kind := .maxPool2dPad kH kW stride padding
        outShape := outShape }
    pushNode (α := α) node
    pure (.node id)
  smoothMaxPool2d := fun {_kH _kW _inH _inW _inC _stride} {_h1} {_h2} _x _temp =>
    fail (α := α) "TorchLean→IR: smooth_max_pool2d is outside the verifier IR fragment"
  avgPool2d := fun {kH kW inH inW inC stride} (_h1 : kH ≠ 0) (_h2 : kW ≠ 0) x => do
    requireContract (α := α) <|
      NN.IR.OpContracts.inferPool2dOutShape "avg_pool2d" kH kW stride 0
        (.dim inC (.dim inH (.dim inW .scalar)))
    let xId ← ensureNode (α := α) (s := .dim inC (.dim inH (.dim inW .scalar))) x
    let id ← freshId (α := α)
    let outShape : Shape := Spec.pool2dMultiOutShape inC inH inW kH kW stride
    let node : Node :=
      { id := id, parents := [xId], kind := .avgPool2d kH kW stride, outShape := outShape }
    pushNode (α := α) node
    pure (.node id)
  avgPool2dPad := fun {kH kW inH inW inC stride padding} (_h1 : kH ≠ 0) (_h2 : kW ≠ 0) x => do
    requireContract (α := α) <|
      NN.IR.OpContracts.inferPool2dOutShape "avg_pool2d_pad" kH kW stride padding
        (.dim inC (.dim inH (.dim inW .scalar)))
    let xId ← ensureNode (α := α) (s := .dim inC (.dim inH (.dim inW .scalar))) x
    let id ← freshId (α := α)
    let outShape : Shape := Spec.pool2dMultiOutShapePad inC inH inW kH kW stride padding
    let node : Node :=
      { id := id
        parents := [xId]
        kind := .avgPool2dPad kH kW stride padding
        outShape := outShape }
    pushNode (α := α) node
    pure (.node id)

  relu := fun {s} x => emitUnary (α := α) (kind := .relu) (x := x) (t := s) (outShape := s)
  sigmoid := fun {s} x => emitUnary (α := α) (kind := .sigmoid) (x := x) (t := s) (outShape := s)
  tanh := fun {s} x => emitUnary (α := α) (kind := .tanh) (x := x) (t := s) (outShape := s)
  gelu := fun {s} x => do
    -- Keep GELU explicit in verifier IR until the IR itself gains a dedicated opcode. Runtime
    -- backends use one tape node; this expansion remains transparent to graph proofs.
    let c0 : α := Activation.Math.geluTanhCoeff
    let c1 : α := MathFunctions.sqrt (Numbers.two / MathFunctions.pi)
    let x2 ← emitBinary (α := α) (kind := .mul_elem) (a := x) (b := x)
    let x3 ← emitBinary (α := α) (kind := .mul_elem) (a := x2) (b := x)
    let c0Tensor : Tensor α s := Spec.fill c0 s
    let scaledX3 ← emitBinary (α := α) (kind := .mul_elem) (a := x3) (b := .const c0Tensor)
    let inner ← emitBinary (α := α) (kind := .add) (a := x) (b := scaledX3)
    let c1Tensor : Tensor α s := Spec.fill c1 s
    let tanhInput ← emitBinary (α := α) (kind := .mul_elem) (a := inner) (b := .const c1Tensor)
    let tanhOut ← emitUnary (α := α) (kind := .tanh) (x := tanhInput) (t := s) (outShape := s)
    let ones : Tensor α s := Spec.fill Numbers.one s
    let onePlus ← emitBinary (α := α) (kind := .add) (a := tanhOut) (b := .const ones)
    let mid ← emitBinary (α := α) (kind := .mul_elem) (a := x) (b := onePlus)
    let halves : Tensor α s := Spec.fill Numbers.half s
    emitBinary (α := α) (kind := .mul_elem) (a := mid) (b := .const halves)
  softmaxLast := fun {s} x => do
    let axis :=
      match Spec.Shape.rank s with
      | 0 => 0
      | Nat.succ r => r
    emitUnary (α := α) (kind := .softmax axis) (x := x) (t := s) (outShape := s)
  logSoftmaxLast := fun {s} x => do
    -- The verifier IR represents `log_softmax` by lowering it through
    -- `softmax` followed by `log` so the semantic graph remains expressible; eager/typed graph
    -- training still uses the stable primitive from the autograd runtime.
    let axis :=
      match Spec.Shape.rank s with
      | 0 => 0
      | Nat.succ r => r
    let probs ← emitUnary (α := α) (kind := .softmax axis) (x := x) (t := s) (outShape := s)
    emitUnary (α := α) (kind := .log) (x := probs) (t := s) (outShape := s)
  softplus := fun {s} x => do
    let oneT : Tensor α s := Spec.fill (α := α) Numbers.one s
    let ex ← emitUnary (α := α) (kind := .exp) (x := x) (t := s) (outShape := s)
    let sum ← emitBinary (α := α) (kind := .add) (a := ex) (b := .const oneT)
    emitUnary (α := α) (kind := .log) (x := sum) (t := s) (outShape := s)
  exp := fun {s} x => emitUnary (α := α) (kind := .exp) (x := x) (t := s) (outShape := s)
  log := fun {s} x => emitUnary (α := α) (kind := .log) (x := x) (t := s) (outShape := s)
  inv := fun {s} x => emitUnary (α := α) (kind := .inv) (x := x) (t := s) (outShape := s)
  detach := fun {s} x => emitUnary (α := α) (kind := .detach) (x := x) (t := s) (outShape := s)
  safeLog := fun {s} x ε => do
    let epsT : Tensor α s := Spec.fill (α := α) ε s
    let shifted ← emitBinary (α := α) (kind := .add) (a := x) (b := .const epsT)
    emitUnary (α := α) (kind := .log) (x := shifted) (t := s) (outShape := s)
  sum := fun {_s} x =>
    emitUnary (α := α) (kind := .sum) (x := x) (t := Shape.scalar) (outShape := Shape.scalar)
  flatten := fun {s} x => do
    let outShape : Shape := .dim (Spec.Shape.size s) .scalar
    emitUnary (α := α) (kind := .flatten s) (x := x) (t := outShape) (outShape := outShape)

  linear := fun {inDim outDim} w b x => do
    let wT ← getConst (α := α) (s := .dim outDim (.dim inDim .scalar)) w
    let bT ← getConst (α := α) (s := .dim outDim .scalar) b
    let xId ← ensureNode (α := α) (s := .dim inDim .scalar) x
    let id ← freshId (α := α)
    let node : Node :=
      { id := id, parents := [xId], kind := .linear, outShape := .dim outDim .scalar }
    modify fun st =>
      { st with
          ps := { st.ps with
            linearWB := st.ps.linearWB.insert id { m := outDim, n := inDim, w := wT, b := bT } } }
    pushNode (α := α) node
    pure (.node id)

  mseLoss := fun {s} yhat target => do
    let yId ← ensureNode (α := α) (s := s) yhat
    let tId ← ensureNode (α := α) (s := s) target
    let id ← freshId (α := α)
    let node : Node := { id := id, parents := [yId, tId], kind := .mseLoss, outShape :=
      Shape.scalar }
    pushNode (α := α) node
    pure (.node id)

  layerNorm := fun {seqLen embedDim} _hSeq _hEmb x gamma beta => do
    let sX : Shape := .dim seqLen (.dim embedDim .scalar)
    let xNorm : Ref α sX ←
      emitUnary (α := α) (kind := .layernorm (axis := 1)) (x := x) (t := sX) (outShape := sX)
    let gammaT ← getConst (α := α) (s := .dim embedDim .scalar) gamma
    let betaT ← getConst (α := α) (s := .dim embedDim .scalar) beta
    let gammaB : Tensor α sX := Tensor.dim (fun _ => gammaT)
    let betaB : Tensor α sX := Tensor.dim (fun _ => betaT)
    let scaled ← emitBinary (α := α) (kind := .mul_elem) (a := xNorm) (b := .const gammaB)
    emitBinary (α := α) (kind := .add) (a := scaled) (b := .const betaB)

  batchNormChannelFirst := fun {_channels _height _width} _hC _hH _hW _x _gamma _beta => do
    fail (α := α)
      ("TorchLean→IR: batchnorm_channel_first (training-style BN: stats from " ++
        "x) is outside the verifier IR fragment. For inference-time BN, " ++
        "use TorchLean.Norm.batch_norm2d_chw_eval / batch_norm2d_nchw_eval (or " ++
        "NN.batchnorm_channel_first_eval).")

  multiHeadAttention := fun {_n _numHeads _dModel _headDim} _h1 wq wk wv wo x mask =>
    emitMultiHeadAttention (α := α) wq wk wv wo x mask

  batchedMultiHeadAttention :=
    fun {_batch _n _numHeads _dModel _headDim} _hBatch _h1 wq wk wv wo x mask =>
      emitBatchedMultiHeadAttention (α := α) wq wk wv wo x mask

  conv := fun {d inC outC} {kernel stride padding} {inSpatial} {hInC} {hKernel} w b x => do
    -- The verifier IR only supports CHW conv2d with symmetric stride/padding.
    if hd : d = 2 then
        let kernel2 : Vector Nat 2 := hd ▸ kernel
        let stride2 : Vector Nat 2 := hd ▸ stride
        let padding2 : Vector Nat 2 := hd ▸ padding
        let inSpatial2 : Vector Nat 2 := hd ▸ inSpatial
        let kH : Nat := kernel2.get ⟨0, by decide⟩
        let kW : Nat := kernel2.get ⟨1, by decide⟩
        let sH : Nat := stride2.get ⟨0, by decide⟩
        let sW : Nat := stride2.get ⟨1, by decide⟩
        let pH : Nat := padding2.get ⟨0, by decide⟩
        let pW : Nat := padding2.get ⟨1, by decide⟩
        let inH : Nat := inSpatial2.get ⟨0, by decide⟩
        let inW : Nat := inSpatial2.get ⟨1, by decide⟩

        -- Cast weights/input to explicit CHW / (outC,inC,kH,kW) so the verifier IR checker can
        -- pattern-match on them.
        have hKernelList : kernel.toList = [kH, kW] := by
          subst d
          simpa [kH, kW, kernel2] using (vector2_toList (v := kernel))
        have hw :
            Shape.ofList (outC :: inC :: kernel.toList) =
              .dim outC (.dim inC (.dim kH (.dim kW .scalar))) := by
          simp [Shape.ofList, hKernelList]
        let w4 : Ref α (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))) :=
          Eq.mp (congrArg (fun sh => Ref α sh) hw) w

        have hinSpatial : inSpatial.toList = [inH, inW] := by
          subst d
          simpa [inH, inW, inSpatial2] using (vector2_toList (v := inSpatial))
        have hx :
            Shape.ofList (inC :: inSpatial.toList) = .dim inC (.dim inH (.dim inW .scalar)) := by
          simp [Shape.ofList, hinSpatial]
        let xCHW : Ref α (.dim inC (.dim inH (.dim inW .scalar))) :=
          Eq.mp (congrArg (fun sh => Ref α sh) hx) x

        if hs : sH = sW then
          if hp : pH = pW then
            if hStride : sH = 0 then
              fail (α := α) "TorchLean→IR: conv: stride must be nonzero"
            else
              requireContract (α := α) <|
                NN.IR.OpContracts.inferConv2dOutShape inC outC kH kW sH pH
                  (.dim inC (.dim inH (.dim inW .scalar)))
              let kT ← getConst (α := α) (s := .dim outC (.dim inC (.dim kH (.dim kW .scalar)))) w4
              let bT ← getConst (α := α) (s := .dim outC .scalar) b
              let xId ← ensureNode (α := α) xCHW
              let id ← freshId (α := α)
              let outShape : Shape :=
                Shape.ofList
                  (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList)
              let node : Node :=
                { id := id
                  parents := [xId]
                  kind := .conv2d inC outC kH kW sH pH
                  outShape := outShape }
              have hkH : kH ≠ 0 := by
                subst d
                exact hKernel ⟨0, by decide⟩
              have hkW : kW ≠ 0 := by
                subst d
                exact hKernel ⟨1, by decide⟩
              let spec : Spec.Conv2dSpec inC outC kH kW sH pH α hInC hkH hkW :=
                { kernel := kT, bias := bT }
              let cfg : NN.IR.Conv2dParams α :=
                { inC := inC, outC := outC, kH := kH, kW := kW
                  stride := sH, padding := pH
                  inH := inH, inW := inW
                  hIn := hInC, hKH := hkH, hKW := hkW, hStride := hStride,
                  spec := spec }
              modify fun st =>
                { st with ps := { st.ps with conv2dCfg := st.ps.conv2dCfg.insert id cfg } }
              pushNode (α := α) node
              pure (.node id)
          else
            fail (α := α) "TorchLean→IR: conv: verifier IR requires uniform padding"
        else
          fail (α := α) "TorchLean→IR: conv: verifier IR requires uniform stride"
    else
      fail (α := α) "TorchLean→IR: conv: verifier IR accepts d=2"

  convTranspose := fun {_d _inC _outC} {_kernel _stride _padding} {_inSpatial} {_hInC} {_hKernel} _w
      _b _x =>
    fail (α := α) "TorchLean→IR: conv_transpose is outside the verifier IR fragment"

  conv2d := fun {inC outC kH kW stride padding inH inW} {h1} {h2} {h3} kernel bias input => do
    if hStride : stride = 0 then
      fail (α := α) "TorchLean→IR: conv2d: stride must be nonzero"
    else
      requireContract (α := α) <|
        NN.IR.OpContracts.inferConv2dOutShape inC outC kH kW stride padding
          (.dim inC (.dim inH (.dim inW .scalar)))
      let kT ← getConst (α := α) (s := .dim outC (.dim inC (.dim kH (.dim kW .scalar)))) kernel
      let bT ← getConst (α := α) (s := .dim outC .scalar) bias
      let xId ← ensureNode (α := α) (s := .dim inC (.dim inH (.dim inW .scalar))) input
      let id ← freshId (α := α)
      let outH : Nat := Spec.Shape.slidingWindowOutDim inH kH stride padding
      let outW : Nat := Spec.Shape.slidingWindowOutDim inW kW stride padding
      let outShape : Shape := .dim outC (.dim outH (.dim outW .scalar))
      let node : Node :=
        { id := id
          parents := [xId]
          kind := .conv2d inC outC kH kW stride padding
          outShape := outShape }
      let spec : Spec.Conv2dSpec inC outC kH kW stride padding α h1 h2 h3 :=
        { kernel := kT, bias := bT }
      let cfg : NN.IR.Conv2dParams α :=
        { inC := inC, outC := outC, kH := kH, kW := kW
          stride := stride, padding := padding
          inH := inH, inW := inW
          hIn := h1, hKH := h2, hKW := h3, hStride := hStride,
          spec := spec }
      modify fun st =>
        { st with ps := { st.ps with conv2dCfg := st.ps.conv2dCfg.insert id cfg } }
      pushNode (α := α) node
      pure (.node id)

  convTranspose2d := fun {_inC _outC _kH _kW _stride _padding _inH _inW} {_h1} {_h2} {_h3} _kernel _bias
      _input =>
    fail (α := α) "TorchLean→IR: conv_transpose2d is outside the verifier IR fragment"

  randUniform := fun {s} seed => do
    let id ← freshId (α := α)
    let node : Node := { id := id, parents := [], kind := .randUniform seed, outShape := s }
    pushNode (α := α) node
    pure (.node id)

  bernoulliMask := fun {s} keepProb seed => do
    let pId ← ensureNode (α := α) keepProb
    let id ← freshId (α := α)
    let node : Node := { id := id, parents := [pId], kind := .bernoulliMask seed, outShape := s }
    pushNode (α := α) node
    pure (.node id)

end NN.Verification.TorchLean

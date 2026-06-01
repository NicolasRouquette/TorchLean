/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness.WellFormed

/-!
# Compiled Forward Evaluation: Shared Invariants
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

namespace Correctness

open NN.Verification.TorchLean

namespace IRStep

/-- Reflexivity for the structural shape equality used by IR runtime guards. -/
theorem shapeBEq_refl (s : Shape) : (s == s) = true := by
  induction s with
  | scalar => rfl
  | dim _ rest ih =>
      have ih' : Shape.areEqual rest rest = true := by
        simpa [BEq.beq] using ih
      simp [BEq.beq, Shape.areEqual, ih']

/-- Reflexivity for the structural shape inequality used by IR runtime guards. -/
theorem shapeBNe_refl (s : Shape) : (s != s) = false := by
  simp [bne, shapeBEq_refl s]

end IRStep

/-! ### Compiler correctness (forward fragment) -/

namespace IRStep

/-! ### Local graph constructors for evaluator lemmas -/

/-- A unary node with parent `0` and an explicit output shape. -/
def unaryNodeOut (kind : OpKind) (outShape : Shape) : NN.IR.Node :=
  { id := 1, parents := [0], kind := kind, outShape := outShape }

/-- A two-node graph for a unary op with explicit input and output shapes. -/
def unaryGraphOut (kind : OpKind) (inShape outShape : Shape) : Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input, outShape := inShape },
      unaryNodeOut kind outShape
    ] }

/-- A unary node whose input and output share the same shape. -/
def unaryNode (kind : OpKind) (s : Shape) : NN.IR.Node :=
  { id := 1, parents := [0], kind := kind, outShape := s }

/-- A two-node graph for a unary op whose input and output share the same shape. -/
def unaryGraph (kind : OpKind) (s : Shape) : Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input, outShape := s },
      unaryNode kind s
    ] }

/-- A binary node with parents `0` and `1` and an explicit output shape. -/
def binaryNodeOut (kind : OpKind) (outShape : Shape) : NN.IR.Node :=
  { id := 2, parents := [0, 1], kind := kind, outShape := outShape }

/-- A three-node graph for a binary op with explicit parent and output shapes. -/
def binaryGraphOut (kind : OpKind) (leftShape rightShape outShape : Shape) : Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input, outShape := leftShape },
      { id := 1, parents := [], kind := .input, outShape := rightShape },
      binaryNodeOut kind outShape
    ] }

/-- A binary node whose inputs and output share the same shape. -/
def binaryNode (kind : OpKind) (s : Shape) : NN.IR.Node :=
  { id := 2, parents := [0, 1], kind := kind, outShape := s }

/-- A three-node graph for a binary op whose inputs and output share the same shape. -/
def binaryGraph (kind : OpKind) (s : Shape) : Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input, outShape := s },
      { id := 1, parents := [], kind := .input, outShape := s },
      binaryNode kind s
    ] }

end IRStep

def evalFGraphVals
    {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : FGraph α paramShapes inShape ss out)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (vals : Array (DVal α)) : Except String (Array (DVal α)) := do
  match g with
  | .ret _y =>
      pure vals
  | .let1 (ss := ss) (mid := mid) (out := out) node gNext =>
      let vOut ←
        evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out := mid)
          node params vals
      evalFGraphVals (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss ++ [mid])
        (out := out)
        gNext params (vals.push vOut)

/-- `Graph.expectShape` returns the stored tensor when the dynamic shape tag matches. -/
theorem expectShape_eq_ok
    {α : Type} [Context α] [DecidableEq Shape]
    {expected : Shape} (v : DVal α) (h : v.shape = expected) :
    NN.IR.Graph.expectShape (α := α) (expected := expected) v =
      Except.ok (h ▸ v.tensor) := by
  cases h
  simp [NN.IR.Graph.expectShape, DVal.shape, DVal.tensor]
  rfl

/-- `getVal` returns the indexed tensor when the runtime value carries the expected shape tag. -/
theorem getVal_eq_ok
    {α : Type} [Context α] [DecidableEq Shape]
    {inShape : Shape} {ss : List Shape} {expected : Shape}
    (vals : Array (DVal α)) (idx : Idx (Ctx inShape ss) expected)
    (h : (vals[idx.id]!).1 = expected) :
    getVal (α := α) (inShape := inShape) (ss := ss) (s := expected) vals idx =
      Except.ok (h ▸ (vals[idx.id]!).snd) := by
  unfold getVal
  simp [DVal.shape]
  rw [dif_pos h]
  rfl

  /--
  Generic prefix-preservation argument for `ParamStore` lookups.

  The forward compiler appends exactly one fresh IR node at each let-binding.  Any payload lookup
  that is preserved by a single `compileNode` step for keys below the fresh id is therefore
  preserved by the whole compiled suffix.
  -/
  private theorem compileFGraph_ps_lookup_get?_lt
      {α β : Type} [Context α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
      (read : NN.MLTheory.CROWN.Graph.ParamStore α → Nat → Option β)
      (hStep :
        ∀ {ss₀ : List Shape} {mid₀ : Shape} {node : Node α paramShapes inShape ss₀ mid₀}
          (id k : Nat) (params : Runtime.Autograd.Torch.TList α paramShapes)
          (ps : NN.MLTheory.CROWN.Graph.ParamStore α),
          k < id →
          read
              (compileNode (α := α) (paramShapes := paramShapes) (inShape := inShape)
                (ss := ss₀) (out := mid₀) id node params ps).2 k =
            read ps k)
      (g : FGraph α paramShapes inShape ss out)
      (params : Runtime.Autograd.Torch.TList α paramShapes)
      (c : NN.Verification.TorchLean.CompiledIR α)
      {k : Nat} (hk : k < c.graph.nodes.size) :
      read
          (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
            (out := out) g params c).ps k =
        read c.ps k := by
    classical
    induction g generalizing c with
    | ret y =>
        simp [compileFGraph]
    | @let1 ss₀ mid₀ out₀ node gNext ih =>
        let id := c.graph.nodes.size
        have hk' : k < id := by simpa [id] using hk
        have hk_succ : k < id + 1 := Nat.lt_succ_of_lt hk'
        let res :=
          compileNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀)
            (out := mid₀) id node params c.ps
        let n : NN.IR.Node := res.1
        let ps' : NN.MLTheory.CROWN.Graph.ParamStore α := res.2
        let c' : NN.Verification.TorchLean.CompiledIR α :=
          { c with graph := { nodes := c.graph.nodes.push n }, ps := ps', outputId := id }
        have hps' : read ps' k = read c.ps k := by
          simpa [res, ps'] using hStep (id := id) (k := k) (params := params) (ps := c.ps) hk'
        have hIH :=
          ih (c := c') (hk := by simpa [c', Array.size_push, id] using hk_succ)
        have : read
            (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape)
              (ss := ss₀ ++ [mid₀]) (out := out₀) gNext params c').ps k =
            read c.ps k := by
          calc
            read
                (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape)
                  (ss := ss₀ ++ [mid₀]) (out := out₀) gNext params c').ps k
                =
              read c'.ps k := hIH
            _ = read c.ps k := by simpa [c'] using hps'
        simpa [compileFGraph, c', id, res] using this

  /--
  Compiling a let-chain does not change `ps.constVals` entries for keys `< c.graph.nodes.size`.
  Compilation only inserts payload at the fresh node id, so older keys are unchanged.
  -/
  theorem compileFGraph_ps_constVals_get?_lt
      {α : Type} [Context α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
      (g : FGraph α paramShapes inShape ss out)
      (params : Runtime.Autograd.Torch.TList α paramShapes)
    (c : NN.Verification.TorchLean.CompiledIR α)
    {k : Nat} (hk : k < c.graph.nodes.size) :
    (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out :=
      out)
        g params c).ps.constVals.get? k = c.ps.constVals.get? k := by
  classical
  exact
    compileFGraph_ps_lookup_get?_lt
      (α := α) (β := NN.MLTheory.CROWN.Graph.FlatVec α)
      (read := fun ps k => ps.constVals.get? k)
      (hStep := by
        intro ss₀ mid₀ node id k params ps hk
        have hidk : id ≠ k := (ne_comm).1 hk.ne
        cases node <;>
          simp [compileNode, Std.HashMap.getElem?_insert, beq_eq_false_iff_ne.mpr hidk])
      g params c hk

  /--
  Compiling a let-chain does not change `ps.linearWB` entries for keys `< c.graph.nodes.size`.
  Compilation only inserts linear payload at the fresh node id, so older keys are unchanged.
  -/
  theorem compileFGraph_ps_linearWB_get?_lt
      {α : Type} [Context α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
      (g : FGraph α paramShapes inShape ss out)
      (params : Runtime.Autograd.Torch.TList α paramShapes)
    (c : NN.Verification.TorchLean.CompiledIR α)
    {k : Nat} (hk : k < c.graph.nodes.size) :
    (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out :=
      out)
        g params c).ps.linearWB.get? k = c.ps.linearWB.get? k := by
  classical
  exact
    compileFGraph_ps_lookup_get?_lt
      (α := α) (β := NN.MLTheory.CROWN.Graph.LinParams α)
      (read := fun ps k => ps.linearWB.get? k)
      (hStep := by
        intro ss₀ mid₀ node id k params ps hk
        have hidk : id ≠ k := (ne_comm).1 hk.ne
        cases node <;>
          simp [compileNode, Std.HashMap.getElem?_insert, beq_eq_false_iff_ne.mpr hidk])
      g params c hk

  /--
  Compiling a let-chain does not change `ps.conv2dCfg` entries for keys `< c.graph.nodes.size`.
  Compilation only inserts convolution payloads at fresh node ids, so older keys are unchanged.
  -/
  theorem compileFGraph_ps_conv2dCfg_get?_lt
      {α : Type} [Context α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
      (g : FGraph α paramShapes inShape ss out)
      (params : Runtime.Autograd.Torch.TList α paramShapes)
    (c : NN.Verification.TorchLean.CompiledIR α)
    {k : Nat} (hk : k < c.graph.nodes.size) :
    (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out :=
      out)
        g params c).ps.conv2dCfg.get? k = c.ps.conv2dCfg.get? k := by
  classical
  exact
    compileFGraph_ps_lookup_get?_lt
      (α := α) (β := NN.MLTheory.CROWN.Graph.Conv2DParams α)
      (read := fun ps k => ps.conv2dCfg.get? k)
      (hStep := by
        intro ss₀ mid₀ node id k params ps hk
        have hidk : id ≠ k := (ne_comm).1 hk.ne
        cases node <;>
          simp [compileNode, Std.HashMap.getElem?_insert, beq_eq_false_iff_ne.mpr hidk])
      g params c hk

  /--
  Compiling a let-chain does not change `ps.batchNorm2dNchwEval` entries for keys below the
  starting graph size.  Eval-mode BatchNorm payloads enter through the broader IR/import bridge,
  not through this proved first-order fragment.
  -/
  theorem compileFGraph_ps_batchNorm2dNchwEval_get?_lt
      {α : Type} [Context α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
      (g : FGraph α paramShapes inShape ss out)
      (params : Runtime.Autograd.Torch.TList α paramShapes)
    (c : NN.Verification.TorchLean.CompiledIR α)
    {k : Nat} (hk : k < c.graph.nodes.size) :
    (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out :=
      out)
        g params c).ps.batchNorm2dNchwEval.get? k =
      c.ps.batchNorm2dNchwEval.get? k := by
  classical
  exact
    compileFGraph_ps_lookup_get?_lt
      (α := α) (β := NN.MLTheory.CROWN.Graph.BatchNorm2DNchwEvalParams α)
      (read := fun ps k => ps.batchNorm2dNchwEval.get? k)
      (hStep := by
        intro ss₀ mid₀ node id k params ps hk
        cases node <;> simp [compileNode])
      g params c hk

  /--
  `compileFGraph` does not change existing nodes at indices `< c.graph.nodes.size`.
  Compilation only appends nodes, so `getNode` agrees on the prefix.
  -/
  theorem compileFGraph_getNode_lt
      {α : Type} [Context α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
      (g : FGraph α paramShapes inShape ss out)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (c : NN.Verification.TorchLean.CompiledIR α)
    {i : Nat} (hi : i < c.graph.nodes.size) :
    (NN.IR.Graph.getNode
        (g := (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
          (out := out)
          g params c).graph) i)
      =
    NN.IR.Graph.getNode (g := c.graph) i := by
  classical
  induction g generalizing c with
  | ret y =>
      simp [compileFGraph]
  | @let1 ss₀ mid₀ out₀ node gNext ih =>
      let id := c.graph.nodes.size
      let res :=
        compileNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out :=
          mid₀)
          id node params c.ps
      let n : NN.IR.Node := res.1
      let ps' : NN.MLTheory.CROWN.Graph.ParamStore α := res.2
      let c' : NN.Verification.TorchLean.CompiledIR α :=
        { c with graph := { nodes := c.graph.nodes.push n }, ps := ps', outputId := id }
      have hi' : i < c'.graph.nodes.size := by
        simpa [c', Array.size_push] using Nat.lt_succ_of_lt hi
      have hNext :
          NN.IR.Graph.getNode
              (g := (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss :=
                ss₀ ++ [mid₀]) (out := out₀)
                gNext params c').graph) i
            =
          NN.IR.Graph.getNode (g := c'.graph) i :=
        ih (c := c') (hi := hi')
      have hPush :
          NN.IR.Graph.getNode (g := c'.graph) i = NN.IR.Graph.getNode (g := c.graph) i := by
        simpa [c', res, id] using getNode_push_lt (g := c.graph) (n := n) (hi := hi)
      simpa [compileFGraph, c', id, res] using Eq.trans hNext hPush

    /-- `compileFGraph` is monotone in `graph.nodes.size` (it only appends nodes). -/
    theorem compileFGraph_nodesSize_le
        {α : Type} [Context α]
        {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
        (g : FGraph α paramShapes inShape ss out)
      (params : Runtime.Autograd.Torch.TList α paramShapes)
      (c : NN.Verification.TorchLean.CompiledIR α) :
      c.graph.nodes.size ≤
        (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out :=
          out)
            g params c).graph.nodes.size := by
    classical
    induction g generalizing c with
    | ret y =>
        simp [compileFGraph]
    | @let1 ss₀ mid₀ out₀ node gNext ih =>
      let id := c.graph.nodes.size
      let res :=
        compileNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out :=
          mid₀)
          id node params c.ps
      let n : NN.IR.Node := res.1
      let ps' : NN.MLTheory.CROWN.Graph.ParamStore α := res.2
      let c' : NN.Verification.TorchLean.CompiledIR α :=
        { c with graph := { nodes := c.graph.nodes.push n }, ps := ps', outputId := id }
      have h1 : c.graph.nodes.size ≤ c'.graph.nodes.size := by
        simp [c', Array.size_push]
      have h2 : c'.graph.nodes.size ≤
          (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀ ++
            [mid₀]) (out := out₀)
            gNext params c').graph.nodes.size := by
        exact ih (c := c')
      simpa [compileFGraph, c', id, res] using Nat.le_trans h1 h2

    /-- Shape lookup through `shapesOfVals` agrees with looking up the dynamic value first. -/
    lemma shapesOfVals_get?_eq
        {α : Type} [Context α] (vals : Array (DVal α)) (i : Nat) :
        (shapesOfVals (α := α) vals)[i]? = (vals[i]?).map (fun v => v.1) := by
    -- Avoid `simp` loops on `Array.getElem?_eq_toList_get?'`.
    have hToList : vals.toList[i]? = vals[i]? := by
      simp
    -- `List.getElem?_map` reduces the `map` and then we rewrite the list lookup to the array
    -- lookup.
    simp [shapesOfVals, List.getElem?_map, hToList]

  @[simp] lemma shapesOfVals_length {α : Type} [Context α] (vals : Array (DVal α)) :
      (shapesOfVals (α := α) vals).length = vals.size := by
    simp [shapesOfVals]

  @[simp] theorem shape_of_vals_of_hShapes
      {α : Type} [Context α]
      {inShape : Shape} {ss : List Shape} {s : Shape}
      (vals : Array (DVal α)) (idx : Idx (Ctx inShape ss) s)
      (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss) :
      (vals[idx.id]!).shape = s := by
    classical
    have hLen : vals.size = (Ctx inShape ss).length := by
      simpa [shapesOfVals_length] using congrArg List.length hShapes
    have hiΓ : idx.id < (Ctx inShape ss).length := idx_id_lt_length (x := idx)
    have hiVals : idx.id < vals.size := by simpa [hLen] using hiΓ
    have hFin : (⟨idx.id, hiΓ⟩ : Fin (Ctx inShape ss).length) = idx.i := by
      apply Fin.ext
      rfl
    have hGetElem : (Ctx inShape ss)[idx.id]'hiΓ = s := by
      -- `l[i]'h` is definitional `l.get ⟨i,h⟩`.
      simpa [Idx.id, List.get, hFin] using idx.h
    have hΓOpt : (Ctx inShape ss)[idx.id]? = some s := by
      have hSome : (Ctx inShape ss)[idx.id]? = some ((Ctx inShape ss)[idx.id]'hiΓ) := by
        simp
      simp [hSome, hGetElem]
    have hShapesAt :
        (shapesOfVals (α := α) vals)[idx.id]? = some s := by
      have hEq : (shapesOfVals (α := α) vals)[idx.id]? = (Ctx inShape ss)[idx.id]? :=
        congrArg (fun l => l[idx.id]?) hShapes
      exact Eq.trans hEq hΓOpt
    -- Convert to an Option statement about the Array lookup.
    have hAt :
        (vals[idx.id]?).map (fun v => v.1) = some s := by
      simpa [shapesOfVals_get?_eq] using hShapesAt
    -- `idx.id < vals.size`, so the Array lookup is `some (vals[idx.id]!)`.
    have hSome : vals[idx.id]? = some (vals[idx.id]!) := by
      simp [getElem?_pos, hiVals]
    -- Extract the shape from the mapped option.
    simpa [hSome] using hAt
end Correctness

end NN.Verification.TorchLean.Proved

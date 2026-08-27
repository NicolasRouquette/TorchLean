/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness.WellFormed

/-!
# Lowered Forward Evaluation: Shared Invariants
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

/-! ### Lowering correctness (forward fragment) -/

namespace IRStep

/-! ### Local graph constructors for evaluator lemmas -/

/-- A unary node with parent `0` and an explicit output shape. -/
def unaryNodeOut (kind : OpKind) (outShape : Shape) : NN.IR.Node :=
  { id := 1, parents := #[0], kind := kind, outShape := outShape }

/-- A two-node graph for a unary op with explicit input and output shapes. -/
def unaryGraphOut (kind : OpKind) (inShape outShape : Shape) : Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input, outShape := inShape },
      unaryNodeOut kind outShape
    ] }

/-- A unary node whose input and output share the same shape. -/
def unaryNode (kind : OpKind) (s : Shape) : NN.IR.Node :=
  { id := 1, parents := #[0], kind := kind, outShape := s }

/-- A two-node graph for a unary op whose input and output share the same shape. -/
def unaryGraph (kind : OpKind) (s : Shape) : Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input, outShape := s },
      unaryNode kind s
    ] }

/-- A binary node with parents `0` and `1` and an explicit output shape. -/
def binaryNodeOut (kind : OpKind) (outShape : Shape) : NN.IR.Node :=
  { id := 2, parents := #[0, 1], kind := kind, outShape := outShape }

/-- A three-node graph for a binary op with explicit parent and output shapes. -/
def binaryGraphOut (kind : OpKind) (leftShape rightShape outShape : Shape) : Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input, outShape := leftShape },
      { id := 1, parents := #[], kind := .input, outShape := rightShape },
      binaryNodeOut kind outShape
    ] }

/-- A binary node whose inputs and output share the same shape. -/
def binaryNode (kind : OpKind) (s : Shape) : NN.IR.Node :=
  { id := 2, parents := #[0, 1], kind := kind, outShape := s }

/-- A three-node graph for a binary op whose inputs and output share the same shape. -/
def binaryGraph (kind : OpKind) (s : Shape) : Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input, outShape := s },
      { id := 1, parents := #[], kind := .input, outShape := s },
      binaryNode kind s
    ] }

/-- A node consuming every preceding entry of a shape array, in order. -/
def variadicNodeOut (kind : OpKind) (parentShapes : Array Shape) (outShape : Shape) : NN.IR.Node :=
  { id := parentShapes.size
    parents := Array.range parentShapes.size
    kind := kind
    outShape := outShape }

/--
Graph fixture for an arbitrary-arity evaluator theorem: one input node per parent shape followed by
a node whose parent ids are the complete preceding range.
-/
def variadicGraphOut (kind : OpKind) (parentShapes : Array Shape) (outShape : Shape) : Graph :=
  { nodes := (parentShapes.mapIdx fun i shape =>
      { id := i, parents := #[], kind := .input, outShape := shape }).push
      (variadicNodeOut kind parentShapes outShape) }

/-- A failure-aware lookup reconstructs an array when it succeeds at every valid index. -/
theorem range_mapM_eq_toList_of_getElem_eq {β : Type} (values : Array β)
    (get : Nat → Except String β)
    (hget : ∀ i (hi : i < values.size), get i = Except.ok values[i]) :
    (List.range values.size).mapM get = Except.ok values.toList := by
  have hLookups :
      (List.range values.size).map get = values.toList.map Except.ok := by
    apply List.ext_getElem
    · simp
    · intro i hRange hValues
      have hi : i < values.size := by simpa using hRange
      simp [hget i hi, Array.getElem_toList]
  have aux : ∀ (ids : List Nat) (result : List β),
      ids.map get = result.map Except.ok → ids.mapM get = Except.ok result := by
    intro ids
    induction ids with
    | nil =>
        intro result hResult
        cases result with
        | nil => rfl
        | cons _ _ => simp at hResult
    | cons i ids ih =>
        intro result hResult
        cases result with
        | nil => simp at hResult
        | cons value result =>
            simp only [List.map_cons, List.cons.injEq] at hResult
            rw [List.mapM_cons, hResult.1, ih result hResult.2]
            rfl
  exact aux (List.range values.size) values.toList hLookups

/-- A failure-aware lookup reconstructs an array when it succeeds at every valid index. -/
theorem rangeArray_mapM_eq_of_getElem_eq {β : Type} (values : Array β)
    (get : Nat → Except String β)
    (hget : ∀ i (hi : i < values.size), get i = Except.ok values[i]) :
    (Array.range values.size).mapM get = Except.ok values := by
  rw [← List.toArray_range, List.mapM_toArray,
    range_mapM_eq_toList_of_getElem_eq values get hget]
  simp

/-- The final node of a variadic evaluator fixture is its variadic operation node. -/
@[simp] theorem variadicGraphOut_getNode
    (kind : OpKind) (parentShapes : Array Shape) (outShape : Shape) :
    (variadicGraphOut kind parentShapes outShape).getNode parentShapes.size =
      .ok (variadicNodeOut kind parentShapes outShape) := by
  let inputNodes : Array NN.IR.Node := parentShapes.mapIdx fun i shape =>
    { id := i, parents := #[], kind := OpKind.input, outShape := shape }
  have hSize : inputNodes.size = parentShapes.size := by simp [inputNodes]
  change ({ nodes := inputNodes.push (variadicNodeOut kind parentShapes outShape) } : Graph).getNode
      parentShapes.size = .ok (variadicNodeOut kind parentShapes outShape)
  rw [← hSize]
  simp [Graph.getNode, Graph.getNode?, variadicNodeOut,
    Bind.bind, Except.bind, Pure.pure, Except.pure]
  exact hSize.symm

end IRStep

/-- Evaluate a typed forward let-chain while accumulating every intermediate dynamic value. -/
def evalForwardLetChainVals
    {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : ForwardLetChain α paramShapes inShape ss out)
    (params : TorchLean.TensorPack α paramShapes)
    (vals : Array (Spec.SomeTensor α)) : Except String (Array (Spec.SomeTensor α)) :=
  match g with
  | .ret _y =>
      pure vals
  | .let1 (ss := ss) (mid := mid) (out := out) node gNext => do
      let vOut ←
        evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out := mid)
          node params vals
      evalForwardLetChainVals (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss ++ [mid])
        (out := out)
        gNext params (vals.push vOut)

/-- `Graph.expectShape` returns the stored tensor when the dynamic shape tag matches. -/
theorem expectShape_eq_ok
    {α : Type} [Context α] [DecidableEq Shape]
    {expected : Shape} (v : Spec.SomeTensor α) (h : v.shape = expected) :
    NN.IR.Graph.expectShape (α := α) (expected := expected) v =
      Except.ok (v.cast h) := by
  cases h
  simp [NN.IR.Graph.expectShape, Spec.SomeTensor.cast, Pure.pure, Except.pure]

/-- A successful shape check certifies the shape stored by the packed tensor. -/
theorem shape_eq_of_expectShape_eq_ok
    {α : Type} [Context α] [DecidableEq Shape]
    {expected : Shape} {v : Spec.SomeTensor α} {t : Tensor α expected}
    (h : NN.IR.Graph.expectShape (α := α) (expected := expected) v = Except.ok t) :
    v.shape = expected := by
  by_contra hShape
  simp [NN.IR.Graph.expectShape, hShape] at h

/-- `getVal` returns the indexed tensor when the runtime value carries the expected shape tag. -/
theorem getVal_eq_ok
    {α : Type} [Context α] [DecidableEq Shape]
    {inShape : Shape} {ss : List Shape} {expected : Shape}
    (vals : Array (Spec.SomeTensor α)) (idx : Idx (Ctx inShape ss) expected)
    (v : Spec.SomeTensor α) (hSome : vals[idx.id]? = some v)
    (h : v.shape = expected) :
    getVal (α := α) (inShape := inShape) (ss := ss) (s := expected) vals idx =
      Except.ok (v.cast h) := by
  cases h
  simp [getVal, getValue?, hSome, Spec.SomeTensor.cast, Bind.bind, Except.bind,
    Pure.pure, Except.pure]

  /--
  Generic prefix-preservation argument for `ParamStore` lookups.

  The forward lowering pass appends exactly one fresh IR node at each let-binding.  Any payload lookup
  that is preserved by a single `lowerNode` step for keys below the fresh id is therefore
  preserved by the whole lowered suffix.
  -/
  private theorem lowerForwardLetChain_ps_lookup_get?_lt
      {α β : Type} [Context α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
      (read : NN.MLTheory.CROWN.Graph.ParamStore α → Nat → Option β)
      (hStep :
        ∀ {ss₀ : List Shape} {mid₀ : Shape} {node : Node α paramShapes inShape ss₀ mid₀}
          (id k : Nat) (params : TorchLean.TensorPack α paramShapes)
          (ps : NN.MLTheory.CROWN.Graph.ParamStore α),
          k < id →
          read
              (lowerNode (α := α) (paramShapes := paramShapes) (inShape := inShape)
                (ss := ss₀) (out := mid₀) id node params ps).2 k =
            read ps k)
      (g : ForwardLetChain α paramShapes inShape ss out)
    (params : TorchLean.TensorPack α paramShapes)
    (c : NN.Verification.TorchLean.LoweredIR α)
      {k : Nat} (hk : k < c.graph.nodes.size) :
      read
          (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
            (out := out) g params c).ps k =
        read c.ps k := by
    classical
      induction g generalizing c with
      | ret y =>
        simp [lowerForwardLetChain]
      | @let1 ss₀ mid₀ out₀ node gNext ih =>
        let id := c.graph.nodes.size
        have hk' : k < id := by simpa [id] using hk
        have hk_succ : k < id + 1 := Nat.lt_succ_of_lt hk'
        let res :=
          lowerNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀)
            (out := mid₀) id node params c.ps
        let n : NN.IR.Node := res.1
        let ps' : NN.MLTheory.CROWN.Graph.ParamStore α := res.2
        let c' : NN.Verification.TorchLean.LoweredIR α :=
          { c with graph := { nodes := c.graph.nodes.push n }, ps := ps', outputId := id }
        have hps' : read ps' k = read c.ps k := by
          simpa [res, ps'] using hStep (id := id) (k := k) (params := params) (ps := c.ps) hk'
        have hIH :=
          ih (c := c') (hk := by simpa [c', Array.size_push, id] using hk_succ)
        have : read
            (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape)
              (ss := ss₀ ++ [mid₀]) (out := out₀) gNext params c').ps k =
            read c.ps k := by
          calc
            read
                (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape)
                  (ss := ss₀ ++ [mid₀]) (out := out₀) gNext params c').ps k
                =
              read c'.ps k := hIH
            _ = read c.ps k := by simpa [c'] using hps'
        simpa [lowerForwardLetChain, c', id, res] using this

  /--
  Lowering a let-chain does not change `ps.constVals` entries for keys `< c.graph.nodes.size`.
  Lowering only inserts payload at the fresh node id, so older keys are unchanged.
  -/
  theorem lowerForwardLetChain_ps_constVals_get?_lt
      {α : Type} [Context α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
      (g : ForwardLetChain α paramShapes inShape ss out)
      (params : TorchLean.TensorPack α paramShapes)
    (c : NN.Verification.TorchLean.LoweredIR α)
    {k : Nat} (hk : k < c.graph.nodes.size) :
    (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out :=
      out)
        g params c).ps.constVals.get? k = c.ps.constVals.get? k := by
    classical
    exact
    lowerForwardLetChain_ps_lookup_get?_lt
      (α := α) (β := NN.MLTheory.CROWN.Graph.FlatTensor α)
      (read := fun ps k => ps.constVals.get? k)
      (hStep := by
        intro ss₀ mid₀ node id k params ps hk
        have hidk : id ≠ k := (ne_comm).1 hk.ne
        cases node <;>
          simp [lowerNode, Std.HashMap.getElem?_insert, beq_eq_false_iff_ne.mpr hidk])
      g params c hk

  /--
  Lowering a let-chain does not change `ps.linearWB` entries for keys `< c.graph.nodes.size`.
  Lowering only inserts linear payload at the fresh node id, so older keys are unchanged.
  -/
  theorem lowerForwardLetChain_ps_linearWB_get?_lt
      {α : Type} [Context α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
      (g : ForwardLetChain α paramShapes inShape ss out)
      (params : TorchLean.TensorPack α paramShapes)
    (c : NN.Verification.TorchLean.LoweredIR α)
    {k : Nat} (hk : k < c.graph.nodes.size) :
    (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out :=
      out)
        g params c).ps.linearWB.get? k = c.ps.linearWB.get? k := by
    classical
    exact
    lowerForwardLetChain_ps_lookup_get?_lt
      (α := α) (β := NN.MLTheory.CROWN.Graph.LinParams α)
      (read := fun ps k => ps.linearWB.get? k)
      (hStep := by
        intro ss₀ mid₀ node id k params ps hk
        have hidk : id ≠ k := (ne_comm).1 hk.ne
        cases node <;>
          simp [lowerNode, Std.HashMap.getElem?_insert, beq_eq_false_iff_ne.mpr hidk])
      g params c hk

  /--
  Lowering a let-chain does not change `ps.convCfg` entries for keys `< c.graph.nodes.size`.
  Lowering only inserts convolution payloads at fresh node ids, so older keys are unchanged.
  -/
  theorem lowerForwardLetChain_ps_convCfg_get?_lt
      {α : Type} [Context α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
      (g : ForwardLetChain α paramShapes inShape ss out)
      (params : TorchLean.TensorPack α paramShapes)
    (c : NN.Verification.TorchLean.LoweredIR α)
    {k : Nat} (hk : k < c.graph.nodes.size) :
    (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out :=
      out)
        g params c).ps.convCfg.get? k = c.ps.convCfg.get? k := by
    classical
    exact
    lowerForwardLetChain_ps_lookup_get?_lt
      (α := α) (β := NN.IR.ConvParams α)
      (read := fun ps k => ps.convCfg.get? k)
      (hStep := by
        intro ss₀ mid₀ node id k params ps hk
        have hidk : id ≠ k := (ne_comm).1 hk.ne
        cases node <;>
          simp [lowerNode, Std.HashMap.getElem?_insert, beq_eq_false_iff_ne.mpr hidk])
      g params c hk

  /--
  Lowering a let-chain does not change `ps.batchNormEval` entries for keys below the
  starting graph size.  Eval-mode BatchNorm payloads enter through the broader IR/import bridge,
  not through this proved first-order fragment.
  -/
  theorem lowerForwardLetChain_ps_batchNormEval_get?_lt
      {α : Type} [Context α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
      (g : ForwardLetChain α paramShapes inShape ss out)
      (params : TorchLean.TensorPack α paramShapes)
    (c : NN.Verification.TorchLean.LoweredIR α)
    {k : Nat} (hk : k < c.graph.nodes.size) :
    (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out :=
      out)
        g params c).ps.batchNormEval.get? k =
      c.ps.batchNormEval.get? k := by
    classical
    exact
    lowerForwardLetChain_ps_lookup_get?_lt
      (α := α) (β := NN.IR.BatchNormEvalParams α)
      (read := fun ps k => ps.batchNormEval.get? k)
      (hStep := by
        intro ss₀ mid₀ node id k params ps hk
        cases node <;> simp [lowerNode])
      g params c hk

  /--
  Lowering a let-chain preserves LayerNorm payloads below the starting graph size.

  A payload-free LayerNorm step erases only its own fresh id, preventing stale future entries from
  changing the source fragment's unit-affine semantics.
  -/
  theorem lowerForwardLetChain_ps_layerNorm_get?_lt
      {α : Type} [Context α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
      (g : ForwardLetChain α paramShapes inShape ss out)
      (params : TorchLean.TensorPack α paramShapes)
      (c : NN.Verification.TorchLean.LoweredIR α)
      {k : Nat} (hk : k < c.graph.nodes.size) :
      (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape)
        (ss := ss) (out := out) g params c).ps.layerNorm.get? k = c.ps.layerNorm.get? k := by
    classical
    exact
      lowerForwardLetChain_ps_lookup_get?_lt
        (α := α) (β := NN.IR.LayerNormParams α)
        (read := fun ps k => ps.layerNorm.get? k)
        (hStep := by
          intro ss₀ mid₀ node id k params ps hk
          have hidk : id ≠ k := (ne_comm).1 hk.ne
          cases node <;>
            simp [lowerNode, Std.HashMap.getElem?_erase, beq_eq_false_iff_ne.mpr hidk])
        g params c hk

  /--
  `lowerForwardLetChain` does not change existing nodes at indices `< c.graph.nodes.size`.
  Lowering only appends nodes, so `getNode` agrees on the prefix.
  -/
  theorem lowerForwardLetChain_getNode_lt
      {α : Type} [Context α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
      (g : ForwardLetChain α paramShapes inShape ss out)
    (params : TorchLean.TensorPack α paramShapes)
    (c : NN.Verification.TorchLean.LoweredIR α)
    {i : Nat} (hi : i < c.graph.nodes.size) :
    (NN.IR.Graph.getNode
      (g := (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
          (out := out)
          g params c).graph) i)
      =
    NN.IR.Graph.getNode (g := c.graph) i := by
    classical
    induction g generalizing c with
    | ret y =>
      simp [lowerForwardLetChain]
    | @let1 ss₀ mid₀ out₀ node gNext ih =>
      let id := c.graph.nodes.size
      let res :=
        lowerNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out :=
          mid₀)
          id node params c.ps
      let n : NN.IR.Node := res.1
      let ps' : NN.MLTheory.CROWN.Graph.ParamStore α := res.2
      let c' : NN.Verification.TorchLean.LoweredIR α :=
        { c with graph := { nodes := c.graph.nodes.push n }, ps := ps', outputId := id }
      have hi' : i < c'.graph.nodes.size := by
        simpa [c', Array.size_push] using Nat.lt_succ_of_lt hi
      have hNext :
          NN.IR.Graph.getNode
              (g := (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss :=
                ss₀ ++ [mid₀]) (out := out₀)
                gNext params c').graph) i
            =
          NN.IR.Graph.getNode (g := c'.graph) i :=
        ih (c := c') (hi := hi')
      have hPush :
          NN.IR.Graph.getNode (g := c'.graph) i = NN.IR.Graph.getNode (g := c.graph) i := by
        simpa [c', res, id] using getNode_push_lt (g := c.graph) (n := n) (hi := hi)
      simpa [lowerForwardLetChain, c', id, res] using Eq.trans hNext hPush

  /-- `lowerForwardLetChain` is monotone in `graph.nodes.size` (it only appends nodes). -/
  theorem lowerForwardLetChain_nodesSize_le
      {α : Type} [Context α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
        (g : ForwardLetChain α paramShapes inShape ss out)
      (params : TorchLean.TensorPack α paramShapes)
      (c : NN.Verification.TorchLean.LoweredIR α) :
    c.graph.nodes.size ≤
      (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out :=
          out)
        g params c).graph.nodes.size := by
    classical
    induction g generalizing c with
    | ret y =>
        simp [lowerForwardLetChain]
    | @let1 ss₀ mid₀ out₀ node gNext ih =>
        let id := c.graph.nodes.size
        let res :=
          lowerNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out :=
            mid₀)
            id node params c.ps
        let n : NN.IR.Node := res.1
        let ps' : NN.MLTheory.CROWN.Graph.ParamStore α := res.2
        let c' : NN.Verification.TorchLean.LoweredIR α :=
          { c with graph := { nodes := c.graph.nodes.push n }, ps := ps', outputId := id }
        have h1 : c.graph.nodes.size ≤ c'.graph.nodes.size := by
          simp [c', Array.size_push]
        have h2 : c'.graph.nodes.size ≤
            (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀ ++
              [mid₀]) (out := out₀)
              gNext params c').graph.nodes.size := by
          exact ih (c := c')
        simpa [lowerForwardLetChain, c', id, res] using Nat.le_trans h1 h2

    /-- Shape lookup through `shapesOfVals` agrees with looking up the dynamic value first. -/
    lemma shapesOfVals_get?_eq
        {α : Type} [Context α] (vals : Array (Spec.SomeTensor α)) (i : Nat) :
        (shapesOfVals (α := α) vals)[i]? = (vals[i]?).map (fun v => v.1) := by
      -- Avoid `simp` loops on `Array.getElem?_eq_toList_get?'`.
      have hToList : vals.toList[i]? = vals[i]? := by
        simp
      -- `List.getElem?_map` reduces the `map` and then we rewrite the list lookup to the array
      -- lookup.
      simp [shapesOfVals, List.getElem?_map, hToList]

  @[simp] lemma shapesOfVals_length {α : Type} [Context α] (vals : Array (Spec.SomeTensor α)) :
      (shapesOfVals (α := α) vals).length = vals.size := by
      simp [shapesOfVals]

  /-- A shape-context invariant proves that a typed index is in bounds for the value array. -/
  theorem index_lt_of_shapesOfVals_eq
      {α : Type} [Context α]
      {inShape : Shape} {ss : List Shape} {s : Shape}
      (vals : Array (Spec.SomeTensor α)) (idx : Idx (Ctx inShape ss) s)
      (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss) : idx.id < vals.size := by
    have hLen : vals.size = (Ctx inShape ss).length := by
      simpa [shapesOfVals_length] using congrArg List.length hShapes
    have hiΓ : idx.id < (Ctx inShape ss).length := idx_id_lt_length (x := idx)
    simpa [hLen] using hiΓ

  /-- The packed runtime value selected by a typed index and a matching shape context. -/
  def packedAt
      {α : Type} [Context α]
      {inShape : Shape} {ss : List Shape} {s : Shape}
      (vals : Array (Spec.SomeTensor α)) (idx : Idx (Ctx inShape ss) s)
      (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss) : Spec.SomeTensor α :=
    vals[idx.id]'(index_lt_of_shapesOfVals_eq vals idx hShapes)

  /-- Safe array lookup returns the packed value selected by `packedAt`. -/
  theorem getElem?_eq_some_packedAt
      {α : Type} [Context α]
      {inShape : Shape} {ss : List Shape} {s : Shape}
      (vals : Array (Spec.SomeTensor α)) (idx : Idx (Ctx inShape ss) s)
      (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss) :
      vals[idx.id]? = some (packedAt vals idx hShapes) := by
    simp [packedAt, index_lt_of_shapesOfVals_eq vals idx hShapes]

  @[simp] theorem packedAt_shape
      {α : Type} [Context α]
      {inShape : Shape} {ss : List Shape} {s : Shape}
      (vals : Array (Spec.SomeTensor α)) (idx : Idx (Ctx inShape ss) s)
      (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss) :
      (packedAt vals idx hShapes).shape = s := by
    classical
    have hLen : vals.size = (Ctx inShape ss).length := by
      simpa [shapesOfVals_length] using congrArg List.length hShapes
    have hiΓ : idx.id < (Ctx inShape ss).length := idx_id_lt_length (x := idx)
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
    have hSome : vals[idx.id]? = some (packedAt vals idx hShapes) :=
      getElem?_eq_some_packedAt vals idx hShapes
    -- Extract the shape from the mapped option.
    simpa [hSome] using hAt

  /-- The typed tensor selected by an index into a runtime context with the expected shapes. -/
  def tensorAt
      {α : Type} [Context α]
      {inShape : Shape} {ss : List Shape} {s : Shape}
      (vals : Array (Spec.SomeTensor α)) (idx : Idx (Ctx inShape ss) s)
      (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss) : Tensor α s :=
    (packedAt vals idx hShapes).cast (packedAt_shape vals idx hShapes)

  /-- Packing the typed tensor recovered from a well-shaped context returns the original value. -/
  theorem ofTensor_tensorAt_eq_packedAt
      {α : Type} [Context α]
      {inShape : Shape} {ss : List Shape} {s : Shape}
      (vals : Array (Spec.SomeTensor α)) (idx : Idx (Ctx inShape ss) s)
      (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss) :
      Spec.SomeTensor.ofTensor (tensorAt vals idx hShapes) = packedAt vals idx hShapes := by
    let value := packedAt vals idx hShapes
    have hShape : value.shape = s := packedAt_shape vals idx hShapes
    change Spec.SomeTensor.ofTensor (value.cast hShape) = value
    exact Spec.SomeTensor.ofTensor_cast value hShape

  /-- Shape checking succeeds for the typed tensor extracted from a well-shaped context. -/
  theorem expectShape_packedAt_eq_ok
      {α : Type} [Context α] [DecidableEq Shape]
      {inShape : Shape} {ss : List Shape} {s : Shape}
      (vals : Array (Spec.SomeTensor α)) (idx : Idx (Ctx inShape ss) s)
      (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss) :
      NN.IR.Graph.expectShape (α := α) (expected := s) (packedAt vals idx hShapes) =
        Except.ok (tensorAt vals idx hShapes) := by
    simpa [tensorAt] using
      expectShape_eq_ok (expected := s) (packedAt vals idx hShapes)
        (packedAt_shape vals idx hShapes)

  /--
  `getVal` succeeds from a well-shaped executable context.

  This is the proof layer form of `getVal_eq_ok`: callers use the semantic invariant
  `shapesOfVals vals = Ctx inShape ss`, and the lemma derives the array-bounds fact internally.
  -/
  theorem getVal_eq_ok_of_shapesOfVals_eq
      {α : Type} [Context α] [DecidableEq Shape]
      {inShape : Shape} {ss : List Shape} {expected : Shape}
      (vals : Array (Spec.SomeTensor α)) (idx : Idx (Ctx inShape ss) expected)
      (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss) :
      getVal (α := α) (inShape := inShape) (ss := ss) (s := expected) vals idx =
        Except.ok (tensorAt vals idx hShapes) :=
    getVal_eq_ok vals idx (packedAt vals idx hShapes)
      (getElem?_eq_some_packedAt vals idx hShapes) (packedAt_shape vals idx hShapes)
end Correctness

end NN.Verification.TorchLean.Proved

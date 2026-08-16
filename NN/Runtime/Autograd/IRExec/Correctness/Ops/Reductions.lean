/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Correctness.SemanticEquivalenceCommon

/-!
# Reductions and Broadcasting

Correctness lemmas for IR nodes whose primary behavior is broadcasting or reduction:

* `broadcastTo s₁ s₂` (explicit broadcasting, used to keep elementwise ops simple),
* `reduceSum axis` and `reduceMean axis` (single-axis reductions),
* `sum` (full reduction to a scalar).

Each lemma matches the lowering control flow closely: we validate the parent structure and the
side-condition checks that `buildFrom` enforces, then construct the forward-graph closure and
show that it matches `NN.IR.Graph.evalAt` at the current node. We finish by appealing to the shared
`buildFrom_denoteAllFrom_finish` lemma for the tail of the graph.

Build note: reductions are among the more expensive op proofs because axes change shapes. Lean has
to track both the input and output shapes, normalize the axis-side conditions, and then compare the
lowered reduction with the IR denotation. Axis/shape arithmetic belongs in
small lemmas so the semantic proof can read more like the lowering code.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace IRExec

open Spec
open Tensor
open Proofs.Autograd.Algebra
open NN.IR
open Internal

/-- Correctness lemma for `.broadcastTo s₁ s₂` lowering. -/
theorem buildFrom_denoteAllFrom_broadcastTo
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (s₁ s₂ : Shape)
    (hN : g.getNode i = .ok n) (hk : n.kind = .broadcastTo s₁ s₂) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := NN.IR.DVal.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := NN.IR.DVal.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (NN.IR.DVal α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TList α ([inShape] ++ ss) :=
    ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
  let input : NN.IR.DVal α := NN.IR.DVal.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp [hi, hN] at hBuild
  simp (config := { failIfUnchanged := false }) [hk] at hBuild
  cases hp : n.parents with
  | nil =>
      simp [hp] at hBuild; try cases hBuild
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          simp [hp] at hBuild; try cases hBuild
      | nil =>
          cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId s₁ with
          | error msg =>
              simp [hp, hIdx] at hBuild; try cases hBuild
          | ok ip =>
              simp [hp, hIdx] at hBuild
              cases hCan : NN.IR.OpContracts.mkCanBroadcastTo? s₁ s₂ with
              | none =>
                  simp [hCan] at hBuild; try cases hBuild
              | some cb =>
                  simp [hCan] at hBuild
                  by_cases hOut : s₂ = n.outShape
                  ·
                    simp [hOut] at hBuild
                    let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                      mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                        let x := getIdx (α := α) (xs := ctx) ip
                        hOut ▸ Tensor.broadcastTo (α := α) (s₁ := s₁) (s₂ := s₂) cb x)
                    let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                    have hRec :
                        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                          (i := i + 1) st1 = .ok st' := by
                      simpa [st1, nodeData] using hBuild
                    have hGet :
                        vals0[pId]! =
                          NN.IR.DVal.mk (α := α) s₁ (getIdx (α := α) (xs := ctx) ip) := by
                      simpa [vals0, ctx] using
                        (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                          (gd := gd) (x := x) (pid := pId) (s := s₁) (idx := ip) hIdx)
                    have hEval :
                        NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                            (input := input) (vals := vals0) (i := i) =
                          .ok
                            (NN.IR.DVal.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                      have hExp :
                          NN.IR.Graph.expectShape (α := α) (expected := s₁) vals0[pId]! =
                            .ok (getIdx (α := α) (xs := ctx) ip) := by
                        simp [hGet, NN.IR.DVal.mk]
                      cases hOut
                      simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput, hN, hk, hp, hExp, hCan,
                        NN.IR.DVal.shape, NN.IR.DVal.tensor, NN.IR.DVal.mk,
                        nodeData, mkForwardNode,
                        throw_eq_error]
                    have hStep :
                        denoteAllState (α := α) inShape st1 x =
                          vals0.push (NN.IR.DVal.mk (α := α) n.outShape
                            (nodeData.eval ctx)) := by
                      simpa [vals0, st1, nodeData, ctx] using
                        (denoteAllState_snoc (α := α) (inShape := inShape) (ss := ss) (τ := n.outShape)
                          (gd := gd) (nodeData := nodeData) (x := x))
                    have hTail := ih st1 hRec
                    exact buildFrom_denoteAllFrom_finish (α := α) (g := g) (payload := payload)
                      (i := i) (x := x) (hi := hi) (τ := n.outShape)
                      (nodeData := nodeData) (st1 := st1) (st' := st')
                      (ctx := ctx) (vals0 := vals0) (input := input) hTail hEval hStep
                  ·
                    simp [hOut] at hBuild
                    try cases hBuild

-- The proof below follows the inline `reduceSum` branch in
-- `...Correctness.SemanticEquivalence`.
/-- Correctness lemma for `.reduceSum axis` lowering. -/
theorem buildFrom_denoteAllFrom_reduceSum
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (axis : Nat)
    (hN : g.getNode i = .ok n) (hk : n.kind = .reduceSum axis) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := NN.IR.DVal.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := NN.IR.DVal.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (NN.IR.DVal α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TList α ([inShape] ++ ss) :=
    ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
  let input : NN.IR.DVal α := NN.IR.DVal.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp [hi, hN] at hBuild
  simp (config := { failIfUnchanged := false }) [hk] at hBuild
  cases hp : n.parents with
  | nil =>
      simp [hp] at hBuild; try cases hBuild
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          simp [hp] at hBuild; try cases hBuild
      | nil =>
          cases hP : g.getNode pId with
          | error msg =>
              simp [hp, hP] at hBuild; try cases hBuild
          | ok pNode =>
              simp [hp, hP] at hBuild
              let s := pNode.outShape
              cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId s with
              | error msg =>
                  simp [s, hIdx] at hBuild
              | ok ip =>
                  simp (config := { failIfUnchanged := false }) [s, hIdx] at hBuild
                  cases hAxis : Spec.Shape.validAxis? (axis := axis) s with
                  | none =>
                      exact False.elim <| throw_bind_ne_ok (h := (by simpa [s, hAxis] using hBuild))
                  | some hAxisPf =>
                      simp (config := { failIfUnchanged := false }) [s, hAxis] at hBuild
                      let hRed := Shape.proveReducibleAlong axis s hAxisPf.down
                      let expected : Shape := Spec.Tensor.shapeAfterSum s axis
                      by_cases hOut : expected = n.outShape
                      ·
                        have hCond :
                            Spec.Tensor.shapeAfterSum pNode.outShape axis = n.outShape := by
                          simpa [expected, s] using hOut
                        simp [hCond] at hBuild
                        let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                          mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                            let x := getIdx (α := α) (xs := ctx) ip
                            let y : Tensor α expected :=
                              Tensor.reduceSum (α := α) (s := s) axis x hRed
                            hOut ▸ y)
                        let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                        have hRec :
                            buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                              (i := i + 1) st1 = .ok st' := by
                          simpa [st1, nodeData] using hBuild
                        have hGet :
                            vals0[pId]! =
                              NN.IR.DVal.mk (α := α) s (getIdx (α := α) (xs := ctx) ip) := by
                          simpa [vals0, ctx] using
                            (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                              (gd := gd) (x := x) (pid := pId) (s := s) (idx := ip) hIdx)
                        have hEval :
                            NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                                (input := input) (vals := vals0) (i := i) =
                              .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                          simpa [nodeData, mkForwardNode] using
                            (evalAt_reduceSum_ok (α := α) (g := g) (payload := payload)
                              (input := input) (vals := vals0) (i := i) (n := n)
                              (pId := pId) (axis := axis) (s := s)
                              (pT := getIdx (α := α) (xs := ctx) ip) (hAxisPf := hAxisPf)
                              (hN := hN) (hk := hk) (hp := hp) (hGet := hGet) (hAxis := hAxis)
                              (hOut := hOut))
                        have hStep :
                            denoteAllState (α := α) inShape st1 x =
                              vals0.push (NN.IR.DVal.mk (α := α) n.outShape
                                (nodeData.eval ctx)) := by
                          simpa [vals0, st1, nodeData, ctx] using
                            (denoteAllState_snoc (α := α) (inShape := inShape) (ss := ss) (τ := n.outShape)
                              (gd := gd) (nodeData := nodeData) (x := x))
                        have hTail := ih st1 hRec
                        exact buildFrom_denoteAllFrom_finish (α := α) (g := g) (payload := payload)
                          (i := i) (x := x) (hi := hi) (τ := n.outShape)
                          (nodeData := nodeData) (st1 := st1) (st' := st')
                          (ctx := ctx) (vals0 := vals0) (input := input) hTail hEval hStep
                      ·
                        have hCondNe :
                            ¬Spec.Tensor.shapeAfterSum pNode.outShape axis = n.outShape := by
                          intro hEq
                          apply hOut
                          simpa [expected, s] using hEq
                        simp [hCondNe] at hBuild
                        try cases hBuild

-- Like `buildFrom_denoteAllFrom_reduceSum`, this proof follows the implementation branch closely.
/-- Correctness lemma for `.reduceMean axis` lowering. -/
theorem buildFrom_denoteAllFrom_reduceMean
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (axis : Nat)
    (hN : g.getNode i = .ok n) (hk : n.kind = .reduceMean axis) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := NN.IR.DVal.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := NN.IR.DVal.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (NN.IR.DVal α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TList α ([inShape] ++ ss) :=
    ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
  let input : NN.IR.DVal α := NN.IR.DVal.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp [hi, hN] at hBuild
  simp (config := { failIfUnchanged := false }) [hk] at hBuild
  cases hp : n.parents with
  | nil =>
      simp [hp] at hBuild; try cases hBuild
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          simp [hp] at hBuild; try cases hBuild
      | nil =>
          cases hP : g.getNode pId with
          | error msg =>
              simp [hp, hP] at hBuild; try cases hBuild
          | ok pNode =>
              simp [hp, hP] at hBuild
              let s := pNode.outShape
              cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId s with
              | error msg =>
                  simp [s, hIdx] at hBuild
              | ok ip =>
                  simp (config := { failIfUnchanged := false }) [s, hIdx] at hBuild
                  cases hAxis : Spec.Shape.validAxis? (axis := axis) s with
                  | none =>
                      exact False.elim <| throw_bind_ne_ok (h := (by simpa [s, hAxis] using hBuild))
                  | some hAxisPf =>
                      simp (config := { failIfUnchanged := false }) [s, hAxis] at hBuild
                      let hRed := Shape.proveReducibleAlong axis s hAxisPf.down
                      let expected : Shape := Spec.Tensor.shapeAfterSum s axis
                      by_cases hOut : expected = n.outShape
                      ·
                        have hCond :
                            Spec.Tensor.shapeAfterSum pNode.outShape axis = n.outShape := by
                          simpa [expected, s] using hOut
                        simp [hCond] at hBuild
                        let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                          mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                            let x := getIdx (α := α) (xs := ctx) ip
                            let y : Tensor α expected :=
                              Tensor.reduceMean (α := α) (s := s) axis x hRed
                            hOut ▸ y)
                        let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                        have hRec :
                            buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                              (i := i + 1) st1 = .ok st' := by
                          simpa [st1, nodeData] using hBuild
                        have hGet :
                            vals0[pId]! =
                              NN.IR.DVal.mk (α := α) s (getIdx (α := α) (xs := ctx) ip) := by
                          simpa [vals0, ctx] using
                            (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                              (gd := gd) (x := x) (pid := pId) (s := s) (idx := ip) hIdx)
                        have hEval :
                            NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                                (input := input) (vals := vals0) (i := i) =
                              .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                          simpa [nodeData, mkForwardNode] using
                            (evalAt_reduceMean_ok (α := α) (g := g) (payload := payload)
                              (input := input) (vals := vals0) (i := i) (n := n)
                              (pId := pId) (axis := axis) (s := s)
                              (pT := getIdx (α := α) (xs := ctx) ip) (hAxisPf := hAxisPf)
                              (hN := hN) (hk := hk) (hp := hp) (hGet := hGet) (hAxis := hAxis)
                              (hOut := hOut))
                        have hStep :
                            denoteAllState (α := α) inShape st1 x =
                              vals0.push (NN.IR.DVal.mk (α := α) n.outShape
                                (nodeData.eval ctx)) := by
                          simpa [vals0, st1, nodeData, ctx] using
                            (denoteAllState_snoc (α := α) (inShape := inShape) (ss := ss) (τ := n.outShape)
                              (gd := gd) (nodeData := nodeData) (x := x))
                        have hTail := ih st1 hRec
                        exact buildFrom_denoteAllFrom_finish (α := α) (g := g) (payload := payload)
                          (i := i) (x := x) (hi := hi) (τ := n.outShape)
                          (nodeData := nodeData) (st1 := st1) (st' := st')
                          (ctx := ctx) (vals0 := vals0) (input := input) hTail hEval hStep
                      ·
                        have hCondNe :
                            ¬Spec.Tensor.shapeAfterSum pNode.outShape axis = n.outShape := by
                          intro hEq
                          apply hOut
                          simpa [expected, s] using hEq
                        simp [hCondNe] at hBuild
                        try cases hBuild

/-- Correctness lemma for `.sum` lowering (sum-reduction to scalar). -/
theorem buildFrom_denoteAllFrom_sum
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .sum) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := NN.IR.DVal.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := NN.IR.DVal.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (NN.IR.DVal α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TList α ([inShape] ++ ss) :=
    ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
  let input : NN.IR.DVal α := NN.IR.DVal.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp [hi, hN] at hBuild
  simp (config := { failIfUnchanged := false }) [hk] at hBuild
  cases hp : n.parents with
  | nil =>
      simp [hp] at hBuild; try cases hBuild
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          simp [hp] at hBuild; try cases hBuild
      | nil =>
          cases hP : g.getNode pId with
          | error msg =>
              simp [hp, hP] at hBuild; try cases hBuild
          | ok pNode =>
              simp [hp, hP] at hBuild
              let s := pNode.outShape
              cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId s with
              | error msg =>
                  simp [s, hIdx] at hBuild
              | ok ip =>
                  simp (config := { failIfUnchanged := false }) [s, hIdx] at hBuild
                  by_cases hOut : Shape.scalar = n.outShape
                  ·
                    simp [hOut] at hBuild
                    let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                      mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                        let x := getIdx (α := α) (xs := ctx) ip
                        hOut ▸ Tensor.scalar (Tensor.sumSpec (α := α) x))
                    let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                    have hRec :
                        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                          (i := i + 1) st1 = .ok st' := by
                      simpa [st1, nodeData] using hBuild
                    have hGet :
                        vals0[pId]! =
                          NN.IR.DVal.mk (α := α) s (getIdx (α := α) (xs := ctx) ip) := by
                      simpa [vals0, ctx] using
                        (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                          (gd := gd) (x := x) (pid := pId) (s := s) (idx := ip) hIdx)
                    have hEval :
                        NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                            (input := input) (vals := vals0) (i := i) =
                          .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                      have hGet' :
                          vals0[pId]! =
                            (⟨s, getIdx (α := α) (xs := ctx) ip⟩ : NN.IR.DVal α) := by
                        simpa [NN.IR.DVal.mk] using hGet
                      simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput, hN, hk, hp]
                      rw [hGet']
                      simp [hOut, nodeData, mkForwardNode, Tensor.eqRec_proof_irrel]
                      -- The remaining obligation (if any) is proof-irrelevance for the cast used to
                      -- type the scalar tensor.
                      rfl
                    have hStep :
                        denoteAllState (α := α) inShape st1 x =
                          vals0.push (NN.IR.DVal.mk (α := α) n.outShape
                            (nodeData.eval ctx)) := by
                      simpa [vals0, st1, nodeData, ctx] using
                        (denoteAllState_snoc (α := α) (inShape := inShape) (ss := ss) (τ := n.outShape)
                          (gd := gd) (nodeData := nodeData) (x := x))
                    have hTail := ih st1 hRec
                    exact buildFrom_denoteAllFrom_finish (α := α) (g := g) (payload := payload)
                      (i := i) (x := x) (hi := hi) (τ := n.outShape)
                      (nodeData := nodeData) (st1 := st1) (st' := st')
                      (ctx := ctx) (vals0 := vals0) (input := input) hTail hEval hStep
                  ·
                    exact False.elim <| throw_bind_ne_ok (h := (by simpa [hOut] using hBuild))

end IRExec
end Autograd
end Runtime

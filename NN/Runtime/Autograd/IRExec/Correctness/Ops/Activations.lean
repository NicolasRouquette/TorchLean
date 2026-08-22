/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Correctness.SemanticEquivalenceCommon

/-!
# Activation Operators

Semantic-preservation lemmas for unary activation operators in the IR-to-forward-executor lowering.

Each lemma mirrors the corresponding branch in the `Correctness.SemanticEquivalence` module and
gives that operator a stable theorem name. The main semantic equivalence proof can then focus on
graph traversal instead of carrying every parent-list and typed-index detail inline.

Build note: these proofs can be slower than the operators look. The activation itself is simple;
the proof cost comes from checking the singleton-parent contract, recovering a typed index from the
IR parent id, and showing that the dynamically evaluated `Spec.PackedTensor` is the same value as the lowered
node output. The shared unary-operator skeleton keeps each activation branch focused on its tensor
function.
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

/-- Semantic-preservation lemma for `.relu` lowering. -/
theorem buildFrom_denoteAllFrom_relu
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .relu) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.PackedTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.PackedTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (Spec.PackedTensor α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TList α ([inShape] ++ ss) :=
    ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
  let input : Spec.PackedTensor α := Spec.PackedTensor.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp (config := { failIfUnchanged := false }) [hi, hN, hk] at hBuild
  cases hp : n.parents with
  | nil =>
      simp [hp] at hBuild
      try cases hBuild
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          simp [hp] at hBuild
          try cases hBuild
      | nil =>
          cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId n.outShape with
          | error msg =>
              simp [hp, hIdx] at hBuild
              try cases hBuild
          | ok ip =>
              simp [hp, hIdx] at hBuild
              let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                  Activation.reluSpec (α := α) (getIdx (α := α) (xs := ctx) ip))
              let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
              have hRec :
                  buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                      (i := i + 1) st1 =
                    .ok st' := by
                simpa [st1, nodeData] using hBuild
              have hTail := ih st1 hRec
              have hGet :
                  vals0[pId]? = some (Spec.PackedTensor.mk (α := α) n.outShape (getIdx (α := α) (xs := ctx) ip)) := by
                simpa [vals0, ctx] using
                  (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                    (gd := gd) (x := x) (pid := pId) (s := n.outShape) (idx := ip) hIdx)
              have hEval :
                  NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                      (input := input) (vals := vals0) (i := i) =
                    .ok (Spec.PackedTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput, hN, hk, hp, hGet,
                  nodeData, mkForwardNode,
                  throw_eq_error]
              exact buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                (τ := n.outShape) (nodeData := nodeData) hTail hEval

/-- Semantic-preservation lemma for `.tanh` lowering. -/
theorem buildFrom_denoteAllFrom_tanh
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .tanh) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.PackedTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.PackedTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (Spec.PackedTensor α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TList α ([inShape] ++ ss) :=
    ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
  let input : Spec.PackedTensor α := Spec.PackedTensor.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp (config := { failIfUnchanged := false }) [hi, hN, hk] at hBuild
  cases hp : n.parents with
  | nil =>
      simp [hp] at hBuild
      try cases hBuild
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          simp [hp] at hBuild
          try cases hBuild
      | nil =>
          cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId n.outShape with
          | error msg =>
              simp [hp, hIdx] at hBuild
              try cases hBuild
          | ok ip =>
              simp [hp, hIdx] at hBuild
              let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                  Activation.tanhSpec (α := α) (getIdx (α := α) (xs := ctx) ip))
              let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
              have hRec :
                  buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                      (i := i + 1) st1 =
                    .ok st' := by
                simpa [st1, nodeData] using hBuild
              have hTail := ih st1 hRec
              have hGet :
                  vals0[pId]? = some (Spec.PackedTensor.mk (α := α) n.outShape (getIdx (α := α) (xs := ctx) ip)) := by
                simpa [vals0, ctx] using
                  (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                    (gd := gd) (x := x) (pid := pId) (s := n.outShape) (idx := ip) hIdx)
              have hEval :
                  NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                      (input := input) (vals := vals0) (i := i) =
                    .ok (Spec.PackedTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput, hN, hk, hp, hGet,
                  nodeData, mkForwardNode,
                  throw_eq_error]
              exact buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                (τ := n.outShape) (nodeData := nodeData) hTail hEval

/-- Semantic-preservation lemma for `.sigmoid` lowering. -/
theorem buildFrom_denoteAllFrom_sigmoid
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .sigmoid) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.PackedTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.PackedTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (Spec.PackedTensor α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TList α ([inShape] ++ ss) :=
    ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
  let input : Spec.PackedTensor α := Spec.PackedTensor.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp (config := { failIfUnchanged := false }) [hi, hN, hk] at hBuild
  cases hp : n.parents with
  | nil =>
      simp [hp] at hBuild
      try cases hBuild
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          simp [hp] at hBuild
          try cases hBuild
      | nil =>
          cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId n.outShape with
          | error msg =>
              simp [hp, hIdx] at hBuild
              try cases hBuild
          | ok ip =>
              simp [hp, hIdx] at hBuild
              let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                  Activation.sigmoidSpec (α := α) (getIdx (α := α) (xs := ctx) ip))
              let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
              have hRec :
                  buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                      (i := i + 1) st1 =
                    .ok st' := by
                simpa [st1, nodeData] using hBuild
              have hTail := ih st1 hRec
              have hGet :
                  vals0[pId]? = some (Spec.PackedTensor.mk (α := α) n.outShape (getIdx (α := α) (xs := ctx) ip)) := by
                simpa [vals0, ctx] using
                  (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                    (gd := gd) (x := x) (pid := pId) (s := n.outShape) (idx := ip) hIdx)
              have hEval :
                  NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                      (input := input) (vals := vals0) (i := i) =
                    .ok (Spec.PackedTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput, hN, hk, hp, hGet,
                  nodeData, mkForwardNode,
                  throw_eq_error]
              exact buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                (τ := n.outShape) (nodeData := nodeData) hTail hEval

/-- Semantic-preservation lemma for `.exp` lowering. -/
theorem buildFrom_denoteAllFrom_exp
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .exp) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.PackedTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.PackedTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (Spec.PackedTensor α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TList α ([inShape] ++ ss) :=
    ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
  let input : Spec.PackedTensor α := Spec.PackedTensor.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp (config := { failIfUnchanged := false }) [hi, hN, hk] at hBuild
  cases hp : n.parents with
  | nil =>
      simp [hp] at hBuild
      try cases hBuild
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          simp [hp] at hBuild
          try cases hBuild
      | nil =>
          cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId n.outShape with
          | error msg =>
              simp [hp, hIdx] at hBuild
              try cases hBuild
          | ok ip =>
              simp [hp, hIdx] at hBuild
              let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                  Tensor.expSpec (α := α) (getIdx (α := α) (xs := ctx) ip))
              let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
              have hRec :
                  buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                      (i := i + 1) st1 =
                    .ok st' := by
                simpa [st1, nodeData] using hBuild
              have hTail := ih st1 hRec
              have hGet :
                  vals0[pId]? = some (Spec.PackedTensor.mk (α := α) n.outShape (getIdx (α := α) (xs := ctx) ip)) := by
                simpa [vals0, ctx] using
                  (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                    (gd := gd) (x := x) (pid := pId) (s := n.outShape) (idx := ip) hIdx)
              have hEval :
                  NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                      (input := input) (vals := vals0) (i := i) =
                    .ok (Spec.PackedTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput, hN, hk, hp, hGet,
                  nodeData, mkForwardNode,
                  throw_eq_error]
              exact buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                (τ := n.outShape) (nodeData := nodeData) hTail hEval

/--
Positive-domain simplification for the lowered raw-log branch.

The end-to-end lowering theorem currently excludes raw `.log` through `NoRawLog`; a future theorem
can use this local fact after it carries per-node positivity facts through the graph.
-/
theorem rawLogForward_positive
    {α : Type} [Context α] {s : Shape} (x : Tensor α s)
    (h : Tensor.allSpec (α := α) (s := s) (fun v => decide (0 < v)) x = true) :
    (if Tensor.allSpec (α := α) (s := s) (fun v => decide (0 < v)) x = true then
      Tensor.logSpec (α := α) x
    else
      (Inhabited.default : Tensor α s)) =
      Tensor.logSpec (α := α) x := by
  simp [h]

/-- Semantic-preservation lemma for `.sin` lowering. -/
theorem buildFrom_denoteAllFrom_sin
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .sin) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.PackedTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.PackedTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (Spec.PackedTensor α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TList α ([inShape] ++ ss) :=
    ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
  let input : Spec.PackedTensor α := Spec.PackedTensor.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp (config := { failIfUnchanged := false }) [hi, hN, hk] at hBuild
  cases hp : n.parents with
  | nil =>
      simp [hp] at hBuild
      try cases hBuild
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          simp [hp] at hBuild
          try cases hBuild
      | nil =>
          cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId n.outShape with
          | error msg =>
              simp [hp, hIdx] at hBuild
              try cases hBuild
          | ok ip =>
              simp [hp, hIdx] at hBuild
              let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                  Tensor.mapSpec (α := α) (s := n.outShape) (fun x => MathFunctions.sin x)
                    (getIdx (α := α) (xs := ctx) ip))
              let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
              have hRec :
                  buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                      (i := i + 1) st1 =
                    .ok st' := by
                simpa [st1, nodeData] using hBuild
              have hTail := ih st1 hRec
              have hGet :
                  vals0[pId]? = some (Spec.PackedTensor.mk (α := α) n.outShape (getIdx (α := α) (xs := ctx) ip)) := by
                simpa [vals0, ctx] using
                  (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                    (gd := gd) (x := x) (pid := pId) (s := n.outShape) (idx := ip) hIdx)
              have hEval :
                  NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                      (input := input) (vals := vals0) (i := i) =
                    .ok (Spec.PackedTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput, hN, hk, hp, hGet,
                  nodeData, mkForwardNode,
                  throw_eq_error]
              exact buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                (τ := n.outShape) (nodeData := nodeData) hTail hEval

/-- Semantic-preservation lemma for `.cos` lowering. -/
theorem buildFrom_denoteAllFrom_cos
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .cos) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.PackedTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.PackedTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (Spec.PackedTensor α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TList α ([inShape] ++ ss) :=
    ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
  let input : Spec.PackedTensor α := Spec.PackedTensor.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp (config := { failIfUnchanged := false }) [hi, hN, hk] at hBuild
  cases hp : n.parents with
  | nil =>
      simp [hp] at hBuild
      try cases hBuild
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          simp [hp] at hBuild
          try cases hBuild
      | nil =>
          cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId n.outShape with
          | error msg =>
              simp [hp, hIdx] at hBuild
              try cases hBuild
          | ok ip =>
              simp [hp, hIdx] at hBuild
              let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                  Tensor.mapSpec (α := α) (s := n.outShape) (fun x => MathFunctions.cos x)
                    (getIdx (α := α) (xs := ctx) ip))
              let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
              have hRec :
                  buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                      (i := i + 1) st1 =
                    .ok st' := by
                simpa [st1, nodeData] using hBuild
              have hTail := ih st1 hRec
              have hGet :
                  vals0[pId]? = some (Spec.PackedTensor.mk (α := α) n.outShape (getIdx (α := α) (xs := ctx) ip)) := by
                simpa [vals0, ctx] using
                  (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                    (gd := gd) (x := x) (pid := pId) (s := n.outShape) (idx := ip) hIdx)
              have hEval :
                  NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                      (input := input) (vals := vals0) (i := i) =
                    .ok (Spec.PackedTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput, hN, hk, hp, hGet,
                  nodeData, mkForwardNode,
                  throw_eq_error]
              exact buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                (τ := n.outShape) (nodeData := nodeData) hTail hEval

/--
Semantic-preservation lemma for `.softmax axis` lowering.

The lowering accepts every axis that names a dimension of the output shape. The resulting typed
node uses the same axis-indexed specification as the denotational IR semantics.
-/
theorem buildFrom_denoteAllFrom_softmax
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (axis : Nat)
    (hN : g.getNode i = .ok n) (hk : n.kind = .softmax axis) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.PackedTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.PackedTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (Spec.PackedTensor α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TList α ([inShape] ++ ss) :=
    ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
  let input : Spec.PackedTensor α := Spec.PackedTensor.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp (config := { failIfUnchanged := false }) [hi, hN, hk] at hBuild
  cases hp : n.parents with
  | nil =>
      simp [hp] at hBuild
      try cases hBuild
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          simp [hp] at hBuild
          try cases hBuild
      | nil =>
          cases hAxis : Spec.Shape.axisInBounds? axis n.outShape with
          | none =>
              simp [hp, hAxis] at hBuild
              try cases hBuild
          | some h =>
              simp (config := { failIfUnchanged := false }) [hp, hAxis] at hBuild
              cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId n.outShape with
              | error msg =>
                  simp [hIdx] at hBuild
                  try cases hBuild
              | ok ip =>
                  simp [hIdx] at hBuild
                  let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                    mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                      @Activation.softmaxSpec α _ n.outShape axis h.down
                        (getIdx (α := α) (xs := ctx) ip))
                  let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                  have hRec :
                      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                        (i := i + 1) st1 = .ok st' := by
                    simpa [st1, nodeData] using hBuild
                  have hGet :
                      vals0[pId]? = some (Spec.PackedTensor.mk (α := α) n.outShape (getIdx (α := α) (xs := ctx) ip)) := by
                    simpa [vals0, ctx] using
                      (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                        (gd := gd) (x := x) (pid := pId) (s := n.outShape) (idx := ip) hIdx)
                  have hEval :
                      NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                          (input := input) (vals := vals0) (i := i) =
                        .ok (Spec.PackedTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                    simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                      NN.IR.Graph.normalizeNodeOutput, hN, hk, hp, hAxis, hGet, throw_eq_error,
                      Pure.pure, Except.pure, nodeData, mkForwardNode]
                  have hStep :
                      denoteAllState (α := α) inShape st1 x =
                        vals0.push (Spec.PackedTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                    simpa [vals0, st1, nodeData, ctx] using
                      (denoteAllState_snoc (α := α) (inShape := inShape) (ss := ss) (τ := n.outShape)
                        (gd := gd) (nodeData := nodeData) (x := x))
                  have hTail := ih st1 hRec
                  exact buildFrom_denoteAllFrom_finish (α := α) (g := g) (payload := payload)
                    (i := i) (x := x) (hi := hi) (τ := n.outShape)
                    (nodeData := nodeData) (st1 := st1) (st' := st')
                    (ctx := ctx) (vals0 := vals0) (input := input) hTail hEval hStep

/-- Semantic preservation for stable last-axis softmax with a hard Boolean mask. -/
theorem buildFrom_denoteAllFrom_hardMaskedSoftmax
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node) (mask : NN.IR.HardMask)
    (hN : g.getNode i = .ok n) (hk : n.kind = .hardMaskedSoftmax mask)
    (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.PackedTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.PackedTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (Spec.PackedTensor α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TList α ([inShape] ++ ss) :=
    ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
  let input : Spec.PackedTensor α := Spec.PackedTensor.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp (config := { failIfUnchanged := false }) [hi, hN, hk] at hBuild
  cases hp : n.parents with
  | nil =>
      simp [hp, throw_eq_error] at hBuild
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          simp [hp, throw_eq_error] at hBuild
      | nil =>
          cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId n.outShape with
          | error msg =>
              simp [hp, hIdx] at hBuild
          | ok ip =>
              cases hMask : NN.IR.HardMask.toTensorAs? mask n.outShape with
              | error msg =>
                  simp [hp, hIdx, hMask, throw_eq_error] at hBuild
              | ok allowed =>
                  simp (config := { failIfUnchanged := false }) [hp, hIdx, hMask] at hBuild
                  let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                    mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                      Spec.hardMaskedSoftmaxLastSpec
                        (getIdx (α := α) (xs := ctx) ip) allowed)
                  let st1 : State α inShape :=
                    ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                  have hRec :
                      buildFrom (α := α) (g := g) (payload := payload)
                          (inShape := inShape) (i := i + 1) st1 = .ok st' := by
                    simpa [st1, nodeData] using hBuild
                  have hGet :
                      vals0[pId]? = some (Spec.PackedTensor.mk (α := α) n.outShape
                          (getIdx (α := α) (xs := ctx) ip)) := by
                    simpa [vals0, ctx] using
                      (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                        (gd := gd) (x := x) (pid := pId) (s := n.outShape)
                        (idx := ip) hIdx)
                  have hEval :
                      NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                          (input := input) (vals := vals0) (i := i) =
                        .ok
                          (Spec.PackedTensor.mk (α := α) n.outShape
                            (nodeData.eval ctx)) := by
                    simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                      NN.IR.Graph.normalizeNodeOutput, hN, hk, hp, hGet, hMask,
                      throw_eq_error, Pure.pure, Except.pure, nodeData, mkForwardNode]
                  have hStep :
                      denoteAllState (α := α) inShape st1 x =
                        vals0.push
                          (Spec.PackedTensor.mk (α := α) n.outShape
                            (nodeData.eval ctx)) := by
                    simpa [vals0, st1, nodeData, ctx] using
                      (denoteAllState_snoc (α := α) (inShape := inShape) (ss := ss)
                        (τ := n.outShape) (gd := gd) (nodeData := nodeData) (x := x))
                  have hTail := ih st1 hRec
                  exact buildFrom_denoteAllFrom_finish (α := α) (g := g)
                    (payload := payload) (i := i) (x := x) (hi := hi)
                    (τ := n.outShape) (nodeData := nodeData) (st1 := st1) (st' := st')
                    (ctx := ctx) (vals0 := vals0) (input := input) hTail hEval hStep

end IRExec
end Autograd
end Runtime

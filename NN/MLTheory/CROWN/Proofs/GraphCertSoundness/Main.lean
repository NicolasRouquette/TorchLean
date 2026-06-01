/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.NonlinearOps

/-!
# Graph IBP Certificate Soundness

The induction theorem: local IBP certificate consistency plus local semantic consistency implies
that every certified node box encloses the corresponding semantic value.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN

namespace CertSoundness

noncomputable section

/-!
## Main theorem: local IBP certificate implies semantic enclosure (supported subset)

We use strong induction on node id, assuming a topological order:
every parent id is strictly smaller than the node id.
-/

/-- Topological order assumption: all parent ids are strictly smaller than the node id. -/
def TopoSorted (g : Graph) : Prop :=
  ∀ id : Nat, id < g.nodes.size →
    ∀ p : Nat, p ∈ (g.nodes[id]!).parents → p < id

/-- A graph is supported by this soundness theorem if every node kind is in our supported subset. -/
def Supported (g : Graph) : Prop :=
  ∀ id : Nat, id < g.nodes.size →
    match (g.nodes[id]!).kind with
    | .input | .const _ | .detach
    | .add | .sub | .mul_elem | .relu
    | .linear | .matmul
    | .tanh | .sigmoid | .sin | .cos => True
    | _ => False

/-- Inputs are well-formed if every `.input` node has a value, and that value is enclosed by
its input box from `ParamStore.inputBoxes`. -/
def InputsEnclosed (g : Graph) (ps : ParamStore ℝ) (inputs : Std.HashMap Nat Val) : Prop :=
  ∀ id : Nat, id < g.nodes.size →
    (g.nodes[id]!).kind = .input →
      ∃ B v, ps.inputBoxes[id]? = some B ∧ inputs[id]? = some v ∧ EnclosesBox B v

/-!
### The enclosure theorem

Assumptions:
* `TopoSorted g`: induction works (parents are earlier).
* `Supported g`: every node kind is handled by the proof.
* `CertLocalOK g ps cert`: the certificate is locally consistent with the IBP step.
* `InputsEnclosed g ps inputs`: semantic inputs are inside the certified input boxes.
* `SemLocalOK g ps inputs vals`: `vals` is a locally-consistent semantic interpretation.

Conclusion:
* For every node `id`, if the semantics produces a value `v` and the certificate has a box `B`,
  then `B` encloses `v`.
-/
theorem cert_encloses_semantics
    (g : Graph) (ps : ParamStore ℝ)
    (cert : Array (Option (FlatBox ℝ)))
    (inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val))
    (htopo : TopoSorted g)
    (hsupp : Supported g)
    (hcert : CertLocalOK (g := g) (ps := ps) cert)
    (hsem : SemLocalOK (g := g) (ps := ps) (inputs := inputs) vals)
    (hinputs : InputsEnclosed g ps inputs) :
    ∀ id : Nat, id < g.nodes.size →
      match cert[id]!, vals[id]! with
      | some B, some v => EnclosesBox B v
      | _, _ => True := by
  classical
  intro id hid
  -- Strong induction on `id` (parents are strictly smaller by `TopoSorted`).
  refine Nat.strong_induction_on id
      (p := fun k =>
        k < g.nodes.size →
          match cert[k]!, vals[k]! with
          | some B, some v => EnclosesBox B v
          | _, _ => True) ?_ hid
  intro k ih hk
  -- If certificate or value is missing, the goal is trivial.
  cases hcert with
  | intro hcertSz hnode =>
    cases hsem with
    | intro hvalsSz hsemNode =>
      have hcertk : cert[k]! = certStepNode? g.nodes ps cert k := hnode k hk
      have hvalk  : vals[k]! = evalNode? g.nodes ps inputs vals k := hsemNode k hk
      cases hck : cert[k]! <;> cases hvk : vals[k]! <;> simp
      case some.some B v =>
        -- From now on, we know both a certificate box and a semantic value exist.
        have hsupk := hsupp k hk
        have hcertStep : certStepNode? g.nodes ps cert k = some B := by
          have : (some B) = certStepNode? g.nodes ps cert k := by
            simpa [hck] using hcertk
          exact this.symm
        have hvalStep : evalNode? g.nodes ps inputs vals k = some v := by
          have : (some v) = evalNode? g.nodes ps inputs vals k := by
            simpa [hvk] using hvalk
          exact this.symm
        -- We'll use IH for parents.
        have parentIH :
            ∀ p : Nat, p ∈ (g.nodes[k]!).parents →
              match cert[p]!, vals[p]! with
              | some Bp, some vp => EnclosesBox Bp vp
              | _, _ => True := by
          intro p hp
          have hpk : p < k := htopo k hk p hp
          have hps : p < g.nodes.size := lt_trans hpk hk
          exact ih p hpk hps
        -- Now do a case split on the op kind.
        -- `Supported` lets us immediately discharge *unsupported* cases.
        cases hkKind : (g.nodes[k]!).kind <;>
          (simp [hkKind] at hsupk <;> try cases hsupk)
        case input =>
          rcases hinputs k hk hkKind with ⟨Bin, vin, hB, hv, hEnc⟩
          -- Use the *step equalities* to identify `B`/`v` with the input box/value.
          have hcertStepIn : certStepNode? g.nodes ps cert k = some Bin := by
            simp [certStepNode?, hkKind, hB]
          have hvalStepIn : evalNode? g.nodes ps inputs vals k = some vin := by
            simp [evalNode?, hkKind, hv]
          have hB_eq : B = Bin := by
            have : some B = some Bin := by simpa [hcertStep] using hcertStepIn
            cases this
            rfl
          have hv_eq : v = vin := by
            have : some v = some vin := by simpa [hvalStep] using hvalStepIn
            cases this
            rfl
          subst hB_eq
          subst hv_eq
          simpa using hEnc
        case const valueShape =>
          -- Both semantics and certificate read the same constant from `ps.constVals`.
          cases hconst : ps.constVals[k]? with
          | none =>
              -- If the parameter store has no constant here, the node cannot evaluate/certify.
              simp [certStepNode?, hkKind, hconst] at hcertStep
          | some val =>
              have hcertStep' := hcertStep
              have hvalStep' := hvalStep
              simp [certStepNode?, hkKind, hconst] at hcertStep'
              simp [evalNode?, hkKind, hconst] at hvalStep'
              cases hcertStep'
              cases hvalStep'
              -- Point box encloses its point (unfold enclosure and finish by simp).
              refine ⟨rfl, ?_⟩
              -- `castDimScalar rfl.symm` is definitional; use the point-box lemma.
              simpa [encloses, castDimScalar] using encloses_point_self_real (n := v.n) (x := v.v)
        case detach =>
          cases hparents : (g.nodes[k]!).parents with
          | nil =>
              -- With no parents, `certStepNode?` is `none`, contradicting `some B`.
              simp [certStepNode?, hkKind, hparents] at hcertStep
          | cons p1 _ =>
              have hp1c : p1 < cert.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hcertSz] using this
              have hp1v : p1 < vals.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hvalsSz] using this
              have hcertStep' : some B = getBox? cert p1 := by
                simpa [certStepNode?, hkKind, hparents] using (Eq.symm hcertStep)
              have hvalStep' : some v = getVal? vals p1 := by
                simpa [evalNode?, hkKind, hparents] using (Eq.symm hvalStep)
              -- Parent boxes/values must exist, otherwise the step would be `none`.
              cases hgb : getBox? cert p1 with
              | none =>
                  simp [hgb] at hcertStep'
              | some B1 =>
                  cases hgv : getVal? vals p1 with
                  | none =>
                      simp [hgv] at hvalStep'
                  | some v1 =>
                      have hB1 : cert[p1]! = some B1 := by
                        simpa [getBox?, hp1c] using hgb
                      have hv1 : vals[p1]! = some v1 := by
                        simpa [getVal?, hp1v] using hgv
                      have hpar' : EnclosesBox B1 v1 := by
                        have h := parentIH p1 (by simp [hparents])
                        simpa [hB1, hv1] using h
                      have hB_eq : B = B1 := by
                        have : some B = some B1 := by simpa [hgb] using hcertStep'
                        cases this
                        rfl
                      have hv_eq : v = v1 := by
                        have : some v = some v1 := by simpa [hgv] using hvalStep'
                        cases this
                        rfl
                      subst hB_eq
                      subst hv_eq
                      simpa using hpar'
        case add =>
          -- Extract parents.
          cases hparents : (g.nodes[k]!).parents with
          | nil =>
              -- `certStepNode?` would be `none`, contradicting `cert[k] = some`.
              simp [certStepNode?, hkKind, hparents] at hcertStep
          | cons p1 rest =>
              cases rest with
              | nil =>
                  simp [certStepNode?, hkKind, hparents] at hcertStep
              | cons p2 _ =>
                  -- From the certificate step: B = box_add Bp1 Bp2.
                  have hp1c : p1 < cert.size := by
                    have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                    simpa [hcertSz] using this
                  have hp2c : p2 < cert.size := by
                    have : p2 < g.nodes.size := lt_trans (htopo k hk p2 (by simp [hparents])) hk
                    simpa [hcertSz] using this
                  have hcertStep' :
                      some B =
                        match getBox? cert p1, getBox? cert p2 with
                        | some B1, some B2 =>
                            if _h : B1.dim = B2.dim then
                              some (box_add (α := ℝ) B1 B2)
                            else
                              none
                        | _, _ => none := by
                    simpa [certStepNode?, hkKind, hparents] using (Eq.symm hcertStep)
                  -- From the semantics step: v = x + y (and dims match).
                  have hp1v : p1 < vals.size := by
                    have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                    simpa [hvalsSz] using this
                  have hp2v : p2 < vals.size := by
                    have : p2 < g.nodes.size := lt_trans (htopo k hk p2 (by simp [hparents])) hk
                    simpa [hvalsSz] using this
                  have hvalStep' :
                      some v =
                        match getVal? vals p1, getVal? vals p2 with
                        | some x, some y =>
                            if h : x.n = y.n then
                              let yv : Tensor ℝ (.dim x.n .scalar) :=
                                castDimScalar (α := ℝ) (Eq.symm h) y.v
                              some { n := x.n, v := Tensor.addSpec (α := ℝ) x.v yv }
                            else none
                        | _, _ => none := by
                    simpa [evalNode?, hkKind, hparents] using (Eq.symm hvalStep)
                  -- Extract concrete parent boxes/values (they must exist since this node produced
                  -- `some`).
                  cases hgb1 : getBox? cert p1 with
                  | none =>
                      simp [hgb1] at hcertStep'
                  | some B1 =>
                      cases hgb2 : getBox? cert p2 with
                      | none =>
                          simp [hgb1, hgb2] at hcertStep'
                      | some B2 =>
                          cases hgv1 : getVal? vals p1 with
                          | none =>
                              simp [hgv1] at hvalStep'
                          | some v1 =>
                              cases hgv2 : getVal? vals p2 with
                              | none =>
                                  simp [hgv1, hgv2] at hvalStep'
                              | some v2 =>
                                  have hB1 : cert[p1]! = some B1 := by
                                    simpa [getBox?, hp1c] using hgb1
                                  have hB2 : cert[p2]! = some B2 := by
                                    simpa [getBox?, hp2c] using hgb2
                                  have hv1 : vals[p1]! = some v1 := by
                                    simpa [getVal?, hp1v] using hgv1
                                  have hv2 : vals[p2]! = some v2 := by
                                    simpa [getVal?, hp2v] using hgv2
                                  have hpar1' : EnclosesBox B1 v1 := by
                                    have h := parentIH p1 (by simp [hparents])
                                    simpa [hB1, hv1] using h
                                  have hpar2' : EnclosesBox B2 v2 := by
                                    have h := parentIH p2 (by simp [hparents])
                                    simpa [hB2, hv2] using h
                                  rcases hpar1' with ⟨hDim1, hxEnc⟩
                                  rcases hpar2' with ⟨hDim2, hyEnc⟩
                                  by_cases hxy : v1.n = v2.n
                                  ·
                                  -- Dimensions agree; prove enclosure using `box_add_sound` plus
                                  -- casts.
                                    have hBB : B1.dim = B2.dim :=
                                      Eq.trans hDim1 (Eq.trans hxy (Eq.symm hDim2))
                                    have hcertStep'' := hcertStep'
                                    simp [hgb1, hgb2, hBB] at hcertStep''
                                    cases hcertStep''
                                    have hvalStep'' := hvalStep'
                                    simp [hgv1, hgv2, hxy] at hvalStep''
                                    cases hvalStep''
                                    -- Make `box_add` reducible by destructing the boxes and using
                                    -- the equal-dimension branch.
                                    cases B1 with
                                    | mk n1 lo1 hi1 =>
                                      cases B2 with
                                      | mk n2 lo2 hi2 =>
                                        have hBB' : n1 = n2 := by
                                          simpa using hBB
                                        cases hBB'
                                        let x : Tensor ℝ (.dim n1 .scalar) :=
                                          castDimScalar (α := ℝ) hDim1.symm v1.v
                                        let y : Tensor ℝ (.dim n1 .scalar) :=
                                          castDimScalar (α := ℝ) hDim2.symm v2.v
                                        have hEncl :
                                            encloses
                                              { dim := n1
                                                lo := Tensor.addSpec (α := ℝ) lo1 lo2
                                                hi := Tensor.addSpec (α := ℝ) hi1 hi2 }
                                              (Tensor.addSpec (α := ℝ) x y) :=
                                          NN.MLTheory.CROWN.Graph.Theorems.Semantics.box_add_sound
                                            (α := ℝ) (n := n1)
                                            (lo1 := lo1) (hi1 := hi1) (lo2 := lo2) (hi2 := hi2)
                                            (add_mono := add_mono_real)
                                            (x := x) (y := y)
                                            (hx := by simpa [x] using hxEnc)
                                            (hy := by simpa [y] using hyEnc)
                                        have hBoxEq :
                                            box_add (α := ℝ)
                                                { dim := n1, lo := lo1, hi := hi1 }
                                                { dim := n1, lo := lo2, hi := hi2 }
                                              =
                                                { dim := n1
                                                  lo := Tensor.addSpec (α := ℝ) lo1 lo2
                                                  hi := Tensor.addSpec (α := ℝ) hi1 hi2 } := by
                                          simpa using
                                            (NN.MLTheory.CROWN.Graph.Theorems.box_add_on_eq (α := ℝ)
                                              n1 lo1 hi1 lo2 hi2)
                                        have hProof :
                                            Eq.trans hxy.symm hDim1.symm = hDim2.symm := by
                                          apply Subsingleton.elim
                                        have hValEq :
                                            castDimScalar (α := ℝ) hDim1.symm
                                                (Tensor.addSpec (α := ℝ) v1.v
                                                  (castDimScalar (α := ℝ) hxy.symm v2.v))
                                              = Tensor.addSpec (α := ℝ) x y := by
                                          have h1 :
                                              castDimScalar (α := ℝ) hDim1.symm
                                                (Tensor.addSpec (α := ℝ) v1.v
                                                  (castDimScalar (α := ℝ) hxy.symm v2.v))
                                                =
                                                  Tensor.addSpec (α := ℝ)
                                                    (castDimScalar (α := ℝ) hDim1.symm v1.v)
                                                    (castDimScalar (α := ℝ) hDim1.symm
                                                      (castDimScalar (α := ℝ) hxy.symm v2.v)) := by
                                            simpa using
                                              castDimScalar_add_spec (h := hDim1.symm) (x := v1.v)
                                                (y := castDimScalar (α := ℝ) hxy.symm v2.v)
                                          have h2 :
                                              castDimScalar (α := ℝ) hDim1.symm
                                                (castDimScalar (α := ℝ) hxy.symm v2.v)
                                                = y := by
                                            have := (castDimScalar_trans (h₁ := hxy.symm) (h₂ :=
                                              hDim1.symm) (t := v2.v)).symm
                                            simpa [y, hProof] using this
                                          simpa [x, y, h2] using h1
                                        -- Prove enclosure for the canonical “equal-dimension”
                                        -- result box,
                                        -- then rewrite the goal box (`box_add …`) using `hBoxEq`.
                                        have hCanon :
                                            EnclosesBox
                                              { dim := n1
                                                lo := Tensor.addSpec (α := ℝ) lo1 lo2
                                                hi := Tensor.addSpec (α := ℝ) hi1 hi2 }
                                              { n := v1.n
                                                v :=
                                                  Tensor.addSpec (α := ℝ) v1.v
                                                    (castDimScalar (α := ℝ) hxy.symm v2.v) } := by
                                          refine ⟨hDim1, ?_⟩
                                          have : encloses
                                              { dim := n1
                                                lo := Tensor.addSpec (α := ℝ) lo1 lo2
                                                hi := Tensor.addSpec (α := ℝ) hi1 hi2 }
                                              (Tensor.addSpec (α := ℝ) x y) := hEncl
                                          simpa [hValEq] using this
                                        simpa [hBoxEq] using hCanon
                                  ·
                                    simp [hgv1, hgv2, hxy] at hvalStep'

        case mul_elem =>
          cases hparents : (g.nodes[k]!).parents with
          | nil =>
              simp [certStepNode?, hkKind, hparents] at hcertStep
          | cons p1 rest =>
              cases rest with
              | nil =>
                  simp [certStepNode?, hkKind, hparents] at hcertStep
              | cons p2 _ =>
                  have hp1c : p1 < cert.size := by
                    have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                    simpa [hcertSz] using this
                  have hp2c : p2 < cert.size := by
                    have : p2 < g.nodes.size := lt_trans (htopo k hk p2 (by simp [hparents])) hk
                    simpa [hcertSz] using this
                  have hp1v : p1 < vals.size := by
                    have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                    simpa [hvalsSz] using this
                  have hp2v : p2 < vals.size := by
                    have : p2 < g.nodes.size := lt_trans (htopo k hk p2 (by simp [hparents])) hk
                    simpa [hvalsSz] using this
                  have hcertStep' :
                      some B =
                        match getBox? cert p1, getBox? cert p2 with
                        | some B1, some B2 => box_mul_elem (α := ℝ) B1 B2
                        | _, _ => none := by
                    simpa [certStepNode?, hkKind, hparents] using (Eq.symm hcertStep)
                  have hvalStep' :
                      some v =
                        match getVal? vals p1, getVal? vals p2 with
                        | some x, some y =>
                            if h : x.n = y.n then
                              let yv : Tensor ℝ (.dim x.n .scalar) :=
                                castDimScalar (α := ℝ) (Eq.symm h) y.v
                              some { n := x.n, v := Tensor.mulSpec (α := ℝ) x.v yv }
                            else none
                        | _, _ => none := by
                    simpa [evalNode?, hkKind, hparents] using (Eq.symm hvalStep)
                  cases hgb1 : getBox? cert p1 with
                  | none =>
                      simp [hgb1] at hcertStep'
                  | some B1 =>
                      cases hgb2 : getBox? cert p2 with
                      | none =>
                          simp [hgb1, hgb2] at hcertStep'
                      | some B2 =>
                          cases hgv1 : getVal? vals p1 with
                          | none =>
                              simp [hgv1] at hvalStep'
                          | some v1 =>
                              cases hgv2 : getVal? vals p2 with
                              | none =>
                                  simp [hgv1, hgv2] at hvalStep'
                              | some v2 =>
                                  have hB1 : cert[p1]! = some B1 := by
                                    simpa [getBox?, hp1c] using hgb1
                                  have hB2 : cert[p2]! = some B2 := by
                                    simpa [getBox?, hp2c] using hgb2
                                  have hv1 : vals[p1]! = some v1 := by
                                    simpa [getVal?, hp1v] using hgv1
                                  have hv2 : vals[p2]! = some v2 := by
                                    simpa [getVal?, hp2v] using hgv2
                                  have hpar1 : EnclosesBox B1 v1 := by
                                    have h := parentIH p1 (by simp [hparents])
                                    simpa [hB1, hv1] using h
                                  have hpar2 : EnclosesBox B2 v2 := by
                                    have h := parentIH p2 (by simp [hparents])
                                    simpa [hB2, hv2] using h
                                  rcases hpar1 with ⟨hDim1, hx1⟩
                                  rcases hpar2 with ⟨hDim2, hx2⟩
                                  have hcertStep'' : some B = box_mul_elem (α := ℝ) B1 B2 := by
                                    simpa [hgb1, hgb2] using hcertStep'
                                  cases hmul : box_mul_elem (α := ℝ) B1 B2 with
                                  | none =>
                                      simp [hmul] at hcertStep''
                                  | some Bmul =>
                                      have hB_eq : B = Bmul := by
                                        have h := hcertStep''
                                        simp [hmul] at h
                                        cases h
                                        rfl
                                      subst B
                                      by_cases hxy : v1.n = v2.n
                                      · -- Unfold the value step under the `dims match` branch.
                                        have hvEq :
                                            v =
                                              ⟨v1.n,
                                                Tensor.mulSpec (α := ℝ) v1.v
                                                  (castDimScalar (α := ℝ) (Eq.symm hxy) v2.v)⟩ := by
                                          have : (some v : Option Val) =
                                              some
                                                ⟨v1.n,
                                                  Tensor.mulSpec (α := ℝ) v1.v
                                                    (castDimScalar (α := ℝ) (Eq.symm hxy) v2.v)⟩ :=
                                                      by
                                            simpa [hgv1, hgv2, hxy] using hvalStep'
                                          cases this
                                          rfl
                                        subst hvEq
                                        -- Reduce to tensors at dimension `Bmul.dim` and apply
                                        -- `box_mul_elem_sound_real`.
                                        cases B1 with
                                        | mk n1 lo1 hi1 =>
                                          cases B2 with
                                          | mk n2 lo2 hi2 =>
                                            -- `Bmul` coming from `box_mul_elem` forces equal
                                            -- dimensions.
                                            have hn12 : n1 = n2 := by
                                              by_contra hne
                                              have : box_mul_elem (α := ℝ)
                                                  { dim := n1, lo := lo1, hi := hi1 }
                                                  { dim := n2, lo := lo2, hi := hi2 } = none := by
                                                unfold box_mul_elem
                                                simp [hne]
                                              have : False := by
                                                simp [this] at hmul
                                              exact this.elim
                                            cases hn12
                                            let x : Tensor ℝ (.dim n1 .scalar) :=
                                              castDimScalar (α := ℝ) hDim1.symm v1.v
                                            let y : Tensor ℝ (.dim n1 .scalar) :=
                                              castDimScalar (α := ℝ) hDim2.symm v2.v
                                            have hx : encloses { dim := n1, lo := lo1, hi := hi1 } x
                                              := by
                                              simpa [x] using hx1
                                            have hy : encloses { dim := n1, lo := lo2, hi := hi2 } y
                                              := by
                                              simpa [y] using hx2
                                            have hMulEncl :
                                                EnclosesBox Bmul ⟨n1, Tensor.mulSpec (α := ℝ) x y⟩
                                                  :=
                                              box_mul_elem_sound_real (n := n1)
                                                (lo1 := lo1) (hi1 := hi1) (lo2 := lo2) (hi2 := hi2)
                                                  (x := x) (y := y) hx hy
                                                (B := Bmul) (by simpa using hmul)
                                            rcases hMulEncl with ⟨hDimMul, hEncMul⟩
                                            -- We need enclosure for the semantic value `v`, whose
                                            -- dimension is `v1.n`.
                                            -- Use `hDimMul : Bmul.dim = n1` and `hDim1 : n1 =
                                            -- v1.n`.
                                            let hW : Bmul.dim = v1.n := Eq.trans hDimMul hDim1
                                            refine ⟨hW, ?_⟩
                                            -- Rewrite the value cast into the `x*y` cast used by
                                            -- `hEncMul`.
                                            have hvCast0 :
                                                castDimScalar (α := ℝ) hDim1.symm
                                                    (Tensor.mulSpec (α := ℝ) v1.v
                                                      (castDimScalar (α := ℝ) (Eq.symm hxy) v2.v))
                                                  =
                                                  Tensor.mulSpec (α := ℝ) x y := by
                                              -- Commute the cast with multiplication and rewrite
                                              -- the second input cast into `y`.
                                              have hyCast :
                                                  castDimScalar (α := ℝ) hDim1.symm
                                                      (castDimScalar (α := ℝ) (Eq.symm hxy) v2.v)
                                                    = y := by
                                                -- Both sides are casts of `v2.v` to dimension `n1`.
                                                have hEq :
                                                    Eq.trans (Eq.symm hxy) hDim1.symm = hDim2.symm
                                                      := by
                                                  apply Subsingleton.elim
                                                -- Reassociate casts and rewrite.
                                                have hnest :
                                                    castDimScalar (α := ℝ) hDim1.symm
                                                        (castDimScalar (α := ℝ) (Eq.symm hxy) v2.v)
                                                      =
                                                      castDimScalar (α := ℝ) (Eq.trans (Eq.symm hxy)
                                                        hDim1.symm) v2.v := by
                                                  simpa using
                                                    (castDimScalar_trans (h₁ := (Eq.symm hxy)) (h₂
                                                      := hDim1.symm) (t := v2.v)).symm
                                                simpa [y, hEq] using hnest
                                              -- Now simplify using `x` and `hyCast`.
                                              simp [x, castDimScalar_mul_spec, hyCast]
                                            have hvCast :
                                                castDimScalar (α := ℝ) hW.symm
                                                    (Tensor.mulSpec (α := ℝ) v1.v
                                                      (castDimScalar (α := ℝ) (Eq.symm hxy) v2.v))
                                                  =
                                                  castDimScalar (α := ℝ) hDimMul.symm
                                                    (Tensor.mulSpec (α := ℝ) x y) := by
                                              -- `hW.symm` and `hDim1.symm.trans hDimMul.symm` are
                                              -- both proofs of `v1.n = Bmul.dim`.
                                              have hts : hW.symm = Eq.trans hDim1.symm hDimMul.symm
                                                := by
                                                apply Subsingleton.elim
                                              -- Expand `hW.symm` into a composite cast, then use
                                              -- `hvCast0`.
                                              calc
                                                castDimScalar (α := ℝ) hW.symm
                                                    (Tensor.mulSpec (α := ℝ) v1.v
                                                      (castDimScalar (α := ℝ) (Eq.symm hxy) v2.v))
                                                    =
                                                    castDimScalar (α := ℝ) hDimMul.symm
                                                      (castDimScalar (α := ℝ) hDim1.symm
                                                        (Tensor.mulSpec (α := ℝ) v1.v
                                                          (castDimScalar (α := ℝ) (Eq.symm hxy)
                                                            v2.v))) := by
                                                      -- Reassociate casts along `hDim1.symm` then
                                                      -- `hDimMul.symm`.
                                                      simpa [hts] using
                                                        (castDimScalar_trans (h₁ := hDim1.symm) (h₂
                                                          := hDimMul.symm)
                                                          (t := Tensor.mulSpec (α := ℝ) v1.v
                                                            (castDimScalar (α := ℝ) (Eq.symm hxy)
                                                              v2.v)))
                                                _ = castDimScalar (α := ℝ) hDimMul.symm
                                                  (Tensor.mulSpec (α := ℝ) x y) := by
                                                      simpa using
                                                        congrArg (fun t => castDimScalar (α := ℝ)
                                                          hDimMul.symm t) hvCast0
                                            -- `hEncMul` is already phrased using `hDimMul`.
                                            simpa [hvCast] using hEncMul
                                      ·
                                        simp [hgv1, hgv2, hxy] at hvalStep'

        case sub =>
          -- Similar to `.add`, using `Theorems.Semantics.box_sub_sound`.
          cases hparents : (g.nodes[k]!).parents with
          | nil =>
              simp [certStepNode?, hkKind, hparents] at hcertStep
          | cons p1 rest =>
              cases rest with
              | nil =>
                  simp [certStepNode?, hkKind, hparents] at hcertStep
              | cons p2 _ =>
                  have hp1c : p1 < cert.size := by
                    have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                    simpa [hcertSz] using this
                  have hp2c : p2 < cert.size := by
                    have : p2 < g.nodes.size := lt_trans (htopo k hk p2 (by simp [hparents])) hk
                    simpa [hcertSz] using this
                  have hp1v : p1 < vals.size := by
                    have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                    simpa [hvalsSz] using this
                  have hp2v : p2 < vals.size := by
                    have : p2 < g.nodes.size := lt_trans (htopo k hk p2 (by simp [hparents])) hk
                    simpa [hvalsSz] using this
                  have hcertStep' :
                      some B =
                        match getBox? cert p1, getBox? cert p2 with
                        | some B1, some B2 =>
                            if _h : B1.dim = B2.dim then
                              some (box_sub (α := ℝ) B1 B2)
                            else
                              none
                        | _, _ => none := by
                    simpa [certStepNode?, hkKind, hparents] using (Eq.symm hcertStep)
                  have hvalStep' :
                      some v =
                        match getVal? vals p1, getVal? vals p2 with
                        | some x, some y =>
                            if h : x.n = y.n then
                              let yv : Tensor ℝ (.dim x.n .scalar) :=
                                castDimScalar (α := ℝ) (Eq.symm h) y.v
                              some { n := x.n, v := Tensor.subSpec (α := ℝ) x.v yv }
                            else none
                        | _, _ => none := by
                    simpa [evalNode?, hkKind, hparents] using (Eq.symm hvalStep)
                  cases hgb1 : getBox? cert p1 with
                    | none =>
                        simp [hgb1] at hcertStep'
                    | some B1 =>
                        cases hgb2 : getBox? cert p2 with
                        | none =>
                            simp [hgb1, hgb2] at hcertStep'
                        | some B2 =>
                            cases hgv1 : getVal? vals p1 with
                            | none =>
                                simp [hgv1] at hvalStep'
                            | some v1 =>
                                cases hgv2 : getVal? vals p2 with
                                | none =>
                                    simp [hgv1, hgv2] at hvalStep'
                                | some v2 =>
                                  have hB1 : cert[p1]! = some B1 := by
                                    simpa [getBox?, hp1c] using hgb1
                                  have hB2 : cert[p2]! = some B2 := by
                                    simpa [getBox?, hp2c] using hgb2
                                  have hv1 : vals[p1]! = some v1 := by
                                    simpa [getVal?, hp1v] using hgv1
                                  have hv2 : vals[p2]! = some v2 := by
                                    simpa [getVal?, hp2v] using hgv2
                                  have hpar1' : EnclosesBox B1 v1 := by
                                    have h := parentIH p1 (by simp [hparents])
                                    simpa [hB1, hv1] using h
                                  have hpar2' : EnclosesBox B2 v2 := by
                                    have h := parentIH p2 (by simp [hparents])
                                    simpa [hB2, hv2] using h
                                  rcases hpar1' with ⟨hDim1, hxEnc⟩
                                  rcases hpar2' with ⟨hDim2, hyEnc⟩
                                  by_cases hxy : v1.n = v2.n
                                  ·
                                    have hBB : B1.dim = B2.dim :=
                                      Eq.trans hDim1 (Eq.trans hxy (Eq.symm hDim2))
                                    have hcertStep'' := hcertStep'
                                    simp [hgb1, hgb2, hBB] at hcertStep''
                                    cases hcertStep''
                                    have hvalStep'' := hvalStep'
                                    simp [hgv1, hgv2, hxy] at hvalStep''
                                    cases hvalStep''
                                    cases B1 with
                                    | mk n1 lo1 hi1 =>
                                      cases B2 with
                                      | mk n2 lo2 hi2 =>
                                        have hBB' : n1 = n2 := by
                                          simpa using hBB
                                        cases hBB'
                                        let x : Tensor ℝ (.dim n1 .scalar) :=
                                          castDimScalar (α := ℝ) hDim1.symm v1.v
                                        let y : Tensor ℝ (.dim n1 .scalar) :=
                                          castDimScalar (α := ℝ) hDim2.symm v2.v
                                        have hEncl :
                                            encloses
                                              { dim := n1
                                                lo := Tensor.subSpec (α := ℝ) lo1 hi2
                                                hi := Tensor.subSpec (α := ℝ) hi1 lo2 }
                                              (Tensor.subSpec (α := ℝ) x y) :=
                                          NN.MLTheory.CROWN.Graph.Theorems.Semantics.box_sub_sound
                                            (α := ℝ) (n := n1)
                                            (lo1 := lo1) (hi1 := hi1) (lo2 := lo2) (hi2 := hi2)
                                            (sub_mono := sub_mono_real)
                                            (x := x) (y := y)
                                            (hx := by simpa [x] using hxEnc)
                                            (hy := by simpa [y] using hyEnc)
                                        have hBoxEq :
                                            box_sub (α := ℝ)
                                                { dim := n1, lo := lo1, hi := hi1 }
                                                { dim := n1, lo := lo2, hi := hi2 }
                                              =
                                                { dim := n1
                                                  lo := Tensor.subSpec (α := ℝ) lo1 hi2
                                                  hi := Tensor.subSpec (α := ℝ) hi1 lo2 } := by
                                          simpa using
                                            (NN.MLTheory.CROWN.Graph.Theorems.box_sub_on_eq (α := ℝ)
                                              n1 lo1 hi1 lo2 hi2)
                                        have hProof :
                                            Eq.trans hxy.symm hDim1.symm = hDim2.symm := by
                                          apply Subsingleton.elim
                                        have hValEq :
                                            castDimScalar (α := ℝ) hDim1.symm
                                                (Tensor.subSpec (α := ℝ) v1.v
                                                  (castDimScalar (α := ℝ) hxy.symm v2.v))
                                              = Tensor.subSpec (α := ℝ) x y := by
                                          have h1 :
                                              castDimScalar (α := ℝ) hDim1.symm
                                                (Tensor.subSpec (α := ℝ) v1.v
                                                  (castDimScalar (α := ℝ) hxy.symm v2.v))
                                                =
                                                  Tensor.subSpec (α := ℝ)
                                                    (castDimScalar (α := ℝ) hDim1.symm v1.v)
                                                    (castDimScalar (α := ℝ) hDim1.symm
                                                      (castDimScalar (α := ℝ) hxy.symm v2.v)) := by
                                            simpa using
                                              castDimScalar_sub_spec (h := hDim1.symm) (x := v1.v)
                                                (y := castDimScalar (α := ℝ) hxy.symm v2.v)
                                          have h2 :
                                              castDimScalar (α := ℝ) hDim1.symm
                                                (castDimScalar (α := ℝ) hxy.symm v2.v)
                                                = y := by
                                            have := (castDimScalar_trans (h₁ := hxy.symm) (h₂ :=
                                              hDim1.symm) (t := v2.v)).symm
                                            simpa [y, hProof] using this
                                          simpa [x, y, h2] using h1
                                        have hCanon :
                                            EnclosesBox
                                              { dim := n1
                                                lo := Tensor.subSpec (α := ℝ) lo1 hi2
                                                hi := Tensor.subSpec (α := ℝ) hi1 lo2 }
                                              { n := v1.n
                                                v :=
                                                  Tensor.subSpec (α := ℝ) v1.v
                                                    (castDimScalar (α := ℝ) hxy.symm v2.v) } := by
                                          refine ⟨hDim1, ?_⟩
                                          have : encloses
                                              { dim := n1
                                                lo := Tensor.subSpec (α := ℝ) lo1 hi2
                                                hi := Tensor.subSpec (α := ℝ) hi1 lo2 }
                                              (Tensor.subSpec (α := ℝ) x y) := hEncl
                                          simpa [hValEq] using this
                                        simpa [hBoxEq] using hCanon
                                  ·
                                    simp [hgv1, hgv2, hxy] at hvalStep'

        case relu =>
          cases hparents : (g.nodes[k]!).parents with
          | nil =>
              simp [certStepNode?, hkKind, hparents] at hcertStep
          | cons p1 _ =>
              have hp1c : p1 < cert.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hcertSz] using this
              have hp1v : p1 < vals.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hvalsSz] using this
              have hcertStep' :
                  some B =
                    match getBox? cert p1 with
                    | some B1 => some (box_relu (α := ℝ) B1)
                    | none => none := by
                simpa [certStepNode?, hkKind, hparents] using (Eq.symm hcertStep)
              have hvalStep' :
                  some v =
                    match getVal? vals p1 with
                    | some x => some { n := x.n, v := Activation.reluSpec (α := ℝ) x.v }
                    | none => none := by
                simpa [evalNode?, hkKind, hparents] using (Eq.symm hvalStep)
              cases hgb : getBox? cert p1 with
              | none =>
                  simp [hgb] at hcertStep'
              | some B1 =>
                  cases hgv : getVal? vals p1 with
                  | none =>
                      simp [hgv] at hvalStep'
                  | some v1 =>
                      have hB1 : cert[p1]! = some B1 := by
                        simpa [getBox?, hp1c] using hgb
                      have hv1 : vals[p1]! = some v1 := by
                        simpa [getVal?, hp1v] using hgv
                      have hpar' : EnclosesBox B1 v1 := by
                        have h := parentIH p1 (by simp [hparents])
                        simpa [hB1, hv1] using h
                      have hB_eq : B = box_relu (α := ℝ) B1 := by
                        have : some B = some (box_relu (α := ℝ) B1) := by
                          simpa [hgb] using hcertStep'
                        cases this
                        rfl
                      have hv_eq : v = { n := v1.n, v := Activation.reluSpec (α := ℝ) v1.v } := by
                        have : some v = some { n := v1.n, v := Activation.reluSpec (α := ℝ) v1.v }
                          := by
                          simpa [hgv] using hvalStep'
                        cases this
                        rfl
                      subst hB_eq
                      subst hv_eq
                      rcases hpar' with ⟨hDim, hxEnc⟩
                      have hDimOut : (box_relu (α := ℝ) B1).dim = v1.n := by
                        exact Eq.trans
                          (NN.MLTheory.CROWN.Graph.Theorems.box_relu_dim (α := ℝ) B1)
                          hDim
                      refine ⟨hDimOut, ?_⟩
                      -- Use the existing `box_relu_sound` lemma at dimension `B1.dim`,
                      -- and then rewrite the semantic value (a cast of `relu_spec v1.v`)
                      -- into `relu_spec` of the casted input.
                      have hrelu :=
                        (NN.MLTheory.CROWN.Graph.Theorems.Semantics.box_relu_sound (α := ℝ)
                          (n := B1.dim) (lo := B1.lo) (hi := B1.hi)
                          (relu_mono := relu_mono_real)
                          (x := castDimScalar (α := ℝ) hDim.symm v1.v)
                          (hx := by
                            simpa [castDimScalar] using hxEnc))
                      have hcastProof : hDimOut.symm = hDim.symm := by
                        -- Both sides are proofs of `v1.n = B1.dim`.
                        apply Subsingleton.elim
                      have hvCast :
                          castDimScalar (α := ℝ) hDimOut.symm (Activation.reluSpec (α := ℝ) v1.v)
                            = Activation.reluSpec (α := ℝ) (castDimScalar (α := ℝ) hDim.symm v1.v)
                              := by
                        -- Commute the cast with `map_spec` (ReLU is elementwise).
                        simpa [hcastProof, Activation.reluSpec] using
                          (castDimScalar_map_spec (h := hDim.symm)
                            (f := Activation.Math.reluSpec (α := ℝ)) (t := v1.v))
                      -- `box_relu_sound` already inserts a cast to match `(box_relu ...).dim`;
                      -- after rewriting the value cast, it is exactly the enclosure we need.
                      simpa [EnclosesBox, hvCast] using hrelu
        case tanh =>
          cases hparents : (g.nodes[k]!).parents with
          | nil =>
              simp [certStepNode?, hkKind, hparents] at hcertStep
          | cons p1 _ =>
              have hp1c : p1 < cert.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hcertSz] using this
              have hp1v : p1 < vals.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hvalsSz] using this
              have hcertStep' :
                  some B =
                    match getBox? cert p1 with
                    | some Xin =>
                        let yB := NN.MLTheory.CROWN.Runtime.Ops.IBP.tanh (α := ℝ) (n := Xin.dim)
                          (ofFlatBox (α := ℝ) Xin)
                        some (toFlatBox (α := ℝ) Xin.dim yB)
                    | none => none := by
                simpa [certStepNode?, hkKind, hparents] using (Eq.symm hcertStep)
              have hvalStep' :
                  some v =
                    match getVal? vals p1 with
                    | some x => some { n := x.n, v := Activation.tanhSpec (α := ℝ) x.v }
                    | none => none := by
                simpa [evalNode?, hkKind, hparents] using (Eq.symm hvalStep)
              cases hgb : getBox? cert p1 with
              | none =>
                  simp [hgb] at hcertStep'
              | some B1 =>
                  cases hgv : getVal? vals p1 with
                  | none =>
                      simp [hgv] at hvalStep'
                  | some v1 =>
                      have hB1 : cert[p1]! = some B1 := by
                        simpa [getBox?, hp1c] using hgb
                      have hv1 : vals[p1]! = some v1 := by
                        simpa [getVal?, hp1v] using hgv
                      have hpar' : EnclosesBox B1 v1 := by
                        have h := parentIH p1 (by simp [hparents])
                        simpa [hB1, hv1] using h
                      rcases hpar' with ⟨hDim, hxEnc⟩
                      have hcertStep'' := hcertStep'
                      have hvalStep'' := hvalStep'
                      simp [hgb] at hcertStep''
                      simp [hgv] at hvalStep''
                      cases hcertStep''
                      cases hvalStep''
                      have hxContains :
                          Box.contains (α := ℝ) (ofFlatBox (α := ℝ) B1)
                            (castDimScalar (α := ℝ) hDim.symm v1.v) :=
                        contains_of_encloses (B := B1)
                          (x := castDimScalar (α := ℝ) hDim.symm v1.v) (by simpa using hxEnc)
                      have houtContains :
                          Box.contains (α := ℝ)
                            (NN.MLTheory.CROWN.Runtime.Ops.IBP.tanh (α := ℝ) (n := B1.dim)
                              (ofFlatBox (α := ℝ) B1))
                            (Activation.tanhSpec (α := ℝ) (castDimScalar (α := ℝ) hDim.symm v1.v))
                              := by
                        simpa [NN.MLTheory.CROWN.Runtime.Ops.IBP.tanh, Activation.tanhSpec,
                          Activation.Math.tanhSpec,
                          MathFunctions.tanh]
                          using map_minmax_sound_real (n := B1.dim) (f := Real.tanh) tanh_mono_real
                            (xB := ofFlatBox (α := ℝ) B1)
                            (x := castDimScalar (α := ℝ) hDim.symm v1.v)
                            hxContains
                      have hvCast :
                          castDimScalar (α := ℝ) hDim.symm (Activation.tanhSpec (α := ℝ) v1.v)
                            = Activation.tanhSpec (α := ℝ) (castDimScalar (α := ℝ) hDim.symm v1.v)
                              := by
                        simpa [Activation.tanhSpec] using
                          (castDimScalar_map_spec (h := hDim.symm)
                            (f := Activation.Math.tanhSpec (α := ℝ)) (t := v1.v))
                      refine ⟨hDim, ?_⟩
                      simpa [hvCast.symm] using
                        encloses_of_contains (n := B1.dim)
                          (B := NN.MLTheory.CROWN.Runtime.Ops.IBP.tanh (α := ℝ) (n := B1.dim)
                            (ofFlatBox (α := ℝ) B1))
                          (x := Activation.tanhSpec (α := ℝ) (castDimScalar (α := ℝ) hDim.symm
                            v1.v))
                          houtContains
        case sigmoid =>
          cases hparents : (g.nodes[k]!).parents with
          | nil =>
              simp [certStepNode?, hkKind, hparents] at hcertStep
          | cons p1 _ =>
              have hp1c : p1 < cert.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hcertSz] using this
              have hp1v : p1 < vals.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hvalsSz] using this
              have hcertStep' :
                  some B =
                    match getBox? cert p1 with
                    | some Xin =>
                        let yB := NN.MLTheory.CROWN.Runtime.Ops.IBP.sigmoid (α := ℝ) (n := Xin.dim)
                          (ofFlatBox (α := ℝ) Xin)
                        some (toFlatBox (α := ℝ) Xin.dim yB)
                    | none => none := by
                simpa [certStepNode?, hkKind, hparents] using (Eq.symm hcertStep)
              have hvalStep' :
                  some v =
                    match getVal? vals p1 with
                    | some x => some { n := x.n, v := Activation.sigmoidSpec (α := ℝ) x.v }
                    | none => none := by
                simpa [evalNode?, hkKind, hparents] using (Eq.symm hvalStep)
              cases hgb : getBox? cert p1 with
              | none =>
                  simp [hgb] at hcertStep'
              | some B1 =>
                  cases hgv : getVal? vals p1 with
                  | none =>
                      simp [hgv] at hvalStep'
                  | some v1 =>
                      have hB1 : cert[p1]! = some B1 := by
                        simpa [getBox?, hp1c] using hgb
                      have hv1 : vals[p1]! = some v1 := by
                        simpa [getVal?, hp1v] using hgv
                      have hpar' : EnclosesBox B1 v1 := by
                        have h := parentIH p1 (by simp [hparents])
                        simpa [hB1, hv1] using h
                      rcases hpar' with ⟨hDim, hxEnc⟩
                      have hcertStep'' := hcertStep'
                      have hvalStep'' := hvalStep'
                      simp [hgb] at hcertStep''
                      simp [hgv] at hvalStep''
                      cases hcertStep''
                      cases hvalStep''
                      have hxContains :
                          Box.contains (α := ℝ) (ofFlatBox (α := ℝ) B1)
                            (castDimScalar (α := ℝ) hDim.symm v1.v) :=
                        contains_of_encloses (B := B1)
                          (x := castDimScalar (α := ℝ) hDim.symm v1.v) (by simpa using hxEnc)
                      have houtContains :
                          Box.contains (α := ℝ)
                            (NN.MLTheory.CROWN.Runtime.Ops.IBP.sigmoid (α := ℝ) (n := B1.dim)
                              (ofFlatBox (α := ℝ) B1))
                            (Activation.sigmoidSpec (α := ℝ) (castDimScalar (α := ℝ) hDim.symm
                              v1.v)) := by
                        have hs : Monotone (Activation.Math.sigmoidSpec (α := ℝ)) :=
                          sigmoid_mono_real
                        simpa [NN.MLTheory.CROWN.Runtime.Ops.IBP.sigmoid, Activation.sigmoidSpec,
                          Tensor.mapSpec]
                          using map_minmax_sound_real (n := B1.dim) (f :=
                            Activation.Math.sigmoidSpec (α := ℝ)) hs
                            (xB := ofFlatBox (α := ℝ) B1)
                            (x := castDimScalar (α := ℝ) hDim.symm v1.v)
                            hxContains
                      have hvCast :
                          castDimScalar (α := ℝ) hDim.symm (Activation.sigmoidSpec (α := ℝ) v1.v)
                            = Activation.sigmoidSpec (α := ℝ) (castDimScalar (α := ℝ) hDim.symm
                              v1.v) := by
                        -- Commute cast with `map_spec` (sigmoid is elementwise).
                        simpa [Activation.sigmoidSpec] using
                          (castDimScalar_map_spec (h := hDim.symm)
                            (f := Activation.Math.sigmoidSpec (α := ℝ)) (t := v1.v))
                      refine ⟨hDim, ?_⟩
                      simpa [hvCast.symm] using
                        encloses_of_contains (n := B1.dim)
                          (B := NN.MLTheory.CROWN.Runtime.Ops.IBP.sigmoid (α := ℝ) (n := B1.dim)
                            (ofFlatBox (α := ℝ) B1))
                          (x := Activation.sigmoidSpec (α := ℝ) (castDimScalar (α := ℝ) hDim.symm
                            v1.v))
                          houtContains
        case sin =>
          cases hparents : (g.nodes[k]!).parents with
          | nil =>
              simp [certStepNode?, hkKind, hparents] at hcertStep
          | cons p1 _ =>
              have hp1c : p1 < cert.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hcertSz] using this
              have hp1v : p1 < vals.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hvalsSz] using this
              have hcertStep' :
                  some B =
                    match getBox? cert p1 with
                    | some Xin =>
                        let yB := NN.MLTheory.CROWN.Runtime.Ops.IBP.sin (α := ℝ) (n := Xin.dim)
                          (ofFlatBox (α := ℝ) Xin)
                        some (toFlatBox (α := ℝ) Xin.dim yB)
                    | none => none := by
                simpa [certStepNode?, hkKind, hparents] using (Eq.symm hcertStep)
              have hvalStep' :
                  some v =
                    match getVal? vals p1 with
                    | some x =>
                        some
                          { n := x.n
                            v :=
                              Tensor.mapSpec (α := ℝ) (s := .dim x.n .scalar) (fun z => Real.sin z)
                                x.v }
                    | none => none := by
                simpa [evalNode?, hkKind, hparents] using (Eq.symm hvalStep)
              cases hgb : getBox? cert p1 with
              | none =>
                  simp [hgb] at hcertStep'
              | some B1 =>
                  cases hgv : getVal? vals p1 with
                  | none =>
                      simp [hgv] at hvalStep'
                  | some v1 =>
                      have hB1 : cert[p1]! = some B1 := by
                        simpa [getBox?, hp1c] using hgb
                      have hv1 : vals[p1]! = some v1 := by
                        simpa [getVal?, hp1v] using hgv
                      have hpar' : EnclosesBox B1 v1 := by
                        have h := parentIH p1 (by simp [hparents])
                        simpa [hB1, hv1] using h
                      rcases hpar' with ⟨hDim, hxEnc⟩
                      have hcertStep'' := hcertStep'
                      have hvalStep'' := hvalStep'
                      simp [hgb] at hcertStep''
                      simp [hgv] at hvalStep''
                      cases hcertStep''
                      cases hvalStep''
                      have hxContains :
                          Box.contains (α := ℝ) (ofFlatBox (α := ℝ) B1)
                            (castDimScalar (α := ℝ) hDim.symm v1.v) :=
                        contains_of_encloses (B := B1)
                          (x := castDimScalar (α := ℝ) hDim.symm v1.v) (by simpa using hxEnc)
                      have houtContains :
                          Box.contains (α := ℝ)
                            (NN.MLTheory.CROWN.Runtime.Ops.IBP.sin (α := ℝ) (n := B1.dim) (ofFlatBox
                              (α := ℝ) B1))
                            (Tensor.mapSpec (α := ℝ) (s := .dim B1.dim .scalar) (fun z => Real.sin
                              z)
                              (castDimScalar (α := ℝ) hDim.symm v1.v)) := by
                        have h' :=
                          ibp_sin_sound_real (n := B1.dim)
                            (xB := ofFlatBox (α := ℝ) B1)
                            (x := castDimScalar (α := ℝ) hDim.symm v1.v)
                            hxContains
                        simpa using h'
                      have hvCast :
                          castDimScalar (α := ℝ) hDim.symm
                              (Tensor.mapSpec (α := ℝ) (s := .dim v1.n .scalar) (fun z => Real.sin
                                z) v1.v)
                            =
                            Tensor.mapSpec (α := ℝ) (s := .dim B1.dim .scalar) (fun z => Real.sin
                              z)
                              (castDimScalar (α := ℝ) hDim.symm v1.v) := by
                        -- Commute cast with `map_spec` (sin is elementwise).
                        simpa using
                          (castDimScalar_map_spec (h := hDim.symm)
                            (f := fun z : ℝ => Real.sin z) (t := v1.v))
                      refine ⟨hDim, ?_⟩
                      simpa [hvCast.symm] using
                        encloses_of_contains (n := B1.dim)
                          (B := NN.MLTheory.CROWN.Runtime.Ops.IBP.sin (α := ℝ) (n := B1.dim)
                            (ofFlatBox (α := ℝ) B1))
                          (x := Tensor.mapSpec (α := ℝ) (s := .dim B1.dim .scalar) Real.sin
                            (castDimScalar (α := ℝ) hDim.symm v1.v))
                          (by simpa using houtContains)
        case cos =>
          cases hparents : (g.nodes[k]!).parents with
          | nil =>
              simp [certStepNode?, hkKind, hparents] at hcertStep
          | cons p1 _ =>
              have hp1c : p1 < cert.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hcertSz] using this
              have hp1v : p1 < vals.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hvalsSz] using this
              have hcertStep' :
                  some B =
                    match getBox? cert p1 with
                    | some Xin =>
                        let yB := NN.MLTheory.CROWN.Runtime.Ops.IBP.cos (α := ℝ) (n := Xin.dim)
                          (ofFlatBox (α := ℝ) Xin)
                        some (toFlatBox (α := ℝ) Xin.dim yB)
                    | none => none := by
                simpa [certStepNode?, hkKind, hparents] using (Eq.symm hcertStep)
              have hvalStep' :
                  some v =
                    match getVal? vals p1 with
                    | some x =>
                        some
                          { n := x.n
                            v :=
                              Tensor.mapSpec (α := ℝ) (s := .dim x.n .scalar) (fun z => Real.cos z)
                                x.v }
                    | none => none := by
                simpa [evalNode?, hkKind, hparents] using (Eq.symm hvalStep)
              cases hgb : getBox? cert p1 with
              | none =>
                  simp [hgb] at hcertStep'
              | some B1 =>
                  cases hgv : getVal? vals p1 with
                  | none =>
                      simp [hgv] at hvalStep'
                  | some v1 =>
                      have hB1 : cert[p1]! = some B1 := by
                        simpa [getBox?, hp1c] using hgb
                      have hv1 : vals[p1]! = some v1 := by
                        simpa [getVal?, hp1v] using hgv
                      have hpar' : EnclosesBox B1 v1 := by
                        have h := parentIH p1 (by simp [hparents])
                        simpa [hB1, hv1] using h
                      rcases hpar' with ⟨hDim, hxEnc⟩
                      have hcertStep'' := hcertStep'
                      have hvalStep'' := hvalStep'
                      simp [hgb] at hcertStep''
                      simp [hgv] at hvalStep''
                      cases hcertStep''
                      cases hvalStep''
                      have hxContains :
                          Box.contains (α := ℝ) (ofFlatBox (α := ℝ) B1)
                            (castDimScalar (α := ℝ) hDim.symm v1.v) :=
                        contains_of_encloses (B := B1)
                          (x := castDimScalar (α := ℝ) hDim.symm v1.v) (by simpa using hxEnc)
                      have houtContains :
                          Box.contains (α := ℝ)
                            (NN.MLTheory.CROWN.Runtime.Ops.IBP.cos (α := ℝ) (n := B1.dim) (ofFlatBox
                              (α := ℝ) B1))
                            (Tensor.mapSpec (α := ℝ) (s := .dim B1.dim .scalar) (fun z => Real.cos
                              z)
                              (castDimScalar (α := ℝ) hDim.symm v1.v)) := by
                        have h' :=
                          ibp_cos_sound_real (n := B1.dim)
                            (xB := ofFlatBox (α := ℝ) B1)
                            (x := castDimScalar (α := ℝ) hDim.symm v1.v)
                            hxContains
                        simpa using h'
                      have hvCast :
                          castDimScalar (α := ℝ) hDim.symm
                              (Tensor.mapSpec (α := ℝ) (s := .dim v1.n .scalar) (fun z => Real.cos
                                z) v1.v)
                            =
                            Tensor.mapSpec (α := ℝ) (s := .dim B1.dim .scalar) (fun z => Real.cos
                              z)
                              (castDimScalar (α := ℝ) hDim.symm v1.v) := by
                        simpa using
                          (castDimScalar_map_spec (h := hDim.symm)
                            (f := fun z : ℝ => Real.cos z) (t := v1.v))
                      refine ⟨hDim, ?_⟩
                      simpa [hvCast.symm] using
                        encloses_of_contains (n := B1.dim)
                          (B := NN.MLTheory.CROWN.Runtime.Ops.IBP.cos (α := ℝ) (n := B1.dim)
                            (ofFlatBox (α := ℝ) B1))
                          (x := Tensor.mapSpec (α := ℝ) (s := .dim B1.dim .scalar) Real.cos
                            (castDimScalar (α := ℝ) hDim.symm v1.v))
                          (by simpa using houtContains)
        case linear =>
          cases hparents : (g.nodes[k]!).parents with
          | nil =>
              simp [certStepNode?, hkKind, hparents] at hcertStep
          | cons p1 _ =>
              have hp1c : p1 < cert.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hcertSz] using this
              have hp1v : p1 < vals.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hvalsSz] using this
              -- Certificate step delegates to `ibp_linear`.
              have hcertStep' :
                  some B =
                    match getBox? cert p1 with
                    | some Xin => ibp_linear (α := ℝ) k ps Xin
                    | none => none := by
                simpa [certStepNode?, hkKind, hparents] using (Eq.symm hcertStep)
              -- Semantics step uses the linear spec from ParamStore.
              have hvalStep' :
                  some v =
                    match getVal? vals p1, ps.linearWB[k]? with
                    | some x, some p =>
                        if h : x.n = p.n then
                          let xv : Tensor ℝ (.dim p.n .scalar) := by simpa [h] using x.v
                          let yv : Tensor ℝ (.dim p.m .scalar) :=
                            Spec.linearSpec (α := ℝ) { weights := p.w, bias := p.b } xv
                          some { n := p.m, v := yv }
                        else none
                    | _, _ => none := by
                simpa [evalNode?, hkKind, hparents] using (Eq.symm hvalStep)
              cases hgb : getBox? cert p1 with
              | none =>
                  simp [hgb] at hcertStep'
              | some B1 =>
                  cases hgv : getVal? vals p1 with
                  | none =>
                      simp [hgv] at hvalStep'
                  | some v1 =>
                      have hB1 : cert[p1]! = some B1 := by
                        simpa [getBox?, hp1c] using hgb
                      have hv1 : vals[p1]! = some v1 := by
                        simpa [getVal?, hp1v] using hgv
                      have hpar' : EnclosesBox B1 v1 := by
                        have h := parentIH p1 (by simp [hparents])
                        simpa [hB1, hv1] using h
                      rcases hpar' with ⟨hDim, hxEnc⟩
                      -- Unfold `ibp_linear` and use the already-proved `IBP.linear` soundness
                      -- theorem.
                      unfold ibp_linear at hcertStep'
                      cases hlin : ps.linearWB[k]? with
                      | none =>
                          have : (none : Option (FlatBox ℝ)) = some B := by
                            simpa [hlin, hgb] using (Eq.symm hcertStep')
                          cases this
                      | some p =>
                          simp [hlin, hgb] at hcertStep'
                          -- Substitute the concrete parent value into the semantic step.
                          simp [hlin, hgv] at hvalStep'
                          by_cases hXin : B1.dim = p.n
                          · simp [ibpLinearParams, hXin] at hcertStep'
                            by_cases hxDim' : v1.n = p.n
                            · simp [hxDim'] at hvalStep'
                              cases hcertStep'
                              cases hvalStep'
                              -- Convert parent enclosure to `Box.contains` for the casted input
                              -- box.
                              have hxContains0 :
                                  Box.contains (α := ℝ) (ofFlatBox (α := ℝ) B1)
                                    (castDimScalar (α := ℝ) hDim.symm v1.v) :=
                                contains_of_encloses (B := B1)
                                  (x := castDimScalar (α := ℝ) hDim.symm v1.v) (by simpa using
                                    hxEnc)
                              have hxContains1 :
                                  Box.contains (α := ℝ)
                                    (castBoxDim (α := ℝ) hXin (ofFlatBox (α := ℝ) B1))
                                    (castDimScalar (α := ℝ) hXin
                                      (castDimScalar (α := ℝ) hDim.symm v1.v)) := by
                                -- Transport `contains` across the dimension cast on the box.
                                exact (contains_castBoxDim_iff (h := hXin) (B := ofFlatBox (α := ℝ)
                                  B1)
                                  (x := castDimScalar (α := ℝ) hDim.symm v1.v)).2 hxContains0
                              have hProof : Eq.trans hDim.symm hXin = hxDim' := by
                                apply Subsingleton.elim
                              have hxCastEq :
                                  castDimScalar (α := ℝ) hxDim' v1.v
                                    = castDimScalar (α := ℝ) hXin
                                      (castDimScalar (α := ℝ) hDim.symm v1.v) := by
                                -- `hxDim'` and `Eq.trans hDim.symm hXin` are the same equality, up
                                -- to proof irrelevance.
                                simpa [hProof] using
                                  (castDimScalar_trans (h₁ := hDim.symm) (h₂ := hXin) (t := v1.v))
                              have hxContains :
                                  Box.contains (α := ℝ)
                                    (castBoxDim (α := ℝ) hXin (ofFlatBox (α := ℝ) B1))
                                    (castDimScalar (α := ℝ) hxDim' v1.v) := by
                                simpa [hxCastEq] using hxContains1
                              have hb : Box.contains (α := ℝ) (Box.point (α := ℝ) p.b) p.b := by
                                cases p.b with
                                | dim fb =>
                                  intro i
                                  cases fb i with
                                  | scalar b =>
                                    simp [Box.contains]
                              have houtContains :
                                  Box.contains (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.linear (α := ℝ) (m := p.m) (n := p.n) p.w
                                      (castBoxDim (α := ℝ) hXin (ofFlatBox (α := ℝ) B1))
                                      (Box.point (α := ℝ) p.b))
                                    (Spec.linearSpec (α := ℝ) { weights := p.w, bias := p.b }
                                      (castDimScalar (α := ℝ) hxDim' v1.v)) :=
                                NN.MLTheory.CROWN.Theorems.ibp_linear_sound_real p.w
                                  (castBoxDim (α := ℝ) hXin (ofFlatBox (α := ℝ) B1))
                                  (Box.point (α := ℝ) p.b)
                                  (castDimScalar (α := ℝ) hxDim' v1.v)
                                  p.b
                                  hxContains hb
                              refine ⟨rfl, ?_⟩
                              exact encloses_of_contains (n := p.m)
                                (B := NN.MLTheory.CROWN.IBP.linear (α := ℝ) (m := p.m) (n := p.n)
                                  p.w
                                  (castBoxDim (α := ℝ) hXin (ofFlatBox (α := ℝ) B1)) (Box.point (α
                                    := ℝ) p.b))
                                (x := Spec.linearSpec (α := ℝ) { weights := p.w, bias := p.b }
                                  (castDimScalar (α := ℝ) hxDim' v1.v))
                                houtContains
                            ·
                              simp [hxDim'] at hvalStep'
                          ·
                            simp [ibpLinearParams, hXin] at hcertStep'
        case matmul =>
          -- Same as `.linear`, but bias is zero and params come from `ParamStore.matmulW`.
          cases hparents : (g.nodes[k]!).parents with
          | nil =>
              simp [certStepNode?, hkKind, hparents] at hcertStep
          | cons p1 _ =>
              have hp1c : p1 < cert.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hcertSz] using this
              have hp1v : p1 < vals.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hvalsSz] using this
              have hcertStep' :
                  some B =
                    match getBox? cert p1 with
                    | some Xin => ibp_matmul (α := ℝ) k ps Xin
                    | none => none := by
                simpa [certStepNode?, hkKind, hparents] using (Eq.symm hcertStep)
              have hvalStep' :
                  some v =
                    match getVal? vals p1, ps.matmulW[k]? with
                    | some x, some p =>
                        if h : x.n = p.n then
                          let xv : Tensor ℝ (.dim p.n .scalar) := by simpa [h] using x.v
                          let z : Tensor ℝ (.dim p.m .scalar) := Spec.fill (α := ℝ) 0 (.dim p.m
                            .scalar)
                          let yv : Tensor ℝ (.dim p.m .scalar) :=
                            Spec.linearSpec (α := ℝ) { weights := p.w, bias := z } xv
                          some { n := p.m, v := yv }
                        else none
                    | _, _ => none := by
                simpa [evalNode?, hkKind, hparents] using (Eq.symm hvalStep)
              cases hgb : getBox? cert p1 with
              | none =>
                  simp [hgb] at hcertStep'
              | some B1 =>
                  cases hgv : getVal? vals p1 with
                  | none =>
                      simp [hgv] at hvalStep'
                  | some v1 =>
                      have hB1 : cert[p1]! = some B1 := by
                        simpa [getBox?, hp1c] using hgb
                      have hv1 : vals[p1]! = some v1 := by
                        simpa [getVal?, hp1v] using hgv
                      have hpar' : EnclosesBox B1 v1 := by
                        have h := parentIH p1 (by simp [hparents])
                        simpa [hB1, hv1] using h
                      rcases hpar' with ⟨hDim, hxEnc⟩
                      unfold ibp_matmul at hcertStep'
                      cases hmat : ps.matmulW[k]? with
                      | none =>
                          have : (none : Option (FlatBox ℝ)) = some B := by
                            simpa [hmat, hgb] using (Eq.symm hcertStep')
                          cases this
                      | some p =>
                          simp [hmat, hgb] at hcertStep'
                          simp [hmat, hgv] at hvalStep'
                          by_cases hXin : B1.dim = p.n
                          · simp [hXin] at hcertStep'
                            by_cases hxDim' : v1.n = p.n
                            · simp [hxDim'] at hvalStep'
                              cases hcertStep'
                              cases hvalStep'
                              have hxContains0 :
                                  Box.contains (α := ℝ) (ofFlatBox (α := ℝ) B1)
                                    (castDimScalar (α := ℝ) hDim.symm v1.v) :=
                                contains_of_encloses (B := B1)
                                  (x := castDimScalar (α := ℝ) hDim.symm v1.v) (by simpa using
                                    hxEnc)
                              have hxContains1 :
                                  Box.contains (α := ℝ)
                                    (castBoxDim (α := ℝ) hXin (ofFlatBox (α := ℝ) B1))
                                    (castDimScalar (α := ℝ) hXin
                                      (castDimScalar (α := ℝ) hDim.symm v1.v)) := by
                                exact (contains_castBoxDim_iff (h := hXin) (B := ofFlatBox (α := ℝ)
                                  B1)
                                  (x := castDimScalar (α := ℝ) hDim.symm v1.v)).2 hxContains0
                              have hProof : Eq.trans hDim.symm hXin = hxDim' := by
                                apply Subsingleton.elim
                              have hxCastEq :
                                  castDimScalar (α := ℝ) hxDim' v1.v
                                    = castDimScalar (α := ℝ) hXin
                                      (castDimScalar (α := ℝ) hDim.symm v1.v) := by
                                simpa [hProof] using
                                  (castDimScalar_trans (h₁ := hDim.symm) (h₂ := hXin) (t := v1.v))
                              have hxContains :
                                  Box.contains (α := ℝ)
                                    (castBoxDim (α := ℝ) hXin (ofFlatBox (α := ℝ) B1))
                                    (castDimScalar (α := ℝ) hxDim' v1.v) := by
                                simpa [hxCastEq] using hxContains1
                              let z : Tensor ℝ (.dim p.m .scalar) :=
                                Spec.fill (α := ℝ) 0 (.dim p.m .scalar)
                              have hz : Box.contains (α := ℝ) (Box.point (α := ℝ) z) z := by
                                cases z with
                                | dim fb =>
                                  intro i
                                  cases fb i with
                                  | scalar b =>
                                    simp [Box.contains]
                              have houtContains :
                                  Box.contains (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.linear (α := ℝ) (m := p.m) (n := p.n) p.w
                                      (castBoxDim (α := ℝ) hXin (ofFlatBox (α := ℝ) B1))
                                      (Box.point (α := ℝ) z))
                                    (Spec.linearSpec (α := ℝ) { weights := p.w, bias := z }
                                      (castDimScalar (α := ℝ) hxDim' v1.v)) :=
                                NN.MLTheory.CROWN.Theorems.ibp_linear_sound_real p.w
                                  (castBoxDim (α := ℝ) hXin (ofFlatBox (α := ℝ) B1))
                                  (Box.point (α := ℝ) z)
                                  (castDimScalar (α := ℝ) hxDim' v1.v)
                                  z
                                  hxContains hz
                              refine ⟨rfl, encloses_of_contains (n := p.m)
                                (B := NN.MLTheory.CROWN.IBP.linear (α := ℝ) (m := p.m) (n := p.n)
                                  p.w
                                  (castBoxDim (α := ℝ) hXin (ofFlatBox (α := ℝ) B1)) (Box.point (α
                                    := ℝ) z))
                                (x := Spec.linearSpec (α := ℝ) { weights := p.w, bias := z }
                                  (castDimScalar (α := ℝ) hxDim' v1.v))
                                houtContains⟩
                            ·
                              simp [hxDim'] at hvalStep'
                          ·
                            simp [hXin] at hcertStep'

end

end CertSoundness

end NN.MLTheory.CROWN.Graph

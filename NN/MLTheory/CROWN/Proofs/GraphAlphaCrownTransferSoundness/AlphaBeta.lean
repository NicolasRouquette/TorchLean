/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Alpha
public import NN.MLTheory.CROWN.Proofs.GraphCrownCertSoundness

/-!
# α/β-CROWN Graph Transfer Soundness

Pointwise soundness theorem for the β-extended graph transfer rule.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec
open _root_.Spec.Tensor
open scoped BigOperators
open Proofs.TensorAlgebra

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Cert

namespace AlphaCrownTransferSoundness

noncomputable section

open CrownCertSoundness
open CertSoundness

/--
Pointwise soundness of the graph-dialect α/β-CROWN transfer rule.

This is the β-extended analog of `alphaCrown_transfer_sound`.

Compared to plain α-CROWN, the step function additionally receives a `beta` array encoding
per-ReLU phase constraints (active/inactive/unstable). A phase consistent with the IBP
pre-activation interval gives an exact affine rule for that unit. An inconsistent phase rejects the
step; it does not silently fall back to another relaxation. Operators outside the affine-transfer
subset use an IBP-derived constant enclosure when that rule is available.

The theorem states that this concrete step function satisfies `CrownTransferSound`, and thus can
be used as the trusted “checker semantics” in `graph_crown_cert_soundness`.
-/
theorem alphaBetaCrown_transfer_sound
    (g : Graph) (ps : ParamStore ℝ)
    (ibp : Array (Option (FlatBox ℝ)))
    (alpha : Array (Option (FlatTensor ℝ)))
    (beta : Array (Option (Array Int)))
    (cert : Array (Option (FlatAffineBounds ℝ)))
    (inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val))
    (ctx : AffineCtx) (x : Tensor ℝ [ctx.inputDim])
    (htopo : TopoSorted g)
    (hsem : SemLocalOK (g := g) (ps := ps) (inputs := inputs) vals)
    (hinputs : InputsMatch (inputs := inputs) (ctx := ctx) x)
    (hibp : IBPEnclosesVals (ibp := ibp) (vals := vals))
    (halpha : AlphaOK (alpha := alpha)) :
    CrownTransferSound
      (g := g) (_ps := ps) (_inputs := inputs) (vals := vals)
      (ctx := ctx) (x := x)
      (step := stepAlphaBeta g ps ibp alpha beta ctx) (cert := cert) := by
  classical
  -- Reuse the α-CROWN transfer theorem for all nodes where `α/β` reduces to plain α.
  have hsoundAlpha :
      CrownTransferSound
        (g := g) (_ps := ps) (_inputs := inputs) (vals := vals)
        (ctx := ctx) (x := x)
        (step := stepAlpha g ps ibp alpha ctx) (cert := cert) :=
    alphaCrown_transfer_sound (g := g) (ps := ps) (ibp := ibp) (alpha := alpha) (cert := cert)
      (inputs := inputs) (vals := vals) (ctx := ctx) (x := x)
      (htopo := htopo) (hsem := hsem) (hinputs := hinputs) (hibp := hibp) (halpha := halpha)

  intro id hid hparents
  cases hs : stepAlphaBeta g ps ibp alpha beta ctx cert id <;> cases hv : vals[id]!
  all_goals simp
  case some.some b v =>
      -- Semantic evaluation at this node.
      have hEvalEq : vals[id]! = evalNode? g.nodes ps inputs vals id := hsem.2 id hid
      have hEvalSome : evalNode? g.nodes ps inputs vals id = some v := by
        have : some v = evalNode? g.nodes ps inputs vals id := by
          simpa [hv] using hEvalEq
        simpa using this.symm

      -- Split by node kind, mirroring `alphaBetaCrownStepNode?`.
      cases hk : (g.nodes[id]!).kind
      case relu =>
          -- If there is no β vector, α/β-CROWN is definitionally α-CROWN.
          cases hbeta : getBeta? (beta := beta) id with
          | none =>
              have hsAlpha : stepAlpha g ps ibp alpha ctx cert id = some b := by
                simpa [stepAlphaBeta, stepAlpha, alphaBetaCrownStepNode?, hk, hbeta] using hs
              have hA := hsoundAlpha id hid hparents
              simpa [hsAlpha, hv] using hA
          | some phases =>
              cases hps : NN.IR.unaryParent? (g.nodes[id]!).parents with
              | none =>
                  -- ReLU needs a parent; the step cannot succeed.
                  simp [stepAlphaBeta, alphaBetaCrownStepNode?, hk, hbeta, hps] at hs
              | some p1 =>
                have hpMem : p1 ∈ (g.nodes[id]!).parents :=
                  NN.IR.mem_of_unaryParent?_eq_some hps

                -- Step-side: extract parent affine bounds + IBP box.
                have hs' := hs
                simp [stepAlphaBeta, alphaBetaCrownStepNode?, hk, hps, hbeta] at hs'
                cases hxin : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 with
                | none =>
                    have : False := by
                      simp [hxin] at hs'
                    exact False.elim this
                | some xin =>
                    cases hpre : ibp[p1]! with
                    | none =>
                        have : False := by
                          simp [hxin, hpre] at hs'
                        exact False.elim this
                    | some preB =>
                        -- Semantic-side: extract the parent value `vp` and identify `v` as
                        -- `relu(vp)`.
                        have hEval' := hEvalSome
                        simp [CertSoundness.evalNode?, hk, hps] at hEval'
                        cases hgv : CertSoundness.getVal? vals p1 with
                        | none =>
                            have : False := by
                              simp [hgv] at hEval'
                            exact False.elim this
                        | some vp =>
                            have hvEq :
                                some { n := vp.n, v := Activation.reluSpec (α := ℝ) vp.v } = some v
                                  := by
                              simpa [hgv] using hEval'
                            cases hvEq

                            -- Connect `getAff?/getVal?` equalities to array lookups to use
                            -- `hparents`.
                            have hcertp : cert[p1]! = some xin := by
                              by_cases hltC : p1 < cert.size
                              · simpa [NN.MLTheory.CROWN.Cert.getAff?, hltC] using hxin
                              ·
                                have : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 = none := by
                                  simp [NN.MLTheory.CROWN.Cert.getAff?, hltC]
                                have : False := by
                                  simp [this] at hxin
                                exact False.elim this
                            have hvpp : vals[p1]! = some vp := by
                              by_cases hltV : p1 < vals.size
                              · simpa [CertSoundness.getVal?, hltV] using hgv
                              ·
                                have : CertSoundness.getVal? vals p1 = none := by
                                  simp [CertSoundness.getVal?, hltV]
                                have : False := by
                                  simp [this] at hgv
                                exact False.elim this

                            have parentEnc : CrownCertSoundness.EnclosesAtInput (α := ℝ) ctx x xin
                              vp := by
                              have := hparents p1 hpMem
                              simpa [hcertp, hvpp] using this
                            rcases parentEnc with ⟨hinDim, hvec⟩
                            dsimp at hvec
                            rcases hvec with ⟨hdimB, hencB⟩
                            have hdn : xin.outDim = vp.n := by
                              simpa [CrownCertSoundness.boundsEvalAt] using hdimB

                            -- IBP enclosure for the parent value.
                            have hencIbp : CertSoundness.EnclosesBox preB vp := by
                              have hltVals : p1 < vals.size := by
                                have : p1 < g.nodes.size := lt_trans (htopo id hid p1 hpMem) hid
                                simpa [hsem.1] using this
                              have hraw := hibp p1 hltVals
                              rw [hpre, hvpp] at hraw
                              exact hraw

                            rcases hencIbp with ⟨hdimIbp, hboxIbp⟩

                            -- `hout` is needed to align the parent affine out-dimension with the
                            -- IBP dimension.
                            by_cases hout : xin.outDim = preB.dim
                            ·
                              -- Common local proof once we have a concrete `αt` + phase
                              -- relaxations.
                              let x' : Tensor ℝ [xin.inDim] :=
                                castDimScalar (α := ℝ) (n := ctx.inputDim) (n' := xin.inDim)
                                  hinDim.symm x
                              let xLo : AffineVec ℝ xin.inDim preB.dim := by
                                simpa [hout] using xin.loAff
                              let xHi : AffineVec ℝ xin.inDim preB.dim := by
                                simpa [hout] using xin.hiAff

                              have relu_beta_common
                                  (αt : Tensor ℝ [preB.dim])
                                  (hαrange : ∀ i : Fin preB.dim,
                                    (0 : ℝ) ≤ getScalar αt i ∧ getScalar αt i ≤ (1 : ℝ))
                                  (relaxLo relaxHi :
                                    Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax ℝ) [preB.dim])
                                  (hrelax :
                                    phaseRelaxVec? (α := ℝ) (n := preB.dim) preB.lo preB.hi αt
                                      phases =
                                      some (relaxLo, relaxHi))
                                  (hbEq :
                                    some
                                        { inDim := xin.inDim
                                          outDim := preB.dim
                                          loAff :=
                                            NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α :=
                                              ℝ)
                                              (inDim := xin.inDim) (hidDim := preB.dim) relaxLo xLo
                                          hiAff :=
                                            NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α :=
                                              ℝ)
                                              (inDim := xin.inDim) (hidDim := preB.dim) relaxHi xHi
                                                } =
                                      some b) :
                                  CrownCertSoundness.EnclosesAtInput (α := ℝ) ctx x b
                                    { n := vp.n, v := Activation.reluSpec (α := ℝ) vp.v } := by
                                have hb :
                                    b =
                                      { inDim := xin.inDim
                                        outDim := preB.dim
                                        loAff :=
                                          NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α :=
                                            ℝ)
                                            (inDim := xin.inDim) (hidDim := preB.dim) relaxLo xLo
                                        hiAff :=
                                          NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α :=
                                            ℝ)
                                            (inDim := xin.inDim) (hidDim := preB.dim) relaxHi xHi }
                                              := by
                                  exact (Option.some.inj hbEq).symm
                                -- Cast the semantic parent value into `preB.dim` so the ReLU is
                                -- well-typed.
                                let z : Tensor ℝ [preB.dim] :=
                                  castDimScalar (α := ℝ) (n := vp.n) (n' := preB.dim) hdimIbp.symm
                                    vp.v
                                have hreluCast :
                                    castDimScalar (α := ℝ) (n := vp.n) (n' := preB.dim) hdimIbp.symm
                                        (Activation.reluSpec (α := ℝ) vp.v)
                                      =
                                      Activation.reluSpec (α := ℝ) z := by
                                  simpa [z] using
                                    (relu_spec_castDimScalar (h := hdimIbp.symm) (t := vp.v))

                                -- Derive the affine enclosure `lAff ≤ z ≤ uAff` from the parent's
                                -- enclosure.
                                let zXin : Tensor ℝ [xin.outDim] :=
                                  castDimScalar (α := ℝ) (n := vp.n) (n' := xin.outDim)
                                    hdn.symm vp.v
                                have hzXin :
                                    Theorems.Semantics.encloses (α := ℝ)
                                      (boundsEvalAt (α := ℝ) xin x') zXin := by
                                  have : castDimScalar (α := ℝ) hdimB.symm vp.v = zXin := by
                                    exact castDimScalar_proof_irrel (h₁ := hdimB.symm) (h₂ :=
                                      hdn.symm) (t := vp.v)
                                  simpa [this, zXin] using hencB

                                have hzCast0 :=
                                  sem_encloses_castDim (B := boundsEvalAt (α := ℝ) xin x') (h :=
                                    hout) (x := zXin) hzXin
                                have hzCastZ :
                                    castDimScalar (α := ℝ) hout zXin = z := by
                                  have htrans : Eq.trans hdn.symm hout = hdimIbp.symm := by
                                    exact Subsingleton.elim _ _
                                  have := (castDimScalar_trans (h₁ := hdn.symm) (h₂ := hout) (t :=
                                    vp.v)).symm
                                  simp [zXin, z] at this ⊢
                                let lAff : Tensor ℝ [preB.dim] :=
                                  affineEvalAt (α := ℝ) (inDim := xin.inDim) (outDim := preB.dim)
                                    xLo x'
                                let uAff : Tensor ℝ [preB.dim] :=
                                  affineEvalAt (α := ℝ) (inDim := xin.inDim) (outDim := preB.dim)
                                    xHi x'
                                have hl :
                                    castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin x').lo =
                                      lAff := by
                                  simpa [CrownCertSoundness.boundsEvalAt,
                                    CrownCertSoundness.affineEvalAt, lAff, x', xLo,
                                    Cert.castAffineOut] using
                                    (affineEvalAt_castAffineOut (h := hout) (aff := xin.loAff) (x :=
                                      x')).symm
                                have hu :
                                    castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin x').hi =
                                      uAff := by
                                  simpa [CrownCertSoundness.boundsEvalAt,
                                    CrownCertSoundness.affineEvalAt, uAff, x', xHi,
                                    Cert.castAffineOut] using
                                    (affineEvalAt_castAffineOut (h := hout) (aff := xin.hiAff) (x :=
                                      x')).symm
                                have hzAff :
                                    Theorems.Semantics.encloses (α := ℝ)
                                      { dim := preB.dim
                                        lo := lAff
                                        hi := uAff } z := by
                                  -- Convert `hzCast0` into the `lAff/uAff` box via `hl/hu`.
                                  have hzCast1 :
                                      Theorems.Semantics.encloses (α := ℝ)
                                        { dim := preB.dim
                                          lo := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ)
                                            xin x').lo
                                          hi := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ)
                                            xin x').hi }
                                        z := by
                                    exact sem_encloses_value_eq
                                      (B := { dim := preB.dim
                                              lo := castDimScalar (α := ℝ) hout (boundsEvalAt (α :=
                                                ℝ) xin x').lo
                                              hi := castDimScalar (α := ℝ) hout (boundsEvalAt (α :=
                                                ℝ) xin x').hi })
                                      (hxy := hzCastZ) hzCast0
                                  have hBoxEq :
                                      ({ dim := preB.dim
                                         lo := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ)
                                           xin x').lo
                                         hi := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ)
                                           xin x').hi } : FlatBox ℝ)
                                        =
                                      ({ dim := preB.dim, lo := lAff, hi := uAff } : FlatBox ℝ) :=
                                        by
                                    refine FlatBox.ext' (hDim := rfl) (hLo := ?_) (hHi := ?_)
                                    · exact heq_of_eq hl
                                    · exact heq_of_eq hu
                                  exact sem_encloses_of_eq (h := hBoxEq) (x := z) hzCast1

                                have hzIbp : Theorems.Semantics.encloses (α := ℝ) preB z := by
                                  simpa [CertSoundness.encloses, z] using hboxIbp

                                let bout : FlatAffineBounds ℝ :=
                                  { inDim := xin.inDim
                                    outDim := preB.dim
                                    loAff :=
                                      NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := ℝ)
                                        (inDim := xin.inDim) (hidDim := preB.dim) relaxLo xLo
                                    hiAff :=
                                      NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := ℝ)
                                        (inDim := xin.inDim) (hidDim := preB.dim) relaxHi xHi }
                                have hzAffI := (encloses_iff_getScalar (n := preB.dim) (lo := lAff) (hi
                                  := uAff) (x := z)).1 hzAff
                                have hzIbpI := (encloses_iff_getScalar (n := preB.dim) (lo := preB.lo)
                                  (hi := preB.hi) (x := z)).1 hzIbp
                                have hphase := phaseRelaxVec?_some_getScalar (n := preB.dim)
                                  (lo := preB.lo) (hi := preB.hi) (αv := αt) (phases := phases)
                                  (relaxLo := relaxLo) (relaxHi := relaxHi) hrelax

                                have hConcrete :
                                    Theorems.Semantics.encloses (α := ℝ)
                                      (boundsEvalAt (α := ℝ) bout x')
                                      (Activation.reluSpec (α := ℝ) z) := by
                                  -- Now it suffices to show enclosure against the concrete lo/hi
                                  -- tensors.
                                  refine (encloses_iff_getScalar (n := preB.dim)
                                    (lo := (boundsEvalAt (α := ℝ) bout x').lo)
                                    (hi := (boundsEvalAt (α := ℝ) bout x').hi)
                                    (x := Activation.reluSpec (α := ℝ) z)).2 ?_
                                  intro i
                                  have hzLo := (hzAffI i).1
                                  have hzHi := (hzAffI i).2
                                  have hzIlo := (hzIbpI i).1
                                  have hzIhi := (hzIbpI i).2
                                  rcases hphase.2 i with ⟨ph, hcons, hrHi, hrLo⟩

                                  let li := getScalar preB.lo i
                                  let ui := getScalar preB.hi i
                                  let zi := getScalar z i
                                  let ai := getScalar αt i
                                  have hai0 : (0 : ℝ) ≤ ai := (hαrange i).1
                                  have hai1 : ai ≤ (1 : ℝ) := (hαrange i).2

                                  let rpLo := getScalar relaxLo i
                                  let rpHi := getScalar relaxHi i
                                  have hsLo : 0 ≤ rpLo.slope := by
                                    have hs :
                                        0 ≤ (phaseRelaxLowerScalar (α := ℝ) li ui ai ph).slope :=
                                      phaseRelaxLowerScalar_slope_nonneg (l := li) (u := ui) (a :=
                                        ai) (ph := ph) hai0
                                    simpa [rpLo, li, ui, ai, hrLo] using hs
                                  have hsHi : 0 ≤ rpHi.slope := by
                                    have hs :
                                        0 ≤ (phaseRelaxUpperScalar (α := ℝ) li ui ph).slope :=
                                      phaseRelaxUpperScalar_slope_nonneg (l := li) (u := ui) (ph :=
                                        ph)
                                    simpa [rpHi, li, ui, hrHi] using hs

                                  have hlo_def :
                                      getScalar (boundsEvalAt (α := ℝ) bout x').lo i
                                        =
                                        let rp := getScalar relaxLo i
                                        rp.slope * getScalar lAff i + rp.bias := by
                                    simpa [CrownCertSoundness.boundsEvalAt,
                                      CrownCertSoundness.affineEvalAt, bout, lAff, x', xLo] using
                                      (getScalar_affineEvalAt_relu_propagate_affine
                                        (relax := relaxLo) (aff := xLo) (x := x') (i := i))
                                  have hhi_def :
                                      getScalar (boundsEvalAt (α := ℝ) bout x').hi i
                                        =
                                        let rp := getScalar relaxHi i
                                        rp.slope * getScalar uAff i + rp.bias := by
                                    simpa [CrownCertSoundness.boundsEvalAt,
                                      CrownCertSoundness.affineEvalAt, bout, uAff, x', xHi] using
                                      (getScalar_affineEvalAt_relu_propagate_affine
                                        (relax := relaxHi) (aff := xHi) (x := x') (i := i))

                                  have hlo1 :
                                      rpLo.slope * getScalar lAff i + rpLo.bias
                                        ≤
                                      rpLo.slope * zi + rpLo.bias := by
                                    have hm : rpLo.slope * getScalar lAff i ≤ rpLo.slope * zi := by
                                      exact mul_le_mul_of_nonneg_left hzLo hsLo
                                    have h' := add_le_add_right hm rpLo.bias
                                    simpa [add_comm, add_left_comm, add_assoc] using h'
                                  have hlo2 :
                                      rpLo.slope * zi + rpLo.bias
                                        ≤
                                      Activation.Math.reluSpec (α := ℝ) zi := by
                                    have :=
                                      NN.MLTheory.CROWN.Proofs.phaseRelaxLowerScalar_sound
                                        (l := li) (u := ui) (a := ai) (x := zi)
                                        (hlx := hzIlo) (hxu := hzIhi) (ha0 := hai0) (ha1 := hai1)
                                        (ph := ph) (hcons := hcons)
                                    simpa [rpLo, li, ui, ai, zi, hrLo] using this
                                  have hlo :
                                      getScalar (boundsEvalAt (α := ℝ) bout x').lo i ≤
                                        Activation.Math.reluSpec (α := ℝ) zi := by
                                    simp [hlo_def]
                                    exact le_trans hlo1 hlo2

                                  have hhi1 :
                                      Activation.Math.reluSpec (α := ℝ) zi ≤
                                        rpHi.slope * zi + rpHi.bias := by
                                    have :=
                                      NN.MLTheory.CROWN.Proofs.phaseRelaxUpperScalar_sound
                                        (l := li) (u := ui) (x := zi)
                                        (hlx := hzIlo) (hxu := hzIhi) (ph := ph) (hcons := hcons)
                                    simpa [rpHi, li, ui, zi, hrHi] using this
                                  have hhi2 :
                                      rpHi.slope * zi + rpHi.bias
                                        ≤
                                      rpHi.slope * getScalar uAff i + rpHi.bias := by
                                    have hm : rpHi.slope * zi ≤ rpHi.slope * getScalar uAff i := by
                                      exact mul_le_mul_of_nonneg_left hzHi hsHi
                                    have h' := add_le_add_right hm rpHi.bias
                                    simpa [add_comm, add_left_comm, add_assoc] using h'
                                  have hhi :
                                      Activation.Math.reluSpec (α := ℝ) zi ≤
                                        getScalar (boundsEvalAt (α := ℝ) bout x').hi i := by
                                    simp [hhi_def]
                                    exact le_trans hhi1 hhi2

                                  have hrelu : getScalar (Activation.reluSpec (α := ℝ) z) i =
                                      Activation.Math.reluSpec (α := ℝ) zi := by
                                    simpa [zi] using (getScalar_relu_spec (t := z) (i := i))
                                  constructor
                                  · rw [hrelu]
                                    exact hlo
                                  · rw [hrelu]
                                    exact hhi
                                rw [hb]
                                refine ⟨hinDim, ?_⟩
                                dsimp [CrownCertSoundness.EnclosesVec]
                                refine ⟨hdimIbp, ?_⟩
                                have hConcrete' :
                                    Theorems.Semantics.encloses (α := ℝ)
                                      (boundsEvalAt (α := ℝ) bout x')
                                      (Activation.reluSpec (α := ℝ) z) := by
                                  simpa [x'] using hConcrete
                                exact sem_encloses_value_eq
                                  (B := boundsEvalAt (α := ℝ) bout x')
                                  (hxy := hreluCast.symm) hConcrete'

                              -- Now instantiate `relu_beta_common` according to whether α is
                              -- present.
                              cases hαopt : NN.MLTheory.CROWN.Cert.getAlpha? (α := ℝ) alpha id with
                              | some αv =>
                                  have hs'' := hs'
                                  simp [hxin, hpre, hαopt, hout] at hs''
                                  by_cases hα : αv.n = preB.dim
                                  ·
                                    let αt : Tensor ℝ [preB.dim] :=
                                      castDimScalar (α := ℝ) (n := αv.n) (n' := preB.dim) hα αv.v
                                    have hαrange : ∀ i : Fin preB.dim,
                                        (0 : ℝ) ≤ getScalar αt i ∧ getScalar αt i ≤ (1 : ℝ) := by
                                      have hidA : id < alpha.size := by
                                        by_cases hltA : id < alpha.size
                                        · exact hltA
                                        ·
                                          have : NN.MLTheory.CROWN.Cert.getAlpha? (α := ℝ) alpha id
                                            = none := by
                                            simp [NN.MLTheory.CROWN.Cert.getAlpha?, hltA]
                                          have : False := by
                                            simp [this] at hαopt
                                          exact False.elim this
                                      have hentry : alpha[id]! = some αv := by
                                        simpa [NN.MLTheory.CROWN.Cert.getAlpha?, hidA] using hαopt
                                      have hrange0 : ∀ i : Fin αv.n, (0 : ℝ) ≤ getScalar αv.v i ∧ getScalar
                                        αv.v i ≤ (1 : ℝ) := by
                                        simpa [hentry] using halpha id hidA
                                      intro i
                                      have hri := hrange0 (Fin.cast hα.symm i)
                                      simpa [αt, hα, getScalar_castDimScalar] using hri
                                    simp [hα] at hs''
                                    cases hrelax : phaseRelaxVec? (α := ℝ) (n := preB.dim) preB.lo
                                      preB.hi αt phases with
                                    | none =>
                                        have : False := by
                                          simp [αt, hrelax] at hs''
                                        exact False.elim this
                                    | some rpair =>
                                        cases rpair with
                                        | mk relaxLo relaxHi =>
                                            have hbEq :
                                                some
                                                    ({ inDim := xin.inDim
                                                       outDim := preB.dim
                                                       loAff :=
                                                         Runtime.Ops.ReLU.propagateAffine
                                                           (α := ℝ)
                                                           (inDim := xin.inDim) (hidDim := preB.dim)
                                                           relaxLo xLo
                                                       hiAff :=
                                                         Runtime.Ops.ReLU.propagateAffine
                                                           (α := ℝ)
                                                           (inDim := xin.inDim) (hidDim := preB.dim)
                                                           relaxHi xHi } : FlatAffineBounds ℝ) =
                                                  some b := by
                                              simpa [αt, hrelax, xLo, xHi, Cert.castAffineOut] using hs''
                                            exact relu_beta_common αt hαrange relaxLo relaxHi hrelax
                                              hbEq
                                  ·
                                    -- Dimension mismatch: step cannot succeed.
                                    simp [hα] at hs''
                              | none =>
                                  have hs'' := hs'
                                  simp [hxin, hpre, hαopt, hout] at hs''
                                  let αt : Tensor ℝ [preB.dim] :=
                                    defaultAlphaVec (α := ℝ) (n := preB.dim) preB.lo preB.hi
                                  have hαrange : ∀ i : Fin preB.dim,
                                      (0 : ℝ) ≤ getScalar αt i ∧ getScalar αt i ≤ (1 : ℝ) := by
                                    simpa [αt] using defaultAlphaVec_range (lo := preB.lo) (hi :=
                                      preB.hi)
                                  cases hrelax : phaseRelaxVec? (α := ℝ) (n := preB.dim) preB.lo
                                    preB.hi αt phases with
                                  | none =>
                                      have : False := by
                                        simp [αt, hrelax] at hs''
                                      exact False.elim this
                                  | some rpair =>
                                      cases rpair with
                                      | mk relaxLo relaxHi =>
                                          have hbEq :
                                              some
                                                  ({ inDim := xin.inDim
                                                     outDim := preB.dim
                                                     loAff :=
                                                       Runtime.Ops.ReLU.propagateAffine
                                                         (α := ℝ)
                                                         (inDim := xin.inDim) (hidDim := preB.dim)
                                                         relaxLo xLo
                                                     hiAff :=
                                                       Runtime.Ops.ReLU.propagateAffine
                                                         (α := ℝ)
                                                         (inDim := xin.inDim) (hidDim := preB.dim)
                                                         relaxHi xHi } : FlatAffineBounds ℝ) =
                                                some b := by
                                            simpa [αt, hrelax, xLo, xHi, Cert.castAffineOut] using hs''
                                          exact relu_beta_common αt hαrange relaxLo relaxHi hrelax
                                            hbEq
                            ·
                              -- If `xin.outDim ≠ preB.dim` then the step returns `none`.
                              have hs'' := hs'
                              -- First reduce the `match` on the known `some xin` / `some preB`.
                              simp [hxin, hpre] at hs''
                              -- Now the outer `if hout : xin.outDim = preB.dim` is forced to take
                              -- the `else` branch.
                              cases hαopt : NN.MLTheory.CROWN.Cert.getAlpha? (α := ℝ) alpha id <;> (
                                have : False := by
                                  simp [hαopt, hout] at hs''
                                exact False.elim this
                              )

      all_goals
        -- All other node kinds delegate to α-CROWN.
        have hsAlpha : stepAlpha g ps ibp alpha ctx cert id = some b := by
          simpa [stepAlphaBeta, stepAlpha, alphaBetaCrownStepNode?, hk] using hs
        have hA := hsoundAlpha id hid hparents
        simpa [hsAlpha, hv] using hA

  end

open CrownCertSoundness
open CertSoundness

/--
A locally replayed α/β-CROWN certificate encloses every corresponding graph value.

This is the user-facing composition of `alphaBetaCrown_transfer_sound` with the generic graph
certificate checker. The certificate producer remains untrusted: `hcert` requires its entries to
agree node-by-node with TorchLean's α/β transfer function.
-/
theorem alphaBetaCrown_cert_encloses_semantics
    (g : Graph) (ps : ParamStore ℝ)
    (ibp : Array (Option (FlatBox ℝ)))
    (alpha : Array (Option (FlatTensor ℝ)))
    (beta : Array (Option (Array Int)))
    (cert : Array (Option (FlatAffineBounds ℝ)))
    (inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val))
    (ctx : AffineCtx) (x : Tensor ℝ [ctx.inputDim])
    (htopo : TopoSorted g)
    (hsem : SemLocalOK (g := g) (ps := ps) (inputs := inputs) vals)
    (hinputs : InputsMatch (inputs := inputs) (ctx := ctx) x)
    (hibp : IBPEnclosesVals (ibp := ibp) (vals := vals))
    (halpha : AlphaOK (alpha := alpha))
    (hcert : CrownCertLocalOK (g := g) (step := stepAlphaBeta g ps ibp alpha beta ctx) cert) :
    ∀ id : Nat, id < g.nodes.size →
      ∀ (b : FlatAffineBounds ℝ) (v : Val),
        cert[id]! = some b →
        vals[id]! = some v →
        EnclosesAtInput (α := ℝ) ctx x b v := by
  apply crown_checker_encloses_semantics
    (g := g) (ps := ps) (step := stepAlphaBeta g ps ibp alpha beta ctx)
    (cert := cert) (inputs := inputs) (vals := vals) (ctx := ctx) (x := x)
    htopo hsem hcert
  exact alphaBetaCrown_transfer_sound
    (g := g) (ps := ps) (ibp := ibp) (alpha := alpha) (beta := beta) (cert := cert)
    (inputs := inputs) (vals := vals) (ctx := ctx) (x := x)
    htopo hsem hinputs hibp halpha

end AlphaCrownTransferSoundness

end NN.MLTheory.CROWN.Graph

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Real.Basic
public import NN.MLTheory.CROWN.Graph
public import NN.MLTheory.CROWN.Models.Mlp

/-!
# GraphIBPBasicTheorems

Basic theorems about the graph-level IBP engine:

* A `Valid` predicate for interval boxes (`lo ≤ hi` componentwise).
* Preservation of validity for core interval ops (add/sub/relu) and runtime monotone-activation IBP.
* Validity of the linear/matmul IBP rules (derived from the existing `ibp_linear_sound_real` proof).
-/

@[expose] public section


namespace NN.MLTheory.CROWN

open _root_.Spec
open _root_.Spec.Tensor

namespace Box

/-- Extract the underlying scalar value at index `i` from a vector tensor. -/
def getScalar {n : Nat} (t : Tensor ℝ (.dim n .scalar)) (i : Fin n) : ℝ :=
  match t with
  | .dim f =>
    match f i with
    | .scalar v => v

/-- Componentwise validity of a 1D interval box: `lo ≤ hi` for every coordinate. -/
def Valid {n : Nat} (B : Box ℝ (.dim n .scalar)) : Prop :=
  ∀ i : Fin n, getScalar B.lo i ≤ getScalar B.hi i

/-- If a box contains any point, then it is componentwise valid (`lo ≤ hi`). -/
theorem valid_of_contains {n : Nat} (B : Box ℝ (.dim n .scalar)) (x : Tensor ℝ (.dim n .scalar))
  (hx : Box.contains (α := ℝ) B x) : Valid B := by
  intro i
  cases B with
  | mk lo hi =>
    cases lo with
    | dim flo =>
      cases hi with
      | dim fhi =>
        cases x with
        | dim fx =>
          have hx_i := hx i
          cases hL : flo i with
          | scalar l =>
            cases hU : fhi i with
            | scalar u =>
              cases hX : fx i with
              | scalar v =>
                have hv : l ≤ v ∧ v ≤ u := by
                  simpa [Box.contains, hL, hU, hX] using hx_i
                simpa [Valid, getScalar, hL, hU] using (le_trans hv.1 hv.2)

/-- A valid box contains its lower endpoint `lo`. -/
theorem contains_lo_of_valid {n : Nat} (B : Box ℝ (.dim n .scalar)) (hB : Valid B) :
    Box.contains (α := ℝ) B B.lo := by
  cases B with
  | mk lo hi =>
    cases lo with
    | dim flo =>
      cases hi with
      | dim fhi =>
        intro i
        cases hL : flo i with
        | scalar l =>
          cases hU : fhi i with
          | scalar u =>
            have hlu : l ≤ u := by
              simpa [Valid, getScalar, hL, hU] using (hB i)
            -- scalar containment: l ≤ l and l ≤ u
            simp [Box.contains, hlu]

/-- Validity is preserved by (definitional) casts of the vector dimension. -/
theorem valid_castBoxDim {n n' : Nat} (h : n = n')
    (B : Box ℝ (.dim n .scalar)) (hB : Valid B) :
    Valid (NN.MLTheory.CROWN.Graph.castBoxDim (α := ℝ) h B) := by
  cases h
  simpa [NN.MLTheory.CROWN.Graph.castBoxDim] using hB

end Box

end NN.MLTheory.CROWN

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN

namespace FlatBoxTheorems

variable {α : Type} [Context α]

/-- Converting a valid `Box` into a `FlatBox` preserves validity. -/
theorem valid_toFlatBox_real {n : Nat} (B : Box ℝ (.dim n .scalar)) (hB :
  NN.MLTheory.CROWN.Box.Valid B) :
    (toFlatBox (α := ℝ) n B).Valid := by
  intro i
  cases B with
  | mk lo hi =>
    cases lo with
    | dim flo =>
      cases hi with
      | dim fhi =>
        cases hL : flo i with
        | scalar l =>
          cases hU : fhi i with
          | scalar u =>
            have hlu : l ≤ u := by
              simpa [NN.MLTheory.CROWN.Box.Valid, NN.MLTheory.CROWN.Box.getScalar, hL, hU] using (hB
                i)
            simpa [NN.MLTheory.CROWN.FlatBox.Valid, NN.MLTheory.CROWN.FlatBox.getScalar, toFlatBox,
              hL, hU] using hlu

/-- Converting a valid `FlatBox` into a `Box` preserves validity. -/
theorem valid_ofFlatBox_real (B : FlatBox ℝ) (hB : B.Valid) :
    NN.MLTheory.CROWN.Box.Valid (ofFlatBox (α := ℝ) B) := by
  intro i
  cases B with
  | mk n lo hi =>
    cases lo with
    | dim flo =>
      cases hi with
      | dim fhi =>
        cases hL : flo i with
        | scalar l =>
          cases hU : fhi i with
          | scalar u =>
            have hlu : l ≤ u := by
              simpa [NN.MLTheory.CROWN.FlatBox.Valid, NN.MLTheory.CROWN.FlatBox.getScalar, hL, hU]
                using (hB i)
            simpa [NN.MLTheory.CROWN.Box.Valid, NN.MLTheory.CROWN.Box.getScalar, ofFlatBox, hL, hU]
              using hlu

/-- Validity is preserved by interval addition on `FlatBox` (over `ℝ`). -/
theorem valid_box_add_real (B1 B2 : FlatBox ℝ) (h1 : B1.Valid) (h2 : B2.Valid) :
    (boxAdd (α := ℝ) B1 B2).Valid := by
  let : BoundOps ℝ := instBoundOpsReal
  cases B1 with
  | mk n1 lo1 hi1 =>
    cases B2 with
    | mk n2 lo2 hi2 =>
      by_cases h : n1 = n2
      · subst h
        -- use the canonical form lemma for the equal-dimension case
        have hEq :
            boxAdd (α := ℝ)
                { dim := n1, lo := lo1, hi := hi1 }
                { dim := n1, lo := lo2, hi := hi2 } =
              { dim := n1
                lo := Tensor.map2Spec BoundOps.addDown lo1 lo2
                hi := Tensor.map2Spec BoundOps.addUp hi1 hi2 } := by
          simpa using (NN.MLTheory.CROWN.Graph.Theorems.box_add_on_eq (α := ℝ) n1 lo1 hi1 lo2 hi2)
        -- reduce the goal to scalar arithmetic
        rw [hEq]
        intro i
        cases lo1 with
        | dim flo1 =>
          cases hi1 with
          | dim fhi1 =>
            cases lo2 with
            | dim flo2 =>
              cases hi2 with
              | dim fhi2 =>
                cases hL1 : flo1 i with
                | scalar l1 =>
                  cases hU1 : fhi1 i with
                  | scalar u1 =>
                    cases hL2 : flo2 i with
                    | scalar l2 =>
                      cases hU2 : fhi2 i with
                      | scalar u2 =>
                        have h1i : l1 ≤ u1 := by
                          simpa [NN.MLTheory.CROWN.FlatBox.Valid,
                            NN.MLTheory.CROWN.FlatBox.getScalar, hL1, hU1]
                            using (h1 i)
                        have h2i : l2 ≤ u2 := by
                          simpa [NN.MLTheory.CROWN.FlatBox.Valid,
                            NN.MLTheory.CROWN.FlatBox.getScalar, hL2, hU2]
                            using (h2 i)
                        have hdown : BoundOps.addDown l1 l2 = l1 + l2 := rfl
                        have hup : BoundOps.addUp u1 u2 = u1 + u2 := rfl
                        -- unfold the scalar projections through `Tensor.add_spec`
                        simpa [NN.MLTheory.CROWN.FlatBox.Valid, NN.MLTheory.CROWN.FlatBox.getScalar,
                          Tensor.map2Spec, hL1, hU1, hL2, hU2, hdown, hup]
                          using add_le_add h1i h2i
      · -- mismatch branch: returns B1 unchanged
        have hEq :
            boxAdd (α := ℝ)
                { dim := n1, lo := lo1, hi := hi1 }
                { dim := n2, lo := lo2, hi := hi2 } =
              { dim := n1, lo := lo1, hi := hi1 } := by
          simp [boxAdd, h]
        simpa [hEq] using h1

/-- Validity is preserved by interval subtraction on `FlatBox` (over `ℝ`). -/
theorem valid_box_sub_real (B1 B2 : FlatBox ℝ) (h1 : B1.Valid) (h2 : B2.Valid) :
    (boxSub (α := ℝ) B1 B2).Valid := by
  let : BoundOps ℝ := instBoundOpsReal
  cases B1 with
  | mk n1 lo1 hi1 =>
    cases B2 with
    | mk n2 lo2 hi2 =>
      by_cases h : n1 = n2
      · subst h
        have hEq :
            boxSub (α := ℝ)
                { dim := n1, lo := lo1, hi := hi1 }
                { dim := n1, lo := lo2, hi := hi2 } =
              { dim := n1
                lo := Tensor.map2Spec BoundOps.subDown lo1 hi2
                hi := Tensor.map2Spec BoundOps.subUp hi1 lo2 } := by
          simpa using (NN.MLTheory.CROWN.Graph.Theorems.box_sub_on_eq (α := ℝ) n1 lo1 hi1 lo2 hi2)
        rw [hEq]
        intro i
        cases lo1 with
        | dim flo1 =>
          cases hi1 with
          | dim fhi1 =>
            cases lo2 with
            | dim flo2 =>
              cases hi2 with
              | dim fhi2 =>
                cases hL1 : flo1 i with
                | scalar l1 =>
                  cases hU1 : fhi1 i with
                  | scalar u1 =>
                    cases hL2 : flo2 i with
                    | scalar l2 =>
                      cases hU2 : fhi2 i with
                      | scalar u2 =>
                        have h1i : l1 ≤ u1 := by
                          simpa [NN.MLTheory.CROWN.FlatBox.Valid,
                            NN.MLTheory.CROWN.FlatBox.getScalar, hL1, hU1]
                            using (h1 i)
                        have h2i : l2 ≤ u2 := by
                          simpa [NN.MLTheory.CROWN.FlatBox.Valid,
                            NN.MLTheory.CROWN.FlatBox.getScalar, hL2, hU2]
                            using (h2 i)
                        have hneg : -u2 ≤ -l2 := neg_le_neg h2i
                        have hadd : l1 + (-u2) ≤ u1 + (-l2) := add_le_add h1i hneg
                        have hdown : BoundOps.subDown l1 u2 = l1 - u2 := rfl
                        have hup : BoundOps.subUp u1 l2 = u1 - l2 := rfl
                        simpa [NN.MLTheory.CROWN.FlatBox.Valid, NN.MLTheory.CROWN.FlatBox.getScalar,
                          Tensor.map2Spec, hL1, hU1, hL2, hU2, hdown, hup,
                          sub_eq_add_neg, add_assoc, add_left_comm, add_comm]
                          using hadd
      ·
        have hEq :
            boxSub (α := ℝ)
                { dim := n1, lo := lo1, hi := hi1 }
                { dim := n2, lo := lo2, hi := hi2 } =
              { dim := n1, lo := lo1, hi := hi1 } := by
          simp [boxSub, h]
        simpa [hEq] using h1

/-- Validity is preserved by ReLU-IBP on `FlatBox` (over `ℝ`). -/
theorem valid_box_relu_real (B : FlatBox ℝ) (hB : B.Valid) :
    (boxRelu (α := ℝ) B).Valid := by
  intro i
  cases B with
  | mk n lo hi =>
    cases lo with
    | dim flo =>
      cases hi with
      | dim fhi =>
        cases hL : flo i with
        | scalar l =>
          cases hU : fhi i with
          | scalar u =>
            have hlu : l ≤ u := by
              have hi := hB i
              simpa [NN.MLTheory.CROWN.FlatBox.Valid, NN.MLTheory.CROWN.FlatBox.getScalar, hL, hU]
                using hi
            have hrelu : max l 0 ≤ max u 0 := max_le_max hlu (le_rfl)
            -- unfold the output scalars after `Tensor.map_spec`
            simpa [boxRelu, NN.MLTheory.CROWN.FlatBox.Valid, NN.MLTheory.CROWN.FlatBox.getScalar,
              Tensor.mapSpec, Activation.Math.reluSpec, hL, hU]
              using hrelu

end FlatBoxTheorems

namespace Theorems

open NN.MLTheory.CROWN.Box

/-- Validity of the `IBP.linear` transfer rule (over `ℝ`). -/
theorem ibp_linear_valid_real {m n : Nat}
  (W : Tensor ℝ (.dim m (.dim n .scalar)))
  (xB : Box ℝ (.dim n .scalar))
  (bB : Box ℝ (.dim m .scalar))
  (hxB : Valid xB) (hbB : Valid bB) :
  Valid (IBP.linear (α := ℝ) W xB bB) := by
  -- pick witnesses `x = xB.lo` and `b = bB.lo`
  have hx : Box.contains (α := ℝ) xB xB.lo := Box.contains_lo_of_valid (B := xB) hxB
  have hb : Box.contains (α := ℝ) bB bB.lo := Box.contains_lo_of_valid (B := bB) hbB
  have hout :
      Box.contains (α := ℝ) (IBP.linear (α := ℝ) W xB bB)
        (Spec.linearSpec (α := ℝ) { weights := W, bias := bB.lo } xB.lo) :=
    NN.MLTheory.CROWN.Theorems.ibp_linear_sound_real W xB bB xB.lo bB.lo hx hb
  exact Box.valid_of_contains (B := IBP.linear (α := ℝ) W xB bB) _ hout

/-- Validity of the graph-level `linear` IBP rule (over `ℝ`), if the parameters are present. -/
theorem graph_ibp_linear_valid_real (id : Nat) (ps : ParamStore ℝ) (Xin : FlatBox ℝ)
  (hXin : FlatBox.Valid (α := ℝ) Xin) :
  match ibpLinear (α := ℝ) id ps Xin with
  | none => True
  | some Bout => FlatBox.Valid (α := ℝ) Bout := by
  classical
  unfold ibpLinear
  cases hlin : ps.linearWB[id]? with
  | none => simp
  | some p =>
      by_cases hdim : Xin.dim = p.n
      · -- dimension match: reduce to `IBP.linear_valid_real`
        simp [ibpLinearParams, hdim]
        -- show the produced box is valid
        -- Convert input to a dimension-safe `Box` and prove it is valid.
        have hxBoxValid : Valid (castBoxDim (α := ℝ) hdim (ofFlatBox (α := ℝ) Xin)) := by
          -- validity is preserved by definitional cast
          have : Valid (ofFlatBox (α := ℝ) Xin) := FlatBoxTheorems.valid_ofFlatBox_real (B := Xin)
            hXin
          exact NN.MLTheory.CROWN.Box.valid_castBoxDim (h := hdim) (B := ofFlatBox (α := ℝ) Xin)
            this
        have hbBoxValid : Valid (Box.point (α := ℝ) p.b) := by
          cases p.b with
          | dim _ =>
            simp [Valid, Box.point]
        have hyValid : Valid (IBP.linear (α := ℝ) (m := p.m) (n := p.n) p.w
            (castBoxDim (α := ℝ) hdim (ofFlatBox (α := ℝ) Xin))
            (Box.point (α := ℝ) p.b)) :=
          ibp_linear_valid_real (W := p.w) (xB := castBoxDim (α := ℝ) hdim (ofFlatBox (α := ℝ) Xin))
            (bB := Box.point (α := ℝ) p.b) hxBoxValid hbBoxValid
        exact FlatBoxTheorems.valid_toFlatBox_real (B := IBP.linear (α := ℝ) (m := p.m) (n := p.n)
          p.w
          (castBoxDim (α := ℝ) hdim (ofFlatBox (α := ℝ) Xin)) (Box.point (α := ℝ) p.b)) hyValid
      · simp [ibpLinearParams, hdim]

/-- Validity of the graph-level `matmul` IBP rule (over `ℝ`), if the parameters are present. -/
theorem graph_ibp_matmul_valid_real (id : Nat) (ps : ParamStore ℝ) (Xin : FlatBox ℝ)
  (hXin : FlatBox.Valid (α := ℝ) Xin) :
  match ibpMatmul (α := ℝ) id ps Xin with
  | none => True
  | some Bout => FlatBox.Valid (α := ℝ) Bout := by
  classical
  unfold ibpMatmul
  cases hmat : ps.matmulW[id]? with
  | none => simp
  | some p =>
      by_cases hdim : Xin.dim = p.n
      · simp [hdim]
        have hxBoxValid : Valid (castBoxDim (α := ℝ) hdim (ofFlatBox (α := ℝ) Xin)) := by
          have : Valid (ofFlatBox (α := ℝ) Xin) := FlatBoxTheorems.valid_ofFlatBox_real (B := Xin)
            hXin
          exact NN.MLTheory.CROWN.Box.valid_castBoxDim (h := hdim) (B := ofFlatBox (α := ℝ) Xin)
            this
        -- zero bias point box is valid
        let z : Tensor ℝ (.dim p.m .scalar) := Spec.fill (α := ℝ) 0 (.dim p.m .scalar)
        have hbBoxValid : Valid (Box.point (α := ℝ) z) := by
          cases z with
          | dim _ =>
            simp [Valid, Box.point]
        have hyValid : Valid (IBP.linear (α := ℝ) (m := p.m) (n := p.n) p.w
            (castBoxDim (α := ℝ) hdim (ofFlatBox (α := ℝ) Xin))
            (Box.point (α := ℝ) z)) :=
          ibp_linear_valid_real (W := p.w) (xB := castBoxDim (α := ℝ) hdim (ofFlatBox (α := ℝ) Xin))
            (bB := Box.point (α := ℝ) z) hxBoxValid hbBoxValid
        exact FlatBoxTheorems.valid_toFlatBox_real (B := IBP.linear (α := ℝ) (m := p.m) (n := p.n)
          p.w
          (castBoxDim (α := ℝ) hdim (ofFlatBox (α := ℝ) Xin)) (Box.point (α := ℝ) z)) hyValid
      · simp [hdim]

end Theorems

end NN.MLTheory.CROWN.Graph

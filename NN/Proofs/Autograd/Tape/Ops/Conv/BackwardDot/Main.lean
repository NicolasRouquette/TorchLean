/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Ops.Conv.BackwardDot.Input

/-!
Main convolution backward-dot theorem layer.

This file combines the input and weight-side algebra needed to prove the tape-level dot-product
form of convolution backpropagation.
-/

@[expose] public section

namespace Proofs
namespace Autograd
namespace Conv2d

open Spec
open Tensor

open scoped BigOperators

noncomputable section


/--
Main dot-level bridge theorem for Conv2d.

It states that the triple returned by `Spec.conv2d_backward_spec` is the adjoint (w.r.t. `Spec.dot`)
of the corresponding forward-mode directional derivatives with respect to `(kernel, bias, input)`.

This is the key lemma connecting the handwritten runtime backward to the analytic “VJP is adjoint
of `fderiv`” theorem in `NN.Proofs.Autograd.Tape.Ops.Conv.FDeriv`.
-/
theorem conv2d_backward_spec_dot
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (layer : Spec.Conv2dSpec inC outC kH kW stride padding ℝ h1 h2 h3)
    (input : Spec.Tensor ℝ (.dim inC (.dim inH (.dim inW .scalar))))
    (δ : Spec.Tensor ℝ (.dim outC (.dim (outH inH kH stride padding) (.dim (outW inW kW stride padding) .scalar))))
    (dKernel : Tensor ℝ (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
    (dBias : Tensor ℝ (.dim outC .scalar))
    (dInput : Spec.Tensor ℝ (.dim inC (.dim inH (.dim inW .scalar)))) :
    let layerK : Spec.Conv2dSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
      { kernel := dKernel, bias := fill (0 : ℝ) (.dim outC .scalar) }
    let layer0 : Spec.Conv2dSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
      { kernel := layer.kernel, bias := fill (0 : ℝ) (.dim outC .scalar) }
    let jvp : Spec.Tensor ℝ (.dim outC (.dim (outH inH kH stride padding) (.dim (outW inW kW stride padding) .scalar))) :=
      addSpec
        (Spec.conv2dSpec (α := ℝ) (layer := layerK) input)
        (addSpec
          (biasBroadcast (outC := outC) (outH := outH inH kH stride padding) (outW := outW inW kW
            stride padding) dBias)
          (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput))
    let grads := Spec.conv2dBackwardSpec (α := ℝ) (layer := layer) (input := input) (grad_output
      := δ)
    dot jvp δ =
      dot dKernel grads.1 + dot dBias grads.2.1 + dot dInput grads.2.2 := by
  intro layerK layer0 jvp grads
  -- Expand the dot across the JVP sum and apply the three component dot-bridge lemmas.
  have hK : dot (Spec.conv2dSpec (α := ℝ) (layer := layerK) input) δ =
      dot dKernel (Spec.conv2dKernelDerivSpec (α := ℝ) (layer := layer) (input := input)
        (grad_output := δ)) := by
    simpa using (dot_conv2d_kernel (layer := layer) (input := input) (dKernel := dKernel) (δ := δ))
  have hB :
      dot (biasBroadcast (outC := outC) (outH := outH inH kH stride padding) (outW := outW inW kW
        stride padding) dBias) δ
        =
      dot dBias (Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layer) (input := input)
        (grad_output := δ)) := by
    exact dot_biasBroadcast_eq_dot_bias_deriv (layer := layer) (input := input) (db := dBias)
      (δ := δ)
  have hX :
      dot (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput) δ
        =
      dot dInput (Spec.conv2dInputDerivSpec (α := ℝ) (layer := layer) (input := input)
        (grad_output := δ)) := by
    simpa using (dot_conv2d_input (layer := layer) (input := input) (dInput := dInput) (δ := δ))
  -- Now finish by splitting the dot on `jvp` and rewriting `grads`.
  subst jvp
  -- Split dot over the nested additions, then rewrite using `hK`/`hB`/`hX` and unfold `grads`.
  have hsplit :
      dot
          (addSpec
            (Spec.conv2dSpec (α := ℝ) (layer := layerK) input)
            ((biasBroadcast (outC := outC) (outH := outH inH kH stride padding) (outW := outW inW kW
              stride padding) dBias).addSpec
              (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput)))
          δ
        =
      dot (Spec.conv2dSpec (α := ℝ) (layer := layerK) input) δ +
        dot (biasBroadcast (outC := outC) (outH := outH inH kH stride padding) (outW := outW inW kW
          stride padding) dBias) δ +
        dot (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput) δ := by
    -- Expand dot across the outer `+`, then across the inner `+`.
    calc
      dot
          (addSpec
            (Spec.conv2dSpec (α := ℝ) (layer := layerK) input)
            ((biasBroadcast (outC := outC) (outH := outH inH kH stride padding) (outW := outW inW kW
              stride padding) dBias).addSpec
              (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput)))
          δ
          =
      dot (Spec.conv2dSpec (α := ℝ) (layer := layerK) input) δ +
          dot
              ((biasBroadcast (outC := outC) (outH := outH inH kH stride padding) (outW := outW inW
                kW stride padding) dBias).addSpec
                (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput))
              δ := by
                -- Apply the dot-add lemma once (don’t rewrite the RHS further).
                exact
                  dot_add_left
                    (a := Spec.conv2dSpec (α := ℝ) (layer := layerK) input)
                    (b :=
                      (biasBroadcast (outC := outC) (outH := outH inH kH stride padding)
                        (outW := outW inW kW stride padding) dBias).addSpec
                        (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput))
                    (c := δ)
      _ =
        dot (Spec.conv2dSpec (α := ℝ) (layer := layerK) input) δ +
          (dot (biasBroadcast (outC := outC) (outH := outH inH kH stride padding) (outW := outW inW
            kW stride padding) dBias) δ +
            dot (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput) δ) := by
              -- Use the dot-add lemma directly for the inner sum.
              rw [dot_add_left
                  (a := biasBroadcast (outC := outC) (outH := outH inH kH stride padding) (outW :=
                    outW inW kW stride padding) dBias)
                  (b := Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput)
                  (c := δ)]
              rfl
      _ =
        dot (Spec.conv2dSpec (α := ℝ) (layer := layerK) input) δ +
          dot (biasBroadcast (outC := outC) (outH := outH inH kH stride padding) (outW := outW inW
            kW stride padding) dBias) δ +
          dot (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput) δ := by
            ring
  -- Rewrite the dot terms using the component bridges, then unfold `grads` and reassociate.
  calc
    dot
        (addSpec
          (Spec.conv2dSpec (α := ℝ) (layer := layerK) input)
          ((biasBroadcast (outC := outC) (outH := outH inH kH stride padding) (outW := outW inW kW
            stride padding) dBias).addSpec
            (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput)))
        δ
        =
      dot (Spec.conv2dSpec (α := ℝ) (layer := layerK) input) δ +
        dot (biasBroadcast (outC := outC) (outH := outH inH kH stride padding) (outW := outW inW kW
          stride padding) dBias) δ +
        dot (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput) δ := by
          simpa using hsplit
    _ =
      dot dKernel (Spec.conv2dKernelDerivSpec (α := ℝ) (layer := layer) (input := input)
        (grad_output := δ)) +
        dot dBias (Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layer) (input := input)
          (grad_output := δ)) +
        dot dInput (Spec.conv2dInputDerivSpec (α := ℝ) (layer := layer) (input := input)
          (grad_output := δ)) := by
          -- rewrite each term using the corresponding dot bridge lemma
          simp [hK, hB, hX, add_assoc, add_comm]
      _ = dot dKernel grads.1 + dot dBias grads.2.1 + dot dInput grads.2.2 := by
            simp [grads, Spec.conv2dBackwardSpec]
end

end Conv2d
end Autograd
end Proofs

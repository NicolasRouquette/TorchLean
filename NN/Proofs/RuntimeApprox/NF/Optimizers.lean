/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.Ops.Elementwise.Unary
public import NN.Proofs.RuntimeApprox.NF.ShapeOps
public import NN.Proofs.RuntimeApprox.Optimizer
public import NN.Runtime.Optim.Optimizers

/-!
# Rounded Optimizer Steps for `NF`

Concrete instances of `RuntimeApprox.Optimizer.NumericalStepContract` for TorchLean's rounded
`NF` runtime. These proofs use the same tensor equations as the public optimizers and the shared
elementwise error transformers; there is no second optimizer implementation in the proof layer.

The first contracts cover SGD and momentum SGD. They already compose with the generic
`NumericalStepContract.run_approx` theorem over arbitrary finite gradient streams and arbitrary
tensor ranks. Adaptive optimizers build on the positive-division and square-root rules and are kept
in this module so all optimizer numerical contracts share one public home.

The Adam recurrence follows Kingma and Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015
(https://arxiv.org/abs/1412.6980). The decoupled decay term follows Loshchilov and Hutter,
*Decoupled Weight Decay Regularization*, ICLR 2019 (https://arxiv.org/abs/1711.05101).
-/

@[expose] public section

namespace Proofs.RuntimeApprox.NFBackend.Optimizer

open Spec
open Tensor
open TorchLean.Floats
open Proofs.RuntimeApprox.Optimizer

noncomputable section

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]
variable {rnd : ℝ → ℤ} [NeuralValidRndToNearest rnd]

local notation "R" => TorchLean.Floats.NF β fexp rnd

/-! ## SGD -/

/-- Error in the runtime learning-rate scalar stored by SGD. -/
abbrev SGDStateBound := ℝ

/-- Exact/runtime relation for SGD state. -/
def sgdStateApprox {s : Shape} (stateS : _root_.Optim.SGD.State ℝ s)
    (stateR : _root_.Optim.SGD.State R s) (error : SGDStateBound) : Prop :=
  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) stateR.lr - stateS.lr) ≤ error

/-- Parameter error after one SGD update, computed from the actual runtime tensors. -/
def sgdStepBound {s : Shape} (lrError paramsError gradsError : ℝ)
    (stateR : _root_.Optim.SGD.State R s) (paramsR gradsR : Tensor R s) :
    StepBound (fun _ => SGDStateBound) s :=
  let scaledGradError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      gradsError lrError stateR.lr gradsR)
  let scaledGradR := scaleSpec gradsR stateR.lr
  { state := lrError
    params := linfNorm
      (subBoundTensor (β := β) (fexp := fexp) paramsError scaledGradError paramsR scaledGradR) }

/-- Numerical refinement contract for TorchLean's plain SGD update `p <- p - lr * g`. -/
def sgdContract : NumericalStepContract R (toSpec (β := β) (fexp := fexp) (rnd := rnd)) where
  name := "SGD"
  StateSpec := _root_.Optim.SGD.State ℝ
  StateRuntime := _root_.Optim.SGD.State R
  StateBound := fun _ => SGDStateBound
  StepData := fun _ => Unit
  stateApprox := sgdStateApprox (β := β) (fexp := fexp) (rnd := rnd)
  stepDataValid := fun _ _ _ _ _ _ _ _ _ _ => True
  updateSpec := fun state params grads =>
    (state, _root_.Optim.SGD.update state params grads)
  updateRuntime := fun state params grads =>
    (state, _root_.Optim.SGD.update state params grads)
  updateBound := fun lrError paramsError gradsError state params grads _ =>
    sgdStepBound (β := β) (fexp := fexp) (rnd := rnd)
      lrError paramsError gradsError state params grads
  stateBoundReport := fun lrError => [("learning rate", lrError)]
  stepDataReport := fun _ => []
  updateSound := by
    intro s stateS stateR lrError paramsS paramsR paramsError gradsS gradsR gradsError
      _stepData hlr hparams hgrads _hvalid
    have hscaled := approxTensor_scale_spec_of_approx
      (β := β) (fexp := fexp) (rnd := rnd) stateS.lr stateR.lr hgrads hlr
    have hnext := approxTensor_sub_spec
      (β := β) (fexp := fexp) (rnd := rnd) hparams hscaled
    constructor
    · exact hlr
    · simpa [sgdStepBound, _root_.Optim.SGD.update] using hnext

/-- One actual TorchLean SGD parameter update refines its exact-real counterpart. -/
theorem approxTensor_sgd_update {s : Shape}
    {stateS : _root_.Optim.SGD.State ℝ s} {stateR : _root_.Optim.SGD.State R s}
    {lrError : ℝ} {paramsS : Tensor ℝ s} {paramsR : Tensor R s} {paramsError : ℝ}
    {gradsS : Tensor ℝ s} {gradsR : Tensor R s} {gradsError : ℝ}
    (hstate : sgdStateApprox (β := β) (fexp := fexp) (rnd := rnd)
      stateS stateR lrError)
    (hparams : approxTensor (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      paramsS paramsR paramsError)
    (hgrads : approxTensor (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      gradsS gradsR gradsError) :
    approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (_root_.Optim.SGD.update stateS paramsS gradsS)
      (_root_.Optim.SGD.update stateR paramsR gradsR)
      (sgdStepBound (β := β) (fexp := fexp) (rnd := rnd)
        lrError paramsError gradsError stateR paramsR gradsR).params := by
  have hscaled := approxTensor_scale_spec_of_approx
    (β := β) (fexp := fexp) (rnd := rnd) stateS.lr stateR.lr hgrads hstate
  have hnext := approxTensor_sub_spec
    (β := β) (fexp := fexp) (rnd := rnd) hparams hscaled
  simpa [sgdStepBound, _root_.Optim.SGD.update] using hnext

/-! ## Momentum SGD -/

/-- Error budgets for momentum SGD's scalar hyperparameters and momentum buffer. -/
structure MomentumSGDStateBound (s : Shape) where
  /-- Learning-rate error. -/
  lr : ℝ
  /-- Momentum-coefficient error. -/
  momentum : ℝ
  /-- Infinity-norm error in the stored momentum buffer. -/
  buf : ℝ

/-- Exact/runtime relation for momentum SGD state. -/
def momentumSGDStateApprox {s : Shape} (stateS : _root_.Optim.MomentumSGD.State ℝ s)
    (stateR : _root_.Optim.MomentumSGD.State R s) (error : MomentumSGDStateBound s) : Prop :=
  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) stateR.lr - stateS.lr) ≤ error.lr ∧
  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) stateR.momentum - stateS.momentum) ≤
    error.momentum ∧
  approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
    stateS.buf stateR.buf error.buf

/-- State and parameter bounds for one momentum-SGD update. -/
def momentumSGDStepBound {s : Shape} (stateError : MomentumSGDStateBound s)
    (paramsError gradsError : ℝ) (stateR : _root_.Optim.MomentumSGD.State R s)
    (paramsR gradsR : Tensor R s) : StepBound MomentumSGDStateBound s :=
  let scaledBufR := scaleSpec stateR.buf stateR.momentum
  let scaledBufError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      stateError.buf stateError.momentum stateR.momentum stateR.buf)
  let newBufR := _root_.Optim.OptimizerUtils.updateMomentumBuf
    stateR.buf stateR.momentum gradsR
  let newBufError := linfNorm
    (addBoundTensor (β := β) (fexp := fexp)
      scaledBufError gradsError scaledBufR gradsR)
  let scaledUpdateError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      newBufError stateError.lr stateR.lr newBufR)
  { state := { stateError with buf := newBufError }
    params := linfNorm
      (subBoundTensor (β := β) (fexp := fexp)
        paramsError scaledUpdateError paramsR (scaleSpec newBufR stateR.lr)) }

/-- Numerical refinement contract for momentum SGD with arbitrary-rank parameter tensors. -/
def momentumSGDContract :
    NumericalStepContract R (toSpec (β := β) (fexp := fexp) (rnd := rnd)) where
  name := "Momentum SGD"
  StateSpec := _root_.Optim.MomentumSGD.State ℝ
  StateRuntime := _root_.Optim.MomentumSGD.State R
  StateBound := MomentumSGDStateBound
  StepData := fun _ => Unit
  stateApprox := momentumSGDStateApprox (β := β) (fexp := fexp) (rnd := rnd)
  stepDataValid := fun _ _ _ _ _ _ _ _ _ _ => True
  updateSpec := _root_.Optim.MomentumSGD.update
  updateRuntime := _root_.Optim.MomentumSGD.update
  updateBound := fun stateError paramsError gradsError state params grads _ =>
    momentumSGDStepBound (β := β) (fexp := fexp) (rnd := rnd)
      stateError paramsError gradsError state params grads
  stateBoundReport := fun error =>
    [("learning rate", error.lr), ("momentum", error.momentum), ("momentum buffer", error.buf)]
  stepDataReport := fun _ => []
  updateSound := by
    intro s stateS stateR stateError paramsS paramsR paramsError gradsS gradsR gradsError
      _stepData hstate hparams hgrads _hvalid
    rcases hstate with ⟨hlr, hmomentum, hbuf⟩
    have hscaledBuf := approxTensor_scale_spec_of_approx
      (β := β) (fexp := fexp) (rnd := rnd)
      stateS.momentum stateR.momentum hbuf hmomentum
    have hnewBuf := approxTensor_add_spec
      (β := β) (fexp := fexp) (rnd := rnd) hscaledBuf hgrads
    have hscaledUpdate := approxTensor_scale_spec_of_approx
      (β := β) (fexp := fexp) (rnd := rnd)
      stateS.lr stateR.lr hnewBuf hlr
    have hnextParams := approxTensor_sub_spec
      (β := β) (fexp := fexp) (rnd := rnd) hparams hscaledUpdate
    constructor
    · exact ⟨hlr, hmomentum, by
        simpa [momentumSGDStepBound, _root_.Optim.MomentumSGD.update,
          _root_.Optim.OptimizerUtils.updateMomentumBuf] using hnewBuf⟩
    · simpa [momentumSGDStepBound, _root_.Optim.MomentumSGD.update,
        _root_.Optim.OptimizerUtils.updateMomentumBuf] using hnextParams

/-- One public momentum-SGD update refines its exact-real counterpart.

This named corollary exposes the useful one-step statement without duplicating its proof; the
generic `momentumSGDContract.updateSound` field remains the source used for finite runs and
graph-level composition. -/
theorem approxTensor_momentumSGD_update {s : Shape}
    {stateS : _root_.Optim.MomentumSGD.State ℝ s}
    {stateR : _root_.Optim.MomentumSGD.State R s}
    {stateError : MomentumSGDStateBound s}
    {paramsS : Tensor ℝ s} {paramsR : Tensor R s} {paramsError : ℝ}
    {gradsS : Tensor ℝ s} {gradsR : Tensor R s} {gradsError : ℝ}
    (hstate : momentumSGDStateApprox (β := β) (fexp := fexp) (rnd := rnd)
      stateS stateR stateError)
    (hparams : approxTensor (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      paramsS paramsR paramsError)
    (hgrads : approxTensor (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      gradsS gradsR gradsError) :
    let nextBound := momentumSGDStepBound (β := β) (fexp := fexp) (rnd := rnd)
      stateError paramsError gradsError stateR paramsR gradsR
    momentumSGDStateApprox (β := β) (fexp := fexp) (rnd := rnd)
        (_root_.Optim.MomentumSGD.update stateS paramsS gradsS).1
        (_root_.Optim.MomentumSGD.update stateR paramsR gradsR).1 nextBound.state ∧
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (_root_.Optim.MomentumSGD.update stateS paramsS gradsS).2
        (_root_.Optim.MomentumSGD.update stateR paramsR gradsR).2 nextBound.params := by
  exact momentumSGDContract.updateSound
    stateS stateR stateError paramsS paramsR paramsError gradsS gradsR gradsError ()
    hstate hparams hgrads trivial

/-! ## AdamW -/

/-- Error budgets relating exact and rounded AdamW state. -/
structure AdamWStateBound (s : Shape) where
  /-- Error in the stored learning rate. -/
  lr : ℝ
  /-- Error in the first-moment decay coefficient. -/
  beta1 : ℝ
  /-- Error in the second-moment decay coefficient. -/
  beta2 : ℝ
  /-- Error in the denominator stabilizer. -/
  epsilon : ℝ
  /-- Error in the decoupled weight-decay coefficient. -/
  weightDecay : ℝ
  /-- Infinity-norm error in the first-moment tensor. -/
  moment1 : ℝ
  /-- Infinity-norm error in the second-moment tensor. -/
  moment2 : ℝ

/-- Exact/runtime relation for the persistent AdamW state. -/
def adamWStateApprox {s : Shape} (stateS : _root_.Optim.AdamW.State ℝ s)
    (stateR : _root_.Optim.AdamW.State R s) (error : AdamWStateBound s) : Prop :=
  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) stateR.lr - stateS.lr) ≤ error.lr ∧
  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) stateR.beta1 - stateS.beta1) ≤ error.beta1 ∧
  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) stateR.beta2 - stateS.beta2) ≤ error.beta2 ∧
  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) stateR.epsilon - stateS.epsilon) ≤
    error.epsilon ∧
  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) stateR.weightDecay -
    stateS.weightDecay) ≤ error.weightDecay ∧
  approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
    stateS.m stateR.m error.moment1 ∧
  approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
    stateS.v stateR.v error.moment2 ∧
  stateS.t = stateR.t

/-- Errors for scalar expressions derived inside one AdamW step.

They are kept separate from persistent state errors because subtraction, powers, reciprocal, and
the product `lr * weightDecay` each round in the runtime scalar model. -/
structure AdamWDerivedErrors where
  /-- Error in the rounded scalar expression `1 - beta1`. -/
  oneMinusBeta1 : ℝ
  /-- Error in the rounded scalar expression `1 - beta2`. -/
  oneMinusBeta2 : ℝ
  /-- Error in the reciprocal first-moment bias correction. -/
  biasInv1 : ℝ
  /-- Error in the reciprocal second-moment bias correction. -/
  biasInv2 : ℝ
  /-- Error in the rounded product `lr * weightDecay`. -/
  decayScale : ℝ

/-- Composed errors for the intermediate tensors in one AdamW step. -/
structure AdamWStepErrorTrace where
  /-- Error after squaring the gradient. -/
  squaredGrad : ℝ
  /-- Error after updating the first moment. -/
  moment1 : ℝ
  /-- Error after updating the second moment. -/
  moment2 : ℝ
  /-- Error after first-moment bias correction. -/
  moment1Hat : ℝ
  /-- Error after second-moment bias correction. -/
  moment2Hat : ℝ
  /-- Error after square root of the corrected second moment. -/
  std : ℝ
  /-- Error after adding epsilon to the square-root denominator. -/
  denominator : ℝ
  /-- Error in the elementwise adaptive learning rate. -/
  adaptiveLR : ℝ
  /-- Error in the Adam update before subtraction from parameters. -/
  adamUpdate : ℝ
  /-- Error in the decoupled weight-decay update. -/
  decayUpdate : ℝ
  /-- Error after applying decoupled weight decay. -/
  decayedParams : ℝ
  /-- Final parameter error after the full AdamW step. -/
  params : ℝ

/-- Compute AdamW's complete one-step error trace from runtime values and scalar subexpression
budgets. The reduction to one infinity-norm number per tensor keeps the trace independent of rank. -/
def adamWStepErrorTrace {s : Shape} (stateError : AdamWStateBound s)
    (derived : AdamWDerivedErrors) (paramsError gradsError η : ℝ)
    (stateR : _root_.Optim.AdamW.State R s) (paramsR gradsR : Tensor R s) :
    AdamWStepErrorTrace :=
  let t' := stateR.t + 1
  let oneMinusBeta1R := 1 - stateR.beta1
  let oneMinusBeta2R := 1 - stateR.beta2
  let moment1LeftR := scaleSpec stateR.m stateR.beta1
  let moment1LeftError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      stateError.moment1 stateError.beta1 stateR.beta1 stateR.m)
  let moment1RightR := scaleSpec gradsR oneMinusBeta1R
  let moment1RightError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      gradsError derived.oneMinusBeta1 oneMinusBeta1R gradsR)
  let moment1R := addSpec moment1LeftR moment1RightR
  let moment1Error := linfNorm
    (addBoundTensor (β := β) (fexp := fexp)
      moment1LeftError moment1RightError moment1LeftR moment1RightR)
  let squaredGradsR := squareSpec gradsR
  let squaredGradError := linfNorm
    (mulBoundTensor (β := β) (fexp := fexp) gradsError gradsError gradsR gradsR)
  let moment2LeftR := scaleSpec stateR.v stateR.beta2
  let moment2LeftError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      stateError.moment2 stateError.beta2 stateR.beta2 stateR.v)
  let moment2RightR := scaleSpec squaredGradsR oneMinusBeta2R
  let moment2RightError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      squaredGradError derived.oneMinusBeta2 oneMinusBeta2R squaredGradsR)
  let moment2R := addSpec moment2LeftR moment2RightR
  let moment2Error := linfNorm
    (addBoundTensor (β := β) (fexp := fexp)
      moment2LeftError moment2RightError moment2LeftR moment2RightR)
  let biasInv1R := 1 / (1 - _root_.Optim.scalarPowNat stateR.beta1 t')
  let biasInv2R := 1 / (1 - _root_.Optim.scalarPowNat stateR.beta2 t')
  let moment1HatR := scaleSpec moment1R biasInv1R
  let moment1HatError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      moment1Error derived.biasInv1 biasInv1R moment1R)
  let moment2HatR := scaleSpec moment2R biasInv2R
  let moment2HatError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      moment2Error derived.biasInv2 biasInv2R moment2R)
  let stdR := sqrtSpec moment2HatR
  let stdError := linfNorm
    (sqrtPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd) η moment2HatError moment2HatR)
  let epsilonR := fill stateR.epsilon s
  let denominatorR := addSpec stdR epsilonR
  let denominatorError := linfNorm
    (addBoundTensor (β := β) (fexp := fexp)
      stdError stateError.epsilon stdR epsilonR)
  let lrR := fill stateR.lr s
  let adaptiveLRR := divSpec lrR denominatorR
  let adaptiveLRError := linfNorm
    (divPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      (Real.sqrt η) stateError.lr denominatorError lrR denominatorR)
  let adamUpdateR := mulSpec adaptiveLRR moment1HatR
  let adamUpdateError := linfNorm
    (mulBoundTensor (β := β) (fexp := fexp)
      adaptiveLRError moment1HatError adaptiveLRR moment1HatR)
  let decayScaleR := stateR.lr * stateR.weightDecay
  let decayUpdateR := scaleSpec paramsR decayScaleR
  let decayUpdateError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      paramsError derived.decayScale decayScaleR paramsR)
  let decayedParamsR := subSpec paramsR decayUpdateR
  let decayedParamsError := linfNorm
    (subBoundTensor (β := β) (fexp := fexp)
      paramsError decayUpdateError paramsR decayUpdateR)
  let paramsError' := linfNorm
    (subBoundTensor (β := β) (fexp := fexp)
      decayedParamsError adamUpdateError decayedParamsR adamUpdateR)
  { squaredGrad := squaredGradError
    moment1 := moment1Error
    moment2 := moment2Error
    moment1Hat := moment1HatError
    moment2Hat := moment2HatError
    std := stdError
    denominator := denominatorError
    adaptiveLR := adaptiveLRError
    adamUpdate := adamUpdateError
    decayUpdate := decayUpdateError
    decayedParams := decayedParamsError
    params := paramsError' }

/-- One AdamW update is numerically sound on a certified positive second-moment domain.

The hypotheses for the derived scalar expressions expose rounding in `1-β`, bias correction, and
the decoupled decay coefficient. `η` keeps `sqrt(vHat)` away from its singular derivative at zero;
the two margin hypotheses ensure the rounded second moment and final denominator remain positive.
-/
theorem approxTensor_adamW_update {s : Shape}
    {stateS : _root_.Optim.AdamW.State ℝ s} {stateR : _root_.Optim.AdamW.State R s}
    {stateError : AdamWStateBound s} {derived : AdamWDerivedErrors}
    {paramsS : Tensor ℝ s} {paramsR : Tensor R s} {paramsError : ℝ}
    {gradsS : Tensor ℝ s} {gradsR : Tensor R s} {gradsError η : ℝ}
    (hstate : adamWStateApprox (β := β) (fexp := fexp) (rnd := rnd)
      stateS stateR stateError)
    (hparams : approxTensor (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) paramsS paramsR paramsError)
    (hgrads : approxTensor (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) gradsS gradsR gradsError)
    (honeMinus1 : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (1 - stateR.beta1) -
      (1 - stateS.beta1)) ≤ derived.oneMinusBeta1)
    (honeMinus2 : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (1 - stateR.beta2) -
      (1 - stateS.beta2)) ≤ derived.oneMinusBeta2)
    (hbias1 : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
      (1 / (1 - _root_.Optim.scalarPowNat stateR.beta1 (stateR.t + 1))) -
      (1 / (1 - _root_.Optim.scalarPowNat stateS.beta1 (stateS.t + 1)))) ≤ derived.biasInv1)
    (hbias2 : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
      (1 / (1 - _root_.Optim.scalarPowNat stateR.beta2 (stateR.t + 1))) -
      (1 / (1 - _root_.Optim.scalarPowNat stateS.beta2 (stateS.t + 1)))) ≤ derived.biasInv2)
    (hdecay : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
      (stateR.lr * stateR.weightDecay) - stateS.lr * stateS.weightDecay) ≤ derived.decayScale)
    (hη : 0 < η)
    (hEpsilon : 0 ≤ stateS.epsilon)
    (hMoment2Hat :
      let t' := stateS.t + 1
      let moment2 := addSpec (scaleSpec stateS.v stateS.beta2)
        (scaleSpec (squareSpec gradsS) (1 - stateS.beta2))
      Tensor.Forall (fun z : ℝ => η ≤ z)
        (scaleSpec moment2 (1 / (1 - _root_.Optim.scalarPowNat stateS.beta2 t'))))
    (hMoment2Margin :
      (adamWStepErrorTrace (β := β) (fexp := fexp) (rnd := rnd)
        stateError derived paramsError gradsError η stateR paramsR gradsR).moment2Hat < η)
    (hDenominatorMargin :
      (adamWStepErrorTrace (β := β) (fexp := fexp) (rnd := rnd)
        stateError derived paramsError gradsError η stateR paramsR gradsR).denominator < Real.sqrt η) :
    let trace := adamWStepErrorTrace (β := β) (fexp := fexp) (rnd := rnd)
      stateError derived paramsError gradsError η stateR paramsR gradsR
    adamWStateApprox (β := β) (fexp := fexp) (rnd := rnd)
        (_root_.Optim.AdamW.update stateS paramsS gradsS).1
        (_root_.Optim.AdamW.update stateR paramsR gradsR).1
        { stateError with moment1 := trace.moment1, moment2 := trace.moment2 } ∧
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (_root_.Optim.AdamW.update stateS paramsS gradsS).2
        (_root_.Optim.AdamW.update stateR paramsR gradsR).2 trace.params := by
  dsimp only
  rcases hstate with ⟨hlr, hbeta1, hbeta2, hepsilon, hweightDecay, hm, hv, ht⟩
  let mS := addSpec (scaleSpec stateS.m stateS.beta1)
    (scaleSpec gradsS (1 - stateS.beta1))
  let mR := addSpec (scaleSpec stateR.m stateR.beta1)
    (scaleSpec gradsR (1 - stateR.beta1))
  let vS := addSpec (scaleSpec stateS.v stateS.beta2)
    (scaleSpec (squareSpec gradsS) (1 - stateS.beta2))
  let vR := addSpec (scaleSpec stateR.v stateR.beta2)
    (scaleSpec (squareSpec gradsR) (1 - stateR.beta2))
  let bias1S := 1 / (1 - _root_.Optim.scalarPowNat stateS.beta1 (stateS.t + 1))
  let bias1R := 1 / (1 - _root_.Optim.scalarPowNat stateR.beta1 (stateR.t + 1))
  let bias2S := 1 / (1 - _root_.Optim.scalarPowNat stateS.beta2 (stateS.t + 1))
  let bias2R := 1 / (1 - _root_.Optim.scalarPowNat stateR.beta2 (stateR.t + 1))
  let mHatS := scaleSpec mS bias1S
  let mHatR := scaleSpec mR bias1R
  let vHatS := scaleSpec vS bias2S
  let vHatR := scaleSpec vR bias2R
  let trace := adamWStepErrorTrace (β := β) (fexp := fexp) (rnd := rnd)
    stateError derived paramsError gradsError η stateR paramsR gradsR
  have hmLeft := approxTensor_scale_spec_of_approx
    (β := β) (fexp := fexp) (rnd := rnd) stateS.beta1 stateR.beta1 hm hbeta1
  have hmRight := approxTensor_scale_spec_of_approx
    (β := β) (fexp := fexp) (rnd := rnd)
    (1 - stateS.beta1) (1 - stateR.beta1) hgrads honeMinus1
  have hm' := approxTensor_add_spec (β := β) (fexp := fexp) (rnd := rnd) hmLeft hmRight
  have hsq := approxTensor_square_spec (β := β) (fexp := fexp) (rnd := rnd) hgrads
  have hvLeft := approxTensor_scale_spec_of_approx
    (β := β) (fexp := fexp) (rnd := rnd) stateS.beta2 stateR.beta2 hv hbeta2
  have hvRight := approxTensor_scale_spec_of_approx
    (β := β) (fexp := fexp) (rnd := rnd)
    (1 - stateS.beta2) (1 - stateR.beta2) hsq honeMinus2
  have hv' := approxTensor_add_spec (β := β) (fexp := fexp) (rnd := rnd) hvLeft hvRight
  have hmHat := approxTensor_scale_spec_of_approx
    (β := β) (fexp := fexp) (rnd := rnd) bias1S bias1R hm'
      (by simpa [bias1S, bias1R] using hbias1)
  have hvHat := approxTensor_scale_spec_of_approx
    (β := β) (fexp := fexp) (rnd := rnd) bias2S bias2R hv'
      (by simpa [bias2S, bias2R] using hbias2)
  have hsqrt := approxTensor_sqrt_spec_of_pos_lb
    (β := β) (fexp := fexp) (rnd := rnd) η hη hvHat
      (by simpa [vHatS, vS, bias2S] using hMoment2Hat)
      hMoment2Margin
  have hepsilonFill := approxTensor_fill_const
    (β := β) (fexp := fexp) (rnd := rnd) hepsilon (s := s)
  have hdenominator := approxTensor_add_spec
    (β := β) (fexp := fexp) (rnd := rnd) hsqrt hepsilonFill
  have hstdLower : Tensor.Forall (fun z : ℝ => Real.sqrt η ≤ z) (sqrtSpec vHatS) := by
    apply Tensor.forall_mapSpec (by simpa [vHatS, vS, bias2S] using hMoment2Hat)
    intro z hz
    change Real.sqrt η ≤ Real.sqrt (max z 0)
    exact Real.sqrt_le_sqrt (le_trans hz (le_max_left z 0))
  have hepsilonLower : Tensor.Forall (fun z : ℝ => 0 ≤ z) (fill stateS.epsilon s) :=
    Tensor.forall_fill hEpsilon
  have hdenominatorLower : Tensor.Forall (fun z : ℝ => Real.sqrt η ≤ z)
      (addSpec (sqrtSpec vHatS) (fill stateS.epsilon s)) := by
    apply Tensor.forall_map2Spec hstdLower hepsilonLower
    intro a b ha hb
    linarith
  have hlrFill := approxTensor_fill_const
    (β := β) (fexp := fexp) (rnd := rnd) hlr (s := s)
  have hadaptive := approxTensor_div_spec_of_pos_lb
    (β := β) (fexp := fexp) (rnd := rnd) (Real.sqrt η)
    hlrFill hdenominator hdenominatorLower hDenominatorMargin
  have hadamUpdate := approxTensor_mul_spec
    (β := β) (fexp := fexp) (rnd := rnd) hadaptive hmHat
  have hdecayUpdate := approxTensor_scale_spec_of_approx
    (β := β) (fexp := fexp) (rnd := rnd)
    (stateS.lr * stateS.weightDecay) (stateR.lr * stateR.weightDecay) hparams hdecay
  have hdecayed := approxTensor_sub_spec
    (β := β) (fexp := fexp) (rnd := rnd) hparams hdecayUpdate
  have hnext := approxTensor_sub_spec
    (β := β) (fexp := fexp) (rnd := rnd) hdecayed hadamUpdate
  constructor
  · refine ⟨hlr, hbeta1, hbeta2, hepsilon, hweightDecay, ?_, ?_, ?_⟩
    · simpa [_root_.Optim.AdamW.update, trace, adamWStepErrorTrace, mS, mR] using hm'
    · simpa [_root_.Optim.AdamW.update, trace, adamWStepErrorTrace, vS, vR] using hv'
    · simpa [_root_.Optim.AdamW.update] using ht
  · simpa [trace, adamWStepErrorTrace, _root_.Optim.AdamW.update,
      _root_.Optim.OptimizerUtils.mkAdaptiveLR, mS, mR, vS, vR, mHatS, mHatR,
      vHatS, vHatR, bias1S, bias1R, bias2S, bias2R] using hnext

/-! ## AdamW contract instance -/

/-- Numerical data and positivity margin for one AdamW update. -/
structure AdamWStepData where
  /-- Bounds for rounded scalar subexpressions used by bias correction and decay. -/
  derived : AdamWDerivedErrors
  /-- Strict lower bound on the exact bias-corrected second moment. -/
  eta : ℝ

/-- Complete validity predicate for one AdamW contract application. -/
def adamWStepDataValid {s : Shape}
    (stateS : _root_.Optim.AdamW.State ℝ s) (stateR : _root_.Optim.AdamW.State R s)
    (stateError : AdamWStateBound s)
    (_paramsS : Tensor ℝ s) (paramsR : Tensor R s) (paramsError : ℝ)
    (gradsS : Tensor ℝ s) (gradsR : Tensor R s) (gradsError : ℝ)
    (data : AdamWStepData) : Prop :=
  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (1 - stateR.beta1) -
      (1 - stateS.beta1)) ≤ data.derived.oneMinusBeta1 ∧
  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (1 - stateR.beta2) -
      (1 - stateS.beta2)) ≤ data.derived.oneMinusBeta2 ∧
  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
      (1 / (1 - _root_.Optim.scalarPowNat stateR.beta1 (stateR.t + 1))) -
      (1 / (1 - _root_.Optim.scalarPowNat stateS.beta1 (stateS.t + 1)))) ≤
    data.derived.biasInv1 ∧
  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
      (1 / (1 - _root_.Optim.scalarPowNat stateR.beta2 (stateR.t + 1))) -
      (1 / (1 - _root_.Optim.scalarPowNat stateS.beta2 (stateS.t + 1)))) ≤
    data.derived.biasInv2 ∧
  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
      (stateR.lr * stateR.weightDecay) - stateS.lr * stateS.weightDecay) ≤
    data.derived.decayScale ∧
  0 < data.eta ∧
  0 ≤ stateS.epsilon ∧
  (let t' := stateS.t + 1
   let moment2 := addSpec (scaleSpec stateS.v stateS.beta2)
     (scaleSpec (squareSpec gradsS) (1 - stateS.beta2))
   Tensor.Forall (fun z : ℝ => data.eta ≤ z)
     (scaleSpec moment2 (1 / (1 - _root_.Optim.scalarPowNat stateS.beta2 t')))) ∧
  (adamWStepErrorTrace (β := β) (fexp := fexp) (rnd := rnd)
      stateError data.derived paramsError gradsError data.eta stateR paramsR gradsR).moment2Hat <
    data.eta ∧
  (adamWStepErrorTrace (β := β) (fexp := fexp) (rnd := rnd)
      stateError data.derived paramsError gradsError data.eta stateR paramsR gradsR).denominator <
    Real.sqrt data.eta

/-- State and parameter error object produced by one AdamW contract step. -/
def adamWStepBound {s : Shape} (stateError : AdamWStateBound s)
    (paramsError gradsError : ℝ) (stateR : _root_.Optim.AdamW.State R s)
    (paramsR gradsR : Tensor R s) (data : AdamWStepData) :
    StepBound AdamWStateBound s :=
  let trace := adamWStepErrorTrace (β := β) (fexp := fexp) (rnd := rnd)
    stateError data.derived paramsError gradsError data.eta stateR paramsR gradsR
  { state := { stateError with moment1 := trace.moment1, moment2 := trace.moment2 }
    params := trace.params }

/-- AdamW instance of the generic numerical optimizer contract.

Its side conditions are data, not a second execution framework. `NumericalStepContract.run_approx`
therefore composes AdamW over finite runs exactly as it does SGD and momentum SGD.
-/
def adamWContract : NumericalStepContract R
    (toSpec (β := β) (fexp := fexp) (rnd := rnd)) where
  name := "AdamW"
  StateSpec := _root_.Optim.AdamW.State ℝ
  StateRuntime := _root_.Optim.AdamW.State R
  StateBound := AdamWStateBound
  StepData := fun _ => AdamWStepData
  stateApprox := adamWStateApprox (β := β) (fexp := fexp) (rnd := rnd)
  stepDataValid := adamWStepDataValid (β := β) (fexp := fexp) (rnd := rnd)
  updateSpec := _root_.Optim.AdamW.update
  updateRuntime := _root_.Optim.AdamW.update
  updateBound := fun stateError paramsError gradsError state params grads data =>
    adamWStepBound (β := β) (fexp := fexp) (rnd := rnd)
      stateError paramsError gradsError state params grads data
  stateBoundReport := fun error =>
    [("learning rate", error.lr), ("beta1", error.beta1), ("beta2", error.beta2),
      ("epsilon", error.epsilon), ("weight decay", error.weightDecay),
      ("first moment", error.moment1), ("second moment", error.moment2)]
  stepDataReport := fun data =>
    [("eta", data.eta), ("1 - beta1", data.derived.oneMinusBeta1),
      ("1 - beta2", data.derived.oneMinusBeta2),
      ("bias inverse 1", data.derived.biasInv1),
      ("bias inverse 2", data.derived.biasInv2),
      ("decay scale", data.derived.decayScale)]
  updateSound := by
    intro s stateS stateR stateError paramsS paramsR paramsError gradsS gradsR gradsError data
      hstate hparams hgrads hvalid
    rcases hvalid with
      ⟨honeMinus1, honeMinus2, hbias1, hbias2, hdecay, hEta, hEpsilon,
        hMoment2Hat, hMoment2Margin, hDenominatorMargin⟩
    simpa [adamWStepBound] using
      (approxTensor_adamW_update (β := β) (fexp := fexp) (rnd := rnd)
        hstate hparams hgrads honeMinus1 honeMinus2 hbias1 hbias2 hdecay hEta hEpsilon
        hMoment2Hat hMoment2Margin hDenominatorMargin)

end
end Proofs.RuntimeApprox.NFBackend.Optimizer

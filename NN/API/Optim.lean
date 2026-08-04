/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural
public import NN.API.Trainer.Manual.Core

/-!
# Optimizers

Optimizer configuration records and runtime optimizer constructors.

The default trainer config exposes self-contained core update rules for SGD, momentum SGD,
AdaGrad, RMSProp, Adam, AdamW, and Adadelta. Runtime-only extension points live here too:

- Muon is an optimizer, but the public runtime constructor requires an explicit orthogonalization
  backend.  The identity backend is available for proofs and fallback behavior.
- GaLore is exposed as gradient-projection machinery around a base update.  The public name is
  therefore `optim.galore.projectedSGD`, which says exactly which update rule owns the state.
-/

@[expose] public section

namespace TorchLean

namespace optim

/-- Optimizer algorithm and hyperparameters used by `Trainer`. -/
abbrev Optimizer := TorchLean.Trainer.Manual.OptimizerConfig

/-- Public SGD optimizer configuration. -/
structure SgdConfig where
  /-- Learning rate. -/
  lr : Float
  /-- Momentum coefficient. -/
  momentum : Float := 0.0
deriving Repr

/-- Public AdaGrad optimizer configuration. -/
structure AdaGradConfig where
  /-- Learning rate. -/
  lr : Float
  /-- Numerical stabilizer. -/
  epsilon : Float := 1e-10
deriving Repr

/-- Public RMSProp optimizer configuration. -/
structure RMSPropConfig where
  /-- Learning rate. -/
  lr : Float
  /-- Decay coefficient for the running average of squared gradients. -/
  decay : Float := 0.99
  /-- Numerical stabilizer. -/
  epsilon : Float := 1e-8
deriving Repr

/-- Public Adam optimizer configuration. -/
structure AdamConfig where
  /-- Learning rate. -/
  lr : Float
  /-- First moment coefficient. -/
  beta1 : Float := 0.9
  /-- Second moment coefficient. -/
  beta2 : Float := 0.999
  /-- Numerical stabilizer. -/
  epsilon : Float := 1e-8
deriving Repr

/-- Public AdamW optimizer configuration. -/
structure AdamWConfig extends AdamConfig where
  /-- Decoupled weight decay. -/
  weightDecay : Float := 0.01
deriving Repr

/-- Public Adadelta optimizer configuration. -/
structure AdadeltaConfig where
  /-- Learning rate. -/
  lr : Float := 1.0
  /-- Decay coefficient for gradient/update accumulators. -/
  rho : Float := 0.9
  /-- Numerical stabilizer. -/
  epsilon : Float := 1e-6
deriving Repr

/-- SGD optimizer config, written `optim.sgd { lr := 0.05 }`. -/
def sgd (cfg : SgdConfig) : Optimizer :=
  .sgd cfg.lr cfg.momentum

/-- Momentum SGD, using momentum `0.9` when the configuration leaves it at zero. -/
def momentumSGD (cfg : SgdConfig) : Optimizer :=
  .sgd cfg.lr
    (if cfg.momentum == 0.0 then 0.9 else cfg.momentum)

/-- AdaGrad optimizer config, written `optim.adagrad { lr := 0.05 }`. -/
def adagrad (cfg : AdaGradConfig) : Optimizer :=
  .adagrad cfg.lr cfg.epsilon

/-- RMSProp optimizer config, written `optim.rmsprop { lr := 1e-3 }`. -/
def rmsprop (cfg : RMSPropConfig) : Optimizer :=
  .rmsprop cfg.lr cfg.decay cfg.epsilon

/-- Adam optimizer config, written `optim.adam { lr := 1e-3 }`. -/
def adam (cfg : AdamConfig) : Optimizer :=
  .adam cfg.lr cfg.beta1 cfg.beta2 cfg.epsilon

/-- AdamW optimizer config, written `optim.adamw { lr := 1e-3, weightDecay := 0.01 }`. -/
def adamw (cfg : AdamWConfig) : Optimizer :=
  .adamw cfg.lr cfg.weightDecay cfg.beta1 cfg.beta2 cfg.epsilon

/-- Adadelta optimizer config, written `optim.adadelta {}`. -/
def adadelta (cfg : AdadeltaConfig) : Optimizer :=
  .adadelta cfg.lr cfg.rho cfg.epsilon

/-- Optimizer names accepted by model-training commands. -/
inductive Kind where
  | sgd
  | adagrad
  | rmsprop
  | adam
  | adamw
  | adadelta
deriving DecidableEq, Repr

namespace Kind

/-- Parse an optimizer name used by `--optim`. -/
def parse (value : String) : Except String Kind :=
  match value with
  | "sgd" => pure .sgd
  | "adagrad" => pure .adagrad
  | "rmsprop" => pure .rmsprop
  | "adam" => pure .adam
  | "adamw" => pure .adamw
  | "adadelta" => pure .adadelta
  | _ =>
      throw s!"bad --optim {value}; expected sgd, adagrad, rmsprop, adam, adamw, or adadelta"

/-- Name written to logs and summaries. -/
def name : Kind → String
  | .sgd => "SGD"
  | .adagrad => "AdaGrad"
  | .rmsprop => "RMSProp"
  | .adam => "Adam"
  | .adamw => "AdamW"
  | .adadelta => "Adadelta"

/-- Build the selected optimizer with the defaults used by training commands. -/
def toOptimizer (kind : Kind) (lr : Float) : Optimizer :=
  match kind with
  | .sgd => optim.sgd { lr }
  | .adagrad => optim.adagrad { lr }
  | .rmsprop => optim.rmsprop { lr }
  | .adam => optim.adam { lr, beta2 := 0.95 }
  | .adamw => optim.adamw { lr, weightDecay := 0.1, beta2 := 0.95 }
  | .adadelta => optim.adadelta { lr }

end Kind

/-- Runtime Adam optimizer for module-level training. -/
def runtimeAdam {α : Type} [_root_.Context α]
    (lr beta1 beta2 epsilon : α) {paramShapes : List Shape} :
    _root_.Runtime.Autograd.TorchLean.Optim.Optimizer α paramShapes :=
  _root_.Runtime.Autograd.TorchLean.Optim.adam (α := α) lr beta1 beta2 epsilon (paramShapes := paramShapes)

/-- Runtime AdamW optimizer for module-level training. -/
def runtimeAdamW {α : Type} [_root_.Context α]
    (lr weightDecay beta1 beta2 epsilon : α) {paramShapes : List Shape} :
    _root_.Runtime.Autograd.TorchLean.Optim.Optimizer α paramShapes :=
  _root_.Runtime.Autograd.TorchLean.Optim.adamw (α := α) lr weightDecay beta1 beta2 epsilon
    (paramShapes := paramShapes)

/-- Runtime SGD optimizer for module-level training. -/
def runtimeSGD {α : Type} [_root_.Context α]
    (lr : α) {paramShapes : List Shape} :
    _root_.Runtime.Autograd.TorchLean.Optim.Optimizer α paramShapes :=
  _root_.Runtime.Autograd.TorchLean.Optim.sgd (α := α) lr (paramShapes := paramShapes)

/-- Runtime momentum-SGD optimizer for module-level training. -/
def runtimeMomentumSGD {α : Type} [_root_.Context α]
    (lr momentum : α) {paramShapes : List Shape} :
    _root_.Runtime.Autograd.TorchLean.Optim.Optimizer α paramShapes :=
  _root_.Runtime.Autograd.TorchLean.Optim.momentumSGD (α := α) lr momentum (paramShapes := paramShapes)

/-- Runtime AdaGrad optimizer for module-level training. -/
def runtimeAdaGrad {α : Type} [_root_.Context α]
    (lr epsilon : α) {paramShapes : List Shape} :
    _root_.Runtime.Autograd.TorchLean.Optim.Optimizer α paramShapes :=
  _root_.Runtime.Autograd.TorchLean.Optim.adagrad (α := α) lr epsilon (paramShapes := paramShapes)

/-- Runtime RMSProp optimizer for module-level training. -/
def runtimeRMSProp {α : Type} [_root_.Context α]
    (lr decay epsilon : α) {paramShapes : List Shape} :
    _root_.Runtime.Autograd.TorchLean.Optim.Optimizer α paramShapes :=
  _root_.Runtime.Autograd.TorchLean.Optim.rmsprop (α := α) lr decay epsilon (paramShapes := paramShapes)

/-- Runtime Adadelta optimizer for module-level training. -/
def runtimeAdadelta {α : Type} [_root_.Context α]
    (lr rho epsilon : α) {paramShapes : List Shape} :
    _root_.Runtime.Autograd.TorchLean.Optim.Optimizer α paramShapes :=
  _root_.Runtime.Autograd.TorchLean.Optim.adadelta (α := α) lr rho epsilon (paramShapes := paramShapes)

@[inherit_doc _root_.Optim.Muon.Orthogonalizer]
abbrev MuonOrthogonalizer := _root_.Optim.Muon.Orthogonalizer

@[inherit_doc _root_.Optim.Muon.identityOrthogonalizer]
def identityMuonOrthogonalizer {α : Type} {s : Shape} :
    MuonOrthogonalizer α s :=
  _root_.Optim.Muon.identityOrthogonalizer (α := α) (s := s)

/--
Runtime Muon-style optimizer for module-level training.

Muon is public at the runtime layer because a meaningful Muon run needs an orthogonalization
backend. The default identity backend supports proofs and fallback behavior; production Muon should
pass a matrix-shaped orthogonalizer.
-/
def runtimeMuon {α : Type} [_root_.Context α]
    (lr momentum : α)
    (orthogonalizer : {s : Shape} → MuonOrthogonalizer α s :=
      fun {s} => identityMuonOrthogonalizer (α := α) (s := s))
    {paramShapes : List Shape} :
    _root_.Runtime.Autograd.TorchLean.Optim.Optimizer α paramShapes :=
  _root_.Runtime.Autograd.TorchLean.Optim.muon (α := α) lr momentum orthogonalizer (paramShapes := paramShapes)

namespace galore

export _root_.Optim.GaLore (Projector)

@[inherit_doc _root_.Optim.GaLore.identityProjector]
def identityProjector {α : Type} {s : Shape} :
    Projector α s s :=
  _root_.Optim.GaLore.identityProjector (α := α) (s := s)

/--
Projected-SGD runtime constructor for GaLore-style gradient projection.

This is a projection strategy wrapped around an SGD update.  Full GaLore also needs a policy that
constructs and refreshes low-rank projectors for matrix parameters; this constructor exposes the
verified update boundary once a same-shape projector is supplied.
-/
def projectedSGD {α : Type} [_root_.Context α]
    (lr : α)
    (projector : {s : Shape} → Projector α s s :=
      fun {s} => identityProjector (α := α) (s := s))
    {paramShapes : List Shape} :
    _root_.Runtime.Autograd.TorchLean.Optim.Optimizer α paramShapes :=
  _root_.Runtime.Autograd.TorchLean.Optim.projectedSGD (α := α) lr projector (paramShapes := paramShapes)

end galore

end optim


end TorchLean

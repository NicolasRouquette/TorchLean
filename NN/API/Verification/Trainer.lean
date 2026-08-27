/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural.Execution
public import NN.Tensor
public import NN.API.Trainer.Predict
public import NN.API.Trainer.Train.Regression
public import NN.MLTheory.CROWN.Flatbox
public import NN.MLTheory.CROWN.Graph
public import NN.Verification.TorchLean.Lowering

/-!
# Trained Regression Verification

Opt-in IBP verification for trained regression models. Import this module and use
`Trainer.trainVerified` to retain a required verifier alongside the ordinary training result.
Importing `NN.API` or `NN.API.Trainer` does not import this extension or the CROWN implementation.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

/-- Result of checking an output box with a trained-model IBP verifier. -/
structure VerificationReport where
  /-- Number of IR nodes in the lowered verifier graph. -/
  nodes : Nat
  /-- Flattened output dimension reported by the verifier. -/
  outputDim : Nat
  /-- Lower bound on the flattened output box. -/
  lo : String
  /-- Upper bound on the flattened output box. -/
  hi : String

namespace VerificationReport

/-- One-line verification summary. -/
def summary (report : VerificationReport) : String :=
  s!"IBP nodes={report.nodes} output_dim={report.outputDim} lo={report.lo} hi={report.hi}"

/-- Print the verification summary. -/
def printSummary (report : VerificationReport) : IO Unit :=
  IO.println report.summary

instance : ToString VerificationReport where
  toString := summary

end VerificationReport

/--
A trained regression model with an attached IBP verifier.

The ordinary training report and prediction operations are inherited from `TrainResult`; the
verification operation is required because this value can only be built by the opt-in verified
training path.
-/
structure VerifiedTrainResult (inputShape outputShape : List Nat)
    extends TrainResult inputShape outputShape where
  /-- Verify a uniform $\ell_\infty$ input ball around the supplied center. -/
  verifyRobustLInf : Tensor Float inputShape → Float → IO VerificationReport

namespace VerifiedTrainResult

/-- Verify a uniform $\ell_\infty$ input ball and print the resulting output interval. -/
def printRobustLInf {inputShape outputShape : List Nat}
    (result : VerifiedTrainResult inputShape outputShape) (center : Tensor Float inputShape)
    (eps : Float) : IO Unit := do
  let report ← result.verifyRobustLInf center eps
  report.printSummary

instance {inputShape outputShape : List Nat} :
    ToString (VerifiedTrainResult inputShape outputShape) where
  toString result := result.report.summary

end VerifiedTrainResult

namespace Internal

namespace Regression

/-- Build the IBP operation over the live parameters in an already-trained runner. -/
def mkVerifyRobustLInf {inputShape outputShape : List Nat} {α : Type}
    [_root_.Context α] [DecidableEq Shape] [ToString α] [Runtime.FromFloat α]
    [NN.MLTheory.CROWN.BoundOps α]
    [Runtime.TensorTransfer α]
    (trainer : SelectedTask inputShape outputShape)
    (runner : TorchLean.Trainer.Manual.Runner α trainer.task) :
    Tensor Float inputShape → Float → IO VerificationReport :=
  fun centerFloat eps => do
    Manual.Runner.eval (task := trainer.task) runner
    let state : _root_.TorchLean.TensorPack α
        trainer.task.stateShapes ←
      Manual.Runner.state (task := trainer.task) runner
    let modelState : _root_.TorchLean.TensorPack α (nn.stateShapes trainer.model) :=
      Eq.mp (congrArg (_root_.TorchLean.TensorPack α) trainer.stateShapes_eq) state
    let lowered ←
      match NN.Verification.TorchLean.lowerForwardToIR
          (TorchLean.nn.forward trainer.model (α := α)) modelState with
      | .ok result => pure result
      | .error message => throw <| IO.userError message
    let center := Tensor.map (Runtime.ofFloat (α := α)) centerFloat
    let ps := lowered.seedLInfBall center (Runtime.ofFloat eps)
    let ibp := lowered.runIBP ps
    let outB ←
      match lowered.outputBox? ibp with
      | .ok box => pure box
      | .error message => throw <| IO.userError message
    pure
      { nodes := lowered.graph.nodes.size
        outputDim := outB.dim
        lo := Tensor.pretty outB.lo
        hi := Tensor.pretty outB.hi }

/-- Train and retain an IBP verifier using an already-instantiated runner. -/
def trainVerifiedWithRunner {inputShape outputShape : List Nat} {α : Type}
    [_root_.Context α] [DecidableEq Shape] [ToString α] [Runtime.FromFloat α]
    [NN.MLTheory.CROWN.BoundOps α]
    [Runtime.TensorTransfer α]
    (trainer : SelectedTask inputShape outputShape)
    (data : Dataset inputShape outputShape)
    (cfg : TorchLean.Trainer.Manual.TrainConfig)
    (probes : Array (Probe inputShape))
    (runner : TorchLean.Trainer.Manual.Runner α trainer.task) :
    IO (VerifiedTrainResult inputShape outputShape) := do
  let (result, _) ←
    trainCoreWithRunner (α := α) trainer data cfg probes runner (fun _ => pure ())
  pure
    { toTrainResult := result
      verifyRobustLInf := mkVerifyRobustLInf trainer runner }

/-- Train the selected regression task and retain its live IBP verifier. -/
def trainVerified {inputShape outputShape : List Nat}
    (trainer : SelectedTask inputShape outputShape)
    (data : Dataset inputShape outputShape)
    (opts : TrainOptions := {}) (probes : Array (Probe inputShape) := #[]) :
    IO (VerifiedTrainResult inputShape outputShape) := do
  let run := trainer.runtime
  let result ← SelectedTask.withRunnerFor trainer run
    (Regression.trainVerifiedWithRunner trainer data
      (opts.toTrainConfig run.optimizer) probes)
    (Regression.trainVerifiedWithRunner trainer data
      (opts.toTrainConfig run.optimizer) probes)
  result.report.writeLog opts.log opts.title opts.notes
  pure result

end Regression

end Internal

/-- Train using the trainer's runtime settings and retain an IBP verifier. -/
def trainVerified {inputShape outputShape : List Nat}
    (trainer : TorchLean.Trainer inputShape outputShape)
    (data : Dataset inputShape outputShape) (opts : TrainOptions := {})
    (probes : Array (Probe inputShape) := #[]) :
    IO (VerifiedTrainResult inputShape outputShape) :=
  match trainer.task with
  | .regression reduction =>
      Internal.Regression.trainVerified
        (Internal.SelectedTask.regression trainer reduction) data opts probes
  | @Task.oneHotCrossEntropy _ _ _ _ =>
      throw <| IO.userError
        "Trainer.trainVerified: indexed classification verification is not implemented"
  | .custom .. =>
      throw <| IO.userError
        "Trainer.trainVerified: custom-loss verification is not implemented"

end Trainer

end TorchLean

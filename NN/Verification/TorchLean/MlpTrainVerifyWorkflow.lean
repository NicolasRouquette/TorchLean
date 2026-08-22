/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.API.Module.Command

/-!
# TorchLean MLP Train-Then-Verify Workflow

This is a native TorchLean workflow: the model is built, trained, lowered to verifier IR, and
checked without importing an external certificate.

Training:
- build a 2-layer ReLU MLP with a scalar MSE loss
- train it for a few SGD steps with reusable typed graph execution

Verification:
- lower the trained model's forward pass to the verifier IR
- run public IBP bounds on a small input box around one sample

Run:
  `lake exe verify -- torchlean-mlp-workflow --scalar float32`
  `lake exe verify -- torchlean-mlp-workflow --scalar ieee32-exec`
-/

@[expose] public section


namespace NN.Verification.TorchLean.MlpTrainVerifyWorkflow

open _root_.Spec
open _root_.Spec.Tensor
open _root_.TorchLean

/-- Input dimension for the workflow model. -/
def inDim : Nat := 2
/-- Hidden width for the workflow model. Kept as local config, not as part of the public name. -/
def hiddenDim : Nat := 100
/-- Output dimension for the workflow model. -/
def outDim : Nat := 1

/-- Input shape for the workflow model. -/
def xShape : Spec.Shape := .dim inDim .scalar
/-- Output shape for the workflow model. -/
def yShape : Spec.Shape := .dim outDim .scalar

/-- Batched inputs for training. -/
def XFloat : Spec.Tensor Float (.dim 3 xShape) :=
  tensor! [[1.0, 0.0],
           [0.0, 1.0],
           [1.0, 1.0]]

/-- Affine regression target used by the workflow. -/
def target (x : Spec.Tensor Float xShape) : Spec.Tensor Float yShape :=
  let x₁ := Spec.Tensor.item (Spec.get x ⟨0, by decide⟩)
  let x₂ := Spec.Tensor.item (Spec.get x ⟨1, by decide⟩)
  Spec.Tensor.dim fun _ => Spec.Tensor.scalar (2.0 * x₁ - 3.0 * x₂)

/-- Batched targets for training. -/
def YFloat : Spec.Tensor Float (.dim 3 yShape) :=
  Spec.Tensor.mapLeading (.dim 3 .scalar) target XFloat

/-- TorchLean model used for training and verification. -/
def mkModel : nn.Builder (nn.Sequential xShape yShape) :=
  nn.Sequential![
    nn.linear inDim hiddenDim,
    nn.relu,
    nn.linear hiddenDim outDim
  ]

/-- Deterministically instantiate the workflow model from initialization seed zero. -/
def model : nn.Sequential xShape yShape :=
  nn.build 0 mkModel

/--
Run training and verification under a chosen scalar backend `α`.

The trained result owns the trained parameters. Calling `trained.verifyRobustLInf` therefore checks the
model that was actually trained, without reopening a polymorphic low-level callback in this example
file.
-/
def runOnce {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    [Runtime.FromFloat α] (opts : Options) : IO Unit := do
  let dataset := Data.tensorDataset XFloat YFloat
  let trainer := Trainer.new model <|
    Trainer.Config.fromRunConfig
      (Trainer.RunConfig.ofRuntimeOptions opts { optimizer := optim.sgd { lr := 0.05 } })
      .regression

  IO.println s!"== TorchLean MLP workflow ({inDim} → {hiddenDim} → {outDim}) =="
  IO.println s!"Training with execution={reprStr opts.execution}, device={opts.device.cliName}"
  let trained ← trainer.train dataset { steps := 10 }
  IO.println s!"avg_loss(on samples)={trained.report.after}"
  IO.println "Checking public IBP bounds on a small input box"
  let center := _root_.Spec.get XFloat ⟨0, by decide⟩
  let cert ← trained.verifyRobustLInf center 0.05
  cert.printSummary

/-- Runtime-selected typed runner used by the CLI entrypoint. -/
def runMain {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    [Runtime.FromFloat α] (opts : Options) (rest : List String) : IO Unit := do
  CLI.requireNoArgs "torchlean-mlp-workflow" rest
  if opts.usesCuda then
    throw <| IO.userError
      "torchlean-mlp-workflow: CUDA eager training is not used here; this workflow keeps trained parameters as Lean tensors so the verifier can lower and check them. Use the model-training examples for CUDA runtime training, or run this verifier workflow without --device cuda."
  runOnce (α := α) opts

/--
CLI entry point for the native TorchLean MLP workflow.

This is wired into `lake exe verify -- torchlean-mlp-workflow`.
-/
def main (args : List String) : IO Unit := do
  let args :=
    if CLI.hasFlagValue args "execution" then
      args
    else
      "--execution=typed-graph" :: args
  Module.withRuntime args
    (fun {α} _ _ _ _ _cast opts rest => runMain (α := α) opts rest)

end NN.Verification.TorchLean.MlpTrainVerifyWorkflow

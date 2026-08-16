/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Examples.Models
public import NN.Examples.DeepDives
public import NN.Examples.Quickstart
public import NN.Examples.Data.Loaders.Csv
public import NN.Examples.Data.Loaders.Npy
public import NN.Examples.Data.Loaders.Cifar10Images
public import NN.Examples.Interop.PyTorch.Roundtrip
public import NN.Examples.Interop.PyTorch.TorchExportCheck

/-!
# TorchLean Example Runner

Executable root for the `torchlean` example runner.

Each example keeps its own namespace, and this module selects which one to run from a subcommand
argument such as `mlp` or `gpt2`.
-/

@[expose] public section

open System
open TorchLean

namespace NN.Examples.Models.Runner

def usage : String :=
  String.intercalate "\n"
    [ "TorchLean runnable examples"
    , ""
    , "Usage:"
    , "  lake exe torchlean <example> [flags...]"
    , "  lake exe torchlean --choose <example> [flags...]"
    , "  lake exe torchlean <example> --help"
    , ""
    , "Start here:"
    , "  lake exe torchlean quickstart_tensors"
    , "  lake exe torchlean quickstart_autograd"
    , "  lake exe torchlean quickstart_mlp --steps 20"
    , "  lake exe torchlean --choose quickstart_mlp --steps 20"
    , "  lake -R -K cuda=true exe torchlean gpt2 --device cuda --tiny-shakespeare --steps 1 --windows 1 --generate 0"
    , ""
    , "Common examples:"
    , "  quickstart_tensors | quickstart_autograd | quickstart_mlp | quickstart_minibatch_mlp | quickstart_cnn"
    , "  mlp | cnn | resnet | transformer | gpt2 | text_gpt2 | mamba | fno1d_burgers"
    , "  autoencoder | vae | gan | diffusion | ppo_cartpole | dqn_replay"
    , "  pytorch_roundtrip | pytorch_export_check"
    , "  data_csv | data_npy | data_cifar10"
    , "  float32_semantics | numerical_certificate | graphspec | ir_axis_ops | one_semantic_universe"
    , ""
    , "Runtime flags:"
    , "  --choose                         ask for runtime choices before running"
    , "  --device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external"
    , "  --scalar float32|ieee32-exec|complex64"
    , "  --execution eager|typed-graph"
    , "  --seed N"
    , "  --show-backend"
    , ""
    , "Verification commands live under `lake exe verify -- list`."
    , "Use `lake exe torchlean <example> --help` for command-specific flags."
    ]

/-- Runtime flags that consume the following command-line token. -/
def prefixFlagTakesValue (a : String) : Bool :=
  a == "--device" ||
  a == "--scalar" ||
  a == "--execution" ||
  a == "--seed"

/--
Split CLI arguments into `(prefixFlags, command, commandArgs)`.

We allow "global" runtime flags before the command name so users can write either:
- `torchlean mlp --device cpu`, or
- `torchlean --device cpu mlp`.
-/
def splitCommandArgs? (args : List String) : Option (List String × String × List String) :=
  let rec go (prefixRev : List String) : List String → Option (List String × String × List String)
    | [] => none
    | a :: b :: rest =>
        if prefixFlagTakesValue a then
          go (b :: a :: prefixRev) rest
        else if a.startsWith "-" then
          go (a :: prefixRev) (b :: rest)
        else
          some (prefixRev.reverse, a, b :: rest)
    | a :: rest =>
        if a.startsWith "-" then
          go (a :: prefixRev) rest
        else
          some (prefixRev.reverse, a, rest)
  go [] args

/-- Detect whether the command line already selects a runtime device. -/
def hasDeviceFlag : List String → Bool
  | [] => false
  | "--device" :: _ :: _ => true
  | a :: rest =>
      a.startsWith "--device=" || hasDeviceFlag rest

/-- Read one line, treating EOF as the default answer. -/
def readPromptLine : IO String := do
  try
    let line ← (← IO.getStdin).getLine
    pure line.trimAscii.toString
  catch _ =>
    pure ""

/-- Ask for the device in the interactive runtime chooser. -/
partial def askDevice : IO (List String) := do
  IO.println "Runtime device:"
  IO.println "  1) CPU    portable default"
  IO.println "  2) CUDA   GPU runtime, requires `lake -R -K cuda=true exe ...`"
  IO.print "Select device [1]: "
  (← IO.getStdout).flush
  match (← readPromptLine).toLower with
  | "" | "1" | "cpu" => pure ["--device", "cpu"]
  | "2" | "cuda" => pure ["--device", "cuda"]
  | _ =>
      IO.println "Please choose 1/cpu or 2/cuda."
      askDevice

/--
Fill in missing runtime choices interactively.

The chooser is explicit (`--choose`) so scripts, tests, and CI never block waiting for stdin.
-/
def chooseRuntimeArgs (args : List String) : IO (List String) := do
  IO.println "TorchLean runtime chooser"
  let deviceArgs ←
    if hasDeviceFlag args then
      pure []
    else
      askDevice
  pure (deviceArgs ++ args)

/-- Strip top-level runner flags that are not meant for individual examples. -/
def stripRunnerFlag (flag : String) : List String → List String
  | [] => []
  | a :: rest =>
      if a == flag then stripRunnerFlag flag rest else a :: stripRunnerFlag flag rest

def runCmd (cmd : String) (args : List String) : IO UInt32 := do
  match cmd with
  | "quickstart_tensors" =>
      NN.Examples.Quickstart.TensorBasics.main args
      pure 0
  | "quickstart_autograd" =>
      NN.Examples.Quickstart.AutogradBasics.main args
      pure 0
  | "quickstart_mlp" =>
      NN.Examples.Quickstart.SimpleMLPTrain.main args
      pure 0
  | "quickstart_minibatch_mlp" =>
      NN.Examples.Quickstart.MinibatchMLPTrain.main args
      pure 0
  | "quickstart_cnn" =>
      NN.Examples.Quickstart.SimpleCNNTrain.main args
      pure 0
  | "mlp" => NN.Examples.Models.Supervised.Mlp.main args
  | "kan" => NN.Examples.Models.Supervised.Kan.main args
  | "cnn" => NN.Examples.Models.Vision.Cnn.main args
  | "resnet" => NN.Examples.Models.Vision.ResNet.main args
  | "diffusion" => NN.Examples.Models.Generative.Diffusion.main args
  | "fno1d_burgers" => NN.Examples.Models.Operators.Fno1dBurgers.main args
  | "autoencoder" => NN.Examples.Models.Generative.Autoencoder.main args
  | "mae" => NN.Examples.Models.Generative.Mae.main args
  | "vae" => NN.Examples.Models.Generative.Vae.main args
  | "vqvae" => NN.Examples.Models.Generative.VqVae.main args
  | "gan" => NN.Examples.Models.Generative.Gan.main args
  | "rnn" => NN.Examples.Models.Sequence.Rnn.main args
  | "lstm" => NN.Examples.Models.Sequence.Lstm.main args
  | "lstm_regression" => NN.Examples.Models.Supervised.LstmRegression.main args
  | "transformer" => NN.Examples.Models.Sequence.Transformer.main args
  | "vit" => NN.Examples.Models.Vision.Vit.main args
  | "gpt2" => NN.Examples.Models.Sequence.Gpt2.main args
  | "gpt2_saved" => NN.Examples.Models.Sequence.Gpt2Saved.main args
  | "text_gpt2" => NN.Examples.Models.Sequence.TextGPT2.main args
  | "chargpt" => NN.Examples.Models.Sequence.CharGpt.main args
  | "gpt_adder" => NN.Examples.Models.Sequence.GptAdder.main args
  | "mamba" => NN.Examples.Models.Sequence.Mamba.main args
  | "ppo_cartpole" => NN.Examples.Models.RL.PPOCartPole.main args
  | "ppo_gridworld" => NN.Examples.Models.RL.PPOGridWorld.main args
  | "ppo_pong_ram" => NN.Examples.Models.RL.PPOPongRam.main args
  | "dqn_replay" => NN.Examples.Models.RL.DQNReplay.main args
  | "data_csv" =>
      NN.Examples.Data.Loaders.Csv.main args
      pure 0
  | "data_npy" =>
      NN.Examples.Data.Loaders.Npy.main args
      pure 0
  | "data_cifar10" =>
      NN.Examples.Data.Loaders.Cifar10Images.main args
      pure 0
  | "pytorch_roundtrip" =>
      NN.Examples.Interop.PyTorch.Roundtrip.main args
      pure 0
  | "pytorch_export_check" => NN.Examples.Interop.PyTorch.TorchExportCheck.main args
  | "floats_arb_ieee_compare" => NN.Examples.DeepDives.Floats.ArbIEEEExecCompare.main args
  | "float32_semantics" =>
      NN.Examples.DeepDives.Floats.Float32Semantics.main args
      pure 0
  | "numerical_certificate" =>
      NN.Examples.DeepDives.Floats.GraphNumericalCertificate.main args
  | "graphspec" =>
      NN.Examples.DeepDives.GraphSpec.Tutorial.main args
      pure 0
  | "ir_axis_ops" =>
      NN.Examples.DeepDives.IRAxisOps.main args
      pure 0
  | "one_semantic_universe" =>
      NN.Examples.DeepDives.OneSemanticUniverse.main args
      pure 0
  | "torch_ir_pytorch" =>
      NN.Examples.DeepDives.TorchIRPyTorch.main args
      pure 0
  | _ =>
      IO.eprintln s!"Unknown example: {cmd}"
      IO.eprintln ""
      IO.eprintln usage
      pure 1

def main (args : List String) : IO UInt32 := do
  let args := CLI.dropDashDash args
  match args with
  | [] =>
      IO.eprintln NN.Examples.Models.Runner.usage
      pure 1
  | "--help" :: _ | "-h" :: _ =>
      IO.println NN.Examples.Models.Runner.usage
      pure 0
  | _ =>
      let choose := args.contains "--choose" && !(args.contains "--help") && !(args.contains "-h")
      let args := NN.Examples.Models.Runner.stripRunnerFlag "--choose" args
      let args ←
        if choose then
          NN.Examples.Models.Runner.chooseRuntimeArgs args
        else
          pure args
      match NN.Examples.Models.Runner.splitCommandArgs? args with
      | none =>
          IO.eprintln NN.Examples.Models.Runner.usage
          pure 1
      | some (pref, cmd, commandArgs) =>
          if pref.contains "--help" || pref.contains "-h" then
            IO.println NN.Examples.Models.Runner.usage
            pure 0
          else
            runCmd cmd (pref ++ commandArgs)

end NN.Examples.Models.Runner

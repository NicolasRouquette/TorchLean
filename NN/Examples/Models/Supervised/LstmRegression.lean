/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.API.Data
public import NN.API.Models.SimpleSeq
public import NN.Examples.Data.RealPaths

/-!
# LSTM Seasonal Regression / Forecasting

This is the runnable supervised sequence example: an LSTM fits a real-valued forecasting task
and works with the same CPU/CUDA runtime flags as the other model commands.

The default data path uses the UCI Individual Household Electric Power Consumption dataset:
minute-level power readings from one household over almost four years. The preparation script turns
that into hourly one-step forecasting windows:

`past 24 hours -> next 24 shifted-by-one-hour targets`

Prepare the real data once:

```bash
python3 scripts/datasets/download_example_data.py --household-power --household-power-windows 512
```

Recommended runs:
- use `--steps 1` to check that the runtime, data loader, and CUDA path agree on shapes;
- use `--steps 200 --windows 96` for a short training run with before/after forecast reports;
- change `--report-offset` to evaluate a different part of the power curve;
- lower `--lr` if the reported forecast error increases.

```bash
lake exe -K cuda=true torchlean lstm_regression --cuda --steps 1 --windows 1
lake exe -K cuda=true torchlean lstm_regression --cuda --steps 200 --windows 96
```

Dataset citation: Hebrail and Berard, "Individual Household Electric Power Consumption", UCI Machine
Learning Repository, DOI `10.24432/C58K54`, CC BY 4.0.
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.Supervised.LstmRegression

/-- Runner subcommand: `lake exe torchlean lstm_regression ...`. -/
def exeName : String := "torchlean lstm_regression"

/--
Default JSON path for the before/after loss.

Pass `--log PATH` to write somewhere else, or `--log disabled` when you only want terminal output.
-/
def defaultLogJson : System.FilePath := "data/model_zoo/lstm_regression_trainlog.json"

/-- Default root for downloaded real datasets. Override with `--data-dir`. -/
def defaultDataDir : System.FilePath :=
  _root_.NN.Examples.Data.RealPaths.defaultDataDir

/-- One day of hourly samples. Try `48` if you want a two-day context, but expect slower runs. -/
def seqLen : Nat := 24

/-- One scalar feature. If you add calendar/weather features, increase this and update `scalarRow`. -/
def inputSize : Nat := 1

/-- Hidden width for the recurrent state used by this example. -/
def hiddenSize : Nat := 32

/--
Shared recurrent-model configuration.

This keeps the model constructor, input shape, and output shape tied to the same three numbers:
`seqLen`, `inputSize`, and `hiddenSize`. When a shape error shows up, this is the first place to
check.
-/
def cfg : nn.models.SeqRnnHeadConfig :=
  { seqLen := seqLen, inputSize := inputSize, hiddenSize := hiddenSize }

/-- Input shape: one scalar observation at each of `seqLen` timesteps. -/
abbrev σ : Shape :=
  nn.models.seqRnnHeadInShape cfg

/-- Target/prediction shape: one next-step scalar at each of `seqLen` timesteps. -/
abbrev τ : Shape :=
  nn.models.seqRnnHeadOutShape cfg

/--
The actual forecaster.

`nn.models.lstmWithLinearHead cfg` expands to:

`nn.lstm seqLen inputSize hiddenSize`
followed by a time-distributed `nn.linear hiddenSize inputSize`.

So every timestep emits a scalar forecast. We are not using only the final hidden state here; the loss
checks the whole output sequence.
-/
def mkModel : nn.M (nn.Sequential σ τ) :=
  nn.models.lstmWithLinearHead cfg

/-- Data source tags for terminal logs and JSON metadata. -/
def dataTags (xPath yPath : System.FilePath) : Array String :=
  #["data=uci-household-power", s!"x={xPath}", s!"y={yPath}"]

/-- Load prepared UCI household-power windows through the shared `.npy` supervised source. -/
def loadRealSamples (xPath yPath : System.FilePath) (windows : Nat) :
    IO (Except String (Array (API.sample.Supervised Float σ τ))) := do
  let xMeta ← Data.fromNpy xPath
  let actualWindows : Except String Nat ←
    match xMeta with
    | .error e => pure (Except.error e)
    | .ok info =>
        match info.shape with
        | n :: rest =>
            if rest = [seqLen, inputSize] then
              pure (Except.ok n)
            else
              pure (Except.error s!"expected X.npy shape (N,{seqLen},{inputSize}), got {info.shape}")
        | _ => pure (Except.error s!"expected X.npy shape (N,{seqLen},{inputSize}), got {info.shape}")
  match actualWindows with
  | .error e => pure (.error e)
  | .ok n =>
      if windows > n then
        pure (.error s!"requested --windows {windows}, but {xPath} only contains {n} windows")
      else
        let src := Data.SupervisedSource.ofPaths .npy xPath yPath n
          [seqLen, inputSize] [seqLen, inputSize]
        let ds ← Data.SupervisedSource.load (α := Float) src
        pure <| ds.map (fun d => ((Data.toList d).take windows).toArray)

/--
Fallback sample used for defensive array access inside reporting and training helpers.

The command-line parser rejects `--windows 0`; this helper keeps the remaining code total and gives
a clear error if a caller provides an empty sample array directly.
-/
def firstSample? (xs : Array (API.sample.Supervised Float σ τ)) :
    Except String (API.sample.Supervised Float σ τ) :=
  match xs[0]? with
  | some x => .ok x
  | none => .error "no training windows loaded"

/--
Validation MSE over a bounded prefix of loaded windows.

The example reports a fixed-size prefix so evaluation cost is predictable even when the training
window count is large. This is an in-sample diagnostic for the loaded window source; a separate
experiment can pass held-out windows here when it needs an out-of-sample validation number.
-/
def meanLossOnSamples
    (model : nn.Sequential σ τ)
    (m : TorchLean.Module.ScalarModule Float (nn.paramShapes model) [σ, τ])
    (xs : Array (API.sample.Supervised Float σ τ)) : IO Float := do
  let fallback ← Common.orThrow exeName (firstSample? xs)
  let evalCount := Nat.min xs.size 32
  let mut total := 0.0
  for i in [0:evalCount] do
    let loss ← TorchLean.Module.forward (α := Float) m (xs.getD i fallback)
    total := total + Tensor.toScalar loss
  pure (total / Float.ofNat (Nat.max 1 evalCount))

/--
Read `t[row,0]` from a `seqLen × 1` forecast tensor.

The row is clamped so the reporting loop remains valid if `seqLen` is changed without also updating
the number of displayed rows.
-/
def readSeriesAt (t : Tensor Float τ) (i : Nat) : Float :=
  let i : Fin seqLen :=
    ⟨Nat.min i (seqLen - 1),
      Nat.lt_of_le_of_lt (Nat.min_le_right i (seqLen - 1)) (by decide)⟩
  match t with
  | Tensor.dim rows =>
      match rows i with
      | Tensor.dim cols =>
          match cols ⟨0, by decide⟩ with
          | Tensor.scalar x => x

/--
Print the first few predicted/target pairs for one dataset offset.

The report is tied to a deterministic offset so before/after comparisons refer to the same window.
Try `--report-offset 0`, `--report-offset 96`, or `--report-offset 144` to inspect different parts
of the seasonal curve.
-/
def printForecastReport
    (opts : Runtime.Autograd.Torch.Options)
    (model : nn.Sequential σ τ)
    (params : TorchLean.ParamList Float (nn.paramShapes model))
    (samples : Array (API.sample.Supervised Float σ τ))
    (offset : Nat) : IO Unit := do
  let fallback ← Common.orThrow exeName (firstSample? samples)
  let sample := samples.getD (offset % Nat.max 1 samples.size) fallback
  let pred ← nn.eval1NoGrad (α := Float) opts model params (NN.API.sample.x sample)
  let target := NN.API.sample.y sample
  IO.println s!"  report_index={offset % Nat.max 1 samples.size}"
  for i in [0:Nat.min seqLen 8] do
    IO.println s!"    t+{i+1}: pred={readSeriesAt pred i} target={readSeriesAt target i}"

/-- Example-specific training options after the shared runtime parser has handled CPU/CUDA flags. -/
structure TrainOptions where
  /-- Optimizer steps. -/
  steps : Nat
  /-- Number of deterministic windows to cycle through. More windows mean more seasonal coverage. -/
  windows : Nat
  /-- Adam learning rate. -/
  lr : Float
  /-- Start offset for the printed before/after forecast report. This does not affect training. -/
  reportOffset : Nat
  /-- Prepared UCI household-power input windows. Override with `--x`. -/
  xPath : System.FilePath
  /-- Prepared UCI household-power target windows. Override with `--y`. -/
  yPath : System.FilePath
  /-- Logging destination: JSON path, stdout-style destination, or disabled. -/
  log : _root_.Runtime.Training.LogDestination
  /-- Concrete default path when `log` is path-backed. -/
  logPath : System.FilePath
  /-- CUDA allocator telemetry cadence. `0` means quiet unless the shared default enables it. -/
  cudaMemWatch : Nat := 0
deriving Repr

/--
Parse example-specific flags.

Runtime flags such as `--cpu`, `--cuda`, `--dtype`, and `--backend` are handled by
`Common.runFloat`; this parser handles data, forecasting, and logging knobs.
-/
def parseTrainOptions (args : List String) : Except String (TrainOptions × List String) := do
  let (train, args) ← Common.parseLoggedTrainFlags exeName args defaultLogJson 100
  let (dataDir, args) ← _root_.NN.Examples.Data.RealPaths.takeDataDir args
  let (windows?, args) ← CLI.takeNatFlagOnce args "windows"
  let (lr?, args) ← CLI.takeFloatFlagOnce args "lr"
  let (reportOffset?, args) ← CLI.takeNatFlagOnce args "report-offset"
  let (x?, args) ← CLI.takePathFlagOnce args "x"
  let (y?, args) ← CLI.takePathFlagOnce args "y"
  let windows := windows?.getD 512
  if windows = 0 then
    throw s!"{exeName}: --windows must be > 0"
  pure ({ steps := train.steps
          windows := windows
          lr := lr?.getD 0.01
          reportOffset := reportOffset?.getD 96
          xPath := x?.getD (_root_.NN.Examples.Data.RealPaths.householdPowerX dataDir)
          yPath := y?.getD (_root_.NN.Examples.Data.RealPaths.householdPowerY dataDir)
          log := train.log
          logPath := train.logPath
          cudaMemWatch := train.cudaMemWatch }, args)

/--
Train the LSTM forecaster and return `(lossBefore, lossAfter)`.

The training recipe is direct:

1. construct the model under `nn.withModel`;
2. wrap it as a scalar MSE module;
3. load UCI household-power windows;
4. print a forecast report before training;
5. run Adam over the loaded windows; and
6. print the same forecast report after training.

Because the data is deterministic, repeated runs are easy to compare: the loss should drop and the
reported predictions should move toward the target values.
-/
def trainForecast (opts : Runtime.Autograd.Torch.Options) (train : TrainOptions) :
    IO (Float × Float) := do
  nn.withModel mkModel fun model => do
    let xs ← Common.orThrow exeName =<< loadRealSamples train.xPath train.yPath train.windows
    let modDef := nn.mseScalarModuleDef model
    let m ← TorchLean.Module.instantiateWithOptions (α := Float) modDef id opts
    let L0 ← meanLossOnSamples model m xs
    IO.println "  before training forecast report:"
    printForecastReport opts model m.trainer.params xs train.reportOffset

    let opt := TorchLean.Optim.adam (α := Float)
      (paramShapes := nn.paramShapes model)
      (lr := train.lr)
      (beta1 := 0.9)
      (beta2 := 0.999)
      (epsilon := 1e-8)
    let optH ← TorchLean.Optim.handle (α := Float) m opt
    let fallback ← Common.orThrow exeName (firstSample? xs)
    let cudaMemWatch := Common.effectiveCudaMemWatch opts train.steps train.cudaMemWatch
    let mut memWatch? ← Common.reportCudaMemWatch opts cudaMemWatch train.steps 0 none
    for step in [0:train.steps] do
      optH.step (xs.getD (step % Nat.max 1 xs.size) fallback)
      memWatch? ← Common.reportCudaMemWatch opts cudaMemWatch train.steps (step + 1) memWatch?

    let L1 ← meanLossOnSamples model m xs
    IO.println "  after training forecast report:"
    printForecastReport opts model m.trainer.params xs train.reportOffset
    IO.println s!"  steps={train.steps} windows={xs.size} lr={train.lr} loss0={L0} loss1={L1}"
    pure (L0, L1)

/-- Executable entrypoint for CPU/CUDA Float training. -/
def main (args : List String) : IO UInt32 := do
  Common.runFloat exeName args
    (banner := fun opts =>
      s!"{exeName}: LSTM time-series regression (device={if opts.useGpu then "cuda" else "cpu"})")
    (k := fun opts rest => do
      let (train, rest) ← Common.orThrow exeName <| parseTrainOptions rest
      Common.orThrow exeName <| CLI.requireNoArgs rest
      let (L0, L1) ← trainForecast opts train
      Common.writeBeforeAfterLossLogTo train.log "LSTM seasonal regression" train.steps L0 L1
        (#[s!"device={if opts.useGpu then "cuda" else "cpu"}", s!"windows={train.windows}",
          s!"lr={train.lr}", s!"report_index={train.reportOffset}",
          s!"cuda_mem_watch={Common.effectiveCudaMemWatch opts train.steps train.cudaMemWatch}",
          "task=next-step household power forecasting"] ++ dataTags train.xPath train.yPath))

end NN.Examples.Models.Supervised.LstmRegression

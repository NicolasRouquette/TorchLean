/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Examples.Data.RealPaths
public import NN.Examples.Models.Common.RealData

/-!
# LSTM Seasonal Regression / Forecasting

This is the runnable supervised sequence example: an LSTM trains on a real-valued forecasting task
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
lake -R -K cuda=true exe torchlean lstm_regression --device cuda --steps 1 --windows 1
lake -R -K cuda=true exe torchlean lstm_regression --device cuda --steps 200 --windows 96
```

Dataset citation: Hebrail and Berard, "Individual Household Electric Power Consumption", UCI Machine
Learning Repository, DOI `10.24432/C58K54`, CC BY 4.0.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Supervised.LstmRegression

/-- Runner subcommand: `lake exe torchlean lstm_regression ...`. -/
def exeName : String := "torchlean lstm_regression"

/--
Default JSON path for the before/after loss.

Pass `--log PATH` to write somewhere else, or `--log disabled` when you only want terminal output.
-/
def defaultLogJson : System.FilePath := ModelZoo.trainLogPath "lstm_regression"

/-- Default root for downloaded real datasets. Override with `--data-dir`. -/
def defaultDataDir : System.FilePath :=
  _root_.NN.Examples.Data.RealPaths.defaultDataDir

/-- Prepared household-power windows contain one day of hourly samples. -/
def rawSeqLen : Nat := 24

/-- Short prefix used by the runnable recurrent example. -/
def seqLen : Nat := 2

/-- One scalar feature. If you add calendar/weather features, increase this and update `scalarRow`. -/
def inputSize : Nat := 1

/-- Hidden width for the recurrent state used by this runnable example. -/
def hiddenSize : Nat := 2

/--
Shared recurrent-model configuration.

The model constructor, input shape, and output shape all read from this value. The single output
feature is the next power-consumption prediction at each time step.
-/
def cfg : nn.models.RecurrentConfig :=
  { seqLen := seqLen
    inputSize := inputSize
    hiddenSize := hiddenSize
    outputSize := 1 }

/-- Input shape: one scalar observation at each of `seqLen` timesteps. -/
abbrev σ : Shape :=
  cfg.inputShape

/-- Target/prediction shape: one next-step scalar at each of `seqLen` timesteps. -/
abbrev τ : Shape :=
  cfg.outputShape

/-- Raw input shape stored by the prepared household-power `.npy` files. -/
abbrev rawσ : Shape :=
  .dim rawSeqLen (.dim inputSize .scalar)

/-- Raw target shape stored by the prepared household-power `.npy` files. -/
abbrev rawτ : Shape :=
  .dim rawSeqLen (.dim inputSize .scalar)

/--
The actual forecaster.

`nn.models.lstmWithLinearHead cfg` expands to:

`nn.LSTM seqLen inputSize hiddenSize`
followed by a time-distributed `nn.linear hiddenSize 1`.

So every timestep emits a scalar forecast. We are not using only the final hidden state here; the loss
checks the whole output sequence.
-/
def mkModel : nn.Builder (nn.Sequential σ τ) :=
  nn.models.lstmWithLinearHead cfg

/-- Data source tags for terminal logs and JSON metadata. -/
def dataTags (xPath yPath : System.FilePath) : Array String :=
  #["data=uci-household-power", s!"x={xPath}", s!"y={yPath}"]

/-- Validate the prepared input file and return its available window count. -/
def availableWindows (xPath : System.FilePath) :
    IO (Except String Nat) :=
  Data.availableNpyRows xPath [rawSeqLen, inputSize] s!"X.npy shape (N,{rawSeqLen},{inputSize})"

/-- Load the Float version once for reporting probes and short training. -/
def loadReportSamples (xPath yPath : System.FilePath) (windows : Nat) :
    IO (Except String (Array (Sample.Supervised Float rawσ rawτ))) := do
  Data.loadSupervisedNpy xPath yPath windows
    [rawSeqLen, inputSize] [rawSeqLen, inputSize]

/-- Keep the first `seqLen` rows of a prepared `rawSeqLen × 1` tensor. -/
def takePrefix (t : Tensor Float rawσ) : Tensor Float σ :=
  match t with
  | Spec.Tensor.dim rows =>
      Spec.Tensor.dim fun i =>
        rows ⟨i.val, Nat.lt_of_lt_of_le i.isLt (by decide : seqLen ≤ rawSeqLen)⟩

/-- Convert one real 24-hour prepared window into the tiny recurrent training sample. -/
def prefixSample (sample : Sample.Supervised Float rawσ rawτ) : Sample.Supervised Float σ τ :=
  Sample.mk (takePrefix (Sample.x sample)) (takePrefix (Sample.y sample))

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
  | Spec.Tensor.dim rows =>
      match rows i with
      | Spec.Tensor.dim cols =>
          match cols ⟨0, by decide⟩ with
          | Spec.Tensor.scalar x => x

/--
Render the first few target values for one forecast window.
-/
def targetSummary (sample : Sample.Supervised Float σ τ) : String :=
  String.intercalate ", " <|
    (List.range (Nat.min seqLen 8)).map (fun i =>
      s!"t+{i+1}={readSeriesAt (Sample.y sample) i}")

/-- Public trainer probe for a deterministic forecast window. -/
def probeOfSample (sample : Sample.Supervised Float σ τ) (idx : Nat) : Trainer.Probe σ :=
  Trainer.Probe.ofFloatTensor
    "forecast"
    (Sample.x sample)
    (inputText := s!"report_index={idx}")
    (expected := some (targetSummary sample))

/-
LSTM regression uses the same public training recipe as the other supervised examples:

1. name the model;
2. name the dataset;
3. choose persistent runtime settings;
4. choose per-training options; and
5. call the configured public trainer session.
-/
def trainForecast (opts : Options) (train : RealData.HouseholdPowerModelTrainFlags) := do
  Data.requirePairedFiles exeName
    "household-power inputs" train.xPath
    "household-power targets" train.yPath
    RealData.missingHouseholdPowerHint
  let available ← ModelZoo.orThrow exeName =<< availableWindows train.xPath
  if train.windows > available then
    throw <| IO.userError
      s!"{exeName}: requested --windows {train.windows}, but {train.xPath} only contains {available} windows"
  let rawSamples ← ModelZoo.orThrow exeName =<< loadReportSamples train.xPath train.yPath train.windows
  let samples := rawSamples.map prefixSample
  let fallback ← ModelZoo.orThrow exeName <|
    match samples[0]? with
    | some sample => .ok sample
    | none => .error "no training windows loaded"
  let probeIdx := train.reportOffset % Nat.max 1 samples.size
  let probe := probeOfSample (samples.getD probeIdx fallback) probeIdx
  let trainer :=
    Trainer.new mkModel <|
      Trainer.Config.fromRunConfig
        (Trainer.RunConfig.ofRuntimeOptions opts { optimizer := optim.adam { lr := train.lr } })
        .regression
        (seed := train.seed)
  trainer.train
    (Data.floatSampleArray samples)
    (CLI.Training.OptimizerOptions.toTrainerOptions train.toOptimizerOptions
      (title := "LSTM seasonal regression")
      (notes := ModelZoo.ForecastWindowDataFlags.trainLogNotes train.toForecastWindowDataFlags ++
        #[s!"lr={train.lr}", s!"cuda_mem_watch={train.cudaMemWatch}",
          "task=next-step household power forecasting"] ++ dataTags train.xPath train.yPath))
    [probe]

/-- Executable entrypoint for CPU/CUDA Float training. -/
def main (args : List String) : IO UInt32 :=
  TrainCommand.forecastWindow exeName args
    (fun rest =>
      RealData.HouseholdPowerModelTrainFlags.parse exeName rest defaultLogJson 100 0.01 512 96)
    (ModelZoo.bannerWithDevice exeName "LSTM time-series regression")
    trainForecast

end NN.Examples.Models.Supervised.LstmRegression

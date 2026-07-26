/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Native TorchLean 1D FNO on the Burgers operator:

  python3 NN/Examples/Data/prepare_fno1d_burgers.py --download --grid 32 --ntrain 128 --ntest 32
  lake -R -K cuda=true build
  lake -R -K cuda=true exe torchlean fno1d_burgers --device cuda --steps 700 --lr 0.003 \
    --plot-csv data/real/fno/predictions.csv --log data/real/fno/trainlog.json
  python3 NN/Examples/Data/plot_fno1d_burgers.py --csv data/real/fno/predictions.csv
-/

module

public import NN.API
public import NN.Examples.ModelZoo
public import NN.Runtime.Autograd.Engine.Cuda.Fno1dRfftFused
public import NN.Spec.Layers.Loss

/-!
# Native TorchLean FNO1D Burgers

Read this after the basic CNN/MLP examples if you want the operator-learning path. The Python
scripts do the two jobs Lean should not own here: download and reshape the public
`burgers_data_R10.mat` file, then plot the prediction CSV. The model, loss, optimizer, and training
loop stay in TorchLean.

This executable uses real-split Fourier arithmetic because the Burgers data are real-valued. The
portable path evaluates the dense multidimensional DFT with separate real and imaginary tensors.
On CUDA, the fused `spectralConv1dRfft` autograd primitive uses the same representation for its
weights and executes the transforms through cuFFT.

The training task follows the standard FNO Burgers setup: learn the operator
$u_0(x)\mapsto u(x,T)$ on a fixed periodic grid. The default grid and row counts are modest enough for a
local run while still exercising the real operator-learning path. Larger runs can raise `--steps`,
export more rows, and bump the constants below.

References for the dataset/training convention:
- Li et al., “Fourier Neural Operator for Parametric Partial Differential Equations”, 2020/2021.
- MathWorks’ Burgers FNO example and the `burgers_data_R10.mat` public dataset.
- SciML FNO tutorials using fields `a` for initial conditions and `u` for final solutions.
-/

@[expose] public section

open Spec Tensor
open TorchLean

namespace NN.Examples.Models.Operators.Fno1dBurgers

/-- CLI subcommand name used in terminal banners and errors. -/
def exeName : String := "torchlean fno1d_burgers"

/-- Spatial grid resolution used by the prepared Burgers `.npy` slices. -/
def grid : Nat := 32

/-- Channel width inside the compact FNO block. -/
def width : Nat := 8

/-- Number of Fourier modes retained on each side of the real FFT spectrum. -/
def modes : Nat := 8

/-- Number of spectral blocks used by the compact training run. -/
def blocks : Nat := 1

/-- Default number of training rows expected from the preparation script. -/
def defaultTrainRows : Nat := 128

/-- Default number of held-out rows expected from the preparation script. -/
def defaultTestRows : Nat := 32

/-- Spec.Shape-level FNO configuration shared by the constructor and sample loaders. -/
def modelCfg : nn.models.FNOConfig 1 :=
  { spatial := Vector.replicate 1 grid
    modes := Vector.replicate 1 modes
    spatialNonzero := by intro i; fin_cases i; decide
    modesFit := by intro i; fin_cases i; decide
    width := width
    widthNonzero := by decide
    blocks := blocks
    seed := 0 }

/-- Model input shape: one sampled initial condition on the fixed grid. -/
abbrev σ : Spec.Shape := .dim grid .scalar

/-- Model output shape: one predicted terminal solution on the same grid. -/
abbrev τ : Spec.Shape := .dim grid .scalar

/-- Directory where the preparation script writes Burgers tensors by default. -/
def defaultDir : System.FilePath := "data/real/fno"

/-- Default training input tensor path. -/
def trainXPath : System.FilePath := defaultDir / "burgers_train_X.npy"

/-- Default training target tensor path. -/
def trainYPath : System.FilePath := defaultDir / "burgers_train_y.npy"

/-- Default held-out input tensor path. -/
def testXPath : System.FilePath := defaultDir / "burgers_test_X.npy"

/-- Default held-out target tensor path. -/
def testYPath : System.FilePath := defaultDir / "burgers_test_y.npy"

/-- Default CSV path for the prediction-vs-target plot script. -/
def defaultPlotCsv : System.FilePath := defaultDir / "predictions.csv"

/-- Default JSON training-log path. -/
def defaultLogJson : System.FilePath := defaultDir / "trainlog.json"

/-- User-facing hint printed when the prepared Burgers tensors are missing. -/
def missingDataHint : String :=
  "Prepare the public Burgers FNO dataset with:\n" ++
  "  python3 NN/Examples/Data/prepare_fno1d_burgers.py --download --grid 32 --ntrain 128 --ntest 32\n" ++
  "The .mat file is large; use --mat PATH if you already downloaded burgers_data_R10.mat."

/--
FNO Burgers command-line options: training flags, data paths, and artifact paths.

Seeded optimizer/log flags come from `ModelZoo`, the Burgers tensor paths use
`ModelZoo.PairedNpyEvalFlags`, and the plot path uses `ModelZoo.CsvArtifactFlags`.
-/
structure BurgersOptions extends
    ModelZoo.SeededTrainFlags,
    ModelZoo.PairedNpyEvalFlags,
    ModelZoo.CsvArtifactFlags where
deriving Repr

/-- All required dataset files for this run. -/
def dataPaths (cfg : BurgersOptions) : List System.FilePath :=
  [cfg.trainX, cfg.trainY, cfg.testX, cfg.testY]

namespace BurgersOptions

/-- Parse the FNO Burgers command-line options. -/
def parse (args : List String) :
    Except String (BurgersOptions × List String) := do
  let (trainBase, args) ← ModelZoo.parseSeededTrainFlags exeName args defaultLogJson 50 5e-3
  let (data, args) ← ModelZoo.PairedNpyEvalFlags.parse args
    trainXPath trainYPath testXPath testYPath defaultTrainRows defaultTestRows
  let (artifact, args) ← ModelZoo.CsvArtifactFlags.parse args defaultPlotCsv
  let cfg : BurgersOptions :=
    { toSeededTrainFlags := trainBase
      toPairedNpyEvalFlags := data
      toCsvArtifactFlags := artifact }
  pure (cfg, args)

/-- Effective CUDA-memory-watch cadence for this run. -/
def effectiveCudaMemWatch (cfg : BurgersOptions) (opts : Options) : Nat :=
  ModelZoo.effectiveCudaMemWatch opts cfg.steps cfg.cudaMemWatch

/-- TrainLog note fields for the fused CUDA execution path. -/
def logNotes (cfg : BurgersOptions) (spectralPath : String) (device : String) : Array String :=
  #[
    s!"model=fno",
    s!"spectral_path={spectralPath}",
    s!"device={device}",
    s!"grid={grid}",
    s!"width={width}",
    s!"modes={modes}",
    s!"blocks={blocks}",
    s!"steps={cfg.steps}",
    s!"lr={cfg.lr}"
  ] ++ ModelZoo.PairedNpyEvalFlags.trainLogNotes cfg.toPairedNpyEvalFlags

end BurgersOptions

def mkModel : nn.M (nn.Sequential σ τ) :=
  by
    simpa [σ, τ, nn.models.fnoInShape, nn.models.fnoOutShape, modelCfg,
      Vector.replicate, Spec.Shape.ofList] using nn.models.fno modelCfg

/-- Load one fixed-grid Burgers split as supervised TorchLean samples. -/
def loadDataset
    (xPath yPath : System.FilePath) (n : Nat) :
    IO (Data.Dataset (SupervisedSample Float σ τ)) := do
  let samples ← ModelZoo.orThrow exeName =<<
    Data.loadSupervisedNpyFloatSamples xPath yPath n [grid] [grid]
  pure <| Data.fromList samples.toList

/-- Write one FNO prediction row to CSV for the companion plotting script. -/
def writePredictionProbe (plotCsv : System.FilePath)
    (x target prediction : Tensor Float σ) : IO Unit := do
  Data.writeVectorPredictionCsv plotCsv x target prediction
  IO.println s!"  wrote prediction CSV: {plotCsv}"
  IO.println s!"  plot with: python3 NN/Examples/Data/plot_fno1d_burgers.py --csv {plotCsv}"

def metricHistory : Training.MetricHistory :=
  Training.MetricHistory.empty #[
    ("train_mse", "#4e79a7"),
    ("test_mse", "#f28e2b")
  ]

/-- Persist the train/test MSE history with model/data metadata attached. -/
def writeMetricLog (dest : Training.LogDestination) (hist : Training.MetricHistory)
    (cfg : BurgersOptions) (spectralPath device : String) : IO Unit := do
  ModelZoo.writeMetricHistoryLog dest "FNO1D Burgers (TorchLean)" hist
    (cfg.logNotes spectralPath device)

/-- Loaded train/test splits before evaluation prefixes and cycling streams are derived. -/
structure LoadedData where
  /-- Training split as supervised samples. -/
  train : Data.Dataset (SupervisedSample Float σ τ)
  /-- Held-out split as supervised samples. -/
  test : Data.Dataset (SupervisedSample Float σ τ)

/-- Validate paths and load both Burgers splits. -/
def loadData (cfg : BurgersOptions) :
    IO LoadedData := do
  Data.requireFiles exeName (dataPaths cfg) missingDataHint
  let train ← loadDataset cfg.trainX cfg.trainY cfg.trainRows
  let test ← loadDataset cfg.testX cfg.testY cfg.testRows
  pure { train, test }

/-- Deterministic evaluation prefixes and cycling stream derived from the loaded train/test sets. -/
structure EvalData where
  trainDatasetSamples : List (SupervisedSample Float σ τ)
  testDatasetSamples : List (SupervisedSample Float σ τ)
  trainSamples : List (Tensor Float σ × Tensor Float τ)
  testSamples : List (Tensor Float σ × Tensor Float τ)
  reportTrainDatasetSamples : List (SupervisedSample Float σ τ)
  reportTestDatasetSamples : List (SupervisedSample Float σ τ)
  reportTrainSamples : List (Tensor Float σ × Tensor Float τ)
  reportTestSamples : List (Tensor Float σ × Tensor Float τ)
  trainCycle : Nat → SupervisedSample Float σ τ

/--
Convert loaded Burgers datasets into the common runtime/evaluation view used by both execution
paths.

The fused CUDA path:
- evaluate on fixed deterministic prefixes,
- train by cycling through the finite dataset with `seed + step`, and
- emit the same train/test MSE metric history.
-/
def mkEvalData (cfg : BurgersOptions) (data : LoadedData) : IO EvalData := do
  let trainDatasetSamples := Data.toList data.train
  let testDatasetSamples := Data.toList data.test
  let trainSamples := trainDatasetSamples.map Sample.toPair
  let testSamples := testDatasetSamples.map Sample.toPair
  let trainCycle ← ModelZoo.orThrow exeName <|
    Data.cycleListOrError trainDatasetSamples "empty Burgers training dataset"
  pure
    { trainDatasetSamples := trainDatasetSamples
      testDatasetSamples := testDatasetSamples
      trainSamples := trainSamples
      testSamples := testSamples
      reportTrainDatasetSamples := trainDatasetSamples.take cfg.evalRows
      reportTestDatasetSamples := testDatasetSamples.take cfg.evalRows
      reportTrainSamples := trainSamples.take cfg.evalRows
      reportTestSamples := testSamples.take cfg.evalRows
      trainCycle := trainCycle }

/-- Push one train/test MSE point into the metric history and print the tagged report line. -/
def pushLossPoint
    (hist : Training.MetricHistory)
    (step : Nat)
    (tag : String)
    (trainLoss testLoss : Float) : IO Training.MetricHistory := do
  IO.println s!"  {tag}: train_mse={trainLoss} test_mse={testLoss}"
  pure <| hist.push step #[trainLoss, testLoss]

namespace FusedCuda

/-- Fused CUDA parameter packet for the real-FFT FNO kernel. -/
abbrev Param := _root_.Runtime.Autograd.Cuda.Fno1dRfftFused.Param

/-- Mean MSE over a finite evaluation prefix using the fused CUDA FNO implementation. -/
def meanLoss (ps : Array Param) (samples : List (Tensor Float σ × Tensor Float τ)) :
    IO Float := do
  let result ←
    _root_.Runtime.Autograd.Cuda.Fno1dRfftFused.meanLoss
      (grid := grid) (width := width) (modes := modes) (blocks := blocks) ps samples
  _root_.Runtime.Autograd.okOrThrow result

/-- Train/test MSE pair for the current fused CUDA parameters. -/
def evalLosses (trainEval testEval : List (Tensor Float σ × Tensor Float τ))
    (ps : Array Param) : IO (Float × Float) := do
  let trainLoss ← meanLoss ps trainEval
  let testLoss ← meanLoss ps testEval
  pure (trainLoss, testLoss)

/-- Append one fused-CUDA evaluation point to the metric history. -/
def recordEval (trainEval testEval : List (Tensor Float σ × Tensor Float τ))
    (hist : Training.MetricHistory) (step : Nat) (ps : Array Param) (tag : String) :
    IO Training.MetricHistory := do
  let (trainLoss, testLoss) ← evalLosses trainEval testEval ps
  pushLossPoint hist step tag trainLoss testLoss

/-- Predict one Burgers terminal field through the fused CUDA spectral path. -/
def predict (ps : Array Param) (x : Tensor Float σ) : IO (Tensor Float τ) := do
  let result ←
    _root_.Runtime.Autograd.Cuda.Fno1dRfftFused.forward
      (grid := grid) (width := width) (modes := modes) (blocks := blocks) ps x none
  let fw ← _root_.Runtime.Autograd.okOrThrow result
  let predictionResult ←
    _root_.Runtime.Autograd.Cuda.Fno1dRfftFused.predFromTape (grid := grid) fw.tape fw.predId
  fw.dispose
  _root_.Runtime.Autograd.okOrThrow predictionResult

/-- One fused CUDA Adam update on a single Burgers sample. -/
def trainStep (lr : Float)
    (ps : Array Param)
    (adamSt : _root_.Runtime.Autograd.Cuda.Fno1dRfftFused.AdamState)
    (sample : Tensor Float σ × Tensor Float τ) :
    IO (Array Param × _root_.Runtime.Autograd.Cuda.Fno1dRfftFused.AdamState) := do
  let (x, y) := sample
  let forwardResult ←
    _root_.Runtime.Autograd.Cuda.Fno1dRfftFused.forward
      (grid := grid) (width := width) (modes := modes) (blocks := blocks) ps x (some y)
  let fw ← _root_.Runtime.Autograd.okOrThrow forwardResult
  let result ←
    _root_.Runtime.Autograd.Cuda.Fno1dRfftFused.updateParamsAdam ps fw lr adamSt
  _root_.Runtime.Autograd.okOrThrow result

/-- Run the fused cuFFT/RFFT training path and emit its training and prediction artifacts. -/
def run (cfg : BurgersOptions) : IO Unit := do
  let data ← loadData cfg
  let eval := ← mkEvalData cfg data
  let mut ps :=
    _root_.Runtime.Autograd.Cuda.Fno1dRfftFused.initParams
      (grid := grid) (width := width) (modes := modes) (blocks := blocks) cfg.seed
  let mut adamSt : _root_.Runtime.Autograd.Cuda.Fno1dRfftFused.AdamState := {}
  let mut hist ← recordEval eval.reportTrainSamples eval.reportTestSamples metricHistory 0 ps "before"
  let cudaOpts : _root_.Runtime.Autograd.Torch.Options :=
    { executionProfile := NN.Backend.BackendProfile.checkedCuda }
  let cudaMemWatch := cfg.effectiveCudaMemWatch cudaOpts
  let mut memWatch? ← ModelZoo.reportCudaMemWatch cudaOpts cudaMemWatch cfg.steps 0 none
  let progressEvery : Nat := Nat.max 1 (cfg.steps / 10)
  for step in [0:cfg.steps] do
    let s := eval.trainCycle (cfg.seed + step)
    let sample := Sample.toPair s
    let (ps', adamSt') ← trainStep cfg.lr ps adamSt sample
    ps := ps'
    adamSt := adamSt'
    memWatch? ← ModelZoo.reportCudaMemWatch cudaOpts cudaMemWatch cfg.steps (step + 1) memWatch?
    if ModelZoo.shouldLogStep progressEvery (step + 1) then
      hist ← recordEval eval.reportTrainSamples eval.reportTestSamples hist (step + 1) ps s!"step {step + 1}"
  hist ← recordEval eval.reportTrainSamples eval.reportTestSamples hist cfg.steps ps "after"
  match eval.testSamples with
  | [] => pure ()
  | sample :: _ =>
      let (x, y) := sample
      let yhat ← predict ps x
      writePredictionProbe cfg.plotCsv x y yhat
  writeMetricLog cfg.log hist cfg "fused cuFFT RFFT autograd op" "cuda"

end FusedCuda

def runPortableDense
    (opts : Options)
    (cfg : BurgersOptions) :
    IO Unit := do
  -- Load the train/test arrays once, then keep the runtime loop purely over typed samples.
  let data ← loadData cfg
  let eval := ← mkEvalData cfg data
  let trainer :=
    Trainer.new mkModel <|
      Trainer.Config.fromRunConfig
        (Trainer.runConfig opts { optimizer := optim.adam { lr := cfg.lr } })
        .regression
        (seed := cfg.seed)
  trainer.printInfo
  let histRef ← IO.mkRef metricHistory
  let meanPredMse (predict : Tensor Float σ → IO (Tensor Float τ))
      (samples : List (Tensor Float σ × Tensor Float τ)) : IO Float := do
    match samples with
    | [] => pure 0.0
    | xs =>
        let mut total : Float := 0.0
        for (x, y) in xs do
          let yhat ← predict x
          total := total + _root_.Spec.mseSpec yhat y
        pure (total / Float.ofNat xs.length)
  let recordEval
      (step : Nat) (tag : String)
      (predict : Tensor Float σ → IO (Tensor Float τ)) : IO Unit := do
    let trainLoss ← meanPredMse predict eval.reportTrainSamples
    let testLoss ← meanPredMse predict eval.reportTestSamples
    let hist ← histRef.get
    histRef.set (← pushLossPoint hist step tag trainLoss testLoss)
  let evalSample ←
    match eval.reportTrainDatasetSamples with
    | sample :: _ => pure sample
    | [] => throw <| IO.userError s!"{exeName}: empty Burgers training evaluation prefix"
  let progressEvery : Nat := Nat.max 1 (cfg.steps / 10)
  let trained ← trainer.trainStreamFloat opts
    (fun step => eval.trainCycle (cfg.seed + step))
    evalSample
    { steps := cfg.steps, log := .disabled }
    (curveEvery := progressEvery)
    (cudaMemWatch := cfg.cudaMemWatch)
    (onEval := recordEval)
  -- Save one prediction slice so the example reports both scalar loss curves and a field-level
  -- Burgers trajectory comparison.
  match eval.testSamples with
  | [] => pure ()
  | sample :: _ =>
      let (x, y) := sample
      let yhat ← trained.predict x
      writePredictionProbe cfg.plotCsv x y yhat
  let hist ← histRef.get
  writeMetricLog cfg.log hist cfg "portable dense DFT ops" (ModelZoo.deviceName opts)


def logRunHeader (opts : Options) (cfg : BurgersOptions) : IO Unit := do
  IO.println s!"{exeName}: native real-split FNO1D Burgers"
  let backendName :=
    match opts.backend with
    | .eager => "eager"
    | .compiled => "compiled"
  IO.println s!"  {ModelZoo.deviceNote opts} backend={backendName}"
  IO.println s!"  grid={grid} width={width} modes={modes} blocks={blocks}"
  IO.println s!"  rows train={cfg.trainRows} test={cfg.testRows} eval_prefix={cfg.evalRows}"
  IO.println s!"  cuda_mem_watch={cfg.effectiveCudaMemWatch opts}"
  IO.println s!"  train={cfg.trainX} / {cfg.trainY}"
  IO.println s!"  test ={cfg.testX} / {cfg.testY}"
  IO.println s!"  log  ={cfg.logPath}"

def main (args : List String) : IO UInt32 := do
  Runtime.runFloat exeName args
    (banner := ModelZoo.bannerWithDevice exeName "native FNO1D Burgers")
    (k := fun opts rest => do
      let (cfg, rest) ← ModelZoo.orThrow exeName <| BurgersOptions.parse rest
      CLI.requireNoArgs exeName rest
      logRunHeader opts cfg
      if opts.usesCuda then
        IO.println "  spectral path=fused cuFFT RFFT autograd op"
        if opts.backend != .eager then
          IO.println "  note: fused CUDA path uses the eager CUDA tape (ignoring --backend compiled)"
        FusedCuda.run cfg
      else
        IO.println "  spectral path=portable dense multidimensional DFT"
        runPortableDense opts cfg)

end NN.Examples.Models.Operators.Fno1dBurgers

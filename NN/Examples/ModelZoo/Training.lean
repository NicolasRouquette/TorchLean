/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Runtime
public import NN.API.Tensor
public import NN.API.TensorPack
public import NN.API.Trainer.FixedSample
public import NN.API.Trainer.Reporting
public import NN.Examples.ModelZoo.Command

/-!
# Model-Zoo Training Support

Logging, runtime, and fixed-sample training helpers used by the runnable model examples.
-/

@[expose] public section

namespace NN.Examples.ModelZoo

open TorchLean

/-- Resolve the CUDA allocator-reporting cadence for a model-zoo run. -/
def effectiveCudaMemWatch (opts : Options) (steps requested : Nat) : Nat :=
  TorchLean.Trainer.Manual.effectiveCudaMemWatch opts steps requested

/-- Training-log note recording the selected CUDA allocator-reporting cadence. -/
def cudaMemWatchNote (opts : Options) (steps requested : Nat) : String :=
  s!"cuda_mem_watch={effectiveCudaMemWatch opts steps requested}"

/-- Sample and report CUDA allocator state when the selected cadence is reached. -/
def reportCudaMemWatch
    (opts : Options)
    (watchEvery totalSteps completed : Nat)
    (state? : Option TorchLean.Trainer.Manual.CudaMemWatchState) :
    IO (Option TorchLean.Trainer.Manual.CudaMemWatchState) :=
  TorchLean.Trainer.Manual.reportCudaMemWatch opts watchEvery totalSteps completed state?

/-- Return whether a completed model-zoo step should emit a periodic report. -/
def shouldLogStep (every completed : Nat) : Bool :=
  Training.shouldReport every completed

/-- Print the first and last losses recorded by a model-zoo training curve. -/
def printCurveLossSummary (steps : Nat) (curve : Training.Curve) : IO Unit :=
  Training.Curve.printLossSummary curve steps

/-!
Shared CLI and logging names for the built-in model-zoo examples.

These are public so examples can stay short and readable. They live under `ModelZoo`; ordinary
library code should usually use `Trainer`, `Data`, and `optim` directly.
-/

/-- Standard location for a model-example training log under `data/model_zoo`. -/
def trainLogPath (stem : String) : System.FilePath :=
  System.FilePath.mk s!"data/model_zoo/{stem}_trainlog.json"

/-- Lift a parser result into `IO`, prefixing failures with the executable name. -/
def orThrow {α : Type} (exeName : String) (result : Except String α) : IO α :=
  CLI.orThrow exeName result

/-- Runtime device label used by example banners and notes. -/
def deviceName (opts : Options) : String :=
  opts.deviceName

/-- `device=...` note string used by example logs. -/
def deviceNote (opts : Options) : String :=
  s!"device={deviceName opts}"

/-- Model-zoo banner with the executable name, a short description, and the selected device. -/
def bannerWithDevice (exeName desc : String) (opts : Options) : String :=
  s!"{exeName}: {desc} (device={deviceName opts})"

/--
Two-line model-zoo banner: a headline with the selected device, then one detail line.
-/
def bannerWithDeviceDetails
    (exeName desc details : String) (opts : Options) : String :=
  bannerWithDevice exeName desc opts ++ "\n" ++ details

/-- Fail with a contextual error when an executable model check is false. -/
def check (exeName msg : String) (b : Bool) : IO Unit :=
  unless b do throw <| IO.userError s!"{exeName}: {msg}"

/-- Write a before-and-after loss comparison to a JSON file. -/
def writeBeforeAfterLossLogPath (path : System.FilePath)
    (title : String) (steps : Nat) (beforeLoss afterLoss : Float) (notes : Array String := #[]) :
    IO Unit :=
  Training.writeLossComparison path title steps beforeLoss afterLoss notes

/-- Write a before-and-after loss comparison to the selected logging destination. -/
def writeBeforeAfterLossLog
    (dest : Training.LogDestination)
    (title : String) (steps : Nat) (beforeLoss afterLoss : Float) (notes : Array String := #[]) :
    IO Unit :=
  Training.writeLossComparisonTo dest title steps beforeLoss afterLoss notes

/-- Write one scalar curve to the selected logging destination. -/
def writeCurveLog
    (dest : Training.LogDestination)
    (title : String) (curve : Training.Curve)
    (seriesName : String := "loss") (notes : Array String := #[]) :
    IO Unit :=
  Training.Curve.writeLogTo curve dest title seriesName notes

/--
Write a single-curve training log with an explicit series color.

Use this when a command already has a `Training.Curve` and wants a `TrainLog` instead of the default
`"loss"` curve writer.
-/
def writeCurveTrainLog
    (dest : Training.LogDestination)
    (title : String)
    (curve : Training.Curve)
    (seriesName : String)
    (color : String := "#4e79a7")
    (notes : Array String := #[]) :
    IO Unit :=
  Training.writeLogTo dest (curve.toTrainLog title seriesName color notes)

/--
Write a multi-series metric history as a TrainLog artifact.

Use this when a command has a `Training.MetricHistory` with named, colored series and wants to write
the usual `TrainLog` artifact without repeating the conversion code in every example.
-/
def writeMetricHistoryLog
    (dest : Training.LogDestination)
    (title : String)
    (history : Training.MetricHistory)
    (notes : Array String := #[]) :
    IO Unit :=
  Training.writeLogTo dest (history.toTrainLog (title := title) (notes := notes))

/-- Write a prepared training log to a JSON file. -/
def writeTrainLogPath (path : System.FilePath) (log : Training.TrainLog) :
    IO Unit :=
  Training.writeLog path log

/-- Write a prepared training log to the selected destination. -/
def writeTrainLog
    (dest : Training.LogDestination)
    (log : Training.TrainLog) :
    IO Unit :=
  Training.writeLogTo dest log

@[inherit_doc TorchLean.Trainer.FixedSample.curveFloat]
def trainFixedCurveFloat
    {σ τ : Shape}
    (mkModel : TorchLean.nn.M (TorchLean.nn.Sequential σ τ))
    (mkModuleDef :
      (model : TorchLean.nn.Sequential σ τ) →
        TorchLean.Module.ScalarModuleDef (TorchLean.nn.paramShapes model) [σ, τ])
    (mkOptim :
      (paramShapes : List Shape) → _root_.Runtime.Autograd.TorchLean.Optim.Optimizer Float paramShapes)
    (opts : Options)
    (sample : SupervisedSample Float σ τ)
    (steps : Nat)
    (cudaMemWatch : Nat := 0) :
    IO Training.Curve :=
  TorchLean.Trainer.FixedSample.curveFloat
    (mkModel := mkModel)
    (mkModuleDef := mkModuleDef)
    (mkOptim := mkOptim)
    (opts := opts)
    (sample := sample)
    (steps := steps)
    (cudaMemWatch := cudaMemWatch)

end NN.Examples.ModelZoo

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Examples.Data.RealPaths
public import NN.Examples.Models.Common.Train

/-!
# Shared Real-Data Helpers for Model Examples

The model examples should exercise real data paths. We keep the shared pieces here:

- loading a prepared CIFAR-10 NPY minibatch,
- reading a local text corpus, and
- printing the same "how to prepare data" hint everywhere.

The data files are prepared by `scripts/datasets/download_example_data.py`; examples report missing inputs
explicitly instead of silently falling back to synthetic tensors.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.RealData

/-- Number of channels in the prepared CIFAR-10 image tensors. -/
def cifarChannels : Nat := 3

/-- Height of the prepared CIFAR-10 image tensors. -/
def cifarHeight : Nat := 32

/-- Width of the prepared CIFAR-10 image tensors. -/
def cifarWidth : Nat := 32

/-- Number of CIFAR-10 classes, hence the width of one-hot targets. -/
def cifarClasses : Nat := 10

/-- Default row budget for CIFAR-10 model-zoo commands. -/
def defaultCifarRows : Nat := 1

/-- Number of channels in converted ImageNet-style image tensors. -/
def imagenet64Channels : Nat := 3

/-- Height of converted ImageNet-style image tensors. -/
def imagenet64Height : Nat := 64

/-- Width of converted ImageNet-style image tensors. -/
def imagenet64Width : Nat := 64

/-- Number of ImageNet-style classes expected by the converted label path. -/
def imagenet64Classes : Nat := 1000

/-- Default row budget for ImageNet64 model-zoo runs. -/
def defaultImageNet64Rows : Nat := 200

instance : NeZero cifarChannels := ⟨by decide⟩
instance : NeZero cifarHeight := ⟨by decide⟩
instance : NeZero cifarWidth := ⟨by decide⟩
instance : NeZero cifarClasses := ⟨by decide⟩
instance : NeZero imagenet64Channels := ⟨by decide⟩
instance : NeZero imagenet64Height := ⟨by decide⟩
instance : NeZero imagenet64Width := ⟨by decide⟩
instance : NeZero imagenet64Classes := ⟨by decide⟩

/-- Shape of one CIFAR-10 image after conversion to CHW layout. -/
abbrev CifarImage : Shape :=
  .dim cifarChannels (.dim cifarHeight (.dim cifarWidth .scalar))

/-- One-hot CIFAR-10 target shape. -/
abbrev CifarTarget : Shape :=
  .dim cifarClasses .scalar

/-- Take the top-left `h × w` view of a CIFAR image batch. -/
def cropCifarImages (batch h w : Nat)
    (hH : h ≤ cifarHeight) (hW : w ≤ cifarWidth)
    (x : Tensor Float (.dim batch (.dim cifarChannels (.dim cifarHeight (.dim cifarWidth .scalar))))) :
    Tensor Float (.dim batch (.dim cifarChannels (.dim h (.dim w .scalar)))) :=
  Spec.Tensor.dim (fun bi =>
    let img := Spec.getAtSpec x bi
    Spec.Tensor.dim (fun ch =>
      let plane := Spec.getAtSpec img ch
      Spec.Tensor.dim (fun row =>
        let srcRow : Fin cifarHeight := ⟨row.val, Nat.lt_of_lt_of_le row.isLt hH⟩
        let line := Spec.getAtSpec plane srcRow
        Spec.Tensor.dim (fun col =>
          let srcCol : Fin cifarWidth := ⟨col.val, Nat.lt_of_lt_of_le col.isLt hW⟩
          Spec.getAtSpec line srcCol))))

/-- Crop a CIFAR minibatch while leaving the one-hot class labels unchanged. -/
def cropCifarBatch (batch h w : Nat)
    (hH : h ≤ cifarHeight) (hW : w ≤ cifarWidth)
    (sample : Sample.Batch Float batch CifarImage CifarTarget) :
    Sample.Supervised Float (.dim batch (.dim cifarChannels (.dim h (.dim w .scalar)))) (.dim batch (.dim cifarClasses .scalar)) :=
  Sample.mk (cropCifarImages batch h w hH hW (Sample.x sample)) (Sample.y sample)

/-- ImageNet-style converted image shape used by the higher-resolution diffusion example. -/
abbrev ImageNet64Image : Shape :=
  .dim imagenet64Channels (.dim imagenet64Height (.dim imagenet64Width .scalar))

/--
One-hot target shape for ImageNet-style folders.

The diffusion example ignores labels, but reusing `Data.LabeledSource` keeps the data path identical
to the supervised examples and lets class-directory conversion catch malformed labels early.
-/
abbrev ImageNet64Target : Shape :=
  .dim imagenet64Classes .scalar

/-- Error message shown when a CIFAR-backed example cannot find the prepared arrays. -/
def missingCifarHint : String :=
  "Prepare real CIFAR-10 arrays with:\n" ++
  "  python3 scripts/datasets/download_example_data.py --cifar10\n" ++
  "Then rerun the model command."

/-- Error message shown when an ImageNet64-backed example cannot find the prepared arrays. -/
def missingImageNet64Hint : String :=
  "Prepare an ImageNet-style 64x64 subset with:\n" ++
  "  python3 scripts/datasets/torchlean_data_convert.py image-folder \\\n" ++
  "    --input /path/to/imagenet/train \\\n" ++
  "    --x-output data/real/imagenet64/imagenet64_train_X.npy \\\n" ++
  "    --y-output data/real/imagenet64/imagenet64_train_y.npy \\\n" ++
  "    --height 64 --width 64 --labels-from-dirs --limit 2000\n" ++
  "This path expects a local ImageNet-style image-folder dataset."

/-- Error message shown when a text-model example cannot find a corpus. -/
def missingTextHint : String :=
  "Prepare the text corpus with:\n" ++
  "  python3 scripts/datasets/download_example_data.py --tiny-shakespeare\n" ++
  "or pass --data-file PATH."

/-- Error message shown when the Auto MPG CSV is missing. -/
def missingAutoMpgHint : String :=
  "Prepare the Auto MPG CSV with:\n" ++
  "  python3 scripts/datasets/download_example_data.py --auto-mpg\n" ++
  "Then rerun the model command."

/-- Error message shown when the household-power forecasting dataset is missing. -/
def missingHouseholdPowerHint : String :=
  "Prepare the household-power forecasting windows with:\n" ++
  "  python3 scripts/datasets/download_example_data.py --household-power --household-power-windows 512\n" ++
  "Then rerun the model command."

/-- Default local path for the Tiny Shakespeare corpus. -/
def tinyShakespearePath : System.FilePath :=
  _root_.NN.Examples.Data.RealPaths.tinyShakespeare

/-- Default local path for the TinyStories validation split. -/
def tinyStoriesValidPath : System.FilePath :=
  _root_.NN.Examples.Data.RealPaths.tinyStoriesValid

/-- Data-preparation hint for commands that only need Tiny Shakespeare. -/
def missingTinyShakespeareHint : String :=
  "Download Tiny Shakespeare with:\n" ++
  "  python3 scripts/datasets/download_example_data.py --tiny-shakespeare"

/-- Data-preparation hint for commands that accept both Tiny Shakespeare and TinyStories. -/
def missingTinyShakespeareOrTinyStoriesHint : String :=
  missingTinyShakespeareHint ++ "\n" ++
  "For TinyStories (valid split):\n" ++
  "  python3 scripts/datasets/download_example_data.py --tinystories-valid"

namespace NpyDatasets

def parseCifar (args : List String) :
    Except String (ModelZoo.NpyDataFlags × List String) := do
  ModelZoo.NpyDataFlags.parse args
    _root_.NN.Examples.Data.RealPaths.cifar10TrainX
    _root_.NN.Examples.Data.RealPaths.cifar10TrainY
    defaultCifarRows

/--
Parse the shared flags for an ImageNet-style 64x64 NPY dataset.

The expected input is produced by `scripts/datasets/torchlean_data_convert.py image-folder`; that converter
handles JPEG/PNG decoding, RGB conversion, resizing, class-directory labels, and the final NCHW
layout. Lean then reads only the simple `.npy` tensors.
-/
def parseImageNet64 (args : List String) :
    Except String (ModelZoo.NpyDataFlags × List String) := do
  ModelZoo.NpyDataFlags.parse args
    _root_.NN.Examples.Data.RealPaths.imagenet64TrainX
    _root_.NN.Examples.Data.RealPaths.imagenet64TrainY
    defaultImageNet64Rows

end NpyDatasets

/-- Parsed CIFAR dataset and fixed-sample training flags for runnable model examples. -/
abbrev CifarLoggedTrainFlags := ModelZoo.NpyLoggedTrainFlags

/-- Parsed CIFAR dataset and optimizer/training flags for classifier examples. -/
abbrev CifarModelTrainFlags := ModelZoo.NpyModelTrainFlags

namespace CifarLoggedTrainFlags

/--
Parse the standard CIFAR plus fixed-step training flags and reject unused arguments.

Generative examples use the same prepared CIFAR arrays and the same loss-curve logging contract;
only the model and target construction differ.
-/
def parse (exeName : String) (args : List String)
    (defaultLogPath : System.FilePath) (defaultSteps : Nat := 10) :
    Except String CifarLoggedTrainFlags :=
  ModelZoo.NpyLoggedTrainFlags.parse exeName args defaultLogPath defaultSteps
    (parseData := NpyDatasets.parseCifar)

end CifarLoggedTrainFlags

namespace CifarModelTrainFlags

/--
Parse the standard CIFAR plus optimizer/training flags.

Vision examples share the same CIFAR data boundary and optimizer controls; architecture files only
need to provide the model constructor and logging title. Any remaining arguments are preserved so
the caller can forward runtime flags such as `--device cpu`, `--device cuda`, or `--execution typed-graph` to the
public `Trainer.RunConfig` parser.
-/
def parse (exeName : String) (args : List String)
    (defaultLogPath : System.FilePath) (defaultSteps : Nat := 1) (defaultLr : Float := 1e-3) :
    Except String (CifarModelTrainFlags × List String) :=
  ModelZoo.NpyModelTrainFlags.parse exeName args defaultLogPath
    (defaultSteps := defaultSteps) (defaultLr := defaultLr)
    (parseData := NpyDatasets.parseCifar)

end CifarModelTrainFlags

/-- Common TrainLog notes for CIFAR-backed examples. -/
def cifarTrainNotes (opts : Options)
    (flags : CifarLoggedTrainFlags) (extra : Array String := #[]) : Array String :=
  ModelZoo.NpyDataFlags.trainLogNotes flags.toNpyDataFlags "cifar10" ++
  #[ModelZoo.deviceNote opts, ModelZoo.cudaMemoryNote opts flags.steps flags.cudaMemWatch]
  ++ extra

namespace ForecastWindowDataFlags

/--
Parse the shared flags for household-power forecasting windows.

Forecasting commands share `--data-dir`, `--x`, `--y`, `--windows`, `--report-offset`, and `--seed`.
-/
def parse
    (exeName : String)
    (args : List String)
    (defaultWindows : Nat := 512)
    (defaultReportOffset : Nat := 96) :
    Except String (ModelZoo.ForecastWindowDataFlags × List String) := do
  let (dataDir, args) ← _root_.NN.Examples.Data.RealPaths.takeDataDir args
  let (seed, args) ← CLI.takeSeed args (default := 0)
  let (windows, args) ← CLI.takePositiveNatFlag args exeName "windows" defaultWindows
  let (reportOffset, args) ← CLI.takeNatFlagDefault args "report-offset" defaultReportOffset
  let (xPath, args) ←
    CLI.takePathFlagDefault args "x" (_root_.NN.Examples.Data.RealPaths.householdPowerX dataDir)
  let (yPath, args) ←
    CLI.takePathFlagDefault args "y" (_root_.NN.Examples.Data.RealPaths.householdPowerY dataDir)
  pure ({ xPath := xPath
          yPath := yPath
          windows := windows
          reportOffset := reportOffset
          seed := seed }, args)

end ForecastWindowDataFlags

/-- Parsed household-power forecasting data plus optimizer/training flags. -/
abbrev HouseholdPowerModelTrainFlags := ModelZoo.ForecastWindowModelTrainFlags

namespace HouseholdPowerModelTrainFlags

/--
Parse the standard household-power forecasting flags plus optimizer/training flags.

The forecasting command still owns the model and reporting logic, but the shared data/runtime flag
surface lives here with the other real-data code.
-/
def parse
    (exeName : String)
    (args : List String)
    (defaultLogPath : System.FilePath)
    (defaultSteps : Nat := 100)
    (defaultLr : Float := 0.01)
    (defaultWindows : Nat := 512)
    (defaultReportOffset : Nat := 96) :
    Except String (HouseholdPowerModelTrainFlags × List String) :=
  ModelZoo.ForecastWindowModelTrainFlags.parse exeName args defaultLogPath
    (defaultSteps := defaultSteps) (defaultLr := defaultLr)
    (parseData :=
      fun args => ForecastWindowDataFlags.parse exeName args defaultWindows defaultReportOffset)

end HouseholdPowerModelTrainFlags

def loadCifarLoader
    (exeName : String) (batch nRows seed : Nat) (xPath yPath : System.FilePath) :
    IO (Data.BatchLoader Float batch CifarImage CifarTarget) := do
  Data.requirePairedFiles
    exeName
    "CIFAR-10 images" xPath
    "CIFAR-10 labels" yPath
    missingCifarHint
  let src := Data.LabeledSource.ofPaths .npy xPath yPath nRows
    [cifarChannels, cifarHeight, cifarWidth] cifarClasses
  let dsE ← src.load (α := Float)
  let ds ←
    match dsE with
    | .ok ds => pure ds
    | .error msg =>
        let hint :=
          s!"{exeName}: failed to load CIFAR-10 arrays for --n-total {nRows}.\n" ++
          s!"{msg}\n" ++
          "If your local .npy files contain fewer rows, pass --n-total with that row count; " ++
          "to regenerate the default 200-row slice, run:\n" ++
          "  python3 scripts/datasets/download_example_data.py --cifar10"
        throw <| IO.userError hint
  -- Return the typed minibatch loader. Callers can take one batch for a fixed-sample check or pass
  -- the loader to the shared training code for shuffled multi-step training.
  let dl := Data.batchLoader ds batch (shuffle := true) (seed := seed) (dropLast := true)
  pure dl

/-- Public trainer dataset for prepared CIFAR-10 NPY image/label arrays. -/
def cifarDataset (nRows : Nat) (xPath yPath : System.FilePath) :
    Trainer.DataSource CifarImage CifarTarget :=
  let src := Data.LabeledSource.ofPaths .npy xPath yPath nRows
    [cifarChannels, cifarHeight, cifarWidth] cifarClasses
  Data.labeledDataset src

/-- Common training-log notes for CIFAR-backed classifier examples. -/
def cifarClassifierNotes (batch : Nat)
    (flags : CifarModelTrainFlags) (extra : Array String := #[]) : Array String :=
  ModelZoo.NpyDataFlags.trainLogNotes flags.toNpyDataFlags "cifar10" ++
  #[s!"lr={flags.lr}", s!"steps={flags.steps}", s!"batch={batch}"]
  ++ extra

/--
Shared `main` entrypoint for CIFAR-backed curve-reporting commands.

Some commands do not match the public trainer result shape because they manage several modules or log
one custom scalar curve instead of a single trainer report. They still share the same CIFAR parsing,
runtime parsing, CUDA-memory notes, and TrainLog boundary.
-/
def cifarCurve
    (exeName : String)
    (args : List String)
    (defaultLogPath : System.FilePath)
    (defaultSteps : Nat := 10)
    (banner : Options → String)
    (seriesName : String)
    (title : String)
    (extraNotes : Options → CifarLoggedTrainFlags → Array String := fun _ _ => #[])
    (train : Options → CifarLoggedTrainFlags → IO Training.Curve) :
    IO UInt32 :=
  TrainCommand.runParsedWith exeName args
    (fun rest => do
      let flags ← CifarLoggedTrainFlags.parse exeName rest defaultLogPath defaultSteps
      pure (flags, []))
    banner
    train
    (fun opts flags curve =>
      ModelZoo.writeCurveTrainLog flags.log title curve seriesName
        (notes := cifarTrainNotes opts flags (extraNotes opts flags)))

/-- Load one shuffled epoch of full CIFAR-10 minibatches from prepared `.npy` arrays. -/
def loadCifarBatches
    (exeName : String) (batch nRows seed : Nat) (xPath yPath : System.FilePath) :
    IO (List (Sample.Batch Float batch CifarImage CifarTarget)) := do
  let dl ← loadCifarLoader exeName batch nRows seed xPath yPath
  let (_dl', batches) ← ModelZoo.orThrow exeName <|
    Data.BatchLoader.nonemptyEpoch exeName dl
  pure batches

/-- Load the first full CIFAR-10 minibatch from the shared CIFAR loader. -/
def loadCifarBatch
    (exeName : String) (batch nRows seed : Nat) (xPath yPath : System.FilePath) :
    IO (Sample.Batch Float batch CifarImage CifarTarget) := do
  let dl ← loadCifarLoader exeName batch nRows seed xPath yPath
  ModelZoo.orThrow exeName <| Data.BatchLoader.firstFullBatch exeName dl

/--
Load a user-prepared ImageNet-style `64x64` minibatch.

This loader reads prepared `.npy` arrays rather than JPEG files. The Python converter is the trust
boundary for filesystem image decoding and resizing; this Lean path checks the resulting tensor shape
and class range before handing the batch to examples.
-/
def loadImageNet64Loader
    (exeName : String) (batch nRows seed : Nat) (xPath yPath : System.FilePath) :
    IO (Data.BatchLoader Float batch ImageNet64Image ImageNet64Target) := do
  unless (← xPath.pathExists) do
    throw <| IO.userError s!"{exeName}: missing ImageNet64 images: {xPath}\n{missingImageNet64Hint}"
  unless (← yPath.pathExists) do
    throw <| IO.userError s!"{exeName}: missing ImageNet64 labels: {yPath}\n{missingImageNet64Hint}"
  let src := Data.LabeledSource.ofPaths .npy xPath yPath nRows
    [imagenet64Channels, imagenet64Height, imagenet64Width] imagenet64Classes
  let dsE ← src.load (α := Float)
  let ds ←
    match dsE with
    | .ok ds => pure ds
    | .error msg =>
        let hint :=
          s!"{exeName}: failed to load ImageNet64 arrays for --n-total {nRows}.\n" ++
          s!"{msg}\n" ++
          "If your local .npy files contain fewer rows, pass --n-total with that row count; " ++
          "to create ImageNet64 arrays, run the image-folder converter described in the error hint."
        throw <| IO.userError hint
  -- Same convention as CIFAR: this is the reusable loader for full-dataset loops.
  -- `loadImageNet64Batch` is for call sites that need a single fixed minibatch.
  let dl := Data.batchLoader ds batch (shuffle := true) (seed := seed) (dropLast := true)
  pure dl

/-- Load one shuffled epoch of full ImageNet64-style minibatches from prepared `.npy` arrays. -/
def loadImageNet64Batches
    (exeName : String) (batch nRows seed : Nat) (xPath yPath : System.FilePath) :
    IO (List (Sample.Batch Float batch ImageNet64Image ImageNet64Target)) := do
  let dl ← loadImageNet64Loader exeName batch nRows seed xPath yPath
  let (_dl', batches) ← ModelZoo.orThrow exeName <|
    Data.BatchLoader.nonemptyEpoch exeName dl
  pure batches

/-- Load the first full ImageNet64-style minibatch from the shared ImageNet64 loader. -/
def loadImageNet64Batch
    (exeName : String) (batch nRows seed : Nat) (xPath yPath : System.FilePath) :
    IO (Sample.Batch Float batch ImageNet64Image ImageNet64Target) := do
  let dl ← loadImageNet64Loader exeName batch nRows seed xPath yPath
  ModelZoo.orThrow exeName <| Data.BatchLoader.firstFullBatch exeName dl

/--
Load a CIFAR minibatch and expose it as a compact flattened vector batch.

The file paths and download hints remain in `NN.Examples`. Tensor flattening uses the ordinary
shape-general operation from the public tensor API, so the data path does not require an
image-specific model helper.
-/
def loadCifarFeatureBatch (batch : Nat) (cfg : nn.models.DenseGenerative.Config)
    (hData : cfg.dataDim ≤ Spec.Shape.size CifarImage)
    (exeName : String) (xPath yPath : System.FilePath) (nRows seed : Nat) :
    IO (Tensor Float (cfg.dataShape (.dim batch .scalar))) := do
  let batchSample ← loadCifarBatch exeName batch nRows seed xPath yPath
  pure (Tensor.flattenPrefix (.dim batch .scalar) cfg.dataDim hData (Sample.x batchSample))

/--
Public singleton dataset for compact vector generative examples over flattened CIFAR batches.

Autoencoder, VAE, and VQ-VAE examples all load one real CIFAR batch, flatten it to the compact
vector boundary, build one supervised sample, and hand that sample to the public trainer API. The
sample itself may be Float-specific; this dataset constructor casts it into the runtime-selected scalar so the
command still works across the ordinary public runtime backends.
-/
def cifarFeatureDataset {τ : Shape}
    (batch : Nat)
    (cfg : nn.models.DenseGenerative.Config)
    (hData : cfg.dataDim ≤ Spec.Shape.size CifarImage)
    (exeName : String)
    (mkSample : Tensor Float (cfg.dataShape (.dim batch .scalar)) →
      Sample.Supervised Float (cfg.dataShape (.dim batch .scalar)) τ)
    (xPath yPath : System.FilePath) (nRows seed : Nat) :
    Trainer.DataSource (cfg.dataShape (.dim batch .scalar)) τ :=
  Data.singletonFloatIO do
    let x ← loadCifarFeatureBatch batch cfg hData exeName xPath yPath nRows seed
    pure (mkSample x)

/-- Shared text-corpus CLI/data boundary for local text-model examples. -/
abbrev TextCorpusFlags := text.TextCorpusPathOptions

namespace TextCorpusFlags

/-- Command-line options shared by local text-corpus examples. -/
def help : List String :=
  [ "  --data-file PATH      read a local UTF-8 text corpus"
  , "  --tiny-shakespeare    use data/real/tinyshakespeare/input.txt"
  ]

/--
Parse the shared `--data-file` flag used by local text-model examples.

`--tiny-shakespeare` is accepted as an explicit shortcut for the default corpus path.
-/
def parse (args : List String) :
    Except String (TextCorpusFlags × List String) := do
  let args := args.filter (fun a => a != "--tiny-shakespeare")
  text.TextCorpusPathOptions.parse args tinyShakespearePath

/-- Read the selected text corpus and fail with a shared preparation hint when it is missing. -/
def read (exeName : String) (flags : TextCorpusFlags) : IO String := do
  unless (← flags.path.pathExists) do
    throw <| IO.userError s!"{exeName}: missing text corpus: {flags.path}\n{missingTextHint}"
  let text ← IO.FS.readFile flags.path
  if text.isEmpty then
    throw <| IO.userError s!"{exeName}: empty text corpus: {flags.path}"
  pure text

end TextCorpusFlags

end NN.Examples.Models.RealData

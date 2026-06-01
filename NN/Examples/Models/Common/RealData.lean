/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.API.Models.Generative
public import NN.API.Models.TrainFixed
public import NN.Examples.Data.RealPaths

/-!
# Shared Real-Data Helpers for Model Examples

`NN/Examples/Models/*` should exercise real data paths. This file centralizes the shared parts:

- loading a prepared CIFAR-10 NPY minibatch,
- reading a local text corpus, and
- printing the same "how to prepare data" hint everywhere.

The data files are prepared by `scripts/datasets/download_example_data.py`; examples report missing inputs
explicitly instead of silently falling back to synthetic tensors.
-/

@[expose] public section

open Spec
open Tensor
open NN.API

namespace NN.Examples.Models.RealData

def cifarChannels : Nat := 3
def cifarHeight : Nat := 32
def cifarWidth : Nat := 32
def cifarClasses : Nat := 10
def defaultCifarRows : Nat := 200

def imagenet64Channels : Nat := 3
def imagenet64Height : Nat := 64
def imagenet64Width : Nat := 64
def imagenet64Classes : Nat := 1000
def defaultImageNet64Rows : Nat := 200

instance : NeZero cifarChannels := ⟨by decide⟩
instance : NeZero cifarHeight := ⟨by decide⟩
instance : NeZero cifarWidth := ⟨by decide⟩
instance : NeZero cifarClasses := ⟨by decide⟩
instance : NeZero imagenet64Channels := ⟨by decide⟩
instance : NeZero imagenet64Height := ⟨by decide⟩
instance : NeZero imagenet64Width := ⟨by decide⟩
instance : NeZero imagenet64Classes := ⟨by decide⟩

abbrev CifarImage : Shape :=
  Shape.Image cifarChannels cifarHeight cifarWidth

abbrev CifarTarget : Shape :=
  Shape.Vec cifarClasses

/-- ImageNet-style converted image shape used by the higher-resolution diffusion example. -/
abbrev ImageNet64Image : Shape :=
  Shape.Image imagenet64Channels imagenet64Height imagenet64Width

/--
One-hot target shape for ImageNet-style folders.

The diffusion example ignores labels, but reusing `Data.LabeledSource` keeps the data path identical
to the supervised examples and lets class-directory conversion catch malformed labels early.
-/
abbrev ImageNet64Target : Shape :=
  Shape.Vec imagenet64Classes

def missingCifarHint : String :=
  "Prepare real CIFAR-10 arrays with:\n" ++
  "  python3 scripts/datasets/download_example_data.py --cifar10\n" ++
  "Then rerun the model command."

def missingImageNet64Hint : String :=
  "Prepare an ImageNet-style 64x64 subset with:\n" ++
  "  python3 scripts/datasets/torchlean_data_convert.py image-folder \\\n" ++
  "    --input /path/to/imagenet/train \\\n" ++
  "    --x-output data/real/imagenet64/imagenet64_train_X.npy \\\n" ++
  "    --y-output data/real/imagenet64/imagenet64_train_y.npy \\\n" ++
  "    --height 64 --width 64 --labels-from-dirs --limit 2000\n" ++
  "This path expects a local ImageNet-style image-folder dataset."

def missingTextHint : String :=
  "Prepare the text corpus with:\n" ++
  "  python3 scripts/datasets/download_example_data.py --tiny-shakespeare\n" ++
  "or pass --data-file PATH."

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

def parseCifarFlags (args : List String) :
    Except String (System.FilePath × System.FilePath × Nat × Nat × List String) := do
  let (seed, args) ← CLI.takeSeed args 0
  let (nRows?, args) ← CLI.takeNatFlagOnce args "n-total"
  let (x?, args) ← CLI.takePathFlagOnce args "x"
  let (y?, args) ← CLI.takePathFlagOnce args "y"
  pure (x?.getD _root_.NN.Examples.Data.RealPaths.cifar10TrainX,
    y?.getD _root_.NN.Examples.Data.RealPaths.cifar10TrainY,
    nRows?.getD defaultCifarRows,
    seed,
    args)

/-- Parsed CIFAR dataset and fixed-sample training flags for runnable model examples. -/
structure CifarLoggedTrainFlags where
  /-- Prepared CIFAR image tensor path. -/
  xPath : System.FilePath
  /-- Prepared CIFAR label tensor path. -/
  yPath : System.FilePath
  /-- Number of rows to read from the prepared arrays. -/
  nRows : Nat
  /-- Data-loader seed. -/
  seed : Nat
  /-- Shared `--steps`, `--log`, and CUDA telemetry flags. -/
  train : Common.LoggedTrainFlags

/-- Parsed CIFAR dataset and optimizer/training flags for classifier examples. -/
structure CifarModelTrainFlags where
  /-- Prepared CIFAR image tensor path. -/
  xPath : System.FilePath
  /-- Prepared CIFAR label tensor path. -/
  yPath : System.FilePath
  /-- Number of rows to read from the prepared arrays. -/
  nRows : Nat
  /-- Data-loader seed. -/
  seed : Nat
  /-- Shared optimizer, step-count, logging, and CUDA telemetry flags. -/
  train : Common.ModelTrainFlags

/--
Parse the standard CIFAR plus fixed-step training flags and reject unused arguments.

Generative examples use the same prepared CIFAR arrays and the same loss-curve logging contract;
only the model and target construction differ.
-/
def parseCifarLoggedTrainFlags (exeName : String) (args : List String)
    (defaultLogPath : System.FilePath) (defaultSteps : Nat := 10) :
    Except String CifarLoggedTrainFlags := do
  let (xPath, yPath, nRows, seed, rest) ← parseCifarFlags args
  let (train, rest) ← Common.parseLoggedTrainFlags exeName rest defaultLogPath defaultSteps
  CLI.requireNoArgs rest
  pure { xPath, yPath, nRows, seed, train }

/--
Parse the standard CIFAR plus optimizer/training flags and reject unused arguments.

Vision examples share the same CIFAR data boundary and optimizer controls; architecture files only
need to provide the model constructor and logging title.
-/
def parseCifarModelTrainFlags (exeName : String) (args : List String)
    (defaultLogPath : System.FilePath) (defaultSteps : Nat := 1) (defaultLr : Float := 1e-3) :
    Except String CifarModelTrainFlags := do
  let (xPath, yPath, nRows, seed, rest) ← parseCifarFlags args
  let (train, rest) ← Common.parseModelTrainFlags exeName rest defaultLogPath defaultSteps defaultLr
  CLI.requireNoArgs rest
  pure { xPath, yPath, nRows, seed, train }

/-- Common TrainLog notes for CIFAR-backed examples. -/
def cifarTrainNotes (opts : Runtime.Autograd.Torch.Options)
    (flags : CifarLoggedTrainFlags) (extra : Array String := #[]) : Array String :=
  #[s!"data=cifar10", s!"nRows={flags.nRows}",
    s!"device={if opts.useGpu then "cuda" else "cpu"}",
    s!"cuda_mem_watch={Common.effectiveCudaMemWatch opts flags.train.steps flags.train.cudaMemWatch}"]
  ++ extra

/--
Parse the shared flags for an ImageNet-style 64x64 NPY dataset.

The expected input is produced by `scripts/datasets/torchlean_data_convert.py image-folder`; that converter
handles JPEG/PNG decoding, RGB conversion, resizing, class-directory labels, and the final NCHW
layout. Lean then reads only the simple `.npy` tensors.
-/
def parseImageNet64Flags (args : List String) :
    Except String (System.FilePath × System.FilePath × Nat × Nat × List String) := do
  let (seed, args) ← CLI.takeSeed args 0
  let (nRows?, args) ← CLI.takeNatFlagOnce args "n-total"
  let (x?, args) ← CLI.takePathFlagOnce args "x"
  let (y?, args) ← CLI.takePathFlagOnce args "y"
  pure (x?.getD _root_.NN.Examples.Data.RealPaths.imagenet64TrainX,
    y?.getD _root_.NN.Examples.Data.RealPaths.imagenet64TrainY,
    nRows?.getD defaultImageNet64Rows,
    seed,
    args)

def loadCifarLoader {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (exeName : String) (batch nRows seed : Nat) (xPath yPath : System.FilePath) :
    IO (Data.BatchLoader α batch CifarImage CifarTarget) := do
  unless (← xPath.pathExists) do
    throw <| IO.userError s!"{exeName}: missing CIFAR-10 images: {xPath}\n{missingCifarHint}"
  unless (← yPath.pathExists) do
    throw <| IO.userError s!"{exeName}: missing CIFAR-10 labels: {yPath}\n{missingCifarHint}"
  let src := Data.LabeledSource.ofPaths .npy xPath yPath nRows
    [cifarChannels, cifarHeight, cifarWidth] cifarClasses
  let dsE ← src.load (α := α)
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
  -- the loader to the shared training helpers for shuffled multi-step training.
  let dl := Data.batchLoader ds batch (shuffle := true) (seed := seed) (dropLast := true)
  pure dl

/--
Train an image classifier on a prepared CIFAR-10 loader.

The vision examples differ in architecture, but the runtime path is the same: load CIFAR, build a
cross-entropy module, train with Adam for the requested number of steps, and write a TrainLog curve.
Keeping that path here makes CNN, ResNet, and ViT use the same data and logging contract.
-/
def fitCifarClassifier
    (exeName title : String) (batch : Nat)
    (mkModel : nn.M (nn.Sequential (Shape.dim batch CifarImage) (Shape.dim batch CifarTarget)))
    (opts : Runtime.Autograd.Torch.Options)
    (flags : CifarModelTrainFlags)
    (notes : Array String := #[]) :
    IO (train.FitReport Float) := do
  let loader ← loadCifarLoader (α := Float) exeName batch flags.nRows flags.seed flags.xPath flags.yPath
  nn.withModel mkModel fun model => do
    let modDef := nn.crossEntropyOneHotScalarModuleDef model (reduction := .mean)
    let module ← TorchLean.Module.instantiateWithOptions (α := Float) modDef id opts
    let opt := Common.adamOptimizer (α := Float) id (nn.paramShapes model) flags.train.lr
    let cudaMemWatch :=
      Common.effectiveCudaMemWatch opts flags.train.train.steps flags.train.cudaMemWatch
    let baseNotes :=
      #[s!"data=cifar10", s!"x={flags.xPath}", s!"y={flags.yPath}", s!"nRows={flags.nRows}",
        s!"device={if opts.useGpu then "cuda" else "cpu"}", s!"lr={flags.train.lr}",
        s!"steps={flags.train.train.steps}", s!"batch={batch}", s!"cuda_mem_watch={cudaMemWatch}"]
    let (report, _loader') ← train.fitModuleLoaderStepsLoggedFloat module opt opts
      flags.train.train.steps loader flags.train.train.log title
      (baseNotes ++ notes) "loss" flags.train.cudaMemWatch
    pure report

/-- Load one shuffled epoch of full CIFAR-10 minibatches from prepared `.npy` arrays. -/
def loadCifarBatches {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (exeName : String) (batch nRows seed : Nat) (xPath yPath : System.FilePath) :
    IO (List (sample.Batch α batch CifarImage CifarTarget)) := do
  let dl ← loadCifarLoader (α := α) exeName batch nRows seed xPath yPath
  let (_dl', batches) ← Common.orThrow exeName <| Data.BatchLoader.epoch exeName dl
  match batches with
  | _ :: _ => pure batches
  | [] =>
      throw <| IO.userError
        s!"{exeName}: no full CIFAR-10 minibatch available (batch={batch}, rows={Data.size dl.raw.dataset})"

/-- Load the first full CIFAR-10 minibatch from the shared CIFAR loader. -/
def loadCifarBatch {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (exeName : String) (batch nRows seed : Nat) (xPath yPath : System.FilePath) :
    IO (sample.Batch α batch CifarImage CifarTarget) := do
  match (← loadCifarBatches (α := α) exeName batch nRows seed xPath yPath) with
  | b :: _ => pure b
  | [] =>
      throw <| IO.userError s!"{exeName}: internal error: expected at least one CIFAR-10 minibatch"

/--
Load a user-prepared ImageNet-style `64x64` minibatch.

This loader reads prepared `.npy` arrays rather than JPEG files. The Python converter is the trust
boundary for filesystem image decoding and resizing; this Lean path checks the resulting tensor shape
and class range before handing the batch to examples.
-/
def loadImageNet64Loader {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (exeName : String) (batch nRows seed : Nat) (xPath yPath : System.FilePath) :
    IO (Data.BatchLoader α batch ImageNet64Image ImageNet64Target) := do
  unless (← xPath.pathExists) do
    throw <| IO.userError s!"{exeName}: missing ImageNet64 images: {xPath}\n{missingImageNet64Hint}"
  unless (← yPath.pathExists) do
    throw <| IO.userError s!"{exeName}: missing ImageNet64 labels: {yPath}\n{missingImageNet64Hint}"
  let src := Data.LabeledSource.ofPaths .npy xPath yPath nRows
    [imagenet64Channels, imagenet64Height, imagenet64Width] imagenet64Classes
  let dsE ← src.load (α := α)
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
  -- Same convention as CIFAR: this is the reusable loader for full-dataset loops. The
  -- `loadImageNet64Batch` wrapper below is for call sites that need a single fixed minibatch.
  let dl := Data.batchLoader ds batch (shuffle := true) (seed := seed) (dropLast := true)
  pure dl

/-- Load one shuffled epoch of full ImageNet64-style minibatches from prepared `.npy` arrays. -/
def loadImageNet64Batches {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (exeName : String) (batch nRows seed : Nat) (xPath yPath : System.FilePath) :
    IO (List (sample.Batch α batch ImageNet64Image ImageNet64Target)) := do
  let dl ← loadImageNet64Loader (α := α) exeName batch nRows seed xPath yPath
  let (_dl', batches) ← Common.orThrow exeName <| Data.BatchLoader.epoch exeName dl
  match batches with
  | _ :: _ => pure batches
  | [] =>
      throw <| IO.userError
        s!"{exeName}: no full ImageNet64 minibatch available (batch={batch}, rows={Data.size dl.raw.dataset})"

/-- Load the first full ImageNet64-style minibatch from the shared ImageNet64 loader. -/
def loadImageNet64Batch {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (exeName : String) (batch nRows seed : Nat) (xPath yPath : System.FilePath) :
    IO (sample.Batch α batch ImageNet64Image ImageNet64Target) := do
  match (← loadImageNet64Batches (α := α) exeName batch nRows seed xPath yPath) with
  | b :: _ => pure b
  | [] =>
      throw <| IO.userError s!"{exeName}: internal error: expected at least one ImageNet64 minibatch"

/--
Load a CIFAR minibatch and expose it as a compact flattened vector batch.

The file paths and download hints remain in `NN.Examples`, but the flattening logic itself lives in
`NN.API.Models.Generative` so users can reuse it with their own image tensors.
-/
def loadCifarVectorBatch (cfg : nn.models.VectorGenerativeConfig)
    (hData : cfg.dataDim ≤ Shape.size CifarImage)
    (exeName : String) (xPath yPath : System.FilePath) (nRows seed : Nat) :
    IO (Tensor Float (nn.models.vectorDataShape cfg)) := do
  let batchSample ← loadCifarBatch (α := Float) exeName cfg.batch nRows seed xPath yPath
  pure (nn.models.flattenBatchPrefix cfg hData (sample.x batchSample))

/--
Train a vector generative model on the shared flattened-CIFAR sample path.

Autoencoder, VAE, and VQ-VAE examples differ in model and target construction, but they share the
same data boundary, Adam settings, curve reporting, and TrainLog format. This helper keeps those
runtime contracts in one place while leaving the theorem-facing model choices in each example file.
-/
def fitCifarVectorCurve {τ : Shape}
    (cfg : nn.models.VectorGenerativeConfig)
    (hData : cfg.dataDim ≤ Shape.size CifarImage)
    (exeName title : String)
    (mkModel : nn.M (nn.Sequential (nn.models.vectorDataShape cfg) τ))
    (mkSample : Tensor Float (nn.models.vectorDataShape cfg) →
      sample.Supervised Float (nn.models.vectorDataShape cfg) τ)
    (opts : Runtime.Autograd.Torch.Options)
    (flags : CifarLoggedTrainFlags)
    (notes : Array String := #[]) :
    IO _root_.Runtime.Training.Curve := do
  let x ← loadCifarVectorBatch cfg hData exeName flags.xPath flags.yPath flags.nRows flags.seed
  let sample := mkSample x
  let curve ←
    _root_.NN.API.Models.TrainFixed.curveFloat
      (mkModel := mkModel)
      (mkModuleDef := fun model => nn.mseScalarModuleDef model)
      (mkOptim := fun ps =>
        TorchLean.Optim.adam (α := Float) (paramShapes := ps)
          (lr := 1e-3) (beta1 := 0.9) (beta2 := 0.999) (epsilon := 1e-8))
      (opts := opts) (sample := sample) (steps := flags.train.steps)
      (cudaMemWatch := flags.train.cudaMemWatch)
  Common.printCurveLossSummary flags.train.steps curve
  Common.writeCurveLogTo flags.train.log title curve "loss"
    (cifarTrainNotes opts flags notes)
  pure curve

def parseTextFlags (args : List String) :
    Except String (System.FilePath × List String) := do
  -- `--tiny-shakespeare` names the default corpus explicitly; `--data-file` selects any local text.
  let args := args.filter (fun a => a != "--tiny-shakespeare")
  let (path?, args) ← CLI.takePathFlagOnce args "data-file"
  pure (path?.getD tinyShakespearePath, args)

def readTextCorpus (exeName : String) (path : System.FilePath) : IO String := do
  unless (← path.pathExists) do
    throw <| IO.userError s!"{exeName}: missing text corpus: {path}\n{missingTextHint}"
  let text ← IO.FS.readFile path
  if text.isEmpty then
    throw <| IO.userError s!"{exeName}: empty text corpus: {path}"
  pure text

def textCausalSample {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (seqLen vocab : Nat) (input : String) :
    sample.Supervised α (Shape.Mat seqLen vocab) (Shape.Mat seqLen vocab) :=
  let bytes := input.toUTF8
  let toks := (NN.API.text.byteTokenWindow bytes (seqLen + 1)).map (fun b => b % vocab)
  let (xF, yF) := NN.API.text.causalLmXYOneHotMatFloat seqLen vocab toks
  let x : Tensor α (Shape.Mat seqLen vocab) := Common.castTensor Runtime.ofFloat xF
  let y : Tensor α (Shape.Mat seqLen vocab) := Common.castTensor Runtime.ofFloat yF
  sample.mk x y

end NN.Examples.Models.RealData

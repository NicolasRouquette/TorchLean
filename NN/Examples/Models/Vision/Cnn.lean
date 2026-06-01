/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Device-agnostic real-data example:
  python3 scripts/datasets/download_example_data.py --cifar10
  lake exe torchlean cnn --cpu
  lake build -R -K cuda=true && lake exe torchlean cnn --cuda
-/

module


public import NN
public import NN.API.Models.Cnn
public import NN.Examples.Models.Common.RealData

/-!
# CNN Training Example

Runnable `torchlean cnn` example. It trains a small convolutional classifier on a prepared CIFAR-10
minibatch.

The reusable model wiring lives in `NN.API.Models.Cnn` (`nn.models.cnn`). This file is the
runnable wrapper: command-line parsing, dataset selection, step-limited loader training, and
TrainLog artifact writing.

```bash
python3 scripts/datasets/download_example_data.py --cifar10
lake build -R -K cuda=true && lake exe torchlean cnn --cuda --steps 1
```
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.Vision.Cnn

def exeName : String := "torchlean cnn"
def defaultLogJson : System.FilePath := Common.modelZooTrainLog "cnn"

def batch : Nat := 4
def inC : Nat := 3
def inH : Nat := RealData.cifarHeight
def inW : Nat := RealData.cifarWidth

def outDim : Nat := RealData.cifarClasses

def cfg : nn.models.CnnConfig :=
  { batch := batch, inC := inC, inH := inH, inW := inW, outDim := outDim }

abbrev σ : Shape :=
  nn.models.cnnInShape cfg

abbrev τ : Shape :=
  nn.models.cnnOutShape cfg

def mkModel : nn.M (nn.Sequential σ τ) :=
  nn.models.cnn cfg

def main (args : List String) : IO UInt32 := do
  Common.runFloat exeName args
    (banner := fun opts =>
      s!"{exeName}: CNN training (device={if opts.useGpu then "cuda" else "cpu"})")
    (k := fun opts rest => do
      let flags ← Common.orThrow exeName <|
        RealData.parseCifarModelTrainFlags exeName rest defaultLogJson 1 1e-3
      let report ← RealData.fitCifarClassifier exeName "CNN training" batch mkModel opts flags
      Common.printFitReport flags.train.train.steps report)

end NN.Examples.Models.Vision.Cnn

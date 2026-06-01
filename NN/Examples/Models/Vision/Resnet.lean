/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Device-agnostic real-data example:
  python3 scripts/datasets/download_example_data.py --cifar10
  lake exe torchlean resnet --cpu
  lake build -R -K cuda=true && lake exe torchlean resnet --cuda

This runs a medium ResNet-style model built from `API.nn.resnetBasicBlock` (Conv+BN+residual).
-/

module


public import NN
public import NN.API.Models.Resnet
public import NN.Examples.Models.Common.RealData

/-!
# ResNet Real-Data Example

Runnable `torchlean resnet` example. It trains a compact ResNet-style classifier built from
`API.nn.resnetBasicBlock` on a prepared CIFAR-10 minibatch.

The reusable model wiring lives in `NN.API.Models.Resnet` (`nn.models.resnet`). This file is the
runnable wrapper (CIFAR loader construction + step-limited training loop).

```bash
python3 scripts/datasets/download_example_data.py --cifar10
lake build -R -K cuda=true && lake exe torchlean resnet --cuda --n-total 200 --steps 1
```

Tip: the defaults are set for a short validation run. For a longer run:

```bash
lake build -R -K cuda=true
lake exe torchlean resnet --cuda --fast-kernels --n-total 5000 --steps 200
```
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.Vision.Resnet

def exeName : String := "torchlean resnet"
def defaultLogJson : System.FilePath := Common.modelZooTrainLog "resnet"

def batch : Nat := 2
def inC : Nat := 3
def inH : Nat := RealData.cifarHeight
def inW : Nat := RealData.cifarWidth

def stemC : Nat := 8
def numClasses : Nat := RealData.cifarClasses

def cfg : nn.models.ResnetConfig :=
  { batch := batch, inC := inC, inH := inH, inW := inW, stemC := stemC, numClasses := numClasses }

abbrev σ : Shape :=
  nn.models.resnetInShape cfg

abbrev τ : Shape :=
  nn.models.resnetOutShape cfg

def mkModel : nn.M (nn.Sequential σ τ) := do
  nn.models.resnet cfg

def main (args : List String) : IO UInt32 := do
  Common.runFloat exeName args
    (banner := fun opts =>
      s!"{exeName}: ResNet CIFAR training (device={if opts.useGpu then "cuda" else "cpu"})")
    (k := fun opts rest => do
      let flags ← Common.orThrow exeName <|
        RealData.parseCifarModelTrainFlags exeName rest defaultLogJson 1 1e-3
      let report ← RealData.fitCifarClassifier exeName "ResNet CIFAR training" batch mkModel opts flags
      Common.printFitReport flags.train.train.steps report)

end NN.Examples.Models.Vision.Resnet

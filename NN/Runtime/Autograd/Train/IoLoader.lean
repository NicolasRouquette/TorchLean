/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Train.IoLoader.Csv
public import NN.Runtime.Autograd.Train.IoLoader.Npy

/-!
# IO loaders for training datasets

`NN.Runtime.Autograd.Train.IoLoader` is the public umbrella for file-backed training loaders.

The loader surface has three parts:

- `IoLoader.Parsing` contains the parser primitives and safety limits shared by those formats.
- `IoLoader.Csv` contains CSV-to-tensor dataset readers.
- `IoLoader.Npy` contains the supported NumPy `.npy` subset for vectors and matrices.

This umbrella keeps the import path stable while the parsing code stays close to the file format it
checks.
-/

@[expose] public section

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Types
public import NN.Runtime.Autograd.Torch.Core.Session
public import NN.Runtime.Autograd.Torch.Core.Ops
public import NN.Runtime.Autograd.Torch.Core.BackwardOptim
public import NN.Runtime.Autograd.Torch.Core.TypedGraph
public import NN.Runtime.Autograd.Torch.Core.Functional
public import NN.Runtime.Autograd.Torch.Core.Trainer

/-!
# Torch Core

Torch-style runtime front-end for eager execution, typed graphs, and training helpers.

- `Core.Types`: public handles, options, and parameter wrappers.
- `Core.Session`: eager session state, CUDA bridge, and tape lifecycle helpers.
- `Core.Ops`: eager tensor operations.
- `Core.BackwardOptim`: eager backward passes and optimizers.
- `Core.TypedGraph`: reusable shape-indexed graph wrappers.
- `Core.Functional`: operation-generic `Ops` interface and curried syntax.
- `Core.Trainer`: `Ops` instances, parameter packs, and scalar trainer construction.

Most callers can import this module directly to bring the Torch runtime API into scope.
-/

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core
public import NN.Runtime.Autograd.Torch.TypedGraphSession
public import NN.Runtime.Autograd.Torch.Utils

/-!
# Torch-style runtime front-end

This is the public umbrella for the low-level PyTorch-style runtime layer.

The split is intentional:

- `Torch.Core` defines imperative tensor references, parameters, eager sessions, operation wrappers,
  typed executable graphs, and simple scalar trainers.
- `Torch.Internal.TypedGraphSession` is the recorder used to implement imperative typed graph
  execution. Its backpropagation agrees with the runtime tape obtained from the recorded graph.
- `Torch.Utils` contains compact example/training conveniences such as deterministic initializers,
  small sample builders, and trainer loops.

`TorchLean/*` builds the higher-level model and program API on top of this layer. `Torch` contains
the low-level session and reference machinery; `TorchLean` contains the public model API.
-/

@[expose] public section

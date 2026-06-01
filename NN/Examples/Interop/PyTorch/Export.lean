/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Interop.PyTorch.CNN.Export
public import NN.Examples.Interop.PyTorch.MLP.Export
public import NN.Examples.Interop.PyTorch.Transformer.Export

/-!
# PyTorch Example Exporters

Example-specific PyTorch `nn.Module` generators used by the round-trip examples.

These stay outside `NN.Runtime.PyTorch` because they fix particular model shapes and naming
conventions. The runtime bridge owns the general IR and `state_dict` paths; these modules live
beside the MLP/CNN/Transformer reference artifacts.
-/

@[expose] public section

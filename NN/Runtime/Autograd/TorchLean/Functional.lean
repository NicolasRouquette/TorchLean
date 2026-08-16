/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.Functional.Core
public import NN.Runtime.Autograd.TorchLean.Functional.Einsum
public import NN.Runtime.Autograd.TorchLean.Functional.ShapeOps

/-!
# Functional

TorchLean functional helpers in the style of `torch.*` building blocks.

These are derived ops built from the small primitive `TorchLean.Ops` surface, so eager execution and
typed graph construction share the same model and loss definitions.

For background, see the PyTorch documentation for `torch.nn.functional`, `torch.autograd`, and
checkpointing, together with the standard reverse-mode AD references by Linnainmaa and by Griewank
and Walther.
-/

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

-- shake: keep-all

public import NN.API.Runtime.Core
public import NN.API.Runtime.Layers
public import NN.API.Runtime.Autograd
public import NN.API.Runtime.Module
public import NN.API.Runtime.Training

/-!
# TorchLean Runtime API

`NN.API.Runtime` exposes the executable runtime API under `NN.API.TorchLean`: tensor
primitives, functional ops, losses, optimizer configs, sequential model APIs, module execution,
autograd, supervised training, and session-level tools.

Most model code should start from `NN.API.Public`; this module is the right layer when the runtime
itself is the subject.
-/

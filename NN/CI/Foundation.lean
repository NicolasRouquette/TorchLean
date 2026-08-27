/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import NN.API.Data
import NN.API.Init
import NN.API.Neural
import NN.API.Trainer.Scheduler
import NN.GraphSpec
import NN.IR
import NN.Spec

/-!
# Additional Foundation Modules

These imports cover focused API, GraphSpec, and IR modules that remain outside the downstream
umbrella or are useful checks of the subsystem import boundaries.
-/

@[expose] public section

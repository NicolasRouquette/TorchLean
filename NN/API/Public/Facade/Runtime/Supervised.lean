/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.NN

/-!
# TorchLean Supervised Runtime API

Supervised runtime names for manual code.
-/

@[expose] public section

namespace TorchLean

namespace Supervised

export NN.API.TorchLean.Supervised
  (SeqLoss SeqTask paramShapes)
export NN.API.TorchLean.Supervised.SeqTask
  (mse crossEntropyOneHot moduleDef moduleDefWithMode)

end Supervised


end TorchLean

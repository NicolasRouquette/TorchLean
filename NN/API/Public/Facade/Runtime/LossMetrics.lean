/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.NN

/-!
# TorchLean Loss and Metrics API

Public loss reductions, losses, and metrics.
-/

@[expose] public section

namespace TorchLean

namespace Loss

@[inherit_doc NN.API.TorchLean.Loss.Reduction]
abbrev Reduction := NN.API.TorchLean.Loss.Reduction

export NN.API.TorchLean.Loss
  (mse
   nllOneHot crossEntropyOneHot
   nllIndex nllNat crossEntropyIndex crossEntropyNat
   bceWithLogits bce)

end Loss

namespace Metrics

export NN.API.TorchLean.Metrics (argmax? classOfOneHot? correctOneHot?)

end Metrics


end TorchLean

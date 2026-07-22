/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.NN

/-!
# TorchLean Runtime Tensor-Pack API

Public tensor-pack names used by runtime layer examples.
-/

@[expose] public section

namespace TorchLean

namespace tensorpack

export NN.API.tensorpack
  (singleton pair triple quad map zipWith append split
   unpackSingleton unpackPair unpackTriple unpackQuad
   first second third fourth)

end tensorpack


end TorchLean

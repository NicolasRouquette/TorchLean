/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Json

/-!
# TorchLean JSON API

Small JSON operations used by public artifact and interop examples.
-/

@[expose] public section

namespace TorchLean

namespace Json

export NN.API.Json
  (fail expectObjE expectFieldE expectStringE expectNatE expectArrayE optionalBoolFieldE
   expectNatArrayE parseFile)

end Json

end TorchLean

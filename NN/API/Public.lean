/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

-- shake: keep-all

public import NN.API.Public.NN
public import NN.API.Public.TensorPack
public import NN.API.Public.Seeded
public import NN.API.Public.Autograd

/-!
# Public API Implementation

Model builders, seeded construction, tensor packs, and autograd operations used to implement the
`TorchLean.*` API.

Application code should use the focused public import:

```lean
import NN.API
open TorchLean
```

The application entrypoint is `NN.API`, which exposes these operations through the short public
names:

- `TorchLean.nn` (model/layer builders)
- `TorchLean.optim` (optimizer configs)
- `TorchLean.Trainer` (train/evaluate APIs)
- `TorchLean.Data` (datasets/loaders + CSV/NPY readers)
- `TorchLean.Loss` and `TorchLean.Metrics`
- `TorchLean.classical` (classical and statistical models)

Import `NN.API.Public` only when extending this implementation layer. Application code should use
`NN.API`.

The callback-heavy training namespace lives in `NN.API.Public.Training`. It is deliberately not
re-exported from this umbrella module: ordinary code should get training through `TorchLean.Trainer`,
while files that need callback runners should import the advanced training
module explicitly.
-/

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.Trainer.Core
public import NN.API.Public.Facade.Trainer.Constructor
public import NN.API.Public.Facade.Trainer.Results
public import NN.API.Public.Facade.Trainer.Run
public import NN.API.Public.Facade.Trainer.Train
public import NN.API.Public.Facade.Trainer.Predict

/-!
# TorchLean Trainer API

Import entrypoint for the public training API:

```lean
let trainer := Trainer.new model
  { task := .regression
    optimizer := optim.adam { lr := 0.03 } }
let y0 ← trainer.predict x
let trained ← trainer.train data { steps := 200, batchSize := 16, logEvery := 25 }
trained.printSummary
```

Implementation is split into focused modules under `NN.API.Public.Facade.Trainer`.
-/

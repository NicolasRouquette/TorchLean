/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Core
public import NN.API.Trainer.Dataset
public import NN.API.Trainer.Constructor
public import NN.API.Trainer.FixedSample
public import NN.API.Trainer.Manual
public import NN.API.Trainer.Results
public import NN.API.Trainer.Run
public import NN.API.Trainer.Scheduler
public import NN.API.Trainer.Train
public import NN.API.Trainer.Predict
public import NN.API.Trainer.Reporting

/-!
# Training

The main training interface:

```lean
let trainer := Trainer.new model
  { task := .regression
    optimizer := optim.adam { lr := 0.03 } }
let y0 ← trainer.predict x
let trained ← trainer.train data { steps := 200, batchSize := 16, logEvery := 25 }
trained.printSummary
```

The same interface supports regression, classification, custom losses, finite datasets, and
streaming batches.
-/

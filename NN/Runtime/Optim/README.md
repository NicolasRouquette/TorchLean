# Runtime Optimizers

This folder owns the executable optimizer and scheduler equations used by TorchLean training loops.
The formulas are pure functions over typed tensors; runtime training code lifts them to parameter
lists, optimizer state maps, checkpoints, and public trainer configuration.

The design is data-first rather than inheritance-first. Each optimizer carries exactly the state its
update equation needs: SGD stores a learning rate, momentum SGD stores a buffer, Adam/AdamW store
moment buffers and a step counter, Adadelta stores gradient/update EMAs, and Muon stores the
orthogonalizer backend used to turn momentum into the update direction. GaLore lives beside the
optimizers as projected-gradient machinery: it stores the projection backend and then applies a
named optimizer to the projected gradient.

## Files

- `Optimizers.lean`: per-tensor update equations for SGD, momentum SGD, AdaGrad, RMSProp, Adam,
  AdamW, Adadelta, Muon-style orthogonalized momentum, and GaLore-style projected updates.
- `Schedulers.lean`: deterministic learning-rate schedules, including constant, step, exponential,
  cosine, cyclic, one-cycle, and PyTorch-compatible variants where step-count conventions matter.
- `GradientUtils.lean`: gradient clipping and utility operations shared by trainer code.

## Public API

Most users should reach standard trainer optimizers through the application API:

```lean
import NN
open TorchLean

let opt := optim.adamw
let sched := schedulers.cosine
```

The high-level trainer config exposes SGD, momentum SGD, AdaGrad, RMSProp, Adam, AdamW, and
Adadelta. Optimizer-adjacent extension points use explicit runtime names:

- `optim.runtimeMuon` is the runtime Muon optimizer. It requires an orthogonalization backend
  because the backend output is part of the mathematical update.
- `optim.galore.projectedSGD` is the public GaLore-style projected-gradient path. The projection is
  explicit, and the optimizer applied after projection is still named.

The exact public names live under `NN/API/Public/Facade/Runtime/Optim.lean`,
`TorchLean.optim` and `NN/API/TorchLean/Schedulers.lean`. Runtime implementation
files should stay focused on the update equations and state transitions.

## Proof Boundary

The proof layer optimizer interface lives in `NN/MLTheory/Optimization/OptimizerLaws.lean`. When a
new optimizer is added, the intended pattern is:

1. define the pure state and update equation here;
2. expose it through the optimizer API if it is user-facing;
3. register it as a `TensorOptimizer` in the theory layer;
4. prove the algebraic laws or reduction facts that make the optimizer reusable in larger proofs.

This keeps runtime performance work, public API ergonomics, and theorem statements connected
without making any one layer own all three jobs.

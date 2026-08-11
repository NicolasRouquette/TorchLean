# Quickstart

This folder is the starting path for TorchLean. It focuses on the moves people copy first:

- build typed tensors,
- inspect values with editor widgets,
- use autograd APIs,
- train small MLP and CNN models,
- and see how ordinary Lean proofs fit into the workflow.

New code should start from the root API:

```lean
import NN.API
open TorchLean
```

The quickstarts should not make you learn subsystem namespaces first. Use `TorchLean.nn`,
`TorchLean.Tensor`, `TorchLean.autograd`, `TorchLean.Trainer`, and `TorchLean.Data` here. Drop into
`NN.*` when you are extending TorchLean itself, proving runtime facts, or working with subsystem
entrypoints.

For larger architectures, use `NN/Examples/Models`. For datasets and loaders backed by files, use
`NN/Examples/Data/Loaders`. Runnable verification examples and bundled certificate artifacts live
under `NN/Examples/Verification`; reusable checkers live under `NN/Verification`.

## Recommended Order

1. `TensorBasics.lean`
   Command: `lake exe torchlean quickstart_tensors`
   Start here if you are new to the syntax. The file shows how TorchLean writes shaped literals,
   how inferred shapes appear in ordinary code, and how tensor printing stays separate from the
   proof-level scalar models.

2. `StarterWorkflow.lean`
   Build check: `lake build NN.Examples.Quickstart.StarterWorkflow`
   This is the API shape we want people to copy first: `import NN.API`, `nn.Sequential![...]`,
   `Data.tensorDataset xs ys`, `Trainer.new model { task := .regression, optimizer := ... }`,
   `trainer.predict x`, `trainer.train data { steps := ..., batchSize := ..., logEvery := ... }`,
   and then one trained-handle prediction plus one $\ell_\infty$ IBP verification call. The task field chooses
   the loss kind: regression means MSE, classification means one-hot cross entropy.
   The example is small enough to read in one sitting and already has the ownership pattern used by
   the larger examples: the trainer owns parameters, optimizer state, backend selection, and
   trained-result methods.

3. `Widgets.lean`
   Editor-only: open the file and put the cursor on the `#tensor_view`, `#ir_view`, or
   `#train_log_view` commands.
   Widgets are inspection tools. They make tensors, graphs, and logs easier to read, but they do not
   change the proof status of an object.

4. `AutogradBasics.lean`
   Command: `lake exe torchlean quickstart_autograd`
   This uses `TorchLean.autograd` directly, so the example shows VJP/Jacobian/HVP without exposing
   the runtime callback machinery.
   Use it to understand the executable differentiation API before reading the autograd proof files
   under `NN/Proofs`.

5. `SimpleMlpTrain.lean`
   Command: `lake exe torchlean quickstart_mlp --steps 200 --dtype float32 --backend compiled`
   Alternate trusted-runtime run: `lake exe torchlean quickstart_mlp --steps 200 --dtype float --backend compiled`
   This is the smallest training loop with command-line scalar/backend choices. It is the right
   place to check that the public trainer path still feels like one model rather than many backend
   functions.

6. `MinibatchMlpTrain.lean`
   Command: `lake exe torchlean quickstart_minibatch_mlp --steps 30 --batch 5 --dtype float --backend eager`
   This adds deterministic minibatching and a trained handle for batched follow-up predictions.

7. `SimpleCnnTrain.lean`
   Command: `lake exe torchlean quickstart_cnn --steps 5 --batch 2 --dtype float --backend eager`
   This moves from vector inputs to image-shaped tensors and convolutional layers without changing
   the public trainer pattern.

8. `Proofs.lean`
   Build check: `lake build NN.Examples.Quickstart.Proofs`
   This is the first quickstart that is about Lean propositions rather than runtime output. It keeps
   the proof examples small so the connection between a model-shaped object and a theorem statement
   is visible.

## Shared Conventions

The quickstarts use the same public conventions:

- user code starts with `import NN.API` and `open TorchLean`;
- tensors carry enough shape information to make common mistakes visible;
- training code goes through `Trainer.new`, `trainer.train`, and trained handles;
- backend and dtype choices are configuration, not separate public forward functions;
- runtime checks and proofs are both useful, but they answer different questions.

Once those moves are familiar, the model zoo and verification examples add scale, file-backed data,
external artifacts, CUDA runs, and theorem-backed checkers.

## Where The Bigger Examples Live

The quickstarts use `NN.API` and keep ordinary training behind `Trainer` and
`Data`. Larger architectures, real datasets, PyTorch interop, RL, and verification examples live in
their specialized folders under `NN/Examples`.

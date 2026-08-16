/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Widgets
public import NN.Floats.IEEEExec.Exec32
public import NN.IR.Graph
public import NN.API

/-!
# Quickstart: Widgets

TorchLean widgets are editor-side inspection tools. They render values already present in Lean
without changing runtime semantics or proofs.

Try these commands in the editor:

- put the cursor on a `#tensor_view`, `#float32_view`, `#ir_view`, or `#train_log_view` command;
- put the cursor on `#pytorch_translate_file` to preview a PyTorch-to-TorchLean skeleton;
- Lean's infoview renders an interactive panel;
- if you want the full gallery, open `NN.Examples.DeepDives.Widgets`.

This quickstart keeps only the smallest useful examples; the full widget gallery lives in
`NN.Examples.DeepDives.Widgets`.
-/

@[expose] public section

namespace NN.Examples.Quickstart.Widgets

open TorchLean
open TorchLean.Floats.IEEE754

/-- A small vector, built with the same typed tensor constructor used in ordinary code. -/
def vector : Tensor Float (shape![4]) :=
  tensor! [1.0, 2.0, 3.0, 4.0]

/-- A small matrix where the shape is visible both in the type and in the widget. -/
def matrix : Tensor Int (shape![2, 3]) :=
  tensor! [
    [1, 2, 3],
    [4, 5, 6]
  ]

/-- A binary32 value; the widget shows sign/exponent/fraction fields and classification flags. -/
def one32 : IEEE32Exec :=
  IEEE32Exec.ofFloat 1.0

/-- A small IR graph: input plus constant, then an add node. -/
def addGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input, outShape := shape![2] },
      { id := 1, parents := [], kind := .const shape![2], outShape := shape![2] },
      { id := 2, parents := [0, 1], kind := .add, outShape := shape![2] }
    ] }

/-- A minimal training log; runtime examples can write the same structure as JSON. -/
def tinyTrainLog : Training.TrainLog :=
  { title := "Quickstart loss"
    steps := #[0, 1, 2, 3]
    series := #[
      { name := "loss", values := #[1.0, 0.45, 0.22, 0.12], color := "#c44" }
    ]
    notes := #["editor-only visualization; runtime training logs use the same schema"] }

/-!
The commands below render editor panels through ProofWidgets. Deleting them leaves the same
tensors, graphs, logs, Python source files, and proofs behind.
-/

#tensor_view vector
#tensor_view matrix
#tensor_stats_view vector

#float32_view one32
#float32_round_view (0.1 : Float)

#ir_view addGraph
#shape_infer_view addGraph

/-!
The translator widget is placed next to the IR/shape widgets. It is a bounded-scope
preview for "what would this PyTorch layer stack look like in TorchLean?" The checked graph-capture
path is still the `torch.export` importer, which parses and validates explicit IR JSON.
-/
#pytorch_translate_file "NN/Examples/Quickstart/pytorch_translator_mlp.py"

#train_log_view tinyTrainLog

end NN.Examples.Quickstart.Widgets

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness.Eval.BatchNorm
public import NN.Verification.TorchLean.Proved.Correctness.Eval.Concat
public import NN.Verification.TorchLean.Proved.Correctness.Eval.MiscOps
public import NN.Verification.TorchLean.Proved.Correctness.Eval.Reductions
public import NN.Verification.TorchLean.Proved.Correctness.Eval.Softmax
public import NN.Verification.TorchLean.Proved.Correctness.Eval.Transpose

/-!
# IR Evaluation Coverage

This file is a proof-facing checklist for the local `Graph.evalAt` bridge lemmas.

The bridge lemmas themselves live in the operation-specific files.  Here we keep a small,
machine-checked summary of the IR operation tags currently covered by those local evaluator facts.
When a new `OpKind` constructor is added, this list should be updated together with the
corresponding `evalAt` theorem.
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open NN.IR

namespace Correctness

namespace IRStep

/-- Canonical representative for each IR constructor covered by a local `Graph.evalAt` theorem. -/
def evalAtCoverageWitnesses : List OpKind :=
  [
    .input,
    .const .scalar,
    .permute [],
    .detach,
    .randUniform 0,
    .bernoulliMask 0,
    .add,
    .sub,
    .mul_elem,
    .abs,
    .sqrt,
    .inv,
    .maxElem,
    .minElem,
    .maxPool2d 1 1 1,
    .maxPool2dPad 1 1 1 0,
    .avgPool2d 1 1 1,
    .avgPool2dPad 1 1 1 0,
    .broadcastTo .scalar .scalar,
    .reduceSum 0,
    .reduceMean 0,
    .sum,
    .matmul,
    .linear,
    .conv2d 1 1 1 1 1 0,
    .batchNorm2dNchwEval 1,
    .relu,
    .tanh,
    .sigmoid,
    .exp,
    .log,
    .sin,
    .cos,
    .softmax 0,
    .layernorm 0,
    .reshape .scalar .scalar,
    .flatten .scalar,
    .concat 0,
    .swap_first_two,
    .transpose3dLastTwo,
    .mseLoss
  ]

/-- Tags for the IR constructors covered by local `Graph.evalAt` bridge lemmas. -/
def evalAtCoverageTags : List String :=
  evalAtCoverageWitnesses.map OpKind.tag

/--
Executable checkpoint for the current local evaluator bridge surface.

This is not a semantic equivalence theorem.  It is a maintained coverage guard: the list names the
IR constructor families that have local `Graph.evalAt` lemmas in this directory.
-/
theorem evalAtCoverageTags_eq :
    evalAtCoverageTags =
      [
        "input",
        "const",
        "permute",
        "detach",
        "rand_uniform",
        "bernoulli_mask",
        "add",
        "sub",
        "mul_elem",
        "abs",
        "sqrt",
        "inv",
        "max_elem",
        "min_elem",
        "max_pool2d",
        "max_pool2d_pad",
        "avg_pool2d",
        "avg_pool2d_pad",
        "broadcastTo",
        "reduce_sum",
        "reduce_mean",
        "sum",
        "matmul",
        "linear",
        "conv2d",
        "batch_norm2d_nchw_eval",
        "relu",
        "tanh",
        "sigmoid",
        "exp",
        "log",
        "sin",
        "cos",
        "softmax",
        "layernorm",
        "reshape",
        "flatten",
        "concat",
        "swap_first_two",
        "transpose3d_last_two",
        "mse_loss"
      ] := by
  rfl

/-- Number of IR constructor families with local `Graph.evalAt` bridge coverage. -/
theorem evalAtCoverageWitnesses_length :
    evalAtCoverageWitnesses.length = 41 := by
  rfl

/-- The coverage checklist records each covered IR constructor tag once. -/
theorem evalAtCoverageTags_nodup :
    evalAtCoverageTags.Nodup := by
  decide

/--
Every current IR constructor family has an entry in the local evaluator bridge checklist.

The statement quantifies over `OpKind`, not only over the representative list above. If a new
constructor is added to the IR, this theorem stops compiling until the bridge coverage surface is
reviewed.
-/
theorem evalAtCoverageTags_complete (kind : OpKind) :
    kind.tag ∈ evalAtCoverageTags := by
  cases kind <;> simp [evalAtCoverageTags, evalAtCoverageWitnesses, OpKind.tag]

/--
The checklist is exactly the set of current IR constructor tags.

Together with `evalAtCoverageTags_nodup`, this keeps the coverage summary honest in both
directions: no missing constructor family and no stale tag unrelated to an `OpKind`.
-/
theorem evalAtCoverageTags_iff (tag : String) :
    tag ∈ evalAtCoverageTags ↔ ∃ kind : OpKind, kind.tag = tag := by
  constructor
  · intro h
    unfold evalAtCoverageTags at h
    rcases List.mem_map.mp h with ⟨kind, _hKind, hTag⟩
    exact ⟨kind, hTag⟩
  · rintro ⟨kind, rfl⟩
    exact evalAtCoverageTags_complete kind

end IRStep

end Correctness

end NN.Verification.TorchLean.Proved

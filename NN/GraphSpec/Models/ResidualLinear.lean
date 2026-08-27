/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.DAG

/-!
# Residual Linear Block

This is the smallest “ResNet-like” example in the directory, and it is intentionally chosen to be
easy to read.

It shows the structural reason we need the DAG IR without dragging in convolution arithmetic:

$$
\begin{aligned}
y &= \operatorname{Linear}(x),\\
\mathrm{out} &= \operatorname{ReLU}(y+x).
\end{aligned}
$$

Because `x` is consumed by both the main path and the skip path, a pure chain would have to
recompute the input path or hide sharing inside a special-purpose combinator. In `GraphSpec.DAG` we
express the sharing directly with `let1`.

This file is best read as a “hello world” for DAG-authored GraphSpec examples:

- one explicit parameter ABI,
- one shared intermediate,
- one multi-input primitive (`add`),
- one final nonlinearity.

References / citations:
- He et al. (2016), “Deep Residual Learning for Image Recognition” (ResNets).
- `NN.GraphSpec.DAG.Core` for the term language and semantics.
-/

@[expose] public section


namespace NN
namespace GraphSpec
namespace Models

open _root_.Spec
open Spec.Tensor
open _root_.TorchLean.Tensor
open NN.GraphSpec.DAG

/--
Parameter ABI for the residual block.

The layout is exactly:

- `W : Tensor α [d, d]`
- `b : Tensor α [d]`

The parameter-free skip path reuses the input `x`.
-/
abbrev ResidualLinearParams (d : Nat) : List Shape :=
  [[d, d], [d]]

/--
Residual linear block in DAG form.

In ordinary math notation, this is

$$
x\mapsto\operatorname{ReLU}(Wx+b+x).
$$

This is a good first DAG example because the only genuinely DAG-specific feature is sharing the
input between the main branch and the skip branch.
-/
def residualLinear (d : Nat) :
    DAG.Model (ps := ResidualLinearParams d) (ins := [[d]]) (τ := [d]) :=
  let Γ : List Shape :=
    [[d, d], [d], [d]]
  let w : DAG.Term Γ [d, d] :=
    DAG.Term.var (Γ := Γ) .head
  let b : DAG.Term Γ [d] :=
    DAG.Term.var (Γ := Γ) (.tail .head)
  let x : DAG.Term Γ [d] :=
    DAG.Term.var (Γ := Γ) (.tail (.tail .head))
  let y : DAG.Term Γ [d] :=
    DAG.Term.op (Γ := Γ) (DAG.PrimOp.linear (inDim := d) (outDim := d))
      (DAG.Args.cons w (DAG.Args.cons b (DAG.Args.cons x (DAG.Args.nil))))
  { initParams :=
      -- Deterministic, simple init: all zeros.
      let W0 : Spec.Tensor Float [d, d] := Spec.zeros (α := Float) [d, d]
      let b0 : Spec.Tensor Float [d] := Spec.zeros (α := Float) [d]
      .cons W0 (.cons b0 .nil)
    body :=
      DAG.Term.let1 y <|
        let Γ' : List Shape :=
          [[d, d], [d], [d], [d]]
        let yv : DAG.Term Γ' [d] :=
          DAG.Term.var (Γ := Γ') (.tail (.tail (.tail .head)))
        let add : DAG.Term Γ' [d] :=
          DAG.Term.op (Γ := Γ') (DAG.PrimOp.add (s := [d]))
            (DAG.Args.cons yv
              (DAG.Args.cons
                (DAG.Term.var (Γ := Γ') (.tail (.tail .head)))
                (DAG.Args.nil)))
        DAG.Term.op (Γ := Γ') (DAG.PrimOp.relu (s := [d])) (DAG.Args.cons add
          (DAG.Args.nil))
  }

end Models
end GraphSpec
end NN

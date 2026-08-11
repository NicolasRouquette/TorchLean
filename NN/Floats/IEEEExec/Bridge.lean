/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Bridge.ERealTotal
public import NN.Floats.IEEEExec.Bridge.Expressions
public import NN.Floats.IEEEExec.Bridge.FP32
public import NN.Floats.IEEEExec.Bridge.FP32Total
public import NN.Floats.IEEEExec.Bridge.LeanFloat32

/-!
# Bridges From Executable Binary32

This umbrella collects the refinement layers around `IEEE32Exec`: finite rounded-real semantics,
total special-value semantics, expression-level composition, extended-real interpretation, and the
bridge to Lean's logical `Float32.Model`. Native execution remains a separate backend
boundary.
-/

@[expose] public section

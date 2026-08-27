/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Bridge.FP32Total.MinMax

/-!
# Total FP32 Refinement Bridge

The modules exported here connect executable binary32 operations to real-valued rounding formulas
while retaining explicit NaN and infinity branches.
-/

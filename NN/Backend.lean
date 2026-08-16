/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Types
public import NN.Backend.Capsule
public import NN.Backend.Availability
public import NN.Backend.Target
public import NN.Backend.Planner
public import NN.Backend.Audit
public import NN.Backend.Recheck
public import NN.Backend.Attention
public import NN.Backend.NativeCUDA
public import NN.Backend.Reference
public import NN.Backend.LibTorch
public import NN.Backend.Registry
public import NN.Backend.IR
public import NN.Backend.Grouping
public import NN.Backend.Gate
public import NN.Backend.Accepted
public import NN.Backend.Profile
public import NN.Backend.Report

/-!
# Backend Contracts

Contract-carrying backend vocabulary for TorchLean runtimes.

The semantic graph and specs stay in Lean. Fast providers such as native CUDA, LibTorch, ATen,
cuBLAS, cuDNN, or cuFFT enter through named capsules that record shape/layout/value/VJP contracts
and an explicit trust level. The planner consumes these capsules under an `KernelPolicy` and
selects an admissible execution provider for each operation or fused operation group.
-/

@[expose] public section

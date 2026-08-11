/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Proofs.Approximation
public import NN.MLTheory.Proofs.Hopfield
public import NN.MLTheory.Proofs.ReLU
public import NN.MLTheory.Proofs.StateSpace
public import NN.MLTheory.Proofs.Verification

/-!
# MLTheory proofs

This is the curated entrypoint for theorem-heavy MLTheory developments. It groups the proof files
by mathematical theme:

- approximation and finite-precision universal approximation;
- Hopfield energy descent and convergence;
- ReLU algebra and compact-set approximation;
- state-space / Mamba scan and causality laws; and
- robustness theorems used by verification workflows.

Model and specification modules define the semantics; these imports collect reusable mathematical
properties of those definitions.
-/

@[expose] public section

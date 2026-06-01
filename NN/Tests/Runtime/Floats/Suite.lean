/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tests.Runtime.Floats.AllAutogradTests
public import NN.Tests.Runtime.Floats.CertificatePreconditions
public import NN.Tests.Runtime.Floats.IRBatchNorm
public import NN.Tests.Runtime.Floats.IRShapeContracts
public import NN.Tests.Runtime.Floats.KnnCheck
public import NN.Tests.Runtime.Floats.ModelsCheck
public import NN.Tests.Runtime.Floats.ONNXBridge
public import NN.Tests.Runtime.Floats.PINNDerivResidual
public import NN.Tests.Runtime.Floats.PyTorchRoundtripParity
public import NN.Tests.Runtime.Floats.RLCheck
public import NN.Tests.Runtime.Floats.RnnGruLstmBpttCheck
public import NN.Tests.Runtime.Floats.TorchLeanIRExecEquivCheck
public import NN.Tests.Runtime.Floats.TorchLeanIndexShapeCheck
public import NN.Tests.Runtime.Floats.TorchLeanOpsCheck
public import NN.Tests.Runtime.Floats.TorchLeanSpecMlpEquivCheck

/-!
# Suite

Aggregates the float runtime and autograd test suites.

These are runtime checks for regressions in the executable float backends
and keep public examples from silently breaking. They complement the proof modules: tests cover
runtime wiring, floating-point behavior, parser glue, and execution paths that sit outside the
kernel of Lean theorems.
-/

@[expose] public section

namespace Tests
namespace Floats

/-- Unified Float test entrypoint (called by `NN/Tests/Suite.lean`). -/
def run : IO Unit := do
  Tests.Floats.runAllAutogradTests
  Tests.Floats.CertificatePreconditions.run
  Tests.Floats.IRBatchNorm.run
  Tests.Floats.IRShapeContracts.run
  Tests.Floats.ModelsCheck.run
  Tests.Floats.ONNXBridge.run
  Tests.Floats.PyTorchRoundtripParity.run
  Tests.Floats.KNN.run
  Tests.Floats.RLCheck.run
  Tests.Floats.BPTT.run
  Tests.Floats.PinnDerivResidual.run
  Tests.Floats.TorchLeanOpsCheck.run
  Tests.Floats.TorchLeanIndexShapeCheck.run
  Tests.Floats.TorchLeanSpecMLPEquivCheck.run
  Tests.Floats.TorchLeanIRExecEquivCheck.run

end Floats
end Tests

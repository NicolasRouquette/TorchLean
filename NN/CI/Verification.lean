/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Cert.AbCrownLeafCert
public import NN.Verification.PINN.CLI
public import NN.Verification.PINN.Certificate
public import NN.Verification.PINN.Core
public import NN.Verification.PINN.DatasetCheck
public import NN.Verification.PINN.PdeAst
public import NN.Verification.PINN.PdeParse
public import NN.Verification.PINN.ResidualAffine
public import NN.Verification.Robustness.Digits

public import NN.Verification.Cert.CROWNNodeCert
public import NN.Verification.Cert.CROWNNodeCertAlphaBeta
public import NN.Verification.Cert.IBPCert
public import NN.Verification.Cert.IBPNodeCert
public import NN.Verification.ODE.Ast
public import NN.Verification.ODE.Parse
public import NN.Verification.PINN.Architecture
public import NN.Verification.TorchLean.Lowering
public import NN.Verification.TorchLean.ExecutableLowering
public import NN.Verification.TorchLean.Correctness
public import NN.Verification.TorchLean.Proved
public import NN.Verification.TorchLean.SpecEval
public import NN.Verification.Util.Array
public import NN.Verification.Util.FloatApprox
public import NN.Verification.Util.Json
public import NN.Verification.Util.Tensor

/-!
# Verification CI Suite

Focused CI import suite. `NN.CI.All` combines every suite for exhaustive repository validation.

Local usage:

```bash
lake build NN.CI.All
```
-/

@[expose] public section

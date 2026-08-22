/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Profile
public import NN.Floats.Interval.IEEEExec32
public import NN.IR.Graph
public import NN.Proofs.RuntimeApprox.Graph.NumericalCertificate
-- The ordinary import exposes certificate data to compiled definitions; this second import keeps
-- executable IR for the `#eval` report at the end of the example.
public meta import NN.Proofs.RuntimeApprox.Graph.NumericalCertificate

/-!
# A graph numerical certificate

This example certifies a four-node scalar graph:

```text
x in [1, 2]       c in [0.5, 1]
       \             /
        y = x + c
             |
        z = y * c
```

The source intervals are executable binary32 endpoints. The checker propagates them with directed
rounding, rejects non-finite intermediate intervals, and records the backend capsules selected when
the portable CPU profile replans the graph. The resulting range trace is an executable check; a
`CheckedRealExecution` supplies the separate proof that an exact-real execution is enclosed. This
file demonstrates the artifact-generation and replay path that a larger graph uses.
-/

@[expose] public section

namespace NN.Examples.DeepDives.Floats.GraphNumericalCertificate

open Proofs.RuntimeApprox.NumericalCertificate
open _root_.Spec
open TorchLean.Floats.IEEE754

/-- The example uses scalar nodes so the interval endpoints remain easy to inspect. The certificate
machinery itself stores only one scalar hull per tensor and is independent of tensor rank. -/
def graph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input, outShape := .scalar },
      { id := 1, parents := [], kind := .const .scalar, outShape := .scalar },
      { id := 2, parents := [0, 1], kind := .add, outShape := .scalar },
      { id := 3, parents := [2, 1], kind := .mul_elem, outShape := .scalar }
    ] }

def interval (lo hi : UInt32) : IEEE32Exec.Interval32 :=
  { lo := IEEE32Exec.ofBits lo, hi := IEEE32Exec.ofBits hi }

/-- Did an executable certificate operation return a checked value? -/
def accepted {α : Type} : Except String α -> Bool
  | .ok _ => true
  | .error _ => false

/-- Input and constant assumptions. Hexadecimal endpoints preserve the exact binary32 artifact. -/
def sources : Array SourceRange := #[
  { nodeId := 0, enclosure := interval 0x3f800000 0x40000000 },
  { nodeId := 1, enclosure := interval 0x3f000000 0x3f800000 }
]

/-- Generate and replay the range trace and the selected kernel plan. -/
def checked : Except String CheckedCertificate :=
  generateChecked NN.Backend.BackendProfile.checkedCpu graph sources

/-- Concrete payload used for bit-level replay. The constant is `0.75`, which lies in the declared
constant range $[0.5,1]$. -/
def payload : NN.IR.Payload IEEE32Exec where
  const? := fun nodeId =>
    if nodeId = 1 then
      some
        { n := 1
          v := .dim (fun _ => .scalar (IEEE32Exec.ofBits 0x3f400000)) }
    else
      none

/-- A concrete input (`1.25`) inside the declared input interval. -/
def input : Spec.PackedTensor IEEE32Exec :=
  Spec.PackedTensor.mk .scalar (.scalar (IEEE32Exec.ofBits 0x3fa00000))

/-- Replay the same graph using the bit-level IEEE32 interpreter and check every intermediate. -/
def replay : Except String CheckedExecution := do
  let certificate <- checked
  executeIEEE32 payload input certificate

/-- Deliberately replace the addition range with $[0,0]$. This models a corrupted or optimistic
external artifact; replay must not accept it merely because $[0,0]$ is itself a valid interval. -/
def tampered : Except String GraphNumericalCertificate := do
  let raw <- generate NN.Backend.BackendProfile.checkedCpu graph sources
  let addition <- match raw.ranges[2]? with
    | some row => pure row
    | none => throw "example certificate is missing its addition row"
  let claimed := { addition with enclosure := interval 0x00000000 0x00000000 }
  pure { raw with ranges := raw.ranges.set! 2 claimed }

/-- Check the deliberately corrupted artifact against the canonical graph transfers. -/
def tamperedCheck : Except String CheckedCertificate := do
  let raw <- tampered
  check NN.Backend.BackendProfile.checkedCpu graph raw

/-- A finite interval attached to an arithmetic node is still an invalid source assumption. -/
def misplacedSourceCheck : Except String GraphNumericalCertificate :=
  generate NN.Backend.BackendProfile.checkedCpu graph <|
    sources.push { nodeId := 2, enclosure := interval 0x00000000 0x3f800000 }

/-- Registries are deterministic maps: registering a second source contract is rejected. -/
def duplicateContractCheck : Except String GraphRangeRegistry := do
  let registry <- defaultRegistry
  registry.register sourceContract

/-- A certificate is bound to the named operation registry used to derive its transfer rows. -/
def registryMismatchCheck : Except String CheckedCertificate := do
  let raw <- generate NN.Backend.BackendProfile.checkedCpu graph sources
  let registry <- defaultRegistry
  let renamed := { registry with name := "example.incompatible-registry" }
  checkWith renamed NN.Backend.BackendProfile.checkedCpu graph raw

/-! ## Coverage before propagation

Coverage is checked after any architecture has lowered to the common IR. An architecture using only
registered primitives needs no architecture-specific checker. A new primitive is rejected with its
node id and operation name until a local range contract is registered.
-/

/-- The base graph is completely covered by the built-in range registry. -/
def baseCoverage : Except String NumericalCoverageReport := do
  let registry <- defaultRegistry
  requireNumericalCoverage registry graph

/-- Exponential is executable in the graph IR, but it intentionally has no built-in interval
transfer yet. This graph demonstrates that unsupported numerical semantics fail explicitly. -/
def unsupportedGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input, outShape := .scalar },
      { id := 1, parents := [0], kind := .exp, outShape := .scalar }
    ] }

/-- Coverage failure occurs before certificate propagation begins. -/
def unsupportedCoverage : Except String NumericalCoverageReport := do
  let registry <- defaultRegistry
  requireNumericalCoverage registry unsupportedGraph

/-! ## A fixed-order reduction

Reduction order is part of the backend audit because floating-point addition is not associative.
The portable profile advertises the same left fold used by `Tensor.sumSpec`, so the checker can
propagate this reduction directly. Native CUDA's implementation-dependent reduction policy is not
silently treated as the same computation.
-/

def reductionGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input,
        outShape := .dim 3 .scalar },
      { id := 1, parents := [0], kind := .sum, outShape := .scalar }
    ] }

def reductionSources : Array SourceRange := #[
  { nodeId := 0, enclosure := interval 0xbf800000 0x40000000 }
]

def reductionInput : Spec.PackedTensor IEEE32Exec :=
  Spec.PackedTensor.mk (.dim 3 .scalar) <| .dim fun i =>
    match i.1 with
    | 0 => .scalar (IEEE32Exec.ofBits 0x3f800000)
    | 1 => .scalar (IEEE32Exec.ofBits 0xbf000000)
    | _ => .scalar (IEEE32Exec.ofBits 0x40000000)

def reductionReplay : Except String CheckedExecution := do
  let certificate <-
    generateChecked NN.Backend.BackendProfile.checkedCpu reductionGraph reductionSources
  executeIEEE32 {} reductionInput certificate

/-! ## Matrix accumulation

The same reduction policy governs matrix multiplication. Each output entry is a fixed-left sum of
products in the portable profile, so the checker combines outward-rounded multiplication with the
existing sum transfer. CUDA profiles advertise an implementation-dependent accumulation and are
not accepted by this particular transfer.
-/

def matmulGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input,
        outShape := .dim 2 (.dim 2 .scalar) },
      { id := 1, parents := [], kind := .const (.dim 2 (.dim 2 .scalar)),
        outShape := .dim 2 (.dim 2 .scalar) },
      { id := 2, parents := [0, 1], kind := .matmul,
        outShape := .dim 2 (.dim 2 .scalar) }
    ] }

def matmulSources : Array SourceRange := #[
  { nodeId := 0, enclosure := interval 0xbf800000 0x3f800000 },
  { nodeId := 1, enclosure := interval 0x00000000 0x3f800000 }
]

def matmulPayload : NN.IR.Payload IEEE32Exec where
  const? := fun nodeId =>
    if nodeId = 1 then
      some
        { n := 4
          v := .dim fun i =>
            match i.1 with
            | 0 | 3 => .scalar IEEE32Exec.posOne
            | _ => .scalar IEEE32Exec.posZero }
    else
      none

def matmulInput : Spec.PackedTensor IEEE32Exec :=
  Spec.PackedTensor.mk (.dim 2 (.dim 2 .scalar)) <| .dim fun i => .dim fun j =>
    if i = j then .scalar IEEE32Exec.posOne else .scalar IEEE32Exec.negOne

def matmulReplay : Except String CheckedExecution := do
  let certificate <-
    generateChecked NN.Backend.BackendProfile.checkedCpu matmulGraph matmulSources
  executeIEEE32 matmulPayload matmulInput certificate

/-- Attempt to use the fixed-left matrix transfer with a CUDA reduction policy. -/
def cudaMatmulCertificate : Except String GraphNumericalCertificate :=
  generate NN.Backend.BackendProfile.checkedCuda matmulGraph matmulSources

/-! ## Domain-sensitive square root

The checker propagates absolute value before checking the square-root domain. Thus an input range
that crosses zero is valid for `abs → sqrt`, while the same range passed directly to `sqrt` is
rejected. The square-root endpoints use TorchLean's proved directed binary32 rounders rather than a
host `libm` call.
-/

def sqrtGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input, outShape := .scalar },
      { id := 1, parents := [0], kind := .abs, outShape := .scalar },
      { id := 2, parents := [1], kind := .sqrt, outShape := .scalar }
    ] }

def sqrtSources : Array SourceRange := #[
  { nodeId := 0, enclosure := interval 0xc0800000 0x41100000 }
]

def sqrtInput : Spec.PackedTensor IEEE32Exec :=
  Spec.PackedTensor.mk .scalar (.scalar (IEEE32Exec.ofBits 0xc0800000))

def sqrtReplay : Except String CheckedExecution := do
  let certificate <-
    generateChecked NN.Backend.BackendProfile.checkedCpu sqrtGraph sqrtSources
  executeIEEE32 {} sqrtInput certificate

def invalidSqrtGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input, outShape := .scalar },
      { id := 1, parents := [0], kind := .sqrt, outShape := .scalar }
    ] }

/-- A source interval containing negative values does not satisfy the real square-root domain. -/
def invalidSqrtCertificate : Except String GraphNumericalCertificate :=
  generate NN.Backend.BackendProfile.checkedCpu invalidSqrtGraph sqrtSources

/-! ## Layer normalization

LayerNorm combines several domain-sensitive steps. The certificate follows the implementation:
mean, centering, squaring, variance, epsilon stabilization, directed square root, and division. The
portable profile fixes the reduction order used by both means.
-/

def layerNormGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input,
        outShape := .dim 2 (.dim 3 .scalar) },
      { id := 1, parents := [0], kind := .layernorm 1,
        outShape := .dim 2 (.dim 3 .scalar) }
    ] }

def layerNormSources : Array SourceRange := #[
  { nodeId := 0, enclosure := interval 0xc0000000 0x40000000 }
]

def layerNormInput : Spec.PackedTensor IEEE32Exec :=
  Spec.PackedTensor.mk (.dim 2 (.dim 3 .scalar)) <| .dim fun row => .dim fun column =>
    match row.1, column.1 with
    | 0, 0 => .scalar IEEE32Exec.negOne
    | 0, 1 => .scalar IEEE32Exec.posZero
    | 0, _ => .scalar IEEE32Exec.posOne
    | _, 0 => .scalar (IEEE32Exec.ofBits 0x40000000)
    | _, 1 => .scalar IEEE32Exec.posOne
    | _, _ => .scalar IEEE32Exec.posZero

def layerNormReplay : Except String CheckedExecution := do
  let certificate <-
    generateChecked NN.Backend.BackendProfile.checkedCpu layerNormGraph layerNormSources
  executeIEEE32 {} layerNormInput certificate

/-- The same fixed-left LayerNorm transfer is not attributed to an unspecified CUDA reduction. -/
def cudaLayerNormCertificate : Except String GraphNumericalCertificate :=
  generate NN.Backend.BackendProfile.checkedCuda layerNormGraph layerNormSources

/-! ## Domain-sensitive activations

Absolute value converts the signed source range to a nonnegative interval. That discharged domain
condition allows the checker to apply the proved directed square-root endpoints. ReLU then
preserves the resulting range.
-/

def activationGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input, outShape := .scalar },
      { id := 1, parents := [0], kind := .abs, outShape := .scalar },
      { id := 2, parents := [1], kind := .sqrt, outShape := .scalar },
      { id := 3, parents := [2], kind := .relu, outShape := .scalar }
    ] }

def activationSources : Array SourceRange := #[
  { nodeId := 0, enclosure := interval 0xc0800000 0x40800000 }
]

def activationInput : Spec.PackedTensor IEEE32Exec :=
  Spec.PackedTensor.mk .scalar (.scalar (IEEE32Exec.ofBits 0xc0800000))

def activationReplay : Except String CheckedExecution := do
  let certificate <-
    generateChecked NN.Backend.BackendProfile.checkedCpu activationGraph activationSources
  executeIEEE32 {} activationInput certificate

/-!
## Stable axis softmax

The real softmax theorem proves that a nonempty row lies in $[0,1]$; the bit-level replay then checks
that the executable implementation stayed finite and respected that range for the concrete input.
-/

def softmaxGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input,
        outShape := .dim 3 .scalar },
      { id := 1, parents := [0], kind := .softmax 0,
        outShape := .dim 3 .scalar }
    ] }

def softmaxSources : Array SourceRange := #[
  { nodeId := 0, enclosure := interval 0xc0000000 0x40000000 }
]

def softmaxInput : Spec.PackedTensor IEEE32Exec :=
  Spec.PackedTensor.mk (.dim 3 .scalar) <| .dim fun i =>
    match i.1 with
    | 0 => .scalar IEEE32Exec.posOne
    | 1 => .scalar IEEE32Exec.posZero
    | _ => .scalar IEEE32Exec.negOne

def softmaxReplay : Except String CheckedExecution := do
  let certificate <-
    generateChecked NN.Backend.BackendProfile.checkedCpu softmaxGraph softmaxSources
  executeIEEE32 {} softmaxInput certificate

/-!
## A complete model pass

The preceding examples isolate individual numerical rules. This final graph runs the same machinery
over a two-layer MLP with matrix weights and explicit bias tensors:

```text
input [1,2]
  -> matmul [2,3]
  -> add bias [1,3]
  -> ReLU
  -> matmul [3,1]
  -> add bias [1,1]
```

Nothing in certificate generation is told that this is an MLP. The checker sees ten ordinary IR
nodes and obtains each transfer from `GraphRangeRegistry`. Kernel selection independently chooses
a capsule for every operation. The final replay executes the stored graph with bit-level binary32
semantics and checks all ten intermediate tensors against the regenerated ranges.
-/

def mlpInputShape : Spec.Shape := .dim 1 (.dim 2 .scalar)
def mlpHiddenShape : Spec.Shape := .dim 1 (.dim 3 .scalar)
def mlpFirstWeightShape : Spec.Shape := .dim 2 (.dim 3 .scalar)
def mlpSecondWeightShape : Spec.Shape := .dim 3 (.dim 1 .scalar)
def mlpOutputShape : Spec.Shape := .dim 1 (.dim 1 .scalar)

/-- A two-layer matrix MLP expressed only in the canonical operation IR. -/
def mlpGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input, outShape := mlpInputShape },
      { id := 1, parents := [], kind := .const mlpFirstWeightShape,
        outShape := mlpFirstWeightShape },
      { id := 2, parents := [0, 1], kind := .matmul, outShape := mlpHiddenShape },
      { id := 3, parents := [], kind := .const mlpHiddenShape, outShape := mlpHiddenShape },
      { id := 4, parents := [2, 3], kind := .add, outShape := mlpHiddenShape },
      { id := 5, parents := [4], kind := .relu, outShape := mlpHiddenShape },
      { id := 6, parents := [], kind := .const mlpSecondWeightShape,
        outShape := mlpSecondWeightShape },
      { id := 7, parents := [5, 6], kind := .matmul, outShape := mlpOutputShape },
      { id := 8, parents := [], kind := .const mlpOutputShape, outShape := mlpOutputShape },
      { id := 9, parents := [7, 8], kind := .add, outShape := mlpOutputShape }
    ] }

/-- Source ranges cover inputs, both weight matrices, and both bias tensors. A single enclosure per
tensor is sufficient for this certificate format; the graph walk remains independent of rank. -/
def mlpSources : Array SourceRange := #[
  { nodeId := 0, enclosure := interval 0xbf800000 0x3f800000 },
  { nodeId := 1, enclosure := interval 0xbf800000 0x3f800000 },
  { nodeId := 3, enclosure := interval 0xbe800000 0x3e800000 },
  { nodeId := 6, enclosure := interval 0xbf800000 0x3f800000 },
  { nodeId := 8, enclosure := interval 0xbe800000 0x3e800000 }
]

/-! Constant payloads use the IR's canonical flat storage ABI; node shapes recover the typed matrix
view during evaluation. The explicit order below is row-major. -/

def mlpFirstWeightFlat : Spec.Tensor IEEE32Exec (.dim 6 .scalar) :=
  .dim fun i =>
    match i.1 with
    | 0 => .scalar (IEEE32Exec.ofBits 0x3f000000)
    | 1 => .scalar (IEEE32Exec.ofBits 0xbe800000)
    | 2 => .scalar (IEEE32Exec.ofBits 0x3f400000)
    | 3 => .scalar (IEEE32Exec.ofBits 0xbf000000)
    | 4 => .scalar IEEE32Exec.posOne
    | _ => .scalar (IEEE32Exec.ofBits 0x3e800000)

def mlpHiddenBiasFlat : Spec.Tensor IEEE32Exec (.dim 3 .scalar) :=
  .dim fun i =>
    match i.1 with
    | 0 => .scalar (IEEE32Exec.ofBits 0x3e000000)
    | 1 => .scalar (IEEE32Exec.ofBits 0xbe000000)
    | _ => .scalar IEEE32Exec.posZero

def mlpSecondWeightFlat : Spec.Tensor IEEE32Exec (.dim 3 .scalar) :=
  .dim fun i =>
    match i.1 with
    | 0 => .scalar (IEEE32Exec.ofBits 0x3f000000)
    | 1 => .scalar (IEEE32Exec.ofBits 0xbf400000)
    | _ => .scalar IEEE32Exec.posOne

def mlpOutputBiasFlat : Spec.Tensor IEEE32Exec (.dim 1 .scalar) :=
  .dim fun _ => .scalar (IEEE32Exec.ofBits 0x3d800000)

/-- Concrete parameters are payloads of the constant nodes, not special fields in the checker. -/
def mlpPayload : NN.IR.Payload IEEE32Exec where
  const? := fun nodeId =>
    match nodeId with
    | 1 => some { n := 6, v := mlpFirstWeightFlat }
    | 3 => some { n := 3, v := mlpHiddenBiasFlat }
    | 6 => some { n := 3, v := mlpSecondWeightFlat }
    | 8 => some { n := 1, v := mlpOutputBiasFlat }
    | _ => none

def mlpInput : Spec.PackedTensor IEEE32Exec :=
  let value : Spec.Tensor IEEE32Exec (.dim 1 (.dim 2 .scalar)) :=
    .dim fun _ => .dim fun column =>
      if column.1 = 0 then
        .scalar (IEEE32Exec.ofBits 0x3f000000) -- 0.5
      else
        .scalar IEEE32Exec.negOne
  Spec.PackedTensor.mk mlpInputShape (by simpa [mlpInputShape] using value)

/-- Generate the operation-local range trace and bind it to the checked CPU capsule plan. -/
def mlpCertificate : Except String CheckedCertificate :=
  generateChecked NN.Backend.BackendProfile.checkedCpu mlpGraph mlpSources

/-- Execute the stored graph in the bit-level binary32 interpreter and check every node. -/
def mlpReplay : Except String CheckedExecution := do
  let certificate <- mlpCertificate
  executeIEEE32 mlpPayload mlpInput certificate

/-- Executable acceptance report. Positive cases should be `true`; deliberately corrupted,
invalid-domain, or wrong-reduction-policy cases should be `false`. This list exercises range
reconstruction and IEEE replay; it does not construct the separate exact-real enclosure proof. -/
def exampleChecks : List (String × Bool) :=
  [ ("base certificate", accepted checked)
  , ("base IEEE replay", accepted replay)
  , ("tampered range rejected", !accepted tamperedCheck)
  , ("misplaced source rejected", !accepted misplacedSourceCheck)
  , ("duplicate contract rejected", !accepted duplicateContractCheck)
  , ("registry mismatch rejected", !accepted registryMismatchCheck)
  , ("base graph coverage", accepted baseCoverage)
  , ("unsupported operation rejected", !accepted unsupportedCoverage)
  , ("fixed-left reduction", accepted reductionReplay)
  , ("portable matmul", accepted matmulReplay)
  , ("CUDA matmul policy rejected", !accepted cudaMatmulCertificate)
  , ("directed sqrt", accepted sqrtReplay)
  , ("negative sqrt domain rejected", !accepted invalidSqrtCertificate)
  , ("portable LayerNorm", accepted layerNormReplay)
  , ("CUDA LayerNorm policy rejected", !accepted cudaLayerNormCertificate)
  , ("activation chain", accepted activationReplay)
  , ("stable softmax", accepted softmaxReplay)
  , ("two-layer MLP certificate", accepted mlpCertificate)
  , ("two-layer MLP IEEE replay", accepted mlpReplay)
  ]

def usage : String :=
  String.intercalate "\n"
    [ "Numerical runtime certificate example"
    , ""
    , "Usage:"
    , "  lake exe torchlean numerical_certificate"
    , ""
    , "Runs operation coverage, range-certificate, backend-audit, tamper-rejection, and bit-level"
    , "binary32 replay checks. The final two checks run a complete two-layer MLP."
    ]

/-- Public runner for the certificate examples. A failed positive check or an accepted negative
check produces a nonzero exit code, so this command is also suitable for regression testing. -/
def main (args : List String) : IO UInt32 := do
  if args.any fun arg => arg = "-h" || arg = "--help" then
    IO.println usage
    return 0
  if !args.isEmpty then
    IO.eprintln s!"unexpected arguments: {String.intercalate " " args}"
    IO.eprintln usage
    return 2
  IO.println "TorchLean numerical runtime certificate"
  let mut failed := false
  for (name, ok) in exampleChecks do
    IO.println s!"  {if ok then "ok" else "FAIL"}  {name}"
    if !ok then
      failed := true
  if failed then
    IO.eprintln "Numerical certificate checks failed."
    return 1
  IO.println "All numerical certificate checks passed."
  return 0

end NN.Examples.DeepDives.Floats.GraphNumericalCertificate

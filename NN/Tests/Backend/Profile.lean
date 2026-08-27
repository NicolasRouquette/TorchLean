/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Report
public import NN.Backend.IR
public import NN.Runtime.Autograd.Torch.Core.Ops
public import NN.Tensor

/-!
# Backend Profile Tests

Regression checks for contract-carrying backend profiles.

These are policy checks, not numerical kernel tests: they make sure backend planning does not
silently cross a trusted boundary or fall back to an unavailable platform provider.
-/

@[expose] public section

namespace NN.Tests.Backend.Profile

open NN.Backend

def expect (tag : String) (ok : Bool) : IO Unit := do
  unless ok do
    throw <| IO.userError s!"backend profile check failed: {tag}"

def expectCapsules (tag : String) (got expected : Array String) : IO Unit := do
  expect tag (got == expected)

def expectOp (tag : String) (kind : NN.IR.OpKind) (expected : Option BackendOp) : IO Unit := do
  expect tag (NN.Backend.IR.op? kind == expected)

def scalarHardMask : NN.IR.HardMask :=
  { shape := Spec.Shape.scalar, allowed := #[true] }

def expectContains (tag needle haystack : String) : IO Unit := do
  expect tag (haystack.contains needle)

def tinyReluGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0
        parents := #[]
        kind := .input
        outShape := Spec.Shape.scalar },
      { id := 1
        parents := #[0]
        kind := .relu
        outShape := Spec.Shape.scalar },
      { id := 2
        parents := #[1]
        kind := .relu
        outShape := Spec.Shape.scalar }
    ] }

def externalReluCapsule : KernelCapsule :=
  { Reference.relu with
    name := "external.relu"
    provider := .external
    device := .external
    trustLevel := .checked
    notes := "Test-only external capsule used to check target gating." }

def disabledForwardReluCapsule : KernelCapsule :=
  { Reference.relu with
    name := "reference.relu_disabled"
    supportsForward := false
    notes := "Test-only capsule used to check planner forward support gating." }

def fuzzedReluCapsule : KernelCapsule :=
  { Reference.relu with
    name := "reference.relu_fuzzed"
    valueContract :=
      { claim := .valueRefinement .relu
        summary := "Test-only fuzz-backed value contract."
        evidence := .fuzzOracle "profile-test-fuzz-oracle" }
    notes := "Test-only capsule used to check strict policy rejection of fuzz-backed evidence." }

def replacementReluCapsule : KernelCapsule :=
  { Reference.relu with
    name := "replacement.relu" }

def malformedReluCapsule : KernelCapsule :=
  { Reference.relu with
    name := "reference.relu_malformed"
    shapeContract := ContractDescriptor.tested
      (.valueRefinement .relu)
      "Deliberately mismatched descriptor for contract-alignment testing."
      "NN.Tests.Backend.Profile" }

def forwardOnlyReluCapsule : KernelCapsule :=
  { Reference.relu with
    name := "reference.relu_forward_only"
    vjpMode := .none
    vjpContract := ContractDescriptor.vjpUnavailable
      .relu "This test capsule intentionally has no VJP." }

/-- Metadata for the proof-carrying planner regression below. -/
def verifiedReluCapsule : KernelCapsule :=
  { forwardOnlyReluCapsule with
    name := "verified.relu"
    trustLevel := .verified }

/-- A small typed implementation used to check that verified planning retains semantic evidence. -/
def verifiedReluKernel : ProofCarryingKernel Int Int .relu (fun x => max x 0) :=
  { capsule := verifiedReluCapsule
    implementation := fun x => max x 0
    operation_matches := by rfl
    trust_verified := by rfl
    contracts_aligned := by decide
    refines := fun _ => rfl }

/-- Test capsule whose declared provider does not match the CPU random executor. -/
def mismatchedRandomCapsule : KernelCapsule :=
  { Reference.randUniform with
    name := "torchlean.rand_uniform_mismatched"
    provider := .torchLean }

def planOrThrow (tag : String) (profile : BackendProfile) (ops : Array BackendOp) :
    IO KernelPlan := do
  match profile.planOps ops with
  | .ok plan => pure plan
  | .error msg => throw <| IO.userError s!"{tag}: planning failed: {msg}"

def profileOrThrow (tag : String) (opts : Runtime.Autograd.Torch.Options) :
    IO BackendProfile := do
  let profile ← match opts.effectiveBackendProfile with
    | .ok profile => pure profile
    | .error msg => throw <| IO.userError s!"{tag}: profile resolution failed: {msg}"
  unless profile.hasDeviceCapsule do
    throw <| IO.userError s!"{tag}: profile `{profile.name}` has no capsule for its device"
  pure profile

def expectPlanningFails (tag : String) (profile : BackendProfile) (ops : Array BackendOp) :
    IO Unit := do
  match profile.planOps ops with
  | .ok plan =>
      throw <| IO.userError
        s!"{tag}: expected planning to fail, got capsules {plan.capsuleNames}"
  | .error _ => pure ()

def acceptedGraphOrThrow (tag : String) (profile : BackendProfile) (g : NN.IR.Graph) :
    IO AcceptedGraphKernelPlan := do
  match profile.acceptGraph g with
  | .ok (.accepted plan) => pure plan
  | .ok (.rejected _ failures) =>
      throw <| IO.userError s!"{tag}: expected accepted graph, got gate failures {repr failures}"
  | .error msg =>
      throw <| IO.userError s!"{tag}: graph planning failed: {msg}"

def expectGraphRejected (tag : String) (profile : BackendProfile) (g : NN.IR.Graph) :
    IO Unit := do
  match profile.acceptGraph g with
  | .ok (.accepted plan) =>
      throw <| IO.userError s!"{tag}: expected rejected graph, got capsules {plan.capsuleNames}"
  | .ok (.rejected _ _) => pure ()
  | .error msg =>
      throw <| IO.userError s!"{tag}: graph planning failed before gate: {msg}"

def expectNativeCudaBindingAccepts (tag : String)
    (opts : Runtime.Autograd.Torch.Options) (op : BackendOp) : IO Unit := do
  let base ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
  let s := { base with opts := opts }
  let result ← s.executeSelected op
    #[({ name := "profile-test native CUDA handler"
         op
         provider := .nativeCuda
         device := .cuda
         execute := fun _ => pure true } : KernelHandler Bool)]
  expect tag result

/-- A handler mismatch must fail before its implementation can run. -/
def expectBindingRejected (tag needle : String) (handler : KernelHandler Bool) : IO Unit := do
  match Reference.relu.bind handler with
  | .ok _ =>
      throw <| IO.userError s!"{tag}: mismatched handler unexpectedly bound"
  | .error message =>
      expectContains tag needle message

def checkHandlerIdentity : IO Unit := do
  let referenceHandler : KernelHandler Bool := {
    name := "reference ReLU test handler"
    op := .relu
    provider := .reference
    device := .cpu
    execute := fun _ => pure true
  }
  match Reference.relu.bind referenceHandler with
  | .ok executable =>
      expect "matching handler executes" (← executable.run)
  | .error message =>
      throw <| IO.userError s!"matching handler did not bind: {message}"
  expectBindingRejected "handler operation mismatch" "implements `add`"
    { referenceHandler with op := .add }
  expectBindingRejected "handler provider mismatch" "uses provider"
    { referenceHandler with provider := .torchLean }
  expectBindingRejected "handler device mismatch" "targets `cuda`"
    { referenceHandler with device := .cuda }

/-- Eager execution must not run a reference implementation under another provider's capsule. -/
def expectRandomProviderRejected : IO Unit := do
  let mismatchedModule : Registry.CapsuleModule :=
    { name := "reference", capsules := #[mismatchedRandomCapsule] }
  let profile := BackendProfile.checkedCpu.withCapsuleModules #[mismatchedModule]
  let opts :=
    Runtime.Autograd.Torch.Options.withBackendProfile
      ({} : Runtime.Autograd.Torch.Options) profile
  let session ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
    opts
  try
    let _ ← Runtime.Autograd.Torch.Internal.EagerSession.randUniform
      (s := session) (sh := Spec.Shape.ofList [2]) 7
    throw <| IO.userError "mismatched random provider unexpectedly executed"
  catch e =>
    expectContains "random provider mismatch is rejected"
      "no matching executable handler is linked" e.toString

def expectRuntimeDeviceRejected (tag : String) (device : NN.Backend.Device) :
    IO Unit := do
  let unavailableProfile : BackendProfile :=
    { BackendProfile.checkedCpu with
      name := s!"unavailable_{device.cliName}"
      policy := { BackendProfile.checkedCpu.policy with device := device } }
  try
    let opts :=
      Runtime.Autograd.Torch.Options.withBackendProfile
        ({} : Runtime.Autograd.Torch.Options) unavailableProfile
    let _ ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
      opts
    throw <| IO.userError s!"{tag}: expected runtime device rejection"
  catch e =>
    let msg := toString e
    expectContains tag s!"has no capsule for device `{device.cliName}`" msg

/-- User-facing CUDA sessions must agree with the implementation linked behind the CUDA symbols. -/
def expectCudaSessionMatchesRuntime : IO Unit := do
  let opts : Runtime.Autograd.Torch.Options :=
    { device := .cuda }
  match Runtime.Autograd.Cuda.Buffer.runtimeStatus with
  | .nativeAvailable =>
      let _ ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float) opts
      pure ()
  | .cpuStub =>
      try
        let _ ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float) opts
        throw <| IO.userError "CPU-stub build unexpectedly admitted a user CUDA session"
      catch e =>
        expectContains "CPU-stub CUDA session rejection" "CPU parity stubs" e.toString
  | .nativeUnavailable =>
      try
        let _ ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float) opts
        throw <| IO.userError "CUDA build without a visible device unexpectedly admitted a session"
      catch e =>
        expectContains "unavailable native CUDA session rejection" "no usable CUDA device" e.toString

def run : IO Unit := do
  checkHandlerIdentity
  let typedGraphCpuOpts : Runtime.Autograd.Torch.Options :=
    { execution := .typedGraph
      device := .cpu }
  expect "typed graph CPU options retain the public execution and device choices"
    (typedGraphCpuOpts.execution == .typedGraph && typedGraphCpuOpts.device == .cpu)
  let typedGraphCpuProfile ← profileOrThrow "typed graph CPU profile" typedGraphCpuOpts
  expect "execution mode does not alter CPU capsule selection"
    (typedGraphCpuProfile.policy.device == .cpu)
  expect "default registry contract fields are aligned"
    ((Registry.flatten Registry.maintainedModules).all KernelCapsule.contractsAligned)
  expect "LibTorch registry contract fields are aligned"
    ((Registry.flatten (#[Registry.libTorchModule] ++ Registry.maintainedModules)).all
      KernelCapsule.contractsAligned)
  let duplicateModuleProfile : BackendProfile :=
    { BackendProfile.checkedCpu with
      capsuleModules := #[Registry.libTorchModule, Registry.libTorchModule] }
  expectPlanningFails "duplicate capsule module names are rejected"
    duplicateModuleProfile #[.relu]
  let replacementModule : Registry.CapsuleModule :=
    { name := "reference", capsules := #[replacementReluCapsule] }
  let replacementProfile :=
    BackendProfile.checkedCpu.withCapsuleModules #[replacementModule]
  expect "same-name capsule modules are replaced instead of duplicated"
    ((replacementProfile.capsuleModules.filter (fun module => module.name == "reference")).size == 1)
  let replaced ← planOrThrow "replacement capsule module" replacementProfile #[.relu]
  expectCapsules "replacement capsule module is selected" replaced.capsuleNames
    #["replacement.relu"]
  let inferenceOpts : Runtime.Autograd.Torch.Options := { gradEnabled := false }
  let inferenceProfile ← profileOrThrow "no-grad default profile" inferenceOpts
  expect "no-grad runtime planning requests no VJP"
    (inferenceProfile.policy.vjpMode == .none)
  let trainingOpts : Runtime.Autograd.Torch.Options := { gradEnabled := true }
  let trainingProfile ← profileOrThrow "training default profile" trainingOpts
  expect "training runtime planning requests the TorchLean tape"
    (trainingProfile.policy.vjpMode == .torchLeanTape)
  expectOp "IR add maps to exact add capsule" .add (some .add)
  expectOp "IR linear maps to exact linear capsule" .linear (some .linear)
  let convConfig : NN.IR.ConvConfig :=
    { spatialRank := 2
      kernel := tensor! [3, 3]
      stride := tensor! [1, 1]
      padding := tensor! [0, 0]
      channelAxis := 0
      inChannels := 1
      outChannels := 1 }
  let poolConfig : NN.IR.WindowConfig :=
    { spatialRank := 2
      kernel := tensor! [2, 2]
      stride := tensor! [2, 2]
      padding := tensor! [0, 0] }
  expectOp "IR convolution maps to the convolution capability"
    (.conv convConfig) (some .conv)
  expectOp "IR max pooling maps to the max-pool capability"
    (.maxPool poolConfig) (some .maxPool)
  expectOp "IR rand uniform maps to exact forward-only capsule"
    (.randUniform 0) (some .randUniform)
  expectOp "IR permute maps to exact permute capsule"
    (.permute #[1, 0]) (some .permute)
  expectOp "IR hard-masked softmax keeps its exact capsule identity"
    (.hardMaskedSoftmax scalarHardMask) (some .hardMaskedSoftmax)
  expectOp "IR input has no backend capsule" .input none

  expect "cpu availability rejects external device capsule"
    (!(Availability.cpu.admitsCapsule externalReluCapsule))
  expect "external target admits external capsule when provider is available"
    (Target.external.declaredAvailability.admitsCapsule externalReluCapsule)
  match planOpsAvailable
      { device := .external, provider := .only .external }
      Target.external.declaredAvailability
      #[externalReluCapsule]
      #[.relu] with
  | .ok plan =>
      expectCapsules "external target can plan explicit external capsule" plan.capsuleNames
        #["external.relu"]
  | .error msg =>
      throw <| IO.userError s!"external capsule planning failed: {msg}"
  match planOps { device := .cpu } #[disabledForwardReluCapsule] #[.relu] with
  | .ok plan =>
      throw <| IO.userError
        s!"disabled forward capsule unexpectedly planned as {plan.capsuleNames}"
  | .error _ => pure ()
  match planOps
      { device := .cpu, vjpMode := .none }
      #[Reference.relu] #[.relu] with
  | .ok inferencePlan =>
      expectCapsules "inference accepts a forward capsule that also supports VJP"
        inferencePlan.capsuleNames #["reference.relu"]
  | .error msg =>
      throw <| IO.userError s!"inference VJP compatibility test: planning failed: {msg}"
  match planOps { device := .cpu, vjpMode := .torchLeanTape }
      #[forwardOnlyReluCapsule] #[.relu] with
  | .ok forwardOnly =>
      throw <| IO.userError
        s!"forward-only differentiable capsule unexpectedly planned as {forwardOnly.capsuleNames}"
  | .error _ => pure ()
  match planOps { device := .cpu, assurance := .verified }
      #[malformedReluCapsule] #[.relu] with
  | .ok malformed =>
      throw <| IO.userError
        s!"malformed capsule unexpectedly planned as {malformed.capsuleNames}"
  | .error _ => pure ()
  match planOps { device := .cpu, assurance := .verified, vjpMode := .none }
      #[verifiedReluCapsule] #[.relu] with
  | .ok erased =>
      throw <| IO.userError
        s!"erased verified metadata unexpectedly planned as {erased.capsuleNames}"
  | .error _ => pure ()
  let verifiedConfig : KernelPolicy :=
    { device := .cpu, assurance := .verified, vjpMode := .none }
  match planVerifiedKernel verifiedConfig #[verifiedReluKernel] with
  | .ok planned =>
      expect "proof-carrying planner did not execute the indexed specification"
        (planned.run (-3) == 0 && planned.run 4 == 4)
  | .error msg =>
      throw <| IO.userError s!"proof-carrying planner rejected a verified kernel: {msg}"
  match planVerifiedKernel { verifiedConfig with assurance := .checked } #[verifiedReluKernel] with
  | .ok _ =>
      throw <| IO.userError "proof-carrying planner accepted a non-verified assurance policy"
  | .error _ => pure ()
  match planOps { device := .cpu } #[fuzzedReluCapsule] #[.relu] with
  | .ok fuzzedRelu =>
      expect "strict gate rejects fuzz-only evidence"
        (!fuzzedRelu.acceptedBy AssurancePolicy.verified)
      expect "runtime gate accepts fuzz-backed checked evidence"
        (fuzzedRelu.acceptedBy AssurancePolicy.checked)
  | .error msg =>
      throw <| IO.userError s!"fuzzed relu policy test: planning failed: {msg}"

  let acceptedCpuGraph ← acceptedGraphOrThrow "checked cpu graph acceptance"
    BackendProfile.checkedCpu tinyReluGraph
  expectCapsules "checked cpu graph coalesces same relu capsule"
    acceptedCpuGraph.capsuleNames #["reference.relu"]
  expectCapsules "checked cpu graph keeps source node ids"
    (acceptedCpuGraph.nodeIds.map (fun n => toString n)) #["1", "2"]
  expect "checked cpu graph has no missing evidence" acceptedCpuGraph.audit.hasNoMissingEvidence
  expect "checked cpu graph has no trusted external" (!acceptedCpuGraph.audit.hasTrustedExternal)

  let singletonCpuProfile : BackendProfile :=
    { BackendProfile.checkedCpu with
      name := "checked_cpu_singleton_test"
      groupingMode := .singleton }
  let singletonCpuGraph ← acceptedGraphOrThrow "checked cpu singleton graph acceptance"
    singletonCpuProfile tinyReluGraph
  expectCapsules "singleton graph keeps repeated relu capsules"
    singletonCpuGraph.capsuleNames #["reference.relu", "reference.relu"]

  -- Operations without a LibTorch capsule use the native provider under the hybrid profile.
  let softmaxFallback ← planOrThrow "hybrid native softmax fallback"
    BackendProfile.libTorchForwardCuda #[.softmax]
  expectCapsules "hybrid profile falls back to native softmax"
    softmaxFallback.capsuleNames #["native_cuda.softmax"]

  let exactOps :=
    #[ BackendOp.matmul, .linear, .mseLoss, .add, .sub, .mul, .scale, .abs, .sqrt
    , .clamp, .max, .min, .relu, .gelu, .sigmoid, .tanh
    , .softmax, .hardMaskedSoftmax, .softplus, .exp, .log, .inv, .safeLog, .logSoftmax
    , .reduceSum
    , .reduceMean, .randUniform, .bernoulliMask, .reshape, .permute, .broadcast
    , .concat, .slice, .gather, .scatterAdd, .layerNorm, .batchNorm, .conv
    , .convTranspose, .maxPool, .smoothMaxPool, .avgPool ]

  let exactReferenceCapsules := exactOps.map fun op => s!"reference.{op.name}"
  let exactNativeCudaCapsules := exactOps.map fun op => s!"native_cuda.{op.name}"
  let cpuOnlyOps := #[BackendOp.sin, .cos]
  let profileOps :=
    #[ BackendOp.matmul, .relu, .softmax, .hardMaskedSoftmax, .layerNorm, .batchNorm
    , .conv, .convTranspose, .maxPool, .smoothMaxPool, .avgPool, .mseLoss
    , .scaledDotProductAttention ]

  let cpu ← planOrThrow "checked cpu" BackendProfile.checkedCpu profileOps
  expectCapsules "checked cpu capsule order" cpu.capsuleNames
    #[ "reference.matmul"
    , "reference.relu"
    , "reference.softmax"
    , "reference.hard_masked_softmax"
    , "reference.layer_norm"
    , "reference.batch_norm"
    , "reference.conv"
    , "reference.conv_transpose"
    , "reference.max_pool"
    , "reference.smooth_max_pool"
    , "reference.avg_pool"
    , "reference.mse_loss"
    , "reference.attention"
    ]
  expect "checked cpu has no trusted external" (!cpu.hasTrustedExternal)

  let cpuExact ← planOrThrow "checked cpu exact ops" BackendProfile.checkedCpu exactOps
  expectCapsules "checked cpu exact capsules" cpuExact.capsuleNames exactReferenceCapsules
  let cpuOnly ← planOrThrow "checked cpu cpu-only ops" BackendProfile.checkedCpu cpuOnlyOps
  expectCapsules "checked cpu cpu-only capsules" cpuOnly.capsuleNames
    #["reference.sin", "reference.cos"]

  let replacementModule : Registry.CapsuleModule :=
    { name := "profile-test-replacement", capsules := #[replacementReluCapsule] }
  let extendedCpu := BackendProfile.checkedCpu.withCapsuleModules #[replacementModule]
  let extendedPlan ← planOrThrow "extended capsule modules" extendedCpu #[.relu, .matmul]
  expectCapsules "extended modules preserve model-independent preference" extendedPlan.capsuleNames
    #["replacement.relu", "reference.matmul"]

  let reportOps := exactOps ++ #[.scaledDotProductAttention]
  match BackendProfile.checkedCpu.planReport reportOps with
  | .ok report =>
      expectContains "checked cpu report names exact add" "add: reference.add" report
      expectContains "checked cpu report names exact reshape" "reshape: reference.reshape" report
      expectContains "checked cpu report names exact batchnorm"
        "batch_norm: reference.batch_norm" report
      expectContains "checked cpu report names exact smooth max pool"
        "smooth_max_pool: reference.smooth_max_pool" report
      expectContains "checked cpu report names exact attention"
        "scaled_dot_product_attention: reference.attention" report
  | .error msg =>
      throw <| IO.userError s!"checked cpu report failed: {msg}"

  let cuda ← planOrThrow "checked cuda" BackendProfile.checkedCuda profileOps
  expectCapsules "checked cuda capsule order" cuda.capsuleNames
    #[ "native_cuda.matmul"
    , "native_cuda.relu"
    , "native_cuda.softmax"
    , "native_cuda.hard_masked_softmax"
    , "native_cuda.layer_norm"
    , "native_cuda.batch_norm"
    , "native_cuda.conv"
    , "native_cuda.conv_transpose"
    , "native_cuda.max_pool"
    , "native_cuda.smooth_max_pool"
    , "native_cuda.avg_pool"
    , "native_cuda.mse_loss"
    , "torchlean.composed_attention"
    ]
  expect "checked cuda has no trusted external" (!cuda.hasTrustedExternal)

  let cudaExact ← planOrThrow "checked cuda exact ops" BackendProfile.checkedCuda exactOps
  expectCapsules "checked cuda exact capsules" cudaExact.capsuleNames exactNativeCudaCapsules
  expectPlanningFails "checked cuda has no sin capsule yet" BackendProfile.checkedCuda #[.sin]
  match BackendProfile.checkedCuda.planReport reportOps with
  | .ok report =>
      expectContains "checked cuda report names exact add" "add: native_cuda.add" report
      expectContains "checked cuda report names exact max pool" "max_pool: native_cuda.max_pool" report
      expectContains "checked cuda report names exact batchnorm"
        "batch_norm: native_cuda.batch_norm" report
      expectContains "checked cuda report names exact smooth max pool"
        "smooth_max_pool: native_cuda.smooth_max_pool" report
      expectContains "checked cuda report names exact attention"
        "scaled_dot_product_attention: torchlean.composed_attention" report
  | .error msg =>
      throw <| IO.userError s!"checked cuda report failed: {msg}"

  let libtorchForward ← planOrThrow "libtorch forward cuda" BackendProfile.libTorchForwardCuda
    #[.scaledDotProductAttention]
  expectCapsules "preferred LibTorch provider wins without registry-order dependence"
    libtorchForward.capsuleNames
    #["libtorch.sdpa_forward"]
  expect "libtorch forward records external boundary" libtorchForward.hasTrustedExternal
  expectCapsules "libtorch forward external op" libtorchForward.trustedExternalOps
    #["scaled_dot_product_attention"]
  expect "strict gate rejects trusted LibTorch forward"
    (!libtorchForward.acceptedBy AssurancePolicy.verified)
  let hybridForward ← planOrThrow "hybrid libtorch forward cuda"
    BackendProfile.libTorchForwardCuda #[.add, .scaledDotProductAttention, .relu]
  expectCapsules "hybrid profile uses native fallback around LibTorch attention"
    hybridForward.capsuleNames
    #["native_cuda.add", "libtorch.sdpa_forward", "native_cuda.relu"]
  match BackendProfile.libTorchForwardCuda.planReport #[.scaledDotProductAttention] with
  | .ok report =>
      expectContains "libtorch forward report names TorchLean tape"
        "vjp=torchlean-tape" report
      expectContains "libtorch forward report names capsule"
        "scaled_dot_product_attention: libtorch.sdpa_forward" report
  | .error msg =>
      throw <| IO.userError s!"libtorch forward report failed: {msg}"

  for (tag, device) in
      [ ("runtime rejects named-but-unimplemented metal", NN.Backend.Device.metal)
      , ("runtime rejects named-but-unimplemented rocm", NN.Backend.Device.rocm)
      , ("runtime rejects named-but-unimplemented wasm", NN.Backend.Device.wasm)
      , ("runtime rejects named-but-unimplemented tpu", NN.Backend.Device.tpu)
      , ("runtime rejects named-but-unimplemented trainium", NN.Backend.Device.trainium)
      , ("runtime rejects named-but-unimplemented custom chip", NN.Backend.Device.custom)
      , ("runtime rejects named-but-unimplemented external", NN.Backend.Device.external)
      ] do
    expectRuntimeDeviceRejected tag device

  expectCudaSessionMatchesRuntime

  let cpuSession ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
    ({ device := .cpu } : Runtime.Autograd.Torch.Options)
  let firstRelu ← cpuSession.selectedCapsule .relu
  let secondRelu ← cpuSession.selectedCapsule .relu
  let cpuSelections ← cpuSession.backendSelections
  expect "session reuses the selected capsule for a repeated operation"
    (firstRelu.name == secondRelu.name && cpuSelections.size == 1)
  expectRandomProviderRejected

  let checkedCudaOpts : Runtime.Autograd.Torch.Options :=
    { device := .cuda }
  for op in
      #[ BackendOp.matmul
      , .batchNorm
      , .maxPool
      , .avgPool
      , .smoothMaxPool
      ] do
    expectNativeCudaBindingAccepts s!"checked cuda runtime binding accepts `{op.name}`"
      checkedCudaOpts op

  IO.println "  backend profiles: ok"

end NN.Tests.Backend.Profile

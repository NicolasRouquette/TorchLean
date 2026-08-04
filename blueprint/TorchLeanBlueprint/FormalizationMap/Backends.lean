import Verso
import VersoManual
import VersoBlueprint
import NN.Backend
import NN.Runtime.Autograd.Engine
import NN.Runtime.Autograd.Torch
import NN.Runtime.Autograd.TorchLean
import NN.Runtime.Autograd.Train
import NN.API.Trainer.Manual
import NN.API.Trainer.Scheduler
import NN.Runtime.PyTorch

open Verso.Genre
open Verso.Genre.Manual
open Informal

#doc (Manual) "Backends and Training" =>

Backend selection is an audited planning step. A capsule names the operation, provider, device,
numerical policy, and evidence. Eager execution matches the selected capsule to a handler with the
same identity. Graph-level lowering produces checked planning data, not an execution engine.

:::group "backend_selection"
Kernel contracts, providers, and dispatch.
:::

:::definition "backend_capsule_contracts" (parent := "backend_selection") (lean := "NN.Backend.KernelCapsule")
A kernel capsule records an operation, provider, device, forward and VJP support, numerical policy,
and evidence for its shape, layout, value, and VJP claims. It states the contract expected from an
implementation; it does not prove that implementation.
:::

:::definition "proof_carrying_kernel" (parent := "backend_selection") (lean := "NN.Backend.ProofCarryingKernel")
A proof-carrying kernel packages a typed implementation with a pointwise proof that it equals one
explicit Lean specification. The typed verified planner retains that proof; converting the object
to ordinary capsule metadata does not.
:::

:::theorem "verified_kernel_execution" (parent := "backend_selection") (lean := "NN.Backend.VerifiedPlannedKernel.run_eq_specification")
Executing a kernel selected by the typed verified planner returns the value of the specification
that indexes the selected kernel.
:::

:::proof "verified_kernel_execution"
The selected object retains the original {uses "proof_carrying_kernel"}[refinement field], so the
result follows by applying that field to the input.
:::

:::definition "cuda_native_boundary" (parent := "backend_selection") (lean := "Runtime.Autograd.Cuda.Buffer")
`Cuda.Buffer` is an opaque handle to a contiguous float32 buffer. A CUDA build stores device
memory behind the handle; the default stub keeps parity storage on the host. Lean code cannot
inspect either representation directly.
:::

:::definition "backend_profile" (parent := "backend_selection") (lean := "NN.Backend.BackendProfile")
A backend profile stores a name, execution configuration, target, capsule modules, and graph
lowering mode. Target availability and the registry are derived from those fields, and capsule
modules are validated when a graph is planned.
:::

:::definition "backend_provider_catalog" (parent := "backend_selection") (lean := "NN.Backend.Registry.maintainedModules")
The maintained registry collects {uses "backend_capsule_contracts"}[contract capsules] contributed
by the attention, native CUDA, and reference modules. It contains planning metadata, not executable
handlers. Profiles add the separate LibTorch module when requested.
:::

:::definition "checked_cpu_backend_profile" (parent := "backend_selection") (lean := "NN.Backend.BackendProfile.checkedCpu")
The checked CPU profile instantiates {uses "backend_profile"}[the profile record] with
{uses "backend_provider_catalog"}[the maintained capsule modules], a portable CPU target,
coalesced lowering, and the checked assurance policy.
:::

:::definition "backend_executable_binding" (parent := "backend_selection") (lean := "NN.Backend.KernelCapsule.bind")
Binding a {uses "backend_capsule_contracts"}[selected capsule] to a handler checks that their
operation, provider, and device agree. The resulting executable kernel carries those identity
equalities; binding does not strengthen the capsule's numerical evidence.
:::

:::definition "backend_planning_and_gating" (parent := "backend_selection") (lean := "NN.Backend.BackendProfile.acceptGraph")
A {uses "backend_profile"}[backend profile] validates its configured capsule modules, requires
{uses "ir_structural_validation"}[a structurally well-formed IR graph], and selects a capsule for
each runtime-relevant node. It then lowers those choices as singleton or coalesced groups and
gates their {uses "backend_capsule_contracts"}[evidence] under the profile's assurance policy.
:::

:::definition "backend_eager_dispatch" (parent := "backend_selection") (lean := "Runtime.Autograd.Torch.Internal.EagerSession.executeSelected")
For one eager operation, the session selects and caches an admitted capsule, finds a handler with
the same operation, provider, and device, and runs it through
{uses "backend_executable_binding"}[the checked binding]. This dispatcher does not append an
autograd node; each operation implementation records its own forward value and VJP.
:::

:::definition "cuda_autograd_tape" (parent := "backend_selection") (lean := "Runtime.Autograd.Cuda.Tape")
The CUDA tape stores {uses "cuda_native_boundary"}[device buffers], parent ids, and local VJP
closures in evaluation order. `requireValue`, `requireGrad`, and backward accumulation check shape
tags and native buffer lengths. Dense backward returns one buffer per node; sparse backward retains
owned buffers only for selected node ids and requires the caller to release them.
:::

:::theorem "cuda_execution_contracts" (parent := "backend_selection") (lean := "Runtime.Autograd.Cuda.Float32Contract.native_add_eq_ieee32")
Given the stated native bit-agreement hypothesis, decoded native scalar addition equals
{uses "executable_binary32"}[`IEEE32Exec.add`]. The hypothesis remains visible in the theorem type.
:::

:::proof "cuda_execution_contracts"
The proof rewrites the supplied native result bits to
{uses "executable_binary32"}[the executable binary32 result]. It does not prove the external kernel
implementation from source.
:::

:::group "training_runtime"
Scalar-loss modules and stateful supervised updates.
:::

:::definition "runtime_module_training" (parent := "training_runtime") (lean := "Runtime.Autograd.TorchLean.Module.ScalarModule")
`ScalarModule` wraps a scalar trainer, its runtime options, and the selected host/device tensor
conversion. The trainer owns a shape-indexed mutable parameter pack and runs its scalar-loss
{uses "runtime_ops_program"}[program] through {uses "backend_eager_dispatch"}[eager execution] or
{uses "runtime_compiled_autograd"}[the compiled backend]. Gradients are returned to callers;
generic optimizer state is passed to and returned from update methods rather than stored here.
:::

:::definition "supervised_training_state" (parent := "training_runtime") (lean := "TorchLean.Trainer.Manual.Stepper")
`Stepper` holds a supervised runner, a one-sample update closure, an epoch-over-list closure, and
a step counter.
:::

:::definition "supervised_training_constructor" (parent := "training_runtime") (lean := "TorchLean.Trainer.Manual.stepper")
The supervised stepper constructor builds
{uses "supervised_training_state"}[the stateful training loop] around
{uses "runtime_module_training"}[a scalar module]. It selects an optimizer, applies an optional
learning-rate schedule, and refreshes mode-dependent model buffers before each update.
:::

:::group "external_graph_bridges"
Checked import of captured PyTorch graph artifacts.
:::

:::definition "pytorch_graph_import" (parent := "external_graph_bridges") (lean := "Import.PyTorch.TorchExport.parseGraph")
The `torch.export` adapter parses TorchLean's captured graph schema, lowers its supported values to
the shared IR, runs {uses "ir_shape_validation"}[shape validation], and checks that the named input
and output nodes exist.
:::

:::theorem "pytorch_graph_import_well_shaped" (parent := "external_graph_bridges") (lean := "Import.PyTorch.TorchExport.parseGraph_wellShaped")
Every graph returned successfully by {uses "pytorch_graph_import"}[the `torch.export` parser]
satisfies {uses "ir_shape_validation"}[TorchLean's executable shape check].
:::

:::proof "pytorch_graph_import_well_shaped"
The proof unfolds {uses "pytorch_graph_import"}[the parser], rules out each rejected branch, and
returns the successful {uses "ir_shape_validation"}[shape check].
:::

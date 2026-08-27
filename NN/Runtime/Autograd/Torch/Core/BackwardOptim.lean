/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.OptimizerCheckpoint

/-!
# Backward Passes and Optimizers

Gradient extraction and optimizer updates for eager sessions, including the CUDA paths that keep
parameter mirrors and moment buffers on device.
-/


@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor
open Proofs.Autograd.Algebra

namespace Internal

namespace EagerSession

/--
Run reverse-mode backprop on the CUDA tape, returning device gradients for all tape entries.

This is the CUDA analogue of `backwardDenseAll`, but it does *not* download gradients back to the
host. This is primarily useful for implementing GPU-native optimizer steps.
-/
def backwardDenseAllCuda {α : Type} [TensorTransfer α] (s : EagerSession α) [Add α] [Zero α]
  [DecidableEq Shape]
  {sh : Shape} (out : TensorRef α sh) (seed : Tensor α sh) :
  IO (Array Runtime.Autograd.Cuda.AnyBuffer) := do
  if Options.device s.opts != .cuda then
    throw <| IO.userError "torch: backwardDenseAllCuda called on non-CUDA eager session"
  let t ← s.cudaTape.get
  let seedAny ← CudaBridge.toAnyBuffer (α := α) (s := sh) seed
  okOrThrow <|
    Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t) (outId := out.id) (seed := seedAny)

/-- Run CUDA backward from a scalar loss with seed `1`, returning device gradient buffers. -/
def backwardScalarDenseAllCuda {α : Type} [TensorTransfer α] (s : EagerSession α) [Add α]
  [Zero α] [One α] [DecidableEq Shape]
  (loss : TensorRef α Shape.scalar) : IO (Array Runtime.Autograd.Cuda.AnyBuffer) := do
  backwardDenseAllCuda (α := α) s (sh := Shape.scalar) loss (Tensor.scalar (1 : α))

/--
Run scalar-loss CUDA backprop and return gradients only for trainable parameter leaves.

The tape walk uses an array indexed by node id rather than a persistent hash map. Only trainable
parameter gradients are packed into the returned map, so optimizer updates stay on device without
paying hash-table costs at every intermediate node.
-/
def backwardScalarParamGradsCuda {α : Type} [TensorTransfer α] (s : EagerSession α)
    [One α] [DecidableEq Shape]
    (loss : TensorRef α Shape.scalar) : IO CudaGradMap := do
  if Options.device s.opts != .cuda then
    throw <| IO.userError "torch: backwardScalarParamGradsCuda called on non-CUDA eager session"
  let t ← s.cudaTape.get
  let params ← s.paramsByLeaf.get
  let seedAny ← CudaBridge.toAnyBuffer (α := α) (s := Shape.scalar)
    (Tensor.scalar (1 : α))
  checkCudaAnyBufferSize "scalar CUDA backward seed" seedAny
  let n := t.nodes.size
  if loss.id ≥ n then
    releaseCudaAnyBuffer seedAny
    throw <| IO.userError "torch: scalar CUDA backward seed id out of bounds"
  let mut initial : Array (Option Runtime.Autograd.Cuda.AnyBuffer) := Array.replicate n none
  initial := initial.set! loss.id (some seedAny)
  let gradsRef ← IO.mkRef initial
  let addGrad (id : Nat) (g : Runtime.Autograd.Cuda.AnyBuffer) : IO Unit := do
    let node ← match t.getNode? id with
      | some node => pure node
      | none =>
          releaseCudaAnyBuffer g
          throw <| IO.userError "torch: invalid parent id during CUDA backward"
    if !node.requiresGrad then
      checkCudaAnyBufferSize s!"discarded gradient for node {id}" g
      releaseCudaAnyBuffer g
    else if _h : g.s = node.value.s then
      let g' : Runtime.Autograd.Cuda.AnyBuffer := { s := node.value.s, buf := g.buf }
      checkCudaAnyBufferSize s!"gradient contribution for node {id} ({node.name})" g'
      let grads ← gradsRef.get
      match grads[id]? with
      | none =>
          releaseCudaAnyBuffer g'
          throw <| IO.userError "torch: CUDA gradient table out of bounds"
      | some none =>
          let owned ← ownedCudaAnyBuffer s!"owned gradient for node {id} ({node.name})" g'
          releaseCudaAnyBuffer g'
          gradsRef.set (grads.set! id (some owned))
      | some (some old) =>
          if _hold : old.s = node.value.s then
            let old' : Runtime.Autograd.Cuda.AnyBuffer := { s := node.value.s, buf := old.buf }
            checkCudaAnyBufferSize s!"accumulated gradient for node {id} ({node.name})" old'
            let summed ← okOrThrow <| Runtime.Autograd.Cuda.AnyBuffer.add old' g'
            releaseCudaAnyBuffer old'
            releaseCudaAnyBuffer g'
            gradsRef.set (grads.set! id (some summed))
          else
            releaseCudaAnyBuffer g'
            throw <| IO.userError "torch: CUDA gradient table has wrong shape for node"
    else
      releaseCudaAnyBuffer g
      throw <| IO.userError "torch: CUDA gradient contribution has wrong shape for parent"
  for off in [0:n] do
    let id := n - 1 - off
    let grads ← gradsRef.get
    match grads[id]? with
    | some (some dLdy) =>
        let node ← match t.getNode? id with
          | some n => pure n
          | none => throw <| IO.userError "torch: internal CUDA tape node missing"
        if node.requiresGrad then
          checkCudaAnyBufferSize s!"upstream gradient for node {id} ({node.name})" dLdy
          let contribs ← okOrThrow <| node.backward dLdy
          for (pid, pg) in contribs do
            addGrad pid pg
        if params.contains id then
          pure ()
        else
          let gradsNow ← gradsRef.get
          match gradsNow[id]? with
          | some (some stale) =>
              releaseCudaAnyBuffer stale
              gradsRef.set (gradsNow.set! id none)
          | _ => pure ()
    | _ => pure ()
  let final ← gradsRef.get
  let mut out : CudaGradMap := Std.HashMap.emptyWithCapacity
  for (id, _p) in params.toList.filter (fun entry => entry.2.requiresGrad) do
    match final[id]? with
    | some (some g) => out := out.insert id g
    | _ => throw <| IO.userError s!"torch: missing CUDA gradient for parameter leaf {id}"
  pure out

/--
Run reverse-mode backprop and return a dense gradient array for all tape entries.

`seed` is the upstream gradient for `out` (like PyTorch's `backward(gradient=...)`).
-/
def backwardDenseAll {α : Type} [TensorTransfer α] (s : EagerSession α) [Add α] [Zero α]
  [DecidableEq Shape]
  {sh : Shape} (out : TensorRef α sh) (seed : Tensor α sh) :
  IO (Array (Spec.SomeTensor α)) := do
  if Options.device s.opts == .cuda then
    let gradsDev ← backwardDenseAllCuda (α := α) s (sh := sh) out seed
    gradsDev.mapM (fun g => CudaBridge.ofAnyBuffer (α := α) g)
  else
    let t ← s.tape.get
    okOrThrow (Runtime.Autograd.Tape.backwardDenseAll (t := t) (outId := out.id)
      (seed := Spec.SomeTensor.ofTensor seed))

/--
Run backward from a scalar loss with seed `1`.

PyTorch comparison: `loss.backward()` for a scalar loss.
-/
def backwardScalarDenseAll {α : Type} [TensorTransfer α] (s : EagerSession α) [Add α]
  [Zero α] [One α] [DecidableEq Shape]
  (loss : TensorRef α Shape.scalar) : IO (Array (Spec.SomeTensor α)) := do
  backwardDenseAll (α := α) s (sh := Shape.scalar) loss (Tensor.scalar (1 : α))

/--
Extract the gradient for a particular `TensorRef` from a dense gradient array.
-/
def grad {α : Type} {sh : Shape} [DecidableEq Shape]
  (grads : Array (Spec.SomeTensor α)) (x : TensorRef α sh) : IO (Tensor α sh) := do
  let gAny ← match grads[x.id]? with
    | some g => pure g
    | none => throw <| IO.userError "torch: gradient array out of bounds"
  if h : gAny.shape = sh then
    pure (gAny.cast h)
  else
    throw <| IO.userError
      s!"torch: grad shape mismatch (expected {Shape.pretty sh}, got {Shape.pretty gAny.shape})"

/--
Apply an SGD update to all parameters recorded via `use`.

PyTorch comparison: `for p in params: p.data -= lr * p.grad`.
-/
def sgdStepAll {α : Type} [TensorTransfer α] (s : EagerSession α)
  [Sub α] [Mul α] [Add α] [Zero α] [DecidableEq Shape]
  (lr : α) (grads : Array (Spec.SomeTensor α)) : IO Unit := do
  if Options.device s.opts == .cuda then
    let lrF ← TensorTransfer.toFloat (α := α) lr
    let t0 ← s.cudaTape.get
    let m ← s.paramsByLeaf.get
    for (id, p) in m.toList.filter (fun entry => entry.2.requiresGrad) do
      let gAny ← match grads[id]? with
        | some g => pure g
        | none => throw <| IO.userError "torch: gradient array out of bounds during SGD"
      if hs : gAny.shape = p.s then
        let gT : Tensor α p.s := gAny.cast hs
        let gDev ← CudaBridge.toAnyBuffer (α := α) (s := p.s) gT
        let pBuf ← okOrThrow <|
          Runtime.Autograd.Cuda.Tape.requireValue (t := t0) (id := id) (s := p.s)
        let updatedDev : Runtime.Autograd.Cuda.AnyBuffer :=
          { s := p.s, buf := Runtime.Autograd.Cuda.Buffer.axpy pBuf gDev.buf (-lrF) }
        p.setCuda updatedDev
        -- The uploaded host gradient is only a transient bridge buffer for this update.
        let released ← Runtime.Autograd.Cuda.Buffer.releaseIO gDev.buf
        AnyParam.observeCudaCleanupFlag released
      else
        throw <| IO.userError "torch: internal grad shape mismatch during SGD"
  else
    let m ← s.paramsByLeaf.get
    for (id, p) in m.toList.filter (fun entry => entry.2.requiresGrad) do
      let gAny ← match grads[id]? with
        | some g => pure g
        | none => throw <| IO.userError "torch: gradient array out of bounds during SGD"
      if hs : gAny.shape = p.s then
        let pv ← p.get
        if hp : pv.shape = p.s then
          let pvT : Tensor α p.s := pv.cast hp
          let gT : Tensor α p.s := gAny.cast hs
          let updated : Tensor α p.s :=
            Tensor.materialize <| subSpec pvT (scaleSpec (α := α) (s := p.s) gT lr)
          p.set (Spec.SomeTensor.ofTensor updated)
        else
          throw <| IO.userError "torch: internal param shape mismatch"
      else
        throw <| IO.userError "torch: internal grad shape mismatch during SGD"

/--
Apply an SGD update to all parameters recorded via `use`, using CUDA device gradients.

This avoids downloading the full dense gradient array and keeps updated parameters in each
`Param`'s CUDA mirror. Host tensors are synchronized later by explicit parameter readback.
-/
def sgdStepAllCuda {α : Type} [TensorTransfer α] (s : EagerSession α) [DecidableEq Shape]
  (lr : α) (grads : Array Runtime.Autograd.Cuda.AnyBuffer) : IO Unit := do
  if Options.device s.opts != .cuda then
    throw <| IO.userError "torch: sgdStepAllCuda called on non-CUDA eager session"
  let lrF ← TensorTransfer.toFloat (α := α) lr
  let t0 ← s.cudaTape.get
  let m ← s.paramsByLeaf.get
  for (id, p) in m.toList.filter (fun entry => entry.2.requiresGrad) do
    let gAny ← match grads[id]? with
      | some g => pure g
      | none => throw <| IO.userError "torch: gradient array out of bounds during SGD"
    if _hs : gAny.s = p.s then
      let pBuf ← okOrThrow <|
        Runtime.Autograd.Cuda.Tape.requireValue (t := t0) (id := id) (s := p.s)
      let updatedDev : Runtime.Autograd.Cuda.AnyBuffer :=
        { s := p.s, buf := Runtime.Autograd.Cuda.Buffer.axpy pBuf gAny.buf (-lrF) }
      p.setCuda updatedDev
    else
      throw <| IO.userError "torch: internal grad shape mismatch during SGD"

/--
Check every trainable parameter, gradient, tape value, and optional Adam state before an optimizer
changes a parameter. This prevents a malformed gradient map or checkpoint state from producing a
partially applied update.
-/
def checkCudaOptimizerInputs {α : Type} (operation : String)
    (tape : Runtime.Autograd.Cuda.Tape) (params : Std.HashMap Nat (AnyParam α))
    (grads : CudaGradMap) (state : Option CudaAdamState := none) : IO Unit := do
  for (id, param) in params.toList.filter (fun entry => entry.2.requiresGrad) do
    let grad ← match grads.get? id with
      | some grad => pure grad
      | none => throw <| IO.userError s!"torch: gradient map missing parameter during {operation}"
    unless grad.s == param.s do
      throw <| IO.userError s!"torch: gradient shape mismatch during {operation}"
    checkCudaAnyBufferSize s!"{operation} gradient for parameter leaf {id}" grad
    let value ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.requireValue (t := tape) (id := id) (s := param.s)
    checkCudaAnyBufferSize s!"{operation} value for parameter leaf {id}"
      { s := param.s, buf := value }
    match state with
    | none => pure ()
    | some states =>
        match states.get? id with
        | none => pure ()
        | some adamState =>
            let expected := Runtime.Autograd.Cuda.Buffer.size value
            if Runtime.Autograd.Cuda.Buffer.size adamState.m != expected ||
                Runtime.Autograd.Cuda.Buffer.size adamState.v != expected then
              throw <| IO.userError
                s!"torch: CUDA Adam state size mismatch for parameter leaf {id}"

/-- Reject a non-finite or negative learning rate before an optimizer mutates device state. -/
def checkCudaLearningRate (operation : String) (learningRate : Float) : IO Unit := do
  unless learningRate.isFinite && 0.0 ≤ learningRate do
    throw <| IO.userError s!"torch: {operation} learning rate must be finite and nonnegative"

/--
Apply SGD from a sparse CUDA gradient map.

This is the path used by the CUDA trainer.  It updates only parameter leaves and avoids allocating
zero gradients for every forward activation in the tape.
-/
def sgdStepAllCudaMap {α : Type} [TensorTransfer α] (s : EagerSession α) [DecidableEq Shape]
    (lr : α) (grads : CudaGradMap) : IO Unit := do
  if Options.device s.opts != .cuda then
    throw <| IO.userError "torch: sgdStepAllCudaMap called on non-CUDA eager session"
  let lrF ← TensorTransfer.toFloat (α := α) lr
  checkCudaLearningRate "CUDA SGD" lrF
  let t0 ← s.cudaTape.get
  let params ← s.paramsByLeaf.get
  checkCudaOptimizerInputs "CUDA SGD" t0 params grads
  for (id, p) in params.toList.filter (fun entry => entry.2.requiresGrad) do
    let gAny ← match grads.get? id with
      | some g => pure g
      | none => throw <| IO.userError "torch: gradient map missing parameter during CUDA SGD"
    if _hs : gAny.s = p.s then
      let pBuf ← okOrThrow <|
        Runtime.Autograd.Cuda.Tape.requireValue (t := t0) (id := id) (s := p.s)
      let updatedDev : Runtime.Autograd.Cuda.AnyBuffer :=
        { s := p.s, buf := Runtime.Autograd.Cuda.Buffer.axpy pBuf gAny.buf (-lrF) }
      p.setCuda updatedDev
    else
      throw <| IO.userError "torch: internal grad shape mismatch during CUDA SGD"

/-- Apply Adam using an already-computed sparse CUDA gradient map. -/
def adamStepAllCudaMap {α : Type} [TensorTransfer α] (s : EagerSession α) [DecidableEq Shape]
    (configRef : IO.Ref (Option CudaAdamConfig)) (stateRef : IO.Ref CudaAdamState)
    (lr beta1 beta2 epsilon : α)
    (grads : CudaGradMap) : IO Unit := do
  if Options.device s.opts != .cuda then
    throw <| IO.userError "torch: adamStepAllCudaMap called on non-CUDA eager session"
  let lrF ← TensorTransfer.toFloat (α := α) lr
  let beta1F ← TensorTransfer.toFloat (α := α) beta1
  let beta2F ← TensorTransfer.toFloat (α := α) beta2
  let epsF ← TensorTransfer.toFloat (α := α) epsilon
  checkCudaLearningRate "CUDA Adam" lrF
  let config : CudaAdamConfig :=
    { kind := .adam, beta1 := beta1F, beta2 := beta2F, epsilon := epsF, weightDecay := 0.0 }
  match config.validate with
  | .ok () => pure ()
  | .error message => throw <| IO.userError message
  let oneMinusBeta1 := 1.0 - beta1F
  let oneMinusBeta2 := 1.0 - beta2F
  let t0 ← s.cudaTape.get
  let params ← s.paramsByLeaf.get
  let mut state ← stateRef.get
  checkCudaOptimizerInputs "CUDA Adam" t0 params grads (some state)
  ensureCudaAdamConfig configRef config
  for (id, p) in params.toList.filter (fun entry => entry.2.requiresGrad) do
    let gAny ← match grads.get? id with
      | some g => pure g
      | none => throw <| IO.userError "torch: gradient map missing parameter during CUDA Adam"
    if _hs : gAny.s = p.s then
      let pBuf ← okOrThrow <|
        Runtime.Autograd.Cuda.Tape.requireValue (t := t0) (id := id) (s := p.s)
      let n := Runtime.Autograd.Cuda.Buffer.size pBuf
      let st :=
        match state.get? id with
        | some st => st
        | none =>
            { m := Runtime.Autograd.Cuda.Buffer.zeros n
              v := Runtime.Autograd.Cuda.Buffer.zeros n
              t := 0 }
      let t' := st.t + 1
      let mHatScale := 1.0 / (1.0 - Float.pow beta1F (Float.ofNat t'))
      let vHatScale := 1.0 / (1.0 - Float.pow beta2F (Float.ofNat t'))
      let (updated, m', v') := Runtime.Autograd.Cuda.Buffer.adamStep
        pBuf gAny.buf st.m st.v
        beta1F oneMinusBeta1 beta2F oneMinusBeta2
        mHatScale vHatScale epsF 0.0 (-lrF)
      let updatedDev : Runtime.Autograd.Cuda.AnyBuffer :=
        { s := p.s, buf := updated }
      p.setCuda updatedDev
      releaseCudaBuffer st.m
      releaseCudaBuffer st.v
      state := state.insert id { m := m', v := v', t := t' }
    else
      throw <| IO.userError "torch: internal grad shape mismatch during CUDA Adam"
  for (id, st) in state.toList do
    if params.contains id then
      pure ()
    else
      releaseCudaBuffer st.m
      releaseCudaBuffer st.v
      state := state.erase id
  stateRef.set state

/--
Apply AdamW from a sparse CUDA gradient map.

Normal training uses this sparse map so activation gradients can be released as soon as their
contributions have been propagated.
-/
def adamWStepAllCudaMap {α : Type} [TensorTransfer α] (s : EagerSession α)
    [DecidableEq Shape]
    (configRef : IO.Ref (Option CudaAdamConfig)) (stateRef : IO.Ref CudaAdamState)
    (lr weightDecay beta1 beta2 epsilon : α)
    (grads : CudaGradMap) : IO Unit := do
  if Options.device s.opts != .cuda then
    throw <| IO.userError "torch: adamWStepAllCudaMap called on non-CUDA eager session"
  let lrF ← TensorTransfer.toFloat (α := α) lr
  let wdF ← TensorTransfer.toFloat (α := α) weightDecay
  let beta1F ← TensorTransfer.toFloat (α := α) beta1
  let beta2F ← TensorTransfer.toFloat (α := α) beta2
  let epsF ← TensorTransfer.toFloat (α := α) epsilon
  checkCudaLearningRate "CUDA AdamW" lrF
  let config : CudaAdamConfig :=
    { kind := .adamW, beta1 := beta1F, beta2 := beta2F, epsilon := epsF, weightDecay := wdF }
  match config.validate with
  | .ok () => pure ()
  | .error message => throw <| IO.userError message
  let oneMinusBeta1 := 1.0 - beta1F
  let oneMinusBeta2 := 1.0 - beta2F
  let t0 ← s.cudaTape.get
  let params ← s.paramsByLeaf.get
  let mut state ← stateRef.get
  checkCudaOptimizerInputs "CUDA AdamW" t0 params grads (some state)
  ensureCudaAdamConfig configRef config
  for (id, p) in params.toList.filter (fun entry => entry.2.requiresGrad) do
    let gAny ← match grads.get? id with
      | some g => pure g
      | none => throw <| IO.userError "torch: gradient map missing parameter during CUDA AdamW"
    if _hs : gAny.s = p.s then
      let pBuf ← okOrThrow <|
        Runtime.Autograd.Cuda.Tape.requireValue (t := t0) (id := id) (s := p.s)
      let n := Runtime.Autograd.Cuda.Buffer.size pBuf
      let st :=
        match state.get? id with
        | some st => st
        | none =>
            { m := Runtime.Autograd.Cuda.Buffer.zeros n
              v := Runtime.Autograd.Cuda.Buffer.zeros n
              t := 0 }
      let t' := st.t + 1
      let mHatScale := 1.0 / (1.0 - Float.pow beta1F (Float.ofNat t'))
      let vHatScale := 1.0 / (1.0 - Float.pow beta2F (Float.ofNat t'))
      let (updated, m', v') := Runtime.Autograd.Cuda.Buffer.adamStep
        pBuf gAny.buf st.m st.v
        beta1F oneMinusBeta1 beta2F oneMinusBeta2
        mHatScale vHatScale epsF (-(lrF * wdF)) (-lrF)
      let updatedDev : Runtime.Autograd.Cuda.AnyBuffer :=
        { s := p.s, buf := updated }
      p.setCuda updatedDev
      releaseCudaBuffer st.m
      releaseCudaBuffer st.v
      state := state.insert id { m := m', v := v', t := t' }
    else
      throw <| IO.userError "torch: internal grad shape mismatch during CUDA AdamW"
  for (id, st) in state.toList do
    if params.contains id then
      pure ()
    else
      releaseCudaBuffer st.m
      releaseCudaBuffer st.v
      state := state.erase id
  stateRef.set state

end EagerSession

end Internal
end Torch
end Autograd
end Runtime

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.HardMask
public import NN.IR.Payload
public import NN.Runtime.Context
public import NN.Runtime.Autograd.TorchLean.Random
public import NN.Spec.Core.Shape
public import NN.Spec.Core.Tensor.Packed
public import NN.Spec.Layers.Activation
public import NN.Spec.Layers.Attention
public import NN.Spec.Layers.Normalization
public import NN.Spec.Layers.Pooling

import NN.Spec.Core.Tensor.Linalg

/-!
# Semantics

Denotational semantics for `NN.IR.Graph`.

This file defines an evaluator for the current IR fragment:
- it evaluates nodes in SSA/topological order,
- each node applies the corresponding *spec-layer* tensor operation to its parents,
- parameter payloads for `const`, `linear`, and `conv2d` are supplied by an explicit `Payload`.

The evaluator is total on well-formed, well-shaped graphs and returns `Except String` on malformed
graphs or missing payloads.

Softmax and layer norm:
- `softmax axis` normalizes independently along the zero-based tensor dimension named by `axis`.
  The canonical specification handles every in-bounds dimension and preserves the tensor shape.
  This matches the meaning of `torch.softmax(x, dim=axis)` in PyTorch.
- `layernorm axis` matches PyTorch's `F.layer_norm(x, normalized_shape=x.shape[axis:])` convention:
  `axis` selects the start of the **normalized suffix**. We implement this by reshaping the tensor
  into a 2D view `(seqLen, embedDim)`, applying the spec 2D LayerNorm (`Spec.layerNorm`), then
  reshaping back.

How this relates to PyTorch:
- `Graph.nodes` is analogous to a topologically-sorted IR like FX/TorchScript.
- `Payload` is analogous to “parameters / buffers / constants” that live outside the pure graph
  structure.
- The evaluator is a pure, denotational model of running the graph. It is designed for clarity and
  for connecting to proofs and verification passes (not for performance).

References / related systems:
- PyTorch FX: https://pytorch.org/docs/stable/fx.html
- TorchScript: https://pytorch.org/docs/stable/jit.html
- ONNX (graph + initializers): https://onnx.ai/
-/

@[expose] public section


namespace NN.IR

open _root_.Spec
open _root_.Spec.Tensor

namespace Graph

/-! ## Permutation lowering -/

/--
Compute a sequence of adjacent swaps that realizes a target permutation.

This is used to implement `.permute` by repeatedly applying `swapAdjacentAtDepth`, which is already
available in the spec tensor library. If the permutation is ill-formed, this returns an error
explaining what went wrong.
-/
def swapDepthsForPerm (perm : List Nat) (r : Nat) : Except String (List Nat) := do
  let mut cur : List Nat := List.range r
  let mut swapsRev : List Nat := []
  for i in [0:r] do
    match perm[i]? with
    | none => throw s!"permute: internal error: missing perm[{i}]"
    | some target =>
        match cur.findIdx? (· == target) with
        | none => throw s!"permute: internal error: target axis {target} not in current axes {cur}"
        | some j =>
            let mut k := j
            while k > i do
              swapsRev := (k - 1) :: swapsRev
              cur := Spec.Shape.swapAdjacentAxes cur (k - 1)
              k := k - 1
  pure swapsRev.reverse

/--
Permute a shape-tagged tensor according to `perm`.

This checks that `perm` is a valid permutation for the input shape (using `Shape.permute?`), then
lowers it to a sequence of adjacent swaps and applies them to the tensor.
-/
def permutePackedTensor {α : Type} [Context α]
    (v : Spec.PackedTensor α) (perm : List Nat) : Except String (Spec.PackedTensor α) := do
  let sIn := v.shape
  match Spec.Shape.permute? sIn perm with
  | none => throw s!"permute: invalid permutation {repr perm} for shape {repr sIn}"
  | some _ =>
      let swaps ← swapDepthsForPerm perm (Spec.Shape.rank sIn)
      pure <| swaps.foldl (fun acc d => Spec.PackedTensor.swapAdjacentAtDepth acc d) v

/-!
## Evaluation helpers

The evaluator itself (`evalAt` / `denoteAll`) is a fold over nodes. These helpers keep the fold
readable:
- `expectShape` checks a value's stored shape against the node's declared `outShape`.
- `evalConst`/`evalLinear`/`evalConv2d` fetch and apply external payloads keyed by node id.
-/

/-- Check that a packed tensor has the expected shape and recover its statically typed tensor. -/
def expectShape {α : Type} [Context α] [DecidableEq Shape]
    (expected : Shape) (v : Spec.PackedTensor α) : Except String (Tensor α expected) := do
  if h : v.shape = expected then
    -- transport across the shape equality
    pure (h ▸ v.tensor)
  else
    throw s!"IR eval: shape mismatch: expected {repr expected}, got {repr v.shape}"

/-- Denominator for totalized mean reductions over a dynamic IR shape.

For nonempty shapes this is the real element count. For empty shapes, the mathematical mean is
undefined; the IR is total, so it uses denominator `1` and the empty sum contributes `0`.
-/
def meanDenom (s : Shape) : Nat :=
  if Spec.Shape.size s = 0 then 1 else Spec.Shape.size s

/-- Evaluate MSE loss on two packed tensors after checking that their stored shapes agree. -/
def mseLossPackedTensor {α : Type} [Context α] [DecidableEq Shape]
    (i : Nat) (yVal tVal : Spec.PackedTensor α) : Except String (Spec.PackedTensor α) := do
  if h : yVal.shape = tVal.shape then
    let yT : Tensor α yVal.shape := yVal.tensor
    let tT : Tensor α yVal.shape := h.symm ▸ tVal.tensor
    let s := yVal.shape
    let diff := Tensor.subSpec (α := α) yT tT
    let sq := Tensor.mulSpec (α := α) diff diff
    let total : α := Tensor.sumSpec (α := α) sq
    let mean : α := total / (↑(meanDenom s) : α)
    pure (Spec.PackedTensor.mk (α := α) Shape.scalar (Tensor.scalar mean))
  else
    throw <|
      s!"IR eval: node {i}: mse_loss expects equal shapes, got " ++
        s!"{repr yVal.shape} vs {repr tVal.shape}"

@[simp] theorem mseLossPackedTensor_mk {α : Type} [Context α] [DecidableEq Shape]
    (i : Nat) {s : Shape} (y t : Tensor α s) :
    mseLossPackedTensor (α := α) i (Spec.PackedTensor.mk (α := α) s y) (Spec.PackedTensor.mk (α := α) s t) =
      .ok (Spec.PackedTensor.mk (α := α) Shape.scalar
        (Tensor.scalar
          (((Tensor.subSpec (α := α) y t).mulSpec (Tensor.subSpec (α := α) y t)).sumSpec /
            (↑(meanDenom s) : α)))) := by
  simp [mseLossPackedTensor]
  rfl

/-- Transport a `Tensor α (dim n scalar)` across an equality `n = n'` (helper for payload casts). -/
def castDimScalar {α : Type} [Context α] {n n' : Nat}
    (h : n = n') (t : Tensor α (Shape.dim n Shape.scalar)) : Tensor α (Shape.dim n' Shape.scalar) :=
      by
  simpa [h] using t

/--
Evaluate a `const` node from the external payload.

Constants are stored “flat” (1D) for convenience, so we check the flattened length matches
`Spec.Shape.size s` and then `unflatten` to the requested shape.
-/
def evalConst {α : Type} [Context α]
    (payload : Payload α) (id : Nat) (s : Shape) : Except String (Tensor α s) := do
  match payload.const? id with
  | none => throw s!"IR eval: missing const payload for node {id}"
  | some c =>
      if h : c.n = Spec.Shape.size s then
        let v' : Tensor α (.dim (Spec.Shape.size s) .scalar) := castDimScalar (α := α) h c.v
        pure (Tensor.unflattenSpec (α := α) (s := s) v')
      else
        throw s!"IR eval: const {id}: flat length mismatch: have {c.n}, expected {Spec.Shape.size s}"

/--
Evaluate a `linear` node from the external payload.

We enforce:
- the input value has shape `(inDim)`, and
- the node's declared outShape matches `(outDim)`.

The actual math is the usual affine map: $y=Wx+b$.
-/
def evalLinear {α : Type} [Context α] [DecidableEq Shape]
    (payload : Payload α) (id : Nat) (x : Spec.PackedTensor α) (outShape : Shape) : Except String (Spec.PackedTensor α) := do
  match payload.linear? id with
  | none => throw s!"IR eval: missing linear payload for node {id}"
  | some p =>
      let expectedIn : Shape := Shape.dim p.inDim Shape.scalar
      let expectedOut : Shape := Shape.dim p.outDim Shape.scalar
      let xT ← expectShape (α := α) (expected := expectedIn) x
      if hOut : outShape = expectedOut then
        let y : Tensor α (Shape.dim p.outDim Shape.scalar) :=
        Tensor.addSpec (α := α) (Spec.matVecMulSpec (α := α) (m := p.outDim) (n := p.inDim) p.W
          xT) p.b
        pure (Spec.PackedTensor.mk (α := α) outShape (hOut ▸ y))
      else
          throw <|
          s!"IR eval: linear {id}: declared outShape mismatch: {repr outShape} vs " ++
            s!"expected {repr expectedOut}"

/--
Evaluate a `conv2d` node from the external payload.

The output shape is computed with the standard (no dilation) formula
$$
\mathrm{out}=\left\lfloor
  \frac{\mathrm{in}+2\,\mathrm{pad}-k}{\mathrm{stride}}
\right\rfloor+1
$$
for each spatial dimension.
-/
def evalConv2d {α : Type} [Context α] [DecidableEq Shape]
    (payload : Payload α) (id : Nat) (x : Spec.PackedTensor α) : Except String (Spec.PackedTensor α) := do
  match payload.conv2d? id with
  | none => throw s!"IR eval: missing conv2d payload for node {id}"
  | some cfg =>
      let _ ←
        OpContracts.inferConv2dOutShape cfg.inC cfg.outC cfg.kH cfg.kW cfg.stride cfg.padding
          x.shape
      let xT ← expectShape (α := α)
        (expected := Shape.dim cfg.inC (Shape.dim cfg.inH (Shape.dim cfg.inW Shape.scalar))) x
      let y := Spec.conv2dSpec (α := α)
        (layer := cfg.spec)
        (input := xT)
      let outH : Nat := Shape.slidingWindowOutDim cfg.inH cfg.kH cfg.stride cfg.padding
      let outW : Nat := Shape.slidingWindowOutDim cfg.inW cfg.kW cfg.stride cfg.padding
      let outShape : Shape := Shape.dim cfg.outC (Shape.dim outH (Shape.dim outW Shape.scalar))
      pure (Spec.PackedTensor.mk (α := α) outShape y)

/--
Apply fixed-statistics BatchNorm2d to a batched channel-first tensor.

The input shape records the batch, channel, height, and width axes. Naming the shared tensor
operation independently of that layout notation keeps lowering and evaluator code concise while
the type retains the exact contract.
-/
def batchNorm2dEvalTensor {α : Type} [Context α]
    (cfg : BatchNorm2dNchwEvalParams α) {n h w : Nat}
    (x : Tensor α (.dim n (.dim cfg.c (.dim h (.dim w .scalar))))) :
    Tensor α (.dim n (.dim cfg.c (.dim h (.dim w .scalar)))) :=
  Tensor.dim fun ni =>
    Tensor.dim fun ci =>
      Tensor.dim fun hi =>
        Tensor.dim fun wi =>
          match getAtSpec (getAtSpec (getAtSpec (getAtSpec x ni) ci) hi) wi,
              getAtSpec cfg.gamma ci, getAtSpec cfg.beta ci,
              getAtSpec cfg.mean ci, getAtSpec cfg.var ci with
          | .scalar xv, .scalar gamma, .scalar beta, .scalar mean, .scalar var =>
              let denom := MathFunctions.sqrt (max var (0 : α) + cfg.eps)
              Tensor.scalar (((xv - mean) / denom) * gamma + beta)

/-- Evaluate eval-mode BatchNorm2d over a batched channel-first tensor. -/
def evalBatchNorm2dNchwEval {α : Type} [Context α] [DecidableEq Shape]
    (payload : Payload α) (id : Nat) (x : Spec.PackedTensor α) : Except String (Spec.PackedTensor α) := do
  match payload.batchNorm2dNchwEval? id with
  | none => throw s!"IR eval: missing batch_norm2d_nchw_eval payload for node {id}"
  | some cfg =>
      match x.shape with
      | .dim n (.dim c (.dim h (.dim w .scalar))) =>
          if _hc : c = cfg.c then
            let xT ← expectShape (α := α)
              (expected := .dim n (.dim cfg.c (.dim h (.dim w .scalar)))) x
            let y : Tensor α (.dim n (.dim cfg.c (.dim h (.dim w .scalar)))) :=
              batchNorm2dEvalTensor (α := α) cfg xT
            pure (Spec.PackedTensor.mk (α := α) (.dim n (.dim cfg.c (.dim h (.dim w .scalar)))) y)
          else
            throw s!"IR eval: batch_norm2d_nchw_eval channel mismatch: input={c}, payload={cfg.c}"
      | s =>
          throw s!"IR eval: batch_norm2d_nchw_eval expects NCHW input, got {repr s}"

/-- Layer normalization without a learned affine transform ($\gamma=1$, $\beta=0$). -/
def layerNormWithoutAffine {α : Type} [Context α]
    (seqLen embedDim : Nat) (x : Tensor α (Shape.dim seqLen (Shape.dim embedDim Shape.scalar))) :
    Except String (Tensor α (Shape.dim seqLen (Shape.dim embedDim Shape.scalar))) := do
  if hSeq : seqLen > 0 then
    if hEmb : embedDim > 0 then
      let gamma : Tensor α (Shape.dim embedDim Shape.scalar) :=
        Spec.fill (α := α) 1 (Shape.dim embedDim Shape.scalar)
      let beta : Tensor α (Shape.dim embedDim Shape.scalar) :=
        Spec.fill (α := α) 0 (Shape.dim embedDim Shape.scalar)
      pure (Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
        (x := x) (gamma := gamma) (beta := beta) (h_seq_pos := hSeq) (h_embed_pos := hEmb))
    else
      throw s!"layernorm: embedDim must be > 0 (got {embedDim})"
  else
    throw s!"layernorm: seqLen must be > 0 (got {seqLen})"

/--
Decode a dynamic concat parent as a tensor with an existential leading dimension and the requested
tail shape. This is the checked boundary shared by list-indexed concat evaluation and its proofs.
-/
def expectLeadingAxisInput {α : Type} [Context α]
    (i : Nat) (rest : Shape) (value : Spec.PackedTensor α) :
    Except String (Sigma fun size => Tensor α (Shape.dim size rest)) := do
  match value with
  | ⟨Shape.dim size actualRest, tensor⟩ =>
      if hRest : actualRest = rest then
        let tensor' : Tensor α (Shape.dim size rest) := by
          simpa [hRest] using tensor
        pure ⟨size, tensor'⟩
      else
        throw <|
          s!"IR eval: node {i}: concat: tail mismatch: {repr actualRest} vs {repr rest}"
  | ⟨_, _⟩ =>
      throw s!"IR eval: node {i}: concat expects rank≥1 parents, got {repr value.shape}"

/-- Fold leading-axis concat over dynamic values that already share the same tail shape. -/
def evalConcatLeadingAxisFold {α : Type} [Context α]
    (i : Nat) (nOut : Nat) (rest : Shape) (parents : List (Spec.PackedTensor α)) :
    Except String (Spec.PackedTensor α) := do
  let sigs ← parents.mapM (expectLeadingAxisInput (α := α) i rest)
  match sigs with
  | [] =>
      throw s!"IR eval: node {i}: concat internal error"
  | s0 :: srest =>
      let outSigma :=
        srest.foldl
          (fun acc nxt =>
            match acc, nxt with
            | ⟨n1, t1⟩, ⟨n2, t2⟩ =>
                ⟨n1 + n2, Tensor.concatLeadingAxisSpec (α := α) (n := n1) (m := n2) (s := rest)
                  t1 t2⟩)
          s0
      match outSigma with
      | ⟨nSum, tSum⟩ =>
          if h : nSum = nOut then
            let y : Tensor α (Shape.dim nOut rest) := by
              simpa [h] using tSum
            pure (Spec.PackedTensor.mk (α := α) (Shape.dim nOut rest) y)
          else
            throw <|
              s!"IR eval: node {i}: concat out dim mismatch: declared {nOut}, " ++
                s!"computed {nSum}"

/--
Evaluate a `concat` node from already evaluated parent values.

The IR concat operation accepts any valid axis.  The tensor primitive concatenates along axis `0`,
so the evaluator implements the generic case by moving the requested axis to the front, folding
`Tensor.concatLeadingAxisSpec` over the permuted parents, and moving the result back.
-/
def evalConcat {α : Type} [Context α] [DecidableEq Shape]
    (i : Nat) (n : Node) (axis : Nat) (parents : List (Spec.PackedTensor α)) : Except String (Spec.PackedTensor α) := do
  let expected ←
    match OpContracts.inferConcatOutShape axis (parents.map (fun pv => pv.shape)) with
    | .ok s => pure s
    | .error msg => throw s!"IR eval: node {i}: {msg} ({n.summary})"
  if expected != n.outShape then
    throw <|
      s!"IR eval: node {i}: concat outShape mismatch: " ++
        s!"expected={repr expected}, declared={repr n.outShape} ({n.summary})"

  if axis = 0 then
    match n.outShape with
    | Shape.dim nOut rest =>
        evalConcatLeadingAxisFold (α := α) i nOut rest parents
    | _ =>
        throw s!"IR eval: node {i}: concat expects rank≥1 outShape, got {repr n.outShape}"
  else
  let permFront ←
    match OpContracts.permMoveAxisToFront axis n.outShape with
    | .ok perm => pure perm
    | .error msg => throw s!"IR eval: node {i}: concat: {msg} ({n.summary})"
  let permBack ←
    match OpContracts.inversePerm permFront with
    | .ok perm => pure perm
    | .error msg => throw s!"IR eval: node {i}: concat: {msg} ({n.summary})"
  let outPermShape ←
    match Spec.Shape.permute? n.outShape permFront with
    | some s => pure s
    | none =>
        throw <|
          s!"IR eval: node {i}: concat: internal error (invalid permutation for " ++
            s!"outShape) ({n.summary})"
  let parentsPerm : List (Spec.PackedTensor α) ←
    parents.mapM (fun pv => do
      match permutePackedTensor (α := α) pv permFront with
      | .ok v => pure v
      | .error msg => throw s!"IR eval: node {i}: concat: {msg} ({n.summary})")
  match outPermShape with
  | Shape.dim nOut rest =>
      let toSigma (pv : Spec.PackedTensor α) : Except String (Sigma fun n => Tensor α (Shape.dim n rest))
        := do
        match pv with
        | ⟨Shape.dim nP restP, t⟩ =>
            if hRest : restP = rest then
              let t' : Tensor α (Shape.dim nP rest) := by
                simpa [hRest] using t
              pure ⟨nP, t'⟩
            else
              throw <|
                s!"IR eval: node {i}: concat: permuted tail mismatch: {repr restP} vs " ++
                  s!"{repr rest}"
        | ⟨_, _⟩ =>
            throw s!"IR eval: node {i}: concat expects rank≥1 parents, got {repr pv.shape}"
      let sigs ← parentsPerm.mapM toSigma
      match sigs with
      | [] =>
          throw s!"IR eval: node {i}: concat internal error"
      | s0 :: srest =>
          let outSigma :=
            srest.foldl
              (fun acc nxt =>
                match acc, nxt with
                | ⟨n1, t1⟩, ⟨n2, t2⟩ =>
                    ⟨n1 + n2, Tensor.concatLeadingAxisSpec (α := α) (n := n1) (m := n2) (s := rest)
                      t1 t2⟩)
              s0
          match outSigma with
          | ⟨nSum, tSum⟩ =>
              if h : nSum = nOut then
                let yPerm : Tensor α (Shape.dim nOut rest) := by
                  simpa [h] using tSum
                let outPerm : Spec.PackedTensor α := Spec.PackedTensor.mk (α := α) (Shape.dim nOut rest) yPerm
                let out0 ←
                  match permutePackedTensor (α := α) outPerm permBack with
                  | .ok v => pure v
                  | .error msg => throw s!"IR eval: node {i}: concat: {msg} ({n.summary})"
                let y ← expectShape (α := α) (expected := n.outShape) out0
                pure (Spec.PackedTensor.mk (α := α) n.outShape y)
              else
                throw <|
                  s!"IR eval: node {i}: concat out dim mismatch: declared {nOut}, " ++
                    s!"computed {nSum}"
  | _ =>
      throw s!"IR eval: node {i}: concat expects rank≥1 outShape, got {repr n.outShape}"

/-- Normalize a node result to the node's declared shape, rejecting inconsistent implementations. -/
def normalizeNodeOutput {α : Type} [Context α] [DecidableEq Shape]
    (i : Nat) (n : Node) (v : Spec.PackedTensor α) : Except String (Spec.PackedTensor α) :=
  if h : v.shape = n.outShape then
    pure (Spec.PackedTensor.mk (α := α) n.outShape (h ▸ v.tensor))
  else
    throw <|
      s!"IR eval: node {i}: produced shape mismatch: produced={repr v.shape}, " ++
        s!"declared={repr n.outShape} ({n.summary})"

@[simp]
theorem normalizeNodeOutput_declared {α : Type} [Context α] [DecidableEq Shape]
    (i : Nat) (n : Node) (t : Tensor α n.outShape) :
    normalizeNodeOutput (α := α) i n (Spec.PackedTensor.mk (α := α) n.outShape t) =
      .ok (Spec.PackedTensor.mk (α := α) n.outShape t) := by
  simp [normalizeNodeOutput, Pure.pure, Except.pure]

@[simp]
theorem normalizeNodeOutput_nodeShape {α : Type} [Context α] [DecidableEq Shape]
    (i id : Nat) (parents : List Nat) (kind : OpKind) (s : Shape) (t : Tensor α s) :
    normalizeNodeOutput (α := α) i { id := id, parents := parents, kind := kind, outShape := s }
        ⟨s, t⟩ =
      .ok ⟨s, t⟩ := by
  simp [normalizeNodeOutput, Pure.pure, Except.pure]

/-- Read an already-evaluated parent, reporting the node and available prefix on failure. -/
@[simp] def getParentValue {α : Type} (vals : Array (Spec.PackedTensor α)) (i : Nat) (n : Node)
    (pid : Nat) :
    Except String (Spec.PackedTensor α) :=
  match vals[pid]? with
  | some parent => pure parent
  | none =>
      throw <|
        s!"IR eval: node {i}: parent {pid} has not been evaluated " ++
          s!"(available values: {vals.size}) ({n.summary})"

/--
Evaluate a known node from its already computed parent values.

Keeping operator dispatch separate from graph lookup lets local correctness proofs reduce only the
selected `OpKind` branch. The caller remains responsible for the graph's topological invariant.
-/
def evalNode
    {α : Type} [Context α] [DecidableEq Shape]
    (payload : Payload α) (input : Spec.PackedTensor α) (vals : Array (Spec.PackedTensor α)) (i : Nat) (n : Node) :
    Except String (Spec.PackedTensor α) := do
  let getParent := getParentValue vals i n
  let v : Spec.PackedTensor α ←
    match n.kind with
    | .input =>
        let t ← expectShape (α := α) (expected := n.outShape) input
        pure (Spec.PackedTensor.mk (α := α) n.outShape t)
    | .const s =>
        let t ← evalConst (α := α) (payload := payload) (id := n.id) (s := s)
        pure (Spec.PackedTensor.mk (α := α) s t)
    | .permute perm =>
        match n.parents with
        | [pId] =>
            let vOut ← permutePackedTensor (α := α) (v := ← getParent pId) perm
            if h : vOut.shape = n.outShape then
              pure (Spec.PackedTensor.mk (α := α) n.outShape (h ▸ vOut.tensor))
            else
              throw <|
                s!"IR eval: node {i}: permute outShape mismatch: " ++
                  s!"computed={repr vOut.shape}, declared={repr n.outShape} ({n.summary})"
        | _ => throw s!"IR eval: node {i}: permute expects 1 parent ({n.summary})"
    | .detach =>
        match n.parents with
        | [pId] =>
            let p ← expectShape (α := α) (expected := n.outShape) (← getParent pId)
            pure (Spec.PackedTensor.mk (α := α) n.outShape p)
        | _ => throw s!"IR eval: node {i}: detach expects 1 parent ({n.summary})"
    | .randUniform seed =>
        match n.parents with
        | [] =>
            let key := Runtime.Autograd.TorchLean.Random.keyOf seed i
            let t : Tensor α n.outShape :=
              Runtime.Autograd.TorchLean.Random.uniform (α := α) key (s := n.outShape)
            pure (Spec.PackedTensor.mk (α := α) n.outShape t)
        | _ => throw s!"IR eval: node {i}: rand_uniform expects 0 parents ({n.summary})"
    | .bernoulliMask seed =>
        match n.parents with
        | [pId] =>
            let pV ← getParent pId
            match pV with
            | ⟨.scalar, Tensor.scalar keepProb⟩ =>
                let key := Runtime.Autograd.TorchLean.Random.keyOf seed i
                let t : Tensor α n.outShape :=
                  Runtime.Autograd.TorchLean.Random.mask (α := α) key keepProb (s := n.outShape)
                pure (Spec.PackedTensor.mk (α := α) n.outShape t)
            | ⟨_, _⟩ =>
                throw
                  s!"IR eval: node {i}: bernoulli_mask expects scalar keepProb parent ({n.summary})"
        | _ => throw s!"IR eval: node {i}: bernoulli_mask expects 1 parent ({n.summary})"
    | .add =>
        match n.parents with
        | [aId, bId] =>
            let a ← expectShape (α := α) (expected := n.outShape) (← getParent aId)
            let b ← expectShape (α := α) (expected := n.outShape) (← getParent bId)
            pure (Spec.PackedTensor.mk (α := α) n.outShape (Tensor.addSpec (α := α) a b))
        | _ => throw s!"IR eval: node {i}: add expects 2 parents ({n.summary})"
    | .sub =>
        match n.parents with
        | [aId, bId] =>
            let a ← expectShape (α := α) (expected := n.outShape) (← getParent aId)
            let b ← expectShape (α := α) (expected := n.outShape) (← getParent bId)
            pure (Spec.PackedTensor.mk (α := α) n.outShape (Tensor.subSpec (α := α) a b))
        | _ => throw s!"IR eval: node {i}: sub expects 2 parents ({n.summary})"
    | .mul_elem =>
        match n.parents with
        | [aId, bId] =>
            let a ← expectShape (α := α) (expected := n.outShape) (← getParent aId)
            let b ← expectShape (α := α) (expected := n.outShape) (← getParent bId)
            pure (Spec.PackedTensor.mk (α := α) n.outShape (Tensor.mulSpec (α := α) a b))
        | _ => throw s!"IR eval: node {i}: mul_elem expects 2 parents ({n.summary})"
    | .abs =>
        match n.parents with
        | [pId] =>
            let p ← expectShape (α := α) (expected := n.outShape) (← getParent pId)
            pure (Spec.PackedTensor.mk (α := α) n.outShape (Tensor.absSpec (α := α) p))
        | _ => throw s!"IR eval: node {i}: abs expects 1 parent ({n.summary})"
    | .sqrt =>
        match n.parents with
        | [pId] =>
            let p ← expectShape (α := α) (expected := n.outShape) (← getParent pId)
            pure (Spec.PackedTensor.mk (α := α) n.outShape (Tensor.sqrtSpec (α := α) p))
        | _ => throw s!"IR eval: node {i}: sqrt expects 1 parent ({n.summary})"
    | .maxElem =>
        match n.parents with
        | [aId, bId] =>
            let a ← expectShape (α := α) (expected := n.outShape) (← getParent aId)
            let b ← expectShape (α := α) (expected := n.outShape) (← getParent bId)
            pure (Spec.PackedTensor.mk (α := α) n.outShape (Tensor.maxSpec (α := α) a b))
        | _ => throw s!"IR eval: node {i}: max_elem expects 2 parents ({n.summary})"
    | .minElem =>
        match n.parents with
        | [aId, bId] =>
            let a ← expectShape (α := α) (expected := n.outShape) (← getParent aId)
            let b ← expectShape (α := α) (expected := n.outShape) (← getParent bId)
            pure (Spec.PackedTensor.mk (α := α) n.outShape (Tensor.minSpec (α := α) a b))
        | _ => throw s!"IR eval: node {i}: min_elem expects 2 parents ({n.summary})"
    | .maxPool2d kH kW stride =>
        match n.parents with
        | [pId] =>
            let pV ← getParent pId
            match pV.shape with
            | .dim inC (.dim inH (.dim inW .scalar)) =>
                if hkH : kH = 0 then
                  throw s!"IR eval: node {i}: max_pool2d requires kH ≠ 0 ({n.summary})"
                else if hkW : kW = 0 then
                  throw s!"IR eval: node {i}: max_pool2d requires kW ≠ 0 ({n.summary})"
                else if hs : stride = 0 then
                  throw s!"IR eval: node {i}: max_pool2d requires stride ≠ 0 ({n.summary})"
                else
                  OpContracts.checkWindowFits "max_pool2d" "height" inH kH 0
                  OpContracts.checkWindowFits "max_pool2d" "width" inW kW 0
                  let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
                  let xCHW ← expectShape (α := α) (expected := sIn) pV
                  let expected : Shape := Spec.pool2dMultiOutShape inC inH inW kH kW stride
                  let layer : Spec.MaxPool2dSpec kH kW stride hkH hkW hs := {}
                  let y : Tensor α expected :=
                    Spec.maxPool2dMultiSpec (α := α) (kH := kH) (kW := kW)
                      (inH := inH) (inW := inW) (inC := inC) (stride := stride)
                      (layer := layer) (input := xCHW)
                  if h : expected = n.outShape then
                    pure (Spec.PackedTensor.mk (α := α) n.outShape (h ▸ y))
                  else
                    throw <|
                      s!"IR eval: node {i}: max_pool2d outShape mismatch: " ++
                        s!"expected={repr expected}, declared={repr n.outShape} ({n.summary})"
            | _ =>
                throw s!"IR eval: node {i}: max_pool2d expects CHW parent shape ({n.summary})"
        | _ => throw s!"IR eval: node {i}: max_pool2d expects 1 parent ({n.summary})"
    | .maxPool2dPad kH kW stride padding =>
        match n.parents with
        | [pId] =>
            let pV ← getParent pId
            match pV.shape with
            | .dim inC (.dim inH (.dim inW .scalar)) =>
                if hkH : kH = 0 then
                  throw s!"IR eval: node {i}: max_pool2d_pad requires kH ≠ 0 ({n.summary})"
                else if hkW : kW = 0 then
                  throw s!"IR eval: node {i}: max_pool2d_pad requires kW ≠ 0 ({n.summary})"
                else if hs : stride = 0 then
                  throw s!"IR eval: node {i}: max_pool2d_pad requires stride ≠ 0 ({n.summary})"
                else
                  OpContracts.checkWindowFits "max_pool2d_pad" "height" inH kH padding
                  OpContracts.checkWindowFits "max_pool2d_pad" "width" inW kW padding
                  let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
                  let xCHW ← expectShape (α := α) (expected := sIn) pV
                  let expected : Shape :=
                    Spec.pool2dMultiOutShapePad inC inH inW kH kW stride padding
                  let layer : Spec.MaxPool2dSpec kH kW stride hkH hkW hs := {}
                  let y : Tensor α expected :=
                    Spec.maxPool2dMultiSpecPad (α := α) (kH := kH) (kW := kW)
                      (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding := padding)
                      (layer := layer) (input := xCHW)
                  if h : expected = n.outShape then
                    pure (Spec.PackedTensor.mk (α := α) n.outShape (h ▸ y))
                  else
                    throw <|
                      s!"IR eval: node {i}: max_pool2d_pad outShape mismatch: " ++
                        s!"expected={repr expected}, declared={repr n.outShape} ({n.summary})"
            | _ =>
                throw s!"IR eval: node {i}: max_pool2d_pad expects CHW parent shape ({n.summary})"
        | _ => throw s!"IR eval: node {i}: max_pool2d_pad expects 1 parent ({n.summary})"
    | .avgPool2d kH kW stride =>
        match n.parents with
        | [pId] =>
            let pV ← getParent pId
            match pV.shape with
            | .dim inC (.dim inH (.dim inW .scalar)) =>
                if hkH : kH = 0 then
                  throw s!"IR eval: node {i}: avg_pool2d requires kH ≠ 0 ({n.summary})"
                else if hkW : kW = 0 then
                  throw s!"IR eval: node {i}: avg_pool2d requires kW ≠ 0 ({n.summary})"
                else if hs : stride = 0 then
                  throw s!"IR eval: node {i}: avg_pool2d requires stride ≠ 0 ({n.summary})"
                else
                  OpContracts.checkWindowFits "avg_pool2d" "height" inH kH 0
                  OpContracts.checkWindowFits "avg_pool2d" "width" inW kW 0
                  let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
                  let xCHW ← expectShape (α := α) (expected := sIn) pV
                  let expected : Shape := Spec.pool2dMultiOutShape inC inH inW kH kW stride
                  let layer : Spec.AvgPool2dSpec kH kW stride hkH hkW hs := {}
                  let y : Tensor α expected :=
                    Spec.avgPool2dMultiSpec (α := α) (kH := kH) (kW := kW)
                      (inH := inH) (inW := inW) (inC := inC) (stride := stride)
                      (h1 := hkH) (h2 := hkW) (layer := layer) (input := xCHW)
                  if h : expected = n.outShape then
                    pure (Spec.PackedTensor.mk (α := α) n.outShape (h ▸ y))
                  else
                    throw <|
                      s!"IR eval: node {i}: avg_pool2d outShape mismatch: " ++
                        s!"expected={repr expected}, declared={repr n.outShape} ({n.summary})"
            | _ =>
                throw s!"IR eval: node {i}: avg_pool2d expects CHW parent shape ({n.summary})"
        | _ => throw s!"IR eval: node {i}: avg_pool2d expects 1 parent ({n.summary})"
    | .avgPool2dPad kH kW stride padding =>
        match n.parents with
        | [pId] =>
            let pV ← getParent pId
            match pV.shape with
            | .dim inC (.dim inH (.dim inW .scalar)) =>
                if hkH : kH = 0 then
                  throw s!"IR eval: node {i}: avg_pool2d_pad requires kH ≠ 0 ({n.summary})"
                else if hkW : kW = 0 then
                  throw s!"IR eval: node {i}: avg_pool2d_pad requires kW ≠ 0 ({n.summary})"
                else if hs : stride = 0 then
                  throw s!"IR eval: node {i}: avg_pool2d_pad requires stride ≠ 0 ({n.summary})"
                else
                  OpContracts.checkWindowFits "avg_pool2d_pad" "height" inH kH padding
                  OpContracts.checkWindowFits "avg_pool2d_pad" "width" inW kW padding
                  let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
                  let xCHW ← expectShape (α := α) (expected := sIn) pV
                  let expected : Shape :=
                    Spec.pool2dMultiOutShapePad inC inH inW kH kW stride padding
                  let layer : Spec.AvgPool2dSpec kH kW stride hkH hkW hs := {}
                  let y : Tensor α expected :=
                    Spec.avgPool2dMultiSpecPad (α := α) (kH := kH) (kW := kW)
                      (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding := padding)
                      (h1 := hkH) (h2 := hkW) (layer := layer) (input := xCHW)
                  if h : expected = n.outShape then
                    pure (Spec.PackedTensor.mk (α := α) n.outShape (h ▸ y))
                  else
                    throw <|
                      s!"IR eval: node {i}: avg_pool2d_pad outShape mismatch: " ++
                        s!"expected={repr expected}, declared={repr n.outShape} ({n.summary})"
            | _ =>
                throw s!"IR eval: node {i}: avg_pool2d_pad expects CHW parent shape ({n.summary})"
        | _ => throw s!"IR eval: node {i}: avg_pool2d_pad expects 1 parent ({n.summary})"
    | .broadcastTo s₁ s₂ =>
        match n.parents with
        | [pId] =>
            let p ← expectShape (α := α) (expected := s₁) (← getParent pId)
            match OpContracts.mkCanBroadcastTo? s₁ s₂ with
            | none => throw s!"IR eval: node {i}: broadcastTo invalid: {repr s₁} → {repr s₂}"
            | some cb =>
                let y := Tensor.broadcastTo (α := α) (s₁ := s₁) (s₂ := s₂) cb p
                pure (Spec.PackedTensor.mk (α := α) s₂ y)
        | _ => throw s!"IR eval: node {i}: broadcastTo expects 1 parent ({n.summary})"
    | .reduceSum axis =>
        match n.parents with
        | [pId] =>
            let pV ← getParent pId
            let s := pV.shape
            let pT : Tensor α s := pV.tensor
            match Spec.Shape.nonemptyAxis? (axis := axis) s with
            | none =>
                let msg :=
                  s!"IR eval: node {i}: reduce_sum invalid axis={axis}" ++
                    s!" for shape {repr s}"
                throw msg
            | some hAxis =>
                let hRed := hAxis.down
                let y := Tensor.reduceSum (α := α) (s := s) axis pT hRed
                pure (Spec.PackedTensor.mk (α := α) (shapeAfterSum s axis) y)
        | _ => throw s!"IR eval: node {i}: reduce_sum expects 1 parent ({n.summary})"
    | .reduceMean axis =>
        match n.parents with
        | [pId] =>
            let pV ← getParent pId
            let s := pV.shape
            let pT : Tensor α s := pV.tensor
            match Spec.Shape.nonemptyAxis? (axis := axis) s with
            | none =>
                let msg :=
                  s!"IR eval: node {i}: reduce_mean invalid axis={axis}" ++
                    s!" for shape {repr s}"
                throw msg
            | some hAxis =>
                let hRed := hAxis.down
                let y := Tensor.reduceMean (α := α) (s := s) axis pT hRed
                pure (Spec.PackedTensor.mk (α := α) (shapeAfterSum s axis) y)
        | _ => throw s!"IR eval: node {i}: reduce_mean expects 1 parent ({n.summary})"
    | .sum =>
        match n.parents with
        | [pId] =>
            let p ← getParent pId
            let s := p.shape
            let t : Tensor α s := p.tensor
            let v : α := Tensor.sumSpec (α := α) t
            pure (Spec.PackedTensor.mk (α := α) .scalar (Tensor.scalar v))
        | _ => throw s!"IR eval: node {i}: sum expects 1 parent ({n.summary})"
    | .matmul =>
        match n.parents with
        | [aId, bId] =>
            let aV ← getParent aId
            let bV ← getParent bId
            match aV.shape, bV.shape with
            | Shape.dim m (Shape.dim n Shape.scalar), Shape.dim n' (Shape.dim p Shape.scalar) =>
                let aT ← expectShape (α := α) (expected := Shape.dim m (Shape.dim n Shape.scalar))
                  aV
                let bT ← expectShape (α := α) (expected := Shape.dim n' (Shape.dim p Shape.scalar))
                  bV
                if h : n = n' then
                  let bT' : Tensor α (Shape.dim n (Shape.dim p Shape.scalar)) := by
                    simpa [h] using bT
                  let y := Spec.matMulSpec (α := α) (m := m) (n := n) (p := p) aT bT'
                  pure (Spec.PackedTensor.mk (α := α) (Shape.dim m (Shape.dim p Shape.scalar)) y)
                else
                  throw s!"IR eval: node {i}: matmul inner dims mismatch: {n} vs {n'}"
            | Shape.dim batch (Shape.dim m (Shape.dim n Shape.scalar)),
              Shape.dim batch' (Shape.dim n' (Shape.dim p Shape.scalar)) =>
                let aT ← expectShape (α := α)
                  (expected := Shape.dim batch (Shape.dim m (Shape.dim n Shape.scalar))) aV
                let bT ← expectShape (α := α)
                  (expected := Shape.dim batch' (Shape.dim n' (Shape.dim p Shape.scalar))) bV
                if hb : batch = batch' then
                  if hn : n = n' then
                    let bT' : Tensor α
                        (Shape.dim batch (Shape.dim n (Shape.dim p Shape.scalar))) := by
                      simpa [hb, hn] using bT
                    let y := Tensor.bmmSpec (α := α) (batch := batch) (m := m) (n := n) (p :=
                      p) aT bT'
                    pure (Spec.PackedTensor.mk (α := α) (Shape.dim batch (Shape.dim m (Shape.dim p
                      Shape.scalar))) y)
                  else
                    throw s!"IR eval: node {i}: matmul inner dims mismatch: {n} vs {n'}"
                else
                  throw s!"IR eval: node {i}: matmul batch dims mismatch: {batch} vs {batch'}"
            | _, _ =>
                throw <|
                  s!"IR eval: node {i}: unsupported matmul shapes: {repr aV.shape} · " ++
                    s!"{repr bV.shape}"
        | _ => throw s!"IR eval: node {i}: matmul expects 2 parents ({n.summary})"
    | .linear =>
        match n.parents with
        | [pId] =>
            evalLinear (α := α) (payload := payload) (id := n.id) (x := ← getParent pId) (outShape :=
              n.outShape)
        | _ => throw s!"IR eval: node {i}: linear expects 1 parent ({n.summary})"
    | .conv2d .. =>
        match n.parents with
        | [pId] =>
            let y ← evalConv2d (α := α) (payload := payload) (id := n.id) (x := ← getParent pId)
            if y.shape != n.outShape then
              throw <|
                s!"IR eval: node {i}: conv2d outShape mismatch: computed={repr y.shape}, " ++
                  s!"declared={repr n.outShape}"
            pure y
        | _ => throw s!"IR eval: node {i}: conv2d expects 1 parent ({n.summary})"
    | .batchNorm2dNchwEval .. =>
        match n.parents with
        | [pId] =>
            let y ← evalBatchNorm2dNchwEval (α := α) (payload := payload) (id := n.id)
              (x := ← getParent pId)
            if y.shape != n.outShape then
              throw <|
                s!"IR eval: node {i}: batch_norm2d_nchw_eval outShape mismatch: " ++
                  s!"computed={repr y.shape}, declared={repr n.outShape}"
            pure y
        | _ => throw s!"IR eval: node {i}: batch_norm2d_nchw_eval expects 1 parent ({n.summary})"
    | .relu =>
        match n.parents with
        | [pId] =>
            let p ← expectShape (α := α) (expected := n.outShape) (← getParent pId)
            pure (Spec.PackedTensor.mk (α := α) n.outShape (Activation.reluSpec (α := α) p))
        | _ => throw s!"IR eval: node {i}: relu expects 1 parent ({n.summary})"
    | .tanh =>
        match n.parents with
        | [pId] =>
            let p ← expectShape (α := α) (expected := n.outShape) (← getParent pId)
            pure (Spec.PackedTensor.mk (α := α) n.outShape (Activation.tanhSpec (α := α) p))
        | _ => throw s!"IR eval: node {i}: tanh expects 1 parent ({n.summary})"
    | .sin =>
        match n.parents with
        | [pId] =>
            let p ← expectShape (α := α) (expected := n.outShape) (← getParent pId)
            pure (Spec.PackedTensor.mk (α := α) n.outShape (Tensor.mapSpec (fun x => MathFunctions.sin x) p))
        | _ => throw s!"IR eval: node {i}: sin expects 1 parent ({n.summary})"
    | .cos =>
        match n.parents with
        | [pId] =>
            let p ← expectShape (α := α) (expected := n.outShape) (← getParent pId)
            pure (Spec.PackedTensor.mk (α := α) n.outShape (Tensor.mapSpec (fun x => MathFunctions.cos x) p))
        | _ => throw s!"IR eval: node {i}: cos expects 1 parent ({n.summary})"
    | .sigmoid =>
        match n.parents with
        | [pId] =>
            let p ← expectShape (α := α) (expected := n.outShape) (← getParent pId)
            pure (Spec.PackedTensor.mk (α := α) n.outShape (Activation.sigmoidSpec (α := α) p))
        | _ => throw s!"IR eval: node {i}: sigmoid expects 1 parent ({n.summary})"
    | .exp =>
        match n.parents with
        | [pId] =>
            let p ← expectShape (α := α) (expected := n.outShape) (← getParent pId)
            pure (Spec.PackedTensor.mk (α := α) n.outShape (Tensor.expSpec (α := α) p))
        | _ => throw s!"IR eval: node {i}: exp expects 1 parent ({n.summary})"
    | .log =>
        match n.parents with
        | [pId] =>
            let p ← expectShape (α := α) (expected := n.outShape) (← getParent pId)
            -- Domain discipline: raw `log` is undefined on nonpositive inputs. The evaluator
            -- rejects that case explicitly; use `safeLogSpec`/`safeLogOp` in models that require
            -- epsilon protection.
            if Tensor.allSpec (α := α) (s := n.outShape) (fun v => decide (0 < v)) p then
              pure (Spec.PackedTensor.mk (α := α) n.outShape (Tensor.logSpec (α := α) p))
            else
              throw
                "IR eval: log: input contains values <= 0 (or NaN); use `safe_log` if you want epsilon protection"
        | _ => throw s!"IR eval: node {i}: log expects 1 parent ({n.summary})"
    | .inv =>
        match n.parents with
        | [pId] =>
            let p ← expectShape (α := α) (expected := n.outShape) (← getParent pId)
            pure (Spec.PackedTensor.mk (α := α) n.outShape (Tensor.invSpec (α := α) p))
        | _ => throw s!"IR eval: node {i}: inv expects 1 parent ({n.summary})"
    | .softmax axis => do
        match n.parents with
        | [pId] =>
            match Spec.Shape.axisInBounds? axis n.outShape with
            | none =>
                throw <| s!"IR eval: node {i}: softmax: invalid axis {axis} for rank " ++
                  s!"{Spec.Shape.rank n.outShape} ({n.summary})"
            | some h =>
                expectShape (α := α) (expected := n.outShape) (← getParent pId) >>= fun p =>
                  pure <| Spec.PackedTensor.mk (α := α) n.outShape <|
                    @Activation.softmaxSpec α _ n.outShape axis h.down p
        | _ =>
            throw s!"IR eval: node {i}: softmax expects 1 parent ({n.summary})"
    | .hardMaskedSoftmax mask =>
        match n.parents with
        | [pId] => do
            let scores ← expectShape (α := α) (expected := n.outShape) (← getParent pId)
            let allowed ←
              match NN.IR.HardMask.toTensorAs? mask n.outShape with
              | .ok value => pure value
              | .error msg =>
                  throw s!"IR eval: node {i}: hard_masked_softmax: {msg} ({n.summary})"
            pure <| Spec.PackedTensor.mk (α := α) n.outShape <|
              hardMaskedSoftmaxLastSpec scores allowed
        | _ =>
            throw s!"IR eval: node {i}: hard_masked_softmax expects 1 parent ({n.summary})"
    | .layernorm axis =>
        match n.parents with
        | [pId] => do
            let x ← expectShape (α := α) (expected := n.outShape) (← getParent pId)
            let (seqLen, embedDim) ←
              match OpContracts.layerNormMatrixDims axis n.outShape with
              | .ok p => pure p
              | .error msg => throw s!"IR eval: node {i}: layernorm: {msg} ({n.summary})"
            let view2d : Shape := Shape.dim seqLen (Shape.dim embedDim Shape.scalar)
            if hNumel : Spec.Shape.size n.outShape = Spec.Shape.size view2d then
              let x2d : Tensor α view2d :=
                Tensor.reshapeSpec (α := α) (s₁ := n.outShape) (s₂ := view2d) x hNumel
              let y2d ← layerNormWithoutAffine (α := α) (seqLen := seqLen) (embedDim := embedDim) x2d
              let y : Tensor α n.outShape :=
                Tensor.reshapeSpec (α := α) (s₁ := view2d) (s₂ := n.outShape) y2d hNumel.symm
              pure (Spec.PackedTensor.mk (α := α) n.outShape y)
            else
              throw <|
                s!"IR eval: node {i}: layernorm internal error: bad reshape sizes " ++
                  s!"({Spec.Shape.size n.outShape} vs {Spec.Shape.size view2d}) ({n.summary})"
        | _ =>
            throw s!"IR eval: node {i}: layernorm expects 1 parent ({n.summary})"
    | .reshape inS outS =>
        match n.parents with
        | [pId] =>
            let pV ← getParent pId
            let pT ← expectShape (α := α) (expected := inS) pV
            if h : Spec.Shape.size inS = Spec.Shape.size outS then
              let y := Tensor.reshapeSpec (α := α) (s₁ := inS) (s₂ := outS) pT h
              pure (Spec.PackedTensor.mk (α := α) outS y)
            else
              throw
                s!"IR eval: node {i}: reshape numel mismatch: {Spec.Shape.size inS} vs {Spec.Shape.size outS}"
        | _ => throw s!"IR eval: node {i}: reshape expects 1 parent ({n.summary})"
    | .flatten s =>
        match n.parents with
        | [pId] =>
            let pV ← getParent pId
            let pT ← expectShape (α := α) (expected := s) pV
            let y := Tensor.flattenSpec (α := α) (s := s) pT
            pure (Spec.PackedTensor.mk (α := α) (.dim (Spec.Shape.size s) .scalar) y)
        | _ => throw s!"IR eval: node {i}: flatten expects 1 parent ({n.summary})"
    | .concat axis => do
        let parents ← n.parents.mapM getParent
        evalConcat (α := α) i n axis parents
      | .swap_first_two =>
          match n.parents with
          | [pId] =>
              match n.outShape with
              | Shape.dim nDim (Shape.dim m rest) =>
                  let p ←
                    expectShape (α := α) (expected := Shape.dim m (Shape.dim nDim rest))
                      (← getParent pId)
                  let y := Tensor.swapFirstTwoSpec (α := α) (m := m) (n := nDim) (s := rest) p
                  pure (Spec.PackedTensor.mk (α := α) (Shape.dim nDim (Shape.dim m rest)) y)
              | _ =>
                  throw s!"IR eval: node {i}: swap_first_two expects rank≥2 outShape ({n.summary})"
          | _ => throw s!"IR eval: node {i}: swap_first_two expects 1 parent ({n.summary})"
      | .transpose3dLastTwo =>
          match n.parents with
          | [pId] =>
              match n.outShape with
              | Shape.dim a (Shape.dim c (Shape.dim b Shape.scalar)) =>
                  let p ←
                    expectShape (α := α) (expected := Shape.dim a (Shape.dim b (Shape.dim c
                      Shape.scalar)))
                      (← getParent pId)
                  let y := Tensor.transpose3DLastTwoSpec (α := α) (a := a) (b := b) (c := c) p
                  pure (Spec.PackedTensor.mk (α := α) (Shape.dim a (Shape.dim c (Shape.dim b Shape.scalar))) y)
              | _ =>
                  throw <|
                    s!"IR eval: node {i}: transpose3d_last_two expects rank=3 with scalar " ++
                      s!"base outShape ({n.summary})"
          | _ => throw s!"IR eval: node {i}: transpose3d_last_two expects 1 parent ({n.summary})"
      | .mseLoss =>
          match n.parents with
          | [yId, tId] =>
              mseLossPackedTensor (α := α) i (← getParent yId) (← getParent tId)
          | _ => throw s!"IR eval: node {i}: mse_loss expects 2 parents ({n.summary})"
  normalizeNodeOutput (α := α) i n v

/--
Evaluate node `i` after checking the graph's id discipline and retrieving the corresponding node.

`denoteAll` checks the full graph structure before repeatedly calling this one-step evaluator.
-/
def evalAt
    {α : Type} [Context α] [DecidableEq Shape]
    (g : Graph) (payload : Payload α) (input : Spec.PackedTensor α) (vals : Array (Spec.PackedTensor α)) (i : Nat) :
    Except String (Spec.PackedTensor α) := do
  let n ← g.getNode i
  evalNode (α := α) payload input vals i n

/--
Evaluate nodes `i, i+1, ...` given already computed prefix values `vals`.

This is written as a structurally recursive function so it is easy to reason about in proofs
(evaluation is “a simple loop over node ids”).
-/
def denoteAllFrom
    {α : Type} [Context α] [DecidableEq Shape]
    (g : Graph) (payload : Payload α) (input : Spec.PackedTensor α) (i : Nat) (vals : Array (Spec.PackedTensor α)) :
    Except String (Array (Spec.PackedTensor α)) := do
  if h : i < g.nodes.size then
    let v ← evalAt (α := α) (g := g) (payload := payload) (input := input) (vals := vals) (i := i)
    denoteAllFrom (α := α) (g := g) (payload := payload) (input := input) (i := i + 1) (vals :=
      vals.push v)
  else
    pure vals
termination_by g.nodes.size - i
decreasing_by
  simpa using Nat.sub_succ_lt_self (a := g.nodes.size) (i := i) h

/--
Evaluate a graph to a table of node values.

This returns an array `vals` of length `g.size` where `vals[i]` is the value of node `i`.

We do a structural well-formedness check once up front (ids/arity/topology). For lowering-produced
graphs, the boolean `Graph.wellFormed` check is a fast path; if it fails we fall back to the
exception-producing `Graph.checkWellFormed` so callers get a readable error message.

The evaluator is total in the sense that it always returns either:
- `.ok vals` (all nodes evaluated successfully), or
- `.error msg` describing the first failure (malformed IR, missing payload, or a local shape error).
-/
def denoteAll
    {α : Type} [Context α] [DecidableEq Shape]
    (g : Graph) (payload : Payload α) (input : Spec.PackedTensor α) : Except String (Array (Spec.PackedTensor α)) := do
  -- Fast path: lowering-produced graphs typically satisfy the boolean `wellFormed` discipline.
  if g.wellFormed then
    pure ()
  else
    g.checkWellFormed
  denoteAllFrom (α := α) (g := g) (payload := payload) (input := input) (i := 0)
    (vals := #[])

/-! ## Scoped notation -/

/--
Scoped notation for evaluating a graph to all node values.

Use with:

```lean
open scoped IR
g⟦payload, input⟧
```
-/
scoped[IR] notation g "⟦" payload ", " input "⟧" =>
  _root_.NN.IR.Graph.denoteAll g payload input

/-- ASCII alternative to `g⟦payload, input⟧`. -/
scoped[IR] notation g "[[" payload ", " input "]]" =>
  _root_.NN.IR.Graph.denoteAll g payload input

/-- Evaluate the graph and return the value at `outputId`. -/
def denote
    {α : Type} [Context α] [DecidableEq Shape]
    (g : Graph) (payload : Payload α) (input : Spec.PackedTensor α) (outputId : Nat) : Except String (Spec.PackedTensor α) :=
      do
  let vals ← denoteAll (α := α) (g := g) (payload := payload) (input := input)
  match vals[outputId]? with
  | none => throw s!"IR eval: outputId out of bounds: {outputId}"
  | some v => pure v

end Graph

end NN.IR

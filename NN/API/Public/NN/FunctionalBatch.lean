/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.API.Public.NN.Core

@[expose] public section

namespace NN
namespace API
namespace nn
namespace pure

/-!
`nn.functional` mirrors `torch.nn.functional`: pure, stateless building blocks.

In TorchLean these are defined as derived ops over the small primitive `Ops` surface, so the same
code works on both the eager backend and the compiled backend.
-/
namespace functional

/-!
PyTorch references:
- `torch.nn.functional`: `https://pytorch.org/docs/stable/nn.functional.html`
-/

export TorchLean.F
  (square checkpoint
   detach stopGrad
   addB mulB
   embedding embeddingRowsNat embeddingBatchSeqNat mean
   dropoutSeeded)

-- Elementwise transcendentals + scalar-affine for scientific forward models
-- (fully qualified to disambiguate the `exp`/`log`/`scale` identifiers, which
-- also name primitives in scope).
export _root_.Runtime.Autograd.TorchLean.F
  (exp log scale shift affine)

end functional

/-!
## Batch Lifting

`batchDim0 n model` wraps a *single-example* model `σ → τ` into a batched model
`(dim n σ) → (dim n τ)` by running the underlying model once per batch element.

This is the correctness-first batch lift used to expose PyTorch-like `N×...` APIs even when a primitive
only exists for the unbatched shape.
-/

/--
Lift a single-example `LayerDef σ τ` to operate on a dimension-0 batch.

This is a correctness-first batch lift: it runs the underlying layer independently on each batch
element. Prefer a primitive batched layer when one exists.
-/
def batchLayerDim0 (n : Nat) {σ τ : Spec.Shape} (l : LayerDef σ τ) :
    LayerDef (.dim n σ) (.dim n τ) :=
  let inSize : Nat := Spec.Shape.size σ
  let outSize : Nat := Spec.Shape.size τ
  { kind := l.kind
    paramShapes := l.paramShapes
    initParams := l.initParams
    paramRequiresGrad := l.paramRequiresGrad
    updateBuffers := none
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := fun sh => TorchLean.RefTy (m := m) (α := α) sh)
          (ss := l.paramShapes ++ [.dim n σ])
          (β := m (TorchLean.RefTy (m := m) (α := α) (.dim n τ)))
          (fun args => do
            let (ps, xBatch) :=
              _root_.Runtime.Autograd.Torch.RefList.splitAppend1
                (Ref := fun sh => TorchLean.RefTy (m := m) (α := α) sh)
                (ss := l.paramShapes) (τ := .dim n σ) args
            let xMat ←
              TorchLean.reshape (m := m) (α := α)
                (s₁ := .dim n σ) (s₂ := .dim n (.dim inSize .scalar))
                xBatch (by simp [Spec.Shape.size, inSize])
            let zeros : Spec.Tensor α (.dim n (.dim outSize .scalar)) :=
              _root_.Spec.Tensor.dim (fun _ =>
                _root_.Spec.Tensor.dim (fun _ =>
                  _root_.Spec.Tensor.scalar (0 : α)))
            let out0 ← TorchLean.const (m := m) (α := α) (s := .dim n (.dim outSize .scalar)) zeros
            let outMat ← (List.finRange n).foldlM (init := out0) (fun acc i => do
              let xRow ← TorchLean.gatherRow (m := m) (α := α) (rows := n) (cols := inSize) xMat i
              let xSample ←
                TorchLean.reshape (m := m) (α := α)
                  (s₁ := .dim inSize .scalar) (s₂ := σ)
                  xRow (by simp [Spec.Shape.size, inSize])
              let ySample ←
                _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
                  (Ref := fun sh => TorchLean.RefTy (m := m) (α := α) sh)
                  (ss := l.paramShapes ++ [σ])
                  (β := m (TorchLean.RefTy (m := m) (α := α) τ))
                  (l.forward mode (α := α) (m := m))
                  (_root_.Runtime.Autograd.Torch.RefList.append ps (.cons xSample .nil))
              let yRow ←
                TorchLean.reshape (m := m) (α := α)
                  (s₁ := τ) (s₂ := .dim outSize .scalar)
                  ySample (by simp [Spec.Shape.size, outSize])
              TorchLean.scatterAddRow (m := m) (α := α) (rows := n) (cols := outSize) acc yRow i)
            TorchLean.reshape (m := m) (α := α)
              (s₁ := .dim n (.dim outSize .scalar)) (s₂ := .dim n τ)
              outMat (by simp [Spec.Shape.size, outSize]))
  }

/-- Lift a sequential model to act pointwise on a leading dim0 batch axis. -/
def batchDim0 (n : Nat) {σ τ : Spec.Shape} : Sequential σ τ → Sequential (.dim n σ) (.dim n τ)
  | .id s => .id (.dim n s)
  | .cons l rest => .cons (batchLayerDim0 n l) (batchDim0 n rest)

/-!
Note: some low-level TorchLean layers (notably conv/pool/norm) have Nat-side well-formedness
proof arguments (e.g. `kH ≠ 0`).

The public path is *record-based specs* that hide those proofs via typeclasses like `NeZero`,
so examples can stay PyTorch-like without relying on positional macros.
-/

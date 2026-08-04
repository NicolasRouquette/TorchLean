/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.API.Neural.Builders

@[expose] public section

namespace TorchLean
namespace nn
namespace Internal

/-!
`nn.functional` mirrors `torch.nn.functional`: pure, stateless building blocks.

In TorchLean these are derived ops over the small primitive `Ops` API, so the same code works on
both the eager backend and the compiled backend.
-/
namespace functional

/-!
PyTorch references:
- `torch.nn.functional`: `https://pytorch.org/docs/stable/nn.functional.html`
-/

export _root_.Runtime.Autograd.TorchLean.F
  (square checkpoint
   detach
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
## Leading-Dimension Mapping

`mapLeading leading model` applies a model independently across every index in an arbitrary leading
shape. A conventional batch is the special case `leading = .dim batch .scalar`; multiple leading
axes work without introducing another tensor or model type.

Correctness-first batch lift for exposing PyTorch-like `N×...` APIs even when a primitive only
exists for the unbatched shape.
-/

namespace Implementation

/--
Expose a runtime layer whose outer axis is a flat batch as a layer over an arbitrary leading
shape. The adapter changes only the view of the input and output; parameters, buffer updates, and
the underlying forward program are preserved.
-/
def adaptFlatBatch (leading : Spec.Shape) {σ τ : Spec.Shape}
    (l : LayerDef (.dim (Spec.Shape.size leading) σ) (.dim (Spec.Shape.size leading) τ)) :
    LayerDef (leading.concat σ) (leading.concat τ) :=
  { kind := l.kind
    paramShapes := l.paramShapes
    initParams := l.initParams
    runtimeInit := l.runtimeInit
    paramRequiresGrad := l.paramRequiresGrad
    updateBuffers := l.updateBuffers.map fun update mode {α} _ _ ps x =>
      update mode ps <| Spec.Tensor.reshapeSpec x (by
        simp [Spec.Shape.size_concat, Spec.Shape.size])
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := fun sh => _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) sh)
          (ss := l.paramShapes ++ [leading.concat σ])
          (β := m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) (leading.concat τ)))
          (fun args => do
            let (ps, x) :=
              _root_.Runtime.Autograd.Torch.RefList.splitLast
                (Ref := fun sh => _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) sh)
                (ss := l.paramShapes) (τ := leading.concat σ) args
            let xBatch ←
              _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
                (s₁ := leading.concat σ) (s₂ := .dim (Spec.Shape.size leading) σ)
                x (by simp [Spec.Shape.size_concat, Spec.Shape.size])
            let yBatch ←
              _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
                (Ref := fun sh => _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) sh)
                (ss := l.paramShapes ++ [.dim (Spec.Shape.size leading) σ])
                (β := m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
                  (.dim (Spec.Shape.size leading) τ)))
                (l.forward mode (α := α) (m := m))
                (_root_.Runtime.Autograd.Torch.RefList.append ps (.cons xBatch .nil))
            _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
              (s₁ := .dim (Spec.Shape.size leading) τ) (s₂ := leading.concat τ)
              yBatch (by simp [Spec.Shape.size_concat, Spec.Shape.size]))
  }

/--
Lift a single-example `LayerDef σ τ` to operate on a leading batch axis.

This is a correctness-first batch lift: it runs the underlying layer independently on each batch
element. Prefer a primitive batched layer when one exists.
-/
def mapLayerOverAxis (n : Nat) {σ τ : Spec.Shape} (l : LayerDef σ τ) :
    LayerDef (.dim n σ) (.dim n τ) :=
  { kind := l.kind
    paramShapes := l.paramShapes
    initParams := l.initParams
    runtimeInit := l.runtimeInit
    paramRequiresGrad := l.paramRequiresGrad
    updateBuffers := l.updateBuffers.map fun update mode {_α} _ _ ps x =>
      match x with
      | Spec.Tensor.dim rows =>
          (List.finRange n).foldlM (init := ps) fun state i => update mode state (rows i)
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := fun sh => _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) sh)
          (ss := l.paramShapes ++ [.dim n σ])
          (β := m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) (.dim n τ)))
          (fun args => do
            let (ps, xBatch) :=
              _root_.Runtime.Autograd.Torch.RefList.splitLast
                (Ref := fun sh => _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) sh)
                (ss := l.paramShapes) (τ := .dim n σ) args
            _root_.Runtime.Autograd.Torch.mapLeadingAxis (m := m) (α := α)
              (fun xSample => l.forwardRef (α := α) (m := m) mode ps xSample)
              xBatch)
  }

/-- Lift a sequential model to act pointwise on a leading batch axis. -/
def mapModelOverAxis (n : Nat) {σ τ : Spec.Shape} :
    Sequential σ τ → Sequential (.dim n σ) (.dim n τ)
  | .id s => .id (.dim n s)
  | .cons l rest => .cons (mapLayerOverAxis n l) (mapModelOverAxis n rest)

end Implementation

/-- Apply a model pointwise over an arbitrary collection of leading dimensions. -/
def mapLeading (leading : Spec.Shape) {σ τ : Spec.Shape} :
    Sequential σ τ → Sequential (leading.concat σ) (leading.concat τ)
  | model =>
      match leading with
      | .scalar => model
      | .dim n rest => Implementation.mapModelOverAxis n (mapLeading rest model)

/-!
Note: some low-level TorchLean layers (notably conv/pool/norm) have Nat-side well-formedness
proof arguments (e.g. `kH ≠ 0`).

The public path is *record-based specs* that hide those proofs via typeclasses like `NeZero`,
so examples can stay PyTorch-like without relying on positional macros.
-/

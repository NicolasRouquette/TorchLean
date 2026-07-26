/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public

/-!
# Kolmogorov-Arnold Network Helpers

KAN layers replace each scalar edge by a small trainable one-dimensional function. TorchLean keeps
that structure visible: an edge family first expands every scalar input into basis features, and the
KAN layer learns one coefficient per `(output, input, basis)` edge.

The first built-in family uses triangular piecewise-linear hats. Users can add another family by
constructing `KANEdgeFamily`: provide a basis dimension and a TorchLean model that maps
`Vec inDim` to `Vec (inDim * basisDim)`.

References:

- Z. Liu et al., "KAN: Kolmogorov-Arnold Networks", arXiv:2404.19756.
- C. de Boor, "A Practical Guide to Splines", Springer, 1978/2001.
-/

@[expose] public section

namespace NN
namespace API

open Spec Tensor

namespace nn
namespace models

/--
Backend-compatible KAN edge family.

An edge family turns each scalar input coordinate into `basisDim` features. A KAN layer then applies
a learned linear map to all expanded features. The basis is a TorchLean model fragment, not an
arbitrary Lean callback, so the resulting KAN can run in eager, compiled, CPU, and CUDA training
paths supported by the underlying operations.
-/
structure KANEdgeFamily where
  /-- Short label shown in model summaries and training metadata. -/
  name : String
  /-- Number of basis features produced per scalar input coordinate. -/
  basisDim : Nat
  /-- Basis expansion for an unbatched vector of length `inDim`. -/
  basis : (inDim : Nat) → nn.Sequential (.dim inDim .scalar)
    (.dim (inDim * basisDim) .scalar)

namespace KANEdgeFamily

/-- Spec.Shape of the edge-basis expansion for `inDim` scalar inputs. -/
abbrev basisShape (edge : KANEdgeFamily) (inDim : Nat) : Spec.Shape :=
  .dim (inDim * edge.basisDim) .scalar

end KANEdgeFamily

/--
Configuration for triangular piecewise-linear KAN edge bases.

The basis functions are hats centered at the integer knots $0,\ldots,\mathrm{gridSize}-1$. The input is
multiplied by `inputScale` before the hats are evaluated. For normalized data in $[0,1]$, setting
$\mathrm{inputScale}=\mathrm{gridSize}-1$ spreads the grid across the full interval.
-/
structure KANPiecewiseLinear where
  /-- Number of knots, hence the number of basis functions per scalar coordinate. -/
  gridSize : Nat
  /-- Scale applied before basis evaluation; use $\mathrm{gridSize}-1$ for normalized $[0,1]$
  inputs. -/
  inputScale : Nat := 1
deriving Repr

namespace KANPiecewiseLinear

/--
Expand `x : Vec inDim` to all triangular basis features.

The output is flattened row-major from a `(gridSize × inDim)` table:
$[\operatorname{basis}_0(x_0),\ldots,\operatorname{basis}_0(x_n),
\operatorname{basis}_1(x_0),\ldots]$.

Each basis value is
$\operatorname{ReLU}(1-|\mathrm{inputScale}\,x_i-k|)$, expressed directly in the ordinary
TorchLean op language rather than through an opaque spline evaluator.
-/
def basisLayer (cfg : KANPiecewiseLinear) (inDim : Nat) :
    nn.Sequential (.dim inDim .scalar) (.dim (inDim * cfg.gridSize) .scalar) :=
  nn.of
    { kind := s!"KANPiecewiseLinear(grid={cfg.gridSize},scale={cfg.inputScale})"
      paramShapes := []
      initParams := .nil
      paramRequiresGrad := []
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun x =>
            ((do
              let zeros : Spec.Tensor α (.dim cfg.gridSize (.dim inDim .scalar)) :=
                Spec.Tensor.dim (fun _ =>
                  Spec.Tensor.dim (fun _ =>
                    Spec.Tensor.scalar (0 : α)))
              let xBasis ← TorchLean.scale (m := m) (α := α) x ((cfg.inputScale : Nat) : α)
              let out0 ← TorchLean.const (m := m) (α := α) zeros
              let out ← (List.finRange cfg.gridSize).foldlM (init := out0) (fun acc k => do
                let centerT : Spec.Tensor α (.dim inDim .scalar) :=
                  Spec.Tensor.dim (fun _ => Spec.Tensor.scalar ((k.val : Nat) : α))
                let oneT : Spec.Tensor α (.dim inDim .scalar) :=
                  Spec.Tensor.dim (fun _ => Spec.Tensor.scalar (1 : α))
                let c ← TorchLean.const (m := m) (α := α) centerT
                let ones ← TorchLean.const (m := m) (α := α) oneT
                let shifted ← TorchLean.sub (m := m) (α := α) xBasis c
                let dist ← TorchLean.abs (m := m) (α := α) shifted
                let raw ← TorchLean.sub (m := m) (α := α) ones dist
                let basis ← TorchLean.relu (m := m) (α := α) raw
                TorchLean.scatterAddRow (m := m) (α := α)
                  (rows := cfg.gridSize) (cols := inDim) acc basis k)
              let flat ← TorchLean.reshape (m := m) (α := α)
                (s₁ := .dim cfg.gridSize (.dim inDim .scalar))
                (s₂ := .dim (cfg.gridSize * inDim) .scalar)
                out (by
                  simp [Spec.Shape.size, Nat.mul_comm])
              TorchLean.reshape (m := m) (α := α)
                (s₁ := .dim (cfg.gridSize * inDim) .scalar)
                (s₂ := .dim (inDim * cfg.gridSize) .scalar)
                flat (by
                  simp [Spec.Shape.size, Nat.mul_comm])
            ) : m (TorchLean.RefTy (m := m) (α := α)
              (.dim (inDim * cfg.gridSize) .scalar)))
    }

/-- Turn piecewise-linear triangular bases into a general KAN edge family. -/
def edgeFamily (cfg : KANPiecewiseLinear) : KANEdgeFamily :=
  { name := s!"piecewise-linear(grid={cfg.gridSize},scale={cfg.inputScale})"
    basisDim := cfg.gridSize
    basis := basisLayer cfg }

end KANPiecewiseLinear

/-- Configuration for a KAN over batched row vectors. -/
structure KANConfig where
  /-- Leading minibatch dimension. -/
  batch : Nat
  /-- Number of scalar input coordinates. -/
  inDim : Nat
  /-- Hidden KAN widths. Each entry creates one KAN layer followed by `tanh`. -/
  hidden : List Nat := []
  /-- Number of output coordinates/classes. -/
  outDim : Nat
  /-- Edge basis family. The default is a compact triangular piecewise-linear basis. -/
  edge : KANEdgeFamily := KANPiecewiseLinear.edgeFamily { gridSize := 8 }

/-- Input shape `(batch × inDim)` for a KAN config. -/
abbrev kanInShape (cfg : KANConfig) : Spec.Shape :=
  .dim cfg.batch (.dim cfg.inDim .scalar)

/-- Output shape `(batch × outDim)` for a KAN config. -/
abbrev kanOutShape (cfg : KANConfig) : Spec.Shape :=
  .dim cfg.batch (.dim cfg.outDim .scalar)

/--
One unbatched KAN layer.

The layer first applies the selected edge basis to every input coordinate, then learns coefficients
with an ordinary linear map from the expanded features to `outDim`.
-/
def kanLayer (inDim outDim : Nat) (edge : KANEdgeFamily) :
    nn.M (nn.Sequential (.dim inDim .scalar) (.dim outDim .scalar)) :=
  nn.Sequential![
    nn.lift (edge.basis inDim),
    nn.linear (inDim * edge.basisDim) outDim
  ]

/-- Recursive unbatched KAN stack. Hidden layers use `tanh`; the final layer is linear in bases. -/
def kanGo (edge : KANEdgeFamily) :
    (inDim : Nat) → (hidden : List Nat) → (outDim : Nat) →
      nn.M (nn.Sequential (.dim inDim .scalar) (.dim outDim .scalar))
  | inDim, [], outDim => kanLayer inDim outDim edge
  | inDim, h :: hs, outDim =>
      nn.Sequential![kanLayer inDim h edge, nn.tanh, kanGo edge h hs outDim]

/--
Build a batched KAN model.

Task semantics are deliberately not baked into the model name: use `Trainer.new` with
`task := .regression`, `.classification`, `.crossEntropy`, or `.custom ...` with the same `KAN`
constructor.
-/
def KAN (cfg : KANConfig) :
    nn.M (nn.Sequential (kanInShape cfg) (kanOutShape cfg)) :=
  do
    let sample ← kanGo cfg.edge cfg.inDim cfg.hidden cfg.outDim
    nn.mapLeading (.dim cfg.batch .scalar) sample

end models
end nn

end API
end NN

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded

/-!
# Kolmogorov-Arnold Networks

KAN layers replace each scalar edge by a small trainable one-dimensional function. TorchLean keeps
that structure visible: an edge family first expands every scalar input into basis features, and the
KAN layer learns one coefficient per `(output, input, basis)` edge.

The first built-in family uses triangular piecewise-linear hats. Users can add another family by
constructing `KanEdgeFamily`: provide a basis dimension and a TorchLean model that maps
`Vec inDim` to `Vec (inDim * basisDim)`.

References:

- Z. Liu et al., "KAN: Kolmogorov-Arnold Networks", arXiv:2404.19756.
- C. de Boor, "A Practical Guide to Splines", Springer, 1978/2001.
-/

@[expose] public section

namespace TorchLean


open Spec Tensor

namespace nn
namespace models

/--
Backend-compatible KAN edge family.

An edge family turns each scalar input coordinate into `basisDim` features. A KAN layer then applies
a learned linear map to all expanded features. The basis is a TorchLean model fragment, not an
arbitrary Lean callback, so the resulting KAN can run in eager, typed graph, CPU, and CUDA training
paths supported by the underlying operations.
-/
structure KanEdgeFamily where
  /-- Short label shown in model summaries and training metadata. -/
  name : String
  /-- Number of basis features produced per scalar input coordinate. -/
  basisDim : Nat
  /-- Basis expansion for an unbatched vector of length `inDim`. -/
  basis : (inDim : Nat) → nn.Sequential (.dim inDim .scalar)
    (.dim (inDim * basisDim) .scalar)

/--
Configuration for triangular piecewise-linear KAN edge bases.

The basis functions are hats centered at the integer knots $0,\ldots,\mathrm{gridSize}-1$. The input is
multiplied by `inputScale` before the hats are evaluated. For normalized data in $[0,1]$, setting
$\mathrm{inputScale}=\mathrm{gridSize}-1$ spreads the grid across the full interval.
-/
structure KanPiecewiseLinear where
  /-- Number of knots, hence the number of basis functions per scalar coordinate. -/
  gridSize : Nat
  /-- Scale applied before basis evaluation; use $\mathrm{gridSize}-1$ for normalized $[0,1]$
  inputs. -/
  inputScale : Nat := 1
deriving Repr

namespace KanPiecewiseLinear

/--
Expand `x : Vec inDim` to all triangular basis features.

The output is flattened row-major from a `(gridSize × inDim)` table:
$[\operatorname{basis}_0(x_0),\ldots,\operatorname{basis}_0(x_n),
\operatorname{basis}_1(x_0),\ldots]$.

Each basis value is
$\operatorname{ReLU}(1-|\mathrm{inputScale}\,x_i-k|)$, expressed directly in the ordinary
TorchLean op language rather than through an opaque spline evaluator.
-/
def basisLayer (cfg : KanPiecewiseLinear) (inDim : Nat) :
    nn.Sequential (.dim inDim .scalar) (.dim (inDim * cfg.gridSize) .scalar) :=
  nn.of
    { kind := s!"KANPiecewiseLinear(grid={cfg.gridSize},scale={cfg.inputScale})"
      stateShapes := []
      initState := .nil
      requiresGrad := []
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun x =>
            ((do
              let zeros : Spec.Tensor α (.dim cfg.gridSize (.dim inDim .scalar)) :=
                Spec.Tensor.dim (fun _ =>
                  Spec.Tensor.dim (fun _ =>
                    Spec.Tensor.scalar (0 : α)))
              let xBasis ← _root_.Runtime.Autograd.Torch.scale (m := m) (α := α) x
                ((cfg.inputScale : Nat) : α)
              let out0 ← _root_.Runtime.Autograd.Torch.const (m := m) (α := α) zeros
              let out ← (List.finRange cfg.gridSize).foldlM (init := out0) (fun acc k => do
                let centerT : Spec.Tensor α (.dim inDim .scalar) :=
                  Spec.Tensor.dim (fun _ => Spec.Tensor.scalar ((k.val : Nat) : α))
                let oneT : Spec.Tensor α (.dim inDim .scalar) :=
                  Spec.Tensor.dim (fun _ => Spec.Tensor.scalar (1 : α))
                let c ← _root_.Runtime.Autograd.Torch.const (m := m) (α := α) centerT
                let ones ← _root_.Runtime.Autograd.Torch.const (m := m) (α := α) oneT
                let shifted ← _root_.Runtime.Autograd.Torch.sub (m := m) (α := α) xBasis c
                let dist ← _root_.Runtime.Autograd.Torch.abs (m := m) (α := α) shifted
                let raw ← _root_.Runtime.Autograd.Torch.sub (m := m) (α := α) ones dist
                let basis ← _root_.Runtime.Autograd.Torch.relu (m := m) (α := α) raw
                _root_.Runtime.Autograd.Torch.scatterAddRow (m := m) (α := α)
                  (rows := cfg.gridSize) (cols := inDim) acc basis k)
              let flat ← _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
                (s₁ := .dim cfg.gridSize (.dim inDim .scalar))
                (s₂ := .dim (cfg.gridSize * inDim) .scalar)
                out (by
                  simp [Spec.Shape.size, Nat.mul_comm])
              _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
                (s₁ := .dim (cfg.gridSize * inDim) .scalar)
                (s₂ := .dim (inDim * cfg.gridSize) .scalar)
                flat (by
                  simp [Spec.Shape.size, Nat.mul_comm])
            ) : m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
              (.dim (inDim * cfg.gridSize) .scalar)))
    }

/-- Turn piecewise-linear triangular bases into a general KAN edge family. -/
def edgeFamily (cfg : KanPiecewiseLinear) : KanEdgeFamily :=
  { name := s!"piecewise-linear(grid={cfg.gridSize},scale={cfg.inputScale})"
    basisDim := cfg.gridSize
    basis := basisLayer cfg }

end KanPiecewiseLinear

/-- Architecture of a Kolmogorov-Arnold network over feature vectors. -/
structure KanConfig where
  /-- Number of scalar input coordinates. -/
  inDim : Nat
  /-- Hidden KAN widths. Each entry creates one KAN layer followed by `tanh`. -/
  hidden : List Nat := []
  /-- Number of output coordinates/classes. -/
  outDim : Nat
  /-- Edge basis family. The default is a compact triangular piecewise-linear basis. -/
  edge : KanEdgeFamily := KanPiecewiseLinear.edgeFamily { gridSize := 8 }

namespace KanConfig

/-- Input shape with arbitrary leading dimensions. -/
abbrev inputShape (cfg : KanConfig) (leading : Spec.Shape := .scalar) : Spec.Shape :=
  leading.concat (.dim cfg.inDim .scalar)

/-- Output shape with the same leading dimensions as the input. -/
abbrev outputShape (cfg : KanConfig) (leading : Spec.Shape := .scalar) : Spec.Shape :=
  leading.concat (.dim cfg.outDim .scalar)

end KanConfig

/--
One KAN layer over a feature vector.

The layer first applies the selected edge basis to every input coordinate, then learns coefficients
with an ordinary linear map from the expanded features to `outDim`.
-/
def kanLayer (inDim outDim : Nat) (edge : KanEdgeFamily) :
    nn.Builder (nn.Sequential (.dim inDim .scalar) (.dim outDim .scalar)) :=
  nn.Sequential![
    nn.lift (edge.basis inDim),
    nn.linear (inDim * edge.basisDim) outDim
  ]

namespace Internal

/-- Recursive KAN stack over one feature vector. Hidden layers use `tanh`. -/
def kanStack (edge : KanEdgeFamily) :
    (inDim : Nat) → (hidden : List Nat) → (outDim : Nat) →
      nn.Builder (nn.Sequential (.dim inDim .scalar) (.dim outDim .scalar))
  | inDim, [], outDim => kanLayer inDim outDim edge
  | inDim, h :: hs, outDim =>
      nn.Sequential![kanLayer inDim h edge, nn.tanh, kanStack edge h hs outDim]

end Internal

/--
Build a KAN over arbitrary leading dimensions.

Task semantics are deliberately not baked into the model name: use `Trainer.new` with
`task := .regression`, `.oneHotCrossEntropy axis`, or `.custom ...` with the same KAN
constructor.
-/
def kan (cfg : KanConfig) (leading : Spec.Shape := .scalar) :
    nn.Builder (nn.Sequential (cfg.inputShape leading) (cfg.outputShape leading)) :=
  do
    let sample ← Internal.kanStack cfg.edge cfg.inDim cfg.hidden cfg.outDim
    nn.mapLeading leading sample

end models
end nn

end TorchLean

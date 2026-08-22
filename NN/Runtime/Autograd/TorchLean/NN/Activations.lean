/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.NN.Recurrent

/-!
# TorchLean NN: Activation and Shape Layers
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor
open Proofs.Autograd.Algebra

namespace NN

/--
ReLU activation layer (no parameters).

PyTorch analogues: `torch.nn.ReLU` / `torch.nn.functional.relu`.
-/
def relu {s : Shape} : Layer s s :=
  { kind := "ReLU"
    stateShapes := []
    initState := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TorchLean.relu (m := m) (α := α) (s := s) x
  }

/--
SiLU (a.k.a. swish) activation layer (no parameters).

PyTorch analogues: `torch.nn.SiLU` / `torch.nn.functional.silu`.
-/
def silu {s : Shape} : Layer s s :=
  { kind := "SiLU"
    stateShapes := []
    initState := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => _root_.Runtime.Autograd.Torch.silu (m := m) (α := α) (s := s) x
  }

/--
GELU activation layer (no parameters).

PyTorch analogues: `torch.nn.GELU` / `torch.nn.functional.gelu`.
-/
def gelu {s : Shape} : Layer s s :=
  { kind := "GELU"
    stateShapes := []
    initState := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => _root_.Runtime.Autograd.Torch.gelu (m := m) (α := α) (s := s) x
  }

/--
Sigmoid activation layer (no parameters).

PyTorch analogy: `torch.sigmoid`.
-/
def sigmoid {s : Shape} : Layer s s :=
  { kind := "Sigmoid"
    stateShapes := []
    initState := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TorchLean.sigmoid (m := m) (α := α) (s := s) x
  }

/--
Hyperbolic tangent activation layer (no parameters).

PyTorch analogy: `torch.tanh`.
-/
def tanh {s : Shape} : Layer s s :=
  { kind := "Tanh"
    stateShapes := []
    initState := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TorchLean.tanh (m := m) (α := α) (s := s) x
  }

/-- Shape-preserving softmax layer along `axis`. -/
def softmax {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s] : Layer s s :=
  { kind := "Softmax"
    stateShapes := []
    initState := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => F.softmax (m := m) (α := α) (s := s) axis x
  }

/-- Shape-preserving stable log-softmax layer along `axis`. -/
def logSoftmax {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s] : Layer s s :=
  { kind := "LogSoftmax"
    stateShapes := []
    initState := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => F.logSoftmax (m := m) (α := α) (s := s) axis x
  }

/--
Pointwise square `x ↦ x^2` (no parameters).

PyTorch analogy: `torch.square(x)` / `x.square()`.
-/
def square {s : Shape} : Layer s s :=
  { kind := "Square"
    stateShapes := []
    initState := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TorchLean.F.square (m := m) (α := α) (s := s) x
  }

/--
Sum-reduce all elements of the input to a scalar (no parameters).

PyTorch analogy: `x.sum()`.
-/
def sum {s : Shape} : Layer s Shape.scalar :=
  { kind := "Sum"
    stateShapes := []
    initState := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TorchLean.sum (m := m) (α := α) (s := s) x
  }

/--
Flatten any tensor to a 1D vector of length `Spec.Shape.size s` (no parameters).

PyTorch analogy: `torch.flatten(x)` or `x.reshape(-1)`.
-/
def flatten {s : Shape} : Layer s (.dim (Spec.Shape.size s) .scalar) :=
  { kind := "Flatten"
    stateShapes := []
    initState := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TorchLean.flatten (m := m) (α := α) (s := s) x
  }

/--
Dropout layer controlled by `Mode`.

- In `Mode.train`, randomly zeroes entries with probability `p`.
- In `Mode.eval`, it is the identity.

We store `p` as a scalar parameter tensor (with `requiresGrad := false`) so it can be threaded
through the unified parameter list without being optimized.

PyTorch analogy: `torch.nn.Dropout(p)` / `torch.nn.functional.dropout(x, p, training=...)`.
-/
def dropout {s : Shape} (p : Float) (seed : Nat := 0) : Layer s s :=
  let pShape : Shape := Shape.scalar
  let p0 : Tensor Float pShape := Tensor.scalar p
  { kind := s!"Dropout(p={p})"
    stateShapes := [pShape]
    initState := .cons p0 .nil
    runtimeInit := some (.cons (.flat (FloatArray.mk #[p])) .nil)
    requiresGrad := [false]
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        fun pRef x =>
          _root_.Runtime.Autograd.TorchLean.F.dropoutRefSeeded (m := m) (α := α) (s := s) x pRef
            seed
            (training := mode == .train)
  }
end NN

end TorchLean
end Autograd
end Runtime

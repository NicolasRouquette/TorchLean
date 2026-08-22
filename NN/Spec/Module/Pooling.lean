/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Pooling
public import NN.Spec.Module.Core

/-!
# Pooling module wrappers

These wrappers expose pooling specs as `Spec.Module`s.

Conventions:

- Channel-first images use shape `(C, H, W)` at the spec level (`.dim C (.dim H (.dim W .scalar))`).
- `Spec.Module.maxPool2d` applies the spatial max-pool independently per channel.
- `Spec.Module.avgPool2d` is provided for a single-channel 2D tensor; multi-channel usage typically
  maps it per channel in the same way as max-pool.

If you want a PyTorch mapping: `nn.MaxPool2d` / `nn.AvgPool2d` on a single `(C,H,W)` image (no
  batch).
-/

@[expose] public section


namespace Spec.Module
open Tensor

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

-- MaxPool2d module specification wrapper
/-- MaxPool2d wrapper (channel-first, pool applied per channel). -/
def maxPool2d {kH kW stride inH inW inC : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  {hStride : stride ≠ 0}
  (m : MaxPool2dSpec kH kW stride h1 h2 hStride) :
  Spec.Module α
    (.dim inC (.dim inH (.dim inW .scalar)))
    (.dim inC
      (.dim (poolOutDim inH kH stride 0)
        (.dim (poolOutDim inW kW stride 0) .scalar))) :=
{ forward := fun x =>
    -- Apply pooling to each channel independently.
    Tensor.dim (fun c => maxPool2dSpec m (getAtSpec x c)),
  kind := "MaxPool2d",
  pythonExpr := s!"nn.MaxPool2d(kernel_size=({kH}, {kW}), stride={stride})" }

-- AvgPool2d module specification wrapper
/-- AvgPool2d wrapper (2D tensor). -/
def avgPool2d {kH kW stride inH inW : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  {hStride : stride ≠ 0}
  (m : AvgPool2dSpec kH kW stride h1 h2 hStride) :
  Spec.Module α
    (.dim inH (.dim inW .scalar))
    (.dim (poolOutDim inH kH stride 0)
      (.dim (poolOutDim inW kW stride 0) .scalar)) :=
{ forward := fun x => avgPool2dSpec (layer := m) x, kind := "AvgPool2d", pythonExpr := s!"nn.AvgPool2d(kernel_size=({kH}, {kW}), stride={stride})" }

end Spec.Module

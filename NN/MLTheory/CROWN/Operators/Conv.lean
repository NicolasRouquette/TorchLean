/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Runtime.Ops
public import NN.Spec.Layers.Conv

/-!
# Conv2d

Conv2d CROWN-IBP bounds in TorchLean.

We provide:
- Interval Bound Propagation (IBP) for Conv2d pre-activations
- The exact flattened affine map for Conv2d

Design notes:
- We flatten the 3D input and convolution output when constructing the exact affine map. This
  reuses `AffineVec` without assigning a special semantic meaning to channel or spatial axes.
- The conv linear operator is explicitly materialized as a matrix Wconv whose rows
  correspond to output positions and columns to input positions. The verifier stays deterministic
  for the tensor sizes targeted by the CROWN operator layer.
-/

@[expose] public section

namespace NN.MLTheory.CROWN

open _root_.Spec
open _root_.Spec.Tensor

variable {α : Type} [Context α]

/-- Flatten a `Box` to a 1D box by flattening both endpoints. -/
def flattenBox {s : Shape} (B : Box α s) : Box α (.dim (Spec.Shape.size s) .scalar) :=
  { lo := Tensor.flattenSpec B.lo, hi := Tensor.flattenSpec B.hi }

/--
Interval Bound Propagation (IBP) for Conv2d pre-activations `y = conv(x, K) + b`.

This computes per-output-position min/max bounds by taking min/max of each product term.
-/
def ibpConv2d
  {inC outC kH kW stride padding inH inW : ℕ}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (layer : Spec.Conv2dSpec inC outC kH kW stride padding α h1 h2 h3)
  (xB : Box α (.dim inC (.dim inH (.dim inW .scalar)))) :
  Box α (.dim outC (.dim (Spec.Shape.slidingWindowOutDim inH kH stride padding) (.dim (Spec.Shape.slidingWindowOutDim inW kW stride padding) .scalar))) :=
  -- Compute lo/hi per output position independently
  let loT := Tensor.dim (fun out_ch =>
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        -- Accumulate lower bound using min over each product term
        let total_lo : α :=
          (List.range inC).foldl (fun acc in_ch =>
            (List.range kH).foldl (fun acc di =>
              (List.range kW).foldl (fun acc dj =>
                -- Map kernel position to input coordinate with padding/stride
                let pi := i * stride + di
                let pj := j * stride + dj
                -- Convert padded coords to input coords if valid
                let valid_i := pi ≥ padding
                let valid_j := pj ≥ padding
                let ii := pi - padding
                let jj := pj - padding
                -- Only contribute if within input bounds
                if valid_i ∧ valid_j ∧ ii < inH ∧ jj < inW then
                  let xlo := getAtOrZero xB.lo [in_ch, ii, jj]
                  let xhi := getAtOrZero xB.hi [in_ch, ii, jj]
                  let a := getAtOrZero layer.kernel [out_ch, in_ch, di, dj]
                  let p1 := a * xlo
                  let p2 := a * xhi
                  let mn := if p1 > p2 then p2 else p1
                  acc + mn
                else acc
              ) acc
            ) acc
          ) 0
        addSpec (Tensor.scalar total_lo) (getAtSpec layer.bias out_ch)
      )
    )
  )
  let hiT := Tensor.dim (fun out_ch =>
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        -- Accumulate upper bound using max over each product term
        let total_hi : α :=
          (List.range inC).foldl (fun acc in_ch =>
            (List.range kH).foldl (fun acc di =>
              (List.range kW).foldl (fun acc dj =>
                let pi := i * stride + di
                let pj := j * stride + dj
                let valid_i := pi ≥ padding
                let valid_j := pj ≥ padding
                let ii := pi - padding
                let jj := pj - padding
                if valid_i ∧ valid_j ∧ ii < inH ∧ jj < inW then
                  let xlo := getAtOrZero xB.lo [in_ch, ii, jj]
                  let xhi := getAtOrZero xB.hi [in_ch, ii, jj]
                  let a := getAtOrZero layer.kernel [out_ch, in_ch, di, dj]
                  let p1 := a * xlo
                  let p2 := a * xhi
                  let mx := if p1 > p2 then p1 else p2
                  acc + mx
                else acc
              ) acc
            ) acc
          ) 0
        addSpec (Tensor.scalar total_hi) (getAtSpec layer.bias out_ch)
      )
    )
  )
  { lo := loT, hi := hiT }

/--
Build the explicit Conv2d linear operator matrix `Wconv` mapping flat input to flat output.

Shapes:
- Input: `inC × inH × inW` (flat size `inC*inH*inW`)
- Output (pre-activation, without bias): `outC × outH × outW` (flat size `outC*outH*outW`)

Entry `Wconv[r,c]` is the contribution of input coordinate `c` to output coordinate `r`.
-/
def conv2dLinearMatrix
  {inC outC kH kW stride padding inH inW : ℕ}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (layer : Spec.Conv2dSpec inC outC kH kW stride padding α h1 h2 h3) :
  let outH := Spec.Shape.slidingWindowOutDim inH kH stride padding
  let outW := Spec.Shape.slidingWindowOutDim inW kW stride padding
  let inShape  := Shape.dim inC (Shape.dim inH (Shape.dim inW Shape.scalar))
  let outShape := Shape.dim outC (Shape.dim outH (Shape.dim outW Shape.scalar))
  let nIn  := Spec.Shape.size inShape
  let nOut := Spec.Shape.size outShape
  Tensor α (.dim nOut (.dim nIn .scalar)) :=
  let outH := Spec.Shape.slidingWindowOutDim inH kH stride padding
  let outW := Spec.Shape.slidingWindowOutDim inW kW stride padding
  -- Helpers to map flat indices to 3D and vice versa
  let decodeIn := fun (c : Nat) =>
    let c0 := c / (inH * inW)
    let r0 := c % (inH * inW)
    let i0 := r0 / inW
    let j0 := r0 % inW
    (c0, i0, j0)
  Tensor.dim (fun r =>
    -- Decode r to (out_ch, i, j)
    let oc := r.val / (outH * outW)
    let rs := r.val % (outH * outW)
    let i := rs / outW
    let j := rs % outW
    Tensor.dim (fun c =>
      let (in_ch, ii, jj) := decodeIn c.val
      -- Sum all kernel positions that map this input (in_ch,ii,jj) to output (oc,i,j)
      let coeff : α :=
        (List.range kH).foldl (fun acc di =>
          (List.range kW).foldl (fun acc dj =>
            let pi := i * stride + di
            let pj := j * stride + dj
            let valid_i := pi ≥ padding
            let valid_j := pj ≥ padding
            let iii := pi - padding
            let jjj := pj - padding
            -- Check alignment and bounds
            if valid_i ∧ valid_j ∧ iii = ii ∧ jjj = jj then
              let a := getAtOrZero layer.kernel [oc, in_ch, di, dj]
              acc + a
            else acc
          ) acc
        ) 0
      Tensor.scalar coeff))

/-- Broadcast a per-channel bias vector across spatial positions, as a flattened output vector. -/
def conv2dBiasBroadcast
  {outC inH inW kH kW stride padding : ℕ}
  (bias : Tensor α (.dim outC .scalar)) :
  let outH := Spec.Shape.slidingWindowOutDim inH kH stride padding
  let outW := Spec.Shape.slidingWindowOutDim inW kW stride padding
  let outShape := Shape.dim outC (Shape.dim outH (Shape.dim outW Shape.scalar))
  Tensor α (.dim (Spec.Shape.size outShape) .scalar) :=
  let outH := Spec.Shape.slidingWindowOutDim inH kH stride padding
  let outW := Spec.Shape.slidingWindowOutDim inW kW stride padding
  Tensor.dim (fun r =>
    let oc := r.val / (outH * outW)
    Tensor.scalar (getAtOrZero bias [oc]))

end NN.MLTheory.CROWN

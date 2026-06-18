/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common
public import NN.Spec.Core.Tensor.Optimizers
meta import NN.Examples.Factorization.Common
meta import NN.Spec.Core.Tensor.Optimizers

/-!
# KernelFlows optimizer steps (S5 examples)

S5 ports the two KernelFlows optimizers of
[`optimizers.jl`](../../../../../KernelFlows.jl/src/optimizers.jl) — `AMSGrad` (with a **scalar** second
moment) and `SGD` (with an optional fixed-step normalization) — and certifies the update arithmetic
(`FactorizationsOptimizers`: the AMSGrad running-max `v̂` is monotone, the fixed-step displacement is
bounded by `ϵ`).

These `#eval`s replay both optimizers over a shared 3-step gradient trajectory and check the resulting
state against **golden values** produced by running the verbatim `iterate!` rules in Julia:

```
AMSGrad (ϵ=0.1, β1=0.9, β2=0.999, δ=1e-8):
  step3 x    = [0.8611431016432155, -2.335025368974013,  0.7807168342892435]
  step3 m    = [-0.0074, 0.0172, 0.0037]   step3 v=vhat = 0.00025216764
SGD fixed=true  (ϵ=0.1): step3 x = [1.0331017445007942, -2.0506334992662176, 0.45206638831810586]
SGD fixed=false (ϵ=0.1): step1 x = [0.99, -2.02, 0.53]   (= x - ϵg)
```

* **Positive** — both trajectories reproduce the Julia golden to machine precision (the steps are pure
  IEEE arithmetic); `v̂` is non-decreasing; the fixed-step length sits at `≈ ϵ`; a zero gradient leaves
  the parameters fixed.
* **Negative** — fixed-step normalization genuinely changes the trajectory (it is not a no-op vs plain
  SGD), and a nonzero gradient genuinely moves the parameters (the fixed-point property is not vacuous).
-/

@[expose] public section

namespace NN.Examples.Factorization.Optimizers

open NN.Examples.Factorization

/-- Length-3 `Float` vector from a list (missing entries `0`). -/
def mkV (xs : List Float) : Fin 3 → Float := fun i => xs.getD i.val 0.0

/-- Read a length-3 function vector as a list (for display). -/
def vlist (v : Fin 3 → Float) : List Float := (List.finRange 3).map v

/-- Max absolute deviation of a length-3 vector from a golden list. -/
def maxVecErr (v : Fin 3 → Float) (gold : List Float) : Float :=
  (List.finRange 3).foldl (fun acc i => max acc (Float.abs (v i - gold.getD i.val 0.0))) 0.0

/-- Initial parameters and the shared gradient trajectory. -/
def x0 : Fin 3 → Float := mkV [1.0, -2.0, 0.5]
def g1 : Fin 3 → Float := mkV [0.1, 0.2, -0.3]
def g2 : Fin 3 → Float := mkV [0.05, -0.1, 0.2]
def g3 : Fin 3 → Float := mkV [-0.2, 0.1, 0.1]

/-! ### AMSGrad (scalar second moment) -/

def A0 : Spec.AMSGradState Float 3 := Spec.amsGradInit x0 0.1 0.9 0.999 1e-8
def A1 : Spec.AMSGradState Float 3 := Spec.amsGradStep A0 g1
def A2 : Spec.AMSGradState Float 3 := Spec.amsGradStep A1 g2
def A3 : Spec.AMSGradState Float 3 := Spec.amsGradStep A2 g3

#eval IO.println s!"AMSGrad step1 x = {vlist A1.x}"
#eval IO.println s!"AMSGrad step3 x = {vlist A3.x}  m = {vlist A3.m}  vhat = {A3.vhat}"

-- Bit-faithful AMSGrad trajectory vs the Julia `iterate!` golden.
#eval assertLt "AMSGrad step1 x vs Julia golden"
  (maxVecErr A1.x [0.9154846459556595, -2.1690307080886813, 0.7535460621330217]) 1e-9
#eval assertLt "AMSGrad step2 x vs Julia golden"
  (maxVecErr A2.x [0.8145430101618032, -2.2267116428280276, 0.8040168800299498]) 1e-9
#eval assertLt "AMSGrad step3 x vs Julia golden"
  (maxVecErr A3.x [0.8611431016432155, -2.335025368974013, 0.7807168342892435]) 1e-9
#eval assertLt "AMSGrad step3 m (first moment) vs Julia golden"
  (maxVecErr A3.m [-0.0074, 0.0172, 0.0037]) 1e-9
#eval assertApproxEq "AMSGrad step3 v=vhat (scalar 2nd moment) vs Julia golden"
  A3.vhat 0.00025216764000000023 1e-12

-- `amsGradStep_vhat_mono`: the running max never decreases (the AMSGrad fix over Adam).
#eval assertLt "AMSGrad vhat monotone non-decreasing (the AMSGrad correction)"
  (if A0.vhat ≤ A1.vhat && A1.vhat ≤ A2.vhat && A2.vhat ≤ A3.vhat then 0.0 else 1.0)

-- `amsGradStep_eq_of_grad_moment_zero`: zero gradient + zero momentum ⟹ parameters unchanged.
#eval assertLt "AMSGrad fixed point: zero grad & momentum leaves x unchanged"
  (maxVecErr (Spec.amsGradStep A0 (fun _ => 0.0)).x (vlist x0)) 1e-15

/-! ### SGD — fixed-step (normalized) and plain -/

def Sf0 : Spec.SGDState Float 3 := Spec.sgdInit x0 0.1 true 1e-9
def Sf1 : Spec.SGDState Float 3 := Spec.sgdStep Sf0 g1
def Sf2 : Spec.SGDState Float 3 := Spec.sgdStep Sf1 g2
def Sf3 : Spec.SGDState Float 3 := Spec.sgdStep Sf2 g3

/-- The fixed-step displacement length `‖Δx‖` (the trust-region radius). -/
def stepNorm (S : Spec.SGDState Float 3) (g : Fin 3 → Float) : Float :=
  Float.sqrt (Spec.dotFn (Spec.sgdDisplacement S g) (Spec.sgdDisplacement S g))

#eval IO.println s!"SGD fixed step3 x = {vlist Sf3.x}  (step1 length = {stepNorm Sf0 g1})"

#eval assertLt "SGD fixed step1 x vs Julia golden"
  (maxVecErr Sf1.x [0.973273875904208, -2.053452248191584, 0.580178372287376]) 1e-9
#eval assertLt "SGD fixed step3 x vs Julia golden"
  (maxVecErr Sf3.x [1.0331017445007942, -2.0506334992662176, 0.45206638831810586]) 1e-9

-- `sgdStep_fixed_displacement_normSq_le`: the fixed step length is bounded by ϵ (trust region)…
#eval assertLt "SGD fixed step length ≤ ϵ (trust region bound)"
  (if stepNorm Sf0 g1 ≤ 0.1 + 1e-12 then 0.0 else 1.0)
-- …and sits at ≈ ϵ for a non-tiny gradient (the normalizer ≈ ‖g‖).
#eval assertApproxEq "SGD fixed step length ≈ ϵ for non-tiny g" (stepNorm Sf0 g1) 0.1 1e-6

def Sn0 : Spec.SGDState Float 3 := Spec.sgdInit x0 0.1 false 1e-9
def Sn1 : Spec.SGDState Float 3 := Spec.sgdStep Sn0 g1

-- `sgdStep_nonfixed_apply`: plain SGD is exactly `x - ϵ g` (= [0.99, -2.02, 0.53]).
#eval assertLt "SGD nonfixed step1 = x - ϵg vs Julia golden"
  (maxVecErr Sn1.x [0.99, -2.02, 0.53]) 1e-12

/-! ### Negative controls -/

-- Fixed-step normalization genuinely changes the trajectory (not a no-op vs plain SGD).
#eval assertGe "SGD fixed ≠ plain SGD (normalization has teeth)"
  (maxVecErr Sf1.x (vlist Sn1.x)) 0.01

-- A nonzero gradient genuinely moves the parameters (the fixed-point property is not vacuous).
#eval assertGe "AMSGrad nonzero gradient genuinely moves x"
  (maxVecErr A1.x (vlist x0)) 0.01

end NN.Examples.Factorization.Optimizers

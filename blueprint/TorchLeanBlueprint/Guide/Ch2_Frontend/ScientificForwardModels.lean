import VersoManual

open Verso.Genre Manual

#doc (Manual) "Scientific Forward Models: Transcendentals and Affine Maps" =>
%%%
tag := "scientific-forward-models"
%%%

TorchLean's functional surface began life NN-flavoured: `square`, `mean`, `relu`,
`sigmoid`, `softmax`. Those cover deep-learning layers, but a large class of
*scientific* forward models — radiative transfer, dielectric mixing, kinetics —
are not built from activations. They are built from an `exp` or `log` of an
*affine* argument. This chapter documents the small set of functional ops added
for that use case, and the reason they matter: a forward model written once is
differentiated by the autograd engine, with no separately maintained gradient.

# The ops

All five are thin lifts of primitives that already carry a registered backward,
so reverse-mode `grad` / `jacrev` works through them unchanged. They live in the
functional namespace alongside `square` and `mean`:

- `nn.functional.exp` — elementwise `eˣ` (`torch.exp`).
- `nn.functional.log` — elementwise `ln x` (`torch.log`). For real-valued
  reasoning, assume positive inputs; it is the real natural log only on `x > 0`,
  and `Float` behavior on nonpositive values (`nan` / `-inf`) follows the backend.
- `nn.functional.scale x c` — multiply by a constant scalar, `c · x`.
- `nn.functional.shift x c` — add a constant scalar, `x + c`.
- `nn.functional.affine x c k` — the affine map `c · x + k`, the single most
  common building block of a linearised physical forward model.

Because they are ordinary functional ops, they compose inside a pure
`autograd.fn1.Fn` exactly like the NN ops do.

# Why this matters: the gradient is derived, not written

The motivating case is the SMAP-NISAR soil-moisture retrieval. Its forward model
relates radar backscatter to soil moisture through

```
σ⁰ = a · NDVI + exp(-2 · b · NDVI) · c · |R|² + d
```

Operationally this is fit per pixel by least squares with a *hand-coded* analytic
Jacobian — and a second, byte-duplicated copy for the JIT path. A sign or factor
error in that Jacobian does not crash anything; it silently degrades the fit, and
the two copies can drift apart. No validation statistic catches it.

With the ops above, the surface term is a one-line `Fn`, and its derivative comes
from autograd. The hand-coded Jacobian becomes *redundant*: instead of trusting a
transcription, the gradient is generated from the forward model and can be checked
against it.

# Worked check (positive and negative controls)

The example `NN.Examples.Functional.Transcendentals` differentiates three tiny
functions and compares the autograd gradient to the closed form. Run it with
`lake exe transcendentals_check`.

```
def expNeg2Fn : autograd.fn1.Fn Spec.Shape.scalar Spec.Shape.scalar :=
  fun x => do
    let u ← nn.functional.scale x (-Numbers.two)   -- -2 · x
    nn.functional.exp u                              -- e^{-2x}
```

The check asserts, at `x = 0.5`:

- *positive* — the autograd gradient equals the analytic `-2 · e^{-2x} = -0.735759`;
- *negative* — it does *not* equal the wrong-*sign* analytic `+2 · e^{-2x} = +0.735759`.

That negative control is the whole point in miniature: it is exactly the
defect — a sign error in a hand-written derivative — that deriving the gradient
by autograd eliminates. The positive controls for `exp` (`grad = eˣ`) and the
affine map (`grad (3x+1) = 3`) round out the suite, and the executable exits
non-zero on any regression.

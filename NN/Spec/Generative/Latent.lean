/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape.Reductions

/-!
# Latent-variable generative helpers

This module contains the shared spec-layer vocabulary for latent generative models:

- continuous latent variables, as used in variational autoencoders (VAEs);
- discrete codebook latents, as used in vector-quantized VAEs (VQ-VAEs).

The definitions are intentionally **model-agnostic**.  A VAE, VQ-VAE, latent diffusion model, or
normalizing-flow model can all reuse these primitives without committing to a particular backbone.

References:
- Kingma and Welling (2014), "Auto-Encoding Variational Bayes" (VAE).
- Rezende, Mohamed, and Wierstra (2014), "Stochastic Backpropagation and Approximate Inference".
- van den Oord, Vinyals, and Kavukcuoglu (2017), "Neural Discrete Representation Learning"
  (VQ-VAE).
-/

@[expose] public section

namespace Generative.Latent

open Spec TorchLean
open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/--
Diagonal-Gaussian reparameterization:

$$
z=\mu+\exp\!\left(\tfrac12\log\sigma^2\right)\odot\varepsilon.
$$

This is the spec-level form of the VAE reparameterization trick. The noise $\varepsilon$ is
explicit, so the
function stays pure and deterministic; runtime examples can supply deterministic or random noise.
-/
def reparameterizeDiag {latent : Shape}
    (mu logvar eps : Tensor α latent) : Tensor α latent :=
  let std := Tensor.expSpec (Tensor.scaleSpec logvar (1 / 2))
  mu + mulSpec std eps

/--
Mean KL term for a diagonal Gaussian posterior against a standard normal prior.

For each latent coordinate:

$$
D_{\mathrm{KL}}\!\left(\mathcal{N}(\mu,\sigma^2)\,\middle\|\,\mathcal{N}(0,1)\right)
=\tfrac12\left(\exp(\log\sigma^2)+\mu^2-1-\log\sigma^2\right).
$$

We return the mean across the latent shape, matching TorchLean's existing loss convention.
For real-like scalars, the small-log-variance branch avoids subtracting one from a nearly
unit exponential. A final nonnegative clamp removes negative roundoff; it is not an error bound.
Complex inputs retain the original algebraic expression when the real-like guard fails.
-/
def diagonalGaussianKlToStandard
    {latent : Shape} (mu logvar : Tensor α latent) : α :=
  let per := Tensor.map2Spec (fun mean logVariance =>
    let magnitude := MathFunctions.abs mean
    let varianceMagnitude := MathFunctions.abs logVariance
    -- Complex magnitudes need not equal either signed input, so do not clamp that expression.
    if (magnitude == mean || magnitude == -mean) &&
        (varianceMagnitude == logVariance || varianceMagnitude == -logVariance) then
      let varianceTerm := if varianceMagnitude < 1 then
          -- exp(v) - 1 - v = 2 sinh²(v/2) + sinh(v) - v.
          -- The quadratic term survives when exp(v) rounds to one near the prior.
          let halfSinh := MathFunctions.sinh (logVariance / 2)
          2 * halfSinh * halfSinh + (MathFunctions.sinh logVariance - logVariance)
        else MathFunctions.exp logVariance - 1 - logVariance
      let value := mean * mean + varianceTerm
      if value == value then max 0 value else value
    else
      MathFunctions.exp logVariance + mean * mean - 1 - logVariance) mu logvar
  (1 / 2) * Tensor.meanSpec per

/-- A finite codebook for vector-quantized latent models. -/
structure Codebook (α : Type) (numCodes : Nat) (latent : Shape) [TorchLean.Storage α] where
  /-- Embedding vector for each code index. -/
  embedding : Fin numCodes → Tensor α latent

/--
Quantize by an explicit code index.

Nearest-neighbor lookup is usually how VQ-VAE chooses this index during execution.  The spec keeps
the index explicit so proofs and verifiers can reason about a fixed code assignment without
depending on an argmin/tie-breaking policy.
-/
def quantizeAt {numCodes : Nat} {latent : Shape}
    (book : Codebook α numCodes latent) (idx : Fin numCodes) : Tensor α latent :=
  book.embedding idx

end Generative.Latent

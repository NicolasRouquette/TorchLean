/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Activation
public import NN.Spec.Core.Tensor.Numerics

/-!
# Gaussian Mixture Model (GMM) (spec model)

This file defines a basic GMM with `nComponents` multivariate Gaussians over `nFeatures`:

- mixing weights $\pi$,
- means $\mu$, and
- covariances $\Sigma$.

`gmmForwardSpec` computes **per-component** log-probabilities for a single input:

$$
\log \pi_k + \log \mathcal{N}(x \mid \mu_k, \Sigma_k).
$$

PyTorch analogies:

- `torch.distributions.MultivariateNormal` for $\mathcal{N}(x \mid \mu, \Sigma)$,
- `torch.distributions.MixtureSameFamily` for mixture distributions,
- `torch.softmax` for turning per-component log-probabilities into responsibilities.

Invalid mixture weights and singular or non-positive-determinant covariance matrices are reported
as `none`. Determinants and inverses are defined via `NN.Spec.Core.Tensor.Numerics`; those
definitions are intended for small feature dimensions and proof/reference usage, not
high-performance clustering on large matrices.

References (background, not required to read the code):

- Dempster, Laird, Rubin (1977), "Maximum Likelihood from Incomplete Data via the EM Algorithm":
  https://www.jstor.org/stable/2984875
- Bishop (2006), "Pattern Recognition and Machine Learning", Chapter 9 (Mixture Models and EM):
  https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/
-/

public section


namespace Spec

open Tensor

variable {α : Type} [Context α]

/-! ## Parameters -/

/-- Parameters of a Gaussian mixture model (GMM). -/
structure GMMSpec (α : Type) (nComponents nFeatures : Nat) where
  /-- Mixing weights $\pi_k$ (typically nonnegative and summing to $1$). -/
  weights : Tensor α [nComponents]
  /-- Component means $\mu_k$. -/
  means : Tensor α [nComponents, nFeatures]
  /-- Component covariance matrices $\Sigma_k$ (typically symmetric positive definite). -/
  covariances : Tensor α [nComponents, nFeatures, nFeatures]

/-- The leading `k × k` principal submatrix of a square matrix. -/
private def leadingPrincipalSubmatrix {n : Nat}
    (matrix : Tensor α [n, n]) (k : Nat) (hk : k ≤ n) :
    Tensor α [k, k] :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      Tensor.scalar (get2 matrix (i.castLE hk) (j.castLE hk))))

/-- Whether a matrix is symmetric under the scalar backend's equality operation. -/
def matrixSymmetricSpec {n : Nat}
    (matrix : Tensor α [n, n]) : Bool :=
  (List.finRange n).all (fun i =>
    (List.finRange n).all (fun j => get2 matrix i j == get2 matrix j i))

/--
Executable Sylvester-criterion check for a symmetric positive-definite covariance matrix.

The matrix must be symmetric and every nonempty leading principal minor must be positive. This is
the domain on which the Gaussian density, inverse, and logarithmic determinant used below have
their usual meaning.
-/
def covariancePositiveDefiniteSpec {n : Nat}
    (matrix : Tensor α [n, n]) : Bool :=
  matrixSymmetricSpec matrix &&
    (List.finRange n).all (fun i =>
      let k := i.val + 1
      have hk : k ≤ n := Nat.succ_le_iff.mpr i.isLt
      let leading := leadingPrincipalSubmatrix matrix k hk
      Context.gtBool (Tensor.item (determinantSpec leading)) 0)

/-- Positive, normalized mixture weights. -/
def mixtureWeightsValidSpec {n : Nat} (weights : Tensor α [n]) : Bool :=
  let positive :=
    match weights with
    | Tensor.dim f =>
        (List.finRange n).all (fun i =>
          match f i with
          | Tensor.scalar w => Context.gtBool w 0)
  positive && (sumSpec weights == 1)

/-- Whether all parameter-domain conditions required by the GMM density hold. -/
def gmmParametersValidSpec {nComponents nFeatures : Nat}
    (m : GMMSpec α nComponents nFeatures) : Bool :=
  nComponents != 0 && nFeatures != 0 &&
    mixtureWeightsValidSpec m.weights &&
    match m.covariances with
    | Tensor.dim covariances =>
        (List.finRange nComponents).all (fun k =>
          covariancePositiveDefiniteSpec (covariances k))

/-- Per-component log-probabilities for a single input.

Given $x \in \mathbb{R}^d$, each component contributes

$$
\log \pi_k-\frac12\left(
  (x-\mu_k)^\mathsf{T}\Sigma_k^{-1}(x-\mu_k)
  +\log\det\Sigma_k+d\log(2\pi)
\right).
$$

This is the natural "logit vector" for responsibilities.  If you want posterior probabilities
$P(z=k\mid x)$, apply `gmmExpectationSpec` (a last-axis softmax).

PyTorch analogy: the returned vector is like per-component `log_prob` values before the final
mixture `logsumexp`.
-/
def gmmForwardSpec {nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (input : Tensor α [nFeatures]) :
  Option (Tensor α [nComponents]) :=
  if gmmParametersValidSpec m then
    match m.weights, m.means, m.covariances with
    | Tensor.dim weights, Tensor.dim means, Tensor.dim covariances =>
      sequenceFin (fun k => do
        let weight := weights k
        let mean := means k
        let covariance := covariances k
        let diff := subSpec input mean
        let covInv ← inverseSpec? covariance
        let det := Tensor.item (determinantSpec covariance)
        let w := Tensor.item weight
        let quadraticForm := dotSpec diff (matVecMulSpec covInv diff)
        let log2pi := MathFunctions.log (Numbers.two * MathFunctions.pi)
        let normalization : α :=
          (nFeatures : α) / Numbers.two * log2pi +
            Numbers.half * MathFunctions.log det
        some (Tensor.scalar
          (MathFunctions.log w + Numbers.negHalf * quadraticForm - normalization)))
  else
    none

/-- E-step responsibilities for a single input.

Mathematically:

$$
\gamma_k
  = P(z=k\mid x)
  = \operatorname{softmax}_k\!\left(
      \log \pi_k+\log\mathcal{N}(x\mid\mu_k,\Sigma_k)
    \right).
$$

PyTorch analogy: `torch.softmax(component_log_probs, dim=-1)` where the logits are the
per-component log-probabilities.
-/
def gmmExpectationSpec {nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (input : Tensor α [nFeatures])
  (_h : nComponents ≠ 0) :
  Option (Tensor α [nComponents]) := do
  let componentLogProbs ← gmmForwardSpec m input
  pure (Activation.softmaxVecSpec (α := α) (n := nComponents) componentLogProbs)

/-- Batched forward pass: apply `gmmForwardSpec` to each sample in a batch. -/
def gmmBatchedForwardSpec {batch nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (input : Tensor α [batch, nFeatures]) :
  Option (Tensor α [batch, nComponents]) :=
  match input with
  | Tensor.dim batch_fn =>
    sequenceFin (fun i => gmmForwardSpec m (batch_fn i))

/-!
## Backward/VJP (for `gmmForwardSpec`)

`gmmForwardSpec` is **vector-valued**: it returns one log-probability per component.

The gradients below are the VJP for that vector function. In particular, responsibilities
$\gamma=\operatorname{softmax}(\text{component log-probabilities})$ do *not* appear in these
formulas by themselves.

Responsibilities show up when you differentiate a **scalar** objective that aggregates components,
like the mixture log-likelihood $\operatorname{logsumexp}(\text{component log-probabilities})$.
In that case, you compute the derivative with respect to the component log-probabilities first
(which will involve $\gamma$), then feed that vector into
`gmmBackwardSpec`.
-/

/-- Gradient/VJP with respect to weights $\pi$ for the output of `gmmForwardSpec`.

For $y_k=\log\pi_k+\cdots$, we have $\partial y_k/\partial\pi_k=1/\pi_k$.
-/
def gmmWeightsDerivSpec {nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (grad_output : Tensor α [nComponents])
  (_h : nComponents ≠ 0) :
  Option (Tensor α [nComponents]) :=
  if gmmParametersValidSpec m then
    sequenceFin (fun k =>
      match get m.weights k, get grad_output k with
      | Tensor.scalar π_k, Tensor.scalar g =>
        some (Tensor.scalar (g / π_k)))
  else
    none

/-- Gradient/VJP with respect to means $\mu$ for the output of `gmmForwardSpec`.

For a single component:

$$
\frac{\partial}{\partial\mu}\log\mathcal{N}(x\mid\mu,\Sigma)
=\frac12\left(\Sigma^{-1}+\Sigma^{-\mathsf{T}}\right)(x-\mu).
$$

For a valid symmetric covariance this reduces to the familiar $\Sigma^{-1}(x-\mu)$.
-/
def gmmMeansDerivSpec {nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (input : Tensor α [nFeatures])
  (grad_output : Tensor α [nComponents])
  (_h : nComponents ≠ 0) :
  Option (Tensor α [nComponents, nFeatures]) :=
  if gmmParametersValidSpec m then
    sequenceFin (fun k => do
      let mean_k := get m.means k
      let covariance_k := get m.covariances k
      let grad_k := get grad_output k
      let diff := subSpec input mean_k
      let covInv ← inverseSpec? covariance_k
      let covInvT := swapAdjacentAxes covInv 0
      let weightedDiff := scaleSpec
        (addSpec (matVecMulSpec covInv diff) (matVecMulSpec covInvT diff))
        Numbers.half
      match grad_k with
      | Tensor.scalar g =>
        pure (scaleSpec weightedDiff g))
  else
    none

/-- Gradient/VJP with respect to the input $x$ for the output of `gmmForwardSpec`.

For one component:

$$
\frac{\partial}{\partial x}\log\mathcal{N}(x\mid\mu,\Sigma)
=-\frac12\left(\Sigma^{-1}+\Sigma^{-\mathsf{T}}\right)(x-\mu).
$$

We sum the contributions from all components, weighted by the upstream gradient $g_k$.
-/
def gmmInputDerivSpec {nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (input : Tensor α [nFeatures])
  (grad_output : Tensor α [nComponents])
  (h : nComponents ≠ 0) :
  Option (Tensor α [nFeatures]) :=
  if gmmParametersValidSpec m then do
    have inst : Shape.HasNonemptyAxis 0 (Shape.dim nComponents (.dim nFeatures .scalar)) := by
      apply Shape.hasNonemptyAxisZeroOfNe h
    let perComponent ← sequenceFin (fun k => do
        let mean_k := get m.means k
        let covariance_k := get m.covariances k
        let gk := get grad_output k
        let diff := subSpec input mean_k
        let covInv ← inverseSpec? covariance_k
        let covInvT := swapAdjacentAxes covInv 0
        let v := scaleSpec
          (addSpec (matVecMulSpec covInv diff) (matVecMulSpec covInvT diff))
          Numbers.half
        match gk with
        | Tensor.scalar g =>
            pure (scaleSpec v (Numbers.negOne * g)))
    pure (reduceSum 0 perComponent inst.proof)
  else
    none

/-- Gradient/VJP with respect to covariances $\Sigma$ for the output of `gmmForwardSpec`.

For one component:

$$
\frac{\partial}{\partial\Sigma}\log\mathcal{N}(x\mid\mu,\Sigma)
=\frac12\left(
  \Sigma^{-\mathsf{T}}(x-\mu)(x-\mu)^\mathsf{T}\Sigma^{-\mathsf{T}}
  -\Sigma^{-\mathsf{T}}
\right).
$$
-/
def gmmCovariancesDerivSpec {nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (input : Tensor α [nFeatures])
  (grad_output : Tensor α [nComponents])
  (_h : nComponents ≠ 0) :
  Option (Tensor α [nComponents, nFeatures, nFeatures]) :=
  if gmmParametersValidSpec m then
    sequenceFin (fun k => do
      let mean_k := get m.means k
      let covariance_k := get m.covariances k
      let grad_k := get grad_output k
      let diff := subSpec input mean_k
      let outerProduct := outerProductSpec diff diff
      let covInv ← inverseSpec? covariance_k
      let covInvT := swapAdjacentAxes covInv 0
      let temp1 := matMulSpec covInvT outerProduct
      let temp2 := matMulSpec temp1 covInvT
      let gradSigma := subSpec temp2 covInvT
      match grad_k with
      | Tensor.scalar g =>
          pure (scaleSpec gradSigma (Numbers.half * g)))
  else
    none

/-- Backward/VJP for `gmmForwardSpec`.

Returns gradients with respect to `(weights, means, covariances, input)`.
-/
def gmmBackwardSpec {nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (input : Tensor α [nFeatures])
  (grad_output : Tensor α [nComponents])
  (h : nComponents ≠ 0) :
  Option (Tensor α [nComponents] ×
   Tensor α [nComponents, nFeatures] ×
   Tensor α [nComponents, nFeatures, nFeatures] ×
   Tensor α [nFeatures]) := do
  let dWeights ← gmmWeightsDerivSpec m grad_output h
  let dMeans ← gmmMeansDerivSpec m input grad_output h
  let dCovariances ← gmmCovariancesDerivSpec m input grad_output h
  let dInput ← gmmInputDerivSpec m input grad_output h
  pure (dWeights, dMeans, dCovariances, dInput)

/-- Uniform mixture weights (all components have probability $1/\mathtt{nComponents}$). -/
private def uniformWeights {nComponents : Nat} : Tensor α [nComponents] :=
  match nComponents with
  | 0 => Tensor.dim (fun k => nomatch k)
  | Nat.succ _ => Tensor.dim (fun _ => Tensor.scalar (1 / (nComponents : α)))

/-- Default initialization for a GMM.

This is kept simple and deterministic:

- uniform weights,
- zero means,
- identity covariances.
-/
def gmmInitSpec {nComponents nFeatures : Nat} :
  GMMSpec α nComponents nFeatures :=
  let weights : Tensor α [nComponents] := uniformWeights (α := α) (nComponents :=
    nComponents)
  let means : Tensor α [nComponents, nFeatures] := Tensor.dim (fun _ =>
    Tensor.dim (fun _ => Tensor.scalar (0 : α)))
  let covariances : Tensor α [nComponents, nFeatures, nFeatures] :=
    Tensor.dim (fun _ => identityTensorSpec nFeatures)
  {
    weights := weights,
    means := means,
    covariances := covariances
  }

/--
Numerically stable log-sum-exp reduction:
$\log\!\left(\sum_i \exp(\mathtt{log\_probs}[i])\right)$.

This is the standard
$m+\log\!\left(\sum_i\exp(x_i-m)\right)$ trick, where $m=\max_i x_i$.
-/
def logSumExpReduce {n : Nat} (log_probs : Tensor α [n]) (h : n ≠ 0) : α :=
  -- Step 1: Find maximum for numerical stability
  have inst : Shape.HasNonemptyAxis 0 (Shape.dim n .scalar) := by
    apply Shape.hasNonemptyAxisZeroOfNe h
  let max_log_prob := reduceMax 0 log_probs inst.proof
  have h_shape : shapeAfterSum (Shape.dim n Shape.scalar) 0 = Shape.scalar := by
    simp [shapeAfterSum]
  let max_log_prob' := item (tensorCast (Shape.scalar) h_shape.symm max_log_prob)

  -- Step 2: Compute sum of exp(log_prob - max_log_prob)
  let shifted_probs := Tensor.dim (fun k =>
    match get log_probs k with
    | Tensor.scalar log_prob =>
      Tensor.scalar (MathFunctions.exp (log_prob - max_log_prob'))
  )
  let sum_shifted := sumSpec shifted_probs

  -- Step 3: Compute log(sum_shifted) + max_log_prob
  if sum_shifted > 0 then
    MathFunctions.log sum_shifted + max_log_prob'
  else
    max_log_prob'  -- Fallback if sum is zero

/--
Mixture log-likelihood $\log p(x)$ computed via log-sum-exp over components.

Mathematically,
$$
\log p(x)=\log\!\left(
  \sum_k \exp\!\left(\log p(x\mid z_k)+\log\pi_k\right)
\right).
$$
-/
def gmmLogLikelihoodSpec {nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (input : Tensor α [nFeatures])
  (h : nComponents ≠ 0) :
  Option α := do
  let componentLogProbs ← gmmForwardSpec m input
  pure (logSumExpReduce componentLogProbs h)

/-!
## Classical training: EM for Gaussian mixtures

For a GMM, “training” is typically done with the Expectation–Maximization (EM) algorithm:

- **E-step**: compute responsibilities $r_{ik}=P(z=k\mid x_i)$ for each sample/component.
- **M-step**: update $\pi$, $\mu$, and $\Sigma$ from the weighted sufficient statistics.

This file already provides `gmmExpectationSpec` (responsibilities for one sample). The helpers
below lift that to a batched dataset and implement a deterministic EM update step.

Numerical notes:
- If a component gets (near) zero total responsibility ($N_k\approx 0$), we keep that component’s
  parameters unchanged (otherwise we’d divide by zero).
- We add a small diagonal “jitter,” $\mathtt{Numbers.epsilon}\,I$, to covariances to keep them
  well-behaved.
-/

/-- Batched responsibilities: apply `gmmExpectationSpec` to each sample. -/
def gmmResponsibilitiesBatchedSpec {nSamples nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (data : Tensor α [nSamples, nFeatures])
  (hK : nComponents ≠ 0) :
  Option (Tensor α [nSamples, nComponents]) :=
  match data with
  | Tensor.dim f =>
      sequenceFin (fun i => gmmExpectationSpec (α := α) (nComponents := nComponents) (nFeatures :=
        nFeatures) m (f i) hK)

/-- Scalar extraction helper for matrices: `t[i,j]` as an `α`. -/
private def getMatrix {n m : Nat} (t : Tensor α [n, m]) (i : Fin n) (j : Fin m) : α
  :=
  match get (get t i) j with
  | Tensor.scalar v => v

/-- Build a vector tensor from a function `Fin n -> α`. -/
private def vecFromFn {n : Nat} (f : Fin n → α) : Tensor α [n] :=
  Tensor.dim (fun i => Tensor.scalar (f i))

/-- Build a matrix tensor from a function `Fin n -> Fin m -> α`. -/
private def matFromFn {n m : Nat} (f : Fin n → Fin m → α) : Tensor α [n, m] :=
  Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (f i j)))

/-- One EM step for a batched dataset. -/
def gmmEmStepSpec {nSamples nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (data : Tensor α [nSamples, nFeatures])
  (hK : nComponents ≠ 0) :
  Option (GMMSpec α nComponents nFeatures) :=
  if _hN : nSamples = 0 then
    some m
  else do
    let resp ← gmmResponsibilitiesBatchedSpec (α := α) (nSamples := nSamples)
      (nComponents := nComponents) (nFeatures := nFeatures) m data hK

    -- N_k = Σ_i r_{ik}
    let Nk : Tensor α [nComponents] :=
      vecFromFn (n := nComponents) (fun k =>
        (List.finRange nSamples).foldl (fun acc i =>
          acc + getMatrix (n := nSamples) (m := nComponents) resp i k
        ) 0)

    -- π_k = N_k / N
    let weights : Tensor α [nComponents] :=
      let wRaw : Tensor α [nComponents] :=
        match Nk, m.weights with
        | Tensor.dim f, Tensor.dim wOld =>
          Tensor.dim (fun k =>
            match f k, wOld k with
            | Tensor.scalar nk, Tensor.scalar w =>
              if nk > 0 then Tensor.scalar (nk / (nSamples : α)) else Tensor.scalar w)
        | _, _ => uniformWeights (α := α) (nComponents := nComponents)
      let s := sumSpec wRaw
      if s > 0 then scaleSpec wRaw (1 / s) else uniformWeights (α := α) (nComponents :=
        nComponents)

    -- μ_k = (1/N_k) Σ_i r_{ik} x_i
    let means : Tensor α [nComponents, nFeatures] :=
      match Nk, m.means with
      | Tensor.dim NkF, Tensor.dim muOld =>
        Tensor.dim (fun k =>
          match NkF k with
          | Tensor.scalar nk =>
            if nk > 0 then
              Tensor.dim (fun f =>
                Tensor.scalar (
                  (List.finRange nSamples).foldl (fun acc i =>
                    let rik := getMatrix (n := nSamples) (m := nComponents) resp i k
                    let xi := get data i
                    acc + rik * Tensor.getScalar xi f
                  ) 0 / nk))
            else
              muOld k)
      | _, _ => m.means

    -- Σ_k = (1/N_k) Σ_i r_{ik} (x_i-μ_k)(x_i-μ_k)ᵀ + εI
    let covariances : Tensor α [nComponents, nFeatures, nFeatures] :=
      match Nk, means with
      | Tensor.dim NkF, Tensor.dim muF =>
        Tensor.dim (fun k =>
          match NkF k, muF k with
          | Tensor.scalar nk, Tensor.dim muVec =>
            if nk > 0 then
              let μ : Tensor α [nFeatures] := Tensor.dim muVec
              let base :=
                matFromFn (n := nFeatures) (m := nFeatures) (fun a b =>
                  (List.finRange nSamples).foldl (fun acc i =>
                    let rik := getMatrix (n := nSamples) (m := nComponents) resp i k
                    let xi := get data i
                    let da := Tensor.getScalar xi a - Tensor.getScalar μ a
                    let db := Tensor.getScalar xi b - Tensor.getScalar μ b
                    acc + rik * da * db
                  ) 0 / nk)
              let jitter := scaleSpec (identityTensorSpec nFeatures) Numbers.epsilon
              addSpec base jitter
            else
              get m.covariances k
          | _, _ => get m.covariances k)
      | _, _ => m.covariances

    pure { weights := weights, means := means, covariances := covariances }

/-- Total negative log-likelihood of a dataset under the current model. -/
def gmmNegLogLikelihoodBatchedSpec {nSamples nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (data : Tensor α [nSamples, nFeatures])
  (hK : nComponents ≠ 0) : Option α :=
  (List.finRange nSamples).foldlM (init := 0) (fun acc i => do
    let xi := get data i
    let ll ← gmmLogLikelihoodSpec (α := α) (nComponents := nComponents) (nFeatures := nFeatures)
      m xi hK
    pure (acc - ll))

/-- Run `epochs` EM steps (deterministic). -/
def gmmEmTrainSpec {nSamples nComponents nFeatures : Nat}
  (epochs : Nat)
  (m : GMMSpec α nComponents nFeatures)
  (data : Tensor α [nSamples, nFeatures])
  (hK : nComponents ≠ 0) :
  Option (GMMSpec α nComponents nFeatures) :=
  (List.finRange epochs).foldlM (init := m) (fun cur _ =>
    gmmEmStepSpec (α := α) cur data hK)

end Spec

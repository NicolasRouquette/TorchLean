/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Sequence

/-!
# Hidden Markov Model (HMM) (spec model)

This file defines an HMM with discrete observations:

- hidden states: `nStates`
- observations: `nObservations` (discrete symbols)

The model parameters are:
- initial distribution $\pi$,
- transition matrix $A$, and
- emission matrix $B$.

We represent observations as `List (Fin nObservations)` to keep the observation alphabet explicit
and avoid mixing “probabilities” with “indices” in the scalar type `α`.

## Notation and shapes

We use the conventional HMM notation:

- $\pi$: initial state distribution,
- $A$: transition matrix, with $A_{ij}=P(z_{t+1}=j\mid z_t=i)$, and
- $B$: emission matrix, with $B_{io}=P(x_t=o\mid z_t=i)$.

An observation sequence is $o_0,o_1,\ldots,o_{T-1}$, where each observation is represented by
`Fin nObservations`.

References:

- Rabiner (1989),
  "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition":
  https://ieeexplore.ieee.org/document/18626
- Baum and Petrie (1966),
  "Statistical Inference for Probabilistic Functions of Finite State Markov Chains":
  https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-37/issue-6/Statistical
    -Inference-for-Probabilistic-Functions-of-Finite-State-Markov-Chains/10.1214/aoms/1177699147.ful
    l

PyTorch analogy:

- emissions are categorical distributions (`torch.distributions.Categorical`),
- the forward algorithm corresponds to multiplying by $A$ and reweighting by the emission vector
  $B_{\mathord{:},o_t}$,
  then summing over previous states (often implemented in log-space in practice).

In practice, PyTorch users often reach for a dedicated HMM library (e.g. `hmmlearn`) or implement
HMMs in log-space with `logsumexp`; TorchLean keeps the spec in a simple, explicit form that is
good for reading and proofs.
-/

public section


namespace Spec

open Tensor

variable {α : Type} [Context α]

/-- A discrete-observation HMM.

We do not enforce probabilistic validity (nonnegativity or rows summing to $1$) at the type level;
that is a modeling assumption, similar to how PyTorch will happily store unconstrained tensors
until you feed them to a distribution or a loss.
-/
structure HMMSpec (α : Type) (nStates nObservations : Nat) where
  /-- Initial distribution $\pi$. -/
  initial : Tensor α (.dim nStates .scalar)
  /-- Transition matrix $A$. -/
  transition : Tensor α (.dim nStates (.dim nStates .scalar))
  /-- Emission matrix $B$. -/
  emission : Tensor α (.dim nStates (.dim nObservations .scalar))

/-- Observation sequence as a list of discrete symbols (indices into the observation alphabet). -/
abbrev ObservationSeq (nObservations : Nat) := List (Fin nObservations)

/-! ## Basic helpers -/

/-- Get the emission probability $B_{\mathtt{state},\mathtt{obs}}$ for a discrete symbol. -/
def getEmissionProbDiscrete
  {nStates nObservations : Nat}
  (m : HMMSpec α nStates nObservations)
  (state : Fin nStates)
  (obs : Fin nObservations) : α :=
  match get m.emission state with
  | Tensor.dim emit_vals =>
    match emit_vals obs with
    | Tensor.scalar prob => prob

/-!
## Baum–Welch (EM) training

The forward-pass APIs above are enough to *use* a fixed HMM, but a “fully implemented” baseline
should also include classical training. For discrete-observation HMMs, the standard training
procedure is the Baum–Welch algorithm (an EM procedure):

- **E-step**: run forward–backward to compute expected state occupancies $\gamma$ and expected
  transition counts $\xi$.
- **M-step**: normalize those expected counts to update $\pi$, $A$, and $B$.

This implementation uses *scaled* forward–backward to reduce numerical underflow:
each forward message $\alpha_t$ is normalized by a scalar $c_t$, and the backward messages divide
by those same scalars. The sequence likelihood is then $\prod_t c_t$, so the log-likelihood is
$\sum_t\log c_t$.

Concretely:

- forward recursion (unnormalized):
  $$
  \widetilde{\alpha}_{t+1}(j)
    = B_{j,o_{t+1}}\sum_i \alpha_t(i)A_{ij};
  $$
- scaling:
  $$
  c_t=\sum_j\widetilde{\alpha}_t(j),
  \qquad
  \alpha_t=\frac{\widetilde{\alpha}_t}{c_t},
  \qquad
  \sum_j\alpha_t(j)=1.
  $$

This is the same basic idea used in many practical HMM implementations (sometimes also expressed as
log-space forward–backward).

This is deterministic and written for clarity; it is not intended to be a high-performance HMM
trainer.
-/

/--
Uniform distribution vector of length `n`.

This is used to keep posterior messages total after an impossible observation has already been
recorded by a zero scaling factor.
-/
private def uniformVec {n : Nat} : Tensor α (.dim n .scalar) :=
  match n with
  | 0 => Tensor.dim (fun k => nomatch k)
  | Nat.succ _ => Tensor.dim (fun _ => Tensor.scalar (1 / (n : α)))

/-- Normalize a nonnegative vector $v$ to sum to $1$, returning
$(v/\sum_i v_i,\sum_i v_i)$.

If the sum is $0$, the normalized message is totalized to a uniform vector, while the returned
scale remains $0$. The zero scale is essential: it records that the observation prefix has
probability zero.
-/
def normalizeVec {n : Nat} (v : Tensor α (.dim n .scalar)) : (Tensor α (.dim n .scalar) × α) :=
  let s := sumSpec v
  if s > 0 then
    (scaleSpec v (1 / s), s)
  else
    (uniformVec (α := α) (n := n), 0)

/-- Sum logarithms of positive scale factors. A nonpositive factor represents zero likelihood. -/
private def logScales? (scales : List α) : Option α :=
  scales.foldl
    (fun acc c =>
      match acc with
      | none => none
      | some total => if c > 0 then some (total + MathFunctions.log c) else none)
    (some 0)

/-- Emission probabilities $B_{\mathord{:},\mathtt{obs}}$ as a vector over states. -/
def emissionVec {nStates nObservations : Nat}
  (m : HMMSpec α nStates nObservations) (obs : Fin nObservations) : Tensor α (.dim nStates .scalar)
    :=
  Tensor.dim (fun s => Tensor.scalar (getEmissionProbDiscrete m s obs))

/--
One forward step (unnormalized) of the scaled forward algorithm.

Given the previous normalized forward message `prev_alpha` and the next observation `obs`, compute
the next unnormalized message `alphaTilde`.
-/
private def forwardStep {nStates nObservations : Nat}
  (m : HMMSpec α nStates nObservations)
  (prev_alpha : Tensor α (.dim nStates .scalar))
  (obs : Fin nObservations) : Tensor α (.dim nStates .scalar) :=
  -- One forward update (without scaling): apply transitions, then reweight by emissions.
  let emission_probs := emissionVec (α := α) m obs
  Tensor.dim (fun s =>
    let trans_sum := Tensor.dim (fun s' =>
      match get prev_alpha s', get m.transition s' with
      | Tensor.scalar alpha_val, Tensor.dim trans_vals =>
        match trans_vals s with
        | Tensor.scalar trans_val => Tensor.scalar (alpha_val * trans_val))
    let trans_total := sumSpec trans_sum
    match get emission_probs s with
    | Tensor.scalar emit_val => Tensor.scalar (emit_val * trans_total))

/-- One timestep of a scaled HMM forward trace. -/
structure HMMForwardStep (α : Type) (nStates nObservations : Nat) where
  /-- Observation consumed at this timestep. -/
  observation : Fin nObservations
  /-- Normalized forward message $\alpha_t$. -/
  message : Tensor α (.dim nStates .scalar)
  /-- Normalization constant $c_t$. -/
  scale : α

private def hmmForwardScaledFrom
    {nStates nObservations : Nat}
    (m : HMMSpec α nStates nObservations) :
    (previous : Tensor α (.dim nStates .scalar)) →
    (observations : ObservationSeq nObservations) →
      Vector (HMMForwardStep α nStates nObservations) observations.length
  | _, [] => #v[]
  | previous, observation :: observations =>
      let raw := forwardStep (α := α) m previous observation
      let (message, scale) := normalizeVec (α := α) raw
      let remaining := hmmForwardScaledFrom m message observations
      ⟨#[{ observation, message, scale }] ++ remaining.toArray, by simp [Nat.add_comm]⟩

/-- Scaled forward pass, returning the observation, $\alpha_t$, and $c_t$ at each timestep.

- Each $\alpha_t$ is normalized to sum to $1$.
- Each $c_t$ is the normalization constant used at step $t$.

If you need the total likelihood, multiply the scales:
$p(o_{0:T-1})=\prod_t c_t$.
-/
def hmmForwardScaled
  {nStates nObservations : Nat}
  (m : HMMSpec α nStates nObservations)
  (observations : ObservationSeq nObservations) :
  Vector (HMMForwardStep α nStates nObservations) observations.length :=
  match observations with
  | [] => #v[]
  | o0 :: os =>
      let alpha0_raw := mulSpec m.initial (emissionVec (α := α) m o0)
      let (alpha0, c0) := normalizeVec (α := α) alpha0_raw
      let remaining := hmmForwardScaledFrom m alpha0 os
      ⟨#[{ observation := o0, message := alpha0, scale := c0 }] ++ remaining.toArray,
        by simp [Nat.add_comm]⟩

/-- Scaled backward pass, producing normalized backward messages $\beta_t$.

The standard backward recursion is:

$$
\beta_t(i)=\sum_j A_{ij}B_{j,o_{t+1}}\beta_{t+1}(j).
$$

In the scaled variant, we divide by the forward scale $c_{t+1}$ so that
$\alpha_t\odot\beta_t$ stays
well-conditioned.
-/
private def hmmBackwardScaled
  {nStates nObservations length : Nat}
  (m : HMMSpec α nStates nObservations)
  (steps : Vector (HMMForwardStep α nStates nObservations) length) :
  Vector (Tensor α (.dim nStates .scalar)) length :=
  let betaLast : Tensor α (.dim nStates .scalar) := fill 1 (.dim nStates .scalar)
  let initial : Option (Tensor α (.dim nStates .scalar) × HMMForwardStep α nStates nObservations) :=
    none
  (Sequence.mapAccumRight length initial fun index next =>
    let current := steps.get index
    match next with
    | none =>
        (some (betaLast, current), betaLast)
    | some (betaNext, nextStep) =>
        let emitNext := emissionVec (α := α) m nextStep.observation
        let betaRaw : Tensor α (.dim nStates .scalar) :=
          Tensor.dim (fun i =>
            let sumOverJ : Tensor α (.dim nStates .scalar) :=
              Tensor.dim (fun j =>
                match get (get m.transition i) j, get emitNext j, get betaNext j with
                | Tensor.scalar aij, Tensor.scalar bj, Tensor.scalar bnext =>
                    Tensor.scalar (aij * bj * bnext))
            Tensor.scalar (sumSpec sumOverJ))
        let beta :=
          if nextStep.scale > 0 then
            scaleSpec betaRaw (1 / nextStep.scale)
          else
            betaRaw
        (some (beta, current), beta)).2

 /-- Elementwise multiplication for state-probability vectors. -/
private def elementwiseMul {n : Nat} (a b : Tensor α (.dim n .scalar)) : Tensor α (.dim n .scalar)
  :=
  mulSpec a b

 /--
Compute the normalized state posterior $\gamma_t$ from forward/backward messages.

$$
\gamma_t(i)\propto\alpha_t(i)\beta_t(i).
$$
 -/
private def gammaAt {nStates : Nat}
  (alpha : Tensor α (.dim nStates .scalar)) (beta : Tensor α (.dim nStates .scalar)) : Tensor α
    (.dim nStates .scalar) :=
  -- γ_t(i) ∝ α_t(i) * β_t(i)
  let g := elementwiseMul (α := α) alpha beta
  (normalizeVec (α := α) g).1

 /--
Compute the normalized transition posterior $\xi_t$ for a single time step.

Unnormalized:
$$
\xi_t(i,j)
  =\alpha_t(i)A_{ij}B_{j,o_{t+1}}\beta_{t+1}(j).
$$
 -/
private def xiAt {nStates nObservations : Nat}
  (m : HMMSpec α nStates nObservations)
  (alpha_t : Tensor α (.dim nStates .scalar))
  (beta_next : Tensor α (.dim nStates .scalar))
  (obs_next : Fin nObservations) : Tensor α (.dim nStates (.dim nStates .scalar)) :=
  let emitNext := emissionVec (α := α) m obs_next
  -- Unnormalized ξ(i,j) = α_t(i) * A(i,j) * B(j, obs_{t+1}) * β_{t+1}(j)
  let xiRaw :=
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        match get alpha_t i, get (get m.transition i) j, get emitNext j, get beta_next j with
        | Tensor.scalar ai, Tensor.scalar aij, Tensor.scalar bj, Tensor.scalar bnext =>
            Tensor.scalar (ai * aij * bj * bnext)))
  -- Normalize so each ξ_t sums to 1 (helps control roundoff).
  let s := sumSpec xiRaw
  if s > 0 then scaleSpec xiRaw (1 / s) else xiRaw

 /-- Sum $\xi_t(i,j)$ over a list of $\xi$ matrices (expected transition count). -/
private def sumXi
  {nStates : Nat}
  (xis : List (Tensor α (.dim nStates (.dim nStates .scalar))))
  (i : Fin nStates) (j : Fin nStates) : α :=
  xis.foldl (fun acc xi =>
    match get (get xi i) j with
    | Tensor.scalar v => acc + v) 0

 /--
Sum $\gamma_t(\mathtt{state})$ over timesteps where the observation equals a given symbol.

This yields the expected emission count for $(\mathtt{state},\mathtt{symbol})$.
 -/
private def sumGammaWhereObs
  {nStates nObservations : Nat} [DecidableEq (Fin nObservations)]
  (observations : ObservationSeq nObservations)
  (gammas : Vector (Tensor α (.dim nStates .scalar)) observations.length)
  (state : Fin nStates) (sym : Fin nObservations) : α :=
  (List.finRange observations.length).foldl (fun acc t =>
    if observations.get t = sym then
      match get (gammas.get t) state with
      | Tensor.scalar v => acc + v
    else
      acc) 0

 /--
Normalize each row of a nonnegative matrix to sum to `1`.

Rows with sum `0` fall back to a uniform row (keeps the EM update total).
 -/
private def normalizeRows {nRows nCols : Nat} (m : Tensor α (.dim nRows (.dim nCols .scalar))) :
    Tensor α (.dim nRows (.dim nCols .scalar)) :=
  Tensor.dim (fun i =>
    let row := get m i
    let s := sumSpec row
    if s > 0 then scaleSpec row (1 / s) else uniformVec (α := α) (n := nCols))

 /--
Compute expected sufficient statistics for one observation sequence.

Returns `(initCounts, transCounts, emitCounts, loglik)` suitable for a Baum-Welch M-step.
 -/
private def expectedCounts
  {nStates nObservations : Nat} [DecidableEq (Fin nObservations)]
  (m : HMMSpec α nStates nObservations)
  (observations : ObservationSeq nObservations) :
  (Tensor α (.dim nStates .scalar) ×
   Tensor α (.dim nStates (.dim nStates .scalar)) ×
   Tensor α (.dim nStates (.dim nObservations .scalar)) ×
   Option α) :=
  -- Returns:
  -- - initial expected occupancies (for π),
  -- - expected transition counts (for A),
  -- - expected emission counts (for B),
  -- - scaled log-likelihood Σ log c_t.
  match observations with
  | [] =>
      (fill (0 : α) (.dim nStates .scalar),
       fill (0 : α) (.dim nStates (.dim nStates .scalar)),
       fill (0 : α) (.dim nStates (.dim nObservations .scalar)),
       let s := sumSpec m.initial
       if s > 0 then some (MathFunctions.log s) else none)
  | observation :: observations =>
      let sequence := observation :: observations
      let steps := hmmForwardScaled (α := α) m sequence
      let betas := hmmBackwardScaled (α := α) m steps
      let gammas : Vector (Tensor α (.dim nStates .scalar)) sequence.length :=
        Vector.ofFn fun t => gammaAt (α := α) (steps.get t).message (betas.get t)
      let xis :=
        (List.finRange observations.length).map fun t =>
          let current : Fin sequence.length :=
            ⟨t.val, by
              simp only [sequence, List.length_cons]
              exact Nat.lt_trans t.isLt (Nat.lt_succ_self observations.length)⟩
          let next : Fin sequence.length :=
            ⟨t.val.succ, by
              simp only [sequence, List.length_cons]
              exact Nat.succ_lt_succ t.isLt⟩
          let nextStep := steps.get next
          xiAt (α := α) m (steps.get current).message (betas.get next) nextStep.observation
      let first : Fin sequence.length := ⟨0, by simp [sequence]⟩
      let initCounts := gammas.get first
      let transCounts :=
        Tensor.dim (fun i =>
          Tensor.dim (fun j =>
            Tensor.scalar (sumXi (α := α) xis i j)))
      let emitCounts :=
        Tensor.dim (fun i =>
          Tensor.dim (fun o =>
            Tensor.scalar (sumGammaWhereObs (α := α) sequence gammas i o)))
      let loglik := logScales? (α := α) (steps.toList.map (·.scale))
      (initCounts, transCounts, emitCounts, loglik)

/-- One Baum–Welch (EM) step on a single sequence. -/
def baumWelchStepSpec
  {nStates nObservations : Nat} [DecidableEq (Fin nObservations)]
  (m : HMMSpec α nStates nObservations)
  (observations : ObservationSeq nObservations) :
  (HMMSpec α nStates nObservations × Option α) :=
  let (initCounts, transCounts, emitCounts, loglik) := expectedCounts (α := α) m observations
  let initial :=
    let (v, _) := normalizeVec (α := α) initCounts
    v
  let transition := normalizeRows (α := α) transCounts
  let emission := normalizeRows (α := α) emitCounts
  ({ initial, transition, emission }, loglik)

/-- One Baum–Welch epoch over a dataset of observation sequences (sums expected counts). -/
def baumWelchEpochSpec
  {nStates nObservations : Nat} [DecidableEq (Fin nObservations)]
  (m : HMMSpec α nStates nObservations)
  (dataset : List (ObservationSeq nObservations)) :
  (HMMSpec α nStates nObservations × Option α) :=
  let init0 := fill (0 : α) (.dim nStates .scalar)
  let trans0 := fill (0 : α) (.dim nStates (.dim nStates .scalar))
  let emit0 := fill (0 : α) (.dim nStates (.dim nObservations .scalar))
  let (initSum, transSum, emitSum, ll) :=
    dataset.foldl (fun (acc : Tensor α (.dim nStates .scalar) ×
                        Tensor α (.dim nStates (.dim nStates .scalar)) ×
                        Tensor α (.dim nStates (.dim nObservations .scalar)) × Option α) obs =>
      let (accInit, accTrans, accEmit, accLL) := acc
      let (iC, tC, eC, llik) := expectedCounts (α := α) m obs
      let nextLL :=
        match accLL, llik with
        | some a, some b => some (a + b)
        | _, _ => none
      (addSpec accInit iC, addSpec accTrans tC, addSpec accEmit eC, nextLL)
    ) (init0, trans0, emit0, some 0)
  let initial := (normalizeVec (α := α) initSum).1
  let transition := normalizeRows (α := α) transSum
  let emission := normalizeRows (α := α) emitSum
  ({ initial, transition, emission }, ll)

/-! ## Forward / likelihood -/

/-- Forward algorithm (scaled) returning the total sequence likelihood.

Implementation note:
we compute the likelihood from the per-timestep scaling factors produced by
`hmmForwardScaled`. This avoids the worst underflow behavior of multiplying many small
probabilities directly.
-/
def hmmForwardSpec
  {nStates nObservations : Nat}
  (m : HMMSpec α nStates nObservations)
  (observations : ObservationSeq nObservations) :
  α :=
  match observations with
  | [] => sumSpec m.initial
  | _ =>
      let steps := hmmForwardScaled (α := α) m observations
      steps.toList.foldl (fun acc step => acc * step.scale) 1

/-- Batched forward pass: compute likelihood for each observation sequence in a list. -/
def hmmBatchedForwardSpec {nStates nObservations : Nat}
  (m : HMMSpec α nStates nObservations)
  (observations : List (ObservationSeq nObservations)) :
  List α :=
  observations.map (hmmForwardSpec m)

/--
Initialize an HMM with uniform (uninformative) parameters.

This is a deterministic uniform initializer (useful for examples/tests); it is not intended as a
statistically meaningful random initialization.
 -/
def hmmInitSpec {nStates nObservations : Nat} :
  HMMSpec α nStates nObservations :=
  let initial : Tensor α (.dim nStates .scalar) := uniformVec (α := α) (n := nStates)
  let transition : Tensor α (.dim nStates (.dim nStates .scalar)) :=
    Tensor.dim (fun _ => uniformVec (α := α) (n := nStates))
  let emission : Tensor α (.dim nStates (.dim nObservations .scalar)) :=
    Tensor.dim (fun _ => uniformVec (α := α) (n := nObservations))
  { initial, transition, emission }

/-- Log-likelihood of an observation sequence.

We compute this from the same scaling factors used in the EM implementation:
$$
\log p(x_{0:T-1})=\sum_t\log c_t.
$$
-/
def hmmLogLikelihoodSpec {nStates nObservations : Nat}
  (m : HMMSpec α nStates nObservations)
  (observations : ObservationSeq nObservations) :
  Option α :=
  match observations with
  | [] =>
      let s := sumSpec m.initial
      if s > 0 then some (MathFunctions.log s) else none
  | _ =>
      let steps := hmmForwardScaled (α := α) m observations
      logScales? (α := α) (steps.toList.map (·.scale))

end Spec

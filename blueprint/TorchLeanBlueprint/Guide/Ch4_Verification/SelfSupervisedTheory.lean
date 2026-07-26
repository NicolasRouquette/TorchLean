import VersoManual

open Verso.Genre Manual

#doc (Manual) "Self-Supervised Objectives" =>
%%%
tag := "self-supervised-theory"
%%%

Self-supervised losses are often short enough to fit in one line of Python. Their meaning is not.
An MAE loss depends on which patches were hidden. A JEPA loss depends on which branch supplies the
target representation. An alignment objective can be minimized by mapping every view to the same
vector unless another term prevents collapse.

TorchLean’s present self-supervised theory isolates this bookkeeping. It is a finite algebraic
skeleton for masks, target views, predictive losses, and collapse guards. It does not formalize
a complete MAE or JEPA training run, and it does not prove that minimizing one of these objectives
learns useful representations.

# Masks And Explicit Index Lists

A Boolean mask over `n` positions is:

```
abbrev Mask (n : Nat) := Fin n → Bool
```

`selected m i` means `m i = true`. The module provides all-false, all-true, and complement
operations with the expected pointwise theorems.

The finite loss itself takes an explicit list of selected indices:

```
def maskedLoss
    (idxs : List (Fin n))
    (perPatchLoss : Fin n → Nat) : Nat :=
  (idxs.map perPatchLoss).sum
```

This design makes ordering and duplication visible. If `idxs = [0, 2]`, then

$$`L=\ell_0+\ell_2.`

Appending index chunks distributes over addition, and reversing the list preserves the sum:

```
theorem maskedLoss_append (xs ys : List (Fin n)) (ell : Fin n → Nat) :
  maskedLoss (xs ++ ys) ell =
    maskedLoss xs ell + maskedLoss ys ell

theorem maskedLoss_reverse (idxs : List (Fin n)) (ell : Fin n → Nat) :
  maskedLoss idxs.reverse ell = maskedLoss idxs ell
```

Run a concrete example:

```
import NN.MLTheory.SelfSupervised

open NN.MLTheory.SelfSupervised

def chosen : List (Fin 4) := [0, 2]

#eval maskedLoss chosen (fun i => i.val + 1)
#check maskedLoss_reverse
```

The output begins:

```
4
NN.MLTheory.SelfSupervised.maskedLoss_reverse ...
```

The two selected losses are `1` and `3`. Now change the list to `[0, 2, 2]`; the result becomes
`7`, because an explicit list may contain duplicate indices. `maskedLoss_reverse` proves
order-insensitivity, not duplicate elimination. A producer that intends a set of masked patches
must prove its exported index list has no duplicates or deliberately accept repeated weighting.

Another boundary is the scalar type. The finite skeleton uses `Nat`, so its “losses” are
already-computed nonnegative summaries. It is convenient for exact list algebra. It is not a
definition of mean-squared error over runtime floats.

# One Contract For Predictive Views

MAE predicts pixels or patches. JEPA predicts a latent target representation. The surrounding
index algebra is almost identical, so
[`PredictiveViewContract`](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/SelfSupervised/PredictiveView.lean)
keeps the target types separate:

```
structure PredictiveViewContract
    (n : Nat)
    (Context Target TargetRep Prediction : Type) where
  targetIdxs : List (Fin n)
  context : Context
  target : Fin n → Target
  targetEncoder : Fin n → Target → TargetRep
  predict : Context → Fin n → Prediction
  distance : TargetRep → Prediction → Nat
  geometryGuard : Nat := 0
```

For a selected index `i`, the predictive term is

$$`\ell\!\left(
  \operatorname{targetEncoder}_i(\operatorname{target}_i),
  \operatorname{predict}(\operatorname{context},i)
\right).`

Summing over `targetIdxs` gives `predictiveLoss`. The full finite objective is

$$`L_{\mathrm{SSL}}
=L_{\mathrm{predictive}}+L_{\mathrm{geometry}}.`

The four separate types prevent an accidental identification:

- `Target` is the raw target-view value;
- `TargetRep` is what the target encoder produces;
- `Prediction` is what the context-side predictor produces;
- `distance` is the operation that compares the last two.

This is where a stopped-gradient target branch would be represented semantically: the contract
contains a target value, but the finite objective does not itself run autograd or prove which
parameters receive gradients. Gradient stopping belongs to a runtime or differentiation theorem.

# MAE Is The Identity-Target Case

In
[`MAE.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/SelfSupervised/MAE.lean),
a patch batch is `Fin n → Patch`, and

```
def maeLoss
    (maskedIdxs : List (Fin n))
    (target : PatchBatch n Patch)
    (pred : Fin n → Pred)
    (patchLoss : Patch → Pred → Nat) : Nat := ...
```

The corresponding predictive-view contract uses the identity target encoder:

$$`\operatorname{targetEncoder}_i(x_i)=x_i.`

`mae_is_predictive_view_loss` proves that `predictiveLoss` for this contract is definitionally the
same finite sum as `maeLoss`. `mae_is_predictive_view_objective` adds that the full objective is
still `maeLoss` because the geometry guard is zero.

The identity reconstruction theorem

```
theorem exactReconstruction_identity (x : PatchBatch n Patch) :
  ExactReconstruction x (reconstruct (fun i patch => patch) x)
```

is a sanity theorem about the abstract reconstruction map. It is not a theorem that a trained MAE
decoder reconstructs its input.

The finite MAE loss also inherits append, reverse, and zero-per-patch theorems. These prove that the
objective is assembled as intended. They say nothing about image patchification, pixel
normalization, or a tensor decoder until those runtime components are connected to this contract.

# JEPA Changes The Target Space

[`JEPA.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/SelfSupervised/JEPA.lean)
starts at the target-representation boundary:

$$`L_{\mathrm{JEPA}}
=\sum_{i\in I}
\ell\!\left(z_i^{\mathrm{target}},
p(z^{\mathrm{context}},i)\right).`

`jepaAsPredictiveViewContract` uses the supplied target representation as `TargetRep`, while
`encodedTargetPredictiveViewContract` exposes a separate target encoder for the more general case.
Theorems `jepa_is_predictive_view_loss` and `jepa_is_predictive_view_objective` identify the JEPA
finite sum with the common predictive contract.

An extensionality theorem, `jepaLoss_target_ext`, says that replacing the target function by one
that agrees at every selected index preserves the loss. This is exactly as strong as it sounds:
values at unselected indices may differ because they are not read by the objective.

As a useful exercise, prove equality after changing an unselected target:

```
import NN.MLTheory.SelfSupervised

open NN.MLTheory.SelfSupervised

#check jepaLoss_target_ext
#check encodedTargetPredictiveViewContract_loss_eq_maskedLoss
```

Then try to use `jepaLoss_target_ext` when the targets differ at a selected index. The missing goal
is the pointwise equality at that index; Lean does not accept the informal claim that the change is
“small.”

# Why Alignment Alone Collapses

The predictive-view file also gives a concrete real-valued graph model. A representation is

$$`z:\operatorname{Fin}(n)\to\mathbb R^d,`

and an `SSLViewGraph n` stores positive pairs of views. The alignment energy is

$$`E_{\mathrm{align}}(z)
=\sum_{(i,k)\in E_+}\|z_i-z_k\|_2^2.`

Every term is nonnegative. But if the representation is collapsed,

$$`\exists c,\;\forall i,\;z_i=c,`

then every squared distance is zero. The theorem
`graphAlignmentEnergy_eq_zero_of_collapsed` proves this for any positive-edge graph. Thus
alignment by itself cannot rule out the constant representation.

TorchLean uses a finite pairwise coordinate-spread summary:

$$`\operatorname{spread}_j(z)
=\sum_i\sum_k(z_{ij}-z_{kj})^2.`

This unnormalized finite spread differs from the sample variance or standard deviation in the
VICReg paper. It captures the property needed here: collapsed representations have zero spread in
every coordinate.

For floor $`\gamma`, the guard is

$$`G_\gamma(z)
=\sum_{j=0}^{d-1}
\max\!\left(0,\gamma-\operatorname{spread}_j(z)\right).`

The complete graph objective is

$$`E_{\mathrm{SSL}}(z)=E_{\mathrm{align}}(z)+G_\gamma(z).`

`graphSSLObjective_collapsed_positive` proves:

```
theorem graphSSLObjective_collapsed_positive
    (graph : SSLViewGraph n)
    (rep : Fin n → EuclideanRep d)
    (hcollapsed : CollapsedRep rep)
    (hd : 0 < d)
    (hgamma : 0 < gamma) :
  0 < graphSSLObjective graph rep gamma
```

In the Infoview:

```
import NN.MLTheory.SelfSupervised

open NN.MLTheory.SelfSupervised

#check graphAlignmentEnergy_eq_zero_of_collapsed
#check graphSSLObjective_collapsed_positive
```

Remove `hgamma`, or set $`\gamma=0`, and the strict positivity claim is false: a collapsed
representation pays zero. Remove `hd`, and a zero-dimensional embedding has no guarded
coordinates. Both hypotheses are mathematically necessary for this theorem.

# The Discrete VICReg Skeleton

[`VICReg.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/SelfSupervised/VICReg.lean)
also contains a deliberately simpler `Nat` model. Its variance-floor penalty is natural-number
subtraction:

$$`\operatorname{varianceFloorPenalty}(\gamma,v)=\gamma-v,`

which is truncated at zero by the semantics of `Nat.sub`. It is therefore the discrete analogue of
$`\max(0,\gamma-v)`, without the square and without a statistical variance estimator.

For `d` collapsed coordinates:

$$`\operatorname{varianceTerm}
  \bigl(\gamma,[0,\ldots,0]\bigr)=d\gamma.`

`varianceTerm_collapsed_positive` proves positivity when there is at least one coordinate and
$`\gamma>0`. `vicregObjective` combines already-computed invariance, variance, and covariance
summaries with natural-number weights. The Barlow-style declarations similarly encode finite
diagonal and off-diagonal penalties, not the full floating cross-correlation computation.

These small exact models are useful for objective algebra, but their names should not be read as a
claim that TorchLean has formalized every estimator and normalization in production VICReg or
Barlow Twins.

# Proof And Runtime Boundary

The current theory establishes:

- finite index and list algebra;
- exact relationships among MAE, JEPA, and a common predictive-view contract;
- zero alignment energy for collapsed real representations;
- positivity of explicit anti-collapse guards under positive dimension and floor.

It does not establish:

- correctness of patch extraction or data augmentation;
- equivalence to a PyTorch MAE, JEPA, VICReg, or Barlow Twins training script;
- stopped-gradient behavior of a target encoder;
- quality, identifiability, or downstream usefulness of learned representations;
- floating-point agreement for the runtime objective.

The architecture is nevertheless useful. A runtime bridge can map tensor masks, encoder outputs,
and loss reductions into the finite contract. Once that bridge exists, the list and collapse
theorems do not need to be reproved.

The maintained `mae` application is one such runtime neighbor, but not yet that bridge: it performs
deterministic patch masking and Float reconstruction training. Its mask construction, tensor
reduction, and stopped-gradient choices must be related explicitly to `maskedLoss` before the exact
list identities on this page become theorems about that command.

The objective shapes are motivated by MAE (He et al.), VICReg (Bardes, Ponce, and LeCun), Barlow
Twins (Zbontar et al.), and joint-embedding predictive architectures. The Lean statements are
narrower than those papers: they formalize the finite algebra and explicit degenerate cases that
the present TorchLean definitions actually express.

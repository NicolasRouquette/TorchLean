import VersoManual

open Verso.Genre Manual

#doc (Manual) "Optimization Theory" =>
%%%
tag := "optimization-theory"
%%%

Training code executes updates. Optimization theory explains when those updates should make
progress.

A TorchLean training run may call SGD or Adam, but a convergence theorem cannot come from the name
of the optimizer alone. It needs an ideal update map, assumptions on the objective or gradient, and
a step size condition.

The runtime can execute SGD and Adam-style updates. The optimization theory material
names the ideal update and the assumptions under which a theorem is allowed to say "this step moves
closer." PyTorch faithfully runs an optimizer step; TorchLean can also state the mathematical
contract that would justify calling the step contractive or convergent.

# The Optimization Contract

The optimization contract has three layers:

- *Runtime update*: the concrete operation that writes new values into parameter tensors, such as
  one SGD, momentum, or Adam-style step.
- *Ideal update*: the mathematical map the runtime update is intended to approximate, for example
  $`x\mapsto x-\eta g(x)`.
- *Convergence theorem*: the conditional theorem saying that iterating the ideal map makes progress
  under assumptions such as strong monotonicity, Lipschitzness, and a safe step size.

Keeping these layers separate stops a common overclaim. A decreasing loss curve is evidence about a
run; it is not itself a proof that the update map is contractive.

# First Order Updates

The first object is a first order optimizer state: parameters, gradients, an optional buffer, and a
time counter. Its executable equation belongs in the definition of the update itself. TorchLean does
not add a theorem merely to restate each record field after reduction. The theory layer begins when
two distinct updates are related. For example,
`Optim.AdamW.update_weight_decay_zero_params_eq_adam_real` proves that the parameter update of
AdamW agrees with Adam when the decoupled weight-decay coefficient is zero.

For tensor optimizers with richer state, the
[optimizer laws API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Optimization/OptimizerLaws.lean) exposes the same pattern:

```
#check NN.MLTheory.Optimization.TensorOptimizer.sgd
#check NN.MLTheory.Optimization.TensorOptimizer.adamw
#check NN.MLTheory.Optimization.TensorOptimizer.runSteps_append
#check NN.MLTheory.Optimization.StepSpec
#check NN.MLTheory.Optimization.StepSpec.runSteps_eq_optimizer_runSteps
```

`TensorOptimizer.runSteps_append` applies to every packaged optimizer. `StepSpec` is reserved for
the stronger situation in which a separately stated mathematical recurrence is proved equal to an
executable update; its generic run theorem then lifts that agreement over a stream of gradients.
Concrete optimizer equations remain in `NN.Runtime.Optim.Optimizers`, while optimizer-specific
theory files establish algebraic invariants and comparisons that are not restatements of those
definitions.

GaLore-style projected SGD follows the same rule. The executable projection interface and the
identity-projection agreement theorem live together in
[NN.Runtime.Optim.Optimizers](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Optim/Optimizers.lean).
The theorem says that choosing the identity projector recovers ordinary SGD; it does not claim that
an arbitrary learned low-rank projector preserves the update.

For AdamW specifically, TorchLean follows the usual decoupled weight decay distinction from
Loshchilov and Hutter,
["Decoupled Weight Decay Regularization"](https://arxiv.org/abs/1711.05101). The Lean theorem is an
update equation, not an empirical claim that AdamW generalizes better on a task.

# Optimizer Extension Points: Muon And GaLore-Style Updates

Modern optimizer work often mixes a familiar base update with a specialized backend. TorchLean
models that explicitly. The runtime layer gives the executable update equation. The theory layer
states what has to be true of the backend output before the update can be cited in a proof.

Muon is represented as momentum followed by an orthogonalization backend. One step first updates the
momentum buffer

$$`m_{t+1}=\beta m_t+g_t,`

then asks a backend for the direction used in the parameter update. With the identity backend,
Muon's parameter update is exactly momentum SGD. With a certified matrix backend, the claim is about
the actual direction returned by that backend: exact column orthogonality

$$`Q^\top Q = I`

or an approximate Gram-residual bound

$$`\|Q^\top Q-I\|_\infty \le \varepsilon.`

The [Muon theory file](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Optimization/Muon.lean)
packages these cases as exact, approximate, and checked-backend contracts. QR-backed directions give
an exact path under positive-pivot hypotheses. Newton-Schulz-style directions give a residual-checked
approximate path, together with fixed-point exact statements when the iteration has reached the
corresponding algebraic condition.

The checked-backend theorem names are deliberately verbose:

```
#check NN.MLTheory.Optimization.Muon.update_has_exact_certified_step_of_checked_backend
#check NN.MLTheory.Optimization.Muon.update_has_approx_certified_step_of_checked_backend
#check NN.MLTheory.Optimization.Muon.update_direction_has_approx_column_gram_of_checked_backend
#check NN.MLTheory.Optimization.Muon.update_has_exact_certified_step_qr
```

They encode the trust boundary. A backend may be fast, randomized, iterative, or external. Lean only
uses the backend as an orthogonalizing step after the backend's success predicate has been checked
or assumed explicitly. That keeps "Muon step executed" separate from "direction has an exact or
approximate Gram certificate."

GaLore-style code is different. GaLore is a gradient-projection strategy, not a single optimizer
name. The runtime object is a projector/lift pair around a base update:

$$`p_{t+1}=p_t-\eta\,\mathrm{lift}(\mathrm{project}(g_t)).`

The current checked baseline says that if the projector is the identity, projected SGD is ordinary
SGD. A future low-rank projector or refresh policy can optimize memory and matrix structure, but it
has to state its own projection contract instead of being hidden inside the word "optimizer."

This naming is reflected in the trainer API. Standard trainer configs use names such as
`optim.sgd`, `optim.adamW`, and `optim.adadelta`. Runtime-level extension points use more explicit
names such as `optim.muon.optimizer` and `optim.galore.sgd`, because those calls must record
their backend or projection as part of the mathematical object.

# Gradient Descent As A Contractive Map

The core convergence theorem is not "SGD always converges." The theorem studies the ideal update
`step eta g x = x - eta * g x`.

In mathematical notation:

$$`x_{t+1}=x_t-\eta g(x_t).`

Before reading the general theorem, run the recurrence for

$$`f(x)=x^2,\qquad \nabla f(x)=2x,\qquad \eta=\frac14.`

Create `GradientDescentDemo.lean`:

```
import Mathlib

def grad (x : ℚ) : ℚ := 2 * x
def step (x : ℚ) : ℚ := x - (1 / 4) * grad x

#eval step 4
#eval step (step 4)

example (x : ℚ) : step x = x / 2 := by
  simp [step, grad]
  ring
```

Lean prints `2` and `1`. The final theorem proves why: each step halves the distance to the
minimizer $`0`, so the squared distance contracts by a factor of $`1/4`. The library theorem replaces
this one-dimensional calculation by assumptions on an abstract gradient map.

If `g` is strongly monotone with parameter $`\mu`, Lipschitz with parameter $`L`, and the step size
$`\eta` is in the safe range, then one step is contractive:

> the distance between `step eta g x` and `step eta g y` is at most $`q` times the distance between
> $`x` and $`y`.

The [linear convergence API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Optimization/GDLinearConvergence.lean) names the
predicate `StrongMonotone mu g`. Informally, it says that the inner product of
$`g(x)-g(y)` with $`x-y` dominates $`\mu\lVert x-y\rVert^2`.

The two analytic hypotheses can be read as:

$$`\langle g(x)-g(y),x-y\rangle\ge \mu\|x-y\|^2`

and

$$`\|g(x)-g(y)\|\le L\|x-y\|.`

Under these hypotheses, `step_norm_sq_le` proves the one-step inequality. The
[strongly convex gradient descent API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Optimization/StronglyConvexGD.lean) then
iterates the inequality. Its theorem `dist_sq_iterate_le_of_q_lt_one` is the statement readers
should remember:

> If the contraction factor $`q(\eta,\mu,L)` is nonnegative and strictly below one, then repeated gradient
> descent steps shrink the squared distance to the reference point geometrically.

The mathematical content is in the hypotheses. The Lean theorem keeps those conditions explicit
because a loss curve cannot infer them.

The contraction and convergence shapes are:

$$`\|T_\eta(x)-T_\eta(y)\|^2\le q\|x-y\|^2`

with a typical monotone/Lipschitz factor

$$`q=1-2\eta\mu+\eta^2L^2,`

and then

$$`\|x_t-x^\star\|^2\le q^t\|x_0-x^\star\|^2.`

Concrete theorem names:

```
#check NN.MLTheory.Optimization.GDLinearConvergence.StrongMonotone
#check NN.MLTheory.Optimization.GDLinearConvergence.step_norm_sq_le
#check NN.MLTheory.Optimization.StronglyConvexGD.dist_sq_iterate_le_of_q_lt_one
#check NN.MLTheory.Optimization.StronglyConvexGD.error_abs_contract_real
```

The result is the standard smooth/strongly-monotone contraction argument found in convex
optimization texts such as Nesterov's
[*Introductory Lectures on Convex Optimization*](https://link.springer.com/book/10.1007/978-1-4419-8853-9).
TorchLean attaches this familiar rate to the exact update map and assumptions used by its training
theorems.

# Smoothness, Strong Convexity, And The Bridge

Most papers state convergence using smoothness and strong convexity of an objective `f`, not
strong monotonicity of an abstract gradient map. TorchLean keeps both vocabularies and proves the
bridge between them.

The informal first order strong convexity condition says that `f y` lies above the tangent model at
$`x` plus a quadratic term with coefficient $`\mu/2`.

The [smooth strong convex bridge API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Optimization/SmoothStrongConvexBridge.lean)
turns that objective level statement into a gradient map statement. The theorem to recognize is
`strongMonotone_gradient_of_firstOrderStrongConvex`: under the first order strong convexity
hypothesis, the gradient is strongly monotone. That bridge lets a training theorem move from "the
loss has these analytic assumptions" to "the update map is contractive."

```
#check NN.MLTheory.Optimization.SmoothStrongConvexBridge.FirstOrderStrongConvex
#check NN.MLTheory.Optimization.SmoothStrongConvexBridge.strongMonotone_gradient_of_firstOrderStrongConvex
```

The bridge is useful because autograd theorems usually speak about derivatives of a loss, while
convergence theorems often speak about a gradient map. The theorem connects those two vocabularies
without treating "gradient" as an informal word.

# How This Connects To Verified Training

The autograd theorems explain why the gradient path computes the intended derivative. Runtime
approximation theorems explain how close a rounded update is to the ideal update. Optimization
theory is where we state what that update means as an algorithm.

For example, a theorem about a full training run can be read as a composition of three facts:

- an autograd theorem saying the gradient is the adjoint derivative of the loss;
- a runtime approximation theorem saying the executable update is close to the ideal update;
- an optimization theorem saying the ideal update contracts under smoothness and strong convexity
  hypotheses.

Those hypotheses matter. The optimization layer avoids turning a loss curve into a convergence
claim: convexity, smoothness, strong monotonicity, and step size conditions remain visible in the
theorem statement.

There is also a state boundary between these equations and a native trainer. Runtime Adam-family
optimizers maintain moment buffers and a step counter for each parameter, while checkpoints must
restore those states with the same parameter identity and ordering. The pure `TensorOptimizer` and
`StepSpec` theorems prove the recurrence they state; they do not by themselves prove allocation,
aliasing, checkpoint restoration, or the mutable runtime state machine. An end-to-end optimizer
claim needs a bridge showing that the trainer's stored state implements the same recurrence.

# Claim Shape

TorchLean formalizes selected first order optimization facts and their assumptions, so future
training theorems can cite named Lean objects rather than prose folklore. A theorem about a
particular run must still name its hypotheses: the gradient being used, the scalar semantics, the
step-size condition, and any backend contract such as a Muon orthogonalizer certificate or a
projector law. The classical lineage is the convex optimization tradition: smoothness, strong
convexity, gradient descent, and contraction arguments. The Lean declarations let us use that
tradition without hiding its assumptions behind a training loop.

# References

- Yurii Nesterov,
  [*Introductory Lectures on Convex Optimization*](https://link.springer.com/book/10.1007/978-1-4419-8853-9),
  Springer 2004.
- Ilya Loshchilov and Frank Hutter,
  ["Decoupled Weight Decay Regularization"](https://arxiv.org/abs/1711.05101),
  ICLR 2019.

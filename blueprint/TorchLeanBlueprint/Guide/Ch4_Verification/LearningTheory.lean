import VersoManual

open Verso.Genre Manual

#doc (Manual) "Learning Theory" =>
%%%
tag := "learning-theory"
%%%

Learning theory belongs in TorchLean because many ML claims become mathematical before they become
experimental. A training script can compute a loss curve, run an attack, add a differentially
private training mechanism, or fit a ridge regression model. The guarantee usually lives in a
predicate: privacy, robustness, stability, convergence, or a finite-precision bridge from an ideal
theorem to executable arithmetic.

TorchLean names those predicates and makes them usable beside models, runtimes, and verification
artifacts. No single certificate checker covers all of learning theory.
The recurring pattern is:

- state a mathematical predicate such as privacy, robustness, or stability;
- compute a runtime diagnostic or artifact when that evidence matters;
- add a bridge theorem when an artifact is used to support a formal claim.

# Five Kinds Of Claim

The learning theory material is organized around five concrete objects:

- *Randomized mechanism*: a map from inputs to probability measures over outputs; Lean states
  $`(\varepsilon,\delta)` privacy, pure privacy, monotonicity in $`\delta`, and post processing.
- *Robustness predicate*: a classifier is stable on a perturbation ball; Lean names tensor norms,
  local balls, Lipschitz predicates, and certified robustness.
- *Algorithmic stability*: replacing one training example changes the learned loss only slightly;
  Lean gives typed datasets, `replaceAt`, `removeAt`, learning maps, and loss change bounds.
- *Dynamical stability*: trajectories stay bounded or converge under stated hypotheses; Lean names
  Lyapunov, ISS, BIBO, incremental, practical, and finite time predicates.
- *Ridge regression case study*: a one-dimensional strongly regularized ERM theorem, with a real
  stability theorem plus an `IEEE32Exec` execution bridge.

These objects do not share one proof method. Differential privacy is an inequality between
measures. Robustness quantifies over a neighborhood of an input. Algorithmic stability compares
two executions of a learning algorithm on neighboring datasets. Dynamical stability concerns an
entire trajectory. The ridge result is a concrete algebraic proof whose constants can be carried
into a numerical analysis. Their common feature is that the quantified object and its boundary are
stated before any runtime evidence is interpreted.

# Differential Privacy

The core privacy definitions live in
[NN.MLTheory.LearningTheory.DifferentialPrivacy.Core API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/LearningTheory/DifferentialPrivacy/Core.lean).
The central abstraction is a randomized mechanism: a function from inputs of type `alpha` to
probability measures over outputs of type `beta`.

An adjacency relation `Adj : α → α → Prop` says which inputs are neighboring datasets or
neighboring records. The definition of $`(\varepsilon,\delta)`-DP is the standard event inequality: for every
adjacent pair and measurable event `S`, the probability of `M a` landing in `S` is at most
$`e^\varepsilon` times the corresponding probability for `M a'`, plus $`\delta`.

The definition returns a `ProbabilityMeasure`, which is general enough for discrete mechanisms,
continuous mechanisms, and randomized training procedures.
The mechanism does not have to be a particular optimizer or sampler; the definition says what any
privacy proof must establish.

In symbols, the definition has the usual event form:

$$`\forall D\sim D',\;\forall S,\qquad
\Pr[M(D)\in S]\le e^\varepsilon \Pr[M(D')\in S]+\delta.`

Two closure facts are especially important:

- `differentialPrivacy_mono_delta`: a mechanism that is private for $`\delta_1` is also private for any
  looser $`\delta_2\ge\delta_1`.
- `differentialPrivacy_postprocess`: measurable post processing preserves DP.

The second theorem is the practical bridge to ordinary ML practice. In training code, we often
train a private model and then pass it through exporters, evaluators, dashboards, or downstream
selection logic. The DP theorem says the post processing step does not need to inspect the private
input again. TorchLean states that as a Lean theorem rather than relying on a comment.

The theorem shape is:

$$`M\ \text{is}\ (\varepsilon,\delta)\text{-DP}
\quad\Longrightarrow\quad
f\circ M\ \text{is}\ (\varepsilon,\delta)\text{-DP}.`

The proof is short for a mathematical reason. For a measurable output event `T`, the
post-processed mechanism lands in `T` exactly when the original mechanism lands in the measurable
preimage `f⁻¹(T)`. Applying the DP inequality to that preimage gives

$$`\begin{aligned}
\Pr[f(M(D))\in T]
&=\Pr[M(D)\in f^{-1}(T)]\\
&\le e^\varepsilon\Pr[M(D')\in f^{-1}(T)]+\delta\\
&=e^\varepsilon\Pr[f(M(D'))\in T]+\delta.
\end{aligned}`

The measurability hypothesis is what licenses the preimage step. No assumption about the internal
representation of the model or report is needed.

The reference point is the standard DP event inequality from Dwork, McSherry, Nissim, and Smith,
["Calibrating Noise to Sensitivity in Private Data Analysis"](https://link.springer.com/chapter/10.1007/11681878_14)
(TCC 2006). TorchLean's current definitions name the property and closure rules; they are not a
claim that a particular optimizer is DP-SGD. A DP-SGD theorem would still need the sampling,
clipping, noise calibration, and composition accounting hypotheses.

In particular, no privacy budget should be inferred from the presence of the DP namespace. The
current source supplies the semantic target and closure laws; it does not contain a runtime privacy
accountant, a Gaussian-mechanism calibration theorem, or an end-to-end theorem for a training
command. Those are separate implementation and proof obligations.

# Robustness: Spec Versus Runtime

Robustness is split by trust boundary into two files:

- [Robustness.Spec](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/LearningTheory/Robustness/Spec.lean) defines the mathematical
  predicates.
- [Robustness.Runtime](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/LearningTheory/Robustness/Runtime.lean) defines executable
  `Float` diagnostics.

The spec side works over TorchLean tensors without committing to one runtime scalar. It defines:

- `tensorLinfNorm` and `tensorL2Norm`,
- `tensorDistance`,
- closed tensor balls,
- global and local Lipschitz continuity,
- adversarial robustness at a point,
- certified robustness for classifiers,
- uniform robustness over a finite dataset,
- contraction mappings,
- local sensitivity ratios.

Proof developments use this vocabulary. For example, a certified robustness claim should say that
the classifier is constant on an $`\varepsilon` ball. A failed search attack is evidence, but it is not the
certificate.

A typical local robustness predicate has the form:

$$`\operatorname{Robust}(f,x,y,\varepsilon)
\Longleftrightarrow
\forall x',\;\|x'-x\|_\infty\le \varepsilon
\Rightarrow
\operatorname*{argmax} f(x')=y.`

The runtime layer specializes norms and distances to `Float` and provides empirical helpers such as
Lipschitz ratios from finite samples and deterministic perturbation sampling. We built that layer because
engineers need fast diagnostics, but its documentation is explicit: a sampled maximum is not a
global certificate. It is evidence, a debugging aid, or a counterexample search tool.

That distinction is one of the main differences from mainstream ML evaluation scripts. A typical
robustness notebook might compute "max observed ratio" and report it as if it were a property of
the model. TorchLean names it as an empirical runtime quantity unless a separate proof connects it
to a certified bound.

The bundled logit-margin artifact makes the difference visible:

```
lake exe verify -- margin-cert
```

It reports:

```
[margin cert] examples=360
[margin cert] nominal_ok=349
[margin cert] certified_ok=318
```

`nominal_ok` counts correctly predicted recorded examples. `certified_ok` counts examples whose
stored bounds imply the required margin. Neither number is automatically a theorem about unseen
data, and the second number depends on the soundness and provenance of the bounds in the artifact.

# Algorithmic Stability

The [algorithmic stability API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/LearningTheory/Stability/Core.lean) uses one
central representation choice: a dataset of size `n` is a `Spec.Vec n Z`, so its shape is part of
the type, not an untyped list whose length must be remembered separately. That lets stability definitions quantify over replace one and remove one
perturbations while keeping the sample size in the type.

The core definitions include:

- coordinate access for datasets,
- `replaceAt` and `removeAt`,
- deterministic learning maps `Dataset n Z → H`,
- losses over the reals,
- empirical error,
- true population error under a probability measure,
- standard stability predicates over algorithms and losses.

In mainstream ML code, "replace one example and retrain" is usually an experiment. In TorchLean, it
is also a formal operation with a type. A theorem about replace one stability can refer to
`replaceAt S i z'` directly and inherit the fact that the modified object is still a dataset of the
same size.

The theorem shape is the standard uniform stability inequality:

$$`\forall S,S^{(i)},z,\qquad
|\ell(A(S),z)-\ell(A(S^{(i)}),z)|\le \beta.`

The representation can be explored without proving a stability bound. Put the following in
`StabilityDataset.lean`:

```
import NN.MLTheory.LearningTheory.Stability.Core

open NN.MLTheory.LearningTheory.Stability

def sample : Dataset 3 Nat :=
  Dataset.ofFn (fun i => i.val + 10)

def changed : Dataset 3 Nat :=
  replaceAt sample ⟨1, by decide⟩ 99

#eval List.ofFn (Dataset.toFn sample)
#eval List.ofFn (Dataset.toFn changed)
```

Elaborating the file prints:

```
[10, 11, 12]
[10, 99, 12]
```

Try to replace coordinate `3`. Lean rejects the index before any learner runs, because an
element of `Fin 3` must carry a proof that its value is smaller than three. The type ensures that
replace-one perturbation preserves dataset size; the stability theorem then reasons about how the
learner responds to that perturbation.

The classical reference is Bousquet and Elisseeff,
["Stability and Generalization"](https://jmlr.org/papers/v2/bousquet02a.html) (JMLR 2002). The
TorchLean definitions follow the same proof habit: first make the perturbation of the dataset
explicit, then state how much the learned loss can change.

# Dynamical Stability

The stability entrypoint also imports
[NN.MLTheory.LearningTheory.Stability.Dynamics API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/LearningTheory/Stability/Dynamics.lean).
This covers recurrence systems, such as $`x_{t+1}=f(x_t)`, and systems driven by inputs, where
the next state depends on an input sequence. The spec file
[Dynamics.Spec](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/LearningTheory/Stability/Dynamics/Spec.lean) names predicates
such as Lyapunov stability, asymptotic stability, exponential stability, input to state stability,
BIBO stability, incremental stability, practical stability, finite time stability, and data/model
stability. The runtime file
[Dynamics.Runtime](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/LearningTheory/Stability/Dynamics/Runtime.lean) provides
`Float` diagnostics for concrete systems.

Neural-network learning theory is not limited to static supervised learning. Recurrent models,
samplers, controllers, RL policies interacting with state, and learned dynamical systems all need
language for trajectories. The runtime side is empirical unless a theorem connects the observed
diagnostic to the spec predicate.

# Ridge Regression As A Worked Theorem

The most concrete learning theory development is
[NN.MLTheory.LearningTheory.Stability.RidgeRegression1D.Real API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/LearningTheory/Stability/RidgeRegression1D/Real.lean).
It proves a replace one uniform stability bound for one dimensional ridge regression with squared
loss under bounded inputs.

The theorem follows the classical strongly convex ERM argument in a small setting, so the proof is
inspectable:

1. Each example is a bounded pair $`(x,y)` with $`\lvert x\rvert\le X` and $`\lvert y\rvert\le Y`.
2. The closed form fit is
   $`\hat w(S) = \frac{\sum_i x_i y_i}{\sum_i x_i^2 + \lambda N}`.
3. Replacing one example changes the numerator and denominator by controlled finite sums.
4. The proof bounds the difference between the two fitted weights.
5. A difference of squares argument converts the weight change bound into a loss change bound.

The final stability statement has the same form as the general predicate. Writing $`N=n+1`,
the proved constant is

$$`\beta
=\frac{4X^2Y^2(\lambda+X^2)^2}{\lambda^3N}.`

Thus the theorem concludes

$$`|\ell(\hat w(S),z)-\ell(\hat w(S'),z)|
\le
\frac{4X^2Y^2(\lambda+X^2)^2}{\lambda^3N}.`

The $`1/N` dependence comes from the ridge denominator. The stronger dependence on $`\lambda` records the
cost of controlling both the fitted weight and the change in its reciprocal denominator. This is
an explicit valid bound, not a claim that the constant is optimal.

The worked theorem is compact enough to inspect. The estimator is not a foreign function.
The dataset is the same `Dataset` representation from the stability core. The boundedness
assumptions are carried by types and hypotheses. The final statement is a Lean theorem, not a prose
claim next to a Python implementation.

A reader should notice what this theorem does and does not say. It proves a real-valued stability
bound for the closed-form ridge estimator under bounded data and positive regularization. It does
not say that an arbitrary minibatch trainer, an iterative solver stopped early, or a float32
implementation has the same bound. Those variants need their own algorithm and numerical bridge.

# Runtime And Spec Splits

The learning theory tree repeats a pattern: a spec predicate or theorem is kept
separate from the executable runtime diagnostic, and an optional float32 or artifact bridge connects
them only when the hypotheses have been stated.

For robustness, the split is `Robustness.Spec` versus `Robustness.Runtime`.

For dynamical stability, the split is `Stability.Dynamics.Spec` versus
`Stability.Dynamics.Runtime`.

For ridge regression, the ideal theorem is in `RidgeRegression1D.Real`, while the executable
float32 development is in
[RidgeRegression1D/IEEE32Exec](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/LearningTheory/Stability/RidgeRegression1D/IEEE32Exec.lean).
The IEEE32Exec side implements the algorithm with TorchLean's executable binary32 model and states
the bridge to a proof level round after each primitive semantics under explicit finiteness
conditions.

The numerics philosophy is the same one used elsewhere in TorchLean. A theorem over the reals is not
automatically a native float theorem. We first prove the clean mathematical result, then separately
state what finite precision execution means and which hypotheses are needed to connect it.

The executable ridge bridge is deliberately narrow:

```
#check NN.MLTheory.LearningTheory.Stability.RidgeRegression1D.IEEE32Exec.ridgeFit1DExec
#check NN.MLTheory.LearningTheory.Stability.RidgeRegression1D.IEEE32Exec.ridgeFit1D_execExpr_toReal_eq_fp32Spec_of_finiteEval
```

The second theorem is a finite-evaluation bridge: if the executable expression evaluates finitely,
its real interpretation agrees with the proof-level FP32 expression. That is not a stability theorem
by itself. It is one bridge that can be composed with a stability theorem when the remaining
rounded-arithmetic error bounds have also been supplied.

This example is a useful model for reading every runtime diagnostic on the page: first identify the
ideal predicate, then the executable quantity, then the theorem that relates them. If the third
piece is absent, the executable result remains a diagnostic even when its value looks favorable.

# Reading A Learning-Theory Result

A learning-theory claim has four visible fields:

```
object       : mechanism, classifier, algorithm, dataset, trajectory, or estimator
property     : privacy, robustness, stability, convergence, or bounded residual
evidence     : theorem, checker, runtime diagnostic, or imported artifact
boundary     : real semantics, FP32/IEEE32Exec bridge, or external producer assumption
```

These four fields separate a theorem from nearby runtime evidence. In particular:

- If a script says an optimizer is differentially private because it used a DP library, the formal
  claim must define a mechanism and prove the DP event inequality or import a
  checked theorem.
- If no attack found an adversarial example, that is runtime evidence unless it implies
  `isCertifiedRobust`.
- If a model seems stable when retrained, the formal statement is a replace-one stability predicate
  over `Dataset n Z`.
- If a dynamical system stayed bounded in simulation, the formal statement is a BIBO, ISS,
  Lyapunov, or related stability predicate.
- If the theorem is over the reals but the code uses float32, the missing link is an
  `IEEE32Exec`/FP32 bridge with explicit finite path hypotheses.

# References

- Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith,
  ["Calibrating Noise to Sensitivity in Private Data Analysis"](https://link.springer.com/chapter/10.1007/11681878_14),
  TCC 2006.
- Olivier Bousquet and Andre Elisseeff,
  ["Stability and Generalization"](https://jmlr.org/papers/v2/bousquet02a.html),
  JMLR 2002.

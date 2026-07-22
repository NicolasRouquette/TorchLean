import VersoManual

open Verso.Genre Manual

#doc (Manual) "Classical and Structural ML Proofs" =>
%%%
tag := "classical-ml-proofs"
%%%

Some neural-network properties are independent of any particular GPU kernel or training loop.
Hopfield networks have an energy argument. ReLU networks can assemble multiplication from
piecewise-linear approximants. Recurrent state-space models are causal because their output at time
`t` is computed before future inputs are seen. These are structural facts about the mathematical
models.

TorchLean formalizes such results beside its runtime developments so that later work can connect
them. The proofs in this chapter do not certify a CUDA implementation, but neither are they
informal descriptions of an architecture. They are Lean theorems about the spec-level definitions.

# Hopfield Dynamics

A TorchLean Hopfield state is a Boolean vector:

```
abbrev State (n : Nat) := Fin n → Bool
```

The numeric activation map interprets `true` as `+1` and `false` as `-1`. Parameters contain a
weight matrix and a threshold vector:

```
structure Params (α : Type) (n : Nat) where
  W : Fin n → Fin n → α
  θ : Fin n → α
```

For state `s`, write `xᵢ ∈ {-1,+1}` for its numeric activation. The net input to neuron `u` is

$$`\operatorname{net}_u(s)=\sum_j W_{uj}x_j.`

`updateAt p s u` changes only coordinate `u`, using

$$`x_u'=
\begin{cases}
+1,&\theta_u\leq\operatorname{net}_u(s),\\
-1,&\operatorname{net}_u(s)<\theta_u.
\end{cases}`

The non-strict comparison fixes a detail often omitted on paper: ties go to `+1`. That convention
becomes important in the convergence proof.

The energy is

$$`E(s)
=-\frac12\sum_i\sum_j W_{ij}x_i x_j
 +\sum_i\theta_i x_i.`

Under symmetric weights and zero diagonal,

$$`W_{ij}=W_{ji},
\qquad W_{ii}=0,`

the theorem
[`energy_updateAt_le`](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/Hopfield/Energy.lean)
proves

$$`E(\operatorname{updateAt}(p,s,u))\leq E(s).`

The proof expands the quadratic energy difference. Symmetry makes the changed row and column
contribute the same net-input term, while the zero diagonal removes the self-interaction. When the
coordinate actually changes and the net input is not tied with the threshold,
`energy_updateAt_lt_of_change_of_ne` strengthens the inequality to a strict decrease.

# Execute A Two-Neuron Update

The spec is executable over rational numbers. This scratch file uses two mutually excitatory
neurons, zero thresholds, and the initial state `[+1,-1]`:

```
import NN.Spec.Models.Hopfield

open Spec.Hopfield

def p : Params Rat 2 where
  W := fun i j => if i = j then 0 else 1
  θ := fun _ => 0

def s : State 2 := fun i => i = 0
def s' : State 2 := updateAt p s 1

#eval List.ofFn s
#eval List.ofFn s'
#eval energy p s
#eval energy p s'
```

The current output is:

```
[true, false]
[true, true]
1
-1
```

The update aligns the second neuron with the first, and the energy decreases from `1` to `-1`.
Changing `W 0 1` without changing `W 1 0` still produces an executable state sequence, but it
prevents use of `energy_updateAt_le`: Lean asks for `SymmetricW p`. Setting a diagonal weight to a
nonzero value similarly leaves the program runnable while invalidating the theorem’s
`DiagonalZero p` premise.

# Why Non-Increasing Energy Is Not Quite Enough

If every state change strictly lowered energy, finiteness would immediately rule out cycles. Ties
make the argument subtler. With the convention “ties go to `+1`,” a state may change while energy
stays equal. TorchLean therefore uses the number of positive neurons,

$$`\operatorname{pluses}(s)
=|\{i\mid s_i=\texttt{true}\}|,`

as a secondary progress measure. For one full cyclic sweep, `cycleUpdate_progress` proves:

- either energy strictly decreases;
- or energy is unchanged and `pluses` strictly increases.

The lexicographic pair

$$`\bigl(E(s),-\operatorname{pluses}(s)\bigr)`

therefore progresses whenever a sweep changes the state. Since `State n` is finite,
[`cycleUpdate_no_nontrivial_cycles`](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/Hopfield/Convergence.lean)
rules out a nontrivial cycle, and `cycleUpdate_exists_fixedpoint_le_card` gives a fixed point within
at most `Fintype.card (State n)` sweeps. The more explicit
`cycleUpdate_exists_fixedpoint_le_pow` states the corresponding `2^n` bound.

Inspect the exact hypotheses in the Infoview:

```
import NN.MLTheory.Proofs.Hopfield

open NN.MLTheory.Proofs.Hopfield

#check energy_updateAt_le
#check cycleUpdate_progress
#check cycleUpdate_exists_fixedpoint_le_pow
```

These are theorems about asynchronous coordinate updates arranged into cyclic sweeps. They do not
apply automatically to synchronous updates, stochastic schedules, modern continuous-state
Hopfield layers, or a floating-point kernel. Each variation needs its own transition relation and
energy argument.

# ReLU Networks As An Algebra Of Approximants

ReLU is piecewise linear, so one hidden layer cannot represent multiplication exactly on all of
`ℝ²`. It can, however, approximate multiplication uniformly on a bounded box.

The identity

$$`xy=\frac{(x+y)^2-(x-y)^2}{4}`

reduces the problem to approximating the square function on `[-2M,2M]`. The file
[`ReLUMulApprox`](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/ReLU/Approx/ReLUMulApprox.lean)
first builds a one-dimensional ReLU approximation to `u²`, lifts copies along the ridge directions

$$`w_+=(1,1),\qquad w_-=(1,-1),`

and combines their outputs with coefficients `1/4` and `-1/4`.

The final theorem is:

```
theorem relu_mul_universal_approximation_box
    {M : ℝ} (hM : 0 < M) :
  ∀ ε > 0,
    ∃ (hidDim : ℕ)
      (l1 : LinearSpec ℝ 2 hidDim)
      (l2 : LinearSpec ℝ hidDim 1),
    ∀ x ∈ box M,
      |mulFun x - mlpEvalNd l1 l2 x| < ε
```

The box hypothesis bounds both coordinates by `M`. Without it, no finite piecewise-linear function
can uniformly approximate the quadratic growth of multiplication on the whole plane.

The bridge module
[`ReLUMlpBridge`](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/ReLU/Bridge/ReLUMlpBridge.lean)
supplies network algebra used in these constructions. A particularly useful exact identity is

$$`\operatorname{ReLU}(u)-\operatorname{ReLU}(-u)=u.`

In Lean:

```
lemma relu_sub_relu_neg (u : ℝ) :
  relu u - relu (-u) = u
```

This identity lets a ReLU network carry affine terms exactly even though individual hidden units
clip negative values.

To explore the proof surface:

```
import NN.MLTheory.Proofs.ReLU.Approx.ReLUMulApprox
import NN.MLTheory.Proofs.ReLU.Bridge.ReLUMlpBridge

open NN.MLTheory.Proofs.ReLUMlpBridge
open NN.MLTheory.Proofs.ReLUMulApprox

#check relu_sub_relu_neg
#check relu_mul_universal_approximation_box
```

Try specializing the multiplication theorem with `M = 0`. The proof cannot supply `0 < M`.
That does not mean multiplication is hard on the singleton zero box; it means this particular
construction and theorem are stated for a positive-radius box. A separate zero-radius lemma would
be trivial but would not strengthen the useful approximation result.

On arbitrary compact subsets of finite-dimensional real space, the larger
[`CompactSet`](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/ReLU/Approximation/CompactSet.lean)
development combines coordinate polynomials, multiplication approximants, and Stone-Weierstrass.
`relu_universal_approximation_compact` proves density of one-hidden-layer ReLU MLPs in continuous
real-valued functions on the compact domain. This is an exact existence theorem over `ℝ`; it is not
a runtime or training guarantee.

# Causality In State-Space Models

A recurrent sequence model should not revise an earlier output after future tokens arrive. For a
simple state-space recurrence,

$$`h_{t+1}=A_t h_t+B_t x_t,\qquad
y_t=C_t h_t+D_t x_t,`

the causal claim can be phrased without derivatives or probability:

$$`\operatorname{take}_{|xs|}
  \bigl(\operatorname{outputs}(\operatorname{run}(xs\mathbin{++}ys))\bigr)
=
\operatorname{outputs}(\operatorname{run}(xs)).`

The theorem says that appending a future suffix `ys` preserves every output already produced for
the prefix `xs`.

[`MambaCausality`](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/StateSpace/MambaCausality.lean)
proves this statement for three increasingly rich specifications:

- `DiagonalS4Spec`;
- `MambaBlockSpec`;
- `SelectiveMambaBlockSpec`, including its carried convolution history.

The public selective theorem is:

```
theorem selectiveMamba_runList_append_outputs_prefix
    (m : SelectiveMambaBlockSpec α
      inputDim innerDim stateDim outputDim convWidth)
    (h0 : Tensor α (.dim innerDim (.dim stateDim .scalar)))
    (xs ys : List (Tensor α (.dim inputDim .scalar))) :
  (m.runList h0 (xs ++ ys)).2.take xs.length =
    (m.runList h0 xs).2
```

The theorem is polymorphic over any scalar `α` with a TorchLean `Context`. Its proof is structural:
induct on `xs`, unfold one recurrent step, and apply the induction hypothesis to the updated state
and history. It does not require commutative or exact arithmetic because causality depends on
evaluation order, not algebraic rearrangement.

Open the declarations:

```
import NN.MLTheory.Proofs.StateSpace.MambaCausality

open NN.MLTheory.StateSpace

#check diagonalS4_runList_append_outputs_prefix
#check compactMamba_runList_append_outputs_prefix
#check selectiveMamba_runList_append_outputs_prefix
```

A useful failed variation is to replace `.take xs.length` by `.take (xs.length + 1)`. The extra
output is the first one allowed to depend on the suffix, so the theorem is false in general. The
prefix length in the checked statement is exactly the causal boundary.

# Proof Boundary

The three developments establish different kinds of structure:

:::table +header
*
  * Development
  * Proved object
  * Not established by that theorem
*
  * Hopfield
  * finite real-valued energy and cyclic asynchronous dynamics
  * floating execution or arbitrary update schedules
*
  * ReLU approximation
  * existence of real MLP parameters with uniform error
  * training convergence or binary32 error
*
  * Mamba/S4
  * prefix preservation of spec-level list runners
  * equality with a particular fused scan kernel
:::

The Hopfield example executes over `Rat`; the energy theorem is stated over `ℝ`. The Mamba
causality theorem works over an abstract `Context`, but a runtime refinement theorem is still
needed to connect a backend kernel to the spec runner. The ReLU theorem constructs real-valued
layers, while quantization and rounded execution require the finite-precision bridge described in
the approximation chapter.

These boundaries are what make the results reusable. A future CUDA proof does not need to reprove
the Hopfield energy algebra, and a future Mamba kernel proof does not need to rediscover the list
causality invariant. It only needs to connect the new executable object to the mathematical one
already named here.

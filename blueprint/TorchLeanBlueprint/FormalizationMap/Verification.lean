import Verso
import VersoManual
import VersoBlueprint
import NN.Proofs.Autograd.Core.SemiringCorrectness
import NN.Proofs.Autograd.Tape.Algebra.Soundness
import NN.Proofs.Autograd.Tape.Algebra.Nodes
import NN.Proofs.Autograd.Tape.Core.FDeriv
import NN.Proofs.Autograd.Runtime.Link.BackwardGraph
import NN.Proofs.Autograd.Runtime.Link.FDeriv
import NN.Verification.TorchLean.Proved.Correctness
import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.Main
import NN.MLTheory.CROWN.Proofs.GraphRunibpEndToEnd
import NN.MLTheory.CROWN.Proofs.GraphCrownCertSoundness
import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Alpha
import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.AlphaBeta
import NN.Verification.Cert.CROWNNodeCertAlphaBeta
import NN.MLTheory.CROWN.Lyapunov.Certificate
import NN.MLTheory.CROWN.Lyapunov.Verification
import NN.MLTheory.Optimization.StronglyConvexGD
import NN.MLTheory.LearningTheory.DifferentialPrivacy.Core
import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximationIEEE32ExecTwoLayerMlp
import NN.Proofs.Verification.ODE.Enclosure
import NN.Verification.Geometry3D.Box3D

open Verso.Genre
open Verso.Genre.Manual
open Informal

#doc (Manual) "Verification" =>

This part of the map follows proofs from local algebra to executable checkers. A solid dependency
edge means the later statement uses the earlier definition or theorem. Runtime checkers that lack a
proved acceptance-to-semantics bridge are described beside the matching proof work, without adding
a dependency edge.

:::group "autograd_correctness"
Algebraic, executable, and analytic accounts of reverse-mode differentiation.
:::

:::definition "autograd_local_adjoint_contract" (parent := "autograd_correctness") (lean := "Proofs.Autograd.Algebra.Node")
Each node in an algebraic proof graph carries forward, JVP, and VJP functions together with the
local dot-product identity relating its JVP and VJP.
:::

:::definition "algebraic_autograd_graph_data" (parent := "autograd_correctness") (lean := "Proofs.Autograd.Algebra.GraphData")
`GraphData` stores a typed sequence of executable forward, JVP, and VJP node operations without
local correctness proofs.
:::

:::definition "algebraic_autograd_graph" (parent := "autograd_correctness") (lean := "Proofs.Autograd.Algebra.Graph")
A proof-carrying algebraic graph is a typed sequence of
{uses "autograd_local_adjoint_contract"}[locally correct nodes]. Its output-shape list records each
intermediate added to the graph.
:::

:::definition "autograd_operation_adjoint_contract" (parent := "autograd_correctness") (lean := "Proofs.Autograd.Algebra.OpSpecCorrect")
`OpSpecCorrect` packages a unary tensor operation, its JVP, and the local inner-product identity
relating that JVP to the operation's VJP.
:::

:::definition "autograd_operation_node_adapter" (parent := "autograd_correctness") (lean := "Proofs.Autograd.Algebra.Node.ofOpSpecCorrect")
The unary-operation adapter places an
{uses "autograd_operation_adjoint_contract"}[`OpSpecCorrect`] value at a typed input index and
produces {uses "autograd_local_adjoint_contract"}[a proof-carrying graph node].
:::

:::theorem "autograd_algebra_adjoint" (parent := "autograd_correctness") (lean := "Proofs.Autograd.Algebra.Graph.backprop_correct")
For an {uses "algebraic_autograd_graph"}[algebraic graph] over a commutative semiring, reverse
accumulation is adjoint to the graph JVP.
:::

:::proof "autograd_algebra_adjoint"
Graph induction expands the JVP and VJP at each node, then closes the new step with
{uses "autograd_local_adjoint_contract"}[that node's local adjoint law].
:::

:::definition "linear_layer_adjoint_contract" (parent := "autograd_correctness") (lean := "Proofs.Autograd.Algebra.linearCorrect")
The linear operation satisfies
{uses "autograd_operation_adjoint_contract"}[the unary operation contract] over any commutative
semiring. Its correctness field applies
{uses "tensor_linear_adjointness"}[matrix-vector adjointness] and commutativity of the tensor dot
product.
:::

:::theorem "autograd_runtime_backward_link" (parent := "autograd_correctness") (lean := "Proofs.Autograd.Algebra.Graph.backwardDenseFrom_lowerGraphToTape_eq_backpropAllCtx")
Lowering an {uses "algebraic_autograd_graph"}[algebraic graph] to
{uses "runtime_autograd_tape"}[the runtime tape] and running dense backward returns the graph's
proved reverse accumulator after its typed tensor context is converted to the runtime array
representation.
:::

:::proof "autograd_runtime_backward_link"
Induction over {uses "algebraic_autograd_graph"}[the graph] maintains the index and
context correspondence of {uses "runtime_autograd_tape"}[the runtime tape] through reverse
accumulation.
:::

:::theorem "autograd_real_tape_inner_product" (parent := "autograd_correctness") (lean := "Proofs.Autograd.Graph.backprop_correct_inner")
For a real proof-carrying tape, the inner product of a JVP with a cotangent seed equals the inner
product of the input tangent with reverse accumulation.
:::

:::proof "autograd_real_tape_inner_product"
Tape induction applies each real node's local vector adjoint law while preserving the Euclidean
inner product across context append and split operations.
:::

:::theorem "autograd_real_graph_adjoint" (parent := "autograd_correctness") (lean := "Proofs.Autograd.Graph.backpropVec_eq_adjoint_fderiv_at")
For differentiable real graph nodes, reverse accumulation equals the adjoint Fréchet derivative at
the chosen point.
:::

:::proof "autograd_real_graph_adjoint"
{uses "autograd_real_tape_inner_product"}[Real tape soundness] supplies the inner-product identity.
The analytic hypotheses identify the graph JVP with a Fréchet derivative, and the adjoint is then
characterized by its inner products.
:::

:::theorem "autograd_lowered_tape_fderiv" (parent := "autograd_correctness") (lean := "Proofs.Autograd.Algebra.Graph.backwardDenseFrom_lowerGraphToTape_adjoint_fderiv_at")
For a real algebraic graph at a differentiable execution point, lowering the graph and running the
exact dense tape returns the full algebraic reverse context. Its input prefix is the adjoint
Fréchet derivative of graph evaluation applied to the output seed.
:::

:::proof "autograd_lowered_tape_fderiv"
The real, environment-free algebraic graph and the analytic graph convert in both directions while
preserving evaluation, JVPs, and reverse accumulation. The tape-lowering correctness theorem gives
the full cotangent context; prefix extraction identifies its input block with analytic backprop,
and {uses "autograd_real_graph_adjoint"}[the analytic graph theorem] identifies that value with the
adjoint Fréchet derivative.
:::

:::group "verified_lowering"
The proved lowering from typed TorchLean programs to verifier IR.
:::

:::definition "verified_program_language" (parent := "verified_lowering") (lean := "NN.Verification.TorchLean.Proved.ForwardProgram")
The verified source language is a typed sequence of supported tensor operations with one input and
shape-indexed intermediate values.
:::

:::definition "verified_forward_lowering" (parent := "verified_lowering") (lean := "NN.Verification.TorchLean.Proved.lowerForwardProgramToIR")
`lowerForwardProgramToIR` lowers {uses "verified_program_language"}[the verified source program] to the
shared IR while preserving its typed node order.
:::

:::theorem "verified_forward_well_formed" (parent := "verified_lowering") (lean := "NN.Verification.TorchLean.Proved.Correctness.lowerForwardProgramToIR_wellFormed")
Every graph produced by {uses "verified_forward_lowering"}[the verified forward lowering] passes
{uses "ir_structural_predicate"}[the IR structural well-formedness predicate].
:::

:::proof "verified_forward_well_formed"
Induction over {uses "verified_program_language"}[the source program] shows that
{uses "verified_forward_lowering"}[the lowering] preserves the node-index invariant at every
append.
:::

:::theorem "verified_forward_correct" (parent := "verified_lowering") (lean := "NN.Verification.TorchLean.Proved.Correctness.runForwardIR_eq_evalForward")
Running the output of {uses "verified_forward_lowering"}[the verified forward lowering] with
{uses "ir_denotation"}[the IR semantics] gives the same result as evaluating the source forward
program.
:::

:::proof "verified_forward_correct"
Induction over {uses "verified_program_language"}[the source program] relates each lowered step to
its matching {uses "ir_denotation"}[IR denotation rule].
:::

:::group "bound_propagation"
Interval and affine certificate soundness.
:::

:::theorem "ibp_local_certificate_sound" (parent := "bound_propagation") (lean := "NN.MLTheory.CROWN.Graph.CertSoundness.cert_encloses_semantics")
A topologically sorted supported graph over {uses "shape_indexed_tensors"}[shape-indexed tensors]
and {uses "scalar_context"}[ordered real scalars] encloses every computed node value when its local
semantic and box certificates are sound.
:::

:::proof "ibp_local_certificate_sound"
Topological induction follows the graph's {uses "shape_indexed_tensors"}[tensor values] and applies
each local enclosure result using the order from {uses "scalar_context"}[the real scalar setting].
:::

:::theorem "ibp_engine_end_to_end" (parent := "bound_propagation") (lean := "NN.MLTheory.CROWN.Graph.CertSoundness.runIBP?_encloses_evalGraphRec")
The concrete real `runIBP?` pass supplies the certificates required by
{uses "ibp_local_certificate_sound"}[the local soundness theorem], so it encloses the recursive
graph semantics whenever it succeeds.
:::

:::proof "ibp_engine_end_to_end"
The executable pass is shown to produce the local certificates required by
{uses "ibp_local_certificate_sound"}[the generic soundness theorem].
:::

:::definition "crown_transfer_contract" (parent := "bound_propagation") (lean := "NN.MLTheory.CROWN.Graph.CrownCertSoundness.CrownTransferSound")
An affine transfer implementation satisfies this contract when each backward transfer preserves
the represented lower and upper bounds.
:::

:::theorem "crown_generic_checker_sound" (parent := "bound_propagation") (lean := "NN.MLTheory.CROWN.Graph.CrownCertSoundness.crown_checker_encloses_semantics")
A locally consistent real affine certificate encloses graph semantics when its transfer step
satisfies {uses "crown_transfer_contract"}[`CrownTransferSound`].
:::

:::proof "crown_generic_checker_sound"
Reverse topological induction composes the certified affine forms and discharges each node with the
{uses "crown_transfer_contract"}[transfer-soundness premise].
:::

:::theorem "alpha_crown_transfer" (parent := "bound_propagation") (lean := "NN.MLTheory.CROWN.Graph.AlphaCrownTransferSoundness.alphaCrown_transfer_sound")
The concrete real α-CROWN transfer step satisfies
{uses "crown_transfer_contract"}[the generic transfer contract].
:::

:::proof "alpha_crown_transfer"
The proof checks the affine relaxation chosen for each supported operation against
{uses "crown_transfer_contract"}[the generic transfer contract].
:::

:::theorem "alpha_beta_crown_transfer" (parent := "bound_propagation") (lean := "NN.MLTheory.CROWN.Graph.AlphaCrownTransferSoundness.alphaBetaCrown_transfer_sound")
The α/β-CROWN transfer step satisfies {uses "crown_transfer_contract"}[the transfer contract].
Unchanged nodes reduce to {uses "alpha_crown_transfer"}[the α-CROWN transfer theorem].
:::

:::proof "alpha_beta_crown_transfer"
Split constraints are handled directly. The remaining operations reuse
{uses "alpha_crown_transfer"}[the α-CROWN transfer proof].
:::

:::definition "alpha_beta_node_checker" (parent := "bound_propagation") (lean := "NN.Verification.CROWNNodeCertAlphaBeta.checkAlphaBetaCROWNNodeCertificate")
The executable checker parses and replays an `IEEE32Exec` node certificate. Its final acceptance
decision has a proved bridge to the proposition-level local replay condition. Connecting that
binary32 condition to the real enclosure in {bpref "crown_generic_checker_sound"}[] still requires
the refinement assumptions for the operations in the graph.
:::

:::theorem "alpha_beta_node_acceptance" (parent := "bound_propagation") (lean := "NN.Verification.CROWNNodeCertAlphaBeta.AlphaBetaCROWNNodeCertificate.accepts_eq_true")
Acceptance of the in-memory α/β-CROWN decision implies `CrownCertLocalOK` for the exact
`IEEE32Exec` replay step used by the checker.
:::

:::proof "alpha_beta_node_acceptance"
The checker compares every dependent affine record bit-for-bit. Soundness of the tensor, matrix,
affine-vector, and optional-record comparisons turns the successful Boolean replay into equality at
every graph node.
:::

:::group "proof_applications"
Selected end-to-end mathematical results and explicit assumptions.
:::

:::definition "lyapunov_certificate_valid" (parent := "proof_applications") (lean := "NN.MLTheory.CROWN.Lyapunov.LyapunovCert.ValidFor")
A Lyapunov certificate is valid for a pair of functions when its two intervals enclose the function
and orbital-derivative values throughout the stated region.
:::

:::theorem "lyapunov_conditions" (parent := "proof_applications") (lean := "NN.MLTheory.CROWN.Lyapunov.Real.lyapunov_conditions")
Certificate thresholds imply positive Lyapunov values and negative derivatives on the certified
region, conditional on {uses "lyapunov_certificate_valid"}[the certificate validity predicate].
:::

:::proof "lyapunov_conditions"
The enclosures in {uses "lyapunov_certificate_valid"}[the validity hypothesis] are compared with
the certificate thresholds and strengthened to strict sign conditions.
:::

:::theorem "gradient_descent_linear_convergence" (parent := "proof_applications") (lean := "Optim.GD.dist_sq_iterate_le_of_q_nonneg")
Under the stated strong-monotonicity, Lipschitz, and step-size hypotheses, the iterates satisfy

$$`\left\lVert \operatorname{step}_{\eta}(g)^{\,k}(x)-x^\star\right\rVert^2
\leq q(\eta,\mu,L)^k\left\lVert x-x^\star\right\rVert^2.`
:::

:::proof "gradient_descent_linear_convergence"
The one-step contraction is iterated, and nonnegativity of $`q(\eta,\mu,L)` controls
multiplication by the geometric factor.
:::

:::theorem "differential_privacy_postprocessing" (parent := "proof_applications") (lean := "NN.MLTheory.LearningTheory.differentialPrivacy_postprocess")
Measurable post-processing preserves $`(\varepsilon,\delta)`-differential privacy.
:::

:::proof "differential_privacy_postprocessing"
The proof rewrites measurable preimages through the post-processing map and reuses the original
privacy inequality.
:::

:::theorem "fp32_relu_approximation_budget" (parent := "proof_applications") (lean := "NN.MLTheory.Proofs.UniversalApproximation.IEEE32ExecTwoLayerMLP.relu_twoLayerMlp_ieee32exec_threeTerm")
Assuming real approximation, parameter quantization, and
{uses "executable_binary32"}[IEEE32 execution] budgets for a two-layer ReLU network, the total
error is bounded by their sum.
:::

:::proof "fp32_relu_approximation_budget"
After interpreting {uses "executable_binary32"}[the IEEE32 result] as a real value, two triangle
inequalities split the target error into the three assumed budgets.
:::

:::theorem "ode_extended_enclosure" (parent := "proof_applications") (lean := "NN.Proofs.Verification.ODE.Enclosure.extendedSolutionEnclosed_fromClampedDynamics")
A comparison argument encloses a clamped scalar ODE solution with constant extension outside the
integration interval.
:::

:::proof "ode_extended_enclosure"
The proof combines the in-interval differential inequality with the two constant-extension cases.
:::

:::theorem "camera_box_checker_sound" (parent := "proof_applications") (lean := "NN.Verification.Geometry3D.Box3D.checkCert_sound")
Acceptance by the boolean 3D box and camera checker over
{uses "shape_indexed_tensors"}[shape-indexed inputs] yields a `Verified3DBox` certificate.
:::

:::proof "camera_box_checker_sound"
Each boolean guard over {uses "shape_indexed_tensors"}[the tensor inputs] is reflected into its
proposition and assembled into the certificate structure.
:::

import Verso
import VersoManual
import VersoBlueprint
import NN.Floats
import NN.Proofs.RuntimeApprox.Graph
import NN.Proofs.RuntimeApprox.NF
import NN.Proofs.RuntimeApprox.Optimizer

open Verso.Genre
open Verso.Genre.Manual
open Informal

#doc (Manual) "Floating-Point Numerics" =>

The numerical map keeps two binary32 models in view. `FP32` is a rounded-real model for proofs;
`IEEE32Exec` computes from raw bits and includes signed zeros, infinities, and NaNs. Bridge theorems
connect their finite cases. The runtime-approximation layer then composes local operator bounds over
whole forward and backward graphs.

:::group "generic_numerics"
Generic formats, rounding, and quantization.
:::

:::definition "radix_float_values" (parent := "generic_numerics") (lean := "TorchLean.Floats.NeuralFloat")
`NeuralFloat` stores an integer mantissa and exponent at a chosen radix. Its real interpretation is
the exact value of that pair.
:::

:::definition "representable_float_grid" (parent := "generic_numerics") (lean := "TorchLean.Floats.neuralGenericFormat")
`neuralGenericFormat` says when a real number belongs to the grid selected by a radix and a valid
exponent policy. Fixed, unbounded, and gradual-underflow formats are instances of this setup.
:::

:::definition "nearest_integer_rounding" (parent := "generic_numerics") (lean := "TorchLean.Floats.NeuralValidRndToNearest")
`NeuralValidRndToNearest` is the contract required of an integer rounding rule: it is monotone,
fixes integers, and stays within one half of its input.
:::

:::definition "generic_grid_rounding" (parent := "generic_numerics") (lean := "TorchLean.Floats.neuralRound")
`neuralRound` scales at the canonical exponent, applies its supplied integer rounding rule, and
maps the resulting {uses "radix_float_values"}[mantissa/exponent pair] back to a real value.
Nearestness and tie handling come from that supplied rule.
:::

:::theorem "nearest_rounding_error" (parent := "generic_numerics") (lean := "TorchLean.Floats.neural_error_bound_ulp")
When its integer rule satisfies {uses "nearest_integer_rounding"}[the nearest-rounding contract],
{uses "generic_grid_rounding"}[generic grid rounding] is within half an ULP.
:::

:::proof "nearest_rounding_error"
The proof applies {uses "nearest_integer_rounding"}[the half-integer error bound] to the scaled
mantissa, then rescales it at the exponent used by {uses "generic_grid_rounding"}[the grid rounder].
:::

:::definition "affine_quantizer" (parent := "generic_numerics") (lean := "TorchLean.Floats.Quantization.AffineQuantizer")
A bounded affine quantizer records its positive scale, zero point, and nonempty integer code range.
Its `quantize` operation accepts the integer rounding rule separately.
:::

:::theorem "affine_quantization_accuracy" (parent := "generic_numerics") (lean := "TorchLean.Floats.Quantization.AffineQuantizer.dequantize_quantize_error_le")
An unclipped value passed through the {uses "affine_quantizer"}[affine quantizer] reconstructs
within half a scale step when its integer rule satisfies
{uses "nearest_integer_rounding"}[the nearest-rounding contract].
:::

:::proof "affine_quantization_accuracy"
The proof applies {uses "nearest_integer_rounding"}[the half-integer error bound] before multiplying
by the positive scale from {uses "affine_quantizer"}[the quantizer].
:::

:::group "binary32_semantics"
Proof-oriented and executable accounts of IEEE 754 binary32.
:::

:::definition "rounded_real_fp32" (parent := "binary32_semantics") (lean := "TorchLean.Floats.FP32")
`FP32` specializes {uses "representable_float_grid"}[the generic format theory] to a proof-oriented
nearest-even model over real values. It leaves out NaNs, infinities, and the upper exponent cutoff,
so claims about those cases belong to `IEEE32Exec`.
:::

:::theorem "fp32_rounding_accuracy" (parent := "binary32_semantics") (lean := "TorchLean.Floats.FP32.round_abs_error")
Rounding in the {uses "rounded_real_fp32"}[rounded-real model] differs from its real input by at
most half an ULP.
:::

:::proof "fp32_rounding_accuracy"
The implementation of {uses "rounded_real_fp32"}[binary32 rounding] reduces to exact integer
arithmetic, and {uses "nearest_rounding_error"}[the generic half-ULP theorem] supplies the error
bound.
:::

:::definition "executable_binary32" (parent := "binary32_semantics") (lean := "TorchLean.Floats.IEEE754.IEEE32Exec")
Raw 32-bit words drive executable addition, multiplication, division, fused multiply-add, and
square root, together with IEEE exception status.
:::

:::theorem "finite_ieee_refinement" (parent := "binary32_semantics") (lean := "TorchLean.Floats.IEEE754.IEEE32Exec.toReal_add_eq_fp32Round_of_isFinite")
On the stated finite-result path, {uses "executable_binary32"}[executable addition] agrees with
{uses "rounded_real_fp32"}[rounded-real binary32 addition]. Matching bridge theorems cover
subtraction, multiplication, fused multiply-add, square root, and division.
:::

:::proof "finite_ieee_refinement"
The proof decodes {uses "executable_binary32"}[finite operands] to dyadics and identifies the
bit-level result with {uses "rounded_real_fp32"}[nearest-even real rounding].
:::

:::theorem "directed_addition_lower_bound" (parent := "binary32_semantics") (lean := "TorchLean.Floats.IEEE754.IEEE32Exec.toEReal_addDown_le")
For finite inputs, downward-rounded {uses "executable_binary32"}[binary32 addition] is no greater
than the exact real sum, including overflow to negative infinity.
:::

:::proof "directed_addition_lower_bound"
The proof decodes the finite {uses "executable_binary32"}[inputs] to dyadics, computes their exact
dyadic sum, and applies soundness of downward rounding in the extended reals.
:::

:::theorem "directed_addition_upper_bound" (parent := "binary32_semantics") (lean := "TorchLean.Floats.IEEE754.IEEE32Exec.toEReal_addUp_ge")
For finite inputs, the exact real sum is no greater than upward-rounded
{uses "executable_binary32"}[binary32 addition], including overflow to positive infinity.
:::

:::proof "directed_addition_upper_bound"
The proof decodes the finite {uses "executable_binary32"}[inputs] to their exact dyadic sum and
applies soundness of upward rounding in the extended reals.
:::

:::group "runtime_error_bounds"
Operator bounds composed across programs.
:::

:::theorem "compositional_forward_approximation" (parent := "runtime_error_bounds") (lean := "Proofs.RuntimeApprox.FwdGraph.eval_approx")
Local approximation contracts over {uses "shape_indexed_tensors"}[typed tensors] compose through
forward graph evaluation.
:::

:::proof "compositional_forward_approximation"
Forward graph induction carries every local contract through
{uses "shape_indexed_tensors"}[the stored typed context].
:::

:::theorem "compositional_reverse_approximation" (parent := "runtime_error_bounds") (lean := "Proofs.RuntimeApprox.RevGraph.backprop_approx")
Given approximate inputs and cotangent seeds, local forward and backward contracts compose into an
approximation bound for every gradient produced by reverse graph evaluation.
:::

:::proof "compositional_reverse_approximation"
Reverse graph induction reuses {uses "compositional_forward_approximation"}[forward composition]
and threads the local backward bounds through the accumulated cotangent context.
:::

:::definition "numerical_optimizer_contract" (parent := "runtime_error_bounds") (lean := "Proofs.RuntimeApprox.Optimizer.NumericalStepContract")
`NumericalStepContract` packages specification and runtime optimizer states, their approximation
relation, an update bound, any domain checks, and the theorem that one update respects the bound.
:::

:::theorem "end_to_end_runtime_error_bounds" (parent := "runtime_error_bounds") (lean := "Proofs.RuntimeApprox.NFBackend.backprop_optimizer_update_approx_graphData")
For one parameter in an already constructed typed reverse graph,
{uses "compositional_reverse_approximation"}[the reverse approximation theorem] and a
{uses "numerical_optimizer_contract"}[valid optimizer contract] carry the stated input, seed,
parameter, state, and step-data bounds through backpropagation and one optimizer update.
:::

:::proof "end_to_end_runtime_error_bounds"
The proof obtains the indexed gradient bound from
{uses "compositional_reverse_approximation"}[reverse composition], then applies the update-soundness
field of {uses "numerical_optimizer_contract"}[the supplied optimizer contract].
:::

:::definition "checked_numerical_certificate" (parent := "runtime_error_bounds") (lean := "Proofs.RuntimeApprox.NumericalCertificate.CheckedCertificate")
A checked numerical certificate retains the submitted artifact together with the canonical source
ranges, node-range trace, accepted backend plan, and proofs that the recomputed trace and audit
match the artifact.
:::

:::definition "proved_real_enclosure_trace" (parent := "runtime_error_bounds") (lean := "Proofs.RuntimeApprox.NumericalCertificate.CheckedRealExecution")
For one {uses "checked_numerical_certificate"}[checked certificate], `CheckedRealExecution` stores a
real payload and input, the complete {uses "ir_denotation"}[IR execution trace], and a proof that
each real node value lies in its checked interval.
:::

:::theorem "checked_numerical_execution" (parent := "runtime_error_bounds") (lean := "Proofs.RuntimeApprox.NumericalCertificate.CheckedExecution.errorTrace")
A checked IEEE replay plus a separately supplied
{uses "proved_real_enclosure_trace"}[real-execution enclosure] for the same
{uses "checked_numerical_certificate"}[certificate] yields a graph-wide pointwise error trace whose
budget at each node is the width of its checked interval.
:::

:::proof "checked_numerical_execution"
The proof combines the IEEE replay's range check with
{uses "proved_real_enclosure_trace"}[the supplied real enclosure], node by node.
:::

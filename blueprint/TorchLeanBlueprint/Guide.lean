import VersoManual
import TorchLeanBlueprint.Guide.Ch1_Introduction.Overview
import TorchLeanBlueprint.Guide.Ch1_Introduction.Motivation
import TorchLeanBlueprint.Guide.Ch1_Introduction.API_Tour
import TorchLeanBlueprint.Guide.Ch1_Introduction.WhyFunctionalProgramming
import TorchLeanBlueprint.Guide.Ch1_Introduction.TheLeanLanguage
import TorchLeanBlueprint.Guide.Ch1_Introduction.TorchLeanVsPyTorch
import TorchLeanBlueprint.Guide.Ch1_Introduction.RunningExample
import TorchLeanBlueprint.Guide.Ch2_Frontend.TensorsAndShapes
import TorchLeanBlueprint.Guide.Ch2_Frontend.BuildingModels
import TorchLeanBlueprint.Guide.Ch2_Frontend.DataAndLoaders
import TorchLeanBlueprint.Guide.Ch2_Frontend.TrainingFromScratch
import TorchLeanBlueprint.Guide.Ch2_Frontend.TorchLeanAPI
import TorchLeanBlueprint.Guide.Ch2_Frontend.ExecutionModes
import TorchLeanBlueprint.Guide.Ch2_Frontend.BackendSelection
import TorchLeanBlueprint.Guide.Ch2_Frontend.AutogradWalkthrough
import TorchLeanBlueprint.Guide.Ch2_Frontend.ScientificForwardModels
import TorchLeanBlueprint.Guide.Ch2_Frontend.RuntimeAndAutograd
import TorchLeanBlueprint.Guide.Ch2_Frontend.PyTorchRoundtrip
import TorchLeanBlueprint.Guide.Ch3_Backend.GraphsAndIR
import TorchLeanBlueprint.Guide.Ch3_Backend.SpecLayer
import TorchLeanBlueprint.Guide.Ch3_Backend.GraphSpec
import TorchLeanBlueprint.Guide.Ch3_Backend.Floats
import TorchLeanBlueprint.Guide.Ch3_Backend.FloatingPointLiterature
import TorchLeanBlueprint.Guide.Ch3_Backend.GPUAndCUDA
import TorchLeanBlueprint.Guide.Ch3_Backend.ExternalToolsAndFFI
import TorchLeanBlueprint.Guide.Ch4_Verification.Verification
import TorchLeanBlueprint.Guide.Ch4_Verification.ProofSystems
import TorchLeanBlueprint.Guide.Ch4_Verification.AutogradProofs
import TorchLeanBlueprint.Guide.Ch4_Verification.RuntimeApproximation
import TorchLeanBlueprint.Guide.Ch4_Verification.LearningTheory
import TorchLeanBlueprint.Guide.Ch4_Verification.OptimizationTheory
import TorchLeanBlueprint.Guide.Ch4_Verification.SelfSupervisedTheory
import TorchLeanBlueprint.Guide.Ch4_Verification.ApproximationTheory
import TorchLeanBlueprint.Guide.Ch4_Verification.ClassicalMLProofs
import TorchLeanBlueprint.Guide.Ch4_Verification.ProbabilityAndGradients
import TorchLeanBlueprint.Guide.Ch4_Verification.ScientificMLVerification
import TorchLeanBlueprint.Guide.Ch4_Verification.FactorizationsCholeskyQR
import TorchLeanBlueprint.Guide.Ch4_Verification.Certificates
import TorchLeanBlueprint.Guide.Ch4_Verification.FP32Soundness
import TorchLeanBlueprint.Guide.Ch4_Verification.TwoStageWorkflows
import TorchLeanBlueprint.Guide.Ch5_Applications.ModernModels
import TorchLeanBlueprint.Guide.Ch5_Applications.ModelZooDeepDive
import TorchLeanBlueprint.Guide.Ch5_Applications.GenerativeModels
import TorchLeanBlueprint.Guide.Ch5_Applications.ReinforcementLearning
import TorchLeanBlueprint.Guide.Ch5_Applications.Widgets
import TorchLeanBlueprint.Guide.Ch5_Applications.BugZooCatalog
import TorchLeanBlueprint.Guide.Ch5_Applications.Examples
import TorchLeanBlueprint.Guide.Ch5_Applications.CLI
import TorchLeanBlueprint.Guide.Ch6_Conclusion.Conclusion

open Verso.Genre Manual

#doc (Manual) "TorchLean" =>
%%%
shortTitle := "TorchLean"
tag := "torchlean"
%%%

TorchLean is a Lean 4 library for writing, training, and verifying neural networks. The quickest
way to understand it is to follow one prediction through the system. A pair of input features enters
a shape-typed model, concrete parameters turn that model into a program, the runtime records the
operations needed for differentiation, and a graph records the computation so that preservation and
verification questions can be stated explicitly.
The value may travel through several representations, but none of those handoffs is meant to be
invisible.

Our companion for that trip is a small nonlinear regression network. At first it looks pleasantly
ordinary: two linear layers with a ReLU between them. We will train it, inspect its parameter shapes,
watch a backward pass accumulate gradients, move selected operations to CUDA, and finally ask what
can be proved about its outputs. By returning to the same model, the guide can explain why a theorem
about an exact real-valued specification is relevant to a particular floating-point GPU run without
automatically proving that run correct.

Once that path feels familiar, the larger examples are variations on the same theme. Transformers
add token and attention structure; ResNets add spatial layouts and skip connections; Fourier neural
operators work with sampled functions; diffusion and reinforcement learning make randomness and
state explicit. The numerical chapters supply generic formats, executable binary32 arithmetic, and
error bounds. The verification chapters build interval, affine, compiler, autograd, and certificate
arguments on top of those definitions.

One reading habit matters throughout: ask what kind of evidence is attached to a claim. An
*implemented* path can be executed. A *tested* path has a regression or conformance check. A
*proved* statement has a Lean theorem with the hypotheses visible in its type. A *planned* feature
is only future work. The guide includes successful runs and deliberately rejected inputs so that
these differences can be observed rather than guessed from an impressive module name.

The examples are meant to be run from the repository root. No theorem-proving background is needed
to begin. Readers new to Lean may also use
[*Functional Programming in Lean*](https://lean-lang.org/functional_programming_in_lean/),
[*Theorem Proving in Lean 4*](https://lean-lang.org/theorem_proving_in_lean4/), and
[*The Lean Language Reference*](https://lean-lang.org/doc/reference/latest/).

# Introduction

Before touching an optimizer, we need a reason to put a neural network in a proof assistant at all.
We begin with a prediction whose surrounding claim is unclear, learn just enough Lean to read the
code, and then write the regression model that will stay with us through the rest of the book.

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.Overview}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.Motivation}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.API_Tour}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.WhyFunctionalProgramming}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.TheLeanLanguage}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.TorchLeanVsPyTorch}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.RunningExample}


# Building Models

An architecture is still only a recipe. It cannot make a prediction until shapes meet data and a
seed produces parameters. Here we turn the model into a small training program, one ordinary step
at a time: load a batch, evaluate the forward map, measure a loss, run backward, and update the
parameters. Each step leaves an object we can inspect rather than hiding the whole loop behind a
single call.

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.TensorsAndShapes}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.BuildingModels}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.DataAndLoaders}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.TrainingFromScratch}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.TorchLeanAPI}


# Runtime, Autograd, and Interop

Training introduces state and hardware. Parameters change, tape nodes save intermediate values, and
optimizer buffers accumulate history. None of that appears in the clean equation

$$`f_\theta(x)=W_2\,\operatorname{ReLU}(W_1x+b_1)+b_2`.

We will follow one step through the runtime, then move the same operation to compiled execution,
CUDA, and LibTorch.

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.ExecutionModes}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.BackendSelection}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.AutogradWalkthrough}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.ScientificForwardModels}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.RuntimeAndAutograd}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.PyTorchRoundtrip}


# Semantics and Graphs

Autograd records one execution, and then that tape can be released. A verification claim must
survive longer. We first recover the model as a mathematical function, then preserve its
architecture in `GraphSpec`, and finally lower it to the ordinary node array used by importers and
verification passes. This is where “the same model” stops being a slogan and becomes a sequence of
explicit correspondence questions.

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.SpecLayer}

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.GraphSpec}

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.GraphsAndIR}


# Floating Point and Native Boundaries

Our graph equations use real numbers, but the program does not. This part starts with that mismatch
and develops TorchLean's floating-point stack. The story begins with Flocq's influential separation
of formats from rounding, continues through TorchLean's generic `NeuralFloat` theory, and ends with
executable binary32 and native kernels.

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.Floats}

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.FloatingPointLiterature}

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.GPUAndCUDA}

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.ExternalToolsAndFFI}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.FP32Soundness}


# Verification and Certificates

We can now state the verification question precisely:

$$`\forall x\in B,\qquad P(\operatorname{denote}(g,\theta,x))`.

This part develops several ways to answer it: interval and affine bounds, compiler-correctness
proofs, autograd theorems, numerical error bounds, optimizer laws, and replayed certificates. We
will also feed bad artifacts to the checkers and see exactly where they fail.

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.Verification}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.ProofSystems}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.AutogradProofs}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.RuntimeApproximation}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.LearningTheory}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.OptimizationTheory}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.SelfSupervisedTheory}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.ApproximationTheory}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.ClassicalMLProofs}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.ProbabilityAndGradients}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.ScientificMLVerification}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.FactorizationsCholeskyQR}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.Certificates}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.TwoStageWorkflows}


# Examples and Applications

Finally we ask whether the machinery still feels natural after leaving the two-layer MLP. A ResNet
introduces spatial layouts and skip connections. GPT turns text into token streams and causal
masks. A Fourier neural operator connects a learned map to PDE data. Diffusion and reinforcement
learning force us to keep randomness and state in view. The examples are organized as runs a reader
can reproduce: what to execute, what artifact appears, what to inspect next, and which claim that
artifact supports and where that support stops.

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.ModernModels}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.ModelZooDeepDive}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.GenerativeModels}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.ReinforcementLearning}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.Widgets}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.BugZooCatalog}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.Examples}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.CLI}

{include 2 TorchLeanBlueprint.Guide.Ch6_Conclusion.Conclusion}

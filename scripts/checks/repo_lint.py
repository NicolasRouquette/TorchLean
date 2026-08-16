#!/usr/bin/env python3
"""
TorchLean repo lints (project-specific).

This linter stays dependency-free so it can run in CI and locally.

Checks are split into:
  - errors: must be fixed (fail CI)
  - warnings: reported for visibility (do not fail by default)
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import urllib.parse
from dataclasses import dataclass
from typing import Iterable


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
LINT_SCOPE_SENTINEL = REPO_ROOT / "NN/MLTheory/CROWN/Lyapunov/Certificate.lean"

# External trees that may exist in a developer checkout but are not part of TorchLean's core sources
# and must not affect repo policy/CI. These are user-cloned repos outside TorchLean's source tree.
#
# Important: `Path.rglob` walks the filesystem, not git-tracked files, so this linter explicitly skips
# these directories to ensure `lake lint` does not start depending on optional checkouts.
VENDORED_DIR_NAMES = {
    "Two-Stage_Neural_Controller_Training",  # optional external checkout (α,β-CROWN workflows)
    "PINN_verification",  # user-cloned external repo (gitignored)
}

# Keep the trusted boundary explicit: axioms must be quarantined, named, and documented.
ALLOWED_AXIOMS = {
    "NN/Runtime/Autograd/Engine/Cuda/Trusted.lean": {"instNonemptyBuffer"},
}

# Narrow allowlist for linter suppressions that are noisy in API entrypoint files but do
# not weaken proofs. Keep this list small and review each addition.
ALLOWED_LINTER_SUPPRESSION_FILES = {
    "NN/Tensor/API.lean",
}

# These modules were pure compatibility routes or duplicate import surfaces. New code must use the
# canonical subsystem umbrellas and namespaces instead of recreating them.
REMOVED_COMPATIBILITY_PATHS = {
    "NN/API/TorchLean/Optimizers.lean",
    "NN/Examples/Verification/LiRPA.lean",
    "NN/GraphSpec/Models/TorchLean/Fno1d.lean",
    "NN/Library.lean",
    "NN/API/TorchLean/Trainer/Verify.lean",
    "NN/API/TorchLean/Data/DotInfo.lean",
    "NN/MLTheory/CROWN/Lyapunov/Oracle.lean",
    "NN/MLTheory/CROWN/Tactics/CrownOracle.lean",
    "NN/Spec/Layers/Pooling/Aliases.lean",
    "NN/Verification/TorchLean/Verified.lean",
}

REMOVED_COMPATIBILITY_PREFIXES = (
    "NN/Entrypoint/",
)

# Documentation may mention producer-side environment variables only when the implementation hook
# exists in source. This prevents guide text from advertising phantom integration flags.
DOCUMENTED_ENV_VAR_IMPLEMENTATIONS = {
    "ABCROWN_ARTIFACT_OUT": "scripts/verification/abcrown/export_leaf_artifact.py",
}

TRUST_BOUNDARY_DECL_REFS = {
    "NN.MLTheory.CROWN.Graph.CrownCertSoundness.CrownTransferSound": (
        "NN/MLTheory/CROWN/Proofs/GraphCrownCertSoundness.lean",
        re.compile(r"\bdef\s+CrownTransferSound\b"),
    ),
    "NN.MLTheory.Proofs.UniversalApproximation.FloatIntervalApprox.OpsExact.Sound": (
        "NN/MLTheory/Proofs/Approximation/FloatInterval/Semantics.lean",
        re.compile(r"\bclass\s+Sound\s*:\s*Prop\b"),
    ),
}

DOC_FACT_BANNED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"leaf (?:artifact )?JSON format exported from (?:α,β-CROWN|alpha-beta-CROWN)",
            flags=re.IGNORECASE,
        ),
        "TorchLean alpha-beta-CROWN leaf JSON is produced by TorchLean's converter from raw terminal-domain data; do not imply the external verifier natively exports the TorchLean schema.",
    ),
    (
        re.compile(
            r"Set\s+`?ABCROWN_ARTIFACT_OUT`?.{0,80}(?:before running|when running)\s+(?:α,β-CROWN|alpha-beta-CROWN)",
            flags=re.IGNORECASE,
        ),
        "`ABCROWN_ARTIFACT_OUT` belongs to TorchLean's exporter/helper boundary around alpha-beta-CROWN.",
    ),
    (
        re.compile(
            r"(?:external tool|alpha-beta-CROWN)\s+writes\s+the\s+JSON",
            flags=re.IGNORECASE,
        ),
        "TorchLean's alpha-beta-CROWN JSON schema is written by the TorchLean exporter/helper, not vanilla alpha-beta-CROWN.",
    ),
    (
        re.compile(
            r"Two-Stage tooling can emit a small JSON \*leaf\s+certificate\*",
            flags=re.IGNORECASE,
        ),
        "Say that an instrumented external verifier exposes terminal domains and TorchLean's helper converts them; do not imply the external Two-Stage tooling natively emits TorchLean's schema.",
    ),
    (
        re.compile(
            r"External JSON artifacts are treated as untrusted\.\s+Checkers parse them, validate shapes, and compare\s+them against Lean recomputation",
            flags=re.IGNORECASE,
        ),
        "Not every JSON artifact is recomputed in Lean; distinguish structural checks, recomputation checks, and theorem-backed checks.",
    ),
    (
        re.compile(
            r"A CUDA run proves that the CUDA path executed",
            flags=re.IGNORECASE,
        ),
        "Reserve `prove` for Lean/checker claims; a CUDA run is runtime evidence, not a proof.",
    ),
    (
        re.compile(
            r"verified reverse mode autograd",
            flags=re.IGNORECASE,
        ),
        "Do not imply all reverse-mode autograd is verified; say selected reverse-mode/autograd proofs.",
    ),
    (
        re.compile(
            r"bundled (?:α,β-CROWN|alpha-beta-CROWN) leaf certificate",
            flags=re.IGNORECASE,
        ),
        "The bundled alpha-beta-CROWN file is a structural leaf artifact, not a proof-backed certificate.",
    ),
    (
        re.compile(
            r"External (?:α,β-CROWN|alpha-beta-CROWN) artifact.*JSON leaf certificate",
            flags=re.IGNORECASE,
        ),
        "Call the alpha-beta-CROWN import a JSON leaf artifact unless the computation is replayed or proof-backed.",
    ),
    (
        re.compile(
            r"\bleaf_cert\.json\b|<cert\.json>|Output certificate path",
            flags=re.IGNORECASE,
        ),
        "Alpha-beta-CROWN-facing docs and CLI help should use `leaf_artifact.json` / artifact wording; the checker is structural unless a separate proof/replay path is named.",
    ),
    (
        re.compile(
            r"Autograd correctness.*backprop computes the adjoint derivative",
            flags=re.IGNORECASE,
        ),
        "Autograd correctness claims must name the supported tape node or graph fragment.",
    ),
    (
        re.compile(
            r"JSON \*leaf\s+certificate\*",
            flags=re.IGNORECASE,
        ),
        "Use `leaf artifact` for alpha-beta-CROWN structural JSON unless the computation is replayed or proof-backed.",
    ),
    (
        re.compile(
            r"abcrown-leaf` to check a JSON certif(?:icate)? against TorchLean's semantics",
            flags=re.IGNORECASE,
        ),
        "Use artifact wording: abcrown-leaf checks a converted structural leaf artifact plus a local witness predicate at the TorchLean boundary.",
    ),
]

TORCHLEAN_SOURCE_LINK_RE = re.compile(
    r"https://github\.com/lean-dojo/TorchLean/blob/main/([^\s\)\]`]+)"
)

DOCGEN_API_LINK_RE = re.compile(
    r"""(?:['"(])(/docs/[A-Za-z0-9_./-]+\.html(?:#[A-Za-z0-9_'.:-]+)?)"""
)

LOCAL_SOURCE_REF_RE = re.compile(
    r"`((?:NN|blueprint|home_page|scripts|csrc)/[^`\s]+"
    r"\.(?:lean|md|py|json|sh|cu|c|h|yml|yaml))`"
)

LEAN_DOC_COMMENT_RE = re.compile(r"/-(?:!|-).*?-/", flags=re.DOTALL)
# A single backslash starts the delimiters that MD4Lean does not recognize.
# The negative lookbehind leaves TeX line breaks such as `\\[1ex]` alone.
DOCGEN_UNSUPPORTED_MATH_DELIMITER_RE = re.compile(r"(?<!\\)\\[\(\[]")
DOCGEN_DISPLAY_MATH_RE = re.compile(r"\$\$(.*?)\$\$", flags=re.DOTALL)
DOCGEN_MARKDOWN_LIST_IN_DISPLAY_MATH_RE = re.compile(
    r"^[ \t]*(?:[-+*]|\d+[.)])[ \t]+",
    flags=re.MULTILINE,
)
VERSO_TEX_IN_ORDINARY_CODE_RE = re.compile(
    r"(?<!\$)`[^`\n]*(?:\\[A-Za-z]+|_\{[^}`]+\})[^`\n]*`"
)
FORMALIZATION_MATH_IN_ORDINARY_CODE_RE = re.compile(
    r"(?<!\$)`[^`\n]*(?:\s[\^+*/<>]=?\s|≤|≥|±)[^`\n]*`"
)

PUBLIC_EXAMPLE_PREFIXES = (
    "NN/Examples/Quickstart/",
    "NN/Examples/Models/",
    "NN/Examples/Data/",
)

PUBLIC_TUTORIAL_PREFIXES = (
    "NN/Examples/Quickstart/",
    "NN/Examples/Data/",
)

PUBLIC_GUIDE_PREFIXES = (
    "README.md",
    "NN/Examples/README.md",
    "NN/Examples/Quickstart/README.md",
    "NN/Examples/Models/README.md",
    "blueprint/TorchLeanBlueprint/Guide/Ch1_Introduction/",
    "blueprint/TorchLeanBlueprint/Guide/Ch2_Frontend/",
    "blueprint/TorchLeanBlueprint/Guide/Ch5_Applications/",
)

PUBLIC_GUIDE_BANNED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\bTrainer\.NewConfig\b|\bNewConfig\b"),
        "public guides should use `Trainer.Config`; `Trainer.NewConfig` was removed during the unified Trainer cleanup.",
    ),
    (
        re.compile(r"\bTrainer\.(regression|classifier|crossEntropy|custom)\b"),
        "public guides should use `Trainer.new ... { task := ... }`; specialized `Trainer.*` constructors are removed.",
    ),
    (
        re.compile(r"\btrainer\.fit\b"),
        "public guides should call `trainer.train`; do not reintroduce the old `fit` public API.",
    ),
    (
        re.compile(r"\bIO\.println\s+(report|fit)\.summary\b"),
        "public guides should use `trained.printSummary` / `report.printSummary` instead of printing `.summary` directly.",
    ),
    (
        re.compile(r"\blet\s+report\s+←\s+trainer\.train\b"),
        "public guides should call the trained handle `trained`; `trainer.train` returns a reusable trained object, not just a report.",
    ),
    (
        re.compile(r"\bRuntimeFit\b|\bparseRuntimeFit\b|\bparsed\.fit\b"),
        "quickstart docs should use `RuntimeTrain`, `parseRuntimeTrain`, and `parsed.trainOptions`.",
    ),
    (
        re.compile(r"\bfitOptionsWhenLogRequested\b|\bfitOptions\b"),
        "public guides should use `trainOptions` terminology, not old `fitOptions` spellings.",
    ),
    (
        re.compile(r"\bTrainer\.FitOptions\b"),
        "`Trainer.FitOptions` was removed; public guides should name `Trainer.TrainOptions`.",
    ),
    (
        re.compile(r"\bTrainer\.FitSummary\b"),
        "`Trainer.FitSummary` was removed; public guides should use `Trainer.TrainSummary`.",
    ),
    (
        re.compile(
            r"\bfitCsvRegression\b|\brunCsvRegressionTrain\b|\bfitNpyRegression\b|"
            r"\brunCifar(Classifier|Regression|Curve)Train\b|"
            r"\brun(RegressionCsv|ClassificationNpy|RegressionNpy|ForecastWindow)\b"
        ),
        "public guides should show `Trainer.new` / `trainer.train`, not removed command-wrapper names.",
    ),
    (
        re.compile(r"\bModelZoo\.Command\b|\bTrainer\.Command\b|\bTrainCommand\.run\b"),
        "public guides should teach `Trainer.new` / `trainer.train`; repository command glue belongs in examples.",
    ),
    (
        re.compile(r"\btrain\.\*"),
        "public guides should teach the `Trainer` API, not `train.*`.",
    ),
    (
        re.compile(r"\btrain\.stepEpochLR\b"),
        "public guides should say `Trainer.stepEpochLR`, not `train.stepEpochLR`.",
    ),
    (
        re.compile(r"\bNN\.API\.nn\b"),
        "public guides should use the canonical `TorchLean.nn` namespace.",
    ),
    (
        re.compile(r"\bNN\.API\.Models\.TrainFixed\b"),
        "fixed-sample training now lives at `TorchLean.Trainer.FixedSample`.",
    ),
]

PUBLIC_EXAMPLE_BANNED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\bNN\.API\.nn\b"),
        "public examples should use the canonical `TorchLean.nn` namespace.",
    ),
    (
        re.compile(r"\bNN\.API\.Models\.TrainFixed\b"),
        "public examples should use `TorchLean.Trainer.FixedSample`.",
    ),
    (
        re.compile(r"\bsample\.Supervised\b"),
        "public examples should use `SupervisedSample`, not the internal `sample.Supervised` spelling.",
    ),
    (
        re.compile(r"_root_\.NN\.API\.sample\.Supervised\b"),
        "public examples should use `SupervisedSample`, not the fully-qualified internal sample type.",
    ),
    (
        re.compile(r"\bSample\.Supervised\b"),
        "public examples should use `SupervisedSample` as the sample type; reserve `Sample.*` for constructors/accessors.",
    ),
    (
        re.compile(r"\bsample\.mk\b"),
        "public examples should use `Sample.mk`, not the internal `sample.mk` spelling.",
    ),
    (
        re.compile(r"\bnn\.sequential!\b"),
        "public examples should use `nn.Sequential!`, not the lowercase macro spelling.",
    ),
    (
        re.compile(r"\bnn\.summary\b"),
        "public examples should print `model.info`; do not introduce a second model-summary spelling.",
    ),
    (
        re.compile(r"\bShape\.(Vec|Mat|Image|Images|NCHW)\b"),
        "public examples should use the canonical lowercase shape helpers: `Shape.vec`, `Shape.mat`, `Shape.image`, `Shape.images`, `Shape.nchw`.",
    ),
    (
        re.compile(r"\bSemantics\.Scalar\b"),
        "public examples should use `Runtime.SemanticScalar`, not the lower internal `Semantics.Scalar` spelling.",
    ),
    (
        re.compile(r"\bTaskRunner\b"),
        "public examples should not expose `TaskRunner`; use `Trainer`/`Module` helpers instead.",
    ),
    (
        re.compile(r"\bTrainer\.Manual\.trainLoaderWith\b"),
        "public examples should use `Trainer.RunConfig` + `Trainer.TrainOptions` with `trainer.train`, not `Trainer.Manual.trainLoaderWith`.",
    ),
    (
        re.compile(r"\bTrainer\.Manual\.logLossEvery\b"),
        "public examples should keep logging inline or use `Trainer.TrainSummary`; do not expose `Trainer.Manual.logLossEvery`.",
    ),
    (
        re.compile(r"\bfitWithParams\b"),
        "public examples should prefer the public trainer/verifier bridges instead of reopening raw post-training parameter callbacks.",
    ),
    (
        re.compile(r"\bTList\b"),
        "public examples should not expose `TList`; use model/trainer/checkpoint helpers instead.",
    ),
    (
        re.compile(r"\btlist\.TList\b|\btlist!\b"),
        "public examples should not expose raw `tlist` packs; use public tensor/model helpers instead.",
    ),
    (
        re.compile(r"\bModule\.instantiateConfigured\b"),
        "public examples should use `Module.instantiate` or the `Trainer` API, not `Module.instantiateConfigured` directly.",
    ),
    (
        re.compile(r"\bTorchLean\.Module\.run\b"),
        "public examples should use the `Trainer` API or `Module.Command.run`, not a removed raw `TorchLean.Module.run` dispatcher.",
    ),
    (
        re.compile(r"\bModule\.(lossValue|optimizerStep)\b"),
        "public model/example training should use trained handles (`trained.predict`, callbacks, or `verify`), not raw module stepping.",
    ),
    (
        re.compile(
            r"\bRealData\.fit(CifarClassifierModel|CifarRegressionModel|CsvRegressionModel|HouseholdPowerRegressionModel)\b"
        ),
        "public model-zoo examples should use the shared example `TrainCommand` runners, not the old `*Model` wrappers.",
    ),
    (
        re.compile(r"\bTrainer\.NewConfig\b|\bNewConfig\b"),
        "public examples should use `Trainer.Config`; `Trainer.NewConfig` was removed during the unified Trainer cleanup.",
    ),
    (
        re.compile(
            r"\bfitCsvRegression\b|\brunCsvRegressionTrain\b|\bfitNpyRegression\b|"
            r"\brunCifar(Classifier|Regression|Curve)Train\b|"
            r"\brun(RegressionCsv|ClassificationNpy|RegressionNpy|ForecastWindow)\b"
        ),
        "public examples should use `Trainer.new` / `trainer.train` or the shared example `TrainCommand` runners, not removed command-wrapper names.",
    ),
    (
        re.compile(r"\bModelZoo\.Command\b|\bTrainer\.Command\b"),
        "repository command glue belongs under `NN.Examples.Models.TrainCommand`, outside the public Trainer namespace.",
    ),
    (
        re.compile(r"\bSimpleText\.main\b"),
        "shared sequence-model code should expose one executable entrypoint, not nested `*.main` actions.",
    ),
    (
        re.compile(r"\.verify\s*\(\s*Trainer\.Verify\.lInfIBP\b"),
        "public examples should prefer `trained.verifyRobustLInf x eps` over manually building a `Trainer.Verify.lInfIBP` request.",
    ),
    (
        re.compile(r"\bTrainer\.(FitOptions|TrainOptions)\.forSteps\b"),
        "public examples should prefer record literals such as `{ steps := n }`, which match the trainer.train API shown in quickstarts.",
    ),
    (
        re.compile(r"\bTrainer\.FitOptions\b"),
        "`Trainer.FitOptions` was removed; public examples should name `Trainer.TrainOptions`.",
    ),
    (
        re.compile(r"\bTrainer\.FitSummary\b"),
        "`Trainer.FitSummary` was removed; public examples should use `Trainer.TrainSummary`.",
    ),
    (
        re.compile(r"\bTrainer\.(regression|classifier|crossEntropy|custom)\b"),
        "public examples should use `Trainer.new ... { task := ... }`; specialized `Trainer.*` constructors are removed.",
    ),
    (
        re.compile(r"\bTrainer\.(Regression|Classifier|OneHotCrossEntropy|Custom)(\.|\b)"),
        "public examples should stay on the unified `Trainer` API, not specialized trainer implementation handles.",
    ),
    (
        re.compile(r"\bstructure\s+RunConfig\b"),
        "public examples should not define their own `RunConfig`; reserve that name for `Trainer.RunConfig` and use domain-specific option names.",
    ),
    (
        re.compile(r"\btrainer\.fit\b"),
        "public examples should call `trainer.train`; `trained` is the conventional local name for the trained result.",
    ),
    (
        re.compile(r"\bRuntimeFit\b|\bparseRuntimeFit\b|\bparsed\.fit\b"),
        "quickstart examples should use `RuntimeTrain`, `parseRuntimeTrain`, and `parsed.trainOptions`.",
    ),
    (
        re.compile(r"\bfitOptionsWhenLogRequested\b|\bfitOptions\b"),
        "public examples should use `trainOptions` terminology, not old `fitOptions` spellings.",
    ),
    (
        re.compile(r"\.fit(StreamFloat|PairStreamFloat|SelectedCrossEntropy)\b|\bfit(StreamFloat|PairStreamFloat|SelectedCrossEntropy)\b"),
        "public examples should use the `train*` trainer methods, not old stream/selected-training helpers.",
    ),
    (
        re.compile(r"\(\s*\{[^}]*optimizer\s*:=.*\}\s*:\s*Trainer\.RunConfig\s*\)\.withOptions\s+opts"),
        "public examples should use `Trainer.RunConfig.ofRuntimeOptions opts { optimizer := ... }`, not a type-ascribed RunConfig followed by `.withRuntimeOptions opts`.",
    ),
    (
        re.compile(
            r"\b(execution := opts\.execution|device := opts\.device|"
            r"backendProfile\? := opts\.backendProfile\?|device := if opts\.usesCuda|"
            r"fastKernels := opts\.fastKernels|"
            r"fastGpuMatmulPrecision := opts\.fastGpuMatmulPrecision)\b"
        ),
        "public examples should use `Trainer.RunConfig.ofRuntimeOptions opts { ... }` instead of manually copying runtime fields from `opts`.",
    ),
    (
        re.compile(r"\bnn\.(mseScalarModuleDef|crossEntropyOneHotScalarModuleDef)\b"),
        "public examples should use public `Module.instantiate*` helpers instead of spelling raw objective definitions.",
    ),
    (
        re.compile(r"\bfun\s+\{α\}"),
        "public examples should avoid raw polymorphic runtime callbacks in user-facing code.",
    ),
    (
        re.compile(r"^\s*open\s+NN\.API\b", flags=re.MULTILINE),
        "public examples should `open TorchLean`, not `open NN.API`.",
    ),
    (
        re.compile(r"^\s*(public\s+)?import\s+NN\s*$", flags=re.MULTILINE),
        "public examples should import the focused `NN.API`, not the complete `NN` umbrella.",
    ),
    (
        re.compile(r"\bNN\.API\."),
        "public examples should go through the `TorchLean` API, not fully-qualified `NN.API.*` implementation paths.",
    ),
    (
        re.compile(r"IO\.println\s+\"model\s*="),
        "public examples should print `model.info`, not a hardcoded model banner.",
    ),
    (
        re.compile(r"\bIO\.println\s+trainer\.info\b"),
        "public examples should use `trainer.printInfo` so model-summary formatting stays consistent.",
    ),
    (
        re.compile(r"\bIO\.println\s+\w*Trainer\.info\b"),
        "public examples should use `trainer.printInfo` / `trainer.printInfoAs`, not direct trainer-info printing.",
    ),
    (
        re.compile(r"\bIO\.println\s+(report|fit)\.summary\b"),
        "public examples should use `report.printSummary` / `trained.printSummary` for trained results.",
    ),
    (
        re.compile(r"\blet\s+report\s+←\s+(trainer\.train|train\s+opts\s+flags)\b"),
        "public examples should call trained results `trained`, not `report`; `trainer.train` returns a trained handle, not just a summary.",
    ),
    (
        re.compile(r"\bfit\.fit\.predict(Batch)?\b"),
        "public stream examples should use `trained.predict` / `trained.predictMany`; do not expose internal training state.",
    ),
    (
        re.compile(r"\bfit\.curve\.values\b"),
        "public paired-stream examples should use `trained.printCurveSummary` for before/after curve summaries.",
    ),
    (
        re.compile(r"\bIO\.println\s+cert\.summary\b"),
        "public examples should use `cert.printSummary` for verification reports.",
    ),
    (
        re.compile(r"\bTrainer\.FitSummary\.parseFloat\?\b"),
        "public examples should use `Trainer.TrainSummary.printFloatLosses` when numeric losses are required, not hand-rolled optional parsing.",
    ),
    (
        re.compile(r"\bTrainer\.FitSummary\.requireFloatLosses\b"),
        "public examples should use `Trainer.TrainSummary.printFloatLosses` when they need numeric losses for logs.",
    ),
]

PUBLIC_TUTORIAL_BANNED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\btrainer\.trainClassifier\b"),
        "public examples should batch classifier datasets with `Data.batchDataset` and call ordinary `trainer.train`; `trainer.trainClassifier` was removed.",
    ),
    (
        re.compile(r"\btrainClassifierWithFlags\b"),
        "public examples should batch classifier datasets with `Data.batchDataset` and call ordinary `trainer.train`; classifier-specific trainer loops were removed.",
    ),
]

TOP_LEVEL_API_DECL_RE = re.compile(
    r"^\s*(def|structure|inductive|class|abbrev|instance|theorem|lemma)\s+",
    flags=re.MULTILINE,
)

PUBLIC_DECL_RE = re.compile(
    r"^\s*(?:public\s+)?(?:def|structure|inductive|class|abbrev)\s+"
    r"(?P<name>[A-Za-z0-9_'.]+)\b",
    flags=re.MULTILINE,
)

# Public tensor and model APIs describe axes through shapes and vectors. Layout spellings and fixed
# spatial ranks belong in low-level kernels or domain examples, not in user-facing declaration names.
PUBLIC_LAYOUT_NAME_RE = re.compile(
    r"(?:chw|nchw|(?:^|_)[123]d(?:_|$)|[a-z][123]d$)",
    flags=re.IGNORECASE,
)

CONTRACT_SOURCE_FILE_RE = re.compile(
    r"\.sourceFile\s*\{(?P<body>[^{}]*)\}",
    flags=re.DOTALL,
)
CONTRACT_NATIVE_SYMBOL_RE = re.compile(
    r"\.nativeSymbol\s*\{(?P<body>[^{}]*)\}",
    flags=re.DOTALL,
)
CONTRACT_GUARD_SOURCE_PATH_RE = re.compile(
    r"\.runtimeGuard\s+\"[^\"]*\.(?:c|cc|cpp|cu|cuh|h|hpp)\"",
)


@dataclass(frozen=True)
class Finding:
    """One repository-lint warning or error."""

    level: str  # "ERROR" | "WARN"
    path: pathlib.Path
    line: int | None
    col: int | None
    message: str

    def render(self) -> str:
        """Format the finding for terminal and CI output."""
        rel = self.path.relative_to(REPO_ROOT)
        if self.line is None:
            return f"{self.level}: {rel}: {self.message}"
        if self.col is None:
            return f"{self.level}: {rel}:{self.line}: {self.message}"
        return f"{self.level}: {rel}:{self.line}:{self.col}: {self.message}"


def _iter_lean_files() -> Iterable[pathlib.Path]:
    """Yield project Lean files while skipping vendored and generated trees."""
    for p in REPO_ROOT.rglob("*.lean"):
        # Vendored dependencies are checked by their own projects.
        if ".lake" in p.parts:
            continue
        if any(d in p.parts for d in VENDORED_DIR_NAMES):
            continue
        # `_out` can contain generated artifacts; keep policy focused on sources.
        if "_out" in p.parts:
            continue
        yield p


def _iter_generated_script_artifacts() -> Iterable[pathlib.Path]:
    """Generated files that stay outside the checked-in `scripts/` tree."""

    scripts_dir = REPO_ROOT / "scripts"
    if not scripts_dir.exists():
        return
    for p in scripts_dir.rglob("*"):
        if "__pycache__" in p.parts or p.suffix in {".pyc", ".pyo"} or p.name == ".DS_Store":
            yield p


def _iter_script_files() -> Iterable[pathlib.Path]:
    """Yield checked-in support scripts and helper files under `scripts/`."""

    scripts_dir = REPO_ROOT / "scripts"
    if not scripts_dir.exists():
        return
    for p in scripts_dir.rglob("*"):
        if p.is_file():
            yield p


def _is_executable(path: pathlib.Path) -> bool:
    """Return whether any executable bit is set for `path`."""

    return bool(path.stat().st_mode & 0o111)


def _has_shebang(text: str) -> bool:
    """Return whether `text` starts with a Unix shebang line."""

    return text.startswith("#!")


def _has_python_module_docstring(text: str) -> bool:
    """Return whether a Python script starts with a module docstring after an optional shebang."""

    lines = text.splitlines()
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]
    body = "\n".join(lines).lstrip()
    return body.startswith(('"""', "'''"))


def _script_needs_readme_entry(path: pathlib.Path) -> bool:
    """Return whether `path` should be mentioned explicitly in `scripts/README.md`."""

    if path.name == "README.md":
        return False
    if path.suffix in {".json", ".txt"}:
        return False
    return path.is_file()


def _line_col(text: str, idx: int) -> tuple[int, int]:
    """Translate a string offset into 1-based line and column coordinates."""
    # 1-based (Lean-style).
    line = text.count("\n", 0, idx) + 1
    last_nl = text.rfind("\n", 0, idx)
    col = idx - last_nl
    return line, col


def _has_nn_header(path: pathlib.Path, text: str) -> bool:
    """Check whether an `NN/` source file carries the standard TorchLean header."""
    # TorchLean policy: NN sources carry a consistent header at the top of the file.
    if not path.is_relative_to(REPO_ROOT / "NN"):
        return True
    head = "\n".join(text.splitlines()[:10])
    return "Copyright (c) 2026 TorchLean" in head


def _has_lean_module_docstring(text: str) -> bool:
    """Return whether a Lean source contains a module docstring (`/-! ... -/`)."""
    return "/-!" in text


def _mask_lean_comments_and_strings(text: str) -> str:
    """
    Return a same-length string where Lean comments/docstrings and string literals are replaced
    with spaces (newlines preserved).

    This prevents repo-lint regexes like `\\bsorry\\b` from triggering on policy mentions in
    docstrings/comments (and avoids false positives in string literals).

    Notes:
      - Lean block comments `/- ... -/` nest; the scanner tracks nesting depth.
      - The scanner is lexical (no full parser), but it avoids treating comment
        markers inside strings as comments.
    """

    out = list(text)
    n = len(text)
    i = 0

    in_line_comment = False
    block_depth = 0
    in_string = False

    while i < n:
        ch = text[i]

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                i += 1
            else:
                out[i] = " "
                i += 1
            continue

        if block_depth > 0:
            if text.startswith("/-", i):
                out[i] = " "
                if i + 1 < n:
                    out[i + 1] = " "
                block_depth += 1
                i += 2
                continue
            if text.startswith("-/", i):
                out[i] = " "
                if i + 1 < n:
                    out[i + 1] = " "
                block_depth -= 1
                i += 2
                continue
            if ch == "\n":
                i += 1
            else:
                out[i] = " "
                i += 1
            continue

        if in_string:
            # Mask string contents while preserving newlines. Lean strings should
            # not contain raw newlines, but the scanner stays defensive so a
            # missing quote does not mask the rest of the file.
            if ch == "\n":
                in_string = False
                i += 1
                continue
            if ch == "\\" and i + 1 < n:
                # Escape sequence: mask both chars.
                out[i] = " "
                if text[i + 1] != "\n":
                    out[i + 1] = " "
                i += 2
                continue
            out[i] = " "
            if ch == '"':
                in_string = False
            i += 1
            continue

        # Outside comments/strings: detect comment/string starts.
        if text.startswith("--", i):
            out[i] = " "
            if i + 1 < n:
                out[i + 1] = " "
            in_line_comment = True
            i += 2
            continue

        if text.startswith("/-", i):
            out[i] = " "
            if i + 1 < n:
                out[i + 1] = " "
            block_depth = 1
            i += 2
            continue

        if ch == '"':
            out[i] = " "
            in_string = True
            i += 1
            continue

        i += 1

    return "".join(out)


def _check_local_source_refs(path: pathlib.Path, text: str, findings: list[Finding]) -> None:
    """Check backtick-quoted local source paths in authored docs/comments."""

    for m in LOCAL_SOURCE_REF_RE.finditer(text):
        raw_target = m.group(1).split("#", 1)[0]
        if any(marker in raw_target for marker in ("*", "<", ">", "...")):
            continue
        target = pathlib.Path(urllib.parse.unquote(raw_target))
        if target.is_absolute():
            continue
        if not (REPO_ROOT / target).exists():
            line, col = _line_col(text, m.start())
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    f"dead local source reference: `{raw_target}` does not exist in this checkout.",
                )
            )


def _check_docgen_api_links(path: pathlib.Path, text: str, findings: list[Finding]) -> None:
    """Check website links to generated API pages against the corresponding Lean source."""

    for m in DOCGEN_API_LINK_RE.finditer(text):
        url = urllib.parse.unquote(m.group(1))
        module_path = url.removeprefix("/docs/").split("#", 1)[0].removesuffix(".html")
        target = REPO_ROOT / f"{module_path}.lean"
        if not target.exists():
            line, col = _line_col(text, m.start())
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    f"dead generated API link: `/docs/{module_path}.html` has no `{module_path}.lean` source.",
                )
            )


def _check_lean_doc_math(path: pathlib.Path, text: str, findings: list[Finding]) -> None:
    """Reject documentation math that DocGen cannot pass intact to MathJax."""

    for comment in LEAN_DOC_COMMENT_RE.finditer(text):
        for match in DOCGEN_UNSUPPORTED_MATH_DELIMITER_RE.finditer(comment.group()):
            offset = comment.start() + match.start()
            line, col = _line_col(text, offset)
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    "DocGen does not preserve this math delimiter; use `$...$` or `$$...$$`.",
                )
            )
        for display in DOCGEN_DISPLAY_MATH_RE.finditer(comment.group()):
            for match in DOCGEN_MARKDOWN_LIST_IN_DISPLAY_MATH_RE.finditer(display.group(1)):
                offset = comment.start() + display.start(1) + match.start()
                line, col = _line_col(text, offset)
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "a display-math line starts like a Markdown list item; move the operator "
                        "to the preceding TeX line so DocGen keeps the equation together.",
                    )
                )


def _check_verso_math_roles(path: pathlib.Path, text: str, findings: list[Finding]) -> None:
    """Catch mathematical TeX that would remain an ordinary monospace code span."""

    patterns = [VERSO_TEX_IN_ORDINARY_CODE_RE]
    if path.is_relative_to(REPO_ROOT / "blueprint/TorchLeanBlueprint/FormalizationMap"):
        patterns.append(FORMALIZATION_MATH_IN_ORDINARY_CODE_RE)
    for pattern in patterns:
        for match in pattern.finditer(text):
            line, col = _line_col(text, match.start())
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    "mathematical prose is in an ordinary code span; use a Verso `$` math role.",
                )
            )


def _lean_string_field(body: str, field: str) -> str | None:
    m = re.search(rf"\b{re.escape(field)}\s*:=\s*\"([^\"]+)\"", body)
    return m.group(1) if m else None


def _lean_optional_string_field(body: str, field: str) -> str | None:
    m = re.search(rf"\b{re.escape(field)}\s*:=\s*some\s+\"([^\"]+)\"", body)
    return m.group(1) if m else None


def _lake_declares_target(lake_text: str, target: str) -> bool:
    return re.search(
        rf"^\s*(?:target|lean_exe|lean_lib)\s+{re.escape(target)}\b",
        lake_text,
        flags=re.MULTILINE,
    ) is not None


def _check_backend_contract_refs(
    path: pathlib.Path,
    text: str,
    lake_text: str,
    findings: list[Finding],
) -> None:
    """Check structured backend contract references to local sources and native symbols."""

    for m in CONTRACT_GUARD_SOURCE_PATH_RE.finditer(text):
        line, col = _line_col(text, m.start())
        findings.append(
            Finding(
                "ERROR",
                path,
                line,
                col,
                "native source paths belong in structured `.sourceFile` or `.nativeSymbol` provenance, not a runtime-guard label.",
            )
        )

    for m in CONTRACT_SOURCE_FILE_RE.finditer(text):
        body = m.group("body")
        raw_path = _lean_string_field(body, "path")
        if raw_path is None:
            line, col = _line_col(text, m.start())
            findings.append(Finding("ERROR", path, line, col, "`.sourceFile` provenance is missing `path := ...`."))
            continue
        source = REPO_ROOT / raw_path
        if not source.exists():
            line, col = _line_col(text, m.start())
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    f"`.sourceFile` provenance points to missing source `{raw_path}`.",
                )
            )

    for m in CONTRACT_NATIVE_SYMBOL_RE.finditer(text):
        body = m.group("body")
        raw_path = _lean_string_field(body, "path")
        symbol = _lean_string_field(body, "symbol")
        build_target = _lean_optional_string_field(body, "buildTarget?")
        line, col = _line_col(text, m.start())
        if raw_path is None:
            findings.append(Finding("ERROR", path, line, col, "`.nativeSymbol` provenance is missing `path := ...`."))
            continue
        if symbol is None:
            findings.append(Finding("ERROR", path, line, col, "`.nativeSymbol` provenance is missing `symbol := ...`."))
            continue
        source = REPO_ROOT / raw_path
        if not source.exists():
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    f"`.nativeSymbol` provenance points to missing source `{raw_path}`.",
                )
            )
            continue
        try:
            source_text = source.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            findings.append(Finding("ERROR", source, None, None, f"failed to read file: {e}"))
            continue
        if symbol not in source_text:
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    f"`.nativeSymbol` provenance names `{symbol}`, but it does not occur in `{raw_path}`.",
                )
            )
        if build_target is not None and not _lake_declares_target(lake_text, build_target):
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    f"`.nativeSymbol` provenance names Lake target `{build_target}`, but `lakefile.lean` does not declare it.",
                )
            )


def lint_repo(*, fail_on_warn: bool) -> list[Finding]:
    """Run TorchLean's repository hygiene checks and return all findings."""
    findings: list[Finding] = []
    try:
        lake_text = (REPO_ROOT / "lakefile.lean").read_text(encoding="utf-8")
    except OSError as e:
        lake_text = ""
        findings.append(Finding("ERROR", REPO_ROOT / "lakefile.lean", None, None, f"failed to read file: {e}"))

    if not LINT_SCOPE_SENTINEL.exists():
        findings.append(
            Finding(
                "ERROR",
                LINT_SCOPE_SENTINEL,
                None,
                None,
                "repo linter is not rooted at TorchLean; expected to see "
                "NN/MLTheory/CROWN/Lyapunov/Certificate.lean.",
            )
        )

    for path in _iter_generated_script_artifacts():
        findings.append(
            Finding(
                "ERROR",
                path,
                None,
                None,
                "generated Python/cache artifact under `scripts/`; remove it from the source tree.",
            )
        )

    scripts_readme = REPO_ROOT / "scripts/README.md"
    try:
        scripts_readme_text = scripts_readme.read_text(encoding="utf-8")
    except OSError as e:
        scripts_readme_text = ""
        findings.append(Finding("ERROR", scripts_readme, None, None, f"failed to read file: {e}"))

    for env_var, rel_impl in DOCUMENTED_ENV_VAR_IMPLEMENTATIONS.items():
        docs_mention = False
        for path in list(REPO_ROOT.glob("README.md")) + list((REPO_ROOT / "blueprint").rglob("*.lean")) + list((REPO_ROOT / "home_page").rglob("*.md")):
            if any(part in {"_site", "docs", "vendor"} for part in path.relative_to(REPO_ROOT).parts):
                continue
            try:
                if env_var in path.read_text(encoding="utf-8"):
                    docs_mention = True
                    break
            except OSError:
                continue
        if docs_mention:
            impl = REPO_ROOT / rel_impl
            try:
                impl_text = impl.read_text(encoding="utf-8")
            except OSError as e:
                findings.append(Finding("ERROR", impl, None, None, f"documented env var `{env_var}` has no readable implementation: {e}"))
                continue
            if env_var not in impl_text:
                findings.append(
                    Finding(
                        "ERROR",
                        impl,
                        None,
                        None,
                        f"documented env var `{env_var}` is not implemented in its declared producer helper.",
                    )
                )

    trust_file = REPO_ROOT / "TRUST_BOUNDARIES.md"
    try:
        trust_text = trust_file.read_text(encoding="utf-8")
    except OSError as e:
        trust_text = ""
        findings.append(Finding("ERROR", trust_file, None, None, f"failed to read file: {e}"))

    for fq_name, (rel_source, decl_re) in TRUST_BOUNDARY_DECL_REFS.items():
        if fq_name not in trust_text:
            findings.append(
                Finding(
                    "ERROR",
                    trust_file,
                    None,
                    None,
                    f"trust-boundary declaration `{fq_name}` is missing from TRUST_BOUNDARIES.md.",
                )
            )
        source = REPO_ROOT / rel_source
        try:
            source_text = source.read_text(encoding="utf-8")
        except OSError as e:
            findings.append(Finding("ERROR", source, None, None, f"failed to read file: {e}"))
            continue
        if not decl_re.search(source_text):
            findings.append(
                Finding(
                    "ERROR",
                    source,
                    None,
                    None,
                    f"TRUST_BOUNDARIES.md cites `{fq_name}`, but the expected declaration was not found.",
                )
            )

    doc_fact_paths = (
        list(REPO_ROOT.glob("README.md"))
        + list(REPO_ROOT.glob("TRUST_BOUNDARIES.md"))
        + list(REPO_ROOT.glob("THIRD_PARTY_NOTICES.md"))
        + list((REPO_ROOT / "blueprint").rglob("*.lean"))
        + list((REPO_ROOT / "home_page").rglob("*.md"))
        + list((REPO_ROOT / "NN").rglob("*.md"))
        + list((REPO_ROOT / "scripts").rglob("*.md"))
    )
    for path in doc_fact_paths:
        if any(part in {"_site", "docs", "vendor"} for part in path.relative_to(REPO_ROOT).parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if path.suffix != ".lean":
            _check_local_source_refs(path, text, findings)
            _check_docgen_api_links(path, text, findings)
        elif path.is_relative_to(REPO_ROOT / "blueprint/TorchLeanBlueprint"):
            _check_verso_math_roles(path, text, findings)
        for rx, msg in DOC_FACT_BANNED_PATTERNS:
            for m in rx.finditer(text):
                line, col = _line_col(text, m.start())
                findings.append(Finding("ERROR", path, line, col, msg))
        for m in TORCHLEAN_SOURCE_LINK_RE.finditer(text):
            raw_target = m.group(1).split("#", 1)[0]
            target = urllib.parse.unquote(raw_target)
            if not (REPO_ROOT / target).exists():
                line, col = _line_col(text, m.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        f"dead TorchLean source link: `{raw_target}` does not exist in this checkout.",
                    )
                )

    for path in _iter_script_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        except OSError as e:
            findings.append(Finding("ERROR", path, None, None, f"failed to read file: {e}"))
            continue

        if path.suffix in {".py", ".sh"}:
            has_shebang = _has_shebang(text)
            is_executable = _is_executable(path)
            if has_shebang and not is_executable:
                findings.append(
                    Finding("ERROR", path, 1, 1, "script has a shebang but is not executable.")
                )
            if is_executable and not has_shebang:
                findings.append(
                    Finding("ERROR", path, 1, 1, "executable script should start with a shebang.")
                )

        if path.suffix == ".py" and not _has_python_module_docstring(text):
            findings.append(
                Finding("ERROR", path, 1, 1, "Python scripts/helpers should start with a module docstring.")
            )

        if _script_needs_readme_entry(path):
            script_rel = rel.removeprefix("scripts/")
            if script_rel not in scripts_readme_text:
                findings.append(
                    Finding("ERROR", path, None, None, "`scripts/README.md` should explain this script.")
                )

    for rel in sorted(REMOVED_COMPATIBILITY_PATHS):
        path = REPO_ROOT / rel
        if path.exists():
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    None,
                    None,
                    "removed compatibility module has been restored; use the canonical API or subsystem import.",
                )
            )
    for prefix in REMOVED_COMPATIBILITY_PREFIXES:
        directory = REPO_ROOT / prefix
        if directory.exists() and any(directory.rglob("*.lean")):
            findings.append(
                Finding(
                    "ERROR",
                    directory,
                    None,
                    None,
                    "removed compatibility import tree has been restored; use the canonical subsystem umbrellas.",
                )
            )

    banned_regexes: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"\bnative_decide\b"), "`native_decide` is banned in TorchLean."),
        (re.compile(r"\bsorry\b"), "`sorry` is banned in TorchLean sources."),
        (re.compile(r"\badmit\b"), "`admit` is banned in TorchLean sources."),
        (
            re.compile(r"\b(FitConfig|LoaderFitConfig|FitReport)\b"),
            "old lower training names are removed; use `TrainConfig`, `LoaderTrainConfig`, and `TrainReport`.",
        ),
        (
            re.compile(r"\beffectiveFitBatchSize\b"),
            "old lower training helper names are removed; use `effectiveTrainBatchSize`.",
        ),
        (
            re.compile(r"\b(FitResult|StreamFitResult|PairStreamFitResult)\b"),
            "old trainer result names are removed; use `TrainResult`, `StreamTrainResult`, and `PairStreamTrainResult`.",
        ),
        (
            re.compile(r"\bverifyLInfIBP\b|\b(Trainer\.)?Verify\.robustLInf\b"),
            "duplicate verification helper names are removed; use `verifyRobustLInf` on trained results or `Trainer.Verify.lInfIBP` for requests.",
        ),
        (
            re.compile(
                r"\b(?:Conv2dLayer|Core\.oneHotAction|scalarValue|CROWNNodeCertificate|"
                r"ToTorchLean\.Sequential)\b"
            ),
            "removed duplicate name found; use the canonical declaration directly.",
        ),
        (
            re.compile(
                r"\b(?:SequentialModel|ModelBuilder|modelParamShapes|LossReduction)\b|"
                r"\bTorchLean\.ParamTensors\b"
            ),
            "removed API alias found; use the canonical `nn`, `Module`, `Loss`, or `TensorPack` name.",
        ),
        (
            re.compile(
                r"\b(?:oneHotNat|oneHotToken|oneHotSequence|matrixPadTo|vectorFromArray|"
                r"flattenKeep0|instantiateHostFloat|curveHostFloat|trainFixedCurveHostFloat|"
                r"RawDataLoader|EmbeddingOptions|oneHotAccuracyBatched|oneHotAccuracyLoader|"
                r"oneHotMetricsBatched|classProbesBatched|ScalarModuleDef|ScalarModule|"
                r"ppoActorCriticScalarModuleDef|ScalarEvaluator|lossScalar|"
                r"LossModuleDef|LossModule|LossEvaluator|lossModuleWithMode|lossModule)\b"
            ),
            "removed misleading API name found; use the canonical name that states its fallback or scalar semantics.",
        ),
        (
            re.compile(r"\b(?:Runtime\.Autograd\.Torch|TorchLean)\.matmul\b"),
            "the rank-two public operation is `mm`; reserve `matmul` for rank-polymorphic broadcasting semantics.",
        ),
        (
            re.compile(
                r"\b(?:flattenBatch|flattenBatchPrefix|classifierBatch|regressorBatch|"
                r"uniformND|maskND|randND|loadCsvTensorND)\b"
            ),
            "removed dimension-specific API name found; use `flattenLeading`, a head with an explicit leading shape, or the `*Dims` random constructor.",
        ),
        (
            re.compile(
                r"\b(?:NativeOptimizerCheckpoint|CudaAdamSchema|saveNativeOptimizerState|"
                r"loadNativeOptimizerState|projectedSGDUpdate_identity_eq_sgd|"
                r"update_identity_param_eq_momentumSGD)\b"
            ),
            "removed checkpoint or definitional-theorem name found; use the shared optimizer-state checkpoint API.",
        ),
        (re.compile(r"\bby\s+omega\b"), "`omega` is banned in TorchLean; prefer `linarith`/`nlinarith`/`grind` or small arithmetic lemmas."),
        (re.compile(r"^\s*omega\b", flags=re.MULTILINE), "`omega` is banned in TorchLean; prefer `linarith`/`nlinarith`/`grind` or small arithmetic lemmas."),
        (re.compile(r"\bsimp\s*\[\s*\*(\s*[,\]])"), "`simp [*]` is banned; prefer `simp [h₁, h₂]` or `simp (config := ...)` with explicit hypotheses."),
        (
            re.compile(r"\bset_option\s+maxHeartbeats\b"),
            "proof-level `maxHeartbeats` overrides are not allowed; split the declaration or isolate expensive normalization behind reusable lemmas.",
        ),
        (
            re.compile(r"^\s*public\s+import\s+Mathlib\.Tactic\b", flags=re.MULTILINE),
            "Do not `public import Mathlib.Tactic.*`; import the specific tactic modules you use (non-public).",
        ),
        (
            re.compile(r"^\s*import\s+Mathlib\.Tactic(?!\.)\b", flags=re.MULTILINE),
            "Do not `import Mathlib.Tactic` (umbrella import). Import the specific `Mathlib.Tactic.*` modules you use.",
        ),
        (re.compile(r"@\[\s*de" r"precated\b"), "`@[de" "precated]` is banned in TorchLean sources."),
        (
            re.compile(
                r"\b(?:compatibility (?:alias|shim|wrapper|layer)|legacy (?:alias|name)|"
                r"deprecated alias|old import path|migration shim|kept for compatibility)\b",
                flags=re.IGNORECASE,
            ),
            "compatibility aliases and shims are not allowed; migrate callers to the canonical API and delete the old route.",
        ),
    ]

    axiom_re = re.compile(r"^\s*axiom\s+([A-Za-z0-9_'.]+)\b", flags=re.MULTILINE)

    for path in _iter_lean_files():
        try:
            raw = path.read_bytes()
        except OSError as e:
            findings.append(Finding("ERROR", path, None, None, f"failed to read file: {e}"))
            continue

        # Enforce LF-only; CRLF and stray CR cause confusing diffs and occasional parser weirdness.
        if b"\r" in raw:
            findings.append(Finding("ERROR", path, None, None, "contains CR (`\\r`) characters (use LF)."))
            continue

        text = raw.decode("utf-8", errors="replace")
        masked = _mask_lean_comments_and_strings(text)
        _check_local_source_refs(path, text, findings)
        _check_lean_doc_math(path, text, findings)
        _check_backend_contract_refs(path, text, lake_text, findings)

        if not _has_nn_header(path, text):
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    1,
                    1,
                    "missing TorchLean header in the first ~10 lines (expected `Copyright (c) 2026 TorchLean`).",
                )
            )

        if path.is_relative_to(REPO_ROOT / "NN") and not _has_lean_module_docstring(text):
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    1,
                    1,
                    "missing Lean module docstring (`/-! ... -/`); add purpose, main declarations, and import guidance.",
                )
            )

        # Line-level whitespace hygiene: Lean rejects tabs, and trailing whitespace is noisy in reviews.
        for i, line in enumerate(text.splitlines(), start=1):
            if "\t" in line:
                col = line.find("\t") + 1
                findings.append(Finding("ERROR", path, i, col, "tab character found (use spaces)."))
            if line.endswith(" "):
                findings.append(Finding("ERROR", path, i, len(line), "trailing whitespace."))

        for rx, msg in banned_regexes:
            for m in rx.finditer(masked):
                line, col = _line_col(text, m.start())
                findings.append(Finding("ERROR", path, line, col, msg))

        rel = path.relative_to(REPO_ROOT).as_posix()

        # Keep the numerical library reusable without importing tensors, models, runtimes, or
        # verification. TorchLean-specific adapters must point into `NN.Floats`, never the reverse.
        if rel == "NN/Floats.lean" or rel.startswith("NN/Floats/"):
            for m in re.finditer(
                r"^\s*(?:public\s+)?import\s+(NN\.[A-Za-z0-9_.]+)\s*$",
                masked,
                flags=re.MULTILINE,
            ):
                imported = m.group(1)
                if not (imported == "NN.Floats" or imported.startswith("NN.Floats.")
                        or imported == "NN.Core" or imported.startswith("NN.Core.")):
                    line, col = _line_col(text, m.start(1))
                    findings.append(
                        Finding(
                            "ERROR",
                            path,
                            line,
                            col,
                            f"floating-point core imports `{imported}`; move this integration to the spec, proof, runtime, or verification layer.",
                        )
                    )

        is_shape_generic_public_api = any(
            rel.startswith(prefix)
            for prefix in (
                "NN/API/TorchLean/",
                "NN/API/Models/",
                "NN/API/Samples/",
            )
        )
        if is_shape_generic_public_api:
            for declaration in PUBLIC_DECL_RE.finditer(masked):
                name = declaration.group("name")
                if PUBLIC_LAYOUT_NAME_RE.search(name):
                    line, col = _line_col(text, declaration.start("name"))
                    findings.append(
                        Finding(
                            "ERROR",
                            path,
                            line,
                            col,
                            f"public declaration `{name}` encodes a fixed rank or memory layout; "
                            "express axes through `Spec.Shape`, `Vector Nat d`, or a domain-specific "
                            "example outside the public tensor/model API.",
                        )
                    )

        if rel.startswith("NN/API/TorchLean/Trainer/Train/") and rel.endswith(".lean"):
            if "(opts : Options)" in masked and "(opts : TrainOptions" in masked:
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        None,
                        None,
                        "trainer train implementation should distinguish runtime `Options` from `TrainOptions` (use names like `runtimeOpts` and `trainOpts`).",
                    )
                )

        if rel == "NN/API/TorchLean/Trainer/Train.lean":
            m = TOP_LEVEL_API_DECL_RE.search(masked)
            if m:
                line, col = _line_col(text, m.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "`NN.API.Runtime.Trainer.Train` must stay an import-only aggregator; put training implementation in `NN.API.Runtime.Trainer.Train.*` modules.",
                    )
                )

        if any(rel == prefix or rel.startswith(prefix) for prefix in PUBLIC_GUIDE_PREFIXES):
            for rx, msg in PUBLIC_GUIDE_BANNED_PATTERNS:
                for m in rx.finditer(text):
                    line, col = _line_col(text, m.start())
                    findings.append(Finding("ERROR", path, line, col, msg))

        if rel == "NN/API/Neural.lean" and re.search(
            r"^\s*public\s+import\s+NN\.API\.Training\s*$", masked, flags=re.MULTILINE
        ):
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    None,
                    None,
                    "`NN.API.Neural` must not re-export `NN.API.Trainer`; use `TorchLean.Trainer` for ordinary code and import the advanced training module explicitly when needed.",
                )
            )
        is_trainer_api = (
            rel == "NN/API/TorchLean/Trainer.lean"
            or rel.startswith("NN/API/TorchLean/Trainer/")
        )
        is_training_entrypoint = rel == "NN/API/Training.lean"
        if re.search(
            r"^\s*public\s+import\s+NN\.API\.Training\s*$", masked, flags=re.MULTILINE
        ) and not (is_trainer_api or is_training_entrypoint):
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    None,
                    None,
                    "`NN.API.Trainer` should only be imported by the Trainer API; keep the callback-heavy training layer out of broad application API imports.",
                )
            )

        if rel.endswith(".lean") and any(rel.startswith(prefix) for prefix in PUBLIC_EXAMPLE_PREFIXES):
            for rx, msg in PUBLIC_EXAMPLE_BANNED_PATTERNS:
                for m in rx.finditer(masked):
                    line, col = _line_col(text, m.start())
                    findings.append(Finding("ERROR", path, line, col, msg))

        if rel.endswith(".lean") and any(rel.startswith(prefix) for prefix in PUBLIC_TUTORIAL_PREFIXES):
            for rx, msg in PUBLIC_TUTORIAL_BANNED_PATTERNS:
                for m in rx.finditer(masked):
                    line, col = _line_col(text, m.start())
                    findings.append(Finding("ERROR", path, line, col, msg))

        # Axioms must be quarantined and named explicitly.
        allowed_axiom_names = ALLOWED_AXIOMS.get(rel, set())
        for m in axiom_re.finditer(masked):
            axiom_name = m.group(1)
            if axiom_name not in allowed_axiom_names:
                line, col = _line_col(text, m.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        f"axiom `{axiom_name}` is not allowlisted; quarantine and document trusted axioms.",
                    )
                )

        # Warn by default; callers can promote these warnings with `--fail-on-warn`.
        if "set_option linter." in masked and " false" in masked:
            suppressions = list(
                re.finditer(r"set_option\s+linter\.([A-Za-z0-9_]+)\s+false(?:\s+in)?", masked)
            )
            disallowed_suppressions = [
                m for m in suppressions
                if not (m.group(1) == "auxLemma" and m.group(0).rstrip().endswith(" in"))
            ]
            # Only a coarse signal; report once per file.
            if disallowed_suppressions:
                rel_posix = path.relative_to(REPO_ROOT).as_posix()
                # Some executable examples, tests, and maintenance scripts scope linter options
                # locally. Keep this warning focused on library-facing code where suppressions are
                # part of the public proof surface.
                if (
                    rel_posix in ALLOWED_LINTER_SUPPRESSION_FILES
                    or
                    rel_posix.startswith("NN/Examples/")
                    or rel_posix.startswith("NN/Tests/")
                    or rel_posix.startswith("scripts/")
                    # The typed-graph correctness proofs scope linter options locally to
                    # keep proof scripts readable; warning here is usually not actionable.
                    or rel_posix.startswith("NN/Runtime/Autograd/TypedGraph/IRExec/Correctness/")
                ):
                    pass
                else:
                    findings.append(
                        Finding(
                            "WARN",
                            path,
                            None,
                            None,
                            "suppresses a linter (`set_option linter.* false`). Prefer fixing the warning or scoping the option tightly.",
                        )
                    )

    if not fail_on_warn:
        return findings
    # Promote warnings to errors.
    return [
        Finding("ERROR" if f.level == "WARN" else f.level, f.path, f.line, f.col, f.message)
        for f in findings
    ]


def main() -> int:
    """CLI entry point used by local checks and CI."""
    ap = argparse.ArgumentParser(description="TorchLean repo lints (project policies).")
    ap.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Treat warnings as errors (useful for tightening policies over time).",
    )
    args = ap.parse_args()

    findings = lint_repo(fail_on_warn=args.fail_on_warn)
    errors = [f for f in findings if f.level == "ERROR"]
    warns = [f for f in findings if f.level == "WARN"]

    for f in findings:
        print(f.render())

    if errors:
        print(f"\nFAILED: {len(errors)} error(s), {len(warns)} warning(s).")
        return 1

    if warns:
        print(f"\nOK (with warnings): {len(warns)} warning(s).")
        return 0

    print("OK: no issues found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

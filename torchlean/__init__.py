"""Capture tensor computations and request explicit checks from TorchLean.

``verify(model, example)`` checks the captured graph's declared structure and
types. Add ``checks=("structure", "types", "compare")`` to compare CPU reference
IR execution with the frontend on that example. Neither result is a proof of
frontend equivalence or correctness on every input.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any


def _command(checker: str | Path | Sequence[str] | None) -> list[str]:
    if isinstance(checker, (str, Path)):
        return [str(checker), "ir"]
    if checker is not None:
        if (not isinstance(checker, Sequence) or not checker
                or not all(isinstance(argument, str) for argument in checker)):
            raise TypeError("checker must be a path or a nonempty sequence of command arguments")
        return list(checker)
    path = os.environ.get("TORCHLEAN_VERIFY")
    if path:
        return [path, "ir"]
    executable = Path(__file__).resolve().parent.parent / ".lake/build/bin/verify"
    if not executable.is_file():
        raise FileNotFoundError(
            "Build the TorchLean `verify` executable first, or supply checker= with "
            "a command ending in `verify ir`. Verification does not start a build implicitly."
        )
    return [str(executable), "ir"]


def _run(command: list[str], arguments: Sequence[str], *, report: bool = False
         ) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command + list(arguments), capture_output=True, text=True, check=False)
    if result.returncode not in ((0, 1) if report else (0,)):
        raise RuntimeError(f"TorchLean checker exited with {result.returncode}: {result.stderr.strip()}")
    return result


def _capture(model: Any, examples: tuple[Any, ...], frontend: str, command: list[str],
             directory: Path, checks: Sequence[str]) -> dict[str, Any]:
    compare = "compare" in checks
    execute = compare or "execute" in checks
    if not examples or examples[0] is None:
        raise ValueError("capturing a computation requires example tensors")
    if frontend == "jax":
        import jax
        from NN.Runtime.JAX.jaxpr import Unsupported, encode, export

        try:
            artifact = export(model, *examples)
        except Unsupported as error:
            raise NotImplementedError(str(error)) from error
        if execute:
            artifact["arguments"] = [encode(value) for value in examples]
        if compare:
            artifact["reference_outputs"] = [encode(value)
                                             for value in jax.tree.leaves(model(*examples))]
        return artifact
    if frontend != "pytorch":
        raise ValueError(f"unknown frontend {frontend!r}; use pytorch, jax, or pass an artifact")
    import torch

    if not all(isinstance(value, torch.Tensor) for value in examples):
        raise NotImplementedError("PyTorch capture requires positional tensor arguments")

    path = directory / "capture.py"
    _run(command, ["--export-pytorch", str(path)])
    spec = importlib.util.spec_from_file_location("torchlean_capture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the emitted PyTorch adapter")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    # Capture and ShapeProp may execute code. Keep the caller's model and input
    # intact; arbitrary external effects in user Python remain outside this API.
    if not isinstance(model, torch.nn.Module):
        if not callable(model):
            raise TypeError("PyTorch capture requires a module or callable")
        function = model

        class Function(torch.nn.Module):
            def forward(self, *inputs):
                return function(*inputs)

        model = Function()
    model = copy.deepcopy(model)
    examples = tuple(value.detach().clone().requires_grad_(value.requires_grad)
                     for value in examples)
    artifact = adapter.export_torchlean_graph_json(
        model, examples, str(directory / "graph.json"), require_torch_export=True)
    if execute:
        artifact["arguments"] = [adapter._constant(value) for value in examples]
    if compare:
        with torch.no_grad():
            outputs, _ = torch.utils._pytree.tree_flatten(model(*examples))
        if not all(isinstance(value, torch.Tensor) for value in outputs):
            raise ValueError("sample comparison requires tensor outputs")
        artifact["reference_outputs"] = [adapter._constant(value) for value in outputs]
    return artifact


def verify(model: Any, example: Any = None, *examples: Any, frontend: str = "pytorch",
           checks: Sequence[str] = ("structure", "types"),
           checker: str | Path | Sequence[str] | None = None) -> dict[str, Any]:
    """Check a captured computation or an existing typed IR artifact.

    ``model`` is a PyTorch module, a JAX function with ``frontend="jax"``, or an
    artifact mapping. Both frontends accept positional tensor arguments, as in
    ``verify(function, x, y)``. ``checks`` may request
    ``structure``, ``types``, ``execute`` or ``compare``. Unknown requests produce
    unsupported entries, not a pass.

    ``checker`` is an executable path (the ``ir`` subcommand is appended), or a
    complete argument sequence already ending in the IR checker command. No shell
    is used. Omitting it uses ``TORCHLEAN_VERIFY`` or the local built executable.

    The returned ``ok`` covers only the requested checks. Inspect ``checks`` for
    their scope. In particular, a bitwise sample comparison is not an all-input
    theorem, and a mismatch may reflect different reduction or transcendental
    implementations rather than a frontend bug.
    """
    if (isinstance(checks, str) or not isinstance(checks, Sequence)
            or not all(isinstance(name, str) for name in checks)):
        raise TypeError("checks must be a sequence of names, not a single string")
    if not checks:
        raise ValueError("request at least one check")
    if isinstance(model, Mapping) and (example is not None or examples):
        raise ValueError("an artifact already contains its arguments; do not supply example tensors")
    command = _command(checker)
    with tempfile.TemporaryDirectory(prefix="torchlean-verify-") as temporary:
        directory = Path(temporary)
        try:
            artifact = dict(model) if isinstance(model, Mapping) else _capture(
                model, (example, *examples), frontend, command, directory, checks)
        except NotImplementedError as error:
            # No Lean checks ran. Keep every request visible rather than fabricating
            # a successful result for a smaller graph or a different frontend.
            return dict(format="torchlean.verification.v1", ok=False, checker="not_run",
                        checks=[dict(name=name, status="unsupported",
                                     message=f"Capture is unsupported: {error}") for name in checks],
                        outputs=None, conversion_flags=None, frontend_equivalence="not_checked",
                        frontend_device_layout_constraints="not_checked",
                        external_kernels="not_checked")
        path = directory / "artifact.json"
        path.write_text(json.dumps(artifact, allow_nan=False), encoding="utf-8")
        result = _run(command, [str(path), *checks], report=True)
        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"checker did not produce a JSON report: {result.stderr.strip()}") from error
        if not isinstance(report, dict) or report.get("format") != "torchlean.verification.v1":
            raise RuntimeError("checker returned an unknown verification report format")
        findings = report.get("checks")
        if (not isinstance(findings, list) or not all(isinstance(item, dict) for item in findings)
                or [item.get("name") for item in findings] != list(checks)
                or type(report.get("ok")) is not bool):
            raise RuntimeError("checker report does not cover the requested checks")
        if (result.returncode == 0) != report["ok"]:
            raise RuntimeError("checker exit status conflicts with its verification report")
        return report


__all__ = ["verify"]

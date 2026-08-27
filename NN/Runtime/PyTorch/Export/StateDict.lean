/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Export.Core

/-!
# PyTorch `state_dict` Bridge

This module is the general weight-interchange path for PyTorch users.

The important split is:

- **Weights** move through PyTorch `state_dict`s. PyTorch’s own documentation recommends saving a
  module’s learned parameters with `torch.save(model.state_dict(), path)` because that is the most
  flexible restoration format.
- **Graphs** move through graph capture (`torch.export`, FX, ONNX, or TorchLean `NN.IR.Graph`).
  A `state_dict` alone does not describe the model architecture; it only names tensors.

Lean should not try to parse PyTorch pickle/zip checkpoints directly. Instead, we emit a small Python
adapter that loads a checkpoint with PyTorch, normalizes common wrappers such as
`{"state_dict": ...}`, and writes shape-checkable JSON:

```json
{
  "params": { "layer.weight": [[...]], "layer.bias": [...] },
  "meta": { "layer.weight": { "shape": [out, in], "dtype": "torch.float32" } }
}
```

`NN.Runtime.PyTorch.Import.Core` then parses the `"params"` object into typed TorchLean tensors.
Architecture-specific loaders are still useful, but only for mapping names and shapes. The transport
format itself is model-agnostic.

References:
- PyTorch documentation, "Saving and Loading Models":
  `https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html`
- PyTorch `torch.export` user guide:
  `https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html`
- PyTorch FX overview:
  `https://docs.pytorch.org/docs/stable/fx.html`
-/

@[expose] public section

namespace Export
namespace PyTorch
namespace StateDict

open Export.PyTorch

/--
Options for the generated Python checkpoint-to-JSON adapter.

This adapter is conservative by design: it accepts tensor-valued entries only, drops common
DataParallel prefixes when requested, and writes plain JSON rather than a PyTorch-specific binary
format. That makes the output easy to inspect, diff, and parse in Lean.
-/
structure JsonBridgeOptions where
  /-- Name of the Python helper function emitted into the generated script. -/
  functionName : String := "export_state_dict_json"
  /-- If true, strip a leading `"module."` from keys produced by `torch.nn.DataParallel`. -/
  stripDataParallelPrefix : Bool := true
  /-- If true, include a `"meta"` object with per-key shape and dtype strings. -/
  includeMeta : Bool := true
  /-- Python expression passed as `weights_only` to `torch.load`. -/
  weightsOnlyExpr : String := "True"
  deriving Repr

/--
Emit a standalone Python script that converts a PyTorch checkpoint into TorchLean JSON.

The script handles three common checkpoint layouts:

- a raw `state_dict`;
- `{ "state_dict": state_dict }`;
- `{ "model_state_dict": state_dict }`.

Usage of the generated script:

```bash
python export_state_dict_json.py model.pt model.json
```

The resulting `model.json` is accepted by `Import.PyTorch.loadWeights?`.
-/
def generateJsonBridgeScript (opts : JsonBridgeOptions := {}) : String :=
  joinLines
    #[ "import argparse"
    , "import json"
    , "import torch"
    , ""
    , "def _unwrap_state_dict(obj):"
    , indentFour "if isinstance(obj, dict):"
    , indentEight "if \"state_dict\" in obj and isinstance(obj[\"state_dict\"], dict):"
    , indentEight "    return obj[\"state_dict\"]"
    , indentEight "if \"model_state_dict\" in obj and isinstance(obj[\"model_state_dict\"], dict):"
    , indentEight "    return obj[\"model_state_dict\"]"
    , indentFour "return obj"
    , ""
    , "def _normalize_key(key: str, strip_data_parallel_prefix: bool) -> str:"
    , indentFour "if strip_data_parallel_prefix and key.startswith(\"module.\"):"
    , indentEight "return key[len(\"module.\"):]"
    , indentFour "return key"
    , ""
    , "def _tensor_payload(t: torch.Tensor):"
    , indentFour "return t.detach().cpu().tolist()"
    , ""
    , s!"def {opts.functionName}(checkpoint_path: str, json_path: str):"
    , indentFour s!"obj = torch.load(checkpoint_path, map_location=\"cpu\", weights_only={opts.weightsOnlyExpr})"
    , indentFour "state_dict = _unwrap_state_dict(obj)"
    , indentFour "if not isinstance(state_dict, dict):"
    , indentEight "raise TypeError(\"expected a state_dict or a checkpoint containing one\")"
    , indentFour "params = {}"
    , indentFour "meta = {}"
    , indentFour "for key, value in state_dict.items():"
    , indentEight "if not torch.is_tensor(value):"
    , indentEight "    continue"
    , indentEight s!"k = _normalize_key(str(key), {pyBool opts.stripDataParallelPrefix})"
    , indentEight "v = value.detach().cpu()"
    , indentEight "params[k] = _tensor_payload(v)"
    , indentEight "meta[k] = {\"shape\": list(v.shape), \"dtype\": str(v.dtype)}"
    , indentFour "payload = {\"params\": params}"
    , indentFour s!"if {pyBool opts.includeMeta}:"
    , indentEight "payload[\"meta\"] = meta"
    , indentFour "with open(json_path, \"w\", encoding=\"utf-8\") as f:"
    , indentEight "json.dump(payload, f, indent=2, sort_keys=True)"
    , indentFour "return payload"
    , ""
    , "def main():"
    , indentFour "parser = argparse.ArgumentParser(description=\"Export a PyTorch state_dict to TorchLean JSON\")"
    , indentFour "parser.add_argument(\"checkpoint\", help=\"Path to a .pt/.pth checkpoint\")"
    , indentFour "parser.add_argument(\"json\", help=\"Output JSON path\")"
    , indentFour "args = parser.parse_args()"
    , indentFour s!"payload = {opts.functionName}(args.checkpoint, args.json)"
    , indentFour "print(f\"wrote {len(payload['params'])} tensor entries to {args.json}\")"
    , ""
    , "if __name__ == \"__main__\":"
    , indentFour "main()"
    ]

end StateDict
end PyTorch
end Export

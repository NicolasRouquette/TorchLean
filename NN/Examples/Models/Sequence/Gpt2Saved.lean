/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Examples.Models.Sequence.Gpt2

/-!
# GPT-2 Checkpoint Example

This is the load-and-sample half of the byte-level GPT example.

1. Train and save a model checkpoint:

```bash
lake -R -K cuda=true build torchlean:exe
lake -R -K cuda=true exe torchlean gpt2 --device cuda --tiny-shakespeare --steps 1 --windows 1 \
  --prompt "First Citizen:" --generate 0 \
  --save-checkpoint data/model_zoo/gpt2_shakespeare.state.json
```

2. Load the checkpoint and sample text (no training loop or optimizer state):

```bash
lake -R -K cuda=true exe torchlean gpt2_saved --device cuda \
  --checkpoint data/model_zoo/gpt2_shakespeare.state.json \
  --prompt "First Citizen:" --generate 0
```

## What A Checkpoint Is Here

This example uses the simplest TorchLean checkpoint format:

- a shape-indexed pack of model state tensors,
- stored as exact `Float.toBits` values in JSON, and
- checked against the model's state layout before inference starts.

So save/load is model-agnostic: if we can name the model, TorchLean can compute the expected
state shapes and reject stale or mismatched checkpoint files.

## Why This Is A Separate Example

The inference-only workflow is direct: load a checkpoint, convert it into runtime
runtime handles, and sample text without building a training loop.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Sequence.Gpt2Saved

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "torchlean gpt2_saved"

/-- Help text for checkpoint-only GPT-2 sampling. -/
def usage : String :=
  String.intercalate "\n"
    [ "Usage:"
    , "  lake -R -K cuda=true exe torchlean gpt2_saved --device cuda --checkpoint PATH [generation flags]"
    , ""
    , "Required:"
    , "  --checkpoint PATH    model state written by `torchlean gpt2 --save-checkpoint`"
    , ""
    , "Common generation flags:"
    , "  --prompt TEXT        prompt prefix"
    , "  --generate N         number of bytes to generate"
    , "  --temperature X      sampling temperature"
    , "  --top-k N            top-k cutoff"
    , "  --seed N             sampling seed"
    ]

/-- Command-local options for loading one checkpoint and sampling from it. -/
structure SavedOptions extends text.GenerationOptions where
  /-- Model checkpoint loaded before sampling starts. -/
  checkpointPath : System.FilePath
deriving Repr

namespace SavedOptions

/-- Parse the checkpoint path followed by the shared generation flags. -/
def parse (args : List String) (defaults : text.GenerationOptions) :
    Except String (SavedOptions × List String) := do
  let (checkpointPath, args) ←
    CLI.takeRequiredPathFlag args "checkpoint" (exeName := exeName)
  let (generation, args) ← text.GenerationOptions.parse exeName args defaults
  pure ({ checkpointPath, toGenerationOptions := generation }, args)

end SavedOptions

/--
Load model state from disk and run sampling with the fixed byte-level GPT architecture.

The checkpoint must match `Gpt2.model`'s state shapes. If the model configuration
in `Gpt2.lean` changes (heads, width, layers, etc.), mismatched checkpoints fail the shape check
before sampling starts.
-/
def sampleCheckpoint
    (load : SavedOptions) :
    IO String := do
  nn.withModel NN.Examples.Models.Sequence.Gpt2.model fun model => do
    -- The checkpoint boundary is shape-indexed: stale files fail before sampling starts.
    let state ← Checkpoint.loadModelState model load.checkpointPath
    let graph ← nn.lowerToTypedGraph model (α := Float)
    let predict : NN.Examples.Models.Sequence.Gpt2.Predictor :=
      fun x => pure <| graph.forward state x
    let outIds ←
      NN.Examples.Models.Sequence.Gpt2.generateSampled predict load.prompt load.generate
        load.temperature load.topK load.seed load.repeatWindow load.repeatPenalty load.asciiOnly
    let txt := text.escapeByteIdsForDisplay outIds
    IO.println s!"  loaded={load.checkpointPath}"
    IO.println s!"  prompt={text.escapeForDisplay load.prompt}"
    IO.println s!"  sampled={txt}"
    pure txt

/-- CLI entrypoint for checkpoint sampling. -/
def main (args : List String) : IO UInt32 := do
  if args.contains "--help" || args.contains "-h" then
    IO.println usage
    return 0
  Module.Command.runFloat32 exeName args
    (banner := fun _ => s!"{exeName}: sample from a model checkpoint")
    (k := fun _opts rest => do
      let (load, rest) ← ModelZoo.orThrow exeName <|
        SavedOptions.parse rest
          { prompt := "First Citizen:"
            generate := 96
            temperature := 0.85
            topK := 12
            repeatPenalty := 1.25
            repeatWindow := 24
            seed := 7
            asciiOnly := false }
      CLI.requireNoArgs exeName rest
      let _ ← sampleCheckpoint load)

end NN.Examples.Models.Sequence.Gpt2Saved

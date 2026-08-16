# Sequence Examples

This folder contains runnable sequence-model examples: recurrent networks, Transformer blocks,
causal language models, a Mamba-style model, and a small arithmetic curriculum. These examples are
where tokenization, sequence length, causal masking, recurrent state, cache-like behavior, and
generation options enter the model zoo.

## Main Entry Points

- `Rnn.lean` (`lake exe torchlean rnn ...`): a small recurrent text-window example.
- `Lstm.lean` (`lake exe torchlean lstm ...`): LSTM text-window training with a time-distributed
  head.
- `Transformer.lean` (`lake exe torchlean transformer ...`): a compact transformer encoder block
  over a real text window.
- `CharGpt.lean` (`lake exe torchlean chargpt ...`): character-level GPT training with a two-update
  smoke preset and a full Tiny Shakespeare lecture preset.
- `Gpt2.lean` (`lake exe torchlean gpt2 ...`): a compact byte-level GPT-2-style causal Transformer
  with save/reload support.
- `Gpt2Saved.lean` (`lake exe torchlean gpt2_saved ...`): load weights saved by `gpt2` and sample.
- `TextGpt2.lean` (`lake exe torchlean text_gpt2 ...`): CUDA corpus trainer with byte-level tokens
  or GPT-2 BPE.
- `GptAdder.lean` (`lake exe torchlean gpt_adder ...`): a synthetic addition curriculum for
  next-token models.
- `Mamba.lean` (`lake exe torchlean mamba ...`): compact text training for the Mamba-style state
  model.

## Why There Are Several GPT Paths

They test different pieces of the stack.

- `chargpt` is the smallest single-file character path. It is good for understanding the data flow
  and for keeping a fast sequence-model check in the runner.
- `gpt2` is the compact byte-level causal Transformer path. It supports saving parameters and
  reloading through `gpt2_saved`.
- `text_gpt2` is the larger corpus trainer. It can use raw byte tokens or GPT-2 BPE tokenizer files
  and is intended for CUDA-backed runs.
- `gpt_adder` uses a controlled arithmetic next-token task to check sequence models, optimizers,
  and generation plumbing on a tiny symbolic curriculum.

## Token And Artifact Boundaries

Sequence examples are especially sensitive to hidden conventions. The docs and command output should
make these objects visible:

- tokenizer choice: byte tokens, character tokens, or GPT-2 BPE files;
- sequence length and batch shape;
- causal mask semantics;
- shifted target construction for next-token prediction;
- generation settings such as temperature, top-k, repeat penalty, and seed;
- saved parameter paths for reloadable runs.

The byte-token examples keep the vocabulary fixed at 256 so the boundary is small. BPE examples use
explicit `vocab.json` and `merges.txt` files under `data/real/gpt2/`. Saved-parameter examples
reload a shape-indexed parameter pack and fail before sampling if the saved shapes no longer match
the model architecture.

The causal-transformer training command uses `CausalTransformer.Indexed`. Its batches contain
`Tensor Nat` token IDs, the embedding layer gathers table rows directly, and repeated IDs
scatter-add into the same weight-gradient row. It does not allocate one-hot token vectors. The
separate `oneHot` constructor remains available for proofs and small examples whose inputs are
already one-hot tensors.

Useful commands:

```bash
lake exe torchlean rnn --device cpu --steps 1
lake -R -K cuda=true exe torchlean transformer --device cuda --tiny-shakespeare --steps 1
lake -R -K cuda=true exe torchlean gpt2 --device cuda --steps 10 --generate 0
lake -R -K cuda=true exe torchlean text_gpt2 --device cuda --data-file data/real/text/tinystories_valid.txt --allow-small-data --steps 1 --generate 0
lake -R -K cuda=true exe torchlean mamba --device cuda --tiny-shakespeare --steps 1 --windows 1 --generate 0
```

For a save/reload check:

```bash
lake -R -K cuda=true exe torchlean gpt2 --device cuda --tiny-shakespeare \
  --steps 1 --windows 1 --generate 0 --save-checkpoint /tmp/gpt2.state.json

lake -R -K cuda=true exe torchlean gpt2_saved --device cuda \
  --checkpoint /tmp/gpt2.state.json --prompt "ROMEO:" --generate 16
```

## What To Inspect

| Artifact | Why it matters |
| --- | --- |
| Training log JSON | Shows loss over steps and records run metadata. |
| Saved params JSON | Tests the shape-indexed parameter boundary used by reload/sampling workflows. |
| Generated text | Qualitative preview of the runtime path; theorem-level claims should cite the owning spec/proof files. |
| Tokenizer files | Define the vocabulary boundary for BPE models. |
| Lean source shapes | Show batch, sequence, vocabulary, and logits dimensions at the type level. |

Sequence models also touch verification problems: causal masks, KV/cache position contracts,
tokenizer boundaries, and batch invariance. Those contracts live in the spec/proof/BugZoo layers.
The model commands here are runtime producers and regression examples; theorem statements should be
cited from the files that own those contracts.

Supervised time-series forecasting lives in
`NN/Examples/Models/Supervised/LstmRegression.lean`, although it uses an LSTM layer, because that
example is about forecasting targets from windows rather than sequence-model mechanics.

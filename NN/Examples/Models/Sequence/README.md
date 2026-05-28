# Sequence Examples (TorchLean)

This folder contains the runnable sequence model examples.

Main entrypoints:

- `CharGpt.lean` (`lake exe torchlean chargpt ...`): the minGPT-style character GPT path.
- `Gpt2.lean` (`lake exe torchlean gpt2 ...`): a compact byte-level GPT-2-style causal Transformer.
- `Gpt2Saved.lean` (`lake exe torchlean gpt2_saved ...`): load weights saved by `gpt2` and sample.
- `TextGpt2.lean` (`lake exe torchlean text_gpt2 ...`): CUDA only corpus trainer, with byte level
  tokens or GPT-2 BPE.
- `Mamba.lean` (`lake exe torchlean mamba ...`): compact text training for the Mamba-style model.

Why there are multiple GPT like files:

- They use different *tokenizers* and *intended scales*.
- `chargpt` is the single-file corpus path with an alphabet tokenizer.
- `gpt2` is the byte-level Transformer path with save/reload support.
- `text_gpt2` is the "trainer interface" for larger corpora and optional GPT-2 BPE.

Other sequence examples are here too (RNN/LSTM layer checks, a transformer block, and the
`gpt_adder` curriculum). Supervised time series forecasting lives in
`NN/Examples/Models/Supervised/LstmRegression.lean`, although it uses an LSTM layer.

# `NN.Spec.Module`

This directory is TorchLean's module-wrapper layer for the spec codebase. Most mathematical
definitions live in `NN/Spec/Layers/` and `NN/Spec/Core/`; this folder packages selected specs into
a uniform shape-indexed module interface:

```lean
ModSpec.NNModuleSpec α inShape outShape
```

That lets TorchLean compose blocks with `ModSpec.SpecChain`, attach export/pretty-printing
metadata, and keep model-shaped specs close to the public `nn.Sequential` style. Evaluation remains
defined by the wrapped specification.

## Files

- `SpecModule.lean`: `NNModuleSpec`, `SpecChain`, composition, and evaluation helpers.
- `Activation.lean`: wrappers for ReLU, sigmoid, tanh, softmax, and related activations.
- `Linear.lean`: linear layer wrappers and small sequence classifier helpers.
- `Conv.lean`: convolution and transpose-convolution wrappers.
- `Pooling.lean`, `GlobalPooling.lean`: rank-generic pooling wrappers over channels-first tensors.
- `Normalization.lean`: layer norm and related normalization wrappers.
- `Attention.lean`: scaled dot-product self-attention wrapper.
- `Embedding.lean`: one-hot/numeric embedding wrapper.
- `PositionalEncoding.lean`: learnable positional encodings.
- `Rnn.lean`, `RnnModels.lean`, `LstmModels.lean`, `GruModels.lean`: recurrent wrappers and
  recurrent model shapes.
- `Dropout.lean`: deterministic inference and explicit-mask dropout wrappers.
- `Flatten.lean`: flattening wrappers used by CNN/model-zoo specs.
- `Gnn.lean`: compact graph-convolution wrapper.
- `Autoencoder.lean`, `Seq2seq.lean`: larger neural model wrappers.
- `LinearRegression.lean`, `LogisticRegression.lean`, `Svm.lean`, `DecisionTree.lean`,
  `GradientBoostedTrees.lean`, `Pca.lean`, `Gmm.lean`, `Hmm.lean`: classical ML and probabilistic
  model specs that share the same typed-tensor vocabulary.

The `kind` and `export_func` fields are metadata for tooling. They are not part of the mathematical
meaning of a module. The meaning comes from the wrapped spec function and its input/output shapes.

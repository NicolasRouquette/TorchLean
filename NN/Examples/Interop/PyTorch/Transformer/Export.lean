/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Export.Core
public import NN.Spec.Models.Transformer

/-!
# Transformer PyTorch Reference Export

PyTorch code generator for the Transformer encoder round-trip reference model.

This file produces a readable Python `nn.Module` implementation that follows the usual PyTorch
structure (MHA + residual + LayerNorm + FFN). In the TorchLean repo we mostly use this as a
round-trip companion: generate a reference implementation, train/tweak in Python if needed, and
optionally export parameters back to Lean via JSON in the importer modules.
-/

@[expose] public section


namespace Export
namespace TransformerPyTorch

open Spec
open Tensor
open Export.PyTorch

/-- Render a small Transformer encoder as a Python `nn.Module` class definition.

This produces readable "reference PyTorch" code (MultiHeadAttention + residual + LayerNorm + FFN),
useful for round-trip examples.
-/
def generateTransformerEncoderPyTorchClass (seqLen embedDim headCount hiddenDim numLayers : Nat)
  (className : String := "TransformerEncoder") : String :=
  joinLines <|
  #[generatePyTorchImports, "import math", ""] ++ #[
    "class MultiHeadAttention(nn.Module):",
    indentTwo s!"def __init__(self, embed_dim={embedDim}, num_heads={headCount}):",
    indentFour "super().__init__()",
    indentFour "self.embed_dim = embed_dim",
    indentFour "self.num_heads = num_heads",
    indentFour "self.head_dim = embed_dim // num_heads",
    indentFour "assert embed_dim % num_heads == 0",
    -- TorchLean's spec uses bias-free attention projections (explicit matrices).
    indentFour "self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)",
    indentFour "self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)",
    indentFour "self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)",
    indentFour "self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)",
    indentTwo "",
    indentTwo "def forward(self, x, mask=None):",
    indentFour "B, S, E = x.shape",
    indentFour "q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)",
    indentFour "k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)",
    indentFour "v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)",
    indentFour "scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)",
    indentFour "if mask is not None:",
    indentSix "scores = scores.masked_fill(mask == 0, float('-inf'))",
    indentFour "attn = torch.softmax(scores, dim=-1)",
    indentFour "out = torch.matmul(attn, v)",
    indentFour "out = out.transpose(1, 2).contiguous().view(B, S, E)",
    indentFour "return self.out_proj(out)",
    "",
    "class FeedForward(nn.Module):",
    indentTwo s!"def __init__(self, embed_dim={embedDim}, hidden_dim={hiddenDim}):",
    indentFour "super().__init__()",
    indentFour "self.fc1 = nn.Linear(embed_dim, hidden_dim)",
    indentFour "self.fc2 = nn.Linear(hidden_dim, embed_dim)",
    indentTwo "",
    indentTwo "def forward(self, x):",
    indentFour "return self.fc2(F.relu(self.fc1(x)))",
    "",
    s!"class {className}(nn.Module):",
    indentTwo (s!"\"\"\"Transformer Encoder with {numLayers} layers, {headCount} heads, " ++
      s!"embed dim {embedDim}, hidden dim {hiddenDim}\"\"\""),
    indentTwo "",
    indentTwo s!"def __init__(self):",
    indentFour "super().__init__()",
    indentFour "self.layers = nn.ModuleList([nn.ModuleDict({",
    indentSix s!"'mha': MultiHeadAttention({embedDim}, {headCount}),",
    indentSix s!"'norm1': nn.LayerNorm({embedDim}),",
    indentSix s!"'ffn': FeedForward({embedDim}, {hiddenDim}),",
    indentSix s!"'norm2': nn.LayerNorm({embedDim})",
    indentFour s!"}) for _ in range({numLayers})])",
    indentTwo "",
    indentTwo "def forward(self, x, mask=None):",
    indentFour "# x: (batch, seq_len, embed_dim)",
    indentFour "for layer in self.layers:",
    indentSix "# Self-attention block",
    indentSix "attn_out = layer['mha'](x, mask)",
    indentSix "x = layer['norm1'](x + attn_out)",
    indentSix "# Feed-forward block",
    indentSix "ffn_out = layer['ffn'](x)",
    indentSix "x = layer['norm2'](x + ffn_out)",
    indentFour "return x",
    indentTwo "",
    indentTwo "@property",
    indentTwo "def input_shape(self):",
    indentFour s!"return ({seqLen}, {embedDim})",
    indentFour "",
    indentTwo "@property",
    indentTwo "def output_shape(self):",
    indentFour s!"return ({seqLen}, {embedDim})",
    indentFour "",
    indentTwo "@property",
    indentTwo "def layer_count(self):",
    indentFour s!"return {numLayers}",
    indentFour "",
    indentTwo "@property",
    indentTwo "def operation_types(self):",
    indentFour
      "return ['MultiHeadAttention', 'LayerNorm', 'FeedForward', 'LayerNorm'] * self.layer_count",
    indentFour ""
  ] ++
    generateGetModelInfoMethodLines className

/--
Generate a single-layer Transformer encoder module with an embedded `state_dict` initializer.

This is meant for round-trip examples where parameters are loaded from TorchLean tensors.

Important convention:
`NN/Spec` transformer weights are stored in the mathematical `(in, out)` orientation because they
are applied as `X * W`. PyTorch stores linear weights as `(out, in)` and applies them as
`X @ W.T + b`. So for all matrix-valued weights we print the transpose when populating the
PyTorch `state_dict`.
-/
def generateTransformerEncoderWithWeights (seqLen embedDim headCount hiddenDim : Nat)
  (queryWeight keyWeight valueWeight outputWeight :
    Tensor Float [embedDim, embedDim])
  (feedForwardInputWeight : Tensor Float [embedDim, hiddenDim])
  (feedForwardOutputWeight : Tensor Float [hiddenDim, embedDim])
  (feedForwardInputBias : Tensor Float [hiddenDim])
  (feedForwardOutputBias norm1Scale norm1Bias norm2Scale norm2Bias :
    Tensor Float [embedDim])
  (className : String := "TransformerEncoder") : String :=
  joinLines #[
    generateTransformerEncoderPyTorchClass seqLen embedDim headCount hiddenDim 1 className,
    "",
    "# Weight initialization helpers",
    "def get_transformer_state_dict():",
    indentTwo "state_dict = {}",
    indentTwo s!"state_dict['layers.0.mha.q_proj.weight'] = torch.tensor({transposedMatrixTensorToPy queryWeight})",
    indentTwo s!"state_dict['layers.0.mha.k_proj.weight'] = torch.tensor({transposedMatrixTensorToPy keyWeight})",
    indentTwo s!"state_dict['layers.0.mha.v_proj.weight'] = torch.tensor({transposedMatrixTensorToPy valueWeight})",
    indentTwo s!"state_dict['layers.0.mha.out_proj.weight'] = torch.tensor({transposedMatrixTensorToPy outputWeight})",
    indentTwo s!"state_dict['layers.0.ffn.fc1.weight'] = torch.tensor({transposedMatrixTensorToPy feedForwardInputWeight})",
    indentTwo s!"state_dict['layers.0.ffn.fc1.bias'] = torch.tensor({tensorToPyString feedForwardInputBias})",
    indentTwo s!"state_dict['layers.0.ffn.fc2.weight'] = torch.tensor({transposedMatrixTensorToPy feedForwardOutputWeight})",
    indentTwo s!"state_dict['layers.0.ffn.fc2.bias'] = torch.tensor({tensorToPyString feedForwardOutputBias})",
    indentTwo s!"state_dict['layers.0.norm1.weight'] = torch.tensor({tensorToPyString norm1Scale})",
    indentTwo s!"state_dict['layers.0.norm1.bias'] = torch.tensor({tensorToPyString norm1Bias})",
    indentTwo s!"state_dict['layers.0.norm2.weight'] = torch.tensor({tensorToPyString norm2Scale})",
    indentTwo s!"state_dict['layers.0.norm2.bias'] = torch.tensor({tensorToPyString norm2Bias})",
    indentTwo "return state_dict",
    indentTwo "",
    "def load_transformer_weights(model):",
    indentTwo "model.load_state_dict(get_transformer_state_dict())",
    indentTwo "return model",
    indentTwo "",
    "# Usage example",
    "if __name__ == \"__main__\":",
    indentTwo s!"model = {className}()",
    indentTwo "model = load_transformer_weights(model)",
    indentTwo s!"x = torch.randn(1, {seqLen}, {embedDim})  # batch=1, seq_len={seqLen}, embed_dim={embedDim}",
    indentTwo "y = model(x)",
    indentTwo "print(f\"Input shape: {x.shape}\")",
    indentTwo "print(f\"Output shape: {y.shape}\")",
    indentTwo "print(f\"Output: {y}\")",
    indentTwo "print(f\"Model info: {model.get_model_info()}\")"
  ]

end TransformerPyTorch
end Export

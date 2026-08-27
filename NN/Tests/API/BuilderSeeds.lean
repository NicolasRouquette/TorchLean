/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Models.FNO
public import NN.API.Models.Generative
public import NN.API.Models.Mamba

/-!
# Builder Seed API Tests

Regression checks for public model constructors whose initialization must advance `nn.Builder`.
-/

@[expose] public section

namespace NN.Tests.API.BuilderSeeds

open TorchLean

universe u

def expectCounter (tag : String) (expected actual : Nat) : IO Unit := do
  unless actual == expected do
    throw <| IO.userError
      s!"builder seed check failed: {tag} (expected {expected}, got {actual})"

def counterAfter {α : Type u} (builder : nn.Builder α) (seed : Nat) : Nat :=
  (builder (rand.SeedStream.init seed)).2.counter

def fnoConfig : nn.models.FnoConfig 1 :=
  { spatial := Spec.fill 2 [1]
    modes := Spec.fill 1 [1]
    spatialNonzero := by intro i; fin_cases i; decide
    modesFit := by intro i; fin_cases i; decide
    width := 1
    widthNonzero := by decide
    blocks := 0 }

def convConfig : nn.Conv 1 :=
  { outChannels := 1
    kernel := Spec.fill 1 [1]
    kernelNonzero := by intro i; fin_cases i; decide
    strideNonzero := by intro i; fin_cases i; decide }

def generativeConfig : nn.models.DenseGenerative.Config :=
  { dataDim := 4, hiddenDim := 3, latentDim := 2 }

def reconstructionModel :=
  nn.models.DenseGenerative.reconstructionWithLatentStatistics generativeConfig

def tanhAutoencoderModel :=
  nn.models.DenseGenerative.tanhBottleneckAutoencoder generativeConfig

def mambaConfig : nn.models.Mamba.Config :=
  { vocab := 4, stateDim := 2 }

def selectiveMambaConfig : nn.models.Mamba.Reference.SelectiveConfig :=
  { vocab := 4, stateDim := 2, ssmStateDim := 2, convWidth := 3 }

def selectiveMambaReference :=
  nn.models.Mamba.Reference.selective selectiveMambaConfig

def run : IO Unit := do
  expectCounter "FNO consumes one base seed" 1 (counterAfter (nn.models.fno fnoConfig) 11)
  expectCounter "convolution consumes kernel and bias seeds" 2 <|
    counterAfter (nn.conv (leading := []) (inChannels := 1) (Spec.fill 2 [1]) convConfig) 11
  expectCounter "RMSNorm consumes one scale seed" 1 (counterAfter (nn.rmsNorm (width := 2)) 11)

end NN.Tests.API.BuilderSeeds

/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.NN

/-!
# TorchLean SSL and Diffusion API

Self-supervised and diffusion objective adapters exposed by the `NN` umbrella.
-/

@[expose] public section

namespace TorchLean

namespace ssl

export NN.API.ssl
  (vectorMaeHiddenMask vectorMaeMask vectorMaeSample
   blockIndex blockHidden blockMask blockMaskBatch blockMaeSample)

end ssl

namespace diffusion

export NN.API.diffusion
  (toMinusOneOne randomEps linearBeta alphaBarsLinear appendTimeChannel noisedSampleFromEps
   noisedSample ddimPrev writeFirstRgbPpm)

end diffusion


end TorchLean

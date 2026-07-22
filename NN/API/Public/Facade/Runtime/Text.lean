/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.NN

/-!
# TorchLean Text Runtime API

Tokenizer, corpus, generation, and interactive text options.
-/

@[expose] public section

namespace TorchLean

namespace text

export NN.API.text
  (Tokenizer oneHotTokenFloat causalLmXOneHotBatch causalLmXOneHotBatchRows causalLmXYOneHotMatFloat
   causalLmSampleOneHotBatchRows causalLmSampleOneHotBatch byteTokenWindow tokenWindow
   decodeWindow decodeArgmaxLogits
   escapeForDisplay escapeByteIdsForDisplay printableAsciiByte argmaxTokenIdsFromBatchLogits
   logitScoresAt batchLogitScoresAt
   topKIndices sampleTopKIndex
   chooseNextToken autoregressiveTokenIds parseGenerationOptions GenerationOptions causalMask
   TextCorpusOptions TextCorpusPathOptions FinetuneOptions BpeCorpusOptions InteractiveOptions
   PromptGenerationOptions SavedParamsGenerationOptions
   generationNotes promptGenerationNotes writeGenerationTrainLog writePromptTrainLog
   LoggedInteractiveOptions InteractiveTrainOptions LoggedPromptInteractiveOptions
   CorpusLoggedPromptInteractiveOptions TrainGenerationOptions WindowedTrainGenerationOptions
   CheckpointedWindowedTrainGenerationOptions
   BatchedCheckpointedWindowedTrainGenerationOptions
   InteractiveCheckpointedWindowedTrainGenerationOptions
   mkLoggedInteractiveOptions mkInteractiveTrainOptions mkLoggedPromptInteractiveOptions
   mkTrainGenerationOptions mkWindowedTrainGenerationOptions
   mkCheckpointedWindowedTrainGenerationOptions
   mkInteractiveCheckpointedWindowedTrainGenerationOptions
   LocalBpeVocab buildLocalBpeVocab localizeBpeTokens)

namespace Tokenizer

export NN.API.text.Tokenizer
  (byte ofAlphabet encodeVec encodeBatchVec)

end Tokenizer

namespace GenerationOptions

export NN.API.text.GenerationOptions
  (toDefaults parse)

end GenerationOptions

namespace TextCorpusOptions

export NN.API.text.TextCorpusOptions
  (parse)

end TextCorpusOptions

namespace TextCorpusPathOptions

export NN.API.text.TextCorpusPathOptions
  (parse)

end TextCorpusPathOptions

namespace FinetuneOptions

export NN.API.text.FinetuneOptions
  (parse)

end FinetuneOptions

namespace BpeCorpusOptions

export NN.API.text.BpeCorpusOptions
  (parse)

end BpeCorpusOptions

namespace InteractiveOptions

export NN.API.text.InteractiveOptions
  (parse)

end InteractiveOptions

namespace PromptGenerationOptions

export NN.API.text.PromptGenerationOptions
  (parse)

end PromptGenerationOptions

namespace SavedParamsGenerationOptions

export NN.API.text.SavedParamsGenerationOptions
  (parse)

end SavedParamsGenerationOptions

namespace InteractiveTrainOptions

@[inherit_doc NN.API.text.InteractiveTrainOptions.parse]
def parse
    (exeName : String)
    (args : List String)
    (defaultLogJson : System.FilePath)
    (defaultSteps : Nat)
    (defaultLr : Float)
    (allowZeroSteps : Bool := false) :
    Except String (InteractiveTrainOptions × List String) :=
  NN.API.text.InteractiveTrainOptions.parse exeName args defaultLogJson defaultSteps defaultLr
    (allowZeroSteps := allowZeroSteps)

end InteractiveTrainOptions

namespace LoggedPromptInteractiveOptions

export NN.API.text.LoggedPromptInteractiveOptions
  (parse)

end LoggedPromptInteractiveOptions

namespace CorpusLoggedPromptInteractiveOptions

export NN.API.text.CorpusLoggedPromptInteractiveOptions
  (parse)

end CorpusLoggedPromptInteractiveOptions

namespace WindowedTrainGenerationOptions

@[inherit_doc NN.API.text.WindowedTrainGenerationOptions.parse]
def parse
    (exeName : String)
    (args : List String)
    (defaultLogJson : System.FilePath)
    (defaultSteps : Nat)
    (defaultLr : Float)
    (defaultWindows : Nat)
    (genDefaults : GenerationOptions)
    (allowZeroSteps : Bool := false) :
    Except String (WindowedTrainGenerationOptions × List String) :=
  NN.API.text.WindowedTrainGenerationOptions.parse exeName args defaultLogJson defaultSteps
    defaultLr defaultWindows genDefaults (allowZeroSteps := allowZeroSteps)

end WindowedTrainGenerationOptions

namespace CheckpointedWindowedTrainGenerationOptions

@[inherit_doc NN.API.text.CheckpointedWindowedTrainGenerationOptions.parse]
def parse
    (exeName : String)
    (args : List String)
    (defaultLogJson : System.FilePath)
    (defaultSteps : Nat)
    (defaultLr : Float)
    (defaultWindows : Nat)
    (genDefaults : GenerationOptions)
    (allowZeroSteps : Bool := false) :
    Except String (CheckpointedWindowedTrainGenerationOptions × List String) :=
  NN.API.text.CheckpointedWindowedTrainGenerationOptions.parse exeName args defaultLogJson
    defaultSteps defaultLr defaultWindows genDefaults (allowZeroSteps := allowZeroSteps)

end CheckpointedWindowedTrainGenerationOptions

namespace BatchedCheckpointedWindowedTrainGenerationOptions

@[inherit_doc NN.API.text.BatchedCheckpointedWindowedTrainGenerationOptions.parse]
def parse
    (exeName : String)
    (args : List String)
    (defaultLogJson : System.FilePath)
    (defaultSteps : Nat)
    (defaultLr : Float)
    (defaultWindows : Nat)
    (defaultBatch : Nat)
    (defaultSeqLen : Nat)
    (genDefaults : GenerationOptions)
    (allowZeroSteps : Bool := false) :
    Except String (BatchedCheckpointedWindowedTrainGenerationOptions × List String) :=
  NN.API.text.BatchedCheckpointedWindowedTrainGenerationOptions.parse exeName args defaultLogJson
    defaultSteps defaultLr defaultWindows defaultBatch defaultSeqLen genDefaults
    (allowZeroSteps := allowZeroSteps)

end BatchedCheckpointedWindowedTrainGenerationOptions

namespace InteractiveCheckpointedWindowedTrainGenerationOptions

@[inherit_doc NN.API.text.InteractiveCheckpointedWindowedTrainGenerationOptions.parse]
def parse
    (exeName : String)
    (args : List String)
    (defaultLogJson : System.FilePath)
    (defaultSteps : Nat)
    (defaultLr : Float)
    (defaultWindows : Nat)
    (genDefaults : GenerationOptions)
    (allowZeroSteps : Bool := false) :
    Except String (InteractiveCheckpointedWindowedTrainGenerationOptions × List String) :=
  NN.API.text.InteractiveCheckpointedWindowedTrainGenerationOptions.parse exeName args
    defaultLogJson defaultSteps defaultLr defaultWindows genDefaults
    (allowZeroSteps := allowZeroSteps)

end InteractiveCheckpointedWindowedTrainGenerationOptions

namespace Corpus

export NN.API.text.Corpus
  (takeUtf8Input readByteFile findWindow? promptAwareOffsets usableTokenStarts
   byteOffset randomBatchTokenWindows tokenArrayWindow)

end Corpus

namespace Gpt2Bpe

export NN.API.text.Gpt2Bpe
  (Tokenizer parseVocabText parseMerges vocabMapOf idMapOf mergeMapOf
   mkTokenizer load loadWithProgress encode decodeD)

end Gpt2Bpe

end text


end TorchLean

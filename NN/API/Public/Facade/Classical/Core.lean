/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models

/-!
# Classical Models

User-facing names for TorchLean's classical and statistical model definitions.

The implementations remain in `NN.Spec.Models`, where their mathematical definitions and proofs
can refer to them directly. This API gives application code a consistent path without creating
a second implementation. Neural-network constructors live under `TorchLean.nn.models`; these
models live under `TorchLean.classical`.

The public families are:

- `knn` and `randomForest` for nearest-neighbor and tree ensembles;
- `naiveBayes`, `svm`, and `logisticRegression` for classification;
- `gmm`, `pca`, and `linearRegression` for statistical modeling and dimensionality reduction;
- `gradientBoostedTrees` for regression and classification ensembles;
- `hmm` for finite-state sequence models; and
- `hopfield` for associative-memory dynamics and their executable counterpart.
-/

@[expose] public section

namespace TorchLean.classical

/-!
## K-nearest neighbors

Stores labeled examples and predicts from the nearest points under the model's distance function.
The API includes unweighted and distance-weighted prediction, classification, confidence values,
and batched evaluation.
-/
namespace knn

@[inherit_doc Spec.KNN]
abbrev Model := Spec.KNN

@[inherit_doc Spec.KNN.fromData]
abbrev fromData := @Spec.KNN.fromData
@[inherit_doc Spec.findKNearest]
abbrev nearest := @Spec.findKNearest
@[inherit_doc Spec.findKNearestWithDistance]
abbrev nearestWithDistance := @Spec.findKNearestWithDistance
@[inherit_doc Spec.classify]
abbrev classify := @Spec.classify
@[inherit_doc Spec.classifyTreeMap]
abbrev classifyTreeMap := @Spec.classifyTreeMap
@[inherit_doc Spec.predict]
abbrev predict := @Spec.predict
@[inherit_doc Spec.predictWeighted]
abbrev predictWeighted := @Spec.predictWeighted
@[inherit_doc Spec.batchPredict]
abbrev batchPredict := @Spec.batchPredict
@[inherit_doc Spec.batchClassify]
abbrev batchClassify := @Spec.batchClassify
@[inherit_doc Spec.classifyWithDistance]
abbrev classifyWithDistance := @Spec.classifyWithDistance
@[inherit_doc Spec.classifyWithConfidence]
abbrev classifyWithConfidence := @Spec.classifyWithConfidence

end knn

/-!
## Random forests

The generic forest API evaluates trees with arbitrary leaf values. The `regression` and
`classification` namespaces specialize it with numeric prediction and fitting objectives.
-/
namespace randomForest

@[inherit_doc random_forest.RandomForest]
abbrev Model := random_forest.RandomForest

@[inherit_doc random_forest.predict]
abbrev predict := @random_forest.predict
@[inherit_doc random_forest.majorityVote]
abbrev majorityVote := @random_forest.majorityVote
@[inherit_doc random_forest.average]
abbrev average := @random_forest.average

namespace regression

@[inherit_doc random_forest.Numeric.RegressionForestSpec]
abbrev Model := random_forest.Numeric.RegressionForestSpec
@[inherit_doc random_forest.Numeric.regressionForestForwardSpec]
abbrev forward := @random_forest.Numeric.regressionForestForwardSpec
@[inherit_doc random_forest.Numeric.regressionForestFitRegressionMseSpec]
abbrev fitMSE := @random_forest.Numeric.regressionForestFitRegressionMseSpec

end regression

namespace classification

@[inherit_doc random_forest.Numeric.ClassificationForestSpec]
abbrev Model := random_forest.Numeric.ClassificationForestSpec
@[inherit_doc random_forest.Numeric.classificationForestPredictSpec]
abbrev predict := @random_forest.Numeric.classificationForestPredictSpec
@[inherit_doc random_forest.Numeric.classificationForestFitClassificationGiniSpec]
abbrev fitGini := @random_forest.Numeric.classificationForestFitClassificationGiniSpec

end classification
end randomForest

/-!
## Naive Bayes

Fits class-conditional feature statistics, scores examples, predicts classes, and evaluates the
negative log-likelihood objective.
-/
namespace naiveBayes

@[inherit_doc NaiveBayes.Example]
abbrev Example := NaiveBayes.Example
@[inherit_doc NaiveBayes.Model]
abbrev Model := NaiveBayes.Model
@[inherit_doc NaiveBayes.fit]
abbrev fit := @NaiveBayes.fit
@[inherit_doc NaiveBayes.score]
abbrev score := @NaiveBayes.score
@[inherit_doc NaiveBayes.predictModel]
abbrev predict := @NaiveBayes.predictModel
@[inherit_doc NaiveBayes.negLogLikelihood]
abbrev negativeLogLikelihood := @NaiveBayes.negLogLikelihood

end naiveBayes

/-!
## Support-vector machines

Provides linear SVM decision functions, hinge objectives, fitting, prediction, support-vector
selection, and the linear, polynomial, and radial-basis kernels used by kernel SVMs.
-/
namespace svm

@[inherit_doc _root_.LinearSVM]
abbrev LinearModel := _root_.LinearSVM
@[inherit_doc _root_.SVM]
abbrev Model := _root_.SVM
@[inherit_doc _root_.LinearSVM.decision]
abbrev decision := @_root_.LinearSVM.decision
@[inherit_doc _root_.LinearSVM.decisionBatch]
abbrev decisionBatch := @_root_.LinearSVM.decisionBatch
@[inherit_doc _root_.hingeLossPerExample]
abbrev hingeLoss := @_root_.hingeLossPerExample
@[inherit_doc _root_.hingeLossMean]
abbrev meanHingeLoss := @_root_.hingeLossMean
@[inherit_doc _root_.LinearSVM.objective]
abbrev objective := @_root_.LinearSVM.objective
@[inherit_doc _root_.LinearSVM.backward]
abbrev backward := @_root_.LinearSVM.backward
@[inherit_doc _root_.fitLinearSVM]
abbrev fit := @_root_.fitLinearSVM
@[inherit_doc _root_.predict]
abbrev predict := @_root_.predict
@[inherit_doc _root_.findSupportVectorIndices]
abbrev supportVectorIndices := @_root_.findSupportVectorIndices

namespace kernel

@[inherit_doc _root_.Kernel.linear]
abbrev linear := @_root_.Kernel.linear
@[inherit_doc _root_.Kernel.polynomial]
abbrev polynomial := @_root_.Kernel.polynomial
@[inherit_doc _root_.Kernel.rbf]
abbrev rbf := @_root_.Kernel.rbf

end kernel
end svm

/-!
## Gaussian mixture models

Exposes mixture evaluation, responsibilities, log likelihood, initialization, backward equations,
and expectation-maximization steps for individual and batched inputs.
-/
namespace gmm

@[inherit_doc Spec.GMMSpec]
abbrev Model := Spec.GMMSpec
@[inherit_doc Spec.gmmForwardSpec]
abbrev forward := @Spec.gmmForwardSpec
@[inherit_doc Spec.gmmExpectationSpec]
abbrev expectation := @Spec.gmmExpectationSpec
@[inherit_doc Spec.gmmBatchedForwardSpec]
abbrev batchForward := @Spec.gmmBatchedForwardSpec
@[inherit_doc Spec.gmmBackwardSpec]
abbrev backward := @Spec.gmmBackwardSpec
@[inherit_doc Spec.gmmInitSpec]
abbrev init := @Spec.gmmInitSpec
@[inherit_doc Spec.gmmLogLikelihoodSpec]
abbrev logLikelihood := @Spec.gmmLogLikelihoodSpec
@[inherit_doc Spec.gmmResponsibilitiesBatchedSpec]
abbrev responsibilities := @Spec.gmmResponsibilitiesBatchedSpec
@[inherit_doc Spec.gmmEmStepSpec]
abbrev emStep := @Spec.gmmEmStepSpec
@[inherit_doc Spec.gmmEmTrainSpec]
abbrev trainEM := @Spec.gmmEmTrainSpec

end gmm

/-!
## Principal component analysis

Provides projection, inverse projection, a one-component power-iteration approximation,
reconstruction error, and explained-variance statistics for PCA models.
-/
namespace pca

@[inherit_doc Spec.PCASpec]
abbrev Model := Spec.PCASpec
@[inherit_doc Spec.pcaForwardSpec]
abbrev forward := @Spec.pcaForwardSpec
@[inherit_doc Spec.pcaInverseSpec]
abbrev inverse := @Spec.pcaInverseSpec
@[inherit_doc Spec.pcaBackwardSpec]
abbrev backward := @Spec.pcaBackwardSpec
@[inherit_doc Spec.pcaFitLeadingComponentApproxSpec]
abbrev fitLeadingComponentApprox := @Spec.pcaFitLeadingComponentApproxSpec
@[inherit_doc Spec.pcaTransformSpec]
abbrev transform := @Spec.pcaTransformSpec
@[inherit_doc Spec.pcaReconstructionErrorSpec]
abbrev reconstructionError := @Spec.pcaReconstructionErrorSpec
@[inherit_doc Spec.pcaCumulativeExplainedVarianceSpec]
abbrev cumulativeExplainedVariance := @Spec.pcaCumulativeExplainedVarianceSpec

end pca

/-!
## Linear regression

Includes single and batched forward/backward equations, training steps, regression metrics, and
ridge, lasso, elastic-net, and polynomial variants.
-/
namespace linearRegression

@[inherit_doc Spec.LinearRegressionSpec]
abbrev Model := Spec.LinearRegressionSpec
@[inherit_doc Spec.linearRegressionForwardSpec]
abbrev forward := @Spec.linearRegressionForwardSpec
@[inherit_doc Spec.linearRegressionBatchedForwardSpec]
abbrev batchForward := @Spec.linearRegressionBatchedForwardSpec
@[inherit_doc Spec.linearRegressionBackwardSpec]
abbrev backward := @Spec.linearRegressionBackwardSpec
@[inherit_doc Spec.linearRegressionBatchedBackwardSpec]
abbrev batchBackward := @Spec.linearRegressionBatchedBackwardSpec
@[inherit_doc Spec.mseLossSpec]
abbrev mseLoss := @Spec.mseLossSpec
@[inherit_doc Spec.linearRegressionTrainStepSpec]
abbrev trainStep := @Spec.linearRegressionTrainStepSpec
@[inherit_doc Spec.rSquaredSpec]
abbrev rSquared := @Spec.rSquaredSpec
@[inherit_doc Spec.ridgeRegressionForwardSpec]
abbrev ridgeForward := @Spec.ridgeRegressionForwardSpec
@[inherit_doc Spec.ridgeLossSpec]
abbrev ridgeLoss := @Spec.ridgeLossSpec
@[inherit_doc Spec.lassoRegressionForwardSpec]
abbrev lassoForward := @Spec.lassoRegressionForwardSpec
@[inherit_doc Spec.lassoLossSpec]
abbrev lassoLoss := @Spec.lassoLossSpec
@[inherit_doc Spec.elasticNetLossSpec]
abbrev elasticNetLoss := @Spec.elasticNetLossSpec
@[inherit_doc Spec.polynomialFeaturesSpec]
abbrev polynomialFeatures := @Spec.polynomialFeaturesSpec
@[inherit_doc Spec.polynomialRegressionForwardSpec]
abbrev polynomialForward := @Spec.polynomialRegressionForwardSpec

end linearRegression

/-!
## Logistic regression

Fits a logistic model and evaluates class probabilities or log probabilities.
-/
namespace logisticRegression

@[inherit_doc _root_.LogisticRegression]
abbrev Model := _root_.LogisticRegression
@[inherit_doc _root_.fitLogistic]
abbrev fit := @_root_.fitLogistic
@[inherit_doc _root_.predictProba]
abbrev predictProbability := @_root_.predictProba
@[inherit_doc _root_.logPredict]
abbrev predictLogProbability := @_root_.logPredict

end logisticRegression

/-!
## Gradient-boosted trees

Defines regression and classification trees, ensemble evaluation, tree fitting, boosting steps,
losses, feature importance, and common regression metrics.
-/
namespace gradientBoostedTrees

@[inherit_doc Spec.TreeNode]
abbrev Tree := Spec.TreeNode
@[inherit_doc Spec.DecisionTreeSpec]
abbrev DecisionTree := Spec.DecisionTreeSpec
@[inherit_doc Spec.GradientBoostedTreesSpec]
abbrev Model := Spec.GradientBoostedTreesSpec
@[inherit_doc Spec.RegressionExample]
abbrev RegressionExample (α : Type) (nFeatures : Nat) :=
  Spec.RegressionExample (α := α) nFeatures
@[inherit_doc Spec.decisionTreeForwardSpecN]
abbrev forwardTree := @Spec.decisionTreeForwardSpecN
@[inherit_doc Spec.decisionTreeBatchedForwardSpecN]
abbrev batchForwardTree := @Spec.decisionTreeBatchedForwardSpecN
@[inherit_doc Spec.gradientBoostedTreesForwardSpec]
abbrev forward := @Spec.gradientBoostedTreesForwardSpec
@[inherit_doc Spec.gradientBoostedTreesBatchedForwardSpec]
abbrev batchForward := @Spec.gradientBoostedTreesBatchedForwardSpec
@[inherit_doc Spec.decisionTreeFitRegressionMseSpec]
abbrev fitRegressionTree := @Spec.decisionTreeFitRegressionMseSpec
@[inherit_doc Spec.gradientBoostedTreesTrainStepSpec]
abbrev trainStep := @Spec.gradientBoostedTreesTrainStepSpec
@[inherit_doc Spec.gradientBoostedTreesTrainStepFitSpec]
abbrev trainStepAndFit := @Spec.gradientBoostedTreesTrainStepFitSpec
@[inherit_doc Spec.gbtMseLossSpec]
abbrev mseLoss := @Spec.gbtMseLossSpec
@[inherit_doc Spec.gbtBinaryCrossentropyLossSpec]
abbrev binaryCrossEntropyLoss := @Spec.gbtBinaryCrossentropyLossSpec
@[inherit_doc Spec.computeFeatureImportanceSpec]
abbrev featureImportance := @Spec.computeFeatureImportanceSpec
@[inherit_doc Spec.gbtRSquaredSpec]
abbrev rSquared := @Spec.gbtRSquaredSpec
@[inherit_doc Spec.gbtMaeSpec]
abbrev meanAbsoluteError := @Spec.gbtMaeSpec
@[inherit_doc Spec.gbtRmseSpec]
abbrev rootMeanSquaredError := @Spec.gbtRmseSpec

namespace classification

@[inherit_doc Spec.ClassifierTreeNode]
abbrev Tree := Spec.ClassifierTreeNode
@[inherit_doc Spec.DecisionTreeClassifierSpec]
abbrev DecisionTree := Spec.DecisionTreeClassifierSpec
@[inherit_doc Spec.ClassificationExample]
abbrev Example (α : Type) (nFeatures : Nat) (β : Type) :=
  Spec.ClassificationExample (α := α) nFeatures β
@[inherit_doc Spec.decisionTreeClassifyForwardSpecN]
abbrev forwardTree := @Spec.decisionTreeClassifyForwardSpecN
@[inherit_doc Spec.decisionTreeFitClassificationGiniListSpec]
abbrev fitGini := @Spec.decisionTreeFitClassificationGiniListSpec

end classification
end gradientBoostedTrees

/-!
## Hidden Markov models

Provides scaled and unscaled forward evaluation, batched evaluation, initialization, likelihood,
and Baum-Welch update equations.
-/
namespace hmm

@[inherit_doc Spec.HMMSpec]
abbrev Model := Spec.HMMSpec
@[inherit_doc Spec.ObservationSeq]
abbrev ObservationSequence := Spec.ObservationSeq
@[inherit_doc Spec.hmmForwardScaled]
abbrev forwardScaled := @Spec.hmmForwardScaled
@[inherit_doc Spec.hmmForwardSpec]
abbrev forward := @Spec.hmmForwardSpec
@[inherit_doc Spec.hmmBatchedForwardSpec]
abbrev batchForward := @Spec.hmmBatchedForwardSpec
@[inherit_doc Spec.baumWelchStepSpec]
abbrev baumWelchStep := @Spec.baumWelchStepSpec
@[inherit_doc Spec.baumWelchEpochSpec]
abbrev baumWelchEpoch := @Spec.baumWelchEpochSpec
@[inherit_doc Spec.hmmInitSpec]
abbrev init := @Spec.hmmInitSpec
@[inherit_doc Spec.hmmLogLikelihoodSpec]
abbrev logLikelihood := @Spec.hmmLogLikelihoodSpec

end hmm

/-!
## Hopfield networks

Exposes the mathematical state and parameter types, asynchronous updates, stability, energy, and
state trajectories, together with an executable counterpart under `hopfield.executable`.
-/
namespace hopfield

@[inherit_doc Spec.Hopfield.State]
abbrev State := Spec.Hopfield.State
@[inherit_doc Spec.Hopfield.StateT]
abbrev StateTensor := Spec.Hopfield.StateT
@[inherit_doc Spec.Hopfield.Params]
abbrev Params := Spec.Hopfield.Params
@[inherit_doc Spec.Hopfield.ParamsT]
abbrev ParamTensors := Spec.Hopfield.ParamsT
@[inherit_doc Spec.Hopfield.updateAt]
abbrev updateAt := @Spec.Hopfield.updateAt
@[inherit_doc Spec.Hopfield.IsStable]
abbrev isStable := @Spec.Hopfield.IsStable
@[inherit_doc Spec.Hopfield.energy]
abbrev energy := @Spec.Hopfield.energy
@[inherit_doc Spec.Hopfield.seqStates]
abbrev states := @Spec.Hopfield.seqStates

namespace executable

@[inherit_doc Spec.Hopfield.Exec.updateAt]
abbrev updateAt := @Spec.Hopfield.Exec.updateAt
@[inherit_doc Spec.Hopfield.Exec.energy]
abbrev energy := @Spec.Hopfield.Exec.energy
@[inherit_doc Spec.Hopfield.Exec.seqStates]
abbrev states := @Spec.Hopfield.Exec.seqStates

end executable
end hopfield

end TorchLean.classical

#ifndef SNNFW_JEPA_TRAINER_H
#define SNNFW_JEPA_TRAINER_H

#include "snnfw/jepa/JepaConfig.h"
#include "snnfw/jepa/JepaSample.h"
#include <string>
#include <vector>

namespace snnfw::jepa {

struct JepaLinearLayer {
    std::vector<std::vector<double>> weights;
    std::vector<double> bias;
};

struct JepaModel {
    size_t projectionDim = 0;
    size_t contextInputDim = 0;
    size_t targetInputDim = 0;
    bool futureSummaryAligned = false;
    bool temporalSpikeCodeAligned = false;
    size_t temporalSpikeCodeDim = 0;
    bool encodeViewMetadata = true;
    double metadataScale = 0.2;
    JepaLinearLayer contextEncoder;
    JepaLinearLayer predictor;
    JepaLinearLayer targetEncoder;
};

struct JepaTrainingSummary {
    size_t sampleCount = 0;
    size_t maskedViewCount = 0;
    size_t trainingExampleCount = 0;
    size_t temporalExampleCount = 0;
    size_t fallbackMaskedExampleCount = 0;
    int epochCount = 0;
    size_t projectionDim = 0;
    double meanLoss = 0.0;
    double meanShuffledLoss = 0.0;
    double meanVariancePenalty = 0.0;
    double meanCovariancePenalty = 0.0;
    double meanVisibleNorm = 0.0;
    double meanContextNorm = 0.0;
    double meanTargetNorm = 0.0;
    double meanPredictionNorm = 0.0;
    double contextVariance = 0.0;
    double targetVariance = 0.0;
    double predictionVariance = 0.0;
    double contextEncoderWeightNorm = 0.0;
    double predictorWeightNorm = 0.0;
    double targetEncoderWeightNorm = 0.0;
};

struct JepaTrainingArtifacts {
    JepaModel model;
    JepaTrainingSummary summary;
};

struct JepaPredictionError {
    bool available = false;
    double cosine = 0.0;
    double error = 0.0;
};

JepaTrainingArtifacts trainMinimalModel(
    const std::vector<JepaLatentSample>& samples,
    const JepaConfig& config,
    const std::string& outputPath);

JepaTrainingSummary runMinimalTrainer(
    const std::vector<JepaLatentSample>& samples,
    const JepaConfig& config,
    const std::string& outputPath);

std::vector<double> encodeSample(
    const JepaLatentSample& sample,
    const JepaModel& model);

JepaPredictionError evaluateTemporalPredictionError(
    const JepaLatentSample& sample,
    const JepaModel& model);

}  // namespace snnfw::jepa

#endif  // SNNFW_JEPA_TRAINER_H

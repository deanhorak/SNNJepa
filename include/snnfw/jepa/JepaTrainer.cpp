#include "snnfw/jepa/JepaTrainer.h"

#include "snnfw/jepa/JepaLoss.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace snnfw::jepa {

namespace {

using json = nlohmann::json;

struct TrainingExample {
    size_t sourceIndex = 0;
    int label = -1;
    std::string hemisphereName;
    std::string surfaceName;
    size_t sourceViewIndex = 0;
    std::string targetMode;
    std::vector<double> visible;
    std::vector<double> target;
};

struct BatchMoments {
    std::vector<double> mean;
    std::vector<double> variance;
};

struct LinearGradients {
    std::vector<std::vector<double>> weightGradients;
    std::vector<double> biasGradients;
};

uint64_t mixFeatureKey(uint64_t value) {
    value ^= value >> 33U;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33U;
    value *= 0xc4ceb9fe1a85ec53ULL;
    value ^= value >> 33U;
    return value;
}

uint64_t hashString(const std::string& value) {
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char ch : value) {
        hash ^= static_cast<uint64_t>(ch);
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::vector<double> projectMetadata(const std::string& hemisphereName,
                                    const std::string& surfaceName,
                                    size_t sourceViewIndex,
                                    size_t dim,
                                    double scale) {
    std::vector<double> projection(dim, 0.0);
    if (dim == 0 || scale <= 0.0) {
        return projection;
    }

    const uint64_t hemisphereHash = hashString(hemisphereName);
    const uint64_t surfaceHash = hashString(surfaceName);
    const uint64_t fixationHash = mixFeatureKey(static_cast<uint64_t>(sourceViewIndex + 1));
    const uint64_t keys[] = {
        hemisphereHash,
        surfaceHash,
        fixationHash,
        mixFeatureKey(hemisphereHash ^ (surfaceHash << 1U) ^ fixationHash),
    };

    for (uint64_t key : keys) {
        const size_t bucket = static_cast<size_t>(mixFeatureKey(key) % dim);
        const double sign = (key & 1ULL) == 0 ? 1.0 : -1.0;
        projection[bucket] += sign * scale;
    }

    return projection;
}

std::vector<double> projectTokens(const std::vector<JepaLatentToken>& tokens, size_t dim) {
    std::vector<double> projection(dim, 0.0);
    if (tokens.empty() || dim == 0) {
        return projection;
    }

    size_t totalValues = 0;
    for (size_t tokenIndex = 0; tokenIndex < tokens.size(); ++tokenIndex) {
        const auto& token = tokens[tokenIndex];
        for (size_t valueIndex = 0; valueIndex < token.values.size(); ++valueIndex) {
            const uint64_t key =
                mixFeatureKey(static_cast<uint64_t>(token.offset + valueIndex) +
                              (static_cast<uint64_t>(tokenIndex + 1) << 32U));
            const size_t bucket = static_cast<size_t>(key % dim);
            const double sign = (key & 1ULL) == 0 ? 1.0 : -1.0;
            projection[bucket] += sign * token.values[valueIndex];
            ++totalValues;
        }
    }

    if (totalValues == 0) {
        return projection;
    }

    const double scale = 1.0 / std::sqrt(static_cast<double>(totalValues));
    for (double& value : projection) {
        value *= scale;
    }
    return projection;
}

std::vector<double> projectVector(const std::vector<double>& values, size_t dim) {
    std::vector<double> projection(dim, 0.0);
    if (values.empty() || dim == 0) {
        return projection;
    }

    for (size_t valueIndex = 0; valueIndex < values.size(); ++valueIndex) {
        const uint64_t key = mixFeatureKey(static_cast<uint64_t>(valueIndex + 1));
        const size_t bucket = static_cast<size_t>(key % dim);
        const double sign = (key & 1ULL) == 0 ? 1.0 : -1.0;
        projection[bucket] += sign * values[valueIndex];
    }

    const double scale = 1.0 / std::sqrt(static_cast<double>(values.size()));
    for (double& value : projection) {
        value *= scale;
    }
    return projection;
}

void addInPlace(std::vector<double>* dst, const std::vector<double>& src) {
    const size_t count = std::min(dst->size(), src.size());
    for (size_t i = 0; i < count; ++i) {
        (*dst)[i] += src[i];
    }
}

std::vector<TrainingExample> buildMaskedExamples(const std::vector<JepaLatentSample>& samples,
                                                 size_t projectionDim,
                                                 const JepaConfig& config) {
    std::vector<TrainingExample> examples;
    for (const auto& sample : samples) {
        for (const auto& maskedView : sample.maskedViews) {
            if (!maskedView.leakageFree ||
                maskedView.visibleBranchTokens.empty() ||
                maskedView.hiddenBranchTokens.empty()) {
                continue;
            }

            TrainingExample example;
            example.sourceIndex = sample.sourceIndex;
            example.label = sample.label;
            example.hemisphereName = maskedView.hemisphereName;
            example.surfaceName = maskedView.surfaceName;
            example.sourceViewIndex = maskedView.sourceViewIndex;
            example.targetMode = "branch_mask";
            example.visible = projectTokens(maskedView.visibleBranchTokens, projectionDim);
            example.target = projectTokens(maskedView.hiddenBranchTokens, projectionDim);
            if (config.encodeViewMetadata) {
                const auto metadata = projectMetadata(maskedView.hemisphereName,
                                                      maskedView.surfaceName,
                                                      maskedView.sourceViewIndex,
                                                      projectionDim,
                                                      config.metadataScale);
                addInPlace(&example.visible, metadata);
                addInPlace(&example.target, metadata);
            }
            examples.push_back(std::move(example));
        }
    }
    return examples;
}

std::vector<TrainingExample> buildTemporalExamples(const std::vector<JepaLatentSample>& samples,
                                                   size_t projectionDim,
                                                   const JepaConfig& config) {
    std::vector<TrainingExample> examples;
    for (const auto& sample : samples) {
        std::unordered_map<std::string, std::vector<size_t>> groupedViewIndices;
        for (size_t viewIndex = 0; viewIndex < sample.hemisphereViews.size(); ++viewIndex) {
            const auto& view = sample.hemisphereViews[viewIndex];
            groupedViewIndices[view.hemisphereName + "|" + view.surfaceName].push_back(viewIndex);
        }

        for (auto& entry : groupedViewIndices) {
            auto& indices = entry.second;
            if (indices.size() < 2) {
                continue;
            }
            std::sort(indices.begin(),
                      indices.end(),
                      [&](size_t lhs, size_t rhs) {
                          return sample.hemisphereViews[lhs].sourceViewIndex <
                                 sample.hemisphereViews[rhs].sourceViewIndex;
                      });
            for (size_t i = 0; i + 1 < indices.size(); ++i) {
                const size_t sourceViewIdx = indices[i];
                const size_t targetViewIdx = indices[i + 1];
                const auto& sourceView = sample.hemisphereViews[sourceViewIdx];
                const auto& targetView = sample.hemisphereViews[targetViewIdx];

                std::vector<double> visible;
                if (sourceViewIdx < sample.maskedViews.size() &&
                    sample.maskedViews[sourceViewIdx].leakageFree &&
                    !sample.maskedViews[sourceViewIdx].visibleBranchTokens.empty()) {
                    visible = projectTokens(sample.maskedViews[sourceViewIdx].visibleBranchTokens,
                                            projectionDim);
                } else if (!sourceView.branchTokens.empty()) {
                    visible = projectTokens(sourceView.branchTokens, projectionDim);
                } else {
                    visible = projectVector(sourceView.hemisphereToken, projectionDim);
                }

                std::vector<double> target;
                if (config.targetMode == TargetMode::TemporalHemisphereSummary) {
                    target = projectVector(targetView.hemisphereToken, projectionDim);
                } else if (!targetView.branchTokens.empty()) {
                    target = projectTokens(targetView.branchTokens, projectionDim);
                } else {
                    target = projectVector(targetView.hemisphereToken, projectionDim);
                }

                TrainingExample example;
                example.sourceIndex = sample.sourceIndex;
                example.label = sample.label;
                example.hemisphereName = sourceView.hemisphereName;
                example.surfaceName = sourceView.surfaceName;
                example.sourceViewIndex = sourceView.sourceViewIndex;
                example.targetMode = targetModeName(config.targetMode);
                example.visible = std::move(visible);
                example.target = std::move(target);
                if (config.encodeViewMetadata) {
                    const auto sourceMetadata = projectMetadata(sourceView.hemisphereName,
                                                                sourceView.surfaceName,
                                                                sourceView.sourceViewIndex,
                                                                projectionDim,
                                                                config.metadataScale);
                    const auto targetMetadata = projectMetadata(targetView.hemisphereName,
                                                                targetView.surfaceName,
                                                                targetView.sourceViewIndex,
                                                                projectionDim,
                                                                config.metadataScale);
                    addInPlace(&example.visible, sourceMetadata);
                    addInPlace(&example.target, targetMetadata);
                }
                examples.push_back(std::move(example));
            }
        }
    }
    return examples;
}

double squaredL2Norm(const std::vector<double>& values) {
    double normSq = 0.0;
    for (double value : values) {
        normSq += value * value;
    }
    return normSq;
}

double l2Norm(const std::vector<double>& values) {
    return std::sqrt(squaredL2Norm(values));
}

double layerWeightNorm(const JepaLinearLayer& layer) {
    double normSq = squaredL2Norm(layer.bias);
    for (const auto& row : layer.weights) {
        normSq += squaredL2Norm(row);
    }
    return std::sqrt(normSq);
}

JepaLinearLayer makeIdentityLayer(size_t dim, double diagonalScale) {
    JepaLinearLayer layer;
    layer.weights.assign(dim, std::vector<double>(dim, 0.0));
    layer.bias.assign(dim, 0.0);
    for (size_t i = 0; i < dim; ++i) {
        layer.weights[i][i] = diagonalScale;
    }
    return layer;
}

std::vector<double> applyLinear(const JepaLinearLayer& layer,
                                const std::vector<double>& input) {
    const size_t dim = layer.bias.size();
    std::vector<double> output(dim, 0.0);
    for (size_t row = 0; row < dim; ++row) {
        double value = layer.bias[row];
        const size_t count = std::min(layer.weights[row].size(), input.size());
        for (size_t col = 0; col < count; ++col) {
            value += layer.weights[row][col] * input[col];
        }
        output[row] = value;
    }
    return output;
}

LinearGradients makeZeroGradients(size_t dim) {
    LinearGradients gradients;
    gradients.weightGradients.assign(dim, std::vector<double>(dim, 0.0));
    gradients.biasGradients.assign(dim, 0.0);
    return gradients;
}

std::vector<double> accumulateLinearGradients(const JepaLinearLayer& layer,
                                              const std::vector<double>& input,
                                              const std::vector<double>& outputGradient,
                                              LinearGradients* gradients) {
    const size_t rows = std::min(layer.weights.size(), outputGradient.size());
    std::vector<double> inputGradient(input.size(), 0.0);
    for (size_t row = 0; row < rows; ++row) {
        const double gradValue = outputGradient[row];
        gradients->biasGradients[row] += gradValue;
        const size_t cols = std::min(layer.weights[row].size(), input.size());
        for (size_t col = 0; col < cols; ++col) {
            gradients->weightGradients[row][col] += gradValue * input[col];
            inputGradient[col] += layer.weights[row][col] * gradValue;
        }
    }
    return inputGradient;
}

void applyGradients(JepaLinearLayer* layer,
                    const LinearGradients& gradients,
                    double learningRate,
                    double weightDecay) {
    const size_t rows = std::min(layer->weights.size(), gradients.weightGradients.size());
    for (size_t row = 0; row < rows; ++row) {
        layer->bias[row] -= learningRate * gradients.biasGradients[row];
        const size_t cols = std::min(layer->weights[row].size(), gradients.weightGradients[row].size());
        for (size_t col = 0; col < cols; ++col) {
            const double grad = gradients.weightGradients[row][col] +
                                (weightDecay * layer->weights[row][col]);
            layer->weights[row][col] -= learningRate * grad;
        }
    }
}

void updateEma(JepaLinearLayer* target, const JepaLinearLayer& source, double decay) {
    const double clampedDecay = std::clamp(decay, 0.0, 0.999999);
    const double sourceGain = 1.0 - clampedDecay;
    const size_t rows = std::min(target->weights.size(), source.weights.size());
    for (size_t row = 0; row < rows; ++row) {
        target->bias[row] =
            (clampedDecay * target->bias[row]) + (sourceGain * source.bias[row]);
        const size_t cols = std::min(target->weights[row].size(), source.weights[row].size());
        for (size_t col = 0; col < cols; ++col) {
            target->weights[row][col] =
                (clampedDecay * target->weights[row][col]) +
                (sourceGain * source.weights[row][col]);
        }
    }
}

double meanEmbeddingVariance(const std::vector<std::vector<double>>& embeddings) {
    if (embeddings.empty() || embeddings.front().empty()) {
        return 0.0;
    }

    const size_t dim = embeddings.front().size();
    std::vector<double> mean(dim, 0.0);
    for (const auto& embedding : embeddings) {
        for (size_t i = 0; i < dim; ++i) {
            mean[i] += embedding[i];
        }
    }
    for (double& value : mean) {
        value /= static_cast<double>(embeddings.size());
    }

    std::vector<double> variance(dim, 0.0);
    for (const auto& embedding : embeddings) {
        for (size_t i = 0; i < dim; ++i) {
            const double delta = embedding[i] - mean[i];
            variance[i] += delta * delta;
        }
    }
    for (double& value : variance) {
        value /= static_cast<double>(embeddings.size());
    }

    return std::accumulate(variance.begin(), variance.end(), 0.0) /
           static_cast<double>(variance.size());
}

double meanNorm(const std::vector<std::vector<double>>& embeddings) {
    if (embeddings.empty()) {
        return 0.0;
    }
    double total = 0.0;
    for (const auto& embedding : embeddings) {
        total += l2Norm(embedding);
    }
    return total / static_cast<double>(embeddings.size());
}

BatchMoments computeBatchMoments(const std::vector<std::vector<double>>& embeddings) {
    BatchMoments moments;
    if (embeddings.empty() || embeddings.front().empty()) {
        return moments;
    }

    const size_t dim = embeddings.front().size();
    moments.mean.assign(dim, 0.0);
    moments.variance.assign(dim, 0.0);
    for (const auto& embedding : embeddings) {
        for (size_t i = 0; i < dim; ++i) {
            moments.mean[i] += embedding[i];
        }
    }
    for (double& value : moments.mean) {
        value /= static_cast<double>(embeddings.size());
    }
    for (const auto& embedding : embeddings) {
        for (size_t i = 0; i < dim; ++i) {
            const double delta = embedding[i] - moments.mean[i];
            moments.variance[i] += delta * delta;
        }
    }
    for (double& value : moments.variance) {
        value /= static_cast<double>(embeddings.size());
    }
    return moments;
}

std::vector<double> cosineGradient(const std::vector<double>& prediction,
                                   const std::vector<double>& target) {
    const size_t count = std::min(prediction.size(), target.size());
    std::vector<double> gradient(count, 0.0);
    if (count == 0) {
        return gradient;
    }

    double dot = 0.0;
    double predNormSq = 0.0;
    double targetNormSq = 0.0;
    for (size_t i = 0; i < count; ++i) {
        dot += prediction[i] * target[i];
        predNormSq += prediction[i] * prediction[i];
        targetNormSq += target[i] * target[i];
    }

    const double predNorm = std::sqrt(std::max(predNormSq, 1e-12));
    const double targetNorm = std::sqrt(std::max(targetNormSq, 1e-12));
    const double denom = predNorm * targetNorm;
    if (denom <= 1e-12) {
        return gradient;
    }

    const double invDenom = 1.0 / denom;
    const double dotOverPredNormSq = dot / std::max(predNormSq, 1e-12);
    for (size_t i = 0; i < count; ++i) {
        gradient[i] = -((target[i] * invDenom) -
                        (dotOverPredNormSq * prediction[i] * invDenom));
    }
    return gradient;
}

double meanSquaredError(const std::vector<double>& lhs, const std::vector<double>& rhs) {
    const size_t count = std::min(lhs.size(), rhs.size());
    if (count == 0) {
        return 0.0;
    }
    double total = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double delta = lhs[i] - rhs[i];
        total += delta * delta;
    }
    return total / static_cast<double>(count);
}

json summaryToJson(const JepaTrainingSummary& summary,
                   const JepaConfig& config,
                   const std::vector<TrainingExample>& examples) {
    json preview = json::array();
    const size_t previewCount = std::min<size_t>(examples.size(), 8);
    for (size_t i = 0; i < previewCount; ++i) {
        preview.push_back({
            {"source_index", examples[i].sourceIndex},
            {"label", examples[i].label},
            {"hemisphere_name", examples[i].hemisphereName},
            {"surface_name", examples[i].surfaceName},
            {"source_view_index", examples[i].sourceViewIndex},
            {"target_mode", examples[i].targetMode},
        });
    }

    return json{
        {"target_mode", targetModeName(config.targetMode)},
        {"projection_dim", summary.projectionDim},
        {"epoch_count", summary.epochCount},
        {"sample_count", summary.sampleCount},
        {"masked_view_count", summary.maskedViewCount},
        {"training_example_count", summary.trainingExampleCount},
        {"temporal_example_count", summary.temporalExampleCount},
        {"fallback_masked_example_count", summary.fallbackMaskedExampleCount},
        {"mean_loss", summary.meanLoss},
        {"mean_shuffled_loss", summary.meanShuffledLoss},
        {"mean_variance_penalty", summary.meanVariancePenalty},
        {"mean_visible_norm", summary.meanVisibleNorm},
        {"mean_context_norm", summary.meanContextNorm},
        {"mean_target_norm", summary.meanTargetNorm},
        {"mean_prediction_norm", summary.meanPredictionNorm},
        {"context_variance", summary.contextVariance},
        {"target_variance", summary.targetVariance},
        {"prediction_variance", summary.predictionVariance},
        {"context_encoder_weight_norm", summary.contextEncoderWeightNorm},
        {"predictor_weight_norm", summary.predictorWeightNorm},
        {"target_encoder_weight_norm", summary.targetEncoderWeightNorm},
        {"variance_floor", config.varianceFloor},
        {"variance_penalty_gain", config.variancePenalty},
        {"learning_rate", config.trainerLearningRate},
        {"weight_decay", config.trainerWeightDecay},
        {"target_ema_decay", config.targetEmaDecay},
        {"mse_loss_weight", config.mseLossWeight},
        {"cosine_loss_weight", config.cosineLossWeight},
        {"encode_view_metadata", config.encodeViewMetadata},
        {"metadata_scale", config.metadataScale},
        {"preview_examples", std::move(preview)},
    };
}

std::vector<const JepaHemisphereView*> sortViews(const JepaLatentSample& sample) {
    std::vector<const JepaHemisphereView*> views;
    views.reserve(sample.hemisphereViews.size());
    for (const auto& view : sample.hemisphereViews) {
        views.push_back(&view);
    }
    std::sort(views.begin(),
              views.end(),
              [](const JepaHemisphereView* lhs, const JepaHemisphereView* rhs) {
                  if (lhs->hemisphereName != rhs->hemisphereName) {
                      return lhs->hemisphereName < rhs->hemisphereName;
                  }
                  if (lhs->surfaceName != rhs->surfaceName) {
                      return lhs->surfaceName < rhs->surfaceName;
                  }
                  return lhs->sourceViewIndex < rhs->sourceViewIndex;
              });
    return views;
}

std::vector<double> meanPool(const std::vector<std::vector<double>>& embeddings, size_t dim) {
    std::vector<double> pooled(dim, 0.0);
    if (embeddings.empty() || dim == 0) {
        return pooled;
    }
    for (const auto& embedding : embeddings) {
        for (size_t i = 0; i < std::min(dim, embedding.size()); ++i) {
            pooled[i] += embedding[i];
        }
    }
    const double scale = 1.0 / static_cast<double>(embeddings.size());
    for (double& value : pooled) {
        value *= scale;
    }
    return pooled;
}

std::vector<double> variancePool(const std::vector<std::vector<double>>& embeddings,
                                 const std::vector<double>& mean,
                                 size_t dim) {
    std::vector<double> pooled(dim, 0.0);
    if (embeddings.empty() || dim == 0) {
        return pooled;
    }
    for (const auto& embedding : embeddings) {
        for (size_t i = 0; i < std::min(dim, embedding.size()); ++i) {
            const double delta = embedding[i] - mean[i];
            pooled[i] += delta * delta;
        }
    }
    const double scale = 1.0 / static_cast<double>(embeddings.size());
    for (double& value : pooled) {
        value *= scale;
    }
    return pooled;
}

void appendVector(std::vector<double>* dst, const std::vector<double>& src) {
    dst->insert(dst->end(), src.begin(), src.end());
}

}  // namespace

JepaTrainingArtifacts trainMinimalModel(const std::vector<JepaLatentSample>& samples,
                                        const JepaConfig& config,
                                        const std::string& outputPath) {
    JepaTrainingArtifacts artifacts;
    JepaTrainingSummary summary;
    summary.sampleCount = samples.size();
    summary.projectionDim = static_cast<size_t>(std::max(1, config.projectionDim));
    for (const auto& sample : samples) {
        summary.maskedViewCount += sample.maskedViews.size();
        if (config.enforceNoLeakage) {
            for (const auto& maskedView : sample.maskedViews) {
                if (!maskedView.leakageFree) {
                    throw std::runtime_error(
                        "JEPA trainer leakage validation failed for source_index=" +
                        std::to_string(sample.sourceIndex) + ", hemisphere=" +
                        maskedView.hemisphereName);
                }
            }
        }
    }

    std::vector<TrainingExample> examples;
    const bool requiresTemporalExamples =
        config.targetMode == TargetMode::TemporalFixation ||
        config.targetMode == TargetMode::TemporalHemisphereSummary;
    if (requiresTemporalExamples) {
        examples = buildTemporalExamples(samples, summary.projectionDim, config);
        summary.temporalExampleCount = examples.size();
    }
    if (requiresTemporalExamples && examples.empty()) {
        throw std::runtime_error(
            "JEPA temporal target mode requested, but no temporal examples were built");
    }
    if (examples.empty()) {
        examples = buildMaskedExamples(samples, summary.projectionDim, config);
        summary.fallbackMaskedExampleCount = examples.size();
    }
    summary.trainingExampleCount = examples.size();
    summary.epochCount = std::max(0, config.trainerEpochs);
    if (examples.empty()) {
        throw std::runtime_error("JEPA minimal trainer found no valid temporal or masked-view examples");
    }

    const size_t dim = summary.projectionDim;
    JepaLinearLayer contextEncoder = makeIdentityLayer(dim, 1.0);
    JepaLinearLayer predictor = makeIdentityLayer(dim, 0.5);
    JepaLinearLayer targetEncoder = contextEncoder;

    const double lr = std::max(0.0, config.trainerLearningRate);
    const double weightDecay = std::max(0.0, config.trainerWeightDecay);
    const double emaDecay = std::clamp(config.targetEmaDecay, 0.0, 0.999999);
    const double mseWeight = std::max(0.0, config.mseLossWeight);
    const double cosineWeight = std::max(0.0, config.cosineLossWeight);
    const double variancePenalty = std::max(0.0, config.variancePenalty);
    const double varianceFloor = std::max(0.0, config.varianceFloor);
    const double gradientScale = 2.0 / static_cast<double>(std::max<size_t>(1, dim));

    for (int epoch = 0; epoch < summary.epochCount; ++epoch) {
        std::vector<std::vector<double>> contextLatents(examples.size());
        std::vector<std::vector<double>> predictions(examples.size());
        std::vector<std::vector<double>> targetLatents(examples.size());
        for (size_t index = 0; index < examples.size(); ++index) {
            contextLatents[index] = applyLinear(contextEncoder, examples[index].visible);
            predictions[index] = applyLinear(predictor, contextLatents[index]);
            targetLatents[index] = applyLinear(targetEncoder, examples[index].target);
        }

        const auto predictionMoments = computeBatchMoments(predictions);
        auto predictorGradients = makeZeroGradients(dim);
        auto contextEncoderGradients = makeZeroGradients(dim);
        const double batchScale = 1.0 / static_cast<double>(std::max<size_t>(1, examples.size()));

        for (size_t index = 0; index < examples.size(); ++index) {
            std::vector<double> predictionGradient(dim, 0.0);
            for (size_t i = 0; i < dim; ++i) {
                predictionGradient[i] +=
                    mseWeight * gradientScale * (predictions[index][i] - targetLatents[index][i]);
            }

            if (cosineWeight > 0.0) {
                const auto cosineGrad = cosineGradient(predictions[index], targetLatents[index]);
                for (size_t i = 0; i < dim; ++i) {
                    predictionGradient[i] += cosineWeight * cosineGrad[i];
                }
            }

            if (!predictionMoments.variance.empty()) {
                for (size_t i = 0; i < dim; ++i) {
                    if (predictionMoments.variance[i] < varianceFloor) {
                        predictionGradient[i] -=
                            variancePenalty * 2.0 * batchScale *
                            (predictions[index][i] - predictionMoments.mean[i]);
                    }
                }
            }

            for (double& value : predictionGradient) {
                value *= batchScale;
            }

            const auto contextGradient =
                accumulateLinearGradients(predictor,
                                          contextLatents[index],
                                          predictionGradient,
                                          &predictorGradients);
            accumulateLinearGradients(contextEncoder,
                                      examples[index].visible,
                                      contextGradient,
                                      &contextEncoderGradients);
        }

        applyGradients(&predictor, predictorGradients, lr, weightDecay);
        applyGradients(&contextEncoder, contextEncoderGradients, lr, weightDecay);
        updateEma(&targetEncoder, contextEncoder, emaDecay);
    }

    std::vector<std::vector<double>> visibleEmbeddings;
    std::vector<std::vector<double>> contextEmbeddings;
    std::vector<std::vector<double>> targetEmbeddings;
    std::vector<std::vector<double>> predictionEmbeddings;
    visibleEmbeddings.reserve(examples.size());
    contextEmbeddings.reserve(examples.size());
    targetEmbeddings.reserve(examples.size());
    predictionEmbeddings.reserve(examples.size());

    double totalLoss = 0.0;
    double totalShuffledLoss = 0.0;
    for (size_t index = 0; index < examples.size(); ++index) {
        const auto& example = examples[index];
        const auto contextLatent = applyLinear(contextEncoder, example.visible);
        const auto prediction = applyLinear(predictor, contextLatent);
        const auto targetLatent = applyLinear(targetEncoder, example.target);
        const auto& shuffledExample = examples[(index + 1) % examples.size()];
        const auto shuffledTargetLatent = applyLinear(targetEncoder, shuffledExample.target);

        totalLoss +=
            (mseWeight * meanSquaredError(prediction, targetLatent)) +
            (cosineWeight * cosineDistance(prediction, targetLatent));
        totalShuffledLoss +=
            (mseWeight * meanSquaredError(prediction, shuffledTargetLatent)) +
            (cosineWeight * cosineDistance(prediction, shuffledTargetLatent));

        visibleEmbeddings.push_back(example.visible);
        contextEmbeddings.push_back(contextLatent);
        targetEmbeddings.push_back(targetLatent);
        predictionEmbeddings.push_back(prediction);
    }

    summary.meanLoss = totalLoss / static_cast<double>(examples.size());
    summary.meanShuffledLoss = totalShuffledLoss / static_cast<double>(examples.size());
    summary.meanVisibleNorm = meanNorm(visibleEmbeddings);
    summary.meanContextNorm = meanNorm(contextEmbeddings);
    summary.meanTargetNorm = meanNorm(targetEmbeddings);
    summary.meanPredictionNorm = meanNorm(predictionEmbeddings);
    summary.contextVariance = meanEmbeddingVariance(contextEmbeddings);
    summary.targetVariance = meanEmbeddingVariance(targetEmbeddings);
    summary.predictionVariance = meanEmbeddingVariance(predictionEmbeddings);
    summary.meanVariancePenalty = std::max(0.0, config.varianceFloor - summary.predictionVariance) *
                                  std::max(0.0, config.variancePenalty);
    summary.contextEncoderWeightNorm = layerWeightNorm(contextEncoder);
    summary.predictorWeightNorm = layerWeightNorm(predictor);
    summary.targetEncoderWeightNorm = layerWeightNorm(targetEncoder);

    if (!outputPath.empty()) {
        const std::filesystem::path path(outputPath);
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }
        std::ofstream out(path);
        if (!out) {
            throw std::runtime_error("Failed to open JEPA trainer dump path: " + outputPath);
        }
        out << summaryToJson(summary, config, examples).dump(2) << '\n';
    }

    artifacts.model.projectionDim = dim;
    artifacts.model.encodeViewMetadata = config.encodeViewMetadata;
    artifacts.model.metadataScale = config.metadataScale;
    artifacts.model.contextEncoder = std::move(contextEncoder);
    artifacts.model.predictor = std::move(predictor);
    artifacts.model.targetEncoder = std::move(targetEncoder);
    artifacts.summary = summary;
    return artifacts;
}

JepaTrainingSummary runMinimalTrainer(const std::vector<JepaLatentSample>& samples,
                                      const JepaConfig& config,
                                      const std::string& outputPath) {
    return trainMinimalModel(samples, config, outputPath).summary;
}

std::vector<double> encodeSample(const JepaLatentSample& sample,
                                 const JepaModel& model) {
    if (model.projectionDim == 0) {
        return {};
    }

    const auto views = sortViews(sample);
    const size_t directProjectionDim = std::max<size_t>(256U, model.projectionDim * 8U);
    std::vector<std::vector<double>> directViews;
    std::vector<std::vector<double>> encodedViews;
    encodedViews.reserve(views.size());
    directViews.reserve(views.size());
    std::unordered_map<std::string, std::vector<std::vector<double>>> directByHemisphere;
    std::unordered_map<std::string, std::vector<std::vector<double>>> byHemisphere;
    std::unordered_map<size_t, std::vector<std::vector<double>>> directByFixation;
    std::unordered_map<size_t, std::vector<std::vector<double>>> byFixation;
    for (const auto* view : views) {
        auto directProjection = !view->branchTokens.empty()
            ? projectTokens(view->branchTokens, directProjectionDim)
            : projectVector(view->hemisphereToken, directProjectionDim);
        auto projection = !view->branchTokens.empty()
            ? projectTokens(view->branchTokens, model.projectionDim)
            : projectVector(view->hemisphereToken, model.projectionDim);
        if (model.encodeViewMetadata) {
            addInPlace(&directProjection,
                       projectMetadata(view->hemisphereName,
                                       view->surfaceName,
                                       view->sourceViewIndex,
                                       directProjectionDim,
                                       model.metadataScale));
            addInPlace(&projection,
                       projectMetadata(view->hemisphereName,
                                       view->surfaceName,
                                       view->sourceViewIndex,
                                       model.projectionDim,
                                       model.metadataScale));
        }
        directViews.push_back(directProjection);
        directByHemisphere[view->hemisphereName].push_back(directProjection);
        directByFixation[view->sourceViewIndex].push_back(directProjection);
        auto encoded = applyLinear(model.contextEncoder, projection);
        encodedViews.push_back(encoded);
        byHemisphere[view->hemisphereName].push_back(encoded);
        byFixation[view->sourceViewIndex].push_back(std::move(encoded));
    }

    if (encodedViews.empty()) {
        return {};
    }

    std::vector<double> embedding;
    embedding.reserve((directProjectionDim * 8U) + (model.projectionDim * 8U));

    const auto globalMean = meanPool(encodedViews, model.projectionDim);
    const auto globalVar = variancePool(encodedViews, globalMean, model.projectionDim);
    const auto directGlobalMean = meanPool(directViews, directProjectionDim);
    const auto directGlobalVar = variancePool(directViews, directGlobalMean, directProjectionDim);
    appendVector(&embedding, directGlobalMean);
    appendVector(&embedding, directGlobalVar);
    appendVector(&embedding, globalMean);
    appendVector(&embedding, globalVar);

    std::vector<std::string> hemisphereNames;
    hemisphereNames.reserve(byHemisphere.size());
    for (const auto& entry : byHemisphere) {
        hemisphereNames.push_back(entry.first);
    }
    std::sort(hemisphereNames.begin(), hemisphereNames.end());
    for (const auto& name : hemisphereNames) {
        appendVector(&embedding, meanPool(directByHemisphere[name], directProjectionDim));
        appendVector(&embedding, meanPool(byHemisphere[name], model.projectionDim));
    }

    std::vector<size_t> fixationIndices;
    fixationIndices.reserve(byFixation.size());
    for (const auto& entry : byFixation) {
        fixationIndices.push_back(entry.first);
    }
    std::sort(fixationIndices.begin(), fixationIndices.end());
    for (size_t fixationIndex : fixationIndices) {
        appendVector(&embedding, meanPool(directByFixation[fixationIndex], directProjectionDim));
        appendVector(&embedding, meanPool(byFixation[fixationIndex], model.projectionDim));
    }
    return embedding;
}

}  // namespace snnfw::jepa

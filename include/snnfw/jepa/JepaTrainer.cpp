#include "snnfw/jepa/JepaTrainer.h"

#include "snnfw/jepa/JepaLoss.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
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

struct ForwardPass {
    std::vector<double> contextPre;
    std::vector<double> contextLatent;
    std::vector<double> predictionPre;
    std::vector<double> prediction;
    std::vector<double> targetPre;
    std::vector<double> targetLatent;
};

struct BatchMoments {
    std::vector<double> mean;
    std::vector<double> variance;
};

struct BatchStandardization {
    std::vector<double> mean;
    std::vector<double> stddev;
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

void normalizeVector(std::vector<double>* values) {
    if (values == nullptr || values->empty()) {
        return;
    }
    const double norm = l2Norm(*values);
    if (norm <= 1e-12) {
        return;
    }
    for (double& value : *values) {
        value /= norm;
    }
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

std::vector<double> projectVectorPreserveNorm(const std::vector<double>& values, size_t dim) {
    auto projection = projectVector(values, dim);
    normalizeVector(&projection);
    return projection;
}

void addInPlace(std::vector<double>* dst, const std::vector<double>& src);

std::vector<double> projectViewTokens(const JepaHemisphereView& view,
                                      size_t dim,
                                      bool encodeMetadata,
                                      double metadataScale) {
    auto projection = !view.branchTokens.empty()
        ? projectTokens(view.branchTokens, dim)
        : projectVector(view.hemisphereToken, dim);
    if (encodeMetadata) {
        addInPlace(&projection,
                   projectMetadata(view.hemisphereName,
                                   view.surfaceName,
                                   view.sourceViewIndex,
                                   dim,
                                   metadataScale));
    }
    return projection;
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

struct TemporalLayout {
    std::vector<const JepaHemisphereView*> views;
    std::vector<std::string> hemisphereNames;
    std::vector<size_t> fixationIndices;
    std::unordered_map<std::string, size_t> hemisphereIndexByName;
    std::unordered_map<size_t, size_t> fixationSlotByIndex;
    std::vector<std::vector<std::vector<double>>> byHemisphereFixation;
};

void appendVector(std::vector<double>* dst, const std::vector<double>& src) {
    dst->insert(dst->end(), src.begin(), src.end());
}

std::vector<double> concatenate(const std::vector<std::vector<double>>& values) {
    size_t totalSize = 0;
    for (const auto& value : values) {
        totalSize += value.size();
    }
    std::vector<double> merged;
    merged.reserve(totalSize);
    for (const auto& value : values) {
        merged.insert(merged.end(), value.begin(), value.end());
    }
    return merged;
}

TemporalLayout buildTemporalLayout(const JepaLatentSample& sample,
                                   size_t projectionDim,
                                   const JepaConfig& config) {
    TemporalLayout layout;
    layout.views = sortViews(sample);
    if (layout.views.empty() || projectionDim == 0) {
        return layout;
    }

    std::unordered_set<size_t> seenFixations;
    for (const auto* view : layout.views) {
        if (layout.hemisphereIndexByName.emplace(view->hemisphereName, layout.hemisphereNames.size())
                .second) {
            layout.hemisphereNames.push_back(view->hemisphereName);
        }
        if (seenFixations.insert(view->sourceViewIndex).second) {
            layout.fixationIndices.push_back(view->sourceViewIndex);
        }
    }
    std::sort(layout.hemisphereNames.begin(), layout.hemisphereNames.end());
    std::sort(layout.fixationIndices.begin(), layout.fixationIndices.end());
    layout.hemisphereIndexByName.clear();
    for (size_t i = 0; i < layout.hemisphereNames.size(); ++i) {
        layout.hemisphereIndexByName.emplace(layout.hemisphereNames[i], i);
    }
    for (size_t i = 0; i < layout.fixationIndices.size(); ++i) {
        layout.fixationSlotByIndex.emplace(layout.fixationIndices[i], i);
    }

    layout.byHemisphereFixation.assign(
        layout.hemisphereNames.size(),
        std::vector<std::vector<double>>(layout.fixationIndices.size()));
    for (const auto* view : layout.views) {
        auto projection = projectViewTokens(
            *view, projectionDim, config.encodeViewMetadata, config.metadataScale);
        const auto hemisphereIt = layout.hemisphereIndexByName.find(view->hemisphereName);
        const auto fixationIt = layout.fixationSlotByIndex.find(view->sourceViewIndex);
        if (hemisphereIt == layout.hemisphereIndexByName.end() ||
            fixationIt == layout.fixationSlotByIndex.end()) {
            continue;
        }
        layout.byHemisphereFixation[hemisphereIt->second][fixationIt->second] = std::move(projection);
    }

    return layout;
}

std::vector<double> buildStructuredTemporalSummaryForRange(const JepaLatentSample& sample,
                                                           size_t projectionDim,
                                                           const JepaConfig& config,
                                                           size_t rangeBegin,
                                                           size_t rangeEnd,
                                                           size_t fixedSlotCount = 0) {
    const auto layout = buildTemporalLayout(sample, projectionDim, config);
    if (layout.views.empty() || layout.fixationIndices.empty() || projectionDim == 0) {
        return {};
    }

    rangeBegin = std::min(rangeBegin, layout.fixationIndices.size());
    rangeEnd = std::min(rangeEnd, layout.fixationIndices.size());
    if (rangeBegin >= rangeEnd) {
        return {};
    }

    const size_t slotCount = fixedSlotCount == 0 ? (rangeEnd - rangeBegin) : fixedSlotCount;
    std::vector<std::vector<double>> selectedViews;
    for (size_t hemisphereIndex = 0; hemisphereIndex < layout.hemisphereNames.size(); ++hemisphereIndex) {
        for (size_t fixationSlot = rangeBegin; fixationSlot < rangeEnd; ++fixationSlot) {
            if (!layout.byHemisphereFixation[hemisphereIndex][fixationSlot].empty()) {
                selectedViews.push_back(layout.byHemisphereFixation[hemisphereIndex][fixationSlot]);
            }
        }
    }
    if (selectedViews.empty()) {
        return {};
    }

    std::vector<double> summary;
    summary.reserve((1 + layout.hemisphereNames.size() + slotCount) * projectionDim);
    appendVector(&summary, meanPool(selectedViews, projectionDim));

    for (size_t hemisphereIndex = 0; hemisphereIndex < layout.hemisphereNames.size(); ++hemisphereIndex) {
        std::vector<std::vector<double>> perHemisphere;
        for (size_t fixationSlot = rangeBegin; fixationSlot < rangeEnd; ++fixationSlot) {
            if (!layout.byHemisphereFixation[hemisphereIndex][fixationSlot].empty()) {
                perHemisphere.push_back(layout.byHemisphereFixation[hemisphereIndex][fixationSlot]);
            }
        }
        appendVector(&summary, meanPool(perHemisphere, projectionDim));
    }

    for (size_t relativeSlot = 0; relativeSlot < slotCount; ++relativeSlot) {
        std::vector<std::vector<double>> perFixation;
        const size_t fixationSlot = rangeBegin + relativeSlot;
        if (fixationSlot < rangeEnd) {
            for (size_t hemisphereIndex = 0; hemisphereIndex < layout.hemisphereNames.size();
                 ++hemisphereIndex) {
                if (!layout.byHemisphereFixation[hemisphereIndex][fixationSlot].empty()) {
                    perFixation.push_back(layout.byHemisphereFixation[hemisphereIndex][fixationSlot]);
                }
            }
        }
        appendVector(&summary, meanPool(perFixation, projectionDim));
    }

    return summary;
}

std::vector<double> buildStructuredTemporalSummary(const JepaLatentSample& sample,
                                                   size_t projectionDim,
                                                   const JepaConfig& config,
                                                   bool useFutureHalf) {
    const auto views = sortViews(sample);
    if (views.empty() || projectionDim == 0) {
        return {};
    }

    std::vector<size_t> fixationIndices;
    fixationIndices.reserve(views.size());
    std::unordered_set<size_t> seenFixations;
    for (const auto* view : views) {
        if (seenFixations.insert(view->sourceViewIndex).second) {
            fixationIndices.push_back(view->sourceViewIndex);
        }
    }
    std::sort(fixationIndices.begin(), fixationIndices.end());
    const size_t split = std::max<size_t>(1, fixationIndices.size() / 2);
    const size_t fixedSlotCount = std::max(split, fixationIndices.size() - split);
    const size_t rangeBegin = useFutureHalf ? split : 0;
    const size_t rangeEnd = useFutureHalf ? fixationIndices.size() : split;
    return buildStructuredTemporalSummaryForRange(
        sample, projectionDim, config, rangeBegin, rangeEnd, fixedSlotCount);
}

void appendNormalizedChannel(std::vector<std::vector<double>>* channels,
                             std::vector<double> channel) {
    if (channels == nullptr || channel.empty()) {
        return;
    }
    normalizeVector(&channel);
    channels->push_back(std::move(channel));
}

std::vector<double> buildFirstSpikeProjectionCode(
    const std::vector<std::vector<double>>& perView,
    double thresholdFraction) {
    if (perView.empty() || perView.front().empty()) {
        return {};
    }
    const size_t dim = perView.front().size();
    std::vector<double> maxAbs(dim, 0.0);
    for (const auto& values : perView) {
        if (values.size() != dim) {
            continue;
        }
        for (size_t i = 0; i < dim; ++i) {
            maxAbs[i] = std::max(maxAbs[i], std::abs(values[i]));
        }
    }

    std::vector<double> code(dim, 0.0);
    const double timeScale = 1.0 / static_cast<double>(std::max<size_t>(1, perView.size()));
    for (size_t feature = 0; feature < dim; ++feature) {
        if (maxAbs[feature] <= 1e-12) {
            continue;
        }
        const double threshold = std::max(1e-12, thresholdFraction * maxAbs[feature]);
        for (size_t t = 0; t < perView.size(); ++t) {
            if (perView[t].size() != dim || std::abs(perView[t][feature]) < threshold) {
                continue;
            }
            const double sign = perView[t][feature] >= 0.0 ? 1.0 : -1.0;
            code[feature] =
                sign * static_cast<double>(perView.size() - t) * timeScale;
            break;
        }
    }
    return code;
}

std::vector<double> buildTraceProjectionCode(
    const std::vector<std::vector<double>>& perView,
    double decay) {
    if (perView.empty() || perView.front().empty()) {
        return {};
    }
    const size_t dim = perView.front().size();
    std::vector<double> trace(dim, 0.0);
    for (const auto& values : perView) {
        if (values.size() != dim) {
            continue;
        }
        for (size_t i = 0; i < dim; ++i) {
            trace[i] = decay * trace[i] + values[i];
        }
    }
    return trace;
}

std::vector<double> buildDeltaProjectionCode(
    const std::vector<std::vector<double>>& perView) {
    if (perView.size() < 2 || perView.front().empty()) {
        return {};
    }
    const size_t dim = perView.front().size();
    std::vector<double> code(dim, 0.0);
    int count = 0;
    for (size_t t = 1; t < perView.size(); ++t) {
        if (perView[t - 1].size() != dim || perView[t].size() != dim) {
            continue;
        }
        for (size_t i = 0; i < dim; ++i) {
            code[i] += perView[t][i] - perView[t - 1][i];
        }
        ++count;
    }
    if (count > 0) {
        const double scale = 1.0 / static_cast<double>(count);
        for (double& value : code) {
            value *= scale;
        }
    }
    return code;
}

std::vector<double> buildPeakProjectionCode(
    const std::vector<std::vector<double>>& perView) {
    if (perView.empty() || perView.front().empty()) {
        return {};
    }
    const size_t dim = perView.front().size();
    std::vector<double> code(dim, 0.0);
    std::vector<double> maxAbs(dim, 0.0);
    for (const auto& values : perView) {
        if (values.size() != dim) {
            continue;
        }
        for (size_t i = 0; i < dim; ++i) {
            const double magnitude = std::abs(values[i]);
            if (magnitude > maxAbs[i]) {
                maxAbs[i] = magnitude;
                code[i] = values[i];
            }
        }
    }
    return code;
}

std::vector<double> buildVarianceProjectionCode(
    const std::vector<std::vector<double>>& perView) {
    if (perView.size() < 2 || perView.front().empty()) {
        return {};
    }
    const size_t dim = perView.front().size();
    std::vector<double> mean(dim, 0.0);
    int count = 0;
    for (const auto& values : perView) {
        if (values.size() != dim) {
            continue;
        }
        for (size_t i = 0; i < dim; ++i) {
            mean[i] += values[i];
        }
        ++count;
    }
    if (count <= 1) {
        return {};
    }
    const double invCount = 1.0 / static_cast<double>(count);
    for (double& value : mean) {
        value *= invCount;
    }

    std::vector<double> code(dim, 0.0);
    for (const auto& values : perView) {
        if (values.size() != dim) {
            continue;
        }
        for (size_t i = 0; i < dim; ++i) {
            const double diff = values[i] - mean[i];
            code[i] += diff * diff;
        }
    }
    const double invDof = 1.0 / static_cast<double>(count - 1);
    for (double& value : code) {
        value *= invDof;
    }
    return code;
}

std::vector<double> buildAdaptationProjectionCode(
    const std::vector<std::vector<double>>& perView) {
    if (perView.size() < 2 || perView.front().empty()) {
        return {};
    }
    const size_t dim = perView.front().size();
    const auto& first = perView.front();
    const auto& last = perView.back();
    if (first.size() != dim || last.size() != dim) {
        return {};
    }
    std::vector<double> code(dim, 0.0);
    for (size_t i = 0; i < dim; ++i) {
        code[i] = last[i] - first[i];
    }
    return code;
}

std::vector<double> projectTemporalSpikeCodeForTrainer(const std::vector<double>& code,
                                                       size_t projectionDim) {
    if (code.empty() || projectionDim == 0U) {
        return {};
    }
    if (code.size() == projectionDim) {
        auto projected = code;
        normalizeVector(&projected);
        return projected;
    }
    return projectVectorPreserveNorm(code, projectionDim);
}

std::vector<double> buildProjectedOnsetSustainedCodeForRange(
    const JepaLatentSample& sample,
    size_t codeDim,
    const JepaConfig& config,
    size_t rangeBegin,
    size_t rangeEnd) {
    const auto layout = buildTemporalLayout(sample, codeDim, config);
    if (layout.views.empty() || layout.fixationIndices.empty() || codeDim == 0U) {
        return {};
    }
    rangeBegin = std::min(rangeBegin, layout.fixationIndices.size());
    rangeEnd = std::min(rangeEnd, layout.fixationIndices.size());
    if (rangeBegin >= rangeEnd) {
        return {};
    }

    std::vector<std::vector<double>> perHemisphere;
    for (size_t hemisphereIndex = 0; hemisphereIndex < layout.hemisphereNames.size();
         ++hemisphereIndex) {
        std::vector<std::vector<double>> perView;
        for (size_t fixationSlot = rangeBegin; fixationSlot < rangeEnd; ++fixationSlot) {
            if (!layout.byHemisphereFixation[hemisphereIndex][fixationSlot].empty()) {
                perView.push_back(layout.byHemisphereFixation[hemisphereIndex][fixationSlot]);
            }
        }
        if (perView.empty()) {
            continue;
        }

        std::vector<std::vector<double>> channels;
        appendNormalizedChannel(&channels, buildFirstSpikeProjectionCode(perView, 0.30));
        appendNormalizedChannel(&channels, buildTraceProjectionCode(perView, 0.70));
        appendNormalizedChannel(&channels, buildDeltaProjectionCode(perView));
        appendNormalizedChannel(&channels, buildPeakProjectionCode(perView));
        appendNormalizedChannel(&channels, buildVarianceProjectionCode(perView));
        appendNormalizedChannel(&channels, buildAdaptationProjectionCode(perView));
        perHemisphere.push_back(projectVectorPreserveNorm(concatenate(channels), codeDim));
    }

    return concatenate(perHemisphere);
}

std::vector<double> buildProjectedOnsetSustainedCode(const JepaLatentSample& sample,
                                                     size_t codeDim,
                                                     const JepaConfig& config,
                                                     bool useFutureHalf) {
    const auto views = sortViews(sample);
    if (views.empty() || codeDim == 0U) {
        return {};
    }

    std::vector<size_t> fixationIndices;
    std::unordered_set<size_t> seenFixations;
    for (const auto* view : views) {
        if (seenFixations.insert(view->sourceViewIndex).second) {
            fixationIndices.push_back(view->sourceViewIndex);
        }
    }
    std::sort(fixationIndices.begin(), fixationIndices.end());
    if (fixationIndices.empty()) {
        return {};
    }

    const size_t split = std::max<size_t>(1, fixationIndices.size() / 2);
    const size_t rangeBegin = useFutureHalf ? split : 0;
    const size_t rangeEnd = useFutureHalf ? fixationIndices.size() : split;
    return buildProjectedOnsetSustainedCodeForRange(sample, codeDim, config, rangeBegin, rangeEnd);
}

std::vector<double> buildSuccessorSummaryTarget(const JepaLatentSample& sample,
                                                size_t projectionDim,
                                                const JepaConfig& config) {
    const auto views = sortViews(sample);
    if (views.empty() || projectionDim == 0) {
        return {};
    }

    std::vector<size_t> fixationIndices;
    fixationIndices.reserve(views.size());
    std::unordered_set<size_t> seenFixations;
    for (const auto* view : views) {
        if (seenFixations.insert(view->sourceViewIndex).second) {
            fixationIndices.push_back(view->sourceViewIndex);
        }
    }
    std::sort(fixationIndices.begin(), fixationIndices.end());
    if (fixationIndices.empty()) {
        return {};
    }

    const size_t split = std::max<size_t>(1, fixationIndices.size() / 2);
    const size_t futureBegin = std::min(split, fixationIndices.size());
    const size_t futureEnd = fixationIndices.size();
    const size_t fixedSlotCount = std::max(split, futureEnd - futureBegin);
    if (futureBegin >= futureEnd) {
        return {};
    }

    std::vector<std::pair<size_t, double>> horizonEnds;
    const auto addHorizon = [&](size_t endSlot, double weight) {
        endSlot = std::clamp(endSlot, futureBegin + 1, futureEnd);
        for (const auto& existing : horizonEnds) {
            if (existing.first == endSlot) {
                return;
            }
        }
        horizonEnds.emplace_back(endSlot, weight);
    };
    const double discount = std::clamp(config.successorDiscount, 0.0, 1.0);
    addHorizon(futureBegin + 1, 1.0);
    addHorizon(futureBegin + 2, discount);
    addHorizon(futureEnd, discount * discount);

    std::vector<double> successorSummary;
    double weightSum = 0.0;
    for (const auto& [endSlot, weight] : horizonEnds) {
        auto horizonSummary = buildStructuredTemporalSummaryForRange(
            sample, projectionDim, config, futureBegin, endSlot, fixedSlotCount);
        if (horizonSummary.empty() || weight <= 0.0) {
            continue;
        }
        if (successorSummary.empty()) {
            successorSummary.assign(horizonSummary.size(), 0.0);
        }
        const size_t count = std::min(successorSummary.size(), horizonSummary.size());
        for (size_t i = 0; i < count; ++i) {
            successorSummary[i] += weight * horizonSummary[i];
        }
        weightSum += weight;
    }
    if (successorSummary.empty() || weightSum <= 0.0) {
        return {};
    }
    for (double& value : successorSummary) {
        value /= weightSum;
    }
    return successorSummary;
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
    const bool useTemporalSpikeCode = config.targetMode == TargetMode::TemporalSpikeCode;
    const size_t temporalSpikeCodeDim =
        static_cast<size_t>(std::max(1, config.temporalSpikeCodeDim));
    for (const auto& sample : samples) {
        auto visible = useTemporalSpikeCode
            ? projectTemporalSpikeCodeForTrainer(
                  buildProjectedOnsetSustainedCode(
                      sample, temporalSpikeCodeDim, config, false),
                  projectionDim)
            : buildStructuredTemporalSummary(sample, projectionDim, config, false);
        auto target = useTemporalSpikeCode
            ? projectTemporalSpikeCodeForTrainer(
                  buildProjectedOnsetSustainedCode(sample, temporalSpikeCodeDim, config, true),
                  projectionDim)
            : buildSuccessorSummaryTarget(sample, projectionDim, config);
        if (visible.empty() || target.empty()) {
            continue;
        }

        TrainingExample example;
        example.sourceIndex = sample.sourceIndex;
        example.label = sample.label;
        example.hemisphereName = "bilateral";
        example.surfaceName = useTemporalSpikeCode
            ? "future_projected_rich_onset_sustained_spike_code"
            : "successor_future_hemisphere_summary";
        example.sourceViewIndex = 0;
        example.targetMode = useTemporalSpikeCode
            ? "future_projected_rich_onset_sustained_spike_code"
            : "successor_future_hemisphere_summary_locked";
        example.visible = std::move(visible);
        example.target = std::move(target);
        examples.push_back(std::move(example));
    }
    return examples;
}

double layerWeightNorm(const JepaLinearLayer& layer) {
    double normSq = squaredL2Norm(layer.bias);
    for (const auto& row : layer.weights) {
        normSq += squaredL2Norm(row);
    }
    return std::sqrt(normSq);
}

JepaLinearLayer makeLinearLayer(size_t outputDim,
                                size_t inputDim,
                                double diagonalScale,
                                uint64_t seed) {
    JepaLinearLayer layer;
    layer.weights.assign(outputDim, std::vector<double>(inputDim, 0.0));
    layer.bias.assign(outputDim, 0.0);
    if (outputDim == 0 || inputDim == 0) {
        return layer;
    }

    if (outputDim == inputDim) {
        for (size_t i = 0; i < outputDim; ++i) {
            layer.weights[i][i] = diagonalScale;
        }
        return layer;
    }

    std::mt19937_64 rng(seed);
    const double scale = std::sqrt(2.0 / static_cast<double>(std::max<size_t>(1, inputDim)));
    std::normal_distribution<double> dist(0.0, scale);
    for (auto& row : layer.weights) {
        for (double& value : row) {
            value = dist(rng);
        }
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

LinearGradients makeZeroGradients(size_t outputDim, size_t inputDim) {
    LinearGradients gradients;
    gradients.weightGradients.assign(outputDim, std::vector<double>(inputDim, 0.0));
    gradients.biasGradients.assign(outputDim, 0.0);
    return gradients;
}

std::vector<double> applyTanh(const std::vector<double>& input) {
    std::vector<double> output(input.size(), 0.0);
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::tanh(input[i]);
    }
    return output;
}

std::vector<double> tanhBackward(const std::vector<double>& activated,
                                 const std::vector<double>& outputGradient) {
    const size_t count = std::min(activated.size(), outputGradient.size());
    std::vector<double> gradient(count, 0.0);
    for (size_t i = 0; i < count; ++i) {
        gradient[i] = outputGradient[i] * (1.0 - (activated[i] * activated[i]));
    }
    return gradient;
}

std::vector<double> layerNormalizeVector(const std::vector<double>& input,
                                         double epsilon = 1e-4) {
    if (input.empty()) {
        return {};
    }
    const double mean =
        std::accumulate(input.begin(), input.end(), 0.0) / static_cast<double>(input.size());
    double variance = 0.0;
    for (double value : input) {
        const double delta = value - mean;
        variance += delta * delta;
    }
    variance /= static_cast<double>(input.size());
    const double invStd = 1.0 / std::sqrt(std::max(variance, 0.0) + std::max(epsilon, 1e-12));

    std::vector<double> output(input.size(), 0.0);
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = (input[i] - mean) * invStd;
    }
    return output;
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

BatchStandardization computeBatchStandardization(
    const std::vector<std::vector<double>>& embeddings,
    double epsilon = 1e-4) {
    BatchStandardization normalization;
    const auto moments = computeBatchMoments(embeddings);
    normalization.mean = moments.mean;
    normalization.stddev.assign(moments.variance.size(), 1.0);
    for (size_t i = 0; i < moments.variance.size(); ++i) {
        normalization.stddev[i] =
            std::sqrt(std::max(moments.variance[i], 0.0) + std::max(epsilon, 1e-12));
    }
    return normalization;
}

std::vector<std::vector<double>> standardizeEmbeddings(
    const std::vector<std::vector<double>>& embeddings,
    const BatchStandardization& normalization) {
    std::vector<std::vector<double>> standardized(
        embeddings.size(),
        embeddings.empty() ? std::vector<double>{}
                           : std::vector<double>(embeddings.front().size(), 0.0));
    for (size_t n = 0; n < embeddings.size(); ++n) {
        for (size_t i = 0; i < std::min(embeddings[n].size(), normalization.mean.size()); ++i) {
            standardized[n][i] =
                (embeddings[n][i] - normalization.mean[i]) / normalization.stddev[i];
        }
    }
    return standardized;
}

std::vector<double> cosineGradient(const std::vector<double>& prediction,
                                   const std::vector<double>& target);

std::vector<double> standardizedPredictionGradient(
    const std::vector<double>& prediction,
    const std::vector<double>& predictionStandardized,
    const std::vector<double>& targetStandardized,
    const BatchStandardization& predictionNormalization,
    double mseWeight,
    double cosineWeight) {
    const size_t count = std::min({prediction.size(),
                                   predictionStandardized.size(),
                                   targetStandardized.size(),
                                   predictionNormalization.stddev.size()});
    std::vector<double> gradient(count, 0.0);
    if (count == 0) {
        return gradient;
    }

    const double mseScale = 2.0 / static_cast<double>(count);
    for (size_t i = 0; i < count; ++i) {
        const double delta = predictionStandardized[i] - targetStandardized[i];
        gradient[i] +=
            mseWeight * mseScale * delta / predictionNormalization.stddev[i];
    }

    if (cosineWeight > 0.0) {
        const auto cosineGrad =
            cosineGradient(predictionStandardized, targetStandardized);
        for (size_t i = 0; i < count; ++i) {
            gradient[i] +=
                cosineWeight * cosineGrad[i] / predictionNormalization.stddev[i];
        }
    }
    return gradient;
}

std::vector<std::vector<double>> covariancePenaltyGradients(
    const std::vector<std::vector<double>>& embeddings,
    double penaltyGain) {
    std::vector<std::vector<double>> gradients(
        embeddings.size(),
        embeddings.empty() ? std::vector<double>{}
                           : std::vector<double>(embeddings.front().size(), 0.0));
    if (penaltyGain <= 0.0 || embeddings.size() < 2 || embeddings.front().size() < 2) {
        return gradients;
    }

    const size_t batchSize = embeddings.size();
    const size_t dim = embeddings.front().size();
    const auto moments = computeBatchMoments(embeddings);
    std::vector<std::vector<double>> centered(batchSize, std::vector<double>(dim, 0.0));
    for (size_t n = 0; n < batchSize; ++n) {
        for (size_t i = 0; i < dim; ++i) {
            centered[n][i] = embeddings[n][i] - moments.mean[i];
        }
    }

    std::vector<std::vector<double>> offdiag(dim, std::vector<double>(dim, 0.0));
    const double batchScale = 1.0 / static_cast<double>(batchSize);
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            if (i == j) {
                continue;
            }
            double covariance = 0.0;
            for (size_t n = 0; n < batchSize; ++n) {
                covariance += centered[n][i] * centered[n][j];
            }
            offdiag[i][j] = covariance * batchScale;
        }
    }

    const double lossScale =
        (4.0 * penaltyGain) /
        (static_cast<double>(batchSize) * static_cast<double>(dim) *
         static_cast<double>(std::max<size_t>(1, dim - 1)));
    for (size_t n = 0; n < batchSize; ++n) {
        for (size_t i = 0; i < dim; ++i) {
            double grad = 0.0;
            for (size_t j = 0; j < dim; ++j) {
                if (i == j) {
                    continue;
                }
                grad += centered[n][j] * offdiag[i][j];
            }
            gradients[n][i] = lossScale * grad;
        }
    }
    return gradients;
}

std::vector<std::vector<double>> variancePenaltyGradients(
    const std::vector<std::vector<double>>& embeddings,
    double varianceFloor,
    double penaltyGain) {
    std::vector<std::vector<double>> gradients(
        embeddings.size(),
        embeddings.empty() ? std::vector<double>{}
                           : std::vector<double>(embeddings.front().size(), 0.0));
    if (penaltyGain <= 0.0 || varianceFloor <= 0.0 || embeddings.size() < 2 || embeddings.front().empty()) {
        return gradients;
    }

    const size_t batchSize = embeddings.size();
    const size_t dim = embeddings.front().size();
    const auto moments = computeBatchMoments(embeddings);
    const double stdFloor = std::sqrt(std::max(varianceFloor, 0.0));
    const double batchScale =
        penaltyGain / (static_cast<double>(batchSize) * static_cast<double>(dim));
    for (size_t i = 0; i < dim; ++i) {
        const double stddev = std::sqrt(std::max(moments.variance[i], 1e-12));
        if (stddev >= stdFloor) {
            continue;
        }
        const double invStd = 1.0 / std::max(stddev, 1e-6);
        for (size_t n = 0; n < batchSize; ++n) {
            gradients[n][i] -= batchScale * invStd *
                               (embeddings[n][i] - moments.mean[i]);
        }
    }
    return gradients;
}

double meanCovariancePenalty(const std::vector<std::vector<double>>& embeddings) {
    if (embeddings.size() < 2 || embeddings.front().size() < 2) {
        return 0.0;
    }
    const auto moments = computeBatchMoments(embeddings);
    const size_t batchSize = embeddings.size();
    const size_t dim = embeddings.front().size();
    double total = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            if (i == j) {
                continue;
            }
            double covariance = 0.0;
            for (size_t n = 0; n < batchSize; ++n) {
                covariance +=
                    (embeddings[n][i] - moments.mean[i]) *
                    (embeddings[n][j] - moments.mean[j]);
            }
            covariance /= static_cast<double>(batchSize);
            total += covariance * covariance;
            ++count;
        }
    }
    return count == 0 ? 0.0 : total / static_cast<double>(count);
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
        {"temporal_spike_code_dim", config.temporalSpikeCodeDim},
        {"future_summary_aligned",
         !examples.empty() &&
             (examples.front().targetMode == "future_hemisphere_summary_locked" ||
              examples.front().targetMode == "successor_future_hemisphere_summary_locked")},
        {"temporal_spike_code_aligned",
         !examples.empty() &&
             examples.front().targetMode ==
                 "future_projected_rich_onset_sustained_spike_code"},
        {"context_input_dim", examples.empty() ? 0 : examples.front().visible.size()},
        {"target_input_dim", examples.empty() ? 0 : examples.front().target.size()},
        {"epoch_count", summary.epochCount},
        {"sample_count", summary.sampleCount},
        {"masked_view_count", summary.maskedViewCount},
        {"training_example_count", summary.trainingExampleCount},
        {"temporal_example_count", summary.temporalExampleCount},
        {"fallback_masked_example_count", summary.fallbackMaskedExampleCount},
        {"mean_loss", summary.meanLoss},
        {"mean_shuffled_loss", summary.meanShuffledLoss},
        {"mean_variance_penalty", summary.meanVariancePenalty},
        {"mean_covariance_penalty", summary.meanCovariancePenalty},
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
        {"covariance_penalty_gain", config.covariancePenalty},
        {"successor_discount", config.successorDiscount},
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
        config.targetMode == TargetMode::TemporalHemisphereSummary ||
        config.targetMode == TargetMode::TemporalSpikeCode;
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
    const size_t contextInputDim = examples.front().visible.size();
    const size_t targetInputDim = examples.front().target.size();
    const uint64_t baseSeed =
        mixFeatureKey(static_cast<uint64_t>(config.maskSeed) ^
                      static_cast<uint64_t>(contextInputDim << 16U) ^
                      static_cast<uint64_t>(targetInputDim << 32U));
    JepaLinearLayer contextEncoder =
        makeLinearLayer(dim, contextInputDim, 1.0, baseSeed ^ 0x51ed2705ULL);
    JepaLinearLayer predictor =
        makeLinearLayer(dim, dim, 0.5, baseSeed ^ 0x9e3779b97f4a7c15ULL);
    JepaLinearLayer targetEncoder =
        makeLinearLayer(dim, targetInputDim, 1.0, baseSeed ^ 0xc4ceb9fe1a85ec53ULL);

    const double baseLr = std::max(0.0, config.trainerLearningRate);
    const double lr = requiresTemporalExamples ? std::min(baseLr, 0.005) : baseLr;
    const double weightDecay = std::max(0.0, config.trainerWeightDecay);
    const double mseWeight = std::max(0.0, config.mseLossWeight);
    const double cosineWeight = std::max(0.0, config.cosineLossWeight);
    const double variancePenalty = std::max(0.0, config.variancePenalty);
    const double varianceFloor = std::max(0.0, config.varianceFloor);
    const double covariancePenalty =
        requiresTemporalExamples ? std::max(0.0, config.covariancePenalty) : 0.0;

    for (int epoch = 0; epoch < summary.epochCount; ++epoch) {
        std::vector<ForwardPass> forward(examples.size());
        std::vector<std::vector<double>> contexts(examples.size());
        std::vector<std::vector<double>> predictions(examples.size());
        for (size_t index = 0; index < examples.size(); ++index) {
            forward[index].contextPre = applyLinear(contextEncoder, examples[index].visible);
            forward[index].contextLatent =
                applyTanh(layerNormalizeVector(forward[index].contextPre));
            forward[index].predictionPre = applyLinear(predictor, forward[index].contextLatent);
            forward[index].prediction =
                applyTanh(layerNormalizeVector(forward[index].predictionPre));
            forward[index].targetPre = applyLinear(targetEncoder, examples[index].target);
            forward[index].targetLatent =
                applyTanh(layerNormalizeVector(forward[index].targetPre));
            contexts[index] = forward[index].contextLatent;
            predictions[index] = forward[index].prediction;
        }

        const auto predictionNormalization = computeBatchStandardization(predictions);
        std::vector<std::vector<double>> targets(examples.size());
        for (size_t index = 0; index < examples.size(); ++index) {
            targets[index] = forward[index].targetLatent;
        }
        const auto targetBatchNormalization = computeBatchStandardization(targets);
        const auto normalizedPredictions =
            standardizeEmbeddings(predictions, predictionNormalization);
        const auto normalizedTargets =
            standardizeEmbeddings(targets, targetBatchNormalization);

        const auto contextVarGradients =
            variancePenaltyGradients(contexts, varianceFloor, variancePenalty);
        const auto predictionVarGradients =
            variancePenaltyGradients(predictions, varianceFloor, variancePenalty);
        const auto contextCovarianceGradients =
            covariancePenaltyGradients(contexts, covariancePenalty);
        const auto covarianceGradients = covariancePenaltyGradients(predictions, covariancePenalty);
        auto predictorGradients = makeZeroGradients(dim, dim);
        auto contextEncoderGradients = makeZeroGradients(dim, contextInputDim);
        const double batchScale = 1.0 / static_cast<double>(std::max<size_t>(1, examples.size()));

        for (size_t index = 0; index < examples.size(); ++index) {
            auto predictionGradient = standardizedPredictionGradient(
                forward[index].prediction,
                normalizedPredictions[index],
                normalizedTargets[index],
                predictionNormalization,
                mseWeight,
                cosineWeight);

            if (index < predictionVarGradients.size()) {
                for (size_t i = 0; i < std::min(dim, predictionVarGradients[index].size()); ++i) {
                    predictionGradient[i] += predictionVarGradients[index][i];
                }
            }
            if (index < covarianceGradients.size()) {
                for (size_t i = 0; i < std::min(dim, covarianceGradients[index].size()); ++i) {
                    predictionGradient[i] += covarianceGradients[index][i];
                }
            }

            for (double& value : predictionGradient) {
                value *= batchScale;
            }

            const auto predictionPreGradient =
                tanhBackward(forward[index].prediction, predictionGradient);
            const auto contextGradient =
                accumulateLinearGradients(predictor,
                                          forward[index].contextLatent,
                                          predictionPreGradient,
                                          &predictorGradients);
            std::vector<double> totalContextGradient = contextGradient;
            if (index < contextVarGradients.size()) {
                for (size_t i = 0; i < std::min(totalContextGradient.size(),
                                                contextVarGradients[index].size());
                     ++i) {
                    totalContextGradient[i] += contextVarGradients[index][i] * batchScale;
                }
            }
            if (index < contextCovarianceGradients.size()) {
                for (size_t i = 0; i < std::min(totalContextGradient.size(),
                                                contextCovarianceGradients[index].size());
                     ++i) {
                    totalContextGradient[i] += contextCovarianceGradients[index][i] * batchScale;
                }
            }
            const auto contextPreGradient =
                tanhBackward(forward[index].contextLatent, totalContextGradient);
            accumulateLinearGradients(contextEncoder,
                                      examples[index].visible,
                                      contextPreGradient,
                                      &contextEncoderGradients);
        }

        applyGradients(&predictor, predictorGradients, lr, weightDecay);
        applyGradients(&contextEncoder, contextEncoderGradients, lr, weightDecay);
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
        const auto contextLatent =
            applyTanh(layerNormalizeVector(applyLinear(contextEncoder, example.visible)));
        const auto prediction =
            applyTanh(layerNormalizeVector(applyLinear(predictor, contextLatent)));
        const auto targetLatent =
            applyTanh(layerNormalizeVector(applyLinear(targetEncoder, example.target)));

        visibleEmbeddings.push_back(example.visible);
        contextEmbeddings.push_back(contextLatent);
        targetEmbeddings.push_back(targetLatent);
        predictionEmbeddings.push_back(prediction);
    }

    const auto predictionNormalization = computeBatchStandardization(predictionEmbeddings);
    const auto targetNormalization = computeBatchStandardization(targetEmbeddings);
    const auto normalizedPredictions =
        standardizeEmbeddings(predictionEmbeddings, predictionNormalization);
    const auto normalizedTargets =
        standardizeEmbeddings(targetEmbeddings, targetNormalization);

    totalLoss = 0.0;
    totalShuffledLoss = 0.0;
    for (size_t index = 0; index < examples.size(); ++index) {
        const auto& shuffledTarget = normalizedTargets[(index + 1) % examples.size()];
        totalLoss +=
            (mseWeight * meanSquaredError(normalizedPredictions[index], normalizedTargets[index])) +
            (cosineWeight * cosineDistance(normalizedPredictions[index], normalizedTargets[index]));
        totalShuffledLoss +=
            (mseWeight * meanSquaredError(normalizedPredictions[index], shuffledTarget)) +
            (cosineWeight * cosineDistance(normalizedPredictions[index], shuffledTarget));
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
    summary.meanVariancePenalty =
        (std::max(0.0, config.varianceFloor - summary.contextVariance) +
         std::max(0.0, config.varianceFloor - summary.predictionVariance)) *
        std::max(0.0, config.variancePenalty);
    summary.meanCovariancePenalty =
        std::max(0.0, config.covariancePenalty) *
        (meanCovariancePenalty(contextEmbeddings) + meanCovariancePenalty(predictionEmbeddings));
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
    artifacts.model.contextInputDim = contextInputDim;
    artifacts.model.targetInputDim = targetInputDim;
    artifacts.model.futureSummaryAligned = requiresTemporalExamples;
    artifacts.model.temporalSpikeCodeAligned = config.targetMode == TargetMode::TemporalSpikeCode;
    artifacts.model.temporalSpikeCodeDim =
        static_cast<size_t>(std::max(1, config.temporalSpikeCodeDim));
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

    if (model.futureSummaryAligned && model.contextInputDim > 0) {
        JepaConfig summaryConfig;
        summaryConfig.encodeViewMetadata = model.encodeViewMetadata;
        summaryConfig.metadataScale = model.metadataScale;
        std::vector<double> contextSummary;
        if (model.temporalSpikeCodeAligned) {
            contextSummary = buildProjectedOnsetSustainedCode(
                sample,
                std::max<size_t>(1U, model.temporalSpikeCodeDim),
                summaryConfig,
                false);
            contextSummary =
                projectTemporalSpikeCodeForTrainer(contextSummary, model.contextInputDim);
        } else {
            contextSummary =
                buildStructuredTemporalSummary(sample, model.projectionDim, summaryConfig, false);
        }
        if (contextSummary.empty()) {
            return {};
        }
        auto contextLatent =
            applyTanh(layerNormalizeVector(applyLinear(model.contextEncoder, contextSummary)));
        auto prediction =
            applyTanh(layerNormalizeVector(applyLinear(model.predictor, contextLatent)));
        normalizeVector(&prediction);
        return prediction;
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
        auto encoded = applyTanh(applyLinear(model.contextEncoder, projection));
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

JepaPredictionError evaluateTemporalPredictionError(
    const JepaLatentSample& sample,
    const JepaModel& model) {
    JepaPredictionError result;
    if (!model.futureSummaryAligned || !model.temporalSpikeCodeAligned ||
        model.contextInputDim == 0 || model.targetInputDim == 0 ||
        model.projectionDim == 0) {
        return result;
    }

    JepaConfig summaryConfig;
    summaryConfig.encodeViewMetadata = model.encodeViewMetadata;
    summaryConfig.metadataScale = model.metadataScale;
    const size_t codeDim = std::max<size_t>(1U, model.temporalSpikeCodeDim);
    auto contextCode = buildProjectedOnsetSustainedCode(sample, codeDim, summaryConfig, false);
    auto targetCode = buildProjectedOnsetSustainedCode(sample, codeDim, summaryConfig, true);
    contextCode = projectTemporalSpikeCodeForTrainer(contextCode, model.contextInputDim);
    targetCode = projectTemporalSpikeCodeForTrainer(targetCode, model.targetInputDim);
    if (contextCode.empty() || targetCode.empty()) {
        return result;
    }

    auto contextLatent =
        applyTanh(layerNormalizeVector(applyLinear(model.contextEncoder, contextCode)));
    auto prediction =
        applyTanh(layerNormalizeVector(applyLinear(model.predictor, contextLatent)));
    auto targetLatent =
        applyTanh(layerNormalizeVector(applyLinear(model.targetEncoder, targetCode)));
    normalizeVector(&prediction);
    normalizeVector(&targetLatent);
    if (prediction.empty() || targetLatent.empty()) {
        return result;
    }

    result.available = true;
    const double distance = std::clamp(cosineDistance(prediction, targetLatent), 0.0, 2.0);
    result.cosine = std::clamp(1.0 - distance, -1.0, 1.0);
    result.error = std::clamp(0.5 * distance, 0.0, 1.0);
    return result;
}

}  // namespace snnfw::jepa

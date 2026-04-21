#ifndef SNNFW_JEPA_CONFIG_H
#define SNNFW_JEPA_CONFIG_H

#include <string>

namespace snnfw::jepa {

enum class TapSurface {
    RawStage1,
    PromotedStage1
};

enum class MaskMode {
    None,
    Branch
};

enum class TargetMode {
    BranchMask,
    TemporalFixation,
    TemporalHemisphereSummary
};

struct JepaConfig {
    bool enabled = false;
    bool exportEnabled = true;
    std::string dumpPath;
    std::string trainerDumpPath;
    TapSurface tapSurface = TapSurface::PromotedStage1;
    int maxSamples = 128;
    bool includeBranchTokens = true;
    bool includeHemisphereToken = true;
    MaskMode maskMode = MaskMode::Branch;
    TargetMode targetMode = TargetMode::TemporalFixation;
    int visibleBranchCount = 1;
    int hiddenBranchCount = 1;
    unsigned int maskSeed = 42;
    bool enforceNoLeakage = true;
    bool trainerEnabled = true;
    bool probeEnabled = false;
    int trainerEpochs = 25;
    int projectionDim = 32;
    double trainerLearningRate = 0.05;
    double trainerWeightDecay = 1e-4;
    double targetEmaDecay = 0.99;
    double varianceFloor = 0.02;
    double variancePenalty = 0.25;
    double mseLossWeight = 1.0;
    double cosineLossWeight = 1.0;
    bool encodeViewMetadata = true;
    double metadataScale = 0.2;
    std::string probeMode = "knn";
    int probeHiddenDim = 64;
    int probeEpochs = 30;
    double probeLearningRate = 0.05;
};

inline const char* tapSurfaceName(TapSurface surface) {
    switch (surface) {
        case TapSurface::RawStage1:
            return "raw_stage1";
        case TapSurface::PromotedStage1:
            return "promoted_stage1";
    }
    return "promoted_stage1";
}

inline const char* maskModeName(MaskMode mode) {
    switch (mode) {
        case MaskMode::None:
            return "none";
        case MaskMode::Branch:
            return "branch";
    }
    return "branch";
}

inline const char* targetModeName(TargetMode mode) {
    switch (mode) {
        case TargetMode::BranchMask:
            return "branch_mask";
        case TargetMode::TemporalFixation:
            return "temporal_fixation";
        case TargetMode::TemporalHemisphereSummary:
            return "temporal_hemisphere_summary";
    }
    return "temporal_fixation";
}

}  // namespace snnfw::jepa

#endif  // SNNFW_JEPA_CONFIG_H

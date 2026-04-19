#ifndef SNNFW_JEPA_SAMPLE_H
#define SNNFW_JEPA_SAMPLE_H

#include <cstddef>
#include <string>
#include <vector>

namespace snnfw::jepa {

struct Stage1TapInput {
    std::string hemisphereName;
    std::string surfaceName;
    int label = -1;
    size_t sourceIndex = 0;
    size_t sourceViewIndex = 0;
    std::vector<double> pattern;
    std::vector<size_t> branchSizes;
};

struct JepaLatentToken {
    std::string name;
    size_t offset = 0;
    size_t size = 0;
    std::vector<double> values;
};

struct JepaHemisphereView {
    std::string hemisphereName;
    std::string surfaceName;
    size_t sourceViewIndex = 0;
    std::vector<JepaLatentToken> branchTokens;
    std::vector<double> hemisphereToken;
};

struct JepaMaskedView {
    std::string hemisphereName;
    std::string surfaceName;
    size_t sourceViewIndex = 0;
    std::vector<size_t> visibleBranchIndices;
    std::vector<size_t> hiddenBranchIndices;
    std::vector<JepaLatentToken> visibleBranchTokens;
    std::vector<JepaLatentToken> hiddenBranchTokens;
    bool leakageFree = true;
    std::vector<std::string> leakageErrors;
};

struct JepaLatentSample {
    size_t sourceIndex = 0;
    int label = -1;
    std::vector<JepaHemisphereView> hemisphereViews;
    std::vector<JepaMaskedView> maskedViews;
};

struct JepaExportSummary {
    size_t sampleCount = 0;
    size_t hemisphereViewCount = 0;
    size_t branchTokenCount = 0;
    size_t hemisphereTokenCount = 0;
    size_t maskedViewCount = 0;
    size_t temporalPairCount = 0;
    size_t leakageViolationCount = 0;
};

}  // namespace snnfw::jepa

#endif  // SNNFW_JEPA_SAMPLE_H

#include "snnfw/jepa/JepaMaskSampler.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_set>

namespace snnfw::jepa {

namespace {

std::mt19937 makeRng(unsigned int baseSeed, size_t sampleSourceIndex, size_t viewIndex) {
    const unsigned int mixed =
        baseSeed ^
        static_cast<unsigned int>((sampleSourceIndex * 0x9e3779b9ULL) & 0xffffffffU) ^
        static_cast<unsigned int>((viewIndex * 0x85ebca6bULL) & 0xffffffffU);
    return std::mt19937(mixed);
}

std::vector<size_t> sampleIndices(size_t totalCount,
                                  size_t requestedCount,
                                  std::mt19937& rng) {
    std::vector<size_t> indices(totalCount);
    std::iota(indices.begin(), indices.end(), static_cast<size_t>(0));
    std::shuffle(indices.begin(), indices.end(), rng);
    if (requestedCount < indices.size()) {
        indices.resize(requestedCount);
    }
    std::sort(indices.begin(), indices.end());
    return indices;
}

std::vector<JepaLatentToken> selectTokens(const std::vector<JepaLatentToken>& source,
                                          const std::vector<size_t>& indices) {
    std::vector<JepaLatentToken> tokens;
    tokens.reserve(indices.size());
    for (size_t index : indices) {
        if (index < source.size()) {
            tokens.push_back(source[index]);
        }
    }
    return tokens;
}

}  // namespace

JepaMaskedView buildMaskedView(const JepaHemisphereView& view,
                               const JepaConfig& config,
                               size_t sampleSourceIndex,
                               size_t viewIndex) {
    JepaMaskedView maskedView;
    maskedView.hemisphereName = view.hemisphereName;
    maskedView.surfaceName = view.surfaceName;
    maskedView.sourceViewIndex = view.sourceViewIndex;

    if (config.maskMode == MaskMode::None || view.branchTokens.empty()) {
        return maskedView;
    }

    auto rng = makeRng(config.maskSeed, sampleSourceIndex, viewIndex);
    const size_t totalBranches = view.branchTokens.size();
    const size_t hiddenCount = std::min(
        totalBranches, static_cast<size_t>(std::max(0, config.hiddenBranchCount)));
    const size_t visibleCount = std::min(
        totalBranches - hiddenCount,
        static_cast<size_t>(std::max(0, config.visibleBranchCount)));

    maskedView.hiddenBranchIndices = sampleIndices(totalBranches, hiddenCount, rng);

    std::vector<size_t> remaining;
    remaining.reserve(totalBranches - maskedView.hiddenBranchIndices.size());
    for (size_t i = 0; i < totalBranches; ++i) {
        if (!std::binary_search(maskedView.hiddenBranchIndices.begin(),
                                maskedView.hiddenBranchIndices.end(), i)) {
            remaining.push_back(i);
        }
    }
    std::shuffle(remaining.begin(), remaining.end(), rng);
    if (visibleCount < remaining.size()) {
        remaining.resize(visibleCount);
    }
    std::sort(remaining.begin(), remaining.end());
    maskedView.visibleBranchIndices = std::move(remaining);

    maskedView.visibleBranchTokens =
        selectTokens(view.branchTokens, maskedView.visibleBranchIndices);
    maskedView.hiddenBranchTokens =
        selectTokens(view.branchTokens, maskedView.hiddenBranchIndices);

    std::vector<std::string> errors;
    maskedView.leakageFree = validateMaskedView(maskedView, &errors);
    maskedView.leakageErrors = std::move(errors);
    return maskedView;
}

bool validateMaskedView(const JepaMaskedView& maskedView,
                        std::vector<std::string>* errors) {
    bool leakageFree = true;

    auto pushError = [&](const std::string& message) {
        leakageFree = false;
        if (errors != nullptr) {
            errors->push_back(message);
        }
    };

    std::unordered_set<size_t> visibleSet(maskedView.visibleBranchIndices.begin(),
                                          maskedView.visibleBranchIndices.end());
    for (size_t hiddenIndex : maskedView.hiddenBranchIndices) {
        if (visibleSet.find(hiddenIndex) != visibleSet.end()) {
            pushError("visible and hidden branch indices overlap at " +
                      std::to_string(hiddenIndex));
        }
    }

    std::unordered_set<std::string> visibleNames;
    for (const auto& token : maskedView.visibleBranchTokens) {
        visibleNames.insert(token.name);
    }
    for (const auto& token : maskedView.hiddenBranchTokens) {
        if (visibleNames.find(token.name) != visibleNames.end()) {
            pushError("visible and hidden branch tokens share name '" + token.name + "'");
        }
    }

    if (maskedView.hiddenBranchIndices.empty()) {
        pushError("masked view has no hidden branches");
    }
    if (maskedView.visibleBranchIndices.empty()) {
        pushError("masked view has no visible branches");
    }

    return leakageFree;
}

void attachMaskedViews(std::vector<JepaLatentSample>& samples,
                       const JepaConfig& config) {
    if (config.maskMode == MaskMode::None) {
        return;
    }

    for (auto& sample : samples) {
        sample.maskedViews.clear();
        sample.maskedViews.reserve(sample.hemisphereViews.size());
        for (size_t viewIndex = 0; viewIndex < sample.hemisphereViews.size(); ++viewIndex) {
            sample.maskedViews.push_back(
                buildMaskedView(sample.hemisphereViews[viewIndex],
                                config,
                                sample.sourceIndex,
                                viewIndex));
        }
    }
}

}  // namespace snnfw::jepa

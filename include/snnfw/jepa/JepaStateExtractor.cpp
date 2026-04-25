#include "snnfw/jepa/JepaStateExtractor.h"
#include "snnfw/jepa/JepaMaskSampler.h"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unordered_map>

namespace snnfw::jepa {

namespace {

using json = nlohmann::json;

std::vector<JepaLatentToken> buildBranchTokens(const Stage1TapInput& tap,
                                               const JepaConfig& config) {
    std::vector<JepaLatentToken> tokens;
    if (tap.pattern.empty()) {
        return tokens;
    }

    size_t offset = 0;
    for (size_t branchIndex = 0; branchIndex < tap.branchSizes.size(); ++branchIndex) {
        const size_t branchSize = tap.branchSizes[branchIndex];
        if (branchSize == 0) {
            continue;
        }
        const size_t end = std::min(offset + branchSize, tap.pattern.size());
        if (end <= offset) {
            break;
        }

        const size_t branchOffset = offset;
        const size_t branchTokenSize = end - offset;
        const size_t subregionCount = std::min(
            branchTokenSize,
            static_cast<size_t>(std::max(1, config.branchSubregionCount)));
        for (size_t subregionIndex = 0; subregionIndex < subregionCount; ++subregionIndex) {
            const size_t localStart = (subregionIndex * branchTokenSize) / subregionCount;
            const size_t localEnd =
                ((subregionIndex + 1) * branchTokenSize) / subregionCount;
            if (localEnd <= localStart) {
                continue;
            }

            JepaLatentToken token;
            token.name = "branch_" + std::to_string(branchIndex) +
                         "_subregion_" + std::to_string(subregionIndex);
            token.offset = branchOffset + localStart;
            token.size = localEnd - localStart;
            token.values.assign(
                tap.pattern.begin() + static_cast<std::ptrdiff_t>(token.offset),
                tap.pattern.begin() + static_cast<std::ptrdiff_t>(token.offset + token.size));
            tokens.push_back(std::move(token));
        }
        offset = end;
    }

    if (tokens.empty()) {
        JepaLatentToken token;
        token.name = "branch_0_subregion_0";
        token.offset = 0;
        token.size = tap.pattern.size();
        token.values = tap.pattern;
        tokens.push_back(std::move(token));
    }

    return tokens;
}

json tokenToJson(const JepaLatentToken& token) {
    return json{
        {"name", token.name},
        {"offset", token.offset},
        {"size", token.size},
        {"values", token.values},
    };
}

json viewToJson(const JepaHemisphereView& view) {
    json branchTokens = json::array();
    for (const auto& token : view.branchTokens) {
        branchTokens.push_back(tokenToJson(token));
    }

    return json{
        {"hemisphere_name", view.hemisphereName},
        {"surface_name", view.surfaceName},
        {"source_view_index", view.sourceViewIndex},
        {"branch_tokens", std::move(branchTokens)},
        {"hemisphere_token", view.hemisphereToken},
    };
}

json maskedViewToJson(const JepaMaskedView& maskedView) {
    json visibleTokens = json::array();
    for (const auto& token : maskedView.visibleBranchTokens) {
        visibleTokens.push_back(tokenToJson(token));
    }

    json hiddenTokens = json::array();
    for (const auto& token : maskedView.hiddenBranchTokens) {
        hiddenTokens.push_back(tokenToJson(token));
    }

    return json{
        {"hemisphere_name", maskedView.hemisphereName},
        {"surface_name", maskedView.surfaceName},
        {"source_view_index", maskedView.sourceViewIndex},
        {"visible_branch_indices", maskedView.visibleBranchIndices},
        {"hidden_branch_indices", maskedView.hiddenBranchIndices},
        {"visible_branch_tokens", std::move(visibleTokens)},
        {"hidden_branch_tokens", std::move(hiddenTokens)},
        {"leakage_free", maskedView.leakageFree},
        {"leakage_errors", maskedView.leakageErrors},
    };
}

}  // namespace

std::vector<JepaLatentSample> buildStage1LatentSamples(
    const std::vector<Stage1TapInput>& taps,
    const JepaConfig& config) {
    std::vector<JepaLatentSample> samples;
    std::unordered_map<size_t, size_t> sampleIndexBySource;
    sampleIndexBySource.reserve(taps.size());

    for (const auto& tap : taps) {
        auto [it, inserted] =
            sampleIndexBySource.emplace(tap.sourceIndex, samples.size());
        if (inserted) {
            JepaLatentSample sample;
            sample.sourceIndex = tap.sourceIndex;
            sample.label = tap.label;
            samples.push_back(std::move(sample));
        }

        auto& sample = samples[it->second];
        JepaHemisphereView view;
        view.hemisphereName = tap.hemisphereName;
        view.surfaceName = tap.surfaceName;
        view.sourceViewIndex = tap.sourceViewIndex;
        if (config.includeBranchTokens) {
            view.branchTokens = buildBranchTokens(tap, config);
        }
        if (config.includeHemisphereToken) {
            view.hemisphereToken = tap.pattern;
        }
        sample.hemisphereViews.push_back(std::move(view));
    }

    attachMaskedViews(samples, config);
    return samples;
}

JepaExportSummary writeStage1LatentSamples(
    const std::vector<Stage1TapInput>& taps,
    const JepaConfig& config,
    const std::string& outputPath) {
    const auto samples = buildStage1LatentSamples(taps, config);

    json root;
    root["tap_surface"] = tapSurfaceName(config.tapSurface);
    root["mask_mode"] = maskModeName(config.maskMode);
    root["target_mode"] = targetModeName(config.targetMode);
    root["enforce_no_leakage"] = config.enforceNoLeakage;
    root["sample_count"] = samples.size();
    root["samples"] = json::array();

    JepaExportSummary summary;
    summary.sampleCount = samples.size();

    for (const auto& sample : samples) {
        json sampleJson{
            {"source_index", sample.sourceIndex},
            {"label", sample.label},
            {"hemisphere_views", json::array()},
            {"masked_views", json::array()},
        };

        for (const auto& view : sample.hemisphereViews) {
            sampleJson["hemisphere_views"].push_back(viewToJson(view));
            ++summary.hemisphereViewCount;
            summary.branchTokenCount += view.branchTokens.size();
            if (!view.hemisphereToken.empty()) {
                ++summary.hemisphereTokenCount;
            }
        }

        for (const auto& maskedView : sample.maskedViews) {
            sampleJson["masked_views"].push_back(maskedViewToJson(maskedView));
            ++summary.maskedViewCount;
            if (!maskedView.leakageFree) {
                ++summary.leakageViolationCount;
                if (config.enforceNoLeakage) {
                    throw std::runtime_error(
                        "JEPA masking leakage validation failed for source_index=" +
                        std::to_string(sample.sourceIndex) + ", hemisphere=" +
                        maskedView.hemisphereName);
                }
            }
        }

        root["samples"].push_back(std::move(sampleJson));
    }

    for (const auto& sample : samples) {
        std::unordered_map<std::string, size_t> counts;
        for (const auto& view : sample.hemisphereViews) {
            const std::string key = view.hemisphereName + "|" + view.surfaceName;
            ++counts[key];
        }
        for (const auto& entry : counts) {
            if (entry.second > 1) {
                summary.temporalPairCount += entry.second;
            }
        }
    }

    const std::filesystem::path path(outputPath);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open JEPA latent dump path: " + outputPath);
    }
    out << root.dump(2) << '\n';
    return summary;
}

}  // namespace snnfw::jepa

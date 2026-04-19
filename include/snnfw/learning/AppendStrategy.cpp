#include "snnfw/learning/AppendStrategy.h"
#include "snnfw/Logger.h"
#include <cstdint>

namespace snnfw {
namespace learning {

namespace {
size_t deterministicReplacementIndex(const std::vector<double>& pattern, size_t patternCount) {
    if (patternCount == 0) return 0;
    uint64_t hash = 1469598103934665603ULL;
    for (double value : pattern) {
        const uint32_t q = static_cast<uint32_t>(value * 1000.0);
        hash ^= static_cast<uint64_t>(q);
        hash *= 1099511628211ULL;
    }
    return static_cast<size_t>(hash % static_cast<uint64_t>(patternCount));
}
} // namespace

AppendStrategy::AppendStrategy(const Config& config)
    : PatternUpdateStrategy(config),
      blendAlpha_(config.getDoubleParam("blend_alpha", 0.2)) {
    
    SNNFW_DEBUG("AppendStrategy created: maxPatterns={}, similarityThreshold={}, blendAlpha={}",
                config.maxPatterns, config.similarityThreshold, blendAlpha_);
}

bool AppendStrategy::updatePatterns(
    std::vector<std::vector<double>>& patterns,
    const std::vector<double>& newPattern,
    std::function<double(const std::vector<double>&, const std::vector<double>&)> similarityMetric) const {
    
    // Case 1: Below capacity - simply add the new pattern
    if (patterns.size() < config_.maxPatterns) {
        patterns.push_back(newPattern);
        
        SNNFW_DEBUG("AppendStrategy: Added new pattern (total: {})", patterns.size());
        return true;
    }

    // Case 2: At capacity - find most similar and blend or replace
    
    auto [bestIdx, bestSim] = findMostSimilar(patterns, newPattern, similarityMetric);
    
    // If similar enough, blend into existing pattern
    if (bestIdx >= 0 && bestSim >= config_.similarityThreshold) {
        blendPattern(patterns[bestIdx], newPattern, blendAlpha_);
        
        SNNFW_DEBUG("AppendStrategy: Blended into pattern {} (similarity={:.3f})",
                    bestIdx, bestSim);
        return true;
    }

    // Not similar enough - deterministic replacement keeps seed-controlled A/B runs stable.
    size_t replaceIdx = deterministicReplacementIndex(newPattern, patterns.size());
    patterns[replaceIdx] = newPattern;
    
    SNNFW_DEBUG("AppendStrategy: Replaced deterministic pattern {} (best similarity={:.3f})",
                replaceIdx, bestSim);
    return true;
}

void AppendStrategy::blendPattern(
    std::vector<double>& target,
    const std::vector<double>& newPattern,
    double alpha) const {
    
    // Ensure patterns are same size
    if (target.size() != newPattern.size()) {
        SNNFW_WARN("AppendStrategy: Cannot blend patterns of different sizes ({} vs {})",
                   target.size(), newPattern.size());
        return;
    }

    // Weighted average: target = (1-alpha)*target + alpha*newPattern
    for (size_t i = 0; i < target.size(); ++i) {
        target[i] = (1.0 - alpha) * target[i] + alpha * newPattern[i];
    }

    SNNFW_TRACE("AppendStrategy: Blended pattern with alpha={:.2f}", alpha);
}

} // namespace learning
} // namespace snnfw

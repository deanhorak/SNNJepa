#include "snnfw/learning/ReplaceWorstStrategy.h"
#include "snnfw/Logger.h"
#include <algorithm>
#include <limits>

namespace snnfw {
namespace learning {

ReplaceWorstStrategy::ReplaceWorstStrategy(const Config& config)
    : PatternUpdateStrategy(config),
      blendAlpha_(config.getDoubleParam("blend_alpha", 0.2)) {
    
    SNNFW_DEBUG("ReplaceWorstStrategy created: maxPatterns={}, similarityThreshold={}, blendAlpha={}",
                config.maxPatterns, config.similarityThreshold, blendAlpha_);
}

bool ReplaceWorstStrategy::updatePatterns(
    std::vector<std::vector<double>>& patterns,
    const std::vector<double>& newPattern,
    std::function<double(const std::vector<double>&, const std::vector<double>&)> similarityMetric) const {
    
    // Ensure usage counts match pattern count
    if (usageCounts_.size() != patterns.size()) {
        usageCounts_.resize(patterns.size(), 0);
    }

    // Case 1: Below capacity - simply add the new pattern
    if (patterns.size() < config_.maxPatterns) {
        patterns.push_back(newPattern);
        usageCounts_.push_back(0);  // Initialize usage count for new pattern
        
        SNNFW_DEBUG("ReplaceWorstStrategy: Added new pattern (total: {})", patterns.size());
        return true;
    }

    // Case 2: At capacity - check if we should blend or replace
    
    // Find most similar existing pattern
    auto [bestIdx, bestSim] = findMostSimilar(patterns, newPattern, similarityMetric);
    
    // If highly similar to existing pattern, blend instead of replacing
    if (bestSim >= config_.similarityThreshold) {
        blendPattern(patterns[bestIdx], newPattern, blendAlpha_);
        // Increment usage count for the pattern we blended into
        usageCounts_[bestIdx]++;
        
        SNNFW_DEBUG("ReplaceWorstStrategy: Blended into pattern {} (similarity={:.3f}, usage={})",
                    bestIdx, bestSim, usageCounts_[bestIdx]);
        return true;
    }

    // Not similar enough to blend - find least-used pattern and replace it
    int worstIdx = findLeastUsed(patterns);
    
    if (worstIdx >= 0) {
        size_t oldUsage = usageCounts_[worstIdx];
        patterns[worstIdx] = newPattern;
        usageCounts_[worstIdx] = 0;  // Reset usage count for replaced pattern
        
        SNNFW_DEBUG("ReplaceWorstStrategy: Replaced pattern {} (old usage={}, similarity to new={:.3f})",
                    worstIdx, oldUsage, bestSim);
        return true;
    }

    // Should never reach here, but handle gracefully
    SNNFW_WARN("ReplaceWorstStrategy: Failed to update patterns (unexpected state)");
    return false;
}

void ReplaceWorstStrategy::recordPatternUsage(size_t patternIndex) const {
    if (patternIndex < usageCounts_.size()) {
        usageCounts_[patternIndex]++;
        SNNFW_TRACE("ReplaceWorstStrategy: Pattern {} usage incremented to {}",
                    patternIndex, usageCounts_[patternIndex]);
    } else {
        SNNFW_WARN("ReplaceWorstStrategy: Invalid pattern index {} (max: {})",
                   patternIndex, usageCounts_.size());
    }
}

size_t ReplaceWorstStrategy::getPatternUsage(size_t patternIndex) const {
    if (patternIndex < usageCounts_.size()) {
        return usageCounts_[patternIndex];
    }
    return 0;
}

void ReplaceWorstStrategy::resetUsageCounters() const {
    std::fill(usageCounts_.begin(), usageCounts_.end(), 0);
    SNNFW_DEBUG("ReplaceWorstStrategy: Reset all usage counters ({} patterns)", usageCounts_.size());
}

int ReplaceWorstStrategy::findLeastUsed(const std::vector<std::vector<double>>& patterns) const {
    if (patterns.empty()) {
        return -1;
    }

    // Ensure usage counts are initialized
    if (usageCounts_.size() != patterns.size()) {
        usageCounts_.resize(patterns.size(), 0);
    }

    // Find pattern with minimum usage count
    int worstIdx = 0;
    size_t minUsage = usageCounts_[0];

    for (size_t i = 1; i < usageCounts_.size(); ++i) {
        if (usageCounts_[i] < minUsage) {
            minUsage = usageCounts_[i];
            worstIdx = static_cast<int>(i);
        }
    }

    SNNFW_TRACE("ReplaceWorstStrategy: Least-used pattern is {} (usage={})", worstIdx, minUsage);
    return worstIdx;
}

void ReplaceWorstStrategy::blendPattern(
    std::vector<double>& target,
    const std::vector<double>& newPattern,
    double alpha) const {
    
    // Ensure patterns are same size
    if (target.size() != newPattern.size()) {
        SNNFW_WARN("ReplaceWorstStrategy: Cannot blend patterns of different sizes ({} vs {})",
                   target.size(), newPattern.size());
        return;
    }

    // Weighted average: target = (1-alpha)*target + alpha*newPattern
    // This implements a form of Hebbian learning where repeated similar
    // patterns strengthen the existing representation
    for (size_t i = 0; i < target.size(); ++i) {
        target[i] = (1.0 - alpha) * target[i] + alpha * newPattern[i];
    }

    SNNFW_TRACE("ReplaceWorstStrategy: Blended pattern with alpha={:.2f}", alpha);
}

} // namespace learning
} // namespace snnfw


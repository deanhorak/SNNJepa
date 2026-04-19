#include "snnfw/learning/MergeSimilarStrategy.h"
#include "snnfw/Logger.h"
#include <algorithm>

namespace snnfw {
namespace learning {

MergeSimilarStrategy::MergeSimilarStrategy(const Config& config)
    : PatternUpdateStrategy(config),
      mergeWeight_(config.getDoubleParam("merge_weight", 0.3)) {
    
    SNNFW_DEBUG("MergeSimilarStrategy created: maxPatterns={}, similarityThreshold={}, mergeWeight={}",
                config.maxPatterns, config.similarityThreshold, mergeWeight_);
}

bool MergeSimilarStrategy::updatePatterns(
    std::vector<std::vector<double>>& patterns,
    const std::vector<double>& newPattern,
    std::function<double(const std::vector<double>&, const std::vector<double>&)> similarityMetric) const {
    
    // Ensure merge counts match pattern count
    if (mergeCounts_.size() != patterns.size()) {
        mergeCounts_.resize(patterns.size(), 0);
    }

    // Case 1: Empty patterns - add first pattern
    if (patterns.empty()) {
        patterns.push_back(newPattern);
        mergeCounts_.push_back(0);
        
        SNNFW_DEBUG("MergeSimilarStrategy: Added first pattern");
        return true;
    }

    // Find most similar existing pattern
    auto [bestIdx, bestSim] = findMostSimilar(patterns, newPattern, similarityMetric);

    // Case 2: Similar enough to merge - consolidate into existing pattern
    if (bestSim >= config_.similarityThreshold) {
        mergeIntoPattern(patterns[bestIdx], newPattern, mergeWeight_);
        mergeCounts_[bestIdx]++;
        
        SNNFW_DEBUG("MergeSimilarStrategy: Merged into pattern {} (similarity={:.3f}, merges={})",
                    bestIdx, bestSim, mergeCounts_[bestIdx]);
        return true;
    }

    // Case 3: Not similar enough - need to add as new pattern
    
    // If below capacity, simply add
    if (patterns.size() < config_.maxPatterns) {
        patterns.push_back(newPattern);
        mergeCounts_.push_back(0);
        
        SNNFW_DEBUG("MergeSimilarStrategy: Added new pattern (total: {}, similarity to closest={:.3f})",
                    patterns.size(), bestSim);
        return true;
    }

    // Case 4: At capacity - replace least representative pattern
    // The least representative pattern is the one with lowest average similarity
    // to all other patterns (i.e., the outlier)
    int worstIdx = findLeastRepresentative(patterns, similarityMetric);
    
    if (worstIdx >= 0) {
        size_t oldMerges = mergeCounts_[worstIdx];
        patterns[worstIdx] = newPattern;
        mergeCounts_[worstIdx] = 0;  // Reset merge count for replaced pattern
        
        SNNFW_DEBUG("MergeSimilarStrategy: Replaced pattern {} (old merges={}, was outlier)",
                    worstIdx, oldMerges);
        return true;
    }

    // Should never reach here, but handle gracefully
    SNNFW_WARN("MergeSimilarStrategy: Failed to update patterns (unexpected state)");
    return false;
}

size_t MergeSimilarStrategy::getMergeCount(size_t patternIndex) const {
    if (patternIndex < mergeCounts_.size()) {
        return mergeCounts_[patternIndex];
    }
    return 0;
}

void MergeSimilarStrategy::resetMergeCounters() const {
    std::fill(mergeCounts_.begin(), mergeCounts_.end(), 0);
    SNNFW_DEBUG("MergeSimilarStrategy: Reset all merge counters ({} patterns)", mergeCounts_.size());
}

void MergeSimilarStrategy::mergeIntoPattern(
    std::vector<double>& target,
    const std::vector<double>& newPattern,
    double weight) const {
    
    // Ensure patterns are same size
    if (target.size() != newPattern.size()) {
        SNNFW_WARN("MergeSimilarStrategy: Cannot merge patterns of different sizes ({} vs {})",
                   target.size(), newPattern.size());
        return;
    }

    // Weighted average: target = (1-weight)*target + weight*newPattern
    // This creates a "prototype" pattern that represents the common features
    // of all merged patterns. The weight parameter controls how much the
    // prototype adapts to new examples:
    // - Low weight (e.g., 0.1): Stable prototype, slow adaptation
    // - High weight (e.g., 0.5): Adaptive prototype, fast adaptation
    //
    // Biological analogy: This mimics how semantic memories in the neocortex
    // are gradually refined through repeated exposure to similar experiences.
    for (size_t i = 0; i < target.size(); ++i) {
        target[i] = (1.0 - weight) * target[i] + weight * newPattern[i];
    }

    SNNFW_TRACE("MergeSimilarStrategy: Merged pattern with weight={:.2f}", weight);
}

} // namespace learning
} // namespace snnfw


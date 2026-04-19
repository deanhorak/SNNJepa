#include "snnfw/learning/HybridStrategy.h"
#include "snnfw/BinaryPattern.h"
#include <algorithm>
#include <limits>
#include <cmath>
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

size_t deterministicReplacementIndex(const BinaryPattern& pattern, size_t patternCount) {
    if (patternCount == 0) return 0;
    uint64_t hash = 1469598103934665603ULL;
    for (uint8_t bin : pattern.getData()) {
        hash ^= static_cast<uint64_t>(bin);
        hash *= 1099511628211ULL;
    }
    return static_cast<size_t>(hash % static_cast<uint64_t>(patternCount));
}
} // namespace

HybridStrategy::HybridStrategy(const Config& config)
    : PatternUpdateStrategy(config)
    , totalMerges_(0)
    , totalPrunes_(0)
    , totalBlends_(0)
    , totalAdds_(0)
    , mergeThreshold_(0.85)
    , mergeWeight_(0.3)
    , blendAlpha_(0.2)
    , pruneThreshold_(2)
{
    // Read merge_threshold from config (should be higher than similarityThreshold)
    auto mergeThreshIt = config.doubleParams.find("merge_threshold");
    if (mergeThreshIt != config.doubleParams.end()) {
        mergeThreshold_ = mergeThreshIt->second;
    }
    
    // Read merge_weight from config
    auto mergeWeightIt = config.doubleParams.find("merge_weight");
    if (mergeWeightIt != config.doubleParams.end()) {
        mergeWeight_ = mergeWeightIt->second;
    }
    
    // Read blend_alpha from config
    auto blendAlphaIt = config.doubleParams.find("blend_alpha");
    if (blendAlphaIt != config.doubleParams.end()) {
        blendAlpha_ = blendAlphaIt->second;
    }
    
    // Read prune_threshold from config
    auto pruneThreshIt = config.intParams.find("prune_threshold");
    if (pruneThreshIt != config.intParams.end()) {
        pruneThreshold_ = static_cast<size_t>(pruneThreshIt->second);
    }
    
    // Validate: merge_threshold should be >= similarityThreshold
    if (mergeThreshold_ < config.similarityThreshold) {
        mergeThreshold_ = config.similarityThreshold + 0.1;
    }
}

bool HybridStrategy::updatePatterns(
    std::vector<std::vector<double>>& patterns,
    const std::vector<double>& newPattern,
    std::function<double(const std::vector<double>&, const std::vector<double>&)> similarityMetric) const
{
    // Ensure tracking vectors match pattern count
    if (usageCounts_.size() != patterns.size()) {
        usageCounts_.resize(patterns.size(), 0);
    }
    if (mergeCounts_.size() != patterns.size()) {
        mergeCounts_.resize(patterns.size(), 0);
    }
    
    // Case 1: Below capacity - just add the pattern
    if (patterns.size() < config_.maxPatterns) {
        patterns.push_back(newPattern);
        usageCounts_.push_back(0);
        mergeCounts_.push_back(0);
        totalAdds_++;
        return true;
    }
    
    // Case 2: At capacity - find most similar pattern
    auto [bestIdx, bestSim] = findMostSimilar(patterns, newPattern, similarityMetric);
    
    if (bestIdx < 0) {
        // No patterns exist (shouldn't happen, but handle gracefully)
        patterns.push_back(newPattern);
        usageCounts_.push_back(0);
        mergeCounts_.push_back(0);
        totalAdds_++;
        return true;
    }
    
    // Case 2a: Very high similarity - MERGE into prototype
    if (bestSim >= mergeThreshold_) {
        mergeIntoPattern(patterns[bestIdx], newPattern, mergeWeight_);
        mergeCounts_[bestIdx]++;
        usageCounts_[bestIdx]++;
        totalMerges_++;
        return true;
    }
    
    // Case 2b: Medium similarity - BLEND (Hebbian strengthening)
    if (bestSim >= config_.similarityThreshold) {
        blendPattern(patterns[bestIdx], newPattern, blendAlpha_);
        usageCounts_[bestIdx]++;
        totalBlends_++;
        return true;
    }
    
    // Case 2c: Low similarity - novel pattern, need to PRUNE and replace
    int leastUsedIdx = findLeastUsed(patterns);
    if (leastUsedIdx >= 0) {
        patterns[leastUsedIdx] = newPattern;
        usageCounts_[leastUsedIdx] = 0;
        mergeCounts_[leastUsedIdx] = 0;
        totalPrunes_++;
        return true;
    }
    
    // Fallback: deterministic replacement keeps runs reproducible.
    size_t replaceIdx = deterministicReplacementIndex(newPattern, patterns.size());
    patterns[replaceIdx] = newPattern;
    usageCounts_[replaceIdx] = 0;
    mergeCounts_[replaceIdx] = 0;
    totalPrunes_++;
    return true;
}

int HybridStrategy::findLeastUsed(const std::vector<std::vector<double>>& patterns) const {
    if (patterns.empty()) {
        return -1;
    }
    
    // Ensure tracking vectors are sized correctly
    if (usageCounts_.size() != patterns.size()) {
        usageCounts_.resize(patterns.size(), 0);
    }
    
    // Find pattern with minimum usage count
    size_t minUsage = std::numeric_limits<size_t>::max();
    int minIdx = -1;
    
    for (size_t i = 0; i < patterns.size(); ++i) {
        if (usageCounts_[i] < minUsage) {
            minUsage = usageCounts_[i];
            minIdx = static_cast<int>(i);
        }
    }
    
    return minIdx;
}

void HybridStrategy::mergeIntoPattern(
    std::vector<double>& target,
    const std::vector<double>& newPattern,
    double weight) const
{
    if (target.size() != newPattern.size()) {
        return;  // Size mismatch - cannot merge
    }
    
    // Weighted average: target = (1-weight)*target + weight*newPattern
    // This creates a prototype that represents both patterns
    for (size_t i = 0; i < target.size(); ++i) {
        target[i] = (1.0 - weight) * target[i] + weight * newPattern[i];
    }
}

void HybridStrategy::blendPattern(
    std::vector<double>& target,
    const std::vector<double>& newPattern,
    double alpha) const
{
    if (target.size() != newPattern.size()) {
        return;  // Size mismatch - cannot blend
    }
    
    // Blend: target = (1-alpha)*target + alpha*newPattern
    // This refines the existing pattern with new information
    for (size_t i = 0; i < target.size(); ++i) {
        target[i] = (1.0 - alpha) * target[i] + alpha * newPattern[i];
    }
}

void HybridStrategy::recordPatternUsage(size_t patternIndex) const {
    if (patternIndex < usageCounts_.size()) {
        usageCounts_[patternIndex]++;
    }
}

size_t HybridStrategy::getPatternUsage(size_t patternIndex) const {
    if (patternIndex < usageCounts_.size()) {
        return usageCounts_[patternIndex];
    }
    return 0;
}

size_t HybridStrategy::getMergeCount(size_t patternIndex) const {
    if (patternIndex < mergeCounts_.size()) {
        return mergeCounts_[patternIndex];
    }
    return 0;
}

void HybridStrategy::resetCounters() const {
    std::fill(usageCounts_.begin(), usageCounts_.end(), 0);
    std::fill(mergeCounts_.begin(), mergeCounts_.end(), 0);
    totalMerges_ = 0;
    totalPrunes_ = 0;
    totalBlends_ = 0;
    totalAdds_ = 0;
}

HybridStrategy::Statistics HybridStrategy::getStatistics() const {
    return Statistics{
        totalMerges_,
        totalPrunes_,
        totalBlends_,
        totalAdds_
    };
}

// ============================================================================
// BinaryPattern version implementations
// ============================================================================

bool HybridStrategy::updatePatterns(
    std::vector<BinaryPattern>& patterns,
    const BinaryPattern& newPattern,
    std::function<double(const BinaryPattern&, const BinaryPattern&)> similarityMetric) const
{
    // Ensure tracking vectors match pattern count
    if (usageCounts_.size() != patterns.size()) {
        usageCounts_.resize(patterns.size(), 0);
    }
    if (mergeCounts_.size() != patterns.size()) {
        mergeCounts_.resize(patterns.size(), 0);
    }

    // Case 1: Below capacity - just add the pattern
    if (patterns.size() < config_.maxPatterns) {
        patterns.push_back(newPattern);
        usageCounts_.push_back(0);
        mergeCounts_.push_back(0);
        totalAdds_++;
        return true;
    }

    // Case 2: At capacity - find most similar pattern
    int bestIdx = -1;
    double bestSim = 0.0;

    if (!patterns.empty()) {
        bestIdx = 0;
        bestSim = similarityMetric(patterns[0], newPattern);

        for (size_t i = 1; i < patterns.size(); ++i) {
            double sim = similarityMetric(patterns[i], newPattern);
            if (sim > bestSim) {
                bestSim = sim;
                bestIdx = static_cast<int>(i);
            }
        }
    }

    if (bestIdx < 0) {
        // No patterns exist (shouldn't happen, but handle gracefully)
        patterns.push_back(newPattern);
        usageCounts_.push_back(0);
        mergeCounts_.push_back(0);
        totalAdds_++;
        return true;
    }

    // Case 2a: Very high similarity - MERGE into prototype
    if (bestSim >= mergeThreshold_) {
        mergeIntoPattern(patterns[bestIdx], newPattern, mergeWeight_);
        mergeCounts_[bestIdx]++;
        usageCounts_[bestIdx]++;
        totalMerges_++;
        return true;
    }

    // Case 2b: Medium similarity - BLEND (Hebbian strengthening)
    if (bestSim >= config_.similarityThreshold) {
        blendPattern(patterns[bestIdx], newPattern, blendAlpha_);
        usageCounts_[bestIdx]++;
        totalBlends_++;
        return true;
    }

    // Case 2c: Low similarity - novel pattern, need to PRUNE and replace
    int leastUsedIdx = findLeastUsed(patterns);
    if (leastUsedIdx >= 0) {
        patterns[leastUsedIdx] = newPattern;
        usageCounts_[leastUsedIdx] = 0;
        mergeCounts_[leastUsedIdx] = 0;
        totalPrunes_++;
        return true;
    }

    // Fallback: deterministic replacement keeps runs reproducible.
    size_t replaceIdx = deterministicReplacementIndex(newPattern, patterns.size());
    patterns[replaceIdx] = newPattern;
    usageCounts_[replaceIdx] = 0;
    mergeCounts_[replaceIdx] = 0;
    totalPrunes_++;
    return true;
}

int HybridStrategy::findLeastUsed(const std::vector<BinaryPattern>& patterns) const {
    if (patterns.empty()) {
        return -1;
    }

    // Ensure tracking vectors are sized correctly
    if (usageCounts_.size() != patterns.size()) {
        usageCounts_.resize(patterns.size(), 0);
    }

    // Find pattern with minimum usage count
    size_t minUsage = std::numeric_limits<size_t>::max();
    int minIdx = -1;

    for (size_t i = 0; i < patterns.size(); ++i) {
        if (usageCounts_[i] < minUsage) {
            minUsage = usageCounts_[i];
            minIdx = static_cast<int>(i);
        }
    }

    return minIdx;
}

void HybridStrategy::mergeIntoPattern(
    BinaryPattern& target,
    const BinaryPattern& newPattern,
    double weight) const
{
    // Use BinaryPattern's built-in merge method
    BinaryPattern::merge(target, newPattern, weight);
}

void HybridStrategy::blendPattern(
    BinaryPattern& target,
    const BinaryPattern& newPattern,
    double alpha) const
{
    // Use BinaryPattern's built-in blend method
    BinaryPattern::blend(target, newPattern, alpha);
}

} // namespace learning
} // namespace snnfw

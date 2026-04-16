#ifndef SNNFW_REPLACE_WORST_STRATEGY_H
#define SNNFW_REPLACE_WORST_STRATEGY_H

#include "snnfw/learning/PatternUpdateStrategy.h"

namespace snnfw {
namespace learning {

/**
 * @brief Pattern update strategy that replaces least-used patterns
 *
 * This strategy implements a biologically-plausible pattern management approach
 * inspired by synaptic pruning and homeostatic plasticity in biological neurons.
 *
 * Biological Basis:
 * ----------------
 * In biological neural networks, synapses that are rarely activated undergo
 * pruning (elimination), while frequently activated synapses are strengthened
 * and maintained. This is a form of "use it or lose it" plasticity that:
 * 
 * 1. Optimizes neural resources by removing unused connections
 * 2. Maintains network efficiency and prevents runaway growth
 * 3. Allows adaptation to changing input statistics
 * 4. Implements a form of competitive learning
 *
 * Implementation:
 * --------------
 * When pattern capacity is reached, this strategy:
 * 1. Tracks usage count for each stored pattern
 * 2. Identifies the least-used (least-activated) pattern
 * 3. Replaces it with the new pattern
 * 4. Resets the usage count for the new pattern
 *
 * This mimics synaptic pruning where weak/unused synapses are eliminated
 * to make room for new, potentially more relevant connections.
 *
 * Parameters:
 * ----------
 * - maxPatterns: Maximum number of patterns to store
 * - similarityThreshold: Threshold for considering patterns similar
 * - blend_alpha: Blending factor when merging similar patterns (default: 0.2)
 *
 * Usage:
 * -----
 * ```cpp
 * PatternUpdateStrategy::Config config;
 * config.name = "replace_worst";
 * config.maxPatterns = 100;
 * config.similarityThreshold = 0.7;
 * config.doubleParams["blend_alpha"] = 0.2;
 * 
 * auto strategy = std::make_unique<ReplaceWorstStrategy>(config);
 * ```
 *
 * References:
 * ----------
 * - Turrigiano & Nelson (2004) - Homeostatic plasticity in the developing nervous system
 * - Chechik et al. (1998) - Synaptic pruning in development: A computational account
 * - Zenke et al. (2013) - Synaptic plasticity in neural networks needs homeostasis
 */
class ReplaceWorstStrategy : public PatternUpdateStrategy {
public:
    /**
     * @brief Constructor
     * @param config Strategy configuration
     */
    explicit ReplaceWorstStrategy(const Config& config);

    /**
     * @brief Destructor
     */
    virtual ~ReplaceWorstStrategy() = default;

    /**
     * @brief Update pattern storage with a new pattern
     * @param patterns Current stored patterns (may be modified)
     * @param newPattern New pattern to add/merge
     * @param similarityMetric Function to compute similarity between patterns
     * @return True if patterns were modified, false otherwise
     *
     * Algorithm:
     * 1. If below capacity, add new pattern
     * 2. If at capacity:
     *    a. Find most similar existing pattern
     *    b. If similarity > threshold, blend new pattern into existing
     *    c. Otherwise, find least-used pattern and replace it
     */
    bool updatePatterns(
        std::vector<std::vector<double>>& patterns,
        const std::vector<double>& newPattern,
        std::function<double(const std::vector<double>&, const std::vector<double>&)> similarityMetric) const override;

    /**
     * @brief Get the strategy name
     */
    std::string getName() const override { return "ReplaceWorst"; }

    /**
     * @brief Track pattern usage (called when pattern is matched during inference)
     * @param patternIndex Index of the pattern that was matched
     *
     * This method should be called externally when a pattern is successfully
     * matched during inference/testing. It increments the usage counter for
     * that pattern, influencing future replacement decisions.
     */
    void recordPatternUsage(size_t patternIndex) const;

    /**
     * @brief Get usage count for a pattern
     * @param patternIndex Index of the pattern
     * @return Usage count (number of times pattern was matched)
     */
    size_t getPatternUsage(size_t patternIndex) const;

    /**
     * @brief Reset all usage counters
     *
     * Useful when starting a new training phase or experiment.
     */
    void resetUsageCounters() const;

    /**
     * @brief Get total number of patterns being tracked
     */
    size_t getTrackedPatternCount() const { return usageCounts_.size(); }

private:
    /**
     * @brief Find the least-used pattern
     * @param patterns Current stored patterns
     * @return Index of least-used pattern, or -1 if empty
     *
     * Biological analogy: Identifies the "weakest synapse" for pruning.
     */
    int findLeastUsed(const std::vector<std::vector<double>>& patterns) const;

    /**
     * @brief Blend new pattern into existing pattern
     * @param target Target pattern to blend into
     * @param newPattern New pattern to blend
     * @param alpha Blending factor (0.0 to 1.0)
     *
     * Biological analogy: Synaptic strengthening through repeated activation.
     * The existing pattern (synapse) is modified by the new input, implementing
     * a form of Hebbian learning ("neurons that fire together, wire together").
     */
    void blendPattern(
        std::vector<double>& target,
        const std::vector<double>& newPattern,
        double alpha) const;

    // Mutable to allow tracking during const operations
    mutable std::vector<size_t> usageCounts_;  ///< Usage count for each pattern
    double blendAlpha_;                         ///< Blending factor for pattern merging
};

} // namespace learning
} // namespace snnfw

#endif // SNNFW_REPLACE_WORST_STRATEGY_H


#ifndef SNNFW_APPEND_STRATEGY_H
#define SNNFW_APPEND_STRATEGY_H

#include "snnfw/learning/PatternUpdateStrategy.h"

namespace snnfw {
namespace learning {

/**
 * @brief Simple append-only pattern update strategy
 *
 * This is the baseline strategy that simply appends new patterns until
 * capacity is reached, then blends new patterns into the most similar
 * existing pattern. This is the current default behavior in SNNFW.
 *
 * Biological Basis:
 * ----------------
 * This strategy mimics early learning where the brain rapidly stores
 * new experiences without much consolidation. It's similar to how the
 * hippocampus quickly encodes episodic memories during initial learning.
 *
 * However, unlike biological systems, this strategy doesn't implement
 * pruning or consolidation, so it may accumulate redundant patterns.
 *
 * Implementation:
 * --------------
 * 1. If below capacity: add new pattern
 * 2. If at capacity:
 *    a. Find most similar existing pattern
 *    b. If similarity >= threshold: blend into existing pattern
 *    c. Otherwise: replace random pattern (fallback)
 *
 * This is the simplest strategy and serves as a baseline for comparison
 * with more sophisticated strategies like ReplaceWorst and MergeSimilar.
 *
 * Parameters:
 * ----------
 * - maxPatterns: Maximum number of patterns to store
 * - similarityThreshold: Threshold for blending patterns
 * - blend_alpha: Blending factor (default: 0.2)
 *
 * Usage:
 * -----
 * ```cpp
 * PatternUpdateStrategy::Config config;
 * config.name = "append";
 * config.maxPatterns = 100;
 * config.similarityThreshold = 0.7;
 * config.doubleParams["blend_alpha"] = 0.2;
 * 
 * auto strategy = std::make_unique<AppendStrategy>(config);
 * ```
 */
class AppendStrategy : public PatternUpdateStrategy {
public:
    /**
     * @brief Constructor
     * @param config Strategy configuration
     */
    explicit AppendStrategy(const Config& config);

    /**
     * @brief Destructor
     */
    virtual ~AppendStrategy() = default;

    /**
     * @brief Update pattern storage with a new pattern
     * @param patterns Current stored patterns (may be modified)
     * @param newPattern New pattern to add/merge
     * @param similarityMetric Function to compute similarity between patterns
     * @return True if patterns were modified, false otherwise
     */
    bool updatePatterns(
        std::vector<std::vector<double>>& patterns,
        const std::vector<double>& newPattern,
        std::function<double(const std::vector<double>&, const std::vector<double>&)> similarityMetric) const override;

    /**
     * @brief Get the strategy name
     */
    std::string getName() const override { return "Append"; }

private:
    /**
     * @brief Blend new pattern into existing pattern
     * @param target Target pattern to blend into
     * @param newPattern New pattern to blend
     * @param alpha Blending factor (0.0 to 1.0)
     */
    void blendPattern(
        std::vector<double>& target,
        const std::vector<double>& newPattern,
        double alpha) const;

    double blendAlpha_;  ///< Blending factor for pattern merging
};

} // namespace learning
} // namespace snnfw

#endif // SNNFW_APPEND_STRATEGY_H


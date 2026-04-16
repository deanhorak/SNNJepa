#ifndef SNNFW_MERGE_SIMILAR_STRATEGY_H
#define SNNFW_MERGE_SIMILAR_STRATEGY_H

#include "snnfw/learning/PatternUpdateStrategy.h"

namespace snnfw {
namespace learning {

/**
 * @brief Pattern update strategy that merges similar patterns
 *
 * This strategy implements a biologically-plausible pattern management approach
 * inspired by synaptic consolidation and memory compression in biological systems.
 *
 * Biological Basis:
 * ----------------
 * In biological neural networks, similar experiences and memories undergo
 * consolidation - a process where related patterns are merged and compressed
 * into more general representations. This process:
 *
 * 1. Reduces memory requirements by eliminating redundancy
 * 2. Enables generalization from specific examples
 * 3. Implements a form of prototype learning
 * 4. Mirrors sleep-dependent memory consolidation in the brain
 *
 * The hippocampus initially stores specific episodic memories, which are
 * gradually consolidated into more general semantic memories in the neocortex.
 * This strategy mimics that consolidation process.
 *
 * Implementation:
 * --------------
 * When a new pattern arrives:
 * 1. Find the most similar existing pattern
 * 2. If similarity exceeds merge_threshold:
 *    - Merge the new pattern into the existing one (weighted average)
 *    - This creates a more general "prototype" pattern
 * 3. If not similar enough:
 *    - Add as new pattern (if space available)
 *    - Or replace least representative pattern (if at capacity)
 *
 * This creates a compressed representation where similar patterns are
 * consolidated into prototypes, analogous to how the brain forms concepts
 * from specific examples.
 *
 * Parameters:
 * ----------
 * - maxPatterns: Maximum number of patterns to store
 * - similarityThreshold: Threshold for considering patterns similar enough to merge
 * - merge_weight: Weight for new pattern in merge (default: 0.3)
 *   Higher values make the prototype adapt faster to new examples
 *
 * Usage:
 * -----
 * ```cpp
 * PatternUpdateStrategy::Config config;
 * config.name = "merge_similar";
 * config.maxPatterns = 100;
 * config.similarityThreshold = 0.75;  // Higher threshold = more selective merging
 * config.doubleParams["merge_weight"] = 0.3;
 * 
 * auto strategy = std::make_unique<MergeSimilarStrategy>(config);
 * ```
 *
 * References:
 * ----------
 * - McClelland et al. (1995) - Why there are complementary learning systems
 * - Kumaran et al. (2016) - What learning systems do intelligent agents need?
 * - Stickgold & Walker (2013) - Sleep-dependent memory triage
 * - Marr (1971) - Simple memory: A theory for archicortex
 */
class MergeSimilarStrategy : public PatternUpdateStrategy {
public:
    /**
     * @brief Constructor
     * @param config Strategy configuration
     */
    explicit MergeSimilarStrategy(const Config& config);

    /**
     * @brief Destructor
     */
    virtual ~MergeSimilarStrategy() = default;

    /**
     * @brief Update pattern storage with a new pattern
     * @param patterns Current stored patterns (may be modified)
     * @param newPattern New pattern to add/merge
     * @param similarityMetric Function to compute similarity between patterns
     * @return True if patterns were modified, false otherwise
     *
     * Algorithm:
     * 1. Find most similar existing pattern
     * 2. If similarity >= threshold:
     *    - Merge new pattern into existing (consolidation)
     *    - Increment merge count for that pattern
     * 3. If similarity < threshold:
     *    a. If below capacity: add as new pattern
     *    b. If at capacity: replace least representative pattern
     */
    bool updatePatterns(
        std::vector<std::vector<double>>& patterns,
        const std::vector<double>& newPattern,
        std::function<double(const std::vector<double>&, const std::vector<double>&)> similarityMetric) const override;

    /**
     * @brief Get the strategy name
     */
    std::string getName() const override { return "MergeSimilar"; }

    /**
     * @brief Get merge count for a pattern
     * @param patternIndex Index of the pattern
     * @return Number of times this pattern has been merged with others
     *
     * Higher merge counts indicate "prototype" patterns that represent
     * many similar examples - analogous to semantic memories in the brain.
     */
    size_t getMergeCount(size_t patternIndex) const;

    /**
     * @brief Reset all merge counters
     */
    void resetMergeCounters() const;

    /**
     * @brief Get total number of patterns being tracked
     */
    size_t getTrackedPatternCount() const { return mergeCounts_.size(); }

private:
    /**
     * @brief Merge new pattern into existing pattern
     * @param target Target pattern to merge into
     * @param newPattern New pattern to merge
     * @param weight Weight for new pattern (0.0 to 1.0)
     *
     * Biological analogy: Synaptic consolidation where repeated similar
     * activation patterns strengthen and refine the synaptic representation.
     * The merged pattern becomes a "prototype" representing the common
     * features of all merged patterns.
     */
    void mergeIntoPattern(
        std::vector<double>& target,
        const std::vector<double>& newPattern,
        double weight) const;

    // Mutable to allow tracking during const operations
    mutable std::vector<size_t> mergeCounts_;  ///< Number of merges for each pattern
    double mergeWeight_;                        ///< Weight for new pattern in merge
};

} // namespace learning
} // namespace snnfw

#endif // SNNFW_MERGE_SIMILAR_STRATEGY_H


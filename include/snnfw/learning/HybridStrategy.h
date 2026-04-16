#ifndef SNNFW_HYBRID_STRATEGY_H
#define SNNFW_HYBRID_STRATEGY_H

#include "snnfw/learning/PatternUpdateStrategy.h"
#include "snnfw/BinaryPattern.h"

namespace snnfw {
namespace learning {

/**
 * @brief Hybrid pattern update strategy combining pruning and consolidation
 *
 * This strategy implements a biologically-plausible pattern management approach
 * that combines synaptic pruning (ReplaceWorst) with selective consolidation
 * (MergeSimilar) to achieve optimal pattern diversity and generalization.
 *
 * Biological Basis:
 * ----------------
 * Real biological neural networks employ multiple complementary mechanisms:
 * 
 * 1. **Synaptic Pruning**: Elimination of unused/weak synapses
 *    - Maintains network efficiency
 *    - Prevents resource waste on irrelevant patterns
 *    - "Use it or lose it" principle
 *
 * 2. **Synaptic Consolidation**: Merging of similar activation patterns
 *    - Reduces redundancy
 *    - Enables generalization
 *    - Forms prototype representations
 *
 * 3. **Homeostatic Plasticity**: Balance between stability and plasticity
 *    - Prevents runaway growth or decay
 *    - Maintains optimal pattern diversity
 *
 * This hybrid strategy implements all three mechanisms in a coordinated way,
 * mimicking how biological systems balance pattern storage, consolidation,
 * and pruning to achieve optimal learning and memory.
 *
 * Implementation Strategy:
 * -----------------------
 * The hybrid approach uses a two-tier similarity threshold system:
 *
 * 1. **High Similarity (>= merge_threshold)**: MERGE
 *    - Patterns are very similar → consolidate into prototype
 *    - Reduces redundancy while preserving information
 *    - Biological: Memory consolidation
 *
 * 2. **Medium Similarity (>= similarity_threshold, < merge_threshold)**: BLEND
 *    - Patterns are somewhat similar → blend into existing
 *    - Refines existing patterns with new information
 *    - Biological: Hebbian learning / synaptic strengthening
 *
 * 3. **Low Similarity (< similarity_threshold)**: ADD or REPLACE
 *    - Pattern is novel → needs separate storage
 *    - If at capacity: replace least-used pattern (pruning)
 *    - Biological: New synapse formation / synaptic pruning
 *
 * This creates a dynamic balance:
 * - Similar patterns are consolidated (reduces redundancy)
 * - Diverse patterns are preserved (maintains discrimination)
 * - Unused patterns are pruned (optimizes resources)
 *
 * Parameters:
 * ----------
 * - maxPatterns: Maximum number of patterns to store
 * - similarityThreshold: Minimum similarity for blending (default: 0.7)
 * - merge_threshold: Minimum similarity for merging (default: 0.85)
 *   Should be higher than similarityThreshold
 * - merge_weight: Weight for new pattern in merge (default: 0.3)
 * - blend_alpha: Blending factor for similar patterns (default: 0.2)
 * - prune_threshold: Minimum usage count to avoid pruning (default: 2)
 *   Patterns used fewer times are candidates for replacement
 *
 * Usage:
 * -----
 * ```cpp
 * PatternUpdateStrategy::Config config;
 * config.name = "hybrid";
 * config.maxPatterns = 100;
 * config.similarityThreshold = 0.7;   // Blend threshold
 * config.doubleParams["merge_threshold"] = 0.85;  // Merge threshold (higher)
 * config.doubleParams["merge_weight"] = 0.3;
 * config.doubleParams["blend_alpha"] = 0.2;
 * config.intParams["prune_threshold"] = 2;
 * 
 * auto strategy = std::make_unique<HybridStrategy>(config);
 * ```
 *
 * Expected Benefits:
 * -----------------
 * 1. **Better Generalization**: Merging similar patterns creates prototypes
 * 2. **Maintained Diversity**: Pruning ensures diverse pattern coverage
 * 3. **Resource Efficiency**: Optimal use of limited pattern storage
 * 4. **Adaptive Learning**: Balances stability (consolidation) and plasticity (pruning)
 *
 * References:
 * ----------
 * - Turrigiano & Nelson (2004) - Homeostatic plasticity
 * - McClelland et al. (1995) - Complementary learning systems
 * - Zenke et al. (2013) - Synaptic plasticity needs homeostasis
 * - Chechik et al. (1998) - Synaptic pruning in development
 */
class HybridStrategy : public PatternUpdateStrategy {
public:
    /**
     * @brief Constructor
     * @param config Strategy configuration
     */
    explicit HybridStrategy(const Config& config);

    /**
     * @brief Destructor
     */
    virtual ~HybridStrategy() = default;

    /**
     * @brief Update pattern storage with a new pattern (vector<double> version)
     * @param patterns Current stored patterns (may be modified)
     * @param newPattern New pattern to add/merge
     * @param similarityMetric Function to compute similarity between patterns
     * @return True if patterns were modified, false otherwise
     *
     * Algorithm:
     * 1. Find most similar existing pattern and its similarity
     * 2. If similarity >= merge_threshold:
     *    - MERGE: Consolidate into prototype (weighted average)
     *    - Increment merge count
     * 3. Else if similarity >= similarity_threshold:
     *    - BLEND: Refine existing pattern (Hebbian strengthening)
     *    - Increment usage count
     * 4. Else (novel pattern):
     *    a. If below capacity: ADD as new pattern
     *    b. If at capacity: PRUNE least-used pattern and replace
     */
    bool updatePatterns(
        std::vector<std::vector<double>>& patterns,
        const std::vector<double>& newPattern,
        std::function<double(const std::vector<double>&, const std::vector<double>&)> similarityMetric) const override;

    /**
     * @brief Update pattern storage with a new pattern (BinaryPattern version)
     * @param patterns Current stored patterns (may be modified)
     * @param newPattern New pattern to add/merge
     * @param similarityMetric Function to compute similarity between patterns
     * @return True if patterns were modified, false otherwise
     *
     * Same algorithm as vector<double> version, but optimized for BinaryPattern
     */
    bool updatePatterns(
        std::vector<BinaryPattern>& patterns,
        const BinaryPattern& newPattern,
        std::function<double(const BinaryPattern&, const BinaryPattern&)> similarityMetric) const override;

    /**
     * @brief Get the strategy name
     */
    std::string getName() const override { return "Hybrid"; }

    /**
     * @brief Track pattern usage (called when pattern is matched during inference)
     * @param patternIndex Index of the pattern that was matched
     */
    void recordPatternUsage(size_t patternIndex) const;

    /**
     * @brief Get usage count for a pattern
     * @param patternIndex Index of the pattern
     * @return Usage count (number of times pattern was matched)
     */
    size_t getPatternUsage(size_t patternIndex) const;

    /**
     * @brief Get merge count for a pattern
     * @param patternIndex Index of the pattern
     * @return Number of times this pattern has been merged with others
     */
    size_t getMergeCount(size_t patternIndex) const;

    /**
     * @brief Reset all counters
     */
    void resetCounters() const;

    /**
     * @brief Get statistics about pattern management
     * @return Struct containing merge count, prune count, blend count, add count
     */
    struct Statistics {
        size_t totalMerges;
        size_t totalPrunes;
        size_t totalBlends;
        size_t totalAdds;
    };
    Statistics getStatistics() const;

private:
    /**
     * @brief Find the least-used pattern (vector<double> version)
     * @param patterns Current stored patterns
     * @return Index of least-used pattern, or -1 if empty
     */
    int findLeastUsed(const std::vector<std::vector<double>>& patterns) const;

    /**
     * @brief Find the least-used pattern (BinaryPattern version)
     * @param patterns Current stored patterns
     * @return Index of least-used pattern, or -1 if empty
     */
    int findLeastUsed(const std::vector<BinaryPattern>& patterns) const;

    /**
     * @brief Merge new pattern into existing pattern (vector<double> version)
     * @param target Target pattern to merge into
     * @param newPattern New pattern to merge
     * @param weight Weight for new pattern (0.0 to 1.0)
     */
    void mergeIntoPattern(
        std::vector<double>& target,
        const std::vector<double>& newPattern,
        double weight) const;

    /**
     * @brief Merge new pattern into existing pattern (BinaryPattern version)
     * @param target Target pattern to merge into
     * @param newPattern New pattern to merge
     * @param weight Weight for new pattern (0.0 to 1.0)
     */
    void mergeIntoPattern(
        BinaryPattern& target,
        const BinaryPattern& newPattern,
        double weight) const;

    /**
     * @brief Blend new pattern into existing pattern (vector<double> version)
     * @param target Target pattern to blend into
     * @param newPattern New pattern to blend
     * @param alpha Blending factor (0.0 to 1.0)
     */
    void blendPattern(
        std::vector<double>& target,
        const std::vector<double>& newPattern,
        double alpha) const;

    /**
     * @brief Blend new pattern into existing pattern (BinaryPattern version)
     * @param target Target pattern to blend into
     * @param newPattern New pattern to blend
     * @param alpha Blending factor (0.0 to 1.0)
     */
    void blendPattern(
        BinaryPattern& target,
        const BinaryPattern& newPattern,
        double alpha) const;

    // Mutable to allow tracking during const operations
    mutable std::vector<size_t> usageCounts_;   ///< Usage count for each pattern
    mutable std::vector<size_t> mergeCounts_;   ///< Merge count for each pattern
    
    // Statistics tracking
    mutable size_t totalMerges_;
    mutable size_t totalPrunes_;
    mutable size_t totalBlends_;
    mutable size_t totalAdds_;
    
    // Configuration parameters
    double mergeThreshold_;     ///< Threshold for merging (higher than similarityThreshold)
    double mergeWeight_;        ///< Weight for new pattern in merge
    double blendAlpha_;         ///< Blending factor for similar patterns
    size_t pruneThreshold_;     ///< Minimum usage count to avoid pruning
};

} // namespace learning
} // namespace snnfw

#endif // SNNFW_HYBRID_STRATEGY_H


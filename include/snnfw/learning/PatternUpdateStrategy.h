#ifndef SNNFW_PATTERN_UPDATE_STRATEGY_H
#define SNNFW_PATTERN_UPDATE_STRATEGY_H

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <functional>

// Forward declaration
namespace snnfw {
    class BinaryPattern;
}

namespace snnfw {
namespace learning {

/**
 * @brief Base class for pattern update strategies
 *
 * Pattern update strategies determine how neurons store and update their
 * reference patterns when learning from spike trains. Different strategies
 * have different trade-offs:
 * - Append: Simple, fast, but may waste memory on redundant patterns
 * - Replace Worst: Maintains diversity by removing least representative patterns
 * - Merge Similar: Combines similar patterns to save memory and generalize
 *
 * Biological Motivation:
 * Real neurons don't have unlimited memory. Synaptic consolidation and
 * homeostatic plasticity mechanisms ensure that important patterns are
 * retained while less important ones fade. This is similar to "replace worst"
 * and "merge similar" strategies.
 *
 * References:
 * - Fusi et al. (2005) - Cascade models of synaptically stored memories
 * - Zenke et al. (2015) - Diverse synaptic plasticity mechanisms
 */
class PatternUpdateStrategy {
public:
    /**
     * @brief Strategy configuration
     */
    struct Config {
        std::string name;                           ///< Strategy name
        size_t maxPatterns;                         ///< Maximum number of patterns to store
        double similarityThreshold;                 ///< Threshold for pattern similarity (0.0 to 1.0)
        std::map<std::string, double> doubleParams; ///< Additional double parameters
        std::map<std::string, int> intParams;       ///< Additional integer parameters
        
        // Helper methods
        double getDoubleParam(const std::string& key, double defaultValue = 0.0) const {
            auto it = doubleParams.find(key);
            return (it != doubleParams.end()) ? it->second : defaultValue;
        }
        
        int getIntParam(const std::string& key, int defaultValue = 0) const {
            auto it = intParams.find(key);
            return (it != intParams.end()) ? it->second : defaultValue;
        }
    };

    /**
     * @brief Constructor
     * @param config Strategy configuration
     */
    explicit PatternUpdateStrategy(const Config& config) : config_(config) {}

    /**
     * @brief Virtual destructor
     */
    virtual ~PatternUpdateStrategy() = default;

    /**
     * @brief Update pattern storage with a new pattern (vector<double> version)
     * @param patterns Current stored patterns (may be modified)
     * @param newPattern New pattern to add/merge
     * @param similarityMetric Function to compute similarity between patterns
     * @return True if patterns were modified, false otherwise
     *
     * The strategy decides whether to:
     * - Add the new pattern (if space available)
     * - Replace an existing pattern
     * - Merge with a similar pattern
     * - Ignore the new pattern
     */
    virtual bool updatePatterns(
        std::vector<std::vector<double>>& patterns,
        const std::vector<double>& newPattern,
        std::function<double(const std::vector<double>&, const std::vector<double>&)> similarityMetric) const = 0;

    /**
     * @brief Update pattern storage with a new pattern (BinaryPattern version)
     * @param patterns Current stored patterns (may be modified)
     * @param newPattern New pattern to add/merge
     * @param similarityMetric Function to compute similarity between patterns
     * @return True if patterns were modified, false otherwise
     *
     * Default implementation: not supported (returns false)
     * Derived classes should override this to support BinaryPattern
     */
    virtual bool updatePatterns(
        std::vector<BinaryPattern>& patterns,
        const BinaryPattern& newPattern,
        std::function<double(const BinaryPattern&, const BinaryPattern&)> similarityMetric) const {
        return false;  // Default: not supported
    }

    /**
     * @brief Get the strategy name
     */
    virtual std::string getName() const = 0;

    /**
     * @brief Get the configuration
     */
    const Config& getConfig() const { return config_; }

    /**
     * @brief Get maximum number of patterns
     */
    size_t getMaxPatterns() const { return config_.maxPatterns; }

    /**
     * @brief Get similarity threshold
     */
    double getSimilarityThreshold() const { return config_.similarityThreshold; }

protected:
    Config config_; ///< Strategy configuration

    /**
     * @brief Helper: Find the most similar pattern
     * @param patterns Stored patterns
     * @param newPattern Pattern to compare
     * @param similarityMetric Similarity function
     * @return Pair of (index, similarity) for most similar pattern, or (-1, 0.0) if empty
     */
    std::pair<int, double> findMostSimilar(
        const std::vector<std::vector<double>>& patterns,
        const std::vector<double>& newPattern,
        std::function<double(const std::vector<double>&, const std::vector<double>&)> similarityMetric) const {
        
        if (patterns.empty()) {
            return {-1, 0.0};
        }

        int bestIdx = 0;
        double bestSim = similarityMetric(patterns[0], newPattern);

        for (size_t i = 1; i < patterns.size(); ++i) {
            double sim = similarityMetric(patterns[i], newPattern);
            if (sim > bestSim) {
                bestSim = sim;
                bestIdx = static_cast<int>(i);
            }
        }

        return {bestIdx, bestSim};
    }

    /**
     * @brief Helper: Find the least representative pattern
     * @param patterns Stored patterns
     * @param similarityMetric Similarity function
     * @return Index of least representative pattern, or -1 if empty
     *
     * The least representative pattern is the one with the lowest average
     * similarity to all other patterns.
     */
    int findLeastRepresentative(
        const std::vector<std::vector<double>>& patterns,
        std::function<double(const std::vector<double>&, const std::vector<double>&)> similarityMetric) const {
        
        if (patterns.empty()) {
            return -1;
        }

        if (patterns.size() == 1) {
            return 0;
        }

        int worstIdx = 0;
        double worstAvgSim = 1.0;

        for (size_t i = 0; i < patterns.size(); ++i) {
            double avgSim = 0.0;
            for (size_t j = 0; j < patterns.size(); ++j) {
                if (i != j) {
                    avgSim += similarityMetric(patterns[i], patterns[j]);
                }
            }
            avgSim /= (patterns.size() - 1);

            if (avgSim < worstAvgSim) {
                worstAvgSim = avgSim;
                worstIdx = static_cast<int>(i);
            }
        }

        return worstIdx;
    }

    /**
     * @brief Helper: Merge two patterns (weighted average)
     * @param pattern1 First pattern
     * @param pattern2 Second pattern
     * @param weight Weight for pattern2 (0.0 to 1.0)
     * @return Merged pattern
     */
    std::vector<double> mergePatterns(
        const std::vector<double>& pattern1,
        const std::vector<double>& pattern2,
        double weight = 0.5) const {
        
        std::vector<double> merged(pattern1.size());
        for (size_t i = 0; i < pattern1.size(); ++i) {
            merged[i] = (1.0 - weight) * pattern1[i] + weight * pattern2[i];
        }
        return merged;
    }
};

/**
 * @brief Factory for creating pattern update strategies
 */
class PatternUpdateStrategyFactory {
public:
    /**
     * @brief Create a pattern update strategy from configuration
     * @param type Strategy type ("append", "replace_worst", "merge_similar")
     * @param config Strategy configuration
     * @return Unique pointer to pattern update strategy
     */
    static std::unique_ptr<PatternUpdateStrategy> create(
        const std::string& type,
        const PatternUpdateStrategy::Config& config);

    /**
     * @brief Get list of available strategies
     */
    static std::vector<std::string> getAvailableStrategies();
};

} // namespace learning
} // namespace snnfw

#endif // SNNFW_PATTERN_UPDATE_STRATEGY_H


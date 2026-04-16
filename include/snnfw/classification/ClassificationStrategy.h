#ifndef SNNFW_CLASSIFICATION_STRATEGY_H
#define SNNFW_CLASSIFICATION_STRATEGY_H

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <functional>
#include <iostream>
#include <omp.h>

namespace snnfw {
namespace classification {

/**
 * @brief Base class for classification strategies
 *
 * Classification strategies determine how to classify a test pattern based on
 * a set of labeled training patterns. Different strategies have different
 * characteristics:
 * - Majority Voting: Simple k-NN with equal votes
 * - Weighted Distance: Closer neighbors have more influence
 * - Weighted Similarity: More similar neighbors have more influence
 *
 * Biological Motivation:
 * The brain doesn't use simple majority voting. Neurons with stronger
 * connections (higher similarity) have more influence on decision-making.
 * This is similar to weighted voting strategies.
 *
 * References:
 * - Cover & Hart (1967) - Nearest neighbor pattern classification
 * - Dudani (1976) - Distance-weighted k-NN rule
 */
class ClassificationStrategy {
public:
    /**
     * @brief Training pattern with label
     */
    struct LabeledPattern {
        std::vector<double> pattern; ///< Pattern (e.g., neuron activations)
        int label;                   ///< Class label
        
        LabeledPattern(const std::vector<double>& p, int l) 
            : pattern(p), label(l) {}
    };

    /**
     * @brief Strategy configuration
     */
    struct Config {
        std::string name;                           ///< Strategy name
        int k;                                      ///< Number of neighbors (for k-NN)
        int numClasses;                             ///< Number of classes
        double distanceExponent;                    ///< Exponent for distance weighting (0 = uniform)
        std::map<std::string, double> doubleParams; ///< Additional double parameters
        std::map<std::string, int> intParams;       ///< Additional integer parameters
        std::map<std::string, std::string> stringParams; ///< Additional string parameters
        
        // Helper methods
        double getDoubleParam(const std::string& key, double defaultValue = 0.0) const {
            auto it = doubleParams.find(key);
            return (it != doubleParams.end()) ? it->second : defaultValue;
        }
        
        int getIntParam(const std::string& key, int defaultValue = 0) const {
            auto it = intParams.find(key);
            return (it != intParams.end()) ? it->second : defaultValue;
        }

        std::string getStringParam(const std::string& key,
                                   const std::string& defaultValue = "") const {
            auto it = stringParams.find(key);
            return (it != stringParams.end()) ? it->second : defaultValue;
        }
    };

    /**
     * @brief Constructor
     * @param config Strategy configuration
     */
    explicit ClassificationStrategy(const Config& config) : config_(config) {}

    /**
     * @brief Virtual destructor
     */
    virtual ~ClassificationStrategy() = default;

    /**
     * @brief Classify a test pattern
     * @param testPattern Pattern to classify
     * @param trainingPatterns Labeled training patterns
     * @param similarityMetric Function to compute similarity between patterns
     * @return Predicted class label
     */
    virtual int classify(
        const std::vector<double>& testPattern,
        const std::vector<LabeledPattern>& trainingPatterns,
        std::function<double(const std::vector<double>&, const std::vector<double>&)> similarityMetric) const = 0;

    /**
     * @brief Classify a test pattern and return confidence scores
     * @param testPattern Pattern to classify
     * @param trainingPatterns Labeled training patterns
     * @param similarityMetric Function to compute similarity between patterns
     * @return Vector of confidence scores (one per class)
     */
    virtual std::vector<double> classifyWithConfidence(
        const std::vector<double>& testPattern,
        const std::vector<LabeledPattern>& trainingPatterns,
        std::function<double(const std::vector<double>&, const std::vector<double>&)> similarityMetric) const = 0;

    /**
     * @brief Get the strategy name
     */
    virtual std::string getName() const = 0;

    /**
     * @brief Get the configuration
     */
    const Config& getConfig() const { return config_; }

    /**
     * @brief Get k (number of neighbors)
     */
    int getK() const { return config_.k; }

    /**
     * @brief Get number of classes
     */
    int getNumClasses() const { return config_.numClasses; }

protected:
    Config config_; ///< Strategy configuration

    /**
     * @brief Helper: Find k nearest neighbors (parallelized)
     * @param testPattern Pattern to classify
     * @param trainingPatterns Labeled training patterns
     * @param similarityMetric Similarity function
     * @param k Number of neighbors to find
     * @return Vector of (index, similarity) pairs for k nearest neighbors
     */
    std::vector<std::pair<int, double>> findKNearestNeighbors(
        const std::vector<double>& testPattern,
        const std::vector<LabeledPattern>& trainingPatterns,
        std::function<double(const std::vector<double>&, const std::vector<double>&)> similarityMetric,
        int k) const {

        // Compute similarities for all training patterns (sequential - parallelization at higher level)
        std::vector<std::pair<int, double>> similarities(trainingPatterns.size());

        for (size_t i = 0; i < trainingPatterns.size(); ++i) {
            double sim = similarityMetric(testPattern, trainingPatterns[i].pattern);
            similarities[i] = {static_cast<int>(i), sim};
        }

        // Sort by similarity (descending)
        std::sort(similarities.begin(), similarities.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });

        // Return top k
        int actualK = std::min(k, static_cast<int>(similarities.size()));
        return std::vector<std::pair<int, double>>(
            similarities.begin(),
            similarities.begin() + actualK);
    }

    /**
     * @brief Helper: Initialize vote vector
     * @return Vector of zeros with size equal to number of classes
     */
    std::vector<double> initializeVotes() const {
        return std::vector<double>(config_.numClasses, 0.0);
    }

    /**
     * @brief Helper: Get class with maximum votes
     * @param votes Vote vector
     * @return Class label with highest vote
     */
    int getMaxVoteClass(const std::vector<double>& votes) const {
        int maxClass = 0;
        double maxVote = votes[0];
        for (size_t i = 1; i < votes.size(); ++i) {
            if (votes[i] > maxVote) {
                maxVote = votes[i];
                maxClass = static_cast<int>(i);
            }
        }
        return maxClass;
    }

    /**
     * @brief Helper: Normalize votes to probabilities
     * @param votes Vote vector
     * @return Normalized probability vector
     */
    std::vector<double> normalizeVotes(const std::vector<double>& votes) const {
        double sum = 0.0;
        for (double v : votes) {
            sum += v;
        }
        
        if (sum > 0.0) {
            std::vector<double> probs(votes.size());
            for (size_t i = 0; i < votes.size(); ++i) {
                probs[i] = votes[i] / sum;
            }
            return probs;
        }
        
        return votes;
    }
};

/**
 * @brief Factory for creating classification strategies
 */
class ClassificationStrategyFactory {
public:
    /**
     * @brief Create a classification strategy from configuration
     * @param type Strategy type ("majority", "weighted_distance", "weighted_similarity")
     * @param config Strategy configuration
     * @return Unique pointer to classification strategy
     */
    static std::unique_ptr<ClassificationStrategy> create(
        const std::string& type,
        const ClassificationStrategy::Config& config);

    /**
     * @brief Get list of available strategies
     */
    static std::vector<std::string> getAvailableStrategies();
};

} // namespace classification
} // namespace snnfw

#endif // SNNFW_CLASSIFICATION_STRATEGY_H

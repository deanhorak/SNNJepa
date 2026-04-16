#ifndef SNNFW_WEIGHTED_DISTANCE_H
#define SNNFW_WEIGHTED_DISTANCE_H

#include "snnfw/classification/ClassificationStrategy.h"
#include <vector>
#include <algorithm>
#include <cmath>

namespace snnfw {
namespace classification {

/**
 * @brief Distance-weighted k-NN classification strategy
 *
 * This strategy weights each neighbor's vote by its distance (or similarity) to
 * the test pattern. Closer neighbors have more influence on the classification.
 *
 * Algorithm:
 * 1. Find k nearest neighbors based on similarity metric
 * 2. Convert similarity to distance: distance = 1 - similarity
 * 3. Compute weight for each neighbor: weight = 1 / (distance^p + epsilon)
 *    where p is the distance exponent (default: 2.0)
 * 4. Each neighbor votes for its class with its weight
 * 5. Return class with highest weighted vote sum
 *
 * Weighting Function:
 * - weight = 1 / (distance^p + epsilon)
 * - p = 1: Linear weighting (Dudani, 1976)
 * - p = 2: Inverse square weighting (default, more emphasis on close neighbors)
 * - epsilon: Small constant to avoid division by zero (default: 1e-6)
 *
 * Performance Expectations:
 * - Should improve over majority voting when neighbor similarities vary
 * - Expected improvement: +0.5-1.5% on MNIST
 * - Works well when closest neighbors are most reliable
 *
 * Biological Motivation:
 * - Neurons with stronger connections (higher similarity) have more influence
 * - Similar to synaptic weight modulation in biological neural networks
 * - Mimics the brain's tendency to trust more similar past experiences
 *
 * References:
 * - Dudani (1976) - The distance-weighted k-nearest-neighbor rule
 * - MacLeod et al. (1987) - A re-examination of the distance-weighted k-NN rule
 */
class WeightedDistance : public ClassificationStrategy {
public:
    /**
     * @brief Construct a distance-weighted classifier
     * @param config Configuration (k, numClasses, distanceExponent)
     */
    explicit WeightedDistance(const Config& config);

    /**
     * @brief Classify using distance-weighted voting
     * @param testPattern Pattern to classify
     * @param trainingPatterns Labeled training patterns
     * @param similarityMetric Similarity function (e.g., cosine similarity)
     * @return Predicted class label
     */
    int classify(
        const std::vector<double>& testPattern,
        const std::vector<LabeledPattern>& trainingPatterns,
        std::function<double(const std::vector<double>&, const std::vector<double>&)> similarityMetric) const override;

    /**
     * @brief Classify with confidence scores
     * @param testPattern Pattern to classify
     * @param trainingPatterns Labeled training patterns
     * @param similarityMetric Similarity function
     * @return Confidence scores for each class (normalized weighted votes)
     */
    std::vector<double> classifyWithConfidence(
        const std::vector<double>& testPattern,
        const std::vector<LabeledPattern>& trainingPatterns,
        std::function<double(const std::vector<double>&, const std::vector<double>&)> similarityMetric) const override;

    /**
     * @brief Get strategy name
     */
    std::string getName() const override { return "WeightedDistance"; }

private:
    /**
     * @brief Compute weight for a neighbor based on distance
     * @param similarity Similarity to test pattern (0 to 1)
     * @return Weight for voting (higher = more influence)
     */
    double computeWeight(double similarity) const;
    
    static constexpr double EPSILON = 1e-6;  ///< Small constant to avoid division by zero
};

} // namespace classification
} // namespace snnfw

#endif // SNNFW_WEIGHTED_DISTANCE_H


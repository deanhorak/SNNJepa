#ifndef SNNFW_WEIGHTED_SIMILARITY_H
#define SNNFW_WEIGHTED_SIMILARITY_H

#include "snnfw/classification/ClassificationStrategy.h"
#include <vector>
#include <algorithm>
#include <cmath>

namespace snnfw {
namespace classification {

/**
 * @brief Similarity-weighted k-NN classification strategy
 *
 * This strategy weights each neighbor's vote directly by its similarity to
 * the test pattern. More similar neighbors have proportionally more influence.
 *
 * Algorithm:
 * 1. Find k nearest neighbors based on similarity metric
 * 2. Each neighbor votes for its class with weight = similarity^p
 *    where p is the similarity exponent (default: 1.0)
 * 3. Return class with highest weighted vote sum
 *
 * Weighting Function:
 * - weight = similarity^p
 * - p = 1: Linear weighting (default, proportional to similarity)
 * - p = 2: Quadratic weighting (more emphasis on highly similar neighbors)
 * - p = 0.5: Square root weighting (less emphasis on similarity differences)
 *
 * Difference from WeightedDistance:
 * - WeightedDistance: weight = 1 / (distance^p + epsilon)
 * - WeightedSimilarity: weight = similarity^p
 * - WeightedSimilarity is simpler and more direct
 * - WeightedDistance can give extreme weights for very close neighbors
 *
 * Performance Expectations:
 * - Should improve over majority voting
 * - Expected improvement: +0.5-1.5% on MNIST
 * - More stable than WeightedDistance (no division by near-zero)
 * - Works well when similarity metric is well-calibrated
 *
 * Biological Motivation:
 * - Neurons with stronger activation (higher similarity) contribute more
 * - Similar to rate coding in biological neural networks
 * - Mimics confidence-weighted decision making in the brain
 *
 * Example (with p=1):
 * - Neighbor 1: similarity = 0.95, weight = 0.95
 * - Neighbor 2: similarity = 0.70, weight = 0.70
 * - Neighbor 1 has 1.36x more influence than Neighbor 2
 *
 * Example (with p=2):
 * - Neighbor 1: similarity = 0.95, weight = 0.90
 * - Neighbor 2: similarity = 0.70, weight = 0.49
 * - Neighbor 1 has 1.84x more influence than Neighbor 2
 */
class WeightedSimilarity : public ClassificationStrategy {
public:
    /**
     * @brief Construct a similarity-weighted classifier
     * @param config Configuration (k, numClasses, distanceExponent used as similarity exponent)
     */
    explicit WeightedSimilarity(const Config& config);

    /**
     * @brief Classify using similarity-weighted voting
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
    std::string getName() const override { return "WeightedSimilarity"; }

private:
    /**
     * @brief Compute weight for a neighbor based on similarity
     * @param similarity Similarity to test pattern (0 to 1)
     * @return Weight for voting (higher = more influence)
     */
    double computeWeight(double similarity) const;
};

} // namespace classification
} // namespace snnfw

#endif // SNNFW_WEIGHTED_SIMILARITY_H


#ifndef SNNFW_MAJORITY_VOTING_H
#define SNNFW_MAJORITY_VOTING_H

#include "snnfw/classification/ClassificationStrategy.h"
#include <vector>
#include <algorithm>

namespace snnfw {
namespace classification {

/**
 * @brief Majority voting k-NN classification strategy
 *
 * This is the baseline k-NN classification approach where each of the k nearest
 * neighbors gets one vote, and the class with the most votes wins.
 *
 * Algorithm:
 * 1. Find k nearest neighbors based on similarity metric
 * 2. Each neighbor votes for its class (equal weight)
 * 3. Return class with most votes
 * 4. Ties are broken by choosing the class with highest average similarity
 *
 * Performance:
 * - Simple and fast
 * - Works well when all neighbors are equally reliable
 * - Baseline for comparison with weighted strategies
 *
 * Current MNIST Performance (8×8 grid, Sobel, Rate):
 * - k=5: 94.63% accuracy
 *
 * Limitations:
 * - Treats all k neighbors equally, even if some are much closer/more similar
 * - Ignores the strength of similarity, only uses rank
 * - Can be suboptimal when neighbor similarities vary widely
 *
 * References:
 * - Cover & Hart (1967) - Nearest neighbor pattern classification
 */
class MajorityVoting : public ClassificationStrategy {
public:
    /**
     * @brief Construct a majority voting classifier
     * @param config Configuration (k, numClasses)
     */
    explicit MajorityVoting(const Config& config);

    /**
     * @brief Classify using majority voting
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
     * @return Confidence scores for each class (normalized vote counts)
     */
    std::vector<double> classifyWithConfidence(
        const std::vector<double>& testPattern,
        const std::vector<LabeledPattern>& trainingPatterns,
        std::function<double(const std::vector<double>&, const std::vector<double>&)> similarityMetric) const override;

    /**
     * @brief Get strategy name
     */
    std::string getName() const override { return "MajorityVoting"; }
};

} // namespace classification
} // namespace snnfw

#endif // SNNFW_MAJORITY_VOTING_H


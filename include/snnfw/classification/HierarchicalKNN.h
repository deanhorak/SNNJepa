#ifndef SNNFW_HIERARCHICAL_KNN_H
#define SNNFW_HIERARCHICAL_KNN_H

#include "snnfw/classification/ClassificationStrategy.h"
#include <vector>

namespace snnfw {
namespace classification {

/**
 * @brief Generic coarse-to-fine k-NN classifier.
 *
 * This strategy first predicts a coarse group, then classifies within that group.
 * The hierarchy itself is configured externally through integer labels, so the
 * mechanism is framework-level while the actual group definitions remain task-level.
 *
 * Supported config params:
 * - stringParams["group_definitions"] = "0,1,2;3,4;5,6,7"
 * - stringParams["coarse_strategy"] = majority|weighted_similarity|weighted_distance
 * - stringParams["fine_strategy"] = majority|weighted_similarity|weighted_distance
 * - intParams["coarse_k"], intParams["fine_k"]
 * - intParams["fallback_to_flat"] = 0|1
 */
class HierarchicalKNN : public ClassificationStrategy {
public:
    explicit HierarchicalKNN(const Config& config);

    int classify(
        const std::vector<double>& testPattern,
        const std::vector<LabeledPattern>& trainingPatterns,
        std::function<double(const std::vector<double>&, const std::vector<double>&)> similarityMetric) const override;

    std::vector<double> classifyWithConfidence(
        const std::vector<double>& testPattern,
        const std::vector<LabeledPattern>& trainingPatterns,
        std::function<double(const std::vector<double>&, const std::vector<double>&)> similarityMetric) const override;

    std::string getName() const override { return "HierarchicalKNN"; }

private:
    enum class VoteMode {
        Majority,
        WeightedSimilarity,
        WeightedDistance
    };

    std::vector<int> labelToGroup_;
    int groupCount_;
    VoteMode coarseMode_;
    VoteMode fineMode_;
    int coarseK_;
    int fineK_;
    bool fallbackToFlat_;

    VoteMode parseVoteMode(const std::string& value, VoteMode defaultMode) const;
    void buildLabelGroups(const std::string& definitions);

    double voteWeight(double similarity, VoteMode mode) const;
    std::vector<double> scoreBuckets(
        const std::vector<std::pair<int, double>>& neighbors,
        const std::vector<LabeledPattern>& patterns,
        const std::vector<int>& bucketMap,
        int numBuckets,
        VoteMode mode) const;
    int selectBucket(
        const std::vector<std::pair<int, double>>& neighbors,
        const std::vector<LabeledPattern>& patterns,
        const std::vector<int>& bucketMap,
        int numBuckets,
        VoteMode mode) const;
    int classifyFlat(
        const std::vector<double>& testPattern,
        const std::vector<LabeledPattern>& trainingPatterns,
        std::function<double(const std::vector<double>&, const std::vector<double>&)> similarityMetric) const;
};

} // namespace classification
} // namespace snnfw

#endif // SNNFW_HIERARCHICAL_KNN_H

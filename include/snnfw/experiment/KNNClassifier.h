#ifndef SNNFW_EXPERIMENT_KNN_CLASSIFIER_H
#define SNNFW_EXPERIMENT_KNN_CLASSIFIER_H

#include "snnfw/experiment/ExperimentConfig.h"
#include "snnfw/classification/ClassificationStrategy.h"
#include <vector>
#include <cstdint>
#include <utility>
#include <limits>
#include <memory>

namespace snnfw {
namespace experiment {

using L5CountVector = std::vector<uint16_t>;
using L5LatencyVector = std::vector<uint16_t>;

/**
 * @brief k-NN + centroid classifier using L5 activation vectors.
 *
 * Stores per-class L5 spike-count patterns during training.
 * At test time, classifies using k-NN voting with cosine similarity,
 * falling back to centroid matching.
 */
class KNNClassifier {
public:
    explicit KNNClassifier(const ExperimentConfig& config);

    /// Reset all stored patterns and centroids
    void reset();

    /// Reset patterns for a new pass (optionally keep history)
    void resetForPass(bool keepHistory);

    /// Store an L5 pattern for a given class label
    void storePattern(int classLabel, const L5CountVector& counts,
                      const L5LatencyVector& latencies = {});

    /// Classify using k-NN voting. Returns (predictedLabel, similarity).
    std::pair<int, double> classifyKNN(const L5CountVector& testCounts,
                                       const L5LatencyVector& testLatencies = {}) const;

    /// Classify using centroid matching. Returns (predictedLabel, similarity).
    std::pair<int, double> classifyCentroid(const L5CountVector& testCounts,
                                            const L5LatencyVector& testLatencies = {}) const;

    /// Compute cosine similarity between two count vectors
    static double cosineSimilarity(const L5CountVector& a, const L5CountVector& b);

    /// Compute centroid similarity for a specific class
    double centroidSimilarity(const L5CountVector& testCounts, int classLabel,
                              const L5LatencyVector& testLatencies = {}) const;

    /// Check if any class is missing patterns
    bool hasMissingClasses() const;

    /// Get number of stored patterns for a class
    size_t getPatternCount(int classLabel) const;

private:
    enum class VoteMode {
        Legacy,
        Majority,
        WeightedSimilarity,
        WeightedDistance,
        Hierarchical
    };

    struct StoredPattern {
        L5CountVector counts;
        L5LatencyVector latencies;
    };

    const ExperimentConfig& config_;
    int numClasses_;
    size_t totalL5Neurons_;

    // Per-class k-NN pattern store
    std::vector<std::vector<StoredPattern>> classPatterns_;

    // Per-class centroid accumulators
    std::vector<std::vector<int64_t>> classCentroids_;
    std::vector<std::vector<double>> classLatencySums_;
    std::vector<std::vector<uint32_t>> classLatencyObsCounts_;
    std::vector<int> classPatternCounts_;
    std::vector<uint32_t> neuronPatternPresence_;
    uint32_t totalPatternsSeen_ = 0;
    VoteMode voteMode_ = VoteMode::Legacy;
    std::unique_ptr<classification::ClassificationStrategy> hierarchicalStrategy_;
    mutable bool genericPatternCacheDirty_ = true;
    mutable std::vector<classification::ClassificationStrategy::LabeledPattern> genericPatternCache_;

    static constexpr uint16_t kNoLatency = std::numeric_limits<uint16_t>::max();
    VoteMode resolveVoteMode() const;
    double knnVoteWeight(double similarity) const;
    std::vector<double> encodePatternForGenericReadout(const L5CountVector& counts,
                                                       const L5LatencyVector& latencies) const;
    const std::vector<classification::ClassificationStrategy::LabeledPattern>&
    buildGenericPatternCache() const;
    double idfWeight(size_t idx) const;
    double weightedCosineSimilarity(const L5CountVector& a, const L5CountVector& b,
                                    const L5LatencyVector* latA = nullptr,
                                    const L5LatencyVector* latB = nullptr) const;
    double weightedCentroidSimilarity(const L5CountVector& testCounts, int classLabel,
                                      const L5LatencyVector* testLatencies = nullptr) const;
    double latencyToSignal(uint16_t latency) const;
    double temporalFeatureScale(uint16_t latency) const;
    double latencySimilarity(const L5LatencyVector& a, const L5LatencyVector& b) const;
    double latencyCentroidSimilarity(const L5LatencyVector& testLatencies, int classLabel) const;
};

} // namespace experiment
} // namespace snnfw

#endif // SNNFW_EXPERIMENT_KNN_CLASSIFIER_H

#include "snnfw/classification/HierarchicalKNN.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace snnfw {
namespace classification {

namespace {

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::vector<int> parseIntList(const std::string& text, char separator) {
    std::vector<int> values;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, separator)) {
        if (token.empty()) {
            continue;
        }
        values.push_back(std::stoi(token));
    }
    return values;
}

} // namespace

HierarchicalKNN::HierarchicalKNN(const Config& config)
    : ClassificationStrategy(config)
    , labelToGroup_(static_cast<size_t>(std::max(0, config.numClasses)), -1)
    , groupCount_(0)
    , coarseMode_(parseVoteMode(config.getStringParam("coarse_strategy", "majority"),
                                VoteMode::Majority))
    , fineMode_(parseVoteMode(config.getStringParam("fine_strategy", "majority"),
                              VoteMode::Majority))
    , coarseK_(config.getIntParam("coarse_k", std::max(1, config.k)))
    , fineK_(config.getIntParam("fine_k", std::max(1, config.k)))
    , fallbackToFlat_(config.getIntParam("fallback_to_flat", 1) != 0) {
    buildLabelGroups(config.getStringParam("group_definitions", ""));
}

HierarchicalKNN::VoteMode HierarchicalKNN::parseVoteMode(const std::string& value,
                                                         VoteMode defaultMode) const {
    const std::string mode = toLower(value);
    if (mode.empty()) {
        return defaultMode;
    }
    if (mode == "majority" || mode == "majority_voting") {
        return VoteMode::Majority;
    }
    if (mode == "weighted_similarity") {
        return VoteMode::WeightedSimilarity;
    }
    if (mode == "weighted_distance") {
        return VoteMode::WeightedDistance;
    }
    throw std::invalid_argument("Unknown hierarchical vote mode: " + value);
}

void HierarchicalKNN::buildLabelGroups(const std::string& definitions) {
    int nextGroup = 0;

    if (!definitions.empty()) {
        std::stringstream groupStream(definitions);
        std::string groupDef;
        while (std::getline(groupStream, groupDef, ';')) {
            if (groupDef.empty()) {
                continue;
            }
            const auto labels = parseIntList(groupDef, ',');
            for (int label : labels) {
                if (label < 0 || label >= config_.numClasses) {
                    throw std::invalid_argument("Hierarchical group label out of range: " +
                                                std::to_string(label));
                }
                if (labelToGroup_[static_cast<size_t>(label)] != -1) {
                    throw std::invalid_argument("Hierarchical group label assigned twice: " +
                                                std::to_string(label));
                }
                labelToGroup_[static_cast<size_t>(label)] = nextGroup;
            }
            nextGroup++;
        }
    }

    for (int label = 0; label < config_.numClasses; ++label) {
        if (labelToGroup_[static_cast<size_t>(label)] == -1) {
            labelToGroup_[static_cast<size_t>(label)] = nextGroup++;
        }
    }

    groupCount_ = nextGroup;
}

double HierarchicalKNN::voteWeight(double similarity, VoteMode mode) const {
    const double clampedSimilarity = std::max(0.0, similarity);
    switch (mode) {
        case VoteMode::Majority:
            return 1.0;
        case VoteMode::WeightedSimilarity:
            return std::pow(clampedSimilarity, config_.distanceExponent);
        case VoteMode::WeightedDistance: {
            constexpr double kEpsilon = 1e-6;
            const double distance = std::max(0.0, 1.0 - clampedSimilarity);
            return 1.0 / (std::pow(distance, config_.distanceExponent) + kEpsilon);
        }
    }
    return 1.0;
}

std::vector<double> HierarchicalKNN::scoreBuckets(
    const std::vector<std::pair<int, double>>& neighbors,
    const std::vector<LabeledPattern>& patterns,
    const std::vector<int>& bucketMap,
    int numBuckets,
    VoteMode mode) const {
    std::vector<double> votes(static_cast<size_t>(numBuckets), 0.0);
    for (const auto& [idx, similarity] : neighbors) {
        const int label = patterns[static_cast<size_t>(idx)].label;
        const int bucket = bucketMap[static_cast<size_t>(label)];
        votes[static_cast<size_t>(bucket)] += voteWeight(similarity, mode);
    }
    return votes;
}

int HierarchicalKNN::selectBucket(
    const std::vector<std::pair<int, double>>& neighbors,
    const std::vector<LabeledPattern>& patterns,
    const std::vector<int>& bucketMap,
    int numBuckets,
    VoteMode mode) const {
    std::vector<double> votes(static_cast<size_t>(numBuckets), 0.0);
    std::vector<double> totalSimilarity(static_cast<size_t>(numBuckets), 0.0);
    std::vector<int> counts(static_cast<size_t>(numBuckets), 0);

    for (const auto& [idx, similarity] : neighbors) {
        const int label = patterns[static_cast<size_t>(idx)].label;
        const int bucket = bucketMap[static_cast<size_t>(label)];
        votes[static_cast<size_t>(bucket)] += voteWeight(similarity, mode);
        totalSimilarity[static_cast<size_t>(bucket)] += similarity;
        counts[static_cast<size_t>(bucket)]++;
    }

    double bestVote = -1.0;
    double bestAvgSimilarity = -1.0;
    int bestBucket = 0;
    for (int bucket = 0; bucket < numBuckets; ++bucket) {
        if (votes[static_cast<size_t>(bucket)] > bestVote) {
            bestVote = votes[static_cast<size_t>(bucket)];
            bestAvgSimilarity =
                counts[static_cast<size_t>(bucket)] > 0
                    ? totalSimilarity[static_cast<size_t>(bucket)] /
                          static_cast<double>(counts[static_cast<size_t>(bucket)])
                    : 0.0;
            bestBucket = bucket;
        } else if (votes[static_cast<size_t>(bucket)] == bestVote) {
            const double avgSimilarity =
                counts[static_cast<size_t>(bucket)] > 0
                    ? totalSimilarity[static_cast<size_t>(bucket)] /
                          static_cast<double>(counts[static_cast<size_t>(bucket)])
                    : 0.0;
            if (avgSimilarity > bestAvgSimilarity) {
                bestAvgSimilarity = avgSimilarity;
                bestBucket = bucket;
            }
        }
    }
    return bestBucket;
}

int HierarchicalKNN::classifyFlat(
    const std::vector<double>& testPattern,
    const std::vector<LabeledPattern>& trainingPatterns,
    std::function<double(const std::vector<double>&, const std::vector<double>&)> similarityMetric) const {
    const auto neighbors = findKNearestNeighbors(testPattern, trainingPatterns, similarityMetric, fineK_);
    std::vector<int> labelIdentity(static_cast<size_t>(config_.numClasses), 0);
    for (int label = 0; label < config_.numClasses; ++label) {
        labelIdentity[static_cast<size_t>(label)] = label;
    }
    return selectBucket(neighbors, trainingPatterns, labelIdentity, config_.numClasses, fineMode_);
}

int HierarchicalKNN::classify(
    const std::vector<double>& testPattern,
    const std::vector<LabeledPattern>& trainingPatterns,
    std::function<double(const std::vector<double>&, const std::vector<double>&)> similarityMetric) const {
    if (trainingPatterns.empty() || groupCount_ <= 0) {
        return 0;
    }

    const auto coarseNeighbors =
        findKNearestNeighbors(testPattern, trainingPatterns, similarityMetric, coarseK_);
    const int coarseGroup =
        selectBucket(coarseNeighbors, trainingPatterns, labelToGroup_, groupCount_, coarseMode_);

    std::vector<LabeledPattern> finePatterns;
    finePatterns.reserve(trainingPatterns.size());
    for (const auto& pattern : trainingPatterns) {
        if (labelToGroup_[static_cast<size_t>(pattern.label)] == coarseGroup) {
            finePatterns.push_back(pattern);
        }
    }

    if (finePatterns.empty()) {
        return fallbackToFlat_ ? classifyFlat(testPattern, trainingPatterns, similarityMetric) : 0;
    }

    std::vector<int> labelIdentity(static_cast<size_t>(config_.numClasses), 0);
    for (int label = 0; label < config_.numClasses; ++label) {
        labelIdentity[static_cast<size_t>(label)] = label;
    }
    const auto fineNeighbors =
        findKNearestNeighbors(testPattern, finePatterns, similarityMetric, fineK_);
    return selectBucket(fineNeighbors, finePatterns, labelIdentity, config_.numClasses, fineMode_);
}

std::vector<double> HierarchicalKNN::classifyWithConfidence(
    const std::vector<double>& testPattern,
    const std::vector<LabeledPattern>& trainingPatterns,
    std::function<double(const std::vector<double>&, const std::vector<double>&)> similarityMetric) const {
    if (trainingPatterns.empty() || groupCount_ <= 0) {
        return std::vector<double>(static_cast<size_t>(config_.numClasses), 0.0);
    }

    const auto coarseNeighbors =
        findKNearestNeighbors(testPattern, trainingPatterns, similarityMetric, coarseK_);
    const auto coarseVotes =
        scoreBuckets(coarseNeighbors, trainingPatterns, labelToGroup_, groupCount_, coarseMode_);
    const auto coarseConfidence = normalizeVotes(coarseVotes);
    const int coarseGroup =
        selectBucket(coarseNeighbors, trainingPatterns, labelToGroup_, groupCount_, coarseMode_);

    std::vector<LabeledPattern> finePatterns;
    finePatterns.reserve(trainingPatterns.size());
    for (const auto& pattern : trainingPatterns) {
        if (labelToGroup_[static_cast<size_t>(pattern.label)] == coarseGroup) {
            finePatterns.push_back(pattern);
        }
    }

    if (finePatterns.empty()) {
        std::vector<double> fallback(static_cast<size_t>(config_.numClasses), 0.0);
        if (fallbackToFlat_) {
            fallback[static_cast<size_t>(classifyFlat(testPattern, trainingPatterns, similarityMetric))] =
                1.0;
        }
        return fallback;
    }

    std::vector<int> labelIdentity(static_cast<size_t>(config_.numClasses), 0);
    for (int label = 0; label < config_.numClasses; ++label) {
        labelIdentity[static_cast<size_t>(label)] = label;
    }
    const auto fineNeighbors =
        findKNearestNeighbors(testPattern, finePatterns, similarityMetric, fineK_);
    const auto fineVotes =
        scoreBuckets(fineNeighbors, finePatterns, labelIdentity, config_.numClasses, fineMode_);
    auto fineConfidence = normalizeVotes(fineVotes);

    const double groupWeight = coarseConfidence[static_cast<size_t>(coarseGroup)];
    for (int label = 0; label < config_.numClasses; ++label) {
        if (labelToGroup_[static_cast<size_t>(label)] == coarseGroup) {
            fineConfidence[static_cast<size_t>(label)] *= groupWeight;
        } else {
            fineConfidence[static_cast<size_t>(label)] = 0.0;
        }
    }
    return normalizeVotes(fineConfidence);
}

} // namespace classification
} // namespace snnfw

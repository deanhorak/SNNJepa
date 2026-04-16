/**
 * @file MajorityVoting.cpp
 * @brief Implementation of majority voting k-NN classification
 *
 * This file implements the baseline k-NN classification strategy where each
 * of the k nearest neighbors gets an equal vote.
 *
 * Algorithm Details:
 * 1. Compute similarity between test pattern and all training patterns
 * 2. Sort by similarity (descending) and select top k
 * 3. Count votes for each class (each neighbor votes once)
 * 4. Return class with most votes
 * 5. Break ties by choosing class with highest average similarity
 *
 * Performance on MNIST (8×8 grid, Sobel, Rate):
 * - k=5: 94.63% accuracy (current baseline)
 *
 * This serves as the baseline for comparison with weighted strategies.
 */

#include "snnfw/classification/MajorityVoting.h"
#include <algorithm>
#include <numeric>

namespace snnfw {
namespace classification {

MajorityVoting::MajorityVoting(const Config& config)
    : ClassificationStrategy(config) {}

int MajorityVoting::classify(
    const std::vector<double>& testPattern,
    const std::vector<LabeledPattern>& trainingPatterns,
    std::function<double(const std::vector<double>&, const std::vector<double>&)> similarityMetric) const {
    
    // Find k nearest neighbors
    auto neighbors = findKNearestNeighbors(testPattern, trainingPatterns, similarityMetric, config_.k);
    
    // Count votes for each class
    std::vector<int> votes(config_.numClasses, 0);
    std::vector<double> totalSimilarity(config_.numClasses, 0.0);
    std::vector<int> classCount(config_.numClasses, 0);
    
    for (const auto& [idx, similarity] : neighbors) {
        int label = trainingPatterns[idx].label;
        votes[label]++;
        totalSimilarity[label] += similarity;
        classCount[label]++;
    }
    
    // Find class with most votes
    int maxVotes = *std::max_element(votes.begin(), votes.end());
    
    // If there's a tie, choose class with highest average similarity
    int bestClass = 0;
    double bestAvgSimilarity = -1.0;
    
    for (int c = 0; c < config_.numClasses; ++c) {
        if (votes[c] == maxVotes) {
            double avgSimilarity = (classCount[c] > 0) ? 
                (totalSimilarity[c] / classCount[c]) : 0.0;
            
            if (avgSimilarity > bestAvgSimilarity) {
                bestAvgSimilarity = avgSimilarity;
                bestClass = c;
            }
        }
    }
    
    return bestClass;
}

std::vector<double> MajorityVoting::classifyWithConfidence(
    const std::vector<double>& testPattern,
    const std::vector<LabeledPattern>& trainingPatterns,
    std::function<double(const std::vector<double>&, const std::vector<double>&)> similarityMetric) const {
    
    // Find k nearest neighbors
    auto neighbors = findKNearestNeighbors(testPattern, trainingPatterns, similarityMetric, config_.k);
    
    // Count votes for each class
    std::vector<double> confidence(config_.numClasses, 0.0);
    
    for (const auto& [idx, similarity] : neighbors) {
        int label = trainingPatterns[idx].label;
        confidence[label] += 1.0;  // Each neighbor gets one vote
    }
    
    // Normalize to get confidence scores (sum to 1.0)
    double totalVotes = static_cast<double>(neighbors.size());
    if (totalVotes > 0.0) {
        for (double& conf : confidence) {
            conf /= totalVotes;
        }
    }
    
    return confidence;
}

} // namespace classification
} // namespace snnfw


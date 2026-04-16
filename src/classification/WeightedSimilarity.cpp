/**
 * @file WeightedSimilarity.cpp
 * @brief Implementation of similarity-weighted k-NN classification
 *
 * This file implements a k-NN classification strategy where each neighbor's
 * vote is weighted directly by its similarity to the test pattern.
 *
 * Algorithm Details:
 * 1. Compute similarity between test pattern and all training patterns
 * 2. Sort by similarity (descending) and select top k
 * 3. For each neighbor:
 *    - Compute weight: w = similarity^p
 *    - Add weighted vote to neighbor's class
 * 4. Return class with highest weighted vote sum
 *
 * Weighting Function:
 * - weight = similarity^p
 * - Default p = 1.0 (linear weighting, proportional to similarity)
 * - p > 1: More emphasis on highly similar neighbors
 * - p < 1: Less emphasis on similarity differences
 *
 * Performance Expectations:
 * - Should outperform majority voting
 * - Expected improvement: +0.5-1.5% on MNIST
 * - More stable than WeightedDistance (no division)
 * - Simpler and more interpretable than WeightedDistance
 *
 * Comparison with WeightedDistance:
 * - WeightedSimilarity is more conservative (smaller weight differences)
 * - WeightedDistance can give extreme weights for very close neighbors
 * - WeightedSimilarity is numerically more stable
 * - Both should improve over majority voting, but may excel in different scenarios
 */

#include "snnfw/classification/WeightedSimilarity.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace snnfw {
namespace classification {

WeightedSimilarity::WeightedSimilarity(const Config& config)
    : ClassificationStrategy(config) {}

double WeightedSimilarity::computeWeight(double similarity) const {
    // Weight = similarity^p
    // Higher similarity → higher weight (direct relationship)
    // Note: We use distanceExponent from config as similarity exponent
    double weight = std::pow(similarity, config_.distanceExponent);
    return weight;
}

int WeightedSimilarity::classify(
    const std::vector<double>& testPattern,
    const std::vector<LabeledPattern>& trainingPatterns,
    std::function<double(const std::vector<double>&, const std::vector<double>&)> similarityMetric) const {
    
    // Find k nearest neighbors
    auto neighbors = findKNearestNeighbors(testPattern, trainingPatterns, similarityMetric, config_.k);
    
    // Compute weighted votes for each class
    std::vector<double> weightedVotes(config_.numClasses, 0.0);
    
    for (const auto& [idx, similarity] : neighbors) {
        int label = trainingPatterns[idx].label;
        double weight = computeWeight(similarity);
        weightedVotes[label] += weight;
    }
    
    // Return class with highest weighted vote
    return std::max_element(weightedVotes.begin(), weightedVotes.end()) - weightedVotes.begin();
}

std::vector<double> WeightedSimilarity::classifyWithConfidence(
    const std::vector<double>& testPattern,
    const std::vector<LabeledPattern>& trainingPatterns,
    std::function<double(const std::vector<double>&, const std::vector<double>&)> similarityMetric) const {
    
    // Find k nearest neighbors
    auto neighbors = findKNearestNeighbors(testPattern, trainingPatterns, similarityMetric, config_.k);
    
    // Compute weighted votes for each class
    std::vector<double> confidence(config_.numClasses, 0.0);
    
    for (const auto& [idx, similarity] : neighbors) {
        int label = trainingPatterns[idx].label;
        double weight = computeWeight(similarity);
        confidence[label] += weight;
    }
    
    // Normalize to get confidence scores (sum to 1.0)
    double totalWeight = std::accumulate(confidence.begin(), confidence.end(), 0.0);
    if (totalWeight > 0.0) {
        for (double& conf : confidence) {
            conf /= totalWeight;
        }
    }
    
    return confidence;
}

} // namespace classification
} // namespace snnfw


/**
 * @file WeightedDistance.cpp
 * @brief Implementation of distance-weighted k-NN classification
 *
 * This file implements a k-NN classification strategy where each neighbor's
 * vote is weighted by its distance (inverse similarity) to the test pattern.
 *
 * Algorithm Details:
 * 1. Compute similarity between test pattern and all training patterns
 * 2. Sort by similarity (descending) and select top k
 * 3. For each neighbor:
 *    - Convert similarity to distance: d = 1 - similarity
 *    - Compute weight: w = 1 / (d^p + epsilon)
 *    - Add weighted vote to neighbor's class
 * 4. Return class with highest weighted vote sum
 *
 * Weighting Function:
 * - weight = 1 / (distance^p + epsilon)
 * - Default p = 2.0 (inverse square weighting)
 * - epsilon = 1e-6 (prevents division by zero)
 *
 * Performance Expectations:
 * - Should outperform majority voting when neighbor similarities vary
 * - Expected improvement: +0.5-1.5% on MNIST
 * - Particularly effective when closest neighbors are highly similar
 *
 * Example:
 * - Neighbor 1: similarity = 0.95, distance = 0.05, weight = 1/(0.05^2 + 1e-6) = 400
 * - Neighbor 2: similarity = 0.70, distance = 0.30, weight = 1/(0.30^2 + 1e-6) = 11.1
 * - Neighbor 1 has ~36x more influence than Neighbor 2
 */

#include "snnfw/classification/WeightedDistance.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace snnfw {
namespace classification {

WeightedDistance::WeightedDistance(const Config& config)
    : ClassificationStrategy(config) {}

double WeightedDistance::computeWeight(double similarity) const {
    // Convert similarity [0,1] to distance [0,1]
    double distance = 1.0 - similarity;
    
    // Compute weight = 1 / (distance^p + epsilon)
    // Higher similarity (lower distance) → higher weight
    double distancePower = std::pow(distance, config_.distanceExponent);
    double weight = 1.0 / (distancePower + EPSILON);
    
    return weight;
}

int WeightedDistance::classify(
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

std::vector<double> WeightedDistance::classifyWithConfidence(
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


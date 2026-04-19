/**
 * @file ClassificationStrategyFactory.cpp
 * @brief Factory for creating classification strategies
 *
 * This file implements the factory pattern for creating classification strategies
 * from configuration. Supported strategies:
 * - "majority" or "majority_voting": Simple k-NN with equal votes
 * - "weighted_distance": Distance-weighted k-NN (closer neighbors have more influence)
 * - "weighted_similarity": Similarity-weighted k-NN (more similar neighbors have more influence)
 *
 * Usage:
 * @code
 * ClassificationStrategy::Config config;
 * config.k = 5;
 * config.numClasses = 10;
 * config.distanceExponent = 2.0;
 * 
 * auto strategy = ClassificationStrategyFactory::create("weighted_distance", config);
 * int label = strategy->classify(testPattern, trainingPatterns, cosineSimilarity);
 * @endcode
 */

#include "snnfw/classification/ClassificationStrategy.h"
#include "snnfw/classification/MajorityVoting.h"
#include "snnfw/classification/WeightedDistance.h"
#include "snnfw/classification/WeightedSimilarity.h"
#include "snnfw/classification/HierarchicalKNN.h"
#include <stdexcept>
#include <algorithm>

namespace snnfw {
namespace classification {

std::unique_ptr<ClassificationStrategy> ClassificationStrategyFactory::create(
    const std::string& type,
    const ClassificationStrategy::Config& config) {
    
    // Convert to lowercase for case-insensitive comparison
    std::string lowerType = type;
    std::transform(lowerType.begin(), lowerType.end(), lowerType.begin(), ::tolower);
    
    if (lowerType == "majority" || lowerType == "majority_voting") {
        return std::make_unique<MajorityVoting>(config);
    }
    else if (lowerType == "weighted_distance") {
        return std::make_unique<WeightedDistance>(config);
    }
    else if (lowerType == "weighted_similarity") {
        return std::make_unique<WeightedSimilarity>(config);
    }
    else if (lowerType == "hierarchical" || lowerType == "hierarchical_knn") {
        return std::make_unique<HierarchicalKNN>(config);
    }
    else {
        throw std::invalid_argument("Unknown classification strategy type: " + type);
    }
}

std::vector<std::string> ClassificationStrategyFactory::getAvailableStrategies() {
    return {
        "majority",
        "majority_voting",
        "hierarchical",
        "hierarchical_knn",
        "weighted_distance",
        "weighted_similarity"
    };
}

} // namespace classification
} // namespace snnfw

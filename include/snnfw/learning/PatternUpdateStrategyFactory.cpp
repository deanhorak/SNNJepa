#include "snnfw/learning/PatternUpdateStrategy.h"
#include "snnfw/learning/AppendStrategy.h"
#include "snnfw/learning/ReplaceWorstStrategy.h"
#include "snnfw/learning/MergeSimilarStrategy.h"
#include "snnfw/learning/HybridStrategy.h"
#include "snnfw/Logger.h"
#include <algorithm>
#include <stdexcept>

namespace snnfw {
namespace learning {

std::unique_ptr<PatternUpdateStrategy> PatternUpdateStrategyFactory::create(
    const std::string& type,
    const PatternUpdateStrategy::Config& config) {
    
    // Convert to lowercase for case-insensitive comparison
    std::string lowerType = type;
    std::transform(lowerType.begin(), lowerType.end(), lowerType.begin(), ::tolower);

    if (lowerType == "append") {
        SNNFW_INFO("Creating AppendStrategy (baseline)");
        return std::make_unique<AppendStrategy>(config);
    }
    else if (lowerType == "replace_worst" || lowerType == "replaceworst") {
        SNNFW_INFO("Creating ReplaceWorstStrategy (synaptic pruning)");
        return std::make_unique<ReplaceWorstStrategy>(config);
    }
    else if (lowerType == "merge_similar" || lowerType == "mergesimilar") {
        SNNFW_INFO("Creating MergeSimilarStrategy (synaptic consolidation)");
        return std::make_unique<MergeSimilarStrategy>(config);
    }
    else if (lowerType == "hybrid") {
        SNNFW_INFO("Creating HybridStrategy (pruning + consolidation)");
        return std::make_unique<HybridStrategy>(config);
    }
    else {
        SNNFW_ERROR("Unknown pattern update strategy: {}", type);
        throw std::invalid_argument("Unknown pattern update strategy: " + type);
    }
}

std::vector<std::string> PatternUpdateStrategyFactory::getAvailableStrategies() {
    return {
        "append",          // Baseline: simple append with blending
        "replace_worst",   // Synaptic pruning: replace least-used patterns
        "merge_similar",   // Synaptic consolidation: merge similar patterns
        "hybrid"           // Hybrid: pruning + consolidation
    };
}

} // namespace learning
} // namespace snnfw


#include "snnfw/encoding/EncodingStrategy.h"
#include "snnfw/encoding/RateEncoder.h"
#include "snnfw/encoding/TemporalEncoder.h"
#include "snnfw/encoding/PopulationEncoder.h"
#include <stdexcept>
#include <algorithm>

namespace snnfw {
namespace encoding {

std::unique_ptr<EncodingStrategy> EncodingStrategyFactory::create(
    const std::string& type,
    const EncodingStrategy::Config& config) {
    
    // Convert type to lowercase for case-insensitive comparison
    std::string lowerType = type;
    std::transform(lowerType.begin(), lowerType.end(), lowerType.begin(), ::tolower);
    
    if (lowerType == "rate" || lowerType == "rate_coding") {
        return std::make_unique<RateEncoder>(config);
    } else if (lowerType == "temporal" || lowerType == "temporal_coding") {
        return std::make_unique<TemporalEncoder>(config);
    } else if (lowerType == "population" || lowerType == "population_coding") {
        return std::make_unique<PopulationEncoder>(config);
    } else {
        throw std::invalid_argument("Unknown encoding strategy type: " + type);
    }
}

std::vector<std::string> EncodingStrategyFactory::getAvailableStrategies() {
    return {
        "rate",
        "rate_coding",
        "temporal",
        "temporal_coding",
        "population",
        "population_coding"
    };
}

} // namespace encoding
} // namespace snnfw


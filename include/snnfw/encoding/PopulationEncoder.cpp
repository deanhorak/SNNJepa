#include "snnfw/encoding/PopulationEncoder.h"
#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace snnfw {
namespace encoding {

PopulationEncoder::PopulationEncoder(const Config& config)
    : EncodingStrategy(config) {
    
    // Parse additional configuration
    populationSize_ = config.getIntParam("population_size", 5);
    tuningWidth_ = config.getDoubleParam("tuning_width", 0.3);
    minResponse_ = config.getDoubleParam("min_response", 0.1);
    
    // Ensure population size is at least 1
    if (populationSize_ < 1) {
        populationSize_ = 1;
    }
}

double PopulationEncoder::computeResponse(double featureValue, double preferredValue) const {
    // Gaussian tuning curve
    // response = exp(-(featureValue - preferredValue)^2 / (2 * sigma^2))
    
    double diff = featureValue - preferredValue;
    double exponent = -(diff * diff) / (2.0 * tuningWidth_ * tuningWidth_);
    double response = std::exp(exponent);
    
    return response;
}

double PopulationEncoder::responseToSpikeTime(double response) const {
    // Convert response strength to spike time
    // Stronger response → earlier spike
    // spikeTime = baselineTime + (1 - response) * intensityScale
    
    return config_.baselineTime + (1.0 - response) * config_.intensityScale;
}

std::vector<double> PopulationEncoder::encode(double featureIntensity, int featureIndex) const {
    if (featureIntensity < 0.0) {
        return {}; // No spikes for negative intensity
    }
    
    // Clamp intensity to [0, 1]
    featureIntensity = std::max(0.0, std::min(1.0, featureIntensity));
    
    std::vector<double> spikeTimes;
    spikeTimes.reserve(populationSize_);
    
    // Generate spike for each neuron in the population
    for (int i = 0; i < populationSize_; ++i) {
        // Compute preferred value for this neuron
        // Evenly distribute preferred values across [0, 1]
        double preferredValue = static_cast<double>(i) / std::max(1.0, static_cast<double>(populationSize_ - 1));
        
        // Compute response based on Gaussian tuning curve
        double response = computeResponse(featureIntensity, preferredValue);
        
        // Only generate spike if response is above threshold
        if (response >= minResponse_) {
            double spikeTime = responseToSpikeTime(response);
            
            // Ensure spike is within temporal window
            if (spikeTime >= 0.0 && spikeTime <= config_.temporalWindow) {
                spikeTimes.push_back(spikeTime);
            } else {
                // Still need to maintain population size consistency
                // Add a placeholder (will be filtered later if needed)
                spikeTimes.push_back(-1.0);
            }
        } else {
            // No spike for this neuron (response too weak)
            // Add placeholder to maintain indexing
            spikeTimes.push_back(-1.0);
        }
    }
    
    return spikeTimes;
}

int PopulationEncoder::getNeuronsPerFeature() const {
    return populationSize_;
}

std::string PopulationEncoder::getName() const {
    return "PopulationEncoder";
}

} // namespace encoding
} // namespace snnfw


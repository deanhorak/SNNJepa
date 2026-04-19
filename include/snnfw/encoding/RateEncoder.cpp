#include "snnfw/encoding/RateEncoder.h"
#include <algorithm>

namespace snnfw {
namespace encoding {

RateEncoder::RateEncoder(const Config& config)
    : EncodingStrategy(config) {}

std::vector<double> RateEncoder::encode(double featureIntensity, int featureIndex) const {
    // Rate coding: stronger features generate earlier spikes
    // This is the current baseline implementation
    
    if (featureIntensity <= 0.0) {
        return {}; // No spike for zero intensity
    }
    
    // Clamp intensity to [0, 1]
    featureIntensity = std::max(0.0, std::min(1.0, featureIntensity));
    
    // Compute spike time: stronger features spike earlier
    // spikeTime = baselineTime + (1 - intensity) * intensityScale
    double spikeTime = config_.baselineTime + 
                       (1.0 - featureIntensity) * config_.intensityScale;
    
    // Ensure spike is within temporal window
    if (spikeTime < 0.0 || spikeTime > config_.temporalWindow) {
        return {}; // Spike outside window
    }
    
    return {spikeTime};
}

int RateEncoder::getNeuronsPerFeature() const {
    return 1; // Rate coding uses one neuron per feature
}

std::string RateEncoder::getName() const {
    return "RateEncoder";
}

} // namespace encoding
} // namespace snnfw


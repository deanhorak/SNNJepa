#include "snnfw/encoding/TemporalEncoder.h"
#include <algorithm>
#include <cmath>
#include <random>

namespace snnfw {
namespace encoding {

TemporalEncoder::TemporalEncoder(const Config& config)
    : EncodingStrategy(config) {
    
    // Parse additional configuration
    dualSpikeMode_ = (config.getIntParam("precision_mode", 0) == 1);
    timingJitter_ = config.getDoubleParam("timing_jitter", 0.0);
    minSpikeInterval_ = config.getDoubleParam("min_spike_interval", 5.0);
}

double TemporalEncoder::computeSpikeTime(double baseTime, int seed) const {
    if (timingJitter_ <= 0.0) {
        return baseTime;
    }
    
    // Add random jitter using seed for reproducibility
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, timingJitter_);
    double jitter = dist(rng);
    
    return baseTime + jitter;
}

std::vector<double> TemporalEncoder::encode(double featureIntensity, int featureIndex) const {
    if (featureIntensity <= 0.0) {
        return {}; // No spike for zero intensity
    }
    
    // Clamp intensity to [0, 1]
    featureIntensity = std::max(0.0, std::min(1.0, featureIntensity));
    
    std::vector<double> spikeTimes;
    
    // First spike: encodes intensity with high precision
    // Use a steeper mapping for better temporal resolution
    // spikeTime = baselineTime + (1 - intensity^2) * intensityScale
    // The squaring makes the mapping more sensitive to high intensities
    double firstSpikeBase = config_.baselineTime + 
                           (1.0 - featureIntensity * featureIntensity) * config_.intensityScale;
    
    double firstSpike = computeSpikeTime(firstSpikeBase, featureIndex * 2);
    
    // Ensure spike is within temporal window
    if (firstSpike >= 0.0 && firstSpike <= config_.temporalWindow) {
        spikeTimes.push_back(firstSpike);
    }
    
    // Optional second spike for enhanced precision
    if (dualSpikeMode_ && !spikeTimes.empty()) {
        // Second spike timing encodes additional information
        // Interval between spikes is proportional to intensity
        double interval = minSpikeInterval_ + 
                         (1.0 - featureIntensity) * (config_.intensityScale * 0.3);
        
        double secondSpikeBase = firstSpikeBase + interval;
        double secondSpike = computeSpikeTime(secondSpikeBase, featureIndex * 2 + 1);
        
        // Ensure second spike is within temporal window
        if (secondSpike >= 0.0 && secondSpike <= config_.temporalWindow) {
            spikeTimes.push_back(secondSpike);
        }
    }
    
    return spikeTimes;
}

int TemporalEncoder::getNeuronsPerFeature() const {
    return 1; // Temporal coding uses one neuron per feature
}

std::string TemporalEncoder::getName() const {
    return "TemporalEncoder";
}

} // namespace encoding
} // namespace snnfw


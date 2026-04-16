#ifndef SNNFW_RATE_ENCODER_H
#define SNNFW_RATE_ENCODER_H

#include "snnfw/encoding/EncodingStrategy.h"

namespace snnfw {
namespace encoding {

/**
 * @brief Rate coding encoder (baseline)
 *
 * Rate coding is the simplest spike encoding scheme where feature intensity
 * is mapped to spike timing: stronger features generate earlier spikes.
 *
 * Formula: spikeTime = baselineTime + (1 - intensity) * intensityScale
 *
 * Characteristics:
 * - Simple and fast
 * - One spike per feature
 * - One neuron per feature
 * - Information encoded in spike latency
 *
 * Biological Motivation:
 * Rate coding is observed in many sensory systems where stimulus intensity
 * correlates with firing rate or latency. However, it's relatively slow
 * compared to temporal coding.
 *
 * References:
 * - Adrian (1928) - The basis of sensation
 * - Thorpe et al. (2001) - Spike-based strategies for rapid processing
 */
class RateEncoder : public EncodingStrategy {
public:
    /**
     * @brief Constructor
     * @param config Encoding configuration
     */
    explicit RateEncoder(const Config& config);

    /**
     * @brief Encode feature intensity into spike time
     * @param featureIntensity Feature intensity (0.0 to 1.0)
     * @param featureIndex Feature index (unused for rate coding)
     * @return Vector with single spike time, or empty if no spike
     */
    std::vector<double> encode(double featureIntensity, int featureIndex = 0) const override;

    /**
     * @brief Get number of neurons per feature (always 1 for rate coding)
     */
    int getNeuronsPerFeature() const override;

    /**
     * @brief Get encoder name
     */
    std::string getName() const override;
};

} // namespace encoding
} // namespace snnfw

#endif // SNNFW_RATE_ENCODER_H


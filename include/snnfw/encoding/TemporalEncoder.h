#ifndef SNNFW_TEMPORAL_ENCODER_H
#define SNNFW_TEMPORAL_ENCODER_H

#include "snnfw/encoding/EncodingStrategy.h"

namespace snnfw {
namespace encoding {

/**
 * @brief Temporal coding encoder
 *
 * Temporal coding uses precise spike timing to encode information with
 * higher temporal resolution than rate coding. Multiple spikes can be
 * generated with precise timing patterns that encode feature intensity.
 *
 * This implementation uses a more sophisticated timing scheme:
 * - First spike latency encodes intensity (like rate coding)
 * - Optional second spike for enhanced precision
 * - Timing jitter can be added for robustness
 *
 * Characteristics:
 * - Fast encoding (first spike carries information)
 * - High temporal precision
 * - Can encode more information per spike
 * - One neuron per feature
 *
 * Biological Motivation:
 * Temporal coding is observed in many sensory systems, particularly in
 * auditory and visual processing. The precise timing of spikes relative
 * to stimulus onset or other spikes carries significant information.
 *
 * References:
 * - Thorpe et al. (2001) - Spike-based strategies for rapid processing
 * - VanRullen & Thorpe (2001) - Rate coding versus temporal order coding
 * - Guyonneau et al. (2005) - Neurons tune to the earliest spikes
 */
class TemporalEncoder : public EncodingStrategy {
public:
    /**
     * @brief Constructor
     * @param config Encoding configuration
     *
     * Additional config parameters:
     * - "precision_mode": "single" or "dual" (default: "single")
     * - "timing_jitter": Amount of random jitter in ms (default: 0.0)
     * - "min_spike_interval": Minimum interval between spikes in ms (default: 5.0)
     */
    explicit TemporalEncoder(const Config& config);

    /**
     * @brief Encode feature intensity into precise spike time(s)
     * @param featureIntensity Feature intensity (0.0 to 1.0)
     * @param featureIndex Feature index (used for jitter seed)
     * @return Vector with one or two spike times
     */
    std::vector<double> encode(double featureIntensity, int featureIndex = 0) const override;

    /**
     * @brief Get number of neurons per feature (always 1 for temporal coding)
     */
    int getNeuronsPerFeature() const override;

    /**
     * @brief Get encoder name
     */
    std::string getName() const override;

private:
    bool dualSpikeMode_;      ///< Whether to generate two spikes
    double timingJitter_;     ///< Amount of timing jitter (ms)
    double minSpikeInterval_; ///< Minimum interval between spikes (ms)

    /**
     * @brief Compute spike time with optional jitter
     */
    double computeSpikeTime(double baseTime, int seed) const;
};

} // namespace encoding
} // namespace snnfw

#endif // SNNFW_TEMPORAL_ENCODER_H


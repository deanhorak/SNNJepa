#ifndef SNNFW_POPULATION_ENCODER_H
#define SNNFW_POPULATION_ENCODER_H

#include "snnfw/encoding/EncodingStrategy.h"

namespace snnfw {
namespace encoding {

/**
 * @brief Population coding encoder
 *
 * Population coding uses multiple neurons with overlapping tuning curves
 * to encode a single feature. Each neuron has a "preferred value" and
 * responds most strongly when the feature is near that value.
 *
 * This provides:
 * - Robustness through redundancy
 * - Higher resolution through population averaging
 * - Noise tolerance
 * - Graceful degradation
 *
 * Implementation:
 * - N neurons per feature (configurable)
 * - Gaussian tuning curves centered at evenly spaced preferred values
 * - Response strength determines spike timing
 *
 * Characteristics:
 * - Multiple neurons per feature (typically 3-10)
 * - Overlapping receptive fields
 * - Robust to noise and neuron loss
 * - Higher memory requirements
 *
 * Biological Motivation:
 * Population coding is ubiquitous in the brain. For example, direction
 * selectivity in visual cortex is encoded by populations of neurons with
 * different preferred directions. The population response provides a
 * robust estimate of the stimulus.
 *
 * References:
 * - Georgopoulos et al. (1986) - Neuronal population coding of movement direction
 * - Pouget et al. (2000) - Information processing with population codes
 * - Averbeck et al. (2006) - Neural correlations, population coding
 */
class PopulationEncoder : public EncodingStrategy {
public:
    /**
     * @brief Constructor
     * @param config Encoding configuration
     *
     * Additional config parameters:
     * - "population_size": Number of neurons per feature (default: 5)
     * - "tuning_width": Width of Gaussian tuning curve (sigma) (default: 0.3)
     * - "min_response": Minimum response to generate spike (default: 0.1)
     */
    explicit PopulationEncoder(const Config& config);

    /**
     * @brief Encode feature intensity into population spike times
     * @param featureIntensity Feature intensity (0.0 to 1.0)
     * @param featureIndex Feature index (used for neuron indexing)
     * @return Vector of spike times (one per neuron in population)
     */
    std::vector<double> encode(double featureIntensity, int featureIndex = 0) const override;

    /**
     * @brief Get number of neurons per feature
     */
    int getNeuronsPerFeature() const override;

    /**
     * @brief Get encoder name
     */
    std::string getName() const override;

private:
    int populationSize_;   ///< Number of neurons per feature
    double tuningWidth_;   ///< Width of Gaussian tuning curve (sigma)
    double minResponse_;   ///< Minimum response to generate spike

    /**
     * @brief Compute Gaussian response for a neuron
     * @param featureValue Actual feature value (0.0 to 1.0)
     * @param preferredValue Neuron's preferred value (0.0 to 1.0)
     * @return Response strength (0.0 to 1.0)
     */
    double computeResponse(double featureValue, double preferredValue) const;

    /**
     * @brief Convert response strength to spike time
     * @param response Response strength (0.0 to 1.0)
     * @return Spike time in ms
     */
    double responseToSpikeTime(double response) const;
};

} // namespace encoding
} // namespace snnfw

#endif // SNNFW_POPULATION_ENCODER_H


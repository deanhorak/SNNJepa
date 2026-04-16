#ifndef SNNFW_ENCODING_STRATEGY_H
#define SNNFW_ENCODING_STRATEGY_H

#include <vector>
#include <string>
#include <memory>
#include <map>

namespace snnfw {
namespace encoding {

/**
 * @brief Base class for spike encoding strategies
 *
 * Encoding strategies convert feature intensities into spike timing patterns.
 * Different encoding schemes can represent information in different ways:
 * - Rate Coding: Feature intensity → spike timing (stronger = earlier)
 * - Temporal Coding: Precise spike timing encodes information
 * - Population Coding: Multiple neurons with overlapping tuning curves
 *
 * Biological Motivation:
 * Real neurons use multiple encoding schemes simultaneously. Rate coding is
 * simple but slow. Temporal coding is fast and precise. Population coding
 * provides robustness through redundancy.
 *
 * References:
 * - Gerstner & Kistler (2002) - Spiking Neuron Models
 * - Thorpe et al. (2001) - Spike-based strategies for rapid processing
 */
class EncodingStrategy {
public:
    /**
     * @brief Encoding configuration parameters
     */
    struct Config {
        std::string name;                           ///< Strategy name
        double temporalWindow;                      ///< Temporal window in milliseconds
        double baselineTime;                        ///< Baseline spike time offset (ms)
        double intensityScale;                      ///< Scaling factor for intensity → time mapping
        std::map<std::string, double> doubleParams; ///< Additional double parameters
        std::map<std::string, int> intParams;       ///< Additional integer parameters
        
        // Helper methods
        double getDoubleParam(const std::string& key, double defaultValue = 0.0) const {
            auto it = doubleParams.find(key);
            return (it != doubleParams.end()) ? it->second : defaultValue;
        }
        
        int getIntParam(const std::string& key, int defaultValue = 0) const {
            auto it = intParams.find(key);
            return (it != intParams.end()) ? it->second : defaultValue;
        }
    };

    /**
     * @brief Constructor
     * @param config Encoding configuration
     */
    explicit EncodingStrategy(const Config& config) : config_(config) {}

    /**
     * @brief Virtual destructor
     */
    virtual ~EncodingStrategy() = default;

    /**
     * @brief Encode a single feature intensity into spike time(s)
     * @param featureIntensity Feature intensity (normalized 0.0 to 1.0)
     * @param featureIndex Index of the feature (for population coding)
     * @return Vector of spike times in milliseconds (may contain multiple spikes)
     *
     * For rate coding: Returns single spike time
     * For temporal coding: Returns single precise spike time
     * For population coding: Returns multiple spike times (one per neuron in population)
     */
    virtual std::vector<double> encode(double featureIntensity, int featureIndex = 0) const = 0;

    /**
     * @brief Encode multiple features into spike times
     * @param features Vector of feature intensities (normalized 0.0 to 1.0)
     * @return Vector of spike times, one per feature (or more for population coding)
     */
    virtual std::vector<double> encodeFeatures(const std::vector<double>& features) const {
        std::vector<double> spikeTimes;
        for (size_t i = 0; i < features.size(); ++i) {
            auto times = encode(features[i], static_cast<int>(i));
            spikeTimes.insert(spikeTimes.end(), times.begin(), times.end());
        }
        return spikeTimes;
    }

    /**
     * @brief Get the number of neurons required per feature
     * @return Number of neurons (1 for rate/temporal coding, >1 for population coding)
     */
    virtual int getNeuronsPerFeature() const = 0;

    /**
     * @brief Get the strategy name
     */
    virtual std::string getName() const = 0;

    /**
     * @brief Get the configuration
     */
    const Config& getConfig() const { return config_; }

protected:
    Config config_; ///< Encoding configuration
};

/**
 * @brief Factory for creating encoding strategies
 */
class EncodingStrategyFactory {
public:
    /**
     * @brief Create an encoding strategy from configuration
     * @param type Strategy type ("rate", "temporal", "population")
     * @param config Strategy configuration
     * @return Unique pointer to encoding strategy
     */
    static std::unique_ptr<EncodingStrategy> create(
        const std::string& type,
        const EncodingStrategy::Config& config);

    /**
     * @brief Get list of available encoding strategies
     */
    static std::vector<std::string> getAvailableStrategies();
};

} // namespace encoding
} // namespace snnfw

#endif // SNNFW_ENCODING_STRATEGY_H


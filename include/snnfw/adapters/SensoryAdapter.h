#ifndef SNNFW_SENSORY_ADAPTER_H
#define SNNFW_SENSORY_ADAPTER_H

#include "snnfw/adapters/BaseAdapter.h"
#include "snnfw/Neuron.h"
#include <vector>
#include <memory>
#include <cstdint>

namespace snnfw {
namespace adapters {

/**
 * @brief Base class for sensory (input) adapters
 *
 * Sensory adapters convert external data into spike trains that can be processed
 * by the spiking neural network. They implement the encoding step of the
 * sense-process-act cycle.
 *
 * Key Responsibilities:
 * - Accept external data in various formats (images, audio, sensor readings, etc.)
 * - Extract relevant features from the data
 * - Encode features as temporal spike patterns
 * - Manage a population of sensory neurons
 * - Provide activation patterns for downstream processing
 *
 * Encoding Strategies:
 * - Rate Coding: Feature intensity → spike timing (stronger = earlier)
 * - Temporal Coding: Feature value → precise spike timing
 * - Population Coding: Feature value → population activity pattern
 * - Phase Coding: Feature value → spike phase relative to oscillation
 *
 * Example Implementations:
 * - RetinaAdapter: Image → edge-oriented spike patterns
 * - AudioAdapter: Sound → frequency-decomposed spike patterns
 * - ProprioceptiveAdapter: Joint angles → position-encoded spikes
 * - TactileAdapter: Touch sensors → pressure-encoded spikes
 */
class SensoryAdapter : public BaseAdapter {
public:
    /**
     * @brief Data sample structure for input data
     */
    struct DataSample {
        std::vector<uint8_t> rawData;     ///< Raw input data
        double timestamp;                  ///< Sample timestamp (ms)
        int rows = 0;                      ///< Optional image rows
        int cols = 0;                      ///< Optional image cols
        int channels = 1;                  ///< Optional image channels
        std::map<std::string, double> metadata; ///< Additional metadata
    };

    /**
     * @brief Feature vector extracted from input data
     */
    struct FeatureVector {
        std::vector<double> features;      ///< Feature values (0.0 to 1.0)
        std::vector<std::string> labels;   ///< Feature labels
        double timestamp;                   ///< Extraction timestamp (ms)
    };

    /**
     * @brief Spike pattern generated from features
     */
    struct SpikePattern {
        std::vector<std::vector<double>> spikeTimes; ///< Spike times per neuron
        double duration;                              ///< Pattern duration (ms)
        double timestamp;                             ///< Pattern timestamp (ms)
    };

    /**
     * @brief Constructor
     * @param config Adapter configuration
     */
    explicit SensoryAdapter(const Config& config) 
        : BaseAdapter(config) {}

    /**
     * @brief Virtual destructor
     */
    virtual ~SensoryAdapter() = default;

    /**
     * @brief Process input data and generate spike patterns
     * @param data Input data sample
     * @return Spike pattern generated from the data
     */
    virtual SpikePattern processData(const DataSample& data) = 0;

    /**
     * @brief Extract features from input data
     * @param data Input data sample
     * @return Feature vector extracted from the data
     */
    virtual FeatureVector extractFeatures(const DataSample& data) = 0;

    /**
     * @brief Encode features as spike patterns
     * @param features Feature vector to encode
     * @return Spike pattern encoding the features
     */
    virtual SpikePattern encodeFeatures(const FeatureVector& features) = 0;

    /**
     * @brief Get the sensory neuron population
     * @return Vector of neurons managed by this adapter
     */
    virtual const std::vector<std::shared_ptr<Neuron>>& getNeurons() const = 0;

    /**
     * @brief Get activation pattern from current neuron state
     * @return Vector of activation values (one per neuron)
     */
    virtual std::vector<double> getActivationPattern() const = 0;

    /**
     * @brief Get the number of sensory neurons
     */
    virtual size_t getNeuronCount() const = 0;

    /**
     * @brief Get the dimensionality of the feature space
     */
    virtual size_t getFeatureDimension() const = 0;

    /**
     * @brief Train/adapt the sensory processing
     * @param data Training data sample
     * @param label Optional label for supervised learning
     */
    virtual void train(const DataSample& data, int label = -1) {
        // Default: no training (stateless processing)
    }

    /**
     * @brief Clear all neuron states
     */
    virtual void clearNeuronStates() = 0;

    /**
     * @brief Reset adapter state
     */
    void reset() override {
        clearNeuronStates();
    }

    /**
     * @brief Get adapter statistics
     */
    std::map<std::string, double> getStatistics() const override {
        std::map<std::string, double> stats;
        stats["neuron_count"] = static_cast<double>(getNeuronCount());
        stats["feature_dimension"] = static_cast<double>(getFeatureDimension());
        return stats;
    }

protected:
    /**
     * @brief Helper: Convert feature value to spike time using rate coding
     * @param featureValue Feature value (0.0 to 1.0)
     * @param duration Total duration (ms)
     * @return Spike time (ms), or -1 if no spike
     */
    double featureToSpikeTime(double featureValue, double duration) const {
        if (featureValue <= 0.0) {
            return -1.0; // No spike
        }
        // Stronger features generate earlier spikes
        return duration * (1.0 - featureValue);
    }

    /**
     * @brief Helper: Convert feature vector to spike times
     * @param features Feature values (0.0 to 1.0)
     * @param duration Total duration (ms)
     * @return Vector of spike times (one per feature)
     */
    std::vector<double> featuresToSpikeTimes(const std::vector<double>& features, 
                                             double duration) const {
        std::vector<double> spikeTimes;
        spikeTimes.reserve(features.size());
        
        for (double feature : features) {
            double spikeTime = featureToSpikeTime(feature, duration);
            if (spikeTime >= 0.0) {
                spikeTimes.push_back(spikeTime);
            }
        }
        
        return spikeTimes;
    }
};

} // namespace adapters
} // namespace snnfw

#endif // SNNFW_SENSORY_ADAPTER_H

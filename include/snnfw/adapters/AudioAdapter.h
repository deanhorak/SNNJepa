#ifndef SNNFW_AUDIO_ADAPTER_H
#define SNNFW_AUDIO_ADAPTER_H

#include "snnfw/adapters/SensoryAdapter.h"
#include <vector>
#include <complex>

namespace snnfw {
namespace adapters {

/**
 * @brief Audio adapter for processing sound input
 *
 * The AudioAdapter mimics the auditory system's frequency decomposition:
 * - Frequency analysis (cochlear filtering)
 * - Temporal encoding (phase locking, rate coding)
 * - Population of frequency-selective neurons
 *
 * Architecture:
 * - Input: Audio samples (PCM format)
 * - Frequency Decomposition: FFT or filterbank
 * - Frequency Channels: Multiple frequency bands (e.g., 32 channels)
 * - Neurons: One or more neurons per frequency channel
 * - Output: Spike patterns encoding frequency content
 *
 * Configuration Parameters:
 * - sample_rate: Audio sample rate (Hz)
 * - num_channels: Number of frequency channels
 * - min_frequency: Minimum frequency (Hz)
 * - max_frequency: Maximum frequency (Hz)
 * - window_size: FFT window size (samples)
 * - hop_size: Hop size between windows (samples)
 * - encoding_mode: "rate" or "temporal"
 *
 * Usage:
 * @code
 * BaseAdapter::Config config;
 * config.type = "audio";
 * config.name = "left_ear";
 * config.setIntParam("sample_rate", 44100);
 * config.setIntParam("num_channels", 32);
 * 
 * AudioAdapter audio(config);
 * audio.initialize();
 * 
 * // Process audio samples
 * SensoryAdapter::DataSample sample;
 * sample.rawData = audioSamples;
 * auto spikes = audio.processData(sample);
 * @endcode
 */
class AudioAdapter : public SensoryAdapter {
public:
    /**
     * @brief Audio encoding mode
     */
    enum class EncodingMode {
        RATE,      ///< Rate coding (amplitude → spike rate)
        TEMPORAL,  ///< Temporal coding (phase → spike timing)
        HYBRID     ///< Combination of rate and temporal
    };

    /**
     * @brief Constructor
     * @param config Adapter configuration
     */
    explicit AudioAdapter(const Config& config);

    /**
     * @brief Destructor
     */
    ~AudioAdapter() override = default;

    /**
     * @brief Initialize the adapter
     */
    bool initialize() override;

    /**
     * @brief Process audio data and generate spike patterns
     * @param data Input audio samples (16-bit PCM)
     * @return Spike pattern from all neurons
     */
    SpikePattern processData(const DataSample& data) override;

    /**
     * @brief Extract frequency features from audio
     * @param data Input audio data
     * @return Feature vector containing frequency amplitudes
     */
    FeatureVector extractFeatures(const DataSample& data) override;

    /**
     * @brief Encode frequency features as spike patterns
     * @param features Frequency feature vector
     * @return Spike pattern encoding the features
     */
    SpikePattern encodeFeatures(const FeatureVector& features) override;

    /**
     * @brief Get the neuron population
     */
    const std::vector<std::shared_ptr<Neuron>>& getNeurons() const override {
        return neurons_;
    }

    /**
     * @brief Get activation pattern from neurons
     */
    std::vector<double> getActivationPattern() const override;

    /**
     * @brief Get neuron count
     */
    size_t getNeuronCount() const override {
        return neurons_.size();
    }

    /**
     * @brief Get feature dimension (number of frequency channels)
     */
    size_t getFeatureDimension() const override {
        return numChannels_;
    }

    /**
     * @brief Clear all neuron states
     */
    void clearNeuronStates() override;

private:
    // Configuration parameters
    int sampleRate_;            ///< Audio sample rate (Hz)
    int numChannels_;           ///< Number of frequency channels
    double minFrequency_;       ///< Minimum frequency (Hz)
    double maxFrequency_;       ///< Maximum frequency (Hz)
    int windowSize_;            ///< FFT window size (samples)
    int hopSize_;               ///< Hop size (samples)
    EncodingMode encodingMode_; ///< Spike encoding mode
    
    // Neuron parameters
    double neuronWindowSize_;   ///< Neuron temporal window (ms)
    double neuronThreshold_;    ///< Neuron similarity threshold
    int neuronMaxPatterns_;     ///< Max patterns per neuron
    
    // Neuron population (one per frequency channel)
    std::vector<std::shared_ptr<Neuron>> neurons_;
    
    // Frequency channel center frequencies
    std::vector<double> channelFrequencies_;
    
    /**
     * @brief Create neuron population
     */
    void createNeurons();
    
    /**
     * @brief Compute FFT of audio window
     * @param samples Audio samples
     * @return Frequency spectrum (magnitude)
     */
    std::vector<double> computeFFT(const std::vector<int16_t>& samples);
    
    /**
     * @brief Map FFT bins to frequency channels
     * @param spectrum FFT magnitude spectrum
     * @return Channel amplitudes
     */
    std::vector<double> mapToChannels(const std::vector<double>& spectrum);
    
    /**
     * @brief Calculate mel-scale frequency
     * @param frequency Frequency in Hz
     * @return Mel-scale value
     */
    double hzToMel(double frequency) const;
    
    /**
     * @brief Calculate frequency from mel-scale
     * @param mel Mel-scale value
     * @return Frequency in Hz
     */
    double melToHz(double mel) const;
};

} // namespace adapters
} // namespace snnfw

#endif // SNNFW_AUDIO_ADAPTER_H


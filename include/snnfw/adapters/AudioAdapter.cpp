#include "snnfw/adapters/AudioAdapter.h"
#include <algorithm>
#include <cmath>

namespace snnfw {
namespace adapters {

AudioAdapter::AudioAdapter(const BaseAdapter::Config& config)
    : SensoryAdapter(config)
    , sampleRate_(config.getIntParam("sample_rate", 44100))
    , numChannels_(config.getIntParam("num_channels", 32))
    , minFrequency_(config.getDoubleParam("min_frequency", 20.0))
    , maxFrequency_(config.getDoubleParam("max_frequency", 20000.0))
    , windowSize_(config.getIntParam("window_size", 1024))
    , hopSize_(config.getIntParam("hop_size", 512))
    , encodingMode_(EncodingMode::RATE)
    , neuronWindowSize_(config.getDoubleParam("neuron_window_size", 100.0))
    , neuronThreshold_(config.getDoubleParam("neuron_threshold", 0.7))
    , neuronMaxPatterns_(config.getIntParam("neuron_max_patterns", 20))
{
    std::string mode = config.getStringParam("encoding_mode", "rate");
    if (mode == "temporal") encodingMode_ = EncodingMode::TEMPORAL;
    else if (mode == "hybrid") encodingMode_ = EncodingMode::HYBRID;
}

bool AudioAdapter::initialize() {
    createNeurons();

    // Calculate channel frequencies (mel scale)
    channelFrequencies_.clear();
    channelFrequencies_.reserve(numChannels_);

    double minMel = hzToMel(minFrequency_);
    double maxMel = hzToMel(maxFrequency_);

    for (int i = 0; i < numChannels_; ++i) {
        double mel = minMel + (maxMel - minMel) * i / (numChannels_ - 1);
        channelFrequencies_.push_back(melToHz(mel));
    }

    return true;
}

SensoryAdapter::SpikePattern AudioAdapter::processData(const DataSample& sample) {
    auto features = extractFeatures(sample);
    return encodeFeatures(features);
}

SensoryAdapter::FeatureVector AudioAdapter::extractFeatures(const DataSample& sample) {
    FeatureVector features;
    features.timestamp = sample.timestamp;
    
    // Convert raw bytes to int16 samples
    std::vector<int16_t> audioSamples;
    for (size_t i = 0; i + 1 < sample.rawData.size(); i += 2) {
        int16_t s = static_cast<int16_t>((sample.rawData[i + 1] << 8) | sample.rawData[i]);
        audioSamples.push_back(s);
    }
    
    // Compute FFT
    auto spectrum = computeFFT(audioSamples);
    
    // Map to frequency channels
    features.features = mapToChannels(spectrum);
    
    // Add labels
    for (int i = 0; i < numChannels_; ++i) {
        features.labels.push_back("channel_" + std::to_string(i));
    }
    
    return features;
}

SensoryAdapter::SpikePattern AudioAdapter::encodeFeatures(const FeatureVector& features) {
    SpikePattern pattern;
    pattern.timestamp = features.timestamp;
    pattern.duration = config_.getDoubleParam("temporal_window", 100.0);

    // Convert each feature to spike times (one spike per feature/channel)
    pattern.spikeTimes.resize(features.features.size());
    for (size_t i = 0; i < features.features.size(); ++i) {
        double spikeTime = featureToSpikeTime(features.features[i], pattern.duration);
        if (spikeTime >= 0.0) {
            pattern.spikeTimes[i].push_back(spikeTime);
        }
    }

    // Insert spikes into neurons
    for (size_t i = 0; i < neurons_.size() && i < pattern.spikeTimes.size(); ++i) {
        for (double spikeTime : pattern.spikeTimes[i]) {
            neurons_[i]->insertSpike(spikeTime);
        }
    }

    return pattern;
}

std::vector<double> AudioAdapter::getActivationPattern() const {
    std::vector<double> activations(neurons_.size(), 0.0);
    
    for (size_t i = 0; i < neurons_.size(); ++i) {
        double similarity = neurons_[i]->getBestSimilarity();
        activations[i] = (similarity >= 0.0) ? similarity : 0.0;
    }
    
    return activations;
}

void AudioAdapter::clearNeuronStates() {
    for (auto& neuron : neurons_) {
        neuron->clearSpikes();
    }
}

void AudioAdapter::createNeurons() {
    neurons_.clear();
    neurons_.reserve(numChannels_);
    
    for (int i = 0; i < numChannels_; ++i) {
        auto neuron = std::make_shared<Neuron>(
            neuronWindowSize_,
            neuronThreshold_,
            neuronMaxPatterns_
        );
        neurons_.push_back(neuron);
    }
}

std::vector<double> AudioAdapter::computeFFT(const std::vector<int16_t>& samples) {
    // Simplified FFT - in production, use a library like FFTW or KissFFT
    int fftSize = std::min(windowSize_, static_cast<int>(samples.size()));
    std::vector<double> spectrum(fftSize / 2 + 1, 0.0);
    
    for (int k = 0; k < fftSize / 2 + 1; ++k) {
        double real = 0.0;
        double imag = 0.0;
        
        for (int n = 0; n < fftSize; ++n) {
            double angle = -2.0 * M_PI * k * n / fftSize;
            double sample = samples[n] / 32768.0; // Normalize
            real += sample * std::cos(angle);
            imag += sample * std::sin(angle);
        }
        
        spectrum[k] = std::sqrt(real * real + imag * imag);
    }
    
    return spectrum;
}

std::vector<double> AudioAdapter::mapToChannels(const std::vector<double>& spectrum) {
    std::vector<double> channelAmplitudes(numChannels_, 0.0);
    
    // Simple mapping: divide spectrum into channels
    int binsPerChannel = spectrum.size() / numChannels_;
    
    for (int i = 0; i < numChannels_; ++i) {
        int startBin = i * binsPerChannel;
        int endBin = (i + 1) * binsPerChannel;
        
        for (int j = startBin; j < endBin && j < static_cast<int>(spectrum.size()); ++j) {
            channelAmplitudes[i] += spectrum[j];
        }
        
        if (binsPerChannel > 0) {
            channelAmplitudes[i] /= binsPerChannel;
        }
    }
    
    return channelAmplitudes;
}

double AudioAdapter::hzToMel(double frequency) const {
    return 2595.0 * std::log10(1.0 + frequency / 700.0);
}

double AudioAdapter::melToHz(double mel) const {
    return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
}

} // namespace adapters
} // namespace snnfw


#ifndef SNNFW_EMNIST_ADAPTER_H
#define SNNFW_EMNIST_ADAPTER_H

#include "snnfw/adapters/SensoryAdapter.h"
#include <vector>

namespace snnfw {
namespace adapters {

/**
 * @brief Sensory adapter that converts EMNIST image pixels into latency-coded spikes.
 *
 * This adapter preserves the existing experiment behavior:
 * - A pixel fires iff normalized_intensity > pixel_threshold
 * - Spike time is (1 - normalized_intensity) * input_latency_ms
 */
class EMNISTAdapter : public SensoryAdapter {
public:
    explicit EMNISTAdapter(const Config& config);
    ~EMNISTAdapter() override = default;

    bool initialize() override;

    SpikePattern processData(const DataSample& data) override;
    FeatureVector extractFeatures(const DataSample& data) override;
    SpikePattern encodeFeatures(const FeatureVector& features) override;

    const std::vector<std::shared_ptr<Neuron>>& getNeurons() const override {
        return neurons_;
    }

    std::vector<double> getActivationPattern() const override {
        return lastActivationPattern_;
    }

    size_t getNeuronCount() const override {
        return featureDimension_;
    }

    size_t getFeatureDimension() const override {
        return featureDimension_;
    }

    void clearNeuronStates() override;

private:
    std::vector<double> preprocessActivations(const std::vector<uint8_t>& pixels) const;
    double computeFeatureSpikeTime(double strength, int orientationBin) const;
    size_t mapPixelToOutputIndex(int partition, size_t pixelIndex) const;
    bool usePartitionedFeatureInputs() const;
    bool isLocalMaximum(const std::vector<double>& map, int row, int col) const;
    bool isHorizontalEdgeBin(int orientationBin) const;

    double pixelThreshold_ = 0.4;
    double inputLatencyMs_ = 15.0;
    int imageRows_ = 28;
    int imageCols_ = 28;
    int outputRows_ = 28;
    int outputCols_ = 28;
    int outputPartitions_ = 1;
    bool partitionsHorizontal_ = true;
    size_t featureDimension_ = 28 * 28;
    bool enableVisualFrontend_ = false;
    double visualFrontendGain_ = 0.45;
    double visualOnCenterWeight_ = 0.60;
    double visualEdgeWeight_ = 0.40;
    bool enableVisualFeatureChannels_ = false;
    double visualFeatureThreshold_ = 0.58;
    double visualOnPartitionThreshold_ = 0.18;
    double visualOffPartitionThreshold_ = 0.22;
    double visualEdgePartitionThreshold_ = 0.55;
    int visualOrientationBins_ = 4;

    std::vector<std::shared_ptr<Neuron>> neurons_;
    std::vector<double> lastActivationPattern_;
};

} // namespace adapters
} // namespace snnfw

#endif // SNNFW_EMNIST_ADAPTER_H

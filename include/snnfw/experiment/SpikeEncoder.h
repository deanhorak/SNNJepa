#ifndef SNNFW_EXPERIMENT_SPIKE_ENCODER_H
#define SNNFW_EXPERIMENT_SPIKE_ENCODER_H

#include "snnfw/Neuron.h"
#include "snnfw/NetworkPropagator.h"
#include "snnfw/SpikeProcessor.h"
#include "snnfw/EMNISTLoader.h"
#include "snnfw/adapters/SensoryAdapter.h"
#include "snnfw/experiment/ExperimentConfig.h"
#include "snnfw/declarative/NetworkConstructor.h"
#include <memory>
#include <vector>

namespace snnfw {
namespace experiment {

/**
 * @brief Encodes pixel images as spike times and injects them into the network.
 *
 * Handles:
 * - Pixel-to-spike latency encoding (brighter = earlier)
 * - Memory cleanup of all network neurons before each image
 * - Waiting for the spike processor schedule horizon
 * - Tracking which input neurons fired
 */
class SpikeEncoder {
public:
    SpikeEncoder(const ExperimentConfig& config,
                 std::shared_ptr<SpikeProcessor> spikeProcessor,
                 std::shared_ptr<NetworkPropagator> propagator,
                 std::shared_ptr<adapters::SensoryAdapter> configuredAdapter = nullptr);

    /**
     * @brief Encode an image and inject spikes into input neurons.
     * @param image The EMNIST image to encode
     * @param inputNeurons The input neuron layer
     * @param columns The cortical columns (for memory cleanup)
     * @param outputPopulations The output populations (for memory cleanup)
     * @return baseTime for this image presentation
     */
    double encodeAndInject(
        const EMNISTLoader::Image& image,
        std::vector<std::shared_ptr<Neuron>>& inputNeurons,
        std::vector<declarative::ConstructedNetwork::ColumnGroup>& columns,
        std::vector<std::vector<std::shared_ptr<Neuron>>>& outputPopulations);

    /// Get which input neurons fired in the last encoding
    const std::vector<bool>& getLastInputFired() const { return lastInputFired_; }

    /// Get the base time of the last encoding
    double getLastBaseTime() const { return lastBaseTime_; }

    /// Wait for simulation time to reach a target
    void waitForSimTime(double targetTimeMs, double timeoutMs);

    /// Get/set the last inference end time (for scheduling)
    double getLastEndTime() const { return lastEndTime_; }
    void setLastEndTime(double t) { lastEndTime_ = t; }

private:
    struct FixationWindow {
        EMNISTLoader::Image image;
        double offsetMs = 0.0;
        double durationMs = 0.0;
    };

    std::vector<FixationWindow> buildFixationWindows(const EMNISTLoader::Image& image) const;
    int injectInputPattern(const adapters::SensoryAdapter::SpikePattern& encodedPattern,
                           double baseTime,
                           double offsetMs,
                           double durationMs,
                           std::vector<std::shared_ptr<Neuron>>& inputNeurons);
    int injectRawInputWindow(const EMNISTLoader::Image& image,
                             double baseTime,
                             double offsetMs,
                             double durationMs,
                             std::vector<std::shared_ptr<Neuron>>& inputNeurons);
    int injectRetinaIntoL4(const adapters::SensoryAdapter::SpikePattern& encodedPattern,
                           double baseTime,
                           double offsetMs,
                           double durationMs,
                           std::vector<declarative::ConstructedNetwork::ColumnGroup>& columns);
    int injectRetinaIntoInput(const EMNISTLoader::Image& image,
                              const adapters::SensoryAdapter::SpikePattern& encodedPattern,
                              double baseTime,
                              double offsetMs,
                              double durationMs,
                              std::vector<std::shared_ptr<Neuron>>& inputNeurons);
    void populateMaskInputActivity(const EMNISTLoader::Image& image,
                                   const std::vector<std::shared_ptr<Neuron>>& inputNeurons);

    const ExperimentConfig& config_;
    std::shared_ptr<SpikeProcessor> spikeProcessor_;
    std::shared_ptr<NetworkPropagator> propagator_;
    std::shared_ptr<adapters::SensoryAdapter> configuredAdapter_;
    std::shared_ptr<adapters::SensoryAdapter> emnistAdapter_;
    bool useEmnistAdapter_ = true;

    double lastEndTime_ = 0.0;
    double lastBaseTime_ = 0.0;
    std::vector<bool> lastInputFired_;
};

} // namespace experiment
} // namespace snnfw

#endif // SNNFW_EXPERIMENT_SPIKE_ENCODER_H

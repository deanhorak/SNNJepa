#ifndef SNNFW_INTERNEURON_ADAPTERS_H
#define SNNFW_INTERNEURON_ADAPTERS_H

#include "snnfw/adapters/SensoryAdapter.h"
#include "snnfw/adapters/MotorAdapter.h"
#include <string>
#include <vector>

namespace snnfw {
namespace adapters {

/// Force-link hook so static registration for interneuron adapters is retained.
void ensureInterneuronAdaptersRegistered();

/**
 * @brief Sensory adapter that receives spike frames over TCP.
 *
 * Protocol (one frame per line):
 *   SNNFW_SPIKES v1 <window_start_ms> <window_end_ms> <index_csv_or_->\n
 */
class InterneuronReceiverAdapter : public SensoryAdapter {
public:
    explicit InterneuronReceiverAdapter(const Config& config);
    ~InterneuronReceiverAdapter() override;

    bool initialize() override;
    void shutdown() override;

    SpikePattern processData(const DataSample& data) override;
    FeatureVector extractFeatures(const DataSample& data) override;
    SpikePattern encodeFeatures(const FeatureVector& features) override;

    const std::vector<std::shared_ptr<Neuron>>& getNeurons() const override {
        return neurons_;
    }
    std::vector<double> getActivationPattern() const override { return lastActivation_; }
    size_t getNeuronCount() const override { return neurons_.size(); }
    size_t getFeatureDimension() const override { return neurons_.size(); }
    void clearNeuronStates() override;

private:
    bool ensureClient();
    bool readAvailableLines(std::vector<std::string>& lines);
    bool parseLine(const std::string& line,
                   std::vector<int>& activeIndices,
                   double& windowStartMs,
                   double& windowEndMs) const;
    void closeSockets();

    std::string bindHost_;
    int bindPort_ = 5000;
    int neuronCount_ = 256;
    int receiveTimeoutMs_ = 5;

    int listenFd_ = -1;
    int clientFd_ = -1;
    std::string rxBuffer_;

    std::vector<std::shared_ptr<Neuron>> neurons_;
    std::vector<double> lastActivation_;
};

/**
 * @brief Motor adapter that sends spike frames over TCP.
 *
 * Protocol (one frame per line):
 *   SNNFW_SPIKES v1 <window_start_ms> <window_end_ms> <index_csv_or_->\n
 */
class InterneuronTransmitterAdapter : public MotorAdapter {
public:
    explicit InterneuronTransmitterAdapter(const Config& config);
    ~InterneuronTransmitterAdapter() override;

    bool initialize() override;
    void shutdown() override;

    MotorCommand decodeActivity(const SpikeActivity& activity) override;
    bool executeCommand(const MotorCommand& command) override;
    bool processNeurons(const std::vector<std::shared_ptr<Neuron>>& neurons,
                        double currentTime) override;

    size_t getChannelCount() const override { return channelCount_; }
    MotorCommand getCurrentCommand() const override { return currentCommand_; }
    void setCommand(const MotorCommand& command) override { currentCommand_ = command; }

private:
    bool ensureConnected(double currentTimeMs);
    bool sendLine(const std::string& line);
    void closeSocket();

    std::string remoteHost_;
    int remotePort_ = 5000;
    int connectTimeoutMs_ = 1000;
    int sendTimeoutMs_ = 100;
    double updateIntervalMs_ = 10.0;
    double startupGraceMs_ = 0.0;
    bool transmitNeuronIds_ = false;

    int socketFd_ = -1;
    double lastUpdateTime_ = 0.0;
    double nextConnectAttemptTimeMs_ = 0.0;
    size_t channelCount_ = 0;
    int connectFailures_ = 0;
    std::string lastConnectError_;
    std::vector<int> pendingActiveIndices_;
    MotorCommand currentCommand_;
};

} // namespace adapters
} // namespace snnfw

#endif // SNNFW_INTERNEURON_ADAPTERS_H

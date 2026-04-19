#include "snnfw/adapters/InterneuronAdapters.h"
#include "snnfw/adapters/AdapterFactory.h"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace snnfw {
namespace adapters {

namespace {

bool setNonBlocking(int fd) {
    const int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return false;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

std::string buildSpikeFrame(double windowStartMs,
                            double windowEndMs,
                            const std::vector<int>& indices) {
    std::ostringstream oss;
    oss << "SNNFW_SPIKES v1 " << windowStartMs << " " << windowEndMs << " ";
    if (indices.empty()) {
        oss << "-";
    } else {
        for (size_t i = 0; i < indices.size(); ++i) {
            if (i > 0) oss << ",";
            oss << indices[i];
        }
    }
    oss << "\n";
    return oss.str();
}

} // namespace

InterneuronReceiverAdapter::InterneuronReceiverAdapter(const Config& config)
    : SensoryAdapter(config)
    , bindHost_(config.getStringParam("bind_host", "0.0.0.0"))
    , bindPort_(config.getIntParam("bind_port", 5000))
    , neuronCount_(std::max(1, config.getIntParam("neuron_count", 256)))
    , receiveTimeoutMs_(std::max(1, config.getIntParam("receive_timeout_ms", 5))) {}

InterneuronReceiverAdapter::~InterneuronReceiverAdapter() {
    closeSockets();
}

bool InterneuronReceiverAdapter::initialize() {
    closeSockets();

    listenFd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listenFd_ < 0) {
        std::cerr << "InterneuronReceiverAdapter socket() failed: " << std::strerror(errno) << "\n";
        return false;
    }

    const int reuse = 1;
    setsockopt(listenFd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(bindPort_));
    if (bindHost_ == "0.0.0.0" || bindHost_.empty()) {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else if (inet_pton(AF_INET, bindHost_.c_str(), &addr.sin_addr) != 1) {
        std::cerr << "InterneuronReceiverAdapter invalid bind_host: " << bindHost_ << "\n";
        closeSockets();
        return false;
    }

    if (bind(listenFd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::cerr << "InterneuronReceiverAdapter bind() failed: " << std::strerror(errno) << "\n";
        closeSockets();
        return false;
    }
    if (listen(listenFd_, 1) != 0 || !setNonBlocking(listenFd_)) {
        std::cerr << "InterneuronReceiverAdapter listen/nonblock failed: " << std::strerror(errno) << "\n";
        closeSockets();
        return false;
    }

    neurons_.clear();
    neurons_.reserve(static_cast<size_t>(neuronCount_));
    for (int i = 0; i < neuronCount_; ++i) {
        neurons_.push_back(std::make_shared<Neuron>(200.0, 0.7, 1, static_cast<uint64_t>(i)));
    }
    lastActivation_.assign(static_cast<size_t>(neuronCount_), 0.0);
    rxBuffer_.clear();
    return true;
}

void InterneuronReceiverAdapter::shutdown() {
    closeSockets();
}

bool InterneuronReceiverAdapter::ensureClient() {
    if (clientFd_ >= 0) return true;
    if (listenFd_ < 0) return false;

    sockaddr_in clientAddr {};
    socklen_t clientLen = sizeof(clientAddr);
    clientFd_ = accept(listenFd_, reinterpret_cast<sockaddr*>(&clientAddr), &clientLen);
    if (clientFd_ < 0) return false;
    if (!setNonBlocking(clientFd_)) {
        close(clientFd_);
        clientFd_ = -1;
        return false;
    }
    return true;
}

bool InterneuronReceiverAdapter::readAvailableLines(std::vector<std::string>& lines) {
    if (!ensureClient()) return false;

    char buf[4096];
    while (true) {
        const ssize_t n = recv(clientFd_, buf, sizeof(buf), 0);
        if (n > 0) {
            rxBuffer_.append(buf, static_cast<size_t>(n));
            continue;
        }
        if (n == 0) {
            close(clientFd_);
            clientFd_ = -1;
            break;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) break;
        close(clientFd_);
        clientFd_ = -1;
        break;
    }

    size_t pos = 0;
    while (true) {
        const size_t nl = rxBuffer_.find('\n', pos);
        if (nl == std::string::npos) break;
        lines.push_back(rxBuffer_.substr(pos, nl - pos));
        pos = nl + 1;
    }
    if (pos > 0) {
        rxBuffer_.erase(0, pos);
    }
    return !lines.empty();
}

bool InterneuronReceiverAdapter::parseLine(const std::string& line,
                                           std::vector<int>& activeIndices,
                                           double& windowStartMs,
                                           double& windowEndMs) const {
    std::istringstream iss(line);
    std::string magic;
    std::string version;
    std::string indices;
    if (!(iss >> magic >> version >> windowStartMs >> windowEndMs >> indices)) {
        return false;
    }
    if (magic != "SNNFW_SPIKES" || version != "v1") return false;

    activeIndices.clear();
    if (indices == "-") return true;

    std::stringstream ss(indices);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        const int idx = std::atoi(tok.c_str());
        if (idx >= 0 && idx < neuronCount_) {
            activeIndices.push_back(idx);
        }
    }
    return true;
}

SensoryAdapter::SpikePattern InterneuronReceiverAdapter::processData(const DataSample&) {
    SpikePattern pattern;
    pattern.timestamp = 0.0;
    pattern.duration = 0.0;
    pattern.spikeTimes.resize(neurons_.size());
    std::fill(lastActivation_.begin(), lastActivation_.end(), 0.0);

    std::vector<std::string> lines;
    if (!readAvailableLines(lines)) {
        return pattern;
    }

    std::vector<int> active;
    for (const auto& line : lines) {
        double startMs = 0.0;
        double endMs = 0.0;
        if (!parseLine(line, active, startMs, endMs)) continue;

        pattern.timestamp = startMs;
        pattern.duration = std::max(0.0, endMs - startMs);
        for (int idx : active) {
            if (idx < 0 || static_cast<size_t>(idx) >= pattern.spikeTimes.size()) continue;
            pattern.spikeTimes[static_cast<size_t>(idx)].push_back(pattern.duration);
            lastActivation_[static_cast<size_t>(idx)] = 1.0;
        }
    }
    return pattern;
}

SensoryAdapter::FeatureVector InterneuronReceiverAdapter::extractFeatures(const DataSample& data) {
    auto pattern = processData(data);
    FeatureVector f;
    f.timestamp = pattern.timestamp;
    f.features = lastActivation_;
    f.labels.reserve(lastActivation_.size());
    for (size_t i = 0; i < lastActivation_.size(); ++i) {
        f.labels.push_back("channel_" + std::to_string(i));
    }
    return f;
}

SensoryAdapter::SpikePattern InterneuronReceiverAdapter::encodeFeatures(const FeatureVector& features) {
    SpikePattern pattern;
    pattern.timestamp = features.timestamp;
    pattern.duration = config_.temporalWindow;
    pattern.spikeTimes.resize(features.features.size());
    for (size_t i = 0; i < features.features.size(); ++i) {
        if (features.features[i] > 0.0) {
            pattern.spikeTimes[i].push_back(0.0);
        }
    }
    return pattern;
}

void InterneuronReceiverAdapter::clearNeuronStates() {
    for (auto& n : neurons_) n->clearSpikes();
    std::fill(lastActivation_.begin(), lastActivation_.end(), 0.0);
}

void InterneuronReceiverAdapter::closeSockets() {
    if (clientFd_ >= 0) {
        close(clientFd_);
        clientFd_ = -1;
    }
    if (listenFd_ >= 0) {
        close(listenFd_);
        listenFd_ = -1;
    }
}

InterneuronTransmitterAdapter::InterneuronTransmitterAdapter(const Config& config)
    : MotorAdapter(config)
    , remoteHost_(config.getStringParam("remote_host", "127.0.0.1"))
    , remotePort_(config.getIntParam("remote_port", 5000))
    , connectTimeoutMs_(std::max(10, config.getIntParam("connect_timeout_ms", 1000)))
    , sendTimeoutMs_(std::max(10, config.getIntParam("send_timeout_ms", 100)))
    , updateIntervalMs_(std::max(1.0, config.getDoubleParam("update_interval_ms", 10.0)))
    , startupGraceMs_(std::max(0.0, config.getDoubleParam("startup_grace_ms", 0.0)))
    , transmitNeuronIds_(config.getIntParam("transmit_neuron_ids", 0) != 0) {}

InterneuronTransmitterAdapter::~InterneuronTransmitterAdapter() {
    closeSocket();
}

bool InterneuronTransmitterAdapter::initialize() {
    closeSocket();
    lastUpdateTime_ = 0.0;
    nextConnectAttemptTimeMs_ = startupGraceMs_;
    connectFailures_ = 0;
    lastConnectError_.clear();
    channelCount_ = 0;
    return true;
}

void InterneuronTransmitterAdapter::shutdown() {
    closeSocket();
}

bool InterneuronTransmitterAdapter::ensureConnected(double currentTimeMs) {
    if (socketFd_ >= 0) return true;
    if (currentTimeMs < nextConnectAttemptTimeMs_) return false;

    socketFd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (socketFd_ < 0) {
        const std::string err = std::strerror(errno);
        ++connectFailures_;
        if (connectFailures_ == 1 || connectFailures_ % 50 == 0 || err != lastConnectError_) {
            std::cerr << "InterneuronTransmitterAdapter socket() failed: " << err << "\n";
        }
        lastConnectError_ = err;
        nextConnectAttemptTimeMs_ = currentTimeMs + 250.0;
        return false;
    }

    timeval tv {};
    tv.tv_sec = sendTimeoutMs_ / 1000;
    tv.tv_usec = (sendTimeoutMs_ % 1000) * 1000;
    setsockopt(socketFd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(remotePort_));
    if (inet_pton(AF_INET, remoteHost_.c_str(), &addr.sin_addr) != 1) {
        std::cerr << "InterneuronTransmitterAdapter invalid remote_host: " << remoteHost_ << "\n";
        closeSocket();
        nextConnectAttemptTimeMs_ = currentTimeMs + 1000.0;
        return false;
    }

    if (connect(socketFd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        const std::string err = std::strerror(errno);
        ++connectFailures_;
        if (connectFailures_ == 1 || connectFailures_ % 50 == 0 || err != lastConnectError_) {
            std::cerr << "InterneuronTransmitterAdapter connect() failed: " << err << "\n";
        }
        lastConnectError_ = err;
        closeSocket();
        nextConnectAttemptTimeMs_ = currentTimeMs + 250.0;
        return false;
    }
    connectFailures_ = 0;
    lastConnectError_.clear();
    nextConnectAttemptTimeMs_ = currentTimeMs;
    return true;
}

MotorAdapter::MotorCommand InterneuronTransmitterAdapter::decodeActivity(const SpikeActivity& activity) {
    MotorCommand cmd;
    cmd.timestamp = activity.windowEnd;
    cmd.metadata["window_start_ms"] = activity.windowStart;
    cmd.metadata["window_end_ms"] = activity.windowEnd;
    cmd.values = activity.firingRates;
    cmd.channels.reserve(activity.firingRates.size());
    for (size_t i = 0; i < activity.firingRates.size(); ++i) {
        cmd.channels.push_back("channel_" + std::to_string(i));
    }
    return cmd;
}

bool InterneuronTransmitterAdapter::executeCommand(const MotorCommand& command) {
    if (!ensureConnected(command.timestamp)) return false;

    std::vector<int> active;
    if (!pendingActiveIndices_.empty()) {
        active = pendingActiveIndices_;
    } else {
        active.reserve(command.values.size());
        for (size_t i = 0; i < command.values.size(); ++i) {
            if (command.values[i] > 0.0) {
                active.push_back(static_cast<int>(i));
            }
        }
    }

    const double start = command.metadata.count("window_start_ms")
                             ? command.metadata.at("window_start_ms")
                             : command.timestamp;
    const std::string line = buildSpikeFrame(start, command.timestamp, active);
    if (!sendLine(line)) return false;

    currentCommand_ = command;
    if (actionCallback_) {
        actionCallback_(command);
    }
    return true;
}

bool InterneuronTransmitterAdapter::processNeurons(
    const std::vector<std::shared_ptr<Neuron>>& neurons,
    double currentTime) {
    if (currentTime - lastUpdateTime_ < updateIntervalMs_) return false;

    SpikeActivity activity;
    activity.windowStart = lastUpdateTime_;
    activity.windowEnd = currentTime;
    activity.firingRates.resize(neurons.size(), 0.0);
    channelCount_ = neurons.size();

    std::vector<int> active;
    active.reserve(neurons.size());
    for (size_t i = 0; i < neurons.size(); ++i) {
        auto spikes = neurons[i]->getSpikes();
        activity.firingRates[i] =
            calculateFiringRate(spikes, activity.windowStart, activity.windowEnd);
        if (activity.firingRates[i] > 0.0) {
            active.push_back(transmitNeuronIds_
                                 ? static_cast<int>(neurons[i]->getId())
                                 : static_cast<int>(i));
        }
    }

    pendingActiveIndices_ = std::move(active);
    auto command = decodeActivity(activity);

    const bool ok = executeCommand(command);
    pendingActiveIndices_.clear();
    lastUpdateTime_ = currentTime;
    return ok;
}

bool InterneuronTransmitterAdapter::sendLine(const std::string& line) {
    if (socketFd_ < 0) return false;

    const char* p = line.c_str();
    size_t remaining = line.size();
    while (remaining > 0) {
        const ssize_t sent = send(socketFd_, p, remaining, MSG_NOSIGNAL);
        if (sent <= 0) {
            closeSocket();
            return false;
        }
        p += sent;
        remaining -= static_cast<size_t>(sent);
    }
    return true;
}

void InterneuronTransmitterAdapter::closeSocket() {
    if (socketFd_ >= 0) {
        close(socketFd_);
        socketFd_ = -1;
    }
}

REGISTER_SENSORY_ADAPTER(InterneuronReceiverAdapter, "interneuron_rx");
REGISTER_MOTOR_ADAPTER(InterneuronTransmitterAdapter, "interneuron_tx");

void ensureInterneuronAdaptersRegistered() {}

} // namespace adapters
} // namespace snnfw

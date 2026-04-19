#include "snnfw/ActivityVisualizer.h"
#include "snnfw/ActivityMonitor.h"
#include "snnfw/NetworkDataAdapter.h"
#include "snnfw/RecordingManager.h"  // For RecordedSpike definition
#include <algorithm>
#include <fstream>
#include <spdlog/spdlog.h>

namespace snnfw {

ActivityVisualizer::ActivityVisualizer(ActivityMonitor& monitor, NetworkDataAdapter& adapter)
    : monitor_(monitor)
    , adapter_(adapter)
    , totalSpikes_(0)
    , lastUpdateTime_(0)
    , recording_(false)
    , playing_(false)
    , playbackLoop_(false)
    , playbackSpeed_(1.0f)
    , playbackStartTime_(0)
    , playbackIndex_(0)
{
    // Initialize neuron activity for all neurons in the network
    const auto& neurons = adapter_.getNeurons();
    neuronActivity_.reserve(neurons.size());
    
    for (size_t i = 0; i < neurons.size(); ++i) {
        NeuronActivity activity;
        activity.neuronId = neurons[i].id;
        activity.activityLevel = 0.0f;
        activity.lastSpikeTime = 0;
        activity.spikeCount = 0;
        activity.decayRate = config_.decayRate;
        
        neuronActivity_.push_back(activity);
        neuronActivityIndex_[activity.neuronId] = i;
    }
    
    spdlog::info("ActivityVisualizer created with {} neurons", neurons.size());
}

ActivityVisualizer::~ActivityVisualizer() {
    spdlog::info("ActivityVisualizer destroyed");
}

void ActivityVisualizer::update(uint64_t currentTime) {
    // Calculate delta time
    float deltaTime = 0.0f;
    if (lastUpdateTime_ > 0) {
        deltaTime = (currentTime - lastUpdateTime_) / 1000.0f; // Convert to seconds
    }
    lastUpdateTime_ = currentTime;

    // Process playback if active
    if (playing_) {
        processPlayback(currentTime);
    }

    // Note: Spike events are added via recordSpike() which is called
    // externally when spikes occur. We don't poll ActivityMonitor here
    // because it only provides aggregated statistics, not individual events.

    // Update particles
    if (deltaTime > 0.0f) {
        updateParticles(currentTime, deltaTime * 1000.0f); // Convert back to ms
        decayActivity(deltaTime);
    }
}

void ActivityVisualizer::recordSpike(uint64_t sourceNeuronId, uint64_t targetNeuronId,
                                     uint64_t synapseId, uint64_t timestamp) {
    onSpikeEvent(sourceNeuronId, targetNeuronId, synapseId, timestamp);
}

const std::vector<SpikeParticle>& ActivityVisualizer::getSpikeParticles() const {
    return particles_;
}

const std::vector<NeuronActivity>& ActivityVisualizer::getNeuronActivity() const {
    return neuronActivity_;
}

float ActivityVisualizer::getNeuronActivityLevel(uint64_t neuronId) const {
    auto it = neuronActivityIndex_.find(neuronId);
    if (it != neuronActivityIndex_.end()) {
        return neuronActivity_[it->second].activityLevel;
    }
    return 0.0f;
}

void ActivityVisualizer::setConfig(const ActivityConfig& config) {
    config_ = config;
    
    // Update decay rates for all neurons
    for (auto& activity : neuronActivity_) {
        activity.decayRate = config_.decayRate;
    }
}

const ActivityConfig& ActivityVisualizer::getConfig() const {
    return config_;
}

void ActivityVisualizer::startRecording() {
    recording_ = true;
    currentRecording_.spikes.clear();
    currentRecording_.startTime = lastUpdateTime_;
    spdlog::info("Started recording activity");
}

void ActivityVisualizer::stopRecording() {
    recording_ = false;
    currentRecording_.endTime = lastUpdateTime_;
    currentRecording_.duration = currentRecording_.endTime - currentRecording_.startTime;
    spdlog::info("Stopped recording activity - {} spikes recorded", 
                 currentRecording_.spikes.size());
}

bool ActivityVisualizer::isRecording() const {
    return recording_;
}

bool ActivityVisualizer::saveRecording(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        spdlog::error("Failed to open file for writing: {}", filename);
        return false;
    }
    
    // Write header
    file.write(reinterpret_cast<const char*>(&currentRecording_.startTime), sizeof(uint64_t));
    file.write(reinterpret_cast<const char*>(&currentRecording_.endTime), sizeof(uint64_t));
    file.write(reinterpret_cast<const char*>(&currentRecording_.duration), sizeof(uint64_t));
    
    // Write spike count
    uint64_t spikeCount = currentRecording_.spikes.size();
    file.write(reinterpret_cast<const char*>(&spikeCount), sizeof(uint64_t));
    
    // Write spikes
    for (const auto& spike : currentRecording_.spikes) {
        file.write(reinterpret_cast<const char*>(&spike), sizeof(RecordedSpike));
    }
    
    spdlog::info("Saved recording to {} - {} spikes", filename, spikeCount);
    return true;
}

bool ActivityVisualizer::loadRecording(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        spdlog::error("Failed to open file for reading: {}", filename);
        return false;
    }
    
    // Read header
    file.read(reinterpret_cast<char*>(&loadedRecording_.startTime), sizeof(uint64_t));
    file.read(reinterpret_cast<char*>(&loadedRecording_.endTime), sizeof(uint64_t));
    file.read(reinterpret_cast<char*>(&loadedRecording_.duration), sizeof(uint64_t));
    
    // Read spike count
    uint64_t spikeCount;
    file.read(reinterpret_cast<char*>(&spikeCount), sizeof(uint64_t));
    
    // Read spikes
    loadedRecording_.spikes.resize(spikeCount);
    for (uint64_t i = 0; i < spikeCount; ++i) {
        file.read(reinterpret_cast<char*>(&loadedRecording_.spikes[i]), sizeof(RecordedSpike));
    }
    
    spdlog::info("Loaded recording from {} - {} spikes", filename, spikeCount);
    return true;
}

void ActivityVisualizer::startPlayback(bool loop) {
    playing_ = true;
    playbackLoop_ = loop;
    playbackStartTime_ = lastUpdateTime_;
    playbackIndex_ = 0;
    clear();
    spdlog::info("Started playback - {} spikes", loadedRecording_.spikes.size());
}

void ActivityVisualizer::stopPlayback() {
    playing_ = false;
    spdlog::info("Stopped playback");
}

bool ActivityVisualizer::isPlaying() const {
    return playing_;
}

void ActivityVisualizer::setPlaybackSpeed(float speed) {
    playbackSpeed_ = speed;
}

uint32_t ActivityVisualizer::getTotalSpikes() const {
    return totalSpikes_;
}

float ActivityVisualizer::getAverageActivityLevel() const {
    if (neuronActivity_.empty()) {
        return 0.0f;
    }
    
    float sum = 0.0f;
    for (const auto& activity : neuronActivity_) {
        sum += activity.activityLevel;
    }
    return sum / neuronActivity_.size();
}

uint32_t ActivityVisualizer::getActiveNeuronCount() const {
    uint32_t count = 0;
    for (const auto& activity : neuronActivity_) {
        if (activity.activityLevel > 0.01f) {
            ++count;
        }
    }
    return count;
}

void ActivityVisualizer::clear() {
    particles_.clear();
    for (auto& activity : neuronActivity_) {
        activity.activityLevel = 0.0f;
        activity.spikeCount = 0;
    }
    totalSpikes_ = 0;
}

void ActivityVisualizer::rebuildFromAdapter() {
    particles_.clear();
    neuronActivity_.clear();
    neuronActivityIndex_.clear();
    totalSpikes_ = 0;

    const auto& neurons = adapter_.getNeurons();
    neuronActivity_.reserve(neurons.size());

    for (size_t i = 0; i < neurons.size(); ++i) {
        NeuronActivity activity;
        activity.neuronId = neurons[i].id;
        activity.activityLevel = 0.0f;
        activity.lastSpikeTime = 0;
        activity.spikeCount = 0;
        activity.decayRate = config_.decayRate;

        neuronActivity_.push_back(activity);
        neuronActivityIndex_[activity.neuronId] = i;
    }
}

void ActivityVisualizer::onSpikeEvent(uint64_t sourceNeuronId, uint64_t targetNeuronId,
                                     uint64_t synapseId, uint64_t timestamp) {
    // Update activity
    updateNeuronActivity(sourceNeuronId, timestamp);
    updateNeuronActivity(targetNeuronId, timestamp);
    
    // Create particle if enabled
    if (config_.showPropagation && particles_.size() < config_.maxParticles) {
        createSpikeParticle(sourceNeuronId, targetNeuronId, synapseId, timestamp);
    }
    
    // Record if enabled
    if (recording_) {
        RecordedSpike spike;
        spike.timestamp = timestamp;
        spike.sourceNeuronId = sourceNeuronId;
        spike.targetNeuronId = targetNeuronId;
        spike.synapseId = synapseId;
        currentRecording_.spikes.push_back(spike);
    }
    
    ++totalSpikes_;
}

void ActivityVisualizer::updateNeuronActivity(uint64_t neuronId, uint64_t timestamp) {
    auto it = neuronActivityIndex_.find(neuronId);
    if (it != neuronActivityIndex_.end()) {
        auto& activity = neuronActivity_[it->second];
        activity.lastSpikeTime = timestamp;
        activity.spikeCount++;
        
        // Increase activity level (clamped to 1.0)
        activity.activityLevel = std::min(1.0f, activity.activityLevel + 0.3f);
    }
}

void ActivityVisualizer::createSpikeParticle(uint64_t sourceNeuronId, uint64_t targetNeuronId,
                                            uint64_t synapseId, uint64_t timestamp) {
    // Get neuron positions from adapter
    const auto& neurons = adapter_.getNeurons();
    
    // Find source and target positions
    glm::vec3 sourcePos(0.0f);
    glm::vec3 targetPos(0.0f);
    bool foundSource = false;
    bool foundTarget = false;
    
    for (const auto& neuron : neurons) {
        if (neuron.id == sourceNeuronId) {
            sourcePos = glm::vec3(neuron.position.x, neuron.position.y, neuron.position.z);
            foundSource = true;
        }
        if (neuron.id == targetNeuronId) {
            targetPos = glm::vec3(neuron.position.x, neuron.position.y, neuron.position.z);
            foundTarget = true;
        }
        if (foundSource && foundTarget) break;
    }
    
    if (!foundSource || !foundTarget) {
        return;
    }
    
    // Create particle
    SpikeParticle particle;
    particle.position = sourcePos;
    particle.velocity = glm::normalize(targetPos - sourcePos) * config_.propagationSpeed;
    particle.color = config_.excitatoryColor; // TODO: Use neuron type
    particle.size = config_.spikeParticleSize;
    particle.lifetime = 0.0f;
    particle.maxLifetime = static_cast<float>(config_.particleLifetime);
    particle.synapseId = synapseId;
    particle.progress = 0.0f;
    particle.sourceNeuronId = sourceNeuronId;
    particle.targetNeuronId = targetNeuronId;
    
    particles_.push_back(particle);
}

void ActivityVisualizer::updateParticles(uint64_t currentTime, float deltaTime) {
    // Update and remove dead particles
    particles_.erase(
        std::remove_if(particles_.begin(), particles_.end(),
            [deltaTime](SpikeParticle& particle) {
                particle.lifetime += deltaTime;
                
                // Update position
                particle.position += particle.velocity * (deltaTime / 1000.0f);
                
                // Update progress
                particle.progress = particle.lifetime / particle.maxLifetime;
                
                // Remove if lifetime exceeded
                return particle.lifetime >= particle.maxLifetime;
            }),
        particles_.end()
    );
}

void ActivityVisualizer::decayActivity(float deltaTime) {
    for (auto& activity : neuronActivity_) {
        if (activity.activityLevel > 0.0f) {
            activity.activityLevel -= activity.decayRate * deltaTime;
            activity.activityLevel = std::max(0.0f, activity.activityLevel);
        }
    }
}

void ActivityVisualizer::processPlayback(uint64_t currentTime) {
    if (loadedRecording_.spikes.empty()) {
        return;
    }
    
    // Calculate playback time
    uint64_t playbackTime = static_cast<uint64_t>(
        (currentTime - playbackStartTime_) * playbackSpeed_
    );
    
    // Process spikes up to current playback time
    while (playbackIndex_ < loadedRecording_.spikes.size()) {
        const auto& spike = loadedRecording_.spikes[playbackIndex_];
        uint64_t relativeTime = spike.timestamp - loadedRecording_.startTime;
        
        if (relativeTime > playbackTime) {
            break;
        }
        
        // Trigger spike event
        onSpikeEvent(spike.sourceNeuronId, spike.targetNeuronId, 
                    spike.synapseId, currentTime);
        
        ++playbackIndex_;
    }
    
    // Check if playback finished
    if (playbackIndex_ >= loadedRecording_.spikes.size()) {
        if (playbackLoop_) {
            playbackIndex_ = 0;
            playbackStartTime_ = currentTime;
            clear();
        } else {
            stopPlayback();
        }
    }
}

} // namespace snnfw

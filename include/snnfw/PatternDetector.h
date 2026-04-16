#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <glm/glm.hpp>

namespace snnfw {

/**
 * @brief Type of detected pattern
 */
enum class PatternType {
    SYNCHRONY,          ///< Synchronous firing of multiple neurons
    OSCILLATION,        ///< Periodic firing pattern
    WAVE,              ///< Propagating wave of activity
    BURST,             ///< Burst of rapid firing
    SILENCE            ///< Period of no activity
};

/**
 * @brief Detected synchrony event
 */
struct SynchronyEvent {
    uint64_t timestamp;                     ///< Time of synchrony
    std::vector<uint64_t> neuronIds;        ///< Neurons involved
    float coherence;                        ///< Coherence measure (0-1)
    float timeWindow;                       ///< Time window used (ms)
    
    SynchronyEvent() : timestamp(0), coherence(0.0f), timeWindow(0.0f) {}
};

/**
 * @brief Detected oscillation pattern
 */
struct OscillationPattern {
    std::vector<uint64_t> neuronIds;        ///< Neurons involved
    float frequency;                        ///< Frequency in Hz
    float amplitude;                        ///< Amplitude (spike rate variation)
    float phase;                            ///< Phase offset
    float confidence;                       ///< Detection confidence (0-1)
    uint64_t startTime;                     ///< Start time
    uint64_t endTime;                       ///< End time
    
    OscillationPattern() : frequency(0.0f), amplitude(0.0f), phase(0.0f), 
                          confidence(0.0f), startTime(0), endTime(0) {}
};

/**
 * @brief Detected wave propagation
 */
struct WaveEvent {
    uint64_t timestamp;                     ///< Wave start time
    glm::vec3 origin;                       ///< Wave origin position
    glm::vec3 direction;                    ///< Wave propagation direction
    float speed;                            ///< Propagation speed (units/ms)
    std::vector<uint64_t> neuronIds;        ///< Neurons involved (in order)
    std::vector<uint64_t> firingTimes;      ///< Firing times for each neuron
    float coherence;                        ///< Wave coherence (0-1)
    
    WaveEvent() : timestamp(0), origin(0.0f), direction(0.0f), 
                 speed(0.0f), coherence(0.0f) {}
};

/**
 * @brief Detected burst event
 */
struct BurstEvent {
    uint64_t neuronId;                      ///< Neuron that burst
    uint64_t startTime;                     ///< Burst start time
    uint64_t endTime;                       ///< Burst end time
    int spikeCount;                         ///< Number of spikes in burst
    float frequency;                        ///< Intra-burst frequency (Hz)
    
    BurstEvent() : neuronId(0), startTime(0), endTime(0), 
                  spikeCount(0), frequency(0.0f) {}
};

/**
 * @brief Configuration for pattern detection
 */
struct PatternDetectorConfig {
    // Synchrony detection
    bool detectSynchrony;
    float synchronyTimeWindow;              ///< Time window for synchrony (ms)
    int synchronyMinNeurons;                ///< Minimum neurons for synchrony
    float synchronyThreshold;               ///< Coherence threshold (0-1)
    
    // Oscillation detection
    bool detectOscillations;
    float oscillationMinFrequency;          ///< Minimum frequency (Hz)
    float oscillationMaxFrequency;          ///< Maximum frequency (Hz)
    float oscillationMinDuration;           ///< Minimum duration (ms)
    float oscillationConfidenceThreshold;   ///< Confidence threshold (0-1)
    
    // Wave detection
    bool detectWaves;
    float waveMinSpeed;                     ///< Minimum wave speed (units/ms)
    float waveMaxSpeed;                     ///< Maximum wave speed (units/ms)
    float waveTimeWindow;                   ///< Time window for wave detection (ms)
    int waveMinNeurons;                     ///< Minimum neurons in wave
    
    // Burst detection
    bool detectBursts;
    float burstTimeWindow;                  ///< Time window for burst (ms)
    int burstMinSpikes;                     ///< Minimum spikes for burst
    float burstMinFrequency;                ///< Minimum intra-burst frequency (Hz)
    
    // General
    uint64_t historyDuration;               ///< How long to keep history (ms)
    
    PatternDetectorConfig()
        : detectSynchrony(true)
        , synchronyTimeWindow(10.0f)
        , synchronyMinNeurons(3)
        , synchronyThreshold(0.7f)
        , detectOscillations(true)
        , oscillationMinFrequency(1.0f)
        , oscillationMaxFrequency(100.0f)
        , oscillationMinDuration(100.0f)
        , oscillationConfidenceThreshold(0.6f)
        , detectWaves(true)
        , waveMinSpeed(0.1f)
        , waveMaxSpeed(10.0f)
        , waveTimeWindow(100.0f)
        , waveMinNeurons(5)
        , detectBursts(true)
        , burstTimeWindow(50.0f)
        , burstMinSpikes(3)
        , burstMinFrequency(20.0f)
        , historyDuration(5000) {}
};

/**
 * @brief Spike event for pattern detection
 */
struct PatternSpikeEvent {
    uint64_t neuronId;
    uint64_t timestamp;
    glm::vec3 position;

    PatternSpikeEvent() : neuronId(0), timestamp(0), position(0.0f) {}
    PatternSpikeEvent(uint64_t id, uint64_t time, const glm::vec3& pos)
        : neuronId(id), timestamp(time), position(pos) {}
};

/**
 * @class PatternDetector
 * @brief Detects temporal and spatial patterns in neural activity
 * 
 * The PatternDetector analyzes spike trains to identify:
 * - Synchronous firing (multiple neurons firing together)
 * - Oscillations (periodic firing patterns)
 * - Propagating waves (spatially organized activity)
 * - Bursts (rapid firing episodes)
 * 
 * Features:
 * - Real-time pattern detection
 * - Configurable detection parameters
 * - Multiple pattern types
 * - Pattern visualization support
 */
class PatternDetector {
public:
    PatternDetector();
    ~PatternDetector();
    
    // Configuration
    void setConfig(const PatternDetectorConfig& config);
    const PatternDetectorConfig& getConfig() const { return config_; }
    
    // Spike recording
    void recordSpike(uint64_t neuronId, uint64_t timestamp, const glm::vec3& position);
    void clearHistory();
    
    // Pattern detection (call periodically)
    void update(uint64_t currentTime);
    
    // Get detected patterns
    const std::vector<SynchronyEvent>& getSynchronyEvents() const { return synchronyEvents_; }
    const std::vector<OscillationPattern>& getOscillations() const { return oscillations_; }
    const std::vector<WaveEvent>& getWaveEvents() const { return waveEvents_; }
    const std::vector<BurstEvent>& getBurstEvents() const { return burstEvents_; }
    
    // Clear detected patterns
    void clearPatterns();
    
    // Statistics
    size_t getSynchronyCount() const { return synchronyEvents_.size(); }
    size_t getOscillationCount() const { return oscillations_.size(); }
    size_t getWaveCount() const { return waveEvents_.size(); }
    size_t getBurstCount() const { return burstEvents_.size(); }
    
    // Get recent spikes for visualization
    const std::vector<PatternSpikeEvent>& getRecentSpikes() const { return spikeHistory_; }

private:
    // Detection methods
    void detectSynchrony(uint64_t currentTime);
    void detectOscillations(uint64_t currentTime);
    void detectWaves(uint64_t currentTime);
    void detectBursts(uint64_t currentTime);
    
    // Helper methods
    void cleanupOldSpikes(uint64_t currentTime);
    std::vector<uint64_t> getSpikesInWindow(uint64_t startTime, uint64_t endTime) const;
    std::vector<PatternSpikeEvent> getNeuronSpikes(uint64_t neuronId, uint64_t startTime, uint64_t endTime) const;
    float calculateCoherence(const std::vector<uint64_t>& timestamps, float timeWindow) const;
    float calculateFrequency(const std::vector<uint64_t>& timestamps) const;
    bool detectOscillationInSpikes(const std::vector<uint64_t>& timestamps,
                                   float& frequency, float& confidence) const;
    glm::vec3 getNeuronPosition(uint64_t neuronId) const;

private:
    PatternDetectorConfig config_;

    // Spike history
    std::vector<PatternSpikeEvent> spikeHistory_;
    std::unordered_map<uint64_t, std::vector<PatternSpikeEvent>> neuronSpikes_;  // Per-neuron spike history
    std::unordered_map<uint64_t, glm::vec3> neuronPositions_;             // Neuron positions
    
    // Detected patterns
    std::vector<SynchronyEvent> synchronyEvents_;
    std::vector<OscillationPattern> oscillations_;
    std::vector<WaveEvent> waveEvents_;
    std::vector<BurstEvent> burstEvents_;
    
    // Last detection time
    uint64_t lastDetectionTime_;
};

} // namespace snnfw


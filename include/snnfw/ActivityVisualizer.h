#ifndef SNNFW_ACTIVITY_VISUALIZER_H
#define SNNFW_ACTIVITY_VISUALIZER_H

#include <glm/glm.hpp>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <string>

namespace snnfw {

// Forward declarations
class ActivityMonitor;
class NetworkDataAdapter;

/**
 * @brief Particle representing a spike event
 */
struct SpikeParticle {
    glm::vec3 position;        // Current position in 3D space
    glm::vec3 velocity;        // Movement direction and speed
    glm::vec4 color;           // RGBA color
    float size;                // Particle size
    float lifetime;            // Time since creation (ms)
    float maxLifetime;         // Total lifetime (ms)
    uint64_t synapseId;        // Associated synapse (0 if none)
    float progress;            // Progress along synapse (0-1)
    uint64_t sourceNeuronId;   // Source neuron
    uint64_t targetNeuronId;   // Target neuron
};

/**
 * @brief Activity state for a single neuron
 */
struct NeuronActivity {
    uint64_t neuronId;
    float activityLevel;       // 0.0 - 1.0 (normalized)
    uint64_t lastSpikeTime;    // Timestamp of last spike (ms)
    uint32_t spikeCount;       // Spikes in current window
    float decayRate;           // Activity decay rate (per second)
};

/**
 * @brief Configuration for activity visualization
 */
struct ActivityConfig {
    uint64_t historyDuration = 1000;      // History window (ms)
    float decayRate = 2.0f;                // Activity decay per second
    float spikeParticleSize = 0.5f;        // Size of spike particles
    uint64_t particleLifetime = 500;       // Particle lifetime (ms)
    bool showPropagation = true;           // Show spike propagation
    bool showHeatmap = true;               // Show activity heatmap
    glm::vec4 excitatoryColor = glm::vec4(1.0f, 0.3f, 0.3f, 1.0f);  // Red
    glm::vec4 inhibitoryColor = glm::vec4(0.3f, 0.3f, 1.0f, 1.0f);  // Blue
    float propagationSpeed = 5.0f;         // Units per second
    uint32_t maxParticles = 10000;         // Maximum active particles
};

// Forward declaration - defined in RecordingManager.h
struct RecordedSpike;

/**
 * @brief Activity recording for playback
 */
struct ActivityRecording {
    std::vector<RecordedSpike> spikes;
    uint64_t startTime;
    uint64_t endTime;
    uint64_t duration;
};

/**
 * @brief Visualizes neural network activity in real-time
 * 
 * This class tracks spike events from ActivityMonitor and generates
 * visual representations including:
 * - Spike particles (billboards)
 * - Activity heatmaps (neuron coloring)
 * - Spike propagation (animated trails)
 * - Temporal patterns (activity over time)
 * 
 * Features:
 * - Real-time spike tracking
 * - Exponential activity decay
 * - Particle system for spike visualization
 * - Recording and playback
 * - Configurable visual parameters
 * 
 * Usage:
 * @code
 * ActivityVisualizer visualizer(activityMonitor, networkAdapter);
 * 
 * // Configure
 * ActivityConfig config;
 * config.showPropagation = true;
 * config.decayRate = 2.0f;
 * visualizer.setConfig(config);
 * 
 * // Update each frame
 * visualizer.update(currentTimeMs);
 * 
 * // Get data for rendering
 * auto& particles = visualizer.getSpikeParticles();
 * auto& activity = visualizer.getNeuronActivity();
 * @endcode
 */
class ActivityVisualizer {
public:
    /**
     * @brief Constructor
     * @param monitor ActivityMonitor to track spike events
     * @param adapter NetworkDataAdapter for neuron positions
     */
    ActivityVisualizer(ActivityMonitor& monitor, NetworkDataAdapter& adapter);
    
    /**
     * @brief Destructor
     */
    ~ActivityVisualizer();
    
    /**
     * @brief Update activity state for current frame
     * @param currentTime Current simulation time (ms)
     */
    void update(uint64_t currentTime);

    /**
     * @brief Record a spike event for visualization
     * @param sourceNeuronId Source neuron ID
     * @param targetNeuronId Target neuron ID
     * @param synapseId Synapse ID
     * @param timestamp Event timestamp (ms)
     */
    void recordSpike(uint64_t sourceNeuronId, uint64_t targetNeuronId,
                    uint64_t synapseId, uint64_t timestamp);
    
    /**
     * @brief Get current spike particles for rendering
     * @return Vector of active spike particles
     */
    const std::vector<SpikeParticle>& getSpikeParticles() const;
    
    /**
     * @brief Get neuron activity levels
     * @return Vector of neuron activity states
     */
    const std::vector<NeuronActivity>& getNeuronActivity() const;
    
    /**
     * @brief Get activity level for specific neuron
     * @param neuronId Neuron ID
     * @return Activity level (0.0 - 1.0), or 0.0 if not found
     */
    float getNeuronActivityLevel(uint64_t neuronId) const;
    
    /**
     * @brief Set configuration
     * @param config Activity configuration
     */
    void setConfig(const ActivityConfig& config);
    
    /**
     * @brief Get current configuration
     * @return Current configuration
     */
    const ActivityConfig& getConfig() const;
    
    /**
     * @brief Start recording activity
     */
    void startRecording();
    
    /**
     * @brief Stop recording activity
     */
    void stopRecording();
    
    /**
     * @brief Check if currently recording
     * @return true if recording
     */
    bool isRecording() const;
    
    /**
     * @brief Save recording to file
     * @param filename Output filename
     * @return true if successful
     */
    bool saveRecording(const std::string& filename);
    
    /**
     * @brief Load recording from file
     * @param filename Input filename
     * @return true if successful
     */
    bool loadRecording(const std::string& filename);
    
    /**
     * @brief Start playback of loaded recording
     * @param loop Whether to loop playback
     */
    void startPlayback(bool loop = false);
    
    /**
     * @brief Stop playback
     */
    void stopPlayback();
    
    /**
     * @brief Check if currently playing back
     * @return true if playing
     */
    bool isPlaying() const;
    
    /**
     * @brief Set playback speed
     * @param speed Speed multiplier (1.0 = normal speed)
     */
    void setPlaybackSpeed(float speed);
    
    /**
     * @brief Get total spike count
     * @return Total spikes processed
     */
    uint32_t getTotalSpikes() const;
    
    /**
     * @brief Get average activity level across all neurons
     * @return Average activity (0.0 - 1.0)
     */
    float getAverageActivityLevel() const;
    
    /**
     * @brief Get count of active neurons (activity > 0)
     * @return Number of active neurons
     */
    uint32_t getActiveNeuronCount() const;
    
    /**
     * @brief Clear all activity and particles
     */
    void clear();

    /**
     * @brief Rebuild activity cache from current NetworkDataAdapter state
     */
    void rebuildFromAdapter();

private:
    /**
     * @brief Handle spike event from ActivityMonitor
     * @param sourceNeuronId Source neuron
     * @param targetNeuronId Target neuron
     * @param synapseId Synapse ID
     * @param timestamp Event timestamp
     */
    void onSpikeEvent(uint64_t sourceNeuronId, uint64_t targetNeuronId, 
                      uint64_t synapseId, uint64_t timestamp);
    
    /**
     * @brief Update neuron activity level
     * @param neuronId Neuron ID
     * @param timestamp Current time
     */
    void updateNeuronActivity(uint64_t neuronId, uint64_t timestamp);
    
    /**
     * @brief Create spike particle for visualization
     * @param sourceNeuronId Source neuron
     * @param targetNeuronId Target neuron
     * @param synapseId Synapse ID
     * @param timestamp Event timestamp
     */
    void createSpikeParticle(uint64_t sourceNeuronId, uint64_t targetNeuronId,
                            uint64_t synapseId, uint64_t timestamp);
    
    /**
     * @brief Update all spike particles
     * @param currentTime Current time
     * @param deltaTime Time since last update (ms)
     */
    void updateParticles(uint64_t currentTime, float deltaTime);
    
    /**
     * @brief Decay activity levels
     * @param deltaTime Time since last update (seconds)
     */
    void decayActivity(float deltaTime);
    
    /**
     * @brief Process playback events
     * @param currentTime Current playback time
     */
    void processPlayback(uint64_t currentTime);

    // References
    ActivityMonitor& monitor_;
    NetworkDataAdapter& adapter_;
    
    // Configuration
    ActivityConfig config_;
    
    // Activity tracking
    std::vector<NeuronActivity> neuronActivity_;
    std::unordered_map<uint64_t, size_t> neuronActivityIndex_;
    
    // Particle system
    std::vector<SpikeParticle> particles_;
    
    // Statistics
    uint32_t totalSpikes_;
    uint64_t lastUpdateTime_;
    
    // Recording
    bool recording_;
    ActivityRecording currentRecording_;
    
    // Playback
    bool playing_;
    bool playbackLoop_;
    float playbackSpeed_;
    uint64_t playbackStartTime_;
    size_t playbackIndex_;
    ActivityRecording loadedRecording_;
};

} // namespace snnfw

#endif // SNNFW_ACTIVITY_VISUALIZER_H

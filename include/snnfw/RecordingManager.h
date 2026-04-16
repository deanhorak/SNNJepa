#ifndef SNNFW_RECORDING_MANAGER_H
#define SNNFW_RECORDING_MANAGER_H

#include <string>
#include <vector>
#include <cstdint>
#include <functional>

namespace snnfw {

// Forward declarations
class ActivityVisualizer;

/**
 * @brief Recorded spike event
 */
struct RecordedSpike {
    uint64_t timestamp;
    uint64_t sourceNeuronId;
    uint64_t targetNeuronId;
    uint64_t synapseId;
};

/**
 * @brief Callback function type for playback spike events
 *
 * Called for each spike during playback.
 * Parameters: sourceNeuronId, targetNeuronId, synapseId, timestamp
 */
using PlaybackSpikeCallback = std::function<void(uint64_t, uint64_t, uint64_t, uint64_t)>;

/**
 * @brief Metadata for a spike recording
 */
struct RecordingMetadata {
    std::string name;           ///< Recording name
    uint64_t startTime;         ///< Start time (ms)
    uint64_t endTime;           ///< End time (ms)
    uint64_t duration;          ///< Duration (ms)
    size_t spikeCount;          ///< Total number of spikes
    size_t neuronCount;         ///< Number of unique neurons
    std::string timestamp;      ///< ISO 8601 timestamp
};

/**
 * @brief Playback state
 */
struct PlaybackState {
    bool playing;               ///< Currently playing
    bool paused;                ///< Currently paused
    bool looping;               ///< Loop playback
    float speed;                ///< Playback speed multiplier (1.0 = normal)
    uint64_t currentTime;       ///< Current playback time (ms)
    uint64_t startTime;         ///< Playback start time (ms)
    uint64_t endTime;           ///< Playback end time (ms)
    
    PlaybackState() 
        : playing(false), paused(false), looping(false), speed(1.0f),
          currentTime(0), startTime(0), endTime(0) {}
};

/**
 * @brief Manages recording and playback of spike activity
 * 
 * RecordingManager allows you to:
 * - Record spike activity to memory
 * - Save recordings to binary files
 * - Load recordings from files
 * - Play back recordings with speed control
 * - Loop playback
 * 
 * Example usage:
 * ```cpp
 * RecordingManager recorder(activityVisualizer);
 * 
 * // Start recording
 * recorder.startRecording();
 * 
 * // ... simulation runs ...
 * 
 * // Stop and save
 * recorder.stopRecording();
 * recorder.saveRecording("my_recording.snnr");
 * 
 * // Later, load and play back
 * recorder.loadRecording("my_recording.snnr");
 * recorder.setSpeed(2.0f);  // 2x speed
 * recorder.setLooping(true);
 * recorder.play();
 * ```
 */
class RecordingManager {
public:
    /**
     * @brief Constructor with visualizer (for backward compatibility)
     * @param visualizer Activity visualizer to record from
     */
    explicit RecordingManager(ActivityVisualizer& visualizer);

    /**
     * @brief Constructor without visualizer (for recording-only mode)
     */
    RecordingManager();

    /**
     * @brief Destructor
     */
    ~RecordingManager();
    
    // Recording control
    
    /**
     * @brief Start recording spike activity
     * @param streamToFile If true, stream spikes directly to file instead of memory
     * @param filename Filename for streaming (required if streamToFile is true)
     */
    void startRecording(bool streamToFile = false, const std::string& filename = "");

    /**
     * @brief Stop recording
     */
    void stopRecording();
    
    /**
     * @brief Check if currently recording
     * @return true if recording
     */
    bool isRecording() const { return recording_; }
    
    /**
     * @brief Add a spike to the current recording
     * @param spike Spike event to record
     */
    void recordSpike(const RecordedSpike& spike);
    
    // Save/load
    
    /**
     * @brief Save recording to file
     * @param filename Path to save file
     * @return true if successful
     */
    bool saveRecording(const std::string& filename);
    
    /**
     * @brief Load recording from file
     * @param filename Path to recording file
     * @param streamingPlayback If true, stream spikes from file during playback instead of loading all into memory
     * @return true if successful
     */
    bool loadRecording(const std::string& filename, bool streamingPlayback = false);
    
    // Playback control
    
    /**
     * @brief Start playback
     */
    void play();
    
    /**
     * @brief Pause playback
     */
    void pause();
    
    /**
     * @brief Stop playback and reset to beginning
     */
    void stop();
    
    /**
     * @brief Set playback speed
     * @param speed Speed multiplier (1.0 = normal, 2.0 = 2x, 0.5 = half speed)
     */
    void setSpeed(float speed);
    
    /**
     * @brief Enable/disable looping
     * @param loop true to enable looping
     */
    void setLooping(bool loop);
    
    /**
     * @brief Seek to specific time in recording
     * @param time Time to seek to (ms)
     */
    void seek(uint64_t time);
    
    /**
     * @brief Update playback (call every frame)
     * @param deltaTime Time since last update (ms)
     */
    void update(uint64_t deltaTime);
    
    // State access
    
    /**
     * @brief Get playback state
     * @return Current playback state
     */
    const PlaybackState& getPlaybackState() const { return playbackState_; }
    
    /**
     * @brief Get recording metadata
     * @return Recording metadata
     */
    const RecordingMetadata& getMetadata() const { return metadata_; }
    
    /**
     * @brief Get all recorded spikes
     * @return Vector of recorded spikes
     */
    const std::vector<RecordedSpike>& getRecording() const { return spikes_; }
    
    /**
     * @brief Clear current recording
     */
    void clearRecording();

    /**
     * @brief Set playback callback for spike events
     * @param callback Function to call for each spike during playback
     */
    void setPlaybackCallback(PlaybackSpikeCallback callback);

private:
    ActivityVisualizer* visualizer_;  // Optional - nullptr if not using visualizer

    // Recording state
    bool recording_;
    uint64_t recordingStartTime_;
    std::vector<RecordedSpike> spikes_;
    RecordingMetadata metadata_;

    // Streaming state
    bool streamingMode_;
    bool wasStreamedToFile_;  // Track if recording was streamed (for saveRecording)
    std::string streamingFilename_;
    std::ofstream* streamingFile_;
    uint64_t streamedSpikeCount_;

    // Playback state
    PlaybackState playbackState_;
    size_t playbackIndex_;  // Current index in recording

    // Streaming playback state
    bool streamingPlayback_;  // If true, stream spikes from file during playback
    std::string playbackFilename_;  // Filename for streaming playback
    std::ifstream* playbackFile_;  // File stream for streaming playback
    uint64_t playbackFileOffset_;  // Current file offset for streaming
    uint64_t totalSpikesInFile_;  // Total spikes in file (for progress tracking)

    // Playback callback
    PlaybackSpikeCallback playbackCallback_;

    // Helper methods
    void updateMetadata();
    std::string getCurrentTimestamp();
    void writeFileHeader();
    void updateFileHeader();
};

} // namespace snnfw

#endif // SNNFW_RECORDING_MANAGER_H


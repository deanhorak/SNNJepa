#ifndef SNNFW_SIMULATION_CONFIG_H
#define SNNFW_SIMULATION_CONFIG_H

#include <string>
#include <cstdint>
#include <chrono>
#include <sstream>
#include <iomanip>

namespace snnfw {

/**
 * @brief Configuration for simulation recording and visualization
 * 
 * This class provides centralized configuration for optional framework features
 * like recording and visualization. It allows experiments to enable/disable
 * these features independently with minimal code changes.
 * 
 * Usage Modes:
 * 1. Performance Mode (default):
 *    - recording=false, visualization=false
 *    - Maximum performance, no overhead
 * 
 * 2. Recording Mode:
 *    - recording=true, visualization=false
 *    - Fast execution with spike recording for later playback
 * 
 * 3. Visualization Mode:
 *    - recording=false, visualization=true
 *    - Live visualization without recording
 * 
 * 4. Debug Mode:
 *    - recording=true, visualization=true
 *    - Full debugging with live visualization and recording
 * 
 * Example:
 * ```cpp
 * SimulationConfig config;
 * config.enableRecording = true;
 * config.recordingFilename = "training_session.snnr";
 * config.enableVisualization = false;  // Record without visualization
 * 
 * ActivityMonitor monitor(datastore, config);
 * // Recording happens automatically when spikes are recorded
 * ```
 */
struct SimulationConfig {
    // Recording configuration
    bool enableRecording = false;           ///< Enable spike recording
    std::string recordingFilename = "";     ///< Output filename for recording (empty = auto-generate)
    std::string recordingPath = "";         ///< Path where recording files are stored (local or remote)
    bool autoSaveRecording = true;          ///< Automatically save recording when stopped

    // Network structure configuration
    std::string networkStructurePath = "";  ///< Path where network structure files (.snnw) are stored (local or remote)

    // Visualization configuration
    bool enableVisualization = false;       ///< Enable live visualization
    int visualizationWidth = 1920;          ///< Visualization window width
    int visualizationHeight = 1080;         ///< Visualization window height
    std::string visualizationTitle = "SNNFW Visualization";  ///< Window title
    
    // Playback configuration
    bool playbackMode = false;              ///< Enable playback mode (mutually exclusive with live mode)
    std::string playbackFilename = "";      ///< Recording file to play back
    float playbackSpeed = 1.0f;             ///< Initial playback speed (1.0 = normal)
    bool playbackLooping = false;           ///< Loop playback when it reaches the end
    
    // Performance configuration
    bool realTimeSync = true;               ///< Synchronize SpikeProcessor with real-time (disable for max speed)
    size_t spikeProcessorThreads = 20;      ///< Number of spike delivery threads

    // Memory management configuration
    bool enableMemoryMonitoring = true;     ///< Enable memory usage monitoring
    size_t maxMemoryMB = 8192;              ///< Maximum memory to use (8GB default)
    double historyDurationMs = 1000.0;      ///< How long to keep spike events in memory (ms)
    bool aggressiveCleanup = true;          ///< Aggressively clean old events when memory is high

    /**
     * @brief Check if any visualization is needed (live or playback)
     * @return true if visualization window should be created
     */
    bool needsVisualization() const {
        return enableVisualization || playbackMode;
    }
    
    /**
     * @brief Check if this is a performance-optimized configuration
     * @return true if no recording or visualization enabled
     */
    bool isPerformanceMode() const {
        return !enableRecording && !enableVisualization && !playbackMode;
    }
    
    /**
     * @brief Validate configuration
     * @return true if configuration is valid
     */
    bool validate() const {
        // Can't have live visualization and playback at the same time
        if (enableVisualization && playbackMode) {
            return false;
        }
        
        // Playback mode requires a filename
        if (playbackMode && playbackFilename.empty()) {
            return false;
        }
        
        // Recording requires a filename (or auto-generate)
        // (empty filename is OK - will be auto-generated)
        
        return true;
    }
    
    /**
     * @brief Generate a default recording filename based on timestamp
     * @return Generated filename
     */
    std::string generateRecordingFilename() const {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << "recording_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ".snnr";
        return ss.str();
    }
};

} // namespace snnfw

#endif // SNNFW_SIMULATION_CONFIG_H


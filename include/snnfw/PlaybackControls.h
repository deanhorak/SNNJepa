#ifndef SNNFW_PLAYBACK_CONTROLS_H
#define SNNFW_PLAYBACK_CONTROLS_H

#include <cstdint>
#include <cstddef>

namespace snnfw {

// Forward declaration
class RecordingManager;

/**
 * @brief ImGui playback controls for RecordingManager
 * 
 * Provides a complete playback UI with play/pause, speed control,
 * timeline scrubbing, and metadata display.
 */
class PlaybackControls {
public:
    /**
     * @brief Constructor
     * 
     * @param recordingManager Recording manager to control
     */
    explicit PlaybackControls(RecordingManager* recordingManager);
    
    /**
     * @brief Render ImGui controls
     * 
     * Call this every frame within ImGui context
     */
    void render();
    
    /**
     * @brief Set window position
     * 
     * @param x X position in pixels
     * @param y Y position in pixels
     */
    void setPosition(float x, float y);
    
    /**
     * @brief Set window size
     * 
     * @param width Width in pixels
     * @param height Height in pixels
     */
    void setSize(float width, float height);
    
    /**
     * @brief Show/hide controls
     * 
     * @param show true to show controls
     */
    void setVisible(bool show);
    
    /**
     * @brief Check if controls are visible
     * 
     * @return true if visible
     */
    bool isVisible() const { return visible_; }
    
private:
    RecordingManager* recordingManager_;
    
    bool visible_;
    float posX_;
    float posY_;
    float width_;
    float height_;
    
    // Helper methods
    void renderPlaybackButtons();
    void renderSpeedControl();
    void renderTimeline();
    void renderMetadata();
    void renderSingleStepButtons();
    
    // Format time for display
    static const char* formatTime(uint64_t milliseconds, char* buffer, size_t bufferSize);
};

} // namespace snnfw

#endif // SNNFW_PLAYBACK_CONTROLS_H


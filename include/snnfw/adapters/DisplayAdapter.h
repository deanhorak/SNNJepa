#ifndef SNNFW_DISPLAY_ADAPTER_H
#define SNNFW_DISPLAY_ADAPTER_H

#include "snnfw/adapters/MotorAdapter.h"
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

namespace snnfw {
namespace adapters {

/**
 * @brief Display adapter for visualizing neural activity
 *
 * The DisplayAdapter converts spike patterns from motor neurons into
 * visual representations. It can display:
 * - Spike raster plots
 * - Firing rate heatmaps
 * - Population activity vectors
 * - ASCII art visualizations
 *
 * Configuration Parameters:
 * - display_width: Width of display in characters
 * - display_height: Height of display in characters
 * - update_interval: Minimum time between updates (ms)
 * - display_mode: "raster", "heatmap", "vector", or "ascii"
 *
 * Usage:
 * @code
 * BaseAdapter::Config config;
 * config.type = "display";
 * config.name = "activity_monitor";
 * config.setIntParam("display_width", 80);
 * config.setIntParam("display_height", 24);
 * 
 * DisplayAdapter display(config);
 * display.initialize();
 * 
 * // Process motor neuron activity
 * display.processNeurons(motorNeurons, currentTime);
 * @endcode
 */
class DisplayAdapter : public MotorAdapter {
public:
    /**
     * @brief Display mode enumeration
     */
    enum class DisplayMode {
        RASTER,    ///< Spike raster plot
        HEATMAP,   ///< Firing rate heatmap
        VECTOR,    ///< Population vector
        ASCII      ///< ASCII art representation
    };

    /**
     * @brief Constructor
     * @param config Adapter configuration
     */
    explicit DisplayAdapter(const Config& config);

    /**
     * @brief Destructor
     */
    ~DisplayAdapter() override = default;

    /**
     * @brief Initialize the adapter
     */
    bool initialize() override;

    /**
     * @brief Decode spike activity into display command
     * @param activity Spike activity from neurons
     * @return Display command (contains ASCII representation)
     */
    MotorCommand decodeActivity(const SpikeActivity& activity) override;

    /**
     * @brief Execute display command (print to console)
     * @param command Display command to execute
     * @return true if successful
     */
    bool executeCommand(const MotorCommand& command) override;

    /**
     * @brief Process motor neurons and update display
     * @param neurons Motor neuron population
     * @param currentTime Current simulation time (ms)
     * @return true if display was updated
     */
    bool processNeurons(const std::vector<std::shared_ptr<Neuron>>& neurons,
                        double currentTime) override;

    /**
     * @brief Get number of display channels
     */
    size_t getChannelCount() const override {
        return displayWidth_ * displayHeight_;
    }

    /**
     * @brief Get current display command
     */
    MotorCommand getCurrentCommand() const override {
        return currentCommand_;
    }

    /**
     * @brief Set display command
     */
    void setCommand(const MotorCommand& command) override {
        currentCommand_ = command;
    }

    /**
     * @brief Set display mode
     */
    void setDisplayMode(DisplayMode mode) {
        displayMode_ = mode;
    }

    /**
     * @brief Get display buffer as string
     */
    std::string getDisplayBuffer() const {
        return displayBuffer_;
    }

private:
    int displayWidth_;          ///< Display width in characters
    int displayHeight_;         ///< Display height in characters
    double updateInterval_;     ///< Minimum update interval (ms)
    DisplayMode displayMode_;   ///< Current display mode
    
    double lastUpdateTime_;     ///< Last update time (ms)
    MotorCommand currentCommand_; ///< Current display command
    std::string displayBuffer_; ///< Current display buffer
    
    /**
     * @brief Generate raster plot
     */
    std::string generateRasterPlot(const SpikeActivity& activity);
    
    /**
     * @brief Generate heatmap
     */
    std::string generateHeatmap(const SpikeActivity& activity);
    
    /**
     * @brief Generate vector display
     */
    std::string generateVectorDisplay(const SpikeActivity& activity);
    
    /**
     * @brief Generate ASCII art
     */
    std::string generateASCII(const SpikeActivity& activity);
    
    /**
     * @brief Extract spike activity from neurons
     */
    SpikeActivity extractActivity(const std::vector<std::shared_ptr<Neuron>>& neurons,
                                   double currentTime);
};

} // namespace adapters
} // namespace snnfw

#endif // SNNFW_DISPLAY_ADAPTER_H


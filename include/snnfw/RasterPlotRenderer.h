#ifndef SNNFW_RASTER_PLOT_RENDERER_H
#define SNNFW_RASTER_PLOT_RENDERER_H

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>
#include <deque>

namespace snnfw {

// Forward declarations
class ShaderManager;
class ActivityVisualizer;

/**
 * @brief Configuration for raster plot rendering
 */
struct RasterPlotConfig {
    // Display settings
    float timeWindowMs = 1000.0f;      ///< Time window to display (ms)
    float neuronHeight = 2.0f;         ///< Height of each neuron row (pixels)
    float spikeMarkerSize = 3.0f;      ///< Size of spike markers (pixels)
    
    // Colors
    glm::vec4 backgroundColor = glm::vec4(0.1f, 0.1f, 0.15f, 1.0f);
    glm::vec4 gridColor = glm::vec4(0.3f, 0.3f, 0.35f, 1.0f);
    glm::vec4 spikeColor = glm::vec4(1.0f, 0.8f, 0.2f, 1.0f);
    glm::vec4 textColor = glm::vec4(0.9f, 0.9f, 0.9f, 1.0f);
    
    // Grid settings
    bool showGrid = true;
    float gridTimeInterval = 100.0f;   ///< Time interval for vertical grid lines (ms)
    bool showNeuronLabels = true;
    bool showTimeLabels = true;
    
    // Scrolling
    bool autoScroll = true;            ///< Auto-scroll to show latest spikes
    float scrollSpeed = 1.0f;          ///< Scroll speed multiplier
    
    // Filtering
    bool colorByNeuronType = true;     ///< Color spikes by neuron type (excitatory/inhibitory)
    glm::vec4 excitatoryColor = glm::vec4(1.0f, 0.3f, 0.3f, 1.0f);
    glm::vec4 inhibitoryColor = glm::vec4(0.3f, 0.3f, 1.0f, 1.0f);
};

/**
 * @brief Spike event for raster plot
 */
struct RasterSpike {
    uint64_t neuronId;
    uint64_t timestamp;
    bool isExcitatory;
    
    RasterSpike(uint64_t nid, uint64_t ts, bool excitatory = true)
        : neuronId(nid), timestamp(ts), isExcitatory(excitatory) {}
};

/**
 * @brief Renders spike raster plots showing neuron firing patterns over time
 * 
 * A raster plot displays spikes as dots/markers on a 2D grid where:
 * - X-axis represents time
 * - Y-axis represents neuron index
 * - Each spike is shown as a marker at (time, neuron_index)
 * 
 * Features:
 * - Time-based scrolling window
 * - Grid lines for time reference
 * - Neuron labels
 * - Color coding by neuron type
 * - Auto-scroll to show latest activity
 */
class RasterPlotRenderer {
public:
    /**
     * @brief Constructor
     * @param shaderManager Shader manager for loading shaders
     */
    explicit RasterPlotRenderer(ShaderManager& shaderManager);
    
    /**
     * @brief Destructor
     */
    ~RasterPlotRenderer();
    
    /**
     * @brief Initialize renderer (load shaders, create buffers)
     * @return true if successful
     */
    bool initialize();
    
    /**
     * @brief Cleanup resources
     */
    void cleanup();
    
    /**
     * @brief Add spike to raster plot
     * @param neuronId Neuron ID
     * @param timestamp Spike timestamp (ms)
     * @param isExcitatory Whether neuron is excitatory
     */
    void addSpike(uint64_t neuronId, uint64_t timestamp, bool isExcitatory = true);
    
    /**
     * @brief Clear all spikes
     */
    void clearSpikes();
    
    /**
     * @brief Render raster plot
     * @param x X position (screen coordinates)
     * @param y Y position (screen coordinates)
     * @param width Width of plot area
     * @param height Height of plot area
     * @param currentTime Current simulation time (ms)
     */
    void render(float x, float y, float width, float height, uint64_t currentTime);
    
    /**
     * @brief Set configuration
     * @param config New configuration
     */
    void setConfig(const RasterPlotConfig& config) { config_ = config; }
    
    /**
     * @brief Get configuration
     * @return Current configuration
     */
    const RasterPlotConfig& getConfig() const { return config_; }
    
    /**
     * @brief Set neuron mapping (neuron ID -> display index)
     * @param neuronIds Ordered list of neuron IDs to display
     */
    void setNeuronMapping(const std::vector<uint64_t>& neuronIds);
    
    /**
     * @brief Get number of neurons being displayed
     * @return Neuron count
     */
    size_t getNeuronCount() const { return neuronMapping_.size(); }
    
    /**
     * @brief Get total spike count
     * @return Total spikes in buffer
     */
    size_t getSpikeCount() const { return spikes_.size(); }
    
    /**
     * @brief Set time offset for manual scrolling
     * @param offset Time offset in ms
     */
    void setTimeOffset(float offset) { timeOffset_ = offset; }
    
    /**
     * @brief Get current time offset
     * @return Time offset in ms
     */
    float getTimeOffset() const { return timeOffset_; }

private:
    /**
     * @brief Update spike buffer (remove old spikes outside time window)
     * @param currentTime Current simulation time
     */
    void updateSpikeBuffer(uint64_t currentTime);
    
    /**
     * @brief Render background and grid
     */
    void renderBackground(float x, float y, float width, float height, uint64_t currentTime);
    
    /**
     * @brief Render spike markers
     */
    void renderSpikes(float x, float y, float width, float height, uint64_t currentTime);
    
    /**
     * @brief Render labels (time and neuron)
     */
    void renderLabels(float x, float y, float width, float height, uint64_t currentTime);
    
    /**
     * @brief Get neuron display index from ID
     * @param neuronId Neuron ID
     * @return Display index, or -1 if not found
     */
    int getNeuronIndex(uint64_t neuronId) const;
    
    /**
     * @brief Setup orthographic projection for 2D rendering
     */
    void setupOrthographicProjection(float width, float height);

private:
    ShaderManager& shaderManager_;
    RasterPlotConfig config_;
    
    // Spike data
    std::deque<RasterSpike> spikes_;
    std::vector<uint64_t> neuronMapping_;  // neuronId -> display index
    
    // Scrolling
    float timeOffset_;
    
    // OpenGL resources
    unsigned int spikeVAO_;
    unsigned int spikeVBO_;
    unsigned int gridVAO_;
    unsigned int gridVBO_;
    unsigned int quadVAO_;
    unsigned int quadVBO_;
    
    bool initialized_;
};

} // namespace snnfw

#endif // SNNFW_RASTER_PLOT_RENDERER_H


#ifndef SNNFW_ACTIVITY_HISTOGRAM_H
#define SNNFW_ACTIVITY_HISTOGRAM_H

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>
#include <unordered_map>

namespace snnfw {

// Forward declarations
class ShaderManager;
class ActivityVisualizer;
class NetworkDataAdapter;

/**
 * @brief Configuration for histogram rendering
 */
struct HistogramConfig {
    // Display settings
    int numBins = 20;                      ///< Number of histogram bins
    float barWidth = 0.8f;                 ///< Width of bars relative to bin width
    float maxBarHeight = 200.0f;           ///< Maximum bar height in pixels
    
    // Colors
    glm::vec4 backgroundColor = glm::vec4(0.1f, 0.1f, 0.15f, 1.0f);
    glm::vec4 barColor = glm::vec4(0.3f, 0.7f, 1.0f, 1.0f);
    glm::vec4 gridColor = glm::vec4(0.3f, 0.3f, 0.35f, 1.0f);
    glm::vec4 textColor = glm::vec4(0.9f, 0.9f, 0.9f, 1.0f);
    
    // Grid settings
    bool showGrid = true;
    bool showLabels = true;
    bool showMean = true;
    bool showMedian = true;
    glm::vec4 meanColor = glm::vec4(1.0f, 0.3f, 0.3f, 1.0f);
    glm::vec4 medianColor = glm::vec4(0.3f, 1.0f, 0.3f, 1.0f);
    
    // Data settings
    float timeWindowMs = 1000.0f;          ///< Time window for analysis (ms)
    bool autoScale = true;                 ///< Auto-scale Y-axis
    float minFrequency = 0.0f;             ///< Minimum frequency (Hz)
    float maxFrequency = 100.0f;           ///< Maximum frequency (Hz)
    
    // Update settings
    float updateInterval = 100.0f;         ///< Update interval (ms)
};

/**
 * @brief Histogram type
 */
enum class HistogramType {
    FIRING_RATE,           ///< Firing rate distribution (Hz)
    SPIKE_COUNT,           ///< Spike count distribution
    ISI,                   ///< Inter-spike interval distribution (ms)
    ACTIVITY_LEVEL         ///< Activity level distribution (0-1)
};

/**
 * @brief Histogram bin data
 */
struct HistogramBin {
    float minValue;        ///< Minimum value in bin
    float maxValue;        ///< Maximum value in bin
    int count;             ///< Number of items in bin
    float frequency;       ///< Normalized frequency (0-1)
    
    HistogramBin() : minValue(0.0f), maxValue(0.0f), count(0), frequency(0.0f) {}
};

/**
 * @brief Statistics for histogram data
 */
struct HistogramStats {
    float mean;
    float median;
    float stddev;
    float min;
    float max;
    int totalCount;
    
    HistogramStats() : mean(0.0f), median(0.0f), stddev(0.0f), 
                       min(0.0f), max(0.0f), totalCount(0) {}
};

/**
 * @brief Renders activity histograms showing frequency distributions
 * 
 * Visualizes various activity metrics as histograms:
 * - Firing rate distribution (Hz)
 * - Spike count distribution
 * - Inter-spike interval distribution
 * - Activity level distribution
 * 
 * Features:
 * - Configurable number of bins
 * - Auto-scaling
 * - Mean and median indicators
 * - Grid lines and labels
 * - Real-time updates
 */
class ActivityHistogram {
public:
    /**
     * @brief Constructor
     * @param shaderManager Shader manager for loading shaders
     */
    explicit ActivityHistogram(ShaderManager& shaderManager);
    
    /**
     * @brief Destructor
     */
    ~ActivityHistogram();
    
    /**
     * @brief Initialize OpenGL resources
     */
    void initialize();
    
    /**
     * @brief Cleanup OpenGL resources
     */
    void cleanup();
    
    /**
     * @brief Update histogram data
     * @param activityVisualizer Activity visualizer with spike data
     * @param adapter Network data adapter
     * @param currentTime Current simulation time (ms)
     */
    void update(const ActivityVisualizer& activityVisualizer,
                const NetworkDataAdapter& adapter,
                uint64_t currentTime);
    
    /**
     * @brief Render histogram
     * @param x X position (pixels)
     * @param y Y position (pixels)
     * @param width Width (pixels)
     * @param height Height (pixels)
     * @param screenWidth Screen width (pixels)
     * @param screenHeight Screen height (pixels)
     */
    void render(float x, float y, float width, float height,
                int screenWidth, int screenHeight);
    
    /**
     * @brief Set histogram type
     */
    void setHistogramType(HistogramType type) { histogramType_ = type; }
    
    /**
     * @brief Get histogram type
     */
    HistogramType getHistogramType() const { return histogramType_; }
    
    /**
     * @brief Set configuration
     */
    void setConfig(const HistogramConfig& config) { config_ = config; }
    
    /**
     * @brief Get configuration
     */
    const HistogramConfig& getConfig() const { return config_; }
    
    /**
     * @brief Get histogram bins
     */
    const std::vector<HistogramBin>& getBins() const { return bins_; }
    
    /**
     * @brief Get statistics
     */
    const HistogramStats& getStats() const { return stats_; }
    
    /**
     * @brief Clear histogram data
     */
    void clear();

private:
    // Helper methods
    void computeHistogram(const std::vector<float>& data);
    void computeStats(const std::vector<float>& data);
    void updateBuffers();
    void renderBackground(float x, float y, float width, float height,
                         int screenWidth, int screenHeight);
    void renderBars(float x, float y, float width, float height,
                   int screenWidth, int screenHeight);
    void renderGrid(float x, float y, float width, float height,
                   int screenWidth, int screenHeight);
    void renderIndicators(float x, float y, float width, float height,
                         int screenWidth, int screenHeight);
    
    // Extract data based on histogram type
    std::vector<float> extractFiringRates(const ActivityVisualizer& activityVisualizer,
                                          const NetworkDataAdapter& adapter,
                                          uint64_t currentTime);
    std::vector<float> extractSpikeCounts(const ActivityVisualizer& activityVisualizer,
                                          const NetworkDataAdapter& adapter,
                                          uint64_t currentTime);
    std::vector<float> extractISIs(const ActivityVisualizer& activityVisualizer,
                                   const NetworkDataAdapter& adapter,
                                   uint64_t currentTime);
    std::vector<float> extractActivityLevels(const ActivityVisualizer& activityVisualizer,
                                             const NetworkDataAdapter& adapter);

private:
    ShaderManager& shaderManager_;
    HistogramConfig config_;
    HistogramType histogramType_;
    
    // Histogram data
    std::vector<HistogramBin> bins_;
    HistogramStats stats_;
    
    // Timing
    uint64_t lastUpdateTime_;
    
    // OpenGL resources
    unsigned int vao_;
    unsigned int vbo_;
    unsigned int ebo_;
    bool initialized_;
    
    // Vertex data for bars
    std::vector<float> vertices_;
    std::vector<unsigned int> indices_;
};

} // namespace snnfw

#endif // SNNFW_ACTIVITY_HISTOGRAM_H


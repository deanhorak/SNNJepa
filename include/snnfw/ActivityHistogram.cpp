#include "snnfw/ActivityHistogram.h"
#include "snnfw/ShaderManager.h"
#include "snnfw/ActivityVisualizer.h"
#include "snnfw/NetworkDataAdapter.h"
#include <glad/glad.h>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace snnfw {

ActivityHistogram::ActivityHistogram(ShaderManager& shaderManager)
    : shaderManager_(shaderManager)
    , histogramType_(HistogramType::FIRING_RATE)
    , lastUpdateTime_(0)
    , vao_(0)
    , vbo_(0)
    , ebo_(0)
    , initialized_(false)
{
    bins_.resize(config_.numBins);
}

ActivityHistogram::~ActivityHistogram() {
    cleanup();
}

void ActivityHistogram::initialize() {
    if (initialized_) return;
    
    // Create OpenGL resources
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    glGenBuffers(1, &ebo_);
    
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    
    // Position attribute (2D)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glBindVertexArray(0);
    
    initialized_ = true;
}

void ActivityHistogram::cleanup() {
    if (!initialized_) return;
    
    if (vao_ != 0) {
        glDeleteVertexArrays(1, &vao_);
        vao_ = 0;
    }
    if (vbo_ != 0) {
        glDeleteBuffers(1, &vbo_);
        vbo_ = 0;
    }
    if (ebo_ != 0) {
        glDeleteBuffers(1, &ebo_);
        ebo_ = 0;
    }
    
    initialized_ = false;
}

void ActivityHistogram::update(const ActivityVisualizer& activityVisualizer,
                               const NetworkDataAdapter& adapter,
                               uint64_t currentTime) {
    // Check if we should update
    if (currentTime - lastUpdateTime_ < static_cast<uint64_t>(config_.updateInterval)) {
        return;
    }
    
    lastUpdateTime_ = currentTime;
    
    // Extract data based on histogram type
    std::vector<float> data;
    switch (histogramType_) {
        case HistogramType::FIRING_RATE:
            data = extractFiringRates(activityVisualizer, adapter, currentTime);
            break;
        case HistogramType::SPIKE_COUNT:
            data = extractSpikeCounts(activityVisualizer, adapter, currentTime);
            break;
        case HistogramType::ISI:
            data = extractISIs(activityVisualizer, adapter, currentTime);
            break;
        case HistogramType::ACTIVITY_LEVEL:
            data = extractActivityLevels(activityVisualizer, adapter);
            break;
    }
    
    // Compute histogram and statistics
    if (!data.empty()) {
        computeHistogram(data);
        computeStats(data);
        updateBuffers();
    }
}

void ActivityHistogram::render(float x, float y, float width, float height,
                               int screenWidth, int screenHeight) {
    if (!initialized_) return;
    
    // Save OpenGL state
    GLboolean depthTestEnabled;
    glGetBooleanv(GL_DEPTH_TEST, &depthTestEnabled);
    glDisable(GL_DEPTH_TEST);
    
    // Render components
    renderBackground(x, y, width, height, screenWidth, screenHeight);
    if (config_.showGrid) {
        renderGrid(x, y, width, height, screenWidth, screenHeight);
    }
    renderBars(x, y, width, height, screenWidth, screenHeight);
    if (config_.showMean || config_.showMedian) {
        renderIndicators(x, y, width, height, screenWidth, screenHeight);
    }
    
    // Restore OpenGL state
    if (depthTestEnabled) {
        glEnable(GL_DEPTH_TEST);
    }
}

void ActivityHistogram::clear() {
    bins_.clear();
    bins_.resize(config_.numBins);
    stats_ = HistogramStats();
    vertices_.clear();
    indices_.clear();
}

void ActivityHistogram::computeHistogram(const std::vector<float>& data) {
    if (data.empty()) {
        clear();
        return;
    }
    
    // Find min and max
    float minVal = *std::min_element(data.begin(), data.end());
    float maxVal = *std::max_element(data.begin(), data.end());
    
    // Use config range if not auto-scaling
    if (!config_.autoScale) {
        minVal = config_.minFrequency;
        maxVal = config_.maxFrequency;
    }
    
    // Avoid division by zero
    if (maxVal - minVal < 1e-6f) {
        maxVal = minVal + 1.0f;
    }
    
    // Initialize bins
    bins_.clear();
    bins_.resize(config_.numBins);
    
    float binWidth = (maxVal - minVal) / config_.numBins;
    for (int i = 0; i < config_.numBins; ++i) {
        bins_[i].minValue = minVal + i * binWidth;
        bins_[i].maxValue = minVal + (i + 1) * binWidth;
        bins_[i].count = 0;
    }
    
    // Count data points in each bin
    for (float value : data) {
        if (value < minVal || value > maxVal) continue;
        
        int binIndex = static_cast<int>((value - minVal) / binWidth);
        binIndex = std::min(binIndex, config_.numBins - 1);
        binIndex = std::max(binIndex, 0);
        
        bins_[binIndex].count++;
    }
    
    // Normalize frequencies
    int maxCount = 0;
    for (const auto& bin : bins_) {
        maxCount = std::max(maxCount, bin.count);
    }
    
    if (maxCount > 0) {
        for (auto& bin : bins_) {
            bin.frequency = static_cast<float>(bin.count) / maxCount;
        }
    }
}

void ActivityHistogram::computeStats(const std::vector<float>& data) {
    if (data.empty()) {
        stats_ = HistogramStats();
        return;
    }
    
    // Mean
    float sum = std::accumulate(data.begin(), data.end(), 0.0f);
    stats_.mean = sum / data.size();
    
    // Min and max
    stats_.min = *std::min_element(data.begin(), data.end());
    stats_.max = *std::max_element(data.begin(), data.end());
    
    // Median
    std::vector<float> sortedData = data;
    std::sort(sortedData.begin(), sortedData.end());
    size_t mid = sortedData.size() / 2;
    if (sortedData.size() % 2 == 0) {
        stats_.median = (sortedData[mid - 1] + sortedData[mid]) / 2.0f;
    } else {
        stats_.median = sortedData[mid];
    }
    
    // Standard deviation
    float variance = 0.0f;
    for (float value : data) {
        float diff = value - stats_.mean;
        variance += diff * diff;
    }
    variance /= data.size();
    stats_.stddev = std::sqrt(variance);
    
    stats_.totalCount = static_cast<int>(data.size());
}

void ActivityHistogram::updateBuffers() {
    if (!initialized_ || bins_.empty()) return;
    
    vertices_.clear();
    indices_.clear();
    
    // Create vertices for each bar (4 vertices per bar)
    for (size_t i = 0; i < bins_.size(); ++i) {
        float x = static_cast<float>(i) / bins_.size();
        float barWidth = config_.barWidth / bins_.size();
        float height = bins_[i].frequency;
        
        // Bottom-left
        vertices_.push_back(x);
        vertices_.push_back(0.0f);
        
        // Bottom-right
        vertices_.push_back(x + barWidth);
        vertices_.push_back(0.0f);
        
        // Top-right
        vertices_.push_back(x + barWidth);
        vertices_.push_back(height);
        
        // Top-left
        vertices_.push_back(x);
        vertices_.push_back(height);
        
        // Indices for two triangles
        unsigned int baseIdx = static_cast<unsigned int>(i * 4);
        indices_.push_back(baseIdx + 0);
        indices_.push_back(baseIdx + 1);
        indices_.push_back(baseIdx + 2);
        indices_.push_back(baseIdx + 0);
        indices_.push_back(baseIdx + 2);
        indices_.push_back(baseIdx + 3);
    }
    
    // Upload to GPU
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, vertices_.size() * sizeof(float),
                 vertices_.data(), GL_DYNAMIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_.size() * sizeof(unsigned int),
                 indices_.data(), GL_DYNAMIC_DRAW);
}

void ActivityHistogram::renderBackground(float x, float y, float width, float height,
                                         int screenWidth, int screenHeight) {
    // Simple background quad
    // This would use a shader to render a colored quad
    // For now, we'll skip this as it requires shader setup
}

void ActivityHistogram::renderBars(float x, float y, float width, float height,
                                   int screenWidth, int screenHeight) {
    if (vertices_.empty() || indices_.empty()) return;
    
    // This would use a shader to render the bars
    // The vertices are in normalized coordinates (0-1)
    // We need to transform them to screen coordinates
    
    glBindVertexArray(vao_);
    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices_.size()),
                   GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void ActivityHistogram::renderGrid(float x, float y, float width, float height,
                                   int screenWidth, int screenHeight) {
    // Render grid lines
    // This would use a line shader
}

void ActivityHistogram::renderIndicators(float x, float y, float width, float height,
                                        int screenWidth, int screenHeight) {
    // Render mean and median lines
    // This would use a line shader
}

std::vector<float> ActivityHistogram::extractFiringRates(
    const ActivityVisualizer& activityVisualizer,
    const NetworkDataAdapter& adapter,
    uint64_t currentTime) {

    std::vector<float> rates;
    const auto& neuronActivity = activityVisualizer.getNeuronActivity();

    // Calculate firing rate for each neuron
    for (const auto& activity : neuronActivity) {
        uint32_t spikeCount = activity.spikeCount;
        float timeWindowSec = config_.timeWindowMs / 1000.0f;
        float firingRate = spikeCount / timeWindowSec;  // Hz
        rates.push_back(firingRate);
    }

    return rates;
}

std::vector<float> ActivityHistogram::extractSpikeCounts(
    const ActivityVisualizer& activityVisualizer,
    const NetworkDataAdapter& adapter,
    uint64_t currentTime) {

    std::vector<float> counts;
    const auto& neuronActivity = activityVisualizer.getNeuronActivity();

    // Get spike count for each neuron
    for (const auto& activity : neuronActivity) {
        counts.push_back(static_cast<float>(activity.spikeCount));
    }

    return counts;
}

std::vector<float> ActivityHistogram::extractISIs(
    const ActivityVisualizer& activityVisualizer,
    const NetworkDataAdapter& adapter,
    uint64_t currentTime) {

    std::vector<float> isis;

    // We would need access to spike history to compute ISIs
    // For now, return empty vector
    // This would require extending ActivityVisualizer to track spike times

    return isis;
}

std::vector<float> ActivityHistogram::extractActivityLevels(
    const ActivityVisualizer& activityVisualizer,
    const NetworkDataAdapter& adapter) {

    std::vector<float> levels;
    const auto& neuronActivity = activityVisualizer.getNeuronActivity();

    // Get activity level for each neuron
    for (const auto& activity : neuronActivity) {
        levels.push_back(activity.activityLevel);
    }

    return levels;
}

} // namespace snnfw


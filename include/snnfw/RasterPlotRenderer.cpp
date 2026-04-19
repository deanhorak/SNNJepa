#include "snnfw/RasterPlotRenderer.h"
#include "snnfw/ShaderManager.h"
#include <glad/glad.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <algorithm>
#include <unordered_map>

namespace snnfw {

RasterPlotRenderer::RasterPlotRenderer(ShaderManager& shaderManager)
    : shaderManager_(shaderManager)
    , timeOffset_(0.0f)
    , spikeVAO_(0)
    , spikeVBO_(0)
    , gridVAO_(0)
    , gridVBO_(0)
    , quadVAO_(0)
    , quadVBO_(0)
    , initialized_(false) {
}

RasterPlotRenderer::~RasterPlotRenderer() {
    cleanup();
}

bool RasterPlotRenderer::initialize() {
    if (initialized_) return true;
    
    // Create spike marker VAO/VBO (point sprites)
    glGenVertexArrays(1, &spikeVAO_);
    glGenBuffers(1, &spikeVBO_);
    
    glBindVertexArray(spikeVAO_);
    glBindBuffer(GL_ARRAY_BUFFER, spikeVBO_);
    
    // Position (x, y) + Color (r, g, b, a)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
    
    // Create grid line VAO/VBO
    glGenVertexArrays(1, &gridVAO_);
    glGenBuffers(1, &gridVBO_);
    
    glBindVertexArray(gridVAO_);
    glBindBuffer(GL_ARRAY_BUFFER, gridVBO_);
    
    // Position (x, y)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glBindVertexArray(0);
    
    // Create quad VAO/VBO for background
    glGenVertexArrays(1, &quadVAO_);
    glGenBuffers(1, &quadVBO_);
    
    glBindVertexArray(quadVAO_);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO_);
    
    // Position (x, y)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glBindVertexArray(0);
    
    initialized_ = true;
    return true;
}

void RasterPlotRenderer::cleanup() {
    if (!initialized_) return;
    
    if (spikeVAO_) glDeleteVertexArrays(1, &spikeVAO_);
    if (spikeVBO_) glDeleteBuffers(1, &spikeVBO_);
    if (gridVAO_) glDeleteVertexArrays(1, &gridVAO_);
    if (gridVBO_) glDeleteBuffers(1, &gridVBO_);
    if (quadVAO_) glDeleteVertexArrays(1, &quadVAO_);
    if (quadVBO_) glDeleteBuffers(1, &quadVBO_);
    
    spikeVAO_ = spikeVBO_ = gridVAO_ = gridVBO_ = quadVAO_ = quadVBO_ = 0;
    initialized_ = false;
}

void RasterPlotRenderer::addSpike(uint64_t neuronId, uint64_t timestamp, bool isExcitatory) {
    spikes_.emplace_back(neuronId, timestamp, isExcitatory);
}

void RasterPlotRenderer::clearSpikes() {
    spikes_.clear();
}

void RasterPlotRenderer::setNeuronMapping(const std::vector<uint64_t>& neuronIds) {
    neuronMapping_ = neuronIds;
}

int RasterPlotRenderer::getNeuronIndex(uint64_t neuronId) const {
    for (size_t i = 0; i < neuronMapping_.size(); ++i) {
        if (neuronMapping_[i] == neuronId) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

void RasterPlotRenderer::updateSpikeBuffer(uint64_t currentTime) {
    // Remove spikes outside the time window
    uint64_t minTime = currentTime > static_cast<uint64_t>(config_.timeWindowMs) 
        ? currentTime - static_cast<uint64_t>(config_.timeWindowMs) 
        : 0;
    
    while (!spikes_.empty() && spikes_.front().timestamp < minTime) {
        spikes_.pop_front();
    }
}

void RasterPlotRenderer::setupOrthographicProjection(float width, float height) {
    glm::mat4 projection = glm::ortho(0.0f, width, height, 0.0f, -1.0f, 1.0f);
    
    // Use basic shader for 2D rendering
    unsigned int shader = shaderManager_.getShader("basic");
    if (shader) {
        glUseProgram(shader);
        glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, 
                          glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, 
                          glm::value_ptr(glm::mat4(1.0f)));
        glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, 
                          glm::value_ptr(glm::mat4(1.0f)));
    }
}

void RasterPlotRenderer::renderBackground(float x, float y, float width, float height, 
                                         uint64_t currentTime) {
    // Render background quad
    float quadVertices[] = {
        x, y,
        x + width, y,
        x + width, y + height,
        x, y,
        x + width, y + height,
        x, y + height
    };
    
    glBindVertexArray(quadVAO_);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_DYNAMIC_DRAW);
    
    unsigned int shader = shaderManager_.getShader("basic");
    if (shader) {
        glUseProgram(shader);
        glUniform4fv(glGetUniformLocation(shader, "color"), 1, 
                    glm::value_ptr(config_.backgroundColor));
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }
    
    // Render grid if enabled
    if (config_.showGrid && neuronMapping_.size() > 0) {
        std::vector<float> gridLines;
        
        // Calculate time range
        uint64_t endTime = currentTime;
        uint64_t startTime = endTime > static_cast<uint64_t>(config_.timeWindowMs)
            ? endTime - static_cast<uint64_t>(config_.timeWindowMs)
            : 0;
        
        // Vertical grid lines (time intervals)
        float timeRange = config_.timeWindowMs;
        for (float t = 0; t <= timeRange; t += config_.gridTimeInterval) {
            float xPos = x + (t / timeRange) * width;
            gridLines.push_back(xPos);
            gridLines.push_back(y);
            gridLines.push_back(xPos);
            gridLines.push_back(y + height);
        }
        
        // Horizontal grid lines (neurons)
        float neuronSpacing = height / static_cast<float>(neuronMapping_.size());
        for (size_t i = 0; i <= neuronMapping_.size(); ++i) {
            float yPos = y + i * neuronSpacing;
            gridLines.push_back(x);
            gridLines.push_back(yPos);
            gridLines.push_back(x + width);
            gridLines.push_back(yPos);
        }
        
        if (!gridLines.empty()) {
            glBindVertexArray(gridVAO_);
            glBindBuffer(GL_ARRAY_BUFFER, gridVBO_);
            glBufferData(GL_ARRAY_BUFFER, gridLines.size() * sizeof(float), 
                        gridLines.data(), GL_DYNAMIC_DRAW);
            
            if (shader) {
                glUseProgram(shader);
                glUniform4fv(glGetUniformLocation(shader, "color"), 1, 
                            glm::value_ptr(config_.gridColor));
                glDrawArrays(GL_LINES, 0, gridLines.size() / 2);
            }
        }
    }
    
    glBindVertexArray(0);
}

void RasterPlotRenderer::renderSpikes(float x, float y, float width, float height, 
                                     uint64_t currentTime) {
    if (spikes_.empty() || neuronMapping_.empty()) return;
    
    // Calculate time range
    uint64_t endTime = currentTime;
    uint64_t startTime = endTime > static_cast<uint64_t>(config_.timeWindowMs)
        ? endTime - static_cast<uint64_t>(config_.timeWindowMs)
        : 0;
    float timeRange = config_.timeWindowMs;
    
    // Build spike vertex data
    std::vector<float> spikeVertices;
    float neuronSpacing = height / static_cast<float>(neuronMapping_.size());
    
    for (const auto& spike : spikes_) {
        // Skip spikes outside time window
        if (spike.timestamp < startTime || spike.timestamp > endTime) continue;
        
        // Get neuron index
        int neuronIdx = getNeuronIndex(spike.neuronId);
        if (neuronIdx < 0) continue;
        
        // Calculate position
        float relativeTime = static_cast<float>(spike.timestamp - startTime);
        float xPos = x + (relativeTime / timeRange) * width;
        float yPos = y + (neuronIdx + 0.5f) * neuronSpacing;
        
        // Choose color
        glm::vec4 color = config_.spikeColor;
        if (config_.colorByNeuronType) {
            color = spike.isExcitatory ? config_.excitatoryColor : config_.inhibitoryColor;
        }
        
        // Add vertex (position + color)
        spikeVertices.push_back(xPos);
        spikeVertices.push_back(yPos);
        spikeVertices.push_back(color.r);
        spikeVertices.push_back(color.g);
        spikeVertices.push_back(color.b);
        spikeVertices.push_back(color.a);
    }
    
    if (spikeVertices.empty()) return;
    
    // Upload and render
    glBindVertexArray(spikeVAO_);
    glBindBuffer(GL_ARRAY_BUFFER, spikeVBO_);
    glBufferData(GL_ARRAY_BUFFER, spikeVertices.size() * sizeof(float), 
                spikeVertices.data(), GL_DYNAMIC_DRAW);
    
    unsigned int shader = shaderManager_.getShader("basic");
    if (shader) {
        glUseProgram(shader);
        glPointSize(config_.spikeMarkerSize);
        glDrawArrays(GL_POINTS, 0, spikeVertices.size() / 6);
    }
    
    glBindVertexArray(0);
}

void RasterPlotRenderer::renderLabels(float x, float y, float width, float height, 
                                     uint64_t currentTime) {
    // Note: Text rendering would require a text rendering system (e.g., FreeType)
    // For now, this is a placeholder. In a full implementation, you would:
    // 1. Render time labels along the bottom
    // 2. Render neuron IDs along the left side
    // This could be done with ImGui or a dedicated text rendering system
}

void RasterPlotRenderer::render(float x, float y, float width, float height, 
                               uint64_t currentTime) {
    if (!initialized_) return;
    
    // Update spike buffer (remove old spikes)
    updateSpikeBuffer(currentTime);
    
    // Setup 2D projection
    setupOrthographicProjection(width, height);
    
    // Save OpenGL state
    GLboolean depthTestEnabled;
    glGetBooleanv(GL_DEPTH_TEST, &depthTestEnabled);
    glDisable(GL_DEPTH_TEST);
    
    // Render components
    renderBackground(x, y, width, height, currentTime);
    renderSpikes(x, y, width, height, currentTime);
    renderLabels(x, y, width, height, currentTime);
    
    // Restore OpenGL state
    if (depthTestEnabled) glEnable(GL_DEPTH_TEST);
}

} // namespace snnfw


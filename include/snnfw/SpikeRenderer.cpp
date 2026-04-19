#include "snnfw/SpikeRenderer.h"
#include "snnfw/ShaderManager.h"
#include "snnfw/Camera.h"
#include "snnfw/NetworkDataAdapter.h"
#include "snnfw/ActivityVisualizer.h"
#include <glad/glad.h>
#include <algorithm>
#include <spdlog/spdlog.h>

namespace snnfw {

SpikeRenderer::SpikeRenderer(ShaderManager& shaderManager)
    : shaderManager_(shaderManager)
    , particleVAO_(0)
    , particleVBO_(0)
    , particleInstanceVBO_(0)
    , trailVAO_(0)
    , trailVBO_(0)
    , particleShader_(0)
    , trailShader_(0)
    , initialized_(false)
{
    // Initialize activity color gradient (blue -> green -> yellow -> red)
    activityGradient_ = {
        glm::vec4(0.0f, 0.0f, 0.5f, 1.0f),  // Dark blue (no activity)
        glm::vec4(0.0f, 0.5f, 1.0f, 1.0f),  // Light blue
        glm::vec4(0.0f, 1.0f, 0.5f, 1.0f),  // Green
        glm::vec4(1.0f, 1.0f, 0.0f, 1.0f),  // Yellow
        glm::vec4(1.0f, 0.5f, 0.0f, 1.0f),  // Orange
        glm::vec4(1.0f, 0.0f, 0.0f, 1.0f)   // Red (high activity)
    };
}

SpikeRenderer::~SpikeRenderer() {
    cleanup();
}

bool SpikeRenderer::initialize() {
    if (initialized_) {
        return true;
    }
    
    // Load shaders
    if (!shaderManager_.loadShader("spike_particle", 
                                   "shaders/spike_particle.vert",
                                   "shaders/spike_particle.frag")) {
        spdlog::error("Failed to load spike particle shader");
        return false;
    }
    
    if (!shaderManager_.loadShader("spike_trail",
                                   "shaders/spike_trail.vert",
                                   "shaders/spike_trail.frag")) {
        spdlog::error("Failed to load spike trail shader");
        return false;
    }
    
    particleShader_ = shaderManager_.getShader("spike_particle");
    trailShader_ = shaderManager_.getShader("spike_trail");
    
    // Create particle buffers
    // Billboard quad vertices (will be instanced)
    float quadVertices[] = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.5f,  0.5f, 0.0f,
        -0.5f,  0.5f, 0.0f
    };
    
    unsigned int quadIndices[] = {
        0, 1, 2,
        2, 3, 0
    };
    
    glGenVertexArrays(1, &particleVAO_);
    glGenBuffers(1, &particleVBO_);
    glGenBuffers(1, &particleInstanceVBO_);
    
    glBindVertexArray(particleVAO_);
    
    // Vertex positions
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    
    // Instance data (will be updated each frame)
    glBindBuffer(GL_ARRAY_BUFFER, particleInstanceVBO_);
    glBufferData(GL_ARRAY_BUFFER, 10000 * sizeof(SpikeParticle), nullptr, GL_DYNAMIC_DRAW);
    
    // Instance position (vec3)
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(SpikeParticle), 
                         (void*)offsetof(SpikeParticle, position));
    glVertexAttribDivisor(1, 1);
    
    // Instance color (vec4)
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(SpikeParticle),
                         (void*)offsetof(SpikeParticle, color));
    glVertexAttribDivisor(2, 1);
    
    // Instance size (float)
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(SpikeParticle),
                         (void*)offsetof(SpikeParticle, size));
    glVertexAttribDivisor(3, 1);
    
    // Instance lifetime/maxLifetime (vec2)
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, sizeof(SpikeParticle),
                         (void*)offsetof(SpikeParticle, lifetime));
    glVertexAttribDivisor(4, 1);
    
    glBindVertexArray(0);
    
    // Create trail buffers
    glGenVertexArrays(1, &trailVAO_);
    glGenBuffers(1, &trailVBO_);
    
    glBindVertexArray(trailVAO_);
    glBindBuffer(GL_ARRAY_BUFFER, trailVBO_);
    glBufferData(GL_ARRAY_BUFFER, 20000 * sizeof(glm::vec3), nullptr, GL_DYNAMIC_DRAW);
    
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    
    glBindVertexArray(0);
    
    initialized_ = true;
    spdlog::info("SpikeRenderer initialized");
    return true;
}

void SpikeRenderer::cleanup() {
    if (!initialized_) {
        return;
    }
    
    glDeleteVertexArrays(1, &particleVAO_);
    glDeleteBuffers(1, &particleVBO_);
    glDeleteBuffers(1, &particleInstanceVBO_);
    glDeleteVertexArrays(1, &trailVAO_);
    glDeleteBuffers(1, &trailVBO_);
    
    initialized_ = false;
    spdlog::info("SpikeRenderer cleaned up");
}

void SpikeRenderer::renderSpikeParticles(
    const std::vector<SpikeParticle>& particles,
    const Camera& camera,
    const SpikeRenderConfig& config)
{
    if (!initialized_ || particles.empty()) {
        return;
    }
    
    resetStats();
    
    // Enable blending for particles
    if (config.enableBlending) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE); // Additive blending for glow
    }
    
    if (config.enableDepthTest) {
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE); // Don't write to depth buffer
    }
    
    // Update particle buffer
    updateParticleBuffer(particles);
    
    // Use particle shader
    shaderManager_.useShader("spike_particle");
    
    // Set uniforms
    shaderManager_.setUniform("view", camera.getViewMatrix());
    shaderManager_.setUniform("projection", camera.getProjectionMatrix(800.0f / 600.0f));
    shaderManager_.setUniform("glowIntensity", config.glowIntensity);
    shaderManager_.setUniform("particleAlpha", config.particleAlpha);
    
    // Render particles
    glBindVertexArray(particleVAO_);
    glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 4, particles.size());
    glBindVertexArray(0);
    
    stats_.particlesRendered = particles.size();
    stats_.drawCalls = 1;
    
    // Restore state
    if (config.enableDepthTest) {
        glDepthMask(GL_TRUE);
    }
    if (config.enableBlending) {
        glDisable(GL_BLEND);
    }
}

void SpikeRenderer::renderSpikeTrails(
    const std::vector<SpikeParticle>& particles,
    const NetworkDataAdapter& adapter,
    const Camera& camera,
    const SpikeRenderConfig& config)
{
    if (!initialized_ || !config.showTrails || particles.empty()) {
        return;
    }
    
    // Enable blending for trails
    if (config.enableBlending) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }
    
    // Update trail buffer
    updateTrailBuffer(particles, adapter);
    
    // Use trail shader
    shaderManager_.useShader("spike_trail");
    
    // Set uniforms
    shaderManager_.setUniform("view", camera.getViewMatrix());
    shaderManager_.setUniform("projection", camera.getProjectionMatrix(800.0f / 600.0f));
    shaderManager_.setUniform("trailLength", config.trailLength);
    
    // Render trails (lines)
    glBindVertexArray(trailVAO_);
    
    // Count particles with trails
    uint32_t trailCount = 0;
    for (const auto& particle : particles) {
        if (particle.synapseId != 0) {
            trailCount++;
        }
    }
    
    if (trailCount > 0) {
        glDrawArrays(GL_LINES, 0, trailCount * 2);
        stats_.trailsRendered = trailCount;
        stats_.drawCalls++;
    }
    
    glBindVertexArray(0);
    
    // Restore state
    if (config.enableBlending) {
        glDisable(GL_BLEND);
    }
}

void SpikeRenderer::applyActivityHeatmap(
    const std::vector<NeuronActivity>& activity,
    NetworkDataAdapter& adapter)
{
    // Get neurons from adapter
    auto& neurons = const_cast<std::vector<NeuronVisualData>&>(adapter.getNeurons());

    // Update neuron colors based on activity
    for (const auto& act : activity) {
        // Find neuron in adapter
        for (auto& neuron : neurons) {
            if (neuron.id == act.neuronId) {
                // Get activity color
                glm::vec4 activityColor = getActivityColor(act.activityLevel);

                // Blend with base color
                glm::vec4 baseColor(neuron.r, neuron.g, neuron.b, neuron.a);
                glm::vec4 blendedColor = glm::mix(baseColor, activityColor, act.activityLevel * 0.7f);

                neuron.r = blendedColor.r;
                neuron.g = blendedColor.g;
                neuron.b = blendedColor.b;
                neuron.a = blendedColor.a;
                break;
            }
        }
    }
}

const SpikeRenderStats& SpikeRenderer::getStats() const {
    return stats_;
}

void SpikeRenderer::resetStats() {
    stats_ = SpikeRenderStats();
}

void SpikeRenderer::updateParticleBuffer(const std::vector<SpikeParticle>& particles) {
    glBindBuffer(GL_ARRAY_BUFFER, particleInstanceVBO_);
    glBufferSubData(GL_ARRAY_BUFFER, 0, 
                   particles.size() * sizeof(SpikeParticle),
                   particles.data());
}

void SpikeRenderer::updateTrailBuffer(const std::vector<SpikeParticle>& particles,
                                     const NetworkDataAdapter& adapter) {
    std::vector<glm::vec3> trailVertices;
    trailVertices.reserve(particles.size() * 2);
    
    const auto& neurons = adapter.getNeurons();
    
    for (const auto& particle : particles) {
        if (particle.synapseId == 0) {
            continue;
        }
        
        // Find source and target positions
        glm::vec3 sourcePos(0.0f);
        glm::vec3 targetPos(0.0f);
        bool foundSource = false;
        bool foundTarget = false;
        
        for (const auto& neuron : neurons) {
            if (neuron.id == particle.sourceNeuronId) {
                sourcePos = glm::vec3(neuron.position.x, neuron.position.y, neuron.position.z);
                foundSource = true;
            }
            if (neuron.id == particle.targetNeuronId) {
                targetPos = glm::vec3(neuron.position.x, neuron.position.y, neuron.position.z);
                foundTarget = true;
            }
            if (foundSource && foundTarget) break;
        }
        
        if (foundSource && foundTarget) {
            // Add line from source to current particle position
            trailVertices.push_back(sourcePos);
            trailVertices.push_back(particle.position);
        }
    }
    
    if (!trailVertices.empty()) {
        glBindBuffer(GL_ARRAY_BUFFER, trailVBO_);
        glBufferSubData(GL_ARRAY_BUFFER, 0,
                       trailVertices.size() * sizeof(glm::vec3),
                       trailVertices.data());
    }
}

glm::vec4 SpikeRenderer::getActivityColor(float activityLevel) const {
    if (activityLevel <= 0.0f) {
        return activityGradient_[0];
    }
    if (activityLevel >= 1.0f) {
        return activityGradient_.back();
    }
    
    // Interpolate between gradient colors
    float scaledLevel = activityLevel * (activityGradient_.size() - 1);
    size_t index = static_cast<size_t>(scaledLevel);
    float t = scaledLevel - index;
    
    if (index >= activityGradient_.size() - 1) {
        return activityGradient_.back();
    }
    
    return glm::mix(activityGradient_[index], activityGradient_[index + 1], t);
}

} // namespace snnfw


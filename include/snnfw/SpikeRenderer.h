#ifndef SNNFW_SPIKE_RENDERER_H
#define SNNFW_SPIKE_RENDERER_H

#include <glm/glm.hpp>
#include <vector>
#include <memory>

namespace snnfw {

// Forward declarations
class ShaderManager;
class Camera;
class NetworkDataAdapter;
struct SpikeParticle;
struct NeuronActivity;

/**
 * @brief Configuration for spike rendering
 */
struct SpikeRenderConfig {
    bool enableBlending = true;        // Enable alpha blending
    bool enableDepthTest = true;       // Enable depth testing
    bool sortParticles = true;         // Sort particles by depth
    float particleAlpha = 0.8f;        // Base particle alpha
    float glowIntensity = 1.5f;        // Glow effect intensity
    bool showTrails = true;            // Show spike trails
    float trailLength = 2.0f;          // Trail length multiplier
    glm::vec3 lightPosition = glm::vec3(10.0f, 10.0f, 10.0f);
};

/**
 * @brief Statistics for spike rendering
 */
struct SpikeRenderStats {
    uint32_t particlesRendered = 0;
    uint32_t particlesCulled = 0;
    uint32_t trailsRendered = 0;
    uint32_t drawCalls = 0;
    float renderTimeMs = 0.0f;
};

/**
 * @brief High-performance renderer for spike particles and propagation effects
 * 
 * This class provides GPU-accelerated rendering of:
 * - Spike particles as billboards (camera-facing quads)
 * - Spike trails showing propagation along synapses
 * - Activity-based neuron coloring (heatmap)
 * 
 * Features:
 * - Instanced rendering for particles (single draw call)
 * - Billboard generation in geometry shader
 * - Depth sorting for correct alpha blending
 * - Frustum and distance culling
 * - Additive blending for glow effects
 * 
 * Usage:
 * @code
 * SpikeRenderer renderer(shaderManager);
 * renderer.initialize();
 * 
 * // Each frame
 * SpikeRenderConfig config;
 * renderer.renderSpikeParticles(particles, camera, config);
 * renderer.renderSpikeTrails(particles, adapter, camera, config);
 * @endcode
 */
class SpikeRenderer {
public:
    /**
     * @brief Constructor
     * @param shaderManager Shader manager for loading shaders
     */
    explicit SpikeRenderer(ShaderManager& shaderManager);
    
    /**
     * @brief Destructor
     */
    ~SpikeRenderer();
    
    /**
     * @brief Initialize renderer (create buffers, load shaders)
     * @return true if successful
     */
    bool initialize();
    
    /**
     * @brief Cleanup resources
     */
    void cleanup();
    
    /**
     * @brief Render spike particles as billboards
     * @param particles Vector of spike particles
     * @param camera Camera for view/projection matrices
     * @param config Rendering configuration
     */
    void renderSpikeParticles(
        const std::vector<SpikeParticle>& particles,
        const Camera& camera,
        const SpikeRenderConfig& config);
    
    /**
     * @brief Render spike trails along synapses
     * @param particles Vector of spike particles
     * @param adapter Network data adapter for synapse positions
     * @param camera Camera for view/projection matrices
     * @param config Rendering configuration
     */
    void renderSpikeTrails(
        const std::vector<SpikeParticle>& particles,
        const NetworkDataAdapter& adapter,
        const Camera& camera,
        const SpikeRenderConfig& config);
    
    /**
     * @brief Apply activity heatmap to neuron colors
     * @param activity Vector of neuron activity levels
     * @param adapter Network data adapter to update neuron colors
     */
    void applyActivityHeatmap(
        const std::vector<NeuronActivity>& activity,
        NetworkDataAdapter& adapter);
    
    /**
     * @brief Get rendering statistics
     * @return Render stats from last frame
     */
    const SpikeRenderStats& getStats() const;
    
    /**
     * @brief Reset statistics
     */
    void resetStats();

private:
    /**
     * @brief Sort particles by distance from camera
     * @param particles Particles to sort
     * @param cameraPos Camera position
     */
    void sortParticlesByDepth(std::vector<SpikeParticle>& particles, 
                             const glm::vec3& cameraPos);
    
    /**
     * @brief Cull particles outside frustum
     * @param particles Particles to cull
     * @param camera Camera for frustum
     * @return Vector of visible particles
     */
    std::vector<SpikeParticle> cullParticles(
        const std::vector<SpikeParticle>& particles,
        const Camera& camera);
    
    /**
     * @brief Update particle instance buffer
     * @param particles Particles to upload
     */
    void updateParticleBuffer(const std::vector<SpikeParticle>& particles);
    
    /**
     * @brief Update trail vertex buffer
     * @param particles Particles with trails
     * @param adapter Network data for synapse positions
     */
    void updateTrailBuffer(const std::vector<SpikeParticle>& particles,
                          const NetworkDataAdapter& adapter);
    
    /**
     * @brief Interpolate color based on activity level
     * @param activityLevel Activity level (0-1)
     * @return Color for activity level
     */
    glm::vec4 getActivityColor(float activityLevel) const;

    // Shader manager reference
    ShaderManager& shaderManager_;
    
    // OpenGL buffers for particles
    unsigned int particleVAO_;
    unsigned int particleVBO_;
    unsigned int particleInstanceVBO_;
    
    // OpenGL buffers for trails
    unsigned int trailVAO_;
    unsigned int trailVBO_;
    
    // Shader programs
    unsigned int particleShader_;
    unsigned int trailShader_;
    
    // Statistics
    SpikeRenderStats stats_;
    
    // Activity color gradient
    std::vector<glm::vec4> activityGradient_;
    
    // Initialization flag
    bool initialized_;
};

} // namespace snnfw

#endif // SNNFW_SPIKE_RENDERER_H


#ifndef SNNFW_NETWORK_GRAPH_RENDERER_H
#define SNNFW_NETWORK_GRAPH_RENDERER_H

#include "snnfw/NetworkDataAdapter.h"
#include "snnfw/ShaderManager.h"
#include "snnfw/Camera.h"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <vector>
#include <memory>
#include <set>
#include <unordered_set>

namespace snnfw {

/**
 * @brief Rendering mode for network visualization
 */
enum class RenderMode {
    NEURONS_ONLY,           ///< Render only neurons
    SYNAPSES_ONLY,          ///< Render only synapses
    NEURONS_AND_SYNAPSES,   ///< Render both neurons and synapses
    HIERARCHICAL_GROUPS     ///< Render hierarchical group boundaries
};

/**
 * @brief Level of detail for rendering
 */
enum class LODLevel {
    FULL,                   ///< Full detail (spheres, curves)
    MEDIUM,                 ///< Medium detail (simplified geometry)
    LOW,                    ///< Low detail (points, lines)
    MINIMAL                 ///< Minimal detail (aggregated representation)
};

/**
 * @brief Rendering configuration
 */
struct RenderConfig {
    RenderMode mode;                ///< Rendering mode
    LODLevel lodLevel;              ///< Level of detail
    
    // Neuron rendering
    bool renderNeurons;             ///< Enable neuron rendering
    float neuronBaseRadius;         ///< Base radius for neurons
    float neuronActivityScale;      ///< Scale factor for activity-based sizing
    int neuronSphereSegments;       ///< Sphere segments (quality)
    bool neuronUseInstancing;       ///< Use instanced rendering
    
    // Synapse rendering
    bool renderSynapses;            ///< Enable synapse rendering
    float synapseBaseThickness;     ///< Base thickness for synapses
    float synapseWeightScale;       ///< Scale factor for weight-based thickness
    bool synapseCurved;             ///< Use curved lines (Bezier)
    int synapseCurveSegments;       ///< Segments for curved synapses
    bool synapseUseInstancing;      ///< Use instanced rendering
    
    // Hierarchical group rendering
    bool renderGroupBoundaries;     ///< Render bounding boxes for groups
    float groupBoundaryThickness;   ///< Thickness of boundary lines
    float groupBoundaryAlpha;       ///< Alpha for boundary rendering

    // Cluster rendering
    bool renderClusters;            ///< Render clusters as colored boxes
    bool clusterActivityColoring;   ///< Color clusters based on activity
    float clusterAlpha;             ///< Alpha for cluster rendering
    
    // Selection and highlighting
    bool enableSelection;           ///< Enable selection rendering
    float selectionGlowIntensity;   ///< Glow intensity for selected objects
    float selectionOutlineWidth;    ///< Outline width for selected objects
    glm::vec4 selectionColor;       ///< Color for selection highlight
    
    // Culling and optimization
    bool frustumCulling;            ///< Enable frustum culling
    bool distanceCulling;           ///< Enable distance-based culling
    float maxRenderDistance;        ///< Maximum render distance
    bool occlusionCulling;          ///< Enable occlusion culling (expensive)
    
    // LOD thresholds
    float lodFullDistance;          ///< Distance for full LOD
    float lodMediumDistance;        ///< Distance for medium LOD
    float lodLowDistance;           ///< Distance for low LOD
    
    // Lighting
    bool enableLighting;            ///< Enable Phong lighting
    glm::vec3 lightPosition;        ///< Light position
    glm::vec3 lightColor;           ///< Light color
    float ambientStrength;          ///< Ambient light strength
    float specularStrength;         ///< Specular light strength
    
    RenderConfig() :
        mode(RenderMode::NEURONS_AND_SYNAPSES),
        lodLevel(LODLevel::FULL),
        renderNeurons(true),
        neuronBaseRadius(0.5f),
        neuronActivityScale(0.5f),
        neuronSphereSegments(16),
        neuronUseInstancing(true),
        renderSynapses(true),
        synapseBaseThickness(0.1f),
        synapseWeightScale(0.5f),
        synapseCurved(false),
        synapseCurveSegments(10),
        synapseUseInstancing(false),
        renderGroupBoundaries(false),
        groupBoundaryThickness(1.0f),
        groupBoundaryAlpha(0.3f),
        renderClusters(true),
        clusterActivityColoring(true),
        clusterAlpha(0.3f),
        enableSelection(true),
        selectionGlowIntensity(1.5f),
        selectionOutlineWidth(2.0f),
        selectionColor(1.0f, 1.0f, 0.0f, 1.0f),
        frustumCulling(true),
        distanceCulling(true),
        maxRenderDistance(1000.0f),
        occlusionCulling(false),
        lodFullDistance(50.0f),
        lodMediumDistance(100.0f),
        lodLowDistance(200.0f),
        enableLighting(true),
        lightPosition(100.0f, 100.0f, 100.0f),
        lightColor(1.0f, 1.0f, 1.0f),
        ambientStrength(0.3f),
        specularStrength(0.5f) {}
};

/**
 * @brief Rendering statistics
 */
struct RenderStats {
    size_t neuronsRendered;         ///< Number of neurons rendered
    size_t synapsesRendered;        ///< Number of synapses rendered
    size_t neuronsCulled;           ///< Number of neurons culled
    size_t synapsesCulled;          ///< Number of synapses culled
    size_t drawCalls;               ///< Number of draw calls
    float renderTimeMs;             ///< Render time in milliseconds
    
    RenderStats() : neuronsRendered(0), synapsesRendered(0),
                    neuronsCulled(0), synapsesCulled(0),
                    drawCalls(0), renderTimeMs(0.0f) {}
};

/**
 * @class NetworkGraphRenderer
 * @brief High-performance renderer for neural network graphs
 * 
 * The NetworkGraphRenderer provides optimized rendering of neural networks
 * with support for large-scale visualization (millions of neurons/synapses).
 * 
 * Key Features:
 * - Instanced rendering for neurons (single draw call for all neurons)
 * - Batch rendering for synapses
 * - Multiple LOD levels for performance
 * - Frustum and distance culling
 * - Selection and highlighting
 * - Hierarchical group visualization
 * - Activity-based coloring and sizing
 * 
 * Performance Optimizations:
 * - GPU instancing reduces draw calls from N to 1
 * - Frustum culling skips off-screen objects
 * - Distance-based LOD reduces geometry complexity
 * - Batch rendering minimizes state changes
 * 
 * Usage:
 * ```cpp
 * NetworkGraphRenderer renderer(shaderManager);
 * renderer.initialize();
 * 
 * RenderConfig config;
 * config.neuronUseInstancing = true;
 * config.frustumCulling = true;
 * 
 * // In render loop:
 * renderer.render(adapter, camera, config);
 * 
 * auto stats = renderer.getStats();
 * std::cout << "Rendered " << stats.neuronsRendered << " neurons\n";
 * ```
 */
class NetworkGraphRenderer {
public:
    /**
     * @brief Constructor
     * @param shaderManager Reference to shader manager
     */
    explicit NetworkGraphRenderer(ShaderManager& shaderManager);
    
    /**
     * @brief Destructor
     */
    ~NetworkGraphRenderer();
    
    /**
     * @brief Initialize renderer (create buffers, load shaders)
     * @return True if successful
     */
    bool initialize();
    
    /**
     * @brief Render network graph
     * @param adapter Network data adapter
     * @param camera Camera for view/projection matrices
     * @param config Rendering configuration
     */
    void render(const NetworkDataAdapter& adapter, const Camera& camera, const RenderConfig& config);
    
    /**
     * @brief Render neurons only
     * @param adapter Network data adapter
     * @param camera Camera for view/projection matrices
     * @param config Rendering configuration
     */
    void renderNeurons(const NetworkDataAdapter& adapter, const Camera& camera, const RenderConfig& config);
    
    /**
     * @brief Render synapses only
     * @param adapter Network data adapter
     * @param camera Camera for view/projection matrices
     * @param config Rendering configuration
     */
    void renderSynapses(const NetworkDataAdapter& adapter, const Camera& camera, const RenderConfig& config);

    /**
     * @brief Render selected neurons with highlighting
     * @param selectedNeurons Set of selected neuron IDs
     * @param adapter Network data adapter
     * @param camera Camera for view/projection matrices
     * @param highlightColor Color for highlighting
     */
    void renderSelectedNeurons(const std::unordered_set<uint64_t>& selectedNeurons,
                              const NetworkDataAdapter& adapter,
                              const Camera& camera,
                              const glm::vec4& highlightColor);

    /**
     * @brief Render hierarchical group boundaries
     * @param adapter Network data adapter
     * @param camera Camera for view/projection matrices
     * @param config Rendering configuration
     */
    void renderGroupBoundaries(const NetworkDataAdapter& adapter, const Camera& camera, const RenderConfig& config);

    /**
     * @brief Render clusters with activity-based coloring
     * @param adapter Network data adapter
     * @param camera Camera for view/projection matrices
     * @param config Rendering configuration
     * @param clusterActivity Map of cluster ID to spike count
     */
    void renderClusters(const NetworkDataAdapter& adapter,
                       const Camera& camera,
                       const RenderConfig& config,
                       const std::map<uint64_t, size_t>& clusterActivity);

    /**
     * @brief Set selected neurons
     * @param neuronIds Set of neuron IDs to highlight
     */
    void setSelectedNeurons(const std::set<uint64_t>& neuronIds);
    
    /**
     * @brief Set selected synapses
     * @param synapseIds Set of synapse IDs to highlight
     */
    void setSelectedSynapses(const std::set<uint64_t>& synapseIds);
    
    /**
     * @brief Clear selection
     */
    void clearSelection();
    
    /**
     * @brief Get rendering statistics
     * @return Rendering statistics from last frame
     */
    const RenderStats& getStats() const { return stats_; }
    
    /**
     * @brief Cleanup OpenGL resources
     */
    void cleanup();

private:
    // Initialization helpers
    bool createNeuronBuffers();
    bool createSynapseBuffers();
    bool createGroupBoundaryBuffers();
    bool createClusterBuffers();
    bool loadShaders();
    
    // Rendering helpers
    void renderNeuronsInstanced(const std::vector<NeuronVisualData>& neurons,
                               const Camera& camera, const RenderConfig& config);
    void renderNeuronsIndividual(const std::vector<NeuronVisualData>& neurons,
                                const Camera& camera, const RenderConfig& config);
    void renderSynapsesLines(const std::vector<SynapseVisualData>& synapses,
                            const Camera& camera, const RenderConfig& config);
    void renderSynapsesCurves(const std::vector<SynapseVisualData>& synapses,
                             const Camera& camera, const RenderConfig& config);
    
    // Culling helpers
    bool isInFrustum(const Position3D& position, float radius, const Camera& camera) const;
    bool isWithinDistance(const Position3D& position, const Camera& camera, float maxDistance) const;
    LODLevel computeLOD(const Position3D& position, const Camera& camera, const RenderConfig& config) const;
    
    // Selection helpers
    bool isSelected(uint64_t neuronId) const;
    bool isSynapseSelected(uint64_t synapseId) const;
    
    // Shader manager reference
    ShaderManager& shaderManager_;
    
    // OpenGL buffers for neurons (instanced rendering)
    GLuint neuronVAO_;
    GLuint neuronVBO_;              // Sphere geometry
    GLuint neuronInstanceVBO_;      // Instance data (position, color, radius)
    GLuint neuronEBO_;              // Element buffer
    int neuronIndexCount_;
    
    // OpenGL buffers for synapses
    GLuint synapseVAO_;
    GLuint synapseVBO_;             // Line/curve geometry
    int synapseVertexCount_;
    
    // OpenGL buffers for group boundaries
    GLuint groupVAO_;
    GLuint groupVBO_;

    // OpenGL buffers for clusters
    GLuint clusterVAO_;
    GLuint clusterVBO_;
    GLuint clusterEBO_;

    // Selection state
    std::set<uint64_t> selectedNeurons_;
    std::set<uint64_t> selectedSynapses_;
    
    // Rendering statistics
    RenderStats stats_;
    
    // Initialization state
    bool initialized_;
};

} // namespace snnfw

#endif // SNNFW_NETWORK_GRAPH_RENDERER_H


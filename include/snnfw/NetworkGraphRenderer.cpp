#include "snnfw/NetworkGraphRenderer.h"
#include "snnfw/GeometryRenderer.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <cmath>

namespace snnfw {

NetworkGraphRenderer::NetworkGraphRenderer(ShaderManager& shaderManager)
    : shaderManager_(shaderManager)
    , neuronVAO_(0)
    , neuronVBO_(0)
    , neuronInstanceVBO_(0)
    , neuronEBO_(0)
    , neuronIndexCount_(0)
    , synapseVAO_(0)
    , synapseVBO_(0)
    , synapseVertexCount_(0)
    , groupVAO_(0)
    , groupVBO_(0)
    , clusterVAO_(0)
    , clusterVBO_(0)
    , clusterEBO_(0)
    , initialized_(false) {
}

NetworkGraphRenderer::~NetworkGraphRenderer() {
    cleanup();
}

bool NetworkGraphRenderer::initialize() {
    if (initialized_) return true;
    
    // Create buffers
    if (!createNeuronBuffers()) {
        std::cerr << "Failed to create neuron buffers" << std::endl;
        return false;
    }
    
    if (!createSynapseBuffers()) {
        std::cerr << "Failed to create synapse buffers" << std::endl;
        return false;
    }
    
    if (!createGroupBoundaryBuffers()) {
        std::cerr << "Failed to create group boundary buffers" << std::endl;
        return false;
    }

    if (!createClusterBuffers()) {
        std::cerr << "Failed to create cluster buffers" << std::endl;
        return false;
    }

    // Load shaders
    if (!loadShaders()) {
        std::cerr << "Failed to load network visualization shaders" << std::endl;
        return false;
    }
    
    initialized_ = true;
    return true;
}

bool NetworkGraphRenderer::createNeuronBuffers() {
    // Generate sphere geometry (unit sphere)
    // Using lower poly count for better performance with instanced rendering
    std::vector<float> vertices;
    std::vector<unsigned int> indices;

    const int segments = 12;  // Reduced from 16 for better performance
    const int rings = 6;      // Reduced from 8 for better performance
    const float radius = 1.0f;
    
    // Generate vertices
    for (int ring = 0; ring <= rings; ++ring) {
        float phi = M_PI * ring / rings;
        float y = radius * std::cos(phi);
        float ringRadius = radius * std::sin(phi);
        
        for (int seg = 0; seg <= segments; ++seg) {
            float theta = 2.0f * M_PI * seg / segments;
            float x = ringRadius * std::cos(theta);
            float z = ringRadius * std::sin(theta);
            
            // Position
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
            
            // Normal (same as position for unit sphere)
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
        }
    }
    
    // Generate indices
    for (int ring = 0; ring < rings; ++ring) {
        for (int seg = 0; seg < segments; ++seg) {
            int current = ring * (segments + 1) + seg;
            int next = current + segments + 1;
            
            // First triangle
            indices.push_back(current);
            indices.push_back(next);
            indices.push_back(current + 1);
            
            // Second triangle
            indices.push_back(current + 1);
            indices.push_back(next);
            indices.push_back(next + 1);
        }
    }
    
    neuronIndexCount_ = indices.size();
    
    // Create VAO
    glGenVertexArrays(1, &neuronVAO_);
    glBindVertexArray(neuronVAO_);
    
    // Create VBO for sphere geometry
    glGenBuffers(1, &neuronVBO_);
    glBindBuffer(GL_ARRAY_BUFFER, neuronVBO_);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
    
    // Position attribute (location 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Normal attribute (location 1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    // Create EBO
    glGenBuffers(1, &neuronEBO_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, neuronEBO_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
    
    // Create instance VBO (pre-allocate for up to 100k neurons)
    glGenBuffers(1, &neuronInstanceVBO_);
    glBindBuffer(GL_ARRAY_BUFFER, neuronInstanceVBO_);
    glBufferData(GL_ARRAY_BUFFER, 100000 * 8 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    
    // Instance position (location 2)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2, 1);  // Advance once per instance
    
    // Instance color (location 3)
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(3);
    glVertexAttribDivisor(3, 1);
    
    // Instance radius (location 4)
    glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(7 * sizeof(float)));
    glEnableVertexAttribArray(4);
    glVertexAttribDivisor(4, 1);
    
    glBindVertexArray(0);
    
    return true;
}

bool NetworkGraphRenderer::createSynapseBuffers() {
    // Create VAO for synapses (lines)
    glGenVertexArrays(1, &synapseVAO_);
    glBindVertexArray(synapseVAO_);

    // Create VBO (pre-allocate for up to 1M synapses)
    glGenBuffers(1, &synapseVBO_);
    glBindBuffer(GL_ARRAY_BUFFER, synapseVBO_);
    glBufferData(GL_ARRAY_BUFFER, 1000000 * 14 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    
    // Position attribute (location 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Color attribute (location 1)
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
    
    return true;
}

bool NetworkGraphRenderer::createGroupBoundaryBuffers() {
    // Create VAO for group boundaries
    glGenVertexArrays(1, &groupVAO_);
    glBindVertexArray(groupVAO_);

    // Create VBO (will be filled per frame)
    glGenBuffers(1, &groupVBO_);
    glBindBuffer(GL_ARRAY_BUFFER, groupVBO_);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);

    return true;
}

bool NetworkGraphRenderer::createClusterBuffers() {
    // Create VAO for clusters (boxes)
    glGenVertexArrays(1, &clusterVAO_);
    glBindVertexArray(clusterVAO_);

    // Create VBO for box vertices (position + color)
    glGenBuffers(1, &clusterVBO_);
    glBindBuffer(GL_ARRAY_BUFFER, clusterVBO_);

    // Position attribute (location 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Color attribute (location 1)
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Create EBO for box indices
    glGenBuffers(1, &clusterEBO_);

    glBindVertexArray(0);

    return true;
}

bool NetworkGraphRenderer::loadShaders() {
    // Load neuron shader
    if (!shaderManager_.loadShader("network_neuron", 
                                   "shaders/network_neuron.vert", 
                                   "shaders/network_neuron.frag")) {
        return false;
    }
    
    // Load synapse shader
    if (!shaderManager_.loadShader("network_synapse",
                                   "shaders/network_synapse.vert",
                                   "shaders/network_synapse.frag")) {
        return false;
    }
    
    return true;
}

void NetworkGraphRenderer::render(const NetworkDataAdapter& adapter, 
                                  const Camera& camera, 
                                  const RenderConfig& config) {
    if (!initialized_) {
        std::cerr << "NetworkGraphRenderer not initialized" << std::endl;
        return;
    }
    
    // Reset stats
    stats_ = RenderStats();
    
    // Render based on mode
    switch (config.mode) {
        case RenderMode::NEURONS_ONLY:
            if (config.renderNeurons) {
                renderNeurons(adapter, camera, config);
            }
            break;
            
        case RenderMode::SYNAPSES_ONLY:
            if (config.renderSynapses) {
                renderSynapses(adapter, camera, config);
            }
            break;
            
        case RenderMode::NEURONS_AND_SYNAPSES:
            if (config.renderSynapses) {
                renderSynapses(adapter, camera, config);
            }
            if (config.renderNeurons) {
                renderNeurons(adapter, camera, config);
            }
            break;
            
        case RenderMode::HIERARCHICAL_GROUPS:
            if (config.renderGroupBoundaries) {
                renderGroupBoundaries(adapter, camera, config);
            }
            break;
    }
}

void NetworkGraphRenderer::renderNeurons(const NetworkDataAdapter& adapter,
                                        const Camera& camera,
                                        const RenderConfig& config) {
    if (config.neuronUseInstancing) {
        renderNeuronsInstanced(adapter.getNeurons(), camera, config);
    } else {
        renderNeuronsIndividual(adapter.getNeurons(), camera, config);
    }
}

void NetworkGraphRenderer::renderSelectedNeurons(const std::unordered_set<uint64_t>& selectedNeurons,
                                                const NetworkDataAdapter& adapter,
                                                const Camera& camera,
                                                const glm::vec4& highlightColor) {
    if (selectedNeurons.empty()) return;

    // Filter neurons to only selected ones
    std::vector<NeuronVisualData> selected;
    const auto& allNeurons = adapter.getNeurons();
    for (const auto& neuron : allNeurons) {
        if (selectedNeurons.count(neuron.id)) {
            selected.push_back(neuron);
        }
    }

    if (selected.empty()) return;

    // Prepare instance data with highlight color
    std::vector<float> instanceData;
    for (const auto& neuron : selected) {
        // Position (x, y, z)
        instanceData.push_back(neuron.position.x);
        instanceData.push_back(neuron.position.y);
        instanceData.push_back(neuron.position.z);

        // Color (use highlight color)
        instanceData.push_back(highlightColor.r);
        instanceData.push_back(highlightColor.g);
        instanceData.push_back(highlightColor.b);
        instanceData.push_back(highlightColor.a);

        // Scale (slightly larger for highlighting)
        float scale = 1.5f;
        instanceData.push_back(scale);
    }

    // Upload instance data
    glBindBuffer(GL_ARRAY_BUFFER, neuronInstanceVBO_);
    glBufferData(GL_ARRAY_BUFFER, instanceData.size() * sizeof(float),
                instanceData.data(), GL_DYNAMIC_DRAW);

    // Setup shader
    unsigned int shader = shaderManager_.getShader("network_neuron");
    if (!shader) return;

    glUseProgram(shader);

    // Set matrices
    glm::mat4 view = camera.getViewMatrix();
    glm::mat4 projection = camera.getProjectionMatrix(1280.0f / 720.0f);  // TODO: Get actual aspect ratio

    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

    // Set lighting
    glUniform3fv(glGetUniformLocation(shader, "lightPos"), 1, glm::value_ptr(glm::vec3(10.0f, 10.0f, 10.0f)));
    glUniform3fv(glGetUniformLocation(shader, "viewPos"), 1, glm::value_ptr(camera.getPosition()));

    // Enable blending for glow effect
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    // Render
    glBindVertexArray(neuronVAO_);
    glDrawElementsInstanced(GL_TRIANGLES, neuronIndexCount_, GL_UNSIGNED_INT, 0, selected.size());
    glBindVertexArray(0);

    // Restore blending
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_BLEND);
}

void NetworkGraphRenderer::renderNeuronsInstanced(const std::vector<NeuronVisualData>& neurons,
                                                 const Camera& camera,
                                                 const RenderConfig& config) {
    if (neurons.empty()) return;

    // Prepare instance data
    std::vector<float> instanceData;
    instanceData.reserve(neurons.size() * 8);  // 3 pos + 4 color + 1 radius = 8 floats per neuron
    
    size_t renderedCount = 0;
    for (const auto& neuron : neurons) {
        // Culling
        if (config.frustumCulling && !isInFrustum(neuron.position, neuron.radius, camera)) {
            stats_.neuronsCulled++;
            continue;
        }
        if (config.distanceCulling && !isWithinDistance(neuron.position, camera, config.maxRenderDistance)) {
            stats_.neuronsCulled++;
            continue;
        }
        
        // Position
        instanceData.push_back(neuron.position.x);
        instanceData.push_back(neuron.position.y);
        instanceData.push_back(neuron.position.z);
        
        // Color
        instanceData.push_back(neuron.r);
        instanceData.push_back(neuron.g);
        instanceData.push_back(neuron.b);
        instanceData.push_back(neuron.a);
        
        // Radius (scaled by activity)
        float radius = config.neuronBaseRadius + neuron.activity * config.neuronActivityScale;
        instanceData.push_back(radius);
        
        renderedCount++;
    }
    
    if (renderedCount == 0) return;

    // Upload instance data (use SubData for better performance if data fits in pre-allocated buffer)
    glBindBuffer(GL_ARRAY_BUFFER, neuronInstanceVBO_);
    size_t dataSize = instanceData.size() * sizeof(float);
    size_t maxBufferSize = 100000 * 8 * sizeof(float);

    if (dataSize <= maxBufferSize) {
        // Use glBufferSubData for better performance (no reallocation)
        glBufferSubData(GL_ARRAY_BUFFER, 0, dataSize, instanceData.data());
    } else {
        // Reallocate if needed (rare case with >100k neurons)
        glBufferData(GL_ARRAY_BUFFER, dataSize, instanceData.data(), GL_DYNAMIC_DRAW);
    }
    
    // Use shader
    shaderManager_.useShader("network_neuron");
    
    // Set uniforms
    shaderManager_.setUniform("view", camera.getViewMatrix());
    shaderManager_.setUniform("projection", camera.getProjectionMatrix(800.0f / 600.0f));  // TODO: get aspect from config
    shaderManager_.setUniform("lightPos", config.lightPosition);
    shaderManager_.setUniform("viewPos", camera.getPosition());
    shaderManager_.setUniform("lightColor", config.lightColor);
    shaderManager_.setUniform("ambientStrength", config.ambientStrength);
    shaderManager_.setUniform("specularStrength", config.specularStrength);
    shaderManager_.setUniform("enableLighting", config.enableLighting ? 1 : 0);
    shaderManager_.setUniform("isSelected", 0);  // TODO: handle selection
    
    // Draw instanced
    glBindVertexArray(neuronVAO_);
    glDrawElementsInstanced(GL_TRIANGLES, neuronIndexCount_, GL_UNSIGNED_INT, 0, renderedCount);
    glBindVertexArray(0);
    
    stats_.neuronsRendered = renderedCount;
    stats_.drawCalls++;
}

void NetworkGraphRenderer::renderNeuronsIndividual(const std::vector<NeuronVisualData>& neurons,
                                                   const Camera& camera,
                                                   const RenderConfig& config) {
    // TODO: Implement individual rendering (for debugging or special cases)
    renderNeuronsInstanced(neurons, camera, config);
}

void NetworkGraphRenderer::renderSynapses(const NetworkDataAdapter& adapter,
                                         const Camera& camera,
                                         const RenderConfig& config) {
    if (config.synapseCurved) {
        renderSynapsesCurves(adapter.getSynapses(), camera, config);
    } else {
        renderSynapsesLines(adapter.getSynapses(), camera, config);
    }
}

void NetworkGraphRenderer::renderSynapsesLines(const std::vector<SynapseVisualData>& synapses,
                                              const Camera& camera,
                                              const RenderConfig& config) {
    if (synapses.empty()) return;
    
    // Prepare line data
    std::vector<float> lineData;
    lineData.reserve(synapses.size() * 14);  // 2 vertices * 7 floats (3 pos + 4 color)
    
    size_t renderedCount = 0;
    for (const auto& synapse : synapses) {
        // TODO: Add culling for synapses
        
        // Source vertex
        lineData.push_back(synapse.sourcePos.x);
        lineData.push_back(synapse.sourcePos.y);
        lineData.push_back(synapse.sourcePos.z);
        lineData.push_back(synapse.r);
        lineData.push_back(synapse.g);
        lineData.push_back(synapse.b);
        lineData.push_back(synapse.a);
        
        // Target vertex
        lineData.push_back(synapse.targetPos.x);
        lineData.push_back(synapse.targetPos.y);
        lineData.push_back(synapse.targetPos.z);
        lineData.push_back(synapse.r);
        lineData.push_back(synapse.g);
        lineData.push_back(synapse.b);
        lineData.push_back(synapse.a);
        
        renderedCount++;
    }
    
    if (renderedCount == 0) return;

    // Upload line data (use SubData for better performance if data fits in pre-allocated buffer)
    glBindBuffer(GL_ARRAY_BUFFER, synapseVBO_);
    size_t dataSize = lineData.size() * sizeof(float);
    size_t maxBufferSize = 1000000 * 14 * sizeof(float);

    if (dataSize <= maxBufferSize) {
        // Use glBufferSubData for better performance (no reallocation)
        glBufferSubData(GL_ARRAY_BUFFER, 0, dataSize, lineData.data());
    } else {
        // Reallocate if needed (rare case with >1M synapses)
        glBufferData(GL_ARRAY_BUFFER, dataSize, lineData.data(), GL_DYNAMIC_DRAW);
    }
    
    // Use shader
    shaderManager_.useShader("network_synapse");
    
    // Set uniforms
    shaderManager_.setUniform("view", camera.getViewMatrix());
    shaderManager_.setUniform("projection", camera.getProjectionMatrix(800.0f / 600.0f));
    
    // Draw lines
    glBindVertexArray(synapseVAO_);
    glDrawArrays(GL_LINES, 0, renderedCount * 2);
    glBindVertexArray(0);
    
    stats_.synapsesRendered = renderedCount;
    stats_.drawCalls++;
}

void NetworkGraphRenderer::renderSynapsesCurves(const std::vector<SynapseVisualData>& synapses,
                                               const Camera& camera,
                                               const RenderConfig& config) {
    // TODO: Implement curved synapse rendering with Bezier curves
    renderSynapsesLines(synapses, camera, config);
}

void NetworkGraphRenderer::renderGroupBoundaries(const NetworkDataAdapter& adapter,
                                                const Camera& camera,
                                                const RenderConfig& config) {
    // TODO: Implement group boundary rendering
}

void NetworkGraphRenderer::renderClusters(const NetworkDataAdapter& adapter,
                                         const Camera& camera,
                                         const RenderConfig& config,
                                         const std::map<uint64_t, size_t>& clusterActivity) {
    const auto& groups = adapter.getGroups();
    if (groups.empty()) return;

    // Find max activity for normalization
    size_t maxActivity = 1;
    for (const auto& [clusterId, count] : clusterActivity) {
        if (count > maxActivity) maxActivity = count;
    }

    // Prepare vertex data for all cluster boxes
    std::vector<float> vertices;
    std::vector<unsigned int> indices;

    for (const auto& group : groups) {
        if (group.typeName != "Cluster") continue;
        if (group.neuronIds.empty()) continue;

        // Get activity level for this cluster (0.0 to 1.0)
        float activityLevel = 0.0f;
        auto it = clusterActivity.find(group.id);
        if (it != clusterActivity.end() && maxActivity > 0) {
            activityLevel = static_cast<float>(it->second) / static_cast<float>(maxActivity);
        }

        // Color interpolation: deep red (0.5, 0.0, 0.0) to bright white (1.0, 1.0, 1.0)
        glm::vec4 color;
        if (config.clusterActivityColoring) {
            // Deep red at 0% activity
            glm::vec4 lowColor(0.5f, 0.0f, 0.0f, config.clusterAlpha);
            // Bright white at 100% activity
            glm::vec4 highColor(1.0f, 1.0f, 1.0f, config.clusterAlpha);
            color = glm::mix(lowColor, highColor, activityLevel);
        } else {
            color = glm::vec4(0.3f, 0.3f, 0.8f, config.clusterAlpha);
        }

        // Get bounding box
        glm::vec3 minPos(group.boundingBoxMin.x, group.boundingBoxMin.y, group.boundingBoxMin.z);
        glm::vec3 maxPos(group.boundingBoxMax.x, group.boundingBoxMax.y, group.boundingBoxMax.z);

        // Create box vertices (8 corners)
        unsigned int baseIndex = vertices.size() / 7;

        // 8 corners of the box
        glm::vec3 corners[8] = {
            glm::vec3(minPos.x, minPos.y, minPos.z),
            glm::vec3(maxPos.x, minPos.y, minPos.z),
            glm::vec3(maxPos.x, maxPos.y, minPos.z),
            glm::vec3(minPos.x, maxPos.y, minPos.z),
            glm::vec3(minPos.x, minPos.y, maxPos.z),
            glm::vec3(maxPos.x, minPos.y, maxPos.z),
            glm::vec3(maxPos.x, maxPos.y, maxPos.z),
            glm::vec3(minPos.x, maxPos.y, maxPos.z)
        };

        // Add vertices
        for (int i = 0; i < 8; ++i) {
            vertices.push_back(corners[i].x);
            vertices.push_back(corners[i].y);
            vertices.push_back(corners[i].z);
            vertices.push_back(color.r);
            vertices.push_back(color.g);
            vertices.push_back(color.b);
            vertices.push_back(color.a);
        }

        // Add indices for 12 edges (lines)
        unsigned int edgeIndices[24] = {
            0, 1,  1, 2,  2, 3,  3, 0,  // Bottom face
            4, 5,  5, 6,  6, 7,  7, 4,  // Top face
            0, 4,  1, 5,  2, 6,  3, 7   // Vertical edges
        };

        for (int i = 0; i < 24; ++i) {
            indices.push_back(baseIndex + edgeIndices[i]);
        }
    }

    if (vertices.empty()) return;

    // Upload data
    glBindBuffer(GL_ARRAY_BUFFER, clusterVBO_);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, clusterEBO_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_DYNAMIC_DRAW);

    // Use synapse shader (simple line shader)
    shaderManager_.useShader("network_synapse");

    // Set uniforms
    shaderManager_.setUniform("view", camera.getViewMatrix());
    shaderManager_.setUniform("projection", camera.getProjectionMatrix(800.0f / 600.0f));

    // Enable blending for transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Draw lines
    glBindVertexArray(clusterVAO_);
    glDrawElements(GL_LINES, indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    glDisable(GL_BLEND);
}

void NetworkGraphRenderer::setSelectedNeurons(const std::set<uint64_t>& neuronIds) {
    selectedNeurons_ = neuronIds;
}

void NetworkGraphRenderer::setSelectedSynapses(const std::set<uint64_t>& synapseIds) {
    selectedSynapses_ = synapseIds;
}

void NetworkGraphRenderer::clearSelection() {
    selectedNeurons_.clear();
    selectedSynapses_.clear();
}

bool NetworkGraphRenderer::isInFrustum(const Position3D& position, float radius, const Camera& camera) const {
    // TODO: Implement proper frustum culling
    return true;  // For now, render everything
}

bool NetworkGraphRenderer::isWithinDistance(const Position3D& position, const Camera& camera, float maxDistance) const {
    glm::vec3 camPos = camera.getPosition();
    Position3D cameraPosition(camPos.x, camPos.y, camPos.z);
    float dist = position.distanceTo(cameraPosition);
    return dist <= maxDistance;
}

LODLevel NetworkGraphRenderer::computeLOD(const Position3D& position, const Camera& camera, const RenderConfig& config) const {
    glm::vec3 camPos = camera.getPosition();
    Position3D cameraPosition(camPos.x, camPos.y, camPos.z);
    float dist = position.distanceTo(cameraPosition);

    if (dist < config.lodFullDistance) return LODLevel::FULL;
    if (dist < config.lodMediumDistance) return LODLevel::MEDIUM;
    if (dist < config.lodLowDistance) return LODLevel::LOW;
    return LODLevel::MINIMAL;
}

bool NetworkGraphRenderer::isSelected(uint64_t neuronId) const {
    return selectedNeurons_.find(neuronId) != selectedNeurons_.end();
}

bool NetworkGraphRenderer::isSynapseSelected(uint64_t synapseId) const {
    return selectedSynapses_.find(synapseId) != selectedSynapses_.end();
}

void NetworkGraphRenderer::cleanup() {
    if (neuronVAO_) glDeleteVertexArrays(1, &neuronVAO_);
    if (neuronVBO_) glDeleteBuffers(1, &neuronVBO_);
    if (neuronInstanceVBO_) glDeleteBuffers(1, &neuronInstanceVBO_);
    if (neuronEBO_) glDeleteBuffers(1, &neuronEBO_);
    if (synapseVAO_) glDeleteVertexArrays(1, &synapseVAO_);
    if (synapseVBO_) glDeleteBuffers(1, &synapseVBO_);
    if (groupVAO_) glDeleteVertexArrays(1, &groupVAO_);
    if (groupVBO_) glDeleteBuffers(1, &groupVBO_);
    if (clusterVAO_) glDeleteVertexArrays(1, &clusterVAO_);
    if (clusterVBO_) glDeleteBuffers(1, &clusterVBO_);
    if (clusterEBO_) glDeleteBuffers(1, &clusterEBO_);

    initialized_ = false;
}

} // namespace snnfw


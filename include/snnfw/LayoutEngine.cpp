#include "snnfw/LayoutEngine.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <queue>

namespace snnfw {

LayoutEngine::LayoutEngine()
    : progressCallback_(nullptr) {
}

bool LayoutEngine::computeLayout(NetworkDataAdapter& adapter, const LayoutConfig& config) {
    switch (config.algorithm) {
        case LayoutAlgorithm::HIERARCHICAL_TREE:
            return computeHierarchicalTreeLayout(adapter, config);
        case LayoutAlgorithm::HIERARCHICAL_GROUPED:
            return computeHierarchicalGroupedLayout(adapter, config);
        case LayoutAlgorithm::FORCE_DIRECTED:
            return computeForceDirectedLayout(adapter, config);
        case LayoutAlgorithm::GRID:
            return computeGridLayout(adapter, config);
        case LayoutAlgorithm::CIRCULAR:
            return computeCircularLayout(adapter, config);
        case LayoutAlgorithm::LAYERED:
            return computeLayeredLayout(adapter, config);
        case LayoutAlgorithm::ANATOMICAL:
            return computeAnatomicalLayout(adapter, config);
        default:
            return false;
    }
}

bool LayoutEngine::computeHierarchicalTreeLayout(NetworkDataAdapter& adapter, const LayoutConfig& config) {
    reportProgress(0.0f);

    // Store config for use in assignTreePositions
    currentConfig_ = config;

    // Build hierarchy tree
    TreeNode* root = buildHierarchyTree(adapter);
    if (!root) return false;

    reportProgress(0.3f);

    // Compute positions recursively
    computeTreePositions(root, config, 0);

    reportProgress(0.7f);

    // Assign positions to neurons
    assignTreePositions(root, adapter, config);

    reportProgress(0.9f);

    // Cleanup
    deleteTree(root);

    // Post-processing
    if (config.centerLayout) {
        centerLayout(adapter);
    }
    if (config.normalizePositions) {
        normalizePositions(adapter, config.boundingBoxSize);
    }

    reportProgress(1.0f);
    return true;
}

LayoutEngine::TreeNode* LayoutEngine::buildHierarchyTree(NetworkDataAdapter& adapter) {
    auto& groups = adapter.getGroups();
    if (groups.empty()) return nullptr;
    
    // Find root (group with no parent)
    const HierarchicalGroup* rootGroup = nullptr;
    for (const auto& group : groups) {
        if (group.parentGroupId == 0) {
            rootGroup = &group;
            break;
        }
    }
    
    if (!rootGroup) return nullptr;
    
    // Build tree recursively
    std::function<TreeNode*(const HierarchicalGroup&)> buildNode;
    buildNode = [&](const HierarchicalGroup& group) -> TreeNode* {
        TreeNode* node = new TreeNode();
        node->id = group.id;
        node->type = group.typeName;
        
        // Find children
        for (const auto& childGroup : groups) {
            if (childGroup.parentGroupId == group.id) {
                TreeNode* child = buildNode(childGroup);
                if (child) {
                    child->parent = node;
                    node->children.push_back(child);
                }
            }
        }
        
        return node;
    };
    
    return buildNode(*rootGroup);
}

void LayoutEngine::computeTreePositions(TreeNode* node, const LayoutConfig& config, int depth) {
    if (!node) return;
    
    // Compute positions for children first (post-order)
    for (TreeNode* child : node->children) {
        computeTreePositions(child, config, depth + 1);
    }
    
    // Compute width needed for this subtree
    if (node->children.empty()) {
        node->width = config.neuronSpacing;
    } else {
        float totalWidth = 0.0f;
        for (TreeNode* child : node->children) {
            totalWidth += child->width;
        }
        node->width = totalWidth * config.treeHorizontalSpread;
    }
    
    // Position children horizontally
    float currentX = -node->width / 2.0f;
    for (TreeNode* child : node->children) {
        child->position.x = currentX + child->width / 2.0f;
        child->position.y = -depth * config.treeVerticalSpacing;
        child->position.z = 0.0f;
        currentX += child->width;
    }
    
    // Position this node at center of children
    if (!node->children.empty()) {
        float minX = node->children.front()->position.x;
        float maxX = node->children.back()->position.x;
        node->position.x = (minX + maxX) / 2.0f;
    } else {
        node->position.x = 0.0f;
    }
    node->position.y = -depth * config.treeVerticalSpacing;
    node->position.z = 0.0f;
}

void LayoutEngine::assignTreePositions(TreeNode* node, NetworkDataAdapter& adapter) {
    assignTreePositions(node, adapter, currentConfig_);
}

void LayoutEngine::assignTreePositions(TreeNode* node, NetworkDataAdapter& adapter, const LayoutConfig& config) {
    if (!node) return;

    // If this is a cluster, assign positions to its neurons
    if (node->type == "Cluster") {
        auto neurons = adapter.getNeuronsByLevel("Cluster", node->id);

        // Arrange neurons in a grid within the cluster
        int gridSize = static_cast<int>(std::ceil(std::sqrt(neurons.size())));
        float spacing = 1.0f;

        for (size_t i = 0; i < neurons.size(); ++i) {
            int row = i / gridSize;
            int col = i % gridSize;

            Position3D pos;
            pos.x = node->position.x + (col - gridSize/2.0f) * spacing;
            pos.y = node->position.y;
            pos.z = node->position.z + (row - gridSize/2.0f) * spacing;

            // Update neuron position in adapter
            auto& allNeurons = const_cast<std::vector<NeuronVisualData>&>(adapter.getNeurons());
            for (auto& neuron : allNeurons) {
                if (neuron.id == neurons[i].id) {
                    // Only override if configured to do so, or if position is not set (all zeros)
                    bool hasPosition = (neuron.position.x != 0.0f || neuron.position.y != 0.0f || neuron.position.z != 0.0f);
                    if (config.overrideStoredPositions || !hasPosition) {
                        neuron.position = pos;
                    }
                    break;
                }
            }
        }
    }

    // Recurse to children
    for (TreeNode* child : node->children) {
        // Offset child position by parent position
        child->position.x += node->position.x;
        child->position.y += node->position.y;
        child->position.z += node->position.z;

        assignTreePositions(child, adapter, config);
    }
}

void LayoutEngine::deleteTree(TreeNode* node) {
    if (!node) return;
    
    for (TreeNode* child : node->children) {
        deleteTree(child);
    }
    
    delete node;
}

bool LayoutEngine::computeForceDirectedLayout(NetworkDataAdapter& adapter, const LayoutConfig& config) {
    reportProgress(0.0f);
    
    // Initialize force nodes
    std::vector<ForceNode> nodes;
    initializeForceNodes(adapter, nodes);
    
    if (nodes.empty()) return false;
    
    // Iterative force simulation
    float dt = 0.1f;
    for (int iter = 0; iter < config.maxIterations; ++iter) {
        // Compute forces
        computeForces(nodes, adapter.getSynapses(), config);
        
        // Update positions
        updatePositions(nodes, config, dt);
        
        // Check convergence
        if (hasConverged(nodes, config.convergenceThreshold)) {
            std::cout << "Force-directed layout converged at iteration " << iter << std::endl;
            break;
        }
        
        // Report progress
        if (iter % 100 == 0) {
            reportProgress(static_cast<float>(iter) / config.maxIterations);
        }
    }
    
    // Copy positions back to adapter
    auto& neurons = const_cast<std::vector<NeuronVisualData>&>(adapter.getNeurons());
    for (size_t i = 0; i < nodes.size(); ++i) {
        neurons[i].position = nodes[i].position;
    }
    
    // Post-processing
    if (config.centerLayout) {
        centerLayout(adapter);
    }
    if (config.normalizePositions) {
        normalizePositions(adapter, config.boundingBoxSize);
    }
    
    reportProgress(1.0f);
    return true;
}

void LayoutEngine::initializeForceNodes(NetworkDataAdapter& adapter, std::vector<ForceNode>& nodes) {
    const auto& neurons = adapter.getNeurons();
    nodes.resize(neurons.size());
    
    // Random initial positions
    for (size_t i = 0; i < neurons.size(); ++i) {
        nodes[i].neuronId = neurons[i].id;
        nodes[i].position.x = (rand() % 1000 - 500) / 10.0f;
        nodes[i].position.y = (rand() % 1000 - 500) / 10.0f;
        nodes[i].position.z = (rand() % 1000 - 500) / 10.0f;
        nodes[i].mass = 1.0f;
    }
}

void LayoutEngine::computeForces(std::vector<ForceNode>& nodes, 
                                 const std::vector<SynapseVisualData>& synapses,
                                 const LayoutConfig& config) {
    // Reset forces
    for (auto& node : nodes) {
        node.force = Position3D(0, 0, 0);
    }
    
    // Repulsion between all pairs
    for (size_t i = 0; i < nodes.size(); ++i) {
        for (size_t j = i + 1; j < nodes.size(); ++j) {
            Position3D delta = nodes[i].position - nodes[j].position;
            float dist = delta.distanceTo(Position3D(0, 0, 0));
            if (dist < 0.1f) dist = 0.1f;  // Avoid division by zero
            
            float repulsion = config.repulsionConstant / (dist * dist);
            Position3D force = delta * (repulsion / dist);
            
            nodes[i].force = nodes[i].force + force;
            nodes[j].force = nodes[j].force - force;
        }
    }
    
    // Attraction along synapses
    for (const auto& synapse : synapses) {
        // Find nodes
        size_t sourceIdx = -1, targetIdx = -1;
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i].neuronId == synapse.sourceNeuronId) sourceIdx = i;
            if (nodes[i].neuronId == synapse.targetNeuronId) targetIdx = i;
        }
        
        if (sourceIdx == static_cast<size_t>(-1) || targetIdx == static_cast<size_t>(-1)) continue;
        
        Position3D delta = nodes[targetIdx].position - nodes[sourceIdx].position;
        float dist = delta.distanceTo(Position3D(0, 0, 0));
        
        float attraction = config.springConstant * dist;
        Position3D force = delta * (attraction / (dist + 0.1f));
        
        nodes[sourceIdx].force = nodes[sourceIdx].force + force;
        nodes[targetIdx].force = nodes[targetIdx].force - force;
    }
}

void LayoutEngine::updatePositions(std::vector<ForceNode>& nodes, const LayoutConfig& config, float dt) {
    for (auto& node : nodes) {
        // Update velocity
        node.velocity = node.velocity * config.dampingFactor + node.force * dt;
        
        // Update position
        node.position = node.position + node.velocity * dt;
    }
}

bool LayoutEngine::hasConverged(const std::vector<ForceNode>& nodes, float threshold) {
    float totalEnergy = 0.0f;
    for (const auto& node : nodes) {
        float speed = node.velocity.distanceTo(Position3D(0, 0, 0));
        totalEnergy += speed * speed;
    }
    return totalEnergy < threshold;
}

bool LayoutEngine::computeGridLayout(NetworkDataAdapter& adapter, const LayoutConfig& config) {
    assignGridPositions(adapter, config);
    
    if (config.centerLayout) centerLayout(adapter);
    if (config.normalizePositions) normalizePositions(adapter, config.boundingBoxSize);
    
    return true;
}

void LayoutEngine::assignGridPositions(NetworkDataAdapter& adapter, const LayoutConfig& config) {
    auto& neurons = const_cast<std::vector<NeuronVisualData>&>(adapter.getNeurons());

    int cols = config.gridColumns;
    float cellSize = config.gridCellSize;

    for (size_t i = 0; i < neurons.size(); ++i) {
        // Skip neurons that already have positions (unless overriding)
        if (!config.overrideStoredPositions) {
            // Check if position is non-zero (indicating it was set from stored data)
            if (neurons[i].position.x != 0.0f || neurons[i].position.y != 0.0f || neurons[i].position.z != 0.0f) {
                continue;
            }
        }

        int row = i / cols;
        int col = i % cols;

        neurons[i].position.x = col * cellSize;
        neurons[i].position.y = 0.0f;
        neurons[i].position.z = row * cellSize;
    }
}

bool LayoutEngine::computeCircularLayout(NetworkDataAdapter& adapter, const LayoutConfig& config) {
    assignCircularPositions(adapter, config);
    
    if (config.centerLayout) centerLayout(adapter);
    if (config.normalizePositions) normalizePositions(adapter, config.boundingBoxSize);
    
    return true;
}

void LayoutEngine::assignCircularPositions(NetworkDataAdapter& adapter, const LayoutConfig& config) {
    auto& neurons = const_cast<std::vector<NeuronVisualData>&>(adapter.getNeurons());
    
    float radius = config.circularRadius;
    size_t count = neurons.size();
    
    for (size_t i = 0; i < count; ++i) {
        float angle = 2.0f * M_PI * i / count;
        neurons[i].position.x = radius * std::cos(angle);
        neurons[i].position.y = 0.0f;
        neurons[i].position.z = radius * std::sin(angle);
    }
}

bool LayoutEngine::computeLayeredLayout(NetworkDataAdapter& adapter, const LayoutConfig& config) {
    assignLayeredPositions(adapter, config);
    
    if (config.centerLayout) centerLayout(adapter);
    if (config.normalizePositions) normalizePositions(adapter, config.boundingBoxSize);
    
    return true;
}

void LayoutEngine::assignLayeredPositions(NetworkDataAdapter& adapter, const LayoutConfig& config) {
    // Simple layered layout: group by layer ID
    auto& neurons = const_cast<std::vector<NeuronVisualData>&>(adapter.getNeurons());
    
    // Group neurons by layer
    std::map<uint64_t, std::vector<size_t>> layerGroups;
    for (size_t i = 0; i < neurons.size(); ++i) {
        layerGroups[neurons[i].layerId].push_back(i);
    }
    
    // Assign positions
    int layerIdx = 0;
    for (auto& [layerId, indices] : layerGroups) {
        float y = layerIdx * config.layerSpacing;
        
        for (size_t i = 0; i < indices.size(); ++i) {
            neurons[indices[i]].position.x = i * config.neuronSpacing;
            neurons[indices[i]].position.y = y;
            neurons[indices[i]].position.z = 0.0f;
        }
        
        layerIdx++;
    }
}

bool LayoutEngine::computeAnatomicalLayout(NetworkDataAdapter& adapter, const LayoutConfig& config) {
    assignAnatomicalPositions(adapter, config);
    
    if (config.centerLayout) centerLayout(adapter);
    if (config.normalizePositions) normalizePositions(adapter, config.boundingBoxSize);
    
    return true;
}

void LayoutEngine::assignAnatomicalPositions(NetworkDataAdapter& adapter, const LayoutConfig& config) {
    // Placeholder: use grid layout for now
    assignGridPositions(adapter, config);
}

void LayoutEngine::getBoundingBox(Position3D& min, Position3D& max) const {
    min = boundingBoxMin_;
    max = boundingBoxMax_;
}

void LayoutEngine::centerLayout(NetworkDataAdapter& adapter) {
    auto& neurons = const_cast<std::vector<NeuronVisualData>&>(adapter.getNeurons());
    if (neurons.empty()) return;
    
    // Compute center
    Position3D center(0, 0, 0);
    for (const auto& neuron : neurons) {
        center.x += neuron.position.x;
        center.y += neuron.position.y;
        center.z += neuron.position.z;
    }
    center.x /= neurons.size();
    center.y /= neurons.size();
    center.z /= neurons.size();
    
    // Translate to origin
    for (auto& neuron : neurons) {
        neuron.position.x -= center.x;
        neuron.position.y -= center.y;
        neuron.position.z -= center.z;
    }
}

void LayoutEngine::normalizePositions(NetworkDataAdapter& adapter, float boxSize) {
    auto& neurons = const_cast<std::vector<NeuronVisualData>&>(adapter.getNeurons());
    if (neurons.empty()) return;
    
    // Find bounding box
    Position3D min = neurons[0].position;
    Position3D max = neurons[0].position;
    
    for (const auto& neuron : neurons) {
        min.x = std::min(min.x, neuron.position.x);
        min.y = std::min(min.y, neuron.position.y);
        min.z = std::min(min.z, neuron.position.z);
        max.x = std::max(max.x, neuron.position.x);
        max.y = std::max(max.y, neuron.position.y);
        max.z = std::max(max.z, neuron.position.z);
    }
    
    // Compute scale
    float rangeX = max.x - min.x;
    float rangeY = max.y - min.y;
    float rangeZ = max.z - min.z;
    float maxRange = std::max({rangeX, rangeY, rangeZ});
    
    if (maxRange < 0.001f) return;  // Avoid division by zero
    
    float scale = boxSize / maxRange;
    
    // Scale positions
    for (auto& neuron : neurons) {
        neuron.position.x *= scale;
        neuron.position.y *= scale;
        neuron.position.z *= scale;
    }
    
    boundingBoxMin_ = min * scale;
    boundingBoxMax_ = max * scale;
}

void LayoutEngine::reportProgress(float progress) {
    if (progressCallback_) {
        progressCallback_(progress);
    }
}

bool LayoutEngine::computeHierarchicalGroupedLayout(NetworkDataAdapter& adapter, const LayoutConfig& config) {
    reportProgress(0.0f);

    assignHierarchicalGroupedPositions(adapter, config);

    reportProgress(0.8f);

    // Post-processing
    if (config.centerLayout) {
        centerLayout(adapter);
    }
    if (config.normalizePositions) {
        normalizePositions(adapter, config.boundingBoxSize);
    }

    reportProgress(1.0f);
    return true;
}

void LayoutEngine::assignHierarchicalGroupedPositions(NetworkDataAdapter& adapter, const LayoutConfig& config) {
    auto& neurons = const_cast<std::vector<NeuronVisualData>&>(adapter.getNeurons());
    const auto& groups = adapter.getGroups();

    // Build maps for quick lookup
    std::map<uint64_t, const HierarchicalGroup*> groupMap;
    for (const auto& group : groups) {
        groupMap[group.id] = &group;
    }

    // Group neurons by each hierarchical level
    std::map<uint64_t, std::vector<NeuronVisualData*>> columnNeurons;
    std::map<uint64_t, std::vector<NeuronVisualData*>> layerNeurons;
    std::map<uint64_t, std::vector<NeuronVisualData*>> clusterNeurons;

    for (auto& neuron : neurons) {
        if (neuron.columnId != 0) columnNeurons[neuron.columnId].push_back(&neuron);
        if (neuron.layerId != 0) layerNeurons[neuron.layerId].push_back(&neuron);
        if (neuron.clusterId != 0) clusterNeurons[neuron.clusterId].push_back(&neuron);
    }

    // Layout columns in a grid
    std::vector<uint64_t> columnIds;
    for (const auto& [columnId, _] : columnNeurons) {
        columnIds.push_back(columnId);
    }
    std::sort(columnIds.begin(), columnIds.end());

    int numColumns = columnIds.size();
    int columnsPerRow = static_cast<int>(std::ceil(std::sqrt(static_cast<float>(numColumns))));

    for (size_t colIdx = 0; colIdx < columnIds.size(); ++colIdx) {
        uint64_t columnId = columnIds[colIdx];

        // Calculate column base position
        int colRow = colIdx / columnsPerRow;
        int colCol = colIdx % columnsPerRow;
        float columnBaseX = colCol * config.columnSpacing;
        float columnBaseZ = colRow * config.columnSpacing;

        // Get layers in this column
        std::vector<uint64_t> layerIdsInColumn;
        for (const auto& group : groups) {
            if (group.typeName == "Layer" && group.parentGroupId == columnId) {
                layerIdsInColumn.push_back(group.id);
            }
        }
        std::sort(layerIdsInColumn.begin(), layerIdsInColumn.end());

        // Layout layers vertically within the column
        for (size_t layerIdx = 0; layerIdx < layerIdsInColumn.size(); ++layerIdx) {
            uint64_t layerId = layerIdsInColumn[layerIdx];
            float layerBaseY = layerIdx * config.layerSpacing;

            // Get clusters in this layer
            std::vector<uint64_t> clusterIdsInLayer;
            for (const auto& group : groups) {
                if (group.typeName == "Cluster" && group.parentGroupId == layerId) {
                    clusterIdsInLayer.push_back(group.id);
                }
            }
            std::sort(clusterIdsInLayer.begin(), clusterIdsInLayer.end());

            // Layout clusters in a small grid within the layer
            int clustersPerRow = static_cast<int>(std::ceil(std::sqrt(static_cast<float>(clusterIdsInLayer.size()))));

            for (size_t clusterIdx = 0; clusterIdx < clusterIdsInLayer.size(); ++clusterIdx) {
                uint64_t clusterId = clusterIdsInLayer[clusterIdx];

                // Calculate cluster offset within layer
                int clusterRow = clusterIdx / clustersPerRow;
                int clusterCol = clusterIdx % clustersPerRow;
                float clusterOffsetX = clusterCol * config.clusterSpacing;
                float clusterOffsetZ = clusterRow * config.clusterSpacing;

                // Position neurons within cluster
                auto it = clusterNeurons.find(clusterId);
                if (it != clusterNeurons.end()) {
                    auto& neuronsInCluster = it->second;
                    int neuronsPerRow = static_cast<int>(std::ceil(std::sqrt(static_cast<float>(neuronsInCluster.size()))));

                    for (size_t neuronIdx = 0; neuronIdx < neuronsInCluster.size(); ++neuronIdx) {
                        auto* neuron = neuronsInCluster[neuronIdx];

                        // Only set position if not already set (unless overriding)
                        bool hasPosition = (neuron->position.x != 0.0f || neuron->position.y != 0.0f || neuron->position.z != 0.0f);
                        if (config.overrideStoredPositions || !hasPosition) {
                            int neuronRow = neuronIdx / neuronsPerRow;
                            int neuronCol = neuronIdx % neuronsPerRow;

                            neuron->position.x = columnBaseX + clusterOffsetX + neuronCol * config.neuronSpacing;
                            neuron->position.y = layerBaseY;
                            neuron->position.z = columnBaseZ + clusterOffsetZ + neuronRow * config.neuronSpacing;
                        }
                    }
                }
            }
        }
    }
}

} // namespace snnfw


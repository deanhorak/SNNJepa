#include "snnfw/NetworkDataAdapter.h"
#include "snnfw/Datastore.h"
#include "snnfw/NetworkInspector.h"
#include "snnfw/ActivityMonitor.h"
#include "snnfw/Neuron.h"
#include "snnfw/Synapse.h"
#include "snnfw/Cluster.h"
#include "snnfw/Layer.h"
#include "snnfw/Column.h"
#include "snnfw/Nucleus.h"
#include "snnfw/Region.h"
#include "snnfw/Lobe.h"
#include "snnfw/Hemisphere.h"
#include "snnfw/Brain.h"
#include "snnfw/NetworkStructureExporter.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace snnfw {

// NetworkDataAdapter implementation
NetworkDataAdapter::NetworkDataAdapter(Datastore& datastore, 
                                     NetworkInspector& inspector,
                                     ActivityMonitor* activityMonitor)
    : datastore_(datastore)
    , inspector_(inspector)
    , activityMonitor_(activityMonitor) {
    
    // Default color scheme: blue for excitatory, red for inhibitory
    excitatoryColor_[0] = 0.2f;
    excitatoryColor_[1] = 0.5f;
    excitatoryColor_[2] = 1.0f;
    
    inhibitoryColor_[0] = 1.0f;
    inhibitoryColor_[1] = 0.2f;
    inhibitoryColor_[2] = 0.2f;
    
    // Default activity colors: dark blue for low, bright yellow for high
    lowActivityColor_[0] = 0.1f;
    lowActivityColor_[1] = 0.1f;
    lowActivityColor_[2] = 0.3f;
    
    highActivityColor_[0] = 1.0f;
    highActivityColor_[1] = 1.0f;
    highActivityColor_[2] = 0.2f;
}

void NetworkDataAdapter::setLayerColor(uint64_t layerId, float r, float g, float b) {
    layerColors_[layerId] = {r, g, b};
}

void NetworkDataAdapter::setClusterLayerIds(const std::vector<uint64_t>& clusterIds, uint64_t layerId) {
    // Update layerId for all neurons in the specified clusters
    for (auto& neuron : neurons_) {
        if (std::find(clusterIds.begin(), clusterIds.end(), neuron.clusterId) != clusterIds.end()) {
            neuron.layerId = layerId;
            // Reapply color with new layer ID
            applyNeuronColor(neuron);
        }
    }
}

bool NetworkDataAdapter::extractNetwork(uint64_t brainId) {
    clearCache();

    // Extract from brain level
    return extractHierarchy(brainId, "Brain");
}

bool NetworkDataAdapter::extractHierarchy(uint64_t rootId, const std::string& typeName) {
    clearCache();

    // Build hierarchical groups FIRST (needed by computeHierarchicalContext)
    buildHierarchicalGroups(rootId, typeName);

    // Extract neurons from this hierarchy
    extractNeurons(rootId, typeName);

    // Extract synapses connecting these neurons
    extractSynapses();

    return neurons_.size() > 0;
}

bool NetworkDataAdapter::extractMultipleClusters(const std::vector<uint64_t>& clusterIds) {
    clearCache();

    // Extract neurons from each cluster
    for (uint64_t clusterId : clusterIds) {
        auto cluster = datastore_.getCluster(clusterId);
        if (!cluster) {
            std::cerr << "Cluster " << clusterId << " not found!" << std::endl;
            continue;
        }

        auto neuronIds = cluster->getNeuronIds();
        for (uint64_t neuronId : neuronIds) {
            auto neuron = datastore_.getNeuron(neuronId);
            if (!neuron) continue;

            NeuronVisualData visualData;
            visualData.id = neuronId;
            visualData.radius = 0.5f;
            visualData.activity = 0.0f;
            visualData.isExcitatory = true;  // Assume excitatory for now
            visualData.position = neuron->getPosition();
            visualData.clusterId = clusterId;
            visualData.layerId = 0;  // Will be set by caller if needed

            // Apply color based on layer
            applyNeuronColor(visualData);

            neurons_.push_back(visualData);
            neuronIndexMap_[neuronId] = neurons_.size() - 1;
        }
    }

    // Extract synapses connecting these neurons
    extractSynapses();

    return neurons_.size() > 0;
}

void NetworkDataAdapter::extractNeurons(uint64_t rootId, const std::string& typeName) {
    // Get hierarchy statistics
    auto stats = inspector_.inspectHierarchy(rootId, typeName, datastore_);

    // Recursively collect all neuron IDs
    std::vector<uint64_t> neuronIds;
    std::function<void(uint64_t, const std::string&)> collectNeurons;

    collectNeurons = [&](uint64_t id, const std::string& type) {
        if (type == "Cluster") {
            // Get cluster and extract neuron IDs
            auto cluster = datastore_.getCluster(id);
            if (cluster) {
                auto clusterNeuronIds = cluster->getNeuronIds();
                neuronIds.insert(neuronIds.end(), clusterNeuronIds.begin(), clusterNeuronIds.end());
            }
        } else if (!type.empty()) {
            // Recurse to children (only if type is not empty)
            auto childStats = inspector_.inspectHierarchy(id, type, datastore_);
            // Make a copy of childIds before iterating, since recursive calls may modify it
            auto childIds = childStats.childIds;
            std::string childType = getChildType(type);
            // Only recurse if we have a valid child type
            if (!childType.empty()) {
                for (uint64_t childId : childIds) {
                    collectNeurons(childId, childType);
                }
            }
        }
    };

    collectNeurons(rootId, typeName);
    
    // Create NeuronVisualData for each neuron
    for (uint64_t neuronId : neuronIds) {
        auto neuron = datastore_.getNeuron(neuronId);
        if (!neuron) continue;

        NeuronVisualData visualData;
        visualData.id = neuronId;
        visualData.radius = 0.5f;  // Default radius, will be updated by layout
        visualData.activity = 0.0f;
        visualData.isExcitatory = true;  // Default, could be determined from neuron type

        // Use stored position if available
        if (neuron->hasPosition()) {
            visualData.position = neuron->getPosition();
        }
        // Otherwise position remains at (0,0,0) and will be set by LayoutEngine

        // Compute hierarchical context
        computeHierarchicalContext(visualData);

        // Apply color
        applyNeuronColor(visualData);

        // Add to collection
        neuronIndexMap_[neuronId] = neurons_.size();
        neurons_.push_back(visualData);
    }
}

void NetworkDataAdapter::extractSynapses() {
    // For each neuron, get its outgoing synapses
    for (const auto& neuron : neurons_) {
        auto neuronObj = datastore_.getNeuron(neuron.id);
        if (!neuronObj) continue;

        uint64_t axonId = neuronObj->getAxonId();
        auto axon = datastore_.getAxon(axonId);
        if (!axon) continue;

        auto synapseIds = axon->getSynapseIds();
        for (uint64_t synapseId : synapseIds) {
            auto synapse = datastore_.getSynapse(synapseId);
            if (!synapse) continue;

            uint64_t dendriteId = synapse->getDendriteId();
            auto dendrite = datastore_.getDendrite(dendriteId);
            if (!dendrite) continue;

            uint64_t targetNeuronId = dendrite->getTargetNeuronId();

            // Check if target neuron is in our visualization
            if (neuronIndexMap_.find(targetNeuronId) == neuronIndexMap_.end()) {
                continue;  // Target not in our set
            }

            SynapseVisualData visualData;
            visualData.id = synapseId;
            visualData.sourceNeuronId = neuron.id;
            visualData.targetNeuronId = targetNeuronId;
            visualData.weight = synapse->getWeight();
            visualData.thickness = 0.1f + std::abs(visualData.weight) * 0.5f;
            visualData.activity = 0.0f;

            // Set positions (will be updated by layout engine)
            visualData.sourcePos = neuron.position;

            // Find target neuron position
            auto targetIt = neuronIndexMap_.find(targetNeuronId);
            if (targetIt != neuronIndexMap_.end()) {
                visualData.targetPos = neurons_[targetIt->second].position;
            }

            // Apply color
            applySynapseColor(visualData);

            // Add to collection
            synapseIndexMap_[synapseId] = synapses_.size();
            synapses_.push_back(visualData);
        }
    }
}

void NetworkDataAdapter::buildHierarchicalGroups(uint64_t rootId, const std::string& typeName) {
    // Build groups recursively
    std::function<void(uint64_t, const std::string&, uint64_t)> buildGroups;

    buildGroups = [&](uint64_t id, const std::string& type, uint64_t parentId) {
        auto stats = inspector_.inspectHierarchy(id, type, datastore_);

        // Debug output
        std::cout << "    Building group: " << type << " " << id << " (" << stats.name << ") parent=" << parentId << std::endl;

        HierarchicalGroup group;
        group.id = id;
        group.typeName = type;
        group.name = stats.name;
        group.parentGroupId = parentId;
        
        // Collect neurons in this group
        if (type == "Cluster") {
            auto cluster = datastore_.getCluster(id);
            if (cluster) {
                // Get ALL neuron IDs from the cluster
                // Don't filter by neuronIndexMap_ because neurons haven't been extracted yet
                group.neuronIds = cluster->getNeuronIds();
            }
        }
        
        // Recurse to children (only if we have a valid child type)
        std::string childType = getChildType(type);
        if (!childType.empty()) {
            for (uint64_t childId : stats.childIds) {
                group.childGroupIds.push_back(childId);
                buildGroups(childId, childType, id);
            }
        }
        
        groups_.push_back(group);
    };
    
    buildGroups(rootId, typeName, 0);
}

void NetworkDataAdapter::computeHierarchicalContext(NeuronVisualData& neuron) {
    // Build a map of group ID to group for quick lookup
    std::unordered_map<uint64_t, const HierarchicalGroup*> groupMap;
    for (const auto& group : groups_) {
        groupMap[group.id] = &group;
    }

    // Find the cluster containing this neuron
    for (const auto& group : groups_) {
        if (group.typeName == "Cluster") {
            auto it = std::find(group.neuronIds.begin(), group.neuronIds.end(), neuron.id);
            if (it != group.neuronIds.end()) {
                neuron.clusterId = group.id;

                // Traverse up the hierarchy to find all parent levels
                uint64_t currentId = group.parentGroupId;
                while (currentId != 0) {
                    auto git = groupMap.find(currentId);
                    if (git == groupMap.end()) break;

                    const HierarchicalGroup* parent = git->second;
                    if (parent->typeName == "Layer") {
                        neuron.layerId = parent->id;
                    } else if (parent->typeName == "Column") {
                        neuron.columnId = parent->id;
                    } else if (parent->typeName == "Nucleus") {
                        neuron.nucleusId = parent->id;
                    } else if (parent->typeName == "Region") {
                        neuron.regionId = parent->id;
                    } else if (parent->typeName == "Lobe") {
                        neuron.lobeId = parent->id;
                    } else if (parent->typeName == "Hemisphere") {
                        neuron.hemisphereId = parent->id;
                    } else if (parent->typeName == "Brain") {
                        neuron.brainId = parent->id;
                    }

                    currentId = parent->parentGroupId;
                }
                break;
            }
        }
    }
}

void NetworkDataAdapter::updateActivity(double startTime, double endTime) {
    if (!activityMonitor_) return;
    
    // Get activity snapshot
    auto snapshot = activityMonitor_->getActivitySnapshot(startTime, endTime);
    
    // Update neuron activity
    for (auto& neuron : neurons_) {
        updateNeuronActivity(neuron, startTime, endTime);
    }
    
    // Update synapse activity
    for (auto& synapse : synapses_) {
        updateSynapseActivity(synapse, startTime, endTime);
    }
}

void NetworkDataAdapter::updateNeuronActivity(NeuronVisualData& neuron, double startTime, double endTime) {
    if (!activityMonitor_) return;
    
    auto snapshot = activityMonitor_->getActivitySnapshot(startTime, endTime);
    
    // Get spike count for this neuron
    auto it = snapshot.neuronSpikeCount.find(neuron.id);
    if (it != snapshot.neuronSpikeCount.end()) {
        // Normalize activity to 0-1 range (assuming max 100 spikes in window)
        neuron.activity = std::min(1.0f, static_cast<float>(it->second) / 100.0f);
    } else {
        neuron.activity = 0.0f;
    }
    
    // Update color based on activity
    applyNeuronColor(neuron);
}

void NetworkDataAdapter::updateSynapseActivity(SynapseVisualData& synapse, double startTime, double endTime) {
    if (!activityMonitor_) return;
    
    // Get spike events
    auto events = activityMonitor_->getSpikeEvents(startTime, endTime);
    
    // Count spikes through this synapse
    size_t spikeCount = 0;
    for (const auto& event : events) {
        if (event.synapseId == synapse.id) {
            spikeCount++;
        }
    }
    
    // Normalize activity
    synapse.activity = std::min(1.0f, static_cast<float>(spikeCount) / 10.0f);
    
    // Update color
    applySynapseColor(synapse);
}

void NetworkDataAdapter::applyNeuronColor(NeuronVisualData& neuron) {
    // Check if this neuron's layer has a custom color
    float baseColor[3];
    auto layerColorIt = layerColors_.find(neuron.layerId);
    if (layerColorIt != layerColors_.end()) {
        // Use layer-specific color
        baseColor[0] = layerColorIt->second[0];
        baseColor[1] = layerColorIt->second[1];
        baseColor[2] = layerColorIt->second[2];
    } else {
        // Use default color based on neuron type
        const float* defaultColor = neuron.isExcitatory ? excitatoryColor_ : inhibitoryColor_;
        baseColor[0] = defaultColor[0];
        baseColor[1] = defaultColor[1];
        baseColor[2] = defaultColor[2];
    }

    // Mix with activity color
    float activityMix = neuron.activity;

    neuron.r = baseColor[0] * (1.0f - activityMix) + highActivityColor_[0] * activityMix;
    neuron.g = baseColor[1] * (1.0f - activityMix) + highActivityColor_[1] * activityMix;
    neuron.b = baseColor[2] * (1.0f - activityMix) + highActivityColor_[2] * activityMix;
    neuron.a = 1.0f;
}

void NetworkDataAdapter::applySynapseColor(SynapseVisualData& synapse) {
    // Color based on weight (positive = bright cyan, negative = red)
    if (synapse.weight > 0) {
        synapse.r = 0.0f;
        synapse.g = 1.0f;
        synapse.b = 1.0f;  // Bright cyan for excitatory
    } else {
        synapse.r = 1.0f;
        synapse.g = 0.0f;
        synapse.b = 0.0f;  // Bright red for inhibitory
    }

    // Mix with activity
    float activityMix = synapse.activity;
    synapse.r = synapse.r * (1.0f - activityMix) + 1.0f * activityMix;
    synapse.g = synapse.g * (1.0f - activityMix) + 1.0f * activityMix;
    synapse.b = synapse.b * (1.0f - activityMix) + 0.2f * activityMix;

    // Higher alpha for better visibility
    synapse.a = 0.6f + std::min(0.4f, std::abs(synapse.weight) * 0.3f);
}

std::vector<NeuronVisualData> NetworkDataAdapter::getNeuronsByLevel(
    const std::string& levelType, uint64_t levelId) const {
    
    std::vector<NeuronVisualData> result;
    
    for (const auto& neuron : neurons_) {
        bool matches = false;
        
        if (levelType == "Cluster" && neuron.clusterId == levelId) matches = true;
        else if (levelType == "Layer" && neuron.layerId == levelId) matches = true;
        else if (levelType == "Column" && neuron.columnId == levelId) matches = true;
        
        if (matches) {
            result.push_back(neuron);
        }
    }
    
    return result;
}

std::vector<SynapseVisualData> NetworkDataAdapter::getSynapsesByNeuron(
    uint64_t neuronId, bool includeIncoming, bool includeOutgoing) const {
    
    std::vector<SynapseVisualData> result;
    
    for (const auto& synapse : synapses_) {
        if (includeOutgoing && synapse.sourceNeuronId == neuronId) {
            result.push_back(synapse);
        }
        if (includeIncoming && synapse.targetNeuronId == neuronId) {
            result.push_back(synapse);
        }
    }
    
    return result;
}

std::vector<LevelStatistics> NetworkDataAdapter::getLevelStatistics() const {
    // TODO: Implement level statistics computation
    return std::vector<LevelStatistics>();
}

void NetworkDataAdapter::setColorScheme(float excitatoryR, float excitatoryG, float excitatoryB,
                                       float inhibitoryR, float inhibitoryG, float inhibitoryB) {
    excitatoryColor_[0] = excitatoryR;
    excitatoryColor_[1] = excitatoryG;
    excitatoryColor_[2] = excitatoryB;
    
    inhibitoryColor_[0] = inhibitoryR;
    inhibitoryColor_[1] = inhibitoryG;
    inhibitoryColor_[2] = inhibitoryB;
}

void NetworkDataAdapter::setActivityColorMap(float lowActivityR, float lowActivityG, float lowActivityB,
                                            float highActivityR, float highActivityG, float highActivityB) {
    lowActivityColor_[0] = lowActivityR;
    lowActivityColor_[1] = lowActivityG;
    lowActivityColor_[2] = lowActivityB;
    
    highActivityColor_[0] = highActivityR;
    highActivityColor_[1] = highActivityG;
    highActivityColor_[2] = highActivityB;
}

void NetworkDataAdapter::updateSynapsePositions() {
    for (auto& synapse : synapses_) {
        // Find source neuron
        auto sourceIt = neuronIndexMap_.find(synapse.sourceNeuronId);
        if (sourceIt != neuronIndexMap_.end()) {
            synapse.sourcePos = neurons_[sourceIt->second].position;
        }

        // Find target neuron
        auto targetIt = neuronIndexMap_.find(synapse.targetNeuronId);
        if (targetIt != neuronIndexMap_.end()) {
            synapse.targetPos = neurons_[targetIt->second].position;
        }
    }
}

void NetworkDataAdapter::clearCache() {
    neurons_.clear();
    synapses_.clear();
    groups_.clear();
    neuronIndexMap_.clear();
    synapseIndexMap_.clear();
}

const NeuronVisualData* NetworkDataAdapter::findNeuron(uint64_t neuronId) const {
    auto it = neuronIndexMap_.find(neuronId);
    if (it != neuronIndexMap_.end()) {
        return &neurons_[it->second];
    }
    return nullptr;
}

const SynapseVisualData* NetworkDataAdapter::findSynapse(uint64_t synapseId) const {
    auto it = synapseIndexMap_.find(synapseId);
    if (it != synapseIndexMap_.end()) {
        return &synapses_[it->second];
    }
    return nullptr;
}

// Helper function to get child type in hierarchy
std::string NetworkDataAdapter::getChildType(const std::string& parentType) const {
    if (parentType == "Brain") return "Hemisphere";
    if (parentType == "Hemisphere") return "Lobe";
    if (parentType == "Lobe") return "Region";
    if (parentType == "Region") return "Nucleus";
    if (parentType == "Nucleus") return "Column";
    if (parentType == "Column") return "Layer";
    if (parentType == "Layer") return "Cluster";
    return "";
}

bool NetworkDataAdapter::exportNetworkStructure(const std::string& filename, const std::string& networkName) {
    NetworkStructureExporter exporter;

    // Set metadata
    NetworkStructureMetadata metadata;
    metadata.name = networkName.empty() ? "SNNFW Network" : networkName;

    // Generate ISO 8601 timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time_t_now), "%Y-%m-%dT%H:%M:%SZ");
    metadata.timestamp = ss.str();

    metadata.neuronCount = neurons_.size();
    metadata.synapseCount = synapses_.size();
    metadata.layerCount = layerColors_.size();

    exporter.setMetadata(metadata);

    // Export neurons
    for (const auto& neuron : neurons_) {
        ExportedNeuron exportedNeuron;
        exportedNeuron.id = neuron.id;
        exportedNeuron.position = neuron.position;
        exportedNeuron.radius = neuron.radius;
        exportedNeuron.r = neuron.r;
        exportedNeuron.g = neuron.g;
        exportedNeuron.b = neuron.b;
        exportedNeuron.a = neuron.a;
        exportedNeuron.isExcitatory = neuron.isExcitatory;
        exportedNeuron.clusterId = neuron.clusterId;
        exportedNeuron.layerId = neuron.layerId;
        exportedNeuron.columnId = neuron.columnId;
        exportedNeuron.nucleusId = neuron.nucleusId;
        exportedNeuron.regionId = neuron.regionId;
        exportedNeuron.lobeId = neuron.lobeId;
        exportedNeuron.hemisphereId = neuron.hemisphereId;
        exportedNeuron.brainId = neuron.brainId;

        exporter.addNeuron(exportedNeuron);
    }

    // Export synapses
    for (const auto& synapse : synapses_) {
        ExportedSynapse exportedSynapse;
        exportedSynapse.id = synapse.id;
        exportedSynapse.sourceNeuronId = synapse.sourceNeuronId;
        exportedSynapse.targetNeuronId = synapse.targetNeuronId;
        exportedSynapse.weight = synapse.weight;
        exportedSynapse.thickness = synapse.thickness;
        exportedSynapse.r = synapse.r;
        exportedSynapse.g = synapse.g;
        exportedSynapse.b = synapse.b;
        exportedSynapse.a = synapse.a;

        exporter.addSynapse(exportedSynapse);
    }

    // Export layers
    for (const auto& layerColor : layerColors_) {
        ExportedLayer exportedLayer;
        exportedLayer.id = layerColor.first;
        exportedLayer.name = "Layer_" + std::to_string(layerColor.first);
        exportedLayer.r = layerColor.second[0];
        exportedLayer.g = layerColor.second[1];
        exportedLayer.b = layerColor.second[2];

        exporter.addLayer(exportedLayer);
    }

    return exporter.save(filename);
}

} // namespace snnfw


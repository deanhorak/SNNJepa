#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Logger.h"
#include <sstream>
#include <algorithm>

namespace snnfw {

NeuralObjectFactory::NeuralObjectFactory()
    : neuronIdCounter(NEURON_ID_START),
      axonIdCounter(AXON_ID_START),
      dendriteIdCounter(DENDRITE_ID_START),
      synapseIdCounter(SYNAPSE_ID_START),
      clusterIdCounter(CLUSTER_ID_START),
      layerIdCounter(LAYER_ID_START),
      columnIdCounter(COLUMN_ID_START),
      nucleusIdCounter(NUCLEUS_ID_START),
      regionIdCounter(REGION_ID_START),
      lobeIdCounter(LOBE_ID_START),
      hemisphereIdCounter(HEMISPHERE_ID_START),
      brainIdCounter(BRAIN_ID_START) {
    SNNFW_DEBUG("NeuralObjectFactory initialized with ID ranges:");
    SNNFW_DEBUG("  Neurons:     {} - {}", NEURON_ID_START, NEURON_ID_END);
    SNNFW_DEBUG("  Axons:       {} - {}", AXON_ID_START, AXON_ID_END);
    SNNFW_DEBUG("  Dendrites:   {} - {}", DENDRITE_ID_START, DENDRITE_ID_END);
    SNNFW_DEBUG("  Synapses:    {} - {}", SYNAPSE_ID_START, SYNAPSE_ID_END);
    SNNFW_DEBUG("  Clusters:    {} - {}", CLUSTER_ID_START, CLUSTER_ID_END);
    SNNFW_DEBUG("  Layers:      {} - {}", LAYER_ID_START, LAYER_ID_END);
    SNNFW_DEBUG("  Columns:     {} - {}", COLUMN_ID_START, COLUMN_ID_END);
    SNNFW_DEBUG("  Nuclei:      {} - {}", NUCLEUS_ID_START, NUCLEUS_ID_END);
    SNNFW_DEBUG("  Regions:     {} - {}", REGION_ID_START, REGION_ID_END);
    SNNFW_DEBUG("  Lobes:       {} - {}", LOBE_ID_START, LOBE_ID_END);
    SNNFW_DEBUG("  Hemispheres: {} - {}", HEMISPHERE_ID_START, HEMISPHERE_ID_END);
    SNNFW_DEBUG("  Brains:      {} - {}", BRAIN_ID_START, BRAIN_ID_END);
}

std::shared_ptr<Neuron> NeuralObjectFactory::createNeuron(
    double windowSizeMs,
    double similarityThreshold,
    size_t maxReferencePatterns) {
    
    uint64_t id = getNextId(ObjectType::NEURON);
    const double clampedThreshold = std::clamp(similarityThreshold, 0.0, 1.0);
    if (clampedThreshold != similarityThreshold) {
        static uint64_t invalidThresholdWarnCount = 0;
        if (invalidThresholdWarnCount < 20) {
            invalidThresholdWarnCount++;
            SNNFW_WARN(
                "NeuralObjectFactory: Clamped invalid similarity threshold {} to {} for neuron {}",
                similarityThreshold, clampedThreshold, id);
        }
    }

    auto neuron = std::make_shared<Neuron>(
        windowSizeMs,
        clampedThreshold,
        maxReferencePatterns,
        id);
    
    SNNFW_DEBUG("Created Neuron with ID: {}", id);
    return neuron;
}

std::shared_ptr<Axon> NeuralObjectFactory::createAxon(uint64_t sourceNeuronId) {
    uint64_t id = getNextId(ObjectType::AXON);
    auto axon = std::make_shared<Axon>(sourceNeuronId, id);
    
    SNNFW_DEBUG("Created Axon with ID: {} (source neuron: {})", id, sourceNeuronId);
    return axon;
}

std::shared_ptr<Dendrite> NeuralObjectFactory::createDendrite(uint64_t targetNeuronId) {
    uint64_t id = getNextId(ObjectType::DENDRITE);
    auto dendrite = std::make_shared<Dendrite>(targetNeuronId, id);
    
    SNNFW_DEBUG("Created Dendrite with ID: {} (target neuron: {})", id, targetNeuronId);
    return dendrite;
}

std::shared_ptr<Synapse> NeuralObjectFactory::createSynapse(
    uint64_t axonId,
    uint64_t dendriteId,
    double weight,
    double delayMs) {
    
    uint64_t id = getNextId(ObjectType::SYNAPSE);
    auto synapse = std::make_shared<Synapse>(axonId, dendriteId, weight, delayMs, id);
    
    SNNFW_DEBUG("Created Synapse with ID: {} (axon: {}, dendrite: {}, weight: {}, delay: {}ms)",
                id, axonId, dendriteId, weight, delayMs);
    return synapse;
}

std::shared_ptr<Cluster> NeuralObjectFactory::createCluster() {
    uint64_t id = getNextId(ObjectType::CLUSTER);
    auto cluster = std::make_shared<Cluster>(id);

    SNNFW_DEBUG("Created Cluster with ID: {}", id);
    return cluster;
}

std::shared_ptr<Layer> NeuralObjectFactory::createLayer() {
    uint64_t id = getNextId(ObjectType::LAYER);
    auto layer = std::make_shared<Layer>(id);

    SNNFW_DEBUG("Created Layer with ID: {}", id);
    return layer;
}

std::shared_ptr<Column> NeuralObjectFactory::createColumn() {
    uint64_t id = getNextId(ObjectType::COLUMN);
    auto column = std::make_shared<Column>(id);

    SNNFW_DEBUG("Created Column with ID: {}", id);
    return column;
}

std::shared_ptr<Nucleus> NeuralObjectFactory::createNucleus() {
    uint64_t id = getNextId(ObjectType::NUCLEUS);
    auto nucleus = std::make_shared<Nucleus>(id);

    SNNFW_DEBUG("Created Nucleus with ID: {}", id);
    return nucleus;
}

std::shared_ptr<Region> NeuralObjectFactory::createRegion() {
    uint64_t id = getNextId(ObjectType::REGION);
    auto region = std::make_shared<Region>(id);

    SNNFW_DEBUG("Created Region with ID: {}", id);
    return region;
}

std::shared_ptr<Lobe> NeuralObjectFactory::createLobe() {
    uint64_t id = getNextId(ObjectType::LOBE);
    auto lobe = std::make_shared<Lobe>(id);

    SNNFW_DEBUG("Created Lobe with ID: {}", id);
    return lobe;
}

std::shared_ptr<Hemisphere> NeuralObjectFactory::createHemisphere() {
    uint64_t id = getNextId(ObjectType::HEMISPHERE);
    auto hemisphere = std::make_shared<Hemisphere>(id);

    SNNFW_DEBUG("Created Hemisphere with ID: {}", id);
    return hemisphere;
}

std::shared_ptr<Brain> NeuralObjectFactory::createBrain() {
    uint64_t id = getNextId(ObjectType::BRAIN);
    auto brain = std::make_shared<Brain>(id);

    SNNFW_DEBUG("Created Brain with ID: {}", id);
    return brain;
}

NeuralObjectFactory::ObjectType NeuralObjectFactory::getObjectType(uint64_t id) {
    if (id >= NEURON_ID_START && id <= NEURON_ID_END) {
        return ObjectType::NEURON;
    } else if (id >= AXON_ID_START && id <= AXON_ID_END) {
        return ObjectType::AXON;
    } else if (id >= DENDRITE_ID_START && id <= DENDRITE_ID_END) {
        return ObjectType::DENDRITE;
    } else if (id >= SYNAPSE_ID_START && id <= SYNAPSE_ID_END) {
        return ObjectType::SYNAPSE;
    } else if (id >= CLUSTER_ID_START && id <= CLUSTER_ID_END) {
        return ObjectType::CLUSTER;
    } else if (id >= LAYER_ID_START && id <= LAYER_ID_END) {
        return ObjectType::LAYER;
    } else if (id >= COLUMN_ID_START && id <= COLUMN_ID_END) {
        return ObjectType::COLUMN;
    } else if (id >= NUCLEUS_ID_START && id <= NUCLEUS_ID_END) {
        return ObjectType::NUCLEUS;
    } else if (id >= REGION_ID_START && id <= REGION_ID_END) {
        return ObjectType::REGION;
    } else if (id >= LOBE_ID_START && id <= LOBE_ID_END) {
        return ObjectType::LOBE;
    } else if (id >= HEMISPHERE_ID_START && id <= HEMISPHERE_ID_END) {
        return ObjectType::HEMISPHERE;
    } else if (id >= BRAIN_ID_START && id <= BRAIN_ID_END) {
        return ObjectType::BRAIN;
    } else {
        return ObjectType::UNKNOWN;
    }
}

const char* NeuralObjectFactory::getObjectTypeName(ObjectType type) {
    switch (type) {
        case ObjectType::NEURON:      return "Neuron";
        case ObjectType::AXON:        return "Axon";
        case ObjectType::DENDRITE:    return "Dendrite";
        case ObjectType::SYNAPSE:     return "Synapse";
        case ObjectType::CLUSTER:     return "Cluster";
        case ObjectType::LAYER:       return "Layer";
        case ObjectType::COLUMN:      return "Column";
        case ObjectType::NUCLEUS:     return "Nucleus";
        case ObjectType::REGION:      return "Region";
        case ObjectType::LOBE:        return "Lobe";
        case ObjectType::HEMISPHERE:  return "Hemisphere";
        case ObjectType::BRAIN:       return "Brain";
        case ObjectType::UNKNOWN:     return "Unknown";
        default:                      return "Invalid";
    }
}

const char* NeuralObjectFactory::getObjectTypeName(uint64_t id) {
    return getObjectTypeName(getObjectType(id));
}

void NeuralObjectFactory::reset() {
    std::lock_guard<std::mutex> lock(idMutex);

    neuronIdCounter = NEURON_ID_START;
    axonIdCounter = AXON_ID_START;
    dendriteIdCounter = DENDRITE_ID_START;
    synapseIdCounter = SYNAPSE_ID_START;
    clusterIdCounter = CLUSTER_ID_START;
    layerIdCounter = LAYER_ID_START;
    columnIdCounter = COLUMN_ID_START;
    nucleusIdCounter = NUCLEUS_ID_START;
    regionIdCounter = REGION_ID_START;
    lobeIdCounter = LOBE_ID_START;
    hemisphereIdCounter = HEMISPHERE_ID_START;
    brainIdCounter = BRAIN_ID_START;

    SNNFW_INFO("NeuralObjectFactory reset - all ID counters reset to start values");
}

uint64_t NeuralObjectFactory::getCurrentId(ObjectType type) const {
    std::lock_guard<std::mutex> lock(idMutex);

    switch (type) {
        case ObjectType::NEURON:      return neuronIdCounter;
        case ObjectType::AXON:        return axonIdCounter;
        case ObjectType::DENDRITE:    return dendriteIdCounter;
        case ObjectType::SYNAPSE:     return synapseIdCounter;
        case ObjectType::CLUSTER:     return clusterIdCounter;
        case ObjectType::LAYER:       return layerIdCounter;
        case ObjectType::COLUMN:      return columnIdCounter;
        case ObjectType::NUCLEUS:     return nucleusIdCounter;
        case ObjectType::REGION:      return regionIdCounter;
        case ObjectType::LOBE:        return lobeIdCounter;
        case ObjectType::HEMISPHERE:  return hemisphereIdCounter;
        case ObjectType::BRAIN:       return brainIdCounter;
        default:                      return 0;
    }
}

uint64_t NeuralObjectFactory::getObjectCount(ObjectType type) const {
    std::lock_guard<std::mutex> lock(idMutex);

    switch (type) {
        case ObjectType::NEURON:
            return neuronIdCounter - NEURON_ID_START;
        case ObjectType::AXON:
            return axonIdCounter - AXON_ID_START;
        case ObjectType::DENDRITE:
            return dendriteIdCounter - DENDRITE_ID_START;
        case ObjectType::SYNAPSE:
            return synapseIdCounter - SYNAPSE_ID_START;
        case ObjectType::CLUSTER:
            return clusterIdCounter - CLUSTER_ID_START;
        case ObjectType::LAYER:
            return layerIdCounter - LAYER_ID_START;
        case ObjectType::COLUMN:
            return columnIdCounter - COLUMN_ID_START;
        case ObjectType::NUCLEUS:
            return nucleusIdCounter - NUCLEUS_ID_START;
        case ObjectType::REGION:
            return regionIdCounter - REGION_ID_START;
        case ObjectType::LOBE:
            return lobeIdCounter - LOBE_ID_START;
        case ObjectType::HEMISPHERE:
            return hemisphereIdCounter - HEMISPHERE_ID_START;
        case ObjectType::BRAIN:
            return brainIdCounter - BRAIN_ID_START;
        default:
            return 0;
    }
}

uint64_t NeuralObjectFactory::getNextId(ObjectType type) {
    std::lock_guard<std::mutex> lock(idMutex);
    
    uint64_t* counter = nullptr;
    uint64_t maxId = 0;
    const char* typeName = getObjectTypeName(type);
    
    switch (type) {
        case ObjectType::NEURON:
            counter = &neuronIdCounter;
            maxId = NEURON_ID_END;
            break;
        case ObjectType::AXON:
            counter = &axonIdCounter;
            maxId = AXON_ID_END;
            break;
        case ObjectType::DENDRITE:
            counter = &dendriteIdCounter;
            maxId = DENDRITE_ID_END;
            break;
        case ObjectType::SYNAPSE:
            counter = &synapseIdCounter;
            maxId = SYNAPSE_ID_END;
            break;
        case ObjectType::CLUSTER:
            counter = &clusterIdCounter;
            maxId = CLUSTER_ID_END;
            break;
        case ObjectType::LAYER:
            counter = &layerIdCounter;
            maxId = LAYER_ID_END;
            break;
        case ObjectType::COLUMN:
            counter = &columnIdCounter;
            maxId = COLUMN_ID_END;
            break;
        case ObjectType::NUCLEUS:
            counter = &nucleusIdCounter;
            maxId = NUCLEUS_ID_END;
            break;
        case ObjectType::REGION:
            counter = &regionIdCounter;
            maxId = REGION_ID_END;
            break;
        case ObjectType::LOBE:
            counter = &lobeIdCounter;
            maxId = LOBE_ID_END;
            break;
        case ObjectType::HEMISPHERE:
            counter = &hemisphereIdCounter;
            maxId = HEMISPHERE_ID_END;
            break;
        case ObjectType::BRAIN:
            counter = &brainIdCounter;
            maxId = BRAIN_ID_END;
            break;
        default:
            SNNFW_ERROR("Invalid object type in getNextId");
            throw std::runtime_error("Invalid object type");
    }
    
    if (*counter > maxId) {
        std::ostringstream oss;
        oss << "ID range exhausted for " << typeName 
            << " (max: " << maxId << ")";
        SNNFW_ERROR("{}", oss.str());
        throw std::runtime_error(oss.str());
    }
    
    return (*counter)++;
}

} // namespace snnfw

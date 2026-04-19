#include "snnfw/ConnectivityBuilder.h"
#include "snnfw/Logger.h"
#include <stdexcept>

namespace snnfw {

ConnectivityBuilder::ConnectivityBuilder(NeuralObjectFactory& factory,
                                       Datastore& datastore,
                                       bool autoPersist)
    : factory_(factory),
      datastore_(datastore),
      autoPersist_(autoPersist) {
    SNNFW_DEBUG("ConnectivityBuilder created");
}

ConnectivityStats ConnectivityBuilder::connect(
    const std::vector<std::shared_ptr<Neuron>>& sourceNeurons,
    const std::vector<std::shared_ptr<Neuron>>& targetNeurons,
    ConnectivityPattern& pattern) {
    
    // Clear previous results
    clearCreatedObjects();
    
    ConnectivityStats stats;
    
    // Extract neuron IDs
    std::vector<uint64_t> sourceIds;
    std::vector<uint64_t> targetIds;
    sourceIds.reserve(sourceNeurons.size());
    targetIds.reserve(targetNeurons.size());
    
    for (const auto& neuron : sourceNeurons) {
        sourceIds.push_back(neuron->getId());
    }
    for (const auto& neuron : targetNeurons) {
        targetIds.push_back(neuron->getId());
    }
    
    // Generate connections using the pattern
    auto connections = pattern.generateConnections(sourceIds, targetIds);
    stats.connectionsRequested = connections.size();
    
    SNNFW_INFO("Applying connectivity pattern: {} connections requested", connections.size());
    
    // Create a map for quick neuron lookup
    std::map<uint64_t, std::shared_ptr<Neuron>> neuronMap;
    for (const auto& neuron : sourceNeurons) {
        neuronMap[neuron->getId()] = neuron;
    }
    for (const auto& neuron : targetNeurons) {
        neuronMap[neuron->getId()] = neuron;
    }
    
    // Create synapses for each connection
    for (const auto& conn : connections) {
        auto sourceIt = neuronMap.find(conn.sourceNeuronId);
        auto targetIt = neuronMap.find(conn.targetNeuronId);
        
        if (sourceIt == neuronMap.end() || targetIt == neuronMap.end()) {
            SNNFW_WARN("Connection references non-existent neuron, skipping");
            stats.failedConnections++;
            continue;
        }
        
        auto sourceNeuron = sourceIt->second;
        auto targetNeuron = targetIt->second;
        
        try {
            // Ensure source neuron has an axon
            uint64_t axonId = ensureAxon(sourceNeuron);
            
            // Create dendrite for target neuron
            uint64_t dendriteId = createDendrite(targetNeuron);
            
            // Create synapse
            auto synapse = factory_.createSynapse(axonId, dendriteId, conn.weight, conn.delay);
            createdSynapses_.push_back(synapse);
            persistObject(synapse);
            
            // Add synapse to axon and dendrite
            auto axon = datastore_.getAxon(axonId);
            auto dendrite = datastore_.getDendrite(dendriteId);
            
            if (axon) {
                axon->addSynapse(synapse->getId());
                persistObject(axon);
            }
            
            if (dendrite) {
                dendrite->addSynapse(synapse->getId());
                persistObject(dendrite);
            }
            
            stats.synapsesCreated++;
            
        } catch (const std::exception& e) {
            SNNFW_ERROR("Failed to create connection: {}", e.what());
            stats.failedConnections++;
        }
    }
    
    SNNFW_INFO("Connectivity complete: {} synapses, {} axons, {} dendrites created",
               stats.synapsesCreated, stats.axonsCreated, stats.dendritesCreated);
    
    return stats;
}

ConnectivityStats ConnectivityBuilder::connect(
    const std::vector<uint64_t>& sourceNeuronIds,
    const std::vector<uint64_t>& targetNeuronIds,
    ConnectivityPattern& pattern) {
    
    // Load neurons from datastore
    std::vector<std::shared_ptr<Neuron>> sourceNeurons;
    std::vector<std::shared_ptr<Neuron>> targetNeurons;
    
    sourceNeurons.reserve(sourceNeuronIds.size());
    targetNeurons.reserve(targetNeuronIds.size());
    
    for (uint64_t id : sourceNeuronIds) {
        auto neuron = datastore_.getNeuron(id);
        if (neuron) {
            sourceNeurons.push_back(neuron);
        } else {
            SNNFW_WARN("Source neuron {} not found in datastore", id);
        }
    }
    
    for (uint64_t id : targetNeuronIds) {
        auto neuron = datastore_.getNeuron(id);
        if (neuron) {
            targetNeurons.push_back(neuron);
        } else {
            SNNFW_WARN("Target neuron {} not found in datastore", id);
        }
    }
    
    return connect(sourceNeurons, targetNeurons, pattern);
}

void ConnectivityBuilder::clearCreatedObjects() {
    createdSynapses_.clear();
    createdAxons_.clear();
    createdDendrites_.clear();
}

uint64_t ConnectivityBuilder::ensureAxon(std::shared_ptr<Neuron> neuron) {
    // Check if neuron already has an axon
    if (neuron->getAxonId() != 0) {
        return neuron->getAxonId();
    }
    
    // Create new axon
    auto axon = factory_.createAxon(neuron->getId());
    neuron->setAxonId(axon->getId());
    
    createdAxons_.push_back(axon);
    persistObject(axon);
    persistObject(neuron);  // Update neuron with axon ID

    return axon->getId();
}

uint64_t ConnectivityBuilder::createDendrite(std::shared_ptr<Neuron> neuron) {
    auto dendrite = factory_.createDendrite(neuron->getId());
    neuron->addDendrite(dendrite->getId());
    
    createdDendrites_.push_back(dendrite);
    persistObject(dendrite);
    persistObject(neuron);  // Update neuron with dendrite
    
    return dendrite->getId();
}

void ConnectivityBuilder::persistObject(const std::shared_ptr<NeuralObject>& obj) {
    if (autoPersist_ && obj) {
        datastore_.put(obj);
        datastore_.markDirty(obj->getId());
    }
}

} // namespace snnfw


#ifndef SNNFW_CONNECTIVITY_BUILDER_H
#define SNNFW_CONNECTIVITY_BUILDER_H

#include "snnfw/ConnectivityPattern.h"
#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Datastore.h"
#include "snnfw/Neuron.h"
#include "snnfw/Axon.h"
#include "snnfw/Dendrite.h"
#include "snnfw/Synapse.h"
#include <vector>
#include <memory>

namespace snnfw {

/**
 * @struct ConnectivityStats
 * @brief Statistics about created connections
 */
struct ConnectivityStats {
    size_t connectionsRequested;  ///< Number of connections requested by pattern
    size_t synapsesCreated;       ///< Number of synapses actually created
    size_t axonsCreated;          ///< Number of axons created
    size_t dendritesCreated;      ///< Number of dendrites created
    size_t failedConnections;     ///< Number of connections that failed
    
    ConnectivityStats()
        : connectionsRequested(0), synapsesCreated(0), axonsCreated(0),
          dendritesCreated(0), failedConnections(0) {}
};

/**
 * @class ConnectivityBuilder
 * @brief Utility class for applying connectivity patterns to create synapses
 * 
 * ConnectivityBuilder takes a connectivity pattern and applies it to groups
 * of neurons, creating the necessary axons, dendrites, and synapses.
 * 
 * Example usage:
 * @code
 * ConnectivityBuilder builder(factory, datastore);
 * 
 * // Create random sparse connectivity
 * RandomSparsePattern pattern(0.2, 1.0, 1.5);  // 20% connectivity
 * auto stats = builder.connect(sourceNeurons, targetNeurons, pattern);
 * 
 * std::cout << "Created " << stats.synapsesCreated << " synapses" << std::endl;
 * @endcode
 */
class ConnectivityBuilder {
public:
    /**
     * @brief Constructor
     * @param factory Neural object factory for creating synapses
     * @param datastore Datastore for persisting objects
     * @param autoPersist Whether to automatically persist created objects (default: true)
     */
    ConnectivityBuilder(NeuralObjectFactory& factory, 
                       Datastore& datastore,
                       bool autoPersist = true);
    
    /**
     * @brief Apply a connectivity pattern to create connections
     * @param sourceNeurons Vector of source neuron shared pointers
     * @param targetNeurons Vector of target neuron shared pointers
     * @param pattern Connectivity pattern to apply
     * @return Statistics about created connections
     */
    ConnectivityStats connect(
        const std::vector<std::shared_ptr<Neuron>>& sourceNeurons,
        const std::vector<std::shared_ptr<Neuron>>& targetNeurons,
        ConnectivityPattern& pattern);
    
    /**
     * @brief Apply a connectivity pattern to create connections (ID-based)
     * @param sourceNeuronIds Vector of source neuron IDs
     * @param targetNeuronIds Vector of target neuron IDs
     * @param pattern Connectivity pattern to apply
     * @return Statistics about created connections
     */
    ConnectivityStats connect(
        const std::vector<uint64_t>& sourceNeuronIds,
        const std::vector<uint64_t>& targetNeuronIds,
        ConnectivityPattern& pattern);
    
    /**
     * @brief Set whether to automatically persist created objects
     * @param autoPersist True to auto-persist, false otherwise
     */
    void setAutoPersist(bool autoPersist) { autoPersist_ = autoPersist; }
    
    /**
     * @brief Get whether auto-persist is enabled
     * @return True if auto-persist is enabled
     */
    bool getAutoPersist() const { return autoPersist_; }
    
    /**
     * @brief Get all created synapses from the last connect() call
     * @return Vector of synapse shared pointers
     */
    const std::vector<std::shared_ptr<Synapse>>& getCreatedSynapses() const {
        return createdSynapses_;
    }
    
    /**
     * @brief Get all created axons from the last connect() call
     * @return Vector of axon shared pointers
     */
    const std::vector<std::shared_ptr<Axon>>& getCreatedAxons() const {
        return createdAxons_;
    }
    
    /**
     * @brief Get all created dendrites from the last connect() call
     * @return Vector of dendrite shared pointers
     */
    const std::vector<std::shared_ptr<Dendrite>>& getCreatedDendrites() const {
        return createdDendrites_;
    }
    
    /**
     * @brief Clear all cached created objects
     */
    void clearCreatedObjects();
    
private:
    /**
     * @brief Ensure a neuron has an axon, creating one if necessary
     * @param neuron Neuron to check
     * @return Axon ID
     */
    uint64_t ensureAxon(std::shared_ptr<Neuron> neuron);
    
    /**
     * @brief Create a dendrite for a neuron
     * @param neuron Neuron to create dendrite for
     * @return Dendrite ID
     */
    uint64_t createDendrite(std::shared_ptr<Neuron> neuron);
    
    /**
     * @brief Persist an object to datastore if auto-persist is enabled
     * @param obj Object to persist
     */
    void persistObject(const std::shared_ptr<NeuralObject>& obj);
    
    NeuralObjectFactory& factory_;
    Datastore& datastore_;
    bool autoPersist_;
    
    // Track created objects from last connect() call
    std::vector<std::shared_ptr<Synapse>> createdSynapses_;
    std::vector<std::shared_ptr<Axon>> createdAxons_;
    std::vector<std::shared_ptr<Dendrite>> createdDendrites_;
};

} // namespace snnfw

#endif // SNNFW_CONNECTIVITY_BUILDER_H


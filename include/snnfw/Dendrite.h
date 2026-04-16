#ifndef SNNFW_DENDRITE_H
#define SNNFW_DENDRITE_H

#include "snnfw/NeuralObject.h"
#include <cstdint>
#include <vector>
#include <memory>
#include <atomic>

namespace snnfw {

// Forward declarations
class ActionPotential;
class NetworkPropagator;

/**
 * @brief Dendrite class representing the input terminal of a neuron
 *
 * In biological neural networks, dendrites are branched projections of a
 * neuron that receive signals from other neurons via synapses. Dendrites
 * conduct the electrochemical stimulation received from other neurons to
 * the cell body (soma) of the neuron.
 *
 * This class models the dendrite as a neural object that:
 * - Belongs to a specific target neuron
 * - Receives action potentials from connected synapses
 * - Maintains a list of synapse IDs connected to it
 * - Processes incoming spikes and forwards them to the parent neuron
 *
 * Reference: Kandel, E. R., et al. (2013). Principles of Neural Science, 5th ed.
 */
class Dendrite : public NeuralObject {
public:
    /**
     * @brief Constructor
     * @param targetNeuronId ID of the neuron this dendrite belongs to
     * @param dendriteId Unique identifier for this dendrite (default: 0)
     */
    explicit Dendrite(uint64_t targetNeuronId, uint64_t dendriteId = 0)
        : NeuralObject(dendriteId), targetNeuronId(targetNeuronId) {}

    /**
     * @brief Get the ID of the target neuron
     * @return ID of the neuron this dendrite belongs to
     */
    uint64_t getTargetNeuronId() const { return targetNeuronId; }

    /**
     * @brief Set the ID of the target neuron
     * @param neuronId ID of the target neuron
     */
    void setTargetNeuronId(uint64_t neuronId) { targetNeuronId = neuronId; }

    /**
     * @brief Add a synapse connection to this dendrite
     * @param synapseId ID of the synapse connecting to this dendrite
     */
    void addSynapse(uint64_t synapseId);

    /**
     * @brief Remove a synapse connection from this dendrite
     * @param synapseId ID of the synapse to remove
     * @return true if synapse was found and removed, false otherwise
     */
    bool removeSynapse(uint64_t synapseId);

    /**
     * @brief Get all synapse IDs connected to this dendrite
     * @return Vector of synapse IDs
     */
    const std::vector<uint64_t>& getSynapseIds() const { return synapseIds; }

    /**
     * @brief Get the number of synapses connected to this dendrite
     * @return Number of connected synapses
     */
    size_t getSynapseCount() const { return synapseIds.size(); }

    /**
     * @brief Receive an action potential (spike) from a synapse
     *
     * This method is called by the SpikeProcessor when an action potential
     * arrives at this dendrite. The dendrite processes the spike and may
     * forward it to the parent neuron for integration.
     *
     * @param actionPotential The action potential being delivered
     */
    void receiveSpike(const std::shared_ptr<ActionPotential>& actionPotential);

    /**
     * @brief Get count of spikes received (thread-safe)
     * @return Number of spikes received since last reset
     */
    size_t getReceivedSpikeCount() const;

    /**
     * @brief Reset received spike counter (thread-safe)
     */
    void resetReceivedSpikeCount();

    /**
     * @brief Set the network propagator for spike delivery
     * @param propagator Weak pointer to the network propagator
     */
    void setNetworkPropagator(std::weak_ptr<NetworkPropagator> propagator) {
        networkPropagator_ = propagator;
    }

    // Serializable interface implementation
    std::string toJson() const override;
    bool fromJson(const std::string& json) override;
    std::string getTypeName() const override { return "Dendrite"; }

private:
    uint64_t targetNeuronId;              ///< ID of the neuron this dendrite belongs to
    std::vector<uint64_t> synapseIds;     ///< IDs of synapses connected to this dendrite
    std::weak_ptr<NetworkPropagator> networkPropagator_; ///< Weak pointer to network propagator for spike delivery
    std::atomic<size_t> receivedSpikeCount_{0}; ///< Count of spikes received since last reset
};

} // namespace snnfw

#endif // SNNFW_DENDRITE_H

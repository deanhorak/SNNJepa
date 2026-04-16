#ifndef SNNFW_AXON_H
#define SNNFW_AXON_H

#include "snnfw/NeuralObject.h"
#include <cstdint>
#include <vector>

namespace snnfw {

/**
 * @brief Axon class representing the output terminal of a neuron
 *
 * In biological neural networks, an axon is the long, slender projection
 * of a neuron that conducts electrical impulses (action potentials) away
 * from the neuron's cell body. A single axon can branch and form synapses
 * with multiple downstream dendrites.
 *
 * This class models the axon as a neural object that:
 * - Belongs to a specific source neuron
 * - Can connect to multiple synapses
 * - Transmits action potentials when the neuron fires
 *
 * Reference: Kandel, E. R., et al. (2013). Principles of Neural Science, 5th ed.
 */
class Axon : public NeuralObject {
public:
    /**
     * @brief Constructor
     * @param sourceNeuronId ID of the neuron this axon belongs to
     * @param axonId Unique identifier for this axon (default: 0)
     */
    explicit Axon(uint64_t sourceNeuronId, uint64_t axonId = 0)
        : NeuralObject(axonId), sourceNeuronId(sourceNeuronId) {}

    /**
     * @brief Get the ID of the source neuron
     * @return ID of the neuron this axon belongs to
     */
    uint64_t getSourceNeuronId() const { return sourceNeuronId; }

    /**
     * @brief Set the ID of the source neuron
     * @param neuronId ID of the source neuron
     */
    void setSourceNeuronId(uint64_t neuronId) { sourceNeuronId = neuronId; }

    /**
     * @brief Add a synapse connection to this axon
     * @param synapseId ID of the synapse connecting to this axon
     */
    void addSynapse(uint64_t synapseId);

    /**
     * @brief Remove a synapse connection from this axon
     * @param synapseId ID of the synapse to remove
     * @return true if synapse was found and removed, false otherwise
     */
    bool removeSynapse(uint64_t synapseId);

    /**
     * @brief Get all synapse IDs connected to this axon
     * @return Const reference to vector of synapse IDs
     */
    const std::vector<uint64_t>& getSynapseIds() const { return synapseIds; }

    /**
     * @brief Get the number of synapses connected to this axon
     * @return Number of connected synapses
     */
    size_t getSynapseCount() const { return synapseIds.size(); }

    // Serializable interface implementation
    std::string toJson() const override;
    bool fromJson(const std::string& json) override;
    std::string getTypeName() const override { return "Axon"; }

private:
    uint64_t sourceNeuronId;              ///< ID of the neuron this axon belongs to
    std::vector<uint64_t> synapseIds;     ///< IDs of synapses connected to this axon
};

} // namespace snnfw

#endif // SNNFW_AXON_H


#ifndef SNNFW_SYNAPSE_H
#define SNNFW_SYNAPSE_H

#include "snnfw/NeuralObject.h"
#include <cstdint>

namespace snnfw {

/**
 * @brief Synapse class representing the connection between an axon and a dendrite
 *
 * In biological neural networks, a synapse is a structure that permits a neuron
 * to pass an electrical or chemical signal to another neuron. Synapses are the
 * fundamental units of neural communication and plasticity.
 *
 * This class models the synapse as a neural object that:
 * - Connects a specific axon (presynaptic) to a specific dendrite (postsynaptic)
 * - Has a synaptic weight that modulates signal strength
 * - Has a synaptic delay representing transmission time
 * - Can be modified for learning (synaptic plasticity)
 *
 * The synapse serves as the link between the upstream axon and the downstream
 * dendrite, implementing the connectivity of the neural network.
 *
 * Reference: 
 * - Kandel, E. R., et al. (2013). Principles of Neural Science, 5th ed.
 * - Gerstner, W., & Kistler, W. M. (2002). Spiking Neuron Models.
 */
class Synapse : public NeuralObject {
public:
    /**
     * @brief Constructor
     * @param axonId ID of the presynaptic axon
     * @param dendriteId ID of the postsynaptic dendrite
     * @param weight Synaptic weight (default: 1.0)
     * @param delayMs Synaptic delay in milliseconds (default: 1.0)
     * @param synapseId Unique identifier for this synapse (default: 0)
     */
    explicit Synapse(uint64_t axonId, 
                    uint64_t dendriteId, 
                    double weight = 1.0,
                    double delayMs = 1.0,
                    uint64_t synapseId = 0)
        : NeuralObject(synapseId), 
          axonId(axonId), 
          dendriteId(dendriteId),
          weight(weight),
          delay(delayMs) {}

    /**
     * @brief Get the ID of the presynaptic axon
     * @return ID of the axon this synapse is connected to
     */
    uint64_t getAxonId() const { return axonId; }

    /**
     * @brief Get the ID of the postsynaptic dendrite
     * @return ID of the dendrite this synapse is connected to
     */
    uint64_t getDendriteId() const { return dendriteId; }

    /**
     * @brief Get the synaptic weight
     * @return Current synaptic weight
     */
    double getWeight() const { return weight; }

    /**
     * @brief Set the synaptic weight
     * @param newWeight New synaptic weight value
     */
    void setWeight(double newWeight) { weight = newWeight; }

    /**
     * @brief Get the synaptic delay
     * @return Synaptic delay in milliseconds
     */
    double getDelay() const { return delay; }

    /**
     * @brief Set the synaptic delay
     * @param delayMs New synaptic delay in milliseconds
     */
    void setDelay(double delayMs) { delay = delayMs; }

    /**
     * @brief Modify the synaptic weight (for learning/plasticity)
     * @param delta Amount to add to the current weight
     */
    void modifyWeight(double delta) { weight += delta; }

    // Serializable interface implementation
    std::string toJson() const override;
    bool fromJson(const std::string& json) override;
    std::string getTypeName() const override { return "Synapse"; }

private:
    uint64_t axonId;      ///< ID of the presynaptic axon
    uint64_t dendriteId;  ///< ID of the postsynaptic dendrite
    double weight;        ///< Synaptic weight (strength of connection)
    double delay;         ///< Synaptic delay in milliseconds
};

} // namespace snnfw

#endif // SNNFW_SYNAPSE_H


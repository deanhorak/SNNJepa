#ifndef SNNFW_SPIKE_ACKNOWLEDGMENT_H
#define SNNFW_SPIKE_ACKNOWLEDGMENT_H

#include "snnfw/EventObject.h"
#include <cstdint>

namespace snnfw {

/**
 * @brief SpikeAcknowledgment represents feedback from a postsynaptic neuron to presynaptic neurons
 *
 * When a postsynaptic neuron fires, it sends acknowledgments back to all presynaptic neurons
 * that contributed spikes within the temporal window (e.g., 200ms). This feedback enables
 * Spike-Timing-Dependent Plasticity (STDP), where synaptic weights are adjusted based on
 * the relative timing of pre- and post-synaptic spikes.
 *
 * STDP Learning Rule:
 * - If presynaptic spike arrives BEFORE postsynaptic spike: strengthen synapse (LTP)
 * - If presynaptic spike arrives AFTER postsynaptic spike: weaken synapse (LTD)
 * - The magnitude of change decreases exponentially with time difference
 *
 * Biological Motivation:
 * STDP is a fundamental learning mechanism in biological neural networks, discovered by
 * Bi & Poo (1998). It implements Hebb's rule: "neurons that fire together, wire together"
 * with precise temporal causality.
 *
 * Reference:
 * - Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons.
 * - Song, S., Miller, K. D., & Abbott, L. F. (2000). Competitive Hebbian learning.
 */
class SpikeAcknowledgment : public EventObject {
public:
    /**
     * @brief Constructor
     * @param synapseId ID of the synapse that delivered the spike
     * @param postsynapticNeuronId ID of the neuron that fired (sending the acknowledgment)
     * @param postsynapticFiringTime Time when the postsynaptic neuron fired
     * @param presynapticSpikeTime Time when the presynaptic spike arrived
     */
    SpikeAcknowledgment(uint64_t synapseId,
                       uint64_t postsynapticNeuronId,
                       double postsynapticFiringTime,
                       double presynapticSpikeTime)
        : EventObject(0),  // Acknowledgments don't need unique IDs
          synapseId_(synapseId),
          postsynapticNeuronId_(postsynapticNeuronId),
          postsynapticFiringTime_(postsynapticFiringTime),
          presynapticSpikeTime_(presynapticSpikeTime) {}

    /**
     * @brief Get the synapse ID
     * @return ID of the synapse that should receive this acknowledgment
     */
    uint64_t getSynapseId() const { return synapseId_; }

    /**
     * @brief Get the postsynaptic neuron ID
     * @return ID of the neuron that fired
     */
    uint64_t getPostsynapticNeuronId() const { return postsynapticNeuronId_; }

    /**
     * @brief Get the postsynaptic firing time
     * @return Time when the postsynaptic neuron fired
     */
    double getPostsynapticFiringTime() const { return postsynapticFiringTime_; }

    /**
     * @brief Get the presynaptic spike time
     * @return Time when the presynaptic spike arrived at the dendrite
     */
    double getPresynapticSpikeTime() const { return presynapticSpikeTime_; }

    /**
     * @brief Calculate the spike time difference (Δt = t_post - t_pre)
     * @return Time difference in milliseconds (positive if pre→post, negative if post→pre)
     */
    double getTimeDifference() const {
        return postsynapticFiringTime_ - presynapticSpikeTime_;
    }

    // EventObject interface implementation
    const char* getEventType() const override { return "SpikeAcknowledgment"; }

private:
    uint64_t synapseId_;                ///< ID of the synapse to update
    uint64_t postsynapticNeuronId_;     ///< ID of the neuron that fired
    double postsynapticFiringTime_;     ///< Time when postsynaptic neuron fired
    double presynapticSpikeTime_;       ///< Time when presynaptic spike arrived
};

} // namespace snnfw

#endif // SNNFW_SPIKE_ACKNOWLEDGMENT_H


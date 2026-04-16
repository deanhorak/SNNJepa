#ifndef SNNFW_RETROGRADE_ACTION_POTENTIAL_H
#define SNNFW_RETROGRADE_ACTION_POTENTIAL_H

#include "snnfw/EventObject.h"
#include <cstdint>

namespace snnfw {

/**
 * @brief RetrogradeActionPotential represents a retrograde signal from postsynaptic neuron to synapse
 *
 * In biological neural networks, retrograde signaling is a form of synaptic plasticity where
 * the postsynaptic neuron sends signals back to the presynaptic terminal. This is crucial for
 * Spike-Timing-Dependent Plasticity (STDP), where synaptic weights are modified based on the
 * relative timing of pre- and post-synaptic spikes.
 *
 * This class models a retrograde signal that:
 * - Originates from a postsynaptic neuron when it fires
 * - Is destined for a specific synapse
 * - Carries timing information for STDP learning
 * - Travels backward through the network (dendrite → synapse)
 *
 * STDP Learning Rule:
 * - If pre-spike arrives BEFORE post-spike (positive Δt): LTP (strengthen synapse)
 * - If pre-spike arrives AFTER post-spike (negative Δt): LTD (weaken synapse)
 * - Magnitude decreases exponentially with |Δt|
 *
 * Timing Calculation:
 * - temporalOffset = lastFiringTime - dispatchTime
 * - If temporalOffset >= 0: neuron fired AFTER the spike was sent → LTP
 * - If temporalOffset < 0: neuron fired BEFORE the spike was sent → LTD
 *
 * Biological Motivation:
 * Retrograde signaling is observed in biological synapses through mechanisms like:
 * - Endocannabinoid signaling
 * - Nitric oxide (NO) diffusion
 * - Brain-Derived Neurotrophic Factor (BDNF)
 * These signals enable the synapse to adjust its strength based on correlation between
 * pre- and post-synaptic activity.
 *
 * Reference:
 * - Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons.
 * - Sjöström, P. J., & Gerstner, W. (2010). Spike-timing dependent plasticity.
 * - Regehr, W. G., et al. (2009). Retrograde signaling in the regulation of synaptic transmission.
 */
class RetrogradeActionPotential : public EventObject {
public:
    /**
     * @brief Constructor
     * @param synapseId ID of the synapse to which this retrograde signal is destined
     * @param postsynapticNeuronId ID of the neuron that fired (sending the retrograde signal)
     * @param scheduledTimeMs Time when this retrograde signal should arrive at the synapse (in ms)
     * @param dispatchTime Time when the original forward spike was dispatched (in ms)
     * @param lastFiringTime Time when the postsynaptic neuron last fired (in ms)
     */
    explicit RetrogradeActionPotential(uint64_t synapseId,
                                      uint64_t postsynapticNeuronId,
                                      double scheduledTimeMs,
                                      double dispatchTime,
                                      double lastFiringTime)
        : EventObject(scheduledTimeMs),
          synapseId(synapseId),
          postsynapticNeuronId(postsynapticNeuronId),
          dispatchTime(dispatchTime),
          lastFiringTime(lastFiringTime) {}

    /**
     * @brief Get the ID of the target synapse
     * @return ID of the synapse that should receive this retrograde signal
     */
    uint64_t getSynapseId() const { return synapseId; }

    /**
     * @brief Get the ID of the postsynaptic neuron
     * @return ID of the neuron that fired and sent this retrograde signal
     */
    uint64_t getPostsynapticNeuronId() const { return postsynapticNeuronId; }

    /**
     * @brief Get the dispatch time of the original forward spike
     * @return Time when the forward ActionPotential was dispatched (in ms)
     */
    double getDispatchTime() const { return dispatchTime; }

    /**
     * @brief Get the last firing time of the postsynaptic neuron
     * @return Time when the postsynaptic neuron last fired (in ms)
     */
    double getLastFiringTime() const { return lastFiringTime; }

    /**
     * @brief Calculate the temporal offset for STDP
     * 
     * Temporal offset = lastFiringTime - dispatchTime
     * - If >= 0: neuron fired AFTER spike was sent → LTP (strengthen)
     * - If < 0: neuron fired BEFORE spike was sent → LTD (weaken)
     * 
     * @return Temporal offset in milliseconds
     */
    double getTemporalOffset() const {
        return lastFiringTime - dispatchTime;
    }

    /**
     * @brief Get the type of this event
     * @return String identifier "RetrogradeActionPotential"
     */
    const char* getEventType() const override { return "RetrogradeActionPotential"; }

private:
    uint64_t synapseId;              ///< ID of the synapse to receive this retrograde signal
    uint64_t postsynapticNeuronId;   ///< ID of the neuron that fired
    double dispatchTime;             ///< Time when the original forward spike was dispatched
    double lastFiringTime;           ///< Time when the postsynaptic neuron last fired
};

} // namespace snnfw

#endif // SNNFW_RETROGRADE_ACTION_POTENTIAL_H


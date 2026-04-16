#ifndef SNNFW_ACTION_POTENTIAL_H
#define SNNFW_ACTION_POTENTIAL_H

#include "snnfw/EventObject.h"
#include <cstdint>

namespace snnfw {

/**
 * @brief ActionPotential class representing a spike traveling through the network
 *
 * In biological neural networks, an action potential (also called a nerve impulse
 * or spike) is a rapid rise and fall in voltage or membrane potential across a
 * cellular membrane. Action potentials are the fundamental units of neural
 * communication, propagating along axons and across synapses.
 *
 * This class models an action potential as an event that:
 * - Originates from a specific synapse
 * - Is destined for a specific dendrite
 * - Has a scheduled delivery time (accounting for synaptic delay)
 * - Carries information about its source and destination
 *
 * The ActionPotential is created when a neuron fires and is scheduled for
 * delivery by the SpikeProcessor at the appropriate time.
 *
 * Reference:
 * - Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of 
 *   membrane current and its application to conduction and excitation in nerve.
 * - Kandel, E. R., et al. (2013). Principles of Neural Science, 5th ed.
 */
class ActionPotential : public EventObject {
public:
    /**
     * @brief Constructor
     * @param synapseId ID of the synapse from which this action potential originated
     * @param dendriteId ID of the dendrite to which this action potential is destined
     * @param scheduledTimeMs Time when this action potential should arrive (in ms)
     * @param amplitude Amplitude/strength of the action potential (default: 1.0)
     * @param dispatchTime Time when this action potential was dispatched (default: 0.0)
     */
    explicit ActionPotential(uint64_t synapseId,
                            uint64_t dendriteId,
                            double scheduledTimeMs,
                            double amplitude = 1.0,
                            double dispatchTime = 0.0)
        : EventObject(scheduledTimeMs),
          synapseId(synapseId),
          dendriteId(dendriteId),
          amplitude(amplitude),
          dispatchTime(dispatchTime) {}

    /**
     * @brief Get the ID of the originating synapse
     * @return ID of the synapse that generated this action potential
     */
    uint64_t getSynapseId() const { return synapseId; }

    /**
     * @brief Get the ID of the destination dendrite
     * @return ID of the dendrite this action potential is traveling to
     */
    uint64_t getDendriteId() const { return dendriteId; }

    /**
     * @brief Get the amplitude of the action potential
     * @return Amplitude value (typically modulated by synaptic weight)
     */
    double getAmplitude() const { return amplitude; }

    /**
     * @brief Set the amplitude of the action potential
     * @param amp New amplitude value
     */
    void setAmplitude(double amp) { amplitude = amp; }

    /**
     * @brief Get the dispatch time of this action potential
     * @return Time when this action potential was dispatched (in ms)
     */
    double getDispatchTime() const { return dispatchTime; }

    /**
     * @brief Set the dispatch time of this action potential
     * @param time Time when this action potential was dispatched (in ms)
     */
    void setDispatchTime(double time) { dispatchTime = time; }

    /**
     * @brief Get the type of this event
     * @return String identifier "ActionPotential"
     */
    const char* getEventType() const override { return "ActionPotential"; }

private:
    uint64_t synapseId;   ///< ID of the synapse from which this AP originated
    uint64_t dendriteId;  ///< ID of the dendrite to which this AP is destined
    double amplitude;     ///< Amplitude/strength of the action potential
    double dispatchTime;  ///< Time when this action potential was dispatched
};

} // namespace snnfw

#endif // SNNFW_ACTION_POTENTIAL_H


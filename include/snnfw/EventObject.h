#ifndef SNNFW_EVENT_OBJECT_H
#define SNNFW_EVENT_OBJECT_H

#include <cstdint>

namespace snnfw {

/**
 * @brief Base class for all event objects in the neural network
 *
 * EventObject serves as the base class for various types of events that
 * occur in the spiking neural network, such as action potentials, synaptic
 * events, and other time-based occurrences.
 *
 * Events are characterized by their scheduled delivery time and can carry
 * additional type-specific information in derived classes.
 */
class EventObject {
public:
    /**
     * @brief Constructor with scheduled time
     * @param scheduledTimeMs The time (in milliseconds) when this event should be delivered
     */
    explicit EventObject(double scheduledTimeMs = 0.0)
        : scheduledTime(scheduledTimeMs) {}

    /**
     * @brief Virtual destructor for proper cleanup of derived classes
     */
    virtual ~EventObject() = default;

    /**
     * @brief Get the scheduled delivery time for this event
     * @return Scheduled time in milliseconds
     */
    double getScheduledTime() const { return scheduledTime; }

    /**
     * @brief Set the scheduled delivery time for this event
     * @param timeMs Scheduled time in milliseconds
     */
    void setScheduledTime(double timeMs) { scheduledTime = timeMs; }

    /**
     * @brief Get the type of this event (for runtime type identification)
     * @return String identifier for the event type
     */
    virtual const char* getEventType() const { return "EventObject"; }

protected:
    double scheduledTime;  ///< Time when this event should be delivered (in ms)
};

} // namespace snnfw

#endif // SNNFW_EVENT_OBJECT_H


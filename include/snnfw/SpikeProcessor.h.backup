#ifndef SNNFW_SPIKE_PROCESSOR_H
#define SNNFW_SPIKE_PROCESSOR_H

#include "snnfw/EventObject.h"
#include "snnfw/ActionPotential.h"
#include "snnfw/Dendrite.h"
#include "snnfw/ThreadPool.h"
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <map>

namespace snnfw {

/**
 * @brief SpikeProcessor manages the delivery of action potentials in the network
 *
 * The SpikeProcessor is a critical component that runs as a background thread,
 * managing the temporal delivery of action potentials (spikes) to their target
 * dendrites. It implements a time-sliced event queue where each time slice
 * represents 1 millisecond of simulation time.
 *
 * Key Features:
 * - Time-sliced event queue (default: 10,000 x 1ms = 10 seconds of buffering)
 * - Parallel spike delivery using thread pool
 * - Even workload distribution across worker threads
 * - Thread-safe spike scheduling and delivery
 * - Configurable number of delivery threads
 *
 * Architecture:
 * - Outer vector: Time slices (each representing 1ms)
 * - Inner vectors: Action potentials scheduled for that time slice
 * - Background thread: Advances simulation time and triggers deliveries
 * - Thread pool: Distributes spike delivery across multiple threads
 *
 * Biological Motivation:
 * In biological neural networks, action potentials propagate with specific
 * delays determined by axon length, myelination, and synaptic transmission
 * time. This processor simulates these delays by scheduling spikes for
 * future delivery.
 *
 * Reference:
 * - Brette, R., et al. (2007). Simulation of networks of spiking neurons.
 * - Gewaltig, M. O., & Diesmann, M. (2007). NEST (NEural Simulation Tool).
 */
class SpikeProcessor {
public:
    /**
     * @brief Constructor
     * @param timeSliceCount Number of time slices to buffer (default: 10000 = 10 seconds)
     * @param deliveryThreads Number of threads for parallel delivery (default: 20)
     */
    explicit SpikeProcessor(size_t timeSliceCount = 10000, size_t deliveryThreads = 20);

    /**
     * @brief Destructor - stops the processor and cleans up
     */
    ~SpikeProcessor();

    // Prevent copying
    SpikeProcessor(const SpikeProcessor&) = delete;
    SpikeProcessor& operator=(const SpikeProcessor&) = delete;

    /**
     * @brief Start the spike processor background thread
     */
    void start();

    /**
     * @brief Stop the spike processor background thread
     */
    void stop();

    /**
     * @brief Check if the processor is running
     * @return true if running, false otherwise
     */
    bool isRunning() const { return running.load(); }

    /**
     * @brief Schedule an action potential for delivery
     * @param actionPotential The action potential to schedule
     * @return true if scheduled successfully, false if time is out of range
     */
    bool scheduleSpike(const std::shared_ptr<ActionPotential>& actionPotential);

    /**
     * @brief Register a dendrite for spike delivery
     * @param dendrite Shared pointer to the dendrite
     */
    void registerDendrite(const std::shared_ptr<Dendrite>& dendrite);

    /**
     * @brief Unregister a dendrite
     * @param dendriteId ID of the dendrite to unregister
     */
    void unregisterDendrite(uint64_t dendriteId);

    /**
     * @brief Get the current simulation time
     * @return Current time in milliseconds
     */
    double getCurrentTime() const { return currentTime.load(); }

    /**
     * @brief Set the simulation time step (default: 1.0 ms)
     * @param stepMs Time step in milliseconds
     */
    void setTimeStep(double stepMs) { timeStep = stepMs; }

    /**
     * @brief Get the time step
     * @return Time step in milliseconds
     */
    double getTimeStep() const { return timeStep; }

    /**
     * @brief Get the number of pending spikes across all time slices
     * @return Total number of scheduled spikes
     */
    size_t getPendingSpikeCount() const;

    /**
     * @brief Get the number of spikes in a specific time slice
     * @param timeSliceIndex Index of the time slice
     * @return Number of spikes in that time slice
     */
    size_t getSpikeCountAtSlice(size_t timeSliceIndex) const;

    /**
     * @brief Enable or disable real-time synchronization
     * @param enable true to sync with wall-clock time, false to run as fast as possible
     */
    void setRealTimeSync(bool enable) { realTimeSync = enable; }

    /**
     * @brief Check if real-time synchronization is enabled
     * @return true if enabled, false otherwise
     */
    bool isRealTimeSync() const { return realTimeSync; }

    /**
     * @brief Get timing statistics
     * @param avgLoopTime Average loop time in microseconds
     * @param maxLoopTime Maximum loop time in microseconds
     * @param driftMs Accumulated drift in milliseconds
     */
    void getTimingStats(double& avgLoopTime, double& maxLoopTime, double& driftMs) const;

private:
    /**
     * @brief Main processing loop (runs in background thread)
     */
    void processingLoop();

    /**
     * @brief Deliver all spikes for the current time slice
     */
    void deliverCurrentSlice();

    /**
     * @brief Get the time slice index for a given time
     * @param timeMs Time in milliseconds
     * @return Time slice index, or -1 if out of range
     */
    int getTimeSliceIndex(double timeMs) const;

    // Configuration
    size_t numTimeSlices;           ///< Number of time slices in the buffer
    size_t numDeliveryThreads;      ///< Number of threads for spike delivery
    double timeStep;                ///< Time step in milliseconds (default: 1.0)

    // Event queue: outer vector = time slices, inner vector = events in that slice
    std::vector<std::vector<std::shared_ptr<ActionPotential>>> eventQueue;
    mutable std::mutex queueMutex;  ///< Protects the event queue

    // Dendrite registry: maps dendrite ID to dendrite object
    std::map<uint64_t, std::shared_ptr<Dendrite>> dendriteRegistry;
    mutable std::mutex registryMutex;  ///< Protects the dendrite registry

    // Thread pool for parallel spike delivery
    std::unique_ptr<ThreadPool> threadPool;

    // Background processing thread
    std::thread processingThread;
    std::atomic<bool> running;      ///< Flag indicating if processor is running
    std::atomic<bool> stopRequested; ///< Flag to request stop
    std::condition_variable cv;     ///< Condition variable for thread synchronization
    std::mutex cvMutex;             ///< Mutex for condition variable

    // Simulation time
    std::atomic<double> currentTime; ///< Current simulation time in milliseconds
    size_t currentSliceIndex;        ///< Current time slice index

    // Real-time synchronization
    bool realTimeSync;               ///< Whether to sync with wall-clock time
    std::chrono::steady_clock::time_point startWallTime; ///< Wall-clock start time

    // Timing statistics
    mutable std::mutex statsMutex;   ///< Protects timing statistics
    double totalLoopTime;            ///< Total accumulated loop time (microseconds)
    double maxLoopTime;              ///< Maximum loop time (microseconds)
    uint64_t loopCount;              ///< Number of loops executed
    double accumulatedDrift;         ///< Accumulated time drift (milliseconds)
};

} // namespace snnfw

#endif // SNNFW_SPIKE_PROCESSOR_H


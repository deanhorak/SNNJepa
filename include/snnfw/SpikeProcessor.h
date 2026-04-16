#ifndef SNNFW_SPIKE_PROCESSOR_H
#define SNNFW_SPIKE_PROCESSOR_H

#include "snnfw/EventObject.h"
#include "snnfw/ActionPotential.h"
#include "snnfw/RetrogradeActionPotential.h"
#include "snnfw/Dendrite.h"
#include "snnfw/Synapse.h"
#include "snnfw/ThreadPool.h"
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <map>
#include <unordered_set>

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
 * - Asynchronous parallel spike delivery using thread pool
 * - Even workload distribution across worker threads
 * - Thread-safe spike scheduling and delivery
 * - Configurable number of delivery threads
 * - Non-blocking time advancement for real-time synchronization
 * - Non-blocking time advancement for real-time synchronization
 *
 * Architecture:
 * - Outer vector: Time slices (each representing 1ms)
 * - Inner vectors: Action potentials scheduled for that time slice
 * - Background thread: Advances simulation time at steady 1ms intervals
 * - Async delivery threads: Process each timeslice independently
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
     * @brief Schedule a retrograde action potential for delivery
     * @param retrogradeAP The retrograde action potential to schedule
     * @return true if scheduled successfully, false if time is out of range
     */
    bool scheduleRetrogradeSpike(const std::shared_ptr<RetrogradeActionPotential>& retrogradeAP);

    /**
     * @brief Register a dendrite for spike delivery
     * @param dendrite Shared pointer to the dendrite
     */
    void registerDendrite(const std::shared_ptr<Dendrite>& dendrite);

    /**
     * @brief Register a synapse for retrograde spike delivery
     * @param synapse Shared pointer to the synapse
     */
    void registerSynapse(const std::shared_ptr<Synapse>& synapse);

    /**
     * @brief Unregister a dendrite
     * @param dendriteId ID of the dendrite to unregister
     */
    void unregisterDendrite(uint64_t dendriteId);

    struct DeliveryStats {
        uint64_t scheduled = 0;
        uint64_t scheduledInputToL4 = 0;
        uint64_t delivered = 0;
        uint64_t deliveredInputToL4 = 0;
        uint64_t missingDendrite = 0;
        uint64_t missingDendriteInputToL4 = 0;
    };

    void resetDeliveryStats();

    DeliveryStats getDeliveryStats() const;

    void setInputToL4NeuronIds(const std::unordered_set<uint64_t>& neuronIds);

    /**
     * @brief Get the current simulation time
     * @return Current time in milliseconds
     */
    double getCurrentTime() const { return currentTime.load(); }

    /**
     * @brief Get the maximum schedule-ahead window in milliseconds
     */
    double getMaxScheduleAheadMs() const { return numTimeSlices * timeStep; }

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
     * @brief Set activity monitor for automatic spike recording
     * @param monitor Pointer to activity monitor (nullptr to disable)
     */
    void setActivityMonitor(class ActivityMonitor* monitor);

    /**
     * @brief Get the number of pending spikes across all time slices
     * @return Total number of scheduled spikes
     */
    size_t getPendingSpikeCount() const;

    /**
     * @brief Purge scheduled action potentials targeting selected dendrites before a cutoff time
     * @param cutoffTime Remove spikes with scheduled time < cutoffTime
     * @param dendriteIds Dendrite IDs to purge
     * @return Number of events removed
     */
    size_t purgeEventsBefore(double cutoffTime, const std::unordered_set<uint64_t>& dendriteIds);

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

    /**
     * @brief Set STDP learning parameters
     * @param aPlus LTP amplitude (default: 0.01)
     * @param aMinus LTD amplitude (default: 0.012)
     * @param tauPlus LTP time constant in ms (default: 20.0)
     * @param tauMinus LTD time constant in ms (default: 20.0)
     */
    void setSTDPParameters(double aPlus, double aMinus, double tauPlus, double tauMinus);

    /**
     * @brief Enable or disable all STDP learning
     * @param enabled When false, no weight updates occur (useful for inference)
     */
    void setStdpEnabled(bool enabled);

    /**
     * @brief Check if STDP learning is currently enabled
     * @return true if STDP is enabled, false otherwise
     */
    bool isStdpEnabled() const;

    /**
     * @brief Get the number of active delivery threads
     * @return Number of threads currently processing timeslices
     */
    size_t getActiveDeliveryThreadCount() const;

private:
    /**
     * @brief Main processing loop (runs in background thread)
     */
    void processingLoop();

    /**
     * @brief Deliver all spikes for a specific time slice (runs asynchronously)
     * @param sliceIndex The time slice index to deliver
     * @param simTime The simulation time for this slice
     */
    void deliverSliceAsync(size_t sliceIndex, double simTime);

    /**
     * @brief Apply STDP weight update to a synapse
     * @param synapse The synapse to update
     * @param temporalOffset Time difference (lastFiringTime - dispatchTime)
     */
    void applySTDPToSynapse(std::shared_ptr<Synapse> synapse, double temporalOffset);

    /**
     * @brief Cleanup completed delivery threads
     */
    void cleanupCompletedThreads();

    /**
     * @brief Wait for all active delivery threads to complete
     * Used in non-real-time mode to ensure spike propagation completes before advancing time
     */
    void waitForDeliveryThreads();

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
    // Events can be either ActionPotential or RetrogradeActionPotential
    std::vector<std::vector<std::shared_ptr<EventObject>>> eventQueue;
    mutable std::mutex queueMutex;  ///< Protects the event queue

    // Dendrite registry: maps dendrite ID to dendrite object
    std::map<uint64_t, std::shared_ptr<Dendrite>> dendriteRegistry;
    mutable std::mutex dendriteRegistryMutex;  ///< Protects the dendrite registry

    // Synapse registry: maps synapse ID to synapse object
    std::map<uint64_t, std::shared_ptr<Synapse>> synapseRegistry;
    mutable std::mutex synapseRegistryMutex;  ///< Protects the synapse registry

    // Thread pool for parallel spike delivery
    std::unique_ptr<ThreadPool> threadPool;

    // Async delivery threads management (DEPRECATED - now using futures)
    std::deque<std::thread> activeDeliveryThreads;  ///< Active timeslice delivery threads (legacy)
    mutable std::mutex deliveryThreadsMutex;        ///< Protects activeDeliveryThreads (legacy)

    // Async delivery futures management (NEW - optimized)
    std::vector<std::vector<std::shared_ptr<std::future<void>>>> activeDeliveryFutures_;
    mutable std::mutex deliveryFuturesMutex_;

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

    // STDP learning parameters
    double stdpAPlus;                ///< LTP amplitude (default: 0.01)
    double stdpAMinus;               ///< LTD amplitude (default: 0.012)
    double stdpTauPlus;              ///< LTP time constant in ms (default: 20.0)
    double stdpTauMinus;             ///< LTD time constant in ms (default: 20.0)
    std::atomic<bool> stdpEnabled_{true};  ///< Master switch for STDP learning

    // Activity monitoring (optional)
    class ActivityMonitor* activityMonitor_;  ///< Optional activity monitor for recording

    // Delivery diagnostics
    DeliveryStats deliveryStats_{};
    std::unordered_set<uint64_t> inputToL4NeuronIds_;
    std::unordered_set<uint64_t> inputToL4Dendrites_;
    mutable std::mutex deliveryStatsMutex_;
};

} // namespace snnfw

#endif // SNNFW_SPIKE_PROCESSOR_H

#include "snnfw/SpikeProcessor.h"
#include "snnfw/ActivityMonitor.h"
#include "snnfw/Logger.h"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

namespace snnfw {

SpikeProcessor::SpikeProcessor(size_t timeSliceCount, size_t deliveryThreads)
    : numTimeSlices(timeSliceCount),
      numDeliveryThreads(deliveryThreads),
      timeStep(1.0),
      running(false),
      stopRequested(false),
      currentTime(0.0),
      currentSliceIndex(0),
      realTimeSync(false),  // Default to false (fast mode) - can be enabled with setRealTimeSync(true)
      totalLoopTime(0.0),
      maxLoopTime(0.0),
      loopCount(0),
      accumulatedDrift(0.0),
      stdpAPlus(0.01),
      stdpAMinus(0.012),
      stdpTauPlus(20.0),
      stdpTauMinus(20.0),
      activityMonitor_(nullptr) {

    // Initialize event queue with empty vectors for each time slice
    eventQueue.resize(numTimeSlices);

    // Create thread pool for spike delivery
    threadPool = std::make_unique<ThreadPool>(numDeliveryThreads);

    SNNFW_INFO("SpikeProcessor created: {} time slices, {} delivery threads, real-time sync: {}",
               numTimeSlices, numDeliveryThreads, realTimeSync);
}

SpikeProcessor::~SpikeProcessor() {
    stop();
}

void SpikeProcessor::start() {
    if (running.load()) {
        SNNFW_WARN("SpikeProcessor already running");
        return;
    }

    stopRequested.store(false);
    running.store(true);

    // Reset timing statistics
    {
        std::lock_guard<std::mutex> lock(statsMutex);
        totalLoopTime = 0.0;
        maxLoopTime = 0.0;
        loopCount = 0;
        accumulatedDrift = 0.0;
    }

    // Record wall-clock start time
    startWallTime = std::chrono::steady_clock::now();

    // Start the background processing thread
    processingThread = std::thread(&SpikeProcessor::processingLoop, this);

    SNNFW_INFO("SpikeProcessor started (real-time sync: {})", realTimeSync);
}

void SpikeProcessor::stop() {
    if (!running.load()) {
        return;
    }

    SNNFW_INFO("SpikeProcessor stopping...");
    
    stopRequested.store(true);
    cv.notify_all();
    
    if (processingThread.joinable()) {
        processingThread.join();
    }
    
    // Join all active delivery threads
    {
        std::lock_guard<std::mutex> lock(deliveryThreadsMutex);
        SNNFW_INFO("SpikeProcessor: Joining {} active delivery threads...", activeDeliveryThreads.size());
        for (auto& thread : activeDeliveryThreads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        activeDeliveryThreads.clear();
    }
    
    running.store(false);

    SNNFW_INFO("SpikeProcessor stopped. Final time: {:.3f}ms", currentTime.load());
}

void SpikeProcessor::setActivityMonitor(ActivityMonitor* monitor) {
    activityMonitor_ = monitor;
    if (monitor) {
        SNNFW_INFO("SpikeProcessor: Activity monitor attached for automatic recording");
    } else {
        SNNFW_INFO("SpikeProcessor: Activity monitor detached");
    }
}

bool SpikeProcessor::scheduleSpike(const std::shared_ptr<ActionPotential>& actionPotential) {
    if (!actionPotential) {
        SNNFW_ERROR("SpikeProcessor: Cannot schedule null action potential");
        return false;
    }

    // Avoid scheduling into the currently-being-delivered slice.
    // In an async delivery model, events scheduled for "now" can otherwise be lost
    // because the current slice's queue has already been moved out for delivery.
    const double currentTimeMs = currentTime.load();
    const double scheduledTime = actionPotential->getScheduledTime();

    // If scheduled time is already in the past, clamp it forward to avoid dropping spikes.
    if (scheduledTime < currentTimeMs) {
        actionPotential->setScheduledTime(currentTimeMs + timeStep);
    }

    // If a spike lands inside the current time step, push it to the next step.
    if (scheduledTime < currentTimeMs + timeStep) {
        actionPotential->setScheduledTime(currentTimeMs + timeStep);
    }

    int sliceIndex = getTimeSliceIndex(actionPotential->getScheduledTime());

    if (sliceIndex < 0) {
        // Silently drop out-of-range spikes (common during high-frequency firing)
        // Use TRACE level for debugging if needed
        SNNFW_TRACE("SpikeProcessor: Spike scheduled for time {:.3f}ms is out of range (current: {:.3f}ms, max: {:.3f}ms)",
                    actionPotential->getScheduledTime(),
                    currentTime.load(),
                    currentTime.load() + numTimeSlices * timeStep);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(queueMutex);
        eventQueue[sliceIndex].push_back(actionPotential);
    }

    {
        std::lock_guard<std::mutex> lock(deliveryStatsMutex_);
        deliveryStats_.scheduled++;
        if (inputToL4Dendrites_.find(actionPotential->getDendriteId()) != inputToL4Dendrites_.end()) {
            deliveryStats_.scheduledInputToL4++;
        }
    }

    SNNFW_TRACE("SpikeProcessor: Scheduled spike for time {:.3f}ms (slice {})",
                actionPotential->getScheduledTime(), sliceIndex);

    return true;
}

bool SpikeProcessor::scheduleRetrogradeSpike(const std::shared_ptr<RetrogradeActionPotential>& retrogradeAP) {
    if (!retrogradeAP) {
        SNNFW_ERROR("SpikeProcessor: Cannot schedule null retrograde action potential");
        return false;
    }

    // Apply the same "no current-slice scheduling" rule as forward spikes.
    const double currentTimeMs = currentTime.load();
    const double scheduledTime = retrogradeAP->getScheduledTime();

    // If scheduled time is already in the past, clamp it forward to avoid dropping spikes.
    if (scheduledTime < currentTimeMs) {
        retrogradeAP->setScheduledTime(currentTimeMs + timeStep);
    }

    if (scheduledTime < currentTimeMs + timeStep) {
        retrogradeAP->setScheduledTime(currentTimeMs + timeStep);
    }

    int sliceIndex = getTimeSliceIndex(retrogradeAP->getScheduledTime());

    if (sliceIndex < 0) {
        // Silently drop out-of-range retrograde spikes (common during high-frequency firing)
        // Use TRACE level for debugging if needed
        SNNFW_TRACE("SpikeProcessor: Retrograde spike scheduled for time {:.3f}ms is out of range (current: {:.3f}ms, max: {:.3f}ms)",
                    retrogradeAP->getScheduledTime(),
                    currentTime.load(),
                    currentTime.load() + numTimeSlices * timeStep);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(queueMutex);
        eventQueue[sliceIndex].push_back(retrogradeAP);
    }

    SNNFW_TRACE("SpikeProcessor: Scheduled retrograde spike for time {:.3f}ms (slice {})",
                retrogradeAP->getScheduledTime(), sliceIndex);

    return true;
}

void SpikeProcessor::registerDendrite(const std::shared_ptr<Dendrite>& dendrite) {
    if (!dendrite) {
        SNNFW_ERROR("SpikeProcessor: Cannot register null dendrite");
        return;
    }

    std::lock_guard<std::mutex> lock(dendriteRegistryMutex);
    dendriteRegistry[dendrite->getId()] = dendrite;
    if (inputToL4NeuronIds_.find(dendrite->getTargetNeuronId()) != inputToL4NeuronIds_.end()) {
        std::lock_guard<std::mutex> statsLock(deliveryStatsMutex_);
        inputToL4Dendrites_.insert(dendrite->getId());
    }

    SNNFW_DEBUG("SpikeProcessor: Registered dendrite {} (total: {})",
                dendrite->getId(), dendriteRegistry.size());
}

void SpikeProcessor::registerSynapse(const std::shared_ptr<Synapse>& synapse) {
    if (!synapse) {
        SNNFW_ERROR("SpikeProcessor: Cannot register null synapse");
        return;
    }

    std::lock_guard<std::mutex> lock(synapseRegistryMutex);
    synapseRegistry[synapse->getId()] = synapse;

    SNNFW_DEBUG("SpikeProcessor: Registered synapse {} (total: {})",
                synapse->getId(), synapseRegistry.size());
}

void SpikeProcessor::unregisterDendrite(uint64_t dendriteId) {
    std::lock_guard<std::mutex> lock(dendriteRegistryMutex);
    auto it = dendriteRegistry.find(dendriteId);

    if (it != dendriteRegistry.end()) {
        dendriteRegistry.erase(it);
        std::lock_guard<std::mutex> statsLock(deliveryStatsMutex_);
        inputToL4Dendrites_.erase(dendriteId);
        SNNFW_DEBUG("SpikeProcessor: Unregistered dendrite {} (remaining: {})",
                    dendriteId, dendriteRegistry.size());
    } else {
        SNNFW_WARN("SpikeProcessor: Dendrite {} not found for unregistration", dendriteId);
    }
}

void SpikeProcessor::resetDeliveryStats() {
    std::lock_guard<std::mutex> lock(deliveryStatsMutex_);
    deliveryStats_ = DeliveryStats{};
}

SpikeProcessor::DeliveryStats SpikeProcessor::getDeliveryStats() const {
    std::lock_guard<std::mutex> lock(deliveryStatsMutex_);
    return deliveryStats_;
}

void SpikeProcessor::setInputToL4NeuronIds(const std::unordered_set<uint64_t>& neuronIds) {
    std::unordered_set<uint64_t> l4Dendrites;
    {
        std::lock_guard<std::mutex> dendLock(dendriteRegistryMutex);
        for (const auto& entry : dendriteRegistry) {
            const auto& dendrite = entry.second;
            if (dendrite && neuronIds.find(dendrite->getTargetNeuronId()) != neuronIds.end()) {
                l4Dendrites.insert(dendrite->getId());
            }
        }
    }

    std::lock_guard<std::mutex> statsLock(deliveryStatsMutex_);
    inputToL4NeuronIds_ = neuronIds;
    inputToL4Dendrites_ = std::move(l4Dendrites);
}

size_t SpikeProcessor::getPendingSpikeCount() const {
    std::lock_guard<std::mutex> lock(queueMutex);
    size_t total = 0;
    for (const auto& slice : eventQueue) {
        total += slice.size();
    }
    return total;
}

size_t SpikeProcessor::getSpikeCountAtSlice(size_t timeSliceIndex) const {
    if (timeSliceIndex >= numTimeSlices) {
        return 0;
    }
    
    std::lock_guard<std::mutex> lock(queueMutex);
    return eventQueue[timeSliceIndex].size();
}

size_t SpikeProcessor::purgeEventsBefore(double cutoffTime, const std::unordered_set<uint64_t>& dendriteIds) {
    if (dendriteIds.empty()) {
        return 0;
    }

    size_t removed = 0;
    std::lock_guard<std::mutex> lock(queueMutex);
    for (auto& slice : eventQueue) {
        slice.erase(
            std::remove_if(slice.begin(), slice.end(),
                           [&](const std::shared_ptr<EventObject>& evt) {
                               if (!evt || evt->getScheduledTime() >= cutoffTime) {
                                   return false;
                               }
                               auto ap = std::dynamic_pointer_cast<ActionPotential>(evt);
                               if (!ap) {
                                   return false;
                               }
                               if (dendriteIds.find(ap->getDendriteId()) == dendriteIds.end()) {
                                   return false;
                               }
                               removed++;
                               return true;
                           }),
            slice.end());
    }
    return removed;
}

int SpikeProcessor::getTimeSliceIndex(double timeMs) const {
    double currentTimeMs = currentTime.load();
    
    // If scheduled time is in the past, clamp it to the next slice
    if (timeMs < currentTimeMs) {
        timeMs = currentTimeMs + timeStep;
    }
    
    // Calculate relative time from current time
    double relativeTime = timeMs - currentTimeMs;
    
    // Calculate slice index
    size_t relativeSliceIndex = static_cast<size_t>(relativeTime / timeStep);
    
    // Check if it's within our buffer
    if (relativeSliceIndex >= numTimeSlices) {
        return -1;
    }
    
    // Calculate absolute slice index (with wraparound)
    size_t absoluteSliceIndex = (currentSliceIndex + relativeSliceIndex) % numTimeSlices;
    
    return static_cast<int>(absoluteSliceIndex);
}

void SpikeProcessor::processingLoop() {
    SNNFW_INFO("SpikeProcessor: Processing loop started");

    auto loopStartTime = std::chrono::steady_clock::now();

    while (!stopRequested.load()) {
        auto iterationStart = std::chrono::steady_clock::now();

        // Cleanup completed delivery threads
        cleanupCompletedThreads();

        // Kick off async delivery for current time slice (non-blocking)
        const double sliceTime = currentTime.load();
        const size_t sliceIndex = currentSliceIndex;
        deliverSliceAsync(sliceIndex, sliceTime);

        if (!realTimeSync) {
            // Non-real-time mode: run as fast as possible, but do not advance the time base
            // until all delivery work for this slice has completed. This prevents "time went
            // backwards" scheduling decisions and reduces dropped spikes.
            waitForDeliveryThreads();
        }

        // Advance time (atomic double doesn't have fetch_add in C++17, so we use store)
        double newTime = currentTime.load() + timeStep;
        currentTime.store(newTime);
        currentSliceIndex = (currentSliceIndex + 1) % numTimeSlices;

        auto iterationEnd = std::chrono::steady_clock::now();

        // Calculate how long this iteration took
        auto iterationDuration = std::chrono::duration_cast<std::chrono::microseconds>(
            iterationEnd - iterationStart);
        double iterationTimeUs = static_cast<double>(iterationDuration.count());

        // Update timing statistics
        {
            std::lock_guard<std::mutex> lock(statsMutex);
            totalLoopTime += iterationTimeUs;
            maxLoopTime = std::max(maxLoopTime, iterationTimeUs);
            loopCount++;
        }

        if (realTimeSync) {
            // Real-time synchronization: each timeslice should take exactly 1ms of wall-clock time
            // Calculate expected wall-clock time for this simulation time
            double expectedWallTimeMs = currentTime.load();
            auto expectedWallTime = startWallTime +
                std::chrono::microseconds(static_cast<int64_t>(expectedWallTimeMs * 1000.0));

            auto now = std::chrono::steady_clock::now();

            // Calculate drift
            auto drift = std::chrono::duration_cast<std::chrono::microseconds>(now - expectedWallTime);
            double driftMs = static_cast<double>(drift.count()) / 1000.0;

            {
                std::lock_guard<std::mutex> lock(statsMutex);
                accumulatedDrift = driftMs;
            }

            // If we're ahead of schedule, sleep to maintain real-time sync
            if (drift.count() < 0) {
                // We're ahead - sleep for the remaining time
                auto sleepDuration = std::chrono::microseconds(-drift.count());
                std::this_thread::sleep_for(sleepDuration);
            } else if (driftMs > 10.0) {
                // We're falling behind by more than 10ms - log a warning
                size_t activeThreads = getActiveDeliveryThreadCount();
                SNNFW_WARN("SpikeProcessor: Falling behind real-time by {:.2f}ms at simulation time {:.1f}ms ({} active delivery threads)",
                          driftMs, currentTime.load(), activeThreads);
            }

            // Log timing info periodically (every 1000 iterations = 1 second of sim time)
            if (loopCount % 1000 == 0) {
                double avgLoopTime = totalLoopTime / loopCount;
                SNNFW_DEBUG("SpikeProcessor: Sim time: {:.1f}ms, Avg loop: {:.1f}μs, Max loop: {:.1f}μs, Drift: {:.2f}ms",
                           currentTime.load(), avgLoopTime, maxLoopTime, driftMs);
            }
        } else {
            // Just a tiny sleep to prevent CPU spinning if there's no work
            if (iterationTimeUs < 10.0) {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        }
    }

    SNNFW_INFO("SpikeProcessor: Processing loop ended at simulation time {:.3f}ms",
               currentTime.load());

    // Log final statistics
    double avgLoopTime, maxLoop, drift;
    getTimingStats(avgLoopTime, maxLoop, drift);
    SNNFW_INFO("SpikeProcessor: Final stats - Avg loop: {:.1f}μs, Max loop: {:.1f}μs, Final drift: {:.2f}ms",
               avgLoopTime, maxLoop, drift);
}

void SpikeProcessor::deliverSliceAsync(size_t sliceIndex, double simTime) {
    // Extract events for this time slice
    std::vector<std::shared_ptr<EventObject>> eventsToDeliver;
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        eventsToDeliver = std::move(eventQueue[sliceIndex]);
        eventQueue[sliceIndex].clear();
    }

    if (eventsToDeliver.empty()) {
        return;
    }

    // DEBUG: Track delivery volume (disabled for performance)
    // static int g_deliveryCount = 0;
    // static size_t g_totalEvents = 0;
    // g_totalEvents += eventsToDeliver.size();
    // if (++g_deliveryCount % 1000 == 0) {
    //     std::cout << "[DEBUG] SpikeProcessor: Delivered " << g_totalEvents << " events in " << g_deliveryCount
    //               << " slices (avg " << (g_totalEvents / g_deliveryCount) << " events/slice)" << std::endl;
    //     g_deliveryCount = 0;
    //     g_totalEvents = 0;
    // }

    SNNFW_TRACE("SpikeProcessor: Async delivering {} events at time {:.3f}ms",
                eventsToDeliver.size(), simTime);

    // OPTIMIZATION: For small numbers of events, deliver synchronously to avoid thread pool overhead
    const size_t ASYNC_THRESHOLD = 50;  // Only use thread pool if we have more than 50 events

    if (eventsToDeliver.size() < ASYNC_THRESHOLD) {
        // Synchronous delivery for small batches
        for (const auto& event : eventsToDeliver) {
            const char* eventType = event->getEventType();

            // OPTIMIZATION: Check first char instead of full string compare
            if (eventType[0] == 'A') { // "ActionPotential"
                // Forward spike - deliver to dendrite
                auto spike = std::static_pointer_cast<ActionPotential>(event);

                std::shared_ptr<Dendrite> dendrite;
                {
                    std::lock_guard<std::mutex> lock(dendriteRegistryMutex);
                    auto it = dendriteRegistry.find(spike->getDendriteId());
                    if (it != dendriteRegistry.end()) {
                        dendrite = it->second;
                    }
                }

                if (dendrite) {
                    dendrite->receiveSpike(spike);
                    {
                        std::lock_guard<std::mutex> lock(deliveryStatsMutex_);
                        deliveryStats_.delivered++;
                        if (inputToL4Dendrites_.find(spike->getDendriteId()) != inputToL4Dendrites_.end()) {
                            deliveryStats_.deliveredInputToL4++;
                        }
                    }

                    // Record spike in activity monitor if attached
                    if (activityMonitor_) {
                        activityMonitor_->recordSpike(spike, spike->getScheduledTime());
                    }
                } else {
                    {
                        std::lock_guard<std::mutex> lock(deliveryStatsMutex_);
                        deliveryStats_.missingDendrite++;
                        if (inputToL4Dendrites_.find(spike->getDendriteId()) != inputToL4Dendrites_.end()) {
                            deliveryStats_.missingDendriteInputToL4++;
                        }
                    }
                    SNNFW_WARN("SpikeProcessor: Dendrite {} not found for spike delivery",
                               spike->getDendriteId());
                }
            }
            else if (eventType[0] == 'R') { // "RetrogradeActionPotential"
                // Retrograde spike - deliver to synapse for STDP
                auto retrogradeSpike = std::static_pointer_cast<RetrogradeActionPotential>(event);

                std::shared_ptr<Synapse> synapse;
                {
                    std::lock_guard<std::mutex> lock(synapseRegistryMutex);
                    auto it = synapseRegistry.find(retrogradeSpike->getSynapseId());
                    if (it != synapseRegistry.end()) {
                        synapse = it->second;
                    }
                }

                if (synapse) {
                    // Apply STDP based on temporal offset
                    double temporalOffset = retrogradeSpike->getTemporalOffset();
                    applySTDPToSynapse(synapse, temporalOffset);
                } else {
                    SNNFW_WARN("SpikeProcessor: Synapse {} not found for retrograde spike delivery",
                               retrogradeSpike->getSynapseId());
                }
            }
            else {
                SNNFW_WARN("SpikeProcessor: Unknown event type: {}", eventType);
            }
        }
        return;  // Done with synchronous delivery
    }

    // OPTIMIZATION: For large batches, use thread pool
    // Instead of creating a new thread, submit work directly to thread pool
    // and store the futures for later synchronization
    std::vector<std::shared_ptr<std::future<void>>> deliveryFutures;

    // OPTIMIZATION: Use shared_ptr to avoid copying the events vector for each task
    auto eventsPtr = std::make_shared<std::vector<std::shared_ptr<EventObject>>>(std::move(eventsToDeliver));

    // Divide events evenly among thread pool workers
    size_t eventsPerThread = (eventsPtr->size() + numDeliveryThreads - 1) / numDeliveryThreads;

    for (size_t threadIdx = 0; threadIdx < numDeliveryThreads; ++threadIdx) {
        size_t startIdx = threadIdx * eventsPerThread;
        size_t endIdx = std::min(startIdx + eventsPerThread, eventsPtr->size());

        if (startIdx >= eventsPtr->size()) {
            break;
        }

        // Submit delivery task to thread pool and capture the future
        auto future = std::make_shared<std::future<void>>(
            threadPool->enqueue([this, eventsPtr, startIdx, endIdx]() {
                for (size_t i = startIdx; i < endIdx; ++i) {
                    const auto& event = (*eventsPtr)[i];

                        // Check event type and deliver accordingly
                        const char* eventType = event->getEventType();

                        if (eventType[0] == 'A') {
                            // Forward spike - deliver to dendrite
                            auto spike = std::static_pointer_cast<ActionPotential>(event);

                            std::shared_ptr<Dendrite> dendrite;
                            {
                                std::lock_guard<std::mutex> lock(dendriteRegistryMutex);
                                auto it = dendriteRegistry.find(spike->getDendriteId());
                                if (it != dendriteRegistry.end()) {
                                    dendrite = it->second;
                                }
                            }

                            if (dendrite) {
                                dendrite->receiveSpike(spike);
                                {
                                    std::lock_guard<std::mutex> lock(deliveryStatsMutex_);
                                    deliveryStats_.delivered++;
                                    if (inputToL4Dendrites_.find(spike->getDendriteId()) != inputToL4Dendrites_.end()) {
                                        deliveryStats_.deliveredInputToL4++;
                                    }
                                }

                                // Record spike in activity monitor if attached
                                if (activityMonitor_) {
                                    activityMonitor_->recordSpike(spike, spike->getScheduledTime());
                                }
                            } else {
                                {
                                    std::lock_guard<std::mutex> lock(deliveryStatsMutex_);
                                    deliveryStats_.missingDendrite++;
                                    if (inputToL4Dendrites_.find(spike->getDendriteId()) != inputToL4Dendrites_.end()) {
                                        deliveryStats_.missingDendriteInputToL4++;
                                    }
                                }
                                SNNFW_WARN("SpikeProcessor: Dendrite {} not found for spike delivery",
                                           spike->getDendriteId());
                            }
                        }
                        else if (eventType[0] == 'R') {
                            // Retrograde spike - deliver to synapse for STDP
                            auto retrogradeSpike = std::static_pointer_cast<RetrogradeActionPotential>(event);

                            std::shared_ptr<Synapse> synapse;
                            {
                                std::lock_guard<std::mutex> lock(synapseRegistryMutex);
                                auto it = synapseRegistry.find(retrogradeSpike->getSynapseId());
                                if (it != synapseRegistry.end()) {
                                    synapse = it->second;
                                }
                            }

                            if (synapse) {
                                // Apply STDP based on temporal offset
                                double temporalOffset = retrogradeSpike->getTemporalOffset();
                                applySTDPToSynapse(synapse, temporalOffset);
                            } else {
                                SNNFW_WARN("SpikeProcessor: Synapse {} not found for retrograde spike delivery",
                                           retrogradeSpike->getSynapseId());
                            }
                        }
                        else {
                            SNNFW_WARN("SpikeProcessor: Unknown event type: {}", eventType);
                        }
                    }
                })
            );
        deliveryFutures.push_back(future);
    }

    // Store futures for synchronization in waitForDeliveryThreads()
    {
        std::lock_guard<std::mutex> lock(deliveryFuturesMutex_);
        activeDeliveryFutures_.push_back(std::move(deliveryFutures));
    }

    SNNFW_TRACE("SpikeProcessor: Submitted {} delivery tasks for time {:.3f}ms",
                deliveryFutures.size(), simTime);
}

void SpikeProcessor::cleanupCompletedThreads() {
    std::lock_guard<std::mutex> lock(deliveryThreadsMutex);
    
    // Join and remove completed threads
    auto it = activeDeliveryThreads.begin();
    while (it != activeDeliveryThreads.end()) {
        if (it->joinable()) {
            // Try to join with zero timeout (non-blocking check)
            // Since C++ doesn't have try_join, we'll just keep accumulating
            // and clean them up periodically or on shutdown
            ++it;
        } else {
            it = activeDeliveryThreads.erase(it);
        }
    }
    
    // Limit the number of threads we keep around
    // Join older threads if we have too many
    const size_t MAX_THREADS = 100;
    while (activeDeliveryThreads.size() > MAX_THREADS && !activeDeliveryThreads.empty()) {
        if (activeDeliveryThreads.front().joinable()) {
            activeDeliveryThreads.front().join();
        }
        activeDeliveryThreads.pop_front();
    }
}

size_t SpikeProcessor::getActiveDeliveryThreadCount() const {
    std::lock_guard<std::mutex> lock(deliveryThreadsMutex);
    return activeDeliveryThreads.size();
}

void SpikeProcessor::waitForDeliveryThreads() {
    std::vector<std::vector<std::shared_ptr<std::future<void>>>> futuresToWait;
    {
        std::lock_guard<std::mutex> lock(deliveryFuturesMutex_);
        futuresToWait = std::move(activeDeliveryFutures_);
        activeDeliveryFutures_.clear();
    }

    // Wait for all delivery futures to complete
    for (auto& futureGroup : futuresToWait) {
        for (auto& future : futureGroup) {
            if (future && future->valid()) {
                future->get();
            }
        }
    }
}

void SpikeProcessor::getTimingStats(double& avgLoopTime, double& maxLoop, double& driftMs) const {
    std::lock_guard<std::mutex> lock(statsMutex);

    if (loopCount > 0) {
        avgLoopTime = totalLoopTime / loopCount;
    } else {
        avgLoopTime = 0.0;
    }

    maxLoop = maxLoopTime;
    driftMs = accumulatedDrift;
}

void SpikeProcessor::setSTDPParameters(double aPlus, double aMinus, double tauPlus, double tauMinus) {
    stdpAPlus = aPlus;
    stdpAMinus = aMinus;
    stdpTauPlus = tauPlus;
    stdpTauMinus = tauMinus;

    SNNFW_INFO("SpikeProcessor: Updated STDP parameters (A+={}, A-={}, τ+={}, τ-={})",
               stdpAPlus, stdpAMinus, stdpTauPlus, stdpTauMinus);
}

void SpikeProcessor::setStdpEnabled(bool enabled) {
    stdpEnabled_.store(enabled, std::memory_order_release);
    SNNFW_INFO("SpikeProcessor: STDP learning {}", enabled ? "enabled" : "disabled");
}

bool SpikeProcessor::isStdpEnabled() const {
    return stdpEnabled_.load(std::memory_order_acquire);
}

void SpikeProcessor::applySTDPToSynapse(std::shared_ptr<Synapse> synapse, double temporalOffset) {
    if (!synapse) {
        return;
    }

    // Early exit if STDP is disabled (inference mode)
    if (!stdpEnabled_.load(std::memory_order_acquire)) {
        return;
    }

    // Classic STDP learning rule:
    // temporalOffset = lastFiringTime - dispatchTime
    // If temporalOffset >= 0: neuron fired AFTER spike was sent → LTP (strengthen)
    // If temporalOffset < 0: neuron fired BEFORE spike was sent → LTD (weaken)

    double weightChange = 0.0;

    if (temporalOffset >= 0) {
        // LTP: strengthen synapse (post-synaptic neuron fired after spike arrived)
        weightChange = stdpAPlus * std::exp(-temporalOffset / stdpTauPlus);
    } else {
        // LTD: weaken synapse (post-synaptic neuron fired before spike arrived)
        weightChange = -stdpAMinus * std::exp(temporalOffset / stdpTauMinus);
    }

    if (weightChange != 0.0) {
        double oldWeight = synapse->getWeight();
        double newWeight = oldWeight + weightChange;

        // Clamp weight to [0, 2] range to prevent runaway growth/decay
        newWeight = std::max(0.0, std::min(2.0, newWeight));

        synapse->setWeight(newWeight);

        SNNFW_TRACE("SpikeProcessor: STDP update for synapse {}: temporalOffset={:.3f}ms, Δw={:.6f}, weight: {:.4f} → {:.4f}",
                   synapse->getId(), temporalOffset, weightChange, oldWeight, newWeight);
    }
}

} // namespace snnfw

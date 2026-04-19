#include "snnfw/NetworkPropagator.h"
#include "snnfw/ActionPotential.h"
#include "snnfw/RetrogradeActionPotential.h"
#include "snnfw/ActivityMonitor.h"
#include "snnfw/RecordingManager.h"
#include "snnfw/Logger.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace snnfw {

NetworkPropagator::NetworkPropagator(std::shared_ptr<SpikeProcessor> spikeProcessor)
    : spikeProcessor_(spikeProcessor),
      stdpAPlus_(0.01),
      stdpAMinus_(0.012),
      stdpTauPlus_(20.0),
      stdpTauMinus_(20.0),
      stdpLtdScale_(1.0),
      stdpLtdWindowMs_(20.0),
      activityMonitor_(nullptr),
      recordingManager_(nullptr) {
    if (!spikeProcessor_) {
        SNNFW_ERROR("NetworkPropagator: SpikeProcessor cannot be null");
        throw std::invalid_argument("SpikeProcessor cannot be null");
    }
    SNNFW_INFO("NetworkPropagator: Initialized with STDP parameters (A+={}, A-={}, τ+={}, τ-={})",
               stdpAPlus_, stdpAMinus_, stdpTauPlus_, stdpTauMinus_);
}

void NetworkPropagator::setActivityMonitor(ActivityMonitor* monitor) {
    activityMonitor_ = monitor;
    if (monitor) {
        SNNFW_INFO("NetworkPropagator: Activity monitor attached for automatic recording");
    } else {
        SNNFW_INFO("NetworkPropagator: Activity monitor detached");
    }
}

void NetworkPropagator::setRecordingManager(RecordingManager* manager) {
    recordingManager_ = manager;
    if (manager) {
        SNNFW_INFO("NetworkPropagator: Recording manager attached for direct recording");
    } else {
        SNNFW_INFO("NetworkPropagator: Recording manager detached");
    }
}

void NetworkPropagator::registerNeuron(const std::shared_ptr<Neuron>& neuron) {
    if (!neuron) {
        SNNFW_ERROR("NetworkPropagator: Cannot register null neuron");
        return;
    }

    std::lock_guard<std::mutex> lock(neuronMutex_);
    neuronRegistry_[neuron->getId()] = neuron;
    SNNFW_DEBUG("NetworkPropagator: Registered neuron {}", neuron->getId());
}

void NetworkPropagator::registerAxon(const std::shared_ptr<Axon>& axon) {
    if (!axon) {
        SNNFW_ERROR("NetworkPropagator: Cannot register null axon");
        return;
    }

    std::lock_guard<std::mutex> lock(axonMutex_);
    axonRegistry_[axon->getId()] = axon;
    SNNFW_DEBUG("NetworkPropagator: Registered axon {}", axon->getId());
}

void NetworkPropagator::registerSynapse(const std::shared_ptr<Synapse>& synapse) {
    if (!synapse) {
        SNNFW_ERROR("NetworkPropagator: Cannot register null synapse");
        return;
    }

    std::lock_guard<std::mutex> lock(synapseMutex_);
    synapseRegistry_[synapse->getId()] = synapse;

    // Build reverse index: dendrite ID -> synapses
    uint64_t dendriteId = synapse->getDendriteId();
    dendriteToSynapsesIndex_[dendriteId].push_back(synapse);

    // Also register with SpikeProcessor for retrograde spike delivery
    spikeProcessor_->registerSynapse(synapse);

    SNNFW_DEBUG("NetworkPropagator: Registered synapse {} targeting dendrite {}",
                synapse->getId(), dendriteId);
}

void NetworkPropagator::registerDendrite(const std::shared_ptr<Dendrite>& dendrite) {
    if (!dendrite) {
        SNNFW_ERROR("NetworkPropagator: Cannot register null dendrite");
        return;
    }

    std::lock_guard<std::mutex> lock(dendriteMutex_);
    dendriteRegistry_[dendrite->getId()] = dendrite;
    
    // Also register with SpikeProcessor for delivery
    spikeProcessor_->registerDendrite(dendrite);
    
    SNNFW_DEBUG("NetworkPropagator: Registered dendrite {}", dendrite->getId());
}

int NetworkPropagator::fireNeuron(uint64_t neuronId, double firingTime) {
    // Record neuron firing if recording manager is attached (before checking axons/synapses)
    if (recordingManager_) {
        RecordedSpike recordedSpike;
        // Clamp negative times to 0 to avoid uint64 overflow
        recordedSpike.timestamp = static_cast<uint64_t>(std::max(0.0, firingTime));
        recordedSpike.sourceNeuronId = neuronId;
        recordedSpike.targetNeuronId = 0;  // No specific target for neuron-level recording
        recordedSpike.synapseId = 0;  // No specific synapse for neuron-level recording
        recordingManager_->recordSpike(recordedSpike);
    }

    // Get the neuron
    std::shared_ptr<Neuron> neuron;
    {
        std::lock_guard<std::mutex> lock(neuronMutex_);
        auto it = neuronRegistry_.find(neuronId);
        if (it == neuronRegistry_.end()) {
            SNNFW_WARN("NetworkPropagator: Neuron {} not found", neuronId);
            return -1;
        }
        neuron = it->second;
    }

    // Get the neuron's axon
    uint64_t axonId = neuron->getAxonId();
    if (axonId == 0) {
        SNNFW_DEBUG("NetworkPropagator: Neuron {} has no axon", neuronId);
        return 0;
    }

    std::shared_ptr<Axon> axon;
    {
        std::lock_guard<std::mutex> lock(axonMutex_);
        auto it = axonRegistry_.find(axonId);
        if (it == axonRegistry_.end()) {
            SNNFW_WARN("NetworkPropagator: Axon {} not found for neuron {}", axonId, neuronId);
            return -1;
        }
        axon = it->second;
    }

    // Get all synapses connected to this axon
    const auto& synapseIds = axon->getSynapseIds();
    if (synapseIds.empty()) {
        static int emptyAxonWarns = 0;
        if (emptyAxonWarns < 20) {
            emptyAxonWarns++;
            SNNFW_WARN("NetworkPropagator: Axon {} (neuron {}) has no synapses; firingTime={:.3f}",
                       axonId, neuronId, firingTime);
        }
        return 0;
    }

    // Get the neuron's temporal signature (unique spike pattern)
    const auto& temporalSignature = neuron->getTemporalSignature();
    if (temporalSignature.empty()) {
        SNNFW_WARN("NetworkPropagator: Neuron {} has no temporal signature", neuronId);
        return 0;
    }

    // Debug: log first few firings to verify synapse/temporal counts
    static int debugFireLogs = 0;
    if (debugFireLogs < 10) {
        debugFireLogs++;
        SNNFW_INFO("NetworkPropagator: fireNeuron neuron={} axon={} synapses={} signatureSpikes={} firingTime={:.3f}",
                   neuronId, axonId, synapseIds.size(), temporalSignature.size(), firingTime);
    }

    int spikesScheduled = 0;
    double minDelay = std::numeric_limits<double>::max();
    double maxDelay = std::numeric_limits<double>::min();

    // For each synapse, create and schedule multiple action potentials based on temporal signature
    for (uint64_t synapseId : synapseIds) {
        std::shared_ptr<Synapse> synapse;
        
        // OPTIMIZATION: Try to get synapse without locking if possible, 
        // or rely on the fact that registries shouldn't change during simulation.
        // Ideally, 'Axon' should hold vector<shared_ptr<Synapse>> instead of vector<uint64_t> IDs.
        // For now, we minimize the scope, but this architecture needs a refactor to cache pointers.
        
        // Current fix: The lookup is unavoidable without changing Axon.h, 
        // but we can at least check if the registry is read-only during run-time 
        // to avoid the lock, or use a shared_mutex (Reader-Writer lock).
        
        // (Applying the lock here as is, but noting this is the #1 performance killer)
        // See recommendation #2 above.
        {
             std::lock_guard<std::mutex> lock(synapseMutex_);
             auto it = synapseRegistry_.find(synapseId);
             if (it == synapseRegistry_.end()) continue;
             synapse = it->second;
        }

        double baseDelay = synapse->getDelay();
        double amplitude = synapse->getWeight();

        // Schedule one spike for each time offset in the temporal signature
        for (double timeOffset : temporalSignature) {
            double totalDelay = baseDelay + timeOffset;
            double arrivalTime = firingTime + totalDelay;

            // Track delay statistics
            minDelay = std::min(minDelay, totalDelay);
            maxDelay = std::max(maxDelay, totalDelay);

            auto actionPotential = std::make_shared<ActionPotential>(
                synapseId,
                synapse->getDendriteId(),
                arrivalTime,
                amplitude,
                firingTime  // Set dispatch time to when neuron fired
            );

            // Schedule for delivery
            if (spikeProcessor_->scheduleSpike(actionPotential)) {
                spikesScheduled++;
                SynapseGroup group = SynapseGroup::Unknown;
                {
                    std::lock_guard<std::mutex> lock(synapseGroupMutex_);
                    auto it = synapseGroupMap_.find(synapseId);
                    if (it != synapseGroupMap_.end()) {
                        group = it->second;
                    }
                }
                if (group == SynapseGroup::InputToL4) {
                    scheduledInputToL4_.fetch_add(1, std::memory_order_relaxed);
                } else {
                    scheduledOther_.fetch_add(1, std::memory_order_relaxed);
                }
                SNNFW_TRACE("NetworkPropagator: Scheduled spike from neuron {} via synapse {} to dendrite {} at time {:.3f}ms (offset: {:.3f}ms, dispatch: {:.3f}ms)",
                           neuronId, synapseId, synapse->getDendriteId(), arrivalTime, timeOffset, firingTime);

                // Record spike in activity monitor if attached
                if (activityMonitor_) {
                    activityMonitor_->recordSpike(actionPotential, arrivalTime);
                }
            } else {
                // Silently drop out-of-range spikes (common during high-frequency firing)
                // Use TRACE level for debugging if needed
                SNNFW_TRACE("NetworkPropagator: Failed to schedule spike from neuron {} via synapse {} (out of time range)",
                           neuronId, synapseId);
            }
        }

        // Trace STDP already updates via pre/post traces in sendAcknowledgment()
        // and deliverSpikeToNeuron(). Scheduling retrograde STDP in that mode
        // would double-apply temporal updates.
        if (!traceStdpEnabled_) {
            // Schedule retrograde action potentials for classic acknowledgment STDP.
            // These travel back to the synapse to update weights based on timing.
            double retrogradeArrivalTime = firingTime + baseDelay;

            auto retrogradeAP = std::make_shared<RetrogradeActionPotential>(
                synapseId,
                neuronId,
                retrogradeArrivalTime,
                firingTime,  // dispatchTime (when the forward spike was sent)
                firingTime   // lastFiringTime (when this neuron fired)
            );

            if (spikeProcessor_->scheduleRetrogradeSpike(retrogradeAP)) {
                SNNFW_TRACE("NetworkPropagator: Scheduled retrograde spike from neuron {} to synapse {} at time {:.3f}ms",
                           neuronId, synapseId, retrogradeArrivalTime);
            } else {
                // Silently drop out-of-range retrograde spikes (common during high-frequency firing)
                // Use TRACE level for debugging if needed
                SNNFW_TRACE("NetworkPropagator: Failed to schedule retrograde spike from neuron {} to synapse {} (out of time range)",
                           neuronId, synapseId);
            }
        }
    }

    SNNFW_DEBUG("NetworkPropagator: Neuron {} fired at {:.3f}ms, scheduled {} spikes across {} synapses (delay range: {:.3f}-{:.3f}ms, temporal spread: {:.1f}ms)",
               neuronId, firingTime, spikesScheduled, synapseIds.size(), minDelay, maxDelay,
               temporalSignature.back() - temporalSignature.front());

    return spikesScheduled;
}

bool NetworkPropagator::deliverSpikeToNeuron(uint64_t neuronId, uint64_t synapseId, double spikeTime, double amplitude, double dispatchTime) {
    std::shared_ptr<Neuron> neuron;
    {
        std::lock_guard<std::mutex> lock(neuronMutex_);
        auto it = neuronRegistry_.find(neuronId);
        if (it == neuronRegistry_.end()) {
            SNNFW_WARN("NetworkPropagator: Cannot deliver spike to neuron {} - not found", neuronId);
            return false;
        }
        neuron = it->second;
    }

    // Insert spike into neuron's buffer
    // Note: amplitude could be used to modulate the spike, but for now we just insert the spike time
    neuron->insertSpike(spikeTime);

    // Record the incoming spike for STDP (with dispatch time)
    neuron->recordIncomingSpike(synapseId, spikeTime, dispatchTime);

    deliveryTotal_.fetch_add(1, std::memory_order_relaxed);
    SynapseGroup group = SynapseGroup::Unknown;
    {
        std::lock_guard<std::mutex> lock(synapseGroupMutex_);
        auto it = synapseGroupMap_.find(synapseId);
        if (it != synapseGroupMap_.end()) {
            group = it->second;
        }
    }
    if (group == SynapseGroup::L5ToOutput) {
        deliveryL5ToOutput_.fetch_add(1, std::memory_order_relaxed);
        double delta = spikeTime - dispatchTime;
        std::lock_guard<std::mutex> lock(deliveryStatsMutex_);
        deliveryL5ToOutputDeltaSum_ += delta;
        deliveryL5ToOutputDeltaMin_ = std::min(deliveryL5ToOutputDeltaMin_, delta);
        deliveryL5ToOutputDeltaMax_ = std::max(deliveryL5ToOutputDeltaMax_, delta);
    }

    if (traceStdpEnabled_) {
        double lastPostTime = 0.0;
        bool hasLastPost = false;
        {
            std::lock_guard<std::mutex> lock(synapseLastPostMutex_);
            auto it = synapseLastPostTime_.find(synapseId);
            if (it != synapseLastPostTime_.end()) {
                lastPostTime = it->second;
                hasLastPost = true;
            }
        }
        if (hasLastPost && spikeTime > lastPostTime) {
            double delta = spikeTime - lastPostTime;
            if (stdpLtdWindowMs_ <= 0.0 || delta <= stdpLtdWindowMs_) {
                double timeDifference = lastPostTime - spikeTime; // negative Δt -> LTD
                applySTDP(synapseId, timeDifference);
            }
        }
        {
            std::lock_guard<std::mutex> lock(synapseLastPreMutex_);
            synapseLastPreTime_[synapseId] = spikeTime;
        }
    }

    SNNFW_TRACE("NetworkPropagator: Delivered spike to neuron {} at time {:.3f}ms (amplitude: {:.3f}, synapse: {}, dispatch: {:.3f}ms)",
               neuronId, spikeTime, amplitude, synapseId, dispatchTime);

    return true;
}

void NetworkPropagator::resetDeliveryStats() {
    deliveryTotal_.store(0, std::memory_order_relaxed);
    deliveryL5ToOutput_.store(0, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(deliveryStatsMutex_);
    deliveryL5ToOutputDeltaSum_ = 0.0;
    deliveryL5ToOutputDeltaMin_ = std::numeric_limits<double>::infinity();
    deliveryL5ToOutputDeltaMax_ = -std::numeric_limits<double>::infinity();
}

void NetworkPropagator::resetScheduleStats() {
    scheduledInputToL4_.store(0, std::memory_order_relaxed);
    scheduledOther_.store(0, std::memory_order_relaxed);
}

NetworkPropagator::DeliveryStats NetworkPropagator::getDeliveryStats() const {
    DeliveryStats stats;
    stats.total = deliveryTotal_.load(std::memory_order_relaxed);
    stats.l5ToOutput = deliveryL5ToOutput_.load(std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(deliveryStatsMutex_);
    stats.l5ToOutputDeltaSum = deliveryL5ToOutputDeltaSum_;
    stats.l5ToOutputDeltaMin = deliveryL5ToOutputDeltaMin_;
    stats.l5ToOutputDeltaMax = deliveryL5ToOutputDeltaMax_;
    return stats;
}

NetworkPropagator::ScheduleStats NetworkPropagator::getScheduleStats() const {
    ScheduleStats stats;
    stats.inputToL4 = scheduledInputToL4_.load(std::memory_order_relaxed);
    stats.other = scheduledOther_.load(std::memory_order_relaxed);
    return stats;
}

std::shared_ptr<Neuron> NetworkPropagator::getNeuron(uint64_t neuronId) const {
    std::lock_guard<std::mutex> lock(neuronMutex_);
    auto it = neuronRegistry_.find(neuronId);
    return (it != neuronRegistry_.end()) ? it->second : nullptr;
}

std::shared_ptr<Axon> NetworkPropagator::getAxon(uint64_t axonId) const {
    std::lock_guard<std::mutex> lock(axonMutex_);
    auto it = axonRegistry_.find(axonId);
    return (it != axonRegistry_.end()) ? it->second : nullptr;
}

std::shared_ptr<Synapse> NetworkPropagator::getSynapse(uint64_t synapseId) const {
    std::lock_guard<std::mutex> lock(synapseMutex_);
    auto it = synapseRegistry_.find(synapseId);
    return (it != synapseRegistry_.end()) ? it->second : nullptr;
}

std::shared_ptr<Dendrite> NetworkPropagator::getDendrite(uint64_t dendriteId) const {
    std::lock_guard<std::mutex> lock(dendriteMutex_);
    auto it = dendriteRegistry_.find(dendriteId);
    return (it != dendriteRegistry_.end()) ? it->second : nullptr;
}

std::vector<double> NetworkPropagator::computeLayerActivation(const std::vector<uint64_t>& neuronIds) const {
    std::vector<double> activations;
    activations.reserve(neuronIds.size());

    for (uint64_t neuronId : neuronIds) {
        std::shared_ptr<Neuron> neuron;
        {
            std::lock_guard<std::mutex> lock(neuronMutex_);
            auto it = neuronRegistry_.find(neuronId);
            if (it != neuronRegistry_.end()) {
                neuron = it->second;
            }
        }

        if (neuron) {
            // Get the best similarity between current spikes and learned patterns
            double activation = neuron->getBestSimilarity();
            activations.push_back(activation);
        } else {
            SNNFW_WARN("NetworkPropagator: Neuron {} not found when computing activation", neuronId);
            activations.push_back(0.0);
        }
    }

    return activations;
}

void NetworkPropagator::clearAllSpikes() {
    std::lock_guard<std::mutex> lock(neuronMutex_);
    for (auto& pair : neuronRegistry_) {
        pair.second->clearSpikes();
    }
    SNNFW_DEBUG("NetworkPropagator: Cleared all spike buffers");
}

size_t NetworkPropagator::getNeuronCount() const {
    std::lock_guard<std::mutex> lock(neuronMutex_);
    return neuronRegistry_.size();
}

size_t NetworkPropagator::getSynapseCount() const {
    std::lock_guard<std::mutex> lock(synapseMutex_);
    return synapseRegistry_.size();
}

void NetworkPropagator::sendAcknowledgment(const std::shared_ptr<SpikeAcknowledgment>& acknowledgment) {
    if (!acknowledgment) {
        SNNFW_ERROR("NetworkPropagator: Cannot send null acknowledgment");
        return;
    }

    {
        std::lock_guard<std::mutex> lock(synapseLastPostMutex_);
        synapseLastPostTime_[acknowledgment->getSynapseId()] = acknowledgment->getPostsynapticFiringTime();
    }

    double timeDifference = acknowledgment->getTimeDifference();
    uint64_t synapseId = acknowledgment->getSynapseId();

    // Track per-neuron STDP eligibility from timing, independent of
    // whether the eventual synaptic update is clipped or deferred.
    accumulateNeuronStdpEligibility(acknowledgment->getPostsynapticNeuronId(), timeDifference);

    if (traceStdpEnabled_) {
        double lastPreTime = 0.0;
        bool hasLastPre = false;
        {
            std::lock_guard<std::mutex> lock(synapseLastPreMutex_);
            auto it = synapseLastPreTime_.find(synapseId);
            if (it != synapseLastPreTime_.end()) {
                lastPreTime = it->second;
                hasLastPre = true;
            }
        }
        if (hasLastPre && acknowledgment->getPostsynapticFiringTime() > lastPreTime) {
            double dt = acknowledgment->getPostsynapticFiringTime() - lastPreTime;
            if (applySTDP(synapseId, dt)) {
                SNNFW_TRACE("NetworkPropagator: Trace STDP LTP for synapse {} (Δt = {:.3f}ms)",
                            synapseId, dt);
            }
        }
        return;
    }

    // Apply STDP to the synapse (acknowledgment timing)
    if (applySTDP(synapseId, timeDifference)) {
        SNNFW_TRACE("NetworkPropagator: Applied STDP to synapse {} (Δt = {:.3f}ms)",
                   synapseId, timeDifference);
    } else {
        SNNFW_WARN("NetworkPropagator: Failed to apply STDP to synapse {}", synapseId);
    }
}

bool NetworkPropagator::applySTDP(uint64_t synapseId, double timeDifference) {
    // Early exit if STDP is disabled (inference mode)
    if (!stdpEnabled_.load(std::memory_order_acquire)) {
        return false;
    }

    std::shared_ptr<Synapse> synapse;
    {
        std::lock_guard<std::mutex> lock(synapseMutex_);
        auto it = synapseRegistry_.find(synapseId);
        if (it == synapseRegistry_.end()) {
            SNNFW_WARN("NetworkPropagator: Synapse {} not found for STDP update", synapseId);
            return false;
        }
        synapse = it->second;
    }

    SynapseGroup group = SynapseGroup::Unknown;
    {
        std::lock_guard<std::mutex> lock(synapseGroupMutex_);
        auto it = synapseGroupMap_.find(synapseId);
        if (it != synapseGroupMap_.end()) {
            group = it->second;
        }
    }

    if (timeDifference > 1e-6) {
        switch (group) {
            case SynapseGroup::InputToL4:
                stdpInputToL4PreBeforePost_.fetch_add(1, std::memory_order_relaxed);
                break;
            case SynapseGroup::L4ToL5:
                stdpL4ToL5PreBeforePost_.fetch_add(1, std::memory_order_relaxed);
                break;
            case SynapseGroup::L5ToOutput:
                stdpL5ToOutputPreBeforePost_.fetch_add(1, std::memory_order_relaxed);
                break;
            case SynapseGroup::Unknown:
            default:
                stdpOtherPreBeforePost_.fetch_add(1, std::memory_order_relaxed);
                break;
        }
    } else if (timeDifference < -1e-6) {
        switch (group) {
            case SynapseGroup::InputToL4:
                stdpInputToL4PostBeforePre_.fetch_add(1, std::memory_order_relaxed);
                break;
            case SynapseGroup::L4ToL5:
                stdpL4ToL5PostBeforePre_.fetch_add(1, std::memory_order_relaxed);
                break;
            case SynapseGroup::L5ToOutput:
                stdpL5ToOutputPostBeforePre_.fetch_add(1, std::memory_order_relaxed);
                break;
            case SynapseGroup::Unknown:
            default:
                stdpOtherPostBeforePre_.fetch_add(1, std::memory_order_relaxed);
                break;
        }
    } else {
        switch (group) {
            case SynapseGroup::InputToL4:
                stdpInputToL4NearZero_.fetch_add(1, std::memory_order_relaxed);
                break;
            case SynapseGroup::L4ToL5:
                stdpL4ToL5NearZero_.fetch_add(1, std::memory_order_relaxed);
                break;
            case SynapseGroup::L5ToOutput:
                stdpL5ToOutputNearZero_.fetch_add(1, std::memory_order_relaxed);
                break;
            case SynapseGroup::Unknown:
            default:
                stdpOtherNearZero_.fetch_add(1, std::memory_order_relaxed);
                break;
        }
    }

    // Classic STDP learning rule:
    // Δw = A+ * exp(-Δt / τ+)  if Δt > 0 (LTP: pre before post)
    // Δw = -A- * exp(Δt / τ-)  if Δt < 0 (LTD: post before pre)

    double weightChange = 0.0;

    if (timeDifference > 0) {
        // LTP: strengthen synapse (pre-synaptic spike arrived before post-synaptic spike)
        weightChange = stdpAPlus_ * std::exp(-timeDifference / stdpTauPlus_);
    } else if (timeDifference < 0) {
        // LTD: weaken synapse (post-synaptic spike arrived before pre-synaptic spike)
        weightChange = -stdpAMinus_ * stdpLtdScale_ * std::exp(timeDifference / stdpTauMinus_);
    }
    // If timeDifference == 0, no change

    if (weightChange != 0.0) {
        double oldWeight = synapse->getWeight();
        double newWeight = oldWeight + weightChange;

        // Clamp weight to [0, 2] range to prevent runaway growth/decay
        newWeight = std::max(0.0, std::min(2.0, newWeight));

        synapse->setWeight(newWeight);
        stdpUpdatesTotal_.fetch_add(1, std::memory_order_relaxed);
        if (weightChange > 0.0) {
            stdpUpdatesLtp_.fetch_add(1, std::memory_order_relaxed);
        } else {
            stdpUpdatesLtd_.fetch_add(1, std::memory_order_relaxed);
        }
        if (weightChange > 0.0) {
            switch (group) {
                case SynapseGroup::InputToL4:
                    stdpInputToL4Ltp_.fetch_add(1, std::memory_order_relaxed);
                    break;
                case SynapseGroup::L4ToL5:
                    stdpL4ToL5Ltp_.fetch_add(1, std::memory_order_relaxed);
                    break;
                case SynapseGroup::L5ToOutput:
                    stdpL5ToOutputLtp_.fetch_add(1, std::memory_order_relaxed);
                    break;
                case SynapseGroup::Unknown:
                default:
                    stdpOtherLtp_.fetch_add(1, std::memory_order_relaxed);
                    break;
            }
        } else {
            switch (group) {
                case SynapseGroup::InputToL4:
                    stdpInputToL4Ltd_.fetch_add(1, std::memory_order_relaxed);
                    break;
                case SynapseGroup::L4ToL5:
                    stdpL4ToL5Ltd_.fetch_add(1, std::memory_order_relaxed);
                    break;
                case SynapseGroup::L5ToOutput:
                    stdpL5ToOutputLtd_.fetch_add(1, std::memory_order_relaxed);
                    break;
                case SynapseGroup::Unknown:
                default:
                    stdpOtherLtd_.fetch_add(1, std::memory_order_relaxed);
                    break;
            }
        }

        SNNFW_TRACE("NetworkPropagator: STDP update for synapse {}: Δt={:.3f}ms, Δw={:.6f}, weight: {:.4f} → {:.4f}",
                   synapseId, timeDifference, weightChange, oldWeight, newWeight);
    }

    return true;
}

void NetworkPropagator::setSTDPParameters(double aPlus, double aMinus, double tauPlus, double tauMinus) {
    stdpAPlus_ = aPlus;
    stdpAMinus_ = aMinus;
    stdpTauPlus_ = tauPlus;
    stdpTauMinus_ = tauMinus;

    SNNFW_INFO("NetworkPropagator: Updated STDP parameters (A+={}, A-={}, τ+={}, τ-={})",
               stdpAPlus_, stdpAMinus_, stdpTauPlus_, stdpTauMinus_);
}

void NetworkPropagator::setStdpLtdScale(double scale) {
    stdpLtdScale_ = std::max(0.0, scale);
    SNNFW_INFO("NetworkPropagator: Updated STDP LTD scale to {}", stdpLtdScale_);
}

void NetworkPropagator::setStdpLtdWindowMs(double windowMs) {
    stdpLtdWindowMs_ = windowMs;
    SNNFW_INFO("NetworkPropagator: Updated STDP LTD window to {} ms", stdpLtdWindowMs_);
}

void NetworkPropagator::setStdpEnabled(bool enabled) {
    stdpEnabled_.store(enabled, std::memory_order_release);
    SNNFW_INFO("NetworkPropagator: STDP learning {}", enabled ? "enabled" : "disabled");
}

bool NetworkPropagator::isStdpEnabled() const {
    return stdpEnabled_.load(std::memory_order_acquire);
}

void NetworkPropagator::applyRewardModulatedSTDP(uint64_t neuronId, double rewardFactor) {
    // Early exit if STDP is disabled (inference mode)
    if (!stdpEnabled_.load(std::memory_order_acquire)) {
        return;
    }

    // Get the target neuron
    std::shared_ptr<Neuron> neuron;
    {
        std::lock_guard<std::mutex> lock(neuronMutex_);
        auto it = neuronRegistry_.find(neuronId);
        if (it == neuronRegistry_.end()) {
            SNNFW_WARN("NetworkPropagator: Cannot apply reward-modulated STDP - neuron {} not found", neuronId);
            return;
        }
        neuron = it->second;
    }

    // Find all synapses targeting this neuron's dendrites using reverse index
    // PERFORMANCE: O(k) where k = number of dendrites, instead of O(n) where n = total synapses
    std::vector<std::shared_ptr<Synapse>> targetSynapses;
    {
        std::lock_guard<std::mutex> lock(synapseMutex_);
        for (uint64_t dendriteId : neuron->getDendriteIds()) {
            auto it = dendriteToSynapsesIndex_.find(dendriteId);
            if (it != dendriteToSynapsesIndex_.end()) {
                // Append all synapses targeting this dendrite
                targetSynapses.insert(targetSynapses.end(), it->second.begin(), it->second.end());
            }
        }
    }

    // Apply reward-modulated weight changes
    int updatedCount = 0;
    for (const auto& synapse : targetSynapses) {
        double currentWeight = synapse->getWeight();

        // Reward-modulated learning: strengthen synapses proportional to reward
        // Positive reward (>1.0) = strengthen, negative reward (<1.0) = weaken
        double weightChange = stdpAPlus_ * (rewardFactor - 1.0);
        double newWeight = currentWeight + weightChange;

        // Clamp to [0, 2] range
        newWeight = std::max(0.0, std::min(2.0, newWeight));

        synapse->setWeight(newWeight);
        updatedCount++;
        stdpUpdatesReward_.fetch_add(1, std::memory_order_relaxed);

        SNNFW_TRACE("NetworkPropagator: Reward-modulated STDP - synapse {} weight: {:.4f} → {:.4f} (reward: {:.2f})",
                   synapse->getId(), currentWeight, newWeight, rewardFactor);
    }

    SNNFW_DEBUG("NetworkPropagator: Applied reward-modulated STDP to {} synapses targeting neuron {} (reward: {:.2f})",
                updatedCount, neuronId, rewardFactor);
}

void NetworkPropagator::resetStdpUpdateStats() {
    stdpUpdatesTotal_.store(0, std::memory_order_relaxed);
    stdpUpdatesLtp_.store(0, std::memory_order_relaxed);
    stdpUpdatesLtd_.store(0, std::memory_order_relaxed);
    stdpUpdatesReward_.store(0, std::memory_order_relaxed);
    resetStdpGroupStats();
    resetStdpTimingStats();
    resetNeuronStdpEligibility();
}

NetworkPropagator::StdpUpdateStats NetworkPropagator::getStdpUpdateStats() const {
    StdpUpdateStats stats;
    stats.total = stdpUpdatesTotal_.load(std::memory_order_relaxed);
    stats.ltp = stdpUpdatesLtp_.load(std::memory_order_relaxed);
    stats.ltd = stdpUpdatesLtd_.load(std::memory_order_relaxed);
    stats.reward = stdpUpdatesReward_.load(std::memory_order_relaxed);
    return stats;
}

void NetworkPropagator::registerSynapseGroup(uint64_t synapseId, SynapseGroup group) {
    std::lock_guard<std::mutex> lock(synapseGroupMutex_);
    synapseGroupMap_[synapseId] = group;
}

void NetworkPropagator::resetStdpGroupStats() {
    stdpInputToL4Ltp_.store(0, std::memory_order_relaxed);
    stdpInputToL4Ltd_.store(0, std::memory_order_relaxed);
    stdpL4ToL5Ltp_.store(0, std::memory_order_relaxed);
    stdpL4ToL5Ltd_.store(0, std::memory_order_relaxed);
    stdpL5ToOutputLtp_.store(0, std::memory_order_relaxed);
    stdpL5ToOutputLtd_.store(0, std::memory_order_relaxed);
    stdpOtherLtp_.store(0, std::memory_order_relaxed);
    stdpOtherLtd_.store(0, std::memory_order_relaxed);
}

NetworkPropagator::StdpGroupStatsSet NetworkPropagator::getStdpGroupStats() const {
    StdpGroupStatsSet stats;
    stats.inputToL4.ltp = stdpInputToL4Ltp_.load(std::memory_order_relaxed);
    stats.inputToL4.ltd = stdpInputToL4Ltd_.load(std::memory_order_relaxed);
    stats.l4ToL5.ltp = stdpL4ToL5Ltp_.load(std::memory_order_relaxed);
    stats.l4ToL5.ltd = stdpL4ToL5Ltd_.load(std::memory_order_relaxed);
    stats.l5ToOutput.ltp = stdpL5ToOutputLtp_.load(std::memory_order_relaxed);
    stats.l5ToOutput.ltd = stdpL5ToOutputLtd_.load(std::memory_order_relaxed);
    stats.other.ltp = stdpOtherLtp_.load(std::memory_order_relaxed);
    stats.other.ltd = stdpOtherLtd_.load(std::memory_order_relaxed);
    return stats;
}

void NetworkPropagator::resetStdpTimingStats() {
    stdpInputToL4PreBeforePost_.store(0, std::memory_order_relaxed);
    stdpInputToL4PostBeforePre_.store(0, std::memory_order_relaxed);
    stdpInputToL4NearZero_.store(0, std::memory_order_relaxed);
    stdpL4ToL5PreBeforePost_.store(0, std::memory_order_relaxed);
    stdpL4ToL5PostBeforePre_.store(0, std::memory_order_relaxed);
    stdpL4ToL5NearZero_.store(0, std::memory_order_relaxed);
    stdpL5ToOutputPreBeforePost_.store(0, std::memory_order_relaxed);
    stdpL5ToOutputPostBeforePre_.store(0, std::memory_order_relaxed);
    stdpL5ToOutputNearZero_.store(0, std::memory_order_relaxed);
    stdpOtherPreBeforePost_.store(0, std::memory_order_relaxed);
    stdpOtherPostBeforePre_.store(0, std::memory_order_relaxed);
    stdpOtherNearZero_.store(0, std::memory_order_relaxed);
}

NetworkPropagator::StdpTimingGroupStats NetworkPropagator::getStdpTimingStats() const {
    StdpTimingGroupStats stats;
    stats.inputToL4.preBeforePost = stdpInputToL4PreBeforePost_.load(std::memory_order_relaxed);
    stats.inputToL4.postBeforePre = stdpInputToL4PostBeforePre_.load(std::memory_order_relaxed);
    stats.inputToL4.nearZero = stdpInputToL4NearZero_.load(std::memory_order_relaxed);
    stats.l4ToL5.preBeforePost = stdpL4ToL5PreBeforePost_.load(std::memory_order_relaxed);
    stats.l4ToL5.postBeforePre = stdpL4ToL5PostBeforePre_.load(std::memory_order_relaxed);
    stats.l4ToL5.nearZero = stdpL4ToL5NearZero_.load(std::memory_order_relaxed);
    stats.l5ToOutput.preBeforePost = stdpL5ToOutputPreBeforePost_.load(std::memory_order_relaxed);
    stats.l5ToOutput.postBeforePre = stdpL5ToOutputPostBeforePre_.load(std::memory_order_relaxed);
    stats.l5ToOutput.nearZero = stdpL5ToOutputNearZero_.load(std::memory_order_relaxed);
    stats.other.preBeforePost = stdpOtherPreBeforePost_.load(std::memory_order_relaxed);
    stats.other.postBeforePre = stdpOtherPostBeforePre_.load(std::memory_order_relaxed);
    stats.other.nearZero = stdpOtherNearZero_.load(std::memory_order_relaxed);
    return stats;
}

void NetworkPropagator::accumulateNeuronStdpEligibility(uint64_t neuronId, double timeDifference) {
    if (neuronId == 0) return;
    if (timeDifference == 0.0) return;

    NeuronStdpEligibilityStats delta;
    if (timeDifference > 0.0) {
        delta.totalUpdates = 1;
        delta.ltpUpdates = 1;
        delta.ltpMagnitude = stdpAPlus_ * std::exp(-timeDifference / stdpTauPlus_);
    } else {
        delta.totalUpdates = 1;
        delta.ltdUpdates = 1;
        delta.ltdMagnitude =
            stdpAMinus_ * stdpLtdScale_ * std::exp(timeDifference / stdpTauMinus_);
    }

    std::lock_guard<std::mutex> lock(neuronStdpEligibilityMutex_);
    auto& stats = neuronStdpEligibility_[neuronId];
    stats.totalUpdates += delta.totalUpdates;
    stats.ltpUpdates += delta.ltpUpdates;
    stats.ltdUpdates += delta.ltdUpdates;
    stats.ltpMagnitude += delta.ltpMagnitude;
    stats.ltdMagnitude += delta.ltdMagnitude;
}

void NetworkPropagator::resetNeuronStdpEligibility() {
    std::lock_guard<std::mutex> lock(neuronStdpEligibilityMutex_);
    neuronStdpEligibility_.clear();
}

NetworkPropagator::NeuronStdpEligibilityStats
NetworkPropagator::getNeuronStdpEligibility(uint64_t neuronId) const {
    std::lock_guard<std::mutex> lock(neuronStdpEligibilityMutex_);
    auto it = neuronStdpEligibility_.find(neuronId);
    if (it == neuronStdpEligibility_.end()) {
        return {};
    }
    return it->second;
}

void NetworkPropagator::setTraceStdpEnabled(bool enabled) {
    traceStdpEnabled_ = enabled;
}

} // namespace snnfw

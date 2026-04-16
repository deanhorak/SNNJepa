#include "snnfw/ActivityMonitor.h"
#include "snnfw/Datastore.h"
#include "snnfw/ActionPotential.h"
#include "snnfw/Neuron.h"
#include "snnfw/Cluster.h"
#include "snnfw/Layer.h"
#include "snnfw/Column.h"
#include "snnfw/Nucleus.h"
#include "snnfw/Region.h"
#include "snnfw/Lobe.h"
#include "snnfw/Hemisphere.h"
#include "snnfw/Brain.h"
#include "snnfw/Synapse.h"
#include "snnfw/Axon.h"
#include "snnfw/Dendrite.h"
#include "snnfw/Logger.h"
#include "snnfw/SimulationConfig.h"
#include "snnfw/RecordingManager.h"
#include "snnfw/ActivityVisualizer.h"
#include <algorithm>

namespace snnfw {

ActivityMonitor::ActivityMonitor(Datastore& datastore)
    : datastore_(datastore),
      monitoring_(false),
      historyDuration_(1000.0),  // Default: 1 second
      snapshotInterval_(10.0),   // Default: 10ms
      nextCallbackId_(1),
      recordingManager_(nullptr) {
    SNNFW_INFO("ActivityMonitor created with {}ms history duration", historyDuration_);
}

ActivityMonitor::ActivityMonitor(Datastore& datastore, const SimulationConfig& config)
    : datastore_(datastore),
      monitoring_(false),
      historyDuration_(1000.0),
      snapshotInterval_(10.0),
      nextCallbackId_(1),
      recordingManager_(nullptr) {

    // Create recording manager if recording is enabled
    if (config.enableRecording) {
        // Note: RecordingManager requires an ActivityVisualizer, which we don't have here
        // We'll need to refactor RecordingManager to work without ActivityVisualizer
        // For now, recording will be set up externally
        SNNFW_INFO("ActivityMonitor created with recording enabled (filename: {})",
                   config.recordingFilename.empty() ? "auto-generated" : config.recordingFilename);
    }

    SNNFW_INFO("ActivityMonitor created with {}ms history duration", historyDuration_);
}

ActivityMonitor::~ActivityMonitor() {
    stopMonitoring();
    SNNFW_INFO("ActivityMonitor destroyed");
}

size_t ActivityMonitor::buildHierarchicalCache(uint64_t brainId) {
    std::lock_guard<std::mutex> lock(cacheMutex_);

    hierarchicalCache_.clear();
    size_t neuronCount = 0;

    auto brain = datastore_.getBrain(brainId);
    if (!brain) {
        SNNFW_ERROR("ActivityMonitor: Brain {} not found", brainId);
        return 0;
    }

    SNNFW_INFO("Building hierarchical context cache from Brain {}", brainId);

    // Traverse: Brain → Hemispheres → Lobes → Regions → Nuclei → Columns → Layers → Clusters → Neurons
    for (size_t hIdx = 0; hIdx < brain->size(); ++hIdx) {
        uint64_t hemisphereId = brain->getHemisphereId(hIdx);
        auto hemisphere = datastore_.getHemisphere(hemisphereId);
        if (!hemisphere) continue;

        for (size_t lIdx = 0; lIdx < hemisphere->size(); ++lIdx) {
            uint64_t lobeId = hemisphere->getLobeId(lIdx);
            auto lobe = datastore_.getLobe(lobeId);
            if (!lobe) continue;

            for (size_t rIdx = 0; rIdx < lobe->size(); ++rIdx) {
                uint64_t regionId = lobe->getRegionId(rIdx);
                auto region = datastore_.getRegion(regionId);
                if (!region) continue;

                for (size_t nIdx = 0; nIdx < region->size(); ++nIdx) {
                    uint64_t nucleusId = region->getNucleusId(nIdx);
                    auto nucleus = datastore_.getNucleus(nucleusId);
                    if (!nucleus) continue;

                    for (size_t colIdx = 0; colIdx < nucleus->size(); ++colIdx) {
                        uint64_t columnId = nucleus->getColumnId(colIdx);
                        auto column = datastore_.getColumn(columnId);
                        if (!column) continue;

                        for (size_t layIdx = 0; layIdx < column->size(); ++layIdx) {
                            uint64_t layerId = column->getLayerId(layIdx);
                            auto layer = datastore_.getLayer(layerId);
                            if (!layer) continue;

                            for (size_t cluIdx = 0; cluIdx < layer->size(); ++cluIdx) {
                                uint64_t clusterId = layer->getClusterId(cluIdx);
                                auto cluster = datastore_.getCluster(clusterId);
                                if (!cluster) continue;

                                // Index all neurons in this cluster
                                const auto& neuronIds = cluster->getNeuronIds();
                                for (uint64_t neuronId : neuronIds) {
                                    HierarchicalContext ctx;
                                    ctx.clusterId = clusterId;
                                    ctx.layerId = layerId;
                                    ctx.columnId = columnId;
                                    ctx.nucleusId = nucleusId;
                                    ctx.regionId = regionId;
                                    ctx.lobeId = lobeId;
                                    ctx.hemisphereId = hemisphereId;
                                    ctx.brainId = brainId;

                                    hierarchicalCache_[neuronId] = ctx;
                                    neuronCount++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    SNNFW_INFO("Hierarchical cache built: {} neurons indexed", neuronCount);
    return neuronCount;
}

void ActivityMonitor::startMonitoring() {
    if (monitoring_) {
        SNNFW_WARN("ActivityMonitor already monitoring");
        return;
    }
    
    monitoring_ = true;
    clearHistory();
    SNNFW_INFO("ActivityMonitor started");
}

void ActivityMonitor::stopMonitoring() {
    if (!monitoring_) {
        return;
    }
    
    monitoring_ = false;
    SNNFW_INFO("ActivityMonitor stopped - recorded {} total spikes", getTotalSpikeCount());
}

void ActivityMonitor::recordSpike(const std::shared_ptr<ActionPotential>& actionPotential, double currentTime) {
    if (!monitoring_ || !actionPotential) {
        return;
    }
    
    SpikeEvent event;
    event.timestamp = currentTime;
    event.synapseId = actionPotential->getSynapseId();
    event.dendriteId = actionPotential->getDendriteId();
    event.amplitude = actionPotential->getAmplitude();
    
    // Resolve synapse to get axon and source neuron
    auto synapse = datastore_.getSynapse(event.synapseId);
    if (!synapse) {
        SNNFW_WARN("ActivityMonitor: Synapse {} not found", event.synapseId);
        return;
    }

    event.axonId = synapse->getAxonId();
    event.delay = synapse->getDelay();

    // Resolve axon to get source neuron
    auto axon = datastore_.getAxon(event.axonId);
    if (!axon) {
        SNNFW_WARN("ActivityMonitor: Axon {} not found", event.axonId);
        return;
    }

    event.sourceNeuronId = axon->getSourceNeuronId();

    // Resolve dendrite to get target neuron
    auto dendrite = datastore_.getDendrite(event.dendriteId);
    if (!dendrite) {
        SNNFW_WARN("ActivityMonitor: Dendrite {} not found", event.dendriteId);
        return;
    }

    event.targetNeuronId = dendrite->getTargetNeuronId();
    
    // Resolve hierarchical context for source and target
    resolveHierarchicalContext(event.sourceNeuronId, event.sourceClusterId, 
                               event.sourceLayerId, event.sourceColumnId);
    resolveHierarchicalContext(event.targetNeuronId, event.targetClusterId,
                               event.targetLayerId, event.targetColumnId);
    
    // Determine propagation type
    event.propagationType = determinePropagationType(
        event.sourceClusterId, event.sourceLayerId, event.sourceColumnId,
        event.targetClusterId, event.targetLayerId, event.targetColumnId);
    
    // Add to event history
    {
        std::lock_guard<std::mutex> lock(eventsMutex_);
        spikeEvents_.push_back(event);
    }
    
    // Notify callbacks
    notifyCallbacks(event);
    
    // Cleanup old events
    cleanupOldEvents(currentTime);
    
    SNNFW_TRACE("ActivityMonitor: Recorded spike from neuron {} to {} at {:.3f}ms (type: {})",
                event.sourceNeuronId, event.targetNeuronId, currentTime,
                static_cast<int>(event.propagationType));
}

void ActivityMonitor::recordNeuronFiring(uint64_t neuronId, double firingTime) {
    if (!monitoring_) {
        return;
    }
    
    SNNFW_TRACE("ActivityMonitor: Neuron {} fired at {:.3f}ms", neuronId, firingTime);
    
    // This could be extended to track neuron firing events separately
    // For now, we primarily track spike propagation through recordSpike()
}

void ActivityMonitor::setHistoryDuration(double durationMs) {
    historyDuration_ = durationMs;
    SNNFW_INFO("ActivityMonitor: History duration set to {:.1f}ms", durationMs);
}

void ActivityMonitor::setSnapshotInterval(double intervalMs) {
    snapshotInterval_ = intervalMs;
    SNNFW_INFO("ActivityMonitor: Snapshot interval set to {:.1f}ms", intervalMs);
}

ActivitySnapshot ActivityMonitor::getActivitySnapshot(double startTime, double endTime) const {
    ActivitySnapshot snapshot;
    snapshot.startTime = startTime;
    snapshot.endTime = endTime;
    
    std::lock_guard<std::mutex> lock(eventsMutex_);
    
    for (const auto& event : spikeEvents_) {
        if (event.timestamp >= startTime && event.timestamp < endTime) {
            snapshot.totalSpikes++;
            
            // Count per neuron
            snapshot.neuronSpikeCount[event.sourceNeuronId]++;
            
            // Count per cluster
            if (event.sourceClusterId != 0) {
                snapshot.clusterSpikeCount[event.sourceClusterId]++;
            }
            if (event.targetClusterId != 0 && event.targetClusterId != event.sourceClusterId) {
                snapshot.clusterSpikeCount[event.targetClusterId]++;
            }
            
            // Count per layer
            if (event.sourceLayerId != 0) {
                snapshot.layerSpikeCount[event.sourceLayerId]++;
            }
            if (event.targetLayerId != 0 && event.targetLayerId != event.sourceLayerId) {
                snapshot.layerSpikeCount[event.targetLayerId]++;
            }
            
            // Count per column
            if (event.sourceColumnId != 0) {
                snapshot.columnSpikeCount[event.sourceColumnId]++;
            }
            if (event.targetColumnId != 0 && event.targetColumnId != event.sourceColumnId) {
                snapshot.columnSpikeCount[event.targetColumnId]++;
            }
            
            // Count propagation types
            snapshot.propagationTypeCount[event.propagationType]++;
        }
    }
    
    // Calculate average spike rate
    double duration = endTime - startTime;
    if (duration > 0.0) {
        snapshot.averageSpikeRate = snapshot.totalSpikes / duration;
    }
    
    return snapshot;
}

ActivitySnapshot ActivityMonitor::getLatestSnapshot() const {
    std::lock_guard<std::mutex> lock(snapshotsMutex_);
    
    if (snapshots_.empty()) {
        return ActivitySnapshot();
    }
    
    return snapshots_.back();
}

std::vector<SpikeEvent> ActivityMonitor::getSpikeEvents(double startTime, double endTime) const {
    std::vector<SpikeEvent> events;
    
    std::lock_guard<std::mutex> lock(eventsMutex_);
    
    for (const auto& event : spikeEvents_) {
        if (event.timestamp >= startTime && event.timestamp < endTime) {
            events.push_back(event);
        }
    }
    
    return events;
}

FlowStatistics ActivityMonitor::getFlowStatistics(double startTime, double endTime, size_t topN) const {
    FlowStatistics stats;
    
    std::lock_guard<std::mutex> lock(eventsMutex_);
    
    // Accumulate flow counts
    for (const auto& event : spikeEvents_) {
        if (event.timestamp >= startTime && event.timestamp < endTime) {
            // Cluster-to-cluster flow
            if (event.sourceClusterId != 0 && event.targetClusterId != 0) {
                auto pair = std::make_pair(event.sourceClusterId, event.targetClusterId);
                stats.clusterToClusterFlow[pair]++;
            }
            
            // Layer-to-layer flow
            if (event.sourceLayerId != 0 && event.targetLayerId != 0) {
                auto pair = std::make_pair(event.sourceLayerId, event.targetLayerId);
                stats.layerToLayerFlow[pair]++;
            }
            
            // Column-to-column flow
            if (event.sourceColumnId != 0 && event.targetColumnId != 0) {
                auto pair = std::make_pair(event.sourceColumnId, event.targetColumnId);
                stats.columnToColumnFlow[pair]++;
            }
        }
    }
    
    // Sort and get top N pathways
    auto sortAndGetTopN = [topN](const auto& flowMap) {
        std::vector<std::pair<std::pair<uint64_t, uint64_t>, size_t>> sorted(
            flowMap.begin(), flowMap.end());
        std::sort(sorted.begin(), sorted.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        if (sorted.size() > topN) {
            sorted.resize(topN);
        }
        return sorted;
    };
    
    stats.topClusterPairs = sortAndGetTopN(stats.clusterToClusterFlow);
    stats.topLayerPairs = sortAndGetTopN(stats.layerToLayerFlow);
    stats.topColumnPairs = sortAndGetTopN(stats.columnToColumnFlow);
    
    return stats;
}

uint64_t ActivityMonitor::registerSpikeCallback(SpikeCallback callback) {
    std::lock_guard<std::mutex> lock(callbacksMutex_);
    
    uint64_t id = nextCallbackId_++;
    callbacks_[id] = callback;
    
    SNNFW_INFO("ActivityMonitor: Registered spike callback with ID {}", id);
    return id;
}

void ActivityMonitor::unregisterSpikeCallback(uint64_t callbackId) {
    std::lock_guard<std::mutex> lock(callbacksMutex_);
    
    auto it = callbacks_.find(callbackId);
    if (it != callbacks_.end()) {
        callbacks_.erase(it);
        SNNFW_INFO("ActivityMonitor: Unregistered spike callback {}", callbackId);
    }
}

void ActivityMonitor::clearHistory() {
    {
        std::lock_guard<std::mutex> lock(eventsMutex_);
        spikeEvents_.clear();
    }
    
    {
        std::lock_guard<std::mutex> lock(snapshotsMutex_);
        snapshots_.clear();
    }
    
    SNNFW_INFO("ActivityMonitor: History cleared");
}

size_t ActivityMonitor::getTotalSpikeCount() const {
    std::lock_guard<std::mutex> lock(eventsMutex_);
    return spikeEvents_.size();
}

double ActivityMonitor::getCurrentSpikeRate(double windowMs) const {
    std::lock_guard<std::mutex> lock(eventsMutex_);
    
    if (spikeEvents_.empty()) {
        return 0.0;
    }
    
    double latestTime = spikeEvents_.back().timestamp;
    double startTime = latestTime - windowMs;
    
    size_t count = 0;
    for (auto it = spikeEvents_.rbegin(); it != spikeEvents_.rend(); ++it) {
        if (it->timestamp < startTime) {
            break;
        }
        count++;
    }
    
    return count / windowMs;
}

bool ActivityMonitor::resolveHierarchicalContext(uint64_t neuronId, uint64_t& clusterId,
                                                 uint64_t& layerId, uint64_t& columnId) const {
    // Initialize to 0 (unknown)
    clusterId = 0;
    layerId = 0;
    columnId = 0;

    // Look up in cache
    std::lock_guard<std::mutex> lock(cacheMutex_);
    auto it = hierarchicalCache_.find(neuronId);
    if (it != hierarchicalCache_.end()) {
        clusterId = it->second.clusterId;
        layerId = it->second.layerId;
        columnId = it->second.columnId;
        return true;
    }

    // Not found in cache
    SNNFW_TRACE("ActivityMonitor: Neuron {} not found in hierarchical cache", neuronId);
    return false;
}

SpikeEvent::PropagationType ActivityMonitor::determinePropagationType(
    uint64_t sourceClusterId, uint64_t sourceLayerId, uint64_t sourceColumnId,
    uint64_t targetClusterId, uint64_t targetLayerId, uint64_t targetColumnId) const {
    
    // If any context is unknown, return UNKNOWN
    if (sourceClusterId == 0 || targetClusterId == 0) {
        return SpikeEvent::PropagationType::UNKNOWN;
    }
    
    // Same cluster
    if (sourceClusterId == targetClusterId) {
        return SpikeEvent::PropagationType::INTRA_CLUSTER;
    }
    
    // Same layer, different cluster
    if (sourceLayerId != 0 && sourceLayerId == targetLayerId) {
        return SpikeEvent::PropagationType::INTRA_LAYER;
    }
    
    // Same column, different layer
    if (sourceColumnId != 0 && sourceColumnId == targetColumnId) {
        return SpikeEvent::PropagationType::INTRA_COLUMN;
    }
    
    // Different columns
    if (sourceColumnId != 0 && targetColumnId != 0 && sourceColumnId != targetColumnId) {
        return SpikeEvent::PropagationType::INTER_COLUMN;
    }
    
    // Default to inter-region if we can't determine more specifically
    return SpikeEvent::PropagationType::INTER_REGION;
}

void ActivityMonitor::cleanupOldEvents(double currentTime) {
    std::lock_guard<std::mutex> lock(eventsMutex_);
    
    double cutoffTime = currentTime - historyDuration_;
    
    // Remove events older than cutoff
    while (!spikeEvents_.empty() && spikeEvents_.front().timestamp < cutoffTime) {
        spikeEvents_.pop_front();
    }
}

void ActivityMonitor::notifyCallbacks(const SpikeEvent& event) {
    std::lock_guard<std::mutex> lock(callbacksMutex_);

    for (const auto& pair : callbacks_) {
        try {
            pair.second(event);
        } catch (const std::exception& e) {
            SNNFW_ERROR("ActivityMonitor: Callback {} threw exception: {}", pair.first, e.what());
        }
    }

    // If recording manager is attached, record the spike
    if (recordingManager_) {
        RecordedSpike recordedSpike;
        recordedSpike.timestamp = static_cast<uint64_t>(event.timestamp);
        recordedSpike.sourceNeuronId = event.sourceNeuronId;
        recordedSpike.targetNeuronId = event.targetNeuronId;
        recordedSpike.synapseId = event.synapseId;
        recordingManager_->recordSpike(recordedSpike);
    }
}

void ActivityMonitor::setRecordingManager(RecordingManager* recordingManager, bool streamToFile, const std::string& filename) {
    recordingManager_ = recordingManager;

    if (recordingManager_) {
        SNNFW_INFO("ActivityMonitor: Recording manager attached");
        if (streamToFile) {
            SNNFW_INFO("ActivityMonitor: Streaming mode enabled - writing to: " + filename);
            recordingManager_->startRecording(true, filename);
        } else {
            SNNFW_INFO("ActivityMonitor: Memory mode enabled");
            recordingManager_->startRecording(false);
        }
    } else {
        SNNFW_INFO("ActivityMonitor: Recording manager detached");
    }
}

bool ActivityMonitor::saveRecording(const std::string& filename) {
    if (!recordingManager_) {
        SNNFW_WARN("ActivityMonitor: No recording manager attached");
        return false;
    }

    recordingManager_->stopRecording();
    bool success = recordingManager_->saveRecording(filename);

    if (success) {
        SNNFW_INFO("ActivityMonitor: Recording saved to {}", filename);
    } else {
        SNNFW_ERROR("ActivityMonitor: Failed to save recording to {}", filename);
    }

    return success;
}

size_t ActivityMonitor::getMemoryUsage() const {
    std::lock_guard<std::mutex> lock(eventsMutex_);

    // Approximate memory: each SpikeEvent is roughly 100 bytes
    // Plus overhead for the deque structure
    size_t eventMemory = spikeEvents_.size() * sizeof(SpikeEvent);
    size_t dequeOverhead = 1024;  // Rough estimate

    return eventMemory + dequeOverhead;
}

size_t ActivityMonitor::aggressiveCleanup(double targetDurationMs) {
    std::lock_guard<std::mutex> lock(eventsMutex_);

    if (spikeEvents_.empty()) {
        return 0;
    }

    double latestTime = spikeEvents_.back().timestamp;
    double cutoffTime = latestTime - targetDurationMs;

    size_t removedCount = 0;
    while (!spikeEvents_.empty() && spikeEvents_.front().timestamp < cutoffTime) {
        spikeEvents_.pop_front();
        removedCount++;
    }

    if (removedCount > 0) {
        SNNFW_INFO("ActivityMonitor: Aggressive cleanup removed {} events, {} remaining",
                   removedCount, spikeEvents_.size());
    }

    return removedCount;
}

bool ActivityMonitor::isMemoryExcessive(size_t maxMemoryMB) const {
    size_t memoryUsage = getMemoryUsage();
    size_t maxMemoryBytes = maxMemoryMB * 1024ULL * 1024ULL;

    return memoryUsage > maxMemoryBytes;
}

} // namespace snnfw


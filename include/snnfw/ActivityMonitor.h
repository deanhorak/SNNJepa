#ifndef SNNFW_ACTIVITY_MONITOR_H
#define SNNFW_ACTIVITY_MONITOR_H

#include <cstdint>
#include <vector>
#include <map>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <functional>

namespace snnfw {

// Forward declarations
class Datastore;
class ActionPotential;
class Neuron;
class RecordingManager;
struct SimulationConfig;

/**
 * @brief Represents a spike event for monitoring and visualization
 * 
 * This structure captures all information needed to visualize spike propagation
 * through the network, including source/target information and hierarchical context.
 */
struct SpikeEvent {
    double timestamp;              ///< When the spike occurred (ms)
    uint64_t sourceNeuronId;       ///< ID of the neuron that fired
    uint64_t targetNeuronId;       ///< ID of the target neuron
    uint64_t synapseId;            ///< ID of the synapse carrying the spike
    uint64_t axonId;               ///< ID of the axon
    uint64_t dendriteId;           ///< ID of the dendrite
    double amplitude;              ///< Spike amplitude (modulated by synaptic weight)
    double delay;                  ///< Synaptic delay (ms)
    
    // Hierarchical context for source neuron
    uint64_t sourceClusterId;      ///< Cluster containing source neuron
    uint64_t sourceLayerId;        ///< Layer containing source cluster
    uint64_t sourceColumnId;       ///< Column containing source layer
    
    // Hierarchical context for target neuron
    uint64_t targetClusterId;      ///< Cluster containing target neuron
    uint64_t targetLayerId;        ///< Layer containing target cluster
    uint64_t targetColumnId;       ///< Column containing target cluster
    
    // Propagation type classification
    enum class PropagationType {
        INTRA_CLUSTER,    ///< Within same cluster
        INTRA_LAYER,      ///< Between clusters in same layer
        INTRA_COLUMN,     ///< Between layers in same column
        INTER_COLUMN,     ///< Between different columns
        INTER_REGION,     ///< Between different regions
        UNKNOWN           ///< Cannot determine
    };
    PropagationType propagationType;
    
    SpikeEvent() : timestamp(0.0), sourceNeuronId(0), targetNeuronId(0),
                   synapseId(0), axonId(0), dendriteId(0),
                   amplitude(0.0), delay(0.0),
                   sourceClusterId(0), sourceLayerId(0), sourceColumnId(0),
                   targetClusterId(0), targetLayerId(0), targetColumnId(0),
                   propagationType(PropagationType::UNKNOWN) {}
};

/**
 * @brief Aggregated activity statistics for a time window
 * 
 * Used for efficient visualization of activity patterns over time.
 */
struct ActivitySnapshot {
    double startTime;              ///< Start of time window (ms)
    double endTime;                ///< End of time window (ms)
    size_t totalSpikes;            ///< Total number of spikes in window
    
    // Per-neuron activity
    std::map<uint64_t, size_t> neuronSpikeCount;  ///< Spike count per neuron
    
    // Per-cluster activity
    std::map<uint64_t, size_t> clusterSpikeCount; ///< Spike count per cluster
    
    // Per-layer activity
    std::map<uint64_t, size_t> layerSpikeCount;   ///< Spike count per layer
    
    // Per-column activity
    std::map<uint64_t, size_t> columnSpikeCount;  ///< Spike count per column
    
    // Propagation type distribution
    std::map<SpikeEvent::PropagationType, size_t> propagationTypeCount;
    
    // Average spike rate (spikes per ms)
    double averageSpikeRate;
    
    ActivitySnapshot() : startTime(0.0), endTime(0.0), totalSpikes(0), averageSpikeRate(0.0) {}
};

/**
 * @brief Information flow statistics between hierarchical structures
 * 
 * Tracks how information flows between different parts of the network,
 * essential for visualizing connectivity patterns.
 */
struct FlowStatistics {
    // Flow between clusters
    std::map<std::pair<uint64_t, uint64_t>, size_t> clusterToClusterFlow;
    
    // Flow between layers
    std::map<std::pair<uint64_t, uint64_t>, size_t> layerToLayerFlow;
    
    // Flow between columns
    std::map<std::pair<uint64_t, uint64_t>, size_t> columnToColumnFlow;
    
    // Most active pathways (sorted by spike count)
    std::vector<std::pair<std::pair<uint64_t, uint64_t>, size_t>> topClusterPairs;
    std::vector<std::pair<std::pair<uint64_t, uint64_t>, size_t>> topLayerPairs;
    std::vector<std::pair<std::pair<uint64_t, uint64_t>, size_t>> topColumnPairs;
};

/**
 * @brief Callback function type for real-time spike notifications
 * 
 * Allows external visualization systems to receive spike events as they occur.
 * The callback receives a const reference to the spike event.
 */
using SpikeCallback = std::function<void(const SpikeEvent&)>;

/**
 * @brief ActivityMonitor - Real-time monitoring of network activity
 * 
 * The ActivityMonitor provides comprehensive tracking of neural activity across
 * the network, with a focus on enabling visualization of information flow.
 * 
 * Key Features:
 * - Real-time spike event tracking with full hierarchical context
 * - Time-windowed activity snapshots for efficient visualization
 * - Information flow analysis between network structures
 * - Callback system for live visualization updates
 * - Configurable history buffer for temporal analysis
 * - Thread-safe operation for concurrent access
 * 
 * Visualization Support:
 * - Captures source/target neuron IDs for drawing connections
 * - Tracks hierarchical context (cluster, layer, column) for grouping
 * - Classifies propagation types for different visualization modes
 * - Provides aggregated statistics for heatmaps and activity plots
 * - Supports real-time callbacks for animated visualizations
 * 
 * Usage:
 * ```cpp
 * ActivityMonitor monitor(datastore);
 * monitor.setHistoryDuration(1000.0);  // Keep 1 second of history
 * monitor.setSnapshotInterval(10.0);   // Snapshot every 10ms
 * 
 * // Register callback for live visualization
 * monitor.registerSpikeCallback([](const SpikeEvent& event) {
 *     // Update visualization with new spike
 *     visualizer.drawSpike(event.sourceNeuronId, event.targetNeuronId);
 * });
 * 
 * // Start monitoring
 * monitor.startMonitoring();
 * 
 * // Record spikes as they occur
 * monitor.recordSpike(actionPotential, currentTime);
 * 
 * // Get activity snapshot for visualization
 * auto snapshot = monitor.getActivitySnapshot(startTime, endTime);
 * visualizer.drawHeatmap(snapshot.clusterSpikeCount);
 * ```
 */
class ActivityMonitor {
public:
    /**
     * @brief Hierarchical context for a neuron
     */
    struct HierarchicalContext {
        uint64_t clusterId = 0;
        uint64_t layerId = 0;
        uint64_t columnId = 0;
        uint64_t nucleusId = 0;
        uint64_t regionId = 0;
        uint64_t lobeId = 0;
        uint64_t hemisphereId = 0;
        uint64_t brainId = 0;
    };

    /**
     * @brief Constructor
     * @param datastore Reference to the datastore for resolving hierarchical context
     */
    explicit ActivityMonitor(Datastore& datastore);

    /**
     * @brief Constructor with configuration
     * @param datastore Reference to the datastore for resolving hierarchical context
     * @param config Simulation configuration (enables recording if configured)
     */
    ActivityMonitor(Datastore& datastore, const SimulationConfig& config);

    /**
     * @brief Destructor
     */
    ~ActivityMonitor();
    
    // Prevent copying
    ActivityMonitor(const ActivityMonitor&) = delete;
    ActivityMonitor& operator=(const ActivityMonitor&) = delete;
    
    /**
     * @brief Build hierarchical context cache from Brain hierarchy
     * @param brainId ID of the Brain object to traverse
     * @return Number of neurons indexed
     *
     * This should be called once after network construction to enable
     * hierarchical activity aggregation. Traverses the entire hierarchy
     * from Brain down to individual neurons and builds a reverse lookup
     * cache for O(1) context resolution.
     */
    size_t buildHierarchicalCache(uint64_t brainId);

    /**
     * @brief Start monitoring network activity
     */
    void startMonitoring();

    /**
     * @brief Stop monitoring network activity
     */
    void stopMonitoring();

    /**
     * @brief Check if monitoring is active
     * @return true if monitoring, false otherwise
     */
    bool isMonitoring() const { return monitoring_; }
    
    /**
     * @brief Record a spike event from an ActionPotential
     * @param actionPotential The action potential representing the spike
     * @param currentTime Current simulation time (ms)
     */
    void recordSpike(const std::shared_ptr<ActionPotential>& actionPotential, double currentTime);
    
    /**
     * @brief Record a neuron firing event
     * @param neuronId ID of the neuron that fired
     * @param firingTime Time when the neuron fired (ms)
     */
    void recordNeuronFiring(uint64_t neuronId, double firingTime);
    
    /**
     * @brief Set the duration of history to keep (in milliseconds)
     * @param durationMs Duration in milliseconds (default: 1000ms = 1 second)
     */
    void setHistoryDuration(double durationMs);
    
    /**
     * @brief Get the current history duration
     * @return History duration in milliseconds
     */
    double getHistoryDuration() const { return historyDuration_; }
    
    /**
     * @brief Set the interval for automatic activity snapshots
     * @param intervalMs Interval in milliseconds (default: 10ms)
     */
    void setSnapshotInterval(double intervalMs);
    
    /**
     * @brief Get the current snapshot interval
     * @return Snapshot interval in milliseconds
     */
    double getSnapshotInterval() const { return snapshotInterval_; }
    
    /**
     * @brief Get activity snapshot for a specific time window
     * @param startTime Start of time window (ms)
     * @param endTime End of time window (ms)
     * @return Activity snapshot containing aggregated statistics
     */
    ActivitySnapshot getActivitySnapshot(double startTime, double endTime) const;
    
    /**
     * @brief Get the most recent activity snapshot
     * @return Most recent snapshot
     */
    ActivitySnapshot getLatestSnapshot() const;
    
    /**
     * @brief Get all spike events in a time window
     * @param startTime Start of time window (ms)
     * @param endTime End of time window (ms)
     * @return Vector of spike events in the time window
     */
    std::vector<SpikeEvent> getSpikeEvents(double startTime, double endTime) const;
    
    /**
     * @brief Get information flow statistics for a time window
     * @param startTime Start of time window (ms)
     * @param endTime End of time window (ms)
     * @param topN Number of top pathways to include (default: 10)
     * @return Flow statistics
     */
    FlowStatistics getFlowStatistics(double startTime, double endTime, size_t topN = 10) const;
    
    /**
     * @brief Register a callback for real-time spike notifications
     * @param callback Function to call when a spike is recorded
     * @return Callback ID for later removal
     */
    uint64_t registerSpikeCallback(SpikeCallback callback);
    
    /**
     * @brief Unregister a spike callback
     * @param callbackId ID returned from registerSpikeCallback
     */
    void unregisterSpikeCallback(uint64_t callbackId);
    
    /**
     * @brief Clear all recorded history
     */
    void clearHistory();
    
    /**
     * @brief Get total number of spikes recorded
     * @return Total spike count
     */
    size_t getTotalSpikeCount() const;
    
    /**
     * @brief Get current spike rate (spikes per millisecond)
     * @param windowMs Time window for rate calculation (default: 100ms)
     * @return Spike rate
     */
    double getCurrentSpikeRate(double windowMs = 100.0) const;

    /**
     * @brief Set recording manager for automatic spike recording
     * @param recordingManager Pointer to recording manager (nullptr to disable)
     * @param streamToFile If true, stream spikes directly to file instead of memory
     * @param filename Filename for streaming (required if streamToFile is true)
     */
    void setRecordingManager(RecordingManager* recordingManager, bool streamToFile = false, const std::string& filename = "");

    /**
     * @brief Get the recording manager
     * @return Pointer to recording manager (nullptr if not set)
     */
    RecordingManager* getRecordingManager() const { return recordingManager_; }

    /**
     * @brief Save current recording to file
     * @param filename Output filename
     * @return true if successful
     */
    bool saveRecording(const std::string& filename);

    /**
     * @brief Get memory usage of spike events buffer
     * @return Approximate memory used in bytes
     */
    size_t getMemoryUsage() const;

    /**
     * @brief Aggressively cleanup old events to reduce memory
     * @param targetDurationMs Target history duration to maintain
     * @return Number of events removed
     */
    size_t aggressiveCleanup(double targetDurationMs);

    /**
     * @brief Check if memory usage is excessive
     * @param maxMemoryMB Maximum allowed memory in MB
     * @return true if memory usage exceeds limit
     */
    bool isMemoryExcessive(size_t maxMemoryMB) const;

private:
    /**
     * @brief Resolve hierarchical context for a neuron
     * @param neuronId ID of the neuron
     * @param clusterId Output: cluster ID
     * @param layerId Output: layer ID
     * @param columnId Output: column ID
     * @return true if context resolved successfully
     */
    bool resolveHierarchicalContext(uint64_t neuronId, uint64_t& clusterId, 
                                   uint64_t& layerId, uint64_t& columnId) const;
    
    /**
     * @brief Determine propagation type between two neurons
     * @param sourceClusterId Source cluster ID
     * @param sourceLayerId Source layer ID
     * @param sourceColumnId Source column ID
     * @param targetClusterId Target cluster ID
     * @param targetLayerId Target layer ID
     * @param targetColumnId Target column ID
     * @return Propagation type classification
     */
    SpikeEvent::PropagationType determinePropagationType(
        uint64_t sourceClusterId, uint64_t sourceLayerId, uint64_t sourceColumnId,
        uint64_t targetClusterId, uint64_t targetLayerId, uint64_t targetColumnId) const;
    
    /**
     * @brief Cleanup old events outside the history window
     * @param currentTime Current simulation time
     */
    void cleanupOldEvents(double currentTime);
    
    /**
     * @brief Notify all registered callbacks of a spike event
     * @param event The spike event to notify
     */
    void notifyCallbacks(const SpikeEvent& event);
    
    Datastore& datastore_;                          ///< Reference to datastore
    bool monitoring_;                               ///< Whether monitoring is active
    double historyDuration_;                        ///< Duration of history to keep (ms)
    double snapshotInterval_;                       ///< Interval for snapshots (ms)

    mutable std::mutex eventsMutex_;                ///< Protects spike events
    std::deque<SpikeEvent> spikeEvents_;            ///< Recorded spike events

    mutable std::mutex snapshotsMutex_;             ///< Protects snapshots
    std::deque<ActivitySnapshot> snapshots_;        ///< Periodic activity snapshots

    mutable std::mutex callbacksMutex_;             ///< Protects callbacks
    std::map<uint64_t, SpikeCallback> callbacks_;   ///< Registered callbacks
    uint64_t nextCallbackId_;                       ///< Next callback ID to assign

    RecordingManager* recordingManager_;            ///< Optional recording manager (nullptr if disabled)

    // Hierarchical context cache: neuronId → context
    std::unordered_map<uint64_t, HierarchicalContext> hierarchicalCache_;
    mutable std::mutex cacheMutex_;
};

} // namespace snnfw

#endif // SNNFW_ACTIVITY_MONITOR_H


#ifndef SNNFW_NETWORK_INSPECTOR_H
#define SNNFW_NETWORK_INSPECTOR_H

#include "snnfw/Datastore.h"
#include "snnfw/NeuralObjectFactory.h"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cstdint>

namespace snnfw {

/**
 * @brief Statistics about connectivity for a neuron
 */
struct ConnectivityStats {
    uint64_t neuronId;
    size_t inDegree;           ///< Number of incoming synapses
    size_t outDegree;          ///< Number of outgoing synapses
    double avgInWeight;        ///< Average weight of incoming synapses
    double avgOutWeight;       ///< Average weight of outgoing synapses
    std::vector<uint64_t> presynapticNeurons;   ///< IDs of neurons providing input
    std::vector<uint64_t> postsynapticNeurons;  ///< IDs of neurons receiving output
};

/**
 * @brief Statistics about a hierarchical structure (Layer, Column, etc.)
 */
struct HierarchyStats {
    std::string typeName;      ///< Type of structure (Brain, Hemisphere, Lobe, etc.)
    uint64_t id = 0;           ///< ID of this structure
    std::string name;          ///< Name of this structure (if available)
    size_t childCount = 0;     ///< Number of direct children
    size_t totalNeurons = 0;   ///< Total neurons in this subtree
    size_t totalSynapses = 0;  ///< Total synapses in this subtree
    size_t totalClusters = 0;  ///< Total clusters in this subtree
    size_t depth = 0;          ///< Depth in hierarchy (0 = Brain)
    std::vector<uint64_t> childIds;  ///< IDs of direct children
};

/**
 * @brief Statistics about a neuron's internal state
 */
struct NeuronStats {
    uint64_t neuronId;
    size_t learnedPatternCount;     ///< Number of learned patterns
    size_t currentSpikeCount;       ///< Number of spikes in rolling window
    double bestSimilarity;          ///< Best similarity to learned patterns
    double windowSizeMs;            ///< Size of rolling time window
    double similarityThreshold;     ///< Threshold for pattern matching
    size_t maxReferencePatterns;    ///< Maximum patterns this neuron can store
    std::string similarityMetric;   ///< Current similarity metric
    uint64_t axonId;                ///< ID of this neuron's axon
    std::vector<uint64_t> dendriteIds;  ///< IDs of this neuron's dendrites
};

/**
 * @brief Report of orphaned or dangling objects
 */
struct OrphanReport {
    std::vector<uint64_t> orphanedNeurons;    ///< Neurons with no connections
    std::vector<uint64_t> orphanedClusters;   ///< Clusters not in any layer
    std::vector<uint64_t> orphanedLayers;     ///< Layers not in any column
    std::vector<uint64_t> orphanedColumns;    ///< Columns not in any nucleus
    std::vector<uint64_t> orphanedNuclei;     ///< Nuclei not in any region
    std::vector<uint64_t> orphanedRegions;    ///< Regions not in any lobe
    std::vector<uint64_t> orphanedLobes;      ///< Lobes not in any hemisphere
    std::vector<uint64_t> orphanedHemispheres; ///< Hemispheres not in any brain
    std::vector<uint64_t> danglingAxons;      ///< Axons with no synapses
    std::vector<uint64_t> danglingSynapses;   ///< Synapses with invalid endpoints
};

/**
 * @brief Format for inspection reports
 */
enum class ReportFormat {
    TEXT,       ///< Human-readable text
    JSON,       ///< JSON format
    MARKDOWN,   ///< Markdown format
    CSV         ///< CSV format (for statistics)
};

/**
 * @class NetworkInspector
 * @brief Comprehensive network introspection and analysis tool
 *
 * The NetworkInspector provides powerful tools to "peek inside the black box"
 * of spiking neural networks. It enables:
 *
 * 1. **Hierarchical Structure Inspection**
 *    - Traverse from Brain → Hemisphere → Lobe → ... → Neuron
 *    - Get statistics at any level (neuron count, synapse count, etc.)
 *    - Visualize network topology
 *
 * 2. **Connectivity Analysis**
 *    - Analyze in-degree and out-degree for neurons
 *    - Find presynaptic and postsynaptic partners
 *    - Compute connectivity matrices
 *    - Identify highly connected hubs
 *
 * 3. **Neuron State Inspection**
 *    - Examine learned patterns
 *    - View current spike history
 *    - Check similarity scores
 *    - Inspect internal parameters
 *
 * 4. **Orphan Detection**
 *    - Find neurons without connections
 *    - Detect dangling references (IDs that don't exist)
 *    - Identify structural inconsistencies
 *
 * 5. **Reporting**
 *    - Generate comprehensive reports in multiple formats
 *    - Export statistics for analysis
 *    - Create visualizations
 *
 * Example Usage:
 * @code
 * NetworkInspector inspector;
 * 
 * // Inspect entire brain hierarchy
 * auto brainStats = inspector.inspectHierarchy(brainId, "Brain", datastore);
 * std::cout << "Total neurons: " << brainStats.totalNeurons << std::endl;
 * 
 * // Analyze connectivity of a specific neuron
 * auto connStats = inspector.analyzeConnectivity(neuronId, datastore);
 * std::cout << "In-degree: " << connStats.inDegree << std::endl;
 * 
 * // Find orphaned objects
 * auto orphans = inspector.findOrphans(datastore);
 * std::cout << "Orphaned neurons: " << orphans.orphanedNeurons.size() << std::endl;
 * 
 * // Generate comprehensive report
 * std::string report = inspector.generateReport(ReportFormat::MARKDOWN);
 * @endcode
 */
class NetworkInspector {
public:
    /**
     * @brief Constructor
     */
    NetworkInspector();

    /**
     * @brief Inspect a complete brain hierarchy
     * @param brainId ID of the brain to inspect
     * @param datastore Reference to the datastore
     * @return Statistics about the brain hierarchy
     */
    HierarchyStats inspectBrain(uint64_t brainId, Datastore& datastore);

    /**
     * @brief Inspect any hierarchical structure (Brain, Hemisphere, Lobe, etc.)
     * @param rootId ID of the root object
     * @param typeName Type of the root object ("Brain", "Hemisphere", etc.)
     * @param datastore Reference to the datastore
     * @return Statistics about the hierarchy rooted at this object
     */
    HierarchyStats inspectHierarchy(uint64_t rootId, const std::string& typeName, Datastore& datastore);

    /**
     * @brief Analyze connectivity for a specific neuron
     * @param neuronId ID of the neuron to analyze
     * @param datastore Reference to the datastore
     * @return Connectivity statistics for the neuron
     */
    ConnectivityStats analyzeConnectivity(uint64_t neuronId, Datastore& datastore);

    /**
     * @brief Get detailed state information for a neuron
     * @param neuronId ID of the neuron to inspect
     * @param datastore Reference to the datastore
     * @return Statistics about the neuron's internal state
     */
    NeuronStats inspectNeuron(uint64_t neuronId, Datastore& datastore);

    /**
     * @brief Find all orphaned and dangling objects in the network
     * @param datastore Reference to the datastore
     * @return Report of all orphaned/dangling objects
     */
    OrphanReport findOrphans(Datastore& datastore);

    /**
     * @brief Get connectivity statistics for all neurons in a cluster
     * @param clusterId ID of the cluster
     * @param datastore Reference to the datastore
     * @return Vector of connectivity statistics for each neuron
     */
    std::vector<ConnectivityStats> analyzeClusterConnectivity(uint64_t clusterId, Datastore& datastore);

    /**
     * @brief Get connectivity statistics for all neurons in a layer
     * @param layerId ID of the layer
     * @param datastore Reference to the datastore
     * @return Vector of connectivity statistics for each neuron
     */
    std::vector<ConnectivityStats> analyzeLayerConnectivity(uint64_t layerId, Datastore& datastore);

    /**
     * @brief Generate a comprehensive inspection report
     * @param format Output format (TEXT, JSON, MARKDOWN, CSV)
     * @return Formatted report string
     */
    std::string generateReport(ReportFormat format) const;

    /**
     * @brief Clear all cached inspection data
     */
    void clearCache();

    /**
     * @brief Get the last inspected hierarchy statistics
     * @return Vector of all hierarchy statistics from last inspection
     */
    const std::vector<HierarchyStats>& getLastHierarchyStats() const { return hierarchyCache_; }

    /**
     * @brief Get the last connectivity analysis results
     * @return Vector of all connectivity statistics from last analysis
     */
    const std::vector<ConnectivityStats>& getLastConnectivityStats() const { return connectivityCache_; }

    /**
     * @brief Get the last orphan detection results
     * @return Last orphan report
     */
    const OrphanReport& getLastOrphanReport() const { return orphanReport_; }

private:
    // Helper methods for hierarchical traversal
    void inspectHemisphere(uint64_t hemisphereId, Datastore& datastore, HierarchyStats& stats);
    void inspectLobe(uint64_t lobeId, Datastore& datastore, HierarchyStats& stats);
    void inspectRegion(uint64_t regionId, Datastore& datastore, HierarchyStats& stats);
    void inspectNucleus(uint64_t nucleusId, Datastore& datastore, HierarchyStats& stats);
    void inspectColumn(uint64_t columnId, Datastore& datastore, HierarchyStats& stats);
    void inspectLayer(uint64_t layerId, Datastore& datastore, HierarchyStats& stats);
    void inspectCluster(uint64_t clusterId, Datastore& datastore, HierarchyStats& stats);

    // Helper methods for connectivity analysis
    std::vector<uint64_t> findPresynapticNeurons(uint64_t neuronId, Datastore& datastore);
    std::vector<uint64_t> findPostsynapticNeurons(uint64_t neuronId, Datastore& datastore);
    double computeAverageWeight(const std::vector<uint64_t>& synapseIds, Datastore& datastore);

    // Helper methods for report generation
    std::string generateTextReport() const;
    std::string generateJsonReport() const;
    std::string generateMarkdownReport() const;
    std::string generateCsvReport() const;

    // Cached inspection results
    std::vector<HierarchyStats> hierarchyCache_;
    std::vector<ConnectivityStats> connectivityCache_;
    std::vector<NeuronStats> neuronStatsCache_;
    OrphanReport orphanReport_;
};

} // namespace snnfw

#endif // SNNFW_NETWORK_INSPECTOR_H

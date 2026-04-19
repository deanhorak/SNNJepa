#include "snnfw/NetworkInspector.h"
#include "snnfw/Brain.h"
#include "snnfw/Hemisphere.h"
#include "snnfw/Lobe.h"
#include "snnfw/Region.h"
#include "snnfw/Nucleus.h"
#include "snnfw/Column.h"
#include "snnfw/Layer.h"
#include "snnfw/Cluster.h"
#include "snnfw/Neuron.h"
#include "snnfw/Axon.h"
#include "snnfw/Dendrite.h"
#include "snnfw/Synapse.h"
#include "snnfw/Logger.h"
#include <nlohmann/json.hpp>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>

namespace snnfw {

NetworkInspector::NetworkInspector() {
    // Constructor
}

HierarchyStats NetworkInspector::inspectBrain(uint64_t brainId, Datastore& datastore) {
    return inspectHierarchy(brainId, "Brain", datastore);
}

HierarchyStats NetworkInspector::inspectHierarchy(uint64_t rootId, const std::string& typeName, Datastore& datastore) {
    HierarchyStats stats;
    stats.id = rootId;
    stats.typeName = typeName;
    stats.childCount = 0;
    stats.totalNeurons = 0;
    stats.totalSynapses = 0;
    stats.totalClusters = 0;
    stats.depth = 0;

    // Dispatch to appropriate inspector based on type
    if (typeName == "Brain") {
        auto brain = datastore.getBrain(rootId);
        if (!brain) {
            SNNFW_ERROR("Brain with ID {} not found", rootId);
            return stats;
        }
        stats.name = brain->getName();
        stats.childIds = brain->getHemisphereIds();
        stats.childCount = stats.childIds.size();

        // Recursively inspect hemispheres (accumulate totals in stats)
        for (uint64_t hemisphereId : stats.childIds) {
            HierarchyStats childStats;
            childStats.id = 0;
            childStats.typeName = "";
            childStats.name = "";
            childStats.childCount = 0;
            childStats.totalNeurons = 0;
            childStats.totalSynapses = 0;
            childStats.totalClusters = 0;
            childStats.depth = 0;
            inspectHemisphere(hemisphereId, datastore, childStats);
            // Accumulate totals from child
            stats.totalNeurons += childStats.totalNeurons;
            stats.totalSynapses += childStats.totalSynapses;
            stats.totalClusters += childStats.totalClusters;
        }
    } else if (typeName == "Hemisphere") {
        auto hemisphere = datastore.getHemisphere(rootId);
        if (!hemisphere) {
            SNNFW_ERROR("Hemisphere with ID {} not found", rootId);
            return stats;
        }
        stats.name = hemisphere->getName();
        stats.childIds = hemisphere->getLobeIds();
        stats.childCount = stats.childIds.size();

        // Recursively inspect lobes
        for (uint64_t lobeId : stats.childIds) {
            HierarchyStats childStats;
            childStats.id = 0;
            childStats.typeName = "";
            childStats.name = "";
            childStats.childCount = 0;
            childStats.totalNeurons = 0;
            childStats.totalSynapses = 0;
            childStats.totalClusters = 0;
            childStats.depth = 0;
            inspectLobe(lobeId, datastore, childStats);
            stats.totalNeurons += childStats.totalNeurons;
            stats.totalSynapses += childStats.totalSynapses;
            stats.totalClusters += childStats.totalClusters;
        }
    } else if (typeName == "Lobe") {
        auto lobe = datastore.getLobe(rootId);
        if (lobe) {
            stats.name = lobe->getName();
            stats.childIds = lobe->getRegionIds();
            stats.childCount = stats.childIds.size();
        }
        inspectLobe(rootId, datastore, stats);
    } else if (typeName == "Region") {
        auto region = datastore.getRegion(rootId);
        if (region) {
            stats.name = region->getName();
            stats.childIds = region->getNucleusIds();
            stats.childCount = stats.childIds.size();
        }
        inspectRegion(rootId, datastore, stats);
    } else if (typeName == "Nucleus") {
        auto nucleus = datastore.getNucleus(rootId);
        if (nucleus) {
            stats.name = nucleus->getName();
            stats.childIds = nucleus->getColumnIds();
            stats.childCount = stats.childIds.size();
        }
        inspectNucleus(rootId, datastore, stats);
    } else if (typeName == "Column") {
        auto column = datastore.getColumn(rootId);
        if (column) {
            stats.childIds = column->getLayerIds();
            stats.childCount = stats.childIds.size();
        }
        inspectColumn(rootId, datastore, stats);
    } else if (typeName == "Layer") {
        auto layer = datastore.getLayer(rootId);
        if (layer) {
            stats.childIds = layer->getClusterIds();
            stats.childCount = stats.childIds.size();
        }
        inspectLayer(rootId, datastore, stats);
    } else if (typeName == "Cluster") {
        auto cluster = datastore.getCluster(rootId);
        if (cluster) {
            stats.childIds = cluster->getNeuronIds();
            stats.childCount = stats.childIds.size();
        }
        inspectCluster(rootId, datastore, stats);
    } else {
        SNNFW_ERROR("Unknown hierarchy type: {}", typeName);
    }

    // Cache the result
    hierarchyCache_.push_back(stats);

    return stats;
}

void NetworkInspector::inspectHemisphere(uint64_t hemisphereId, Datastore& datastore, HierarchyStats& stats) {
    auto hemisphere = datastore.getHemisphere(hemisphereId);
    if (!hemisphere) {
        SNNFW_ERROR("Hemisphere with ID {} not found", hemisphereId);
        return;
    }

    stats.id = hemisphereId;
    stats.typeName = "Hemisphere";
    stats.name = hemisphere->getName();
    stats.childIds = hemisphere->getLobeIds();
    stats.childCount = stats.childIds.size();

    // Make a copy of childIds before iterating
    auto lobeIds = stats.childIds;

    // Recursively inspect lobes
    for (uint64_t lobeId : lobeIds) {
        HierarchyStats lobeStats;  // Create new stats object for child
        inspectLobe(lobeId, datastore, lobeStats);
        // Accumulate child statistics
        stats.totalNeurons += lobeStats.totalNeurons;
        stats.totalSynapses += lobeStats.totalSynapses;
        stats.totalClusters += lobeStats.totalClusters;
        stats.depth = std::max(stats.depth, lobeStats.depth + 1);
    }
}

void NetworkInspector::inspectLobe(uint64_t lobeId, Datastore& datastore, HierarchyStats& stats) {
    auto lobe = datastore.getLobe(lobeId);
    if (!lobe) {
        SNNFW_ERROR("Lobe with ID {} not found", lobeId);
        return;
    }

    stats.id = lobeId;
    stats.typeName = "Lobe";
    stats.name = lobe->getName();
    stats.childIds = lobe->getRegionIds();
    stats.childCount = stats.childIds.size();

    // Make a copy of childIds before iterating
    auto regionIds = stats.childIds;

    // Recursively inspect regions
    for (uint64_t regionId : regionIds) {
        HierarchyStats regionStats;  // Create new stats object for child
        inspectRegion(regionId, datastore, regionStats);
        // Accumulate child statistics
        stats.totalNeurons += regionStats.totalNeurons;
        stats.totalSynapses += regionStats.totalSynapses;
        stats.totalClusters += regionStats.totalClusters;
        stats.depth = std::max(stats.depth, regionStats.depth + 1);
    }
}

void NetworkInspector::inspectRegion(uint64_t regionId, Datastore& datastore, HierarchyStats& stats) {
    auto region = datastore.getRegion(regionId);
    if (!region) {
        SNNFW_ERROR("Region with ID {} not found", regionId);
        return;
    }

    stats.id = regionId;
    stats.typeName = "Region";
    stats.name = region->getName();
    stats.childIds = region->getNucleusIds();
    stats.childCount = stats.childIds.size();

    // Make a copy of childIds before iterating
    auto nucleusIds = stats.childIds;

    // Recursively inspect nuclei
    for (uint64_t nucleusId : nucleusIds) {
        HierarchyStats nucleusStats;  // Create new stats object for child
        inspectNucleus(nucleusId, datastore, nucleusStats);
        // Accumulate child statistics
        stats.totalNeurons += nucleusStats.totalNeurons;
        stats.totalSynapses += nucleusStats.totalSynapses;
        stats.totalClusters += nucleusStats.totalClusters;
        stats.depth = std::max(stats.depth, nucleusStats.depth + 1);
    }
}

void NetworkInspector::inspectNucleus(uint64_t nucleusId, Datastore& datastore, HierarchyStats& stats) {
    auto nucleus = datastore.getNucleus(nucleusId);
    if (!nucleus) {
        SNNFW_ERROR("Nucleus with ID {} not found", nucleusId);
        return;
    }

    stats.id = nucleusId;
    stats.typeName = "Nucleus";
    stats.name = nucleus->getName();
    stats.childIds = nucleus->getColumnIds();
    stats.childCount = stats.childIds.size();

    // Make a copy of childIds before iterating
    auto columnIds = stats.childIds;

    // Recursively inspect columns
    for (uint64_t columnId : columnIds) {
        HierarchyStats columnStats;  // Create new stats object for child
        inspectColumn(columnId, datastore, columnStats);
        // Accumulate child statistics
        stats.totalNeurons += columnStats.totalNeurons;
        stats.totalSynapses += columnStats.totalSynapses;
        stats.totalClusters += columnStats.totalClusters;
        stats.depth = std::max(stats.depth, columnStats.depth + 1);
    }
}

void NetworkInspector::inspectColumn(uint64_t columnId, Datastore& datastore, HierarchyStats& stats) {
    auto column = datastore.getColumn(columnId);
    if (!column) {
        SNNFW_ERROR("Column with ID {} not found", columnId);
        return;
    }

    stats.id = columnId;
    stats.typeName = "Column";
    stats.name = "Column_" + std::to_string(columnId);  // Column doesn't have a name field
    stats.childIds = column->getLayerIds();
    stats.childCount = stats.childIds.size();

    // Make a copy of childIds before iterating
    auto layerIds = stats.childIds;

    // Recursively inspect layers
    for (uint64_t layerId : layerIds) {
        HierarchyStats layerStats;  // Create new stats object for child
        inspectLayer(layerId, datastore, layerStats);
        // Accumulate child statistics
        stats.totalNeurons += layerStats.totalNeurons;
        stats.totalSynapses += layerStats.totalSynapses;
        stats.totalClusters += layerStats.totalClusters;
        stats.depth = std::max(stats.depth, layerStats.depth + 1);
    }
}

void NetworkInspector::inspectLayer(uint64_t layerId, Datastore& datastore, HierarchyStats& stats) {
    auto layer = datastore.getLayer(layerId);
    if (!layer) {
        SNNFW_ERROR("Layer with ID {} not found", layerId);
        return;
    }

    stats.id = layerId;
    stats.typeName = "Layer";
    stats.name = "Layer_" + std::to_string(layerId);  // Layer doesn't have a name field
    stats.childIds = layer->getClusterIds();
    stats.childCount = stats.childIds.size();

    // Make a copy of childIds before iterating
    auto clusterIds = stats.childIds;

    // Recursively inspect clusters
    for (uint64_t clusterId : clusterIds) {
        HierarchyStats clusterStats;  // Create new stats object for child
        inspectCluster(clusterId, datastore, clusterStats);
        // Accumulate child statistics
        stats.totalNeurons += clusterStats.totalNeurons;
        stats.totalSynapses += clusterStats.totalSynapses;
        stats.totalClusters += clusterStats.totalClusters;
        stats.depth = std::max(stats.depth, clusterStats.depth + 1);
    }
}

void NetworkInspector::inspectCluster(uint64_t clusterId, Datastore& datastore, HierarchyStats& stats) {
    auto cluster = datastore.getCluster(clusterId);
    if (!cluster) {
        SNNFW_ERROR("Cluster with ID {} not found", clusterId);
        return;
    }

    stats.id = clusterId;
    stats.typeName = "Cluster";
    stats.childIds = cluster->getNeuronIds();
    stats.childCount = stats.childIds.size();
    stats.totalClusters++;
    stats.totalNeurons += stats.childIds.size();

    // Count synapses for each neuron
    for (uint64_t neuronId : stats.childIds) {
        auto neuron = datastore.getNeuron(neuronId);
        if (neuron) {
            uint64_t axonId = neuron->getAxonId();
            if (axonId != 0) {
                auto axon = datastore.getAxon(axonId);
                if (axon) {
                    stats.totalSynapses += axon->getSynapseIds().size();
                }
            }
        }
    }
}

ConnectivityStats NetworkInspector::analyzeConnectivity(uint64_t neuronId, Datastore& datastore) {
    ConnectivityStats stats;
    stats.neuronId = neuronId;
    stats.inDegree = 0;
    stats.outDegree = 0;
    stats.avgInWeight = 0.0;
    stats.avgOutWeight = 0.0;

    auto neuron = datastore.getNeuron(neuronId);
    if (!neuron) {
        SNNFW_ERROR("Neuron with ID {} not found", neuronId);
        return stats;
    }

    // Analyze outgoing connections (via axon)
    uint64_t axonId = neuron->getAxonId();
    if (axonId != 0) {
        auto axon = datastore.getAxon(axonId);
        if (axon) {
            const auto& synapseIds = axon->getSynapseIds();
            stats.outDegree = synapseIds.size();
            
            std::vector<double> weights;
            for (uint64_t synapseId : synapseIds) {
                auto synapse = datastore.getSynapse(synapseId);
                if (synapse) {
                    weights.push_back(synapse->getWeight());

                    // Find postsynaptic neuron
                    uint64_t dendriteId = synapse->getDendriteId();
                    auto dendrite = datastore.getDendrite(dendriteId);
                    if (dendrite) {
                        uint64_t postNeuronId = dendrite->getTargetNeuronId();
                        if (std::find(stats.postsynapticNeurons.begin(),
                                    stats.postsynapticNeurons.end(),
                                    postNeuronId) == stats.postsynapticNeurons.end()) {
                            stats.postsynapticNeurons.push_back(postNeuronId);
                        }
                    }
                }
            }
            
            if (!weights.empty()) {
                stats.avgOutWeight = std::accumulate(weights.begin(), weights.end(), 0.0) / weights.size();
            }
        }
    }

    // Analyze incoming connections (via dendrites)
    const auto& dendriteIds = neuron->getDendriteIds();
    std::vector<double> inWeights;
    
    for (uint64_t dendriteId : dendriteIds) {
        auto dendrite = datastore.getDendrite(dendriteId);
        if (dendrite) {
            const auto& synapseIds = dendrite->getSynapseIds();
            stats.inDegree += synapseIds.size();
            
            for (uint64_t synapseId : synapseIds) {
                auto synapse = datastore.getSynapse(synapseId);
                if (synapse) {
                    inWeights.push_back(synapse->getWeight());

                    // Find presynaptic neuron
                    uint64_t axonId = synapse->getAxonId();
                    auto axon = datastore.getAxon(axonId);
                    if (axon) {
                        uint64_t preNeuronId = axon->getSourceNeuronId();
                        if (std::find(stats.presynapticNeurons.begin(),
                                    stats.presynapticNeurons.end(),
                                    preNeuronId) == stats.presynapticNeurons.end()) {
                            stats.presynapticNeurons.push_back(preNeuronId);
                        }
                    }
                }
            }
        }
    }
    
    if (!inWeights.empty()) {
        stats.avgInWeight = std::accumulate(inWeights.begin(), inWeights.end(), 0.0) / inWeights.size();
    }

    // Cache the result
    connectivityCache_.push_back(stats);
    
    return stats;
}

NeuronStats NetworkInspector::inspectNeuron(uint64_t neuronId, Datastore& datastore) {
    NeuronStats stats;
    stats.neuronId = neuronId;
    stats.learnedPatternCount = 0;
    stats.currentSpikeCount = 0;
    stats.bestSimilarity = -1.0;
    stats.windowSizeMs = 0.0;
    stats.similarityThreshold = 0.0;
    stats.maxReferencePatterns = 0;
    stats.axonId = 0;

    auto neuron = datastore.getNeuron(neuronId);
    if (!neuron) {
        SNNFW_ERROR("Neuron with ID {} not found", neuronId);
        return stats;
    }

    stats.learnedPatternCount = neuron->getLearnedPatternCount();
    stats.currentSpikeCount = neuron->getSpikes().size();
    stats.bestSimilarity = neuron->getBestSimilarity();
    stats.windowSizeMs = neuron->getWindowSize();
    stats.similarityThreshold = neuron->getSimilarityThreshold();
    stats.maxReferencePatterns = neuron->getMaxReferencePatterns();
    stats.axonId = neuron->getAxonId();
    stats.dendriteIds = neuron->getDendriteIds();
    
    // Convert similarity metric enum to string
    switch (neuron->getSimilarityMetric()) {
        case SimilarityMetric::COSINE:
            stats.similarityMetric = "COSINE";
            break;
        case SimilarityMetric::HISTOGRAM:
            stats.similarityMetric = "HISTOGRAM";
            break;
        case SimilarityMetric::EUCLIDEAN:
            stats.similarityMetric = "EUCLIDEAN";
            break;
        case SimilarityMetric::CORRELATION:
            stats.similarityMetric = "CORRELATION";
            break;
        case SimilarityMetric::WAVEFORM:
            stats.similarityMetric = "WAVEFORM";
            break;
        default:
            stats.similarityMetric = "UNKNOWN";
    }

    // Cache the result
    neuronStatsCache_.push_back(stats);

    return stats;
}

OrphanReport NetworkInspector::findOrphans(Datastore& datastore) {
    OrphanReport report;

    // This is a simplified implementation
    // A full implementation would scan the entire datastore
    // For now, we'll just initialize empty vectors

    SNNFW_WARN("findOrphans() is not yet fully implemented - returns empty report");

    orphanReport_ = report;
    return report;
}

std::vector<ConnectivityStats> NetworkInspector::analyzeClusterConnectivity(
    uint64_t clusterId, Datastore& datastore) {

    std::vector<ConnectivityStats> results;

    auto cluster = datastore.getCluster(clusterId);
    if (!cluster) {
        SNNFW_ERROR("Cluster with ID {} not found", clusterId);
        return results;
    }

    const auto& neuronIds = cluster->getNeuronIds();
    for (uint64_t neuronId : neuronIds) {
        results.push_back(analyzeConnectivity(neuronId, datastore));
    }

    return results;
}

std::vector<ConnectivityStats> NetworkInspector::analyzeLayerConnectivity(
    uint64_t layerId, Datastore& datastore) {

    std::vector<ConnectivityStats> results;

    auto layer = datastore.getLayer(layerId);
    if (!layer) {
        SNNFW_ERROR("Layer with ID {} not found", layerId);
        return results;
    }

    const auto& clusterIds = layer->getClusterIds();
    for (uint64_t clusterId : clusterIds) {
        auto clusterStats = analyzeClusterConnectivity(clusterId, datastore);
        results.insert(results.end(), clusterStats.begin(), clusterStats.end());
    }

    return results;
}

void NetworkInspector::clearCache() {
    hierarchyCache_.clear();
    connectivityCache_.clear();
    neuronStatsCache_.clear();
    orphanReport_ = OrphanReport();
}

std::string NetworkInspector::generateReport(ReportFormat format) const {
    switch (format) {
        case ReportFormat::TEXT:
            return generateTextReport();
        case ReportFormat::JSON:
            return generateJsonReport();
        case ReportFormat::MARKDOWN:
            return generateMarkdownReport();
        case ReportFormat::CSV:
            return generateCsvReport();
        default:
            return "Unknown report format";
    }
}

std::string NetworkInspector::generateTextReport() const {
    std::ostringstream oss;

    oss << "=== Network Inspection Report ===\n\n";

    // Hierarchy statistics
    if (!hierarchyCache_.empty()) {
        oss << "--- Hierarchy Statistics ---\n";
        for (const auto& stats : hierarchyCache_) {
            oss << stats.typeName << " (ID: " << stats.id << ")\n";
            if (!stats.name.empty()) {
                oss << "  Name: " << stats.name << "\n";
            }
            oss << "  Children: " << stats.childCount << "\n";
            oss << "  Total Neurons: " << stats.totalNeurons << "\n";
            oss << "  Total Synapses: " << stats.totalSynapses << "\n";
            oss << "  Total Clusters: " << stats.totalClusters << "\n";
            oss << "\n";
        }
    }

    // Connectivity statistics
    if (!connectivityCache_.empty()) {
        oss << "--- Connectivity Statistics ---\n";
        for (const auto& stats : connectivityCache_) {
            oss << "Neuron ID: " << stats.neuronId << "\n";
            oss << "  In-degree: " << stats.inDegree << "\n";
            oss << "  Out-degree: " << stats.outDegree << "\n";
            oss << "  Avg In Weight: " << std::fixed << std::setprecision(4)
                << stats.avgInWeight << "\n";
            oss << "  Avg Out Weight: " << std::fixed << std::setprecision(4)
                << stats.avgOutWeight << "\n";
            oss << "\n";
        }
    }

    // Neuron statistics
    if (!neuronStatsCache_.empty()) {
        oss << "--- Neuron State Statistics ---\n";
        for (const auto& stats : neuronStatsCache_) {
            oss << "Neuron ID: " << stats.neuronId << "\n";
            oss << "  Learned Patterns: " << stats.learnedPatternCount << "\n";
            oss << "  Current Spikes: " << stats.currentSpikeCount << "\n";
            oss << "  Best Similarity: " << std::fixed << std::setprecision(4)
                << stats.bestSimilarity << "\n";
            oss << "  Similarity Metric: " << stats.similarityMetric << "\n";
            oss << "\n";
        }
    }

    return oss.str();
}

std::string NetworkInspector::generateJsonReport() const {
    nlohmann::json report;

    // Hierarchy statistics
    nlohmann::json hierarchyArray = nlohmann::json::array();
    for (const auto& stats : hierarchyCache_) {
        nlohmann::json h;
        h["type"] = stats.typeName;
        h["id"] = stats.id;
        h["name"] = stats.name;
        h["childCount"] = stats.childCount;
        h["totalNeurons"] = stats.totalNeurons;
        h["totalSynapses"] = stats.totalSynapses;
        h["totalClusters"] = stats.totalClusters;
        hierarchyArray.push_back(h);
    }
    report["hierarchy"] = hierarchyArray;

    // Connectivity statistics
    nlohmann::json connectivityArray = nlohmann::json::array();
    for (const auto& stats : connectivityCache_) {
        nlohmann::json c;
        c["neuronId"] = stats.neuronId;
        c["inDegree"] = stats.inDegree;
        c["outDegree"] = stats.outDegree;
        c["avgInWeight"] = stats.avgInWeight;
        c["avgOutWeight"] = stats.avgOutWeight;
        connectivityArray.push_back(c);
    }
    report["connectivity"] = connectivityArray;

    // Neuron statistics
    nlohmann::json neuronArray = nlohmann::json::array();
    for (const auto& stats : neuronStatsCache_) {
        nlohmann::json n;
        n["neuronId"] = stats.neuronId;
        n["learnedPatternCount"] = stats.learnedPatternCount;
        n["currentSpikeCount"] = stats.currentSpikeCount;
        n["bestSimilarity"] = stats.bestSimilarity;
        n["similarityMetric"] = stats.similarityMetric;
        neuronArray.push_back(n);
    }
    report["neurons"] = neuronArray;

    return report.dump(2);
}

std::string NetworkInspector::generateMarkdownReport() const {
    std::ostringstream oss;

    oss << "# Network Inspection Report\n\n";

    // Hierarchy statistics
    if (!hierarchyCache_.empty()) {
        oss << "## Hierarchy Statistics\n\n";
        oss << "| Type | ID | Name | Children | Neurons | Synapses | Clusters |\n";
        oss << "|------|----|----- |----------|---------|----------|----------|\n";
        for (const auto& stats : hierarchyCache_) {
            oss << "| " << stats.typeName
                << " | " << stats.id
                << " | " << stats.name
                << " | " << stats.childCount
                << " | " << stats.totalNeurons
                << " | " << stats.totalSynapses
                << " | " << stats.totalClusters
                << " |\n";
        }
        oss << "\n";
    }

    // Connectivity statistics
    if (!connectivityCache_.empty()) {
        oss << "## Connectivity Statistics\n\n";
        oss << "| Neuron ID | In-Degree | Out-Degree | Avg In Weight | Avg Out Weight |\n";
        oss << "|-----------|-----------|------------|---------------|----------------|\n";
        for (const auto& stats : connectivityCache_) {
            oss << "| " << stats.neuronId
                << " | " << stats.inDegree
                << " | " << stats.outDegree
                << " | " << std::fixed << std::setprecision(4) << stats.avgInWeight
                << " | " << std::fixed << std::setprecision(4) << stats.avgOutWeight
                << " |\n";
        }
        oss << "\n";
    }

    // Neuron statistics
    if (!neuronStatsCache_.empty()) {
        oss << "## Neuron State Statistics\n\n";
        oss << "| Neuron ID | Patterns | Spikes | Best Similarity | Metric |\n";
        oss << "|-----------|----------|--------|-----------------|--------|\n";
        for (const auto& stats : neuronStatsCache_) {
            oss << "| " << stats.neuronId
                << " | " << stats.learnedPatternCount
                << " | " << stats.currentSpikeCount
                << " | " << std::fixed << std::setprecision(4) << stats.bestSimilarity
                << " | " << stats.similarityMetric
                << " |\n";
        }
        oss << "\n";
    }

    return oss.str();
}

std::string NetworkInspector::generateCsvReport() const {
    std::ostringstream oss;

    // Hierarchy CSV
    if (!hierarchyCache_.empty()) {
        oss << "# Hierarchy Statistics\n";
        oss << "Type,ID,Name,Children,Neurons,Synapses,Clusters\n";
        for (const auto& stats : hierarchyCache_) {
            oss << stats.typeName << ","
                << stats.id << ","
                << stats.name << ","
                << stats.childCount << ","
                << stats.totalNeurons << ","
                << stats.totalSynapses << ","
                << stats.totalClusters << "\n";
        }
        oss << "\n";
    }

    // Connectivity CSV
    if (!connectivityCache_.empty()) {
        oss << "# Connectivity Statistics\n";
        oss << "NeuronID,InDegree,OutDegree,AvgInWeight,AvgOutWeight\n";
        for (const auto& stats : connectivityCache_) {
            oss << stats.neuronId << ","
                << stats.inDegree << ","
                << stats.outDegree << ","
                << std::fixed << std::setprecision(4) << stats.avgInWeight << ","
                << std::fixed << std::setprecision(4) << stats.avgOutWeight << "\n";
        }
        oss << "\n";
    }

    // Neuron CSV
    if (!neuronStatsCache_.empty()) {
        oss << "# Neuron State Statistics\n";
        oss << "NeuronID,Patterns,Spikes,BestSimilarity,Metric\n";
        for (const auto& stats : neuronStatsCache_) {
            oss << stats.neuronId << ","
                << stats.learnedPatternCount << ","
                << stats.currentSpikeCount << ","
                << std::fixed << std::setprecision(4) << stats.bestSimilarity << ","
                << stats.similarityMetric << "\n";
        }
    }

    return oss.str();
}

} // namespace snnfw


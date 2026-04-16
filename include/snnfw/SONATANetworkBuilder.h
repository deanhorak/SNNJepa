#ifndef SNNFW_SONATA_NETWORK_BUILDER_H
#define SNNFW_SONATA_NETWORK_BUILDER_H

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <bbp/sonata/nodes.h>
#include <bbp/sonata/edges.h>
#include "snnfw/Neuron.h"
#include "snnfw/Synapse.h"
#include "snnfw/Logger.h"

namespace snnfw {

/**
 * @brief Builder for neural networks from SONATA format files
 * 
 * SONATA (Scalable Open Network Architecture TemplAte) is a standard format
 * for representing large-scale neural network models. This class reads SONATA
 * files and constructs SNNFW neural network objects.
 * 
 * SONATA Format:
 * - Nodes (neurons): Stored in HDF5 files with properties (type, position, parameters)
 * - Edges (synapses): Stored in HDF5 files with connectivity and weights
 * 
 * Example usage:
 * @code
 * SONATANetworkBuilder builder("network.h5");
 * auto neurons = builder.buildNeurons("layer1_neurons");
 * auto synapses = builder.buildSynapses("layer1_connections");
 * @endcode
 * 
 * For MNIST experiments, the SONATA file describes:
 * - 392 neurons (49 regions × 8 orientations)
 * - Neuron properties: window_size, threshold, max_patterns
 * - Spatial organization: region_row, region_col, orientation
 */
class SONATANetworkBuilder {
public:
    /**
     * @brief Construct a SONATA network builder
     * @param nodesFile Path to SONATA nodes file (HDF5)
     * @param edgesFile Path to SONATA edges file (HDF5, optional)
     */
    explicit SONATANetworkBuilder(const std::string& nodesFile, 
                                  const std::string& edgesFile = "")
        : nodesFile_(nodesFile), edgesFile_(edgesFile) {
        
        SNNFW_INFO("Initializing SONATA network builder");
        SNNFW_INFO("  Nodes file: {}", nodesFile);
        if (!edgesFile.empty()) {
            SNNFW_INFO("  Edges file: {}", edgesFile);
        }
    }
    
    /**
     * @brief Build neurons from SONATA nodes
     * @param populationName Name of the node population to load
     * @return Vector of neurons organized by [region][orientation][neuron_idx]
     */
    std::vector<std::vector<std::vector<std::shared_ptr<Neuron>>>> 
    buildNeurons(const std::string& populationName,
                 int gridSize, int numOrientations, int neuronsPerFeature) {
        
        SNNFW_INFO("Building neurons from SONATA population: {}", populationName);
        
        // Initialize 3D structure
        int numRegions = gridSize * gridSize;
        std::vector<std::vector<std::vector<std::shared_ptr<Neuron>>>> neurons(
            numRegions,
            std::vector<std::vector<std::shared_ptr<Neuron>>>(
                numOrientations,
                std::vector<std::shared_ptr<Neuron>>(neuronsPerFeature)
            )
        );
        
        try {
            // Open SONATA nodes file
            bbp::sonata::NodeStorage storage(nodesFile_);
            auto population = storage.openPopulation(populationName);
            
            // Get node count
            auto selection = population->selectAll();
            size_t nodeCount = selection.flatSize();
            SNNFW_INFO("  Found {} nodes in population", nodeCount);

            // Read neuron properties
            auto windowSizes = population->getAttribute<double>("window_size_ms", selection);
            auto thresholds = population->getAttribute<double>("similarity_threshold", selection);
            auto maxPatterns = population->getAttribute<uint64_t>("max_patterns", selection);
            auto regionRows = population->getAttribute<int32_t>("region_row", selection);
            auto regionCols = population->getAttribute<int32_t>("region_col", selection);
            auto orientations = population->getAttribute<int32_t>("orientation", selection);
            auto neuronIndices = population->getAttribute<int32_t>("neuron_index", selection);
            
            // Create neurons
            for (size_t i = 0; i < nodeCount; ++i) {
                int regionRow = regionRows[i];
                int regionCol = regionCols[i];
                int orientation = orientations[i];
                int neuronIdx = neuronIndices[i];
                
                int regionId = regionRow * gridSize + regionCol;
                
                auto neuron = std::make_shared<Neuron>(
                    windowSizes[i],
                    thresholds[i],
                    maxPatterns[i],
                    i  // Use SONATA node ID as neuron ID
                );
                
                neurons[regionId][orientation][neuronIdx] = neuron;
            }
            
            SNNFW_INFO("  Successfully created {} neurons", nodeCount);
            
        } catch (const std::exception& e) {
            SNNFW_ERROR("Failed to build neurons from SONATA: {}", e.what());
            throw;
        }
        
        return neurons;
    }
    
    /**
     * @brief Build synapses from SONATA edges
     * @param populationName Name of the edge population to load
     * @return Vector of synapses
     */
    std::vector<std::shared_ptr<Synapse>> buildSynapses(const std::string& populationName) {
        std::vector<std::shared_ptr<Synapse>> synapses;
        
        if (edgesFile_.empty()) {
            SNNFW_WARN("No edges file specified, skipping synapse creation");
            return synapses;
        }
        
        SNNFW_INFO("Building synapses from SONATA population: {}", populationName);
        
        try {
            // Open SONATA edges file
            bbp::sonata::EdgeStorage storage(edgesFile_);
            auto population = storage.openPopulation(populationName);
            
            // Get edge count
            auto selection = population->selectAll();
            size_t edgeCount = selection.flatSize();
            SNNFW_INFO("  Found {} edges in population", edgeCount);
            
            // Read edge properties
            auto sourceNodeIds = population->sourceNodeIDs(selection);
            auto targetNodeIds = population->targetNodeIDs(selection);
            auto weights = population->getAttribute<double>("weight", selection);
            auto delays = population->getAttribute<double>("delay", selection);
            
            // Create synapses
            for (size_t i = 0; i < edgeCount; ++i) {
                auto synapse = std::make_shared<Synapse>(
                    sourceNodeIds[i],  // axonId
                    targetNodeIds[i],  // dendriteId
                    weights[i],
                    delays[i]
                );
                synapses.push_back(synapse);
            }
            
            SNNFW_INFO("  Successfully created {} synapses", edgeCount);
            
        } catch (const std::exception& e) {
            SNNFW_ERROR("Failed to build synapses from SONATA: {}", e.what());
            throw;
        }
        
        return synapses;
    }
    
    /**
     * @brief Get neuron metadata from SONATA file
     * @param populationName Name of the node population
     * @return Map of metadata key-value pairs
     */
    std::map<std::string, std::string> getNeuronMetadata(const std::string& populationName) {
        std::map<std::string, std::string> metadata;
        
        try {
            bbp::sonata::NodeStorage storage(nodesFile_);
            auto population = storage.openPopulation(populationName);
            
            // Get available attributes
            auto attributes = population->attributeNames();
            metadata["attribute_count"] = std::to_string(attributes.size());
            metadata["node_count"] = std::to_string(population->size());
            
            SNNFW_INFO("SONATA population '{}' metadata:", populationName);
            SNNFW_INFO("  Nodes: {}", metadata["node_count"]);
            SNNFW_INFO("  Attributes: {}", metadata["attribute_count"]);
            
        } catch (const std::exception& e) {
            SNNFW_ERROR("Failed to read metadata: {}", e.what());
        }
        
        return metadata;
    }

private:
    std::string nodesFile_;  ///< Path to SONATA nodes file
    std::string edgesFile_;  ///< Path to SONATA edges file
};

} // namespace snnfw

#endif // SNNFW_SONATA_NETWORK_BUILDER_H


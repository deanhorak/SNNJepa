#ifndef SNNFW_DECLARATIVE_NETWORK_CONSTRUCTOR_H
#define SNNFW_DECLARATIVE_NETWORK_CONSTRUCTOR_H

#include "snnfw/declarative/NetworkIR.h"
#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Datastore.h"
#include "snnfw/Brain.h"
#include "snnfw/SpikeProcessor.h"
#include "snnfw/NetworkPropagator.h"
#include "snnfw/ConnectivityPattern.h"
#include "snnfw/Neuron.h"
#include "snnfw/Synapse.h"
#include "snnfw/adapters/SensoryAdapter.h"
#include "snnfw/adapters/MotorAdapter.h"
#include <memory>
#include <vector>
#include <map>
#include <unordered_map>
#include <string>

namespace snnfw {
namespace declarative {

/**
 * @struct ConstructedNetwork
 * @brief Result of network construction from a NetworkIR
 *
 * Provides access to all constructed components: brain hierarchy,
 * runtime processors, named neuron groups, and synapse groups.
 */
struct ConstructedNetwork {
    struct PathGroup {
        std::string path;
        std::vector<std::shared_ptr<Neuron>> neurons;
    };

    std::shared_ptr<Brain> brain;
    std::shared_ptr<SpikeProcessor> spikeProcessor;
    std::shared_ptr<NetworkPropagator> propagator;

    /// Input layer neurons (row-major order)
    std::vector<std::shared_ptr<Neuron>> inputNeurons;

    /// Output populations (one vector per class)
    std::vector<std::vector<std::shared_ptr<Neuron>>> outputPopulations;

    /// Per-column structures for feature processing
    struct ColumnGroup {
        std::string name;
        std::string path;
        double orientation = 0.0;
        double spatialFrequency = 0.0;
        /// Neurons keyed by layer name: "L4" -> [...], "L5" -> [...]
        std::map<std::string, std::vector<std::shared_ptr<Neuron>>> layerNeurons;

        /// Tiled receptive field: selected tile indices for this column
        std::vector<int> tileIndices;
        /// Tiled receptive field: input pixel indices in this column's receptive field
        std::vector<int> inputMaskActiveIdx;
    };
    std::vector<ColumnGroup> columns;

    /// Exact full-path layer groups in construction order
    std::vector<PathGroup> layerGroups;
    /// Exact full-path population groups in construction order
    std::vector<PathGroup> populationGroups;
    /// Exact path lookups for layer groups
    std::unordered_map<std::string, size_t> layerGroupIndex;
    /// Exact path lookups for population groups
    std::unordered_map<std::string, size_t> populationGroupIndex;

    /// All synapses grouped by synapse group name
    std::map<std::string, std::vector<std::shared_ptr<Synapse>>> synapseGroups;

    /// Adapters instantiated from declarative config
    std::vector<std::shared_ptr<adapters::SensoryAdapter>> sensoryAdapters;
    std::vector<std::shared_ptr<adapters::MotorAdapter>> motorAdapters;

    /// All neuron IDs for bulk operations
    std::vector<uint64_t> allNeuronIds;

    /// All synapses (flat list)
    std::vector<std::shared_ptr<Synapse>> allSynapses;

    /// All axons
    std::vector<std::shared_ptr<Axon>> allAxons;

    /// All dendrites
    std::vector<std::shared_ptr<Dendrite>> allDendrites;

    /// The IR that was used to construct this network (for reference)
    NetworkIR sourceIR;
};

/**
 * @class NetworkConstructor
 * @brief Builds a running SNNFrame network from a NetworkIR
 *
 * Uses NetworkBuilder and ConnectivityBuilder internally. Construction
 * proceeds in four phases:
 * 1. Build hierarchy (Brain -> Hemisphere -> ... -> Layer)
 * 2. Create neurons (expand column templates, populate layers)
 * 3. Create connectivity (resolve paths, apply patterns)
 * 4. Initialize runtime (SpikeProcessor, NetworkPropagator)
 */
class NetworkConstructor {
public:
    NetworkConstructor(NeuralObjectFactory& factory, Datastore& datastore);

    /// Construct the full network from a validated IR
    ConstructedNetwork construct(const NetworkIR& ir);

private:
    NeuralObjectFactory& factory_;
    Datastore& datastore_;
    unsigned int connectivitySeed_ = 0;

    // Phase 1: Build hierarchy using NetworkBuilder
    void buildHierarchy(const NetworkIR& ir, ConstructedNetwork& result);

    // Phase 2: Create neurons from populations/templates
    void createNeurons(const NetworkIR& ir, ConstructedNetwork& result);

    // Phase 3: Create connectivity from projection rules
    void createConnectivity(const NetworkIR& ir, ConstructedNetwork& result);

    // Apply parity safeguards used by the legacy C++ experiment builder
    void applyConnectivitySafeguards(const NetworkIR& ir, ConstructedNetwork& result);

    // Phase 4: Initialize SpikeProcessor and NetworkPropagator
    void initializeRuntime(const NetworkIR& ir, ConstructedNetwork& result);

    // Helpers
    /// Expand a ColumnTemplateIR into explicit ColumnIR instances
    std::vector<ColumnIR> expandColumnTemplate(const ColumnTemplateIR& tmpl);

    /// Resolve a glob path like "V1/*/L4" to vectors of neurons per column
    std::vector<std::vector<std::shared_ptr<Neuron>>>
    resolveNeuronPath(const std::string& path, const ConstructedNetwork& result);

    /// Resolve a special path like "InputGrid" or "OutputLayer"
    std::vector<std::shared_ptr<Neuron>>
    resolveSpecialPath(const std::string& path, const ConstructedNetwork& result);

    /// Create a ConnectivityPattern from a ProjectionIR specification
    std::unique_ptr<ConnectivityPattern>
    createPattern(const ProjectionIR& proj);

    /// Map synapse group string to NetworkPropagator::SynapseGroup enum
    NetworkPropagator::SynapseGroup
    mapSynapseGroup(const std::string& groupName);

    /// Get NeuronParamsIR by name from the IR, falling back to defaults
    NeuronParamsIR resolveNeuronParams(const std::string& name, const NetworkIR& ir);
};

} // namespace declarative
} // namespace snnfw

#endif // SNNFW_DECLARATIVE_NETWORK_CONSTRUCTOR_H

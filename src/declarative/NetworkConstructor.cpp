#include "snnfw/declarative/NetworkConstructor.h"
#include "snnfw/NetworkBuilder.h"
#include "snnfw/ConnectivityBuilder.h"
#include "snnfw/ConnectivityPattern.h"
#include "snnfw/Logger.h"
#include <regex>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <random>
#include <unordered_set>
#include <cstdint>

namespace snnfw {
namespace declarative {

namespace {

uint32_t hashStringFnv1a(const std::string& text) {
    uint32_t h = 2166136261u;
    for (unsigned char c : text) {
        h ^= static_cast<uint32_t>(c);
        h *= 16777619u;
    }
    return h;
}

uint32_t mixSeed(uint32_t seed, uint32_t value) {
    // 32-bit mix from boost hash_combine to decorrelate sequential values.
    return seed ^ (value + 0x9e3779b9u + (seed << 6) + (seed >> 2));
}

uint32_t makeProjectionSeed(uint32_t baseSeed, const ProjectionIR& proj, size_t projectionIndex) {
    uint32_t seed = mixSeed(baseSeed, static_cast<uint32_t>(projectionIndex));
    seed = mixSeed(seed, hashStringFnv1a(proj.name));
    seed = mixSeed(seed, hashStringFnv1a(proj.source));
    seed = mixSeed(seed, hashStringFnv1a(proj.target));
    seed = mixSeed(seed, hashStringFnv1a(proj.pattern));
    seed = mixSeed(seed, hashStringFnv1a(proj.scope));
    return seed;
}

std::string joinHierarchyPath(std::initializer_list<std::string> parts) {
    std::ostringstream oss;
    bool first = true;
    for (const auto& part : parts) {
        if (part.empty()) {
            continue;
        }
        if (!first) {
            oss << "/";
        }
        oss << part;
        first = false;
    }
    return oss.str();
}

std::vector<std::string> splitPath(const std::string& path) {
    std::vector<std::string> parts;
    std::istringstream iss(path);
    std::string part;
    while (std::getline(iss, part, '/')) {
        if (!part.empty()) {
            parts.push_back(part);
        }
    }
    return parts;
}

bool pathMatchesPattern(const std::string& pattern, const std::string& candidate) {
    const auto patternParts = splitPath(pattern);
    const auto candidateParts = splitPath(candidate);
    if (patternParts.empty() || candidateParts.empty()) {
        return false;
    }
    if (patternParts.size() > candidateParts.size()) {
        return false;
    }

    const size_t offset = candidateParts.size() - patternParts.size();
    for (size_t i = 0; i < patternParts.size(); ++i) {
        if (patternParts[i] == "*") {
            continue;
        }
        if (patternParts[i] != candidateParts[i + offset]) {
            return false;
        }
    }
    return true;
}

} // namespace

NetworkConstructor::NetworkConstructor(NeuralObjectFactory& factory, Datastore& datastore)
    : factory_(factory), datastore_(datastore) {}

ConstructedNetwork NetworkConstructor::construct(const NetworkIR& ir) {
    // Validate first
    auto errors = ir.getValidationErrors();
    if (!errors.empty()) {
        std::string msg = "NetworkIR validation failed:\n";
        for (const auto& e : errors) {
            msg += "  - " + e + "\n";
        }
        throw std::runtime_error(msg);
    }

    ConstructedNetwork result;
    result.sourceIR = ir;
    connectivitySeed_ = ir.simulation.connectivitySeed;

    SNNFW_INFO("NetworkConstructor: Phase 1 - Building hierarchy...");
    buildHierarchy(ir, result);

    SNNFW_INFO("NetworkConstructor: Phase 2 - Creating neurons...");
    createNeurons(ir, result);

    SNNFW_INFO("NetworkConstructor: Phase 3 - Creating connectivity...");
    createConnectivity(ir, result);

    SNNFW_INFO("NetworkConstructor: Phase 3b - Applying connectivity safeguards...");
    applyConnectivitySafeguards(ir, result);

    SNNFW_INFO("NetworkConstructor: Phase 4 - Initializing runtime...");
    initializeRuntime(ir, result);

    SNNFW_INFO("NetworkConstructor: Construction complete. {} neurons, {} synapses, {} columns",
               result.allNeuronIds.size(), result.allSynapses.size(), result.columns.size());

    return result;
}

// ============================================================================
// Phase 1: Build hierarchy
// ============================================================================
void NetworkConstructor::buildHierarchy(const NetworkIR& ir, ConstructedNetwork& result) {
    NetworkBuilder builder(factory_, datastore_, false);  // no auto-validate

    builder.createBrain(ir.brain.name);

    for (const auto& hemiIR : ir.brain.hemispheres) {
        builder.addHemisphere(hemiIR.name);
        for (const auto& lobeIR : hemiIR.lobes) {
            builder.addLobe(lobeIR.name);
            for (const auto& regionIR : lobeIR.regions) {
                builder.addRegion(regionIR.name);
                for (const auto& nucleusIR : regionIR.nuclei) {
                    builder.addNucleus(nucleusIR.name);

                    // Expand column template if present
                    std::vector<ColumnIR> allColumns = nucleusIR.columns;
                    if (nucleusIR.columnTemplate.has_value()) {
                        auto expanded = expandColumnTemplate(nucleusIR.columnTemplate.value());
                        allColumns.insert(allColumns.end(), expanded.begin(), expanded.end());
                    }

                    for (const auto& colIR : allColumns) {
                        builder.addColumn(colIR.name);
                        for (const auto& layerIR : colIR.layers) {
                            builder.addLayer(layerIR.name);
                            // Populations are created in Phase 2
                            builder.up();  // back to column
                        }
                        builder.up();  // back to nucleus
                    }
                    builder.up();  // back to region
                }
                builder.up();  // back to lobe
            }
            builder.up();  // back to hemisphere
        }
        builder.up();  // back to brain
    }

    result.brain = builder.build();
}

// ============================================================================
// Phase 2: Create neurons
// ============================================================================
void NetworkConstructor::createNeurons(const NetworkIR& ir, ConstructedNetwork& result) {
    // Create input layer neurons
    {
        const auto& il = ir.inputLayer;
        auto params = il.neuronParams;
        int totalInput = il.rows * il.cols;
        result.inputNeurons.reserve(totalInput);
        for (int i = 0; i < totalInput; ++i) {
            auto neuron = factory_.createNeuron(
                params.windowSizeMs, params.similarityThreshold, params.maxReferencePatterns);
            result.inputNeurons.push_back(neuron);
            result.allNeuronIds.push_back(neuron->getId());
        }
        SNNFW_INFO("  Created {} input neurons ({}x{})", totalInput, il.rows, il.cols);
    }

    // Create output layer neurons
    {
        const auto& ol = ir.outputLayer;
        auto params = ol.neuronParams;
        result.outputPopulations.resize(ol.numClasses);
        for (int c = 0; c < ol.numClasses; ++c) {
            result.outputPopulations[c].reserve(ol.neuronsPerClass);
            for (int n = 0; n < ol.neuronsPerClass; ++n) {
                auto neuron = factory_.createNeuron(
                    params.windowSizeMs, params.similarityThreshold, params.maxReferencePatterns);
                result.outputPopulations[c].push_back(neuron);
                result.allNeuronIds.push_back(neuron->getId());
            }
        }
        SNNFW_INFO("  Created {} output neurons ({} classes x {} per class)",
                   ol.numClasses * ol.neuronsPerClass, ol.numClasses, ol.neuronsPerClass);
    }

    // Create column neurons from hierarchy
    for (const auto& hemiIR : ir.brain.hemispheres) {
        for (const auto& lobeIR : hemiIR.lobes) {
            for (const auto& regionIR : lobeIR.regions) {
                for (const auto& nucleusIR : regionIR.nuclei) {
                    // Combine explicit columns + expanded template
                    std::vector<ColumnIR> allColumns = nucleusIR.columns;
                    if (nucleusIR.columnTemplate.has_value()) {
                        auto expanded = expandColumnTemplate(nucleusIR.columnTemplate.value());
                        allColumns.insert(allColumns.end(), expanded.begin(), expanded.end());
                    }

                    for (const auto& colIR : allColumns) {
                        ConstructedNetwork::ColumnGroup cg;
                        cg.name = colIR.name;
                        cg.path = joinHierarchyPath(
                            {hemiIR.name, lobeIR.name, regionIR.name, nucleusIR.name, colIR.name});

                        // Extract orientation/frequency from properties
                        if (colIR.properties.count("orientation")) {
                            cg.orientation = colIR.properties.at("orientation");
                        }
                        if (colIR.properties.count("spatial_frequency")) {
                            cg.spatialFrequency = colIR.properties.at("spatial_frequency");
                        }

                        // Create neurons for each layer
                        for (const auto& layerIR : colIR.layers) {
                            auto& layerNeurons = cg.layerNeurons[layerIR.name];
                            for (const auto& popIR : layerIR.populations) {
                                auto params = resolveNeuronParams(popIR.neuronParams, ir);
                                std::vector<std::shared_ptr<Neuron>> populationNeurons;
                                populationNeurons.reserve(popIR.count);
                                for (int i = 0; i < popIR.count; ++i) {
                                    auto neuron = factory_.createNeuron(
                                        params.windowSizeMs,
                                        params.similarityThreshold,
                                        params.maxReferencePatterns);
                                    populationNeurons.push_back(neuron);
                                    result.allNeuronIds.push_back(neuron->getId());
                                }
                                layerNeurons.insert(layerNeurons.end(),
                                                    populationNeurons.begin(),
                                                    populationNeurons.end());

                                ConstructedNetwork::PathGroup populationGroup;
                                populationGroup.path = joinHierarchyPath(
                                    {cg.path, layerIR.name, popIR.name});
                                populationGroup.neurons = std::move(populationNeurons);
                                result.populationGroupIndex[populationGroup.path] =
                                    result.populationGroups.size();
                                result.populationGroups.push_back(std::move(populationGroup));
                            }

                            ConstructedNetwork::PathGroup layerGroup;
                            layerGroup.path = joinHierarchyPath({cg.path, layerIR.name});
                            layerGroup.neurons = layerNeurons;
                            result.layerGroupIndex[layerGroup.path] = result.layerGroups.size();
                            result.layerGroups.push_back(std::move(layerGroup));
                        }

                        result.columns.push_back(std::move(cg));
                    }
                }
            }
        }
    }

    SNNFW_INFO("  Created {} cortical columns with {} total neuron IDs",
               result.columns.size(), result.allNeuronIds.size());
}

// ============================================================================
// Phase 3: Create connectivity
// ============================================================================
void NetworkConstructor::createConnectivity(const NetworkIR& ir, ConstructedNetwork& result) {
    ConnectivityBuilder connBuilder(factory_, datastore_, true);

    for (size_t projIdx = 0; projIdx < ir.projections.size(); ++projIdx) {
        const auto& proj = ir.projections[projIdx];
        auto synapseGroup = mapSynapseGroup(proj.synapseGroup);

        // Special handling for tiled receptive field connectivity
        if (proj.pattern == "tiled_receptive_field") {
            auto inputNeurons = resolveSpecialPath(proj.source, result);
            auto columnL4 = resolveNeuronPath(proj.target, result);
            size_t totalSynapses = 0;

            // Convert input neurons to IDs for the pattern
            std::vector<uint64_t> inputIds;
            inputIds.reserve(inputNeurons.size());
            for (auto& n : inputNeurons) inputIds.push_back(n->getId());

            for (size_t colIdx = 0; colIdx < columnL4.size(); ++colIdx) {
                // Create per-column tiled pattern
                TiledReceptiveFieldPattern tiledPattern(
                    proj.inputGridSize, ir.inputLayer.rows, ir.inputLayer.cols,
                    proj.tilesPerSide, proj.tilesPerColumn,
                    proj.targetGridSize, static_cast<int>(colIdx),
                    proj.weight, proj.delay);

                // Store tile/mask info on the ColumnGroup
                if (colIdx < result.columns.size()) {
                    result.columns[colIdx].tileIndices = tiledPattern.getTileIndices();
                    result.columns[colIdx].inputMaskActiveIdx = tiledPattern.getInputMaskActiveIdx();
                }

                connBuilder.clearCreatedObjects();
                auto stats = connBuilder.connect(inputNeurons, columnL4[colIdx], tiledPattern);
                for (auto& syn : connBuilder.getCreatedSynapses()) {
                    result.allSynapses.push_back(syn);
                    if (!proj.synapseGroup.empty()) {
                        result.synapseGroups[proj.synapseGroup].push_back(syn);
                    }
                }
                result.allAxons.insert(result.allAxons.end(),
                    connBuilder.getCreatedAxons().begin(), connBuilder.getCreatedAxons().end());
                result.allDendrites.insert(result.allDendrites.end(),
                    connBuilder.getCreatedDendrites().begin(), connBuilder.getCreatedDendrites().end());
                totalSynapses += stats.synapsesCreated;
            }
            SNNFW_INFO("  Projection '{}': {} synapses (tiled_receptive_field, {} columns)",
                       proj.name, totalSynapses, columnL4.size());
            continue;
        }

        auto pattern = createPattern(proj);
        if (connectivitySeed_ != 0) {
            pattern->setSeed(makeProjectionSeed(connectivitySeed_, proj, projIdx));
        }

        // Try to resolve as special paths first (InputGrid, OutputLayer)
        auto srcSpecial = resolveSpecialPath(proj.source, result);
        auto tgtSpecial = resolveSpecialPath(proj.target, result);

        if (!srcSpecial.empty() && !tgtSpecial.empty()) {
            // Both are special (unlikely but supported)
            auto stats = connBuilder.connect(srcSpecial, tgtSpecial, *pattern);
            auto& created = connBuilder.getCreatedSynapses();
            for (auto& syn : created) {
                result.allSynapses.push_back(syn);
                if (!proj.synapseGroup.empty()) {
                    result.synapseGroups[proj.synapseGroup].push_back(syn);
                }
            }
            result.allAxons.insert(result.allAxons.end(),
                connBuilder.getCreatedAxons().begin(), connBuilder.getCreatedAxons().end());
            result.allDendrites.insert(result.allDendrites.end(),
                connBuilder.getCreatedDendrites().begin(), connBuilder.getCreatedDendrites().end());
            SNNFW_INFO("  Projection '{}': {} synapses", proj.name, stats.synapsesCreated);
        } else if (!srcSpecial.empty() || !tgtSpecial.empty()) {
            // One special, one glob - connect to/from each column
            auto columnSrc = srcSpecial.empty() ? resolveNeuronPath(proj.source, result)
                : std::vector<std::vector<std::shared_ptr<Neuron>>>{srcSpecial};
            auto columnTgt = tgtSpecial.empty() ? resolveNeuronPath(proj.target, result)
                : std::vector<std::vector<std::shared_ptr<Neuron>>>{tgtSpecial};

            size_t totalSynapses = 0;
            // For scope "intra_column": pair up src[i] with tgt[i]
            // For others: connect all to all
            if (proj.scope == "intra_column" && columnSrc.size() == columnTgt.size()) {
                for (size_t i = 0; i < columnSrc.size(); ++i) {
                    connBuilder.clearCreatedObjects();
                    auto stats = connBuilder.connect(columnSrc[i], columnTgt[i], *pattern);
                    for (auto& syn : connBuilder.getCreatedSynapses()) {
                        result.allSynapses.push_back(syn);
                        if (!proj.synapseGroup.empty()) {
                            result.synapseGroups[proj.synapseGroup].push_back(syn);
                        }
                    }
                    result.allAxons.insert(result.allAxons.end(),
                        connBuilder.getCreatedAxons().begin(), connBuilder.getCreatedAxons().end());
                    result.allDendrites.insert(result.allDendrites.end(),
                        connBuilder.getCreatedDendrites().begin(), connBuilder.getCreatedDendrites().end());
                    totalSynapses += stats.synapsesCreated;
                }
            } else {
                // Flatten all and connect
                std::vector<std::shared_ptr<Neuron>> flatSrc, flatTgt;
                for (auto& v : columnSrc) flatSrc.insert(flatSrc.end(), v.begin(), v.end());
                for (auto& v : columnTgt) flatTgt.insert(flatTgt.end(), v.begin(), v.end());
                connBuilder.clearCreatedObjects();
                auto stats = connBuilder.connect(flatSrc, flatTgt, *pattern);
                for (auto& syn : connBuilder.getCreatedSynapses()) {
                    result.allSynapses.push_back(syn);
                    if (!proj.synapseGroup.empty()) {
                        result.synapseGroups[proj.synapseGroup].push_back(syn);
                    }
                }
                result.allAxons.insert(result.allAxons.end(),
                    connBuilder.getCreatedAxons().begin(), connBuilder.getCreatedAxons().end());
                result.allDendrites.insert(result.allDendrites.end(),
                    connBuilder.getCreatedDendrites().begin(), connBuilder.getCreatedDendrites().end());
                totalSynapses = stats.synapsesCreated;
            }
            SNNFW_INFO("  Projection '{}': {} synapses ({})", proj.name, totalSynapses, proj.scope);
        } else {
            // Both are glob paths
            auto columnSrc = resolveNeuronPath(proj.source, result);
            auto columnTgt = resolveNeuronPath(proj.target, result);
            size_t totalSynapses = 0;

            if (proj.scope == "intra_column" && columnSrc.size() == columnTgt.size()) {
                for (size_t i = 0; i < columnSrc.size(); ++i) {
                    connBuilder.clearCreatedObjects();
                    auto stats = connBuilder.connect(columnSrc[i], columnTgt[i], *pattern);
                    for (auto& syn : connBuilder.getCreatedSynapses()) {
                        result.allSynapses.push_back(syn);
                        if (!proj.synapseGroup.empty()) {
                            result.synapseGroups[proj.synapseGroup].push_back(syn);
                        }
                    }
                    result.allAxons.insert(result.allAxons.end(),
                        connBuilder.getCreatedAxons().begin(), connBuilder.getCreatedAxons().end());
                    result.allDendrites.insert(result.allDendrites.end(),
                        connBuilder.getCreatedDendrites().begin(), connBuilder.getCreatedDendrites().end());
                    totalSynapses += stats.synapsesCreated;
                }
            } else {
                std::vector<std::shared_ptr<Neuron>> flatSrc, flatTgt;
                for (auto& v : columnSrc) flatSrc.insert(flatSrc.end(), v.begin(), v.end());
                for (auto& v : columnTgt) flatTgt.insert(flatTgt.end(), v.begin(), v.end());
                connBuilder.clearCreatedObjects();
                auto stats = connBuilder.connect(flatSrc, flatTgt, *pattern);
                for (auto& syn : connBuilder.getCreatedSynapses()) {
                    result.allSynapses.push_back(syn);
                    if (!proj.synapseGroup.empty()) {
                        result.synapseGroups[proj.synapseGroup].push_back(syn);
                    }
                }
                result.allAxons.insert(result.allAxons.end(),
                    connBuilder.getCreatedAxons().begin(), connBuilder.getCreatedAxons().end());
                result.allDendrites.insert(result.allDendrites.end(),
                    connBuilder.getCreatedDendrites().begin(), connBuilder.getCreatedDendrites().end());
                totalSynapses = stats.synapsesCreated;
            }
            SNNFW_INFO("  Projection '{}': {} synapses ({})", proj.name, totalSynapses, proj.scope);
        }
    }

    SNNFW_INFO("  Total: {} synapses, {} axons, {} dendrites",
               result.allSynapses.size(), result.allAxons.size(), result.allDendrites.size());
}

// ============================================================================
// Phase 4: Initialize runtime
// ============================================================================
void NetworkConstructor::initializeRuntime(const NetworkIR& ir, ConstructedNetwork& result) {
    const auto& sim = ir.simulation;

    // Create SpikeProcessor
    result.spikeProcessor = std::make_shared<SpikeProcessor>(
        sim.spikeProcessorTimeSlices, sim.spikeProcessorThreads);
    result.spikeProcessor->setRealTimeSync(sim.realTimeSync);

    // Create NetworkPropagator
    result.propagator = std::make_shared<NetworkPropagator>(result.spikeProcessor);
    result.propagator->setTraceStdpEnabled(sim.traceStdp);
    result.propagator->setStdpLtdScale(sim.stdpLtdScale);
    result.propagator->setStdpLtdWindowMs(sim.stdpLtdWindowMs);
    result.propagator->setStdpEnabled(sim.stdpEnabled);
    result.spikeProcessor->setStdpEnabled(sim.stdpEnabled);

    // Register all dendrites with SpikeProcessor
    for (const auto& dendrite : result.allDendrites) {
        result.spikeProcessor->registerDendrite(dendrite);
    }

    // Register all neurons with NetworkPropagator
    for (const auto& id : result.allNeuronIds) {
        // We need actual Neuron objects - collect from all groups
        // Input neurons
        for (const auto& n : result.inputNeurons) {
            if (n->getId() == id) {
                result.propagator->registerNeuron(n);
                n->setNetworkPropagator(result.propagator);
                goto next_id;
            }
        }
        // Output neurons
        for (const auto& pop : result.outputPopulations) {
            for (const auto& n : pop) {
                if (n->getId() == id) {
                    result.propagator->registerNeuron(n);
                    n->setNetworkPropagator(result.propagator);
                    goto next_id;
                }
            }
        }
        // Column neurons
        for (const auto& col : result.columns) {
            for (const auto& [layerName, neurons] : col.layerNeurons) {
                for (const auto& n : neurons) {
                    if (n->getId() == id) {
                        result.propagator->registerNeuron(n);
                        n->setNetworkPropagator(result.propagator);
                        goto next_id;
                    }
                }
            }
        }
        next_id:;
    }

    // Register axons
    for (const auto& axon : result.allAxons) {
        result.propagator->registerAxon(axon);
    }

    // Register dendrites
    for (const auto& dendrite : result.allDendrites) {
        result.propagator->registerDendrite(dendrite);
        dendrite->setNetworkPropagator(result.propagator);
    }

    // Register synapses and assign synapse groups
    for (const auto& synapse : result.allSynapses) {
        result.propagator->registerSynapse(synapse);
    }

    // Register synapse groups
    for (const auto& [groupName, synapses] : result.synapseGroups) {
        auto group = mapSynapseGroup(groupName);
        for (const auto& syn : synapses) {
            result.propagator->registerSynapseGroup(syn->getId(), group);
        }
    }

    // Keep spike-delivery diagnostics parity with emnist_letters_training.cpp.
    {
        std::unordered_set<uint64_t> l4NeuronIds;
        for (const auto& col : result.columns) {
            auto it = col.layerNeurons.find("L4");
            if (it == col.layerNeurons.end()) continue;
            for (const auto& neuron : it->second) {
                l4NeuronIds.insert(neuron->getId());
            }
        }
        result.spikeProcessor->setInputToL4NeuronIds(l4NeuronIds);
    }

    SNNFW_INFO("  Runtime initialized: SpikeProcessor({} slices, {} threads), "
               "NetworkPropagator registered {} neurons",
               sim.spikeProcessorTimeSlices, sim.spikeProcessorThreads,
               result.allNeuronIds.size());
}

void NetworkConstructor::applyConnectivitySafeguards(
    const NetworkIR& ir, ConstructedNetwork& result) {
    std::unordered_set<uint64_t> knownAxons;
    std::unordered_set<uint64_t> knownDendrites;
    for (const auto& ax : result.allAxons) {
        if (ax) knownAxons.insert(ax->getId());
    }
    for (const auto& d : result.allDendrites) {
        if (d) knownDendrites.insert(d->getId());
    }

    // Gather all neurons once.
    std::vector<std::shared_ptr<Neuron>> allNeurons;
    allNeurons.reserve(result.allNeuronIds.size());
    allNeurons.insert(allNeurons.end(), result.inputNeurons.begin(), result.inputNeurons.end());
    for (const auto& pop : result.outputPopulations) {
        allNeurons.insert(allNeurons.end(), pop.begin(), pop.end());
    }
    for (const auto& col : result.columns) {
        for (const auto& [layerName, neurons] : col.layerNeurons) {
            allNeurons.insert(allNeurons.end(), neurons.begin(), neurons.end());
        }
    }

    // Ensure every neuron has an axon and at least one dendrite.
    for (const auto& neuron : allNeurons) {
        if (!neuron) continue;

        std::shared_ptr<Axon> axon;
        if (neuron->getAxonId() == 0) {
            axon = factory_.createAxon(neuron->getId());
            neuron->setAxonId(axon->getId());
            datastore_.put(axon);
            if (knownAxons.insert(axon->getId()).second) {
                result.allAxons.push_back(axon);
            }
            datastore_.put(neuron);
        } else {
            axon = datastore_.getAxon(neuron->getAxonId());
            if (axon && knownAxons.insert(axon->getId()).second) {
                result.allAxons.push_back(axon);
            }
        }

        if (neuron->getDendriteIds().empty()) {
            auto dendrite = factory_.createDendrite(neuron->getId());
            neuron->addDendrite(dendrite->getId());
            datastore_.put(dendrite);
            if (knownDendrites.insert(dendrite->getId()).second) {
                result.allDendrites.push_back(dendrite);
            }
            datastore_.put(neuron);
        } else {
            for (uint64_t did : neuron->getDendriteIds()) {
                auto dendrite = datastore_.getDendrite(did);
                if (dendrite && knownDendrites.insert(dendrite->getId()).second) {
                    result.allDendrites.push_back(dendrite);
                }
            }
        }
    }

    // Identify L5 neurons and output dendrites for L5->Output fallback wiring.
    std::unordered_set<uint64_t> l5NeuronIds;
    for (const auto& col : result.columns) {
        auto it = col.layerNeurons.find("L5");
        if (it == col.layerNeurons.end()) continue;
        for (const auto& n : it->second) {
            if (n) l5NeuronIds.insert(n->getId());
        }
    }
    std::vector<uint64_t> outputDendrites;
    for (const auto& pop : result.outputPopulations) {
        for (const auto& n : pop) {
            if (n && !n->getDendriteIds().empty()) {
                outputDendrites.push_back(n->getDendriteIds().front());
            }
        }
    }

    bool hasL5ToOutputProjection = false;
    double l5OutputWeight = 0.02;
    double l5OutputDelay = 1.5;
    std::string l5OutputGroup = "L5ToOutput";
    for (const auto& proj : ir.projections) {
        bool targetsOutput = (proj.target == ir.outputLayer.name || proj.target == "OutputLayer");
        bool sourceIsL5 = (proj.source.find("/L5") != std::string::npos);
        if (!targetsOutput || !sourceIsL5) continue;
        hasL5ToOutputProjection = true;
        l5OutputWeight = proj.weight;
        l5OutputDelay = proj.delay;
        if (!proj.synapseGroup.empty()) {
            l5OutputGroup = proj.synapseGroup;
        }
        break;
    }

    std::mt19937 rng(connectivitySeed_ == 0 ? 42u : connectivitySeed_);
    size_t addedL5Output = 0;
    size_t addedAutapse = 0;
    for (const auto& neuron : allNeurons) {
        if (!neuron) continue;
        auto axon = datastore_.getAxon(neuron->getAxonId());
        if (!axon) continue;
        if (axon->getSynapseCount() > 0) continue;

        bool wired = false;
        if (hasL5ToOutputProjection &&
            l5NeuronIds.find(neuron->getId()) != l5NeuronIds.end() &&
            !outputDendrites.empty()) {
            std::uniform_int_distribution<size_t> pick(0, outputDendrites.size() - 1);
            uint64_t targetDendriteId = outputDendrites[pick(rng)];
            auto dendrite = datastore_.getDendrite(targetDendriteId);
            if (dendrite) {
                auto syn = factory_.createSynapse(axon->getId(), targetDendriteId,
                                                  l5OutputWeight, l5OutputDelay);
                axon->addSynapse(syn->getId());
                dendrite->addSynapse(syn->getId());
                datastore_.put(syn);
                datastore_.put(axon);
                datastore_.put(dendrite);
                result.allSynapses.push_back(syn);
                result.synapseGroups[l5OutputGroup].push_back(syn);
                wired = true;
                addedL5Output++;
            }
        }

        if (!wired && !neuron->getDendriteIds().empty()) {
            uint64_t targetDendriteId = neuron->getDendriteIds().front();
            auto dendrite = datastore_.getDendrite(targetDendriteId);
            if (dendrite) {
                auto syn = factory_.createSynapse(axon->getId(), targetDendriteId, 0.1, 1.0);
                axon->addSynapse(syn->getId());
                dendrite->addSynapse(syn->getId());
                datastore_.put(syn);
                datastore_.put(axon);
                datastore_.put(dendrite);
                result.allSynapses.push_back(syn);
                addedAutapse++;
            }
        }
    }

    if (addedL5Output > 0 || addedAutapse > 0) {
        SNNFW_INFO("  Connectivity safeguards added {} L5->Output synapses and {} fallback autapses",
                   addedL5Output, addedAutapse);
    }
}

// ============================================================================
// Helper: Expand column template
// ============================================================================
std::vector<ColumnIR> NetworkConstructor::expandColumnTemplate(const ColumnTemplateIR& tmpl) {
    std::vector<ColumnIR> result;

    for (double orient : tmpl.orientations) {
        for (double freq : tmpl.frequencies) {
            ColumnIR col;

            // Generate name from pattern
            std::string name = tmpl.namingPattern;
            // Replace {orientation} token
            auto pos = name.find("{orientation}");
            if (pos != std::string::npos) {
                // Format as integer if whole number, else decimal
                std::ostringstream oss;
                if (orient == static_cast<int>(orient)) {
                    oss << static_cast<int>(orient);
                } else {
                    oss << orient;
                }
                name.replace(pos, 13, oss.str());
            }
            // Replace {frequency} token
            pos = name.find("{frequency}");
            if (pos != std::string::npos) {
                std::string freqStr = (freq >= 5.0) ? "High" : "Low";
                name.replace(pos, 11, freqStr);
            }

            col.name = name;
            col.properties["orientation"] = orient;
            col.properties["spatial_frequency"] = freq;
            col.layers = tmpl.layers;  // Copy the layer blueprint

            result.push_back(std::move(col));
        }
    }

    return result;
}

// ============================================================================
// Helper: Resolve neuron path
// ============================================================================
std::vector<std::vector<std::shared_ptr<Neuron>>>
NetworkConstructor::resolveNeuronPath(const std::string& path, const ConstructedNetwork& result) {
    std::vector<std::vector<std::shared_ptr<Neuron>>> resolved;
    if (path.empty()) {
        return resolved;
    }

    const bool hasWildcard = path.find('*') != std::string::npos;
    if (!hasWildcard) {
        const auto layerIt = result.layerGroupIndex.find(path);
        if (layerIt != result.layerGroupIndex.end()) {
            resolved.push_back(result.layerGroups[layerIt->second].neurons);
            return resolved;
        }
        const auto popIt = result.populationGroupIndex.find(path);
        if (popIt != result.populationGroupIndex.end()) {
            resolved.push_back(result.populationGroups[popIt->second].neurons);
            return resolved;
        }
    }

    for (const auto& group : result.layerGroups) {
        if (pathMatchesPattern(path, group.path)) {
            resolved.push_back(group.neurons);
        }
    }
    if (!resolved.empty()) {
        return resolved;
    }
    for (const auto& group : result.populationGroups) {
        if (pathMatchesPattern(path, group.path)) {
            resolved.push_back(group.neurons);
        }
    }
    if (!resolved.empty()) {
        return resolved;
    }

    // Legacy fallback for older short-form paths that only imply a layer name.
    const auto parts = splitPath(path);
    std::string layerName;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (parts[i] == "*" && i + 1 < parts.size()) {
            layerName = parts[i + 1];
            break;
        }
    }
    if (layerName.empty() && !parts.empty()) {
        layerName = parts.back();
    }
    if (layerName.empty()) {
        SNNFW_WARN("Cannot resolve path '{}': no matching layer or population", path);
        return resolved;
    }

    for (const auto& col : result.columns) {
        auto it = col.layerNeurons.find(layerName);
        if (it != col.layerNeurons.end()) {
            resolved.push_back(it->second);
        }
    }

    return resolved;
}

// ============================================================================
// Helper: Resolve special paths (InputGrid, OutputLayer)
// ============================================================================
std::vector<std::shared_ptr<Neuron>>
NetworkConstructor::resolveSpecialPath(const std::string& path, const ConstructedNetwork& result) {
    if (path == result.sourceIR.inputLayer.name || path == "InputGrid") {
        return result.inputNeurons;
    }
    if (path == result.sourceIR.outputLayer.name || path == "OutputLayer") {
        std::vector<std::shared_ptr<Neuron>> flat;
        for (const auto& pop : result.outputPopulations) {
            flat.insert(flat.end(), pop.begin(), pop.end());
        }
        return flat;
    }
    return {};  // Not a special path
}

// ============================================================================
// Helper: Create connectivity pattern
// ============================================================================
std::unique_ptr<ConnectivityPattern>
NetworkConstructor::createPattern(const ProjectionIR& proj) {
    if (proj.pattern == "random_sparse") {
        return std::make_unique<RandomSparsePattern>(proj.probability, proj.weight, proj.delay);
    } else if (proj.pattern == "explicit") {
        return std::make_unique<AllToAllPattern>(proj.weight, proj.delay);
    } else if (proj.pattern == "all_to_all") {
        return std::make_unique<AllToAllPattern>(proj.weight, proj.delay);
    } else if (proj.pattern == "one_to_one") {
        return std::make_unique<OneToOnePattern>(proj.weight, proj.delay);
    } else if (proj.pattern == "many_to_one") {
        return std::make_unique<ManyToOnePattern>(proj.weight, proj.delay);
    } else if (proj.pattern == "topographic") {
        return std::make_unique<TopographicPattern>(1.0, proj.weight, proj.delay);
    } else {
        // Default to random sparse
        SNNFW_WARN("Unknown pattern type '{}', defaulting to random_sparse", proj.pattern);
        return std::make_unique<RandomSparsePattern>(proj.probability, proj.weight, proj.delay);
    }
}

// ============================================================================
// Helper: Map synapse group string to enum
// ============================================================================
NetworkPropagator::SynapseGroup
NetworkConstructor::mapSynapseGroup(const std::string& groupName) {
    if (groupName == "InputToL4") return NetworkPropagator::SynapseGroup::InputToL4;
    if (groupName == "L4ToL5") return NetworkPropagator::SynapseGroup::L4ToL5;
    if (groupName == "L5ToOutput") return NetworkPropagator::SynapseGroup::L5ToOutput;
    return NetworkPropagator::SynapseGroup::Unknown;
}

// ============================================================================
// Helper: Resolve neuron params by name
// ============================================================================
NeuronParamsIR NetworkConstructor::resolveNeuronParams(const std::string& name, const NetworkIR& ir) {
    if (!name.empty()) {
        auto it = ir.neuronParamSets.find(name);
        if (it != ir.neuronParamSets.end()) {
            return it->second;
        }
        SNNFW_WARN("Neuron params '{}' not found, using defaults", name);
    }
    return NeuronParamsIR{};  // defaults
}

} // namespace declarative
} // namespace snnfw

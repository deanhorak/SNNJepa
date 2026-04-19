#include "snnfw/declarative/SONATAParser.h"
#include "snnfw/Logger.h"
#include <bbp/sonata/nodes.h>
#include <bbp/sonata/edges.h>
#include <fstream>
#include <filesystem>
#include <algorithm>

namespace snnfw {
namespace declarative {

bool SONATAParser::canParse(const std::string& filename) const {
    // Detect circuit_config.json or *.sonata.json
    namespace fs = std::filesystem;
    auto fn = fs::path(filename).filename().string();
    if (fn == "circuit_config.json") return true;
    const std::string ext = ".sonata.json";
    if (fn.size() >= ext.size()) {
        return fn.compare(fn.size() - ext.size(), ext.size(), ext) == 0;
    }
    return false;
}

NetworkIR SONATAParser::parse(const std::string& primaryFile,
                               const std::string& /*auxiliaryFile*/) {
    std::ifstream file(primaryFile);
    if (!file.is_open()) {
        throw ParseError("Cannot open SONATA config file: " + primaryFile);
    }
    nlohmann::json config;
    try {
        file >> config;
    } catch (const nlohmann::json::parse_error& e) {
        throw ParseError("JSON parse error in " + primaryFile + ": " + e.what());
    }

    auto baseDir = std::filesystem::path(primaryFile).parent_path().string();
    if (baseDir.empty()) baseDir = ".";

    return parseCircuitConfig(config, baseDir);
}

NetworkIR SONATAParser::parseCircuitConfig(const nlohmann::json& config,
                                            const std::string& baseDir) {
    NetworkIR ir;
    ir.sourceFormat = "sonata";

    // Parse manifest for variable resolution
    auto manifest = parseManifest(config);
    if (manifest.find("$BASE_DIR") == manifest.end()) {
        manifest["$BASE_DIR"] = baseDir;
    }

    // Extract network name
    std::string networkName = config.value("network_name", "SONATANetwork");

    // Parse SNNFrame-specific extensions if present
    if (config.contains("snnframe")) {
        const auto& snnfw = config["snnframe"];

        // Neuron parameter sets
        if (snnfw.contains("neuron_params")) {
            for (auto& [key, val] : snnfw["neuron_params"].items()) {
                auto params = parseNeuronParams(val);
                params.name = key;
                ir.neuronParamSets[key] = params;
            }
        }

        // Input / output layers
        if (snnfw.contains("input_layer")) {
            ir.inputLayer = parseInputLayer(snnfw["input_layer"]);
        }
        if (snnfw.contains("output_layer")) {
            ir.outputLayer = parseOutputLayer(snnfw["output_layer"]);
        }

        // Full brain hierarchy (column templates, layers, etc.)
        if (snnfw.contains("brain")) {
            ir.brain = parseBrain(snnfw["brain"]);
        }

        // Projections (connectivity rules)
        if (snnfw.contains("projections")) {
            for (const auto& proj : snnfw["projections"]) {
                ir.projections.push_back(parseProjection(proj));
            }
        }

        // Feature extraction
        if (snnfw.contains("gabor")) {
            const auto& g = snnfw["gabor"];
            ir.gabor.freqLow = g.value("freq_low", 8.0);
            ir.gabor.freqHigh = g.value("freq_high", 3.0);
            ir.gabor.sigma = g.value("sigma", 2.0);
            ir.gabor.gamma = g.value("gamma", 0.5);
            ir.gabor.threshold = g.value("threshold", 0.5);
            ir.gabor.kernelSize = g.value("kernel_size", 7);
        }

        // Saccades
        if (snnfw.contains("saccades")) {
            ir.saccades = parseSaccades(snnfw["saccades"]);
        }

        // Simulation config
        if (snnfw.contains("simulation")) {
            ir.simulation = parseSimulation(snnfw["simulation"]);
        }

        // External adapters
        if (snnfw.contains("adapters") && snnfw["adapters"].is_array()) {
            for (const auto& adapter : snnfw["adapters"]) {
                ir.adapters.push_back(parseAdapter(adapter));
            }
        }

        // Classifier/readout config
        if (snnfw.contains("classification") && snnfw["classification"].is_object()) {
            ir.classification = parseClassification(snnfw["classification"]);
        }
    }

    // Load nodes and edges from HDF5 files if specified
    if (config.contains("networks")) {
        const auto& networks = config["networks"];
        if (networks.contains("nodes")) {
            for (const auto& nodeEntry : networks["nodes"]) {
                std::string nodesFile = resolveManifest(
                    nodeEntry.value("nodes_file", ""), manifest);
                std::string nodeTypesFile = resolveManifest(
                    nodeEntry.value("node_types_file", ""), manifest);
                if (!nodesFile.empty()) {
                    loadNodes(nodesFile, nodeTypesFile, ir);
                }
            }
        }
        if (networks.contains("edges")) {
            for (const auto& edgeEntry : networks["edges"]) {
                std::string edgesFile = resolveManifest(
                    edgeEntry.value("edges_file", ""), manifest);
                std::string edgeTypesFile = resolveManifest(
                    edgeEntry.value("edge_types_file", ""), manifest);
                if (!edgesFile.empty()) {
                    loadEdges(edgesFile, edgeTypesFile, ir);
                }
            }
        }
    }

    // Build default hierarchy if none was created by node loading
    if (ir.brain.hemispheres.empty()) {
        buildDefaultHierarchy(ir, networkName);
    }

    return ir;
}

std::map<std::string, std::string> SONATAParser::parseManifest(
        const nlohmann::json& config) const {
    std::map<std::string, std::string> manifest;
    if (config.contains("manifest")) {
        for (auto& [key, val] : config["manifest"].items()) {
            manifest[key] = val.get<std::string>();
        }
    }
    return manifest;
}

std::string SONATAParser::resolveManifest(
        const std::string& path,
        const std::map<std::string, std::string>& manifest) const {
    std::string resolved = path;
    // Iteratively resolve variables (handles nested references like $NETWORK_DIR -> $BASE_DIR/networks)
    bool changed = true;
    int iterations = 0;
    while (changed && iterations < 10) {
        changed = false;
        for (const auto& [var, val] : manifest) {
            auto pos = resolved.find(var);
            if (pos != std::string::npos) {
                resolved.replace(pos, var.size(), val);
                changed = true;
            }
        }
        ++iterations;
    }
    return resolved;
}

void SONATAParser::loadNodes(const std::string& nodesFile,
                              const std::string& /*nodeTypesFile*/,
                              NetworkIR& ir) const {
    if (!std::filesystem::exists(nodesFile)) {
        SNNFW_WARN("SONATA nodes file not found: {}", nodesFile);
        return;
    }

    try {
        bbp::sonata::NodeStorage storage(nodesFile);
        auto populationNames = storage.populationNames();

        for (const auto& popName : populationNames) {
            auto population = storage.openPopulation(popName);
            auto selection = population->selectAll();
            size_t nodeCount = selection.flatSize();

            SNNFW_INFO("SONATA: Loading node population '{}' with {} nodes",
                       popName, nodeCount);

            // Read SNNFrame-specific attributes if available
            auto attrNames = population->attributeNames();

            NeuronParamsIR params;
            params.name = popName + "_params";

            // Try to read SNNFrame neuron parameters from node attributes
            bool hasWindowSize = attrNames.count("window_size_ms") > 0;
            bool hasThreshold = attrNames.count("similarity_threshold") > 0;
            bool hasMaxPatterns = attrNames.count("max_patterns") > 0;

            if (hasWindowSize) {
                auto values = population->getAttribute<double>("window_size_ms", selection);
                if (!values.empty()) params.windowSizeMs = values[0];
            }
            if (hasThreshold) {
                auto values = population->getAttribute<double>("similarity_threshold", selection);
                if (!values.empty()) params.similarityThreshold = values[0];
            }
            if (hasMaxPatterns) {
                auto values = population->getAttribute<uint64_t>("max_patterns", selection);
                if (!values.empty()) params.maxReferencePatterns = static_cast<int>(values[0]);
            }

            ir.neuronParamSets[params.name] = params;

            // Create a population IR entry (will be placed in hierarchy by buildDefaultHierarchy)
            PopulationIR popIR;
            popIR.name = popName;
            popIR.count = static_cast<int>(nodeCount);
            popIR.neuronParams = params.name;

            // Store in a temporary layer (buildDefaultHierarchy will organize)
            LayerIR layer;
            layer.name = popName;
            layer.populations.push_back(popIR);

            // Add as a column in the default hierarchy
            ColumnIR col;
            col.name = popName + "_col";
            col.layers.push_back(layer);

            // Check for orientation/frequency attributes for column properties
            if (attrNames.count("orientation") > 0) {
                auto orientations = population->getAttribute<double>("orientation", selection);
                if (!orientations.empty()) {
                    col.properties["orientation"] = orientations[0];
                }
            }

            // Store columns temporarily - buildDefaultHierarchy will place them
            if (ir.brain.hemispheres.empty()) {
                buildDefaultHierarchy(ir, "SONATANetwork");
            }
            auto& nucleus = ir.brain.hemispheres[0].lobes[0].regions[0].nuclei[0];
            nucleus.columns.push_back(col);
        }
    } catch (const std::exception& e) {
        SNNFW_ERROR("Failed to load SONATA nodes from '{}': {}", nodesFile, e.what());
        throw ParseError("Failed to load SONATA nodes: " + std::string(e.what()));
    }
}

void SONATAParser::loadEdges(const std::string& edgesFile,
                              const std::string& /*edgeTypesFile*/,
                              NetworkIR& ir) const {
    if (!std::filesystem::exists(edgesFile)) {
        SNNFW_WARN("SONATA edges file not found: {}", edgesFile);
        return;
    }

    try {
        bbp::sonata::EdgeStorage storage(edgesFile);
        auto populationNames = storage.populationNames();

        for (const auto& popName : populationNames) {
            auto population = storage.openPopulation(popName);
            auto selection = population->selectAll();
            size_t edgeCount = selection.flatSize();

            SNNFW_INFO("SONATA: Loading edge population '{}' with {} edges",
                       popName, edgeCount);

            // Read edge attributes
            auto attrNames = population->attributeNames();

            ProjectionIR proj;
            proj.name = popName;
            proj.pattern = "explicit";  // SONATA has explicit connectivity
            proj.scope = "global";

            // Source and target come from the population metadata
            proj.source = population->source();
            proj.target = population->target();

            // Read weight and delay if available
            if (attrNames.count("weight") > 0) {
                auto weights = population->getAttribute<double>("weight", selection);
                if (!weights.empty()) {
                    // Use average weight as the projection weight
                    double sum = 0.0;
                    for (double w : weights) sum += w;
                    proj.weight = sum / weights.size();
                }
            }
            if (attrNames.count("delay") > 0) {
                auto delays = population->getAttribute<double>("delay", selection);
                if (!delays.empty()) {
                    double sum = 0.0;
                    for (double d : delays) sum += d;
                    proj.delay = sum / delays.size();
                }
            }

            // Estimate connectivity probability
            // (edges / (source_count * target_count))
            proj.probability = 1.0;  // Explicit connectivity

            proj.synapseGroup = popName;
            ir.projections.push_back(proj);
        }
    } catch (const std::exception& e) {
        SNNFW_ERROR("Failed to load SONATA edges from '{}': {}", edgesFile, e.what());
        throw ParseError("Failed to load SONATA edges: " + std::string(e.what()));
    }
}

void SONATAParser::buildDefaultHierarchy(NetworkIR& ir,
                                          const std::string& networkName) const {
    // Create a minimal hierarchy: Brain → Hemisphere → Lobe → Region → Nucleus
    BrainIR brain;
    brain.name = networkName;

    HemisphereIR hemi;
    hemi.name = "Default";

    LobeIR lobe;
    lobe.name = "Primary";

    RegionIR region;
    region.name = "Main";

    NucleusIR nucleus;
    nucleus.name = "Core";

    region.nuclei.push_back(nucleus);
    lobe.regions.push_back(region);
    hemi.lobes.push_back(lobe);
    brain.hemispheres.push_back(hemi);

    ir.brain = brain;
}

// ---------------------------------------------------------------------------
// Extended snnframe section parsers
// ---------------------------------------------------------------------------

NeuronParamsIR SONATAParser::parseNeuronParams(const nlohmann::json& j) const {
    NeuronParamsIR p;
    p.windowSizeMs = j.value("window_size_ms", 500.0);
    p.similarityThreshold = j.value("similarity_threshold", 0.93);
    p.maxReferencePatterns = j.value("max_reference_patterns", 500);
    p.similarityMetric = j.value("similarity_metric", "cosine");
    return p;
}

PopulationIR SONATAParser::parsePopulation(const nlohmann::json& j) const {
    PopulationIR p;
    p.name = j.value("name", "");
    p.count = j.value("count", 0);
    p.neuronParams = j.value("neuron_params", "");
    p.gridLayout = j.value("grid_layout", "");
    return p;
}

LayerIR SONATAParser::parseLayer(const nlohmann::json& j) const {
    LayerIR l;
    l.name = j.value("name", "");
    if (j.contains("populations")) {
        for (const auto& pop : j["populations"]) {
            l.populations.push_back(parsePopulation(pop));
        }
    }
    return l;
}

ColumnIR SONATAParser::parseColumn(const nlohmann::json& j) const {
    ColumnIR c;
    c.name = j.value("name", "");
    if (j.contains("properties")) {
        for (auto& [key, val] : j["properties"].items()) {
            c.properties[key] = val.get<double>();
        }
    }
    if (j.contains("layers")) {
        for (const auto& layer : j["layers"]) {
            c.layers.push_back(parseLayer(layer));
        }
    }
    return c;
}

ColumnTemplateIR SONATAParser::parseColumnTemplate(const nlohmann::json& j) const {
    ColumnTemplateIR t;
    t.templateName = j.value("template_name", "");
    t.namingPattern = j.value("naming_pattern", "Orient_{orientation}_Freq_{frequency}");
    if (j.contains("orientations")) {
        t.orientations = j["orientations"].get<std::vector<double>>();
    }
    if (j.contains("frequencies")) {
        t.frequencies = j["frequencies"].get<std::vector<double>>();
    }
    if (j.contains("layers")) {
        for (const auto& layer : j["layers"]) {
            t.layers.push_back(parseLayer(layer));
        }
    }
    return t;
}

NucleusIR SONATAParser::parseNucleus(const nlohmann::json& j) const {
    NucleusIR n;
    n.name = j.value("name", "");
    if (j.contains("column_template")) {
        n.columnTemplate = parseColumnTemplate(j["column_template"]);
    }
    if (j.contains("columns")) {
        for (const auto& col : j["columns"]) {
            n.columns.push_back(parseColumn(col));
        }
    }
    return n;
}

RegionIR SONATAParser::parseRegion(const nlohmann::json& j) const {
    RegionIR r;
    r.name = j.value("name", "");
    if (j.contains("nuclei")) {
        for (const auto& nuc : j["nuclei"]) {
            r.nuclei.push_back(parseNucleus(nuc));
        }
    }
    return r;
}

LobeIR SONATAParser::parseLobe(const nlohmann::json& j) const {
    LobeIR l;
    l.name = j.value("name", "");
    if (j.contains("regions")) {
        for (const auto& reg : j["regions"]) {
            l.regions.push_back(parseRegion(reg));
        }
    }
    return l;
}

HemisphereIR SONATAParser::parseHemisphere(const nlohmann::json& j) const {
    HemisphereIR h;
    h.name = j.value("name", "");
    if (j.contains("lobes")) {
        for (const auto& lobe : j["lobes"]) {
            h.lobes.push_back(parseLobe(lobe));
        }
    }
    return h;
}

BrainIR SONATAParser::parseBrain(const nlohmann::json& j) const {
    BrainIR b;
    b.name = j.value("name", "");
    if (j.contains("hemispheres")) {
        for (const auto& hemi : j["hemispheres"]) {
            b.hemispheres.push_back(parseHemisphere(hemi));
        }
    }
    return b;
}

ProjectionIR SONATAParser::parseProjection(const nlohmann::json& j) const {
    ProjectionIR p;
    p.name = j.value("name", "");
    p.source = j.value("source", "");
    p.target = j.value("target", "");
    p.pattern = j.value("pattern", "random_sparse");
    p.probability = j.value("probability", 1.0);
    p.weight = j.value("weight", 1.0);
    p.maxWeight = j.value("max_weight", 10.0);
    p.delay = j.value("delay", 1.5);
    p.scope = j.value("scope", "intra_column");
    p.synapseGroup = j.value("synapse_group", "");

    // Tiled receptive field parameters
    p.tilesPerSide = j.value("tiles_per_side", 4);
    p.tilesPerColumn = j.value("tiles_per_column", 3);
    p.inputGridSize = j.value("input_grid_size", 28);
    p.targetGridSize = j.value("target_grid_size", 7);
    return p;
}

InputLayerIR SONATAParser::parseInputLayer(const nlohmann::json& j) const {
    InputLayerIR il;
    il.name = j.value("name", "InputGrid");
    il.rows = j.value("rows", 28);
    il.cols = j.value("cols", 28);
    il.latencyMs = j.value("latency_ms", 15.0);
    il.pixelThreshold = j.value("pixel_threshold", 0.4);
    if (j.contains("neuron_params") && j["neuron_params"].is_object()) {
        il.neuronParams = parseNeuronParams(j["neuron_params"]);
    }
    return il;
}

OutputLayerIR SONATAParser::parseOutputLayer(const nlohmann::json& j) const {
    OutputLayerIR ol;
    ol.name = j.value("name", "OutputLayer");
    ol.numClasses = j.value("num_classes", 26);
    ol.neuronsPerClass = j.value("neurons_per_class", 3);
    if (j.contains("neuron_params") && j["neuron_params"].is_object()) {
        ol.neuronParams = parseNeuronParams(j["neuron_params"]);
    }
    return ol;
}

SimulationConfigIR SONATAParser::parseSimulation(const nlohmann::json& j) const {
    SimulationConfigIR sim;
    sim.connectivitySeed = j.value("connectivity_seed", 0u);
    if (j.contains("spike_processor")) {
        const auto& sp = j["spike_processor"];
        sim.spikeProcessorTimeSlices = sp.value("time_slices", 10000);
        sim.spikeProcessorThreads = sp.value("threads", 24);
        sim.realTimeSync = sp.value("real_time_sync", false);
    }
    if (j.contains("stdp")) {
        const auto& stdp = j["stdp"];
        sim.stdpEnabled = stdp.value("enabled", true);
        sim.stdpLtdScale = stdp.value("ltd_scale", 0.3);
        sim.stdpLtdWindowMs = stdp.value("ltd_window_ms", 70.0);
        sim.traceStdp = stdp.value("trace_stdp", true);
        sim.freezeStdpDuringTesting = stdp.value("freeze_during_testing", true);
        sim.enableStdpEligibilityGate = stdp.value("eligibility_gate", true);
        sim.stdpEligibilityMinUpdates = stdp.value("eligibility_min_updates", 1);
        sim.stdpEligibilityMinLtp = stdp.value("eligibility_min_ltp", 0);
        sim.stdpEligibilityThreshold = stdp.value("eligibility_threshold", -0.002);
        sim.stdpEligibilityLtdPenalty = stdp.value("eligibility_ltd_penalty", 0.5);
    }
    if (j.contains("competition")) {
        const auto& comp = j["competition"];
        sim.l4Keep = comp.value("l4_keep", 8);
        sim.l5Keep = comp.value("l5_keep", 8);
        sim.enableSimilarityCompetition = comp.value("enable_similarity_competition", true);
        sim.l4SimilarityWeight = comp.value("l4_similarity_weight", 0.25);
        sim.l5SimilarityWeight = comp.value("l5_similarity_weight", 0.60);
        sim.traceSimilarityCompetition =
            comp.value("trace_similarity_competition", false);
        sim.enableL5Inhibition = comp.value("enable_l5_inhibition", true);
        sim.enableL5InterColumnInhibition =
            comp.value("enable_l5_inter_column_inhibition", false);
        sim.l5InterColumnInhibit = comp.value("l5_inter_column_inhibit", 0.1);
        sim.l5InterColumnMinOverlap = comp.value("l5_inter_column_min_overlap", 0.2);
        sim.l5InterColumnWinnerScale = comp.value("l5_inter_column_winner_scale", 0.25);
        sim.l5InterColumnMaxInhibit = comp.value("l5_inter_column_max_inhibit", 1.0);
        sim.l5InterColumnMaxOrientationDeltaDeg =
            comp.value("l5_inter_column_max_orientation_delta_deg", 45.0);
        sim.l5InterColumnMaxFrequencyOctaveDelta =
            comp.value("l5_inter_column_max_frequency_octave_delta", 0.75);
        sim.l5InterColumnMaxNeighbors =
            comp.value("l5_inter_column_max_neighbors", 8);
        sim.maskMinActive = comp.value("mask_min_active", 6);
    }
    sim.interImageGapMs = j.value("inter_image_gap_ms", 550.0);
    return sim;
}

SaccadeConfigIR SONATAParser::parseSaccades(const nlohmann::json& j) const {
    SaccadeConfigIR s;
    s.enabled = j.value("enabled", false);
    s.numFixations = j.value("num_fixations", 4);
    if (j.contains("regions")) {
        for (const auto& reg : j["regions"]) {
            FixationRegionIR fr;
            fr.name = reg.value("name", "");
            fr.rowStart = reg.value("row_start", 0);
            fr.rowEnd = reg.value("row_end", 0);
            fr.colStart = reg.value("col_start", 0);
            fr.colEnd = reg.value("col_end", 0);
            s.regions.push_back(fr);
        }
    }
    return s;
}

AdapterConfigIR SONATAParser::parseAdapter(const nlohmann::json& j) const {
    AdapterConfigIR a;
    a.name = j.value("name", "");
    a.type = j.value("type", "");
    a.role = j.value("role", "");
    a.bindTo = j.value("bind_to", "");
    a.temporalWindowMs = j.value("temporal_window_ms", 10.0);

    if (j.contains("double_params") && j["double_params"].is_object()) {
        for (auto& [key, val] : j["double_params"].items()) {
            a.doubleParams[key] = val.get<double>();
        }
    }
    if (j.contains("int_params") && j["int_params"].is_object()) {
        for (auto& [key, val] : j["int_params"].items()) {
            a.intParams[key] = val.get<int>();
        }
    }
    if (j.contains("string_params") && j["string_params"].is_object()) {
        for (auto& [key, val] : j["string_params"].items()) {
            a.stringParams[key] = val.get<std::string>();
        }
    }
    return a;
}

ClassificationConfigIR SONATAParser::parseClassification(const nlohmann::json& j) const {
    ClassificationConfigIR c;
    c.type = j.value("type", "");
    c.k = j.value("k", 5);
    c.distanceExponent = j.value("distance_exponent", 1.0);

    if (j.contains("double_params") && j["double_params"].is_object()) {
        for (auto& [key, val] : j["double_params"].items()) {
            c.doubleParams[key] = val.get<double>();
        }
    }
    if (j.contains("int_params") && j["int_params"].is_object()) {
        for (auto& [key, val] : j["int_params"].items()) {
            c.intParams[key] = val.get<int>();
        }
    }
    if (j.contains("string_params") && j["string_params"].is_object()) {
        for (auto& [key, val] : j["string_params"].items()) {
            c.stringParams[key] = val.get<std::string>();
        }
    }

    if (j.contains("group_definitions")) {
        c.stringParams["group_definitions"] = j["group_definitions"].get<std::string>();
    }
    if (j.contains("coarse_strategy")) {
        c.stringParams["coarse_strategy"] = j["coarse_strategy"].get<std::string>();
    }
    if (j.contains("fine_strategy")) {
        c.stringParams["fine_strategy"] = j["fine_strategy"].get<std::string>();
    }
    if (j.contains("coarse_k")) {
        c.intParams["coarse_k"] = j["coarse_k"].get<int>();
    }
    if (j.contains("fine_k")) {
        c.intParams["fine_k"] = j["fine_k"].get<int>();
    }
    if (j.contains("fallback_to_flat")) {
        if (j["fallback_to_flat"].is_boolean()) {
            c.intParams["fallback_to_flat"] = j["fallback_to_flat"].get<bool>() ? 1 : 0;
        } else {
            c.intParams["fallback_to_flat"] = j["fallback_to_flat"].get<int>();
        }
    }

    return c;
}

} // namespace declarative
} // namespace snnfw

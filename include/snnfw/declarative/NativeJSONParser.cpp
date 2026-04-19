#include "snnfw/declarative/NativeJSONParser.h"
#include <fstream>
#include <sstream>
#include <algorithm>

namespace snnfw {
namespace declarative {

bool NativeJSONParser::canParse(const std::string& filename) const {
    // Check for .snnf.json extension
    const std::string ext = ".snnf.json";
    if (filename.size() >= ext.size()) {
        return filename.compare(filename.size() - ext.size(), ext.size(), ext) == 0;
    }
    return false;
}

NetworkIR NativeJSONParser::parse(const std::string& primaryFile,
                                   const std::string& /*auxiliaryFile*/) {
    std::ifstream file(primaryFile);
    if (!file.is_open()) {
        throw ParseError("Cannot open file: " + primaryFile);
    }
    nlohmann::json root;
    try {
        file >> root;
    } catch (const nlohmann::json::parse_error& e) {
        throw ParseError("JSON parse error in " + primaryFile + ": " + e.what());
    }
    return parseJson(root);
}

NetworkIR NativeJSONParser::parseJson(const nlohmann::json& root) {
    NetworkIR ir;
    ir.sourceFormat = "snnframe_json";

    if (root.contains("snnframe_version")) {
        ir.formatVersion = root["snnframe_version"].get<std::string>();
    }

    // Parse named neuron parameter sets
    if (root.contains("neuron_params")) {
        for (auto& [key, val] : root["neuron_params"].items()) {
            auto params = parseNeuronParams(val);
            params.name = key;
            ir.neuronParamSets[key] = params;
        }
    }

    // Parse special layers
    if (root.contains("input_layer")) {
        ir.inputLayer = parseInputLayer(root["input_layer"]);
    }
    if (root.contains("output_layer")) {
        ir.outputLayer = parseOutputLayer(root["output_layer"]);
    }

    // Parse brain hierarchy
    if (root.contains("brain")) {
        ir.brain = parseBrain(root["brain"]);
    }

    // Parse projections
    if (root.contains("projections")) {
        for (const auto& proj : root["projections"]) {
            ir.projections.push_back(parseProjection(proj));
        }
    }

    // Parse feature extraction
    if (root.contains("gabor")) {
        ir.gabor = parseGabor(root["gabor"]);
    }
    if (root.contains("saccades")) {
        ir.saccades = parseSaccades(root["saccades"]);
    }

    // Parse simulation config
    if (root.contains("simulation")) {
        ir.simulation = parseSimulation(root["simulation"]);
    }

    // Parse external adapters
    if (root.contains("adapters") && root["adapters"].is_array()) {
        for (const auto& adapter : root["adapters"]) {
            ir.adapters.push_back(parseAdapter(adapter));
        }
    }

    // Parse classifier/readout config
    if (root.contains("classification") && root["classification"].is_object()) {
        ir.classification = parseClassification(root["classification"]);
    }

    return ir;
}

NeuronParamsIR NativeJSONParser::parseNeuronParams(const nlohmann::json& j) {
    NeuronParamsIR p;
    if (j.contains("window_size_ms")) p.windowSizeMs = j["window_size_ms"].get<double>();
    if (j.contains("similarity_threshold")) p.similarityThreshold = j["similarity_threshold"].get<double>();
    if (j.contains("max_reference_patterns")) p.maxReferencePatterns = j["max_reference_patterns"].get<int>();
    if (j.contains("similarity_metric")) p.similarityMetric = j["similarity_metric"].get<std::string>();
    return p;
}

PopulationIR NativeJSONParser::parsePopulation(const nlohmann::json& j) {
    PopulationIR p;
    p.name = j.value("name", "");
    p.count = j.value("count", 0);
    p.neuronParams = j.value("neuron_params", "");
    p.gridLayout = j.value("grid_layout", "");
    return p;
}

LayerIR NativeJSONParser::parseLayer(const nlohmann::json& j) {
    LayerIR l;
    l.name = j.value("name", "");
    if (j.contains("populations")) {
        for (const auto& pop : j["populations"]) {
            l.populations.push_back(parsePopulation(pop));
        }
    }
    return l;
}

ColumnIR NativeJSONParser::parseColumn(const nlohmann::json& j) {
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

ColumnTemplateIR NativeJSONParser::parseColumnTemplate(const nlohmann::json& j) {
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

NucleusIR NativeJSONParser::parseNucleus(const nlohmann::json& j) {
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

RegionIR NativeJSONParser::parseRegion(const nlohmann::json& j) {
    RegionIR r;
    r.name = j.value("name", "");
    if (j.contains("nuclei")) {
        for (const auto& nuc : j["nuclei"]) {
            r.nuclei.push_back(parseNucleus(nuc));
        }
    }
    return r;
}

LobeIR NativeJSONParser::parseLobe(const nlohmann::json& j) {
    LobeIR l;
    l.name = j.value("name", "");
    if (j.contains("regions")) {
        for (const auto& reg : j["regions"]) {
            l.regions.push_back(parseRegion(reg));
        }
    }
    return l;
}

HemisphereIR NativeJSONParser::parseHemisphere(const nlohmann::json& j) {
    HemisphereIR h;
    h.name = j.value("name", "");
    if (j.contains("lobes")) {
        for (const auto& lobe : j["lobes"]) {
            h.lobes.push_back(parseLobe(lobe));
        }
    }
    return h;
}

BrainIR NativeJSONParser::parseBrain(const nlohmann::json& j) {
    BrainIR b;
    b.name = j.value("name", "");
    if (j.contains("hemispheres")) {
        for (const auto& hemi : j["hemispheres"]) {
            b.hemispheres.push_back(parseHemisphere(hemi));
        }
    }
    return b;
}

ProjectionIR NativeJSONParser::parseProjection(const nlohmann::json& j) {
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
    return p;
}

InputLayerIR NativeJSONParser::parseInputLayer(const nlohmann::json& j) {
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

OutputLayerIR NativeJSONParser::parseOutputLayer(const nlohmann::json& j) {
    OutputLayerIR ol;
    ol.name = j.value("name", "OutputLayer");
    ol.numClasses = j.value("num_classes", 26);
    ol.neuronsPerClass = j.value("neurons_per_class", 3);
    if (j.contains("neuron_params") && j["neuron_params"].is_object()) {
        ol.neuronParams = parseNeuronParams(j["neuron_params"]);
    }
    return ol;
}

GaborConfigIR NativeJSONParser::parseGabor(const nlohmann::json& j) {
    GaborConfigIR g;
    g.freqLow = j.value("freq_low", 8.0);
    g.freqHigh = j.value("freq_high", 3.0);
    g.sigma = j.value("sigma", 2.0);
    g.gamma = j.value("gamma", 0.5);
    g.threshold = j.value("threshold", 0.5);
    g.kernelSize = j.value("kernel_size", 7);
    return g;
}

SaccadeConfigIR NativeJSONParser::parseSaccades(const nlohmann::json& j) {
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

SimulationConfigIR NativeJSONParser::parseSimulation(const nlohmann::json& j) {
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

AdapterConfigIR NativeJSONParser::parseAdapter(const nlohmann::json& j) {
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

ClassificationConfigIR NativeJSONParser::parseClassification(const nlohmann::json& j) {
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

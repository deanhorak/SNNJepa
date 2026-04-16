#include "snnfw/declarative/NeuroMLParser.h"
#include "snnfw/Logger.h"
#include <boost/property_tree/detail/rapidxml.hpp>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>

namespace snnfw {
namespace declarative {

using namespace boost::property_tree::detail::rapidxml;

bool NeuroMLParser::canParse(const std::string& filename) const {
    const std::string ext1 = ".nml";
    const std::string ext2 = ".neuroml";
    if (filename.size() >= ext1.size() &&
        filename.compare(filename.size() - ext1.size(), ext1.size(), ext1) == 0) {
        return true;
    }
    if (filename.size() >= ext2.size() &&
        filename.compare(filename.size() - ext2.size(), ext2.size(), ext2) == 0) {
        return true;
    }
    return false;
}

NetworkIR NeuroMLParser::parse(const std::string& primaryFile,
                                const std::string& /*auxiliaryFile*/) {
    std::ifstream file(primaryFile);
    if (!file.is_open()) {
        throw ParseError("Cannot open NeuroML file: " + primaryFile);
    }
    std::string xmlContent((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    file.close();

    if (xmlContent.empty()) {
        throw ParseError("NeuroML file is empty: " + primaryFile);
    }

    return parseXML(xmlContent);
}

NetworkIR NeuroMLParser::parseXML(const std::string& xmlContent) {
    // RapidXML requires non-const char*
    std::vector<char> xmlData(xmlContent.begin(), xmlContent.end());
    xmlData.push_back('\0');

    xml_document<> doc;
    try {
        doc.parse<0>(&xmlData[0]);
    } catch (const parse_error& e) {
        throw ParseError(std::string("NeuroML XML parse error: ") + e.what());
    }

    // Find <neuroml> root
    xml_node<>* root = doc.first_node("neuroml");
    if (!root) {
        throw ParseError("No <neuroml> root element found");
    }

    // Parse cell types
    std::map<std::string, CellType> cellTypes;
    parseCellTypes(root, cellTypes);

    // Parse synapse types
    std::map<std::string, SynapseType> synapseTypes;
    parseSynapseTypes(root, synapseTypes);

    // Parse network
    xml_node<>* network = root->first_node("network");
    std::vector<ParsedPopulation> populations;
    std::vector<ParsedProjection> projections;

    if (network) {
        parseNetwork(network, populations, projections);
    }

    // Determine network name
    std::string networkName = "NeuroMLNetwork";
    if (network) {
        xml_attribute<>* idAttr = network->first_attribute("id");
        if (idAttr) networkName = idAttr->value();
    }

    return buildIR(cellTypes, synapseTypes, populations, projections, networkName);
}

// Helper to safely get attribute value
static const char* getAttr(xml_node<>* node, const char* name) {
    xml_attribute<>* attr = node->first_attribute(name);
    return attr ? attr->value() : nullptr;
}

template<typename XmlNode>
void NeuroMLParser::parseCellTypes(XmlNode* root,
                                    std::map<std::string, CellType>& cellTypes) const {
    for (auto* cell = root->first_node("cell"); cell; cell = cell->next_sibling("cell")) {
        CellType ct;
        const char* id = getAttr(cell, "id");
        if (!id) continue;
        ct.id = id;

        // Look for SNNFrame-specific properties in <property> elements
        for (auto* prop = cell->first_node("property"); prop;
             prop = prop->next_sibling("property")) {
            const char* tag = getAttr(prop, "tag");
            const char* val = getAttr(prop, "value");
            if (!tag || !val) continue;

            std::string tagStr(tag);
            if (tagStr == "snnfw:window_size_ms") {
                ct.windowSizeMs = std::stod(val);
            } else if (tagStr == "snnfw:similarity_threshold") {
                ct.similarityThreshold = std::stod(val);
            } else if (tagStr == "snnfw:max_reference_patterns") {
                ct.maxReferencePatterns = std::stoi(val);
            } else if (tagStr == "snnfw:similarity_metric") {
                ct.similarityMetric = val;
            }
        }

        cellTypes[ct.id] = ct;
        SNNFW_DEBUG("NeuroML: Parsed cell type '{}'", ct.id);
    }
}

template<typename XmlNode>
void NeuroMLParser::parseSynapseTypes(XmlNode* root,
                                       std::map<std::string, SynapseType>& synapseTypes) const {
    // Parse various synapse type elements
    const char* synapseElements[] = {
        "expOneSynapse", "expTwoSynapse", "alphaSynapse",
        "blockingPlasticSynapse", "gapJunction", nullptr
    };

    for (int i = 0; synapseElements[i]; ++i) {
        for (auto* syn = root->first_node(synapseElements[i]); syn;
             syn = syn->next_sibling(synapseElements[i])) {
            SynapseType st;
            const char* id = getAttr(syn, "id");
            if (!id) continue;
            st.id = id;

            // Look for weight/delay in properties
            for (auto* prop = syn->first_node("property"); prop;
                 prop = prop->next_sibling("property")) {
                const char* tag = getAttr(prop, "tag");
                const char* val = getAttr(prop, "value");
                if (!tag || !val) continue;
                if (std::strcmp(tag, "snnfw:weight") == 0) st.weight = std::stod(val);
                if (std::strcmp(tag, "snnfw:delay") == 0) st.delay = std::stod(val);
            }

            synapseTypes[st.id] = st;
        }
    }
}

template<typename XmlNode>
void NeuroMLParser::parseNetwork(XmlNode* network,
                                  std::vector<ParsedPopulation>& populations,
                                  std::vector<ParsedProjection>& projections) const {
    // Parse populations
    for (auto* pop = network->first_node("population"); pop;
         pop = pop->next_sibling("population")) {
        ParsedPopulation pp;
        const char* id = getAttr(pop, "id");
        if (id) pp.id = id;

        const char* component = getAttr(pop, "component");
        if (component) pp.component = component;

        const char* sizeAttr = getAttr(pop, "size");
        if (sizeAttr) {
            pp.size = std::stoi(sizeAttr);
        } else {
            // Count <instance> elements
            int count = 0;
            for (auto* inst = pop->first_node("instance"); inst;
                 inst = inst->next_sibling("instance")) {
                ++count;
            }
            pp.size = count;
        }

        // Parse property elements on populations
        for (auto* prop = pop->first_node("property"); prop;
             prop = prop->next_sibling("property")) {
            const char* tag = getAttr(prop, "tag");
            const char* val = getAttr(prop, "value");
            if (tag && val) pp.properties[tag] = val;
        }

        populations.push_back(pp);
        SNNFW_DEBUG("NeuroML: Parsed population '{}' size={}", pp.id, pp.size);
    }

    // Parse projections
    for (auto* proj = network->first_node("projection"); proj;
         proj = proj->next_sibling("projection")) {
        ParsedProjection pp;
        const char* id = getAttr(proj, "id");
        if (id) pp.id = id;

        const char* pre = getAttr(proj, "presynapticPopulation");
        if (pre) pp.prePopulation = pre;

        const char* post = getAttr(proj, "postsynapticPopulation");
        if (post) pp.postPopulation = post;

        const char* syn = getAttr(proj, "synapse");
        if (syn) pp.synapse = syn;

        // Count connections
        int connCount = 0;
        for (auto* conn = proj->first_node("connection"); conn;
             conn = conn->next_sibling("connection")) {
            ++connCount;
        }
        // Also count connectionWD (connection with delay)
        for (auto* conn = proj->first_node("connectionWD"); conn;
             conn = conn->next_sibling("connectionWD")) {
            ++connCount;
        }
        pp.connectionCount = connCount;

        projections.push_back(pp);
        SNNFW_DEBUG("NeuroML: Parsed projection '{}' ({} -> {}, {} connections)",
                    pp.id, pp.prePopulation, pp.postPopulation, pp.connectionCount);
    }
}

NetworkIR NeuroMLParser::buildIR(
        const std::map<std::string, CellType>& cellTypes,
        const std::map<std::string, SynapseType>& synapseTypes,
        const std::vector<ParsedPopulation>& populations,
        const std::vector<ParsedProjection>& projections,
        const std::string& networkName) const {

    NetworkIR ir;
    ir.sourceFormat = "neuroml";

    // Create neuron parameter sets from cell types
    for (const auto& [id, ct] : cellTypes) {
        NeuronParamsIR params;
        params.name = id;
        params.windowSizeMs = ct.windowSizeMs;
        params.similarityThreshold = ct.similarityThreshold;
        params.maxReferencePatterns = ct.maxReferencePatterns;
        params.similarityMetric = ct.similarityMetric;
        ir.neuronParamSets[id] = params;
    }

    // Ensure there's at least a default param set
    if (ir.neuronParamSets.empty()) {
        NeuronParamsIR defaultParams;
        defaultParams.name = "default";
        ir.neuronParamSets["default"] = defaultParams;
    }

    // Build hierarchy: Brain → Hemisphere → Lobe → Region → Nucleus
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

    // Create columns from populations
    for (const auto& pop : populations) {
        ColumnIR col;
        col.name = pop.id + "_col";

        LayerIR layer;
        layer.name = pop.id;

        PopulationIR popIR;
        popIR.name = pop.id;
        popIR.count = pop.size;

        // Map cell component to neuron params
        if (!pop.component.empty() && ir.neuronParamSets.count(pop.component) > 0) {
            popIR.neuronParams = pop.component;
        } else {
            popIR.neuronParams = ir.neuronParamSets.begin()->first;
        }

        layer.populations.push_back(popIR);
        col.layers.push_back(layer);
        nucleus.columns.push_back(col);
    }

    region.nuclei.push_back(nucleus);
    lobe.regions.push_back(region);
    hemi.lobes.push_back(lobe);
    brain.hemispheres.push_back(hemi);
    ir.brain = brain;

    // Map projections
    for (const auto& proj : projections) {
        ProjectionIR projIR;
        projIR.name = proj.id;
        projIR.source = "Main/*/" + proj.prePopulation;
        projIR.target = "Main/*/" + proj.postPopulation;

        // Determine connectivity pattern from connection count
        if (proj.connectionCount > 0) {
            projIR.pattern = "all_to_all";  // Explicit connections
        } else {
            projIR.pattern = "random_sparse";
        }

        // Apply synapse type properties
        if (!proj.synapse.empty() && synapseTypes.count(proj.synapse) > 0) {
            const auto& st = synapseTypes.at(proj.synapse);
            projIR.weight = st.weight;
            projIR.delay = st.delay;
        }

        projIR.scope = "global";
        projIR.synapseGroup = proj.id;
        ir.projections.push_back(projIR);
    }

    return ir;
}

} // namespace declarative
} // namespace snnfw


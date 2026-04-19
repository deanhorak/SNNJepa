#include "snnfw/declarative/HOCParser.h"
#include "snnfw/Logger.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <cctype>

namespace snnfw {
namespace declarative {

bool HOCParser::canParse(const std::string& filename) const {
    const std::string ext = ".hoc";
    if (filename.size() >= ext.size()) {
        return filename.compare(filename.size() - ext.size(), ext.size(), ext) == 0;
    }
    return false;
}

NetworkIR HOCParser::parse(const std::string& primaryFile,
                            const std::string& /*auxiliaryFile*/) {
    std::ifstream file(primaryFile);
    if (!file.is_open()) {
        throw ParseError("Cannot open HOC file: " + primaryFile);
    }
    std::string source((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    file.close();

    if (source.empty()) {
        throw ParseError("HOC file is empty: " + primaryFile);
    }

    return parseSource(source);
}

NetworkIR HOCParser::parseSource(const std::string& source) {
    auto lines = preprocessSource(source);

    std::vector<HOCTemplate> templates;
    extractTemplates(lines, templates);

    std::vector<HOCInstantiation> instantiations;
    extractInstantiations(lines, templates, instantiations);

    std::vector<HOCConnection> connections;
    extractConnections(lines, connections);

    return buildIR(templates, instantiations, connections);
}

std::vector<std::string> HOCParser::preprocessSource(const std::string& source) const {
    std::vector<std::string> result;
    std::istringstream stream(source);
    std::string line;

    while (std::getline(stream, line)) {
        // Strip single-line comments (// style)
        auto commentPos = line.find("//");
        if (commentPos != std::string::npos) {
            line = line.substr(0, commentPos);
        }

        // Strip block comments (/* ... */ on single line)
        auto blockStart = line.find("/*");
        if (blockStart != std::string::npos) {
            auto blockEnd = line.find("*/", blockStart);
            if (blockEnd != std::string::npos) {
                line = line.substr(0, blockStart) + line.substr(blockEnd + 2);
            } else {
                line = line.substr(0, blockStart);
            }
        }

        // Trim whitespace
        auto start = line.find_first_not_of(" \t\r\n");
        auto end = line.find_last_not_of(" \t\r\n");
        if (start != std::string::npos) {
            result.push_back(line.substr(start, end - start + 1));
        }
    }

    return result;
}

void HOCParser::extractTemplates(const std::vector<std::string>& lines,
                                  std::vector<HOCTemplate>& templates) const {
    bool inTemplate = false;
    HOCTemplate current;

    for (size_t i = 0; i < lines.size(); ++i) {
        const auto& line = lines[i];

        if (line.find("begintemplate") == 0) {
            inTemplate = true;
            current = HOCTemplate();
            // Extract template name
            auto spacePos = line.find_first_of(" \t");
            if (spacePos != std::string::npos) {
                current.name = line.substr(spacePos + 1);
                // Trim the name
                auto nameEnd = current.name.find_first_of(" \t{");
                if (nameEnd != std::string::npos) {
                    current.name = current.name.substr(0, nameEnd);
                }
            }
            SNNFW_DEBUG("HOC: Found template '{}'", current.name);
            continue;
        }

        if (line.find("endtemplate") == 0) {
            if (inTemplate && !current.name.empty()) {
                templates.push_back(current);
            }
            inTemplate = false;
            continue;
        }

        if (inTemplate) {
            // Look for property assignments: variable = value
            std::string variable;
            double value;
            if (parseAssignment(line, variable, value)) {
                current.properties[variable] = value;
            }

            // Look for create statements: create soma, axon, dendrite
            if (line.find("create") == 0) {
                std::string rest = line.substr(6); // after "create"
                std::istringstream ss(rest);
                std::string section;
                while (std::getline(ss, section, ',')) {
                    auto s = section.find_first_not_of(" \t");
                    auto e = section.find_last_not_of(" \t");
                    if (s != std::string::npos) {
                        current.sections.push_back(section.substr(s, e - s + 1));
                    }
                }
            }
        }
    }
}

void HOCParser::extractInstantiations(const std::vector<std::string>& lines,
                                       const std::vector<HOCTemplate>& templates,
                                       std::vector<HOCInstantiation>& instantiations) const {
    // Build set of known template names
    std::map<std::string, bool> templateNames;
    for (const auto& t : templates) {
        templateNames[t.name] = true;
    }

    int currentLoopCount = 0;
    std::string currentLoopVar;

    for (const auto& line : lines) {
        // Check for loop: "for i = start, end {"  or  "for (i = start; i <= end; i++)"
        std::string var;
        int start, end;
        if (parseForLoop(line, var, start, end)) {
            currentLoopCount = end - start + 1;
            currentLoopVar = var;
            continue;
        }

        // Check for instantiation: "new TemplateName()" or "cells.append(new TemplateName())"
        for (const auto& [tName, _] : templateNames) {
            auto pos = line.find("new " + tName);
            if (pos != std::string::npos) {
                HOCInstantiation inst;
                inst.templateName = tName;
                inst.count = (currentLoopCount > 0) ? currentLoopCount : 1;

                // Try to extract variable name (e.g., "objref cells" or assignment)
                inst.variableName = tName + "_instances";

                instantiations.push_back(inst);
                SNNFW_DEBUG("HOC: Found instantiation of '{}' count={}",
                           tName, inst.count);

                // Reset loop count after use
                if (currentLoopCount > 0) {
                    currentLoopCount = 0;
                }
                break;
            }
        }
    }
}

void HOCParser::extractConnections(const std::vector<std::string>& lines,
                                    std::vector<HOCConnection>& connections) const {
    int currentLoopCount = 0;

    for (const auto& line : lines) {
        std::string var;
        int start, end;
        if (parseForLoop(line, var, start, end)) {
            currentLoopCount = end - start + 1;
            continue;
        }

        // Look for NetCon: "nc = new NetCon(source, target, threshold, delay, weight)"
        auto netconPos = line.find("new NetCon");
        if (netconPos != std::string::npos) {
            HOCConnection conn;
            conn.connectionCount = (currentLoopCount > 0) ? currentLoopCount : 1;

            // Try to parse weight and delay from arguments
            // Use matching-paren logic since args may contain nested parens
            // e.g. NetCon(cells.o(i).soma(0.5), outputs.o(i).syn, 0, 1.5, 0.5)
            auto parenStart = line.find('(', netconPos);
            std::string::size_type parenEnd = std::string::npos;
            if (parenStart != std::string::npos) {
                int depth = 1;
                for (auto j = parenStart + 1; j < line.size() && depth > 0; ++j) {
                    if (line[j] == '(') ++depth;
                    else if (line[j] == ')') { --depth; if (depth == 0) parenEnd = j; }
                }
            }
            if (parenStart != std::string::npos && parenEnd != std::string::npos) {
                std::string args = line.substr(parenStart + 1, parenEnd - parenStart - 1);
                std::istringstream ss(args);
                std::string token;
                std::vector<std::string> tokens;
                while (std::getline(ss, token, ',')) {
                    auto s = token.find_first_not_of(" \t");
                    auto e = token.find_last_not_of(" \t");
                    if (s != std::string::npos) {
                        tokens.push_back(token.substr(s, e - s + 1));
                    }
                }
                // NetCon(source, target, threshold, delay, weight)
                if (tokens.size() >= 5) {
                    try { conn.delay = std::stod(tokens[3]); } catch (...) {}
                    try { conn.weight = std::stod(tokens[4]); } catch (...) {}
                }
            }

            connections.push_back(conn);
            if (currentLoopCount > 0) currentLoopCount = 0;
            continue;
        }

        // Look for connect statement: "connect section1(x), section2(y)"
        if (line.find("connect ") == 0 || line.find("connect\t") == 0) {
            HOCConnection conn;
            conn.connectionCount = (currentLoopCount > 0) ? currentLoopCount : 1;
            connections.push_back(conn);
            if (currentLoopCount > 0) currentLoopCount = 0;
        }
    }
}

bool HOCParser::parseAssignment(const std::string& line,
                                 std::string& variable, double& value) const {
    auto eqPos = line.find('=');
    if (eqPos == std::string::npos || eqPos == 0) return false;

    // Make sure it's not == or !=
    if (eqPos > 0 && (line[eqPos - 1] == '!' || line[eqPos - 1] == '<' ||
                       line[eqPos - 1] == '>' || line[eqPos - 1] == '=')) {
        return false;
    }
    if (eqPos + 1 < line.size() && line[eqPos + 1] == '=') return false;

    // Extract variable name
    std::string varPart = line.substr(0, eqPos);
    auto vs = varPart.find_last_not_of(" \t");
    if (vs == std::string::npos) return false;
    auto vStart = varPart.find_last_of(" \t.", vs);
    vStart = (vStart == std::string::npos) ? 0 : vStart + 1;
    variable = varPart.substr(vStart, vs - vStart + 1);

    // Extract value
    std::string valPart = line.substr(eqPos + 1);
    auto valS = valPart.find_first_not_of(" \t");
    if (valS == std::string::npos) return false;

    try {
        value = std::stod(valPart.substr(valS));
        return true;
    } catch (...) {
        return false;
    }
}

bool HOCParser::parseForLoop(const std::string& line,
                              std::string& var, int& start, int& end) const {
    // Pattern 1: "for i = start, end"
    if (line.find("for ") != 0 && line.find("for\t") != 0) return false;

    std::string rest = line.substr(3);
    auto s = rest.find_first_not_of(" \t");
    if (s == std::string::npos) return false;
    rest = rest.substr(s);

    // Try "var = start, end" pattern
    auto eqPos = rest.find('=');
    auto commaPos = rest.find(',');
    if (eqPos != std::string::npos && commaPos != std::string::npos && commaPos > eqPos) {
        var = rest.substr(0, eqPos);
        auto ve = var.find_last_not_of(" \t");
        if (ve != std::string::npos) var = var.substr(0, ve + 1);

        std::string startStr = rest.substr(eqPos + 1, commaPos - eqPos - 1);
        std::string endStr = rest.substr(commaPos + 1);

        // Strip trailing { or whitespace
        auto bracePos = endStr.find('{');
        if (bracePos != std::string::npos) endStr = endStr.substr(0, bracePos);

        try {
            start = std::stoi(startStr);
            end = std::stoi(endStr);
            return true;
        } catch (...) {
            return false;
        }
    }

    return false;
}

NetworkIR HOCParser::buildIR(const std::vector<HOCTemplate>& templates,
                              const std::vector<HOCInstantiation>& instantiations,
                              const std::vector<HOCConnection>& connections) const {
    NetworkIR ir;
    ir.sourceFormat = "hoc";

    // Create neuron parameter sets from templates
    for (const auto& tmpl : templates) {
        NeuronParamsIR params;
        params.name = tmpl.name;

        // Map HOC properties to NeuronParamsIR
        if (tmpl.properties.count("window_size") > 0 ||
            tmpl.properties.count("window_size_ms") > 0) {
            params.windowSizeMs = tmpl.properties.count("window_size_ms") > 0
                ? tmpl.properties.at("window_size_ms")
                : tmpl.properties.at("window_size");
        }
        if (tmpl.properties.count("threshold") > 0 ||
            tmpl.properties.count("similarity_threshold") > 0) {
            params.similarityThreshold = tmpl.properties.count("similarity_threshold") > 0
                ? tmpl.properties.at("similarity_threshold")
                : tmpl.properties.at("threshold");
        }
        if (tmpl.properties.count("max_patterns") > 0 ||
            tmpl.properties.count("max_reference_patterns") > 0) {
            params.maxReferencePatterns = static_cast<int>(
                tmpl.properties.count("max_reference_patterns") > 0
                    ? tmpl.properties.at("max_reference_patterns")
                    : tmpl.properties.at("max_patterns"));
        }

        ir.neuronParamSets[tmpl.name] = params;
    }

    // Ensure at least a default param set
    if (ir.neuronParamSets.empty()) {
        NeuronParamsIR defaultParams;
        defaultParams.name = "default";
        ir.neuronParamSets["default"] = defaultParams;
    }

    // Build hierarchy: Brain → Hemisphere → Lobe → Region → Nucleus
    BrainIR brain;
    brain.name = "HOCNetwork";

    HemisphereIR hemi;
    hemi.name = "Default";

    LobeIR lobe;
    lobe.name = "Primary";

    RegionIR region;
    region.name = "Main";

    NucleusIR nucleus;
    nucleus.name = "Core";

    // Create columns from instantiations
    for (const auto& inst : instantiations) {
        ColumnIR col;
        col.name = inst.templateName + "_col";

        LayerIR layer;
        layer.name = inst.templateName;

        PopulationIR pop;
        pop.name = inst.templateName + "_neurons";
        pop.count = inst.count;

        // Link to the template's neuron params
        if (ir.neuronParamSets.count(inst.templateName) > 0) {
            pop.neuronParams = inst.templateName;
        } else {
            pop.neuronParams = ir.neuronParamSets.begin()->first;
        }

        layer.populations.push_back(pop);
        col.layers.push_back(layer);
        nucleus.columns.push_back(col);
    }

    region.nuclei.push_back(nucleus);
    lobe.regions.push_back(region);
    hemi.lobes.push_back(lobe);
    brain.hemispheres.push_back(hemi);
    ir.brain = brain;

    // Create projections from connections
    for (size_t i = 0; i < connections.size(); ++i) {
        const auto& conn = connections[i];
        ProjectionIR proj;
        proj.name = "connection_" + std::to_string(i);
        proj.pattern = "all_to_all";
        proj.weight = conn.weight;
        proj.delay = conn.delay;
        proj.scope = "global";
        proj.synapseGroup = proj.name;

        // Source/target from template references if available
        if (!conn.sourceTemplate.empty()) {
            proj.source = "Main/*/" + conn.sourceTemplate;
        }
        if (!conn.targetTemplate.empty()) {
            proj.target = "Main/*/" + conn.targetTemplate;
        }

        ir.projections.push_back(proj);
    }

    return ir;
}

} // namespace declarative
} // namespace snnfw


#ifndef SNNFW_DECLARATIVE_NEUROML_PARSER_H
#define SNNFW_DECLARATIVE_NEUROML_PARSER_H

#include "snnfw/declarative/FormatParser.h"
#include <string>
#include <map>

namespace snnfw {
namespace declarative {

/**
 * @class NeuroMLParser
 * @brief Parser for NeuroML v2 XML neural network descriptions
 *
 * NeuroML is an XML-based model description language for computational
 * neuroscience developed by the NeuroML community (neuroml.org).
 *
 * The parser reads NeuroML XML files containing:
 *   - <cell> definitions → NeuronParamsIR
 *   - <synapse> definitions → projection weight/delay defaults
 *   - <network> with <population> elements → hierarchy populations
 *   - <projection> with <connection> elements → ProjectionIR
 *
 * File detection: *.nml or *.neuroml
 *
 * The mapping from NeuroML to NetworkIR:
 *   - <cell> → NeuronParamsIR (via property annotations)
 *   - <population> → Layer populations in default hierarchy
 *   - <projection> → ProjectionIR
 *   - <connection> → individual connectivity (aggregated into patterns)
 *   - <inputList>/<input> → InputLayerIR hints
 *
 * NeuroML-specific attributes are mapped via the <annotation> or <property>
 * mechanism, using the "snnfw:" namespace prefix for SNNFrame-specific
 * neuron parameters (e.g., snnfw:window_size_ms).
 */
class NeuroMLParser : public FormatParser {
public:
    NetworkIR parse(const std::string& primaryFile,
                    const std::string& auxiliaryFile = "") override;

    bool canParse(const std::string& filename) const override;
    std::string formatName() const override { return "neuroml"; }

    /// Parse from raw XML string (useful for testing)
    NetworkIR parseXML(const std::string& xmlContent);

private:
    /// Struct to hold parsed cell type info
    struct CellType {
        std::string id;
        double windowSizeMs = 500.0;
        double similarityThreshold = 0.93;
        int maxReferencePatterns = 500;
        std::string similarityMetric = "cosine";
    };

    /// Struct to hold parsed synapse type info
    struct SynapseType {
        std::string id;
        double weight = 1.0;
        double delay = 1.5;
    };

    /// Struct for a parsed population
    struct ParsedPopulation {
        std::string id;
        std::string component;  // cell type reference
        int size = 0;
        std::map<std::string, std::string> properties;
    };

    /// Struct for a parsed projection
    struct ParsedProjection {
        std::string id;
        std::string prePopulation;
        std::string postPopulation;
        std::string synapse;
        int connectionCount = 0;
    };

    /// Parse cell types from <cell> elements
    template<typename XmlNode>
    void parseCellTypes(XmlNode* root, std::map<std::string, CellType>& cellTypes) const;

    /// Parse synapse types from synapse elements
    template<typename XmlNode>
    void parseSynapseTypes(XmlNode* root, std::map<std::string, SynapseType>& synapseTypes) const;

    /// Parse network populations and projections
    template<typename XmlNode>
    void parseNetwork(XmlNode* network,
                      std::vector<ParsedPopulation>& populations,
                      std::vector<ParsedProjection>& projections) const;

    /// Build NetworkIR from parsed elements
    NetworkIR buildIR(const std::map<std::string, CellType>& cellTypes,
                      const std::map<std::string, SynapseType>& synapseTypes,
                      const std::vector<ParsedPopulation>& populations,
                      const std::vector<ParsedProjection>& projections,
                      const std::string& networkName) const;
};

} // namespace declarative
} // namespace snnfw

#endif // SNNFW_DECLARATIVE_NEUROML_PARSER_H


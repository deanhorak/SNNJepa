#ifndef SNNFW_DECLARATIVE_HOC_PARSER_H
#define SNNFW_DECLARATIVE_HOC_PARSER_H

#include "snnfw/declarative/FormatParser.h"
#include <string>
#include <vector>
#include <map>

namespace snnfw {
namespace declarative {

/**
 * @class HOCParser
 * @brief Parser for NEURON HOC scripting language network descriptions
 *
 * HOC is the scripting language used by the NEURON simulator. Unlike SONATA
 * and NeuroML which are data formats, HOC is a procedural language. This
 * parser extracts declarative information (cell templates, counts,
 * connectivity patterns) from HOC scripts.
 *
 * File detection: *.hoc
 *
 * Supported HOC constructs:
 *   - begintemplate / endtemplate → NeuronParamsIR (cell type definitions)
 *   - create → section creation (identifies cell compartments)
 *   - connect → section connectivity
 *   - new <Template>() → cell instantiation (population counts)
 *   - NetCon / syn = new <Synapse> → synaptic connections
 *   - for loops → population/connectivity iteration patterns
 *   - proc / func → procedure/function definitions (skipped)
 *
 * The mapping from HOC to NetworkIR:
 *   - Cell templates → NeuronParamsIR
 *   - Cell instantiation counts → PopulationIR
 *   - NetCon/connect patterns → ProjectionIR
 *   - Template properties (window_size, threshold) → NeuronParamsIR fields
 *
 * Limitations:
 *   - Only extracts structural information; procedural logic is not executed
 *   - Complex HOC with conditional template selection may not parse correctly
 *   - External file loading (xopen, load_file) is not followed
 */
class HOCParser : public FormatParser {
public:
    NetworkIR parse(const std::string& primaryFile,
                    const std::string& auxiliaryFile = "") override;

    bool canParse(const std::string& filename) const override;
    std::string formatName() const override { return "hoc"; }

    /// Parse from raw HOC source string (useful for testing)
    NetworkIR parseSource(const std::string& source);

private:
    /// Represents a parsed HOC cell template
    struct HOCTemplate {
        std::string name;
        std::map<std::string, double> properties;  // e.g., window_size, threshold
        std::vector<std::string> sections;          // created sections (soma, axon, etc.)
    };

    /// Represents a parsed cell instantiation
    struct HOCInstantiation {
        std::string templateName;
        std::string variableName;
        int count = 1;  // from loop analysis
    };

    /// Represents a parsed connection
    struct HOCConnection {
        std::string sourceTemplate;
        std::string targetTemplate;
        double weight = 1.0;
        double delay = 1.5;
        int connectionCount = 0;
    };

    /// Tokenize HOC source into lines, stripping comments
    std::vector<std::string> preprocessSource(const std::string& source) const;

    /// Extract cell templates from preprocessed lines
    void extractTemplates(const std::vector<std::string>& lines,
                          std::vector<HOCTemplate>& templates) const;

    /// Extract cell instantiations (new Template() calls)
    void extractInstantiations(const std::vector<std::string>& lines,
                               const std::vector<HOCTemplate>& templates,
                               std::vector<HOCInstantiation>& instantiations) const;

    /// Extract connections (NetCon, connect statements)
    void extractConnections(const std::vector<std::string>& lines,
                            std::vector<HOCConnection>& connections) const;

    /// Build NetworkIR from extracted HOC constructs
    NetworkIR buildIR(const std::vector<HOCTemplate>& templates,
                      const std::vector<HOCInstantiation>& instantiations,
                      const std::vector<HOCConnection>& connections) const;

    /// Parse a numeric value from HOC assignment (e.g., "window_size = 200")
    bool parseAssignment(const std::string& line, std::string& variable, double& value) const;

    /// Extract loop bounds from a for statement (e.g., "for i = 0, 48")
    bool parseForLoop(const std::string& line, std::string& var, int& start, int& end) const;
};

} // namespace declarative
} // namespace snnfw

#endif // SNNFW_DECLARATIVE_HOC_PARSER_H


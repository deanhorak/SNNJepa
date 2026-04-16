#ifndef SNNFW_DECLARATIVE_NATIVE_JSON_PARSER_H
#define SNNFW_DECLARATIVE_NATIVE_JSON_PARSER_H

#include "snnfw/declarative/FormatParser.h"
#include <nlohmann/json.hpp>

namespace snnfw {
namespace declarative {

/**
 * @class NativeJSONParser
 * @brief Parser for the native SNNFrame JSON description format (.snnf.json)
 *
 * This is the primary and most expressive format for describing SNNFrame networks.
 * It provides a direct JSON serialization of the NetworkIR, supporting all features
 * including column templates, path-based connectivity, named neuron parameters,
 * Gabor configuration, saccade attention, and STDP settings.
 */
class NativeJSONParser : public FormatParser {
public:
    NetworkIR parse(const std::string& primaryFile,
                    const std::string& auxiliaryFile = "") override;

    bool canParse(const std::string& filename) const override;
    std::string formatName() const override { return "snnframe_json"; }

    /// Parse from an already-loaded JSON object (useful for testing)
    NetworkIR parseJson(const nlohmann::json& root);

private:
    NeuronParamsIR parseNeuronParams(const nlohmann::json& j);
    PopulationIR parsePopulation(const nlohmann::json& j);
    LayerIR parseLayer(const nlohmann::json& j);
    ColumnIR parseColumn(const nlohmann::json& j);
    ColumnTemplateIR parseColumnTemplate(const nlohmann::json& j);
    NucleusIR parseNucleus(const nlohmann::json& j);
    RegionIR parseRegion(const nlohmann::json& j);
    LobeIR parseLobe(const nlohmann::json& j);
    HemisphereIR parseHemisphere(const nlohmann::json& j);
    BrainIR parseBrain(const nlohmann::json& j);
    ProjectionIR parseProjection(const nlohmann::json& j);
    InputLayerIR parseInputLayer(const nlohmann::json& j);
    OutputLayerIR parseOutputLayer(const nlohmann::json& j);
    GaborConfigIR parseGabor(const nlohmann::json& j);
    SaccadeConfigIR parseSaccades(const nlohmann::json& j);
    SimulationConfigIR parseSimulation(const nlohmann::json& j);
    AdapterConfigIR parseAdapter(const nlohmann::json& j);
    ClassificationConfigIR parseClassification(const nlohmann::json& j);
};

} // namespace declarative
} // namespace snnfw

#endif // SNNFW_DECLARATIVE_NATIVE_JSON_PARSER_H

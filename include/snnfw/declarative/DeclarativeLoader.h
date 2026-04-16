#ifndef SNNFW_DECLARATIVE_DECLARATIVE_LOADER_H
#define SNNFW_DECLARATIVE_DECLARATIVE_LOADER_H

#include "snnfw/declarative/NetworkIR.h"
#include "snnfw/declarative/FormatParser.h"
#include "snnfw/declarative/NetworkConstructor.h"
#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Datastore.h"
#include <memory>
#include <vector>
#include <string>

namespace snnfw {
namespace declarative {

/**
 * @class DeclarativeLoader
 * @brief User-facing API for loading networks from description files
 *
 * Auto-detects the format from the file extension and delegates to the
 * appropriate FormatParser. The parsed NetworkIR is then passed to
 * NetworkConstructor to build the actual network.
 *
 * Supported extensions:
 *   .snnf.json  → Native SNNFrame JSON format
 *   (Future: .sonata.json, .nml, .neuroml, .hoc)
 *
 * Example usage:
 * @code
 * NeuralObjectFactory factory;
 * Datastore datastore("./db");
 * DeclarativeLoader loader(factory, datastore);
 * auto network = loader.loadNetwork("configs/my_network.snnf.json");
 * // network.brain, network.spikeProcessor, network.propagator are ready
 * @endcode
 */
class DeclarativeLoader {
public:
    DeclarativeLoader(NeuralObjectFactory& factory, Datastore& datastore);

    /**
     * @brief Load and construct a network from a description file
     * @param descriptionFile Path to the network description file
     * @param auxiliaryFile Optional path to secondary file (e.g., simulation config)
     * @return Fully constructed network ready for use
     * @throws ParseError if parsing fails
     * @throws std::runtime_error if construction fails
     */
    ConstructedNetwork loadNetwork(const std::string& descriptionFile,
                                    const std::string& auxiliaryFile = "");

    /**
     * @brief Construct a network from a pre-built IR
     * @param ir The NetworkIR to construct from
     * @return Fully constructed network
     */
    ConstructedNetwork loadFromIR(const NetworkIR& ir);

    /**
     * @brief Parse a file without constructing (useful for validation)
     * @param descriptionFile Path to the file to parse
     * @return The parsed NetworkIR
     * @throws ParseError if parsing fails
     */
    NetworkIR parseOnly(const std::string& descriptionFile);

    /**
     * @brief Register a custom format parser
     * @param parser Unique pointer to the parser to register
     */
    void registerParser(std::unique_ptr<FormatParser> parser);

    /**
     * @brief Get the list of registered format names
     * @return Vector of format name strings
     */
    std::vector<std::string> getRegisteredFormats() const;

private:
    NeuralObjectFactory& factory_;
    Datastore& datastore_;
    std::vector<std::unique_ptr<FormatParser>> parsers_;

    /// Find a parser that can handle the given file
    FormatParser* findParser(const std::string& filename);
};

} // namespace declarative
} // namespace snnfw

#endif // SNNFW_DECLARATIVE_DECLARATIVE_LOADER_H


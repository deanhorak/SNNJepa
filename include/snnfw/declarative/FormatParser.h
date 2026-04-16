#ifndef SNNFW_DECLARATIVE_FORMAT_PARSER_H
#define SNNFW_DECLARATIVE_FORMAT_PARSER_H

#include "snnfw/declarative/NetworkIR.h"
#include <string>
#include <stdexcept>

namespace snnfw {
namespace declarative {

/**
 * @class ParseError
 * @brief Exception thrown when a format parser encounters an error
 */
class ParseError : public std::runtime_error {
public:
    explicit ParseError(const std::string& message)
        : std::runtime_error(message) {}
};

/**
 * @class FormatParser
 * @brief Abstract base class for network description file parsers
 *
 * Each supported format (native JSON, SONATA, NeuroML, HOC) provides
 * a concrete implementation that translates the format into a NetworkIR.
 */
class FormatParser {
public:
    virtual ~FormatParser() = default;

    /**
     * @brief Parse the file(s) and produce a NetworkIR
     * @param primaryFile Path to the main description file
     * @param auxiliaryFile Optional path to a secondary file (e.g., simulation config)
     * @return Populated NetworkIR
     * @throws ParseError if parsing fails
     */
    virtual NetworkIR parse(const std::string& primaryFile,
                            const std::string& auxiliaryFile = "") = 0;

    /**
     * @brief Check if this parser can handle the given file
     * @param filename Path to the file to check
     * @return true if this parser can handle the file
     */
    virtual bool canParse(const std::string& filename) const = 0;

    /**
     * @brief Get the format name (e.g., "snnframe_json", "sonata", "neuroml")
     * @return Format name string
     */
    virtual std::string formatName() const = 0;
};

} // namespace declarative
} // namespace snnfw

#endif // SNNFW_DECLARATIVE_FORMAT_PARSER_H


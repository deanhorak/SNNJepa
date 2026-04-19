#include "snnfw/declarative/DeclarativeLoader.h"
#include "snnfw/declarative/NativeJSONParser.h"
#include "snnfw/declarative/SONATAParser.h"
#include "snnfw/declarative/NeuroMLParser.h"
#include "snnfw/declarative/HOCParser.h"
#include "snnfw/Logger.h"
#include <stdexcept>

namespace snnfw {
namespace declarative {

DeclarativeLoader::DeclarativeLoader(NeuralObjectFactory& factory, Datastore& datastore)
    : factory_(factory), datastore_(datastore) {
    // Register built-in parsers
    parsers_.push_back(std::make_unique<NativeJSONParser>());
    parsers_.push_back(std::make_unique<SONATAParser>());
    parsers_.push_back(std::make_unique<NeuroMLParser>());
    parsers_.push_back(std::make_unique<HOCParser>());
}

ConstructedNetwork DeclarativeLoader::loadNetwork(const std::string& descriptionFile,
                                                    const std::string& auxiliaryFile) {
    SNNFW_INFO("DeclarativeLoader: Loading network from '{}'", descriptionFile);

    // Parse
    auto ir = parseOnly(descriptionFile);

    // Validate
    auto errors = ir.getValidationErrors();
    if (!errors.empty()) {
        std::string msg = "Validation errors in '" + descriptionFile + "':\n";
        for (const auto& e : errors) {
            msg += "  - " + e + "\n";
        }
        throw std::runtime_error(msg);
    }

    SNNFW_INFO("DeclarativeLoader: IR validated successfully");

    // Construct
    return loadFromIR(ir);
}

ConstructedNetwork DeclarativeLoader::loadFromIR(const NetworkIR& ir) {
    NetworkConstructor constructor(factory_, datastore_);
    return constructor.construct(ir);
}

NetworkIR DeclarativeLoader::parseOnly(const std::string& descriptionFile) {
    auto* parser = findParser(descriptionFile);
    if (!parser) {
        throw ParseError("No parser found for file: " + descriptionFile +
                         ". Registered formats: " + [this]() {
                             std::string s;
                             for (const auto& p : parsers_) {
                                 if (!s.empty()) s += ", ";
                                 s += p->formatName();
                             }
                             return s;
                         }());
    }

    SNNFW_INFO("DeclarativeLoader: Using '{}' parser for '{}'",
               parser->formatName(), descriptionFile);

    return parser->parse(descriptionFile);
}

void DeclarativeLoader::registerParser(std::unique_ptr<FormatParser> parser) {
    if (parser) {
        SNNFW_INFO("DeclarativeLoader: Registered parser for format '{}'",
                   parser->formatName());
        parsers_.push_back(std::move(parser));
    }
}

std::vector<std::string> DeclarativeLoader::getRegisteredFormats() const {
    std::vector<std::string> formats;
    formats.reserve(parsers_.size());
    for (const auto& p : parsers_) {
        formats.push_back(p->formatName());
    }
    return formats;
}

FormatParser* DeclarativeLoader::findParser(const std::string& filename) {
    for (const auto& parser : parsers_) {
        if (parser->canParse(filename)) {
            return parser.get();
        }
    }
    return nullptr;
}

} // namespace declarative
} // namespace snnfw


#include "snnfw/Synapse.h"
#include "snnfw/Logger.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace snnfw {

std::string Synapse::toJson() const {
    json j;
    j["type"] = "Synapse";
    j["id"] = getId();
    j["axonId"] = axonId;
    j["dendriteId"] = dendriteId;
    j["weight"] = weight;
    j["delay"] = delay;
    return j.dump();
}

bool Synapse::fromJson(const std::string& jsonStr) {
    try {
        json j = json::parse(jsonStr);

        if (j["type"] != "Synapse") {
            SNNFW_ERROR("Invalid type in JSON: expected 'Synapse', got '{}'", j["type"].get<std::string>());
            return false;
        }

        // Deserialize ID from base class
        setId(j["id"]);

        axonId = j["axonId"];
        dendriteId = j["dendriteId"];
        weight = j["weight"];
        delay = j["delay"];

        return true;
    } catch (const std::exception& e) {
        SNNFW_ERROR("Failed to deserialize Synapse from JSON: {}", e.what());
        return false;
    }
}

} // namespace snnfw


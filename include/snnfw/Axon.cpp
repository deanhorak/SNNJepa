#include "snnfw/Axon.h"
#include "snnfw/Logger.h"
#include <nlohmann/json.hpp>
#include <algorithm>

using json = nlohmann::json;

namespace snnfw {

void Axon::addSynapse(uint64_t synapseId) {
    // Check if synapse is already connected
    auto it = std::find(synapseIds.begin(), synapseIds.end(), synapseId);
    if (it == synapseIds.end()) {
        synapseIds.push_back(synapseId);
        SNNFW_DEBUG("Axon {}: Added synapse {} (total: {})", 
                    getId(), synapseId, synapseIds.size());
    } else {
        SNNFW_WARN("Axon {}: Synapse {} already connected", getId(), synapseId);
    }
}

bool Axon::removeSynapse(uint64_t synapseId) {
    auto it = std::find(synapseIds.begin(), synapseIds.end(), synapseId);
    if (it != synapseIds.end()) {
        synapseIds.erase(it);
        SNNFW_DEBUG("Axon {}: Removed synapse {} (remaining: {})",
                    getId(), synapseId, synapseIds.size());
        return true;
    }
    SNNFW_WARN("Axon {}: Synapse {} not found for removal", getId(), synapseId);
    return false;
}

std::string Axon::toJson() const {
    json j;
    j["type"] = "Axon";
    j["id"] = getId();
    j["sourceNeuronId"] = sourceNeuronId;
    j["synapseIds"] = synapseIds;
    return j.dump();
}

bool Axon::fromJson(const std::string& jsonStr) {
    try {
        json j = json::parse(jsonStr);

        if (j["type"] != "Axon") {
            SNNFW_ERROR("Invalid type in JSON: expected 'Axon', got '{}'", j["type"].get<std::string>());
            return false;
        }

        // Deserialize ID from base class
        setId(j["id"]);

        sourceNeuronId = j["sourceNeuronId"];
        synapseIds = j["synapseIds"].get<std::vector<uint64_t>>();

        return true;
    } catch (const std::exception& e) {
        SNNFW_ERROR("Failed to deserialize Axon from JSON: {}", e.what());
        return false;
    }
}

} // namespace snnfw


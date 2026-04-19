#include "snnfw/Cluster.h"
#include "snnfw/Logger.h"
#include <nlohmann/json.hpp>
#include <algorithm>

using json = nlohmann::json;

namespace snnfw {

Cluster::Cluster(uint64_t clusterId) : NeuralObject(clusterId) {}

void Cluster::addNeuron(uint64_t neuronId) {
    neuronIds.push_back(neuronId);
}

uint64_t Cluster::getNeuronId(size_t index) const {
    if (index < neuronIds.size()) {
        return neuronIds[index];
    }
    return 0;
}

bool Cluster::removeNeuron(uint64_t neuronId) {
    auto it = std::find(neuronIds.begin(), neuronIds.end(), neuronId);
    if (it != neuronIds.end()) {
        neuronIds.erase(it);
        return true;
    }
    return false;
}

void Cluster::clear() {
    neuronIds.clear();
}

void Cluster::printInfo() const {
    SNNFW_INFO("Cluster ID: {}", getId());
    SNNFW_INFO("Number of neurons: {}", neuronIds.size());

    for (size_t i = 0; i < neuronIds.size(); ++i) {
        SNNFW_INFO("  Neuron[{}] ID: {}", i, neuronIds[i]);
    }
}

std::string Cluster::toJson() const {
    json j;
    j["type"] = "Cluster";
    j["id"] = getId();
    j["neuronIds"] = neuronIds;

    // Serialize position if set
    if (hasPosition()) {
        const Position3D& pos = getPosition();
        j["position"] = {
            {"x", pos.x},
            {"y", pos.y},
            {"z", pos.z}
        };
    }

    return j.dump();
}

bool Cluster::fromJson(const std::string& jsonStr) {
    try {
        json j = json::parse(jsonStr);

        if (j["type"] != "Cluster") {
            SNNFW_ERROR("Invalid type in JSON: expected 'Cluster', got '{}'", j["type"].get<std::string>());
            return false;
        }

        // Deserialize ID from base class
        setId(j["id"]);

        // Deserialize position if present
        if (j.contains("position")) {
            float x = j["position"]["x"].get<float>();
            float y = j["position"]["y"].get<float>();
            float z = j["position"]["z"].get<float>();
            setPosition(x, y, z);
        } else {
            clearPosition();
        }

        neuronIds = j["neuronIds"].get<std::vector<uint64_t>>();

        return true;
    } catch (const std::exception& e) {
        SNNFW_ERROR("Failed to deserialize Cluster from JSON: {}", e.what());
        return false;
    }
}

} // namespace snnfw


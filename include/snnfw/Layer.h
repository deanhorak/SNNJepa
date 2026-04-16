#ifndef SNNFW_LAYER_H
#define SNNFW_LAYER_H

#include "snnfw/NeuralObject.h"
#include "snnfw/Logger.h"
#include <nlohmann/json.hpp>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace snnfw {

/**
 * @class Layer
 * @brief Represents a layer containing a collection of Cluster IDs
 *
 * A Layer is a structural component in the neural hierarchy that groups
 * multiple clusters together. This follows biological organization where
 * neurons are organized into layers (e.g., cortical layers I-VI).
 *
 * Note: Stores cluster IDs rather than cluster objects for memory efficiency.
 */
class Layer : public NeuralObject {
public:
    /**
     * @brief Construct a new Layer object
     * @param layerId Unique identifier for this layer
     */
    explicit Layer(uint64_t layerId = 0) : NeuralObject(layerId) {}

    /**
     * @brief Add a cluster to this layer
     * @param clusterId ID of the cluster to add
     */
    void addCluster(uint64_t clusterId) {
        clusterIds.push_back(clusterId);
    }

    /**
     * @brief Get a cluster ID by index
     * @param index Index of the cluster
     * @return Cluster ID, or 0 if index is out of range
     */
    uint64_t getClusterId(size_t index) const {
        if (index < clusterIds.size()) {
            return clusterIds[index];
        }
        return 0;
    }

    /**
     * @brief Remove a cluster by its ID
     * @param clusterId ID of the cluster to remove
     * @return true if cluster was found and removed, false otherwise
     */
    bool removeCluster(uint64_t clusterId) {
        auto it = std::remove(clusterIds.begin(), clusterIds.end(), clusterId);

        if (it != clusterIds.end()) {
            clusterIds.erase(it, clusterIds.end());
            return true;
        }
        return false;
    }

    /**
     * @brief Get the number of clusters in this layer
     * @return Number of clusters
     */
    size_t size() const {
        return clusterIds.size();
    }

    /**
     * @brief Clear all clusters from this layer
     */
    void clear() {
        clusterIds.clear();
    }

    /**
     * @brief Get all cluster IDs in this layer
     * @return Const reference to the vector of cluster IDs
     */
    const std::vector<uint64_t>& getClusterIds() const {
        return clusterIds;
    }

    // Serializable interface implementation
    std::string toJson() const override {
        nlohmann::json j;
        j["type"] = "Layer";
        j["id"] = getId();
        j["clusterIds"] = clusterIds;

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

    bool fromJson(const std::string& jsonStr) override {
        try {
            nlohmann::json j = nlohmann::json::parse(jsonStr);
            if (j["type"] != "Layer") {
                SNNFW_ERROR("Invalid type in JSON: expected 'Layer', got '{}'", j["type"].get<std::string>());
                return false;
            }
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

            clusterIds = j["clusterIds"].get<std::vector<uint64_t>>();
            return true;
        } catch (const std::exception& e) {
            SNNFW_ERROR("Failed to deserialize Layer from JSON: {}", e.what());
            return false;
        }
    }

    std::string getTypeName() const override { return "Layer"; }

private:
    std::vector<uint64_t> clusterIds;
};

} // namespace snnfw

#endif // SNNFW_LAYER_H


#ifndef SNNFW_COLUMN_H
#define SNNFW_COLUMN_H

#include "snnfw/NeuralObject.h"
#include "snnfw/Logger.h"
#include <nlohmann/json.hpp>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace snnfw {

/**
 * @class Column
 * @brief Represents a column containing a collection of Layer IDs
 *
 * A Column is a structural component that groups multiple layers together.
 * This follows biological organization such as cortical columns, which are
 * vertical arrangements of neurons spanning multiple cortical layers.
 *
 * Note: Stores layer IDs rather than layer objects for memory efficiency.
 */
class Column : public NeuralObject {
public:
    /**
     * @brief Construct a new Column object
     * @param columnId Unique identifier for this column
     */
    explicit Column(uint64_t columnId = 0) : NeuralObject(columnId) {}

    /**
     * @brief Add a layer to this column
     * @param layerId ID of the layer to add
     */
    void addLayer(uint64_t layerId) {
        layerIds.push_back(layerId);
    }

    /**
     * @brief Get a layer ID by index
     * @param index Index of the layer
     * @return Layer ID, or 0 if index is out of range
     */
    uint64_t getLayerId(size_t index) const {
        if (index < layerIds.size()) {
            return layerIds[index];
        }
        return 0;
    }

    /**
     * @brief Remove a layer by its ID
     * @param layerId ID of the layer to remove
     * @return true if layer was found and removed, false otherwise
     */
    bool removeLayer(uint64_t layerId) {
        auto it = std::remove(layerIds.begin(), layerIds.end(), layerId);

        if (it != layerIds.end()) {
            layerIds.erase(it, layerIds.end());
            return true;
        }
        return false;
    }

    /**
     * @brief Get the number of layers in this column
     * @return Number of layers
     */
    size_t size() const {
        return layerIds.size();
    }

    /**
     * @brief Clear all layers from this column
     */
    void clear() {
        layerIds.clear();
    }

    /**
     * @brief Get all layer IDs in this column
     * @return Const reference to the vector of layer IDs
     */
    const std::vector<uint64_t>& getLayerIds() const {
        return layerIds;
    }

    // Serializable interface implementation
    std::string toJson() const override {
        nlohmann::json j;
        j["type"] = "Column";
        j["id"] = getId();
        j["layerIds"] = layerIds;

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
            if (j["type"] != "Column") {
                SNNFW_ERROR("Invalid type in JSON: expected 'Column', got '{}'", j["type"].get<std::string>());
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

            layerIds = j["layerIds"].get<std::vector<uint64_t>>();
            return true;
        } catch (const std::exception& e) {
            SNNFW_ERROR("Failed to deserialize Column from JSON: {}", e.what());
            return false;
        }
    }

    std::string getTypeName() const override { return "Column"; }

private:
    std::vector<uint64_t> layerIds;
};

} // namespace snnfw

#endif // SNNFW_COLUMN_H


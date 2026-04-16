#ifndef SNNFW_LOBE_H
#define SNNFW_LOBE_H

#include "snnfw/NeuralObject.h"
#include "snnfw/Logger.h"
#include <nlohmann/json.hpp>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace snnfw {

/**
 * @class Lobe
 * @brief Represents a lobe containing a collection of Region IDs
 *
 * A Lobe is a structural component that groups multiple regions together.
 * In biological terms, lobes are major divisions of the brain
 * (e.g., frontal lobe, parietal lobe, temporal lobe, occipital lobe).
 *
 * Note: Stores region IDs rather than region objects for memory efficiency.
 */
class Lobe : public NeuralObject {
public:
    /**
     * @brief Construct a new Lobe object
     * @param lobeId Unique identifier for this lobe
     * @param lobeName Optional name for this lobe
     */
    explicit Lobe(uint64_t lobeId = 0, const std::string& lobeName = "")
        : NeuralObject(lobeId), name(lobeName) {}

    /**
     * @brief Set the name of this lobe
     * @param lobeName Name to set
     */
    void setName(const std::string& lobeName) {
        name = lobeName;
    }

    /**
     * @brief Get the name of this lobe
     * @return Name of the lobe
     */
    const std::string& getName() const {
        return name;
    }

    /**
     * @brief Add a region to this lobe
     * @param regionId ID of the region to add
     */
    void addRegion(uint64_t regionId) {
        regionIds.push_back(regionId);
    }

    /**
     * @brief Get a region ID by index
     * @param index Index of the region
     * @return Region ID, or 0 if index is out of range
     */
    uint64_t getRegionId(size_t index) const {
        if (index < regionIds.size()) {
            return regionIds[index];
        }
        return 0;
    }

    /**
     * @brief Remove a region by its ID
     * @param regionId ID of the region to remove
     * @return true if region was found and removed, false otherwise
     */
    bool removeRegion(uint64_t regionId) {
        auto it = std::remove(regionIds.begin(), regionIds.end(), regionId);

        if (it != regionIds.end()) {
            regionIds.erase(it, regionIds.end());
            return true;
        }
        return false;
    }

    /**
     * @brief Get the number of regions in this lobe
     * @return Number of regions
     */
    size_t size() const {
        return regionIds.size();
    }

    /**
     * @brief Clear all regions from this lobe
     */
    void clear() {
        regionIds.clear();
    }

    /**
     * @brief Get all region IDs in this lobe
     * @return Const reference to the vector of region IDs
     */
    const std::vector<uint64_t>& getRegionIds() const {
        return regionIds;
    }

    // Serializable interface implementation
    std::string toJson() const override {
        nlohmann::json j;
        j["type"] = "Lobe";
        j["id"] = getId();
        j["name"] = name;
        j["regionIds"] = regionIds;

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
            if (j["type"] != "Lobe") {
                SNNFW_ERROR("Invalid type in JSON: expected 'Lobe', got '{}'", j["type"].get<std::string>());
                return false;
            }
            setId(j["id"]);
            name = j.value("name", "");

            // Deserialize position if present
            if (j.contains("position")) {
                float x = j["position"]["x"].get<float>();
                float y = j["position"]["y"].get<float>();
                float z = j["position"]["z"].get<float>();
                setPosition(x, y, z);
            } else {
                clearPosition();
            }

            regionIds = j["regionIds"].get<std::vector<uint64_t>>();
            return true;
        } catch (const std::exception& e) {
            SNNFW_ERROR("Failed to deserialize Lobe from JSON: {}", e.what());
            return false;
        }
    }

    std::string getTypeName() const override { return "Lobe"; }

private:
    std::string name;
    std::vector<uint64_t> regionIds;
};

} // namespace snnfw

#endif // SNNFW_LOBE_H


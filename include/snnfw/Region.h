#ifndef SNNFW_REGION_H
#define SNNFW_REGION_H

#include "snnfw/NeuralObject.h"
#include "snnfw/Logger.h"
#include <nlohmann/json.hpp>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace snnfw {

/**
 * @class Region
 * @brief Represents a region containing a collection of Nucleus IDs
 *
 * A Region is a structural component that groups multiple nuclei together.
 * In biological terms, a region represents a functional area of the brain
 * (e.g., visual cortex, motor cortex, hippocampus).
 *
 * Note: Stores nucleus IDs rather than nucleus objects for memory efficiency.
 */
class Region : public NeuralObject {
public:
    /**
     * @brief Construct a new Region object
     * @param regionId Unique identifier for this region
     * @param regionName Optional name for this region
     */
    explicit Region(uint64_t regionId = 0, const std::string& regionName = "")
        : NeuralObject(regionId), name(regionName) {}

    /**
     * @brief Set the name of this region
     * @param regionName Name to set
     */
    void setName(const std::string& regionName) {
        name = regionName;
    }

    /**
     * @brief Get the name of this region
     * @return Name of the region
     */
    const std::string& getName() const {
        return name;
    }

    /**
     * @brief Add a nucleus to this region
     * @param nucleusId ID of the nucleus to add
     */
    void addNucleus(uint64_t nucleusId) {
        nucleusIds.push_back(nucleusId);
    }

    /**
     * @brief Get a nucleus ID by index
     * @param index Index of the nucleus
     * @return Nucleus ID, or 0 if index is out of range
     */
    uint64_t getNucleusId(size_t index) const {
        if (index < nucleusIds.size()) {
            return nucleusIds[index];
        }
        return 0;
    }

    /**
     * @brief Remove a nucleus by its ID
     * @param nucleusId ID of the nucleus to remove
     * @return true if nucleus was found and removed, false otherwise
     */
    bool removeNucleus(uint64_t nucleusId) {
        auto it = std::remove(nucleusIds.begin(), nucleusIds.end(), nucleusId);

        if (it != nucleusIds.end()) {
            nucleusIds.erase(it, nucleusIds.end());
            return true;
        }
        return false;
    }

    /**
     * @brief Get the number of nuclei in this region
     * @return Number of nuclei
     */
    size_t size() const {
        return nucleusIds.size();
    }

    /**
     * @brief Clear all nuclei from this region
     */
    void clear() {
        nucleusIds.clear();
    }

    /**
     * @brief Get all nucleus IDs in this region
     * @return Const reference to the vector of nucleus IDs
     */
    const std::vector<uint64_t>& getNucleusIds() const {
        return nucleusIds;
    }

    // Serializable interface implementation
    std::string toJson() const override {
        nlohmann::json j;
        j["type"] = "Region";
        j["id"] = getId();
        j["name"] = name;
        j["nucleusIds"] = nucleusIds;

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
            if (j["type"] != "Region") {
                SNNFW_ERROR("Invalid type in JSON: expected 'Region', got '{}'", j["type"].get<std::string>());
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

            nucleusIds = j["nucleusIds"].get<std::vector<uint64_t>>();
            return true;
        } catch (const std::exception& e) {
            SNNFW_ERROR("Failed to deserialize Region from JSON: {}", e.what());
            return false;
        }
    }

    std::string getTypeName() const override { return "Region"; }

private:
    std::string name;
    std::vector<uint64_t> nucleusIds;
};

} // namespace snnfw

#endif // SNNFW_REGION_H


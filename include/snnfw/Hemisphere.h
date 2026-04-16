#ifndef SNNFW_HEMISPHERE_H
#define SNNFW_HEMISPHERE_H

#include "snnfw/NeuralObject.h"
#include "snnfw/Logger.h"
#include <nlohmann/json.hpp>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace snnfw {

/**
 * @class Hemisphere
 * @brief Represents a hemisphere containing a collection of Lobe IDs
 *
 * A Hemisphere is a structural component that groups multiple lobes together.
 * In biological terms, the brain is divided into left and right hemispheres,
 * each containing multiple lobes.
 *
 * Note: Stores lobe IDs rather than lobe objects for memory efficiency.
 */
class Hemisphere : public NeuralObject {
public:
    /**
     * @brief Construct a new Hemisphere object
     * @param hemisphereId Unique identifier for this hemisphere
     * @param hemisphereName Optional name for this hemisphere
     */
    explicit Hemisphere(uint64_t hemisphereId = 0, const std::string& hemisphereName = "")
        : NeuralObject(hemisphereId), name(hemisphereName) {}

    /**
     * @brief Set the name of this hemisphere
     * @param hemisphereName Name to set
     */
    void setName(const std::string& hemisphereName) {
        name = hemisphereName;
    }

    /**
     * @brief Get the name of this hemisphere
     * @return Name of the hemisphere
     */
    const std::string& getName() const {
        return name;
    }

    /**
     * @brief Add a lobe to this hemisphere
     * @param lobeId ID of the lobe to add
     */
    void addLobe(uint64_t lobeId) {
        lobeIds.push_back(lobeId);
    }

    /**
     * @brief Get a lobe ID by index
     * @param index Index of the lobe
     * @return Lobe ID, or 0 if index is out of range
     */
    uint64_t getLobeId(size_t index) const {
        if (index < lobeIds.size()) {
            return lobeIds[index];
        }
        return 0;
    }

    /**
     * @brief Remove a lobe by its ID
     * @param lobeId ID of the lobe to remove
     * @return true if lobe was found and removed, false otherwise
     */
    bool removeLobe(uint64_t lobeId) {
        auto it = std::remove(lobeIds.begin(), lobeIds.end(), lobeId);

        if (it != lobeIds.end()) {
            lobeIds.erase(it, lobeIds.end());
            return true;
        }
        return false;
    }

    /**
     * @brief Get the number of lobes in this hemisphere
     * @return Number of lobes
     */
    size_t size() const {
        return lobeIds.size();
    }

    /**
     * @brief Clear all lobes from this hemisphere
     */
    void clear() {
        lobeIds.clear();
    }

    /**
     * @brief Get all lobe IDs in this hemisphere
     * @return Const reference to the vector of lobe IDs
     */
    const std::vector<uint64_t>& getLobeIds() const {
        return lobeIds;
    }

    // Serializable interface implementation
    std::string toJson() const override {
        nlohmann::json j;
        j["type"] = "Hemisphere";
        j["id"] = getId();
        j["name"] = name;
        j["lobeIds"] = lobeIds;

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
            if (j["type"] != "Hemisphere") {
                SNNFW_ERROR("Invalid type in JSON: expected 'Hemisphere', got '{}'", j["type"].get<std::string>());
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

            lobeIds = j["lobeIds"].get<std::vector<uint64_t>>();
            return true;
        } catch (const std::exception& e) {
            SNNFW_ERROR("Failed to deserialize Hemisphere from JSON: {}", e.what());
            return false;
        }
    }

    std::string getTypeName() const override { return "Hemisphere"; }

private:
    std::string name;
    std::vector<uint64_t> lobeIds;
};

} // namespace snnfw

#endif // SNNFW_HEMISPHERE_H


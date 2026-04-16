#ifndef SNNFW_BRAIN_H
#define SNNFW_BRAIN_H

#include "snnfw/NeuralObject.h"
#include "snnfw/Logger.h"
#include <nlohmann/json.hpp>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace snnfw {

/**
 * @class Brain
 * @brief Represents a brain containing a collection of Hemisphere IDs
 *
 * A Brain is the top-level structural component in the neural hierarchy.
 * It contains hemispheres, which in turn contain the entire hierarchical
 * structure of the neural network (lobes, regions, nuclei, columns, layers,
 * clusters, and neurons).
 *
 * Note: Stores hemisphere IDs rather than hemisphere objects for memory efficiency.
 */
class Brain : public NeuralObject {
public:
    /**
     * @brief Construct a new Brain object
     * @param brainId Unique identifier for this brain
     * @param brainName Optional name for this brain
     */
    explicit Brain(uint64_t brainId = 0, const std::string& brainName = "")
        : NeuralObject(brainId), name(brainName) {}

    /**
     * @brief Set the name of this brain
     * @param brainName Name to set
     */
    void setName(const std::string& brainName) {
        name = brainName;
    }

    /**
     * @brief Get the name of this brain
     * @return Name of the brain
     */
    const std::string& getName() const {
        return name;
    }

    /**
     * @brief Add a hemisphere to this brain
     * @param hemisphereId ID of the hemisphere to add
     */
    void addHemisphere(uint64_t hemisphereId) {
        hemisphereIds.push_back(hemisphereId);
    }

    /**
     * @brief Get a hemisphere ID by index
     * @param index Index of the hemisphere
     * @return Hemisphere ID, or 0 if index is out of range
     */
    uint64_t getHemisphereId(size_t index) const {
        if (index < hemisphereIds.size()) {
            return hemisphereIds[index];
        }
        return 0;
    }

    /**
     * @brief Remove a hemisphere by its ID
     * @param hemisphereId ID of the hemisphere to remove
     * @return true if hemisphere was found and removed, false otherwise
     */
    bool removeHemisphere(uint64_t hemisphereId) {
        auto it = std::remove(hemisphereIds.begin(), hemisphereIds.end(), hemisphereId);

        if (it != hemisphereIds.end()) {
            hemisphereIds.erase(it, hemisphereIds.end());
            return true;
        }
        return false;
    }

    /**
     * @brief Get the number of hemispheres in this brain
     * @return Number of hemispheres
     */
    size_t size() const {
        return hemisphereIds.size();
    }

    /**
     * @brief Clear all hemispheres from this brain
     */
    void clear() {
        hemisphereIds.clear();
    }

    /**
     * @brief Get all hemisphere IDs in this brain
     * @return Const reference to the vector of hemisphere IDs
     */
    const std::vector<uint64_t>& getHemisphereIds() const {
        return hemisphereIds;
    }

    // Serializable interface implementation
    std::string toJson() const override {
        nlohmann::json j;
        j["type"] = "Brain";
        j["id"] = getId();
        j["name"] = name;
        j["hemisphereIds"] = hemisphereIds;

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
            if (j["type"] != "Brain") {
                SNNFW_ERROR("Invalid type in JSON: expected 'Brain', got '{}'", j["type"].get<std::string>());
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

            hemisphereIds = j["hemisphereIds"].get<std::vector<uint64_t>>();
            return true;
        } catch (const std::exception& e) {
            SNNFW_ERROR("Failed to deserialize Brain from JSON: {}", e.what());
            return false;
        }
    }

    std::string getTypeName() const override { return "Brain"; }

private:
    std::string name;
    std::vector<uint64_t> hemisphereIds;
};

} // namespace snnfw

#endif // SNNFW_BRAIN_H


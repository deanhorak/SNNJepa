#ifndef SNNFW_NUCLEUS_H
#define SNNFW_NUCLEUS_H

#include "snnfw/NeuralObject.h"
#include "snnfw/Logger.h"
#include <nlohmann/json.hpp>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace snnfw {

/**
 * @class Nucleus
 * @brief Represents a nucleus containing a collection of Column IDs
 *
 * A Nucleus is a structural component that groups multiple columns together.
 * In biological terms, a nucleus is a cluster of neurons in the central
 * nervous system (e.g., thalamic nuclei, basal ganglia nuclei).
 *
 * Note: Stores column IDs rather than column objects for memory efficiency.
 */
class Nucleus : public NeuralObject {
public:
    /**
     * @brief Construct a new Nucleus object
     * @param nucleusId Unique identifier for this nucleus
     * @param nucleusName Optional name for this nucleus
     */
    explicit Nucleus(uint64_t nucleusId = 0, const std::string& nucleusName = "")
        : NeuralObject(nucleusId), name(nucleusName) {}

    /**
     * @brief Set the name of this nucleus
     * @param nucleusName Name to set
     */
    void setName(const std::string& nucleusName) {
        name = nucleusName;
    }

    /**
     * @brief Get the name of this nucleus
     * @return Name of the nucleus
     */
    const std::string& getName() const {
        return name;
    }

    /**
     * @brief Add a column to this nucleus
     * @param columnId ID of the column to add
     */
    void addColumn(uint64_t columnId) {
        columnIds.push_back(columnId);
    }

    /**
     * @brief Get a column ID by index
     * @param index Index of the column
     * @return Column ID, or 0 if index is out of range
     */
    uint64_t getColumnId(size_t index) const {
        if (index < columnIds.size()) {
            return columnIds[index];
        }
        return 0;
    }

    /**
     * @brief Remove a column by its ID
     * @param columnId ID of the column to remove
     * @return true if column was found and removed, false otherwise
     */
    bool removeColumn(uint64_t columnId) {
        auto it = std::remove(columnIds.begin(), columnIds.end(), columnId);

        if (it != columnIds.end()) {
            columnIds.erase(it, columnIds.end());
            return true;
        }
        return false;
    }

    /**
     * @brief Get the number of columns in this nucleus
     * @return Number of columns
     */
    size_t size() const {
        return columnIds.size();
    }

    /**
     * @brief Clear all columns from this nucleus
     */
    void clear() {
        columnIds.clear();
    }

    /**
     * @brief Get all column IDs in this nucleus
     * @return Const reference to the vector of column IDs
     */
    const std::vector<uint64_t>& getColumnIds() const {
        return columnIds;
    }

    // Serializable interface implementation
    std::string toJson() const override {
        nlohmann::json j;
        j["type"] = "Nucleus";
        j["id"] = getId();
        j["name"] = name;
        j["columnIds"] = columnIds;

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
            if (j["type"] != "Nucleus") {
                SNNFW_ERROR("Invalid type in JSON: expected 'Nucleus', got '{}'", j["type"].get<std::string>());
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

            columnIds = j["columnIds"].get<std::vector<uint64_t>>();
            return true;
        } catch (const std::exception& e) {
            SNNFW_ERROR("Failed to deserialize Nucleus from JSON: {}", e.what());
            return false;
        }
    }

    std::string getTypeName() const override { return "Nucleus"; }

private:
    std::string name;
    std::vector<uint64_t> columnIds;
};

} // namespace snnfw

#endif // SNNFW_NUCLEUS_H


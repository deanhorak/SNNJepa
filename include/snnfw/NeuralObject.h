#ifndef SNNFW_NEURAL_OBJECT_H
#define SNNFW_NEURAL_OBJECT_H

#include "snnfw/Serializable.h"
#include <cstdint>
#include <cmath>

namespace snnfw {

/**
 * @brief 3D position for spatial layout
 *
 * This structure represents a 3D position in space, used for storing
 * the physical location of neural objects. Positions can be imported
 * from anatomically realistic models (e.g., NEURON, NEST) or computed
 * by layout algorithms.
 */
struct Position3D {
    float x, y, z;

    Position3D() : x(0.0f), y(0.0f), z(0.0f) {}
    Position3D(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    /**
     * @brief Calculate Euclidean distance to another position
     * @param other The other position
     * @return Distance in 3D space
     */
    float distanceTo(const Position3D& other) const {
        float dx = x - other.x;
        float dy = y - other.y;
        float dz = z - other.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }

    /**
     * @brief Add two positions (vector addition)
     */
    Position3D operator+(const Position3D& other) const {
        return Position3D(x + other.x, y + other.y, z + other.z);
    }

    /**
     * @brief Subtract two positions (vector subtraction)
     */
    Position3D operator-(const Position3D& other) const {
        return Position3D(x - other.x, y - other.y, z - other.z);
    }

    /**
     * @brief Multiply position by scalar
     */
    Position3D operator*(float scalar) const {
        return Position3D(x * scalar, y * scalar, z * scalar);
    }

    /**
     * @brief Check if two positions are equal (within epsilon)
     */
    bool operator==(const Position3D& other) const {
        const float epsilon = 1e-6f;
        return std::abs(x - other.x) < epsilon &&
               std::abs(y - other.y) < epsilon &&
               std::abs(z - other.z) < epsilon;
    }

    /**
     * @brief Check if two positions are not equal
     */
    bool operator!=(const Position3D& other) const {
        return !(*this == other);
    }
};

/**
 * @brief Base class for all neural objects in the framework
 *
 * This class provides a unique identifier and optional 3D position for
 * neural objects such as Neurons, Clusters, Layers, etc. The ID is a
 * 64-bit integer that uniquely identifies the object within its class.
 *
 * The position field allows storing physical 3D coordinates, which can be:
 * - Imported from anatomically realistic models (NEURON, NEST, etc.)
 * - Computed by layout algorithms for visualization
 * - Left unset for objects without spatial representation
 *
 * All NeuralObjects implement the Serializable interface for persistence.
 */
class NeuralObject : public Serializable {
public:
    /**
     * @brief Virtual destructor for proper cleanup of derived classes
     */
    virtual ~NeuralObject() = default;

    /**
     * @brief Get the unique identifier for this neural object
     * @return 64-bit unique identifier
     */
    uint64_t getId() const { return id; }

    /**
     * @brief Check if this object has a 3D position set
     * @return true if position has been set, false otherwise
     */
    bool hasPosition() const { return hasPosition_; }

    /**
     * @brief Get the 3D position of this object
     * @return Position3D structure with x, y, z coordinates
     * @note Check hasPosition() first to ensure position is valid
     */
    const Position3D& getPosition() const { return position_; }

    /**
     * @brief Set the 3D position of this object
     * @param pos The position to set
     */
    void setPosition(const Position3D& pos) {
        position_ = pos;
        hasPosition_ = true;
    }

    /**
     * @brief Set the 3D position of this object
     * @param x X coordinate
     * @param y Y coordinate
     * @param z Z coordinate
     */
    void setPosition(float x, float y, float z) {
        position_ = Position3D(x, y, z);
        hasPosition_ = true;
    }

    /**
     * @brief Clear the position (mark as not set)
     */
    void clearPosition() {
        hasPosition_ = false;
        position_ = Position3D();
    }

protected:
    /**
     * @brief Protected constructor - only derived classes can create instances
     * @param objectId The unique identifier for this object
     */
    explicit NeuralObject(uint64_t objectId = 0)
        : id(objectId), hasPosition_(false), position_() {}

    /**
     * @brief Set the unique identifier
     * @param objectId The unique identifier to set
     */
    void setId(uint64_t objectId) { id = objectId; }

private:
    uint64_t id;              ///< Unique identifier for this neural object
    bool hasPosition_;        ///< Flag indicating if position has been set
    Position3D position_;     ///< 3D position in space (if hasPosition_ is true)
};

} // namespace snnfw

#endif // SNNFW_NEURAL_OBJECT_H


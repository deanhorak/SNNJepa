#ifndef SNNFW_SERIALIZABLE_H
#define SNNFW_SERIALIZABLE_H

#include <string>

namespace snnfw {

/**
 * @brief Interface for objects that can be serialized to/from JSON
 *
 * All NeuralObject-derived classes should implement this interface
 * to support persistence in the Datastore.
 */
class Serializable {
public:
    virtual ~Serializable() = default;

    /**
     * @brief Serialize the object to a JSON string
     * @return JSON string representation of the object's state
     */
    virtual std::string toJson() const = 0;

    /**
     * @brief Deserialize the object from a JSON string
     * @param json JSON string containing the object's state
     * @return true if deserialization was successful, false otherwise
     */
    virtual bool fromJson(const std::string& json) = 0;

    /**
     * @brief Get the type name of the object for deserialization
     * @return String identifying the object type (e.g., "Neuron", "Axon")
     */
    virtual std::string getTypeName() const = 0;
};

} // namespace snnfw

#endif // SNNFW_SERIALIZABLE_H


#ifndef SNNFW_CLUSTER_H
#define SNNFW_CLUSTER_H

#include "snnfw/NeuralObject.h"
#include <vector>
#include <cstdint>

namespace snnfw {

/**
 * @brief Cluster class - a container for managing multiple neurons
 *
 * This class represents a cluster of neurons, providing functionality
 * to manage and organize groups of neurons within the spiking neural
 * network framework. Uses ID-based storage for memory efficiency.
 */
class Cluster : public NeuralObject {
public:
    /**
     * @brief Default constructor
     * @param clusterId Unique identifier for this cluster (default: 0)
     */
    explicit Cluster(uint64_t clusterId = 0);

    /**
     * @brief Add a neuron to the cluster by ID
     * @param neuronId ID of the neuron to add
     */
    void addNeuron(uint64_t neuronId);

    /**
     * @brief Get the number of neurons in the cluster
     * @return Number of neurons
     */
    size_t size() const { return neuronIds.size(); }

    /**
     * @brief Get a neuron ID by index
     * @param index Index of the neuron (0-based)
     * @return Neuron ID, or 0 if index is out of bounds
     */
    uint64_t getNeuronId(size_t index) const;

    /**
     * @brief Get all neuron IDs in the cluster
     * @return Const reference to the vector of neuron IDs
     */
    const std::vector<uint64_t>& getNeuronIds() const { return neuronIds; }

    /**
     * @brief Remove a neuron by ID
     * @param neuronId ID of the neuron to remove
     * @return true if neuron was removed, false if not found
     */
    bool removeNeuron(uint64_t neuronId);

    /**
     * @brief Clear all neurons from the cluster
     */
    void clear();

    /**
     * @brief Print information about the cluster
     */
    void printInfo() const;

    // Serializable interface implementation
    std::string toJson() const override;
    bool fromJson(const std::string& json) override;
    std::string getTypeName() const override { return "Cluster"; }

private:
    std::vector<uint64_t> neuronIds;  ///< Vector of neuron IDs in this cluster
};

} // namespace snnfw

#endif // SNNFW_CLUSTER_H


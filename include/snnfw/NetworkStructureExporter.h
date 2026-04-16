#ifndef SNNFW_NETWORK_STRUCTURE_EXPORTER_H
#define SNNFW_NETWORK_STRUCTURE_EXPORTER_H

#include <string>
#include <vector>
#include <cstdint>
#include <fstream>
#include "snnfw/NeuralObject.h"

namespace snnfw {

/**
 * @brief Network structure data for visualization playback
 * 
 * This file format (.snnw) stores the complete network structure including
 * neuron positions, synapse connections, and hierarchical organization.
 * It's designed to be loaded alongside spike recordings (.snnr) to enable
 * full 3D visualization during playback.
 */

/**
 * @brief Neuron structure data
 */
struct ExportedNeuron {
    uint64_t id;                    ///< Neuron ID
    Position3D position;            ///< 3D position
    float radius;                   ///< Visual radius
    float r, g, b, a;              ///< Color (RGBA)
    bool isExcitatory;             ///< Excitatory vs inhibitory
    
    // Hierarchical context
    uint64_t clusterId;
    uint64_t layerId;
    uint64_t columnId;
    uint64_t nucleusId;
    uint64_t regionId;
    uint64_t lobeId;
    uint64_t hemisphereId;
    uint64_t brainId;
};

/**
 * @brief Synapse structure data
 */
struct ExportedSynapse {
    uint64_t id;                    ///< Synapse ID
    uint64_t sourceNeuronId;        ///< Source neuron ID
    uint64_t targetNeuronId;        ///< Target neuron ID
    float weight;                   ///< Synaptic weight
    float thickness;                ///< Visual thickness
    float r, g, b, a;              ///< Color (RGBA)
};

/**
 * @brief Layer metadata
 */
struct ExportedLayer {
    uint64_t id;                    ///< Layer ID
    std::string name;               ///< Layer name
    float r, g, b;                  ///< Layer color
};

/**
 * @brief Network structure metadata
 */
struct NetworkStructureMetadata {
    std::string name;               ///< Network name
    std::string timestamp;          ///< ISO 8601 timestamp
    size_t neuronCount;             ///< Total neurons
    size_t synapseCount;            ///< Total synapses
    size_t layerCount;              ///< Total layers
};

/**
 * @class NetworkStructureExporter
 * @brief Exports and imports network structure for visualization playback
 * 
 * File format (.snnw):
 * - Magic number: "SNNW" (4 bytes)
 * - Version: uint32_t (4 bytes)
 * - Metadata length: uint32_t (4 bytes)
 * - Metadata: JSON string (variable, padded to 512 bytes)
 * - Neuron count: uint64_t (8 bytes)
 * - Neurons: array of ExportedNeuron structs
 * - Synapse count: uint64_t (8 bytes)
 * - Synapses: array of ExportedSynapse structs
 * - Layer count: uint64_t (8 bytes)
 * - Layers: array of ExportedLayer structs (variable size due to strings)
 */
class NetworkStructureExporter {
public:
    NetworkStructureExporter();
    ~NetworkStructureExporter();

    /**
     * @brief Set network metadata
     */
    void setMetadata(const NetworkStructureMetadata& metadata);

    /**
     * @brief Add a neuron to export
     */
    void addNeuron(const ExportedNeuron& neuron);

    /**
     * @brief Add a synapse to export
     */
    void addSynapse(const ExportedSynapse& synapse);

    /**
     * @brief Add a layer to export
     */
    void addLayer(const ExportedLayer& layer);

    /**
     * @brief Save network structure to file
     * @param filename Output filename (.snnw)
     * @return True if successful
     */
    bool save(const std::string& filename);

    /**
     * @brief Load network structure from file
     * @param filename Input filename (.snnw)
     * @return True if successful
     */
    bool load(const std::string& filename);

    /**
     * @brief Get loaded metadata
     */
    const NetworkStructureMetadata& getMetadata() const { return metadata_; }

    /**
     * @brief Get loaded neurons
     */
    const std::vector<ExportedNeuron>& getNeurons() const { return neurons_; }

    /**
     * @brief Get loaded synapses
     */
    const std::vector<ExportedSynapse>& getSynapses() const { return synapses_; }

    /**
     * @brief Get loaded layers
     */
    const std::vector<ExportedLayer>& getLayers() const { return layers_; }

    /**
     * @brief Clear all data
     */
    void clear();

private:
    NetworkStructureMetadata metadata_;
    std::vector<ExportedNeuron> neurons_;
    std::vector<ExportedSynapse> synapses_;
    std::vector<ExportedLayer> layers_;

    static constexpr uint32_t FILE_VERSION = 1;
    static constexpr size_t METADATA_SIZE = 512;
};

} // namespace snnfw

#endif // SNNFW_NETWORK_STRUCTURE_EXPORTER_H


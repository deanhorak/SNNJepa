#ifndef SNNFW_MODEL_IMPORTER_H
#define SNNFW_MODEL_IMPORTER_H

#include "snnfw/NeuralObject.h"
#include "snnfw/Datastore.h"
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace snnfw {

/**
 * @brief Import result containing statistics and mapping information
 */
struct ImportResult {
    size_t neuronsImported = 0;
    size_t positionsSet = 0;
    size_t synapsesImported = 0;
    bool success = false;
    std::string errorMessage;
    
    // Mapping from external IDs to SNNFW IDs
    std::map<std::string, uint64_t> neuronIdMap;
    std::map<std::string, uint64_t> synapseIdMap;
};

/**
 * @brief Configuration for model import
 */
struct ImportConfig {
    bool importPositions = true;        ///< Import 3D positions
    bool importConnectivity = true;     ///< Import synaptic connections
    bool importWeights = true;          ///< Import synaptic weights
    bool createMissingNeurons = true;   ///< Create neurons if they don't exist
    float positionScale = 1.0f;         ///< Scale factor for positions (e.g., convert μm to mm)
    
    // Position offset (useful for aligning multiple imports)
    float offsetX = 0.0f;
    float offsetY = 0.0f;
    float offsetZ = 0.0f;
};

/**
 * @brief ModelImporter - Import neural network models from external simulators
 * 
 * This class provides utilities to import neural network models from various
 * external formats, including:
 * - NEURON simulator (.hoc, .swc morphology files)
 * - NEST simulator (.json, .dat position files)
 * - Generic CSV format (neuron_id, x, y, z)
 * - NeuroML (.nml)
 * 
 * The importer can:
 * - Import 3D positions for existing neurons
 * - Create new neurons with positions
 * - Import connectivity patterns
 * - Map external neuron IDs to SNNFW IDs
 */
class ModelImporter {
public:
    /**
     * @brief Constructor
     * @param datastore Reference to the datastore for loading/saving neurons
     */
    explicit ModelImporter(Datastore& datastore);
    
    /**
     * @brief Import positions from a CSV file
     * 
     * CSV format: neuron_id,x,y,z
     * Example:
     *   1,100.5,200.3,50.2
     *   2,105.1,198.7,51.8
     * 
     * @param filename Path to CSV file
     * @param config Import configuration
     * @return Import result with statistics
     */
    ImportResult importFromCSV(const std::string& filename, const ImportConfig& config = ImportConfig());
    
    /**
     * @brief Import positions from NEURON SWC morphology file
     * 
     * SWC format (space-separated):
     *   n T x y z R P
     * Where:
     *   n = point number (neuron ID)
     *   T = type (1=soma, 2=axon, 3=dendrite, etc.)
     *   x,y,z = position in 3D space
     *   R = radius
     *   P = parent point number
     * 
     * @param filename Path to SWC file
     * @param config Import configuration
     * @return Import result with statistics
     */
    ImportResult importFromSWC(const std::string& filename, const ImportConfig& config = ImportConfig());
    
    /**
     * @brief Import positions from NEST position file
     * 
     * NEST format (JSON):
     * {
     *   "neurons": [
     *     {"id": 1, "position": [100.5, 200.3, 50.2]},
     *     {"id": 2, "position": [105.1, 198.7, 51.8]}
     *   ]
     * }
     * 
     * @param filename Path to NEST JSON file
     * @param config Import configuration
     * @return Import result with statistics
     */
    ImportResult importFromNEST(const std::string& filename, const ImportConfig& config = ImportConfig());
    
    /**
     * @brief Import positions from NeuroML file
     *
     * NeuroML format (XML):
     * <neuroml>
     *   <population id="pop1">
     *     <instance id="0">
     *       <location x="100.5" y="200.3" z="50.2"/>
     *     </instance>
     *   </population>
     * </neuroml>
     *
     * @param filename Path to NeuroML file
     * @param config Import configuration
     * @return Import result with statistics
     */
    ImportResult importFromNeuroML(const std::string& filename, const ImportConfig& config = ImportConfig());

    /**
     * @brief Import positions from HDF5 file
     *
     * Supports multiple HDF5 formats:
     * - SONATA format: /nodes/<population>/positions dataset (Nx3 array)
     * - NWB format: /processing/ophys/ImageSegmentation/PlaneSegmentation/voxel_mask
     * - Generic format: /positions dataset (Nx3 array) with optional /neuron_ids
     *
     * @param filename Path to HDF5 file
     * @param config Import configuration
     * @param datasetPath Optional path to positions dataset (default: auto-detect)
     * @return Import result with statistics
     */
    ImportResult importFromHDF5(const std::string& filename,
                                const ImportConfig& config = ImportConfig(),
                                const std::string& datasetPath = "");
    
    /**
     * @brief Set position for a specific neuron by ID
     * @param neuronId SNNFW neuron ID
     * @param x X coordinate
     * @param y Y coordinate
     * @param z Z coordinate
     * @return true if successful, false if neuron not found
     */
    bool setNeuronPosition(uint64_t neuronId, float x, float y, float z);
    
    /**
     * @brief Set position for a specific neuron by ID
     * @param neuronId SNNFW neuron ID
     * @param position Position3D structure
     * @return true if successful, false if neuron not found
     */
    bool setNeuronPosition(uint64_t neuronId, const Position3D& position);
    
    /**
     * @brief Get position for a specific neuron by ID
     * @param neuronId SNNFW neuron ID
     * @param position Output position (only valid if return is true)
     * @return true if neuron has position, false otherwise
     */
    bool getNeuronPosition(uint64_t neuronId, Position3D& position) const;
    
    /**
     * @brief Export positions to CSV file
     * @param filename Output CSV file path
     * @param neuronIds List of neuron IDs to export (empty = all neurons)
     * @return true if successful
     */
    bool exportToCSV(const std::string& filename, const std::vector<uint64_t>& neuronIds = {}) const;
    
    /**
     * @brief Get the last import result
     * @return Reference to the last import result
     */
    const ImportResult& getLastResult() const { return lastResult_; }

private:
    Datastore& datastore_;
    ImportResult lastResult_;
    
    /**
     * @brief Apply position offset and scale from config
     */
    Position3D transformPosition(float x, float y, float z, const ImportConfig& config) const;
    
    /**
     * @brief Parse a CSV line into tokens
     */
    std::vector<std::string> parseCSVLine(const std::string& line) const;
    
    /**
     * @brief Trim whitespace from string
     */
    std::string trim(const std::string& str) const;
};

} // namespace snnfw

#endif // SNNFW_MODEL_IMPORTER_H


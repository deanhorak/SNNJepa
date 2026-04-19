#include "snnfw/ModelImporter.h"
#include "snnfw/Neuron.h"
#include "snnfw/Logger.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <boost/property_tree/detail/rapidxml.hpp>
#include <hdf5.h>
#include <vector>
#include <cstring>

using json = nlohmann::json;

namespace snnfw {

ModelImporter::ModelImporter(Datastore& datastore)
    : datastore_(datastore) {
}

Position3D ModelImporter::transformPosition(float x, float y, float z, const ImportConfig& config) const {
    return Position3D(
        x * config.positionScale + config.offsetX,
        y * config.positionScale + config.offsetY,
        z * config.positionScale + config.offsetZ
    );
}

std::string ModelImporter::trim(const std::string& str) const {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

std::vector<std::string> ModelImporter::parseCSVLine(const std::string& line) const {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;
    
    while (std::getline(ss, token, ',')) {
        tokens.push_back(trim(token));
    }
    
    return tokens;
}

ImportResult ModelImporter::importFromCSV(const std::string& filename, const ImportConfig& config) {
    lastResult_ = ImportResult();
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        lastResult_.errorMessage = "Failed to open file: " + filename;
        SNNFW_ERROR("{}", lastResult_.errorMessage);
        return lastResult_;
    }
    
    std::string line;
    size_t lineNumber = 0;
    
    // Skip header if present
    if (std::getline(file, line)) {
        lineNumber++;
        // Check if first line is a header (contains non-numeric characters)
        if (line.find("neuron") != std::string::npos || 
            line.find("id") != std::string::npos ||
            line.find("x") != std::string::npos) {
            // Skip header line
        } else {
            // First line is data, process it
            file.seekg(0);
            lineNumber = 0;
        }
    }
    
    while (std::getline(file, line)) {
        lineNumber++;
        
        // Skip empty lines and comments
        std::string trimmedLine = trim(line);
        if (trimmedLine.empty() || trimmedLine[0] == '#') {
            continue;
        }
        
        auto tokens = parseCSVLine(trimmedLine);
        if (tokens.size() < 4) {
            SNNFW_WARN("Line {}: Expected at least 4 columns (id,x,y,z), got {}", lineNumber, tokens.size());
            continue;
        }
        
        try {
            // Parse neuron ID and position
            uint64_t neuronId = std::stoull(tokens[0]);
            float x = std::stof(tokens[1]);
            float y = std::stof(tokens[2]);
            float z = std::stof(tokens[3]);
            
            // Transform position
            Position3D pos = transformPosition(x, y, z, config);
            
            // Set position on neuron
            if (setNeuronPosition(neuronId, pos)) {
                lastResult_.positionsSet++;
                lastResult_.neuronIdMap[tokens[0]] = neuronId;
            } else {
                SNNFW_WARN("Line {}: Neuron {} not found in datastore", lineNumber, neuronId);
            }
            
        } catch (const std::exception& e) {
            SNNFW_WARN("Line {}: Failed to parse: {}", lineNumber, e.what());
        }
    }
    
    lastResult_.neuronsImported = lastResult_.positionsSet;
    lastResult_.success = lastResult_.positionsSet > 0;
    
    if (lastResult_.success) {
        SNNFW_INFO("Imported {} neuron positions from {}", lastResult_.positionsSet, filename);
    }
    
    return lastResult_;
}

ImportResult ModelImporter::importFromSWC(const std::string& filename, const ImportConfig& config) {
    lastResult_ = ImportResult();
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        lastResult_.errorMessage = "Failed to open file: " + filename;
        SNNFW_ERROR("{}", lastResult_.errorMessage);
        return lastResult_;
    }
    
    std::string line;
    size_t lineNumber = 0;
    
    while (std::getline(file, line)) {
        lineNumber++;
        
        // Skip empty lines and comments
        std::string trimmedLine = trim(line);
        if (trimmedLine.empty() || trimmedLine[0] == '#') {
            continue;
        }
        
        std::istringstream iss(trimmedLine);
        uint64_t n, type, parent;
        float x, y, z, radius;
        
        // SWC format: n T x y z R P
        if (!(iss >> n >> type >> x >> y >> z >> radius >> parent)) {
            SNNFW_WARN("Line {}: Invalid SWC format", lineNumber);
            continue;
        }
        
        // Transform position
        Position3D pos = transformPosition(x, y, z, config);
        
        // Set position on neuron (use point number as neuron ID)
        if (setNeuronPosition(n, pos)) {
            lastResult_.positionsSet++;
            lastResult_.neuronIdMap[std::to_string(n)] = n;
        } else {
            SNNFW_WARN("Line {}: Neuron {} not found in datastore", lineNumber, n);
        }
    }
    
    lastResult_.neuronsImported = lastResult_.positionsSet;
    lastResult_.success = lastResult_.positionsSet > 0;
    
    if (lastResult_.success) {
        SNNFW_INFO("Imported {} neuron positions from SWC file {}", lastResult_.positionsSet, filename);
    }
    
    return lastResult_;
}

ImportResult ModelImporter::importFromNEST(const std::string& filename, const ImportConfig& config) {
    lastResult_ = ImportResult();
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        lastResult_.errorMessage = "Failed to open file: " + filename;
        SNNFW_ERROR("{}", lastResult_.errorMessage);
        return lastResult_;
    }
    
    try {
        json j;
        file >> j;
        
        if (!j.contains("neurons")) {
            lastResult_.errorMessage = "JSON file missing 'neurons' array";
            SNNFW_ERROR("{}", lastResult_.errorMessage);
            return lastResult_;
        }
        
        for (const auto& neuronData : j["neurons"]) {
            if (!neuronData.contains("id") || !neuronData.contains("position")) {
                SNNFW_WARN("Neuron entry missing 'id' or 'position' field");
                continue;
            }
            
            uint64_t neuronId = neuronData["id"].get<uint64_t>();
            const auto& posArray = neuronData["position"];
            
            if (posArray.size() < 3) {
                SNNFW_WARN("Neuron {}: Position array has less than 3 elements", neuronId);
                continue;
            }
            
            float x = posArray[0].get<float>();
            float y = posArray[1].get<float>();
            float z = posArray[2].get<float>();
            
            // Transform position
            Position3D pos = transformPosition(x, y, z, config);
            
            // Set position on neuron
            if (setNeuronPosition(neuronId, pos)) {
                lastResult_.positionsSet++;
                lastResult_.neuronIdMap[std::to_string(neuronId)] = neuronId;
            } else {
                SNNFW_WARN("Neuron {} not found in datastore", neuronId);
            }
        }
        
        lastResult_.neuronsImported = lastResult_.positionsSet;
        lastResult_.success = lastResult_.positionsSet > 0;
        
        if (lastResult_.success) {
            SNNFW_INFO("Imported {} neuron positions from NEST file {}", lastResult_.positionsSet, filename);
        }
        
    } catch (const std::exception& e) {
        lastResult_.errorMessage = std::string("Failed to parse JSON: ") + e.what();
        SNNFW_ERROR("{}", lastResult_.errorMessage);
    }
    
    return lastResult_;
}

ImportResult ModelImporter::importFromNeuroML(const std::string& filename, const ImportConfig& config) {
    using namespace boost::property_tree::detail::rapidxml;

    lastResult_ = ImportResult();

    // Read the entire file into memory
    std::ifstream file(filename);
    if (!file.is_open()) {
        lastResult_.errorMessage = "Failed to open file: " + filename;
        SNNFW_ERROR("{}", lastResult_.errorMessage);
        return lastResult_;
    }

    std::string xmlContent((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    if (xmlContent.empty()) {
        lastResult_.errorMessage = "File is empty: " + filename;
        SNNFW_ERROR("{}", lastResult_.errorMessage);
        return lastResult_;
    }

    try {
        // RapidXML requires non-const char* (it modifies the string during parsing)
        std::vector<char> xmlData(xmlContent.begin(), xmlContent.end());
        xmlData.push_back('\0');

        xml_document<> doc;
        doc.parse<0>(&xmlData[0]);

        // Find the root <neuroml> element
        xml_node<>* root = doc.first_node("neuroml");
        if (!root) {
            lastResult_.errorMessage = "No <neuroml> root element found";
            SNNFW_ERROR("{}", lastResult_.errorMessage);
            return lastResult_;
        }

        // Look for <network> element
        xml_node<>* network = root->first_node("network");
        if (!network) {
            lastResult_.errorMessage = "No <network> element found in NeuroML file";
            SNNFW_WARN("{}", lastResult_.errorMessage);
            return lastResult_;
        }

        // Parse populations
        for (xml_node<>* population = network->first_node("population");
             population;
             population = population->next_sibling("population")) {

            xml_attribute<>* componentAttr = population->first_attribute("component");
            if (!componentAttr) continue;

            // Look for instances with positions
            for (xml_node<>* instance = population->first_node("instance");
                 instance;
                 instance = instance->next_sibling("instance")) {

                xml_attribute<>* idAttr = instance->first_attribute("id");
                if (!idAttr) continue;

                // Look for <location> element with x, y, z attributes
                xml_node<>* location = instance->first_node("location");
                if (!location) continue;

                xml_attribute<>* xAttr = location->first_attribute("x");
                xml_attribute<>* yAttr = location->first_attribute("y");
                xml_attribute<>* zAttr = location->first_attribute("z");

                if (!xAttr || !yAttr || !zAttr) continue;

                try {
                    float x = std::stof(xAttr->value());
                    float y = std::stof(yAttr->value());
                    float z = std::stof(zAttr->value());

                    // Apply transformation
                    Position3D pos = transformPosition(x, y, z, config);

                    // Try to parse neuron ID from the instance id
                    uint64_t neuronId = 0;
                    try {
                        neuronId = std::stoull(idAttr->value());
                    } catch (...) {
                        // If ID is not a number, skip this instance
                        SNNFW_WARN("Skipping instance with non-numeric ID: {}", idAttr->value());
                        continue;
                    }

                    if (setNeuronPosition(neuronId, pos)) {
                        lastResult_.positionsSet++;
                        lastResult_.neuronIdMap[idAttr->value()] = neuronId;
                    }
                } catch (const std::exception& e) {
                    SNNFW_WARN("Failed to parse position for instance {}: {}",
                              idAttr->value(), e.what());
                }
            }
        }

        lastResult_.success = (lastResult_.positionsSet > 0);
        lastResult_.neuronsImported = lastResult_.positionsSet;

        if (lastResult_.success) {
            SNNFW_INFO("Imported {} neuron positions from NeuroML file: {}",
                      lastResult_.positionsSet, filename);
        } else {
            lastResult_.errorMessage = "No positions found in NeuroML file";
            SNNFW_WARN("{}", lastResult_.errorMessage);
        }

    } catch (const parse_error& e) {
        lastResult_.errorMessage = std::string("XML parse error: ") + e.what();
        SNNFW_ERROR("{}", lastResult_.errorMessage);
    } catch (const std::exception& e) {
        lastResult_.errorMessage = std::string("Error parsing NeuroML: ") + e.what();
        SNNFW_ERROR("{}", lastResult_.errorMessage);
    }

    return lastResult_;
}

ImportResult ModelImporter::importFromHDF5(const std::string& filename,
                                          const ImportConfig& config,
                                          const std::string& datasetPath) {
    lastResult_ = ImportResult();

    // Open HDF5 file
    hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        lastResult_.errorMessage = "Failed to open HDF5 file: " + filename;
        SNNFW_ERROR("{}", lastResult_.errorMessage);
        return lastResult_;
    }

    // List of dataset paths to try (in order of preference)
    std::vector<std::string> pathsToTry;

    if (!datasetPath.empty()) {
        // User specified a path
        pathsToTry.push_back(datasetPath);
    } else {
        // Auto-detect common formats
        pathsToTry.push_back("/positions");                                    // Generic
        pathsToTry.push_back("/nodes/default/positions");                      // SONATA
        pathsToTry.push_back("/nodes/positions");                              // SONATA variant
        pathsToTry.push_back("/processing/ophys/ImageSegmentation/PlaneSegmentation/voxel_mask"); // NWB
    }

    hid_t dataset_id = -1;
    std::string foundPath;

    // Try to open one of the dataset paths
    for (const auto& path : pathsToTry) {
        H5E_BEGIN_TRY {
            dataset_id = H5Dopen(file_id, path.c_str(), H5P_DEFAULT);
        } H5E_END_TRY;

        if (dataset_id >= 0) {
            foundPath = path;
            SNNFW_INFO("Found positions dataset at: {}", path);
            break;
        }
    }

    if (dataset_id < 0) {
        H5Fclose(file_id);
        lastResult_.errorMessage = "No positions dataset found in HDF5 file. Tried: " +
                                   std::to_string(pathsToTry.size()) + " paths";
        SNNFW_ERROR("{}", lastResult_.errorMessage);
        return lastResult_;
    }

    // Get dataspace and dimensions
    hid_t dataspace_id = H5Dget_space(dataset_id);
    int ndims = H5Sget_simple_extent_ndims(dataspace_id);

    if (ndims != 2) {
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        lastResult_.errorMessage = "Positions dataset must be 2D (Nx3), found " +
                                   std::to_string(ndims) + "D";
        SNNFW_ERROR("{}", lastResult_.errorMessage);
        return lastResult_;
    }

    hsize_t dims[2];
    H5Sget_simple_extent_dims(dataspace_id, dims, nullptr);

    if (dims[1] != 3) {
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        lastResult_.errorMessage = "Positions dataset must have 3 columns (x,y,z), found " +
                                   std::to_string(dims[1]);
        SNNFW_ERROR("{}", lastResult_.errorMessage);
        return lastResult_;
    }

    size_t numNeurons = dims[0];
    SNNFW_INFO("Reading {} neuron positions from HDF5", numNeurons);

    // Read positions data
    std::vector<float> positions(numNeurons * 3);
    herr_t status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, positions.data());

    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);

    if (status < 0) {
        H5Fclose(file_id);
        lastResult_.errorMessage = "Failed to read positions data from HDF5";
        SNNFW_ERROR("{}", lastResult_.errorMessage);
        return lastResult_;
    }

    // Try to read neuron IDs (optional)
    std::vector<uint64_t> neuronIds;
    std::string idPath = foundPath.substr(0, foundPath.find_last_of('/')) + "/neuron_ids";

    hid_t id_dataset = -1;
    H5E_BEGIN_TRY {
        id_dataset = H5Dopen(file_id, idPath.c_str(), H5P_DEFAULT);
    } H5E_END_TRY;

    if (id_dataset >= 0) {
        neuronIds.resize(numNeurons);
        H5Dread(id_dataset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, neuronIds.data());
        H5Dclose(id_dataset);
        SNNFW_INFO("Found neuron IDs dataset");
    } else {
        // Generate sequential IDs starting from 100000000000000 (neuron ID base)
        SNNFW_INFO("No neuron IDs found, using sequential IDs");
        neuronIds.resize(numNeurons);
        for (size_t i = 0; i < numNeurons; ++i) {
            neuronIds[i] = 100000000000000ULL + i;
        }
    }

    H5Fclose(file_id);

    // Import positions
    for (size_t i = 0; i < numNeurons; ++i) {
        float x = positions[i * 3 + 0];
        float y = positions[i * 3 + 1];
        float z = positions[i * 3 + 2];

        Position3D pos = transformPosition(x, y, z, config);

        if (setNeuronPosition(neuronIds[i], pos)) {
            lastResult_.positionsSet++;
            lastResult_.neuronIdMap[std::to_string(neuronIds[i])] = neuronIds[i];
        }
    }

    lastResult_.success = (lastResult_.positionsSet > 0);
    lastResult_.neuronsImported = lastResult_.positionsSet;

    if (lastResult_.success) {
        SNNFW_INFO("Imported {} neuron positions from HDF5 file: {}",
                  lastResult_.positionsSet, filename);
    } else {
        lastResult_.errorMessage = "Failed to import any positions from HDF5";
        SNNFW_WARN("{}", lastResult_.errorMessage);
    }

    return lastResult_;
}

bool ModelImporter::setNeuronPosition(uint64_t neuronId, float x, float y, float z) {
    return setNeuronPosition(neuronId, Position3D(x, y, z));
}

bool ModelImporter::setNeuronPosition(uint64_t neuronId, const Position3D& position) {
    auto neuronObj = datastore_.get(neuronId);
    if (!neuronObj) {
        return false;
    }

    auto neuron = std::dynamic_pointer_cast<Neuron>(neuronObj);
    if (!neuron) {
        return false;
    }

    neuron->setPosition(position);
    datastore_.put(neuron);
    datastore_.markDirty(neuronId);

    return true;
}

bool ModelImporter::getNeuronPosition(uint64_t neuronId, Position3D& position) const {
    auto neuronObj = datastore_.get(neuronId);
    if (!neuronObj) {
        return false;
    }

    if (!neuronObj->hasPosition()) {
        return false;
    }

    position = neuronObj->getPosition();
    return true;
}

bool ModelImporter::exportToCSV(const std::string& filename, const std::vector<uint64_t>& neuronIds) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        SNNFW_ERROR("Failed to open file for writing: {}", filename);
        return false;
    }
    
    // Write header
    file << "neuron_id,x,y,z\n";
    
    // TODO: If neuronIds is empty, we need a way to iterate all neurons in datastore
    // For now, just export the provided IDs
    
    size_t exported = 0;
    for (uint64_t neuronId : neuronIds) {
        Position3D pos;
        if (getNeuronPosition(neuronId, pos)) {
            file << neuronId << "," << pos.x << "," << pos.y << "," << pos.z << "\n";
            exported++;
        }
    }
    
    SNNFW_INFO("Exported {} neuron positions to {}", exported, filename);
    return true;
}

} // namespace snnfw


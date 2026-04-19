#include "snnfw/NetworkStructureExporter.h"
#include <iostream>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace snnfw {

NetworkStructureExporter::NetworkStructureExporter() {
    clear();
}

NetworkStructureExporter::~NetworkStructureExporter() {
}

void NetworkStructureExporter::setMetadata(const NetworkStructureMetadata& metadata) {
    metadata_ = metadata;
}

void NetworkStructureExporter::addNeuron(const ExportedNeuron& neuron) {
    neurons_.push_back(neuron);
}

void NetworkStructureExporter::addSynapse(const ExportedSynapse& synapse) {
    synapses_.push_back(synapse);
}

void NetworkStructureExporter::addLayer(const ExportedLayer& layer) {
    layers_.push_back(layer);
}

void NetworkStructureExporter::clear() {
    metadata_ = NetworkStructureMetadata();
    neurons_.clear();
    synapses_.clear();
    layers_.clear();
}

bool NetworkStructureExporter::save(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return false;
    }

    // Write magic number
    const char magic[4] = {'S', 'N', 'N', 'W'};
    file.write(magic, 4);

    // Write version
    uint32_t version = FILE_VERSION;
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // Create metadata JSON
    json metadataJson;
    metadataJson["name"] = metadata_.name;
    metadataJson["timestamp"] = metadata_.timestamp;
    metadataJson["neuronCount"] = neurons_.size();
    metadataJson["synapseCount"] = synapses_.size();
    metadataJson["layerCount"] = layers_.size();

    std::string metadataStr = metadataJson.dump();
    
    // Write metadata length
    uint32_t metadataLength = static_cast<uint32_t>(metadataStr.length());
    file.write(reinterpret_cast<const char*>(&metadataLength), sizeof(metadataLength));

    // Write metadata (padded to METADATA_SIZE)
    char metadataBuffer[METADATA_SIZE];
    std::memset(metadataBuffer, ' ', METADATA_SIZE);
    std::memcpy(metadataBuffer, metadataStr.c_str(), std::min(metadataStr.length(), METADATA_SIZE));
    file.write(metadataBuffer, METADATA_SIZE);

    // Write neurons
    uint64_t neuronCount = neurons_.size();
    file.write(reinterpret_cast<const char*>(&neuronCount), sizeof(neuronCount));
    file.write(reinterpret_cast<const char*>(neurons_.data()), 
               neurons_.size() * sizeof(ExportedNeuron));

    // Write synapses
    uint64_t synapseCount = synapses_.size();
    file.write(reinterpret_cast<const char*>(&synapseCount), sizeof(synapseCount));
    file.write(reinterpret_cast<const char*>(synapses_.data()), 
               synapses_.size() * sizeof(ExportedSynapse));

    // Write layers (variable size due to strings)
    uint64_t layerCount = layers_.size();
    file.write(reinterpret_cast<const char*>(&layerCount), sizeof(layerCount));
    
    for (const auto& layer : layers_) {
        // Write layer ID
        file.write(reinterpret_cast<const char*>(&layer.id), sizeof(layer.id));
        
        // Write name length and name
        uint32_t nameLength = static_cast<uint32_t>(layer.name.length());
        file.write(reinterpret_cast<const char*>(&nameLength), sizeof(nameLength));
        file.write(layer.name.c_str(), nameLength);
        
        // Write color
        file.write(reinterpret_cast<const char*>(&layer.r), sizeof(layer.r));
        file.write(reinterpret_cast<const char*>(&layer.g), sizeof(layer.g));
        file.write(reinterpret_cast<const char*>(&layer.b), sizeof(layer.b));
    }

    file.close();
    
    std::cout << "Saved network structure to " << filename << std::endl;
    std::cout << "  Neurons: " << neurons_.size() << std::endl;
    std::cout << "  Synapses: " << synapses_.size() << std::endl;
    std::cout << "  Layers: " << layers_.size() << std::endl;
    
    return true;
}

bool NetworkStructureExporter::load(const std::string& filename) {
    clear();

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return false;
    }

    // Read and verify magic number
    char magic[4];
    file.read(magic, 4);
    if (std::memcmp(magic, "SNNW", 4) != 0) {
        std::cerr << "Invalid file format (bad magic number)" << std::endl;
        return false;
    }

    // Read version
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != FILE_VERSION) {
        std::cerr << "Unsupported file version: " << version << std::endl;
        return false;
    }

    // Read metadata
    uint32_t metadataLength;
    file.read(reinterpret_cast<char*>(&metadataLength), sizeof(metadataLength));

    char metadataBuffer[METADATA_SIZE];
    file.read(metadataBuffer, METADATA_SIZE);
    std::string metadataStr(metadataBuffer, metadataLength);

    try {
        json metadataJson = json::parse(metadataStr);
        metadata_.name = metadataJson.value("name", "");
        metadata_.timestamp = metadataJson.value("timestamp", "");
        metadata_.neuronCount = metadataJson.value("neuronCount", 0);
        metadata_.synapseCount = metadataJson.value("synapseCount", 0);
        metadata_.layerCount = metadataJson.value("layerCount", 0);
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse metadata JSON: " << e.what() << std::endl;
        return false;
    }

    // Read neurons
    uint64_t neuronCount;
    file.read(reinterpret_cast<char*>(&neuronCount), sizeof(neuronCount));
    neurons_.resize(neuronCount);
    file.read(reinterpret_cast<char*>(neurons_.data()),
              neuronCount * sizeof(ExportedNeuron));

    // Read synapses
    uint64_t synapseCount;
    file.read(reinterpret_cast<char*>(&synapseCount), sizeof(synapseCount));
    synapses_.resize(synapseCount);
    file.read(reinterpret_cast<char*>(synapses_.data()),
              synapseCount * sizeof(ExportedSynapse));

    // Read layers
    uint64_t layerCount;
    file.read(reinterpret_cast<char*>(&layerCount), sizeof(layerCount));

    for (uint64_t i = 0; i < layerCount; ++i) {
        ExportedLayer layer;

        // Read layer ID
        file.read(reinterpret_cast<char*>(&layer.id), sizeof(layer.id));

        // Read name
        uint32_t nameLength;
        file.read(reinterpret_cast<char*>(&nameLength), sizeof(nameLength));
        layer.name.resize(nameLength);
        file.read(&layer.name[0], nameLength);

        // Read color
        file.read(reinterpret_cast<char*>(&layer.r), sizeof(layer.r));
        file.read(reinterpret_cast<char*>(&layer.g), sizeof(layer.g));
        file.read(reinterpret_cast<char*>(&layer.b), sizeof(layer.b));

        layers_.push_back(layer);
    }

    file.close();

    std::cout << "Loaded network structure from " << filename << std::endl;
    std::cout << "  Neurons: " << neurons_.size() << std::endl;
    std::cout << "  Synapses: " << synapses_.size() << std::endl;
    std::cout << "  Layers: " << layers_.size() << std::endl;

    return true;
}

} // namespace snnfw


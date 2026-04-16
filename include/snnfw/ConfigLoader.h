#ifndef SNNFW_CONFIG_LOADER_H
#define SNNFW_CONFIG_LOADER_H

#include <string>
#include <iostream>
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
#include <vector>
#include "snnfw/Logger.h"
#include "snnfw/adapters/BaseAdapter.h"

namespace snnfw {

using json = nlohmann::json;

/**
 * @brief Configuration loader for experiment hyperparameters
 * 
 * This class loads configuration from JSON files, providing type-safe access
 * to hyperparameters for neural network experiments.
 * 
 * Example configuration file (mnist_config.json):
 * {
 *   "experiment": {
 *     "name": "mnist_optimized",
 *     "description": "MNIST digit recognition with k-NN classification"
 *   },
 *   "network": {
 *     "grid_size": 7,
 *     "region_size": 4,
 *     "num_orientations": 8,
 *     "neurons_per_feature": 1,
 *     "temporal_window_ms": 200.0,
 *     "edge_threshold": 0.15
 *   },
 *   "neuron": {
 *     "window_size_ms": 200.0,
 *     "similarity_threshold": 0.7,
 *     "max_patterns": 100
 *   },
 *   "training": {
 *     "examples_per_digit": 5000,
 *     "test_images": 10000
 *   },
 *   "classification": {
 *     "method": "knn",
 *     "k_neighbors": 5
 *   },
 *   "data": {
 *     "train_images": "data/train-images-idx3-ubyte",
 *     "train_labels": "data/train-labels-idx1-ubyte",
 *     "test_images": "data/t10k-images-idx3-ubyte",
 *     "test_labels": "data/t10k-labels-idx1-ubyte"
 *   },
 *   "sonata": {
 *     "network_file": "configs/mnist_network.h5",
 *     "use_sonata": true
 *   }
 * }
 */
class ConfigLoader {
public:
    /**
     * @brief Load configuration from JSON file
     * @param configPath Path to JSON configuration file
     * @throws std::runtime_error if file cannot be loaded or parsed
     */
    explicit ConfigLoader(const std::string& configPath) {
        std::ifstream file(configPath);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open config file: " + configPath);
        }
        
        try {
            file >> config_;
            SNNFW_INFO("Loaded configuration from: {}", configPath);
        } catch (const json::exception& e) {
            throw std::runtime_error("Failed to parse config file: " + std::string(e.what()));
        }
    }
    
    /**
     * @brief Get a value from the configuration
     * @tparam T Type of the value to retrieve
     * @param path JSON pointer path (e.g., "/network/grid_size")
     * @param defaultValue Default value if path doesn't exist
     * @return The value at the path, or defaultValue if not found
     */
    template<typename T>
    T get(const std::string& path, const T& defaultValue = T{}) const {
        try {
            return config_.at(json::json_pointer(path)).get<T>();
        } catch (const json::exception&) {
            SNNFW_WARN("Config path '{}' not found, using default value", path);
            return defaultValue;
        }
    }
    
    /**
     * @brief Get a required value from the configuration
     * @tparam T Type of the value to retrieve
     * @param path JSON pointer path (e.g., "/network/grid_size")
     * @return The value at the path
     * @throws std::runtime_error if path doesn't exist
     */
    template<typename T>
    T getRequired(const std::string& path) const {
        try {
            return config_.at(json::json_pointer(path)).get<T>();
        } catch (const json::exception& e) {
            throw std::runtime_error("Required config path '" + path + "' not found: " + e.what());
        }
    }
    
    /**
     * @brief Check if a path exists in the configuration
     * @param path JSON pointer path
     * @return true if path exists, false otherwise
     */
    bool has(const std::string& path) const {
        try {
            config_.at(json::json_pointer(path));
            return true;
        } catch (const json::exception&) {
            return false;
        }
    }
    
    /**
     * @brief Get the raw JSON object
     * @return Reference to the JSON configuration
     */
    const json& getJson() const {
        return config_;
    }
    
    /**
     * @brief Get a section of the configuration as JSON
     * @param path JSON pointer path to section
     * @return JSON object at the path
     * @throws std::runtime_error if path doesn't exist
     */
    json getSection(const std::string& path) const {
        try {
            return config_.at(json::json_pointer(path));
        } catch (const json::exception& e) {
            throw std::runtime_error("Config section '" + path + "' not found: " + e.what());
        }
    }
    
    /**
     * @brief Print the configuration to stdout
     */
    void print() const {
        std::cout << "=== Configuration ===" << std::endl;
        std::cout << config_.dump(2) << std::endl;
    }
    
    /**
     * @brief Save configuration to file
     * @param outputPath Path to save the configuration
     * @return true if successful, false otherwise
     */
    bool save(const std::string& outputPath) const {
        try {
            std::ofstream file(outputPath);
            if (!file.is_open()) {
                SNNFW_ERROR("Failed to open output file: {}", outputPath);
                return false;
            }
            file << config_.dump(2);
            SNNFW_INFO("Saved configuration to: {}", outputPath);
            return true;
        } catch (const std::exception& e) {
            SNNFW_ERROR("Failed to save configuration: {}", e.what());
            return false;
        }
    }

    /**
     * @brief Create adapter configuration from JSON section
     * @param adapterName Name of the adapter in the config (e.g., "retina", "audio")
     * @return BaseAdapter::Config structure populated from JSON
     * @throws std::runtime_error if adapter section doesn't exist
     */
    adapters::BaseAdapter::Config getAdapterConfig(const std::string& adapterName) const {
        std::string adapterPath = "/adapters/" + adapterName;

        if (!has(adapterPath)) {
            throw std::runtime_error("Adapter configuration '" + adapterName + "' not found");
        }

        json adapterJson = getSection(adapterPath);

        adapters::BaseAdapter::Config config;
        config.name = adapterName;
        config.type = adapterJson.value("type", adapterName);
        config.temporalWindow = adapterJson.value("temporal_window_ms", 200.0);

        // Extract parameters by type
        if (adapterJson.contains("parameters")) {
            json params = adapterJson["parameters"];

            for (auto& [key, value] : params.items()) {
                if (value.is_number_float()) {
                    config.doubleParams[key] = value.get<double>();
                } else if (value.is_number_integer()) {
                    config.intParams[key] = value.get<int>();
                } else if (value.is_string()) {
                    config.stringParams[key] = value.get<std::string>();
                }
            }
        }

        return config;
    }

    /**
     * @brief Get list of all configured adapters
     * @return Vector of adapter names
     */
    std::vector<std::string> getAdapterNames() const {
        std::vector<std::string> names;

        if (!has("/adapters")) {
            return names;
        }

        json adaptersJson = getSection("/adapters");
        for (auto& [key, value] : adaptersJson.items()) {
            names.push_back(key);
        }

        return names;
    }

    /**
     * @brief Check if adapters are configured
     * @return true if adapters section exists, false otherwise
     */
    bool hasAdapters() const {
        return has("/adapters");
    }

private:
    json config_;  ///< The loaded configuration
};

} // namespace snnfw

#endif // SNNFW_CONFIG_LOADER_H


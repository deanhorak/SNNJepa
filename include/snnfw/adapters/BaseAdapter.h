#ifndef SNNFW_BASE_ADAPTER_H
#define SNNFW_BASE_ADAPTER_H

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <functional>
#include "snnfw/Neuron.h"

namespace snnfw {
namespace adapters {

/**
 * @brief Base class for all input/output adapters
 *
 * Adapters provide a standardized interface for connecting the spiking neural network
 * to external data sources (sensory input) and actuators (motor output). They handle:
 * - Data format conversion (images, audio, video, sensor data, etc.)
 * - Spike encoding (converting external data to spike trains)
 * - Spike decoding (converting spike trains to external actions)
 * - Temporal synchronization
 * - Configuration management
 *
 * Design Philosophy:
 * - Adapters are the boundary between the SNN and the external world
 * - They encapsulate domain-specific processing (e.g., edge detection for vision)
 * - They provide a clean separation of concerns
 * - They enable reusable, composable components
 *
 * Example Adapters:
 * - RetinaAdapter: Converts images to spike trains (edge detection, rate coding)
 * - AudioAdapter: Converts audio to spike trains (frequency decomposition)
 * - VideoStreamAdapter: Processes video frames in real-time
 * - ServoAdapter: Converts spike trains to motor commands
 * - DisplayAdapter: Visualizes network activity
 */
class BaseAdapter {
public:
    /**
     * @brief Adapter configuration parameters
     */
    struct Config {
        std::string name;                           ///< Adapter instance name
        std::string type;                           ///< Adapter type (e.g., "retina", "audio")
        double temporalWindow;                      ///< Temporal window in milliseconds
        std::map<std::string, double> doubleParams; ///< Double parameters
        std::map<std::string, int> intParams;       ///< Integer parameters
        std::map<std::string, std::string> stringParams; ///< String parameters

        // Helper methods for accessing parameters
        double getDoubleParam(const std::string& key, double defaultValue = 0.0) const {
            auto it = doubleParams.find(key);
            return (it != doubleParams.end()) ? it->second : defaultValue;
        }

        int getIntParam(const std::string& key, int defaultValue = 0) const {
            auto it = intParams.find(key);
            return (it != intParams.end()) ? it->second : defaultValue;
        }

        std::string getStringParam(const std::string& key, const std::string& defaultValue = "") const {
            auto it = stringParams.find(key);
            return (it != stringParams.end()) ? it->second : defaultValue;
        }
    };

    /**
     * @brief Constructor
     * @param config Adapter configuration
     */
    explicit BaseAdapter(const Config& config) 
        : config_(config), initialized_(false) {}

    /**
     * @brief Virtual destructor
     */
    virtual ~BaseAdapter() = default;

    /**
     * @brief Initialize the adapter
     * @return true if successful, false otherwise
     */
    virtual bool initialize() {
        initialized_ = true;
        return true;
    }

    /**
     * @brief Shutdown the adapter and release resources
     */
    virtual void shutdown() {
        initialized_ = false;
    }

    /**
     * @brief Check if adapter is initialized
     */
    bool isInitialized() const { return initialized_; }

    /**
     * @brief Get adapter name
     */
    const std::string& getName() const { return config_.name; }

    /**
     * @brief Get adapter type
     */
    const std::string& getType() const { return config_.type; }

    /**
     * @brief Get adapter configuration
     */
    const Config& getConfig() const { return config_; }

    /**
     * @brief Update configuration parameter
     * @param key Parameter name
     * @param value Parameter value
     */
    void setDoubleParam(const std::string& key, double value) {
        config_.doubleParams[key] = value;
    }

    void setIntParam(const std::string& key, int value) {
        config_.intParams[key] = value;
    }

    void setStringParam(const std::string& key, const std::string& value) {
        config_.stringParams[key] = value;
    }

    /**
     * @brief Get configuration parameter
     */
    double getDoubleParam(const std::string& key, double defaultValue = 0.0) const {
        auto it = config_.doubleParams.find(key);
        return (it != config_.doubleParams.end()) ? it->second : defaultValue;
    }

    int getIntParam(const std::string& key, int defaultValue = 0) const {
        auto it = config_.intParams.find(key);
        return (it != config_.intParams.end()) ? it->second : defaultValue;
    }

    std::string getStringParam(const std::string& key, const std::string& defaultValue = "") const {
        auto it = config_.stringParams.find(key);
        return (it != config_.stringParams.end()) ? it->second : defaultValue;
    }

    /**
     * @brief Reset adapter state
     */
    virtual void reset() = 0;

    /**
     * @brief Get adapter statistics
     * @return Map of statistic name to value
     */
    virtual std::map<std::string, double> getStatistics() const {
        return std::map<std::string, double>();
    }

protected:
    Config config_;        ///< Adapter configuration
    bool initialized_;     ///< Initialization state
};

} // namespace adapters
} // namespace snnfw

#endif // SNNFW_BASE_ADAPTER_H


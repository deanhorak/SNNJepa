#ifndef SNNFW_ADAPTER_FACTORY_H
#define SNNFW_ADAPTER_FACTORY_H

#include "snnfw/adapters/BaseAdapter.h"
#include "snnfw/adapters/SensoryAdapter.h"
#include "snnfw/adapters/MotorAdapter.h"
#include <memory>
#include <string>
#include <map>
#include <functional>

namespace snnfw {
namespace adapters {

/**
 * @brief Factory for creating and managing adapters
 *
 * The AdapterFactory provides:
 * - Registration of adapter types
 * - Dynamic adapter creation from configuration
 * - Adapter lifecycle management
 * - Type-safe adapter retrieval
 *
 * Usage:
 * @code
 * // Register adapter types
 * AdapterFactory factory;
 * factory.registerSensoryAdapter("retina", 
 *     [](const BaseAdapter::Config& cfg) { 
 *         return std::make_shared<RetinaAdapter>(cfg); 
 *     });
 *
 * // Create adapter from configuration
 * BaseAdapter::Config config;
 * config.name = "left_eye";
 * config.type = "retina";
 * auto adapter = factory.createSensoryAdapter(config);
 * @endcode
 */
class AdapterFactory {
public:
    /**
     * @brief Factory function type for sensory adapters
     */
    using SensoryAdapterCreator = std::function<std::shared_ptr<SensoryAdapter>(const BaseAdapter::Config&)>;

    /**
     * @brief Factory function type for motor adapters
     */
    using MotorAdapterCreator = std::function<std::shared_ptr<MotorAdapter>(const BaseAdapter::Config&)>;

    /**
     * @brief Constructor
     */
    AdapterFactory() = default;

    /**
     * @brief Register a sensory adapter type
     * @param type Adapter type name (e.g., "retina", "audio")
     * @param creator Factory function to create the adapter
     */
    void registerSensoryAdapter(const std::string& type, SensoryAdapterCreator creator) {
        sensoryCreators_[type] = creator;
    }

    /**
     * @brief Register a motor adapter type
     * @param type Adapter type name (e.g., "servo", "display")
     * @param creator Factory function to create the adapter
     */
    void registerMotorAdapter(const std::string& type, MotorAdapterCreator creator) {
        motorCreators_[type] = creator;
    }

    /**
     * @brief Create a sensory adapter
     * @param config Adapter configuration
     * @return Shared pointer to the created adapter, or nullptr if type not found
     */
    std::shared_ptr<SensoryAdapter> createSensoryAdapter(const BaseAdapter::Config& config) {
        auto it = sensoryCreators_.find(config.type);
        if (it == sensoryCreators_.end()) {
            return nullptr;
        }
        return it->second(config);
    }

    /**
     * @brief Create a motor adapter
     * @param config Adapter configuration
     * @return Shared pointer to the created adapter, or nullptr if type not found
     */
    std::shared_ptr<MotorAdapter> createMotorAdapter(const BaseAdapter::Config& config) {
        auto it = motorCreators_.find(config.type);
        if (it == motorCreators_.end()) {
            return nullptr;
        }
        return it->second(config);
    }

    /**
     * @brief Check if a sensory adapter type is registered
     * @param type Adapter type name
     * @return true if registered, false otherwise
     */
    bool hasSensoryAdapter(const std::string& type) const {
        return sensoryCreators_.find(type) != sensoryCreators_.end();
    }

    /**
     * @brief Check if a motor adapter type is registered
     * @param type Adapter type name
     * @return true if registered, false otherwise
     */
    bool hasMotorAdapter(const std::string& type) const {
        return motorCreators_.find(type) != motorCreators_.end();
    }

    /**
     * @brief Get list of registered sensory adapter types
     */
    std::vector<std::string> getSensoryAdapterTypes() const {
        std::vector<std::string> types;
        for (const auto& pair : sensoryCreators_) {
            types.push_back(pair.first);
        }
        return types;
    }

    /**
     * @brief Get list of registered motor adapter types
     */
    std::vector<std::string> getMotorAdapterTypes() const {
        std::vector<std::string> types;
        for (const auto& pair : motorCreators_) {
            types.push_back(pair.first);
        }
        return types;
    }

    /**
     * @brief Get the global adapter factory instance
     */
    static AdapterFactory& getInstance() {
        static AdapterFactory instance;
        return instance;
    }

private:
    std::map<std::string, SensoryAdapterCreator> sensoryCreators_;
    std::map<std::string, MotorAdapterCreator> motorCreators_;
};

/**
 * @brief Helper macro for registering sensory adapters
 * 
 * Usage:
 * @code
 * REGISTER_SENSORY_ADAPTER(RetinaAdapter, "retina");
 * @endcode
 */
#define REGISTER_SENSORY_ADAPTER(AdapterClass, typeName) \
    namespace { \
        struct AdapterClass##Registrar { \
            AdapterClass##Registrar() { \
                snnfw::adapters::AdapterFactory::getInstance().registerSensoryAdapter( \
                    typeName, \
                    [](const snnfw::adapters::BaseAdapter::Config& cfg) { \
                        return std::make_shared<AdapterClass>(cfg); \
                    } \
                ); \
            } \
        }; \
        static AdapterClass##Registrar global_##AdapterClass##Registrar; \
    }

/**
 * @brief Helper macro for registering motor adapters
 * 
 * Usage:
 * @code
 * REGISTER_MOTOR_ADAPTER(ServoAdapter, "servo");
 * @endcode
 */
#define REGISTER_MOTOR_ADAPTER(AdapterClass, typeName) \
    namespace { \
        struct AdapterClass##Registrar { \
            AdapterClass##Registrar() { \
                snnfw::adapters::AdapterFactory::getInstance().registerMotorAdapter( \
                    typeName, \
                    [](const snnfw::adapters::BaseAdapter::Config& cfg) { \
                        return std::make_shared<AdapterClass>(cfg); \
                    } \
                ); \
            } \
        }; \
        static AdapterClass##Registrar global_##AdapterClass##Registrar; \
    }

} // namespace adapters
} // namespace snnfw

#endif // SNNFW_ADAPTER_FACTORY_H


#ifndef SNNFW_NETWORK_VALIDATOR_H
#define SNNFW_NETWORK_VALIDATOR_H

#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace snnfw {

// Forward declarations
class Datastore;

/**
 * @brief Severity level for validation errors
 */
enum class ValidationSeverity {
    INFO,       // Informational message (not an error)
    WARNING,    // Potential issue but not critical
    ERROR,      // Serious issue that should be fixed
    CRITICAL    // Critical issue that will cause failures
};

/**
 * @brief Type of validation error
 */
enum class ValidationErrorType {
    // ID and existence errors
    MISSING_ID,
    INVALID_ID_RANGE,
    ORPHANED_OBJECT,
    
    // Connectivity errors
    DANGLING_REFERENCE,
    INVALID_CONNECTION,
    CIRCULAR_DEPENDENCY,
    
    // Hierarchy errors
    INVALID_HIERARCHY,
    MISSING_PARENT,
    MISSING_CHILD,
    
    // Resource limit errors
    RESOURCE_LIMIT_EXCEEDED,
    MEMORY_LIMIT_EXCEEDED,
    
    // Consistency errors
    INCONSISTENT_STATE,
    DUPLICATE_ID,
    
    // Other
    UNKNOWN_ERROR
};

/**
 * @brief Represents a single validation error or warning
 */
struct ValidationError {
    ValidationSeverity severity;
    ValidationErrorType type;
    std::string message;
    uint64_t objectId;          // ID of the object with the error (0 if not applicable)
    std::string objectType;     // Type of the object (e.g., "Neuron", "Synapse")
    std::string context;        // Additional context information
    
    ValidationError(ValidationSeverity sev, ValidationErrorType t, const std::string& msg,
                   uint64_t id = 0, const std::string& objType = "", const std::string& ctx = "")
        : severity(sev), type(t), message(msg), objectId(id), objectType(objType), context(ctx) {}
};

/**
 * @brief Result of a validation operation
 */
struct ValidationResult {
    bool isValid;                           // Overall validation status
    std::vector<ValidationError> errors;    // All errors and warnings
    size_t criticalCount;                   // Number of critical errors
    size_t errorCount;                      // Number of errors
    size_t warningCount;                    // Number of warnings
    size_t infoCount;                       // Number of info messages
    
    ValidationResult() : isValid(true), criticalCount(0), errorCount(0), warningCount(0), infoCount(0) {}
    
    void addError(const ValidationError& error) {
        errors.push_back(error);
        
        switch (error.severity) {
            case ValidationSeverity::CRITICAL:
                criticalCount++;
                isValid = false;
                break;
            case ValidationSeverity::ERROR:
                errorCount++;
                isValid = false;
                break;
            case ValidationSeverity::WARNING:
                warningCount++;
                break;
            case ValidationSeverity::INFO:
                infoCount++;
                break;
        }
    }
    
    std::string getSummary() const;
    std::string getDetailedReport() const;
};

/**
 * @brief Configuration for validation behavior
 */
struct ValidationConfig {
    // What to validate
    bool checkIdExistence = true;
    bool checkConnectivity = true;
    bool checkHierarchy = true;
    bool checkResourceLimits = true;
    bool checkCycles = false;  // Expensive, disabled by default
    
    // Resource limits
    size_t maxSynapsesPerNeuron = 100000;
    size_t maxPatternsPerNeuron = 10000;
    size_t maxDendritesPerNeuron = 100000;
    size_t maxAxonsPerNeuron = 1;  // Typically 1 axon per neuron
    
    // Behavior
    bool stopOnFirstCritical = false;
    bool verbose = false;
};

/**
 * @brief Validates network structure and integrity
 * 
 * The NetworkValidator ensures that neural networks are structurally sound
 * and consistent. It checks for:
 * - ID existence (all referenced IDs exist in the datastore)
 * - Connectivity validity (synapses connect valid neurons)
 * - Hierarchy integrity (proper parent-child relationships)
 * - Resource limits (neurons don't exceed limits)
 * - Consistency (no duplicate IDs, orphaned objects, etc.)
 */
class NetworkValidator {
public:
    /**
     * @brief Construct a validator with default configuration
     */
    NetworkValidator();
    
    /**
     * @brief Construct a validator with custom configuration
     */
    explicit NetworkValidator(const ValidationConfig& config);
    
    /**
     * @brief Validate an entire network starting from a root object
     * 
     * @param rootId ID of the root object (typically a Brain or Hemisphere)
     * @param datastore Datastore containing the network
     * @return ValidationResult with all errors and warnings
     */
    ValidationResult validateNetwork(uint64_t rootId, Datastore& datastore);
    
    /**
     * @brief Validate a single neuron
     * 
     * @param neuronId ID of the neuron to validate
     * @param datastore Datastore containing the neuron
     * @return ValidationResult with errors specific to this neuron
     */
    ValidationResult validateNeuron(uint64_t neuronId, Datastore& datastore);
    
    /**
     * @brief Validate a single synapse
     * 
     * @param synapseId ID of the synapse to validate
     * @param datastore Datastore containing the synapse
     * @return ValidationResult with errors specific to this synapse
     */
    ValidationResult validateSynapse(uint64_t synapseId, Datastore& datastore);
    
    /**
     * @brief Validate a hierarchical structure (Brain, Hemisphere, Lobe, etc.)
     * 
     * @param structureId ID of the hierarchical structure
     * @param datastore Datastore containing the structure
     * @return ValidationResult with errors in the hierarchy
     */
    ValidationResult validateHierarchy(uint64_t structureId, Datastore& datastore);
    
    /**
     * @brief Check if an ID exists in the datastore
     * 
     * @param id ID to check
     * @param datastore Datastore to check in
     * @return true if the ID exists, false otherwise
     */
    bool checkIdExists(uint64_t id, Datastore& datastore);
    
    /**
     * @brief Check connectivity of a synapse
     * 
     * @param synapseId ID of the synapse
     * @param datastore Datastore containing the synapse
     * @param result ValidationResult to add errors to
     */
    void checkConnectivity(uint64_t synapseId, Datastore& datastore, ValidationResult& result);
    
    /**
     * @brief Check resource limits for a neuron
     * 
     * @param neuronId ID of the neuron
     * @param datastore Datastore containing the neuron
     * @param result ValidationResult to add errors to
     */
    void checkResourceLimits(uint64_t neuronId, Datastore& datastore, ValidationResult& result);
    
    /**
     * @brief Check for circular dependencies in the network
     * 
     * @param rootId ID of the root object
     * @param datastore Datastore containing the network
     * @param result ValidationResult to add errors to
     */
    void checkCycles(uint64_t rootId, Datastore& datastore, ValidationResult& result);
    
    /**
     * @brief Get the current validation configuration
     */
    const ValidationConfig& getConfig() const { return config_; }
    
    /**
     * @brief Set a new validation configuration
     */
    void setConfig(const ValidationConfig& config) { config_ = config; }
    
private:
    ValidationConfig config_;
    
    // Helper methods
    void validateHierarchyRecursive(uint64_t structureId, const std::string& structureType,
                                   Datastore& datastore, ValidationResult& result);
    
    std::string getObjectType(uint64_t id);
    bool isInIdRange(uint64_t id, const std::string& expectedType);
};

} // namespace snnfw

#endif // SNNFW_NETWORK_VALIDATOR_H


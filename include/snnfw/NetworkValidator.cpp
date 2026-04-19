#include "snnfw/NetworkValidator.h"
#include "snnfw/Datastore.h"
#include "snnfw/Neuron.h"
#include "snnfw/Synapse.h"
#include "snnfw/Axon.h"
#include "snnfw/Dendrite.h"
#include "snnfw/Cluster.h"
#include "snnfw/Layer.h"
#include "snnfw/Column.h"
#include "snnfw/Nucleus.h"
#include "snnfw/Region.h"
#include "snnfw/Lobe.h"
#include "snnfw/Hemisphere.h"
#include "snnfw/Brain.h"
#include "snnfw/Logger.h"
#include <sstream>
#include <unordered_set>
#include <algorithm>

namespace snnfw {

// ValidationResult methods
std::string ValidationResult::getSummary() const {
    std::ostringstream oss;
    oss << "Validation " << (isValid ? "PASSED" : "FAILED") << ": ";
    oss << criticalCount << " critical, ";
    oss << errorCount << " errors, ";
    oss << warningCount << " warnings, ";
    oss << infoCount << " info";
    return oss.str();
}

std::string ValidationResult::getDetailedReport() const {
    std::ostringstream oss;
    oss << "=== Validation Report ===\n";
    oss << "Status: " << (isValid ? "PASSED" : "FAILED") << "\n";
    oss << "Critical: " << criticalCount << "\n";
    oss << "Errors: " << errorCount << "\n";
    oss << "Warnings: " << warningCount << "\n";
    oss << "Info: " << infoCount << "\n";
    oss << "\n";
    
    if (!errors.empty()) {
        oss << "=== Issues ===\n";
        for (const auto& error : errors) {
            const char* severityStr = "";
            switch (error.severity) {
                case ValidationSeverity::CRITICAL: severityStr = "CRITICAL"; break;
                case ValidationSeverity::ERROR: severityStr = "ERROR"; break;
                case ValidationSeverity::WARNING: severityStr = "WARNING"; break;
                case ValidationSeverity::INFO: severityStr = "INFO"; break;
            }
            
            oss << "[" << severityStr << "] ";
            if (error.objectId != 0) {
                oss << error.objectType << " " << error.objectId << ": ";
            }
            oss << error.message;
            if (!error.context.empty()) {
                oss << " (" << error.context << ")";
            }
            oss << "\n";
        }
    }
    
    return oss.str();
}

// NetworkValidator implementation
NetworkValidator::NetworkValidator() : config_() {
    SNNFW_INFO("NetworkValidator created with default configuration");
}

NetworkValidator::NetworkValidator(const ValidationConfig& config) : config_(config) {
    SNNFW_INFO("NetworkValidator created with custom configuration");
}

ValidationResult NetworkValidator::validateNetwork(uint64_t rootId, Datastore& datastore) {
    SNNFW_INFO("Validating network starting from root ID: {}", rootId);
    
    ValidationResult result;
    
    // Check if root exists
    if (!checkIdExists(rootId, datastore)) {
        result.addError(ValidationError(
            ValidationSeverity::CRITICAL,
            ValidationErrorType::MISSING_ID,
            "Root object does not exist",
            rootId,
            "Unknown",
            "Cannot validate network without root object"
        ));
        return result;
    }
    
    // Determine root type and validate hierarchy
    std::string rootType = getObjectType(rootId);
    if (config_.checkHierarchy) {
        validateHierarchyRecursive(rootId, rootType, datastore, result);
    }
    
    SNNFW_INFO("Network validation complete: {}", result.getSummary());
    return result;
}

ValidationResult NetworkValidator::validateNeuron(uint64_t neuronId, Datastore& datastore) {
    ValidationResult result;
    
    // Check if neuron exists
    if (!checkIdExists(neuronId, datastore)) {
        result.addError(ValidationError(
            ValidationSeverity::CRITICAL,
            ValidationErrorType::MISSING_ID,
            "Neuron does not exist",
            neuronId,
            "Neuron"
        ));
        return result;
    }
    
    auto neuron = datastore.getNeuron(neuronId);
    if (!neuron) {
        result.addError(ValidationError(
            ValidationSeverity::CRITICAL,
            ValidationErrorType::MISSING_ID,
            "Failed to retrieve neuron from datastore",
            neuronId,
            "Neuron"
        ));
        return result;
    }
    
    // Check resource limits
    if (config_.checkResourceLimits) {
        checkResourceLimits(neuronId, datastore, result);
    }
    
    // Check axon exists
    uint64_t axonId = neuron->getAxonId();
    if (axonId != 0 && !checkIdExists(axonId, datastore)) {
        result.addError(ValidationError(
            ValidationSeverity::ERROR,
            ValidationErrorType::DANGLING_REFERENCE,
            "Neuron references non-existent axon",
            neuronId,
            "Neuron",
            "Axon ID: " + std::to_string(axonId)
        ));
    }
    
    // Check dendrites exist
    const auto& dendriteIds = neuron->getDendriteIds();
    for (uint64_t dendriteId : dendriteIds) {
        if (!checkIdExists(dendriteId, datastore)) {
            result.addError(ValidationError(
                ValidationSeverity::ERROR,
                ValidationErrorType::DANGLING_REFERENCE,
                "Neuron references non-existent dendrite",
                neuronId,
                "Neuron",
                "Dendrite ID: " + std::to_string(dendriteId)
            ));
        }
    }
    
    return result;
}

ValidationResult NetworkValidator::validateSynapse(uint64_t synapseId, Datastore& datastore) {
    ValidationResult result;
    
    // Check if synapse exists
    if (!checkIdExists(synapseId, datastore)) {
        result.addError(ValidationError(
            ValidationSeverity::CRITICAL,
            ValidationErrorType::MISSING_ID,
            "Synapse does not exist",
            synapseId,
            "Synapse"
        ));
        return result;
    }
    
    // Check connectivity
    if (config_.checkConnectivity) {
        checkConnectivity(synapseId, datastore, result);
    }
    
    return result;
}

ValidationResult NetworkValidator::validateHierarchy(uint64_t structureId, Datastore& datastore) {
    ValidationResult result;
    
    // Check if structure exists
    if (!checkIdExists(structureId, datastore)) {
        result.addError(ValidationError(
            ValidationSeverity::CRITICAL,
            ValidationErrorType::MISSING_ID,
            "Hierarchical structure does not exist",
            structureId,
            "Unknown"
        ));
        return result;
    }
    
    std::string structureType = getObjectType(structureId);
    validateHierarchyRecursive(structureId, structureType, datastore, result);
    
    return result;
}

bool NetworkValidator::checkIdExists(uint64_t id, Datastore& datastore) {
    if (id == 0) return false;
    
    // Try to get the object from the datastore
    auto obj = datastore.get(id);
    return obj != nullptr;
}

void NetworkValidator::checkConnectivity(uint64_t synapseId, Datastore& datastore, ValidationResult& result) {
    auto synapse = datastore.getSynapse(synapseId);
    if (!synapse) {
        result.addError(ValidationError(
            ValidationSeverity::ERROR,
            ValidationErrorType::MISSING_ID,
            "Failed to retrieve synapse from datastore",
            synapseId,
            "Synapse"
        ));
        return;
    }
    
    // Check axon exists
    uint64_t axonId = synapse->getAxonId();
    if (!checkIdExists(axonId, datastore)) {
        result.addError(ValidationError(
            ValidationSeverity::ERROR,
            ValidationErrorType::DANGLING_REFERENCE,
            "Synapse references non-existent axon",
            synapseId,
            "Synapse",
            "Axon ID: " + std::to_string(axonId)
        ));
    }
    
    // Check dendrite exists
    uint64_t dendriteId = synapse->getDendriteId();
    if (!checkIdExists(dendriteId, datastore)) {
        result.addError(ValidationError(
            ValidationSeverity::ERROR,
            ValidationErrorType::DANGLING_REFERENCE,
            "Synapse references non-existent dendrite",
            synapseId,
            "Synapse",
            "Dendrite ID: " + std::to_string(dendriteId)
        ));
    }
}

void NetworkValidator::checkResourceLimits(uint64_t neuronId, Datastore& datastore, ValidationResult& result) {
    auto neuron = datastore.getNeuron(neuronId);
    if (!neuron) return;
    
    // Check dendrite count
    size_t dendriteCount = neuron->getDendriteIds().size();
    if (dendriteCount > config_.maxDendritesPerNeuron) {
        result.addError(ValidationError(
            ValidationSeverity::WARNING,
            ValidationErrorType::RESOURCE_LIMIT_EXCEEDED,
            "Neuron exceeds maximum dendrite count",
            neuronId,
            "Neuron",
            "Count: " + std::to_string(dendriteCount) + ", Max: " + std::to_string(config_.maxDendritesPerNeuron)
        ));
    }
    
    // Check pattern count
    size_t patternCount = neuron->getReferencePatterns().size();
    if (patternCount > config_.maxPatternsPerNeuron) {
        result.addError(ValidationError(
            ValidationSeverity::WARNING,
            ValidationErrorType::RESOURCE_LIMIT_EXCEEDED,
            "Neuron exceeds maximum pattern count",
            neuronId,
            "Neuron",
            "Count: " + std::to_string(patternCount) + ", Max: " + std::to_string(config_.maxPatternsPerNeuron)
        ));
    }
}

void NetworkValidator::checkCycles(uint64_t rootId, Datastore& datastore, ValidationResult& result) {
    // TODO: Implement cycle detection using DFS
    // This is expensive and should only be run when explicitly requested
    SNNFW_TRACE("Cycle detection not yet implemented");
}

std::string NetworkValidator::getObjectType(uint64_t id) {
    // Determine type based on ID range
    if (id >= 1200000000000000ULL && id < 1300000000000000ULL) return "Brain";
    if (id >= 1100000000000000ULL && id < 1200000000000000ULL) return "Hemisphere";
    if (id >= 1000000000000000ULL && id < 1100000000000000ULL) return "Lobe";
    if (id >= 900000000000000ULL && id < 1000000000000000ULL) return "Region";
    if (id >= 800000000000000ULL && id < 900000000000000ULL) return "Nucleus";
    if (id >= 700000000000000ULL && id < 800000000000000ULL) return "Column";
    if (id >= 600000000000000ULL && id < 700000000000000ULL) return "Layer";
    if (id >= 500000000000000ULL && id < 600000000000000ULL) return "Cluster";
    if (id >= 100000000000000ULL && id < 200000000000000ULL) return "Neuron";
    if (id >= 200000000000000ULL && id < 300000000000000ULL) return "Axon";
    if (id >= 300000000000000ULL && id < 400000000000000ULL) return "Dendrite";
    if (id >= 400000000000000ULL && id < 500000000000000ULL) return "Synapse";
    return "Unknown";
}

bool NetworkValidator::isInIdRange(uint64_t id, const std::string& expectedType) {
    return getObjectType(id) == expectedType;
}

void NetworkValidator::validateHierarchyRecursive(uint64_t structureId, const std::string& structureType,
                                                  Datastore& datastore, ValidationResult& result) {
    if (config_.verbose) {
        SNNFW_TRACE("Validating {} {}", structureType, structureId);
    }

    // Validate based on type
    if (structureType == "Brain") {
        auto brain = datastore.getBrain(structureId);
        if (!brain) {
            result.addError(ValidationError(
                ValidationSeverity::ERROR,
                ValidationErrorType::MISSING_ID,
                "Failed to retrieve Brain from datastore",
                structureId,
                "Brain"
            ));
            return;
        }

        // Validate all hemispheres
        const auto& hemisphereIds = brain->getHemisphereIds();
        for (uint64_t hemId : hemisphereIds) {
            if (!checkIdExists(hemId, datastore)) {
                result.addError(ValidationError(
                    ValidationSeverity::ERROR,
                    ValidationErrorType::DANGLING_REFERENCE,
                    "Brain references non-existent hemisphere",
                    structureId,
                    "Brain",
                    "Hemisphere ID: " + std::to_string(hemId)
                ));
            } else {
                validateHierarchyRecursive(hemId, "Hemisphere", datastore, result);
            }
        }
    }
    else if (structureType == "Hemisphere") {
        auto hemisphere = datastore.getHemisphere(structureId);
        if (!hemisphere) {
            result.addError(ValidationError(
                ValidationSeverity::ERROR,
                ValidationErrorType::MISSING_ID,
                "Failed to retrieve Hemisphere from datastore",
                structureId,
                "Hemisphere"
            ));
            return;
        }

        // Validate all lobes
        const auto& lobeIds = hemisphere->getLobeIds();
        for (uint64_t lobeId : lobeIds) {
            if (!checkIdExists(lobeId, datastore)) {
                result.addError(ValidationError(
                    ValidationSeverity::ERROR,
                    ValidationErrorType::DANGLING_REFERENCE,
                    "Hemisphere references non-existent lobe",
                    structureId,
                    "Hemisphere",
                    "Lobe ID: " + std::to_string(lobeId)
                ));
            } else {
                validateHierarchyRecursive(lobeId, "Lobe", datastore, result);
            }
        }
    }
    else if (structureType == "Lobe") {
        auto lobe = datastore.getLobe(structureId);
        if (!lobe) {
            result.addError(ValidationError(
                ValidationSeverity::ERROR,
                ValidationErrorType::MISSING_ID,
                "Failed to retrieve Lobe from datastore",
                structureId,
                "Lobe"
            ));
            return;
        }

        // Validate all regions
        const auto& regionIds = lobe->getRegionIds();
        for (uint64_t regionId : regionIds) {
            if (!checkIdExists(regionId, datastore)) {
                result.addError(ValidationError(
                    ValidationSeverity::ERROR,
                    ValidationErrorType::DANGLING_REFERENCE,
                    "Lobe references non-existent region",
                    structureId,
                    "Lobe",
                    "Region ID: " + std::to_string(regionId)
                ));
            } else {
                validateHierarchyRecursive(regionId, "Region", datastore, result);
            }
        }
    }
    else if (structureType == "Region") {
        auto region = datastore.getRegion(structureId);
        if (!region) {
            result.addError(ValidationError(
                ValidationSeverity::ERROR,
                ValidationErrorType::MISSING_ID,
                "Failed to retrieve Region from datastore",
                structureId,
                "Region"
            ));
            return;
        }

        // Validate all nuclei
        const auto& nucleusIds = region->getNucleusIds();
        for (uint64_t nucleusId : nucleusIds) {
            if (!checkIdExists(nucleusId, datastore)) {
                result.addError(ValidationError(
                    ValidationSeverity::ERROR,
                    ValidationErrorType::DANGLING_REFERENCE,
                    "Region references non-existent nucleus",
                    structureId,
                    "Region",
                    "Nucleus ID: " + std::to_string(nucleusId)
                ));
            } else {
                validateHierarchyRecursive(nucleusId, "Nucleus", datastore, result);
            }
        }
    }
    else if (structureType == "Nucleus") {
        auto nucleus = datastore.getNucleus(structureId);
        if (!nucleus) {
            result.addError(ValidationError(
                ValidationSeverity::ERROR,
                ValidationErrorType::MISSING_ID,
                "Failed to retrieve Nucleus from datastore",
                structureId,
                "Nucleus"
            ));
            return;
        }

        // Validate all columns
        const auto& columnIds = nucleus->getColumnIds();
        for (uint64_t columnId : columnIds) {
            if (!checkIdExists(columnId, datastore)) {
                result.addError(ValidationError(
                    ValidationSeverity::ERROR,
                    ValidationErrorType::DANGLING_REFERENCE,
                    "Nucleus references non-existent column",
                    structureId,
                    "Nucleus",
                    "Column ID: " + std::to_string(columnId)
                ));
            } else {
                validateHierarchyRecursive(columnId, "Column", datastore, result);
            }
        }
    }
    else if (structureType == "Column") {
        auto column = datastore.getColumn(structureId);
        if (!column) {
            result.addError(ValidationError(
                ValidationSeverity::ERROR,
                ValidationErrorType::MISSING_ID,
                "Failed to retrieve Column from datastore",
                structureId,
                "Column"
            ));
            return;
        }

        // Validate all layers
        const auto& layerIds = column->getLayerIds();
        for (uint64_t layerId : layerIds) {
            if (!checkIdExists(layerId, datastore)) {
                result.addError(ValidationError(
                    ValidationSeverity::ERROR,
                    ValidationErrorType::DANGLING_REFERENCE,
                    "Column references non-existent layer",
                    structureId,
                    "Column",
                    "Layer ID: " + std::to_string(layerId)
                ));
            } else {
                validateHierarchyRecursive(layerId, "Layer", datastore, result);
            }
        }
    }
    else if (structureType == "Layer") {
        auto layer = datastore.getLayer(structureId);
        if (!layer) {
            result.addError(ValidationError(
                ValidationSeverity::ERROR,
                ValidationErrorType::MISSING_ID,
                "Failed to retrieve Layer from datastore",
                structureId,
                "Layer"
            ));
            return;
        }

        // Validate all clusters
        const auto& clusterIds = layer->getClusterIds();
        for (uint64_t clusterId : clusterIds) {
            if (!checkIdExists(clusterId, datastore)) {
                result.addError(ValidationError(
                    ValidationSeverity::ERROR,
                    ValidationErrorType::DANGLING_REFERENCE,
                    "Layer references non-existent cluster",
                    structureId,
                    "Layer",
                    "Cluster ID: " + std::to_string(clusterId)
                ));
            } else {
                validateHierarchyRecursive(clusterId, "Cluster", datastore, result);
            }
        }
    }
    else if (structureType == "Cluster") {
        auto cluster = datastore.getCluster(structureId);
        if (!cluster) {
            result.addError(ValidationError(
                ValidationSeverity::ERROR,
                ValidationErrorType::MISSING_ID,
                "Failed to retrieve Cluster from datastore",
                structureId,
                "Cluster"
            ));
            return;
        }

        // Validate all neurons
        const auto& neuronIds = cluster->getNeuronIds();
        for (uint64_t neuronId : neuronIds) {
            if (!checkIdExists(neuronId, datastore)) {
                result.addError(ValidationError(
                    ValidationSeverity::ERROR,
                    ValidationErrorType::DANGLING_REFERENCE,
                    "Cluster references non-existent neuron",
                    structureId,
                    "Cluster",
                    "Neuron ID: " + std::to_string(neuronId)
                ));
            } else if (config_.checkResourceLimits) {
                // Validate each neuron
                auto neuronResult = validateNeuron(neuronId, datastore);
                for (const auto& error : neuronResult.errors) {
                    result.addError(error);
                }
            }
        }
    }
}

} // namespace snnfw


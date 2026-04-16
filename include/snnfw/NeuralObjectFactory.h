#ifndef SNNFW_NEURAL_OBJECT_FACTORY_H
#define SNNFW_NEURAL_OBJECT_FACTORY_H

#include "snnfw/Neuron.h"
#include "snnfw/Cluster.h"
#include "snnfw/Axon.h"
#include "snnfw/Dendrite.h"
#include "snnfw/Synapse.h"
#include "snnfw/Layer.h"
#include "snnfw/Column.h"
#include "snnfw/Nucleus.h"
#include "snnfw/Region.h"
#include "snnfw/Lobe.h"
#include "snnfw/Hemisphere.h"
#include "snnfw/Brain.h"
#include <memory>
#include <mutex>
#include <stdexcept>
#include <cstdint>

namespace snnfw {

/**
 * @brief Factory class for creating NeuralObject-derived instances with managed IDs
 *
 * The NeuralObjectFactory is responsible for creating all NeuralObject-derived
 * classes and assigning unique IDs from type-specific ranges. This allows object
 * type identification based on the ID value alone.
 *
 * ID Range Assignments (100 trillion IDs per type):
 * - Neurons:     100,000,000,000,000 - 199,999,999,999,999
 * - Axons:       200,000,000,000,000 - 299,999,999,999,999
 * - Dendrites:   300,000,000,000,000 - 399,999,999,999,999
 * - Synapses:    400,000,000,000,000 - 499,999,999,999,999
 * - Clusters:    500,000,000,000,000 - 599,999,999,999,999
 * - Layers:      600,000,000,000,000 - 699,999,999,999,999
 * - Columns:     700,000,000,000,000 - 799,999,999,999,999
 * - Nuclei:      800,000,000,000,000 - 899,999,999,999,999
 * - Regions:     900,000,000,000,000 - 999,999,999,999,999
 * - Lobes:     1,000,000,000,000,000 - 1,099,999,999,999,999
 * - Hemispheres: 1,100,000,000,000,000 - 1,199,999,999,999,999
 * - Brains:    1,200,000,000,000,000 - 1,299,999,999,999,999
 *
 * The factory is thread-safe and can be used from multiple threads concurrently.
 *
 * Design Pattern: Factory Pattern with Singleton-like usage (though not enforced)
 *
 * Example Usage:
 * @code
 * NeuralObjectFactory factory;
 *
 * // Create hierarchical structure
 * auto brain = factory.createBrain();
 * auto hemisphere = factory.createHemisphere();
 * brain->addHemisphere(hemisphere->getId());
 *
 * // Create neurons and connections
 * auto neuron = factory.createNeuron(50.0, 0.95, 20);
 * auto axon = factory.createAxon(neuron->getId());
 * auto dendrite = factory.createDendrite(neuron->getId());
 * auto synapse = factory.createSynapse(axon->getId(), dendrite->getId(), 1.0, 1.0);
 * @endcode
 */
class NeuralObjectFactory {
public:
    /**
     * @brief Object type enumeration for ID range identification
     */
    enum class ObjectType {
        NEURON,
        AXON,
        DENDRITE,
        SYNAPSE,
        CLUSTER,
        LAYER,
        COLUMN,
        NUCLEUS,
        REGION,
        LOBE,
        HEMISPHERE,
        BRAIN,
        UNKNOWN
    };

    /**
     * @brief Constructor - initializes ID counters to range start values
     */
    NeuralObjectFactory();

    /**
     * @brief Create a new Neuron with auto-assigned ID
     * @param windowSizeMs Size of the rolling time window in milliseconds
     * @param similarityThreshold Threshold for pattern similarity (0.0 to 1.0)
     * @param maxReferencePatterns Maximum number of reference patterns to store
     * @return Shared pointer to the newly created Neuron
     * @throws std::runtime_error if ID range is exhausted
     */
    std::shared_ptr<Neuron> createNeuron(
        double windowSizeMs,
        double similarityThreshold,
        size_t maxReferencePatterns = 20);

    /**
     * @brief Create a new Axon with auto-assigned ID
     * @param sourceNeuronId ID of the neuron this axon belongs to
     * @return Shared pointer to the newly created Axon
     * @throws std::runtime_error if ID range is exhausted
     */
    std::shared_ptr<Axon> createAxon(uint64_t sourceNeuronId);

    /**
     * @brief Create a new Dendrite with auto-assigned ID
     * @param targetNeuronId ID of the neuron this dendrite belongs to
     * @return Shared pointer to the newly created Dendrite
     * @throws std::runtime_error if ID range is exhausted
     */
    std::shared_ptr<Dendrite> createDendrite(uint64_t targetNeuronId);

    /**
     * @brief Create a new Synapse with auto-assigned ID
     * @param axonId ID of the presynaptic axon
     * @param dendriteId ID of the postsynaptic dendrite
     * @param weight Synaptic weight (default: 1.0)
     * @param delayMs Synaptic delay in milliseconds (default: 1.0)
     * @return Shared pointer to the newly created Synapse
     * @throws std::runtime_error if ID range is exhausted
     */
    std::shared_ptr<Synapse> createSynapse(
        uint64_t axonId,
        uint64_t dendriteId,
        double weight = 1.0,
        double delayMs = 1.0);

    /**
     * @brief Create a new Cluster with auto-assigned ID
     * @return Shared pointer to the newly created Cluster
     * @throws std::runtime_error if ID range is exhausted
     */
    std::shared_ptr<Cluster> createCluster();

    /**
     * @brief Create a new Layer with auto-assigned ID
     * @return Shared pointer to the newly created Layer
     * @throws std::runtime_error if ID range is exhausted
     */
    std::shared_ptr<Layer> createLayer();

    /**
     * @brief Create a new Column with auto-assigned ID
     * @return Shared pointer to the newly created Column
     * @throws std::runtime_error if ID range is exhausted
     */
    std::shared_ptr<Column> createColumn();

    /**
     * @brief Create a new Nucleus with auto-assigned ID
     * @return Shared pointer to the newly created Nucleus
     * @throws std::runtime_error if ID range is exhausted
     */
    std::shared_ptr<Nucleus> createNucleus();

    /**
     * @brief Create a new Region with auto-assigned ID
     * @return Shared pointer to the newly created Region
     * @throws std::runtime_error if ID range is exhausted
     */
    std::shared_ptr<Region> createRegion();

    /**
     * @brief Create a new Lobe with auto-assigned ID
     * @return Shared pointer to the newly created Lobe
     * @throws std::runtime_error if ID range is exhausted
     */
    std::shared_ptr<Lobe> createLobe();

    /**
     * @brief Create a new Hemisphere with auto-assigned ID
     * @return Shared pointer to the newly created Hemisphere
     * @throws std::runtime_error if ID range is exhausted
     */
    std::shared_ptr<Hemisphere> createHemisphere();

    /**
     * @brief Create a new Brain with auto-assigned ID
     * @return Shared pointer to the newly created Brain
     * @throws std::runtime_error if ID range is exhausted
     */
    std::shared_ptr<Brain> createBrain();

    /**
     * @brief Get the object type from an ID value
     * @param id The ID to check
     * @return ObjectType enum value
     */
    static ObjectType getObjectType(uint64_t id);

    /**
     * @brief Get a human-readable string for an object type
     * @param type The ObjectType enum value
     * @return String representation of the type
     */
    static const char* getObjectTypeName(ObjectType type);

    /**
     * @brief Get the object type name from an ID value
     * @param id The ID to check
     * @return String representation of the object type
     */
    static const char* getObjectTypeName(uint64_t id);

    /**
     * @brief Reset all ID counters to their initial values
     * WARNING: This should only be used for testing or when starting a new simulation
     */
    void reset();

    /**
     * @brief Get the current ID counter for a specific object type
     * @param type The object type
     * @return Current ID counter value
     */
    uint64_t getCurrentId(ObjectType type) const;

    /**
     * @brief Get the number of objects created of a specific type
     * @param type The object type
     * @return Number of objects created
     */
    uint64_t getObjectCount(ObjectType type) const;

    // ID Range Constants (100 trillion IDs per type)
    // Human brain scale: ~86 billion neurons, ~100 trillion synapses
    static constexpr uint64_t NEURON_ID_START    = 100000000000000ULL;  // 100 trillion
    static constexpr uint64_t NEURON_ID_END      = 199999999999999ULL;  // 200 trillion - 1
    static constexpr uint64_t AXON_ID_START      = 200000000000000ULL;  // 200 trillion
    static constexpr uint64_t AXON_ID_END        = 299999999999999ULL;  // 300 trillion - 1
    static constexpr uint64_t DENDRITE_ID_START  = 300000000000000ULL;  // 300 trillion
    static constexpr uint64_t DENDRITE_ID_END    = 399999999999999ULL;  // 400 trillion - 1
    static constexpr uint64_t SYNAPSE_ID_START   = 400000000000000ULL;  // 400 trillion
    static constexpr uint64_t SYNAPSE_ID_END     = 499999999999999ULL;  // 500 trillion - 1
    static constexpr uint64_t CLUSTER_ID_START   = 500000000000000ULL;  // 500 trillion
    static constexpr uint64_t CLUSTER_ID_END     = 599999999999999ULL;  // 600 trillion - 1
    static constexpr uint64_t LAYER_ID_START     = 600000000000000ULL;  // 600 trillion
    static constexpr uint64_t LAYER_ID_END       = 699999999999999ULL;  // 700 trillion - 1
    static constexpr uint64_t COLUMN_ID_START    = 700000000000000ULL;  // 700 trillion
    static constexpr uint64_t COLUMN_ID_END      = 799999999999999ULL;  // 800 trillion - 1
    static constexpr uint64_t NUCLEUS_ID_START   = 800000000000000ULL;  // 800 trillion
    static constexpr uint64_t NUCLEUS_ID_END     = 899999999999999ULL;  // 900 trillion - 1
    static constexpr uint64_t REGION_ID_START    = 900000000000000ULL;  // 900 trillion
    static constexpr uint64_t REGION_ID_END      = 999999999999999ULL;  // 1 quadrillion - 1
    static constexpr uint64_t LOBE_ID_START      = 1000000000000000ULL; // 1 quadrillion
    static constexpr uint64_t LOBE_ID_END        = 1099999999999999ULL; // 1.1 quadrillion - 1
    static constexpr uint64_t HEMISPHERE_ID_START= 1100000000000000ULL; // 1.1 quadrillion
    static constexpr uint64_t HEMISPHERE_ID_END  = 1199999999999999ULL; // 1.2 quadrillion - 1
    static constexpr uint64_t BRAIN_ID_START     = 1200000000000000ULL; // 1.2 quadrillion
    static constexpr uint64_t BRAIN_ID_END       = 1299999999999999ULL; // 1.3 quadrillion - 1

private:
    /**
     * @brief Get the next ID for a specific object type
     * @param type The object type
     * @return Next available ID
     * @throws std::runtime_error if ID range is exhausted
     */
    uint64_t getNextId(ObjectType type);

    // ID counters for each object type (protected by mutex)
    uint64_t neuronIdCounter;
    uint64_t axonIdCounter;
    uint64_t dendriteIdCounter;
    uint64_t synapseIdCounter;
    uint64_t clusterIdCounter;
    uint64_t layerIdCounter;
    uint64_t columnIdCounter;
    uint64_t nucleusIdCounter;
    uint64_t regionIdCounter;
    uint64_t lobeIdCounter;
    uint64_t hemisphereIdCounter;
    uint64_t brainIdCounter;

    // Mutex for thread-safe ID generation
    mutable std::mutex idMutex;
};

} // namespace snnfw

#endif // SNNFW_NEURAL_OBJECT_FACTORY_H


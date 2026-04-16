#ifndef SNNFW_NETWORK_BUILDER_H
#define SNNFW_NETWORK_BUILDER_H

#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Datastore.h"
#include "snnfw/Brain.h"
#include "snnfw/Hemisphere.h"
#include "snnfw/Lobe.h"
#include "snnfw/Region.h"
#include "snnfw/Nucleus.h"
#include "snnfw/Column.h"
#include "snnfw/Layer.h"
#include "snnfw/Cluster.h"
#include "snnfw/Neuron.h"
#include "snnfw/Axon.h"
#include "snnfw/Dendrite.h"
#include "snnfw/Synapse.h"
#include "snnfw/NetworkValidator.h"
#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace snnfw {

/**
 * @class NetworkBuilder
 * @brief Fluent API for building hierarchical neural networks
 * 
 * NetworkBuilder provides a convenient fluent interface for constructing
 * complex hierarchical neural networks. It handles automatic ID management,
 * object creation, and validation.
 * 
 * Example usage:
 * @code
 * NetworkBuilder builder(factory, datastore);
 * auto brain = builder.createBrain("HumanBrain")
 *     .addHemisphere("Left")
 *         .addLobe("Occipital")
 *             .addRegion("V1")
 *                 .addNucleus("Layer4C")
 *                     .addColumns(12, "Orientation")
 *                         .addLayers(6, "Cortical")
 *                             .addClusters(10, 100)  // 10 clusters, 100 neurons each
 *     .build();
 * @endcode
 */
class NetworkBuilder {
public:
    /**
     * @brief Construct a NetworkBuilder
     * @param factory Reference to NeuralObjectFactory for creating objects
     * @param datastore Reference to Datastore for persisting objects
     * @param autoValidate Whether to automatically validate on build (default: true)
     */
    NetworkBuilder(NeuralObjectFactory& factory, 
                   Datastore& datastore,
                   bool autoValidate = true);

    /**
     * @brief Create a new brain and start building
     * @param name Name for the brain
     * @return Reference to this builder for chaining
     */
    NetworkBuilder& createBrain(const std::string& name = "");

    /**
     * @brief Add a hemisphere to the current brain
     * @param name Name for the hemisphere
     * @return Reference to this builder for chaining
     */
    NetworkBuilder& addHemisphere(const std::string& name = "");

    /**
     * @brief Add a lobe to the current hemisphere
     * @param name Name for the lobe
     * @return Reference to this builder for chaining
     */
    NetworkBuilder& addLobe(const std::string& name = "");

    /**
     * @brief Add a region to the current lobe
     * @param name Name for the region
     * @return Reference to this builder for chaining
     */
    NetworkBuilder& addRegion(const std::string& name = "");

    /**
     * @brief Add a nucleus to the current region
     * @param name Name for the nucleus
     * @return Reference to this builder for chaining
     */
    NetworkBuilder& addNucleus(const std::string& name = "");

    /**
     * @brief Add a column to the current nucleus
     * @param name Name for the column
     * @return Reference to this builder for chaining
     */
    NetworkBuilder& addColumn(const std::string& name = "");

    /**
     * @brief Add multiple columns to the current nucleus
     * @param count Number of columns to create
     * @param namePrefix Prefix for column names (will append index)
     * @return Reference to this builder for chaining
     */
    NetworkBuilder& addColumns(size_t count, const std::string& namePrefix = "Column");

    /**
     * @brief Add a layer to the current column
     * @param name Name for the layer
     * @return Reference to this builder for chaining
     */
    NetworkBuilder& addLayer(const std::string& name = "");

    /**
     * @brief Add multiple layers to the current column
     * @param count Number of layers to create
     * @param namePrefix Prefix for layer names (will append index)
     * @return Reference to this builder for chaining
     */
    NetworkBuilder& addLayers(size_t count, const std::string& namePrefix = "Layer");

    /**
     * @brief Add a cluster to the current layer
     * @param neuronCount Number of neurons in the cluster
     * @param windowSizeMs Neuron window size in milliseconds (0 = use default)
     * @param similarityThreshold Neuron similarity threshold (0 = use default)
     * @param maxReferencePatterns Maximum reference patterns per neuron (0 = use default)
     * @return Reference to this builder for chaining
     */
    NetworkBuilder& addCluster(size_t neuronCount,
                               double windowSizeMs = 0.0,
                               double similarityThreshold = 0.0,
                               size_t maxReferencePatterns = 0);

    /**
     * @brief Add multiple clusters to the current layer
     * @param clusterCount Number of clusters to create
     * @param neuronsPerCluster Number of neurons per cluster
     * @param windowSizeMs Neuron window size in milliseconds
     * @param similarityThreshold Neuron similarity threshold
     * @param maxReferencePatterns Maximum reference patterns per neuron
     * @return Reference to this builder for chaining
     */
    NetworkBuilder& addClusters(size_t clusterCount,
                                size_t neuronsPerCluster,
                                double windowSizeMs = 50.0,
                                double similarityThreshold = 0.85,
                                size_t maxReferencePatterns = 100);

    /**
     * @brief Navigate up one level in the hierarchy
     * @return Reference to this builder for chaining
     */
    NetworkBuilder& up();

    /**
     * @brief Navigate to the root (brain) level
     * @return Reference to this builder for chaining
     */
    NetworkBuilder& toRoot();

    /**
     * @brief Build and return the brain
     * @return Shared pointer to the built brain
     * @throws std::runtime_error if validation fails (when autoValidate is true)
     */
    std::shared_ptr<Brain> build();

    /**
     * @brief Get the current brain being built
     * @return Shared pointer to the brain (may be null if not created yet)
     */
    std::shared_ptr<Brain> getBrain() const { return brain_; }

    /**
     * @brief Get all neurons created during building
     * @return Vector of neuron shared pointers
     */
    const std::vector<std::shared_ptr<Neuron>>& getNeurons() const { return neurons_; }

    /**
     * @brief Get all clusters created during building
     * @return Vector of cluster shared pointers
     */
    const std::vector<std::shared_ptr<Cluster>>& getClusters() const { return clusters_; }

    /**
     * @brief Get all layers created during building
     * @return Vector of layer shared pointers
     */
    const std::vector<std::shared_ptr<Layer>>& getLayers() const { return layers_; }

    /**
     * @brief Get all columns created during building
     * @return Vector of column shared pointers
     */
    const std::vector<std::shared_ptr<Column>>& getColumns() const { return columns_; }

    /**
     * @brief Set whether to automatically persist objects to datastore
     * @param autoPersist Whether to auto-persist (default: true)
     * @return Reference to this builder for chaining
     */
    NetworkBuilder& setAutoPersist(bool autoPersist);

    /**
     * @brief Set neuron parameters for subsequent neuron creation
     * @param windowSizeMs Neuron window size in milliseconds
     * @param similarityThreshold Neuron similarity threshold
     * @param maxReferencePatterns Maximum reference patterns per neuron
     * @return Reference to this builder for chaining
     */
    NetworkBuilder& setNeuronParams(double windowSizeMs,
                                    double similarityThreshold,
                                    size_t maxReferencePatterns);

private:
    // References to factory and datastore
    NeuralObjectFactory& factory_;
    Datastore& datastore_;
    
    // Configuration
    bool autoValidate_;
    bool autoPersist_;
    
    // Default neuron parameters
    double defaultWindowSizeMs_;
    double defaultSimilarityThreshold_;
    size_t defaultMaxReferencePatterns_;
    
    // Current hierarchy context
    std::shared_ptr<Brain> brain_;
    std::shared_ptr<Hemisphere> currentHemisphere_;
    std::shared_ptr<Lobe> currentLobe_;
    std::shared_ptr<Region> currentRegion_;
    std::shared_ptr<Nucleus> currentNucleus_;
    std::shared_ptr<Column> currentColumn_;
    std::shared_ptr<Layer> currentLayer_;
    
    // Collections of created objects
    std::vector<std::shared_ptr<Neuron>> neurons_;
    std::vector<std::shared_ptr<Cluster>> clusters_;
    std::vector<std::shared_ptr<Layer>> layers_;
    std::vector<std::shared_ptr<Column>> columns_;
    std::vector<std::shared_ptr<Nucleus>> nuclei_;
    std::vector<std::shared_ptr<Region>> regions_;
    std::vector<std::shared_ptr<Lobe>> lobes_;
    std::vector<std::shared_ptr<Hemisphere>> hemispheres_;
    
    // Navigation stack for up() operation
    enum class ContextLevel {
        BRAIN,
        HEMISPHERE,
        LOBE,
        REGION,
        NUCLEUS,
        COLUMN,
        LAYER
    };
    std::vector<ContextLevel> contextStack_;
    
    // Helper methods
    void persistObject(const std::shared_ptr<NeuralObject>& obj);
    void pushContext(ContextLevel level);
    void popContext();
};

} // namespace snnfw

#endif // SNNFW_NETWORK_BUILDER_H


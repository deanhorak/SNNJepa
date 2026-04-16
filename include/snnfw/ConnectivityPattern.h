#ifndef SNNFW_CONNECTIVITY_PATTERN_H
#define SNNFW_CONNECTIVITY_PATTERN_H

#include <vector>
#include <memory>
#include <random>
#include <functional>
#include <cstdint>
#include <map>

namespace snnfw {

// Forward declarations
class Neuron;
class NeuralObjectFactory;
class Datastore;

/**
 * @struct Connection
 * @brief Represents a connection between two neurons
 */
struct Connection {
    uint64_t sourceNeuronId;  ///< ID of the presynaptic neuron
    uint64_t targetNeuronId;  ///< ID of the postsynaptic neuron
    double weight;            ///< Synaptic weight
    double delay;             ///< Synaptic delay in milliseconds
    
    Connection(uint64_t src, uint64_t tgt, double w = 1.0, double d = 1.0)
        : sourceNeuronId(src), targetNeuronId(tgt), weight(w), delay(d) {}
};

/**
 * @struct SpatialPosition
 * @brief Represents a 3D spatial position for neurons
 */
struct SpatialPosition {
    double x;
    double y;
    double z;
    
    SpatialPosition(double x_ = 0.0, double y_ = 0.0, double z_ = 0.0)
        : x(x_), y(y_), z(z_) {}
    
    /// Calculate Euclidean distance to another position
    double distanceTo(const SpatialPosition& other) const;
};

/**
 * @class ConnectivityPattern
 * @brief Base class for connectivity pattern generators
 * 
 * ConnectivityPattern provides an interface for generating connections
 * between groups of neurons according to different patterns (random,
 * distance-dependent, topographic, etc.).
 */
class ConnectivityPattern {
public:
    virtual ~ConnectivityPattern() = default;
    
    /**
     * @brief Generate connections between source and target neurons
     * @param sourceNeurons Vector of source (presynaptic) neuron IDs
     * @param targetNeurons Vector of target (postsynaptic) neuron IDs
     * @return Vector of connections to create
     */
    virtual std::vector<Connection> generateConnections(
        const std::vector<uint64_t>& sourceNeurons,
        const std::vector<uint64_t>& targetNeurons) = 0;
    
    /**
     * @brief Set random seed for reproducibility
     * @param seed Random seed value
     */
    virtual void setSeed(unsigned int seed) {
        rng_.seed(seed);
    }
    
protected:
    std::mt19937 rng_{std::random_device{}()};  ///< Random number generator
};

/**
 * @class RandomSparsePattern
 * @brief Random sparse connectivity with specified connection probability
 * 
 * Creates connections randomly between source and target neurons with
 * a given probability. Each potential connection is independently
 * considered with the specified probability.
 */
class RandomSparsePattern : public ConnectivityPattern {
public:
    /**
     * @brief Constructor
     * @param probability Connection probability (0.0 to 1.0)
     * @param weight Synaptic weight for all connections
     * @param delay Synaptic delay in milliseconds for all connections
     */
    RandomSparsePattern(double probability, double weight = 1.0, double delay = 1.0);
    
    std::vector<Connection> generateConnections(
        const std::vector<uint64_t>& sourceNeurons,
        const std::vector<uint64_t>& targetNeurons) override;
    
private:
    double probability_;
    double weight_;
    double delay_;
};

/**
 * @class AllToAllPattern
 * @brief Fully connected pattern (all-to-all connectivity)
 * 
 * Creates connections from every source neuron to every target neuron.
 */
class AllToAllPattern : public ConnectivityPattern {
public:
    /**
     * @brief Constructor
     * @param weight Synaptic weight for all connections
     * @param delay Synaptic delay in milliseconds for all connections
     */
    AllToAllPattern(double weight = 1.0, double delay = 1.0);
    
    std::vector<Connection> generateConnections(
        const std::vector<uint64_t>& sourceNeurons,
        const std::vector<uint64_t>& targetNeurons) override;
    
private:
    double weight_;
    double delay_;
};

/**
 * @class OneToOnePattern
 * @brief One-to-one connectivity pattern
 * 
 * Creates connections from source[i] to target[i]. Requires equal
 * number of source and target neurons.
 */
class OneToOnePattern : public ConnectivityPattern {
public:
    /**
     * @brief Constructor
     * @param weight Synaptic weight for all connections
     * @param delay Synaptic delay in milliseconds for all connections
     */
    OneToOnePattern(double weight = 1.0, double delay = 1.0);
    
    std::vector<Connection> generateConnections(
        const std::vector<uint64_t>& sourceNeurons,
        const std::vector<uint64_t>& targetNeurons) override;
    
private:
    double weight_;
    double delay_;
};

/**
 * @class ManyToOnePattern
 * @brief Many-to-one connectivity pattern
 * 
 * Creates connections from all source neurons to each target neuron.
 * This is useful for convergent connections.
 */
class ManyToOnePattern : public ConnectivityPattern {
public:
    /**
     * @brief Constructor
     * @param weight Synaptic weight for all connections
     * @param delay Synaptic delay in milliseconds for all connections
     */
    ManyToOnePattern(double weight = 1.0, double delay = 1.0);
    
    std::vector<Connection> generateConnections(
        const std::vector<uint64_t>& sourceNeurons,
        const std::vector<uint64_t>& targetNeurons) override;
    
private:
    double weight_;
    double delay_;
};

/**
 * @class DistanceDependentPattern
 * @brief Distance-dependent connectivity with Gaussian falloff
 * 
 * Connection probability decreases with distance according to a
 * Gaussian function: P(d) = exp(-d^2 / (2*sigma^2))
 * 
 * Requires spatial positions to be provided for neurons.
 */
class DistanceDependentPattern : public ConnectivityPattern {
public:
    /**
     * @brief Constructor
     * @param sigma Standard deviation of Gaussian falloff
     * @param weight Synaptic weight for all connections
     * @param delay Synaptic delay in milliseconds for all connections
     */
    DistanceDependentPattern(double sigma, double weight = 1.0, double delay = 1.0);
    
    /**
     * @brief Set spatial positions for neurons
     * @param positions Map from neuron ID to spatial position
     */
    void setPositions(const std::map<uint64_t, SpatialPosition>& positions);
    
    std::vector<Connection> generateConnections(
        const std::vector<uint64_t>& sourceNeurons,
        const std::vector<uint64_t>& targetNeurons) override;
    
private:
    double sigma_;
    double weight_;
    double delay_;
    std::map<uint64_t, SpatialPosition> positions_;
};

/**
 * @class TopographicPattern
 * @brief Topographic connectivity preserving spatial relationships
 * 
 * Creates connections that preserve spatial relationships between
 * source and target populations. Neurons at similar positions in
 * their respective populations are more likely to connect.
 */
class TopographicPattern : public ConnectivityPattern {
public:
    /**
     * @brief Constructor
     * @param sigma Standard deviation for position matching
     * @param weight Synaptic weight for all connections
     * @param delay Synaptic delay in milliseconds for all connections
     */
    TopographicPattern(double sigma, double weight = 1.0, double delay = 1.0);
    
    std::vector<Connection> generateConnections(
        const std::vector<uint64_t>& sourceNeurons,
        const std::vector<uint64_t>& targetNeurons) override;
    
private:
    double sigma_;
    double weight_;
    double delay_;
};

/**
 * @class SmallWorldPattern
 * @brief Small-world connectivity (local + long-range connections)
 * 
 * Combines local connectivity (high probability for nearby neurons)
 * with sparse long-range connections. This creates the "small-world"
 * property found in many biological neural networks.
 */
class SmallWorldPattern : public ConnectivityPattern {
public:
    /**
     * @brief Constructor
     * @param localProbability Probability of local connections
     * @param longRangeProbability Probability of long-range connections
     * @param localRadius Radius defining "local" neighborhood
     * @param weight Synaptic weight for all connections
     * @param delay Synaptic delay in milliseconds for all connections
     */
    SmallWorldPattern(double localProbability, 
                     double longRangeProbability,
                     double localRadius,
                     double weight = 1.0, 
                     double delay = 1.0);
    
    /**
     * @brief Set spatial positions for neurons
     * @param positions Map from neuron ID to spatial position
     */
    void setPositions(const std::map<uint64_t, SpatialPosition>& positions);
    
    std::vector<Connection> generateConnections(
        const std::vector<uint64_t>& sourceNeurons,
        const std::vector<uint64_t>& targetNeurons) override;
    
private:
    double localProbability_;
    double longRangeProbability_;
    double localRadius_;
    double weight_;
    double delay_;
    std::map<uint64_t, SpatialPosition> positions_;
};

/**
 * @class TiledReceptiveFieldPattern
 * @brief Tile-based receptive field connectivity for input-to-L4 projections
 *
 * Divides the input grid into tiles and assigns a subset of tiles to each
 * column. Each L4 neuron (on a grid layout) connects to specific pixels
 * within its column's tile regions, providing spatial locality and
 * breaking symmetry across columns.
 *
 * This is a per-column pattern: a new instance must be created for each
 * column with the appropriate column index.
 */
class TiledReceptiveFieldPattern : public ConnectivityPattern {
public:
    /**
     * @brief Constructor
     * @param partitionInputSize Input partition side length (e.g. 28 for EMNIST)
     * @param inputRows Total input grid rows
     * @param inputCols Total input grid cols
     * @param tilesPerSide Number of tiles along each axis (e.g. 4 for 4×4)
     * @param tilesPerColumn Number of tiles assigned per column
     * @param l4GridSize L4 grid side length (e.g. 7 for 7×7 = 49 neurons)
     * @param columnIndex Index of the column (for deterministic tile selection)
     * @param weight Synaptic weight
     * @param delay Synaptic delay in milliseconds
     */
    TiledReceptiveFieldPattern(int partitionInputSize, int inputRows, int inputCols,
                               int tilesPerSide, int tilesPerColumn,
                               int l4GridSize, int columnIndex,
                               double weight = 0.1, double delay = 1.5);

    std::vector<Connection> generateConnections(
        const std::vector<uint64_t>& sourceNeurons,
        const std::vector<uint64_t>& targetNeurons) override;

    /// Get the input pixel indices that belong to this column's receptive field
    const std::vector<int>& getInputMaskActiveIdx() const { return inputMaskActiveIdx_; }

    /// Get the selected tile indices for this column
    const std::vector<int>& getTileIndices() const { return tileIndices_; }

private:
    int partitionInputSize_;
    int inputRows_;
    int inputCols_;
    int partitionCount_;
    bool partitionsHorizontal_;
    int tilesPerSide_;
    int tilesPerColumn_;
    int l4GridSize_;
    int columnIndex_;
    double weight_;
    double delay_;

    std::vector<int> tileIndices_;       // selected tile indices
    std::vector<int> inputMaskActiveIdx_; // input pixel indices in receptive field

    void computeTileSelection();
    int mapPartitionIndex(int partition, int localRow, int localCol) const;
};

} // namespace snnfw

#endif // SNNFW_CONNECTIVITY_PATTERN_H

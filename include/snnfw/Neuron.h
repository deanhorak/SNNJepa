#ifndef SNNFW_NEURON_H
#define SNNFW_NEURON_H

#include "snnfw/NeuralObject.h"
#include "snnfw/BinaryPattern.h"
#include <vector>
#include <cstddef>
#include <memory>
#include <deque>
#include <mutex>

namespace snnfw {

// Forward declarations
namespace learning {
    class PatternUpdateStrategy;
}
class NetworkPropagator;

/**
 * @brief Similarity metric types for pattern matching
 */
enum class SimilarityMetric {
    COSINE,              ///< Cosine similarity (default, current behavior)
    HISTOGRAM,           ///< Histogram intersection (Jaccard-like)
    EUCLIDEAN,           ///< Euclidean distance converted to similarity
    CORRELATION,         ///< Pearson correlation
    WAVEFORM             ///< Cross-correlation of Gaussian-smoothed waveforms (temporal shape)
};

/**
 * @brief Neuron class for spiking neural network with temporal pattern learning
 *
 * This class implements a biologically-inspired neuron that learns temporal spike patterns
 * rather than using traditional weight-based learning. Key features:
 *
 * Pattern Learning:
 * - Stores temporal spike patterns using fixed-size BinaryPattern (200 bytes each)
 * - Up to maxReferencePatterns patterns per neuron (default: 20, MNIST uses 100)
 * - When capacity is reached, new patterns are blended into most similar existing pattern
 * - Uses cosine similarity for pattern matching
 * - Memory: 200 bytes per pattern (vs ~800 bytes for vector<double>)
 *
 * Spike Processing:
 * - Rolling time window maintains recent spikes (vector<double>)
 * - Spikes outside the window are automatically removed
 * - Pattern learning via learnCurrentPattern() converts spikes to BinaryPattern
 *
 * Similarity Metrics:
 * - Cosine similarity for binary pattern matching (drop-in replacement)
 * - Histogram intersection for spike count overlap
 * - Euclidean and correlation metrics also available
 *
 * Connectivity:
 * - One axon (output terminal) - stored as axonId
 * - Multiple dendrites (input terminals) - stored as dendriteIds
 *
 * Usage in MNIST Experiments:
 * - 392 neurons (49 regions × 8 orientations)
 * - Each neuron learns edge patterns at specific orientation
 * - Activation vectors used for k-NN classification
 * - Achieves 81.20% accuracy on MNIST digit recognition
 */
class Neuron : public NeuralObject {
public:
    /**
     * @brief Constructor with parameters
     * @param windowSizeMs Size of the rolling time window in milliseconds
     * @param similarityThreshold Threshold for pattern similarity (0.0 to 1.0)
     * @param maxReferencePatterns Maximum number of reference patterns to store
     * @param neuronId Unique identifier for this neuron (default: 0)
     */
    Neuron(double windowSizeMs, double similarityThreshold, size_t maxReferencePatterns = 20, uint64_t neuronId = 0);

    /**
     * @brief Insert a spike with a given timestamp
     * @param spikeTime Timestamp of the spike
     */
    void insertSpike(double spikeTime);

    /**
     * @brief Learn the current spike pattern (either add or blend)
     */
    void learnCurrentPattern();

    /**
     * @brief Set the pattern update strategy for learning
     * @param strategy Shared pointer to the pattern update strategy
     */
    void setPatternUpdateStrategy(std::shared_ptr<learning::PatternUpdateStrategy> strategy);

    /**
     * @brief Set the similarity metric for pattern matching
     * @param metric Similarity metric to use (COSINE, HISTOGRAM, EUCLIDEAN, CORRELATION)
     */
    void setSimilarityMetric(SimilarityMetric metric) { similarityMetric_ = metric; }

    /**
     * @brief Get the current similarity metric
     * @return Current similarity metric
     */
    SimilarityMetric getSimilarityMetric() const { return similarityMetric_; }

    /**
     * @brief Get the window size in milliseconds
     * @return Window size in milliseconds
     */
    double getWindowSize() const { return windowSize; }

    /**
     * @brief Get the similarity threshold
     * @return Similarity threshold (0.0 to 1.0)
     */
    double getSimilarityThreshold() const { return threshold; }

    /**
     * @brief Get the maximum number of reference patterns
     * @return Maximum number of reference patterns
     */
    size_t getMaxReferencePatterns() const { return maxPatterns; }

    /**
     * @brief Print current rolling window of spikes
     */
    void printSpikes() const;

    /**
     * @brief Print all reference patterns
     */
    void printReferencePatterns() const;

    /**
     * @brief Check if current pattern matches any learned pattern
     * @return true if neuron should fire, false otherwise
     */
    bool checkShouldFire() const { return shouldFire(); }

    /**
     * @brief Get the best similarity score between current spikes and learned patterns
     * @return Best similarity score (0.0 to 1.0), or -1.0 if no patterns learned
     */
    double getBestSimilarity() const;

    /**
     * @brief Get the number of learned patterns
     * @return Number of reference patterns stored
     */
    size_t getLearnedPatternCount() const {
        return prototypePatterns_.size() + exemplarPatterns_.size();
    }

    /**
     * @brief Get the number of prototype memories
     */
    size_t getPrototypePatternCount() const { return prototypePatterns_.size(); }

    /**
     * @brief Get the number of exemplar memories
     */
    size_t getExemplarPatternCount() const { return exemplarPatterns_.size(); }

    /**
     * @brief Get all learned patterns (as BinaryPattern)
     * @return Const reference to vector of learned BinaryPatterns
     */
    const std::vector<BinaryPattern>& getLearnedPatterns() const;

    /**
     * @brief Get all spikes from the rolling window (thread-safe copy)
     * @return Vector of spike times
     */
    std::vector<double> getSpikes() const {
        std::lock_guard<std::mutex> lock(spikesMutex_);
        return spikes;
    }

    /**
     * @brief Clear all spikes from the rolling window (thread-safe)
     */
    void clearSpikes() {
        std::lock_guard<std::mutex> lock(spikesMutex_);
        spikes.clear();
        maxSpikeTime_ = -std::numeric_limits<double>::infinity();
    }

    /**
     * @brief Remove spikes older than a cutoff time (thread-safe)
     * @param cutoffTime Spikes with time < cutoffTime are removed
     */
    void removeSpikesBefore(double cutoffTime) {
        std::lock_guard<std::mutex> lock(spikesMutex_);
        spikes.erase(
            std::remove_if(spikes.begin(), spikes.end(),
                           [cutoffTime](double t) { return t < cutoffTime; }),
            spikes.end());
    }

    /**
     * @brief Set the axon ID for this neuron
     * @param id ID of the axon
     */
    void setAxonId(uint64_t id) { axonId = id; }

    /**
     * @brief Get the axon ID for this neuron
     * @return ID of the axon (0 if not set)
     */
    uint64_t getAxonId() const { return axonId; }

    /**
     * @brief Add a dendrite to this neuron
     * @param dendriteId ID of the dendrite to add
     */
    void addDendrite(uint64_t dendriteId);

    /**
     * @brief Remove a dendrite from this neuron
     * @param dendriteId ID of the dendrite to remove
     * @return true if dendrite was removed, false if not found
     */
    bool removeDendrite(uint64_t dendriteId);

    /**
     * @brief Get all reference patterns learned by this neuron (as BinaryPattern)
     * @return Const reference to vector of reference BinaryPatterns
     */
    const std::vector<BinaryPattern>& getReferencePatterns() const { return getLearnedPatterns(); }

    /**
     * @brief Get all dendrite IDs for this neuron
     * @return Const reference to vector of dendrite IDs
     */
    const std::vector<uint64_t>& getDendriteIds() const { return dendriteIds; }

    /**
     * @brief Get the temporal signature (spike timing offsets) for this neuron
     * @return Vector of time offsets in milliseconds
     */
    const std::vector<double>& getTemporalSignature() const { return temporalSignature_; }

    /**
     * @brief Fire this neuron's intrinsic temporal signature pattern
     * Inserts spikes according to the neuron's unique temporal signature
     * @param baseTime Base time for the spike pattern (signature offsets are added to this)
     */
    void fireSignature(double baseTime);

    /**
     * @brief Disable the neuron's temporal signature (no intrinsic spike pattern)
     */
    void disableTemporalSignature() { temporalSignature_.clear(); }

    /**
     * @brief Get the number of dendrites
     * @return Number of dendrites
     */
    size_t getDendriteCount() const { return dendriteIds.size(); }

    /**
     * @brief Record an incoming spike from a synapse (for STDP)
     * @param synapseId ID of the synapse that delivered the spike
     * @param spikeTime Time when the spike arrived
     * @param dispatchTime Time when the spike was originally dispatched
     */
    void recordIncomingSpike(uint64_t synapseId, double spikeTime, double dispatchTime = 0.0);

    /**
     * @brief Get count of incoming spikes recorded (thread-safe)
     * @return Number of incoming spikes recorded since last reset
     */
    size_t getIncomingSpikeCount() const;

    /**
     * @brief Reset incoming spike counter (thread-safe)
     */
    void resetIncomingSpikeCount();

    /**
     * @brief Apply inhibition to this neuron
     * @param amount Amount of inhibition to apply (reduces activation)
     */
    void applyInhibition(double amount);

    /**
     * @brief Get the current inhibition level
     * @return Current inhibition amount
     */
    double getInhibition() const { return inhibition_; }

    /**
     * @brief Reset inhibition to zero
     */
    void resetInhibition() { inhibition_ = 0.0; }

    /**
     * @brief Get similarity-driven activation before inhibition is applied
     * @return Similarity activation in [0, 1]
     */
    double getSimilarityActivation() const;

    /**
     * @brief Get activation level (best similarity minus inhibition)
     * @return Effective activation level
     */
    double getActivation() const;

    /**
     * @brief Update firing rate statistics (for homeostatic plasticity)
     * @param currentTime Current simulation time
     */
    void updateFiringRate(double currentTime);

    /**
     * @brief Get the current firing rate (Hz)
     * @return Firing rate in Hz
     */
    double getFiringRate() const { return firingRate_; }

    /**
     * @brief Set target firing rate for homeostatic plasticity
     * @param targetRate Target firing rate in Hz
     */
    void setTargetFiringRate(double targetRate) { targetFiringRate_ = targetRate; }

    /**
     * @brief Apply homeostatic plasticity to adjust intrinsic excitability
     * Increases excitability if firing rate is too low, decreases if too high
     */
    void applyHomeostaticPlasticity();

    /**
     * @brief Fire the neuron and send acknowledgments to presynaptic neurons
     * @param firingTime Time when this neuron fires
     * @return Number of acknowledgments sent
     */
    int fireAndAcknowledge(double firingTime);

    /**
     * @brief Set the NetworkPropagator for sending acknowledgments
     * @param propagator Weak pointer to the NetworkPropagator
     */
    void setNetworkPropagator(std::weak_ptr<NetworkPropagator> propagator) {
        networkPropagator_ = propagator;
    }

    /**
     * @brief Clear old incoming spike records outside the temporal window
     * @param currentTime Current simulation time
     */
    void clearOldIncomingSpikes(double currentTime);

    /**
     * @brief Perform periodic memory cleanup to prevent memory leaks
     * Clears old spikes, shrinks containers, and resets counters
     * @param currentTime Current simulation time
     */
    void periodicMemoryCleanup(double currentTime);

    // Serializable interface implementation
    std::string toJson() const override;
    bool fromJson(const std::string& json) override;
    std::string getTypeName() const override { return "Neuron"; }

private:
    /**
     * @brief Structure to track incoming spikes for STDP
     */
    struct IncomingSpike {
        uint64_t synapseId;   ///< ID of the synapse that delivered the spike
        double arrivalTime;   ///< Time when the spike arrived at this neuron
        double dispatchTime;  ///< Time when the spike was originally dispatched

        IncomingSpike(uint64_t synId, double arrTime, double dispTime = 0.0)
            : synapseId(synId), arrivalTime(arrTime), dispatchTime(dispTime) {}
    };

    std::vector<double> spikes;                          ///< Rolling spike window (temporary, converted to BinaryPattern)
    mutable std::mutex spikesMutex_;                     ///< Mutex to protect spikes vector from concurrent access
    double maxSpikeTime_;                                ///< Maximum spike time in the buffer (for efficient cleanup)
    std::vector<BinaryPattern> prototypePatterns_;       ///< Consolidated memories for strong repeated motifs
    std::vector<uint16_t> prototypeSupports_;            ///< Reinforcement counts for prototype memories
    std::vector<BinaryPattern> exemplarPatterns_;        ///< Specific memories for variant exemplars
    std::vector<uint16_t> exemplarSupports_;             ///< Reinforcement counts for exemplar memories
    mutable std::vector<BinaryPattern> combinedPatternsCache_; ///< Scratch cache for read-only accessors
    double windowSize;                                   ///< Size of rolling window in ms
    double threshold;                                    ///< Similarity threshold for firing
    size_t maxPatterns;                                  ///< Maximum number of reference patterns

    uint64_t axonId;                                     ///< ID of the axon for this neuron (0 if not set)
    std::vector<uint64_t> dendriteIds;                   ///< IDs of dendrites connected to this neuron

    std::shared_ptr<learning::PatternUpdateStrategy> patternStrategy_;  ///< Strategy for updating patterns (optional)
    SimilarityMetric similarityMetric_;                  ///< Similarity metric for pattern matching (default: COSINE)

    // STDP-related members
    std::deque<IncomingSpike> incomingSpikes_;           ///< Recent incoming spikes for STDP (within window)
    mutable std::mutex incomingSpikesMutex_;             ///< Mutex to protect incomingSpikes_ from concurrent access
    size_t incomingSpikeCount_;                          ///< Count of incoming spikes recorded since last reset
    std::weak_ptr<NetworkPropagator> networkPropagator_; ///< Reference to NetworkPropagator for sending acknowledgments

    // Temporal signature - unique spike pattern for this neuron
    std::vector<double> temporalSignature_;              ///< Unique temporal offsets (ms) for multi-spike pattern

    // Inhibition for winner-take-all and lateral inhibition
    double inhibition_;                                  ///< Current inhibition level (reduces activation)

    // Homeostatic plasticity - firing rate regulation
    double firingRate_;                                  ///< Current firing rate (Hz)
    double targetFiringRate_;                            ///< Target firing rate for homeostasis (Hz)
    double intrinsicExcitability_;                       ///< Intrinsic excitability multiplier (default: 1.0)
    double lastFiringTime_;                              ///< Time of last firing event
    int firingCount_;                                    ///< Number of firings in current window
    double firingWindowStart_;                           ///< Start time of firing rate measurement window

    /**
     * @brief Generate a unique temporal signature for this neuron
     * Creates a pattern of 1-10 spikes spread over ~100ms
     */
    void generateTemporalSignature();

    /**
     * @brief Remove spikes outside the rolling window
     * @param currentTime Current timestamp
     */
    void removeOldSpikes(double currentTime);

    /**
     * @brief Remove old spikes from the rolling window (unsafe - caller must hold spikesMutex_)
     * @param currentTime Current timestamp
     */
    void removeOldSpikesUnsafe(double currentTime);

    /**
     * @brief Compute similarity between two patterns using the selected metric
     * @param a First pattern
     * @param b Second pattern
     * @return Similarity score (0.0 to 1.0)
     */
    double computeSimilarity(const BinaryPattern& a, const BinaryPattern& b) const;

    double findBestSimilarity(
        const BinaryPattern& currentPattern,
        const std::vector<BinaryPattern>& patterns,
        int* bestIndex = nullptr) const;

    /**
     * @brief Check if current pattern is similar to any reference pattern
     * @return true if neuron should fire, false otherwise
     */
    bool shouldFire() const;

    /**
     * @brief Find the most similar reference pattern
     * @param newPattern Pattern to compare against
     * @return Index of most similar pattern, or -1 if none found
     */
    int findMostSimilarPattern(const std::vector<double>& newPattern) const;

    /**
     * @brief Blend the new pattern into the target reference pattern
     * @param target Target pattern to blend into
     * @param newPattern New pattern to blend
     * @param alpha Blending factor (0.0 to 1.0)
     */
    void blendPattern(std::vector<double>& target, const std::vector<double>& newPattern, double alpha = 0.2);

    /**
     * @brief Calculate cosine similarity between two equal-length vectors
     * @param a First vector
     * @param b Second vector
     * @return Cosine similarity value (0.0 to 1.0)
     */
    static double cosineSimilarity(const std::vector<double>& a, const std::vector<double>& b);

    /**
     * @brief Compute spike distance between two spike trains
     * @param spikes1 First spike train
     * @param spikes2 Second spike train
     * @return Distance (lower is more similar)
     */
    double spikeDistance(const std::vector<double>& spikes1, const std::vector<double>& spikes2) const;

    /**
     * @brief Convert spike pattern to temporal histogram for fuzzy matching
     * @param pattern Spike times to convert
     * @return Histogram representation
     */
    std::vector<double> spikeToHistogram(const std::vector<double>& pattern) const;

    /**
     * @brief Compute similarity between two histograms
     * @param hist1 First histogram
     * @param hist2 Second histogram
     * @return Similarity score (0.0 to 1.0)
     */
    double histogramSimilarity(const std::vector<double>& hist1, const std::vector<double>& hist2) const;
};

} // namespace snnfw

#endif // SNNFW_NEURON_H

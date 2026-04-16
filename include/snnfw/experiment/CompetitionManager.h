#ifndef SNNFW_EXPERIMENT_COMPETITION_MANAGER_H
#define SNNFW_EXPERIMENT_COMPETITION_MANAGER_H

#include "snnfw/Neuron.h"
#include "snnfw/NetworkPropagator.h"
#include "snnfw/experiment/ExperimentConfig.h"
#include "snnfw/declarative/NetworkConstructor.h"
#include <memory>
#include <vector>
#include <array>
#include <random>
#include <cstddef>
#include <cstdint>

namespace snnfw {
namespace experiment {

/**
 * @brief Manages winner-take-all competition in L4, L5, and output layers.
 *
 * Extracted from the competition logic in emnist_letters_training.cpp.
 */
class CompetitionManager {
public:
    explicit CompetitionManager(const ExperimentConfig& config);

    /**
     * @brief Run L4 competition for all columns.
     * Selects top-k L4 neurons per column and fires them.
     * @param columns The cortical columns
     * @param inputFired Which input neurons fired
     * @param baseTime Current base time
     * @param propagator Network propagator for firing
     * @return colHasL4 - which columns had L4 activity
     */
    std::vector<bool> runL4Competition(
        std::vector<declarative::ConstructedNetwork::ColumnGroup>& columns,
        const std::vector<bool>& inputFired,
        double baseTime,
        std::shared_ptr<NetworkPropagator> propagator);

    /**
     * @brief Run L5 competition for all columns.
     * Selects top-k L5 neurons per column, applies inhibition to losers.
     * @param columns The cortical columns
     * @param colHasL4 Which columns had L4 activity
     * @param baseTime Current base time
     * @param propagator Network propagator for firing
     * @return l5WinnerGlobal - which L5 neurons won globally
     */
    std::vector<bool> runL5Competition(
        std::vector<declarative::ConstructedNetwork::ColumnGroup>& columns,
        const std::vector<bool>& colHasL4,
        double baseTime,
        std::shared_ptr<NetworkPropagator> propagator,
        bool trainingPhase);

    /**
     * @brief Apply output competition (keep top-k classes by spike count).
     * @param outSpikeCounts Per-class spike counts (modified in place)
     */
    void applyOutputCompetition(std::vector<int>& outSpikeCounts);

    /// Set the random seed for jitter
    void setSeed(unsigned int seed);

private:
    struct InterColumnEdge {
        size_t sourceColumn = 0;
        double weight = 0.0;
    };

    void rebuildInterColumnCache(
        const std::vector<declarative::ConstructedNetwork::ColumnGroup>& columns);

    static double computeOverlapRatio(
        const std::vector<int>& a,
        const std::vector<int>& b);

    static double computeFeatureGate(
        const declarative::ConstructedNetwork::ColumnGroup& target,
        const declarative::ConstructedNetwork::ColumnGroup& source,
        double maxOrientationDeltaDeg,
        double maxFrequencyOctaveDelta);

    static double computeHybridCompetitionScore(
        size_t spikeCount,
        size_t maxSpikeCount,
        double bestSimilarity,
        double similarityWeight);

    const ExperimentConfig& config_;
    std::mt19937 gen_;
    std::normal_distribution<double> l4PostJitterDist_;
    std::vector<std::vector<InterColumnEdge>> interColumnIncoming_;
    size_t interColumnCacheColumns_ = 0;
    bool interColumnCacheValid_ = false;
    uint64_t l4CompetitionCalls_ = 0;
    uint64_t l5CompetitionCalls_ = 0;
};

} // namespace experiment
} // namespace snnfw

#endif // SNNFW_EXPERIMENT_COMPETITION_MANAGER_H

#ifndef SNNFW_EXPERIMENT_TRAINING_PIPELINE_H
#define SNNFW_EXPERIMENT_TRAINING_PIPELINE_H

#include "snnfw/experiment/ExperimentConfig.h"
#include "snnfw/experiment/SpikeEncoder.h"
#include "snnfw/experiment/CompetitionManager.h"
#include "snnfw/experiment/KNNClassifier.h"
#include "snnfw/experiment/SupervisedTeacher.h"
#include "snnfw/declarative/NetworkConstructor.h"
#include "snnfw/EMNISTLoader.h"
#include "snnfw/SpikeProcessor.h"
#include "snnfw/NetworkPropagator.h"
#include <memory>
#include <vector>
#include <functional>

namespace snnfw {
namespace experiment {

/**
 * @brief Result of a single inference pass over one image.
 */
struct InferenceResult {
    std::vector<uint16_t> l5Counts;       // spike counts per L5 neuron
    std::vector<uint16_t> l5Latencies;    // first-spike latency per L5 neuron
    std::vector<int> outSpikeCounts;      // per-class output spike counts
    std::vector<int> rawOutSpikeCounts;   // before output competition
    size_t totalL5Spikes = 0;
};

/**
 * @brief Multi-pass training and testing pipeline for SNN classification.
 *
 * Orchestrates the full training loop:
 * 1. For each pass: encode images, run L4/L5 competition, supervised teach, store patterns
 * 2. After each pass: run testing phase with STDP disabled, measure accuracy
 * 3. Check convergence and stop when accuracy stabilizes
 */
class TrainingPipeline {
public:
    TrainingPipeline(const ExperimentConfig& config,
                     declarative::ConstructedNetwork& network);

    /// Run the full multi-pass training + testing loop. Returns final accuracy.
    double run(EMNISTLoader& trainLoader, EMNISTLoader& testLoader);

    /// Run inference on a single image (used internally and for testing).
    InferenceResult runInference(const EMNISTLoader::Image& image);

    /// Get per-pass accuracy history
    const std::vector<double>& getPassAccuracies() const { return passAccuracies_; }

    /// Get total patterns learned
    int getTotalPatternsLearned() const { return totalPatternsLearned_; }

private:
    const ExperimentConfig& config_;
    declarative::ConstructedNetwork& network_;

    // Sub-components
    SpikeEncoder encoder_;
    CompetitionManager competition_;
    KNNClassifier classifier_;
    SupervisedTeacher teacher_;

    // State
    std::vector<double> passAccuracies_;
    int totalPatternsLearned_ = 0;

    // Helpers
    void collectL5Readout(std::vector<uint16_t>& counts,
                          std::vector<uint16_t>& latencies,
                          const std::vector<bool>& l5Winners,
                          double baseTime);
    void applyL5DivisiveNormalization(std::vector<uint16_t>& counts) const;

    void applyHomeostasis();
    void processMotorAdapters(double currentTimeMs);

    double runTestingPhase(EMNISTLoader& testLoader);
};

} // namespace experiment
} // namespace snnfw

#endif // SNNFW_EXPERIMENT_TRAINING_PIPELINE_H

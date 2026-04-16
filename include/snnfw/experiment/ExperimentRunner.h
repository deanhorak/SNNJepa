#ifndef SNNFW_EXPERIMENT_RUNNER_H
#define SNNFW_EXPERIMENT_RUNNER_H

#include "snnfw/experiment/ExperimentConfig.h"
#include "snnfw/experiment/TrainingPipeline.h"
#include "snnfw/declarative/DeclarativeLoader.h"
#include "snnfw/declarative/NetworkConstructor.h"
#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Datastore.h"
#include "snnfw/EMNISTLoader.h"
#include <memory>
#include <string>

namespace snnfw {
namespace experiment {

/**
 * @brief Top-level experiment runner: loads config, builds network, runs training + testing.
 *
 * Usage:
 *   ExperimentConfig config;
 *   // fill config fields...
 *   ExperimentRunner runner(config);
 *   double accuracy = runner.run();
 */
class ExperimentRunner {
public:
    explicit ExperimentRunner(const ExperimentConfig& config);

    /// Run the full experiment. Returns final accuracy.
    double run();

    /// Get per-pass accuracy history
    const std::vector<double>& getPassAccuracies() const;

private:
    ExperimentConfig config_;

    // Owned infrastructure
    std::unique_ptr<Datastore> datastore_;
    std::unique_ptr<NeuralObjectFactory> factory_;

    // Network
    std::unique_ptr<declarative::ConstructedNetwork> network_;

    // Pipeline
    std::unique_ptr<TrainingPipeline> pipeline_;

    // Data
    std::unique_ptr<EMNISTLoader> trainLoader_;
    std::unique_ptr<EMNISTLoader> testLoader_;

    // Results
    std::vector<double> passAccuracies_;

    /// Load network from SONATA/JSON config
    void buildNetwork();

    /// Load EMNIST dataset
    void loadData();

    /// Populate config fields from the loaded IR (numColumns, layer sizes, etc.)
    void syncConfigFromIR(const declarative::NetworkIR& ir);

    /// Instantiate declarative adapters and attach to the constructed network.
    void instantiateAdapters(const declarative::NetworkIR& ir);
};

} // namespace experiment
} // namespace snnfw

#endif // SNNFW_EXPERIMENT_RUNNER_H

/**
 * @file HyperparameterOptimizer.h
 * @brief General-purpose hyperparameter optimization framework
 * 
 * Supports multiple optimization strategies:
 * - Grid Search: Exhaustive search over parameter grid
 * - Random Search: Random sampling of parameter space
 * - Bayesian Optimization: Model-based optimization
 * - Genetic Algorithm: Evolutionary optimization
 */

#ifndef SNNFW_HYPERPARAMETER_OPTIMIZER_H
#define SNNFW_HYPERPARAMETER_OPTIMIZER_H

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <memory>
#include <random>
#include <nlohmann/json.hpp>

namespace snnfw {

/**
 * @brief Parameter specification for optimization
 */
struct ParameterSpec {
    enum class Type {
        INTEGER,
        DOUBLE,
        CATEGORICAL
    };
    
    std::string name;           ///< Parameter name (e.g., "neuron.similarity_threshold")
    Type type;                  ///< Parameter type
    
    // For INTEGER and DOUBLE
    double minValue;            ///< Minimum value
    double maxValue;            ///< Maximum value
    double step;                ///< Step size (for grid search)
    
    // For CATEGORICAL
    std::vector<std::string> categories; ///< Possible categorical values
    
    // For INTEGER
    int minInt() const { return static_cast<int>(minValue); }
    int maxInt() const { return static_cast<int>(maxValue); }
    int stepInt() const { return static_cast<int>(step); }
};

/**
 * @brief Parameter configuration (one point in parameter space)
 */
struct ParameterConfig {
    std::map<std::string, double> doubleParams;
    std::map<std::string, int> intParams;
    std::map<std::string, std::string> stringParams;
    
    // Convert to JSON for saving
    nlohmann::json toJson() const;
    
    // Create from JSON
    static ParameterConfig fromJson(const nlohmann::json& j);
};

/**
 * @brief Result of a single experiment run
 */
struct ExperimentResult {
    ParameterConfig config;     ///< Parameter configuration used
    double score;               ///< Objective score (e.g., accuracy)
    double trainingTime;        ///< Training time in seconds
    double testingTime;         ///< Testing time in seconds
    std::map<std::string, double> metrics; ///< Additional metrics
    std::string timestamp;      ///< When experiment was run
    
    nlohmann::json toJson() const;
    static ExperimentResult fromJson(const nlohmann::json& j);
};

/**
 * @brief Objective function type
 * 
 * Takes a parameter configuration and returns a score.
 * Higher scores are better.
 */
using ObjectiveFunction = std::function<ExperimentResult(const ParameterConfig&)>;

/**
 * @brief Optimization strategy
 */
enum class OptimizationStrategy {
    GRID_SEARCH,        ///< Exhaustive grid search
    RANDOM_SEARCH,      ///< Random sampling
    BAYESIAN,           ///< Bayesian optimization (Gaussian Process)
    GENETIC_ALGORITHM   ///< Genetic algorithm
};

/**
 * @brief Hyperparameter optimizer
 */
class HyperparameterOptimizer {
public:
    /**
     * @brief Constructor
     * @param strategy Optimization strategy to use
     * @param seed Random seed for reproducibility
     */
    HyperparameterOptimizer(OptimizationStrategy strategy = OptimizationStrategy::GRID_SEARCH,
                           unsigned int seed = 42);
    
    /**
     * @brief Add a parameter to optimize
     */
    void addParameter(const ParameterSpec& param);
    
    /**
     * @brief Set the objective function
     */
    void setObjective(ObjectiveFunction objective);
    
    /**
     * @brief Set maximum number of iterations
     */
    void setMaxIterations(int maxIter) { maxIterations_ = maxIter; }
    
    /**
     * @brief Set number of parallel workers
     */
    void setNumWorkers(int workers) { numWorkers_ = workers; }
    
    /**
     * @brief Set results directory
     */
    void setResultsDir(const std::string& dir) { resultsDir_ = dir; }
    
    /**
     * @brief Enable/disable saving intermediate results
     */
    void setSaveIntermediateResults(bool save) { saveIntermediateResults_ = save; }
    
    /**
     * @brief Run optimization
     * @return Best result found
     */
    ExperimentResult optimize();
    
    /**
     * @brief Get all results
     */
    const std::vector<ExperimentResult>& getResults() const { return results_; }
    
    /**
     * @brief Get best result
     */
    ExperimentResult getBestResult() const;
    
    /**
     * @brief Save results to JSON file
     */
    void saveResults(const std::string& filename) const;
    
    /**
     * @brief Load results from JSON file
     */
    void loadResults(const std::string& filename);
    
    /**
     * @brief Resume optimization from saved results
     */
    ExperimentResult resume(const std::string& filename);
    
private:
    OptimizationStrategy strategy_;
    std::vector<ParameterSpec> parameters_;
    ObjectiveFunction objective_;
    std::vector<ExperimentResult> results_;
    
    int maxIterations_;
    int numWorkers_;
    std::string resultsDir_;
    bool saveIntermediateResults_;
    
    std::mt19937 rng_;
    
    // Strategy-specific methods
    ExperimentResult gridSearch();
    ExperimentResult randomSearch();
    ExperimentResult bayesianOptimization();
    ExperimentResult geneticAlgorithm();
    
    // Helper methods
    std::vector<ParameterConfig> generateGridConfigs();
    ParameterConfig generateRandomConfig();
    ParameterConfig sampleFromPosterior();
    std::vector<ParameterConfig> crossover(const ParameterConfig& parent1,
                                           const ParameterConfig& parent2);
    ParameterConfig mutate(const ParameterConfig& config, double mutationRate);
    
    void saveIntermediateResult(const ExperimentResult& result);
    std::string getCurrentTimestamp() const;
};

/**
 * @brief Helper class for MNIST-specific optimization
 */
class MNISTOptimizer {
public:
    /**
     * @brief Create optimizer with MNIST-specific parameters
     */
    static std::shared_ptr<HyperparameterOptimizer> create(
        const std::string& baseConfigPath,
        const std::string& resultsDir = "optimization_results");
    
    /**
     * @brief Add common MNIST parameters to optimizer
     */
    static void addMNISTParameters(HyperparameterOptimizer& optimizer);
    
    /**
     * @brief Create MNIST objective function
     */
    static ObjectiveFunction createMNISTObjective(const std::string& baseConfigPath);
};

} // namespace snnfw

#endif // SNNFW_HYPERPARAMETER_OPTIMIZER_H


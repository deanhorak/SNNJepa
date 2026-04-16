#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <random>
#include <nlohmann/json.hpp>

namespace snnfw {

/**
 * @brief Parameter definition for optimization
 */
struct ParameterDef {
    enum class Type {
        INTEGER,
        DOUBLE,
        DISCRETE_INT,
        DISCRETE_DOUBLE
    };
    
    std::string name;           // JSON path (e.g., "/neuron/window_size_ms")
    Type type;
    
    // For continuous parameters
    double minValue;
    double maxValue;
    double step;                // For grid search
    
    // For discrete parameters
    std::vector<double> discreteValues;
    
    // Current value
    double currentValue;
    
    ParameterDef(const std::string& n, Type t, double min, double max, double s = 0.0)
        : name(n), type(t), minValue(min), maxValue(max), step(s), currentValue(min) {}
    
    ParameterDef(const std::string& n, const std::vector<double>& values)
        : name(n), type(Type::DISCRETE_DOUBLE), discreteValues(values), 
          currentValue(values.empty() ? 0.0 : values[0]) {}
};

/**
 * @brief Result of a single optimization trial
 */
struct OptimizationResult {
    int trialId;
    std::map<std::string, double> parameters;
    double accuracy;
    double trainingTime;
    double testingTime;
    int correctPredictions;
    int totalPredictions;
    std::string timestamp;
    
    nlohmann::json toJson() const;
    static OptimizationResult fromJson(const nlohmann::json& j);
};

/**
 * @brief Optimization strategy
 */
enum class OptimizationStrategy {
    GRID_SEARCH,        // Exhaustive grid search
    RANDOM_SEARCH,      // Random sampling
    COORDINATE_ASCENT,  // Optimize one parameter at a time
    ADAPTIVE_RANDOM     // Random search with adaptive sampling around best results
};

/**
 * @brief Configuration optimizer for automatic parameter tuning
 * 
 * This class manages the optimization process:
 * 1. Defines parameter search space
 * 2. Generates parameter configurations
 * 3. Tracks results and convergence
 * 4. Identifies best configurations
 */
class ConfigOptimizer {
public:
    /**
     * @brief Constructor
     * @param baseConfigPath Path to base configuration file
     * @param parameterSpacePath Path to parameter space definition JSON
     * @param resultsDir Directory to store optimization results
     */
    ConfigOptimizer(const std::string& baseConfigPath,
                   const std::string& parameterSpacePath,
                   const std::string& resultsDir);
    
    /**
     * @brief Load parameter space definition from JSON
     */
    void loadParameterSpace();
    
    /**
     * @brief Set optimization strategy
     */
    void setStrategy(OptimizationStrategy strategy);
    
    /**
     * @brief Set convergence criteria
     * @param maxTrials Maximum number of trials
     * @param minImprovement Minimum improvement to continue (e.g., 0.01 = 1%)
     * @param patienceTrials Number of trials without improvement before stopping
     */
    void setConvergenceCriteria(int maxTrials, double minImprovement, int patienceTrials);
    
    /**
     * @brief Generate next configuration to try
     * @return JSON configuration, or empty if optimization is complete
     */
    nlohmann::json generateNextConfig();
    
    /**
     * @brief Record result of a trial
     * @param result The optimization result
     */
    void recordResult(const OptimizationResult& result);
    
    /**
     * @brief Check if optimization should continue
     * @return true if more trials should be run
     */
    bool shouldContinue() const;
    
    /**
     * @brief Get best result so far
     */
    OptimizationResult getBestResult() const;
    
    /**
     * @brief Get all results
     */
    const std::vector<OptimizationResult>& getAllResults() const { return results_; }
    
    /**
     * @brief Save optimization state to disk
     */
    void saveState();
    
    /**
     * @brief Load optimization state from disk
     */
    void loadState();
    
    /**
     * @brief Generate summary report
     */
    std::string generateReport() const;
    
    /**
     * @brief Get current trial number
     */
    int getCurrentTrial() const { return currentTrial_; }
    
private:
    // Configuration paths
    std::string baseConfigPath_;
    std::string parameterSpacePath_;
    std::string resultsDir_;
    
    // Base configuration
    nlohmann::json baseConfig_;
    
    // Parameter space
    std::vector<ParameterDef> parameters_;
    
    // Optimization state
    OptimizationStrategy strategy_;
    int currentTrial_;
    int maxTrials_;
    double minImprovement_;
    int patienceTrials_;
    int trialsWithoutImprovement_;
    
    // Results
    std::vector<OptimizationResult> results_;
    double bestAccuracy_;
    int bestTrialId_;
    
    // Grid search state
    std::vector<size_t> gridIndices_;
    bool gridSearchComplete_;
    
    // Random search state
    std::mt19937 rng_;
    
    // Helper methods
    void initializeGridSearch();
    bool advanceGridSearch();
    nlohmann::json generateGridConfig();
    nlohmann::json generateRandomConfig();
    nlohmann::json generateCoordinateAscentConfig();
    nlohmann::json generateAdaptiveRandomConfig();
    
    void updateParametersFromGrid();
    void setParameterInJson(nlohmann::json& config, const std::string& path, double value);
    double getRandomValue(const ParameterDef& param);
    
    void saveResults();
    void loadResults();
};

} // namespace snnfw


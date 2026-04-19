/**
 * @file HyperparameterOptimizer.cpp
 * @brief Implementation of hyperparameter optimization framework
 */

#include "snnfw/HyperparameterOptimizer.h"
#include "snnfw/ConfigLoader.h"
#include "snnfw/Logger.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <filesystem>

namespace snnfw {

// ============================================================================
// ParameterConfig
// ============================================================================

nlohmann::json ParameterConfig::toJson() const {
    nlohmann::json j;
    j["doubleParams"] = doubleParams;
    j["intParams"] = intParams;
    j["stringParams"] = stringParams;
    return j;
}

ParameterConfig ParameterConfig::fromJson(const nlohmann::json& j) {
    ParameterConfig config;
    if (j.contains("doubleParams")) {
        config.doubleParams = j["doubleParams"].get<std::map<std::string, double>>();
    }
    if (j.contains("intParams")) {
        config.intParams = j["intParams"].get<std::map<std::string, int>>();
    }
    if (j.contains("stringParams")) {
        config.stringParams = j["stringParams"].get<std::map<std::string, std::string>>();
    }
    return config;
}

// ============================================================================
// ExperimentResult
// ============================================================================

nlohmann::json ExperimentResult::toJson() const {
    nlohmann::json j;
    j["config"] = config.toJson();
    j["score"] = score;
    j["trainingTime"] = trainingTime;
    j["testingTime"] = testingTime;
    j["metrics"] = metrics;
    j["timestamp"] = timestamp;
    return j;
}

ExperimentResult ExperimentResult::fromJson(const nlohmann::json& j) {
    ExperimentResult result;
    result.config = ParameterConfig::fromJson(j["config"]);
    result.score = j["score"];
    result.trainingTime = j.value("trainingTime", 0.0);
    result.testingTime = j.value("testingTime", 0.0);
    if (j.contains("metrics")) {
        result.metrics = j["metrics"].get<std::map<std::string, double>>();
    }
    result.timestamp = j.value("timestamp", "");
    return result;
}

// ============================================================================
// HyperparameterOptimizer
// ============================================================================

HyperparameterOptimizer::HyperparameterOptimizer(OptimizationStrategy strategy,
                                                 unsigned int seed)
    : strategy_(strategy)
    , maxIterations_(100)
    , numWorkers_(1)
    , resultsDir_("optimization_results")
    , saveIntermediateResults_(true)
    , rng_(seed) {
}

void HyperparameterOptimizer::addParameter(const ParameterSpec& param) {
    parameters_.push_back(param);
}

void HyperparameterOptimizer::setObjective(ObjectiveFunction objective) {
    objective_ = objective;
}

ExperimentResult HyperparameterOptimizer::optimize() {
    if (!objective_) {
        throw std::runtime_error("Objective function not set");
    }
    
    if (parameters_.empty()) {
        throw std::runtime_error("No parameters to optimize");
    }
    
    // Create results directory
    std::filesystem::create_directories(resultsDir_);
    
    SNNFW_INFO("Starting optimization with {} strategy", 
               strategy_ == OptimizationStrategy::GRID_SEARCH ? "Grid Search" :
               strategy_ == OptimizationStrategy::RANDOM_SEARCH ? "Random Search" :
               strategy_ == OptimizationStrategy::BAYESIAN ? "Bayesian" : "Genetic Algorithm");
    
    SNNFW_INFO("Optimizing {} parameters", parameters_.size());
    SNNFW_INFO("Max iterations: {}", maxIterations_);
    
    // Run optimization based on strategy
    ExperimentResult best;
    switch (strategy_) {
        case OptimizationStrategy::GRID_SEARCH:
            best = gridSearch();
            break;
        case OptimizationStrategy::RANDOM_SEARCH:
            best = randomSearch();
            break;
        case OptimizationStrategy::BAYESIAN:
            best = bayesianOptimization();
            break;
        case OptimizationStrategy::GENETIC_ALGORITHM:
            best = geneticAlgorithm();
            break;
    }
    
    SNNFW_INFO("Optimization complete. Best score: {}", best.score);
    
    // Save final results
    saveResults(resultsDir_ + "/final_results.json");
    
    return best;
}

ExperimentResult HyperparameterOptimizer::getBestResult() const {
    if (results_.empty()) {
        throw std::runtime_error("No results available");
    }
    
    auto best = std::max_element(results_.begin(), results_.end(),
        [](const ExperimentResult& a, const ExperimentResult& b) {
            return a.score < b.score;
        });
    
    return *best;
}

void HyperparameterOptimizer::saveResults(const std::string& filename) const {
    nlohmann::json j;
    j["strategy"] = static_cast<int>(strategy_);
    j["parameters"] = nlohmann::json::array();
    
    for (const auto& param : parameters_) {
        nlohmann::json p;
        p["name"] = param.name;
        p["type"] = static_cast<int>(param.type);
        if (param.type != ParameterSpec::Type::CATEGORICAL) {
            p["minValue"] = param.minValue;
            p["maxValue"] = param.maxValue;
            p["step"] = param.step;
        } else {
            p["categories"] = param.categories;
        }
        j["parameters"].push_back(p);
    }
    
    j["results"] = nlohmann::json::array();
    for (const auto& result : results_) {
        j["results"].push_back(result.toJson());
    }
    
    std::ofstream file(filename);
    file << j.dump(2);
    
    SNNFW_INFO("Saved {} results to {}", results_.size(), filename);
}

void HyperparameterOptimizer::loadResults(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    nlohmann::json j;
    file >> j;
    
    results_.clear();
    for (const auto& r : j["results"]) {
        results_.push_back(ExperimentResult::fromJson(r));
    }
    
    SNNFW_INFO("Loaded {} results from {}", results_.size(), filename);
}

ExperimentResult HyperparameterOptimizer::resume(const std::string& filename) {
    loadResults(filename);
    return optimize();
}

// ============================================================================
// Grid Search
// ============================================================================

ExperimentResult HyperparameterOptimizer::gridSearch() {
    auto configs = generateGridConfigs();
    
    SNNFW_INFO("Grid search: {} configurations to evaluate", configs.size());
    
    int iteration = 0;
    for (const auto& config : configs) {
        if (iteration >= maxIterations_) {
            SNNFW_INFO("Reached max iterations ({})", maxIterations_);
            break;
        }
        
        SNNFW_INFO("Iteration {}/{}: Evaluating configuration", 
                   iteration + 1, std::min(maxIterations_, static_cast<int>(configs.size())));
        
        auto result = objective_(config);
        result.timestamp = getCurrentTimestamp();
        results_.push_back(result);
        
        SNNFW_INFO("Score: {}", result.score);
        
        if (saveIntermediateResults_) {
            saveIntermediateResult(result);
        }
        
        iteration++;
    }
    
    return getBestResult();
}

std::vector<ParameterConfig> HyperparameterOptimizer::generateGridConfigs() {
    std::vector<ParameterConfig> configs;
    
    // Generate all combinations
    std::function<void(size_t, ParameterConfig)> generate;
    generate = [&](size_t paramIdx, ParameterConfig current) {
        if (paramIdx >= parameters_.size()) {
            configs.push_back(current);
            return;
        }
        
        const auto& param = parameters_[paramIdx];
        
        if (param.type == ParameterSpec::Type::INTEGER) {
            for (int val = param.minInt(); val <= param.maxInt(); val += param.stepInt()) {
                current.intParams[param.name] = val;
                generate(paramIdx + 1, current);
            }
        } else if (param.type == ParameterSpec::Type::DOUBLE) {
            for (double val = param.minValue; val <= param.maxValue; val += param.step) {
                current.doubleParams[param.name] = val;
                generate(paramIdx + 1, current);
            }
        } else { // CATEGORICAL
            for (const auto& cat : param.categories) {
                current.stringParams[param.name] = cat;
                generate(paramIdx + 1, current);
            }
        }
    };
    
    generate(0, ParameterConfig());
    return configs;
}

// ============================================================================
// Random Search
// ============================================================================

ExperimentResult HyperparameterOptimizer::randomSearch() {
    SNNFW_INFO("Random search: {} iterations", maxIterations_);
    
    for (int i = 0; i < maxIterations_; ++i) {
        SNNFW_INFO("Iteration {}/{}", i + 1, maxIterations_);
        
        auto config = generateRandomConfig();
        auto result = objective_(config);
        result.timestamp = getCurrentTimestamp();
        results_.push_back(result);
        
        SNNFW_INFO("Score: {}", result.score);
        
        if (saveIntermediateResults_) {
            saveIntermediateResult(result);
        }
    }
    
    return getBestResult();
}

ParameterConfig HyperparameterOptimizer::generateRandomConfig() {
    ParameterConfig config;
    
    for (const auto& param : parameters_) {
        if (param.type == ParameterSpec::Type::INTEGER) {
            std::uniform_int_distribution<int> dist(param.minInt(), param.maxInt());
            config.intParams[param.name] = dist(rng_);
        } else if (param.type == ParameterSpec::Type::DOUBLE) {
            std::uniform_real_distribution<double> dist(param.minValue, param.maxValue);
            config.doubleParams[param.name] = dist(rng_);
        } else { // CATEGORICAL
            std::uniform_int_distribution<size_t> dist(0, param.categories.size() - 1);
            config.stringParams[param.name] = param.categories[dist(rng_)];
        }
    }
    
    return config;
}

// ============================================================================
// Bayesian Optimization (Simplified)
// ============================================================================

ExperimentResult HyperparameterOptimizer::bayesianOptimization() {
    SNNFW_INFO("Bayesian optimization: {} iterations", maxIterations_);
    SNNFW_INFO("Note: Using simplified acquisition function (Expected Improvement)");
    
    // Start with random exploration
    int numInitial = std::min(10, maxIterations_ / 2);
    for (int i = 0; i < numInitial; ++i) {
        SNNFW_INFO("Initial exploration {}/{}", i + 1, numInitial);
        auto config = generateRandomConfig();
        auto result = objective_(config);
        result.timestamp = getCurrentTimestamp();
        results_.push_back(result);
        
        if (saveIntermediateResults_) {
            saveIntermediateResult(result);
        }
    }
    
    // Continue with exploitation/exploration balance
    for (int i = numInitial; i < maxIterations_; ++i) {
        SNNFW_INFO("Iteration {}/{}", i + 1, maxIterations_);
        
        // Sample from posterior (simplified: sample near best results)
        auto config = sampleFromPosterior();
        auto result = objective_(config);
        result.timestamp = getCurrentTimestamp();
        results_.push_back(result);
        
        SNNFW_INFO("Score: {}", result.score);
        
        if (saveIntermediateResults_) {
            saveIntermediateResult(result);
        }
    }
    
    return getBestResult();
}

ParameterConfig HyperparameterOptimizer::sampleFromPosterior() {
    // Simplified: Sample near best configurations with some noise
    if (results_.empty()) {
        return generateRandomConfig();
    }
    
    // Get top 3 results
    auto sortedResults = results_;
    std::sort(sortedResults.begin(), sortedResults.end(),
        [](const ExperimentResult& a, const ExperimentResult& b) {
            return a.score > b.score;
        });
    
    int numTop = std::min(3, static_cast<int>(sortedResults.size()));
    std::uniform_int_distribution<int> dist(0, numTop - 1);
    const auto& baseConfig = sortedResults[dist(rng_)].config;
    
    // Add noise
    return mutate(baseConfig, 0.2);
}

// ============================================================================
// Genetic Algorithm
// ============================================================================

ExperimentResult HyperparameterOptimizer::geneticAlgorithm() {
    SNNFW_INFO("Genetic algorithm: {} generations", maxIterations_);
    
    const int populationSize = 20;
    const double mutationRate = 0.1;
    const double eliteRatio = 0.2;
    
    // Initialize population
    std::vector<ParameterConfig> population;
    for (int i = 0; i < populationSize; ++i) {
        population.push_back(generateRandomConfig());
    }
    
    for (int gen = 0; gen < maxIterations_; ++gen) {
        SNNFW_INFO("Generation {}/{}", gen + 1, maxIterations_);
        
        // Evaluate population
        std::vector<std::pair<double, ParameterConfig>> scored;
        for (const auto& config : population) {
            auto result = objective_(config);
            result.timestamp = getCurrentTimestamp();
            results_.push_back(result);
            scored.push_back({result.score, config});
            
            if (saveIntermediateResults_) {
                saveIntermediateResult(result);
            }
        }
        
        // Sort by score
        std::sort(scored.begin(), scored.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        
        SNNFW_INFO("Best score in generation: {}", scored[0].first);
        
        // Create next generation
        std::vector<ParameterConfig> nextGen;
        
        // Elitism: keep top performers
        int numElite = static_cast<int>(populationSize * eliteRatio);
        for (int i = 0; i < numElite; ++i) {
            nextGen.push_back(scored[i].second);
        }
        
        // Crossover and mutation
        while (nextGen.size() < populationSize) {
            // Tournament selection
            std::uniform_int_distribution<int> dist(0, populationSize / 2);
            int idx1 = dist(rng_);
            int idx2 = dist(rng_);
            
            auto offspring = crossover(scored[idx1].second, scored[idx2].second);
            for (auto& child : offspring) {
                if (nextGen.size() < populationSize) {
                    nextGen.push_back(mutate(child, mutationRate));
                }
            }
        }
        
        population = nextGen;
    }
    
    return getBestResult();
}

std::vector<ParameterConfig> HyperparameterOptimizer::crossover(
    const ParameterConfig& parent1,
    const ParameterConfig& parent2) {
    
    ParameterConfig child1, child2;
    
    // Uniform crossover
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (const auto& [key, val] : parent1.intParams) {
        if (dist(rng_) < 0.5) {
            child1.intParams[key] = val;
            child2.intParams[key] = parent2.intParams.at(key);
        } else {
            child1.intParams[key] = parent2.intParams.at(key);
            child2.intParams[key] = val;
        }
    }
    
    for (const auto& [key, val] : parent1.doubleParams) {
        if (dist(rng_) < 0.5) {
            child1.doubleParams[key] = val;
            child2.doubleParams[key] = parent2.doubleParams.at(key);
        } else {
            child1.doubleParams[key] = parent2.doubleParams.at(key);
            child2.doubleParams[key] = val;
        }
    }
    
    for (const auto& [key, val] : parent1.stringParams) {
        if (dist(rng_) < 0.5) {
            child1.stringParams[key] = val;
            child2.stringParams[key] = parent2.stringParams.at(key);
        } else {
            child1.stringParams[key] = parent2.stringParams.at(key);
            child2.stringParams[key] = val;
        }
    }
    
    return {child1, child2};
}

ParameterConfig HyperparameterOptimizer::mutate(const ParameterConfig& config,
                                                double mutationRate) {
    ParameterConfig mutated = config;
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    
    for (const auto& param : parameters_) {
        if (prob(rng_) < mutationRate) {
            if (param.type == ParameterSpec::Type::INTEGER) {
                std::uniform_int_distribution<int> dist(param.minInt(), param.maxInt());
                mutated.intParams[param.name] = dist(rng_);
            } else if (param.type == ParameterSpec::Type::DOUBLE) {
                std::uniform_real_distribution<double> dist(param.minValue, param.maxValue);
                mutated.doubleParams[param.name] = dist(rng_);
            } else {
                std::uniform_int_distribution<size_t> dist(0, param.categories.size() - 1);
                mutated.stringParams[param.name] = param.categories[dist(rng_)];
            }
        }
    }
    
    return mutated;
}

// ============================================================================
// Helper Methods
// ============================================================================

void HyperparameterOptimizer::saveIntermediateResult(const ExperimentResult& result) {
    std::string filename = resultsDir_ + "/result_" + result.timestamp + ".json";
    std::ofstream file(filename);
    file << result.toJson().dump(2);
}

std::string HyperparameterOptimizer::getCurrentTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
    ss << "_" << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

// ============================================================================
// MNIST Optimizer
// ============================================================================

std::shared_ptr<HyperparameterOptimizer> MNISTOptimizer::create(
    const std::string& baseConfigPath,
    const std::string& resultsDir) {

    auto optimizer = std::make_shared<HyperparameterOptimizer>(
        OptimizationStrategy::RANDOM_SEARCH);

    optimizer->setResultsDir(resultsDir);
    addMNISTParameters(*optimizer);
    optimizer->setObjective(createMNISTObjective(baseConfigPath));

    return optimizer;
}

void MNISTOptimizer::addMNISTParameters(HyperparameterOptimizer& optimizer) {
    // Grid size
    ParameterSpec gridSize;
    gridSize.name = "network.grid_size";
    gridSize.type = ParameterSpec::Type::INTEGER;
    gridSize.minValue = 5;
    gridSize.maxValue = 9;
    gridSize.step = 1;
    optimizer.addParameter(gridSize);

    // Number of orientations
    ParameterSpec numOrientations;
    numOrientations.name = "network.num_orientations";
    numOrientations.type = ParameterSpec::Type::INTEGER;
    numOrientations.minValue = 4;
    numOrientations.maxValue = 12;
    numOrientations.step = 2;
    optimizer.addParameter(numOrientations);

    // Edge threshold
    ParameterSpec edgeThreshold;
    edgeThreshold.name = "network.edge_threshold";
    edgeThreshold.type = ParameterSpec::Type::DOUBLE;
    edgeThreshold.minValue = 0.05;
    edgeThreshold.maxValue = 0.3;
    edgeThreshold.step = 0.05;
    optimizer.addParameter(edgeThreshold);

    // Neuron similarity threshold
    ParameterSpec similarityThreshold;
    similarityThreshold.name = "neuron.similarity_threshold";
    similarityThreshold.type = ParameterSpec::Type::DOUBLE;
    similarityThreshold.minValue = 0.5;
    similarityThreshold.maxValue = 0.9;
    similarityThreshold.step = 0.1;
    optimizer.addParameter(similarityThreshold);

    // Max patterns per neuron
    ParameterSpec maxPatterns;
    maxPatterns.name = "neuron.max_patterns";
    maxPatterns.type = ParameterSpec::Type::INTEGER;
    maxPatterns.minValue = 20;
    maxPatterns.maxValue = 200;
    maxPatterns.step = 20;
    optimizer.addParameter(maxPatterns);

    // k neighbors for k-NN
    ParameterSpec kNeighbors;
    kNeighbors.name = "classification.k_neighbors";
    kNeighbors.type = ParameterSpec::Type::INTEGER;
    kNeighbors.minValue = 1;
    kNeighbors.maxValue = 15;
    kNeighbors.step = 2;
    optimizer.addParameter(kNeighbors);

    // Temporal window
    ParameterSpec temporalWindow;
    temporalWindow.name = "neuron.window_size_ms";
    temporalWindow.type = ParameterSpec::Type::DOUBLE;
    temporalWindow.minValue = 50.0;
    temporalWindow.maxValue = 300.0;
    temporalWindow.step = 50.0;
    optimizer.addParameter(temporalWindow);
}

ObjectiveFunction MNISTOptimizer::createMNISTObjective(const std::string& baseConfigPath) {
    return [baseConfigPath](const ParameterConfig& params) -> ExperimentResult {
        // This will be implemented in the optimization program
        // For now, return a placeholder
        ExperimentResult result;
        result.config = params;
        result.score = 0.0;
        result.trainingTime = 0.0;
        result.testingTime = 0.0;
        return result;
    };
}

} // namespace snnfw


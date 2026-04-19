#include "snnfw/ConfigOptimizer.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <chrono>
#include <random>
#include <algorithm>
#include <filesystem>
#include <iostream>

namespace snnfw {

namespace fs = std::filesystem;

// ============================================================================
// OptimizationResult implementation
// ============================================================================

nlohmann::json OptimizationResult::toJson() const {
    nlohmann::json j;
    j["trial_id"] = trialId;
    j["parameters"] = parameters;
    j["accuracy"] = accuracy;
    j["training_time"] = trainingTime;
    j["testing_time"] = testingTime;
    j["correct_predictions"] = correctPredictions;
    j["total_predictions"] = totalPredictions;
    j["timestamp"] = timestamp;
    return j;
}

OptimizationResult OptimizationResult::fromJson(const nlohmann::json& j) {
    OptimizationResult result;
    result.trialId = j["trial_id"];
    result.parameters = j["parameters"].get<std::map<std::string, double>>();
    result.accuracy = j["accuracy"];
    result.trainingTime = j["training_time"];
    result.testingTime = j["testing_time"];
    result.correctPredictions = j["correct_predictions"];
    result.totalPredictions = j["total_predictions"];
    result.timestamp = j["timestamp"];
    return result;
}

// ============================================================================
// ConfigOptimizer implementation
// ============================================================================

ConfigOptimizer::ConfigOptimizer(const std::string& baseConfigPath,
                                 const std::string& parameterSpacePath,
                                 const std::string& resultsDir)
    : baseConfigPath_(baseConfigPath)
    , parameterSpacePath_(parameterSpacePath)
    , resultsDir_(resultsDir)
    , strategy_(OptimizationStrategy::RANDOM_SEARCH)
    , currentTrial_(0)
    , maxTrials_(100)
    , minImprovement_(0.001)  // 0.1%
    , patienceTrials_(10)
    , trialsWithoutImprovement_(0)
    , bestAccuracy_(0.0)
    , bestTrialId_(-1)
    , gridSearchComplete_(false)
    , rng_(std::random_device{}())
{
    // Create results directory if it doesn't exist
    fs::create_directories(resultsDir_);
    
    // Load base configuration
    std::ifstream configFile(baseConfigPath_);
    if (!configFile.is_open()) {
        throw std::runtime_error("Failed to open base config: " + baseConfigPath_);
    }
    configFile >> baseConfig_;
}

void ConfigOptimizer::loadParameterSpace() {
    std::ifstream paramFile(parameterSpacePath_);
    if (!paramFile.is_open()) {
        throw std::runtime_error("Failed to open parameter space: " + parameterSpacePath_);
    }
    
    nlohmann::json paramSpace;
    paramFile >> paramSpace;
    
    // Parse parameter definitions
    for (const auto& param : paramSpace["parameters"]) {
        std::string name = param["name"];
        std::string type = param["type"];
        
        if (type == "integer_range") {
            int min = param["min"];
            int max = param["max"];
            int step = param.value("step", 1);
            parameters_.emplace_back(name, ParameterDef::Type::INTEGER, min, max, step);
        }
        else if (type == "double_range") {
            double min = param["min"];
            double max = param["max"];
            double step = param.value("step", 0.0);
            parameters_.emplace_back(name, ParameterDef::Type::DOUBLE, min, max, step);
        }
        else if (type == "discrete") {
            std::vector<double> values = param["values"].get<std::vector<double>>();
            parameters_.emplace_back(name, values);
        }
    }
    
    std::cout << "Loaded " << parameters_.size() << " parameters for optimization" << std::endl;
}

void ConfigOptimizer::setStrategy(OptimizationStrategy strategy) {
    strategy_ = strategy;
    
    if (strategy_ == OptimizationStrategy::GRID_SEARCH) {
        initializeGridSearch();
    }
}

void ConfigOptimizer::setConvergenceCriteria(int maxTrials, double minImprovement, int patienceTrials) {
    maxTrials_ = maxTrials;
    minImprovement_ = minImprovement;
    patienceTrials_ = patienceTrials;
}

void ConfigOptimizer::initializeGridSearch() {
    gridIndices_.clear();
    gridIndices_.resize(parameters_.size(), 0);
    gridSearchComplete_ = false;
}

bool ConfigOptimizer::advanceGridSearch() {
    // Increment grid indices (like a multi-digit counter)
    for (int i = parameters_.size() - 1; i >= 0; --i) {
        const auto& param = parameters_[i];
        size_t maxIndex;
        
        if (param.type == ParameterDef::Type::DISCRETE_DOUBLE ||
            param.type == ParameterDef::Type::DISCRETE_INT) {
            maxIndex = param.discreteValues.size() - 1;
        } else {
            // Calculate number of steps
            maxIndex = static_cast<size_t>((param.maxValue - param.minValue) / param.step);
        }
        
        if (gridIndices_[i] < maxIndex) {
            gridIndices_[i]++;
            return true;
        } else {
            gridIndices_[i] = 0;
        }
    }
    
    gridSearchComplete_ = true;
    return false;
}

void ConfigOptimizer::updateParametersFromGrid() {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param = parameters_[i];
        size_t idx = gridIndices_[i];
        
        if (param.type == ParameterDef::Type::DISCRETE_DOUBLE ||
            param.type == ParameterDef::Type::DISCRETE_INT) {
            param.currentValue = param.discreteValues[idx];
        } else {
            param.currentValue = param.minValue + idx * param.step;
        }
    }
}

double ConfigOptimizer::getRandomValue(const ParameterDef& param) {
    if (param.type == ParameterDef::Type::DISCRETE_DOUBLE ||
        param.type == ParameterDef::Type::DISCRETE_INT) {
        std::uniform_int_distribution<size_t> dist(0, param.discreteValues.size() - 1);
        return param.discreteValues[dist(rng_)];
    } else if (param.type == ParameterDef::Type::INTEGER) {
        std::uniform_int_distribution<int> dist(
            static_cast<int>(param.minValue),
            static_cast<int>(param.maxValue)
        );
        return static_cast<double>(dist(rng_));
    } else {
        std::uniform_real_distribution<double> dist(param.minValue, param.maxValue);
        return dist(rng_);
    }
}

void ConfigOptimizer::setParameterInJson(nlohmann::json& config, const std::string& path, double value) {
    // Parse JSON path (e.g., "/neuron/window_size_ms")
    std::vector<std::string> parts;
    std::stringstream ss(path);
    std::string part;
    
    while (std::getline(ss, part, '/')) {
        if (!part.empty()) {
            parts.push_back(part);
        }
    }
    
    // Navigate to the correct location and set value
    nlohmann::json* current = &config;
    for (size_t i = 0; i < parts.size() - 1; ++i) {
        if (!current->contains(parts[i])) {
            (*current)[parts[i]] = nlohmann::json::object();
        }
        current = &(*current)[parts[i]];
    }
    
    // Set the value (convert to int if parameter is integer type)
    const std::string& lastPart = parts.back();
    for (const auto& param : parameters_) {
        if (param.name == path) {
            if (param.type == ParameterDef::Type::INTEGER ||
                param.type == ParameterDef::Type::DISCRETE_INT) {
                (*current)[lastPart] = static_cast<int>(value);
            } else {
                (*current)[lastPart] = value;
            }
            break;
        }
    }
}

nlohmann::json ConfigOptimizer::generateGridConfig() {
    updateParametersFromGrid();
    
    nlohmann::json config = baseConfig_;
    for (const auto& param : parameters_) {
        setParameterInJson(config, param.name, param.currentValue);
    }
    
    return config;
}

nlohmann::json ConfigOptimizer::generateRandomConfig() {
    nlohmann::json config = baseConfig_;
    
    for (auto& param : parameters_) {
        param.currentValue = getRandomValue(param);
        setParameterInJson(config, param.name, param.currentValue);
    }
    
    return config;
}

nlohmann::json ConfigOptimizer::generateCoordinateAscentConfig() {
    // Start from best configuration
    if (bestTrialId_ >= 0) {
        const auto& bestResult = results_[bestTrialId_];
        for (auto& param : parameters_) {
            if (bestResult.parameters.count(param.name)) {
                param.currentValue = bestResult.parameters.at(param.name);
            }
        }
    }
    
    // Vary one parameter at a time
    int paramIndex = currentTrial_ % parameters_.size();
    parameters_[paramIndex].currentValue = getRandomValue(parameters_[paramIndex]);
    
    nlohmann::json config = baseConfig_;
    for (const auto& param : parameters_) {
        setParameterInJson(config, param.name, param.currentValue);
    }
    
    return config;
}

nlohmann::json ConfigOptimizer::generateAdaptiveRandomConfig() {
    // Use best results to guide sampling
    nlohmann::json config = baseConfig_;

    for (auto& param : parameters_) {
        if (bestTrialId_ >= 0 && results_.size() > 5) {
            // For discrete parameters, sample from nearby values
            if (param.type == ParameterDef::Type::DISCRETE_INT ||
                param.type == ParameterDef::Type::DISCRETE_DOUBLE) {

                const auto& bestResult = results_[bestTrialId_];
                double bestValue = bestResult.parameters.at(param.name);

                // Find index of best value in discrete set
                size_t bestIdx = 0;
                for (size_t i = 0; i < param.discreteValues.size(); ++i) {
                    if (std::abs(param.discreteValues[i] - bestValue) < 1e-6) {
                        bestIdx = i;
                        break;
                    }
                }

                // Sample from nearby indices with decreasing window
                int window = std::max(1, static_cast<int>(param.discreteValues.size() / (2.0 + currentTrial_ / 10.0)));
                int minIdx = std::max(0, static_cast<int>(bestIdx) - window);
                int maxIdx = std::min(static_cast<int>(param.discreteValues.size()) - 1, static_cast<int>(bestIdx) + window);

                std::uniform_int_distribution<int> dist(minIdx, maxIdx);
                param.currentValue = param.discreteValues[dist(rng_)];
            } else {
                // For continuous parameters, use Gaussian sampling
                const auto& bestResult = results_[bestTrialId_];
                double bestValue = bestResult.parameters.at(param.name);
                double range = (param.maxValue - param.minValue) / (1.0 + currentTrial_ / 10.0);

                std::normal_distribution<double> dist(bestValue, range * 0.2);
                param.currentValue = std::clamp(dist(rng_), param.minValue, param.maxValue);
            }
        } else {
            param.currentValue = getRandomValue(param);
        }

        setParameterInJson(config, param.name, param.currentValue);
    }

    return config;
}

nlohmann::json ConfigOptimizer::generateNextConfig() {
    if (!shouldContinue()) {
        return nlohmann::json();
    }
    
    currentTrial_++;
    
    switch (strategy_) {
        case OptimizationStrategy::GRID_SEARCH:
            if (gridSearchComplete_) {
                return nlohmann::json();
            }
            return generateGridConfig();
            
        case OptimizationStrategy::RANDOM_SEARCH:
            return generateRandomConfig();
            
        case OptimizationStrategy::COORDINATE_ASCENT:
            return generateCoordinateAscentConfig();
            
        case OptimizationStrategy::ADAPTIVE_RANDOM:
            return generateAdaptiveRandomConfig();
    }
    
    return nlohmann::json();
}

void ConfigOptimizer::recordResult(const OptimizationResult& result) {
    results_.push_back(result);
    
    // Check if this is a new best
    if (result.accuracy > bestAccuracy_) {
        double improvement = result.accuracy - bestAccuracy_;
        
        if (improvement >= minImprovement_) {
            trialsWithoutImprovement_ = 0;
        } else {
            trialsWithoutImprovement_++;
        }
        
        bestAccuracy_ = result.accuracy;
        bestTrialId_ = results_.size() - 1;
        
        std::cout << "🎉 New best accuracy: " << (bestAccuracy_ * 100.0) << "% "
                  << "(improvement: +" << (improvement * 100.0) << "%)" << std::endl;
    } else {
        trialsWithoutImprovement_++;
    }
    
    // Advance grid search if using that strategy
    if (strategy_ == OptimizationStrategy::GRID_SEARCH) {
        advanceGridSearch();
    }
    
    // Save results periodically
    if (results_.size() % 5 == 0) {
        saveResults();
    }
}

bool ConfigOptimizer::shouldContinue() const {
    if (currentTrial_ >= maxTrials_) {
        return false;
    }
    
    if (trialsWithoutImprovement_ >= patienceTrials_) {
        return false;
    }
    
    if (strategy_ == OptimizationStrategy::GRID_SEARCH && gridSearchComplete_) {
        return false;
    }
    
    return true;
}

OptimizationResult ConfigOptimizer::getBestResult() const {
    if (bestTrialId_ < 0 || bestTrialId_ >= static_cast<int>(results_.size())) {
        throw std::runtime_error("No results available");
    }
    return results_[bestTrialId_];
}

void ConfigOptimizer::saveResults() {
    std::string resultsPath = resultsDir_ + "/optimization_results.json";
    
    nlohmann::json j;
    j["results"] = nlohmann::json::array();
    
    for (const auto& result : results_) {
        j["results"].push_back(result.toJson());
    }
    
    std::ofstream outFile(resultsPath);
    outFile << j.dump(2);
}

void ConfigOptimizer::loadResults() {
    std::string resultsPath = resultsDir_ + "/optimization_results.json";
    
    std::ifstream inFile(resultsPath);
    if (!inFile.is_open()) {
        return;  // No previous results
    }
    
    nlohmann::json j;
    inFile >> j;
    
    results_.clear();
    for (const auto& resultJson : j["results"]) {
        results_.push_back(OptimizationResult::fromJson(resultJson));
    }
    
    // Find best result
    for (size_t i = 0; i < results_.size(); ++i) {
        if (results_[i].accuracy > bestAccuracy_) {
            bestAccuracy_ = results_[i].accuracy;
            bestTrialId_ = i;
        }
    }
    
    currentTrial_ = results_.size();
}

void ConfigOptimizer::saveState() {
    saveResults();
    
    // Save optimizer state
    std::string statePath = resultsDir_ + "/optimizer_state.json";
    nlohmann::json j;
    j["current_trial"] = currentTrial_;
    j["best_accuracy"] = bestAccuracy_;
    j["best_trial_id"] = bestTrialId_;
    j["trials_without_improvement"] = trialsWithoutImprovement_;
    
    std::ofstream outFile(statePath);
    outFile << j.dump(2);
}

void ConfigOptimizer::loadState() {
    loadResults();
    
    std::string statePath = resultsDir_ + "/optimizer_state.json";
    std::ifstream inFile(statePath);
    if (!inFile.is_open()) {
        return;
    }
    
    nlohmann::json j;
    inFile >> j;
    
    currentTrial_ = j["current_trial"];
    bestAccuracy_ = j["best_accuracy"];
    bestTrialId_ = j["best_trial_id"];
    trialsWithoutImprovement_ = j["trials_without_improvement"];
}

std::string ConfigOptimizer::generateReport() const {
    std::stringstream ss;
    
    ss << "=== Optimization Report ===\n\n";
    ss << "Strategy: ";
    switch (strategy_) {
        case OptimizationStrategy::GRID_SEARCH: ss << "Grid Search\n"; break;
        case OptimizationStrategy::RANDOM_SEARCH: ss << "Random Search\n"; break;
        case OptimizationStrategy::COORDINATE_ASCENT: ss << "Coordinate Ascent\n"; break;
        case OptimizationStrategy::ADAPTIVE_RANDOM: ss << "Adaptive Random\n"; break;
    }
    
    ss << "Trials completed: " << results_.size() << " / " << maxTrials_ << "\n";
    ss << "Best accuracy: " << (bestAccuracy_ * 100.0) << "%\n";
    ss << "Trials without improvement: " << trialsWithoutImprovement_ << " / " << patienceTrials_ << "\n\n";
    
    if (bestTrialId_ >= 0) {
        const auto& best = results_[bestTrialId_];
        ss << "Best configuration (Trial #" << best.trialId << "):\n";
        for (const auto& [name, value] : best.parameters) {
            ss << "  " << name << ": " << value << "\n";
        }
        ss << "  Accuracy: " << (best.accuracy * 100.0) << "%\n";
        ss << "  Training time: " << best.trainingTime << "s\n";
        ss << "  Testing time: " << best.testingTime << "s\n";
    }
    
    return ss.str();
}

} // namespace snnfw


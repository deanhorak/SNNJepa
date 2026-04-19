#include "snnfw/experiment/ExperimentRunner.h"
#include "snnfw/adapters/AdapterFactory.h"
#include "snnfw/adapters/RetinaAdapter.h"
#include "snnfw/adapters/InterneuronAdapters.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <stdexcept>

namespace snnfw {
namespace experiment {

namespace {

void ensureCoreSensoryAdaptersRegistered() {
    auto& factory = adapters::AdapterFactory::getInstance();
    if (!factory.hasSensoryAdapter("retina")) {
        factory.registerSensoryAdapter(
            "retina",
            [](const adapters::BaseAdapter::Config& cfg) {
                return std::make_shared<adapters::RetinaAdapter>(cfg);
            });
    }
}

std::string joinColumnFrequencies(
    const std::vector<declarative::ConstructedNetwork::ColumnGroup>& columns) {
    std::vector<double> frequencies;
    frequencies.reserve(columns.size());
    for (const auto& col : columns) {
        frequencies.push_back(col.spatialFrequency);
    }
    std::sort(frequencies.begin(), frequencies.end());
    frequencies.erase(
        std::unique(frequencies.begin(), frequencies.end(),
                    [](double a, double b) { return std::abs(a - b) < 1e-6; }),
        frequencies.end());

    std::ostringstream ss;
    for (size_t i = 0; i < frequencies.size(); ++i) {
        if (i > 0) {
            ss << ",";
        }
        ss << frequencies[i];
    }
    return ss.str();
}

} // namespace

ExperimentRunner::ExperimentRunner(const ExperimentConfig& config)
    : config_(config)
{
}

double ExperimentRunner::run() {
    std::cout << "=== EMNIST Letters Training (SONATA Model) ===" << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();

    // Step 1: Load data
    std::cout << "\n[1/3] Loading EMNIST data..." << std::endl;
    loadData();

    // Step 2: Build network from config
    std::cout << "\n[2/3] Building network from " << config_.networkConfigPath << "..." << std::endl;
    buildNetwork();

    // Step 3: Run training pipeline
    std::cout << "\n[3/3] Starting training pipeline..." << std::endl;
    pipeline_ = std::make_unique<TrainingPipeline>(config_, *network_);
    double finalAccuracy = pipeline_->run(*trainLoader_, *testLoader_);
    passAccuracies_ = pipeline_->getPassAccuracies();

    auto endTime = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double>(endTime - startTime).count();

    std::cout << "\n=== Experiment Complete ===" << std::endl;
    std::cout << "  Final accuracy: " << std::fixed << std::setprecision(2)
              << finalAccuracy << "%" << std::endl;
    std::cout << "  Total passes: " << passAccuracies_.size() << std::endl;
    std::cout << "  Total patterns learned: " << pipeline_->getTotalPatternsLearned() << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(1) << totalTime << "s" << std::endl;

    // Print per-pass accuracies
    std::cout << "  Per-pass accuracies: ";
    for (size_t i = 0; i < passAccuracies_.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(2) << passAccuracies_[i] << "%";
    }
    std::cout << std::endl;

    return finalAccuracy;
}

void ExperimentRunner::loadData() {
    trainLoader_ = std::make_unique<EMNISTLoader>();
    testLoader_ = std::make_unique<EMNISTLoader>();

    if (!trainLoader_->load(config_.trainImagesPath, config_.trainLabelsPath)) {
        throw std::runtime_error("Failed to load training data from: " +
                                 config_.trainImagesPath);
    }
    std::cout << "  Training set: " << trainLoader_->size() << " images" << std::endl;

    if (!testLoader_->load(config_.testImagesPath, config_.testLabelsPath)) {
        throw std::runtime_error("Failed to load test data from: " +
                                 config_.testImagesPath);
    }
    std::cout << "  Test set: " << testLoader_->size() << " images" << std::endl;
}

void ExperimentRunner::buildNetwork() {
    // Create datastore and factory first (needed by DeclarativeLoader)
    datastore_ = std::make_unique<Datastore>(config_.datastorePath);
    factory_ = std::make_unique<NeuralObjectFactory>();

    // Load and parse the network description
    declarative::DeclarativeLoader loader(*factory_, *datastore_);
    auto ir = loader.parseOnly(config_.networkConfigPath);

    // Sync config from the IR
    syncConfigFromIR(ir);

    // Make declarative connectivity construction deterministic when --seed is set.
    if (config_.seed != 0) {
        ir.simulation.connectivitySeed = config_.seed;
    }

    // Construct the network from the parsed IR
    network_ = std::make_unique<declarative::ConstructedNetwork>(loader.loadFromIR(ir));
    instantiateAdapters(ir);

    std::cout << "  Network built: " << network_->inputNeurons.size() << " input neurons, "
              << network_->columns.size() << " columns, "
              << network_->outputPopulations.size() << " output classes" << std::endl;
    std::cout << "  Total neurons: " << network_->allNeuronIds.size() << std::endl;
    std::cout << "  Total synapses: " << network_->allSynapses.size() << std::endl;
    if (!config_.classificationType.empty()) {
        std::cout << "  Readout classifier: " << config_.classificationType
                  << " (k=" << config_.knnK << ", exponent=" << config_.knnSimilarityExponent
                  << ")" << std::endl;
    }

    // Start the spike processor
    network_->spikeProcessor->start();
    std::cout << "  Spike processor started" << std::endl;
}

void ExperimentRunner::syncConfigFromIR(const declarative::NetworkIR& ir) {
    // Sync encoder/input parameters from the parsed IR
    config_.inputRows = ir.inputLayer.rows;
    config_.inputCols = ir.inputLayer.cols;
    config_.pixelThreshold = ir.inputLayer.pixelThreshold;
    config_.inputLatencyMs = ir.inputLayer.latencyMs;
    if (!config_.hasSaccadeRuntimeOverride) {
        config_.enableSaccades = ir.saccades.enabled;
    }
    config_.saccadeNumFixations = std::max(1, ir.saccades.numFixations);
    config_.saccadeRegions.clear();
    config_.saccadeRegions.reserve(ir.saccades.regions.size());
    for (const auto& region : ir.saccades.regions) {
        ExperimentConfig::FixationRegion fix;
        fix.name = region.name;
        fix.rowStart = region.rowStart;
        fix.rowEnd = region.rowEnd;
        fix.colStart = region.colStart;
        fix.colEnd = region.colEnd;
        config_.saccadeRegions.push_back(std::move(fix));
    }

    // Sync architecture parameters from the parsed IR
    config_.numClasses = ir.outputLayer.numClasses;
    config_.neuronsPerOutputClass = ir.outputLayer.neuronsPerClass;
    config_.numColumns = 0;

    // Count columns from the brain hierarchy
    for (auto& hemi : ir.brain.hemispheres) {
        for (auto& lobe : hemi.lobes) {
            for (auto& region : lobe.regions) {
                for (auto& nucleus : region.nuclei) {
                    config_.numColumns += static_cast<int>(nucleus.columns.size());
                    if (nucleus.columnTemplate.has_value()) {
                        auto& tmpl = nucleus.columnTemplate.value();
                        config_.numColumns += static_cast<int>(
                            tmpl.orientations.size() * tmpl.frequencies.size());
                    }
                }
            }
        }
    }

    // Sync simulation parameters
    config_.interImageGapMs = ir.simulation.interImageGapMs;
    config_.l4Keep = ir.simulation.l4Keep;
    config_.l5Keep = ir.simulation.l5Keep;
    if (!config_.hasSimilarityRuntimeOverrides) {
        config_.enableSimilarityCompetition = ir.simulation.enableSimilarityCompetition;
        config_.l4SimilarityWeight = ir.simulation.l4SimilarityWeight;
        config_.l5SimilarityWeight = ir.simulation.l5SimilarityWeight;
        config_.traceSimilarityCompetition = ir.simulation.traceSimilarityCompetition;
    }
    config_.enableL5Inhibition = ir.simulation.enableL5Inhibition;
    config_.enableL5InterColumnInhibition = ir.simulation.enableL5InterColumnInhibition;
    config_.l5InterColumnInhibit = ir.simulation.l5InterColumnInhibit;
    config_.l5InterColumnMinOverlap = ir.simulation.l5InterColumnMinOverlap;
    config_.l5InterColumnWinnerScale = ir.simulation.l5InterColumnWinnerScale;
    config_.l5InterColumnMaxInhibit = ir.simulation.l5InterColumnMaxInhibit;
    config_.l5InterColumnMaxOrientationDeltaDeg =
        ir.simulation.l5InterColumnMaxOrientationDeltaDeg;
    config_.l5InterColumnMaxFrequencyOctaveDelta =
        ir.simulation.l5InterColumnMaxFrequencyOctaveDelta;
    config_.l5InterColumnMaxNeighbors = ir.simulation.l5InterColumnMaxNeighbors;
    config_.maskMinActive = ir.simulation.maskMinActive;
    config_.stdpLtdScale = ir.simulation.stdpLtdScale;
    config_.stdpLtdWindowMs = ir.simulation.stdpLtdWindowMs;
    config_.traceStdp = ir.simulation.traceStdp;
    config_.enableStdpEligibilityGate = ir.simulation.enableStdpEligibilityGate;
    config_.stdpEligibilityMinUpdates = ir.simulation.stdpEligibilityMinUpdates;
    config_.stdpEligibilityMinLtp = ir.simulation.stdpEligibilityMinLtp;
    config_.stdpEligibilityThreshold = ir.simulation.stdpEligibilityThreshold;
    config_.stdpEligibilityLtdPenalty = ir.simulation.stdpEligibilityLtdPenalty;
    config_.numThreads = ir.simulation.spikeProcessorThreads;

    if (!ir.classification.type.empty()) {
        config_.classificationType = ir.classification.type;
        config_.knnK = ir.classification.k;
        config_.knnSimilarityExponent = ir.classification.distanceExponent;
        config_.classificationIntParams = ir.classification.intParams;
        config_.classificationDoubleParams = ir.classification.doubleParams;
        config_.classificationStringParams = ir.classification.stringParams;

        std::string normalizedType = ir.classification.type;
        std::transform(normalizedType.begin(), normalizedType.end(), normalizedType.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (normalizedType == "majority" || normalizedType == "majority_voting") {
            config_.enableKnnSimilarityWeightedVote = false;
        } else if (normalizedType == "weighted_similarity") {
            config_.enableKnnSimilarityWeightedVote = true;
        }
    }

    // Determine layer sizes from column template or first column
    for (auto& hemi : ir.brain.hemispheres) {
        for (auto& lobe : hemi.lobes) {
            for (auto& region : lobe.regions) {
                for (auto& nucleus : region.nuclei) {
                    std::vector<declarative::LayerIR> const* layers = nullptr;
                    if (nucleus.columnTemplate.has_value()) {
                        layers = &nucleus.columnTemplate.value().layers;
                    } else if (!nucleus.columns.empty()) {
                        layers = &nucleus.columns.front().layers;
                    }
                    if (layers) {
                        for (auto& layer : *layers) {
                            for (auto& pop : layer.populations) {
                                if (layer.name == "L4" && !pop.gridLayout.empty()) {
                                    // Parse "7x7" -> 7
                                    auto xPos = pop.gridLayout.find('x');
                                    if (xPos != std::string::npos) {
                                        config_.layer4Size = std::stoi(
                                            pop.gridLayout.substr(0, xPos));
                                    }
                                } else if (layer.name == "L5") {
                                    config_.layer5Neurons = pop.count;
                                }
                            }
                        }
                        return; // Got what we need
                    }
                }
            }
        }
    }
}

void ExperimentRunner::instantiateAdapters(const declarative::NetworkIR& ir) {
    if (!network_) {
        throw std::runtime_error("Cannot instantiate adapters before network is constructed");
    }

    adapters::ensureInterneuronAdaptersRegistered();
    ensureCoreSensoryAdaptersRegistered();
    auto& factory = adapters::AdapterFactory::getInstance();
    for (const auto& cfgIR : ir.adapters) {
        adapters::BaseAdapter::Config cfg;
        cfg.name = cfgIR.name;
        cfg.type = cfgIR.type;
        cfg.temporalWindow = cfgIR.temporalWindowMs;
        cfg.doubleParams = cfgIR.doubleParams;
        cfg.intParams = cfgIR.intParams;
        cfg.stringParams = cfgIR.stringParams;
        if (cfgIR.type == "retina" && cfgIR.bindTo == "l4") {
            cfg.stringParams["frequency_values"] = joinColumnFrequencies(network_->columns);
        }

        std::string role = cfgIR.role;
        if (role.empty()) {
            if (cfgIR.type == "interneuron_rx") {
                role = "sensory";
            } else if (cfgIR.type == "interneuron_tx") {
                role = "motor";
            }
        }
        if (!cfgIR.bindTo.empty()) {
            cfg.stringParams["bind_to"] = cfgIR.bindTo;
        }

        if (role == "sensory") {
            auto adapter = factory.createSensoryAdapter(cfg);
            if (!adapter) {
                throw std::runtime_error("No sensory adapter registered for type '" + cfg.type + "'");
            }
            if (!adapter->initialize()) {
                throw std::runtime_error("Failed to initialize sensory adapter '" + cfg.name + "'");
            }
            network_->sensoryAdapters.push_back(std::move(adapter));
            continue;
        }
        if (role == "motor") {
            auto adapter = factory.createMotorAdapter(cfg);
            if (!adapter) {
                throw std::runtime_error("No motor adapter registered for type '" + cfg.type + "'");
            }
            if (!adapter->initialize()) {
                throw std::runtime_error("Failed to initialize motor adapter '" + cfg.name + "'");
            }
            network_->motorAdapters.push_back(std::move(adapter));
            continue;
        }
        throw std::runtime_error("Adapter '" + cfg.name + "' has unresolved role");
    }

    if (!ir.adapters.empty()) {
        std::cout << "  Adapters initialized: "
                  << network_->sensoryAdapters.size() << " sensory, "
                  << network_->motorAdapters.size() << " motor" << std::endl;
    }
}

const std::vector<double>& ExperimentRunner::getPassAccuracies() const {
    return passAccuracies_;
}

} // namespace experiment
} // namespace snnfw

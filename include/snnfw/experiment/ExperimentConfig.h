#ifndef SNNFW_EXPERIMENT_CONFIG_H
#define SNNFW_EXPERIMENT_CONFIG_H

#include <string>
#include <vector>
#include <array>
#include <map>

namespace snnfw {
namespace experiment {

/**
 * @brief Configuration for an SNN classification experiment.
 *
 * Extracted from the hardcoded TrainingConfig in emnist_letters_training.cpp.
 * All parameters that control the training/testing pipeline live here.
 */
struct ExperimentConfig {
    struct FixationRegion {
        std::string name;
        int rowStart = 0;
        int rowEnd = 0;
        int colStart = 0;
        int colEnd = 0;
    };

    // --- Data paths ---
    std::string trainImagesPath;
    std::string trainLabelsPath;
    std::string testImagesPath;
    std::string testLabelsPath;
    std::string datastorePath = "./experiment_db";
    std::string networkConfigPath;  // SONATA / JSON config file

    // --- Random seed ---
    unsigned int seed = 0;

    // --- Neuron parameters ---
    double neuronWindow = 500.0;
    double neuronThreshold = 1.2;
    int neuronMaxPatterns = 500;
    double inputLatencyMs = 15.0;

    // --- Training parameters ---
    int numClasses = 26;
    int trainingExamplesPerClass = 200;
    int maxTrainingImages = 5200;
    int maxPasses = 20;

    // --- Spike processor ---
    int numThreads = 24;

    // --- Architecture (read from SONATA, but kept here for pipeline logic) ---
    int numColumns = 16;
    int layer4Size = 7;       // grid side length (7×7 = 49)
    int layer5Neurons = 80;

    // --- Synaptic parameters ---
    double initialWeight = 0.1;
    double maxWeight = 0.5;
    double outputConnectProb = 0.02;
    double outputInitialWeight = 0.02;

    // --- Input / competition ---
    int inputRows = 28;
    int inputCols = 28;
    double pixelThreshold = 0.4;
    bool enableVisualFrontend = false;
    double visualFrontendGain = 0.45;
    double visualOnCenterWeight = 0.60;
    double visualEdgeWeight = 0.40;
    bool enableVisualFeatureChannels = false;
    double visualFeatureThreshold = 0.58;
    double visualOnPartitionThreshold = 0.18;
    double visualOffPartitionThreshold = 0.28;
    double visualEdgePartitionThreshold = 0.60;
    int maskMinActive = 6;
    int tilesPerColumn = 3;
    int l4Keep = 8;
    int l5Keep = 8;
    bool enableSimilarityCompetition = true;
    double l4IncomingWeight = 0.0;
    double l4ActivationWeight = 0.0;
    double l4SimilarityWeight = 0.25;
    double l5IncomingWeight = 0.0;
    double l5ActivationWeight = 0.0;
    double l5SimilarityWeight = 0.60;
    bool traceSimilarityCompetition = false;
    bool hasSimilarityRuntimeOverrides = false;
    bool hasSaccadeRuntimeOverride = false;
    double l4RowDelay = 0.3;
    double l4ColDelay = 0.2;
    double interImageGapMs = 550.0;
    double l4PostShiftMs = -6.0;
    double l4PostJitterMs = 2.0;
    bool enableSaccades = false;
    int saccadeNumFixations = 1;
    std::vector<FixationRegion> saccadeRegions;
    int saccadeMaxRegionsPerImage = 2;
    int saccadeAttentionGrid = 4;
    int saccadeTopKRegions = 3;
    bool saccadeDropFullFixation = true;
    double saccadeFullRegionGain = 0.35;
    double saccadeAttentionFloor = 0.35;
    double saccadeAttentionPower = 1.0;
    bool saccadeUseTileAttention = false;

    // --- L5 inhibition ---
    bool enableL5Inhibition = true;
    double l5InhibitLoser = 1.2;
    double l5InhibitThreshold = 0.5;
    int l5MinSpikes = 1;
    bool enableL5InferenceSimilarityGate = true;
    double l5InferenceSimilarityBias = 0.0;
    double l5WinnerMinSimilarity = 0.0;
    double l5WinnerMinScoreMargin = 0.0;
    int l5WinnerGateBootstrapPatterns = 24;
    bool enableL5InterColumnInhibition = false;
    double l5InterColumnInhibit = 0.1;
    double l5InterColumnMinOverlap = 0.2;
    double l5InterColumnWinnerScale = 0.25;
    double l5InterColumnMaxInhibit = 1.0;
    double l5InterColumnMaxOrientationDeltaDeg = 45.0;
    double l5InterColumnMaxFrequencyOctaveDelta = 0.75;
    int l5InterColumnMaxNeighbors = 8;

    // --- STDP ---
    double stdpLtdScale = 0.3;
    double stdpLtdWindowMs = 70.0;
    bool traceStdp = true;
    bool freezeStdpAfterPass1 = false;
    bool enableStdpEligibilityGate = true;
    int stdpEligibilityMinUpdates = 1;
    int stdpEligibilityMinLtp = 0;
    double stdpEligibilityThreshold = -0.002;
    double stdpEligibilityLtdPenalty = 0.5;
    int outputFallbackMinEligibleL5 = 1;
    double outputFallbackMinEligibleL5Fraction = 0.0;
    int outputFallbackBootstrapPatterns = 24;

    // --- Homeostasis ---
    double l4TargetRate = 8.0;
    double l5TargetRate = 5.0;
    double outputTargetRate = 2.0;

    // --- Output competition ---
    bool enableOutputCompetition = true;
    int outputCompetitionKeep = 3;
    int outputCompetitionMinSpikes = 3;
    int outputVoteMinTopSpikes = 5;
    double outputVoteMinTopRatio = 1.1;

    // --- Classification ---
    int knnK = 5;
    bool enableKnnSimilarityWeightedVote = false;
    double knnSimilarityExponent = 1.0;
    size_t maxPatternsPerClass = 1024;
    bool keepL5History = false;
    bool enableOutputVote = true;
    bool enableFullPropagation = true;
    bool disableOutputTeach = false;
    bool enablePairDisambiguation = false;
    double pairDisambMarginTI = 0.0;
    double pairDisambMarginGQ = 0.0;
    double pairDisambMarginIL = 0.0;
    double pairDisambMarginTL = 0.0;
    double pairDisambMarginCE = 0.0;
    double pairDisambToIMarginBoost = 0.0;
    double pairDisambFromIMarginRelax = 0.0;
    bool enableTemporalLatencyReadout = false;
    double temporalLatencyWeight = 0.35;
    bool enableTemporalFeatureCoding = false;
    double temporalFeatureGain = 0.35;
    double temporalFeaturePower = 1.0;
    bool enableReadoutIdfWeighting = false;
    double readoutIdfPower = 1.0;
    bool enableL5DivisiveNormalization = false;
    int l5DivisiveTargetPerColumn = 64;
    std::string classificationType;
    std::map<std::string, int> classificationIntParams;
    std::map<std::string, double> classificationDoubleParams;
    std::map<std::string, std::string> classificationStringParams;

    // --- Convergence ---
    double accuracyEpsilon = 0.001;
    int stablePassesRequired = 3;

    // --- Per-class inclusion mask (true = include) ---
    std::vector<bool> includeClasses;

    // --- Recording ---
    bool enableRecording = false;
    std::string recordingFilename;

    // --- Test limit (0 = all) ---
    int testLimit = 0;

    // --- Neurons per output class ---
    int neuronsPerOutputClass = 3;
};

} // namespace experiment
} // namespace snnfw

#endif // SNNFW_EXPERIMENT_CONFIG_H

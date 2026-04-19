#include "snnfw/Logger.h"
#include "snnfw/ThreadPool.h"
#include "snnfw/adapters/RetinaAdapter.h"
#include "snnfw/classification/ClassificationStrategy.h"
#include "snnfw/declarative/NativeJSONParser.h"
#include "snnfw/declarative/SONATAParser.h"
#include "snnfw/domain/VisualDomainAdapter.h"
#include "snnfw/jepa/JepaConfig.h"
#include "snnfw/jepa/JepaStateExtractor.h"
#include "snnfw/jepa/JepaTrainer.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <random>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

using snnfw::adapters::BaseAdapter;
using snnfw::adapters::RetinaAdapter;
using snnfw::classification::ClassificationStrategy;
using snnfw::declarative::AdapterConfigIR;
using snnfw::declarative::ClassificationConfigIR;
using snnfw::declarative::ColumnIR;
using snnfw::declarative::ColumnTemplateIR;
using snnfw::declarative::NativeJSONParser;
using snnfw::declarative::NetworkIR;
using snnfw::declarative::SONATAParser;
using snnfw::domain::VisualDomainAdapter;
using snnfw::domain::VisualDomainConfig;
using snnfw::domain::VisualStimulus;
using snnfw::jepa::JepaConfig;
using snnfw::jepa::Stage1TapInput;

using IndexBuckets = std::vector<std::vector<size_t>>;
using IntMatrix = std::vector<std::vector<int>>;
using DoubleMatrix = std::vector<std::vector<double>>;

struct Config {
    std::string configPath;
    std::string trainImagesPath;
    std::string trainLabelsPath;
    std::string testImagesPath;
    std::string testLabelsPath;
    std::string inputDomain = "emnist";
    std::string inputVariant = "letters";
    bool inputApplyTransform = true;
    int numClasses = 26;
    std::vector<std::string> classNames;
    int examplesPerClass = 200;
    int testLimit = 1000;
    unsigned int seed = 42;
    int gridSize = 8;
    std::vector<int> gridSizes{8};
    int numOrientations = 8;
    double edgeThreshold = 0.15;
    double temporalWindowMs = 200.0;
    std::string edgeOperator = "sobel";
    std::string encodingStrategy = "rate";
    std::string classifier = "majority";
    std::string activationMode = "binary";
    std::string hierarchicalGroups;
    std::string hierarchicalCoarseStrategy = "majority";
    std::string hierarchicalFineStrategy = "majority";
    int knnK = 5;
    int hierarchicalCoarseK = 0;
    int hierarchicalFineK = 0;
    double classifierExponent = 1.0;
    bool useFeatures = false;
    std::vector<BaseAdapter::Config> retinaConfigs;
    std::vector<std::vector<int>> focusGroups;
    std::string focusGroupSpec;
    int focusLimitPerLabel = 0;
    bool focusOnly = false;
    bool bilateralFusion = false;
    std::string stage1Classifier = "majority";
    std::string fusionClassifier = "majority";
    int stage1K = 0;
    int fusionK = 0;
    double stage1Exponent = 1.0;
    double fusionExponent = 1.0;
    int fusionHoldoutPerClass = 0;
    std::string fusionFeatureMode = "interaction";
    double corpusVoteGain = 0.35;
    double corpusMarginGain = 0.50;
    double corpusCentroidGain = 0.0;
    double corpusNeighborGain = 0.0;
    double corpusDisagreementGain = 0.0;
    bool hemisphereConvergentCodeEnabled = false;
    int hemisphereConvergentSummaryBins = 12;
    double hemisphereConvergentResidualGain = 0.35;
    double hemisphereConvergentInteractionGain = 1.0;
    bool hemisphereTopographicStageEnabled = false;
    double hemisphereTopographicResidualGain = 0.35;
    double hemisphereTopographicContinuityGain = 0.60;
    double hemisphereTopographicJunctionGain = 0.35;
    double hemisphereTopographicAuxiliaryGain = 0.25;
    bool figureGroundStageEnabled = false;
    bool figureGroundMaskEnabled = false;
    bool figureGroundClassifierEnabled = false;
    bool figureGroundObjectMemoryEnabled = false;
    bool recurrentSensoryStateEnabled = false;
    bool recurrentPopulationReadoutEnabled = false;
    bool recurrentObjectStateReadoutEnabled = false;
    int figureGroundCycles = 3;
    double figureGroundBorderGain = 0.35;
    double figureGroundSurfaceGain = 0.25;
    double figureGroundCompetitionGain = 0.20;
    double figureGroundFeedbackGain = 0.15;
    double figureGroundResidualGain = 0.50;
    double figureGroundMaskGain = 0.35;
    double figureGroundClassifierGain = 0.35;
    double figureGroundObjectMemoryGain = 0.25;
    double figureGroundObjectMemoryKeepFraction = 0.30;
    int recurrentSensoryCycles = 3;
    double recurrentSensoryFeedforwardGain = 0.55;
    double recurrentSensoryStateGain = 0.35;
    double recurrentSensoryFigureGroundGain = 0.25;
    double recurrentSensoryContinuityGain = 0.20;
    double recurrentSensoryCallosalGain = 0.15;
    int recurrentObjectStateUnitsPerClass = 3;
    int recurrentObjectStateCycles = 4;
    double recurrentObjectStateInputGain = 1.0;
    double recurrentObjectStateSelfGain = 0.35;
    double recurrentObjectStateClassSupportGain = 0.40;
    double recurrentObjectStateCompetitionGain = 0.55;
    bool predictionErrorFeedbackEnabled = false;
    double predictionErrorFeedbackGain = 0.12;
    double predictionErrorFeedbackThreshold = 0.20;
    double predictionErrorFeedbackMaxPenalty = 0.18;
    double predictionErrorFeedbackMinConfidence = 0.05;
    int onlineCorrectionRepeats = 0;
    int onlineExemplarBudgetPerClass = 16;
    double onlineCentroidLr = 0.25;
    double onlinePositiveRewardGain = 0.35;
    double onlineNegativeRewardGain = 1.0;
    bool onlineRewardStdpEnabled = false;
    double onlineRewardStdpGain = 0.0;
    double onlineRewardStdpLtp = 0.12;
    double onlineRewardStdpLtd = 0.08;
    bool onlineTripletStdpEnabled = false;
    double onlineTripletStdpGain = 0.0;
    double onlineTripletStdpLtp = 0.10;
    double onlineTripletStdpLtd = 0.06;
    double onlineTripletStdpFastDecay = 0.88;
    double onlineTripletStdpSlowDecay = 0.97;
    bool onlineVoltagePlasticityEnabled = false;
    double onlineVoltagePlasticityGain = 0.0;
    double onlineVoltagePlasticityLtp = 0.12;
    double onlineVoltagePlasticityLtd = 0.08;
    double onlineVoltagePlasticityDecay = 0.94;
    double onlineVoltagePlasticityThreshold = 0.30;
    int onlineReplayQueueCapacity = 256;
    int onlineReplayDelaySteps = 0;
    int onlineReplayPauseInterval = 1;
    int onlineReplayBatchSize = 1;
    double onlineReplayUncertaintyThreshold = 0.35;
    double onlineEligibilityTraceDecay = 1.0;
    bool onlineNeuromodulatorEligibilityEnabled = false;
    double onlineNeuromodulatorUncertaintyGain = 0.75;
    double onlineNeuromodulatorPositiveRewardGain = 0.35;
    double onlineNeuromodulatorEligibilityMax = 1.75;
    bool onlineReplayVariableDelayEnabled = false;
    bool onlineReplaySuccessConsolidationEnabled = false;
    int onlineReplaySuccessConsolidationCapacity = 64;
    int onlineReplaySuccessConsolidationBatchSize = 1;
    int onlineReplaySuccessConsolidationRepeats = 1;
    double onlineContextUncertaintyGain = 0.75;
    double onlineContextDisagreementGain = 0.50;
    double onlineConfusionClusterGain = 0.0;
    double onlineConfusionClusterDecay = 0.97;
    bool focusAdjustmentEnabled = false;
    double focusAdjustmentMarginThreshold = 0.10;
    double focusAdjustmentZoom = 1.08;
    double focusAdjustmentShiftPx = 0.35;
    int saccadeFixations = 1;
    bool saccadeTrainingOnly = true;
    double saccadeJitterPx = 0.0;
    double saccadeZoom = 1.0;
    bool guidedSaccadesEnabled = false;
    bool guidedSaccadeTrainingOnly = true;
    int guidedSaccadeFixations = 0;
    int guidedSaccadeCandidatePeaks = 8;
    double guidedSaccadeJitterPx = 0.0;
    double guidedSaccadeZoom = 1.0;
    double guidedSaccadeIorStrength = 0.35;
    std::string trainingCurriculumSpec;
    std::vector<std::vector<int>> trainingCurriculumGroups;
    double trainingReviewFraction = 0.0;
    int trainingAugmentationVariants = 0;
    double trainingAugmentationShiftPx = 0.0;
    double trainingAugmentationRotationDeg = 0.0;
    double trainingAugmentationNoiseStd = 0.0;
    bool activeInferenceEnabled = false;
    int activeInferenceFixations = 1;
    int activeInferenceMinFixations = 1;
    bool activeInferenceRemapEnabled = true;
    double activeInferenceShiftPx = 3.0;
    double activeInferenceZoom = 1.08;
    double activeInferenceUncertaintyThreshold = 0.35;
    double activeInferenceIorStrength = 0.35;
    bool hierarchicalRetinaLayout = false;
    std::vector<std::string> declaredHemisphereOrder;
    std::string fusionPath;
    bool separabilityDiagnostics = false;
    int separabilitySampleLimit = 0;
    bool flowAuditEnabled = false;
    int flowAuditSampleLimit = 0;
    std::string flowAuditOutputPrefix;
    JepaConfig jepa;
};

struct JepaProbeResult {
    IntMatrix confusion;
    int correct = 0;
    int tested = 0;
    double seconds = 0.0;
};

struct ObjectStateAttractor {
    int label = -1;
    std::vector<double> prototype;
};

struct HemisphereRuntime {
    std::string name;
    std::vector<std::unique_ptr<RetinaAdapter>> retinas;
    std::vector<ClassificationStrategy::LabeledPattern> rawTrainingPatterns;
    std::vector<ClassificationStrategy::LabeledPattern> trainingPatterns;
    std::vector<size_t> trainingSourceIndices;
    std::unique_ptr<ClassificationStrategy> classifier;
    double overallWeight = 1.0;
    std::vector<double> classWeights;
    std::vector<double> predictionWeights;
    std::vector<std::vector<double>> classCentroids;
    std::vector<std::vector<double>> rewardStdpPrototypes;
    std::vector<std::vector<double>> tripletStdpPrototypes;
    std::vector<double> tripletPreTrace;
    std::vector<double> tripletPostTraceFast;
    std::vector<double> tripletPostTraceSlow;
    size_t tripletTraceStep = 0;
    std::vector<std::vector<double>> voltagePlasticityPrototypes;
    std::vector<double> voltageDepolarizationTrace;
    size_t voltageTraceStep = 0;
    std::vector<std::vector<size_t>> onlinePatternIndices;
    std::vector<size_t> retinaPatternSizes;
    std::vector<double> lastFigureGroundPattern;
    std::vector<ClassificationStrategy::LabeledPattern> figureGroundObjectMemoryPatterns;
    std::vector<double> lastFigureGroundObjectPattern;
    std::vector<ObjectStateAttractor> recurrentObjectStateAttractors;
};

struct FusionRuntime {
    std::vector<ClassificationStrategy::LabeledPattern> trainingPatterns;
    std::vector<std::vector<double>> rewardStdpPrototypes;
    std::vector<std::vector<double>> tripletStdpPrototypes;
    std::vector<double> tripletPreTrace;
    std::vector<double> tripletPostTraceFast;
    std::vector<double> tripletPostTraceSlow;
    size_t tripletTraceStep = 0;
    std::vector<std::vector<double>> voltagePlasticityPrototypes;
    std::vector<double> voltageDepolarizationTrace;
    size_t voltageTraceStep = 0;
    std::vector<std::vector<size_t>> onlinePatternIndices;
};

struct LabelTrace {
    int label = -1;
    double score = 0.0;
};

struct HemisphereDecisionTrace {
    std::vector<double> pattern;
    std::vector<double> classifierPattern;
    std::vector<double> figureGroundPattern;
    std::vector<double> figureGroundObjectPattern;
    std::vector<double> figureGroundObjectConfidence;
    std::vector<double> confidence;
    std::vector<LabelTrace> topHypotheses;
    int predicted = -1;
    double margin = 0.0;
};

struct FigureGroundBranchSummary {
    std::string name;
    std::vector<double> pattern;
    double ownedLeftMass = 0.0;
    double ownedRightMass = 0.0;
    double surfaceMass = 0.0;
    double junctionMass = 0.0;
    double borderSurfaceRatio = 0.0;
    double leftRightBalance = 0.0;
    int componentCount = 0;
};

struct FigureGroundState {
    bool valid = false;
    std::vector<FigureGroundBranchSummary> branches;
    std::vector<double> pattern;
    double ownedBorderMass = 0.0;
    double surfaceMass = 0.0;
    double junctionMass = 0.0;
    double borderSurfaceRatio = 0.0;
    double leftRightBalance = 0.0;
    int componentCount = 0;
};

struct RecurrentSensoryPatternResult {
    std::vector<double> pattern;
    std::vector<double> figureGroundPattern;
    std::vector<std::vector<double>> fixationPatterns;
    int fixationCount = 0;
};

struct BilateralDecisionTrace {
    std::vector<HemisphereDecisionTrace> hemisphereTraces;
    std::vector<double> combinedConfidence;
    std::vector<double> fusionPattern;
    std::vector<LabelTrace> topHypotheses;
    int predicted = -1;
    int fixationCount = 1;
    std::vector<std::pair<double, double>> fixationShifts;
};

struct DecisionContext {
    double uncertainty = 0.0;
    double disagreement = 0.0;
    double confusionCluster = 0.0;
    double plasticity = 1.0;
    double replayPriority = 0.0;
};

struct ConfusionClusterMemory {
    DoubleMatrix strengths;
};

struct OnlineSampleRecord {
    size_t imageIndex = 0;
    int truth = -1;
    int leftInitialPredicted = -1;
    int rightInitialPredicted = -1;
    int initialPredicted = -1;
    int finalPredicted = -1;
    bool correctionSucceeded = false;
};

struct ReplayItem {
    size_t recordIndex = 0;
    int remainingReplays = 0;
    double priority = 0.0;
    double eligibilityScale = 1.0;
    size_t traceAgeSteps = 0;
    size_t readyStep = 0;
    size_t sequence = 0;
};

struct FlowAuditVectorStats {
    double meanAbs = 0.0;
    double l2 = 0.0;
    double activeFraction = 0.0;
};

struct RetinaLayerBinding {
    std::string hemisphere;
    std::string lobe;
    std::string region;
    std::string nucleus;
    std::string column;
    std::string layer;
    std::string path;
};

struct HemisphereTrainingArtifacts {
    std::vector<ClassificationStrategy::LabeledPattern> trainingPatterns;
    std::vector<size_t> trainingSourceIndices;
    struct FlowAuditBranchTrainingStats {
        std::string name;
        std::vector<std::vector<double>> preCentroids;
        std::vector<std::vector<double>> postCentroids;
        std::vector<int> preCounts;
        std::vector<int> postCounts;
        double fixationVarianceSum = 0.0;
        int fixationVarianceSamples = 0;
    };
    bool flowAuditEnabled = false;
    std::vector<FlowAuditBranchTrainingStats> flowAuditBranches;
};

struct TrainingSchedulePlan {
    std::vector<size_t> primaryIndices;
    std::vector<size_t> reviewIndices;
};

std::vector<double> buildFusionPattern(std::vector<HemisphereRuntime>& hemispheres,
                                       const VisualStimulus& image,
                                       const Config& config);
struct FixationSpec;
struct HemispherePatternAccumulator;
struct ContextualGroupingLayout;
struct FlowAuditSampleCapture;
struct RecurrentSensoryPatternResult;
void appendFixationToAccumulator(HemisphereRuntime& hemisphere,
                                 HemispherePatternAccumulator& accumulator,
                                 const FixationSpec& fixation,
                                 const Config& config);
std::vector<double> buildPatternFromAccumulator(HemisphereRuntime& hemisphere,
                                                const HemispherePatternAccumulator& accumulator);
std::vector<double> buildHemisphereConvergentPattern(const HemisphereRuntime& hemisphere,
                                                     const std::vector<double>& pattern,
                                                     const Config& config);
FigureGroundState buildHemisphereFigureGroundState(const HemisphereRuntime& hemisphere,
                                                   const std::vector<double>& pattern,
                                                   const Config& config);
bool useFigureGroundStage(const Config& config);
bool useFigureGroundMask(const Config& config);
bool useFigureGroundClassifier(const Config& config);
bool useFigureGroundObjectMemory(const Config& config);
bool useRecurrentSensoryState(const Config& config);
bool useRecurrentPopulationReadout(const Config& config);
bool useRecurrentObjectStateReadout(const Config& config);
std::vector<double> computeHemisphereReadoutConfidence(
    const HemisphereRuntime& hemisphere,
    const std::vector<double>& pattern,
    const Config& config);
std::vector<double> computeHemisphereSupportReadoutConfidence(
    const HemisphereRuntime& hemisphere,
    const std::vector<double>& pattern,
    const std::vector<ClassificationStrategy::LabeledPattern>& supportPatterns,
    const Config& config);
std::vector<double> buildHemisphereClassifierPattern(const HemisphereRuntime& hemisphere,
                                                     const std::vector<double>& pattern,
                                                     const Config& config,
                                                     const FigureGroundState* figureGroundState =
                                                         nullptr);
std::vector<double> buildFigureGroundObjectMemoryPattern(
    const HemisphereRuntime& hemisphere,
    const std::vector<double>& pattern,
    const FigureGroundState& state,
    const Config& config);
std::vector<double> computeFigureGroundObjectMemoryConfidence(
    HemisphereRuntime& hemisphere,
    const std::vector<double>& pattern,
    const FigureGroundState& state,
    const Config& config,
    std::vector<double>* objectPatternOut = nullptr);
std::vector<double> buildFigureGroundDrivePattern(const HemisphereRuntime& hemisphere,
                                                  const std::vector<double>& pattern,
                                                  const FigureGroundState& state);
std::vector<VisualStimulus> buildRecurrentExtractionStimuli(
    const VisualStimulus& image,
    const Config& config,
    bool trainingPhase,
    size_t sampleSeed,
    std::vector<std::unique_ptr<RetinaAdapter>>* retinas = nullptr);
RecurrentSensoryPatternResult buildRecurrentSensoryPattern(
    HemisphereRuntime& hemisphere,
    const VisualStimulus& image,
    const Config& config,
    bool trainingPhase,
    size_t sampleSeed);
std::vector<RecurrentSensoryPatternResult> buildRecurrentSensoryPatterns(
    std::vector<HemisphereRuntime>& hemispheres,
    const VisualStimulus& image,
    const Config& config,
    bool trainingPhase,
    size_t sampleSeed);
std::vector<std::vector<double>> computeClassCentroids(
    const std::vector<ClassificationStrategy::LabeledPattern>& patterns,
    int numClasses);
std::vector<ObjectStateAttractor> buildObjectStateAttractors(
    const std::vector<ClassificationStrategy::LabeledPattern>& patterns,
    int numClasses,
    int unitsPerClass);
void rebuildHemisphereObjectStateAttractors(HemisphereRuntime& hemisphere,
                                            const Config& config);
size_t inferAuxiliaryChannelCount(const RetinaAdapter& retina);
size_t inferFrequencyBandCount(const RetinaAdapter& retina);
std::vector<size_t> inferHemisphereBranchPatternSizes(
    std::vector<std::unique_ptr<RetinaAdapter>>& retinas,
    const VisualStimulus& image,
    bool useFeatures);
std::vector<double> applyContextualGrouping(const RetinaAdapter& retina,
                                            std::vector<double> part);
ContextualGroupingLayout inferContextualGroupingLayout(const RetinaAdapter& retina,
                                                       size_t totalSize);
std::pair<double, double> computeRetinaChannelL2Energies(const RetinaAdapter& retina,
                                                         const std::vector<double>& part);
std::vector<double> extractRetinaPartRawPattern(RetinaAdapter& retina,
                                                const VisualStimulus& image,
                                                bool useFeatures,
                                                bool learnPatterns);
std::vector<double> extractPattern(std::vector<std::unique_ptr<RetinaAdapter>>& retinas,
                                   const VisualStimulus& image,
                                   const Config& config,
                                   bool trainingPhase,
                                   bool learnPatterns,
                                   size_t sampleSeed,
                                   FlowAuditSampleCapture* flowAuditCapture);
std::vector<HemisphereDecisionTrace> inferHemisphereDecisions(
    std::vector<HemisphereRuntime>& hemispheres,
    const VisualStimulus& image,
    const Config& config,
    std::vector<FlowAuditSampleCapture>* flowAuditCaptures = nullptr);
std::unique_ptr<ClassificationStrategy> makeClassifierStrategy(const std::string& type,
                                                               int k,
                                                               double exponent,
                                                               const Config& config);
std::string toLower(std::string value);
std::vector<Stage1TapInput> buildJepaStage1TapInputs(
    std::vector<HemisphereRuntime>& hemispheres,
    const VisualDomainAdapter& trainLoader,
    const Config& config);
std::vector<Stage1TapInput> buildJepaStage1EvalTapInputs(
    std::vector<HemisphereRuntime>& hemispheres,
    const VisualDomainAdapter& loader,
    const std::vector<size_t>& indices,
    const Config& config);
void maybeExportJepaStage1Taps(std::vector<HemisphereRuntime>& hemispheres,
                               const VisualDomainAdapter& trainLoader,
                               const Config& config);
bool maybeTrainJepaModel(std::vector<HemisphereRuntime>& hemispheres,
                         const VisualDomainAdapter& trainLoader,
                         const Config& config,
                         snnfw::jepa::JepaTrainingArtifacts* artifactsOut = nullptr);
JepaProbeResult evaluateJepaProbe(std::vector<HemisphereRuntime>& hemispheres,
                                  const VisualDomainAdapter& trainLoader,
                                  const VisualDomainAdapter& testLoader,
                                  const std::vector<size_t>& indices,
                                  const Config& config,
                                  const snnfw::jepa::JepaModel& model,
                                  const std::string& label);

size_t hemisphereWorkerCount(size_t hemisphereCount) {
    const size_t hardwareThreads =
        std::max<size_t>(1, static_cast<size_t>(std::thread::hardware_concurrency()));
    return std::max<size_t>(1, std::min(hemisphereCount, hardwareThreads));
}

snnfw::ThreadPool& getHemisphereThreadPool(size_t hemisphereCount) {
    static std::unique_ptr<snnfw::ThreadPool> pool;
    static size_t poolSize = 0;

    const size_t desiredSize = hemisphereWorkerCount(hemisphereCount);
    if (!pool || poolSize != desiredSize) {
        pool = std::make_unique<snnfw::ThreadPool>(desiredSize);
        poolSize = desiredSize;
    }
    return *pool;
}

template <typename Result, typename Fn>
std::vector<Result> runHemisphereTasks(size_t hemisphereCount, Fn&& fn) {
    std::vector<Result> results(hemisphereCount);
    if (hemisphereCount <= 1) {
        for (size_t i = 0; i < hemisphereCount; ++i) {
            results[i] = fn(i);
        }
        return results;
    }

    auto& pool = getHemisphereThreadPool(hemisphereCount);
    std::vector<std::future<Result>> futures;
    futures.reserve(hemisphereCount);
    for (size_t i = 0; i < hemisphereCount; ++i) {
        futures.push_back(pool.enqueue([&fn, i]() { return fn(i); }));
    }
    for (size_t i = 0; i < hemisphereCount; ++i) {
        results[i] = futures[i].get();
    }
    return results;
}

template <typename T>
std::vector<T> makeClassVector(int numClasses, const T& value = T()) {
    return std::vector<T>(static_cast<size_t>(std::max(0, numClasses)), value);
}

template <typename T>
std::vector<std::vector<T>> makeClassMatrix(int numClasses, const T& value = T()) {
    return std::vector<std::vector<T>>(
        static_cast<size_t>(std::max(0, numClasses)),
        std::vector<T>(static_cast<size_t>(std::max(0, numClasses)), value));
}

std::string classLabel(const Config& config, int label) {
    if (label >= 0 && static_cast<size_t>(label) < config.classNames.size() &&
        !config.classNames[static_cast<size_t>(label)].empty()) {
        return config.classNames[static_cast<size_t>(label)];
    }
    return std::to_string(label);
}

void configureClassMetadata(Config& config, const VisualDomainAdapter& adapter) {
    config.numClasses = adapter.numClasses();
    config.classNames = adapter.classNames();
    if (config.classNames.size() != static_cast<size_t>(config.numClasses)) {
        config.classNames.clear();
        for (int label = 0; label < config.numClasses; ++label) {
            config.classNames.push_back(std::to_string(label));
        }
    }
}

void initializeHemisphereRuntime(HemisphereRuntime& hemisphere, int numClasses) {
    hemisphere.classWeights.assign(static_cast<size_t>(numClasses), 1.0);
    hemisphere.predictionWeights.assign(static_cast<size_t>(numClasses), 1.0);
    hemisphere.classCentroids.assign(static_cast<size_t>(numClasses), {});
    hemisphere.rewardStdpPrototypes.assign(static_cast<size_t>(numClasses), {});
    hemisphere.tripletStdpPrototypes.assign(static_cast<size_t>(numClasses), {});
    hemisphere.tripletPreTrace.clear();
    hemisphere.tripletPostTraceFast.assign(static_cast<size_t>(numClasses), 0.0);
    hemisphere.tripletPostTraceSlow.assign(static_cast<size_t>(numClasses), 0.0);
    hemisphere.tripletTraceStep = 0;
    hemisphere.voltagePlasticityPrototypes.assign(static_cast<size_t>(numClasses), {});
    hemisphere.voltageDepolarizationTrace.assign(static_cast<size_t>(numClasses), 0.0);
    hemisphere.voltageTraceStep = 0;
    hemisphere.onlinePatternIndices.assign(static_cast<size_t>(numClasses), {});
    hemisphere.figureGroundObjectMemoryPatterns.clear();
    hemisphere.lastFigureGroundPattern.clear();
    hemisphere.lastFigureGroundObjectPattern.clear();
}

void initializeFusionRuntime(FusionRuntime& fusionRuntime, int numClasses) {
    fusionRuntime.rewardStdpPrototypes.assign(static_cast<size_t>(numClasses), {});
    fusionRuntime.tripletStdpPrototypes.assign(static_cast<size_t>(numClasses), {});
    fusionRuntime.tripletPreTrace.clear();
    fusionRuntime.tripletPostTraceFast.assign(static_cast<size_t>(numClasses), 0.0);
    fusionRuntime.tripletPostTraceSlow.assign(static_cast<size_t>(numClasses), 0.0);
    fusionRuntime.tripletTraceStep = 0;
    fusionRuntime.voltagePlasticityPrototypes.assign(static_cast<size_t>(numClasses), {});
    fusionRuntime.voltageDepolarizationTrace.assign(static_cast<size_t>(numClasses), 0.0);
    fusionRuntime.voltageTraceStep = 0;
    fusionRuntime.onlinePatternIndices.assign(static_cast<size_t>(numClasses), {});
}

ConfusionClusterMemory makeConfusionClusterMemory(const Config& config) {
    ConfusionClusterMemory memory;
    memory.strengths = makeClassMatrix<double>(config.numClasses, 0.0);
    return memory;
}

double cosineSimilarity(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size() || a.empty()) {
        return 0.0;
    }

    double dot = 0.0;
    double normA = 0.0;
    double normB = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    if (normA <= 0.0 || normB <= 0.0) {
        return 0.0;
    }
    return dot / (std::sqrt(normA) * std::sqrt(normB));
}

std::string normalizeLabelToken(std::string token) {
    token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
    std::transform(token.begin(), token.end(), token.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return token;
}

std::vector<int> parseCsvInts(const std::string& csv) {
    std::vector<int> values;
    std::stringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            values.push_back(std::stoi(token));
        }
    }
    return values;
}

int parseLabelToken(std::string token, const Config& config) {
    token = normalizeLabelToken(std::move(token));
    if (token.empty()) {
        throw std::runtime_error("Empty label token in focus group definition");
    }

    for (size_t label = 0; label < config.classNames.size(); ++label) {
        if (normalizeLabelToken(config.classNames[label]) == token) {
            return static_cast<int>(label);
        }
    }

    const int numeric = std::stoi(token);
    if (numeric >= 0 && numeric < config.numClasses) {
        return numeric;
    }
    if (numeric >= 1 && numeric <= config.numClasses) {
        return numeric - 1;
    }
    throw std::runtime_error("Label token out of range: " + token);
}

std::vector<std::vector<int>> parseLabelGroups(const std::string& spec, const Config& config) {
    std::vector<std::vector<int>> groups;
    std::stringstream outer(spec);
    std::string groupToken;
    while (std::getline(outer, groupToken, ';')) {
        if (groupToken.empty()) {
            continue;
        }
        std::stringstream inner(groupToken);
        std::string labelToken;
        std::vector<int> group;
        while (std::getline(inner, labelToken, ',')) {
            if (!labelToken.empty()) {
                const int label = parseLabelToken(labelToken, config);
                if (std::find(group.begin(), group.end(), label) == group.end()) {
                    group.push_back(label);
                }
            }
        }
        if (!group.empty()) {
            groups.push_back(std::move(group));
        }
    }
    return groups;
}

std::vector<int> flattenLabelGroups(const std::vector<std::vector<int>>& groups) {
    std::vector<int> labels;
    for (const auto& group : groups) {
        for (int label : group) {
            if (std::find(labels.begin(), labels.end(), label) == labels.end()) {
                labels.push_back(label);
            }
        }
    }
    std::sort(labels.begin(), labels.end());
    return labels;
}

std::string labelGroupToString(const std::vector<int>& group, const Config& config) {
    std::ostringstream oss;
    for (size_t i = 0; i < group.size(); ++i) {
        if (i > 0) {
            oss << "/";
        }
        oss << classLabel(config, group[i]);
    }
    return oss.str();
}

std::string labelGroupsToString(const std::vector<std::vector<int>>& groups, const Config& config) {
    std::ostringstream oss;
    for (size_t i = 0; i < groups.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << "[" << labelGroupToString(groups[i], config) << "]";
    }
    return oss.str();
}

void normalizeL2(std::vector<double>& values) {
    double norm = 0.0;
    for (double value : values) {
        norm += value * value;
    }
    if (norm <= 0.0) {
        return;
    }
    norm = std::sqrt(norm);
    for (double& value : values) {
        value /= norm;
    }
}

void normalizeSum(std::vector<double>& values) {
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    if (sum <= 0.0) {
        return;
    }
    for (double& value : values) {
        value /= sum;
    }
}

FlowAuditVectorStats computeFlowAuditVectorStats(const std::vector<double>& pattern) {
    FlowAuditVectorStats stats;
    if (pattern.empty()) {
        return stats;
    }

    double l1 = 0.0;
    double l2 = 0.0;
    int active = 0;
    for (double value : pattern) {
        const double absValue = std::abs(value);
        l1 += absValue;
        l2 += value * value;
        if (absValue > 1e-6) {
            ++active;
        }
    }

    stats.meanAbs = l1 / static_cast<double>(pattern.size());
    stats.l2 = std::sqrt(l2);
    stats.activeFraction = static_cast<double>(active) / static_cast<double>(pattern.size());
    return stats;
}

double computeFlowAuditFixationVariance(const std::vector<std::vector<double>>& fixationParts) {
    if (fixationParts.size() <= 1 || fixationParts.front().empty()) {
        return 0.0;
    }

    const size_t dim = fixationParts.front().size();
    std::vector<double> mean(dim, 0.0);
    size_t valid = 0;
    for (const auto& part : fixationParts) {
        if (part.size() != dim) {
            continue;
        }
        for (size_t i = 0; i < dim; ++i) {
            mean[i] += part[i];
        }
        ++valid;
    }
    if (valid <= 1) {
        return 0.0;
    }
    const double invValid = 1.0 / static_cast<double>(valid);
    for (double& value : mean) {
        value *= invValid;
    }

    double variance = 0.0;
    for (const auto& part : fixationParts) {
        if (part.size() != dim) {
            continue;
        }
        double distanceSq = 0.0;
        for (size_t i = 0; i < dim; ++i) {
            const double delta = part[i] - mean[i];
            distanceSq += delta * delta;
        }
        variance += distanceSq;
    }
    return variance / static_cast<double>(valid);
}

double computeFlowAuditFixationConsistency(
    const std::vector<std::vector<double>>& fixationPatterns) {
    if (fixationPatterns.empty()) {
        return 0.0;
    }

    std::vector<const std::vector<double>*> validPatterns;
    validPatterns.reserve(fixationPatterns.size());
    for (const auto& pattern : fixationPatterns) {
        if (!pattern.empty()) {
            validPatterns.push_back(&pattern);
        }
    }
    if (validPatterns.empty()) {
        return 0.0;
    }
    if (validPatterns.size() == 1) {
        return 1.0;
    }

    double similaritySum = 0.0;
    int pairCount = 0;
    for (size_t a = 0; a < validPatterns.size(); ++a) {
        for (size_t b = a + 1; b < validPatterns.size(); ++b) {
            const auto& lhs = *validPatterns[a];
            const auto& rhs = *validPatterns[b];
            if (lhs.size() != rhs.size()) {
                continue;
            }
            similaritySum += cosineSimilarity(lhs, rhs);
            ++pairCount;
        }
    }
    if (pairCount <= 0) {
        return 1.0;
    }
    return similaritySum / static_cast<double>(pairCount);
}

std::string sanitizeFlowAuditStem(std::string value) {
    for (char& c : value) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_' || c == '-')) {
            c = '_';
        }
    }
    value.erase(std::unique(value.begin(), value.end(),
                            [](char a, char b) { return a == '_' && b == '_'; }),
                value.end());
    while (!value.empty() && value.front() == '_') {
        value.erase(value.begin());
    }
    while (!value.empty() && value.back() == '_') {
        value.pop_back();
    }
    return value.empty() ? "flow_audit" : value;
}

std::string defaultFlowAuditOutputPrefix(const Config& config) {
    if (!config.flowAuditOutputPrefix.empty()) {
        return config.flowAuditOutputPrefix;
    }

    std::filesystem::path baseDir("build");
    std::string stem = "flow_audit";
    if (!config.configPath.empty()) {
        stem = std::filesystem::path(config.configPath).stem().string();
    } else if (!config.inputDomain.empty()) {
        stem = config.inputDomain + "_" + config.inputVariant + "_flow_audit";
    }
    stem = sanitizeFlowAuditStem(stem);
    return (baseDir / stem).string();
}

VisualStimulus applyImageFocusTransform(const VisualStimulus& image,
                                        double scale,
                                        double shiftXPx,
                                        double shiftYPx = 0.0,
                                        double rotationDeg = 0.0) {
    const bool hasTransform = std::abs(scale - 1.0) > 1e-6 ||
                              std::abs(shiftXPx) > 1e-6 ||
                              std::abs(shiftYPx) > 1e-6 ||
                              std::abs(rotationDeg) > 1e-6;
    if (!hasTransform) {
        return image;
    }

    VisualStimulus transformed;
    transformed.label = image.label;
    transformed.rows = image.rows;
    transformed.cols = image.cols;
    transformed.channels = std::max(1, image.channels);
    transformed.pixels.assign(
        static_cast<size_t>(image.rows * image.cols * transformed.channels), 0);

    const double centerX = 0.5 * static_cast<double>(image.cols - 1);
    const double centerY = 0.5 * static_cast<double>(image.rows - 1);
    const double safeScale = std::max(1e-3, scale);
    const double theta = rotationDeg * 3.14159265358979323846 / 180.0;
    const double cosTheta = std::cos(theta);
    const double sinTheta = std::sin(theta);

    auto sampleBilinear = [&](double row, double col, int channel) -> uint8_t {
        if (row < 0.0 || col < 0.0 ||
            row > static_cast<double>(image.rows - 1) ||
            col > static_cast<double>(image.cols - 1)) {
            return 0;
        }

        const int r0 = static_cast<int>(std::floor(row));
        const int c0 = static_cast<int>(std::floor(col));
        const int r1 = std::min(r0 + 1, image.rows - 1);
        const int c1 = std::min(c0 + 1, image.cols - 1);
        const double fr = row - static_cast<double>(r0);
        const double fc = col - static_cast<double>(c0);

        const double p00 = static_cast<double>(image.getPixel(r0, c0, channel));
        const double p01 = static_cast<double>(image.getPixel(r0, c1, channel));
        const double p10 = static_cast<double>(image.getPixel(r1, c0, channel));
        const double p11 = static_cast<double>(image.getPixel(r1, c1, channel));
        const double top = p00 * (1.0 - fc) + p01 * fc;
        const double bottom = p10 * (1.0 - fc) + p11 * fc;
        const double value = top * (1.0 - fr) + bottom * fr;
        return static_cast<uint8_t>(std::clamp(std::lround(value), 0L, 255L));
    };

    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            double x = static_cast<double>(col) - centerX;
            double y = static_cast<double>(row) - centerY;
            x -= shiftXPx;
            y -= shiftYPx;
            const double rotatedX = (x * cosTheta + y * sinTheta) / safeScale;
            const double rotatedY = (-x * sinTheta + y * cosTheta) / safeScale;
            const double srcX = rotatedX + centerX;
            const double srcY = rotatedY + centerY;
            for (int channel = 0; channel < transformed.channels; ++channel) {
                transformed.pixels[(static_cast<size_t>(row * image.cols + col) *
                                    static_cast<size_t>(transformed.channels)) +
                                   static_cast<size_t>(channel)] =
                    sampleBilinear(srcY, srcX, channel);
            }
        }
    }

    return transformed;
}

std::mt19937 makeTrainingAugmentationRng(const VisualStimulus& image,
                                         const Config& config,
                                         size_t sampleSeed) {
    std::seed_seq seedSeq{
        config.seed,
        static_cast<unsigned int>(sampleSeed & 0xffffffffU),
        static_cast<unsigned int>((sampleSeed >> 32U) & 0xffffffffU),
        static_cast<unsigned int>(std::max(0, image.label)),
        static_cast<unsigned int>(std::max(1, image.rows)),
        static_cast<unsigned int>(std::max(1, image.cols)),
        static_cast<unsigned int>(std::max(1, image.channels)),
    };
    return std::mt19937(seedSeq);
}

void applyPhotonLikeNoise(VisualStimulus& image,
                          double normalizedNoiseStd,
                          std::mt19937& rng) {
    if (normalizedNoiseStd <= 0.0 || image.pixels.empty()) {
        return;
    }

    std::normal_distribution<double> noise(0.0, 1.0);
    for (uint8_t& pixel : image.pixels) {
        const double value = static_cast<double>(pixel) / 255.0;
        const double sigma = normalizedNoiseStd * std::sqrt(std::max(0.05, value));
        const double noisyValue = std::clamp(value + sigma * noise(rng), 0.0, 1.0);
        pixel = static_cast<uint8_t>(std::clamp(std::lround(noisyValue * 255.0), 0L, 255L));
    }
}

std::vector<VisualStimulus> buildTrainingAugmentationStimuli(const VisualStimulus& image,
                                                             const Config& config,
                                                             bool trainingPhase,
                                                             size_t sampleSeed) {
    const bool enabled = trainingPhase &&
        config.trainingAugmentationVariants > 0 &&
        (config.trainingAugmentationShiftPx > 0.0 ||
         config.trainingAugmentationRotationDeg > 0.0 ||
         config.trainingAugmentationNoiseStd > 0.0);
    if (!enabled) {
        return {image};
    }

    std::mt19937 rng = makeTrainingAugmentationRng(image, config, sampleSeed);
    std::uniform_real_distribution<double> shiftDistribution(
        -config.trainingAugmentationShiftPx, config.trainingAugmentationShiftPx);
    std::uniform_real_distribution<double> rotationDistribution(
        -config.trainingAugmentationRotationDeg, config.trainingAugmentationRotationDeg);

    std::vector<VisualStimulus> stimuli;
    stimuli.reserve(static_cast<size_t>(1 + config.trainingAugmentationVariants));
    stimuli.push_back(image);
    for (int variant = 0; variant < config.trainingAugmentationVariants; ++variant) {
        const double shiftX = config.trainingAugmentationShiftPx > 0.0
            ? shiftDistribution(rng)
            : 0.0;
        const double shiftY = config.trainingAugmentationShiftPx > 0.0
            ? shiftDistribution(rng)
            : 0.0;
        const double rotation = config.trainingAugmentationRotationDeg > 0.0
            ? rotationDistribution(rng)
            : 0.0;
        auto augmented = applyImageFocusTransform(image, 1.0, shiftX, shiftY, rotation);
        applyPhotonLikeNoise(augmented, config.trainingAugmentationNoiseStd, rng);
        stimuli.push_back(std::move(augmented));
    }
    return stimuli;
}

std::vector<VisualStimulus> buildSaccadeStimuli(const VisualStimulus& image,
                                                const Config& config,
                                                bool trainingPhase,
                                                std::vector<std::unique_ptr<RetinaAdapter>>* retinas = nullptr) {
    static const std::array<std::pair<double, double>, 8> kOffsets = {{
        {-1.0, 0.0},
        {1.0, 0.0},
        {0.0, -1.0},
        {0.0, 1.0},
        {-0.7, -0.7},
        {0.7, -0.7},
        {-0.7, 0.7},
        {0.7, 0.7},
    }};

    const bool wantsGuided =
        config.guidedSaccadesEnabled && (trainingPhase || !config.guidedSaccadeTrainingOnly);
    const int requestedGuidedFixations =
        config.guidedSaccadeFixations > 0
            ? config.guidedSaccadeFixations
            : std::max(1, config.saccadeFixations);
    const bool enableSampling = trainingPhase || !config.saccadeTrainingOnly || wantsGuided;
    const int fixations = enableSampling
        ? std::max(1, wantsGuided ? requestedGuidedFixations : config.saccadeFixations)
        : 1;
    const double effectiveJitter = wantsGuided && config.guidedSaccadeJitterPx > 0.0
        ? config.guidedSaccadeJitterPx
        : config.saccadeJitterPx;
    const double effectiveZoom = wantsGuided && config.guidedSaccadeZoom > 1e-6
        ? config.guidedSaccadeZoom
        : config.saccadeZoom;
    if (fixations <= 1 ||
        (effectiveJitter <= 0.0 && std::abs(effectiveZoom - 1.0) <= 1e-6)) {
        return {image};
    }

    const bool guidedEnabled =
        wantsGuided && retinas != nullptr &&
        !retinas->empty();
    if (guidedEnabled) {
        struct GuidedCandidate {
            double offsetX = 0.0;
            double offsetY = 0.0;
            double score = 0.0;
        };

        static const std::array<std::pair<double, double>, 24> kGuidedOffsets = {{
            {-1.0, 0.0}, {1.0, 0.0}, {0.0, -1.0}, {0.0, 1.0},
            {-0.7, -0.7}, {0.7, -0.7}, {-0.7, 0.7}, {0.7, 0.7},
            {-1.5, 0.0}, {1.5, 0.0}, {0.0, -1.5}, {0.0, 1.5},
            {-1.05, -1.05}, {1.05, -1.05}, {-1.05, 1.05}, {1.05, 1.05},
            {-0.5, -1.3}, {0.5, -1.3}, {-0.5, 1.3}, {0.5, 1.3},
            {-1.3, -0.5}, {1.3, -0.5}, {-1.3, 0.5}, {1.3, 0.5},
        }};

        const int guidedFixations =
            config.guidedSaccadeFixations > 0
                ? config.guidedSaccadeFixations
                : std::max(1, config.saccadeFixations);
        const int candidateLimit = std::clamp(
            config.guidedSaccadeCandidatePeaks,
            1,
            static_cast<int>(kGuidedOffsets.size()));
        const double guidedJitter = config.guidedSaccadeJitterPx > 0.0
            ? config.guidedSaccadeJitterPx
            : config.saccadeJitterPx;
        const double guidedZoom = config.guidedSaccadeZoom > 1e-6
            ? config.guidedSaccadeZoom
            : config.saccadeZoom;

        std::vector<GuidedCandidate> candidates;
        candidates.reserve(static_cast<size_t>(candidateLimit));
        for (int i = 0; i < candidateLimit; ++i) {
            const auto& [offsetX, offsetY] = kGuidedOffsets[static_cast<size_t>(i)];
            auto shifted = applyImageFocusTransform(
                image, guidedZoom, guidedJitter * offsetX, guidedJitter * offsetY);
            double score = 0.0;
            int scoredBranches = 0;
            for (auto& retina : *retinas) {
                if (retina == nullptr || toLower(retina->getName()).find("g10") == std::string::npos) {
                    continue;
                }
                auto rawPart = extractRetinaPartRawPattern(*retina, shifted, config.useFeatures, false);
                const auto [orientationL2, auxiliaryL2] =
                    computeRetinaChannelL2Energies(*retina, rawPart);
                score += orientationL2 + (0.15 * auxiliaryL2);
                scoredBranches++;
            }
            if (scoredBranches > 0) {
                score /= static_cast<double>(scoredBranches);
            }
            candidates.push_back({offsetX, offsetY, score});
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.score > rhs.score;
        });

        std::vector<VisualStimulus> stimuli;
        stimuli.reserve(static_cast<size_t>(guidedFixations));
        stimuli.push_back(image);
        std::vector<std::pair<double, double>> selectedOffsets;
        selectedOffsets.reserve(static_cast<size_t>(guidedFixations));
        selectedOffsets.push_back({0.0, 0.0});
        for (int fixation = 1; fixation < guidedFixations; ++fixation) {
            size_t bestIndex = 0;
            double bestScore = -std::numeric_limits<double>::infinity();
            for (size_t candidateIndex = 0; candidateIndex < candidates.size(); ++candidateIndex) {
                const auto& candidate = candidates[candidateIndex];
                double minDistance = std::numeric_limits<double>::infinity();
                for (const auto& [selectedX, selectedY] : selectedOffsets) {
                    const double dx = candidate.offsetX - selectedX;
                    const double dy = candidate.offsetY - selectedY;
                    minDistance = std::min(minDistance, std::sqrt((dx * dx) + (dy * dy)));
                }
                const double iorPenalty =
                    config.guidedSaccadeIorStrength / std::max(0.25, minDistance);
                const double adjustedScore = candidate.score - iorPenalty;
                if (adjustedScore > bestScore) {
                    bestScore = adjustedScore;
                    bestIndex = candidateIndex;
                }
            }
            const auto selected = candidates[bestIndex];
            selectedOffsets.push_back({selected.offsetX, selected.offsetY});
            stimuli.push_back(applyImageFocusTransform(
                image,
                guidedZoom,
                guidedJitter * selected.offsetX,
                guidedJitter * selected.offsetY));
        }
        return stimuli;
    }

    std::vector<VisualStimulus> stimuli;
    stimuli.reserve(static_cast<size_t>(fixations));
    stimuli.push_back(image);
    for (int fixation = 1; fixation < fixations; ++fixation) {
        const auto& [offsetX, offsetY] =
            kOffsets[static_cast<size_t>((fixation - 1) % static_cast<int>(kOffsets.size()))];
        stimuli.push_back(applyImageFocusTransform(
            image,
            config.saccadeZoom,
            config.saccadeJitterPx * offsetX,
            config.saccadeJitterPx * offsetY));
    }
    return stimuli;
}

std::vector<VisualStimulus> buildExtractionStimuli(const VisualStimulus& image,
                                                   const Config& config,
                                                   bool trainingPhase,
                                                   size_t sampleSeed,
                                                   std::vector<std::unique_ptr<RetinaAdapter>>* retinas = nullptr) {
    const auto augmentedImages =
        buildTrainingAugmentationStimuli(image, config, trainingPhase, sampleSeed);
    if (augmentedImages.size() == 1) {
        return buildSaccadeStimuli(augmentedImages.front(), config, trainingPhase, retinas);
    }

    std::vector<VisualStimulus> stimuli;
    const bool enableSampling = trainingPhase || !config.saccadeTrainingOnly;
    const size_t fixationsPerImage = static_cast<size_t>(
        enableSampling ? std::max(1, config.saccadeFixations) : 1);
    stimuli.reserve(augmentedImages.size() * fixationsPerImage);
    for (const auto& augmentedImage : augmentedImages) {
        auto fixationImages = buildSaccadeStimuli(augmentedImage, config, trainingPhase, retinas);
        stimuli.insert(stimuli.end(), fixationImages.begin(), fixationImages.end());
    }
    return stimuli;
}

std::vector<VisualStimulus> buildRecurrentExtractionStimuli(
    const VisualStimulus& image,
    const Config& config,
    bool trainingPhase,
    size_t sampleSeed,
    std::vector<std::unique_ptr<RetinaAdapter>>* retinas) {
    Config recurrentConfig = config;
    recurrentConfig.saccadeTrainingOnly = false;
    recurrentConfig.guidedSaccadeTrainingOnly = false;
    return buildExtractionStimuli(image, recurrentConfig, trainingPhase, sampleSeed, retinas);
}

struct FixationSpec {
    VisualStimulus image;
    double shiftXPx = 0.0;
    double shiftYPx = 0.0;
    double zoom = 1.0;
    double saliency = 0.0;
};

struct HemispherePatternAccumulator {
    std::vector<std::vector<double>> retinaSums;
    int fixationCount = 0;
};

double computeCentralFixationSaliency(const VisualStimulus& image) {
    if (image.rows <= 0 || image.cols <= 0 || image.pixels.empty()) {
        return 0.0;
    }

    const int rowStart = image.rows / 4;
    const int rowEnd = std::max(rowStart + 1, (3 * image.rows) / 4);
    const int colStart = image.cols / 4;
    const int colEnd = std::max(colStart + 1, (3 * image.cols) / 4);

    double luminanceSum = 0.0;
    double luminanceSqSum = 0.0;
    double gradientSum = 0.0;
    int count = 0;
    for (int row = rowStart; row < rowEnd; ++row) {
        for (int col = colStart; col < colEnd; ++col) {
            const auto luminanceAt = [&](int r, int c) {
                r = std::clamp(r, 0, image.rows - 1);
                c = std::clamp(c, 0, image.cols - 1);
                const size_t base = (static_cast<size_t>(r * image.cols + c) *
                                     static_cast<size_t>(std::max(1, image.channels)));
                if (image.channels >= 3) {
                    const double red = static_cast<double>(image.pixels[base]) / 255.0;
                    const double green = static_cast<double>(image.pixels[base + 1]) / 255.0;
                    const double blue = static_cast<double>(image.pixels[base + 2]) / 255.0;
                    return 0.299 * red + 0.587 * green + 0.114 * blue;
                }
                return static_cast<double>(image.pixels[base]) / 255.0;
            };

            const double center = luminanceAt(row, col);
            const double left = luminanceAt(row, col - 1);
            const double right = luminanceAt(row, col + 1);
            const double up = luminanceAt(row - 1, col);
            const double down = luminanceAt(row + 1, col);
            const double dx = right - left;
            const double dy = down - up;
            luminanceSum += center;
            luminanceSqSum += center * center;
            gradientSum += std::sqrt(dx * dx + dy * dy);
            ++count;
        }
    }

    if (count <= 0) {
        return 0.0;
    }
    const double mean = luminanceSum / static_cast<double>(count);
    const double variance =
        std::max(0.0, (luminanceSqSum / static_cast<double>(count)) - (mean * mean));
    const double stddev = std::sqrt(variance);
    const double meanGradient = gradientSum / static_cast<double>(count);
    return (0.65 * meanGradient) + (0.35 * stddev);
}

double computeFixationRegionSaliency(const VisualStimulus& image,
                                     int rowStart,
                                     int rowEnd,
                                     int colStart,
                                     int colEnd) {
    if (image.rows <= 0 || image.cols <= 0 || image.pixels.empty()) {
        return 0.0;
    }

    rowStart = std::clamp(rowStart, 0, image.rows - 1);
    rowEnd = std::clamp(rowEnd, rowStart + 1, image.rows);
    colStart = std::clamp(colStart, 0, image.cols - 1);
    colEnd = std::clamp(colEnd, colStart + 1, image.cols);
    if (rowStart >= rowEnd || colStart >= colEnd) {
        return 0.0;
    }

    const auto luminanceAt = [&](int r, int c) {
        r = std::clamp(r, 0, image.rows - 1);
        c = std::clamp(c, 0, image.cols - 1);
        const size_t base = (static_cast<size_t>(r * image.cols + c) *
                             static_cast<size_t>(std::max(1, image.channels)));
        if (image.channels >= 3) {
            const double red = static_cast<double>(image.pixels[base]) / 255.0;
            const double green = static_cast<double>(image.pixels[base + 1]) / 255.0;
            const double blue = static_cast<double>(image.pixels[base + 2]) / 255.0;
            return 0.299 * red + 0.587 * green + 0.114 * blue;
        }
        return static_cast<double>(image.pixels[base]) / 255.0;
    };

    double luminanceSum = 0.0;
    double luminanceSqSum = 0.0;
    double gradientSum = 0.0;
    int count = 0;
    for (int row = rowStart; row < rowEnd; ++row) {
        for (int col = colStart; col < colEnd; ++col) {
            const double center = luminanceAt(row, col);
            const double left = luminanceAt(row, col - 1);
            const double right = luminanceAt(row, col + 1);
            const double up = luminanceAt(row - 1, col);
            const double down = luminanceAt(row + 1, col);
            const double dx = right - left;
            const double dy = down - up;
            luminanceSum += center;
            luminanceSqSum += center * center;
            gradientSum += std::sqrt(dx * dx + dy * dy);
            ++count;
        }
    }

    if (count <= 0) {
        return 0.0;
    }
    const double mean = luminanceSum / static_cast<double>(count);
    const double variance =
        std::max(0.0, (luminanceSqSum / static_cast<double>(count)) - (mean * mean));
    const double stddev = std::sqrt(variance);
    const double meanGradient = gradientSum / static_cast<double>(count);
    return (0.65 * meanGradient) + (0.35 * stddev);
}

std::vector<FixationSpec> buildActiveInferenceFixationCandidates(const VisualStimulus& image,
                                                                 const Config& config) {
    struct TileCandidate {
        int row = 0;
        int col = 0;
        double saliency = 0.0;
        double normalizedDx = 0.0;
        double normalizedDy = 0.0;
    };

    const int grid = 5;
    const double centerX = 0.5 * static_cast<double>(std::max(1, image.cols - 1));
    const double centerY = 0.5 * static_cast<double>(std::max(1, image.rows - 1));
    const double halfWidth = std::max(1.0, 0.5 * static_cast<double>(image.cols));
    const double halfHeight = std::max(1.0, 0.5 * static_cast<double>(image.rows));

    std::vector<TileCandidate> rankedTiles;
    rankedTiles.reserve(static_cast<size_t>(grid * grid));
    for (int row = 0; row < grid; ++row) {
        const int rowStart = (row * image.rows) / grid;
        const int rowEnd = std::max(rowStart + 1, ((row + 1) * image.rows) / grid);
        const double tileCenterY = 0.5 * static_cast<double>(rowStart + rowEnd - 1);
        for (int col = 0; col < grid; ++col) {
            const int colStart = (col * image.cols) / grid;
            const int colEnd = std::max(colStart + 1, ((col + 1) * image.cols) / grid);
            const double tileCenterX = 0.5 * static_cast<double>(colStart + colEnd - 1);
            TileCandidate tile;
            tile.row = row;
            tile.col = col;
            tile.saliency = computeFixationRegionSaliency(
                image, rowStart, rowEnd, colStart, colEnd);
            tile.normalizedDx = (tileCenterX - centerX) / halfWidth;
            tile.normalizedDy = (tileCenterY - centerY) / halfHeight;
            rankedTiles.push_back(tile);
        }
    }

    std::sort(rankedTiles.begin(), rankedTiles.end(),
              [](const TileCandidate& lhs, const TileCandidate& rhs) {
                  if (lhs.saliency != rhs.saliency) {
                      return lhs.saliency > rhs.saliency;
                  }
                  if (lhs.row != rhs.row) {
                      return lhs.row < rhs.row;
                  }
                  return lhs.col < rhs.col;
              });

    std::vector<FixationSpec> candidates;
    candidates.reserve(1 + rankedTiles.size());
    candidates.push_back({image, 0.0, 0.0, 1.0, computeCentralFixationSaliency(image)});
    for (const auto& tile : rankedTiles) {
        if (std::abs(tile.normalizedDx) < 1e-6 && std::abs(tile.normalizedDy) < 1e-6) {
            continue;
        }
        FixationSpec fixation;
        fixation.shiftXPx = -config.activeInferenceShiftPx * tile.normalizedDx;
        fixation.shiftYPx = -config.activeInferenceShiftPx * tile.normalizedDy;
        fixation.zoom = config.activeInferenceZoom;
        fixation.image = applyImageFocusTransform(
            image, fixation.zoom, fixation.shiftXPx, fixation.shiftYPx);
        fixation.saliency = tile.saliency;
        candidates.push_back(std::move(fixation));
    }
    return candidates;
}

size_t selectNextActiveFixation(const std::vector<FixationSpec>& candidates,
                                const std::vector<size_t>& selected,
                                const Config& config) {
    size_t bestIndex = candidates.size();
    double bestScore = -std::numeric_limits<double>::infinity();
    const double sigma = std::max(1.0, config.activeInferenceShiftPx);
    const double sigmaSq = sigma * sigma;

    for (size_t candidateIndex = 1; candidateIndex < candidates.size(); ++candidateIndex) {
        if (std::find(selected.begin(), selected.end(), candidateIndex) != selected.end()) {
            continue;
        }
        double score = candidates[candidateIndex].saliency;
        for (size_t chosenIndex : selected) {
            if (chosenIndex >= candidates.size()) {
                continue;
            }
            const double dx = candidates[candidateIndex].shiftXPx - candidates[chosenIndex].shiftXPx;
            const double dy = candidates[candidateIndex].shiftYPx - candidates[chosenIndex].shiftYPx;
            const double distSq = dx * dx + dy * dy;
            score -= config.activeInferenceIorStrength * std::exp(-0.5 * distSq / sigmaSq);
        }
        if (score > bestScore) {
            bestScore = score;
            bestIndex = candidateIndex;
        }
    }

    return bestIndex;
}

double computeDecisionUncertainty(const BilateralDecisionTrace& decision) {
    if (decision.topHypotheses.empty()) {
        return 1.0;
    }
    const double best = decision.topHypotheses.front().score;
    const double second = decision.topHypotheses.size() > 1 ? decision.topHypotheses[1].score : 0.0;
    if (best <= 1e-6) {
        return 1.0;
    }
    return 1.0 - std::clamp((best - second) / best, 0.0, 1.0);
}

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::vector<Stage1TapInput> buildJepaStage1TapInputs(
    std::vector<HemisphereRuntime>& hemispheres,
    const VisualDomainAdapter& trainLoader,
    const Config& config) {
    std::vector<Stage1TapInput> taps;
    if (!config.jepa.enabled) {
        return taps;
    }

    const bool useRawStage1 =
        config.jepa.tapSurface == snnfw::jepa::TapSurface::RawStage1;
    const std::string surfaceName = useRawStage1 ? "raw_stage1" : "promoted_stage1";
    const bool temporalFixationMode =
        config.jepa.targetMode == snnfw::jepa::TargetMode::TemporalFixation &&
        config.saccadeFixations > 1;

    for (auto& hemisphere : hemispheres) {
        if (temporalFixationMode && !hemisphere.trainingSourceIndices.empty()) {
            std::vector<size_t> uniqueSourceIndices;
            uniqueSourceIndices.reserve(hemisphere.trainingSourceIndices.size());
            std::unordered_set<size_t> seenSources;
            for (size_t sourceIndex : hemisphere.trainingSourceIndices) {
                if (seenSources.insert(sourceIndex).second) {
                    uniqueSourceIndices.push_back(sourceIndex);
                }
            }
            size_t available = uniqueSourceIndices.size();
            if (config.jepa.maxSamples > 0) {
                available = std::min(
                    available, static_cast<size_t>(std::max(0, config.jepa.maxSamples)));
            }
            for (size_t i = 0; i < available; ++i) {
                const size_t sourceIndex = uniqueSourceIndices[i];
                const auto& image = trainLoader.getStimulus(sourceIndex);
                if (hemisphere.retinaPatternSizes.empty()) {
                    hemisphere.retinaPatternSizes = inferHemisphereBranchPatternSizes(
                        hemisphere.retinas, image, config.useFeatures);
                }
                const auto fixationImages =
                    buildExtractionStimuli(image, config, true, sourceIndex, &hemisphere.retinas);
                for (size_t fixationIndex = 0; fixationIndex < fixationImages.size(); ++fixationIndex) {
                    Stage1TapInput tap;
                    tap.hemisphereName = hemisphere.name;
                    tap.surfaceName = surfaceName;
                    tap.label = image.label;
                    tap.sourceIndex = sourceIndex;
                    tap.sourceViewIndex = fixationIndex;
                    tap.pattern = extractPattern(
                        hemisphere.retinas,
                        fixationImages[fixationIndex],
                        config,
                        false,
                        false,
                        0U,
                        nullptr);
                    tap.branchSizes = hemisphere.retinaPatternSizes;
                    const size_t branchTotal = std::accumulate(
                        tap.branchSizes.begin(), tap.branchSizes.end(), static_cast<size_t>(0));
                    if (branchTotal != tap.pattern.size()) {
                        tap.branchSizes.clear();
                    }
                    taps.push_back(std::move(tap));
                }
            }
            continue;
        }

        std::unordered_map<size_t, size_t> sourceViewCounts;
        const auto& patterns =
            useRawStage1 ? hemisphere.rawTrainingPatterns : hemisphere.trainingPatterns;
        if (patterns.empty()) {
            continue;
        }

        if (hemisphere.retinaPatternSizes.empty() && !hemisphere.trainingSourceIndices.empty()) {
            const auto& referenceImage =
                trainLoader.getStimulus(hemisphere.trainingSourceIndices.front());
            hemisphere.retinaPatternSizes = inferHemisphereBranchPatternSizes(
                hemisphere.retinas, referenceImage, config.useFeatures);
        }

        size_t available = patterns.size();
        if (!hemisphere.trainingSourceIndices.empty()) {
            available = std::min(available, hemisphere.trainingSourceIndices.size());
        }
        if (config.jepa.maxSamples > 0) {
            available = std::min(
                available, static_cast<size_t>(std::max(0, config.jepa.maxSamples)));
        }

        for (size_t i = 0; i < available; ++i) {
            Stage1TapInput tap;
            tap.hemisphereName = hemisphere.name;
            tap.surfaceName = surfaceName;
            tap.label = patterns[i].label;
            tap.sourceIndex =
                hemisphere.trainingSourceIndices.empty() ? i : hemisphere.trainingSourceIndices[i];
            tap.sourceViewIndex = sourceViewCounts[tap.sourceIndex]++;
            tap.pattern = patterns[i].pattern;
            tap.branchSizes = hemisphere.retinaPatternSizes;
            const size_t branchTotal = std::accumulate(
                tap.branchSizes.begin(), tap.branchSizes.end(), static_cast<size_t>(0));
            if (branchTotal != tap.pattern.size()) {
                tap.branchSizes.clear();
            }
            taps.push_back(std::move(tap));
        }
    }

    return taps;
}

std::vector<Stage1TapInput> buildJepaStage1EvalTapInputs(
    std::vector<HemisphereRuntime>& hemispheres,
    const VisualDomainAdapter& loader,
    const std::vector<size_t>& indices,
    const Config& config) {
    std::vector<Stage1TapInput> taps;
    if (!config.jepa.enabled) {
        return taps;
    }

    const bool useRawStage1 =
        config.jepa.tapSurface == snnfw::jepa::TapSurface::RawStage1;
    const std::string surfaceName = useRawStage1 ? "raw_stage1" : "promoted_stage1";
    taps.reserve(indices.size() * hemispheres.size());

    for (size_t sourceIndex : indices) {
        const auto& image = loader.getStimulus(sourceIndex);
        const int label = image.label;
        if (label < 0 || label >= config.numClasses) {
            continue;
        }

        if (useRawStage1) {
            for (auto& hemisphere : hemispheres) {
                if (hemisphere.retinaPatternSizes.empty()) {
                    hemisphere.retinaPatternSizes = inferHemisphereBranchPatternSizes(
                        hemisphere.retinas, image, config.useFeatures);
                }

                Stage1TapInput tap;
                tap.hemisphereName = hemisphere.name;
                tap.surfaceName = surfaceName;
                tap.label = label;
                tap.sourceIndex = sourceIndex;
                tap.sourceViewIndex = 0;
                tap.pattern = extractPattern(
                    hemisphere.retinas, image, config, false, false, 0U, nullptr);
                tap.branchSizes = hemisphere.retinaPatternSizes;
                const size_t branchTotal = std::accumulate(
                    tap.branchSizes.begin(), tap.branchSizes.end(), static_cast<size_t>(0));
                if (branchTotal != tap.pattern.size()) {
                    tap.branchSizes.clear();
                }
                taps.push_back(std::move(tap));
            }
            continue;
        }

        const auto traces = inferHemisphereDecisions(hemispheres, image, config);
        for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
            Stage1TapInput tap;
            tap.hemisphereName = hemispheres[hemisphereIndex].name;
            tap.surfaceName = surfaceName;
            tap.label = label;
            tap.sourceIndex = sourceIndex;
            tap.sourceViewIndex = 0;
            if (hemisphereIndex < traces.size()) {
                tap.pattern = traces[hemisphereIndex].classifierPattern.empty()
                    ? traces[hemisphereIndex].pattern
                    : traces[hemisphereIndex].classifierPattern;
            }
            taps.push_back(std::move(tap));
        }
    }

    return taps;
}

void maybeExportJepaStage1Taps(std::vector<HemisphereRuntime>& hemispheres,
                               const VisualDomainAdapter& trainLoader,
                               const Config& config) {
    if (!config.jepa.enabled) {
        return;
    }

    auto taps = buildJepaStage1TapInputs(hemispheres, trainLoader, config);
    if (taps.empty()) {
        std::cout << "  JEPA tap export skipped: no stage-1 patterns available" << std::endl;
        return;
    }

    const std::string outputPath = config.jepa.dumpPath.empty()
        ? "build/jepa_stage1_latents.json"
        : config.jepa.dumpPath;
    const auto summary =
        snnfw::jepa::writeStage1LatentSamples(taps, config.jepa, outputPath);
    std::cout << "  JEPA stage1 taps exported: samples=" << summary.sampleCount
              << ", hemisphere_views=" << summary.hemisphereViewCount
              << ", masked_views=" << summary.maskedViewCount
              << ", temporal_pairs=" << summary.temporalPairCount
              << ", branch_tokens=" << summary.branchTokenCount
              << ", hemisphere_tokens=" << summary.hemisphereTokenCount
              << ", leakage_violations=" << summary.leakageViolationCount
              << ", path=" << outputPath << std::endl;
}

bool maybeTrainJepaModel(std::vector<HemisphereRuntime>& hemispheres,
                         const VisualDomainAdapter& trainLoader,
                         const Config& config,
                         snnfw::jepa::JepaTrainingArtifacts* artifactsOut) {
    if (!config.jepa.enabled || !config.jepa.trainerEnabled) {
        return false;
    }

    auto taps = buildJepaStage1TapInputs(hemispheres, trainLoader, config);
    if (taps.empty()) {
        std::cout << "  JEPA trainer skipped: no stage-1 patterns available" << std::endl;
        return false;
    }

    const auto samples = snnfw::jepa::buildStage1LatentSamples(taps, config.jepa);
    const std::string outputPath = config.jepa.trainerDumpPath.empty()
        ? "build/jepa_minimal_trainer.json"
        : config.jepa.trainerDumpPath;
    auto artifacts = snnfw::jepa::trainMinimalModel(samples, config.jepa, outputPath);
    const auto& summary = artifacts.summary;
    std::cout << "  JEPA trainer summary: examples=" << summary.trainingExampleCount
              << ", target_mode=" << snnfw::jepa::targetModeName(config.jepa.targetMode)
              << ", temporal_examples=" << summary.temporalExampleCount
              << ", fallback_masked_examples=" << summary.fallbackMaskedExampleCount
              << ", epochs=" << summary.epochCount
              << ", projection_dim=" << summary.projectionDim
              << ", loss=" << std::fixed << std::setprecision(4) << summary.meanLoss
              << ", shuffled_loss=" << summary.meanShuffledLoss
              << ", pred_var=" << summary.predictionVariance
              << ", target_var=" << summary.targetVariance
              << ", variance_penalty=" << summary.meanVariancePenalty
              << ", path=" << outputPath << std::endl;
    if (artifactsOut != nullptr) {
        *artifactsOut = std::move(artifacts);
    }
    return true;
}

JepaProbeResult evaluateJepaProbe(std::vector<HemisphereRuntime>& hemispheres,
                                  const VisualDomainAdapter& trainLoader,
                                  const VisualDomainAdapter& testLoader,
                                  const std::vector<size_t>& indices,
                                  const Config& config,
                                  const snnfw::jepa::JepaModel& model,
                                  const std::string& label) {
    JepaProbeResult result;
    result.confusion.assign(
        static_cast<size_t>(config.numClasses),
        std::vector<int>(static_cast<size_t>(config.numClasses), 0));
    const auto start = std::chrono::high_resolution_clock::now();

    const auto trainTaps = buildJepaStage1TapInputs(hemispheres, trainLoader, config);
    const auto trainSamples = snnfw::jepa::buildStage1LatentSamples(trainTaps, config.jepa);
    std::vector<ClassificationStrategy::LabeledPattern> trainingPatterns;
    trainingPatterns.reserve(trainSamples.size());
    for (const auto& sample : trainSamples) {
        auto embedding = snnfw::jepa::encodeSample(sample, model);
        if (embedding.empty()) {
            continue;
        }
        trainingPatterns.emplace_back(embedding, sample.label);
    }
    if (trainingPatterns.empty()) {
        throw std::runtime_error("JEPA probe found no valid training embeddings");
    }

    auto classifier = makeClassifierStrategy(
        config.classifier, config.knnK, config.classifierExponent, config);
    const auto evalTaps = buildJepaStage1EvalTapInputs(hemispheres, testLoader, indices, config);
    const auto evalSamples = snnfw::jepa::buildStage1LatentSamples(evalTaps, config.jepa);
    const int maxTests = static_cast<int>(evalSamples.size());
    const int progressStep = maxTests >= 1000 ? 200 : (maxTests >= 200 ? 50 : 25);

    for (const auto& sample : evalSamples) {
        if (sample.label < 0 || sample.label >= config.numClasses) {
            continue;
        }
        auto embedding = snnfw::jepa::encodeSample(sample, model);
        if (embedding.empty()) {
            continue;
        }

        const int predicted =
            classifier->classify(embedding, trainingPatterns, cosineSimilarity);
        result.confusion[static_cast<size_t>(sample.label)][static_cast<size_t>(predicted)]++;
        result.correct += (predicted == sample.label) ? 1 : 0;
        result.tested++;

        if (result.tested % progressStep == 0) {
            const double acc =
                100.0 * static_cast<double>(result.correct) /
                static_cast<double>(std::max(1, result.tested));
            std::cout << "  " << label << ": " << result.tested << "/" << maxTests
                      << " (" << std::fixed << std::setprecision(2) << acc << "%)" << std::endl;
        }
    }

    const auto end = std::chrono::high_resolution_clock::now();
    result.seconds = std::chrono::duration<double>(end - start).count();
    return result;
}

std::string trim(std::string value) {
    const auto isSpace = [](unsigned char c) { return std::isspace(c) != 0; };
    value.erase(value.begin(),
                std::find_if(value.begin(), value.end(),
                             [&](unsigned char c) { return !isSpace(c); }));
    value.erase(std::find_if(value.rbegin(), value.rend(),
                             [&](unsigned char c) { return !isSpace(c); }).base(),
                value.end());
    return value;
}

std::string normalizeHierarchyPath(const std::string& path) {
    std::stringstream ss(path);
    std::string token;
    std::vector<std::string> parts;
    while (std::getline(ss, token, '/')) {
        token = trim(token);
        if (!token.empty()) {
            parts.push_back(token);
        }
    }

    std::ostringstream oss;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) {
            oss << "/";
        }
        oss << parts[i];
    }
    return oss.str();
}

std::vector<ColumnIR> expandColumnTemplateForPaths(const ColumnTemplateIR& tmpl) {
    std::vector<ColumnIR> result;
    for (double orientation : tmpl.orientations) {
        for (double frequency : tmpl.frequencies) {
            ColumnIR column;
            std::string name = tmpl.namingPattern;

            auto pos = name.find("{orientation}");
            if (pos != std::string::npos) {
                std::ostringstream oss;
                if (orientation == static_cast<int>(orientation)) {
                    oss << static_cast<int>(orientation);
                } else {
                    oss << orientation;
                }
                name.replace(pos, 13, oss.str());
            }

            pos = name.find("{frequency}");
            if (pos != std::string::npos) {
                std::ostringstream oss;
                if (frequency == static_cast<int>(frequency)) {
                    oss << static_cast<int>(frequency);
                } else {
                    oss << frequency;
                }
                name.replace(pos, 11, oss.str());
            }

            column.name = name;
            column.properties["orientation"] = orientation;
            column.properties["spatial_frequency"] = frequency;
            column.layers = tmpl.layers;
            result.push_back(std::move(column));
        }
    }
    return result;
}

std::unordered_map<std::string, RetinaLayerBinding> buildRetinaLayerIndex(const NetworkIR& ir) {
    std::unordered_map<std::string, RetinaLayerBinding> index;

    for (const auto& hemisphere : ir.brain.hemispheres) {
        for (const auto& lobe : hemisphere.lobes) {
            for (const auto& region : lobe.regions) {
                for (const auto& nucleus : region.nuclei) {
                    std::vector<ColumnIR> columns = nucleus.columns;
                    if (nucleus.columnTemplate.has_value()) {
                        auto expanded = expandColumnTemplateForPaths(nucleus.columnTemplate.value());
                        columns.insert(columns.end(), expanded.begin(), expanded.end());
                    }

                    for (const auto& column : columns) {
                        for (const auto& layer : column.layers) {
                            RetinaLayerBinding binding;
                            binding.hemisphere = hemisphere.name;
                            binding.lobe = lobe.name;
                            binding.region = region.name;
                            binding.nucleus = nucleus.name;
                            binding.column = column.name;
                            binding.layer = layer.name;
                            binding.path = normalizeHierarchyPath(
                                hemisphere.name + "/" + lobe.name + "/" + region.name + "/" +
                                nucleus.name + "/" + column.name + "/" + layer.name);
                            index[binding.path] = binding;
                        }
                    }
                }
            }
        }
    }

    return index;
}

BaseAdapter::Config makeDefaultRetinaConfig(const Config& config, int gridSize, size_t index) {
    BaseAdapter::Config retinaConfig;
    retinaConfig.name = "emnist_retina_" + std::to_string(index);
    retinaConfig.type = "retina";
    retinaConfig.temporalWindow = config.temporalWindowMs;
    retinaConfig.intParams["grid_size"] = gridSize;
    retinaConfig.intParams["num_orientations"] = config.numOrientations;
    retinaConfig.doubleParams["edge_threshold"] = config.edgeThreshold;
    retinaConfig.doubleParams["neuron_window_size"] = config.temporalWindowMs;
    retinaConfig.doubleParams["neuron_threshold"] = 0.7;
    retinaConfig.intParams["neuron_max_patterns"] = 100;
    retinaConfig.stringParams["edge_operator"] = config.edgeOperator;
    retinaConfig.stringParams["encoding_strategy"] = config.encodingStrategy;
    retinaConfig.stringParams["activation_mode"] = config.activationMode;
    return retinaConfig;
}

void applyClassificationConfig(const ClassificationConfigIR& irConfig, Config& config) {
    if (!irConfig.type.empty()) {
        config.classifier = irConfig.type;
    }
    config.knnK = irConfig.k;
    config.classifierExponent = irConfig.distanceExponent;
    config.stage1Classifier = config.classifier;
    config.fusionClassifier = config.classifier;
    config.stage1K = config.knnK;
    config.fusionK = config.knnK;
    config.stage1Exponent = config.classifierExponent;
    config.fusionExponent = config.classifierExponent;

    const auto groupIt = irConfig.stringParams.find("group_definitions");
    if (groupIt != irConfig.stringParams.end()) {
        config.hierarchicalGroups = groupIt->second;
    }
    const auto coarseStrategyIt = irConfig.stringParams.find("coarse_strategy");
    if (coarseStrategyIt != irConfig.stringParams.end()) {
        config.hierarchicalCoarseStrategy = coarseStrategyIt->second;
    }
    const auto fineStrategyIt = irConfig.stringParams.find("fine_strategy");
    if (fineStrategyIt != irConfig.stringParams.end()) {
        config.hierarchicalFineStrategy = fineStrategyIt->second;
    }
    const auto coarseKIt = irConfig.intParams.find("coarse_k");
    if (coarseKIt != irConfig.intParams.end()) {
        config.hierarchicalCoarseK = coarseKIt->second;
    }
    const auto fineKIt = irConfig.intParams.find("fine_k");
    if (fineKIt != irConfig.intParams.end()) {
        config.hierarchicalFineK = fineKIt->second;
    }
    const auto fusionModeIt = irConfig.stringParams.find("fusion_mode");
    if (fusionModeIt != irConfig.stringParams.end()) {
        std::string value = fusionModeIt->second;
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        config.bilateralFusion = (value == "bilateral");
    }
    const auto stage1StrategyIt = irConfig.stringParams.find("stage1_strategy");
    if (stage1StrategyIt != irConfig.stringParams.end()) {
        config.stage1Classifier = stage1StrategyIt->second;
    }
    const auto fusionStrategyIt = irConfig.stringParams.find("fusion_strategy");
    if (fusionStrategyIt != irConfig.stringParams.end()) {
        config.fusionClassifier = fusionStrategyIt->second;
    }
    const auto fusionFeatureModeIt = irConfig.stringParams.find("fusion_feature_mode");
    if (fusionFeatureModeIt != irConfig.stringParams.end()) {
        config.fusionFeatureMode = fusionFeatureModeIt->second;
    }
    const auto representationIt = irConfig.stringParams.find("representation");
    if (representationIt != irConfig.stringParams.end()) {
        const std::string value = toLower(representationIt->second);
        if (value == "features") {
            config.useFeatures = true;
        } else if (value == "activations") {
            config.useFeatures = false;
        }
    }
    const auto fusionPathIt = irConfig.stringParams.find("fusion_path");
    if (fusionPathIt != irConfig.stringParams.end()) {
        config.fusionPath = normalizeHierarchyPath(fusionPathIt->second);
    }
    const auto trainingCurriculumIt =
        irConfig.stringParams.find("training_curriculum_groups");
    if (trainingCurriculumIt != irConfig.stringParams.end()) {
        config.trainingCurriculumSpec = trainingCurriculumIt->second;
    }
    const auto stage1KIt = irConfig.intParams.find("stage1_k");
    if (stage1KIt != irConfig.intParams.end()) {
        config.stage1K = stage1KIt->second;
    }
    const auto fusionKIt = irConfig.intParams.find("fusion_k");
    if (fusionKIt != irConfig.intParams.end()) {
        config.fusionK = fusionKIt->second;
    }
    const auto fusionHoldoutIt = irConfig.intParams.find("fusion_holdout_per_class");
    if (fusionHoldoutIt != irConfig.intParams.end()) {
        config.fusionHoldoutPerClass = fusionHoldoutIt->second;
    }
    const auto stage1ExpIt = irConfig.doubleParams.find("stage1_distance_exponent");
    if (stage1ExpIt != irConfig.doubleParams.end()) {
        config.stage1Exponent = stage1ExpIt->second;
    }
    const auto fusionExpIt = irConfig.doubleParams.find("fusion_distance_exponent");
    if (fusionExpIt != irConfig.doubleParams.end()) {
        config.fusionExponent = fusionExpIt->second;
    }
    const auto corpusVoteGainIt = irConfig.doubleParams.find("corpus_vote_gain");
    if (corpusVoteGainIt != irConfig.doubleParams.end()) {
        config.corpusVoteGain = corpusVoteGainIt->second;
    }
    const auto corpusMarginGainIt = irConfig.doubleParams.find("corpus_margin_gain");
    if (corpusMarginGainIt != irConfig.doubleParams.end()) {
        config.corpusMarginGain = corpusMarginGainIt->second;
    }
    const auto corpusCentroidGainIt = irConfig.doubleParams.find("corpus_centroid_gain");
    if (corpusCentroidGainIt != irConfig.doubleParams.end()) {
        config.corpusCentroidGain = corpusCentroidGainIt->second;
    }
    const auto corpusNeighborGainIt = irConfig.doubleParams.find("corpus_neighbor_gain");
    if (corpusNeighborGainIt != irConfig.doubleParams.end()) {
        config.corpusNeighborGain = corpusNeighborGainIt->second;
    }
    const auto corpusDisagreementGainIt =
        irConfig.doubleParams.find("corpus_disagreement_gain");
    if (corpusDisagreementGainIt != irConfig.doubleParams.end()) {
        config.corpusDisagreementGain = corpusDisagreementGainIt->second;
    }
    const auto predictionErrorFeedbackEnabledIt =
        irConfig.intParams.find("prediction_error_feedback_enabled");
    if (predictionErrorFeedbackEnabledIt != irConfig.intParams.end()) {
        config.predictionErrorFeedbackEnabled =
            predictionErrorFeedbackEnabledIt->second != 0;
    }
    const auto predictionErrorFeedbackGainIt =
        irConfig.doubleParams.find("prediction_error_feedback_gain");
    if (predictionErrorFeedbackGainIt != irConfig.doubleParams.end()) {
        config.predictionErrorFeedbackGain =
            std::max(0.0, predictionErrorFeedbackGainIt->second);
    }
    const auto predictionErrorFeedbackThresholdIt =
        irConfig.doubleParams.find("prediction_error_feedback_threshold");
    if (predictionErrorFeedbackThresholdIt != irConfig.doubleParams.end()) {
        config.predictionErrorFeedbackThreshold =
            std::clamp(predictionErrorFeedbackThresholdIt->second, 0.0, 1.0);
    }
    const auto predictionErrorFeedbackMaxPenaltyIt =
        irConfig.doubleParams.find("prediction_error_feedback_max_penalty");
    if (predictionErrorFeedbackMaxPenaltyIt != irConfig.doubleParams.end()) {
        config.predictionErrorFeedbackMaxPenalty =
            std::clamp(predictionErrorFeedbackMaxPenaltyIt->second, 0.0, 1.0);
    }
    const auto predictionErrorFeedbackMinConfidenceIt =
        irConfig.doubleParams.find("prediction_error_feedback_min_confidence");
    if (predictionErrorFeedbackMinConfidenceIt != irConfig.doubleParams.end()) {
        config.predictionErrorFeedbackMinConfidence =
            std::clamp(predictionErrorFeedbackMinConfidenceIt->second, 0.0, 1.0);
    }
    const auto onlineRepeatsIt = irConfig.intParams.find("online_correction_repeats");
    if (onlineRepeatsIt != irConfig.intParams.end()) {
        config.onlineCorrectionRepeats = onlineRepeatsIt->second;
    }
    const auto onlineBudgetIt = irConfig.intParams.find("online_exemplar_budget_per_class");
    if (onlineBudgetIt != irConfig.intParams.end()) {
        config.onlineExemplarBudgetPerClass = onlineBudgetIt->second;
    }
    const auto onlineCentroidLrIt = irConfig.doubleParams.find("online_centroid_lr");
    if (onlineCentroidLrIt != irConfig.doubleParams.end()) {
        config.onlineCentroidLr = onlineCentroidLrIt->second;
    }
    const auto onlinePositiveGainIt = irConfig.doubleParams.find("online_positive_reward_gain");
    if (onlinePositiveGainIt != irConfig.doubleParams.end()) {
        config.onlinePositiveRewardGain = onlinePositiveGainIt->second;
    }
    const auto onlineNegativeGainIt = irConfig.doubleParams.find("online_negative_reward_gain");
    if (onlineNegativeGainIt != irConfig.doubleParams.end()) {
        config.onlineNegativeRewardGain = onlineNegativeGainIt->second;
    }
    const auto onlineRewardStdpEnabledIt =
        irConfig.intParams.find("online_reward_stdp_enabled");
    if (onlineRewardStdpEnabledIt != irConfig.intParams.end()) {
        config.onlineRewardStdpEnabled = onlineRewardStdpEnabledIt->second != 0;
    }
    const auto onlineRewardStdpGainIt =
        irConfig.doubleParams.find("online_reward_stdp_gain");
    if (onlineRewardStdpGainIt != irConfig.doubleParams.end()) {
        config.onlineRewardStdpGain = onlineRewardStdpGainIt->second;
    }
    const auto onlineRewardStdpLtpIt =
        irConfig.doubleParams.find("online_reward_stdp_ltp");
    if (onlineRewardStdpLtpIt != irConfig.doubleParams.end()) {
        config.onlineRewardStdpLtp = onlineRewardStdpLtpIt->second;
    }
    const auto onlineRewardStdpLtdIt =
        irConfig.doubleParams.find("online_reward_stdp_ltd");
    if (onlineRewardStdpLtdIt != irConfig.doubleParams.end()) {
        config.onlineRewardStdpLtd = onlineRewardStdpLtdIt->second;
    }
    const auto onlineTripletStdpEnabledIt =
        irConfig.intParams.find("online_triplet_stdp_enabled");
    if (onlineTripletStdpEnabledIt != irConfig.intParams.end()) {
        config.onlineTripletStdpEnabled = onlineTripletStdpEnabledIt->second != 0;
    }
    const auto onlineTripletStdpGainIt =
        irConfig.doubleParams.find("online_triplet_stdp_gain");
    if (onlineTripletStdpGainIt != irConfig.doubleParams.end()) {
        config.onlineTripletStdpGain = onlineTripletStdpGainIt->second;
    }
    const auto onlineTripletStdpLtpIt =
        irConfig.doubleParams.find("online_triplet_stdp_ltp");
    if (onlineTripletStdpLtpIt != irConfig.doubleParams.end()) {
        config.onlineTripletStdpLtp = onlineTripletStdpLtpIt->second;
    }
    const auto onlineTripletStdpLtdIt =
        irConfig.doubleParams.find("online_triplet_stdp_ltd");
    if (onlineTripletStdpLtdIt != irConfig.doubleParams.end()) {
        config.onlineTripletStdpLtd = onlineTripletStdpLtdIt->second;
    }
    const auto onlineTripletStdpFastDecayIt =
        irConfig.doubleParams.find("online_triplet_stdp_fast_decay");
    if (onlineTripletStdpFastDecayIt != irConfig.doubleParams.end()) {
        config.onlineTripletStdpFastDecay = onlineTripletStdpFastDecayIt->second;
    }
    const auto onlineTripletStdpSlowDecayIt =
        irConfig.doubleParams.find("online_triplet_stdp_slow_decay");
    if (onlineTripletStdpSlowDecayIt != irConfig.doubleParams.end()) {
        config.onlineTripletStdpSlowDecay = onlineTripletStdpSlowDecayIt->second;
    }
    const auto onlineVoltagePlasticityEnabledIt =
        irConfig.intParams.find("online_voltage_plasticity_enabled");
    if (onlineVoltagePlasticityEnabledIt != irConfig.intParams.end()) {
        config.onlineVoltagePlasticityEnabled = onlineVoltagePlasticityEnabledIt->second != 0;
    }
    const auto onlineVoltagePlasticityGainIt =
        irConfig.doubleParams.find("online_voltage_plasticity_gain");
    if (onlineVoltagePlasticityGainIt != irConfig.doubleParams.end()) {
        config.onlineVoltagePlasticityGain = onlineVoltagePlasticityGainIt->second;
    }
    const auto onlineVoltagePlasticityLtpIt =
        irConfig.doubleParams.find("online_voltage_plasticity_ltp");
    if (onlineVoltagePlasticityLtpIt != irConfig.doubleParams.end()) {
        config.onlineVoltagePlasticityLtp = onlineVoltagePlasticityLtpIt->second;
    }
    const auto onlineVoltagePlasticityLtdIt =
        irConfig.doubleParams.find("online_voltage_plasticity_ltd");
    if (onlineVoltagePlasticityLtdIt != irConfig.doubleParams.end()) {
        config.onlineVoltagePlasticityLtd = onlineVoltagePlasticityLtdIt->second;
    }
    const auto onlineVoltagePlasticityDecayIt =
        irConfig.doubleParams.find("online_voltage_plasticity_decay");
    if (onlineVoltagePlasticityDecayIt != irConfig.doubleParams.end()) {
        config.onlineVoltagePlasticityDecay = onlineVoltagePlasticityDecayIt->second;
    }
    const auto onlineVoltagePlasticityThresholdIt =
        irConfig.doubleParams.find("online_voltage_plasticity_threshold");
    if (onlineVoltagePlasticityThresholdIt != irConfig.doubleParams.end()) {
        config.onlineVoltagePlasticityThreshold = onlineVoltagePlasticityThresholdIt->second;
    }
    const auto onlineReplayCapacityIt = irConfig.intParams.find("online_replay_queue_capacity");
    if (onlineReplayCapacityIt != irConfig.intParams.end()) {
        config.onlineReplayQueueCapacity = onlineReplayCapacityIt->second;
    }
    const auto onlineReplayDelayIt = irConfig.intParams.find("online_replay_delay_steps");
    if (onlineReplayDelayIt != irConfig.intParams.end()) {
        config.onlineReplayDelaySteps = onlineReplayDelayIt->second;
    }
    const auto onlineReplayPauseIt = irConfig.intParams.find("online_replay_pause_interval");
    if (onlineReplayPauseIt != irConfig.intParams.end()) {
        config.onlineReplayPauseInterval = onlineReplayPauseIt->second;
    }
    const auto onlineReplayBatchIt = irConfig.intParams.find("online_replay_batch_size");
    if (onlineReplayBatchIt != irConfig.intParams.end()) {
        config.onlineReplayBatchSize = onlineReplayBatchIt->second;
    }
    const auto onlineReplayThresholdIt =
        irConfig.doubleParams.find("online_replay_uncertainty_threshold");
    if (onlineReplayThresholdIt != irConfig.doubleParams.end()) {
        config.onlineReplayUncertaintyThreshold = onlineReplayThresholdIt->second;
    }
    const auto onlineEligibilityDecayIt =
        irConfig.doubleParams.find("online_eligibility_trace_decay");
    if (onlineEligibilityDecayIt != irConfig.doubleParams.end()) {
        config.onlineEligibilityTraceDecay = onlineEligibilityDecayIt->second;
    }
    const auto onlineNeuromodulatorEligibilityEnabledIt =
        irConfig.intParams.find("online_neuromodulator_eligibility_enabled");
    if (onlineNeuromodulatorEligibilityEnabledIt != irConfig.intParams.end()) {
        config.onlineNeuromodulatorEligibilityEnabled =
            onlineNeuromodulatorEligibilityEnabledIt->second != 0;
    }
    const auto onlineNeuromodulatorUncertaintyGainIt =
        irConfig.doubleParams.find("online_neuromodulator_uncertainty_gain");
    if (onlineNeuromodulatorUncertaintyGainIt != irConfig.doubleParams.end()) {
        config.onlineNeuromodulatorUncertaintyGain =
            std::max(0.0, onlineNeuromodulatorUncertaintyGainIt->second);
    }
    const auto onlineNeuromodulatorPositiveRewardGainIt =
        irConfig.doubleParams.find("online_neuromodulator_positive_reward_gain");
    if (onlineNeuromodulatorPositiveRewardGainIt != irConfig.doubleParams.end()) {
        config.onlineNeuromodulatorPositiveRewardGain =
            std::max(0.0, onlineNeuromodulatorPositiveRewardGainIt->second);
    }
    const auto onlineNeuromodulatorEligibilityMaxIt =
        irConfig.doubleParams.find("online_neuromodulator_eligibility_max");
    if (onlineNeuromodulatorEligibilityMaxIt != irConfig.doubleParams.end()) {
        config.onlineNeuromodulatorEligibilityMax =
            std::max(1.0, onlineNeuromodulatorEligibilityMaxIt->second);
    }
    const auto onlineReplayVariableDelayEnabledIt =
        irConfig.intParams.find("online_replay_variable_delay_enabled");
    if (onlineReplayVariableDelayEnabledIt != irConfig.intParams.end()) {
        config.onlineReplayVariableDelayEnabled =
            onlineReplayVariableDelayEnabledIt->second != 0;
    }
    const auto onlineReplaySuccessConsolidationEnabledIt =
        irConfig.intParams.find("online_replay_success_consolidation_enabled");
    if (onlineReplaySuccessConsolidationEnabledIt != irConfig.intParams.end()) {
        config.onlineReplaySuccessConsolidationEnabled =
            onlineReplaySuccessConsolidationEnabledIt->second != 0;
    }
    const auto onlineReplaySuccessConsolidationCapacityIt =
        irConfig.intParams.find("online_replay_success_consolidation_capacity");
    if (onlineReplaySuccessConsolidationCapacityIt != irConfig.intParams.end()) {
        config.onlineReplaySuccessConsolidationCapacity =
            std::max(0, onlineReplaySuccessConsolidationCapacityIt->second);
    }
    const auto onlineReplaySuccessConsolidationBatchSizeIt =
        irConfig.intParams.find("online_replay_success_consolidation_batch_size");
    if (onlineReplaySuccessConsolidationBatchSizeIt != irConfig.intParams.end()) {
        config.onlineReplaySuccessConsolidationBatchSize =
            std::max(1, onlineReplaySuccessConsolidationBatchSizeIt->second);
    }
    const auto onlineReplaySuccessConsolidationRepeatsIt =
        irConfig.intParams.find("online_replay_success_consolidation_repeats");
    if (onlineReplaySuccessConsolidationRepeatsIt != irConfig.intParams.end()) {
        config.onlineReplaySuccessConsolidationRepeats =
            std::max(1, onlineReplaySuccessConsolidationRepeatsIt->second);
    }
    const auto onlineContextUncertaintyGainIt =
        irConfig.doubleParams.find("online_context_uncertainty_gain");
    if (onlineContextUncertaintyGainIt != irConfig.doubleParams.end()) {
        config.onlineContextUncertaintyGain = onlineContextUncertaintyGainIt->second;
    }
    const auto onlineContextDisagreementGainIt =
        irConfig.doubleParams.find("online_context_disagreement_gain");
    if (onlineContextDisagreementGainIt != irConfig.doubleParams.end()) {
        config.onlineContextDisagreementGain = onlineContextDisagreementGainIt->second;
    }
    const auto hemisphereConvergentCodeEnabledIt =
        irConfig.intParams.find("hemisphere_convergent_code_enabled");
    if (hemisphereConvergentCodeEnabledIt != irConfig.intParams.end()) {
        config.hemisphereConvergentCodeEnabled =
            hemisphereConvergentCodeEnabledIt->second != 0;
    }
    const auto hemisphereConvergentSummaryBinsIt =
        irConfig.intParams.find("hemisphere_convergent_summary_bins");
    if (hemisphereConvergentSummaryBinsIt != irConfig.intParams.end()) {
        config.hemisphereConvergentSummaryBins =
            std::max(2, hemisphereConvergentSummaryBinsIt->second);
    }
    const auto hemisphereConvergentResidualGainIt =
        irConfig.doubleParams.find("hemisphere_convergent_residual_gain");
    if (hemisphereConvergentResidualGainIt != irConfig.doubleParams.end()) {
        config.hemisphereConvergentResidualGain =
            std::max(0.0, hemisphereConvergentResidualGainIt->second);
    }
    const auto hemisphereConvergentInteractionGainIt =
        irConfig.doubleParams.find("hemisphere_convergent_interaction_gain");
    if (hemisphereConvergentInteractionGainIt != irConfig.doubleParams.end()) {
        config.hemisphereConvergentInteractionGain =
            std::max(0.0, hemisphereConvergentInteractionGainIt->second);
    }
    const auto hemisphereTopographicStageEnabledIt =
        irConfig.intParams.find("hemisphere_topographic_stage_enabled");
    if (hemisphereTopographicStageEnabledIt != irConfig.intParams.end()) {
        config.hemisphereTopographicStageEnabled =
            hemisphereTopographicStageEnabledIt->second != 0;
    }
    const auto hemisphereTopographicResidualGainIt =
        irConfig.doubleParams.find("hemisphere_topographic_residual_gain");
    if (hemisphereTopographicResidualGainIt != irConfig.doubleParams.end()) {
        config.hemisphereTopographicResidualGain =
            std::max(0.0, hemisphereTopographicResidualGainIt->second);
    }
    const auto hemisphereTopographicContinuityGainIt =
        irConfig.doubleParams.find("hemisphere_topographic_continuity_gain");
    if (hemisphereTopographicContinuityGainIt != irConfig.doubleParams.end()) {
        config.hemisphereTopographicContinuityGain =
            std::max(0.0, hemisphereTopographicContinuityGainIt->second);
    }
    const auto hemisphereTopographicJunctionGainIt =
        irConfig.doubleParams.find("hemisphere_topographic_junction_gain");
    if (hemisphereTopographicJunctionGainIt != irConfig.doubleParams.end()) {
        config.hemisphereTopographicJunctionGain =
            std::max(0.0, hemisphereTopographicJunctionGainIt->second);
    }
    const auto hemisphereTopographicAuxiliaryGainIt =
        irConfig.doubleParams.find("hemisphere_topographic_auxiliary_gain");
    if (hemisphereTopographicAuxiliaryGainIt != irConfig.doubleParams.end()) {
        config.hemisphereTopographicAuxiliaryGain =
            std::max(0.0, hemisphereTopographicAuxiliaryGainIt->second);
    }
    const auto figureGroundStageEnabledIt =
        irConfig.intParams.find("figure_ground_stage_enabled");
    if (figureGroundStageEnabledIt != irConfig.intParams.end()) {
        config.figureGroundStageEnabled = figureGroundStageEnabledIt->second != 0;
    }
    const auto figureGroundMaskEnabledIt =
        irConfig.intParams.find("figure_ground_mask_enabled");
    if (figureGroundMaskEnabledIt != irConfig.intParams.end()) {
        config.figureGroundMaskEnabled = figureGroundMaskEnabledIt->second != 0;
    }
    const auto figureGroundClassifierEnabledIt =
        irConfig.intParams.find("figure_ground_classifier_enabled");
    if (figureGroundClassifierEnabledIt != irConfig.intParams.end()) {
        config.figureGroundClassifierEnabled =
            figureGroundClassifierEnabledIt->second != 0;
    }
    const auto figureGroundObjectMemoryEnabledIt =
        irConfig.intParams.find("figure_ground_object_memory_enabled");
    if (figureGroundObjectMemoryEnabledIt != irConfig.intParams.end()) {
        config.figureGroundObjectMemoryEnabled =
            figureGroundObjectMemoryEnabledIt->second != 0;
    }
    const auto recurrentSensoryStateEnabledIt =
        irConfig.intParams.find("recurrent_sensory_state_enabled");
    if (recurrentSensoryStateEnabledIt != irConfig.intParams.end()) {
        config.recurrentSensoryStateEnabled =
            recurrentSensoryStateEnabledIt->second != 0;
    }
    const auto recurrentPopulationReadoutEnabledIt =
        irConfig.intParams.find("recurrent_population_readout_enabled");
    if (recurrentPopulationReadoutEnabledIt != irConfig.intParams.end()) {
        config.recurrentPopulationReadoutEnabled =
            recurrentPopulationReadoutEnabledIt->second != 0;
    }
    const auto recurrentObjectStateReadoutEnabledIt =
        irConfig.intParams.find("recurrent_object_state_readout_enabled");
    if (recurrentObjectStateReadoutEnabledIt != irConfig.intParams.end()) {
        config.recurrentObjectStateReadoutEnabled =
            recurrentObjectStateReadoutEnabledIt->second != 0;
    }
    const auto figureGroundCyclesIt = irConfig.intParams.find("figure_ground_cycles");
    if (figureGroundCyclesIt != irConfig.intParams.end()) {
        config.figureGroundCycles = std::max(1, figureGroundCyclesIt->second);
    }
    const auto figureGroundBorderGainIt =
        irConfig.doubleParams.find("figure_ground_border_gain");
    if (figureGroundBorderGainIt != irConfig.doubleParams.end()) {
        config.figureGroundBorderGain = std::max(0.0, figureGroundBorderGainIt->second);
    }
    const auto figureGroundSurfaceGainIt =
        irConfig.doubleParams.find("figure_ground_surface_gain");
    if (figureGroundSurfaceGainIt != irConfig.doubleParams.end()) {
        config.figureGroundSurfaceGain = std::max(0.0, figureGroundSurfaceGainIt->second);
    }
    const auto figureGroundCompetitionGainIt =
        irConfig.doubleParams.find("figure_ground_competition_gain");
    if (figureGroundCompetitionGainIt != irConfig.doubleParams.end()) {
        config.figureGroundCompetitionGain =
            std::max(0.0, figureGroundCompetitionGainIt->second);
    }
    const auto figureGroundFeedbackGainIt =
        irConfig.doubleParams.find("figure_ground_feedback_gain");
    if (figureGroundFeedbackGainIt != irConfig.doubleParams.end()) {
        config.figureGroundFeedbackGain =
            std::max(0.0, figureGroundFeedbackGainIt->second);
    }
    const auto figureGroundResidualGainIt =
        irConfig.doubleParams.find("figure_ground_residual_gain");
    if (figureGroundResidualGainIt != irConfig.doubleParams.end()) {
        config.figureGroundResidualGain =
            std::max(0.0, figureGroundResidualGainIt->second);
    }
    const auto figureGroundMaskGainIt =
        irConfig.doubleParams.find("figure_ground_mask_gain");
    if (figureGroundMaskGainIt != irConfig.doubleParams.end()) {
        config.figureGroundMaskGain =
            std::max(0.0, figureGroundMaskGainIt->second);
    }
    const auto figureGroundClassifierGainIt =
        irConfig.doubleParams.find("figure_ground_classifier_gain");
    if (figureGroundClassifierGainIt != irConfig.doubleParams.end()) {
        config.figureGroundClassifierGain =
            std::max(0.0, figureGroundClassifierGainIt->second);
    }
    const auto figureGroundObjectMemoryGainIt =
        irConfig.doubleParams.find("figure_ground_object_memory_gain");
    if (figureGroundObjectMemoryGainIt != irConfig.doubleParams.end()) {
        config.figureGroundObjectMemoryGain =
            std::max(0.0, figureGroundObjectMemoryGainIt->second);
    }
    const auto figureGroundObjectMemoryKeepFractionIt =
        irConfig.doubleParams.find("figure_ground_object_memory_keep_fraction");
    if (figureGroundObjectMemoryKeepFractionIt != irConfig.doubleParams.end()) {
        config.figureGroundObjectMemoryKeepFraction = std::clamp(
            figureGroundObjectMemoryKeepFractionIt->second, 0.05, 1.0);
    }
    const auto recurrentSensoryCyclesIt =
        irConfig.intParams.find("recurrent_sensory_cycles");
    if (recurrentSensoryCyclesIt != irConfig.intParams.end()) {
        config.recurrentSensoryCycles = std::max(1, recurrentSensoryCyclesIt->second);
    }
    const auto recurrentSensoryFeedforwardGainIt =
        irConfig.doubleParams.find("recurrent_sensory_feedforward_gain");
    if (recurrentSensoryFeedforwardGainIt != irConfig.doubleParams.end()) {
        config.recurrentSensoryFeedforwardGain =
            std::max(0.0, recurrentSensoryFeedforwardGainIt->second);
    }
    const auto recurrentSensoryStateGainIt =
        irConfig.doubleParams.find("recurrent_sensory_state_gain");
    if (recurrentSensoryStateGainIt != irConfig.doubleParams.end()) {
        config.recurrentSensoryStateGain =
            std::max(0.0, recurrentSensoryStateGainIt->second);
    }
    const auto recurrentSensoryFigureGroundGainIt =
        irConfig.doubleParams.find("recurrent_sensory_figure_ground_gain");
    if (recurrentSensoryFigureGroundGainIt != irConfig.doubleParams.end()) {
        config.recurrentSensoryFigureGroundGain =
            std::max(0.0, recurrentSensoryFigureGroundGainIt->second);
    }
    const auto recurrentSensoryContinuityGainIt =
        irConfig.doubleParams.find("recurrent_sensory_continuity_gain");
    if (recurrentSensoryContinuityGainIt != irConfig.doubleParams.end()) {
        config.recurrentSensoryContinuityGain =
            std::max(0.0, recurrentSensoryContinuityGainIt->second);
    }
    const auto recurrentSensoryCallosalGainIt =
        irConfig.doubleParams.find("recurrent_sensory_callosal_gain");
    if (recurrentSensoryCallosalGainIt != irConfig.doubleParams.end()) {
        config.recurrentSensoryCallosalGain =
            std::max(0.0, recurrentSensoryCallosalGainIt->second);
    }
    const auto recurrentObjectStateUnitsPerClassIt =
        irConfig.intParams.find("recurrent_object_state_units_per_class");
    if (recurrentObjectStateUnitsPerClassIt != irConfig.intParams.end()) {
        config.recurrentObjectStateUnitsPerClass =
            std::max(1, recurrentObjectStateUnitsPerClassIt->second);
    }
    const auto recurrentObjectStateCyclesIt =
        irConfig.intParams.find("recurrent_object_state_cycles");
    if (recurrentObjectStateCyclesIt != irConfig.intParams.end()) {
        config.recurrentObjectStateCycles =
            std::max(1, recurrentObjectStateCyclesIt->second);
    }
    const auto recurrentObjectStateInputGainIt =
        irConfig.doubleParams.find("recurrent_object_state_input_gain");
    if (recurrentObjectStateInputGainIt != irConfig.doubleParams.end()) {
        config.recurrentObjectStateInputGain =
            std::max(0.0, recurrentObjectStateInputGainIt->second);
    }
    const auto recurrentObjectStateSelfGainIt =
        irConfig.doubleParams.find("recurrent_object_state_self_gain");
    if (recurrentObjectStateSelfGainIt != irConfig.doubleParams.end()) {
        config.recurrentObjectStateSelfGain =
            std::max(0.0, recurrentObjectStateSelfGainIt->second);
    }
    const auto recurrentObjectStateClassSupportGainIt =
        irConfig.doubleParams.find("recurrent_object_state_class_support_gain");
    if (recurrentObjectStateClassSupportGainIt != irConfig.doubleParams.end()) {
        config.recurrentObjectStateClassSupportGain =
            std::max(0.0, recurrentObjectStateClassSupportGainIt->second);
    }
    const auto recurrentObjectStateCompetitionGainIt =
        irConfig.doubleParams.find("recurrent_object_state_competition_gain");
    if (recurrentObjectStateCompetitionGainIt != irConfig.doubleParams.end()) {
        config.recurrentObjectStateCompetitionGain =
            std::max(0.0, recurrentObjectStateCompetitionGainIt->second);
    }
    const auto onlineConfusionClusterGainIt =
        irConfig.doubleParams.find("online_confusion_cluster_gain");
    if (onlineConfusionClusterGainIt != irConfig.doubleParams.end()) {
        config.onlineConfusionClusterGain = onlineConfusionClusterGainIt->second;
    }
    const auto onlineConfusionClusterDecayIt =
        irConfig.doubleParams.find("online_confusion_cluster_decay");
    if (onlineConfusionClusterDecayIt != irConfig.doubleParams.end()) {
        config.onlineConfusionClusterDecay = onlineConfusionClusterDecayIt->second;
    }
    const auto separabilityDiagnosticsIt =
        irConfig.intParams.find("separability_diagnostics");
    if (separabilityDiagnosticsIt != irConfig.intParams.end()) {
        config.separabilityDiagnostics = separabilityDiagnosticsIt->second != 0;
    }
    const auto separabilitySampleLimitIt =
        irConfig.intParams.find("separability_sample_limit");
    if (separabilitySampleLimitIt != irConfig.intParams.end()) {
        config.separabilitySampleLimit = std::max(0, separabilitySampleLimitIt->second);
    }
    const auto flowAuditEnabledIt = irConfig.intParams.find("flow_audit_enabled");
    if (flowAuditEnabledIt != irConfig.intParams.end()) {
        config.flowAuditEnabled = flowAuditEnabledIt->second != 0;
    }
    const auto flowAuditSampleLimitIt = irConfig.intParams.find("flow_audit_sample_limit");
    if (flowAuditSampleLimitIt != irConfig.intParams.end()) {
        config.flowAuditSampleLimit = std::max(0, flowAuditSampleLimitIt->second);
    }
    const auto flowAuditOutputPrefixIt =
        irConfig.stringParams.find("flow_audit_output_prefix");
    if (flowAuditOutputPrefixIt != irConfig.stringParams.end()) {
        config.flowAuditOutputPrefix = trim(flowAuditOutputPrefixIt->second);
    }
    const auto jepaDumpPathIt = irConfig.stringParams.find("jepa_dump_path");
    if (jepaDumpPathIt != irConfig.stringParams.end()) {
        config.jepa.dumpPath = trim(jepaDumpPathIt->second);
    }
    const auto jepaTapSurfaceIt = irConfig.stringParams.find("jepa_tap_surface");
    if (jepaTapSurfaceIt != irConfig.stringParams.end()) {
        const std::string value = toLower(trim(jepaTapSurfaceIt->second));
        if (value == "raw_stage1") {
            config.jepa.tapSurface = snnfw::jepa::TapSurface::RawStage1;
        } else if (value == "promoted_stage1") {
            config.jepa.tapSurface = snnfw::jepa::TapSurface::PromotedStage1;
        } else {
            throw std::runtime_error("Unsupported jepa_tap_surface: " + value);
        }
    }
    const auto jepaEnabledIt = irConfig.intParams.find("jepa_enabled");
    if (jepaEnabledIt != irConfig.intParams.end()) {
        config.jepa.enabled = jepaEnabledIt->second != 0;
    }
    const auto jepaTrainerEnabledIt = irConfig.intParams.find("jepa_trainer_enabled");
    if (jepaTrainerEnabledIt != irConfig.intParams.end()) {
        config.jepa.trainerEnabled = jepaTrainerEnabledIt->second != 0;
    }
    const auto jepaProbeEnabledIt = irConfig.intParams.find("jepa_probe_enabled");
    if (jepaProbeEnabledIt != irConfig.intParams.end()) {
        config.jepa.probeEnabled = jepaProbeEnabledIt->second != 0;
    }
    const auto jepaTrainerEpochsIt = irConfig.intParams.find("jepa_trainer_epochs");
    if (jepaTrainerEpochsIt != irConfig.intParams.end()) {
        config.jepa.trainerEpochs = std::max(0, jepaTrainerEpochsIt->second);
    }
    const auto jepaProjectionDimIt = irConfig.intParams.find("jepa_projection_dim");
    if (jepaProjectionDimIt != irConfig.intParams.end()) {
        config.jepa.projectionDim = std::max(1, jepaProjectionDimIt->second);
    }
    const auto jepaVisibleBranchCountIt =
        irConfig.intParams.find("jepa_visible_branch_count");
    if (jepaVisibleBranchCountIt != irConfig.intParams.end()) {
        config.jepa.visibleBranchCount = std::max(0, jepaVisibleBranchCountIt->second);
    }
    const auto jepaHiddenBranchCountIt =
        irConfig.intParams.find("jepa_hidden_branch_count");
    if (jepaHiddenBranchCountIt != irConfig.intParams.end()) {
        config.jepa.hiddenBranchCount = std::max(0, jepaHiddenBranchCountIt->second);
    }
    const auto jepaMaxSamplesIt = irConfig.intParams.find("jepa_max_samples");
    if (jepaMaxSamplesIt != irConfig.intParams.end()) {
        config.jepa.maxSamples = std::max(0, jepaMaxSamplesIt->second);
    }
    const auto jepaMaskSeedIt = irConfig.intParams.find("jepa_mask_seed");
    if (jepaMaskSeedIt != irConfig.intParams.end()) {
        config.jepa.maskSeed = static_cast<unsigned int>(std::max(0, jepaMaskSeedIt->second));
    }
    const auto jepaIncludeBranchTokensIt =
        irConfig.intParams.find("jepa_include_branch_tokens");
    if (jepaIncludeBranchTokensIt != irConfig.intParams.end()) {
        config.jepa.includeBranchTokens = jepaIncludeBranchTokensIt->second != 0;
    }
    const auto jepaIncludeHemisphereTokenIt =
        irConfig.intParams.find("jepa_include_hemisphere_token");
    if (jepaIncludeHemisphereTokenIt != irConfig.intParams.end()) {
        config.jepa.includeHemisphereToken = jepaIncludeHemisphereTokenIt->second != 0;
    }
    const auto jepaEnforceNoLeakageIt =
        irConfig.intParams.find("jepa_enforce_no_leakage");
    if (jepaEnforceNoLeakageIt != irConfig.intParams.end()) {
        config.jepa.enforceNoLeakage = jepaEnforceNoLeakageIt->second != 0;
    }
    const auto jepaMaskModeIt = irConfig.stringParams.find("jepa_mask_mode");
    if (jepaMaskModeIt != irConfig.stringParams.end()) {
        const std::string value = toLower(trim(jepaMaskModeIt->second));
        if (value == "none") {
            config.jepa.maskMode = snnfw::jepa::MaskMode::None;
        } else if (value == "branch") {
            config.jepa.maskMode = snnfw::jepa::MaskMode::Branch;
        } else {
            throw std::runtime_error("Unsupported jepa_mask_mode: " + value);
        }
    }
    const auto jepaTargetModeIt = irConfig.stringParams.find("jepa_target_mode");
    if (jepaTargetModeIt != irConfig.stringParams.end()) {
        const std::string value = toLower(trim(jepaTargetModeIt->second));
        if (value == "branch_mask") {
            config.jepa.targetMode = snnfw::jepa::TargetMode::BranchMask;
        } else if (value == "temporal_fixation") {
            config.jepa.targetMode = snnfw::jepa::TargetMode::TemporalFixation;
        } else {
            throw std::runtime_error("Unsupported jepa_target_mode: " + value);
        }
    }
    const auto jepaTrainerDumpPathIt =
        irConfig.stringParams.find("jepa_trainer_dump_path");
    if (jepaTrainerDumpPathIt != irConfig.stringParams.end()) {
        config.jepa.trainerDumpPath = trim(jepaTrainerDumpPathIt->second);
    }
    const auto jepaTrainerLearningRateIt =
        irConfig.doubleParams.find("jepa_trainer_learning_rate");
    if (jepaTrainerLearningRateIt != irConfig.doubleParams.end()) {
        config.jepa.trainerLearningRate = std::max(0.0, jepaTrainerLearningRateIt->second);
    }
    const auto jepaTrainerWeightDecayIt =
        irConfig.doubleParams.find("jepa_trainer_weight_decay");
    if (jepaTrainerWeightDecayIt != irConfig.doubleParams.end()) {
        config.jepa.trainerWeightDecay = std::max(0.0, jepaTrainerWeightDecayIt->second);
    }
    const auto jepaTargetEmaDecayIt =
        irConfig.doubleParams.find("jepa_target_ema_decay");
    if (jepaTargetEmaDecayIt != irConfig.doubleParams.end()) {
        config.jepa.targetEmaDecay =
            std::clamp(jepaTargetEmaDecayIt->second, 0.0, 0.999999);
    }
    const auto jepaVarianceFloorIt =
        irConfig.doubleParams.find("jepa_variance_floor");
    if (jepaVarianceFloorIt != irConfig.doubleParams.end()) {
        config.jepa.varianceFloor = std::max(0.0, jepaVarianceFloorIt->second);
    }
    const auto jepaVariancePenaltyIt =
        irConfig.doubleParams.find("jepa_variance_penalty");
    if (jepaVariancePenaltyIt != irConfig.doubleParams.end()) {
        config.jepa.variancePenalty = std::max(0.0, jepaVariancePenaltyIt->second);
    }
    const auto focusAdjustmentEnabledIt = irConfig.intParams.find("focus_adjustment_enabled");
    if (focusAdjustmentEnabledIt != irConfig.intParams.end()) {
        config.focusAdjustmentEnabled = focusAdjustmentEnabledIt->second != 0;
    }
    const auto saccadeFixationsIt = irConfig.intParams.find("saccade_fixations");
    if (saccadeFixationsIt != irConfig.intParams.end()) {
        config.saccadeFixations = std::max(1, saccadeFixationsIt->second);
    }
    const auto trainingAugVariantsIt =
        irConfig.intParams.find("training_augmentation_variants");
    if (trainingAugVariantsIt != irConfig.intParams.end()) {
        config.trainingAugmentationVariants = std::max(0, trainingAugVariantsIt->second);
    }
    const auto saccadeTrainingOnlyIt = irConfig.intParams.find("saccade_training_only");
    if (saccadeTrainingOnlyIt != irConfig.intParams.end()) {
        config.saccadeTrainingOnly = saccadeTrainingOnlyIt->second != 0;
    }
    const auto guidedSaccadesEnabledIt = irConfig.intParams.find("guided_saccades_enabled");
    if (guidedSaccadesEnabledIt != irConfig.intParams.end()) {
        config.guidedSaccadesEnabled = guidedSaccadesEnabledIt->second != 0;
    }
    const auto guidedSaccadeTrainingOnlyIt =
        irConfig.intParams.find("guided_saccade_training_only");
    if (guidedSaccadeTrainingOnlyIt != irConfig.intParams.end()) {
        config.guidedSaccadeTrainingOnly = guidedSaccadeTrainingOnlyIt->second != 0;
    }
    const auto guidedSaccadeFixationsIt =
        irConfig.intParams.find("guided_saccade_fixations");
    if (guidedSaccadeFixationsIt != irConfig.intParams.end()) {
        config.guidedSaccadeFixations = std::max(1, guidedSaccadeFixationsIt->second);
    }
    const auto guidedSaccadeCandidatePeaksIt =
        irConfig.intParams.find("guided_saccade_candidate_peaks");
    if (guidedSaccadeCandidatePeaksIt != irConfig.intParams.end()) {
        config.guidedSaccadeCandidatePeaks =
            std::max(1, guidedSaccadeCandidatePeaksIt->second);
    }
    const auto activeInferenceEnabledIt = irConfig.intParams.find("active_inference_enabled");
    if (activeInferenceEnabledIt != irConfig.intParams.end()) {
        config.activeInferenceEnabled = activeInferenceEnabledIt->second != 0;
    }
    const auto activeInferenceFixationsIt = irConfig.intParams.find("active_inference_fixations");
    if (activeInferenceFixationsIt != irConfig.intParams.end()) {
        config.activeInferenceFixations = std::max(1, activeInferenceFixationsIt->second);
    }
    const auto activeInferenceMinFixationsIt =
        irConfig.intParams.find("active_inference_min_fixations");
    if (activeInferenceMinFixationsIt != irConfig.intParams.end()) {
        config.activeInferenceMinFixations = std::max(1, activeInferenceMinFixationsIt->second);
    }
    const auto activeInferenceRemapIt = irConfig.intParams.find("active_inference_remap_enabled");
    if (activeInferenceRemapIt != irConfig.intParams.end()) {
        config.activeInferenceRemapEnabled = activeInferenceRemapIt->second != 0;
    }
    const auto focusAdjustmentMarginIt =
        irConfig.doubleParams.find("focus_adjustment_margin_threshold");
    if (focusAdjustmentMarginIt != irConfig.doubleParams.end()) {
        config.focusAdjustmentMarginThreshold = focusAdjustmentMarginIt->second;
    }
    const auto focusAdjustmentZoomIt = irConfig.doubleParams.find("focus_adjustment_zoom");
    if (focusAdjustmentZoomIt != irConfig.doubleParams.end()) {
        config.focusAdjustmentZoom = focusAdjustmentZoomIt->second;
    }
    const auto focusAdjustmentShiftIt = irConfig.doubleParams.find("focus_adjustment_shift_px");
    if (focusAdjustmentShiftIt != irConfig.doubleParams.end()) {
        config.focusAdjustmentShiftPx = focusAdjustmentShiftIt->second;
    }
    const auto saccadeJitterIt = irConfig.doubleParams.find("saccade_jitter_px");
    if (saccadeJitterIt != irConfig.doubleParams.end()) {
        config.saccadeJitterPx = std::max(0.0, saccadeJitterIt->second);
    }
    const auto saccadeZoomIt = irConfig.doubleParams.find("saccade_zoom");
    if (saccadeZoomIt != irConfig.doubleParams.end()) {
        config.saccadeZoom = std::max(1e-3, saccadeZoomIt->second);
    }
    const auto guidedSaccadeJitterIt =
        irConfig.doubleParams.find("guided_saccade_jitter_px");
    if (guidedSaccadeJitterIt != irConfig.doubleParams.end()) {
        config.guidedSaccadeJitterPx = std::max(0.0, guidedSaccadeJitterIt->second);
    }
    const auto guidedSaccadeZoomIt = irConfig.doubleParams.find("guided_saccade_zoom");
    if (guidedSaccadeZoomIt != irConfig.doubleParams.end()) {
        config.guidedSaccadeZoom = std::max(1e-3, guidedSaccadeZoomIt->second);
    }
    const auto guidedSaccadeIorIt =
        irConfig.doubleParams.find("guided_saccade_ior_strength");
    if (guidedSaccadeIorIt != irConfig.doubleParams.end()) {
        config.guidedSaccadeIorStrength = std::max(0.0, guidedSaccadeIorIt->second);
    }
    const auto trainingReviewFractionIt =
        irConfig.doubleParams.find("training_review_fraction");
    if (trainingReviewFractionIt != irConfig.doubleParams.end()) {
        config.trainingReviewFraction =
            std::clamp(trainingReviewFractionIt->second, 0.0, 1.0);
    }
    const auto trainingAugShiftIt =
        irConfig.doubleParams.find("training_augmentation_shift_px");
    if (trainingAugShiftIt != irConfig.doubleParams.end()) {
        config.trainingAugmentationShiftPx = std::max(0.0, trainingAugShiftIt->second);
    }
    const auto trainingAugRotationIt =
        irConfig.doubleParams.find("training_augmentation_rotation_deg");
    if (trainingAugRotationIt != irConfig.doubleParams.end()) {
        config.trainingAugmentationRotationDeg = std::max(0.0, trainingAugRotationIt->second);
    }
    const auto trainingAugNoiseIt =
        irConfig.doubleParams.find("training_augmentation_noise_std");
    if (trainingAugNoiseIt != irConfig.doubleParams.end()) {
        config.trainingAugmentationNoiseStd = std::max(0.0, trainingAugNoiseIt->second);
    }
    const auto activeInferenceShiftIt = irConfig.doubleParams.find("active_inference_shift_px");
    if (activeInferenceShiftIt != irConfig.doubleParams.end()) {
        config.activeInferenceShiftPx = std::max(0.0, activeInferenceShiftIt->second);
    }
    const auto activeInferenceZoomIt = irConfig.doubleParams.find("active_inference_zoom");
    if (activeInferenceZoomIt != irConfig.doubleParams.end()) {
        config.activeInferenceZoom = std::max(1e-3, activeInferenceZoomIt->second);
    }
    const auto activeInferenceThresholdIt =
        irConfig.doubleParams.find("active_inference_uncertainty_threshold");
    if (activeInferenceThresholdIt != irConfig.doubleParams.end()) {
        config.activeInferenceUncertaintyThreshold =
            std::clamp(activeInferenceThresholdIt->second, 0.0, 1.0);
    }
    const auto activeInferenceIorIt = irConfig.doubleParams.find("active_inference_ior_strength");
    if (activeInferenceIorIt != irConfig.doubleParams.end()) {
        config.activeInferenceIorStrength = std::max(0.0, activeInferenceIorIt->second);
    }
    const auto inputDomainIt = irConfig.stringParams.find("input_domain");
    if (inputDomainIt != irConfig.stringParams.end()) {
        config.inputDomain = inputDomainIt->second;
    }
    const auto inputVariantIt = irConfig.stringParams.find("input_variant");
    if (inputVariantIt != irConfig.stringParams.end()) {
        config.inputVariant = inputVariantIt->second;
    }
    const auto inputApplyTransformIt = irConfig.intParams.find("input_apply_transform");
    if (inputApplyTransformIt != irConfig.intParams.end()) {
        config.inputApplyTransform = inputApplyTransformIt->second != 0;
    }
}

BaseAdapter::Config makeRetinaAdapterConfig(const AdapterConfigIR& adapter,
                                            const Config& defaults,
                                            size_t index) {
    BaseAdapter::Config retinaConfig = makeDefaultRetinaConfig(defaults, defaults.gridSize, index);
    retinaConfig.name = adapter.name.empty()
        ? ("emnist_retina_" + std::to_string(index))
        : adapter.name;
    retinaConfig.type = "retina";
    retinaConfig.temporalWindow = adapter.temporalWindowMs > 0.0
        ? adapter.temporalWindowMs
        : defaults.temporalWindowMs;

    for (const auto& [key, value] : adapter.doubleParams) {
        retinaConfig.doubleParams[key] = value;
    }
    for (const auto& [key, value] : adapter.intParams) {
        retinaConfig.intParams[key] = value;
    }
    for (const auto& [key, value] : adapter.stringParams) {
        retinaConfig.stringParams[key] = value;
    }

    if (retinaConfig.intParams.find("grid_size") == retinaConfig.intParams.end()) {
        retinaConfig.intParams["grid_size"] = defaults.gridSize;
    }
    if (retinaConfig.intParams.find("num_orientations") == retinaConfig.intParams.end()) {
        retinaConfig.intParams["num_orientations"] = defaults.numOrientations;
    }
    if (retinaConfig.doubleParams.find("edge_threshold") == retinaConfig.doubleParams.end()) {
        retinaConfig.doubleParams["edge_threshold"] = defaults.edgeThreshold;
    }
    if (retinaConfig.doubleParams.find("neuron_window_size") == retinaConfig.doubleParams.end()) {
        retinaConfig.doubleParams["neuron_window_size"] = retinaConfig.temporalWindow;
    }
    if (retinaConfig.doubleParams.find("neuron_threshold") == retinaConfig.doubleParams.end()) {
        retinaConfig.doubleParams["neuron_threshold"] = 0.7;
    }
    if (retinaConfig.intParams.find("neuron_max_patterns") == retinaConfig.intParams.end()) {
        retinaConfig.intParams["neuron_max_patterns"] = 100;
    }
    if (retinaConfig.stringParams.find("edge_operator") == retinaConfig.stringParams.end()) {
        retinaConfig.stringParams["edge_operator"] = defaults.edgeOperator;
    }
    if (retinaConfig.stringParams.find("encoding_strategy") == retinaConfig.stringParams.end()) {
        retinaConfig.stringParams["encoding_strategy"] = defaults.encodingStrategy;
    }
    if (retinaConfig.stringParams.find("activation_mode") == retinaConfig.stringParams.end()) {
        retinaConfig.stringParams["activation_mode"] = defaults.activationMode;
    }

    return retinaConfig;
}

NetworkIR parseDeclarativeConfig(const std::string& configPath) {
    NativeJSONParser nativeParser;
    if (nativeParser.canParse(configPath)) {
        return nativeParser.parse(configPath);
    }

    SONATAParser sonataParser;
    if (sonataParser.canParse(configPath)) {
        return sonataParser.parse(configPath);
    }

    throw std::runtime_error("Unsupported declarative config format: " + configPath);
}

void loadDeclarativeConfig(Config& config, const std::string& configPath) {
    const NetworkIR ir = parseDeclarativeConfig(configPath);

    if (!ir.classification.type.empty()) {
        applyClassificationConfig(ir.classification, config);
    }

    const auto layerIndex = buildRetinaLayerIndex(ir);
    const bool wantsHierarchicalLayout =
        !config.fusionPath.empty() ||
        std::any_of(ir.adapters.begin(), ir.adapters.end(), [](const AdapterConfigIR& adapter) {
            auto it = adapter.stringParams.find("attach_path");
            return it != adapter.stringParams.end() && !trim(it->second).empty();
        });

    if (wantsHierarchicalLayout) {
        if (layerIndex.empty()) {
            throw std::runtime_error(
                "Hierarchical Retina declarative layout requested, but the config does not "
                "define any brain layer paths");
        }
        config.hierarchicalRetinaLayout = true;
        config.declaredHemisphereOrder.clear();
        for (const auto& hemisphere : ir.brain.hemispheres) {
            config.declaredHemisphereOrder.push_back(hemisphere.name);
        }
        if (!config.fusionPath.empty() &&
            config.fusionPath != "OutputLayer" &&
            layerIndex.find(config.fusionPath) == layerIndex.end()) {
            throw std::runtime_error(
                "fusion_path '" + config.fusionPath + "' does not resolve to a declared brain layer");
        }
    }

    std::vector<BaseAdapter::Config> retinaConfigs;
    retinaConfigs.reserve(ir.adapters.size());
    for (const auto& adapter : ir.adapters) {
        if (adapter.type != "retina") {
            continue;
        }
        auto retinaConfig = makeRetinaAdapterConfig(adapter, config, retinaConfigs.size());
        if (wantsHierarchicalLayout) {
            const std::string attachPath =
                normalizeHierarchyPath(retinaConfig.getStringParam("attach_path", ""));
            if (attachPath.empty()) {
                throw std::runtime_error(
                    "Retina adapter '" + retinaConfig.name +
                    "' is missing string_params.attach_path while hierarchical Retina layout "
                    "is enabled");
            }
            const auto bindingIt = layerIndex.find(attachPath);
            if (bindingIt == layerIndex.end()) {
                throw std::runtime_error(
                    "Retina adapter '" + retinaConfig.name +
                    "' attach_path '" + attachPath + "' does not resolve to a declared brain layer");
            }
            retinaConfig.stringParams["attach_path"] = attachPath;
            retinaConfig.stringParams["hemisphere"] = bindingIt->second.hemisphere;
            retinaConfig.stringParams["layer_name"] = bindingIt->second.layer;
        }
        retinaConfigs.push_back(std::move(retinaConfig));
    }

    if (!retinaConfigs.empty()) {
        config.retinaConfigs = std::move(retinaConfigs);
        config.gridSizes.clear();
        for (const auto& retinaConfig : config.retinaConfigs) {
            config.gridSizes.push_back(retinaConfig.getIntParam("grid_size", config.gridSize));
        }
        config.gridSize = config.gridSizes.front();
        config.numOrientations =
            config.retinaConfigs.front().getIntParam("num_orientations", config.numOrientations);
        config.edgeThreshold =
            config.retinaConfigs.front().getDoubleParam("edge_threshold", config.edgeThreshold);
        config.temporalWindowMs = config.retinaConfigs.front().temporalWindow;
        config.edgeOperator =
            config.retinaConfigs.front().getStringParam("edge_operator", config.edgeOperator);
        config.encodingStrategy =
            config.retinaConfigs.front().getStringParam("encoding_strategy", config.encodingStrategy);
        config.activationMode =
            config.retinaConfigs.front().getStringParam("activation_mode", config.activationMode);
    }
}

std::vector<BaseAdapter::Config> buildRetinaConfigs(const Config& config) {
    if (!config.retinaConfigs.empty()) {
        return config.retinaConfigs;
    }

    std::vector<BaseAdapter::Config> retinaConfigs;
    retinaConfigs.reserve(config.gridSizes.size());
    for (size_t i = 0; i < config.gridSizes.size(); ++i) {
        retinaConfigs.push_back(makeDefaultRetinaConfig(config, config.gridSizes[i], i));
    }
    return retinaConfigs;
}

std::vector<double> extractPattern(std::vector<std::unique_ptr<RetinaAdapter>>& retinas,
                                   const VisualStimulus& image,
                                   const Config& config,
                                   bool trainingPhase,
                                   bool learnPatterns,
                                   size_t sampleSeed = 0U,
                                   FlowAuditSampleCapture* flowAuditCapture = nullptr);

std::vector<std::pair<std::string, std::vector<BaseAdapter::Config>>>
groupRetinaConfigsByHemisphere(const std::vector<BaseAdapter::Config>& retinaConfigs,
                              const Config& config) {
    std::vector<std::pair<std::string, std::vector<BaseAdapter::Config>>> groups;
    if (config.hierarchicalRetinaLayout) {
        for (const auto& hemisphereName : config.declaredHemisphereOrder) {
            groups.push_back({hemisphereName, {}});
        }
    }
    for (const auto& retinaConfig : retinaConfigs) {
        const std::string hemisphere =
            retinaConfig.getStringParam("hemisphere", "default");
        auto it = std::find_if(groups.begin(), groups.end(),
                               [&](const auto& group) { return group.first == hemisphere; });
        if (it == groups.end()) {
            groups.push_back({hemisphere, {retinaConfig}});
        } else {
            it->second.push_back(retinaConfig);
        }
    }
    if (config.hierarchicalRetinaLayout) {
        for (auto& group : groups) {
            std::sort(group.second.begin(), group.second.end(),
                      [](const BaseAdapter::Config& lhs, const BaseAdapter::Config& rhs) {
                          return lhs.getStringParam("attach_path", lhs.name) <
                                 rhs.getStringParam("attach_path", rhs.name);
                      });
        }
        groups.erase(std::remove_if(groups.begin(), groups.end(),
                                    [](const auto& group) { return group.second.empty(); }),
                     groups.end());
    }
    return groups;
}

std::unique_ptr<ClassificationStrategy> makeClassifierStrategy(const std::string& type,
                                                               int k,
                                                               double exponent,
                                                               const Config& config) {
    ClassificationStrategy::Config clsConfig;
    clsConfig.name = type;
    clsConfig.k = k;
    clsConfig.numClasses = config.numClasses;
    clsConfig.distanceExponent = exponent;
    clsConfig.stringParams["group_definitions"] = config.hierarchicalGroups;
    clsConfig.stringParams["coarse_strategy"] = config.hierarchicalCoarseStrategy;
    clsConfig.stringParams["fine_strategy"] = config.hierarchicalFineStrategy;
    if (config.hierarchicalCoarseK > 0) {
        clsConfig.intParams["coarse_k"] = config.hierarchicalCoarseK;
    }
    if (config.hierarchicalFineK > 0) {
        clsConfig.intParams["fine_k"] = config.hierarchicalFineK;
    }
    return snnfw::classification::ClassificationStrategyFactory::create(type, clsConfig);
}

std::vector<std::unique_ptr<RetinaAdapter>>
createRetinaAdapters(const std::vector<BaseAdapter::Config>& retinaConfigs) {
    std::vector<std::unique_ptr<RetinaAdapter>> retinas;
    retinas.reserve(retinaConfigs.size());
    for (const auto& retinaConfig : retinaConfigs) {
        auto retina = std::make_unique<RetinaAdapter>(retinaConfig);
        if (!retina->initialize()) {
            throw std::runtime_error("Failed to initialize RetinaAdapter '" + retinaConfig.name + "'");
        }
        retinas.push_back(std::move(retina));
    }
    return retinas;
}

void setRetinaHomeostaticLearning(std::vector<std::unique_ptr<RetinaAdapter>>& retinas,
                                  bool enabled) {
    for (auto& retina : retinas) {
        retina->setHomeostaticLearningEnabled(enabled);
    }
}

struct TrainingSplit {
    std::vector<size_t> stage1Indices;
    std::vector<size_t> fusionIndices;
};

struct EvaluationResult {
    IntMatrix initialConfusion;
    IntMatrix confusion;
    IntMatrix leftHemisphereConfusion;
    IntMatrix rightHemisphereConfusion;
    IntMatrix correctedInitialPairs;
    IntMatrix unresolvedInitialPairs;
    int tested = 0;
    int correct = 0;
    int initialCorrect = 0;
    int hemisphereComparable = 0;
    int hemisphereAgreements = 0;
    int hemisphereAgreementCorrect = 0;
    int hemisphereDisagreements = 0;
    int leftOnlyCorrectOnDisagreement = 0;
    int rightOnlyCorrectOnDisagreement = 0;
    int bothWrongOnDisagreement = 0;
    int focusAdjustmentTriggers = 0;
    int focusAdjustmentOverrides = 0;
    int focusAdjustmentCorrections = 0;
    int focusAdjustmentRegressions = 0;
    int focusOriginalSelections = 0;
    int focusLeftSelections = 0;
    int focusCenterSelections = 0;
    int focusRightSelections = 0;
    int activeInferenceSamples = 0;
    int activeInferenceExtraFixationSamples = 0;
    int activeInferenceEarlyStops = 0;
    double activeInferenceFixationSum = 0.0;
    int fusionRescuesOneHemisphereRight = 0;
    int fusionRescuesBothHemispheresWrong = 0;
    int fusionMissesWhenLeftWasRight = 0;
    int fusionMissesWhenRightWasRight = 0;
    int initialWrongBothHemispheresWrong = 0;
    int correctedFromOneHemisphereRight = 0;
    int correctedFromBothHemispheresWrong = 0;
    int correctionEvents = 0;
    int correctionSuccesses = 0;
    int correctionReplays = 0;
    int replayDelaySamples = 0;
    double replayDelaySteps = 0.0;
    double replayEligibilityScaleSum = 0.0;
    double confusionClusterScoreSum = 0.0;
    int confusionClusterScoreSamples = 0;
    DoubleMatrix confusionClusterStrengths;
    struct SeparabilityStats {
        std::string name;
        int samples = 0;
        int centroidCorrect = 0;
        int nonzeroSamples = 0;
        int centroidCorrectNonzero = 0;
        int zeroVectorSamples = 0;
        double ownScoreSum = 0.0;
        double otherScoreSum = 0.0;
        double marginSum = 0.0;
        double rawL1Sum = 0.0;
        double rawL2Sum = 0.0;
        double rawActiveFractionSum = 0.0;
        IntMatrix centroidConfusion;
        int hemisphereWrongSamples = 0;
        int branchSupportsHemisphereWrong = 0;
        int branchSupportsTruth = 0;
        std::vector<int> hemisphereWrongTargetCounts;
    };
    std::vector<SeparabilityStats> separabilityStats;
    double seconds = 0.0;
};

struct PatternSlice {
    std::string name;
    size_t offset = 0;
    size_t size = 0;
    std::vector<size_t> indices;
};

struct SeparabilityDiagnosticsRuntime {
    bool enabled = false;
    int sampleLimit = 0;
    int recordedSamples = 0;
    int numClasses = 0;
    std::vector<std::vector<PatternSlice>> hemisphereSlices;
    std::vector<std::vector<std::vector<double>>> hemisphereCentroids;
    std::vector<std::vector<std::vector<std::vector<double>>>> branchCentroids;
    std::vector<std::vector<double>> fusionCentroids;
};

struct FlowAuditPartCapture {
    std::string name;
    std::vector<double> preNormPattern;
    std::vector<double> postNormPattern;
    FlowAuditVectorStats preNorm;
    FlowAuditVectorStats postNorm;
    double preOrientationL2 = 0.0;
    double preAuxiliaryL2 = 0.0;
    double postOrientationL2 = 0.0;
    double postAuxiliaryL2 = 0.0;
    int fixationCount = 1;
    double fixationVariance = 0.0;
    double fixationConsistency = 1.0;
};

struct FlowAuditSampleCapture {
    std::vector<FlowAuditPartCapture> parts;
    std::vector<std::vector<double>> fixationPatterns;
    double fixationConsistency = 1.0;
};

struct FlowAuditBranchReference {
    std::string name;
    std::vector<std::vector<double>> preCentroids;
    std::vector<std::vector<double>> postCentroids;
    double meanTrainingFixationVariance = 0.0;
};

struct FlowAuditCsvRow {
    size_t sampleOrdinal = 0;
    size_t imageIndex = 0;
    int truth = -1;
    std::string stage;
    std::string hemisphere;
    std::string component;
    int predicted = -1;
    int initialPredicted = -1;
    int finalPredicted = -1;
    int top1Label = -1;
    int top2Label = -1;
    int bestNeighborLabel = -1;
    int centroidPredicted = -1;
    double top1Score = 0.0;
    double top2Score = 0.0;
    double confidenceMargin = 0.0;
    double bestNeighborSimilarity = 0.0;
    double topkPurity = 0.0;
    double preNormMeanAbs = 0.0;
    double preNormL2 = 0.0;
    double preNormActiveFraction = 0.0;
    double preOrientationL2 = 0.0;
    double preAuxiliaryL2 = 0.0;
    double preOwnScore = 0.0;
    double preOtherScore = 0.0;
    double preMargin = 0.0;
    double postNormMeanAbs = 0.0;
    double postNormL2 = 0.0;
    double postNormActiveFraction = 0.0;
    double postOrientationL2 = 0.0;
    double postAuxiliaryL2 = 0.0;
    double postOwnScore = 0.0;
    double postOtherScore = 0.0;
    double postMargin = 0.0;
    int fixationCount = 1;
    double fixationVariance = 0.0;
    double fixationConsistency = 1.0;
    double fgOwnedBorderMass = 0.0;
    double fgSurfaceMass = 0.0;
    double fgJunctionMass = 0.0;
    double fgBorderSurfaceRatio = 0.0;
    double fgLeftRightBalance = 0.0;
    int fgComponentCount = 0;
    double fgFixationConsistency = 1.0;
};

struct FlowAuditBranchAggregate {
    int samples = 0;
    int zeroPreSamples = 0;
    int zeroPostSamples = 0;
    double preMarginSum = 0.0;
    double postMarginSum = 0.0;
    double preActiveFractionSum = 0.0;
    double postActiveFractionSum = 0.0;
    double preNormL2Sum = 0.0;
    double postNormL2Sum = 0.0;
    double preOrientationL2Sum = 0.0;
    double preAuxiliaryL2Sum = 0.0;
    double postOrientationL2Sum = 0.0;
    double postAuxiliaryL2Sum = 0.0;
    double fixationVarianceSum = 0.0;
    double fixationConsistencySum = 0.0;
    double meanTrainingFixationVariance = 0.0;
};

struct FlowAuditHemisphereAggregate {
    int samples = 0;
    double confidenceMarginSum = 0.0;
    double centroidMarginSum = 0.0;
    double topkPuritySum = 0.0;
    double bestNeighborSimilaritySum = 0.0;
    double fixationConsistencySum = 0.0;
};

struct FlowAuditFigureGroundAggregate {
    int samples = 0;
    double ownedBorderMassSum = 0.0;
    double surfaceMassSum = 0.0;
    double junctionMassSum = 0.0;
    double borderSurfaceRatioSum = 0.0;
    double leftRightBalanceSum = 0.0;
    double componentCountSum = 0.0;
    double fixationConsistencySum = 0.0;
    double fgFixationConsistencySum = 0.0;
};

struct FlowAuditFusionAggregate {
    int samples = 0;
    int interactionCentroidCorrect = 0;
    int confidenceConcatCentroidCorrect = 0;
    int hemisphereConcatCentroidCorrect = 0;
};

struct FlowAuditReplayAggregate {
    int replaySuccessSamples = 0;
    int replayFailureSamples = 0;
    int positiveClassWeightUpdates = 0;
    int positivePredictionWeightUpdates = 0;
    int positiveCentroidUpdates = 0;
    int positiveExemplarInsertions = 0;
    int positiveFusionExemplarInsertions = 0;
    int negativeClassWeightUpdates = 0;
    int negativePredictionWeightUpdates = 0;
};

struct FlowAuditRuntime {
    bool enabled = false;
    int sampleLimit = 0;
    int numClasses = 0;
    int recordedSamples = 0;
    size_t droppedRows = 0;
    std::string outputPrefix;
    std::vector<std::vector<FlowAuditBranchReference>> hemisphereBranches;
    std::vector<std::vector<std::vector<double>>> hemisphereCentroids;
    std::vector<std::vector<double>> fusionInteractionCentroids;
    std::vector<std::vector<double>> fusionConfidenceConcatCentroids;
    std::vector<std::vector<double>> fusionHemisphereConcatCentroids;
    std::vector<FlowAuditCsvRow> rows;
    std::unordered_map<std::string, FlowAuditBranchAggregate> branchAggregates;
    std::unordered_map<std::string, FlowAuditHemisphereAggregate> hemisphereAggregates;
    std::unordered_map<std::string, FlowAuditFigureGroundAggregate> figureGroundAggregates;
    FlowAuditFusionAggregate fusionAggregate;
    FlowAuditReplayAggregate replayAggregate;
};

bool flowAuditShouldStoreRows(FlowAuditRuntime& runtime);
void recordFlowAuditFusionAlternatives(FlowAuditRuntime& runtime,
                                       const BilateralDecisionTrace& decision,
                                       int truth);
void recordFlowAuditBranchCapture(FlowAuditRuntime& runtime,
                                  size_t sampleOrdinal,
                                  size_t imageIndex,
                                  int truth,
                                  int initialPredicted,
                                  int finalPredicted,
                                  const std::string& hemisphereName,
                                  const HemisphereDecisionTrace& hemisphereTrace,
                                  const FlowAuditSampleCapture& capture,
                                  size_t hemisphereIndex,
                                  bool storeRows);
void recordFlowAuditHemisphereCapture(FlowAuditRuntime& runtime,
                                      size_t sampleOrdinal,
                                      size_t imageIndex,
                                      int truth,
                                      int initialPredicted,
                                      int finalPredicted,
                                      const std::string& hemisphereName,
                                      const HemisphereDecisionTrace& trace,
                                      const FlowAuditSampleCapture* sampleCapture,
                                      const HemisphereRuntime& hemisphere,
                                      size_t hemisphereIndex,
                                      const Config& config,
                                      bool storeRows);
void recordFlowAuditFusionRow(FlowAuditRuntime& runtime,
                              size_t sampleOrdinal,
                              size_t imageIndex,
                              int truth,
                              int initialPredicted,
                              int finalPredicted,
                              const BilateralDecisionTrace& decision,
                              bool storeRows);
void recordFlowAuditReplayMechanisms(FlowAuditRuntime& runtime,
                                     const BilateralDecisionTrace& replayDecision,
                                     bool replaySucceeded);
void writeFlowAuditArtifacts(const FlowAuditRuntime& runtime, const std::string& label);

struct TrainingSplit;

SeparabilityDiagnosticsRuntime buildSeparabilityDiagnosticsRuntime(
    std::vector<HemisphereRuntime>& hemispheres,
    const VisualDomainAdapter& loader,
    const TrainingSplit& split,
    const Config& config,
    const FusionRuntime& fusionRuntime);
void initializeSeparabilityStats(EvaluationResult& result,
                                 const SeparabilityDiagnosticsRuntime& runtime,
                                 const std::vector<HemisphereRuntime>& hemispheres);
void recordSeparabilityDiagnostics(EvaluationResult& result,
                                   SeparabilityDiagnosticsRuntime& runtime,
                                   const std::vector<HemisphereRuntime>& hemispheres,
                                   const BilateralDecisionTrace& decision,
                                   int truth);

EvaluationResult makeEvaluationResult(const Config& config) {
    EvaluationResult result;
    result.initialConfusion = makeClassMatrix<int>(config.numClasses, 0);
    result.confusion = makeClassMatrix<int>(config.numClasses, 0);
    result.leftHemisphereConfusion = makeClassMatrix<int>(config.numClasses, 0);
    result.rightHemisphereConfusion = makeClassMatrix<int>(config.numClasses, 0);
    result.correctedInitialPairs = makeClassMatrix<int>(config.numClasses, 0);
    result.unresolvedInitialPairs = makeClassMatrix<int>(config.numClasses, 0);
    result.confusionClusterStrengths = makeClassMatrix<double>(config.numClasses, 0.0);
    return result;
}

TrainingSplit splitTrainingIndices(const IndexBuckets& indicesByLabel,
                                   int examplesPerClass,
                                   int fusionHoldoutPerClass,
                                   unsigned int seed,
                                   bool retainFusionInStage1 = false) {
    TrainingSplit split;
    std::mt19937 rng(seed);
    for (size_t label = 0; label < indicesByLabel.size(); ++label) {
        auto shuffled = indicesByLabel[label];
        std::shuffle(shuffled.begin(), shuffled.end(), rng);
        const int keep = examplesPerClass > 0
            ? std::min<int>(examplesPerClass, static_cast<int>(shuffled.size()))
            : static_cast<int>(shuffled.size());
        if (keep <= 0) {
            continue;
        }

        int holdout = fusionHoldoutPerClass;
        if (holdout <= 0) {
            holdout = std::min(100, std::max(20, keep / 10));
        }
        holdout = std::min(holdout, std::max(1, keep / 4));
        if (holdout >= keep) {
            holdout = std::max(0, keep - 1);
        }

        const int stage1Count = retainFusionInStage1 ? keep : (keep - holdout);
        const int fusionStart = keep - holdout;
        split.stage1Indices.insert(split.stage1Indices.end(),
                                   shuffled.begin(),
                                   shuffled.begin() + stage1Count);
        split.fusionIndices.insert(split.fusionIndices.end(),
                                   shuffled.begin() + fusionStart,
                                   shuffled.begin() + keep);
    }

    std::shuffle(split.stage1Indices.begin(), split.stage1Indices.end(), rng);
    std::shuffle(split.fusionIndices.begin(), split.fusionIndices.end(), rng);
    return split;
}

IndexBuckets collectSelectedIndicesByLabel(const std::vector<size_t>& selectedIndices,
                                          const VisualDomainAdapter& loader,
                                          int numClasses) {
    IndexBuckets indicesByLabel(static_cast<size_t>(std::max(0, numClasses)));
    for (size_t index : selectedIndices) {
        if (index >= loader.size()) {
            continue;
        }
        const int label = loader.getStimulus(index).label;
        if (label >= 0 && label < numClasses) {
            indicesByLabel[static_cast<size_t>(label)].push_back(index);
        }
    }
    return indicesByLabel;
}

std::vector<std::vector<int>> resolvedCurriculumStages(const Config& config) {
    if (config.trainingCurriculumGroups.empty()) {
        return {};
    }

    std::vector<std::vector<int>> stages = config.trainingCurriculumGroups;
    const auto coveredLabels = flattenLabelGroups(stages);
    std::vector<int> remainder;
    for (int label = 0; label < config.numClasses; ++label) {
        if (std::find(coveredLabels.begin(), coveredLabels.end(), label) == coveredLabels.end()) {
            remainder.push_back(label);
        }
    }
    if (!remainder.empty()) {
        stages.push_back(std::move(remainder));
    }
    return stages;
}

std::vector<size_t> interleaveIndicesForLabels(const IndexBuckets& indicesByLabel,
                                               const std::vector<int>& labels) {
    std::vector<size_t> scheduled;
    std::vector<size_t> offsets(indicesByLabel.size(), 0);

    bool addedAny = true;
    while (addedAny) {
        addedAny = false;
        for (int label : labels) {
            if (label < 0 || static_cast<size_t>(label) >= indicesByLabel.size()) {
                continue;
            }
            const auto& bucket = indicesByLabel[static_cast<size_t>(label)];
            if (offsets[static_cast<size_t>(label)] >= bucket.size()) {
                continue;
            }
            scheduled.push_back(bucket[offsets[static_cast<size_t>(label)]++]);
            addedAny = true;
        }
    }

    return scheduled;
}

TrainingSchedulePlan buildTrainingSchedule(const std::vector<size_t>& selectedIndices,
                                           const VisualDomainAdapter& loader,
                                           const Config& config,
                                           unsigned int seed) {
    TrainingSchedulePlan plan;
    plan.primaryIndices = selectedIndices;

    const bool useCurriculum = !config.trainingCurriculumGroups.empty();
    const bool useReview = config.trainingReviewFraction > 0.0;
    if (!useCurriculum && !useReview) {
        return plan;
    }

    auto indicesByLabel = collectSelectedIndicesByLabel(selectedIndices, loader, config.numClasses);
    std::mt19937 rng(seed);
    for (auto& bucket : indicesByLabel) {
        std::shuffle(bucket.begin(), bucket.end(), rng);
    }

    const auto curriculumStages = resolvedCurriculumStages(config);
    if (useCurriculum && !curriculumStages.empty()) {
        plan.primaryIndices.clear();
        for (const auto& stage : curriculumStages) {
            auto stageIndices = interleaveIndicesForLabels(indicesByLabel, stage);
            plan.primaryIndices.insert(
                plan.primaryIndices.end(), stageIndices.begin(), stageIndices.end());
        }
    }

    if (!useReview) {
        return plan;
    }

    IndexBuckets reviewByLabel(static_cast<size_t>(std::max(0, config.numClasses)));
    std::mt19937 reviewRng(seed ^ 0x9e3779b9U);
    for (size_t label = 0; label < indicesByLabel.size(); ++label) {
        auto reviewCandidates = indicesByLabel[label];
        std::shuffle(reviewCandidates.begin(), reviewCandidates.end(), reviewRng);
        const int keep = static_cast<int>(std::floor(
            config.trainingReviewFraction * static_cast<double>(reviewCandidates.size()) + 1e-6));
        if (keep <= 0) {
            continue;
        }
        reviewByLabel[label].insert(reviewByLabel[label].end(),
                                    reviewCandidates.begin(),
                                    reviewCandidates.begin() +
                                        std::min<int>(keep, static_cast<int>(reviewCandidates.size())));
    }

    if (useCurriculum && !curriculumStages.empty()) {
        for (const auto& stage : curriculumStages) {
            auto stageIndices = interleaveIndicesForLabels(reviewByLabel, stage);
            plan.reviewIndices.insert(
                plan.reviewIndices.end(), stageIndices.begin(), stageIndices.end());
        }
    } else {
        std::vector<int> allLabels(config.numClasses);
        std::iota(allLabels.begin(), allLabels.end(), 0);
        plan.reviewIndices = interleaveIndicesForLabels(reviewByLabel, allLabels);
    }

    return plan;
}

std::vector<ClassificationStrategy::LabeledPattern> buildSupportPatterns(
    const HemisphereRuntime& hemisphere,
    const std::vector<size_t>& excludedSourceIndices) {
    if (excludedSourceIndices.empty()) {
        return hemisphere.trainingPatterns;
    }

    const std::unordered_set<size_t> excluded(excludedSourceIndices.begin(),
                                              excludedSourceIndices.end());
    std::vector<ClassificationStrategy::LabeledPattern> supportPatterns;
    supportPatterns.reserve(hemisphere.trainingPatterns.size());
    for (size_t i = 0; i < hemisphere.trainingPatterns.size(); ++i) {
        if (i < hemisphere.trainingSourceIndices.size() &&
            excluded.find(hemisphere.trainingSourceIndices[i]) != excluded.end()) {
            continue;
        }
        supportPatterns.push_back(hemisphere.trainingPatterns[i]);
    }
    return supportPatterns;
}

bool useOnlineCorrection(const Config& config) {
    return config.onlineCorrectionRepeats > 0;
}

bool useOnlineRewardStdp(const Config& config) {
    return config.onlineRewardStdpEnabled && config.onlineRewardStdpGain > 1e-6 &&
           (config.onlineRewardStdpLtp > 1e-6 || config.onlineRewardStdpLtd > 1e-6);
}

bool useOnlineTripletStdp(const Config& config) {
    return config.onlineTripletStdpEnabled && config.onlineTripletStdpGain > 1e-6 &&
           (config.onlineTripletStdpLtp > 1e-6 || config.onlineTripletStdpLtd > 1e-6);
}

bool useOnlineVoltagePlasticity(const Config& config) {
    return config.onlineVoltagePlasticityEnabled && config.onlineVoltagePlasticityGain > 1e-6 &&
           (config.onlineVoltagePlasticityLtp > 1e-6 ||
            config.onlineVoltagePlasticityLtd > 1e-6);
}

double onlineRewardStdpGain(const Config& config) {
    return std::clamp(config.onlineRewardStdpGain, 0.0, 1.0);
}

double onlineRewardStdpLtp(const Config& config) {
    return std::clamp(config.onlineRewardStdpLtp, 0.0, 1.0);
}

double onlineRewardStdpLtd(const Config& config) {
    return std::clamp(config.onlineRewardStdpLtd, 0.0, 1.0);
}

double onlineTripletStdpGain(const Config& config) {
    return std::clamp(config.onlineTripletStdpGain, 0.0, 1.0);
}

double onlineTripletStdpLtp(const Config& config) {
    return std::clamp(config.onlineTripletStdpLtp, 0.0, 1.0);
}

double onlineTripletStdpLtd(const Config& config) {
    return std::clamp(config.onlineTripletStdpLtd, 0.0, 1.0);
}

double onlineTripletStdpFastDecay(const Config& config) {
    return std::clamp(config.onlineTripletStdpFastDecay, 0.10, 0.999);
}

double onlineTripletStdpSlowDecay(const Config& config) {
    return std::clamp(config.onlineTripletStdpSlowDecay, 0.10, 0.999);
}

double onlineVoltagePlasticityGain(const Config& config) {
    return std::clamp(config.onlineVoltagePlasticityGain, 0.0, 1.0);
}

double onlineVoltagePlasticityLtp(const Config& config) {
    return std::clamp(config.onlineVoltagePlasticityLtp, 0.0, 1.0);
}

double onlineVoltagePlasticityLtd(const Config& config) {
    return std::clamp(config.onlineVoltagePlasticityLtd, 0.0, 1.0);
}

double onlineVoltagePlasticityDecay(const Config& config) {
    return std::clamp(config.onlineVoltagePlasticityDecay, 0.10, 0.999);
}

double onlineVoltagePlasticityThreshold(const Config& config) {
    return std::clamp(config.onlineVoltagePlasticityThreshold, 0.05, 0.95);
}

int onlineTraceTopK(const Config& config) {
    return std::max(1, std::min(3, config.knnK > 0 ? config.knnK : 3));
}

int onlineReplayDelaySteps(const Config& config) {
    return std::max(0, config.onlineReplayDelaySteps);
}

int onlineReplayPauseInterval(const Config& config) {
    return std::max(1, config.onlineReplayPauseInterval);
}

int onlineReplayBatchSize(const Config& config) {
    return std::max(1, config.onlineReplayBatchSize);
}

double onlineNeuromodulatorEligibilityMax(const Config& config) {
    if (!config.onlineNeuromodulatorEligibilityEnabled) {
        return 1.0;
    }
    return std::clamp(config.onlineNeuromodulatorEligibilityMax, 1.0, 3.0);
}

bool useOnlineReplaySuccessConsolidation(const Config& config) {
    return config.onlineReplaySuccessConsolidationEnabled &&
           config.onlineReplaySuccessConsolidationCapacity > 0 &&
           config.onlineReplaySuccessConsolidationRepeats > 0;
}

int onlineReplaySuccessConsolidationBatchSize(const Config& config) {
    return std::max(1, config.onlineReplaySuccessConsolidationBatchSize);
}

double computeNeuromodulatorEligibilityScale(const DecisionContext& context,
                                             bool positiveReward,
                                             const Config& config) {
    if (!config.onlineNeuromodulatorEligibilityEnabled) {
        return 1.0;
    }

    double scale = 1.0 + config.onlineNeuromodulatorUncertaintyGain * context.uncertainty;
    if (positiveReward) {
        scale += config.onlineNeuromodulatorPositiveRewardGain *
                 std::max(0.0, context.plasticity - 1.0);
    }
    return std::clamp(scale, 1.0, onlineNeuromodulatorEligibilityMax(config));
}

size_t computeReplayDelaySteps(double eligibilityScale, const Config& config) {
    const int configuredDelay = onlineReplayDelaySteps(config);
    if (!config.onlineReplayVariableDelayEnabled || configuredDelay <= 0) {
        return static_cast<size_t>(configuredDelay);
    }

    const double maxEligibility = onlineNeuromodulatorEligibilityMax(config);
    if (maxEligibility <= 1.0 + 1e-9) {
        return static_cast<size_t>(configuredDelay);
    }

    const double normalized =
        std::clamp((eligibilityScale - 1.0) / (maxEligibility - 1.0), 0.0, 1.0);
    return static_cast<size_t>(
        std::clamp(static_cast<int>(std::lround(normalized * configuredDelay)), 0, configuredDelay));
}

double onlineConfusionClusterGain(const Config& config) {
    return std::clamp(config.onlineConfusionClusterGain, 0.0, 3.0);
}

double onlineConfusionClusterDecay(const Config& config) {
    return std::clamp(config.onlineConfusionClusterDecay, 0.50, 0.999);
}

double decayEligibilityTrace(double currentScale, size_t delaySteps, const Config& config) {
    const double decay = std::clamp(config.onlineEligibilityTraceDecay, 0.10, 1.0);
    const double delayed =
        currentScale * std::pow(decay, static_cast<double>(std::max<size_t>(1, delaySteps)));
    return std::clamp(delayed, 0.05, onlineNeuromodulatorEligibilityMax(config));
}

void decayConfusionClusterMemory(ConfusionClusterMemory& memory, const Config& config) {
    const double decay = onlineConfusionClusterDecay(config);
    for (auto& row : memory.strengths) {
        for (double& value : row) {
            value *= decay;
        }
    }
}

void reinforceConfusionPair(ConfusionClusterMemory& memory, int a, int b, double amount) {
    if (a < 0 || b < 0 ||
        static_cast<size_t>(a) >= memory.strengths.size() ||
        static_cast<size_t>(b) >= memory.strengths.size() ||
        a == b || amount <= 0.0) {
        return;
    }
    auto& ab = memory.strengths[static_cast<size_t>(a)][static_cast<size_t>(b)];
    auto& ba = memory.strengths[static_cast<size_t>(b)][static_cast<size_t>(a)];
    ab = std::clamp(ab + amount, 0.0, 10.0);
    ba = std::clamp(ba + amount, 0.0, 10.0);
}

void updateConfusionClusterMemory(ConfusionClusterMemory& memory,
                                  const BilateralDecisionTrace& decision,
                                  bool negativeReward,
                                  const Config& config) {
    decayConfusionClusterMemory(memory, config);
    if (!negativeReward) {
        return;
    }

    const double gain = onlineConfusionClusterGain(config);
    if (gain <= 0.0) {
        return;
    }

    if (decision.hemisphereTraces.size() >= 2) {
        const int leftPredicted = decision.hemisphereTraces[0].predicted;
        const int rightPredicted = decision.hemisphereTraces[1].predicted;
        if (leftPredicted >= 0 && rightPredicted >= 0 && leftPredicted != rightPredicted) {
            reinforceConfusionPair(memory, leftPredicted, rightPredicted, 1.00 * gain);
        }
    }

    if (decision.topHypotheses.size() >= 2) {
        reinforceConfusionPair(memory,
                               decision.topHypotheses[0].label,
                               decision.topHypotheses[1].label,
                               0.75 * gain);
    }
}

double computeConfusionClusterScore(const ConfusionClusterMemory& memory,
                                    const BilateralDecisionTrace& decision) {
    double score = 0.0;
    if (decision.hemisphereTraces.size() >= 2) {
        const int leftPredicted = decision.hemisphereTraces[0].predicted;
        const int rightPredicted = decision.hemisphereTraces[1].predicted;
        if (leftPredicted >= 0 && rightPredicted >= 0 && leftPredicted != rightPredicted) {
            score = std::max(score,
                             memory.strengths[static_cast<size_t>(leftPredicted)]
                                             [static_cast<size_t>(rightPredicted)]);
        }
    }

    if (decision.topHypotheses.size() >= 2) {
        const int best = decision.topHypotheses[0].label;
        const int second = decision.topHypotheses[1].label;
        if (best >= 0 && second >= 0 &&
            static_cast<size_t>(best) < memory.strengths.size() &&
            static_cast<size_t>(second) < memory.strengths.size() &&
            best != second) {
            score = std::max(score,
                             memory.strengths[static_cast<size_t>(best)]
                                             [static_cast<size_t>(second)]);
        }
    }
    return score;
}

void recordHemisphereAgreement(EvaluationResult& result,
                               const BilateralDecisionTrace& decision,
                               int truth) {
    if (decision.hemisphereTraces.size() < 2) {
        return;
    }

    const int leftPredicted = decision.hemisphereTraces[0].predicted;
    const int rightPredicted = decision.hemisphereTraces[1].predicted;
    if (leftPredicted < 0 || rightPredicted < 0 || truth < 0 ||
        static_cast<size_t>(leftPredicted) >= result.leftHemisphereConfusion.size() ||
        static_cast<size_t>(rightPredicted) >= result.rightHemisphereConfusion.size() ||
        static_cast<size_t>(truth) >= result.leftHemisphereConfusion.size()) {
        return;
    }

    result.leftHemisphereConfusion[static_cast<size_t>(truth)]
                                 [static_cast<size_t>(leftPredicted)]++;
    result.rightHemisphereConfusion[static_cast<size_t>(truth)]
                                  [static_cast<size_t>(rightPredicted)]++;
    result.hemisphereComparable++;
    if (leftPredicted == rightPredicted) {
        result.hemisphereAgreements++;
        if (leftPredicted == truth) {
            result.hemisphereAgreementCorrect++;
        }
        return;
    }

    result.hemisphereDisagreements++;
    const bool leftCorrect = (leftPredicted == truth);
    const bool rightCorrect = (rightPredicted == truth);
    if (leftCorrect && !rightCorrect) {
        result.leftOnlyCorrectOnDisagreement++;
    } else if (rightCorrect && !leftCorrect) {
        result.rightOnlyCorrectOnDisagreement++;
    } else if (!leftCorrect && !rightCorrect) {
        result.bothWrongOnDisagreement++;
    }
}

DecisionContext buildDecisionContext(const BilateralDecisionTrace& decision,
                                     const Config& config,
                                     const ConfusionClusterMemory* confusionMemory = nullptr) {
    DecisionContext context;
    if (!decision.topHypotheses.empty()) {
        const double best = decision.topHypotheses.front().score;
        const double second =
            decision.topHypotheses.size() > 1 ? decision.topHypotheses[1].score : 0.0;
        const double normalizedMargin =
            best > 1e-6 ? std::clamp((best - second) / best, 0.0, 1.0) : 0.0;
        context.uncertainty = 1.0 - normalizedMargin;
    }

    int validVotes = 0;
    int disagreements = 0;
    for (const auto& hemisphereTrace : decision.hemisphereTraces) {
        if (hemisphereTrace.predicted < 0) {
            continue;
        }
        ++validVotes;
        if (decision.predicted >= 0 && hemisphereTrace.predicted != decision.predicted) {
            ++disagreements;
        }
    }
    context.disagreement = validVotes > 0
        ? static_cast<double>(disagreements) / static_cast<double>(validVotes)
        : 0.0;

    if (confusionMemory != nullptr) {
        context.confusionCluster = computeConfusionClusterScore(*confusionMemory, decision);
    }

    context.plasticity = 1.0 +
        config.onlineContextUncertaintyGain * context.uncertainty +
        config.onlineContextDisagreementGain * context.disagreement;
    context.plasticity = std::clamp(context.plasticity, 0.5, 3.0);
    context.replayPriority = context.plasticity + context.uncertainty +
                             0.5 * context.disagreement + context.confusionCluster;
    return context;
}

void recordActiveInferenceUsage(EvaluationResult& result,
                                const BilateralDecisionTrace& decision,
                                const Config& config) {
    if (!config.activeInferenceEnabled || config.activeInferenceFixations <= 1) {
        return;
    }
    result.activeInferenceSamples++;
    result.activeInferenceFixationSum +=
        static_cast<double>(std::max(1, decision.fixationCount));
    if (decision.fixationCount > 1) {
        result.activeInferenceExtraFixationSamples++;
    }
    if (decision.fixationCount < std::max(1, config.activeInferenceFixations)) {
        result.activeInferenceEarlyStops++;
    }
}

void enqueueReplayItemWithCapacity(std::vector<ReplayItem>& replayQueue,
                                   ReplayItem item,
                                   int capacity) {
    if (item.remainingReplays <= 0 || capacity <= 0) {
        return;
    }

    replayQueue.push_back(item);
    if (static_cast<int>(replayQueue.size()) <= capacity) {
        return;
    }

    const auto worstIt = std::min_element(
        replayQueue.begin(), replayQueue.end(),
        [](const ReplayItem& lhs, const ReplayItem& rhs) {
            if (lhs.priority != rhs.priority) {
                return lhs.priority < rhs.priority;
            }
            return lhs.sequence > rhs.sequence;
        });
    if (worstIt != replayQueue.end()) {
        replayQueue.erase(worstIt);
    }
}

void enqueueReplayItem(std::vector<ReplayItem>& replayQueue,
                       ReplayItem item,
                       const Config& config) {
    enqueueReplayItemWithCapacity(replayQueue, std::move(item), config.onlineReplayQueueCapacity);
}

bool popReplayItem(std::vector<ReplayItem>& replayQueue,
                   ReplayItem& item,
                   size_t currentStep,
                   bool flushAll = false) {
    if (replayQueue.empty()) {
        return false;
    }

    auto bestIt = replayQueue.end();
    for (auto it = replayQueue.begin(); it != replayQueue.end(); ++it) {
        if (!flushAll && it->readyStep > currentStep) {
            continue;
        }
        if (bestIt == replayQueue.end()) {
            bestIt = it;
            continue;
        }
        if (it->priority > bestIt->priority ||
            (it->priority == bestIt->priority && it->sequence < bestIt->sequence)) {
            bestIt = it;
        }
    }
    if (bestIt == replayQueue.end()) {
        return false;
    }
    item = *bestIt;
    replayQueue.erase(bestIt);
    return true;
}

void recordReplayTiming(EvaluationResult& result, const ReplayItem& item, size_t currentStep) {
    result.replayDelaySamples++;
    result.replayDelaySteps += static_cast<double>(item.traceAgeSteps);
    result.replayEligibilityScaleSum += item.eligibilityScale;
}

ReplayItem makeReplayItem(size_t recordIndex,
                          int remainingReplays,
                          double priority,
                          size_t currentStep,
                          double eligibilityScale,
                          const Config& config,
                          size_t sequence) {
    const size_t delaySteps = computeReplayDelaySteps(eligibilityScale, config);
    return ReplayItem{recordIndex,
                      remainingReplays,
                      priority,
                      delaySteps > 0 ? decayEligibilityTrace(eligibilityScale, delaySteps, config)
                                     : std::clamp(eligibilityScale,
                                                  0.05,
                                                  onlineNeuromodulatorEligibilityMax(config)),
                      delaySteps,
                      currentStep + delaySteps,
                      sequence};
}

void finalizeEvaluationFromRecords(EvaluationResult& result,
                                   const std::vector<OnlineSampleRecord>& records,
                                   double seconds) {
    const int numClasses = static_cast<int>(result.confusion.size());
    result.initialConfusion = makeClassMatrix<int>(numClasses, 0);
    result.confusion = makeClassMatrix<int>(numClasses, 0);
    result.correctedInitialPairs = makeClassMatrix<int>(numClasses, 0);
    result.unresolvedInitialPairs = makeClassMatrix<int>(numClasses, 0);
    result.tested = 0;
    result.correct = 0;
    result.initialCorrect = 0;
    result.fusionRescuesOneHemisphereRight = 0;
    result.fusionRescuesBothHemispheresWrong = 0;
    result.fusionMissesWhenLeftWasRight = 0;
    result.fusionMissesWhenRightWasRight = 0;
    result.initialWrongBothHemispheresWrong = 0;
    result.correctedFromOneHemisphereRight = 0;
    result.correctedFromBothHemispheresWrong = 0;

    for (const auto& record : records) {
        if (record.truth < 0 || record.initialPredicted < 0 || record.finalPredicted < 0 ||
            static_cast<size_t>(record.truth) >= result.confusion.size() ||
            static_cast<size_t>(record.initialPredicted) >= result.confusion.size() ||
            static_cast<size_t>(record.finalPredicted) >= result.confusion.size()) {
            continue;
        }
        result.initialConfusion[static_cast<size_t>(record.truth)]
                               [static_cast<size_t>(record.initialPredicted)]++;
        result.confusion[static_cast<size_t>(record.truth)]
                        [static_cast<size_t>(record.finalPredicted)]++;
        result.initialCorrect += (record.initialPredicted == record.truth) ? 1 : 0;
        result.correct += (record.finalPredicted == record.truth) ? 1 : 0;
        result.tested++;

        const bool leftCorrect = record.leftInitialPredicted == record.truth;
        const bool rightCorrect = record.rightInitialPredicted == record.truth;
        const bool exactlyOneHemisphereRight = leftCorrect != rightCorrect;
        const bool bothHemispheresWrong = !leftCorrect && !rightCorrect;
        const bool initialCorrect = record.initialPredicted == record.truth;
        const bool finalCorrect = record.finalPredicted == record.truth;

        if (exactlyOneHemisphereRight) {
            if (initialCorrect) {
                result.fusionRescuesOneHemisphereRight++;
            } else if (leftCorrect) {
                result.fusionMissesWhenLeftWasRight++;
            } else {
                result.fusionMissesWhenRightWasRight++;
            }
        } else if (bothHemispheresWrong) {
            if (initialCorrect) {
                result.fusionRescuesBothHemispheresWrong++;
            } else if (!initialCorrect) {
                result.initialWrongBothHemispheresWrong++;
            }
        }

        if (!initialCorrect) {
            if (finalCorrect) {
                result.correctedInitialPairs[static_cast<size_t>(record.truth)]
                                            [static_cast<size_t>(record.initialPredicted)]++;
                if (exactlyOneHemisphereRight) {
                    result.correctedFromOneHemisphereRight++;
                } else if (bothHemispheresWrong) {
                    result.correctedFromBothHemispheresWrong++;
                }
            } else {
                result.unresolvedInitialPairs[static_cast<size_t>(record.truth)]
                                             [static_cast<size_t>(record.initialPredicted)]++;
            }
        }
    }
    result.seconds = seconds;
}

void updateCentroid(std::vector<double>& centroid,
                    const std::vector<double>& pattern,
                    double learningRate) {
    if (pattern.empty()) {
        return;
    }
    if (centroid.empty()) {
        centroid = pattern;
        normalizeL2(centroid);
        return;
    }

    const size_t dim = std::min(centroid.size(), pattern.size());
    const double keep = std::clamp(1.0 - learningRate, 0.0, 1.0);
    const double learn = std::clamp(learningRate, 0.0, 1.0);
    for (size_t i = 0; i < dim; ++i) {
        centroid[i] = keep * centroid[i] + learn * pattern[i];
    }
    normalizeL2(centroid);
}

void updateRewardStdpPrototype(std::vector<double>& prototype,
                               const std::vector<double>& pattern,
                               double learningRate,
                               bool potentiation) {
    if (pattern.empty() || learningRate <= 0.0) {
        return;
    }

    if (prototype.empty()) {
        if (!potentiation) {
            return;
        }
        prototype = pattern;
        normalizeL2(prototype);
        return;
    }

    const size_t dim = std::min(prototype.size(), pattern.size());
    if (potentiation) {
        const double keep = std::clamp(1.0 - learningRate, 0.0, 1.0);
        const double learn = std::clamp(learningRate, 0.0, 1.0);
        for (size_t i = 0; i < dim; ++i) {
            prototype[i] = keep * prototype[i] + learn * pattern[i];
        }
    } else {
        const double ltd = std::clamp(learningRate, 0.0, 1.0);
        for (size_t i = 0; i < dim; ++i) {
            prototype[i] *= std::clamp(1.0 - ltd * std::max(0.0, pattern[i]), 0.0, 1.0);
        }
    }

    normalizeL2(prototype);
}

std::vector<double> computeRewardStdpEvidence(
    const std::vector<std::vector<double>>& prototypes,
    const std::vector<double>& pattern) {
    std::vector<double> scores(prototypes.size(), 0.0);
    for (size_t label = 0; label < prototypes.size(); ++label) {
        const auto& prototype = prototypes[label];
        if (prototype.empty() || prototype.size() != pattern.size()) {
            continue;
        }
        scores[label] = std::max(0.0, cosineSimilarity(pattern, prototype));
    }
    normalizeSum(scores);
    return scores;
}

template <typename RuntimeT>
void decayTripletStdpState(RuntimeT& runtime, size_t currentStep, const Config& config) {
    if (currentStep <= runtime.tripletTraceStep) {
        return;
    }

    const size_t deltaSteps = currentStep - runtime.tripletTraceStep;
    const double fastDecay =
        std::pow(onlineTripletStdpFastDecay(config), static_cast<double>(deltaSteps));
    const double slowDecay =
        std::pow(onlineTripletStdpSlowDecay(config), static_cast<double>(deltaSteps));
    for (double& value : runtime.tripletPreTrace) {
        value *= fastDecay;
    }
    for (double& value : runtime.tripletPostTraceFast) {
        value *= fastDecay;
    }
    for (double& value : runtime.tripletPostTraceSlow) {
        value *= slowDecay;
    }
    runtime.tripletTraceStep = currentStep;
}

std::vector<double> buildTripletStdpPattern(const std::vector<double>& pattern,
                                            const std::vector<double>& preTrace) {
    if (pattern.empty()) {
        return {};
    }

    std::vector<double> combined = pattern;
    if (preTrace.size() == pattern.size()) {
        for (size_t i = 0; i < combined.size(); ++i) {
            combined[i] = 0.75 * pattern[i] + 0.25 * std::max(0.0, preTrace[i]);
        }
    }
    normalizeL2(combined);
    return combined;
}

template <typename RuntimeT>
void updateTripletStdpTraces(RuntimeT& runtime,
                             const std::vector<double>& pattern,
                             int label,
                             double support) {
    if (pattern.empty() || label < 0 ||
        static_cast<size_t>(label) >= runtime.tripletPostTraceFast.size()) {
        return;
    }

    if (runtime.tripletPreTrace.size() != pattern.size()) {
        runtime.tripletPreTrace.assign(pattern.size(), 0.0);
    }
    for (size_t i = 0; i < pattern.size(); ++i) {
        runtime.tripletPreTrace[i] = 0.75 * runtime.tripletPreTrace[i] + 0.25 * pattern[i];
    }
    normalizeL2(runtime.tripletPreTrace);

    double& fast = runtime.tripletPostTraceFast[static_cast<size_t>(label)];
    double& slow = runtime.tripletPostTraceSlow[static_cast<size_t>(label)];
    const double previousFast = fast;
    fast = std::clamp(fast + support, 0.0, 4.0);
    slow = std::clamp(slow + 0.35 * previousFast + 0.15 * support, 0.0, 4.0);
}

std::vector<double> computeTripletStdpEvidence(
    const std::vector<std::vector<double>>& prototypes,
    const std::vector<double>& pattern) {
    std::vector<double> scores(prototypes.size(), 0.0);
    for (size_t label = 0; label < prototypes.size(); ++label) {
        const auto& prototype = prototypes[label];
        if (prototype.empty() || prototype.size() != pattern.size()) {
            continue;
        }
        scores[label] = std::max(0.0, cosineSimilarity(pattern, prototype));
    }
    normalizeSum(scores);
    return scores;
}

template <typename RuntimeT>
void decayVoltagePlasticityState(RuntimeT& runtime, size_t currentStep, const Config& config) {
    if (currentStep <= runtime.voltageTraceStep) {
        return;
    }

    const size_t deltaSteps = currentStep - runtime.voltageTraceStep;
    const double decay =
        std::pow(onlineVoltagePlasticityDecay(config), static_cast<double>(deltaSteps));
    for (double& value : runtime.voltageDepolarizationTrace) {
        value *= decay;
    }
    runtime.voltageTraceStep = currentStep;
}

double computeVoltagePlasticityDepolarization(const std::vector<double>& prototype,
                                              const std::vector<double>& pattern,
                                              double support,
                                              double persistentTrace) {
    double similarity = std::max(0.0, support);
    if (!prototype.empty() && prototype.size() == pattern.size()) {
        similarity = std::max(0.0, cosineSimilarity(pattern, prototype));
    }
    return std::clamp(0.60 * std::max(0.0, support) + 0.25 * similarity +
                          0.15 * std::min(1.0, std::max(0.0, persistentTrace)),
                      0.0,
                      2.0);
}

template <typename RuntimeT>
void updateVoltagePlasticityTrace(RuntimeT& runtime, int label, double depolarization) {
    if (label < 0 || static_cast<size_t>(label) >= runtime.voltageDepolarizationTrace.size()) {
        return;
    }
    double& trace = runtime.voltageDepolarizationTrace[static_cast<size_t>(label)];
    trace = std::clamp(trace + 0.50 * std::max(0.0, depolarization), 0.0, 4.0);
}

std::vector<double> computeVoltagePlasticityEvidence(
    const std::vector<std::vector<double>>& prototypes,
    const std::vector<double>& pattern) {
    std::vector<double> scores(prototypes.size(), 0.0);
    for (size_t label = 0; label < prototypes.size(); ++label) {
        const auto& prototype = prototypes[label];
        if (prototype.empty() || prototype.size() != pattern.size()) {
            continue;
        }
        scores[label] = std::max(0.0, cosineSimilarity(pattern, prototype));
    }
    normalizeSum(scores);
    return scores;
}

void insertOrReplaceOnlinePattern(
    std::vector<ClassificationStrategy::LabeledPattern>& patterns,
    std::vector<size_t>* sourceIndices,
    std::vector<std::vector<size_t>>& onlineIndices,
    const std::vector<double>& pattern,
    int label,
    int budgetPerClass) {
    if (label < 0 || static_cast<size_t>(label) >= onlineIndices.size() ||
        pattern.empty() || budgetPerClass <= 0) {
        return;
    }

    auto& labelOnlineIndices = onlineIndices[static_cast<size_t>(label)];
    if (static_cast<int>(labelOnlineIndices.size()) < budgetPerClass) {
        patterns.emplace_back(pattern, label);
        labelOnlineIndices.push_back(patterns.size() - 1);
        if (sourceIndices != nullptr) {
            sourceIndices->push_back(std::numeric_limits<size_t>::max());
        }
        return;
    }

    double worstSimilarity = std::numeric_limits<double>::infinity();
    size_t replaceIndex = labelOnlineIndices.front();
    for (size_t candidateIndex : labelOnlineIndices) {
        if (candidateIndex >= patterns.size()) {
            continue;
        }
        const double similarity = cosineSimilarity(pattern, patterns[candidateIndex].pattern);
        if (similarity < worstSimilarity) {
            worstSimilarity = similarity;
            replaceIndex = candidateIndex;
        }
    }
    if (replaceIndex < patterns.size()) {
        patterns[replaceIndex].pattern = pattern;
        patterns[replaceIndex].label = label;
    }
}

std::vector<LabelTrace> collectTopHypotheses(const std::vector<double>& confidence, int k) {
    std::vector<LabelTrace> traces;
    traces.reserve(confidence.size());
    for (size_t i = 0; i < confidence.size(); ++i) {
        traces.push_back({static_cast<int>(i), confidence[i]});
    }
    std::partial_sort(
        traces.begin(),
        traces.begin() + std::min<int>(k, static_cast<int>(traces.size())),
        traces.end(),
        [](const LabelTrace& lhs, const LabelTrace& rhs) { return lhs.score > rhs.score; });
    if (static_cast<int>(traces.size()) > k) {
        traces.resize(static_cast<size_t>(k));
    }
    return traces;
}

HemisphereDecisionTrace inferHemisphereDecision(HemisphereRuntime& hemisphere,
                                                const VisualStimulus& image,
                                                const Config& config,
                                                FlowAuditSampleCapture* flowAuditCapture = nullptr) {
    HemisphereDecisionTrace trace;
    trace.pattern = extractPattern(
        hemisphere.retinas, image, config, false, false, 0U, flowAuditCapture);
    FigureGroundState figureGroundState;
    if (useFigureGroundStage(config)) {
        figureGroundState = buildHemisphereFigureGroundState(hemisphere, trace.pattern, config);
        trace.figureGroundPattern = std::move(figureGroundState.pattern);
        hemisphere.lastFigureGroundPattern = trace.figureGroundPattern;
    } else {
        hemisphere.lastFigureGroundPattern.clear();
    }
    trace.classifierPattern = buildHemisphereClassifierPattern(
        hemisphere,
        trace.pattern,
        config,
        figureGroundState.valid ? &figureGroundState : nullptr);
    const auto& classifierPattern =
        trace.classifierPattern.empty() ? trace.pattern : trace.classifierPattern;
    trace.confidence = computeHemisphereReadoutConfidence(
        hemisphere, classifierPattern, config);
    normalizeSum(trace.confidence);
    if (figureGroundState.valid) {
        trace.figureGroundObjectConfidence = computeFigureGroundObjectMemoryConfidence(
            hemisphere,
            trace.pattern,
            figureGroundState,
            config,
            &trace.figureGroundObjectPattern);
        if (!trace.figureGroundObjectConfidence.empty() &&
            trace.figureGroundObjectConfidence.size() == trace.confidence.size()) {
            for (size_t label = 0; label < trace.confidence.size(); ++label) {
                trace.confidence[label] +=
                    config.figureGroundObjectMemoryGain *
                    trace.figureGroundObjectConfidence[label];
            }
            normalizeSum(trace.confidence);
        }
    } else {
        hemisphere.lastFigureGroundObjectPattern.clear();
    }
    trace.topHypotheses = collectTopHypotheses(trace.confidence, onlineTraceTopK(config));
    if (!trace.topHypotheses.empty()) {
        trace.predicted = trace.topHypotheses.front().label;
        const double best = trace.topHypotheses.front().score;
        const double second = trace.topHypotheses.size() > 1 ? trace.topHypotheses[1].score : 0.0;
        trace.margin = std::max(0.0, best - second);
    }
    return trace;
}

HemisphereDecisionTrace inferHemisphereDecisionFromPattern(HemisphereRuntime& hemisphere,
                                                           std::vector<double> pattern,
                                                           const Config& config) {
    HemisphereDecisionTrace trace;
    trace.pattern = std::move(pattern);
    FigureGroundState figureGroundState;
    if (useFigureGroundStage(config)) {
        figureGroundState = buildHemisphereFigureGroundState(hemisphere, trace.pattern, config);
        trace.figureGroundPattern = std::move(figureGroundState.pattern);
        hemisphere.lastFigureGroundPattern = trace.figureGroundPattern;
    } else {
        hemisphere.lastFigureGroundPattern.clear();
    }
    trace.classifierPattern = buildHemisphereClassifierPattern(
        hemisphere,
        trace.pattern,
        config,
        figureGroundState.valid ? &figureGroundState : nullptr);
    const auto& classifierPattern =
        trace.classifierPattern.empty() ? trace.pattern : trace.classifierPattern;
    trace.confidence = computeHemisphereReadoutConfidence(
        hemisphere, classifierPattern, config);
    normalizeSum(trace.confidence);
    if (figureGroundState.valid) {
        trace.figureGroundObjectConfidence = computeFigureGroundObjectMemoryConfidence(
            hemisphere,
            trace.pattern,
            figureGroundState,
            config,
            &trace.figureGroundObjectPattern);
        if (!trace.figureGroundObjectConfidence.empty() &&
            trace.figureGroundObjectConfidence.size() == trace.confidence.size()) {
            for (size_t label = 0; label < trace.confidence.size(); ++label) {
                trace.confidence[label] +=
                    config.figureGroundObjectMemoryGain *
                    trace.figureGroundObjectConfidence[label];
            }
            normalizeSum(trace.confidence);
        }
    } else {
        hemisphere.lastFigureGroundObjectPattern.clear();
    }
    trace.topHypotheses = collectTopHypotheses(trace.confidence, onlineTraceTopK(config));
    if (!trace.topHypotheses.empty()) {
        trace.predicted = trace.topHypotheses.front().label;
        const double best = trace.topHypotheses.front().score;
        const double second = trace.topHypotheses.size() > 1 ? trace.topHypotheses[1].score : 0.0;
        trace.margin = std::max(0.0, best - second);
    }
    return trace;
}

std::vector<HemisphereDecisionTrace> inferHemisphereDecisions(
    std::vector<HemisphereRuntime>& hemispheres,
    const VisualStimulus& image,
    const Config& config,
    std::vector<FlowAuditSampleCapture>* flowAuditCaptures) {
    if (useRecurrentSensoryState(config)) {
        if (flowAuditCaptures != nullptr) {
            flowAuditCaptures->assign(hemispheres.size(), {});
            for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
                extractPattern(hemispheres[hemisphereIndex].retinas,
                               image,
                               config,
                               false,
                               false,
                               0U,
                               &(*flowAuditCaptures)[hemisphereIndex]);
            }
        }

        auto recurrentResults =
            buildRecurrentSensoryPatterns(hemispheres, image, config, false, 0U);
        std::vector<HemisphereDecisionTrace> traces;
        traces.reserve(hemispheres.size());
        for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
            auto& hemisphere = hemispheres[hemisphereIndex];
            HemisphereDecisionTrace trace;
            if (hemisphereIndex < recurrentResults.size()) {
                trace.pattern = recurrentResults[hemisphereIndex].pattern;
                trace.classifierPattern = trace.pattern;
                trace.figureGroundPattern = recurrentResults[hemisphereIndex].figureGroundPattern;
            }
            if (trace.pattern.empty()) {
                FlowAuditSampleCapture* capture = nullptr;
                if (flowAuditCaptures != nullptr && hemisphereIndex < flowAuditCaptures->size()) {
                    capture = &(*flowAuditCaptures)[hemisphereIndex];
                }
                traces.push_back(inferHemisphereDecision(hemisphere, image, config, capture));
                continue;
            }
            hemisphere.lastFigureGroundPattern = trace.figureGroundPattern;
            hemisphere.lastFigureGroundObjectPattern.clear();
            trace.confidence = computeHemisphereReadoutConfidence(
                hemisphere, trace.classifierPattern, config);
            normalizeSum(trace.confidence);
            trace.topHypotheses = collectTopHypotheses(trace.confidence, onlineTraceTopK(config));
            if (!trace.topHypotheses.empty()) {
                trace.predicted = trace.topHypotheses.front().label;
                const double best = trace.topHypotheses.front().score;
                const double second =
                    trace.topHypotheses.size() > 1 ? trace.topHypotheses[1].score : 0.0;
                trace.margin = std::max(0.0, best - second);
            }
            traces.push_back(std::move(trace));
        }
        return traces;
    }

    return runHemisphereTasks<HemisphereDecisionTrace>(
        hemispheres.size(),
        [&](size_t hemisphereIndex) {
            FlowAuditSampleCapture* capture = nullptr;
            if (flowAuditCaptures != nullptr && hemisphereIndex < flowAuditCaptures->size()) {
                capture = &(*flowAuditCaptures)[hemisphereIndex];
            }
            return inferHemisphereDecision(hemispheres[hemisphereIndex], image, config, capture);
        });
}

std::vector<double> buildFusionPatternFromTraces(const std::vector<HemisphereDecisionTrace>& traces,
                                                 const Config& config) {
    std::vector<double> fusionPattern;
    std::vector<std::vector<double>> confidences;
    confidences.reserve(traces.size());
    for (const auto& trace : traces) {
        std::vector<double> confidence = trace.confidence;
        normalizeL2(confidence);
        confidences.push_back(std::move(confidence));
    }

    for (const auto& confidence : confidences) {
        fusionPattern.insert(fusionPattern.end(), confidence.begin(), confidence.end());
    }

    std::string mode = config.fusionFeatureMode;
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (mode == "interaction") {
        for (size_t a = 0; a < confidences.size(); ++a) {
            for (size_t b = a + 1; b < confidences.size(); ++b) {
                const size_t dim = std::min(confidences[a].size(), confidences[b].size());
                for (size_t i = 0; i < dim; ++i) {
                    fusionPattern.push_back(confidences[a][i] * confidences[b][i]);
                }
                for (size_t i = 0; i < dim; ++i) {
                    fusionPattern.push_back(std::abs(confidences[a][i] - confidences[b][i]));
                }
            }
        }
    }

    normalizeL2(fusionPattern);
    return fusionPattern;
}

std::vector<double> buildConfidenceConcatPatternFromTraces(
    const std::vector<HemisphereDecisionTrace>& traces) {
    std::vector<double> fusionPattern;
    for (const auto& trace : traces) {
        std::vector<double> confidence = trace.confidence;
        normalizeL2(confidence);
        fusionPattern.insert(fusionPattern.end(), confidence.begin(), confidence.end());
    }
    normalizeL2(fusionPattern);
    return fusionPattern;
}

std::vector<double> buildHemisphereConcatPatternFromTraces(
    const std::vector<HemisphereDecisionTrace>& traces) {
    std::vector<double> fusionPattern;
    for (const auto& trace : traces) {
        const auto& hemispherePattern =
            trace.classifierPattern.empty() ? trace.pattern : trace.classifierPattern;
        fusionPattern.insert(fusionPattern.end(), hemispherePattern.begin(), hemispherePattern.end());
    }
    normalizeL2(fusionPattern);
    return fusionPattern;
}

std::vector<double> buildFusionPattern(std::vector<HemisphereRuntime>& hemispheres,
                                       const VisualStimulus& image,
                                       const Config& config) {
    return buildFusionPatternFromTraces(
        inferHemisphereDecisions(hemispheres, image, config),
        config);
}

bool useCorpusCallosumFusion(const Config& config) {
    return toLower(config.fusionClassifier) == "corpus_callosum";
}

void buildHemisphereClassCentroids(HemisphereRuntime& hemisphere) {
    for (auto& centroid : hemisphere.classCentroids) {
        centroid.clear();
    }

    std::vector<int> counts(hemisphere.classCentroids.size(), 0);
    for (const auto& labeledPattern : hemisphere.trainingPatterns) {
        if (labeledPattern.label < 0 ||
            static_cast<size_t>(labeledPattern.label) >= hemisphere.classCentroids.size() ||
            labeledPattern.pattern.empty()) {
            continue;
        }
        auto& centroid = hemisphere.classCentroids[static_cast<size_t>(labeledPattern.label)];
        if (centroid.empty()) {
            centroid.assign(labeledPattern.pattern.size(), 0.0);
        }
        const size_t dim = std::min(centroid.size(), labeledPattern.pattern.size());
        for (size_t i = 0; i < dim; ++i) {
            centroid[i] += labeledPattern.pattern[i];
        }
        counts[static_cast<size_t>(labeledPattern.label)]++;
    }

    for (size_t label = 0; label < hemisphere.classCentroids.size(); ++label) {
        auto& centroid = hemisphere.classCentroids[label];
        if (centroid.empty() || counts[label] <= 0) {
            continue;
        }
        const double invCount = 1.0 / static_cast<double>(counts[label]);
        for (double& value : centroid) {
            value *= invCount;
        }
        normalizeL2(centroid);
    }
}

std::vector<double> computeCentroidEvidence(const HemisphereRuntime& hemisphere,
                                            const std::vector<double>& pattern) {
    std::vector<double> centroidScores(hemisphere.classCentroids.size(), 0.0);
    for (size_t label = 0; label < hemisphere.classCentroids.size(); ++label) {
        const auto& centroid = hemisphere.classCentroids[label];
        if (centroid.empty() || centroid.size() != pattern.size()) {
            continue;
        }
        centroidScores[label] = std::max(0.0, cosineSimilarity(pattern, centroid));
    }
    normalizeSum(centroidScores);
    return centroidScores;
}

std::vector<double> computeCentroidEvidence(
    const std::vector<std::vector<double>>& centroids,
    const std::vector<double>& pattern) {
    std::vector<double> centroidScores(centroids.size(), 0.0);
    for (size_t label = 0; label < centroids.size(); ++label) {
        const auto& centroid = centroids[label];
        if (centroid.empty() || centroid.size() != pattern.size()) {
            continue;
        }
        centroidScores[label] = std::max(0.0, cosineSimilarity(pattern, centroid));
    }
    normalizeSum(centroidScores);
    return centroidScores;
}

std::vector<ObjectStateAttractor> buildObjectStateAttractors(
    const std::vector<ClassificationStrategy::LabeledPattern>& patterns,
    int numClasses,
    int unitsPerClass) {
    std::vector<ObjectStateAttractor> attractors;
    if (numClasses <= 0 || unitsPerClass <= 0 || patterns.empty()) {
        return attractors;
    }

    const auto classCentroids = computeClassCentroids(patterns, numClasses);
    std::vector<std::vector<size_t>> byLabel(static_cast<size_t>(numClasses));
    for (size_t patternIndex = 0; patternIndex < patterns.size(); ++patternIndex) {
        const auto& labeledPattern = patterns[patternIndex];
        if (labeledPattern.label < 0 || labeledPattern.label >= numClasses ||
            labeledPattern.pattern.empty()) {
            continue;
        }
        byLabel[static_cast<size_t>(labeledPattern.label)].push_back(patternIndex);
    }

    for (int label = 0; label < numClasses; ++label) {
        const auto& indices = byLabel[static_cast<size_t>(label)];
        if (indices.empty()) {
            continue;
        }

        const int targetUnits =
            std::min<int>(unitsPerClass, static_cast<int>(indices.size()));
        std::vector<size_t> selected;
        selected.reserve(static_cast<size_t>(targetUnits));

        size_t seedIndex = indices.front();
        const auto& centroid = classCentroids[static_cast<size_t>(label)];
        if (!centroid.empty()) {
            double bestSeedScore = -std::numeric_limits<double>::infinity();
            for (size_t candidateIndex : indices) {
                const double score = cosineSimilarity(patterns[candidateIndex].pattern, centroid);
                if (score > bestSeedScore) {
                    bestSeedScore = score;
                    seedIndex = candidateIndex;
                }
            }
        }
        selected.push_back(seedIndex);

        while (static_cast<int>(selected.size()) < targetUnits) {
            double bestDistance = -1.0;
            size_t bestIndex = std::numeric_limits<size_t>::max();
            for (size_t candidateIndex : indices) {
                if (std::find(selected.begin(), selected.end(), candidateIndex) != selected.end()) {
                    continue;
                }
                double minDistance = std::numeric_limits<double>::infinity();
                for (size_t chosenIndex : selected) {
                    const double similarity =
                        std::max(0.0,
                                 cosineSimilarity(patterns[candidateIndex].pattern,
                                                  patterns[chosenIndex].pattern));
                    minDistance = std::min(minDistance, 1.0 - similarity);
                }
                if (minDistance > bestDistance) {
                    bestDistance = minDistance;
                    bestIndex = candidateIndex;
                }
            }
            if (bestIndex == std::numeric_limits<size_t>::max()) {
                break;
            }
            selected.push_back(bestIndex);
        }

        for (size_t selectedIndex : selected) {
            ObjectStateAttractor attractor;
            attractor.label = label;
            attractor.prototype = patterns[selectedIndex].pattern;
            normalizeL2(attractor.prototype);
            attractors.push_back(std::move(attractor));
        }
    }

    return attractors;
}

std::vector<double> computeObjectStateReadoutConfidence(
    const std::vector<ObjectStateAttractor>& attractors,
    const std::vector<double>& pattern,
    int numClasses,
    const Config& config) {
    if (!useRecurrentObjectStateReadout(config) || attractors.empty() || pattern.empty() ||
        numClasses <= 0) {
        return {};
    }

    std::vector<double> inputs(attractors.size(), 0.0);
    for (size_t i = 0; i < attractors.size(); ++i) {
        const auto& attractor = attractors[i];
        if (attractor.prototype.empty() || attractor.prototype.size() != pattern.size()) {
            continue;
        }
        inputs[i] = std::max(0.0, cosineSimilarity(pattern, attractor.prototype));
    }

    std::vector<double> state = inputs;
    const int cycles = std::max(1, config.recurrentObjectStateCycles);
    for (int cycle = 0; cycle < cycles; ++cycle) {
        std::vector<double> classSupport(static_cast<size_t>(numClasses), 0.0);
        for (size_t i = 0; i < attractors.size(); ++i) {
            const int label = attractors[i].label;
            if (label < 0 || label >= numClasses) {
                continue;
            }
            classSupport[static_cast<size_t>(label)] =
                std::max(classSupport[static_cast<size_t>(label)], state[i]);
        }
        const double totalSupport =
            std::accumulate(classSupport.begin(), classSupport.end(), 0.0);

        std::vector<double> next(state.size(), 0.0);
        for (size_t i = 0; i < attractors.size(); ++i) {
            const int label = attractors[i].label;
            if (label < 0 || label >= numClasses) {
                continue;
            }
            const double ownSupport = classSupport[static_cast<size_t>(label)];
            const double otherSupport =
                (totalSupport - ownSupport) /
                static_cast<double>(std::max(1, numClasses - 1));
            double value =
                config.recurrentObjectStateInputGain * inputs[i] +
                config.recurrentObjectStateSelfGain * state[i] +
                config.recurrentObjectStateClassSupportGain * ownSupport -
                config.recurrentObjectStateCompetitionGain * otherSupport;
            next[i] = std::max(0.0, value);
        }
        const double maxValue =
            *std::max_element(next.begin(), next.end());
        if (maxValue > 1e-9) {
            for (double& value : next) {
                value /= maxValue;
            }
        }
        state = std::move(next);
    }

    std::vector<double> classScores(static_cast<size_t>(numClasses), 0.0);
    for (size_t i = 0; i < attractors.size(); ++i) {
        const int label = attractors[i].label;
        if (label < 0 || label >= numClasses) {
            continue;
        }
        classScores[static_cast<size_t>(label)] =
            std::max(classScores[static_cast<size_t>(label)], state[i]);
    }
    normalizeSum(classScores);
    return classScores;
}

void rebuildHemisphereObjectStateAttractors(HemisphereRuntime& hemisphere,
                                            const Config& config) {
    if (!useRecurrentObjectStateReadout(config)) {
        hemisphere.recurrentObjectStateAttractors.clear();
        return;
    }
    hemisphere.recurrentObjectStateAttractors = buildObjectStateAttractors(
        hemisphere.trainingPatterns,
        static_cast<int>(hemisphere.classCentroids.size()),
        config.recurrentObjectStateUnitsPerClass);
}

std::vector<double> computeHemisphereReadoutConfidence(
    const HemisphereRuntime& hemisphere,
    const std::vector<double>& pattern,
    const Config& config) {
    if (useRecurrentObjectStateReadout(config)) {
        auto objectConfidence = computeObjectStateReadoutConfidence(
            hemisphere.recurrentObjectStateAttractors,
            pattern,
            static_cast<int>(hemisphere.classCentroids.size()),
            config);
        const double objectMass =
            std::accumulate(objectConfidence.begin(), objectConfidence.end(), 0.0);
        if (objectMass > 1e-9) {
            return objectConfidence;
        }
    }
    if (useRecurrentPopulationReadout(config)) {
        auto centroidConfidence = computeCentroidEvidence(hemisphere, pattern);
        const double centroidMass =
            std::accumulate(centroidConfidence.begin(), centroidConfidence.end(), 0.0);
        if (centroidMass > 1e-9) {
            return centroidConfidence;
        }
    }
    return hemisphere.classifier->classifyWithConfidence(
        pattern, hemisphere.trainingPatterns, cosineSimilarity);
}

std::vector<double> computeHemisphereSupportReadoutConfidence(
    const HemisphereRuntime& hemisphere,
    const std::vector<double>& pattern,
    const std::vector<ClassificationStrategy::LabeledPattern>& supportPatterns,
    const Config& config) {
    if (useRecurrentObjectStateReadout(config)) {
        const auto attractors = buildObjectStateAttractors(
            supportPatterns,
            static_cast<int>(hemisphere.classCentroids.size()),
            config.recurrentObjectStateUnitsPerClass);
        auto objectConfidence = computeObjectStateReadoutConfidence(
            attractors,
            pattern,
            static_cast<int>(hemisphere.classCentroids.size()),
            config);
        const double objectMass =
            std::accumulate(objectConfidence.begin(), objectConfidence.end(), 0.0);
        if (objectMass > 1e-9) {
            return objectConfidence;
        }
    }
    if (useRecurrentPopulationReadout(config)) {
        const auto supportCentroids =
            computeClassCentroids(supportPatterns, static_cast<int>(hemisphere.classCentroids.size()));
        auto centroidConfidence = computeCentroidEvidence(supportCentroids, pattern);
        const double centroidMass =
            std::accumulate(centroidConfidence.begin(), centroidConfidence.end(), 0.0);
        if (centroidMass > 1e-9) {
            return centroidConfidence;
        }
    }
    return hemisphere.classifier->classifyWithConfidence(
        pattern, supportPatterns, cosineSimilarity);
}

double computePredictionErrorFeedbackScale(const HemisphereRuntime& hemisphere,
                                           const HemisphereDecisionTrace& trace,
                                           const std::vector<double>& pattern,
                                           const Config& config) {
    if (!config.predictionErrorFeedbackEnabled ||
        config.predictionErrorFeedbackGain <= 0.0 ||
        config.predictionErrorFeedbackMaxPenalty <= 0.0 ||
        trace.predicted < 0 ||
        static_cast<size_t>(trace.predicted) >= hemisphere.classCentroids.size() ||
        pattern.empty()) {
        return 1.0;
    }

    const auto& centroid = hemisphere.classCentroids[static_cast<size_t>(trace.predicted)];
    if (centroid.empty() || centroid.size() != pattern.size()) {
        return 1.0;
    }

    const double topScore =
        trace.topHypotheses.empty() ? 0.0 : trace.topHypotheses.front().score;
    if (topScore < config.predictionErrorFeedbackMinConfidence) {
        return 1.0;
    }

    const double expectedSimilarity =
        std::clamp(std::max(0.0, cosineSimilarity(pattern, centroid)), 0.0, 1.0);
    const double predictionError = 1.0 - expectedSimilarity;
    const double errorGate =
        (predictionError - config.predictionErrorFeedbackThreshold) /
        std::max(1e-9, 1.0 - config.predictionErrorFeedbackThreshold);
    const double penalty = std::clamp(config.predictionErrorFeedbackGain * errorGate,
                                      0.0,
                                      config.predictionErrorFeedbackMaxPenalty);
    return std::clamp(1.0 - penalty, 0.0, 1.0);
}

std::vector<double> computeNeighborSignature(const HemisphereRuntime& hemisphere,
                                             const std::vector<double>& pattern,
                                             int k) {
    std::vector<std::pair<int, double>> neighbors;
    neighbors.reserve(hemisphere.trainingPatterns.size());

    for (size_t i = 0; i < hemisphere.trainingPatterns.size(); ++i) {
        const auto& labeledPattern = hemisphere.trainingPatterns[i];
        if (labeledPattern.label < 0 ||
            static_cast<size_t>(labeledPattern.label) >= hemisphere.classCentroids.size()) {
            continue;
        }
        neighbors.emplace_back(static_cast<int>(i),
                               cosineSimilarity(pattern, labeledPattern.pattern));
    }

    const int actualK = std::min<int>(std::max(1, k), static_cast<int>(neighbors.size()));
    if (actualK <= 0) {
        return std::vector<double>(hemisphere.classCentroids.size(), 0.0);
    }

    std::partial_sort(neighbors.begin(), neighbors.begin() + actualK, neighbors.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });

    std::vector<double> signature(hemisphere.classCentroids.size(), 0.0);
    for (int i = 0; i < actualK; ++i) {
        const auto& [index, similarity] = neighbors[static_cast<size_t>(i)];
        const int label = hemisphere.trainingPatterns[static_cast<size_t>(index)].label;
        signature[static_cast<size_t>(label)] += std::max(0.0, similarity);
    }
    normalizeSum(signature);
    return signature;
}

void calibrateCorpusCallosumWeights(std::vector<HemisphereRuntime>& hemispheres,
                                    const VisualDomainAdapter& loader,
                                    const std::vector<size_t>& indices,
                                    const Config& config) {
    struct CorpusStats {
        double overall = 1.0;
        std::vector<double> trueLabelAccuracy;
        std::vector<double> predictionPrecision;
    };
    const auto rawStats = runHemisphereTasks<CorpusStats>(
        hemispheres.size(),
        [&](size_t hemisphereIndex) {
            CorpusStats stats;
            auto& hemisphere = hemispheres[hemisphereIndex];
            stats.trueLabelAccuracy.assign(hemisphere.classWeights.size(), 1.0);
            stats.predictionPrecision.assign(hemisphere.classWeights.size(), 1.0);
            const auto supportPatterns = buildSupportPatterns(hemisphere, indices);
            std::vector<int> totals(hemisphere.classWeights.size(), 0);
            std::vector<int> corrects(hemisphere.classWeights.size(), 0);
            std::vector<int> predictedTotals(hemisphere.classWeights.size(), 0);
            std::vector<int> predictedCorrects(hemisphere.classWeights.size(), 0);
            int totalSamples = 0;
            int totalCorrect = 0;

            for (size_t index : indices) {
                const auto& image = loader.getStimulus(index);
                const int truth = image.label;
                if (truth < 0 ||
                    static_cast<size_t>(truth) >= hemisphere.classWeights.size() ||
                    supportPatterns.empty()) {
                    continue;
                }

                const auto pattern =
                    extractPattern(hemisphere.retinas, image, config, false, false);
                const auto confidence = computeHemisphereSupportReadoutConfidence(
                    hemisphere, pattern, supportPatterns, config);
                const int predicted = static_cast<int>(
                    std::distance(confidence.begin(),
                                  std::max_element(confidence.begin(), confidence.end())));
                totals[static_cast<size_t>(truth)]++;
                predictedTotals[static_cast<size_t>(predicted)]++;
                totalSamples++;
                if (predicted == truth) {
                    corrects[static_cast<size_t>(truth)]++;
                    predictedCorrects[static_cast<size_t>(predicted)]++;
                    totalCorrect++;
                }
            }

            stats.overall =
                (static_cast<double>(totalCorrect) + 1.0) /
                (static_cast<double>(std::max(1, totalSamples)) + 2.0);
            for (size_t label = 0; label < hemisphere.classWeights.size(); ++label) {
                stats.trueLabelAccuracy[label] =
                    (static_cast<double>(corrects[label]) + 1.0) /
                    (static_cast<double>(std::max(1, totals[label])) + 2.0);
                stats.predictionPrecision[label] =
                    (static_cast<double>(predictedCorrects[label]) + 1.0) /
                    (static_cast<double>(std::max(1, predictedTotals[label])) + 2.0);
            }
            return stats;
        });

    double meanOverall = 0.0;
    for (const auto& stats : rawStats) {
        meanOverall += stats.overall;
    }
    meanOverall /= std::max<size_t>(1, rawStats.size());

    for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
        auto& hemisphere = hemispheres[hemisphereIndex];
        hemisphere.overallWeight =
            std::clamp(rawStats[hemisphereIndex].overall / std::max(1e-6, meanOverall), 0.85, 1.15);
        for (size_t label = 0; label < hemisphere.classWeights.size(); ++label) {
            double meanClassAcc = 0.0;
            double meanPredictionPrecision = 0.0;
            for (const auto& stats : rawStats) {
                meanClassAcc += stats.trueLabelAccuracy[label];
                meanPredictionPrecision += stats.predictionPrecision[label];
            }
            meanClassAcc /= std::max<size_t>(1, rawStats.size());
            meanPredictionPrecision /= std::max<size_t>(1, rawStats.size());
            hemisphere.classWeights[label] = std::clamp(
                rawStats[hemisphereIndex].trueLabelAccuracy[label] /
                    std::max(1e-6, meanClassAcc),
                0.85, 1.15);
            hemisphere.predictionWeights[label] = std::clamp(
                rawStats[hemisphereIndex].predictionPrecision[label] /
                    std::max(1e-6, meanPredictionPrecision),
                0.85, 1.15);
        }
    }
}

const LabelTrace* findLabelTrace(const std::vector<LabelTrace>& traces, int label) {
    for (const auto& trace : traces) {
        if (trace.label == label) {
            return &trace;
        }
    }
    return nullptr;
}

void scaleClamped(double& value, double factor, double minValue, double maxValue) {
    value = std::clamp(value * factor, minValue, maxValue);
}

BilateralDecisionTrace buildCorpusCallosumDecisionFromTraces(
    std::vector<HemisphereRuntime>& hemispheres,
    std::vector<HemisphereDecisionTrace> hemisphereTraces,
    const Config& config) {
    BilateralDecisionTrace trace;
    trace.hemisphereTraces = std::move(hemisphereTraces);
    trace.fusionPattern = buildFusionPatternFromTraces(trace.hemisphereTraces, config);
    trace.combinedConfidence.assign(static_cast<size_t>(config.numClasses), 0.0);
    std::vector<double> disagreementScores(hemispheres.size(), 0.0);

    for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
        auto& hemisphere = hemispheres[hemisphereIndex];
        const auto& hemisphereTrace = trace.hemisphereTraces[hemisphereIndex];
        const auto& classifierPattern =
            hemisphereTrace.classifierPattern.empty() ? hemisphereTrace.pattern
                                                      : hemisphereTrace.classifierPattern;
        const double feedbackScale = computePredictionErrorFeedbackScale(
            hemisphere, hemisphereTrace, classifierPattern, config);
        std::vector<double> centroidEvidence;
        if (config.corpusCentroidGain > 0.0) {
            centroidEvidence = computeCentroidEvidence(hemisphere, classifierPattern);
        }
        std::vector<double> neighborSignature;
        if (config.corpusNeighborGain > 0.0) {
            const int neighborK = config.stage1K > 0 ? config.stage1K : config.knnK;
            neighborSignature = computeNeighborSignature(hemisphere, classifierPattern, neighborK);
        }
        std::vector<double> rewardStdpEvidence;
        if (useOnlineRewardStdp(config)) {
            rewardStdpEvidence =
                computeRewardStdpEvidence(hemisphere.rewardStdpPrototypes, classifierPattern);
        }
        std::vector<double> tripletStdpEvidence;
        if (useOnlineTripletStdp(config)) {
            tripletStdpEvidence =
                computeTripletStdpEvidence(hemisphere.tripletStdpPrototypes, classifierPattern);
        }
        std::vector<double> voltagePlasticityEvidence;
        if (useOnlineVoltagePlasticity(config)) {
            voltagePlasticityEvidence = computeVoltagePlasticityEvidence(
                hemisphere.voltagePlasticityPrototypes, classifierPattern);
        }

        for (size_t label = 0;
             label < trace.combinedConfidence.size() && label < hemisphereTrace.confidence.size();
             ++label) {
            const double labelFeedbackScale =
                static_cast<int>(label) == hemisphereTrace.predicted ? feedbackScale : 1.0;
            trace.combinedConfidence[label] += labelFeedbackScale *
                                               hemisphere.overallWeight *
                                               hemisphere.classWeights[label] *
                                               hemisphereTrace.confidence[label];
            if (!centroidEvidence.empty()) {
                trace.combinedConfidence[label] += config.corpusCentroidGain *
                                                   labelFeedbackScale *
                                                   hemisphere.overallWeight *
                                                   hemisphere.classWeights[label] *
                                                   centroidEvidence[label];
            }
            if (!neighborSignature.empty()) {
                trace.combinedConfidence[label] += config.corpusNeighborGain *
                                                   labelFeedbackScale *
                                                   hemisphere.overallWeight *
                                                   hemisphere.classWeights[label] *
                                                   neighborSignature[label];
            }
            if (!rewardStdpEvidence.empty()) {
                trace.combinedConfidence[label] += onlineRewardStdpGain(config) *
                                                   labelFeedbackScale *
                                                   hemisphere.overallWeight *
                                                   hemisphere.classWeights[label] *
                                                   rewardStdpEvidence[label];
            }
            if (!tripletStdpEvidence.empty()) {
                trace.combinedConfidence[label] += onlineTripletStdpGain(config) *
                                                   labelFeedbackScale *
                                                   hemisphere.overallWeight *
                                                   hemisphere.classWeights[label] *
                                                   tripletStdpEvidence[label];
            }
            if (!voltagePlasticityEvidence.empty()) {
                trace.combinedConfidence[label] += onlineVoltagePlasticityGain(config) *
                                                   labelFeedbackScale *
                                                   hemisphere.overallWeight *
                                                   hemisphere.classWeights[label] *
                                                   voltagePlasticityEvidence[label];
            }
        }

        if (hemisphereTrace.predicted >= 0 &&
            static_cast<size_t>(hemisphereTrace.predicted) < trace.combinedConfidence.size()) {
            const double topScore =
                hemisphereTrace.topHypotheses.empty() ? 0.0 : hemisphereTrace.topHypotheses.front().score;
            const double voteEvidence =
                topScore * (1.0 + config.corpusMarginGain * hemisphereTrace.margin);
            trace.combinedConfidence[static_cast<size_t>(hemisphereTrace.predicted)] +=
                config.corpusVoteGain *
                feedbackScale *
                hemisphere.overallWeight *
                hemisphere.predictionWeights[static_cast<size_t>(hemisphereTrace.predicted)] *
                voteEvidence;
            disagreementScores[hemisphereIndex] =
                feedbackScale *
                hemisphere.overallWeight *
                hemisphere.predictionWeights[static_cast<size_t>(hemisphereTrace.predicted)] *
                hemisphere.classWeights[static_cast<size_t>(hemisphereTrace.predicted)] *
                voteEvidence;
        }
    }

    if (config.corpusDisagreementGain > 0.0 && trace.hemisphereTraces.size() >= 2) {
        bool disagreement = false;
        int validVotes = 0;
        int previousPredicted = -1;
        for (const auto& hemisphereTrace : trace.hemisphereTraces) {
            if (hemisphereTrace.predicted < 0 || hemisphereTrace.predicted >= config.numClasses) {
                continue;
            }
            ++validVotes;
            if (previousPredicted >= 0 && hemisphereTrace.predicted != previousPredicted) {
                disagreement = true;
            }
            previousPredicted = hemisphereTrace.predicted;
        }

        if (disagreement && validVotes >= 2) {
            const double sumScores =
                std::accumulate(disagreementScores.begin(), disagreementScores.end(), 0.0);
            for (size_t hemisphereIndex = 0; hemisphereIndex < trace.hemisphereTraces.size(); ++hemisphereIndex) {
                const auto& hemisphereTrace = trace.hemisphereTraces[hemisphereIndex];
                if (hemisphereTrace.predicted < 0 ||
                    static_cast<size_t>(hemisphereTrace.predicted) >= trace.combinedConfidence.size()) {
                    continue;
                }
                const double topScore = hemisphereTrace.topHypotheses.empty()
                    ? 0.0
                    : hemisphereTrace.topHypotheses.front().score;
                const double normalizedScore = sumScores > 1e-9
                    ? (disagreementScores[hemisphereIndex] / sumScores)
                    : (1.0 / static_cast<double>(validVotes));
                trace.combinedConfidence[static_cast<size_t>(hemisphereTrace.predicted)] +=
                    config.corpusDisagreementGain * normalizedScore * std::max(0.0, topScore);
            }
        }
    }

    normalizeSum(trace.combinedConfidence);
    trace.topHypotheses = collectTopHypotheses(trace.combinedConfidence, onlineTraceTopK(config));
    if (!trace.topHypotheses.empty()) {
        trace.predicted = trace.topHypotheses.front().label;
    }
    return trace;
}

template <typename BuilderFn>
BilateralDecisionTrace inferActiveVisionDecision(std::vector<HemisphereRuntime>& hemispheres,
                                                 const VisualStimulus& image,
                                                 const Config& config,
                                                 BuilderFn builder) {
    auto candidates = buildActiveInferenceFixationCandidates(image, config);
    if (candidates.empty()) {
        return builder(inferHemisphereDecisions(hemispheres, image, config));
    }

    std::vector<HemispherePatternAccumulator> accumulators(hemispheres.size());
    std::vector<size_t> selectedFixations;
    selectedFixations.reserve(static_cast<size_t>(std::max(1, config.activeInferenceFixations)));
    BilateralDecisionTrace decision;

    const int maxFixations = std::max(1, config.activeInferenceFixations);
    const int minFixations = std::max(1, std::min(maxFixations, config.activeInferenceMinFixations));
    for (int fixationIndex = 0; fixationIndex < maxFixations; ++fixationIndex) {
        const size_t candidateIndex =
            fixationIndex == 0 ? 0u : selectNextActiveFixation(candidates, selectedFixations, config);
        if (candidateIndex >= candidates.size()) {
            break;
        }
        selectedFixations.push_back(candidateIndex);
        for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
            appendFixationToAccumulator(
                hemispheres[hemisphereIndex],
                accumulators[hemisphereIndex],
                candidates[candidateIndex],
                config);
        }

        std::vector<HemisphereDecisionTrace> hemisphereTraces;
        hemisphereTraces.reserve(hemispheres.size());
        for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
            hemisphereTraces.push_back(inferHemisphereDecisionFromPattern(
                hemispheres[hemisphereIndex],
                buildPatternFromAccumulator(hemispheres[hemisphereIndex], accumulators[hemisphereIndex]),
                config));
        }

        decision = builder(std::move(hemisphereTraces));
        decision.fixationCount = static_cast<int>(selectedFixations.size());
        decision.fixationShifts.clear();
        decision.fixationShifts.reserve(selectedFixations.size());
        for (size_t chosenIndex : selectedFixations) {
            decision.fixationShifts.push_back(
                {candidates[chosenIndex].shiftXPx, candidates[chosenIndex].shiftYPx});
        }

        if (decision.fixationCount >= minFixations &&
            computeDecisionUncertainty(decision) <= config.activeInferenceUncertaintyThreshold) {
            break;
        }
    }

    if (decision.hemisphereTraces.empty()) {
        decision = builder(inferHemisphereDecisions(hemispheres, image, config));
    }
    return decision;
}

BilateralDecisionTrace inferSingleViewCorpusCallosumDecision(
    std::vector<HemisphereRuntime>& hemispheres,
    const VisualStimulus& image,
    const Config& config) {
    return buildCorpusCallosumDecisionFromTraces(
        hemispheres, inferHemisphereDecisions(hemispheres, image, config), config);
}

BilateralDecisionTrace inferSingleViewCorpusCallosumDecisionWithAudit(
    std::vector<HemisphereRuntime>& hemispheres,
    const VisualStimulus& image,
    const Config& config,
    std::vector<FlowAuditSampleCapture>& flowAuditCaptures) {
    flowAuditCaptures.assign(hemispheres.size(), {});
    return buildCorpusCallosumDecisionFromTraces(
        hemispheres, inferHemisphereDecisions(hemispheres, image, config, &flowAuditCaptures), config);
}

BilateralDecisionTrace inferCorpusCallosumDecision(std::vector<HemisphereRuntime>& hemispheres,
                                                   const VisualStimulus& image,
                                                   const Config& config) {
    if (!config.activeInferenceEnabled || config.activeInferenceFixations <= 1) {
        return inferSingleViewCorpusCallosumDecision(hemispheres, image, config);
    }
    return inferActiveVisionDecision(
        hemispheres,
        image,
        config,
        [&](std::vector<HemisphereDecisionTrace> traces) {
            return buildCorpusCallosumDecisionFromTraces(hemispheres, std::move(traces), config);
        });
}

std::vector<double> buildCorpusCallosumConfidence(std::vector<HemisphereRuntime>& hemispheres,
                                                  const VisualStimulus& image,
                                                  const Config& config) {
    return inferCorpusCallosumDecision(hemispheres, image, config).combinedConfidence;
}

BilateralDecisionTrace buildFusionDecisionFromTraces(
    std::vector<HemisphereDecisionTrace> hemisphereTraces,
    const Config& config,
    const ClassificationStrategy& fusionClassifier,
    const FusionRuntime& fusionRuntime) {
    BilateralDecisionTrace trace;
    trace.hemisphereTraces = std::move(hemisphereTraces);
    trace.fusionPattern = buildFusionPatternFromTraces(trace.hemisphereTraces, config);
    trace.combinedConfidence = fusionClassifier.classifyWithConfidence(
        trace.fusionPattern, fusionRuntime.trainingPatterns, cosineSimilarity);
    normalizeSum(trace.combinedConfidence);
    if (useOnlineRewardStdp(config)) {
        const auto rewardStdpEvidence =
            computeRewardStdpEvidence(fusionRuntime.rewardStdpPrototypes, trace.fusionPattern);
        for (size_t label = 0;
             label < trace.combinedConfidence.size() && label < rewardStdpEvidence.size();
             ++label) {
            trace.combinedConfidence[label] +=
                onlineRewardStdpGain(config) * rewardStdpEvidence[label];
        }
        normalizeSum(trace.combinedConfidence);
    }
    if (useOnlineTripletStdp(config)) {
        const auto tripletStdpEvidence =
            computeTripletStdpEvidence(fusionRuntime.tripletStdpPrototypes, trace.fusionPattern);
        for (size_t label = 0;
             label < trace.combinedConfidence.size() && label < tripletStdpEvidence.size();
             ++label) {
            trace.combinedConfidence[label] +=
                onlineTripletStdpGain(config) * tripletStdpEvidence[label];
        }
        normalizeSum(trace.combinedConfidence);
    }
    if (useOnlineVoltagePlasticity(config)) {
        const auto voltagePlasticityEvidence = computeVoltagePlasticityEvidence(
            fusionRuntime.voltagePlasticityPrototypes, trace.fusionPattern);
        for (size_t label = 0;
             label < trace.combinedConfidence.size() &&
                 label < voltagePlasticityEvidence.size();
             ++label) {
            trace.combinedConfidence[label] +=
                onlineVoltagePlasticityGain(config) * voltagePlasticityEvidence[label];
        }
        normalizeSum(trace.combinedConfidence);
    }
    trace.topHypotheses = collectTopHypotheses(trace.combinedConfidence, onlineTraceTopK(config));
    if (!trace.topHypotheses.empty()) {
        trace.predicted = trace.topHypotheses.front().label;
    }
    return trace;
}

BilateralDecisionTrace inferSingleViewFusionDecision(std::vector<HemisphereRuntime>& hemispheres,
                                                     const VisualStimulus& image,
                                                     const Config& config,
                                                     const ClassificationStrategy& fusionClassifier,
                                                     const FusionRuntime& fusionRuntime) {
    return buildFusionDecisionFromTraces(
        inferHemisphereDecisions(hemispheres, image, config),
        config,
        fusionClassifier,
        fusionRuntime);
}

BilateralDecisionTrace inferSingleViewFusionDecisionWithAudit(
    std::vector<HemisphereRuntime>& hemispheres,
    const VisualStimulus& image,
    const Config& config,
    const ClassificationStrategy& fusionClassifier,
    const FusionRuntime& fusionRuntime,
    std::vector<FlowAuditSampleCapture>& flowAuditCaptures) {
    flowAuditCaptures.assign(hemispheres.size(), {});
    return buildFusionDecisionFromTraces(
        inferHemisphereDecisions(hemispheres, image, config, &flowAuditCaptures),
        config,
        fusionClassifier,
        fusionRuntime);
}

BilateralDecisionTrace inferFusionDecision(std::vector<HemisphereRuntime>& hemispheres,
                                           const VisualStimulus& image,
                                           const Config& config,
                                           const ClassificationStrategy& fusionClassifier,
                                           const FusionRuntime& fusionRuntime) {
    if (!config.activeInferenceEnabled || config.activeInferenceFixations <= 1) {
        return inferSingleViewFusionDecision(
            hemispheres, image, config, fusionClassifier, fusionRuntime);
    }
    return inferActiveVisionDecision(
        hemispheres,
        image,
        config,
        [&](std::vector<HemisphereDecisionTrace> traces) {
            return buildFusionDecisionFromTraces(
                std::move(traces), config, fusionClassifier, fusionRuntime);
        });
}

enum class FocusPreset {
    Original,
    Left,
    Center,
    Right,
};

bool isFocusPair(int labelA, int labelB, int first, int second) {
    return (labelA == first && labelB == second) || (labelA == second && labelB == first);
}

FocusPreset chooseFamilyAwareFocusPreset(const BilateralDecisionTrace& decision) {
    int leftPredicted = -1;
    int rightPredicted = -1;
    if (decision.hemisphereTraces.size() >= 2) {
        leftPredicted = decision.hemisphereTraces[0].predicted;
        rightPredicted = decision.hemisphereTraces[1].predicted;
    }

    if (leftPredicted < 0 || rightPredicted < 0 || leftPredicted == rightPredicted) {
        return FocusPreset::Original;
    }

    if (isFocusPair(leftPredicted, rightPredicted, 'C' - 'A', 'E' - 'A') ||
        isFocusPair(leftPredicted, rightPredicted, 'I' - 'A', 'L' - 'A') ||
        isFocusPair(leftPredicted, rightPredicted, 'X' - 'A', 'Y' - 'A')) {
        return FocusPreset::Left;
    }
    if (isFocusPair(leftPredicted, rightPredicted, 'G' - 'A', 'O' - 'A') ||
        isFocusPair(leftPredicted, rightPredicted, 'X' - 'A', 'Z' - 'A')) {
        return FocusPreset::Right;
    }

    return FocusPreset::Original;
}

double decisionQuality(const BilateralDecisionTrace& decision) {
    if (decision.topHypotheses.empty()) {
        return -1.0;
    }
    const double best = decision.topHypotheses.front().score;
    const double second = decision.topHypotheses.size() > 1 ? decision.topHypotheses[1].score : 0.0;
    return best + 0.5 * std::max(0.0, best - second);
}

bool shouldRunFocusAdjustment(const BilateralDecisionTrace& decision, const Config& config) {
    return config.focusAdjustmentEnabled &&
           chooseFamilyAwareFocusPreset(decision) != FocusPreset::Original;
}

void recordFocusSelection(EvaluationResult& result, FocusPreset preset) {
    switch (preset) {
        case FocusPreset::Original:
            result.focusOriginalSelections++;
            break;
        case FocusPreset::Left:
            result.focusLeftSelections++;
            break;
        case FocusPreset::Center:
            result.focusCenterSelections++;
            break;
        case FocusPreset::Right:
            result.focusRightSelections++;
            break;
    }
}

template <typename InferFn>
BilateralDecisionTrace maybeApplyFocusAdjustment(const VisualStimulus& image,
                                                 const BilateralDecisionTrace& baseDecision,
                                                 int truth,
                                                 const Config& config,
                                                 EvaluationResult& result,
                                                 InferFn inferFn) {
    if (!shouldRunFocusAdjustment(baseDecision, config)) {
        return baseDecision;
    }

    result.focusAdjustmentTriggers++;
    BilateralDecisionTrace bestDecision = baseDecision;
    FocusPreset bestPreset = FocusPreset::Original;
    double bestQuality = decisionQuality(baseDecision);

    const FocusPreset routedPreset = chooseFamilyAwareFocusPreset(baseDecision);
    double shiftXPx = 0.0;
    switch (routedPreset) {
        case FocusPreset::Left:
            shiftXPx = -std::abs(config.focusAdjustmentShiftPx);
            break;
        case FocusPreset::Right:
            shiftXPx = std::abs(config.focusAdjustmentShiftPx);
            break;
        case FocusPreset::Center:
        case FocusPreset::Original:
            recordFocusSelection(result, FocusPreset::Original);
            return baseDecision;
    }

    const auto focusedImage =
        applyImageFocusTransform(image, config.focusAdjustmentZoom, shiftXPx);
    auto candidate = inferFn(focusedImage);
    const double candidateQuality = decisionQuality(candidate);
    const double minimumGain = 0.01;
    if (candidateQuality > bestQuality + minimumGain) {
        bestQuality = candidateQuality;
        bestDecision = std::move(candidate);
        bestPreset = routedPreset;
    }

    recordFocusSelection(result, bestPreset);
    if (bestPreset == FocusPreset::Original) {
        return baseDecision;
    }

    if (bestDecision.predicted != baseDecision.predicted) {
        result.focusAdjustmentOverrides++;
    }
    if (baseDecision.predicted != truth && bestDecision.predicted == truth) {
        result.focusAdjustmentCorrections++;
    } else if (baseDecision.predicted == truth && bestDecision.predicted != truth) {
        result.focusAdjustmentRegressions++;
    }
    return bestDecision;
}

void applyRewardToHemisphere(HemisphereRuntime& hemisphere,
                             const HemisphereDecisionTrace& trace,
                             int rewardedLabel,
                             double reward,
                             const Config& config) {
    if (rewardedLabel < 0 ||
        static_cast<size_t>(rewardedLabel) >= hemisphere.classWeights.size() ||
        trace.topHypotheses.empty()) {
        return;
    }

    const double rewardMagnitude = std::clamp(std::abs(reward), 0.0, 3.0);
    const LabelTrace* rewardedTrace = findLabelTrace(trace.topHypotheses, rewardedLabel);

    if (reward > 0.0) {
        if (rewardedTrace == nullptr) {
            return;
        }
        const auto& classifierPattern =
            trace.classifierPattern.empty() ? trace.pattern : trace.classifierPattern;
        const double rewardGain =
            std::clamp(config.onlinePositiveRewardGain * rewardMagnitude, 0.05, 2.5);
        const double support = std::max(0.05, rewardedTrace->score);
        const double baseLr =
            std::clamp(config.onlineCentroidLr * 0.35 * rewardGain, 0.01, 0.25);
        scaleClamped(hemisphere.classWeights[static_cast<size_t>(rewardedLabel)],
                     1.0 + baseLr * support,
                     0.5, 1.5);
        scaleClamped(hemisphere.overallWeight,
                     1.0 + baseLr * support * 0.4,
                     0.75, 1.25);
        if (trace.predicted == rewardedLabel) {
            scaleClamped(hemisphere.predictionWeights[static_cast<size_t>(rewardedLabel)],
                         1.0 + baseLr * support * (1.0 + trace.margin),
                         0.5, 1.5);
        }
        updateCentroid(hemisphere.classCentroids[static_cast<size_t>(rewardedLabel)],
                       classifierPattern,
                       std::clamp(config.onlineCentroidLr * support * rewardGain, 0.02, 0.50));
        insertOrReplaceOnlinePattern(hemisphere.trainingPatterns,
                                     &hemisphere.trainingSourceIndices,
                                     hemisphere.onlinePatternIndices,
                                     classifierPattern,
                                     rewardedLabel,
                                     config.onlineExemplarBudgetPerClass);
        rebuildHemisphereObjectStateAttractors(hemisphere, config);
        return;
    }

    const LabelTrace* penalizedTrace = findLabelTrace(trace.topHypotheses, rewardedLabel);
    if (penalizedTrace == nullptr) {
        return;
    }
    const double rewardGain =
        std::clamp(config.onlineNegativeRewardGain * rewardMagnitude, 0.05, 2.5);
    const double support = std::max(0.05, penalizedTrace->score);
    const double baseLr =
        std::clamp(config.onlineCentroidLr * 0.35 * rewardGain, 0.01, 0.25);
    const double penalty = std::clamp(baseLr * support * (1.0 + trace.margin), 0.02, 0.24);
    scaleClamped(hemisphere.classWeights[static_cast<size_t>(rewardedLabel)],
                 1.0 - penalty,
                 0.5, 1.5);
    scaleClamped(hemisphere.overallWeight,
                 1.0 - penalty * 0.35,
                 0.75, 1.25);
    if (trace.predicted == rewardedLabel) {
        scaleClamped(hemisphere.predictionWeights[static_cast<size_t>(rewardedLabel)],
                     1.0 - penalty * 1.15,
                     0.5, 1.5);
    }
}

void applyRewardStdpToHemisphere(HemisphereRuntime& hemisphere,
                                 const HemisphereDecisionTrace& trace,
                                 int rewardedLabel,
                                 double reward,
                                 const Config& config) {
    if (!useOnlineRewardStdp(config) ||
        rewardedLabel < 0 ||
        static_cast<size_t>(rewardedLabel) >= hemisphere.rewardStdpPrototypes.size()) {
        return;
    }

    const auto& classifierPattern =
        trace.classifierPattern.empty() ? trace.pattern : trace.classifierPattern;
    if (classifierPattern.empty()) {
        return;
    }

    const LabelTrace* labelTrace = findLabelTrace(trace.topHypotheses, rewardedLabel);
    if (labelTrace == nullptr) {
        return;
    }

    const double rewardMagnitude = std::clamp(std::abs(reward), 0.0, 3.0);
    const double support = std::max(0.05, labelTrace->score);
    if (reward > 0.0) {
        updateRewardStdpPrototype(
            hemisphere.rewardStdpPrototypes[static_cast<size_t>(rewardedLabel)],
            classifierPattern,
            std::clamp(onlineRewardStdpLtp(config) * rewardMagnitude * support, 0.02, 0.35),
            true);
        return;
    }

    updateRewardStdpPrototype(
        hemisphere.rewardStdpPrototypes[static_cast<size_t>(rewardedLabel)],
        classifierPattern,
        std::clamp(onlineRewardStdpLtd(config) * rewardMagnitude * support * (1.0 + trace.margin),
                   0.02,
                   0.35),
        false);
}

void applyTripletStdpToHemisphere(HemisphereRuntime& hemisphere,
                                  const HemisphereDecisionTrace& trace,
                                  int rewardedLabel,
                                  double reward,
                                  size_t currentStep,
                                  const Config& config) {
    if (!useOnlineTripletStdp(config) ||
        rewardedLabel < 0 ||
        static_cast<size_t>(rewardedLabel) >= hemisphere.tripletStdpPrototypes.size()) {
        return;
    }

    const auto& classifierPattern =
        trace.classifierPattern.empty() ? trace.pattern : trace.classifierPattern;
    if (classifierPattern.empty()) {
        return;
    }

    const LabelTrace* labelTrace = findLabelTrace(trace.topHypotheses, rewardedLabel);
    if (labelTrace == nullptr) {
        return;
    }

    decayTripletStdpState(hemisphere, currentStep, config);
    const double rewardMagnitude = std::clamp(std::abs(reward), 0.0, 3.0);
    const double support = std::max(0.05, labelTrace->score);
    const double fastTrace =
        hemisphere.tripletPostTraceFast[static_cast<size_t>(rewardedLabel)];
    const double slowTrace =
        hemisphere.tripletPostTraceSlow[static_cast<size_t>(rewardedLabel)];
    const auto tripletPattern =
        buildTripletStdpPattern(classifierPattern, hemisphere.tripletPreTrace);

    if (reward > 0.0) {
        updateRewardStdpPrototype(
            hemisphere.tripletStdpPrototypes[static_cast<size_t>(rewardedLabel)],
            tripletPattern,
            std::clamp(onlineTripletStdpLtp(config) * rewardMagnitude * support *
                           (1.0 + 0.30 * fastTrace + 0.55 * slowTrace),
                       0.02,
                       0.35),
            true);
    } else {
        updateRewardStdpPrototype(
            hemisphere.tripletStdpPrototypes[static_cast<size_t>(rewardedLabel)],
            classifierPattern,
            std::clamp(onlineTripletStdpLtd(config) * rewardMagnitude * support *
                           (1.0 + trace.margin + 0.25 * fastTrace),
                       0.02,
                       0.35),
            false);
    }

    updateTripletStdpTraces(hemisphere, classifierPattern, rewardedLabel, support);
}

void applyVoltagePlasticityToHemisphere(HemisphereRuntime& hemisphere,
                                        const HemisphereDecisionTrace& trace,
                                        int rewardedLabel,
                                        double reward,
                                        size_t currentStep,
                                        const Config& config) {
    if (!useOnlineVoltagePlasticity(config) ||
        rewardedLabel < 0 ||
        static_cast<size_t>(rewardedLabel) >= hemisphere.voltagePlasticityPrototypes.size()) {
        return;
    }

    const auto& classifierPattern =
        trace.classifierPattern.empty() ? trace.pattern : trace.classifierPattern;
    if (classifierPattern.empty()) {
        return;
    }

    const LabelTrace* labelTrace = findLabelTrace(trace.topHypotheses, rewardedLabel);
    if (labelTrace == nullptr) {
        return;
    }

    decayVoltagePlasticityState(hemisphere, currentStep, config);
    const double rewardMagnitude = std::clamp(std::abs(reward), 0.0, 3.0);
    const double support = std::max(0.05, labelTrace->score);
    const double persistentTrace =
        hemisphere.voltageDepolarizationTrace[static_cast<size_t>(rewardedLabel)];
    auto& prototype =
        hemisphere.voltagePlasticityPrototypes[static_cast<size_t>(rewardedLabel)];
    const double depolarization = computeVoltagePlasticityDepolarization(
        prototype, classifierPattern, support, persistentTrace);
    const double threshold = onlineVoltagePlasticityThreshold(config);
    const double supraThreshold = std::max(0.0, depolarization - threshold);

    if (reward > 0.0) {
        if (supraThreshold <= 1e-6) {
            updateVoltagePlasticityTrace(hemisphere, rewardedLabel, depolarization * 0.5);
            return;
        }
        updateRewardStdpPrototype(
            prototype,
            classifierPattern,
            std::clamp(onlineVoltagePlasticityLtp(config) * rewardMagnitude *
                           std::max(0.05, supraThreshold) * (1.0 + 0.35 * trace.margin),
                       0.02,
                       0.35),
            true);
        updateVoltagePlasticityTrace(hemisphere, rewardedLabel, depolarization);
        return;
    }

    updateRewardStdpPrototype(
        prototype,
        classifierPattern,
        std::clamp(onlineVoltagePlasticityLtd(config) * rewardMagnitude *
                       std::max(0.05, depolarization + 0.35 * trace.margin +
                                         0.25 * supraThreshold),
                   0.02,
                   0.35),
        false);
    updateVoltagePlasticityTrace(hemisphere, rewardedLabel, depolarization);
}

void applyRewardToFusion(FusionRuntime& fusionRuntime,
                         const std::vector<double>& fusionPattern,
                         const std::vector<LabelTrace>& topHypotheses,
                         int rewardedLabel,
                         double reward,
                         const Config& config) {
    if (reward <= 0.0 ||
        reward * config.onlinePositiveRewardGain < 0.05 ||
        rewardedLabel < 0 ||
        fusionPattern.empty()) {
        return;
    }

    insertOrReplaceOnlinePattern(fusionRuntime.trainingPatterns,
                                 nullptr,
                                 fusionRuntime.onlinePatternIndices,
                                 fusionPattern,
                                 rewardedLabel,
                                 config.onlineExemplarBudgetPerClass);
}

void applyRewardStdpToFusion(FusionRuntime& fusionRuntime,
                             const std::vector<double>& fusionPattern,
                             const std::vector<LabelTrace>& topHypotheses,
                             int rewardedLabel,
                             double reward,
                             const Config& config) {
    if (!useOnlineRewardStdp(config) ||
        rewardedLabel < 0 ||
        static_cast<size_t>(rewardedLabel) >= fusionRuntime.rewardStdpPrototypes.size() ||
        fusionPattern.empty()) {
        return;
    }

    const LabelTrace* labelTrace = findLabelTrace(topHypotheses, rewardedLabel);
    if (labelTrace == nullptr) {
        return;
    }

    const double rewardMagnitude = std::clamp(std::abs(reward), 0.0, 3.0);
    const double support = std::max(0.05, labelTrace->score);
    if (reward > 0.0) {
        updateRewardStdpPrototype(
            fusionRuntime.rewardStdpPrototypes[static_cast<size_t>(rewardedLabel)],
            fusionPattern,
            std::clamp(onlineRewardStdpLtp(config) * rewardMagnitude * support, 0.02, 0.35),
            true);
        return;
    }

    updateRewardStdpPrototype(
        fusionRuntime.rewardStdpPrototypes[static_cast<size_t>(rewardedLabel)],
        fusionPattern,
        std::clamp(onlineRewardStdpLtd(config) * rewardMagnitude * support, 0.02, 0.35),
        false);
}

void applyTripletStdpToFusion(FusionRuntime& fusionRuntime,
                              const std::vector<double>& fusionPattern,
                              const std::vector<LabelTrace>& topHypotheses,
                              int rewardedLabel,
                              double reward,
                              size_t currentStep,
                              const Config& config) {
    if (!useOnlineTripletStdp(config) ||
        rewardedLabel < 0 ||
        static_cast<size_t>(rewardedLabel) >= fusionRuntime.tripletStdpPrototypes.size() ||
        fusionPattern.empty()) {
        return;
    }

    const LabelTrace* labelTrace = findLabelTrace(topHypotheses, rewardedLabel);
    if (labelTrace == nullptr) {
        return;
    }

    decayTripletStdpState(fusionRuntime, currentStep, config);
    const double rewardMagnitude = std::clamp(std::abs(reward), 0.0, 3.0);
    const double support = std::max(0.05, labelTrace->score);
    const double fastTrace =
        fusionRuntime.tripletPostTraceFast[static_cast<size_t>(rewardedLabel)];
    const double slowTrace =
        fusionRuntime.tripletPostTraceSlow[static_cast<size_t>(rewardedLabel)];
    const auto tripletPattern =
        buildTripletStdpPattern(fusionPattern, fusionRuntime.tripletPreTrace);

    if (reward > 0.0) {
        updateRewardStdpPrototype(
            fusionRuntime.tripletStdpPrototypes[static_cast<size_t>(rewardedLabel)],
            tripletPattern,
            std::clamp(onlineTripletStdpLtp(config) * rewardMagnitude * support *
                           (1.0 + 0.30 * fastTrace + 0.55 * slowTrace),
                       0.02,
                       0.35),
            true);
    } else {
        updateRewardStdpPrototype(
            fusionRuntime.tripletStdpPrototypes[static_cast<size_t>(rewardedLabel)],
            fusionPattern,
            std::clamp(onlineTripletStdpLtd(config) * rewardMagnitude * support *
                           (1.0 + 0.25 * fastTrace),
                       0.02,
                       0.35),
            false);
    }

    updateTripletStdpTraces(fusionRuntime, fusionPattern, rewardedLabel, support);
}

void applyVoltagePlasticityToFusion(FusionRuntime& fusionRuntime,
                                    const std::vector<double>& fusionPattern,
                                    const std::vector<LabelTrace>& topHypotheses,
                                    int rewardedLabel,
                                    double reward,
                                    size_t currentStep,
                                    const Config& config) {
    if (!useOnlineVoltagePlasticity(config) ||
        rewardedLabel < 0 ||
        static_cast<size_t>(rewardedLabel) >= fusionRuntime.voltagePlasticityPrototypes.size() ||
        fusionPattern.empty()) {
        return;
    }

    const LabelTrace* labelTrace = findLabelTrace(topHypotheses, rewardedLabel);
    if (labelTrace == nullptr) {
        return;
    }

    decayVoltagePlasticityState(fusionRuntime, currentStep, config);
    const double rewardMagnitude = std::clamp(std::abs(reward), 0.0, 3.0);
    const double support = std::max(0.05, labelTrace->score);
    const double persistentTrace =
        fusionRuntime.voltageDepolarizationTrace[static_cast<size_t>(rewardedLabel)];
    auto& prototype =
        fusionRuntime.voltagePlasticityPrototypes[static_cast<size_t>(rewardedLabel)];
    const double depolarization = computeVoltagePlasticityDepolarization(
        prototype, fusionPattern, support, persistentTrace);
    const double threshold = onlineVoltagePlasticityThreshold(config);
    const double supraThreshold = std::max(0.0, depolarization - threshold);

    if (reward > 0.0) {
        if (supraThreshold <= 1e-6) {
            updateVoltagePlasticityTrace(fusionRuntime, rewardedLabel, depolarization * 0.5);
            return;
        }
        updateRewardStdpPrototype(
            prototype,
            fusionPattern,
            std::clamp(onlineVoltagePlasticityLtp(config) * rewardMagnitude *
                           std::max(0.05, supraThreshold),
                       0.02,
                       0.35),
            true);
        updateVoltagePlasticityTrace(fusionRuntime, rewardedLabel, depolarization);
        return;
    }

    updateRewardStdpPrototype(
        prototype,
        fusionPattern,
        std::clamp(onlineVoltagePlasticityLtd(config) * rewardMagnitude *
                       std::max(0.05, depolarization + 0.25 * supraThreshold),
                   0.02,
                   0.35),
        false);
    updateVoltagePlasticityTrace(fusionRuntime, rewardedLabel, depolarization);
}

EvaluationResult evaluateCorpusCallosumPatterns(std::vector<HemisphereRuntime>& hemispheres,
                                                const VisualDomainAdapter& loader,
                                                const std::vector<size_t>& indices,
                                                const Config& config,
                                                SeparabilityDiagnosticsRuntime* separabilityRuntime,
                                                FlowAuditRuntime* flowAuditRuntime,
                                                const std::string& label) {
    EvaluationResult result = makeEvaluationResult(config);
    if (separabilityRuntime != nullptr) {
        initializeSeparabilityStats(result, *separabilityRuntime, hemispheres);
    }
    const auto start = std::chrono::high_resolution_clock::now();
    const int maxTests = static_cast<int>(indices.size());
    const int progressStep = maxTests >= 1000 ? 200 : (maxTests >= 200 ? 50 : 25);
    std::vector<OnlineSampleRecord> records;
    records.reserve(indices.size());
    std::vector<ReplayItem> replayQueue;
    replayQueue.reserve(static_cast<size_t>(std::max(16, config.onlineReplayQueueCapacity)));
    std::vector<ReplayItem> consolidationQueue;
    consolidationQueue.reserve(static_cast<size_t>(
        std::max(8, config.onlineReplaySuccessConsolidationCapacity)));
    size_t replaySequence = 0;
    ConfusionClusterMemory confusionMemory = makeConfusionClusterMemory(config);

    auto enqueueSuccessfulConsolidation = [&](size_t recordIndex,
                                              const DecisionContext& context,
                                              size_t currentStep) {
        if (!useOnlineReplaySuccessConsolidation(config)) {
            return;
        }
        enqueueReplayItemWithCapacity(
            consolidationQueue,
            makeReplayItem(recordIndex,
                           config.onlineReplaySuccessConsolidationRepeats,
                           context.replayPriority + 0.25 * context.plasticity,
                           currentStep,
                           computeNeuromodulatorEligibilityScale(context, true, config),
                           config,
                           replaySequence++),
            config.onlineReplaySuccessConsolidationCapacity);
    };

    auto processReplayItem = [&](ReplayItem item, size_t currentStep) {
        if (item.recordIndex >= records.size()) {
            return;
        }
        auto& record = records[item.recordIndex];
        const auto& replayImage = loader.getStimulus(record.imageIndex);
        auto replayDecision = inferCorpusCallosumDecision(hemispheres, replayImage, config);
        if (replayDecision.predicted < 0 || replayDecision.predicted >= config.numClasses) {
            return;
        }

        record.finalPredicted = replayDecision.predicted;
        result.correctionReplays++;
        recordReplayTiming(result, item, currentStep);
        const auto replayContext = buildDecisionContext(replayDecision, config, &confusionMemory);
        result.confusionClusterScoreSum += replayContext.confusionCluster;
        result.confusionClusterScoreSamples++;
        const double effectiveReward =
            std::max(0.05, replayContext.plasticity * item.eligibilityScale);

        if (record.finalPredicted == record.truth) {
            if (flowAuditRuntime != nullptr) {
                recordFlowAuditReplayMechanisms(*flowAuditRuntime, replayDecision, true);
            }
            if (!record.correctionSucceeded) {
                result.correctionSuccesses++;
                record.correctionSucceeded = true;
            }
            for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
                applyRewardToHemisphere(hemispheres[hemisphereIndex],
                                        replayDecision.hemisphereTraces[hemisphereIndex],
                                        replayDecision.predicted,
                                        effectiveReward,
                                        config);
                applyRewardStdpToHemisphere(hemispheres[hemisphereIndex],
                                            replayDecision.hemisphereTraces[hemisphereIndex],
                                            replayDecision.predicted,
                                            effectiveReward,
                                            config);
                applyTripletStdpToHemisphere(hemispheres[hemisphereIndex],
                                             replayDecision.hemisphereTraces[hemisphereIndex],
                                             replayDecision.predicted,
                                             effectiveReward,
                                             currentStep,
                                             config);
                applyVoltagePlasticityToHemisphere(
                    hemispheres[hemisphereIndex],
                    replayDecision.hemisphereTraces[hemisphereIndex],
                    replayDecision.predicted,
                    effectiveReward,
                    currentStep,
                    config);
            }
            updateConfusionClusterMemory(confusionMemory, replayDecision, false, config);
            enqueueSuccessfulConsolidation(item.recordIndex, replayContext, currentStep);
            return;
        }

        if (flowAuditRuntime != nullptr) {
            recordFlowAuditReplayMechanisms(*flowAuditRuntime, replayDecision, false);
        }

        for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
            applyRewardToHemisphere(hemispheres[hemisphereIndex],
                                    replayDecision.hemisphereTraces[hemisphereIndex],
                                    replayDecision.predicted,
                                    -effectiveReward,
                                    config);
            applyRewardStdpToHemisphere(hemispheres[hemisphereIndex],
                                        replayDecision.hemisphereTraces[hemisphereIndex],
                                        replayDecision.predicted,
                                        -effectiveReward,
                                        config);
            applyTripletStdpToHemisphere(hemispheres[hemisphereIndex],
                                         replayDecision.hemisphereTraces[hemisphereIndex],
                                         replayDecision.predicted,
                                         -effectiveReward,
                                         currentStep,
                                         config);
            applyVoltagePlasticityToHemisphere(
                hemispheres[hemisphereIndex],
                replayDecision.hemisphereTraces[hemisphereIndex],
                replayDecision.predicted,
                -effectiveReward,
                currentStep,
                config);
        }
        updateConfusionClusterMemory(confusionMemory, replayDecision, true, config);
        if (item.remainingReplays > 1) {
            const double replayEligibilityScale = std::max(
                item.eligibilityScale,
                computeNeuromodulatorEligibilityScale(replayContext, false, config));
            enqueueReplayItem(replayQueue,
                              makeReplayItem(item.recordIndex,
                                             item.remainingReplays - 1,
                                             replayContext.replayPriority,
                                             currentStep,
                                             replayEligibilityScale,
                                             config,
                                             replaySequence++),
                              config);
        }
    };

    auto processConsolidationItem = [&](ReplayItem item, size_t currentStep) {
        if (item.recordIndex >= records.size()) {
            return;
        }
        auto& record = records[item.recordIndex];
        const auto& replayImage = loader.getStimulus(record.imageIndex);
        auto replayDecision = inferCorpusCallosumDecision(hemispheres, replayImage, config);
        if (replayDecision.predicted < 0 || replayDecision.predicted >= config.numClasses ||
            replayDecision.predicted != record.truth) {
            return;
        }

        result.correctionReplays++;
        recordReplayTiming(result, item, currentStep);
        const auto replayContext = buildDecisionContext(replayDecision, config, &confusionMemory);
        result.confusionClusterScoreSum += replayContext.confusionCluster;
        result.confusionClusterScoreSamples++;
        const double effectiveReward =
            std::max(0.05, replayContext.plasticity * item.eligibilityScale);

        if (flowAuditRuntime != nullptr) {
            recordFlowAuditReplayMechanisms(*flowAuditRuntime, replayDecision, true);
        }
        for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
            applyRewardToHemisphere(hemispheres[hemisphereIndex],
                                    replayDecision.hemisphereTraces[hemisphereIndex],
                                    replayDecision.predicted,
                                    effectiveReward,
                                    config);
            applyRewardStdpToHemisphere(hemispheres[hemisphereIndex],
                                        replayDecision.hemisphereTraces[hemisphereIndex],
                                        replayDecision.predicted,
                                        effectiveReward,
                                        config);
            applyTripletStdpToHemisphere(hemispheres[hemisphereIndex],
                                         replayDecision.hemisphereTraces[hemisphereIndex],
                                         replayDecision.predicted,
                                         effectiveReward,
                                         currentStep,
                                         config);
            applyVoltagePlasticityToHemisphere(
                hemispheres[hemisphereIndex],
                replayDecision.hemisphereTraces[hemisphereIndex],
                replayDecision.predicted,
                effectiveReward,
                currentStep,
                config);
        }
        updateConfusionClusterMemory(confusionMemory, replayDecision, false, config);
        if (item.remainingReplays > 1) {
            enqueueReplayItemWithCapacity(
                consolidationQueue,
                makeReplayItem(item.recordIndex,
                               item.remainingReplays - 1,
                               replayContext.replayPriority + 0.25 * replayContext.plasticity,
                               currentStep,
                               item.eligibilityScale,
                               config,
                               replaySequence++),
                config.onlineReplaySuccessConsolidationCapacity);
        }
    };

    auto processReplayBudget = [&](size_t currentStep, int budget, bool flushAll = false) {
        if (!flushAll &&
            (currentStep == 0 || (currentStep % static_cast<size_t>(onlineReplayPauseInterval(config))) != 0)) {
            return;
        }
        const int batchBudget = flushAll ? budget : std::min(budget, onlineReplayBatchSize(config));
        ReplayItem item;
        int remaining = batchBudget;
        while (remaining-- > 0 && popReplayItem(replayQueue, item, currentStep, flushAll)) {
            processReplayItem(item, currentStep);
        }
    };

    auto processConsolidationBudget = [&](size_t currentStep, int budget, bool flushAll = false) {
        if (!useOnlineReplaySuccessConsolidation(config)) {
            return;
        }
        if (!flushAll &&
            (currentStep == 0 || (currentStep % static_cast<size_t>(onlineReplayPauseInterval(config))) != 0)) {
            return;
        }
        const int batchBudget = flushAll
            ? budget
            : std::min(budget, onlineReplaySuccessConsolidationBatchSize(config));
        ReplayItem item;
        int remaining = batchBudget;
        while (remaining-- > 0 && popReplayItem(consolidationQueue, item, currentStep, flushAll)) {
            processConsolidationItem(item, currentStep);
        }
    };

    auto currentAccuracy = [&]() {
        const int currentCorrect = static_cast<int>(std::count_if(
            records.begin(), records.end(), [](const OnlineSampleRecord& record) {
                return record.truth >= 0 && record.truth == record.finalPredicted;
            }));
        return 100.0 * static_cast<double>(currentCorrect) /
               static_cast<double>(std::max<size_t>(1, records.size()));
    };

    for (size_t index : indices) {
        const auto& image = loader.getStimulus(index);
        const int truth = image.label;
        if (truth < 0 || truth >= config.numClasses) {
            continue;
        }

        std::vector<FlowAuditSampleCapture> flowAuditCaptures;
        const bool captureBranchAudit =
            flowAuditRuntime != nullptr &&
            (!config.activeInferenceEnabled || config.activeInferenceFixations <= 1) &&
            !config.focusAdjustmentEnabled;
        auto decision = captureBranchAudit
            ? inferSingleViewCorpusCallosumDecisionWithAudit(
                  hemispheres, image, config, flowAuditCaptures)
            : inferCorpusCallosumDecision(hemispheres, image, config);
        const int initialPredicted = decision.predicted;
        if (initialPredicted < 0 || initialPredicted >= config.numClasses) {
            continue;
        }
        const int leftInitialPredicted =
            decision.hemisphereTraces.size() > 0 ? decision.hemisphereTraces[0].predicted : -1;
        const int rightInitialPredicted =
            decision.hemisphereTraces.size() > 1 ? decision.hemisphereTraces[1].predicted : -1;
        recordHemisphereAgreement(result, decision, truth);
        if (separabilityRuntime != nullptr) {
            recordSeparabilityDiagnostics(
                result, *separabilityRuntime, hemispheres, decision, truth);
        }
        decision = maybeApplyFocusAdjustment(
            image,
            decision,
            truth,
            config,
            result,
            [&](const VisualStimulus& focusedImage) {
                return inferCorpusCallosumDecision(hemispheres, focusedImage, config);
            });
        recordActiveInferenceUsage(result, decision, config);
        const size_t sampleOrdinal = records.size() + 1;
        const bool storeFlowAuditRows =
            flowAuditRuntime != nullptr ? flowAuditShouldStoreRows(*flowAuditRuntime) : false;
        if (flowAuditRuntime != nullptr) {
            recordFlowAuditFusionAlternatives(*flowAuditRuntime, decision, truth);
            if (captureBranchAudit && flowAuditCaptures.size() == hemispheres.size()) {
                for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
                    recordFlowAuditBranchCapture(*flowAuditRuntime,
                                                 sampleOrdinal,
                                                 index,
                                                 truth,
                                                 initialPredicted,
                                                 decision.predicted,
                                                 hemispheres[hemisphereIndex].name,
                                                 decision.hemisphereTraces[hemisphereIndex],
                                                 flowAuditCaptures[hemisphereIndex],
                                                 hemisphereIndex,
                                                 storeFlowAuditRows);
                    recordFlowAuditHemisphereCapture(*flowAuditRuntime,
                                                     sampleOrdinal,
                                                     index,
                                                     truth,
                                                     initialPredicted,
                                                     decision.predicted,
                                                     hemispheres[hemisphereIndex].name,
                                                     decision.hemisphereTraces[hemisphereIndex],
                                                     &flowAuditCaptures[hemisphereIndex],
                                                     hemispheres[hemisphereIndex],
                                                     hemisphereIndex,
                                                     config,
                                                     storeFlowAuditRows);
                }
            }
            recordFlowAuditFusionRow(*flowAuditRuntime,
                                     sampleOrdinal,
                                     index,
                                     truth,
                                     initialPredicted,
                                     decision.predicted,
                                     decision,
                                     storeFlowAuditRows);
        }
        records.push_back(
            {index, truth, leftInitialPredicted, rightInitialPredicted, initialPredicted,
             initialPredicted, false});
        auto& record = records.back();
        record.finalPredicted = decision.predicted;
        const auto context = buildDecisionContext(decision, config, &confusionMemory);
        result.confusionClusterScoreSum += context.confusionCluster;
        result.confusionClusterScoreSamples++;

        if (useOnlineCorrection(config)) {
            if (initialPredicted == truth) {
                for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
                    applyRewardToHemisphere(hemispheres[hemisphereIndex],
                                            decision.hemisphereTraces[hemisphereIndex],
                                            decision.predicted,
                                            std::max(0.25, 0.5 * context.plasticity),
                                            config);
                    applyRewardStdpToHemisphere(hemispheres[hemisphereIndex],
                                                decision.hemisphereTraces[hemisphereIndex],
                                                decision.predicted,
                                                std::max(0.25, 0.5 * context.plasticity),
                                                config);
                    applyTripletStdpToHemisphere(hemispheres[hemisphereIndex],
                                                 decision.hemisphereTraces[hemisphereIndex],
                                                 decision.predicted,
                                                 std::max(0.25, 0.5 * context.plasticity),
                                                 records.size(),
                                                 config);
                    applyVoltagePlasticityToHemisphere(
                        hemispheres[hemisphereIndex],
                        decision.hemisphereTraces[hemisphereIndex],
                        decision.predicted,
                        std::max(0.25, 0.5 * context.plasticity),
                                                records.size(),
                                                config);
                }
                enqueueSuccessfulConsolidation(records.size() - 1, context, records.size());
            } else {
                result.correctionEvents++;
                for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
                    applyRewardToHemisphere(hemispheres[hemisphereIndex],
                                            decision.hemisphereTraces[hemisphereIndex],
                                            decision.predicted,
                                            -context.plasticity,
                                            config);
                    applyRewardStdpToHemisphere(hemispheres[hemisphereIndex],
                                                decision.hemisphereTraces[hemisphereIndex],
                                                decision.predicted,
                                                -context.plasticity,
                                                config);
                    applyTripletStdpToHemisphere(hemispheres[hemisphereIndex],
                                                 decision.hemisphereTraces[hemisphereIndex],
                                                 decision.predicted,
                                                 -context.plasticity,
                                                 records.size(),
                                                 config);
                    applyVoltagePlasticityToHemisphere(
                        hemispheres[hemisphereIndex],
                        decision.hemisphereTraces[hemisphereIndex],
                        decision.predicted,
                        -context.plasticity,
                        records.size(),
                        config);
                }
                enqueueReplayItem(replayQueue,
                                  makeReplayItem(records.size() - 1,
                                                 config.onlineCorrectionRepeats,
                                                 context.replayPriority +
                                                     (context.uncertainty >=
                                                              config.onlineReplayUncertaintyThreshold
                                                          ? 0.5
                                                          : 0.0),
                                                 records.size(),
                                                 computeNeuromodulatorEligibilityScale(
                                                     context, false, config),
                                                 config,
                                                 replaySequence++),
                                  config);
            }
        }
        updateConfusionClusterMemory(confusionMemory, decision, decision.predicted != truth, config);

        processReplayBudget(records.size(), std::numeric_limits<int>::max());
        processConsolidationBudget(records.size(), std::numeric_limits<int>::max());

        if (static_cast<int>(records.size()) % progressStep == 0) {
            const double acc = currentAccuracy();
            std::cout << "  " << label << ": " << records.size() << "/" << maxTests
                      << " (" << std::fixed << std::setprecision(2) << acc << "%)" << std::endl;
        }
    }

    processReplayBudget(records.size() + static_cast<size_t>(onlineReplayDelaySteps(config)),
                        std::numeric_limits<int>::max(),
                        true);
    processConsolidationBudget(records.size() +
                                   static_cast<size_t>(onlineReplayDelaySteps(config)),
                               std::numeric_limits<int>::max(),
                               true);

    const auto end = std::chrono::high_resolution_clock::now();
    finalizeEvaluationFromRecords(
        result, records, std::chrono::duration<double>(end - start).count());
    result.confusionClusterStrengths = confusionMemory.strengths;
    return result;
}

std::vector<double> extractRetinaPartRawPattern(RetinaAdapter& retina,
                                                const VisualStimulus& image,
                                                bool useFeatures,
                                                bool learnPatterns) {
    snnfw::adapters::SensoryAdapter::DataSample sample;
    sample.rawData = image.pixels;
    sample.timestamp = 0.0;
    sample.rows = image.rows;
    sample.cols = image.cols;
    sample.channels = std::max(1, image.channels);

    std::vector<double> part;
    if (useFeatures) {
        auto features = retina.extractFeatures(sample);
        part = std::move(features.features);
    } else {
        retina.processData(sample);
        if (learnPatterns) {
            for (const auto& neuron : retina.getNeurons()) {
                neuron->learnCurrentPattern();
            }
        }
        part = retina.getActivationPattern();
    }
    retina.clearNeuronStates();
    return part;
}

std::vector<double> finalizeRetinaPartPattern(const RetinaAdapter& retina,
                                              std::vector<double> part) {
    part = applyContextualGrouping(retina, std::move(part));
    normalizeL2(part);
    const double fusionWeight = std::max(0.0, retina.getDoubleParam("fusion_weight", 1.0));
    if (fusionWeight != 1.0) {
        for (double& value : part) {
            value *= fusionWeight;
        }
    }
    return part;
}

bool usePerFixationStage1Memory(const RetinaAdapter& retina,
                                const Config& config,
                                bool trainingPhase) {
    if (!trainingPhase || config.saccadeFixations <= 1) {
        return false;
    }
    return toLower(retina.getStringParam("stage1_fixation_memory_mode", "mean")) ==
           "per_fixation_exemplar";
}

bool useMeanMaxFixationSummary(const RetinaAdapter& retina) {
    return toLower(retina.getStringParam("stage1_fixation_memory_mode", "mean")) ==
           "mean_max_summary";
}

std::vector<double> extractPattern(std::vector<std::unique_ptr<RetinaAdapter>>& retinas,
                                   const VisualStimulus& image,
                                   const Config& config,
                                   bool trainingPhase,
                                   bool learnPatterns,
                                   size_t sampleSeed,
                                   FlowAuditSampleCapture* flowAuditCapture) {
    std::vector<double> combined;
    const auto fixationImages =
        buildExtractionStimuli(image, config, trainingPhase, sampleSeed, &retinas);
    if (flowAuditCapture != nullptr) {
        flowAuditCapture->parts.clear();
        flowAuditCapture->parts.reserve(retinas.size());
        flowAuditCapture->fixationPatterns.assign(fixationImages.size(), {});
        flowAuditCapture->fixationConsistency =
            fixationImages.empty() ? 0.0 : 1.0;
    }

    for (auto& retina : retinas) {
        std::vector<double> part;
        std::vector<std::vector<double>> fixationParts;
        fixationParts.reserve(fixationImages.size());
        for (const auto& fixationImage : fixationImages) {
            auto rawPart = extractRetinaPartRawPattern(
                *retina, fixationImage, config.useFeatures, learnPatterns);
            fixationParts.push_back(std::move(rawPart));
        }
        std::vector<std::vector<double>> finalizedFixationParts;
        if (flowAuditCapture != nullptr) {
            finalizedFixationParts.reserve(fixationParts.size());
            for (const auto& rawPart : fixationParts) {
                finalizedFixationParts.push_back(
                    finalizeRetinaPartPattern(*retina, rawPart));
            }
        }
        const bool meanMaxSummary = useMeanMaxFixationSummary(*retina);
        std::vector<double> preNormPart;
        std::vector<double> finalizedPart;
        double preOrientationL2 = 0.0;
        double preAuxiliaryL2 = 0.0;
        double postOrientationL2 = 0.0;
        double postAuxiliaryL2 = 0.0;

        if (meanMaxSummary && !fixationParts.empty()) {
            std::vector<double> meanPart = fixationParts.front();
            std::vector<double> maxPart = fixationParts.front();
            for (size_t fixationIndex = 1; fixationIndex < fixationParts.size(); ++fixationIndex) {
                const auto& rawPart = fixationParts[fixationIndex];
                if (rawPart.size() != meanPart.size()) {
                    continue;
                }
                for (size_t i = 0; i < meanPart.size(); ++i) {
                    meanPart[i] += rawPart[i];
                    maxPart[i] = std::max(maxPart[i], rawPart[i]);
                }
            }
            if (fixationParts.size() > 1) {
                const double invFixations = 1.0 / static_cast<double>(fixationParts.size());
                for (double& value : meanPart) {
                    value *= invFixations;
                }
            }
            auto finalizedMean = finalizeRetinaPartPattern(*retina, meanPart);
            auto finalizedMax = finalizeRetinaPartPattern(*retina, maxPart);
            preNormPart.reserve(meanPart.size() + maxPart.size());
            preNormPart.insert(preNormPart.end(), meanPart.begin(), meanPart.end());
            preNormPart.insert(preNormPart.end(), maxPart.begin(), maxPart.end());
            finalizedPart.reserve(finalizedMean.size() + finalizedMax.size());
            finalizedPart.insert(finalizedPart.end(), finalizedMean.begin(), finalizedMean.end());
            finalizedPart.insert(finalizedPart.end(), finalizedMax.begin(), finalizedMax.end());
            const auto [meanPreOrientationL2, meanPreAuxiliaryL2] =
                computeRetinaChannelL2Energies(*retina, meanPart);
            const auto [maxPreOrientationL2, maxPreAuxiliaryL2] =
                computeRetinaChannelL2Energies(*retina, maxPart);
            const auto [meanPostOrientationL2, meanPostAuxiliaryL2] =
                computeRetinaChannelL2Energies(*retina, finalizedMean);
            const auto [maxPostOrientationL2, maxPostAuxiliaryL2] =
                computeRetinaChannelL2Energies(*retina, finalizedMax);
            preOrientationL2 = std::sqrt(meanPreOrientationL2 * meanPreOrientationL2 +
                                         maxPreOrientationL2 * maxPreOrientationL2);
            preAuxiliaryL2 = std::sqrt(meanPreAuxiliaryL2 * meanPreAuxiliaryL2 +
                                       maxPreAuxiliaryL2 * maxPreAuxiliaryL2);
            postOrientationL2 = std::sqrt(meanPostOrientationL2 * meanPostOrientationL2 +
                                          maxPostOrientationL2 * maxPostOrientationL2);
            postAuxiliaryL2 = std::sqrt(meanPostAuxiliaryL2 * meanPostAuxiliaryL2 +
                                        maxPostAuxiliaryL2 * maxPostAuxiliaryL2);
        } else {
            for (const auto& rawPart : fixationParts) {
                if (part.empty()) {
                    part.assign(rawPart.size(), 0.0);
                }
                if (rawPart.size() != part.size()) {
                    continue;
                }
                for (size_t i = 0; i < part.size(); ++i) {
                    part[i] += rawPart[i];
                }
            }
            if (!part.empty() && fixationImages.size() > 1) {
                const double invFixations = 1.0 / static_cast<double>(fixationImages.size());
                for (double& value : part) {
                    value *= invFixations;
                }
            }
            preNormPart = part;
            finalizedPart = finalizeRetinaPartPattern(*retina, std::move(part));
            const auto [basePreOrientationL2, basePreAuxiliaryL2] =
                computeRetinaChannelL2Energies(*retina, preNormPart);
            const auto [basePostOrientationL2, basePostAuxiliaryL2] =
                computeRetinaChannelL2Energies(*retina, finalizedPart);
            preOrientationL2 = basePreOrientationL2;
            preAuxiliaryL2 = basePreAuxiliaryL2;
            postOrientationL2 = basePostOrientationL2;
            postAuxiliaryL2 = basePostAuxiliaryL2;
        }

        if (flowAuditCapture != nullptr) {
            FlowAuditPartCapture capture;
            capture.name = retina->getName();
            capture.preNormPattern = preNormPart;
            capture.postNormPattern = finalizedPart;
            capture.preNorm = computeFlowAuditVectorStats(preNormPart);
            capture.postNorm = computeFlowAuditVectorStats(finalizedPart);
            capture.preOrientationL2 = preOrientationL2;
            capture.preAuxiliaryL2 = preAuxiliaryL2;
            capture.postOrientationL2 = postOrientationL2;
            capture.postAuxiliaryL2 = postAuxiliaryL2;
            capture.fixationCount = static_cast<int>(std::max<size_t>(1, fixationImages.size()));
            capture.fixationVariance = computeFlowAuditFixationVariance(fixationParts);
            capture.fixationConsistency =
                computeFlowAuditFixationConsistency(finalizedFixationParts);
            flowAuditCapture->parts.push_back(std::move(capture));
            for (size_t fixationIndex = 0;
                 fixationIndex < finalizedFixationParts.size() &&
                 fixationIndex < flowAuditCapture->fixationPatterns.size();
                 ++fixationIndex) {
                auto& combinedFixationPattern =
                    flowAuditCapture->fixationPatterns[fixationIndex];
                const auto& fixationPart = finalizedFixationParts[fixationIndex];
                combinedFixationPattern.insert(combinedFixationPattern.end(),
                                              fixationPart.begin(),
                                              fixationPart.end());
            }
        }
        combined.insert(combined.end(), finalizedPart.begin(), finalizedPart.end());
    }

    if (flowAuditCapture != nullptr) {
        for (auto& fixationPattern : flowAuditCapture->fixationPatterns) {
            normalizeL2(fixationPattern);
        }
        flowAuditCapture->fixationConsistency =
            computeFlowAuditFixationConsistency(flowAuditCapture->fixationPatterns);
    }

    return combined;
}

struct Stage1PatternBatch {
    std::vector<std::vector<double>> patterns;
    std::vector<FlowAuditSampleCapture> captures;
};

Stage1PatternBatch extractStage1TrainingPatternBatch(
    std::vector<std::unique_ptr<RetinaAdapter>>& retinas,
    const VisualStimulus& image,
    const Config& config,
    bool learnPatterns,
    size_t sampleSeed,
    bool captureFlowAudit) {
    Stage1PatternBatch batch;
    const bool hasPerFixationMemory = std::any_of(
        retinas.begin(), retinas.end(), [&](const auto& retina) {
            return usePerFixationStage1Memory(*retina, config, true);
        });
    if (!hasPerFixationMemory) {
        FlowAuditSampleCapture capture;
        batch.patterns.push_back(extractPattern(retinas,
                                                image,
                                                config,
                                                true,
                                                learnPatterns,
                                                sampleSeed,
                                                captureFlowAudit ? &capture : nullptr));
        if (captureFlowAudit) {
            batch.captures.push_back(std::move(capture));
        }
        return batch;
    }

    const auto fixationImages = buildExtractionStimuli(image, config, true, sampleSeed, &retinas);

    std::vector<std::vector<std::vector<double>>> retinaPatternOptions;
    std::vector<std::vector<FlowAuditPartCapture>> retinaCaptureOptions;
    retinaPatternOptions.reserve(retinas.size());
    retinaCaptureOptions.reserve(retinas.size());

    size_t outputPatternCount = 1;
    for (auto& retina : retinas) {
        const bool perFixationMemory =
            usePerFixationStage1Memory(*retina, config, true) && fixationImages.size() > 1;
        std::vector<std::vector<double>> fixationRawParts;
        fixationRawParts.reserve(fixationImages.size());
        for (const auto& fixationImage : fixationImages) {
            fixationRawParts.push_back(
                extractRetinaPartRawPattern(*retina, fixationImage, config.useFeatures, learnPatterns));
        }

        std::vector<std::vector<double>> patternOptions;
        std::vector<FlowAuditPartCapture> captureOptions;
        if (perFixationMemory) {
            patternOptions.reserve(fixationRawParts.size());
            captureOptions.reserve(fixationRawParts.size());
            const double fixationVariance = computeFlowAuditFixationVariance(fixationRawParts);
            for (const auto& rawPart : fixationRawParts) {
                auto finalizedPart = finalizeRetinaPartPattern(*retina, rawPart);
                patternOptions.push_back(finalizedPart);
                if (captureFlowAudit) {
                    FlowAuditPartCapture capture;
                    capture.name = retina->getName();
                    capture.preNormPattern = rawPart;
                    capture.postNormPattern = finalizedPart;
                    capture.preNorm = computeFlowAuditVectorStats(rawPart);
                    capture.postNorm = computeFlowAuditVectorStats(finalizedPart);
                    const auto [preOrientationL2, preAuxiliaryL2] =
                        computeRetinaChannelL2Energies(*retina, rawPart);
                    capture.preOrientationL2 = preOrientationL2;
                    capture.preAuxiliaryL2 = preAuxiliaryL2;
                    const auto [postOrientationL2, postAuxiliaryL2] =
                        computeRetinaChannelL2Energies(*retina, finalizedPart);
                    capture.postOrientationL2 = postOrientationL2;
                    capture.postAuxiliaryL2 = postAuxiliaryL2;
                    capture.fixationCount = 1;
                    capture.fixationVariance = fixationVariance;
                    captureOptions.push_back(std::move(capture));
                }
            }
        } else {
            std::vector<double> meanPart;
            for (const auto& rawPart : fixationRawParts) {
                if (meanPart.empty()) {
                    meanPart.assign(rawPart.size(), 0.0);
                }
                if (rawPart.size() != meanPart.size()) {
                    continue;
                }
                for (size_t i = 0; i < meanPart.size(); ++i) {
                    meanPart[i] += rawPart[i];
                }
            }
            if (!meanPart.empty() && fixationRawParts.size() > 1) {
                const double invFixations = 1.0 / static_cast<double>(fixationRawParts.size());
                for (double& value : meanPart) {
                    value *= invFixations;
                }
            }
            auto finalizedPart = finalizeRetinaPartPattern(*retina, meanPart);
            patternOptions.push_back(finalizedPart);
            if (captureFlowAudit) {
                FlowAuditPartCapture capture;
                capture.name = retina->getName();
                capture.preNormPattern = meanPart;
                capture.postNormPattern = finalizedPart;
                capture.preNorm = computeFlowAuditVectorStats(meanPart);
                capture.postNorm = computeFlowAuditVectorStats(finalizedPart);
                const auto [preOrientationL2, preAuxiliaryL2] =
                    computeRetinaChannelL2Energies(*retina, meanPart);
                capture.preOrientationL2 = preOrientationL2;
                capture.preAuxiliaryL2 = preAuxiliaryL2;
                const auto [postOrientationL2, postAuxiliaryL2] =
                    computeRetinaChannelL2Energies(*retina, finalizedPart);
                capture.postOrientationL2 = postOrientationL2;
                capture.postAuxiliaryL2 = postAuxiliaryL2;
                capture.fixationCount = static_cast<int>(std::max<size_t>(1, fixationRawParts.size()));
                capture.fixationVariance = computeFlowAuditFixationVariance(fixationRawParts);
                captureOptions.push_back(std::move(capture));
            }
        }

        outputPatternCount = std::max(outputPatternCount, patternOptions.size());
        retinaPatternOptions.push_back(std::move(patternOptions));
        retinaCaptureOptions.push_back(std::move(captureOptions));
    }

    batch.patterns.resize(outputPatternCount);
    if (captureFlowAudit) {
        batch.captures.resize(outputPatternCount);
    }
    for (size_t optionIndex = 0; optionIndex < outputPatternCount; ++optionIndex) {
        for (size_t retinaIndex = 0; retinaIndex < retinaPatternOptions.size(); ++retinaIndex) {
            const auto& options = retinaPatternOptions[retinaIndex];
            if (options.empty()) {
                continue;
            }
            const size_t chosenIndex = optionIndex < options.size() ? optionIndex : 0;
            const auto& chosenPart = options[chosenIndex];
            batch.patterns[optionIndex].insert(batch.patterns[optionIndex].end(),
                                               chosenPart.begin(),
                                               chosenPart.end());
            if (captureFlowAudit) {
                const auto& captures = retinaCaptureOptions[retinaIndex];
                if (!captures.empty()) {
                    const size_t captureIndex = optionIndex < captures.size() ? optionIndex : 0;
                    batch.captures[optionIndex].parts.push_back(captures[captureIndex]);
                }
            }
        }
    }
    return batch;
}

std::vector<double> extractRetinaPartPattern(RetinaAdapter& retina,
                                             const VisualStimulus& image,
                                             bool useFeatures) {
    return finalizeRetinaPartPattern(
        retina, extractRetinaPartRawPattern(retina, image, useFeatures, false));
}

std::vector<double> remapRetinaPartPattern(const RetinaAdapter& retina,
                                           std::vector<double> part,
                                           const VisualStimulus& image,
                                           double shiftXPx,
                                           double shiftYPx,
                                           bool enabled) {
    if (!enabled || part.empty() || image.rows <= 0 || image.cols <= 0) {
        return part;
    }

    const int gridSize = std::max(1, retina.getIntParam("grid_size", 1));
    const size_t regionCount = static_cast<size_t>(gridSize * gridSize);
    if (regionCount == 0 || (part.size() % regionCount) != 0) {
        return part;
    }

    const int shiftCols = static_cast<int>(std::lround(
        (-shiftXPx) * static_cast<double>(gridSize) / static_cast<double>(image.cols)));
    const int shiftRows = static_cast<int>(std::lround(
        (-shiftYPx) * static_cast<double>(gridSize) / static_cast<double>(image.rows)));
    if (shiftCols == 0 && shiftRows == 0) {
        return part;
    }

    const size_t blockSize = part.size() / regionCount;
    std::vector<double> remapped(part.size(), 0.0);
    for (int row = 0; row < gridSize; ++row) {
        for (int col = 0; col < gridSize; ++col) {
            const int dstRow = row + shiftRows;
            const int dstCol = col + shiftCols;
            if (dstRow < 0 || dstRow >= gridSize || dstCol < 0 || dstCol >= gridSize) {
                continue;
            }
            const size_t srcBase =
                (static_cast<size_t>(row * gridSize + col) * blockSize);
            const size_t dstBase =
                (static_cast<size_t>(dstRow * gridSize + dstCol) * blockSize);
            std::copy_n(part.begin() + static_cast<std::ptrdiff_t>(srcBase),
                        static_cast<std::ptrdiff_t>(blockSize),
                        remapped.begin() + static_cast<std::ptrdiff_t>(dstBase));
        }
    }
    return remapped;
}

void appendFixationToAccumulator(HemisphereRuntime& hemisphere,
                                 HemispherePatternAccumulator& accumulator,
                                 const FixationSpec& fixation,
                                 const Config& config) {
    if (accumulator.retinaSums.empty()) {
        accumulator.retinaSums.resize(hemisphere.retinas.size());
    }

    for (size_t retinaIndex = 0; retinaIndex < hemisphere.retinas.size(); ++retinaIndex) {
        auto& retina = hemisphere.retinas[retinaIndex];
        auto rawPart = extractRetinaPartRawPattern(
            *retina, fixation.image, config.useFeatures, false);
        rawPart = remapRetinaPartPattern(
            *retina,
            std::move(rawPart),
            fixation.image,
            fixation.shiftXPx,
            fixation.shiftYPx,
            config.activeInferenceRemapEnabled);
        auto& sum = accumulator.retinaSums[retinaIndex];
        if (sum.empty()) {
            sum.assign(rawPart.size(), 0.0);
        }
        if (sum.size() != rawPart.size()) {
            continue;
        }
        for (size_t i = 0; i < sum.size(); ++i) {
            sum[i] += rawPart[i];
        }
    }

    accumulator.fixationCount++;
}

std::vector<double> buildPatternFromAccumulator(HemisphereRuntime& hemisphere,
                                                const HemispherePatternAccumulator& accumulator) {
    std::vector<double> combined;
    const double invFixations =
        1.0 / static_cast<double>(std::max(1, accumulator.fixationCount));
    for (size_t retinaIndex = 0; retinaIndex < hemisphere.retinas.size(); ++retinaIndex) {
        if (retinaIndex >= accumulator.retinaSums.size()) {
            continue;
        }
        std::vector<double> part = accumulator.retinaSums[retinaIndex];
        for (double& value : part) {
            value *= invFixations;
        }
        part = finalizeRetinaPartPattern(*hemisphere.retinas[retinaIndex], std::move(part));
        combined.insert(combined.end(), part.begin(), part.end());
    }
    return combined;
}

std::vector<PatternSlice> buildRetinaProjectionSlices(const RetinaAdapter& retina,
                                                      const PatternSlice& branchSlice);

std::vector<PatternSlice> buildPatternSlices(std::vector<std::unique_ptr<RetinaAdapter>>& retinas,
                                             const VisualStimulus& image,
                                             bool useFeatures) {
    std::vector<PatternSlice> slices;
    size_t offset = 0;
    slices.reserve(retinas.size());
    for (auto& retina : retinas) {
        const auto part = extractRetinaPartPattern(*retina, image, useFeatures);
        PatternSlice branchSlice;
        branchSlice.name = retina->getName();
        branchSlice.offset = offset;
        branchSlice.size = part.size();
        slices.push_back(branchSlice);
        if (useFeatures) {
            auto projections = buildRetinaProjectionSlices(*retina, branchSlice);
            slices.insert(slices.end(),
                          std::make_move_iterator(projections.begin()),
                          std::make_move_iterator(projections.end()));
        }
        offset += part.size();
    }
    return slices;
}

std::vector<double> slicePattern(const std::vector<double>& pattern, const PatternSlice& slice) {
    std::vector<double> result;
    if (!slice.indices.empty()) {
        result.reserve(slice.indices.size());
        for (size_t index : slice.indices) {
            if (index < pattern.size()) {
                result.push_back(pattern[index]);
            }
        }
        normalizeL2(result);
        return result;
    }
    if (slice.offset >= pattern.size()) {
        return {};
    }
    const size_t end = std::min(pattern.size(), slice.offset + slice.size);
    if (end <= slice.offset) {
        return {};
    }
    result.assign(pattern.begin() + static_cast<std::ptrdiff_t>(slice.offset),
                  pattern.begin() + static_cast<std::ptrdiff_t>(end));
    normalizeL2(result);
    return result;
}

std::vector<double> slicePatternRaw(const std::vector<double>& pattern, const PatternSlice& slice) {
    if (!slice.indices.empty()) {
        std::vector<double> result;
        result.reserve(slice.indices.size());
        for (size_t index : slice.indices) {
            if (index < pattern.size()) {
                result.push_back(pattern[index]);
            }
        }
        return result;
    }
    if (slice.offset >= pattern.size()) {
        return {};
    }
    const size_t end = std::min(pattern.size(), slice.offset + slice.size);
    if (end <= slice.offset) {
        return {};
    }
    return std::vector<double>(pattern.begin() + static_cast<std::ptrdiff_t>(slice.offset),
                               pattern.begin() + static_cast<std::ptrdiff_t>(end));
}

bool useHemisphereConvergentCode(const Config& config) {
    return config.hemisphereConvergentCodeEnabled;
}

bool useHemisphereTopographicStage(const Config& config) {
    return config.hemisphereTopographicStageEnabled;
}

bool useFigureGroundStage(const Config& config) {
    return config.figureGroundStageEnabled;
}

bool useFigureGroundMask(const Config& config) {
    return config.figureGroundStageEnabled && config.figureGroundMaskEnabled;
}

bool useFigureGroundClassifier(const Config& config) {
    return config.figureGroundStageEnabled && config.figureGroundClassifierEnabled;
}

bool useFigureGroundObjectMemory(const Config& config) {
    return config.figureGroundStageEnabled && config.figureGroundObjectMemoryEnabled &&
           config.figureGroundObjectMemoryGain > 1e-6;
}

bool useRecurrentSensoryState(const Config& config) {
    return config.recurrentSensoryStateEnabled &&
           config.recurrentSensoryCycles > 0 &&
           (config.recurrentSensoryFeedforwardGain > 1e-6 ||
            config.recurrentSensoryStateGain > 1e-6 ||
            config.recurrentSensoryFigureGroundGain > 1e-6 ||
            config.recurrentSensoryContinuityGain > 1e-6 ||
            config.recurrentSensoryCallosalGain > 1e-6);
}

bool useRecurrentPopulationReadout(const Config& config) {
    return useRecurrentSensoryState(config) && config.recurrentPopulationReadoutEnabled;
}

bool useRecurrentObjectStateReadout(const Config& config) {
    return useRecurrentSensoryState(config) &&
           config.recurrentObjectStateReadoutEnabled &&
           config.recurrentObjectStateUnitsPerClass > 0 &&
           config.recurrentObjectStateCycles > 0;
}

std::vector<size_t> inferHemisphereBranchPatternSizes(
    std::vector<std::unique_ptr<RetinaAdapter>>& retinas,
    const VisualStimulus& image,
    bool useFeatures) {
    std::vector<size_t> sizes;
    sizes.reserve(retinas.size());
    const auto slices = buildPatternSlices(retinas, image, useFeatures);
    for (const auto& retina : retinas) {
        const auto it = std::find_if(slices.begin(), slices.end(), [&](const PatternSlice& slice) {
            return slice.name == retina->getName();
        });
        sizes.push_back(it != slices.end() ? it->size : 0u);
    }
    return sizes;
}

std::vector<double> poolPatternToBins(const std::vector<double>& pattern, int bins) {
    if (pattern.empty()) {
        return {};
    }
    const int actualBins = std::max(1, std::min<int>(bins, static_cast<int>(pattern.size())));
    std::vector<double> pooled(static_cast<size_t>(actualBins), 0.0);
    std::vector<int> counts(static_cast<size_t>(actualBins), 0);
    for (size_t i = 0; i < pattern.size(); ++i) {
        const int bin = std::min(
            actualBins - 1,
            static_cast<int>((i * static_cast<size_t>(actualBins)) / pattern.size()));
        pooled[static_cast<size_t>(bin)] += pattern[i];
        counts[static_cast<size_t>(bin)]++;
    }
    for (size_t i = 0; i < pooled.size(); ++i) {
        if (counts[i] > 0) {
            pooled[i] /= static_cast<double>(counts[i]);
        }
    }
    normalizeL2(pooled);
    return pooled;
}

void appendScaledPattern(std::vector<double>& destination,
                         const std::vector<double>& source,
                         double gain) {
    if (source.empty() || std::abs(gain) <= 1e-9) {
        return;
    }
    destination.reserve(destination.size() + source.size());
    for (double value : source) {
        destination.push_back(gain * value);
    }
}

std::vector<double> buildConvergentBranchSummary(const RetinaAdapter& retina,
                                                 const std::vector<double>& branchPattern,
                                                 const Config& config) {
    if (branchPattern.empty()) {
        return {};
    }

    std::vector<double> summary;
    const int bins = std::max(2, config.hemisphereConvergentSummaryBins);
    appendScaledPattern(summary, poolPatternToBins(branchPattern, bins), 1.0);

    PatternSlice branchSlice;
    branchSlice.name = retina.getName();
    branchSlice.offset = 0;
    branchSlice.size = branchPattern.size();
    const auto projections = buildRetinaProjectionSlices(retina, branchSlice);
    for (const auto& projection : projections) {
        const auto projectionPattern = slicePatternRaw(branchPattern, projection);
        appendScaledPattern(summary, poolPatternToBins(projectionPattern, bins), 1.0);
    }

    normalizeL2(summary);
    return summary;
}

std::vector<double> buildHemisphereConvergentPattern(const HemisphereRuntime& hemisphere,
                                                     const std::vector<double>& pattern,
                                                     const Config& config) {
    if (!useHemisphereConvergentCode(config) ||
        hemisphere.retinas.empty() ||
        hemisphere.retinaPatternSizes.size() != hemisphere.retinas.size()) {
        return {};
    }

    std::vector<std::vector<double>> branchSummaries;
    branchSummaries.reserve(hemisphere.retinas.size());

    size_t offset = 0;
    for (size_t retinaIndex = 0; retinaIndex < hemisphere.retinas.size(); ++retinaIndex) {
        const size_t partSize = hemisphere.retinaPatternSizes[retinaIndex];
        if (partSize == 0 || offset >= pattern.size()) {
            offset += partSize;
            continue;
        }
        const size_t end = std::min(pattern.size(), offset + partSize);
        if (end <= offset) {
            offset += partSize;
            continue;
        }

        std::vector<double> branchPattern(pattern.begin() + static_cast<std::ptrdiff_t>(offset),
                                          pattern.begin() + static_cast<std::ptrdiff_t>(end));
        normalizeL2(branchPattern);
        auto summary =
            buildConvergentBranchSummary(*hemisphere.retinas[retinaIndex], branchPattern, config);
        if (!summary.empty()) {
            branchSummaries.push_back(std::move(summary));
        }
        offset += partSize;
    }

    if (branchSummaries.empty()) {
        return {};
    }

    std::vector<double> convergentPattern;
    for (const auto& summary : branchSummaries) {
        appendScaledPattern(convergentPattern,
                            summary,
                            std::max(0.0, config.hemisphereConvergentResidualGain));
    }

    for (size_t a = 0; a < branchSummaries.size(); ++a) {
        for (size_t b = a + 1; b < branchSummaries.size(); ++b) {
            const size_t dim = std::min(branchSummaries[a].size(), branchSummaries[b].size());
            for (size_t i = 0; i < dim; ++i) {
                convergentPattern.push_back(config.hemisphereConvergentInteractionGain *
                                            branchSummaries[a][i] * branchSummaries[b][i]);
            }
            for (size_t i = 0; i < dim; ++i) {
                convergentPattern.push_back(config.hemisphereConvergentInteractionGain *
                                            std::abs(branchSummaries[a][i] -
                                                     branchSummaries[b][i]));
            }
        }
    }

    normalizeL2(convergentPattern);
    return convergentPattern;
}

size_t inferAuxiliaryChannelCount(const RetinaAdapter& retina) {
    const std::string mode = toLower(retina.getStringParam("auxiliary_feature_mode", "none"));
    if (mode == "none") {
        return 0u;
    }
    if (mode == "closure_bank") {
        return 3u;
    }
    if (mode == "gap_bank") {
        return 4u;
    }
    if (mode == "topology_maps") {
        return 8u;
    }
    if (mode == "contour_graph") {
        return 8u;
    }
    if (mode == "contour_sequence") {
        return 8u;
    }
    if (mode == "color_opponent") {
        return 3u;
    }
    if (mode == "appearance_bank") {
        return 6u;
    }
    if (mode == "appearance_stream_bank") {
        return 10u;
    }
    if (mode == "luminance_stream_bank") {
        return 7u;
    }
    if (mode == "contour_support_bank") {
        return 6u;
    }
    return 1u;
}

size_t inferRetinaPartPatternSize(const RetinaAdapter& retina) {
    const size_t gridSize =
        static_cast<size_t>(std::max(1, retina.getIntParam("grid_size", 1)));
    const size_t regionCount = gridSize * gridSize;
    const size_t numOrientations =
        static_cast<size_t>(std::max(1, retina.getIntParam("num_orientations", 8)));
    const int subfieldGridSize = std::max(1, retina.getIntParam("subfield_grid_size", 1));
    const bool subfieldIncludePooled = retina.getIntParam("subfield_include_pooled", 1) != 0;
    const size_t subfieldCount =
        subfieldGridSize <= 1 ? 0u : static_cast<size_t>(subfieldGridSize * subfieldGridSize);
    const size_t orientationBlocks =
        subfieldCount == 0 ? 1u : (subfieldIncludePooled ? 1u + subfieldCount : subfieldCount);
    const size_t colorEdgeChannels =
        toLower(retina.getStringParam("color_edge_mode", "none")) == "opponent" ? 3u : 1u;
    const size_t auxiliaryChannels = inferAuxiliaryChannelCount(retina);
    const size_t frequencyBandCount = inferFrequencyBandCount(retina);
    const size_t orientationFeatureCount =
        numOrientations * orientationBlocks * colorEdgeChannels;
    const size_t perRegionEntries =
        (orientationFeatureCount + auxiliaryChannels) * frequencyBandCount;
    return regionCount * perRegionEntries;
}

size_t inferFrequencyBandCount(const RetinaAdapter& retina) {
    const std::string csv = retina.getStringParam("frequency_values", "");
    if (csv.empty()) {
        return 1u;
    }
    std::set<double> values;
    std::stringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            continue;
        }
        try {
            values.insert(std::stod(token));
        } catch (...) {
        }
    }
    return std::max<size_t>(1u, values.size());
}

struct ContextualGroupingLayout {
    size_t gridSize = 0;
    size_t regionCount = 0;
    size_t numOrientations = 0;
    size_t orientationBlocks = 0;
    size_t colorEdgeChannels = 0;
    size_t auxiliaryChannels = 0;
    size_t frequencyBandCount = 0;
    size_t orientationFeatureCount = 0;
    size_t perRegionOrientationEntries = 0;
    size_t perRegionEntries = 0;

    bool valid() const {
        return gridSize > 0 &&
               regionCount > 0 &&
               numOrientations > 0 &&
               orientationBlocks > 0 &&
               colorEdgeChannels > 0 &&
               frequencyBandCount > 0 &&
               orientationFeatureCount > 0 &&
               perRegionOrientationEntries > 0 &&
               perRegionEntries > 0;
    }

    size_t orientationChannelIndex(size_t block, size_t color, size_t orientation) const {
        return ((block * colorEdgeChannels) + color) * numOrientations + orientation;
    }

    size_t regionValueIndex(size_t region,
                            size_t orientationChannel,
                            size_t band) const {
        return region * perRegionEntries + orientationChannel * frequencyBandCount + band;
    }
};

ContextualGroupingLayout inferContextualGroupingLayout(const RetinaAdapter& retina,
                                                       size_t totalSize) {
    ContextualGroupingLayout layout;
    layout.gridSize =
        static_cast<size_t>(std::max(1, retina.getIntParam("grid_size", 1)));
    layout.regionCount = layout.gridSize * layout.gridSize;
    layout.numOrientations =
        static_cast<size_t>(std::max(1, retina.getIntParam("num_orientations", 8)));
    const int subfieldGridSize = std::max(1, retina.getIntParam("subfield_grid_size", 1));
    const bool subfieldIncludePooled = retina.getIntParam("subfield_include_pooled", 1) != 0;
    const size_t subfieldCount =
        subfieldGridSize <= 1 ? 0u : static_cast<size_t>(subfieldGridSize * subfieldGridSize);
    layout.orientationBlocks =
        subfieldCount == 0 ? 1u : (subfieldIncludePooled ? 1u + subfieldCount : subfieldCount);
    layout.colorEdgeChannels =
        toLower(retina.getStringParam("color_edge_mode", "none")) == "opponent" ? 3u : 1u;
    layout.auxiliaryChannels = inferAuxiliaryChannelCount(retina);
    layout.frequencyBandCount = inferFrequencyBandCount(retina);
    layout.orientationFeatureCount =
        layout.numOrientations * layout.orientationBlocks * layout.colorEdgeChannels;
    layout.perRegionOrientationEntries =
        layout.orientationFeatureCount * layout.frequencyBandCount;
    const size_t perRegionAuxEntries = layout.auxiliaryChannels * layout.frequencyBandCount;
    layout.perRegionEntries = layout.perRegionOrientationEntries + perRegionAuxEntries;
    if (layout.regionCount == 0 ||
        layout.perRegionEntries == 0 ||
        layout.regionCount * layout.perRegionEntries != totalSize) {
        return {};
    }
    return layout;
}

std::pair<double, double> computeRetinaChannelL2Energies(const RetinaAdapter& retina,
                                                         const std::vector<double>& part) {
    const auto layout = inferContextualGroupingLayout(retina, part.size());
    if (!layout.valid()) {
        const double l2 = computeFlowAuditVectorStats(part).l2;
        return {l2, 0.0};
    }

    double orientationSq = 0.0;
    double auxiliarySq = 0.0;
    for (size_t region = 0; region < layout.regionCount; ++region) {
        const size_t base = region * layout.perRegionEntries;
        for (size_t i = 0; i < layout.perRegionOrientationEntries; ++i) {
            const double value = part[base + i];
            orientationSq += value * value;
        }
        for (size_t i = layout.perRegionOrientationEntries; i < layout.perRegionEntries; ++i) {
            const double value = part[base + i];
            auxiliarySq += value * value;
        }
    }
    return {std::sqrt(orientationSq), std::sqrt(auxiliarySq)};
}

void initializeFlowAuditTraining(HemisphereTrainingArtifacts& artifacts,
                                 const std::vector<std::unique_ptr<RetinaAdapter>>& retinas,
                                 int numClasses) {
    artifacts.flowAuditEnabled = true;
    artifacts.flowAuditBranches.clear();
    artifacts.flowAuditBranches.reserve(retinas.size());
    for (const auto& retina : retinas) {
        HemisphereTrainingArtifacts::FlowAuditBranchTrainingStats stats;
        stats.name = retina->getName();
        stats.preCentroids.assign(static_cast<size_t>(std::max(0, numClasses)), {});
        stats.postCentroids.assign(static_cast<size_t>(std::max(0, numClasses)), {});
        stats.preCounts.assign(static_cast<size_t>(std::max(0, numClasses)), 0);
        stats.postCounts.assign(static_cast<size_t>(std::max(0, numClasses)), 0);
        artifacts.flowAuditBranches.push_back(std::move(stats));
    }
}

void accumulateFlowAuditCentroid(std::vector<double>& centroid,
                                 int& count,
                                 const std::vector<double>& pattern) {
    if (pattern.empty()) {
        return;
    }
    if (centroid.empty()) {
        centroid.assign(pattern.size(), 0.0);
    }
    if (centroid.size() != pattern.size()) {
        return;
    }
    for (size_t i = 0; i < pattern.size(); ++i) {
        centroid[i] += pattern[i];
    }
    ++count;
}

void accumulateFlowAuditTrainingCapture(HemisphereTrainingArtifacts& artifacts,
                                        const FlowAuditSampleCapture& capture,
                                        int label) {
    if (!artifacts.flowAuditEnabled || label < 0) {
        return;
    }
    for (size_t branchIndex = 0;
         branchIndex < capture.parts.size() && branchIndex < artifacts.flowAuditBranches.size();
         ++branchIndex) {
        auto& branch = artifacts.flowAuditBranches[branchIndex];
        if (label < 0 || static_cast<size_t>(label) >= branch.preCentroids.size()) {
            continue;
        }
        accumulateFlowAuditCentroid(branch.preCentroids[static_cast<size_t>(label)],
                                    branch.preCounts[static_cast<size_t>(label)],
                                    capture.parts[branchIndex].preNormPattern);
        accumulateFlowAuditCentroid(branch.postCentroids[static_cast<size_t>(label)],
                                    branch.postCounts[static_cast<size_t>(label)],
                                    capture.parts[branchIndex].postNormPattern);
        branch.fixationVarianceSum += capture.parts[branchIndex].fixationVariance;
        branch.fixationVarianceSamples++;
    }
}

void finalizeFlowAuditTraining(HemisphereTrainingArtifacts& artifacts) {
    if (!artifacts.flowAuditEnabled) {
        return;
    }
    for (auto& branch : artifacts.flowAuditBranches) {
        for (size_t label = 0; label < branch.preCentroids.size(); ++label) {
            if (branch.preCounts[label] > 0) {
                const double invCount = 1.0 / static_cast<double>(branch.preCounts[label]);
                for (double& value : branch.preCentroids[label]) {
                    value *= invCount;
                }
                normalizeL2(branch.preCentroids[label]);
            }
            if (branch.postCounts[label] > 0) {
                const double invCount = 1.0 / static_cast<double>(branch.postCounts[label]);
                for (double& value : branch.postCentroids[label]) {
                    value *= invCount;
                }
                normalizeL2(branch.postCentroids[label]);
            }
        }
    }
}

int quantizeDirectionComponent(double value) {
    if (std::abs(value) < 0.33) {
        return 0;
    }
    return value > 0.0 ? 1 : -1;
}

std::pair<int, int> quantizeDirection(double angleRadians) {
    int dx = quantizeDirectionComponent(std::cos(angleRadians));
    int dy = quantizeDirectionComponent(std::sin(angleRadians));
    if (dx == 0 && dy == 0) {
        if (std::abs(std::cos(angleRadians)) >= std::abs(std::sin(angleRadians))) {
            dx = std::cos(angleRadians) >= 0.0 ? 1 : -1;
        } else {
            dy = std::sin(angleRadians) >= 0.0 ? 1 : -1;
        }
    }
    return {dx, dy};
}

double sampleRegionValue(const std::vector<double>& values,
                         size_t gridSize,
                         int row,
                         int col) {
    if (row < 0 || col < 0 ||
        row >= static_cast<int>(gridSize) ||
        col >= static_cast<int>(gridSize)) {
        return 0.0;
    }
    return values[static_cast<size_t>(row) * gridSize + static_cast<size_t>(col)];
}

double sampleEnergy(const std::vector<double>& energy,
                    const ContextualGroupingLayout& layout,
                    int row,
                    int col,
                    size_t orientation,
                    size_t band) {
    if (row < 0 || col < 0 ||
        row >= static_cast<int>(layout.gridSize) ||
        col >= static_cast<int>(layout.gridSize) ||
        orientation >= layout.numOrientations ||
        band >= layout.frequencyBandCount) {
        return 0.0;
    }
    const size_t region =
        static_cast<size_t>(row) * layout.gridSize + static_cast<size_t>(col);
    return energy[(region * layout.numOrientations + orientation) * layout.frequencyBandCount + band];
}

std::vector<double> buildTopographicBranchStage(const RetinaAdapter& retina,
                                                const std::vector<double>& branchPattern,
                                                const Config& config) {
    const ContextualGroupingLayout layout =
        inferContextualGroupingLayout(retina, branchPattern.size());
    if (!layout.valid()) {
        return branchPattern;
    }

    std::vector<double> energy(
        layout.regionCount * layout.numOrientations * layout.frequencyBandCount, 0.0);
    std::vector<double> auxiliary(layout.regionCount * layout.frequencyBandCount, 0.0);

    for (size_t region = 0; region < layout.regionCount; ++region) {
        for (size_t orientation = 0; orientation < layout.numOrientations; ++orientation) {
            for (size_t band = 0; band < layout.frequencyBandCount; ++band) {
                double sum = 0.0;
                int count = 0;
                for (size_t block = 0; block < layout.orientationBlocks; ++block) {
                    for (size_t color = 0; color < layout.colorEdgeChannels; ++color) {
                        const size_t orientationChannel =
                            layout.orientationChannelIndex(block, color, orientation);
                        const size_t index =
                            layout.regionValueIndex(region, orientationChannel, band);
                        sum += branchPattern[index];
                        ++count;
                    }
                }
                energy[(region * layout.numOrientations + orientation) *
                           layout.frequencyBandCount +
                       band] =
                    count > 0 ? (sum / static_cast<double>(count)) : 0.0;
            }
        }

        for (size_t band = 0; band < layout.frequencyBandCount; ++band) {
            double sum = 0.0;
            int count = 0;
            for (size_t aux = 0; aux < layout.auxiliaryChannels; ++aux) {
                const size_t index =
                    region * layout.perRegionEntries + layout.perRegionOrientationEntries +
                    aux * layout.frequencyBandCount + band;
                sum += branchPattern[index];
                ++count;
            }
            auxiliary[region * layout.frequencyBandCount + band] =
                count > 0 ? (sum / static_cast<double>(count)) : 0.0;
        }
    }

    std::vector<double> stage;
    stage.reserve(layout.regionCount *
                  (layout.numOrientations * layout.frequencyBandCount * 3 +
                   layout.frequencyBandCount));
    for (size_t row = 0; row < layout.gridSize; ++row) {
        for (size_t col = 0; col < layout.gridSize; ++col) {
            const size_t region = row * layout.gridSize + col;
            for (size_t orientation = 0; orientation < layout.numOrientations; ++orientation) {
                const double angle =
                    (static_cast<double>(orientation) * M_PI) /
                    static_cast<double>(layout.numOrientations);
                const auto [dx, dy] = quantizeDirection(angle);
                const auto [odx, ody] = quantizeDirection(angle + (M_PI * 0.5));
                const size_t leftOrientation =
                    (orientation + layout.numOrientations - 1) % layout.numOrientations;
                const size_t rightOrientation = (orientation + 1) % layout.numOrientations;

                for (size_t band = 0; band < layout.frequencyBandCount; ++band) {
                    const double center =
                        sampleEnergy(energy, layout, static_cast<int>(row), static_cast<int>(col),
                                     orientation, band);
                    const double alongA =
                        sampleEnergy(energy, layout, static_cast<int>(row) + dy,
                                     static_cast<int>(col) + dx, orientation, band);
                    const double alongB =
                        sampleEnergy(energy, layout, static_cast<int>(row) - dy,
                                     static_cast<int>(col) - dx, orientation, band);
                    const double orthoA =
                        sampleEnergy(energy, layout, static_cast<int>(row) + ody,
                                     static_cast<int>(col) + odx, orientation, band);
                    const double orthoB =
                        sampleEnergy(energy, layout, static_cast<int>(row) - ody,
                                     static_cast<int>(col) - odx, orientation, band);
                    const double continuity =
                        center * std::max(0.0, 0.5 * (alongA + alongB) - 0.5 * (orthoA + orthoB));
                    const double junction =
                        center * 0.5 *
                        (sampleEnergy(energy, layout, static_cast<int>(row), static_cast<int>(col),
                                      leftOrientation, band) +
                         sampleEnergy(energy, layout, static_cast<int>(row), static_cast<int>(col),
                                      rightOrientation, band));

                    stage.push_back(config.hemisphereTopographicResidualGain * center);
                    stage.push_back(config.hemisphereTopographicContinuityGain * continuity);
                    stage.push_back(config.hemisphereTopographicJunctionGain * junction);
                }
            }

            for (size_t band = 0; band < layout.frequencyBandCount; ++band) {
                stage.push_back(config.hemisphereTopographicAuxiliaryGain *
                                auxiliary[region * layout.frequencyBandCount + band]);
            }
        }
    }

    normalizeL2(stage);
    return stage;
}

std::vector<double> buildHemisphereTopographicPattern(const HemisphereRuntime& hemisphere,
                                                      const std::vector<double>& pattern,
                                                      const Config& config) {
    if (!useHemisphereTopographicStage(config) ||
        hemisphere.retinas.empty() ||
        hemisphere.retinaPatternSizes.size() != hemisphere.retinas.size()) {
        return {};
    }

    std::vector<double> topographicPattern;
    size_t offset = 0;
    for (size_t retinaIndex = 0; retinaIndex < hemisphere.retinas.size(); ++retinaIndex) {
        const size_t partSize = hemisphere.retinaPatternSizes[retinaIndex];
        if (partSize == 0 || offset >= pattern.size()) {
            offset += partSize;
            continue;
        }
        const size_t end = std::min(pattern.size(), offset + partSize);
        if (end <= offset) {
            offset += partSize;
            continue;
        }
        std::vector<double> branchPattern(pattern.begin() + static_cast<std::ptrdiff_t>(offset),
                                          pattern.begin() + static_cast<std::ptrdiff_t>(end));
        normalizeL2(branchPattern);
        auto stage =
            buildTopographicBranchStage(*hemisphere.retinas[retinaIndex], branchPattern, config);
        topographicPattern.insert(topographicPattern.end(), stage.begin(), stage.end());
        offset += partSize;
    }

    normalizeL2(topographicPattern);
    return topographicPattern;
}

int countActiveGridComponents(const std::vector<double>& values,
                              size_t gridSize,
                              double threshold) {
    if (values.size() != gridSize * gridSize || gridSize == 0) {
        return 0;
    }

    std::vector<uint8_t> visited(values.size(), 0u);
    int components = 0;
    for (size_t index = 0; index < values.size(); ++index) {
        if (visited[index] != 0u || values[index] <= threshold) {
            continue;
        }
        ++components;
        std::vector<size_t> stack{index};
        visited[index] = 1u;
        while (!stack.empty()) {
            const size_t current = stack.back();
            stack.pop_back();
            const int row = static_cast<int>(current / gridSize);
            const int col = static_cast<int>(current % gridSize);
            constexpr std::array<std::pair<int, int>, 4> kOffsets{
                {{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
            for (const auto& [dRow, dCol] : kOffsets) {
                const int nextRow = row + dRow;
                const int nextCol = col + dCol;
                if (nextRow < 0 || nextCol < 0 ||
                    nextRow >= static_cast<int>(gridSize) ||
                    nextCol >= static_cast<int>(gridSize)) {
                    continue;
                }
                const size_t nextIndex =
                    static_cast<size_t>(nextRow) * gridSize + static_cast<size_t>(nextCol);
                if (visited[nextIndex] != 0u || values[nextIndex] <= threshold) {
                    continue;
                }
                visited[nextIndex] = 1u;
                stack.push_back(nextIndex);
            }
        }
    }

    return components;
}

FigureGroundBranchSummary buildFigureGroundBranchSummary(const RetinaAdapter& retina,
                                                         const std::vector<double>& branchPattern,
                                                         const Config& config) {
    FigureGroundBranchSummary summary;
    summary.name = retina.getName();
    if (branchPattern.empty()) {
        return summary;
    }

    const ContextualGroupingLayout layout =
        inferContextualGroupingLayout(retina, branchPattern.size());
    if (!layout.valid()) {
        summary.pattern = branchPattern;
        normalizeL2(summary.pattern);
        summary.surfaceMass =
            std::accumulate(summary.pattern.begin(), summary.pattern.end(), 0.0,
                            [](double acc, double value) { return acc + std::abs(value); });
        summary.ownedLeftMass = 0.5 * summary.surfaceMass;
        summary.ownedRightMass = 0.5 * summary.surfaceMass;
        summary.borderSurfaceRatio = summary.surfaceMass > 1e-9 ? 1.0 : 0.0;
        summary.componentCount = summary.surfaceMass > 1e-9 ? 1 : 0;
        return summary;
    }

    std::vector<double> energy(
        layout.regionCount * layout.numOrientations * layout.frequencyBandCount, 0.0);
    std::vector<double> auxiliary(
        layout.regionCount * layout.auxiliaryChannels * layout.frequencyBandCount, 0.0);
    std::vector<double> regionSurface(layout.regionCount, 0.0);
    std::vector<double> seedLeft(layout.regionCount, 0.0);
    std::vector<double> seedRight(layout.regionCount, 0.0);
    std::vector<double> junctionSeed(layout.regionCount, 0.0);
    std::vector<double> dominantSupportByRegion(layout.regionCount, 0.0);
    std::vector<size_t> dominantIndexByRegion(layout.regionCount, 0u);

    const std::string auxiliaryMode =
        toLower(retina.getStringParam("auxiliary_feature_mode", "none"));
    for (size_t region = 0; region < layout.regionCount; ++region) {
        double pooledSurface = 0.0;
        double pooledJunction = 0.0;
        double orientationMass = 0.0;
        std::vector<double> pooledOrientation(layout.numOrientations, 0.0);

        for (size_t orientation = 0; orientation < layout.numOrientations; ++orientation) {
            for (size_t band = 0; band < layout.frequencyBandCount; ++band) {
                double sum = 0.0;
                int count = 0;
                for (size_t block = 0; block < layout.orientationBlocks; ++block) {
                    for (size_t color = 0; color < layout.colorEdgeChannels; ++color) {
                        const size_t orientationChannel =
                            layout.orientationChannelIndex(block, color, orientation);
                        const size_t index =
                            layout.regionValueIndex(region, orientationChannel, band);
                        sum += branchPattern[index];
                        ++count;
                    }
                }
                const double mean = count > 0 ? (sum / static_cast<double>(count)) : 0.0;
                energy[(region * layout.numOrientations + orientation) *
                           layout.frequencyBandCount +
                       band] = mean;
                pooledOrientation[orientation] += mean;
                orientationMass += mean;
            }
        }

        for (size_t orientation = 0; orientation < layout.numOrientations; ++orientation) {
            pooledOrientation[orientation] /=
                static_cast<double>(std::max<size_t>(1u, layout.frequencyBandCount));
        }

        for (size_t aux = 0; aux < layout.auxiliaryChannels; ++aux) {
            for (size_t band = 0; band < layout.frequencyBandCount; ++band) {
                const size_t sourceIndex =
                    region * layout.perRegionEntries + layout.perRegionOrientationEntries +
                    aux * layout.frequencyBandCount + band;
                const double value = branchPattern[sourceIndex];
                auxiliary[(region * layout.auxiliaryChannels + aux) *
                              layout.frequencyBandCount +
                          band] = value;
                pooledSurface += value;
                if (aux == 3) {
                    pooledJunction += value;
                }
            }
        }

        const double dominantSupport = pooledOrientation.empty()
                                           ? 0.0
                                           : *std::max_element(pooledOrientation.begin(),
                                                               pooledOrientation.end());
        const size_t dominantIndex =
            pooledOrientation.empty()
                ? 0u
                : static_cast<size_t>(std::distance(
                      pooledOrientation.begin(),
                      std::max_element(pooledOrientation.begin(), pooledOrientation.end())));
        const size_t orthIndex =
            (dominantIndex + (layout.numOrientations / 2)) % layout.numOrientations;
        const double orthSupport =
            pooledOrientation.empty() ? 0.0 : pooledOrientation[orthIndex];
        pooledSurface = layout.auxiliaryChannels > 0
                            ? (pooledSurface /
                               static_cast<double>(layout.auxiliaryChannels *
                                                   layout.frequencyBandCount))
                            : (orientationMass /
                               static_cast<double>(layout.numOrientations *
                                                   layout.frequencyBandCount));
        const double junctionValue =
            std::max(pooledJunction /
                         static_cast<double>(std::max<size_t>(
                             1u, layout.frequencyBandCount)),
                     dominantSupport * orthSupport);
        regionSurface[region] = std::clamp(0.6 * pooledSurface + 0.4 * dominantSupport,
                                           0.0,
                                           2.0);
        dominantSupportByRegion[region] = dominantSupport;
        dominantIndexByRegion[region] = dominantIndex;
        junctionSeed[region] = std::clamp(junctionValue, 0.0, 2.0);
    }

    for (size_t region = 0; region < layout.regionCount; ++region) {
        const double theta =
            (static_cast<double>(dominantIndexByRegion[region]) * M_PI) /
            static_cast<double>(std::max<size_t>(1u, layout.numOrientations));
        const auto [normalDx, normalDy] = quantizeDirection(theta + (M_PI * 0.5));
        const int row = static_cast<int>(region / layout.gridSize);
        const int col = static_cast<int>(region % layout.gridSize);
        const double positiveSide =
            sampleRegionValue(regionSurface, layout.gridSize, row + normalDy, col + normalDx);
        const double negativeSide =
            sampleRegionValue(regionSurface, layout.gridSize, row - normalDy, col - normalDx);
        const double genericBalance =
            std::clamp(positiveSide - negativeSide, -1.0, 1.0);

        double borderLeft = 0.0;
        double borderRight = 0.0;
        if (auxiliaryMode == "contour_support_bank" && layout.auxiliaryChannels >= 6) {
            double leftSum = 0.0;
            double rightSum = 0.0;
            for (size_t band = 0; band < layout.frequencyBandCount; ++band) {
                leftSum += auxiliary[(region * layout.auxiliaryChannels + 4) *
                                         layout.frequencyBandCount +
                                     band];
                rightSum += auxiliary[(region * layout.auxiliaryChannels + 5) *
                                          layout.frequencyBandCount +
                                      band];
            }
            borderLeft = leftSum / static_cast<double>(layout.frequencyBandCount);
            borderRight = rightSum / static_cast<double>(layout.frequencyBandCount);
        } else {
            borderLeft = dominantSupportByRegion[region] *
                         std::clamp(0.5 - 0.5 * genericBalance, 0.0, 1.0);
            borderRight = dominantSupportByRegion[region] *
                          std::clamp(0.5 + 0.5 * genericBalance, 0.0, 1.0);
        }
        seedLeft[region] = std::clamp(borderLeft, 0.0, 2.0);
        seedRight[region] = std::clamp(borderRight, 0.0, 2.0);
    }

    std::vector<double> left = seedLeft;
    std::vector<double> right = seedRight;
    std::vector<double> surface = regionSurface;
    for (int cycle = 0; cycle < std::max(1, config.figureGroundCycles); ++cycle) {
        std::vector<double> nextLeft(left.size(), 0.0);
        std::vector<double> nextRight(right.size(), 0.0);
        std::vector<double> nextSurface(surface.size(), 0.0);
        for (size_t region = 0; region < layout.regionCount; ++region) {
            const int row = static_cast<int>(region / layout.gridSize);
            const int col = static_cast<int>(region % layout.gridSize);
            double neighborSurface = 0.0;
            int neighborCount = 0;
            for (int dRow = -1; dRow <= 1; ++dRow) {
                for (int dCol = -1; dCol <= 1; ++dCol) {
                    if (dRow == 0 && dCol == 0) {
                        continue;
                    }
                    const int nextRow = row + dRow;
                    const int nextCol = col + dCol;
                    if (nextRow < 0 || nextCol < 0 ||
                        nextRow >= static_cast<int>(layout.gridSize) ||
                        nextCol >= static_cast<int>(layout.gridSize)) {
                        continue;
                    }
                    neighborSurface += sampleRegionValue(surface,
                                                         layout.gridSize,
                                                         nextRow,
                                                         nextCol);
                    ++neighborCount;
                }
            }
            if (neighborCount > 0) {
                neighborSurface /= static_cast<double>(neighborCount);
            }

            const double leftDrive = seedLeft[region] +
                                     config.figureGroundFeedbackGain *
                                         std::max(0.0, surface[region] - right[region]);
            const double rightDrive = seedRight[region] +
                                      config.figureGroundFeedbackGain *
                                          std::max(0.0, surface[region] - left[region]);
            nextLeft[region] = std::clamp(
                config.figureGroundResidualGain * left[region] +
                    config.figureGroundBorderGain * leftDrive -
                    config.figureGroundCompetitionGain * right[region],
                0.0,
                4.0);
            nextRight[region] = std::clamp(
                config.figureGroundResidualGain * right[region] +
                    config.figureGroundBorderGain * rightDrive -
                    config.figureGroundCompetitionGain * left[region],
                0.0,
                4.0);
            nextSurface[region] = std::clamp(
                config.figureGroundResidualGain * surface[region] +
                    config.figureGroundSurfaceGain *
                        (0.55 * regionSurface[region] +
                         0.30 * neighborSurface +
                         0.15 * std::max(nextLeft[region], nextRight[region])),
                0.0,
                4.0);
        }
        left.swap(nextLeft);
        right.swap(nextRight);
        surface.swap(nextSurface);
    }

    summary.pattern.reserve(layout.regionCount * 4u);
    for (size_t region = 0; region < layout.regionCount; ++region) {
        summary.pattern.push_back(left[region]);
        summary.pattern.push_back(right[region]);
        summary.pattern.push_back(surface[region]);
        summary.pattern.push_back(junctionSeed[region]);
    }
    normalizeL2(summary.pattern);

    summary.ownedLeftMass = std::accumulate(left.begin(), left.end(), 0.0);
    summary.ownedRightMass = std::accumulate(right.begin(), right.end(), 0.0);
    summary.surfaceMass = std::accumulate(surface.begin(), surface.end(), 0.0);
    summary.junctionMass = std::accumulate(junctionSeed.begin(), junctionSeed.end(), 0.0);
    const double borderMass = summary.ownedLeftMass + summary.ownedRightMass;
    summary.borderSurfaceRatio =
        summary.surfaceMass > 1e-9 ? (borderMass / summary.surfaceMass) : 0.0;
    summary.leftRightBalance =
        borderMass > 1e-9
            ? ((summary.ownedRightMass - summary.ownedLeftMass) / borderMass)
            : 0.0;
    const double meanSurface =
        summary.surfaceMass / static_cast<double>(std::max<size_t>(1u, surface.size()));
    const double componentThreshold = std::max(0.01, 0.75 * meanSurface);
    summary.componentCount =
        countActiveGridComponents(surface, layout.gridSize, componentThreshold);
    return summary;
}

FigureGroundState buildHemisphereFigureGroundState(const HemisphereRuntime& hemisphere,
                                                   const std::vector<double>& pattern,
                                                   const Config& config) {
    FigureGroundState state;
    if ((!config.figureGroundStageEnabled && !useRecurrentSensoryState(config)) ||
        hemisphere.retinas.empty() || pattern.empty()) {
        return state;
    }

    std::vector<size_t> patternSizes = hemisphere.retinaPatternSizes;
    if (patternSizes.size() != hemisphere.retinas.size()) {
        patternSizes.clear();
        patternSizes.reserve(hemisphere.retinas.size());
        for (const auto& retina : hemisphere.retinas) {
            patternSizes.push_back(inferRetinaPartPatternSize(*retina));
        }
    }
    if (patternSizes.size() != hemisphere.retinas.size()) {
        return state;
    }

    size_t offset = 0;
    for (size_t retinaIndex = 0; retinaIndex < hemisphere.retinas.size(); ++retinaIndex) {
        const size_t partSize = patternSizes[retinaIndex];
        if (partSize == 0 || offset >= pattern.size()) {
            offset += partSize;
            continue;
        }
        const size_t end = std::min(pattern.size(), offset + partSize);
        if (end <= offset) {
            offset += partSize;
            continue;
        }
        std::vector<double> branchPattern(pattern.begin() + static_cast<std::ptrdiff_t>(offset),
                                          pattern.begin() + static_cast<std::ptrdiff_t>(end));
        normalizeL2(branchPattern);
        auto branchSummary = buildFigureGroundBranchSummary(
            *hemisphere.retinas[retinaIndex], branchPattern, config);
        if (!branchSummary.pattern.empty()) {
            state.pattern.insert(state.pattern.end(),
                                 branchSummary.pattern.begin(),
                                 branchSummary.pattern.end());
        }
        state.ownedBorderMass +=
            branchSummary.ownedLeftMass + branchSummary.ownedRightMass;
        state.surfaceMass += branchSummary.surfaceMass;
        state.junctionMass += branchSummary.junctionMass;
        state.leftRightBalance += branchSummary.leftRightBalance;
        state.componentCount += branchSummary.componentCount;
        state.branches.push_back(std::move(branchSummary));
        offset += partSize;
    }

    if (state.branches.empty() || state.pattern.empty()) {
        state.pattern.clear();
        return state;
    }

    state.borderSurfaceRatio =
        state.surfaceMass > 1e-9 ? (state.ownedBorderMass / state.surfaceMass) : 0.0;
    state.leftRightBalance /=
        static_cast<double>(std::max<size_t>(1u, state.branches.size()));
    normalizeL2(state.pattern);
    state.valid = true;
    return state;
}

std::vector<double> buildFigureGroundDrivePattern(const HemisphereRuntime& hemisphere,
                                                  const std::vector<double>& pattern,
                                                  const FigureGroundState& state) {
    if (!state.valid || state.branches.empty() || hemisphere.retinas.empty() ||
        hemisphere.retinaPatternSizes.size() != hemisphere.retinas.size() || pattern.empty()) {
        return {};
    }

    std::vector<double> drive(pattern.size(), 0.0);
    bool wroteAny = false;
    size_t offset = 0;
    for (size_t retinaIndex = 0;
         retinaIndex < hemisphere.retinas.size() && retinaIndex < state.branches.size();
         ++retinaIndex) {
        const size_t partSize = hemisphere.retinaPatternSizes[retinaIndex];
        const size_t end = std::min(pattern.size(), offset + partSize);
        if (partSize == 0 || end <= offset) {
            offset += partSize;
            continue;
        }

        const auto& branchSummary = state.branches[retinaIndex];
        const ContextualGroupingLayout layout =
            inferContextualGroupingLayout(*hemisphere.retinas[retinaIndex], partSize);
        if (!layout.valid() ||
            branchSummary.pattern.size() != layout.regionCount * 4u ||
            layout.perRegionEntries == 0u) {
            offset += partSize;
            continue;
        }

        double maxSignal = 0.0;
        std::vector<double> regionSignals(layout.regionCount, 0.0);
        for (size_t region = 0; region < layout.regionCount; ++region) {
            const size_t fgBase = region * 4u;
            const double left = branchSummary.pattern[fgBase];
            const double right = branchSummary.pattern[fgBase + 1u];
            const double surface = branchSummary.pattern[fgBase + 2u];
            const double junction = branchSummary.pattern[fgBase + 3u];
            const double signal =
                std::max(0.0, surface + 0.30 * std::max(left, right) + 0.15 * junction);
            regionSignals[region] = signal;
            maxSignal = std::max(maxSignal, signal);
        }
        if (maxSignal <= 1e-9) {
            offset += partSize;
            continue;
        }

        for (size_t region = 0; region < layout.regionCount; ++region) {
            const double gate = std::clamp(regionSignals[region] / maxSignal, 0.0, 1.0);
            if (gate <= 1e-9) {
                continue;
            }
            const size_t regionBase = offset + region * layout.perRegionEntries;
            const size_t regionEnd =
                std::min(end, regionBase + layout.perRegionEntries);
            if (regionBase >= regionEnd) {
                continue;
            }
            for (size_t index = regionBase; index < regionEnd; ++index) {
                drive[index] = pattern[index] * gate;
            }
            wroteAny = true;
        }
        offset += partSize;
    }

    if (!wroteAny) {
        return {};
    }
    normalizeL2(drive);
    return drive;
}

RecurrentSensoryPatternResult buildRecurrentSensoryPattern(HemisphereRuntime& hemisphere,
                                                           const VisualStimulus& image,
                                                           const Config& config,
                                                           bool trainingPhase,
                                                           size_t sampleSeed) {
    RecurrentSensoryPatternResult result;
    if (!useRecurrentSensoryState(config) || hemisphere.retinas.empty()) {
        return result;
    }

    if (hemisphere.retinaPatternSizes.size() != hemisphere.retinas.size()) {
        hemisphere.retinaPatternSizes =
            inferHemisphereBranchPatternSizes(hemisphere.retinas, image, config.useFeatures);
    }

    const auto fixationImages = buildRecurrentExtractionStimuli(
        image, config, trainingPhase, sampleSeed, &hemisphere.retinas);
    if (fixationImages.empty()) {
        return result;
    }

    result.fixationPatterns.reserve(fixationImages.size());
    std::vector<double> previousSettled;
    for (const auto& fixationImage : fixationImages) {
        std::vector<double> fixationPattern;
        for (auto& retina : hemisphere.retinas) {
            auto rawPart =
                extractRetinaPartRawPattern(*retina, fixationImage, config.useFeatures, false);
            auto finalizedPart = finalizeRetinaPartPattern(*retina, std::move(rawPart));
            fixationPattern.insert(
                fixationPattern.end(), finalizedPart.begin(), finalizedPart.end());
        }
        normalizeL2(fixationPattern);
        if (fixationPattern.empty()) {
            continue;
        }

        const auto figureGroundState =
            buildHemisphereFigureGroundState(hemisphere, fixationPattern, config);
        auto figureGroundDrive =
            buildFigureGroundDrivePattern(hemisphere, fixationPattern, figureGroundState);
        std::vector<double> settled = fixationPattern;
        for (int cycle = 0; cycle < std::max(1, config.recurrentSensoryCycles); ++cycle) {
            std::vector<double> next(settled.size(), 0.0);
            for (size_t i = 0; i < settled.size(); ++i) {
                double value =
                    config.recurrentSensoryFeedforwardGain * fixationPattern[i] +
                    config.recurrentSensoryStateGain * settled[i];
                if (!figureGroundDrive.empty() && i < figureGroundDrive.size()) {
                    value += config.recurrentSensoryFigureGroundGain * figureGroundDrive[i];
                }
                if (!previousSettled.empty() && i < previousSettled.size()) {
                    value += config.recurrentSensoryContinuityGain * previousSettled[i];
                }
                next[i] = value;
            }
            normalizeL2(next);
            settled.swap(next);
        }

        previousSettled = settled;
        result.fixationPatterns.push_back(settled);
        result.figureGroundPattern = figureGroundState.pattern;
    }

    result.fixationCount = static_cast<int>(result.fixationPatterns.size());
    if (result.fixationPatterns.empty()) {
        return result;
    }

    result.pattern.assign(result.fixationPatterns.front().size(), 0.0);
    size_t validCount = 0;
    for (const auto& fixationPattern : result.fixationPatterns) {
        if (fixationPattern.size() != result.pattern.size()) {
            continue;
        }
        for (size_t i = 0; i < result.pattern.size(); ++i) {
            result.pattern[i] += fixationPattern[i];
        }
        ++validCount;
    }
    if (validCount > 0) {
        const double invCount = 1.0 / static_cast<double>(validCount);
        for (double& value : result.pattern) {
            value *= invCount;
        }
        normalizeL2(result.pattern);
    } else {
        result.pattern = result.fixationPatterns.back();
    }
    return result;
}

std::vector<RecurrentSensoryPatternResult> buildRecurrentSensoryPatterns(
    std::vector<HemisphereRuntime>& hemispheres,
    const VisualStimulus& image,
    const Config& config,
    bool trainingPhase,
    size_t sampleSeed) {
    std::vector<RecurrentSensoryPatternResult> results;
    results.reserve(hemispheres.size());
    for (auto& hemisphere : hemispheres) {
        results.push_back(
            buildRecurrentSensoryPattern(hemisphere, image, config, trainingPhase, sampleSeed));
    }

    const double callosalGain = std::max(0.0, config.recurrentSensoryCallosalGain);
    if (callosalGain <= 1e-9 || results.size() < 2) {
        return results;
    }

    std::vector<std::vector<double>> originalPatterns;
    originalPatterns.reserve(results.size());
    for (const auto& result : results) {
        originalPatterns.push_back(result.pattern);
    }

    for (size_t hemisphereIndex = 0; hemisphereIndex < results.size(); ++hemisphereIndex) {
        auto& pattern = results[hemisphereIndex].pattern;
        if (pattern.empty()) {
            continue;
        }
        std::vector<double> support(pattern.size(), 0.0);
        int supportCount = 0;
        for (size_t otherIndex = 0; otherIndex < originalPatterns.size(); ++otherIndex) {
            if (otherIndex == hemisphereIndex ||
                originalPatterns[otherIndex].size() != pattern.size()) {
                continue;
            }
            for (size_t i = 0; i < pattern.size(); ++i) {
                support[i] += originalPatterns[otherIndex][i];
            }
            ++supportCount;
        }
        if (supportCount <= 0) {
            continue;
        }
        const double invSupport = 1.0 / static_cast<double>(supportCount);
        for (double& value : support) {
            value *= invSupport;
        }
        for (size_t i = 0; i < pattern.size(); ++i) {
            pattern[i] = (1.0 - callosalGain) * pattern[i] + callosalGain * support[i];
        }
        normalizeL2(pattern);
    }

    return results;
}

double computeFigureGroundFixationConsistency(const HemisphereRuntime& hemisphere,
                                              const FlowAuditSampleCapture* sampleCapture,
                                              const Config& config) {
    if (sampleCapture == nullptr) {
        return 1.0;
    }
    if (sampleCapture->fixationPatterns.empty()) {
        return 0.0;
    }
    if (sampleCapture->fixationPatterns.size() == 1) {
        return 1.0;
    }

    std::vector<std::vector<double>> figureGroundFixationPatterns;
    figureGroundFixationPatterns.reserve(sampleCapture->fixationPatterns.size());
    for (const auto& fixationPattern : sampleCapture->fixationPatterns) {
        const auto state =
            buildHemisphereFigureGroundState(hemisphere, fixationPattern, config);
        if (state.valid && !state.pattern.empty()) {
            figureGroundFixationPatterns.push_back(state.pattern);
        }
    }
    return computeFlowAuditFixationConsistency(figureGroundFixationPatterns);
}

std::vector<double> buildFigureGroundMaskedPattern(const HemisphereRuntime& hemisphere,
                                                   const std::vector<double>& pattern,
                                                   const FigureGroundState& state,
                                                   const Config& config) {
    if (!useFigureGroundMask(config) || !state.valid || state.branches.empty() ||
        hemisphere.retinas.empty() || hemisphere.retinaPatternSizes.empty() ||
        hemisphere.retinaPatternSizes.size() != hemisphere.retinas.size()) {
        return {};
    }

    std::vector<double> maskedPattern = pattern;
    const double maskGain = std::max(0.0, config.figureGroundMaskGain);
    size_t offset = 0;
    for (size_t retinaIndex = 0;
         retinaIndex < hemisphere.retinas.size() && retinaIndex < state.branches.size();
         ++retinaIndex) {
        const size_t partSize = hemisphere.retinaPatternSizes[retinaIndex];
        const size_t end = std::min(maskedPattern.size(), offset + partSize);
        if (partSize == 0 || end <= offset) {
            offset += partSize;
            continue;
        }

        const auto& branchSummary = state.branches[retinaIndex];
        const auto layout = inferContextualGroupingLayout(*hemisphere.retinas[retinaIndex], partSize);
        if (!layout.valid() || branchSummary.pattern.size() != layout.regionCount * 4u) {
            offset += partSize;
            continue;
        }

        std::vector<double> regionSignals(layout.regionCount, 0.0);
        double meanSignal = 0.0;
        for (size_t region = 0; region < layout.regionCount; ++region) {
            const size_t base = region * 4u;
            const double left = branchSummary.pattern[base];
            const double right = branchSummary.pattern[base + 1u];
            const double surface = branchSummary.pattern[base + 2u];
            const double junction = branchSummary.pattern[base + 3u];
            const double signal =
                std::max(0.0, surface + 0.30 * std::max(left, right) + 0.15 * junction);
            regionSignals[region] = signal;
            meanSignal += signal;
        }
        meanSignal /= static_cast<double>(std::max<size_t>(1u, regionSignals.size()));
        if (meanSignal <= 1e-9) {
            offset += partSize;
            continue;
        }

        for (size_t region = 0; region < layout.regionCount; ++region) {
            const double normalizedSignal = regionSignals[region] / meanSignal;
            const double weight = std::clamp(
                1.0 + maskGain * (normalizedSignal - 1.0),
                0.25,
                2.50);
            const size_t regionBase = offset + region * layout.perRegionEntries;
            for (size_t valueIndex = 0; valueIndex < layout.perRegionEntries; ++valueIndex) {
                maskedPattern[regionBase + valueIndex] *= weight;
            }
        }

        offset += partSize;
    }

    normalizeL2(maskedPattern);
    return maskedPattern;
}

std::vector<double> buildFigureGroundObjectMemoryPattern(
    const HemisphereRuntime& hemisphere,
    const std::vector<double>& pattern,
    const FigureGroundState& state,
    const Config& config) {
    if (!useFigureGroundObjectMemory(config) || !state.valid || state.branches.empty() ||
        hemisphere.retinas.empty() || hemisphere.retinaPatternSizes.size() != hemisphere.retinas.size() ||
        pattern.empty()) {
        return {};
    }

    std::vector<double> objectPattern(pattern.size(), 0.0);
    bool keptAny = false;
    size_t offset = 0;
    const double keepFraction =
        std::clamp(config.figureGroundObjectMemoryKeepFraction, 0.05, 1.0);

    for (size_t branchIndex = 0;
         branchIndex < hemisphere.retinas.size() && branchIndex < hemisphere.retinaPatternSizes.size();
         ++branchIndex) {
        const size_t branchSize = hemisphere.retinaPatternSizes[branchIndex];
        if (offset >= pattern.size()) {
            break;
        }
        if (branchSize == 0 || offset + branchSize > pattern.size() ||
            branchIndex >= state.branches.size()) {
            offset += branchSize;
            continue;
        }

        const auto& retina = *hemisphere.retinas[branchIndex];
        const ContextualGroupingLayout layout =
            inferContextualGroupingLayout(retina, branchSize);
        const auto& branchSummary = state.branches[branchIndex];
        if (!layout.valid() ||
            branchSummary.pattern.size() != layout.regionCount * 4u ||
            layout.perRegionEntries == 0u) {
            offset += branchSize;
            continue;
        }

        std::vector<double> regionSignals(layout.regionCount, 0.0);
        double maxSignal = 0.0;
        for (size_t region = 0; region < layout.regionCount; ++region) {
            const size_t fgBase = region * 4u;
            const double left = branchSummary.pattern[fgBase];
            const double right = branchSummary.pattern[fgBase + 1u];
            const double surface = branchSummary.pattern[fgBase + 2u];
            const double junction = branchSummary.pattern[fgBase + 3u];
            const double signal =
                std::max(0.0, surface + 0.30 * std::max(left, right) + 0.15 * junction);
            regionSignals[region] = signal;
            maxSignal = std::max(maxSignal, signal);
        }
        if (maxSignal <= 1e-9) {
            offset += branchSize;
            continue;
        }

        std::vector<double> sortedSignals = regionSignals;
        std::sort(sortedSignals.begin(), sortedSignals.end(), std::greater<double>());
        const size_t keepCount = std::min<size_t>(
            layout.regionCount,
            std::max<size_t>(
                1u,
                static_cast<size_t>(std::llround(
                    keepFraction * static_cast<double>(layout.regionCount)))));
        const double keepThreshold = sortedSignals[keepCount - 1u];

        for (size_t region = 0; region < layout.regionCount; ++region) {
            const double signal = regionSignals[region];
            if (signal <= 1e-9 || signal + 1e-9 < keepThreshold) {
                continue;
            }
            const double gate = std::clamp(signal / maxSignal, 0.5, 1.0);
            const size_t regionBase = offset + region * layout.perRegionEntries;
            const size_t regionEnd =
                std::min(offset + branchSize, regionBase + layout.perRegionEntries);
            if (regionBase >= regionEnd || regionEnd > pattern.size()) {
                continue;
            }
            for (size_t index = regionBase; index < regionEnd; ++index) {
                objectPattern[index] = pattern[index] * gate;
            }
            keptAny = true;
        }

        offset += branchSize;
    }

    if (!keptAny) {
        return {};
    }
    normalizeL2(objectPattern);
    return objectPattern;
}

std::vector<double> buildFigureGroundClassifierPattern(const std::vector<double>& pattern,
                                                       const FigureGroundState& state,
                                                       const Config& config) {
    if (!useFigureGroundClassifier(config) || !state.valid || state.pattern.empty()) {
        return {};
    }

    std::vector<double> classifierPattern = pattern;
    normalizeL2(classifierPattern);

    std::vector<double> figureGroundPattern = state.pattern;
    normalizeL2(figureGroundPattern);
    const double fgGain = std::max(0.0, config.figureGroundClassifierGain);
    for (double& value : figureGroundPattern) {
        value *= fgGain;
    }

    const double totalMass =
        std::max(1e-9, state.ownedBorderMass + state.surfaceMass + state.junctionMass);
    std::vector<double> summary{
        state.ownedBorderMass / totalMass,
        state.surfaceMass / totalMass,
        state.junctionMass / totalMass,
        std::clamp(state.borderSurfaceRatio / 4.0, 0.0, 1.0),
        std::clamp(0.5 + 0.5 * state.leftRightBalance, 0.0, 1.0),
        std::clamp(static_cast<double>(state.componentCount) / 16.0, 0.0, 1.0),
    };
    normalizeL2(summary);
    for (double& value : summary) {
        value *= fgGain;
    }

    classifierPattern.insert(
        classifierPattern.end(), figureGroundPattern.begin(), figureGroundPattern.end());
    classifierPattern.insert(classifierPattern.end(), summary.begin(), summary.end());
    normalizeL2(classifierPattern);
    return classifierPattern;
}

std::vector<double> buildHemisphereClassifierPattern(const HemisphereRuntime& hemisphere,
                                                     const std::vector<double>& pattern,
                                                     const Config& config,
                                                     const FigureGroundState* figureGroundState) {
    if (useHemisphereTopographicStage(config)) {
        auto topographic = buildHemisphereTopographicPattern(hemisphere, pattern, config);
        if (!topographic.empty()) {
            return topographic;
        }
    }
    if (useHemisphereConvergentCode(config)) {
        auto convergent = buildHemisphereConvergentPattern(hemisphere, pattern, config);
        if (!convergent.empty()) {
            return convergent;
        }
    }
    if (useFigureGroundMask(config)) {
        FigureGroundState ownedState;
        const FigureGroundState* state = figureGroundState;
        if (state == nullptr || !state->valid) {
            ownedState = buildHemisphereFigureGroundState(hemisphere, pattern, config);
            state = &ownedState;
        }
        auto masked = buildFigureGroundMaskedPattern(hemisphere, pattern, *state, config);
        if (!masked.empty()) {
            return masked;
        }
    }
    if (useFigureGroundClassifier(config)) {
        FigureGroundState ownedState;
        const FigureGroundState* state = figureGroundState;
        if (state == nullptr || !state->valid) {
            ownedState = buildHemisphereFigureGroundState(hemisphere, pattern, config);
            state = &ownedState;
        }
        auto figureGround = buildFigureGroundClassifierPattern(pattern, *state, config);
        if (!figureGround.empty()) {
            return figureGround;
        }
    }
    return {};
}

std::vector<double> computeFigureGroundObjectMemoryConfidence(
    HemisphereRuntime& hemisphere,
    const std::vector<double>& pattern,
    const FigureGroundState& state,
    const Config& config,
    std::vector<double>* objectPatternOut) {
    hemisphere.lastFigureGroundObjectPattern.clear();
    if (!useFigureGroundObjectMemory(config) || hemisphere.classifier == nullptr ||
        hemisphere.figureGroundObjectMemoryPatterns.empty() || !state.valid) {
        return {};
    }

    auto objectPattern =
        buildFigureGroundObjectMemoryPattern(hemisphere, pattern, state, config);
    if (objectPattern.empty()) {
        return {};
    }

    hemisphere.lastFigureGroundObjectPattern = objectPattern;
    if (objectPatternOut != nullptr) {
        *objectPatternOut = objectPattern;
    }

    auto confidence = hemisphere.classifier->classifyWithConfidence(
        objectPattern, hemisphere.figureGroundObjectMemoryPatterns, cosineSimilarity);
    normalizeSum(confidence);
    return confidence;
}

std::vector<double> applyContextualGrouping(const RetinaAdapter& retina,
                                            std::vector<double> part) {
    if (retina.getIntParam("contextual_grouping_enabled", 0) == 0 || part.empty()) {
        return part;
    }

    const ContextualGroupingLayout layout =
        inferContextualGroupingLayout(retina, part.size());
    if (!layout.valid()) {
        return part;
    }

    const int iterations =
        std::max(1, retina.getIntParam("contextual_grouping_iterations", 3));
    const double contourGain =
        std::max(0.0, retina.getDoubleParam("contextual_grouping_contour_gain", 0.35));
    const double surroundGain =
        std::max(0.0, retina.getDoubleParam("contextual_grouping_surround_gain", 0.18));
    const double divisiveGain =
        std::max(0.0, retina.getDoubleParam("contextual_grouping_divisive_gain", 0.55));
    const double ownershipGain =
        std::max(0.0, retina.getDoubleParam("contextual_grouping_ownership_gain", 0.45));
    const double coarseBiasBase = std::clamp(
        retina.getDoubleParam("contextual_grouping_coarse_bias", 0.60), 0.0, 1.0);

    std::vector<double> energy(
        layout.regionCount * layout.numOrientations * layout.frequencyBandCount, 0.0);
    std::vector<double> regionSurface(layout.regionCount, 0.0);

    for (size_t region = 0; region < layout.regionCount; ++region) {
        double orientationMean = 0.0;
        for (size_t orientation = 0; orientation < layout.numOrientations; ++orientation) {
            for (size_t band = 0; band < layout.frequencyBandCount; ++band) {
                double sum = 0.0;
                int count = 0;
                for (size_t block = 0; block < layout.orientationBlocks; ++block) {
                    for (size_t color = 0; color < layout.colorEdgeChannels; ++color) {
                        const size_t orientationChannel =
                            layout.orientationChannelIndex(block, color, orientation);
                        const size_t index =
                            layout.regionValueIndex(region, orientationChannel, band);
                        sum += part[index];
                        ++count;
                    }
                }
                const double mean = count > 0 ? (sum / static_cast<double>(count)) : 0.0;
                energy[(region * layout.numOrientations + orientation) * layout.frequencyBandCount + band] =
                    mean;
                orientationMean += mean;
            }
        }

        double auxiliaryMean = 0.0;
        if (layout.auxiliaryChannels > 0) {
            for (size_t aux = 0; aux < layout.auxiliaryChannels; ++aux) {
                for (size_t band = 0; band < layout.frequencyBandCount; ++band) {
                    const size_t index =
                        region * layout.perRegionEntries +
                        layout.perRegionOrientationEntries +
                        aux * layout.frequencyBandCount + band;
                    auxiliaryMean += part[index];
                }
            }
            auxiliaryMean /=
                static_cast<double>(layout.auxiliaryChannels * layout.frequencyBandCount);
        }

        orientationMean /=
            static_cast<double>(layout.numOrientations * layout.frequencyBandCount);
        regionSurface[region] =
            layout.auxiliaryChannels > 0 ? (0.60 * auxiliaryMean + 0.40 * orientationMean)
                                         : orientationMean;
    }

    const std::vector<double> baseEnergy = energy;
    std::vector<double> updated = energy;
    const double kPi = 3.14159265358979323846;
    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<double> regionTotals(layout.regionCount * layout.frequencyBandCount, 0.0);
        for (size_t region = 0; region < layout.regionCount; ++region) {
            for (size_t band = 0; band < layout.frequencyBandCount; ++band) {
                double total = 0.0;
                for (size_t orientation = 0; orientation < layout.numOrientations; ++orientation) {
                    const double value = energy[(region * layout.numOrientations + orientation) *
                                                layout.frequencyBandCount +
                                                band];
                    total += value;
                }
                regionTotals[region * layout.frequencyBandCount + band] = total;
            }
        }

        for (size_t region = 0; region < layout.regionCount; ++region) {
            const int row = static_cast<int>(region / layout.gridSize);
            const int col = static_cast<int>(region % layout.gridSize);
            for (size_t orientation = 0; orientation < layout.numOrientations; ++orientation) {
                const double theta = (kPi * static_cast<double>(orientation)) /
                                     static_cast<double>(layout.numOrientations);
                const auto [tangentDx, tangentDy] = quantizeDirection(theta);
                const auto [normalDx, normalDy] = quantizeDirection(theta + (0.5 * kPi));
                const size_t prevOrientation =
                    orientation == 0 ? (layout.numOrientations - 1) : (orientation - 1);
                const size_t nextOrientation =
                    (orientation + 1) % layout.numOrientations;

                for (size_t band = 0; band < layout.frequencyBandCount; ++band) {
                    const size_t currentIndex =
                        (region * layout.numOrientations + orientation) *
                            layout.frequencyBandCount +
                        band;
                    const double currentValue = energy[currentIndex];
                    const double coarseBias =
                        layout.frequencyBandCount > 1 && iterations > 1
                            ? (coarseBiasBase *
                               (1.0 - static_cast<double>(iter) /
                                          static_cast<double>(iterations - 1)))
                            : 0.0;
                    const double lowBandReference = energy[(region * layout.numOrientations +
                                                            orientation) *
                                                               layout.frequencyBandCount];
                    const double referenceValue =
                        ((1.0 - coarseBias) * currentValue) + (coarseBias * lowBandReference);

                    const double contourSupport =
                        0.35 * (sampleEnergy(energy, layout, row + tangentDy, col + tangentDx,
                                             prevOrientation, band) +
                                sampleEnergy(energy, layout, row - tangentDy, col - tangentDx,
                                             prevOrientation, band)) +
                        0.30 * (sampleEnergy(energy, layout, row + tangentDy, col + tangentDx,
                                             orientation, band) +
                                sampleEnergy(energy, layout, row - tangentDy, col - tangentDx,
                                             orientation, band)) +
                        0.35 * (sampleEnergy(energy, layout, row + tangentDy, col + tangentDx,
                                             nextOrientation, band) +
                                sampleEnergy(energy, layout, row - tangentDy, col - tangentDx,
                                             nextOrientation, band));

                    double localPool = 0.0;
                    double orthogonalPool = 0.0;
                    int localCount = 0;
                    for (int dr = -1; dr <= 1; ++dr) {
                        for (int dc = -1; dc <= 1; ++dc) {
                            const int sampleRow = row + dr;
                            const int sampleCol = col + dc;
                            if (sampleRow < 0 || sampleCol < 0 ||
                                sampleRow >= static_cast<int>(layout.gridSize) ||
                                sampleCol >= static_cast<int>(layout.gridSize)) {
                                continue;
                            }
                            localPool += regionTotals[(static_cast<size_t>(sampleRow) *
                                                       layout.gridSize +
                                                       static_cast<size_t>(sampleCol)) *
                                                          layout.frequencyBandCount +
                                                      band];
                            ++localCount;
                        }
                    }
                    if (localCount > 0) {
                        localPool /= static_cast<double>(localCount);
                    }
                    orthogonalPool =
                        std::max(0.0,
                                 regionTotals[region * layout.frequencyBandCount + band] -
                                     currentValue);
                    orthogonalPool /=
                        std::max(1.0, static_cast<double>(layout.numOrientations - 1));

                    const auto computeSideEvidence = [&](int side) {
                        const int sideRow = row + side * normalDy;
                        const int sideCol = col + side * normalDx;
                        const double center = sampleRegionValue(regionSurface, layout.gridSize,
                                                                sideRow, sideCol);
                        const double forward = sampleRegionValue(
                            regionSurface, layout.gridSize, sideRow + tangentDy,
                            sideCol + tangentDx);
                        const double backward = sampleRegionValue(
                            regionSurface, layout.gridSize, sideRow - tangentDy,
                            sideCol - tangentDx);
                        return center + 0.5 * (forward + backward);
                    };

                    const double positiveSide = computeSideEvidence(1);
                    const double negativeSide = computeSideEvidence(-1);
                    const double sideDelta =
                        std::clamp(positiveSide - negativeSide, -1.0, 1.0);
                    const double drive =
                        referenceValue * (1.0 + contourGain * contourSupport);
                    const double positiveOwnership =
                        drive * std::clamp(1.0 + ownershipGain * sideDelta, 0.25, 2.0);
                    const double negativeOwnership =
                        drive * std::clamp(1.0 - ownershipGain * sideDelta, 0.25, 2.0);
                    const double ownedDrive = std::max(positiveOwnership, negativeOwnership);
                    const double suppressed =
                        std::max(0.0, ownedDrive - surroundGain * orthogonalPool);
                    const double normalized =
                        suppressed / (1.0 + divisiveGain * std::max(0.0, localPool));

                    updated[currentIndex] =
                        std::clamp(0.35 * currentValue + 0.65 * normalized, 0.0, 4.0);
                }
            }
        }
        energy.swap(updated);
    }

    for (size_t region = 0; region < layout.regionCount; ++region) {
        double regionScaleSum = 0.0;
        int regionScaleCount = 0;
        for (size_t orientation = 0; orientation < layout.numOrientations; ++orientation) {
            for (size_t band = 0; band < layout.frequencyBandCount; ++band) {
                const size_t energyIndex =
                    (region * layout.numOrientations + orientation) *
                        layout.frequencyBandCount +
                    band;
                const double originalMean = baseEnergy[energyIndex];
                const double targetMean = energy[energyIndex];
                const double scale =
                    originalMean > 1e-6
                        ? std::clamp(targetMean / originalMean, 0.45, 2.75)
                        : (targetMean > 1e-6 ? 1.25 : 1.0);
                regionScaleSum += scale;
                ++regionScaleCount;

                for (size_t block = 0; block < layout.orientationBlocks; ++block) {
                    for (size_t color = 0; color < layout.colorEdgeChannels; ++color) {
                        const size_t orientationChannel =
                            layout.orientationChannelIndex(block, color, orientation);
                        const size_t index =
                            layout.regionValueIndex(region, orientationChannel, band);
                        part[index] *= scale;
                    }
                }
            }
        }

        if (layout.auxiliaryChannels == 0 || regionScaleCount <= 0) {
            continue;
        }
        const double regionScale =
            std::clamp(regionScaleSum / static_cast<double>(regionScaleCount), 0.60, 1.80);
        const double auxiliaryScale = 0.85 + 0.15 * regionScale;
        for (size_t aux = 0; aux < layout.auxiliaryChannels; ++aux) {
            for (size_t band = 0; band < layout.frequencyBandCount; ++band) {
                const size_t index =
                    region * layout.perRegionEntries +
                    layout.perRegionOrientationEntries +
                    aux * layout.frequencyBandCount + band;
                part[index] *= auxiliaryScale;
            }
        }
    }

    return part;
}

std::vector<PatternSlice> buildRetinaProjectionSlices(const RetinaAdapter& retina,
                                                      const PatternSlice& branchSlice) {
    std::vector<PatternSlice> projections;
    const size_t totalSize = branchSlice.size;
    if (totalSize == 0) {
        return projections;
    }

    const size_t gridSize = static_cast<size_t>(std::max(1, retina.getIntParam("grid_size", 1)));
    const size_t regionCount = gridSize * gridSize;
    const size_t numOrientations =
        static_cast<size_t>(std::max(1, retina.getIntParam("num_orientations", 8)));
    const int subfieldGridSize = std::max(1, retina.getIntParam("subfield_grid_size", 1));
    const bool subfieldIncludePooled = retina.getIntParam("subfield_include_pooled", 1) != 0;
    const size_t subfieldCount =
        subfieldGridSize <= 1 ? 0u : static_cast<size_t>(subfieldGridSize * subfieldGridSize);
    const size_t orientationBlocks =
        subfieldCount == 0 ? 1u : (subfieldIncludePooled ? 1u + subfieldCount : subfieldCount);
    const size_t colorEdgeChannels =
        toLower(retina.getStringParam("color_edge_mode", "none")) == "opponent" ? 3u : 1u;
    const size_t auxiliaryChannels = inferAuxiliaryChannelCount(retina);
    const size_t frequencyBandCount = inferFrequencyBandCount(retina);
    const size_t orientationFeatureCount =
        numOrientations * orientationBlocks * colorEdgeChannels;
    const size_t perRegionOrientationEntries = orientationFeatureCount * frequencyBandCount;
    const size_t perRegionAuxEntries = auxiliaryChannels * frequencyBandCount;
    const size_t perRegionEntries = perRegionOrientationEntries + perRegionAuxEntries;
    if (perRegionEntries == 0 || regionCount * perRegionEntries != totalSize) {
        return projections;
    }

    PatternSlice orientationSlice;
    orientationSlice.name = branchSlice.name + "/orientation";
    PatternSlice auxiliarySlice;
    auxiliarySlice.name = branchSlice.name + "/auxiliary";
    PatternSlice lowBandSlice;
    lowBandSlice.name = branchSlice.name + "/low_band";
    PatternSlice highBandSlice;
    highBandSlice.name = branchSlice.name + "/high_band";

    orientationSlice.indices.reserve(regionCount * perRegionOrientationEntries);
    auxiliarySlice.indices.reserve(regionCount * perRegionAuxEntries);
    lowBandSlice.indices.reserve(regionCount * (orientationFeatureCount + auxiliaryChannels));
    highBandSlice.indices.reserve(regionCount * (orientationFeatureCount + auxiliaryChannels));

    for (size_t region = 0; region < regionCount; ++region) {
        const size_t regionBase = branchSlice.offset + region * perRegionEntries;
        for (size_t orientChannel = 0; orientChannel < orientationFeatureCount; ++orientChannel) {
            const size_t orientBase = regionBase + orientChannel * frequencyBandCount;
            for (size_t band = 0; band < frequencyBandCount; ++band) {
                orientationSlice.indices.push_back(orientBase + band);
            }
            lowBandSlice.indices.push_back(orientBase);
            highBandSlice.indices.push_back(orientBase + (frequencyBandCount - 1));
        }
        for (size_t auxChannel = 0; auxChannel < auxiliaryChannels; ++auxChannel) {
            const size_t auxBase =
                regionBase + perRegionOrientationEntries + auxChannel * frequencyBandCount;
            for (size_t band = 0; band < frequencyBandCount; ++band) {
                auxiliarySlice.indices.push_back(auxBase + band);
            }
            lowBandSlice.indices.push_back(auxBase);
            highBandSlice.indices.push_back(auxBase + (frequencyBandCount - 1));
        }
    }

    projections.push_back(std::move(orientationSlice));
    if (auxiliaryChannels > 0) {
        projections.push_back(std::move(auxiliarySlice));
    }
    if (frequencyBandCount > 1) {
        projections.push_back(std::move(lowBandSlice));
        projections.push_back(std::move(highBandSlice));
    }
    return projections;
}

std::vector<std::vector<double>> computeClassCentroids(
    const std::vector<ClassificationStrategy::LabeledPattern>& patterns,
    int numClasses) {
    std::vector<std::vector<double>> centroids(static_cast<size_t>(std::max(0, numClasses)));
    std::vector<int> counts(static_cast<size_t>(std::max(0, numClasses)), 0);
    for (const auto& labeledPattern : patterns) {
        if (labeledPattern.label < 0 || labeledPattern.label >= numClasses) {
            continue;
        }
        auto& centroid = centroids[static_cast<size_t>(labeledPattern.label)];
        if (centroid.empty()) {
            centroid.assign(labeledPattern.pattern.size(), 0.0);
        }
        if (centroid.size() != labeledPattern.pattern.size()) {
            continue;
        }
        for (size_t i = 0; i < centroid.size(); ++i) {
            centroid[i] += labeledPattern.pattern[i];
        }
        counts[static_cast<size_t>(labeledPattern.label)]++;
    }
    for (size_t label = 0; label < centroids.size(); ++label) {
        const int count = counts[label];
        if (count <= 0) {
            continue;
        }
        for (double& value : centroids[label]) {
            value /= static_cast<double>(count);
        }
        normalizeL2(centroids[label]);
    }
    return centroids;
}

std::vector<std::vector<std::vector<double>>> computeBranchCentroids(
    const std::vector<ClassificationStrategy::LabeledPattern>& patterns,
    const std::vector<PatternSlice>& slices,
    int numClasses) {
    std::vector<std::vector<std::vector<double>>> centroids(
        slices.size(),
        std::vector<std::vector<double>>(static_cast<size_t>(std::max(0, numClasses))));
    std::vector<std::vector<int>> counts(
        slices.size(), std::vector<int>(static_cast<size_t>(std::max(0, numClasses)), 0));

    for (const auto& labeledPattern : patterns) {
        if (labeledPattern.label < 0 || labeledPattern.label >= numClasses) {
            continue;
        }
        const size_t labelIndex = static_cast<size_t>(labeledPattern.label);
        for (size_t sliceIndex = 0; sliceIndex < slices.size(); ++sliceIndex) {
            const auto part = slicePattern(labeledPattern.pattern, slices[sliceIndex]);
            if (part.empty()) {
                continue;
            }
            auto& centroid = centroids[sliceIndex][labelIndex];
            if (centroid.empty()) {
                centroid.assign(part.size(), 0.0);
            }
            if (centroid.size() != part.size()) {
                continue;
            }
            for (size_t i = 0; i < part.size(); ++i) {
                centroid[i] += part[i];
            }
            counts[sliceIndex][labelIndex]++;
        }
    }

    for (size_t sliceIndex = 0; sliceIndex < centroids.size(); ++sliceIndex) {
        for (size_t labelIndex = 0; labelIndex < centroids[sliceIndex].size(); ++labelIndex) {
            const int count = counts[sliceIndex][labelIndex];
            if (count <= 0) {
                continue;
            }
            for (double& value : centroids[sliceIndex][labelIndex]) {
                value /= static_cast<double>(count);
            }
            normalizeL2(centroids[sliceIndex][labelIndex]);
        }
    }
    return centroids;
}

struct CentroidAuditMetrics {
    int predicted = -1;
    double ownScore = 0.0;
    double bestOtherScore = 0.0;
    double margin = 0.0;
};

struct NeighborAuditMetrics {
    int bestNeighborLabel = -1;
    double bestNeighborSimilarity = 0.0;
    double topkPurity = 0.0;
};

CentroidAuditMetrics computeCentroidAuditMetrics(const std::vector<double>& pattern,
                                                 int truth,
                                                 const std::vector<std::vector<double>>& centroids) {
    CentroidAuditMetrics metrics;
    if (pattern.empty() || truth < 0 || static_cast<size_t>(truth) >= centroids.size()) {
        return metrics;
    }

    const auto& ownCentroid = centroids[static_cast<size_t>(truth)];
    if (ownCentroid.empty()) {
        return metrics;
    }

    metrics.ownScore = cosineSimilarity(pattern, ownCentroid);
    double bestScore = -std::numeric_limits<double>::infinity();
    for (size_t label = 0; label < centroids.size(); ++label) {
        if (centroids[label].empty()) {
            continue;
        }
        const double score = cosineSimilarity(pattern, centroids[label]);
        if (static_cast<int>(label) != truth) {
            metrics.bestOtherScore = std::max(metrics.bestOtherScore, score);
        }
        if (score > bestScore) {
            bestScore = score;
            metrics.predicted = static_cast<int>(label);
        }
    }
    metrics.margin = metrics.ownScore - metrics.bestOtherScore;
    return metrics;
}

NeighborAuditMetrics computeNeighborAuditMetrics(const HemisphereRuntime& hemisphere,
                                                 const std::vector<double>& pattern,
                                                 int truth,
                                                 int k) {
    NeighborAuditMetrics metrics;
    if (pattern.empty() || hemisphere.trainingPatterns.empty()) {
        return metrics;
    }

    std::vector<std::pair<int, double>> neighbors;
    neighbors.reserve(hemisphere.trainingPatterns.size());
    for (size_t i = 0; i < hemisphere.trainingPatterns.size(); ++i) {
        neighbors.emplace_back(static_cast<int>(i),
                               cosineSimilarity(pattern, hemisphere.trainingPatterns[i].pattern));
    }

    const int actualK = std::min<int>(std::max(1, k), static_cast<int>(neighbors.size()));
    if (actualK <= 0) {
        return metrics;
    }

    std::partial_sort(neighbors.begin(), neighbors.begin() + actualK, neighbors.end(),
                      [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

    int truthMatches = 0;
    for (int i = 0; i < actualK; ++i) {
        const auto& [index, similarity] = neighbors[static_cast<size_t>(i)];
        const int label = hemisphere.trainingPatterns[static_cast<size_t>(index)].label;
        if (i == 0) {
            metrics.bestNeighborLabel = label;
            metrics.bestNeighborSimilarity = similarity;
        }
        if (label == truth) {
            ++truthMatches;
        }
    }
    metrics.topkPurity =
        static_cast<double>(truthMatches) / static_cast<double>(std::max(1, actualK));
    return metrics;
}

FlowAuditRuntime buildFlowAuditRuntime(
    std::vector<HemisphereRuntime>& hemispheres,
    const std::vector<HemisphereTrainingArtifacts>& trainingArtifacts,
    const VisualDomainAdapter& loader,
    const TrainingSplit& split,
    const Config& config,
    const FusionRuntime& fusionRuntime) {
    FlowAuditRuntime runtime;
    runtime.enabled = config.flowAuditEnabled;
    runtime.sampleLimit = config.flowAuditSampleLimit;
    runtime.numClasses = config.numClasses;
    runtime.outputPrefix = defaultFlowAuditOutputPrefix(config);
    if (!runtime.enabled) {
        return runtime;
    }

    runtime.hemisphereBranches.resize(hemispheres.size());
    runtime.hemisphereCentroids.resize(hemispheres.size());
    for (size_t hemisphereIndex = 0;
         hemisphereIndex < hemispheres.size() && hemisphereIndex < trainingArtifacts.size();
         ++hemisphereIndex) {
        runtime.hemisphereCentroids[hemisphereIndex] =
            computeClassCentroids(hemispheres[hemisphereIndex].trainingPatterns, config.numClasses);
        const auto& artifact = trainingArtifacts[hemisphereIndex];
        runtime.hemisphereBranches[hemisphereIndex].reserve(artifact.flowAuditBranches.size());
        for (const auto& branch : artifact.flowAuditBranches) {
            FlowAuditBranchReference ref;
            ref.name = branch.name;
            ref.preCentroids = branch.preCentroids;
            ref.postCentroids = branch.postCentroids;
            ref.meanTrainingFixationVariance =
                branch.fixationVarianceSamples > 0
                    ? (branch.fixationVarianceSum /
                       static_cast<double>(branch.fixationVarianceSamples))
                    : 0.0;
            runtime.hemisphereBranches[hemisphereIndex].push_back(std::move(ref));
        }
    }

    std::vector<ClassificationStrategy::LabeledPattern> interactionPatterns;
    std::vector<ClassificationStrategy::LabeledPattern> confidenceConcatPatterns;
    std::vector<ClassificationStrategy::LabeledPattern> hemisphereConcatPatterns;
    if (!split.fusionIndices.empty()) {
        interactionPatterns.reserve(split.fusionIndices.size());
        confidenceConcatPatterns.reserve(split.fusionIndices.size());
        hemisphereConcatPatterns.reserve(split.fusionIndices.size());
        for (size_t index : split.fusionIndices) {
            const auto& image = loader.getStimulus(index);
            const int label = image.label;
            if (label < 0 || label >= config.numClasses) {
                continue;
            }
            const auto traces = inferHemisphereDecisions(hemispheres, image, config);
            interactionPatterns.emplace_back(buildFusionPatternFromTraces(traces, config), label);
            confidenceConcatPatterns.emplace_back(
                buildConfidenceConcatPatternFromTraces(traces), label);
            hemisphereConcatPatterns.emplace_back(
                buildHemisphereConcatPatternFromTraces(traces), label);
        }
    } else if (!fusionRuntime.trainingPatterns.empty()) {
        interactionPatterns = fusionRuntime.trainingPatterns;
    }

    runtime.fusionInteractionCentroids =
        computeClassCentroids(interactionPatterns, config.numClasses);
    runtime.fusionConfidenceConcatCentroids =
        computeClassCentroids(confidenceConcatPatterns, config.numClasses);
    runtime.fusionHemisphereConcatCentroids =
        computeClassCentroids(hemisphereConcatPatterns, config.numClasses);
    return runtime;
}

bool flowAuditShouldStoreRows(FlowAuditRuntime& runtime) {
    if (!runtime.enabled) {
        return false;
    }
    if (runtime.sampleLimit <= 0) {
        runtime.recordedSamples++;
        return true;
    }
    if (runtime.recordedSamples < runtime.sampleLimit) {
        runtime.recordedSamples++;
        return true;
    }
    return false;
}

void appendFlowAuditRow(FlowAuditRuntime& runtime, const FlowAuditCsvRow& row, bool storeRow) {
    if (!runtime.enabled) {
        return;
    }
    if (storeRow) {
        runtime.rows.push_back(row);
    } else {
        runtime.droppedRows++;
    }
}

void recordFlowAuditFusionAlternatives(FlowAuditRuntime& runtime,
                                       const BilateralDecisionTrace& decision,
                                       int truth) {
    if (!runtime.enabled || truth < 0 || truth >= runtime.numClasses) {
        return;
    }
    runtime.fusionAggregate.samples++;

    const auto interactionMetrics = computeCentroidAuditMetrics(
        decision.fusionPattern, truth, runtime.fusionInteractionCentroids);
    if (interactionMetrics.predicted == truth) {
        runtime.fusionAggregate.interactionCentroidCorrect++;
    }

    const auto confidenceConcatPattern =
        buildConfidenceConcatPatternFromTraces(decision.hemisphereTraces);
    const auto confidenceConcatMetrics = computeCentroidAuditMetrics(
        confidenceConcatPattern, truth, runtime.fusionConfidenceConcatCentroids);
    if (confidenceConcatMetrics.predicted == truth) {
        runtime.fusionAggregate.confidenceConcatCentroidCorrect++;
    }

    const auto hemisphereConcatPattern =
        buildHemisphereConcatPatternFromTraces(decision.hemisphereTraces);
    const auto hemisphereConcatMetrics = computeCentroidAuditMetrics(
        hemisphereConcatPattern, truth, runtime.fusionHemisphereConcatCentroids);
    if (hemisphereConcatMetrics.predicted == truth) {
        runtime.fusionAggregate.hemisphereConcatCentroidCorrect++;
    }
}

void populateFlowAuditTopHypotheses(const std::vector<LabelTrace>& topHypotheses,
                                    FlowAuditCsvRow& row) {
    if (!topHypotheses.empty()) {
        row.top1Label = topHypotheses.front().label;
        row.top1Score = topHypotheses.front().score;
    }
    if (topHypotheses.size() > 1) {
        row.top2Label = topHypotheses[1].label;
        row.top2Score = topHypotheses[1].score;
    }
    row.confidenceMargin = std::max(0.0, row.top1Score - row.top2Score);
}

void recordFlowAuditBranchCapture(FlowAuditRuntime& runtime,
                                  size_t sampleOrdinal,
                                  size_t imageIndex,
                                  int truth,
                                  int initialPredicted,
                                  int finalPredicted,
                                  const std::string& hemisphereName,
                                  const HemisphereDecisionTrace& hemisphereTrace,
                                  const FlowAuditSampleCapture& capture,
                                  size_t hemisphereIndex,
                                  bool storeRows) {
    if (!runtime.enabled || hemisphereIndex >= runtime.hemisphereBranches.size()) {
        return;
    }

    for (size_t branchIndex = 0;
         branchIndex < capture.parts.size() &&
         branchIndex < runtime.hemisphereBranches[hemisphereIndex].size();
         ++branchIndex) {
        const auto& partCapture = capture.parts[branchIndex];
        const auto& branchRef = runtime.hemisphereBranches[hemisphereIndex][branchIndex];
        const auto preMetrics = computeCentroidAuditMetrics(
            partCapture.preNormPattern, truth, branchRef.preCentroids);
        const auto postMetrics = computeCentroidAuditMetrics(
            partCapture.postNormPattern, truth, branchRef.postCentroids);

        auto& aggregate = runtime.branchAggregates[hemisphereName + "/" + branchRef.name];
        aggregate.samples++;
        aggregate.zeroPreSamples += partCapture.preNorm.l2 <= 1e-6 ? 1 : 0;
        aggregate.zeroPostSamples += partCapture.postNorm.l2 <= 1e-6 ? 1 : 0;
        aggregate.preMarginSum += preMetrics.margin;
        aggregate.postMarginSum += postMetrics.margin;
        aggregate.preActiveFractionSum += partCapture.preNorm.activeFraction;
        aggregate.postActiveFractionSum += partCapture.postNorm.activeFraction;
        aggregate.preNormL2Sum += partCapture.preNorm.l2;
        aggregate.postNormL2Sum += partCapture.postNorm.l2;
        aggregate.preOrientationL2Sum += partCapture.preOrientationL2;
        aggregate.preAuxiliaryL2Sum += partCapture.preAuxiliaryL2;
        aggregate.postOrientationL2Sum += partCapture.postOrientationL2;
        aggregate.postAuxiliaryL2Sum += partCapture.postAuxiliaryL2;
        aggregate.fixationVarianceSum += partCapture.fixationVariance;
        aggregate.fixationConsistencySum += partCapture.fixationConsistency;
        aggregate.meanTrainingFixationVariance = branchRef.meanTrainingFixationVariance;

        FlowAuditCsvRow row;
        row.sampleOrdinal = sampleOrdinal;
        row.imageIndex = imageIndex;
        row.truth = truth;
        row.stage = "branch";
        row.hemisphere = hemisphereName;
        row.component = branchRef.name;
        row.predicted = hemisphereTrace.predicted;
        row.initialPredicted = initialPredicted;
        row.finalPredicted = finalPredicted;
        populateFlowAuditTopHypotheses(hemisphereTrace.topHypotheses, row);
        row.preNormMeanAbs = partCapture.preNorm.meanAbs;
        row.preNormL2 = partCapture.preNorm.l2;
        row.preNormActiveFraction = partCapture.preNorm.activeFraction;
        row.preOrientationL2 = partCapture.preOrientationL2;
        row.preAuxiliaryL2 = partCapture.preAuxiliaryL2;
        row.preOwnScore = preMetrics.ownScore;
        row.preOtherScore = preMetrics.bestOtherScore;
        row.preMargin = preMetrics.margin;
        row.postNormMeanAbs = partCapture.postNorm.meanAbs;
        row.postNormL2 = partCapture.postNorm.l2;
        row.postNormActiveFraction = partCapture.postNorm.activeFraction;
        row.postOrientationL2 = partCapture.postOrientationL2;
        row.postAuxiliaryL2 = partCapture.postAuxiliaryL2;
        row.postOwnScore = postMetrics.ownScore;
        row.postOtherScore = postMetrics.bestOtherScore;
        row.postMargin = postMetrics.margin;
        row.centroidPredicted = postMetrics.predicted;
        row.fixationCount = partCapture.fixationCount;
        row.fixationVariance = partCapture.fixationVariance;
        row.fixationConsistency = partCapture.fixationConsistency;
        appendFlowAuditRow(runtime, row, storeRows);
    }
}

void recordFlowAuditHemisphereCapture(FlowAuditRuntime& runtime,
                                      size_t sampleOrdinal,
                                      size_t imageIndex,
                                      int truth,
                                      int initialPredicted,
                                      int finalPredicted,
                                      const std::string& hemisphereName,
                                      const HemisphereDecisionTrace& trace,
                                      const FlowAuditSampleCapture* sampleCapture,
                                      const HemisphereRuntime& hemisphere,
                                      size_t hemisphereIndex,
                                      const Config& config,
                                      bool storeRows) {
    if (!runtime.enabled || hemisphereIndex >= runtime.hemisphereCentroids.size()) {
        return;
    }

    const auto& hemispherePattern =
        trace.classifierPattern.empty() ? trace.pattern : trace.classifierPattern;
    const auto centroidMetrics = computeCentroidAuditMetrics(
        hemispherePattern, truth, runtime.hemisphereCentroids[hemisphereIndex]);
    const auto neighborMetrics = computeNeighborAuditMetrics(
        hemisphere, hemispherePattern, truth, config.stage1K > 0 ? config.stage1K : config.knnK);

    auto& aggregate = runtime.hemisphereAggregates[hemisphereName];
    aggregate.samples++;
    aggregate.confidenceMarginSum += trace.margin;
    aggregate.centroidMarginSum += centroidMetrics.margin;
    aggregate.topkPuritySum += neighborMetrics.topkPurity;
    aggregate.bestNeighborSimilaritySum += neighborMetrics.bestNeighborSimilarity;
    const int fixationCount = sampleCapture != nullptr
                                  ? static_cast<int>(std::max<size_t>(
                                        1u, sampleCapture->fixationPatterns.size()))
                                  : 1;
    const double fixationVariance =
        sampleCapture != nullptr
            ? computeFlowAuditFixationVariance(sampleCapture->fixationPatterns)
            : 0.0;
    const double fixationConsistency =
        sampleCapture != nullptr ? sampleCapture->fixationConsistency : 1.0;
    aggregate.fixationConsistencySum += fixationConsistency;

    FlowAuditCsvRow row;
    row.sampleOrdinal = sampleOrdinal;
    row.imageIndex = imageIndex;
    row.truth = truth;
    row.stage = "hemisphere";
    row.hemisphere = hemisphereName;
    row.component = "combined";
    row.predicted = trace.predicted;
    row.initialPredicted = initialPredicted;
    row.finalPredicted = finalPredicted;
    populateFlowAuditTopHypotheses(trace.topHypotheses, row);
    const auto postStats = computeFlowAuditVectorStats(hemispherePattern);
    row.postNormMeanAbs = postStats.meanAbs;
    row.postNormL2 = postStats.l2;
    row.postNormActiveFraction = postStats.activeFraction;
    row.postOwnScore = centroidMetrics.ownScore;
    row.postOtherScore = centroidMetrics.bestOtherScore;
    row.postMargin = centroidMetrics.margin;
    row.centroidPredicted = centroidMetrics.predicted;
    row.bestNeighborLabel = neighborMetrics.bestNeighborLabel;
    row.bestNeighborSimilarity = neighborMetrics.bestNeighborSimilarity;
    row.topkPurity = neighborMetrics.topkPurity;
    row.fixationCount = fixationCount;
    row.fixationVariance = fixationVariance;
    row.fixationConsistency = fixationConsistency;
    appendFlowAuditRow(runtime, row, storeRows);

    if (!trace.figureGroundPattern.empty()) {
        const auto figureGroundStats = computeFlowAuditVectorStats(trace.figureGroundPattern);
        const auto figureGroundState =
            buildHemisphereFigureGroundState(hemisphere, trace.pattern, config);
        if (figureGroundState.valid) {
            const double fgFixationConsistency =
                computeFigureGroundFixationConsistency(hemisphere, sampleCapture, config);
            auto& fgAggregate = runtime.figureGroundAggregates[hemisphereName];
            fgAggregate.samples++;
            fgAggregate.ownedBorderMassSum += figureGroundState.ownedBorderMass;
            fgAggregate.surfaceMassSum += figureGroundState.surfaceMass;
            fgAggregate.junctionMassSum += figureGroundState.junctionMass;
            fgAggregate.borderSurfaceRatioSum += figureGroundState.borderSurfaceRatio;
            fgAggregate.leftRightBalanceSum += figureGroundState.leftRightBalance;
            fgAggregate.componentCountSum +=
                static_cast<double>(figureGroundState.componentCount);
            fgAggregate.fixationConsistencySum += fixationConsistency;
            fgAggregate.fgFixationConsistencySum += fgFixationConsistency;

            FlowAuditCsvRow fgRow;
            fgRow.sampleOrdinal = sampleOrdinal;
            fgRow.imageIndex = imageIndex;
            fgRow.truth = truth;
            fgRow.stage = "figure_ground";
            fgRow.hemisphere = hemisphereName;
            fgRow.component = "explicit_state";
            fgRow.predicted = trace.predicted;
            fgRow.initialPredicted = initialPredicted;
            fgRow.finalPredicted = finalPredicted;
            populateFlowAuditTopHypotheses(trace.topHypotheses, fgRow);
            fgRow.postNormMeanAbs = figureGroundStats.meanAbs;
            fgRow.postNormL2 = figureGroundStats.l2;
            fgRow.postNormActiveFraction = figureGroundStats.activeFraction;
            fgRow.fgOwnedBorderMass = figureGroundState.ownedBorderMass;
            fgRow.fgSurfaceMass = figureGroundState.surfaceMass;
            fgRow.fgJunctionMass = figureGroundState.junctionMass;
            fgRow.fgBorderSurfaceRatio = figureGroundState.borderSurfaceRatio;
            fgRow.fgLeftRightBalance = figureGroundState.leftRightBalance;
            fgRow.fgComponentCount = figureGroundState.componentCount;
            fgRow.fixationCount = fixationCount;
            fgRow.fixationVariance = fixationVariance;
            fgRow.fixationConsistency = fixationConsistency;
            fgRow.fgFixationConsistency = fgFixationConsistency;
            appendFlowAuditRow(runtime, fgRow, storeRows);
        }
    }
}

void recordFlowAuditFusionRow(FlowAuditRuntime& runtime,
                              size_t sampleOrdinal,
                              size_t imageIndex,
                              int truth,
                              int initialPredicted,
                              int finalPredicted,
                              const BilateralDecisionTrace& decision,
                              bool storeRows) {
    if (!runtime.enabled) {
        return;
    }

    const auto centroidMetrics = computeCentroidAuditMetrics(
        decision.fusionPattern, truth, runtime.fusionInteractionCentroids);
    FlowAuditCsvRow row;
    row.sampleOrdinal = sampleOrdinal;
    row.imageIndex = imageIndex;
    row.truth = truth;
    row.stage = "fusion";
    row.hemisphere = "bilateral";
    row.component = "interaction";
    row.predicted = decision.predicted;
    row.initialPredicted = initialPredicted;
    row.finalPredicted = finalPredicted;
    populateFlowAuditTopHypotheses(decision.topHypotheses, row);
    const auto postStats = computeFlowAuditVectorStats(decision.fusionPattern);
    row.postNormMeanAbs = postStats.meanAbs;
    row.postNormL2 = postStats.l2;
    row.postNormActiveFraction = postStats.activeFraction;
    row.postOwnScore = centroidMetrics.ownScore;
    row.postOtherScore = centroidMetrics.bestOtherScore;
    row.postMargin = centroidMetrics.margin;
    row.centroidPredicted = centroidMetrics.predicted;
    appendFlowAuditRow(runtime, row, storeRows);
}

void recordFlowAuditReplayMechanisms(FlowAuditRuntime& runtime,
                                     const BilateralDecisionTrace& replayDecision,
                                     bool replaySucceeded) {
    if (!runtime.enabled) {
        return;
    }

    if (replaySucceeded) {
        runtime.replayAggregate.replaySuccessSamples++;
        for (const auto& trace : replayDecision.hemisphereTraces) {
            if (trace.predicted < 0) {
                continue;
            }
            runtime.replayAggregate.positiveClassWeightUpdates++;
            runtime.replayAggregate.positiveCentroidUpdates++;
            runtime.replayAggregate.positiveExemplarInsertions++;
            if (trace.predicted == replayDecision.predicted) {
                runtime.replayAggregate.positivePredictionWeightUpdates++;
            }
        }
        if (replayDecision.predicted >= 0 && !replayDecision.fusionPattern.empty()) {
            runtime.replayAggregate.positiveFusionExemplarInsertions++;
        }
        return;
    }

    runtime.replayAggregate.replayFailureSamples++;
    for (const auto& trace : replayDecision.hemisphereTraces) {
        if (trace.predicted < 0) {
            continue;
        }
        runtime.replayAggregate.negativeClassWeightUpdates++;
        if (trace.predicted == replayDecision.predicted) {
            runtime.replayAggregate.negativePredictionWeightUpdates++;
        }
    }
}

void updateSeparabilityStats(EvaluationResult::SeparabilityStats& stats,
                             const std::vector<double>& pattern,
                             const std::vector<double>& rawPattern,
                             int truth,
                             const std::vector<std::vector<double>>& centroids) {
    if (truth < 0 || static_cast<size_t>(truth) >= centroids.size() || pattern.empty()) {
        return;
    }
    const auto& ownCentroid = centroids[static_cast<size_t>(truth)];
    if (ownCentroid.empty()) {
        return;
    }

    double ownScore = cosineSimilarity(pattern, ownCentroid);
    double bestScore = ownScore;
    double bestOther = -1.0;
    int bestLabel = truth;
    for (size_t label = 0; label < centroids.size(); ++label) {
        if (static_cast<int>(label) == truth || centroids[label].empty()) {
            continue;
        }
        const double score = cosineSimilarity(pattern, centroids[label]);
        if (score > bestOther) {
            bestOther = score;
        }
        if (score > bestScore) {
            bestScore = score;
            bestLabel = static_cast<int>(label);
        }
    }

    stats.samples++;
    stats.ownScoreSum += ownScore;
    stats.otherScoreSum += std::max(0.0, bestOther);
    stats.marginSum += ownScore - std::max(0.0, bestOther);
    bool nonzeroRawPattern = false;
    if (!rawPattern.empty()) {
        double l1 = 0.0;
        double l2 = 0.0;
        int active = 0;
        for (double value : rawPattern) {
            const double absValue = std::abs(value);
            l1 += absValue;
            l2 += value * value;
            if (absValue > 1e-6) {
                active++;
                nonzeroRawPattern = true;
            }
        }
        stats.rawL1Sum += l1 / static_cast<double>(rawPattern.size());
        stats.rawL2Sum += std::sqrt(l2);
        stats.rawActiveFractionSum +=
            static_cast<double>(active) / static_cast<double>(rawPattern.size());
    }
    if (nonzeroRawPattern) {
        stats.nonzeroSamples++;
    } else {
        stats.zeroVectorSamples++;
    }
    if (truth >= 0 &&
        static_cast<size_t>(truth) < stats.centroidConfusion.size() &&
        bestLabel >= 0 &&
        static_cast<size_t>(bestLabel) < stats.centroidConfusion[static_cast<size_t>(truth)].size()) {
        stats.centroidConfusion[static_cast<size_t>(truth)][static_cast<size_t>(bestLabel)]++;
    }
    if (bestLabel == truth) {
        stats.centroidCorrect++;
        if (nonzeroRawPattern) {
            stats.centroidCorrectNonzero++;
        }
    }
}

int classifyByCentroid(const std::vector<double>& pattern,
                       const std::vector<std::vector<double>>& centroids) {
    if (pattern.empty()) {
        return -1;
    }
    double bestScore = -std::numeric_limits<double>::infinity();
    int bestLabel = -1;
    for (size_t label = 0; label < centroids.size(); ++label) {
        if (centroids[label].empty()) {
            continue;
        }
        const double score = cosineSimilarity(pattern, centroids[label]);
        if (score > bestScore) {
            bestScore = score;
            bestLabel = static_cast<int>(label);
        }
    }
    return bestLabel;
}

SeparabilityDiagnosticsRuntime buildSeparabilityDiagnosticsRuntime(
    std::vector<HemisphereRuntime>& hemispheres,
    const VisualDomainAdapter& loader,
    const TrainingSplit& split,
    const Config& config,
    const FusionRuntime& fusionRuntime) {
    SeparabilityDiagnosticsRuntime runtime;
    runtime.enabled = config.separabilityDiagnostics;
    runtime.sampleLimit = config.separabilitySampleLimit;
    runtime.numClasses = config.numClasses;
    if (!runtime.enabled || hemispheres.empty()) {
        return runtime;
    }

    size_t sampleIndex = 0;
    if (!split.stage1Indices.empty()) {
        sampleIndex = split.stage1Indices.front();
    } else if (!split.fusionIndices.empty()) {
        sampleIndex = split.fusionIndices.front();
    } else if (loader.size() > 0) {
        sampleIndex = 0;
    } else {
        runtime.enabled = false;
        return runtime;
    }

    const auto& sampleImage = loader.getStimulus(sampleIndex);
    runtime.hemisphereSlices.reserve(hemispheres.size());
    runtime.hemisphereCentroids.reserve(hemispheres.size());
    runtime.branchCentroids.reserve(hemispheres.size());

    for (auto& hemisphere : hemispheres) {
        auto slices = buildPatternSlices(hemisphere.retinas, sampleImage, config.useFeatures);
        const auto& diagnosticPatterns =
            hemisphere.rawTrainingPatterns.empty() ? hemisphere.trainingPatterns
                                                   : hemisphere.rawTrainingPatterns;
        runtime.branchCentroids.push_back(
            computeBranchCentroids(diagnosticPatterns, slices, config.numClasses));
        runtime.hemisphereCentroids.push_back(
            computeClassCentroids(diagnosticPatterns, config.numClasses));
        runtime.hemisphereSlices.push_back(std::move(slices));
    }

    if (!fusionRuntime.trainingPatterns.empty()) {
        runtime.fusionCentroids =
            computeClassCentroids(fusionRuntime.trainingPatterns, config.numClasses);
    } else if (!split.fusionIndices.empty()) {
        std::vector<ClassificationStrategy::LabeledPattern> fusionPatterns;
        fusionPatterns.reserve(split.fusionIndices.size());
        for (size_t index : split.fusionIndices) {
            const auto& image = loader.getStimulus(index);
            const int label = image.label;
            if (label < 0 || label >= config.numClasses) {
                continue;
            }
            fusionPatterns.emplace_back(buildFusionPattern(hemispheres, image, config), label);
        }
        runtime.fusionCentroids = computeClassCentroids(fusionPatterns, config.numClasses);
    }

    return runtime;
}

void initializeSeparabilityStats(EvaluationResult& result,
                                 const SeparabilityDiagnosticsRuntime& runtime,
                                 const std::vector<HemisphereRuntime>& hemispheres) {
    if (!runtime.enabled) {
        return;
    }

    result.separabilityStats.clear();
    for (size_t hemisphereIndex = 0; hemisphereIndex < runtime.hemisphereSlices.size(); ++hemisphereIndex) {
        for (const auto& slice : runtime.hemisphereSlices[hemisphereIndex]) {
            EvaluationResult::SeparabilityStats stats;
            stats.name = hemispheres[hemisphereIndex].name + "/" + slice.name;
            stats.centroidConfusion = makeClassMatrix<int>(runtime.numClasses, 0);
            stats.hemisphereWrongTargetCounts.assign(static_cast<size_t>(runtime.numClasses), 0);
            result.separabilityStats.push_back(std::move(stats));
        }
        EvaluationResult::SeparabilityStats combined;
        combined.name = hemispheres[hemisphereIndex].name + "/combined";
        combined.centroidConfusion = makeClassMatrix<int>(runtime.numClasses, 0);
        combined.hemisphereWrongTargetCounts.assign(static_cast<size_t>(runtime.numClasses), 0);
        result.separabilityStats.push_back(std::move(combined));
    }
    EvaluationResult::SeparabilityStats fusion;
    fusion.name = "fusion";
    fusion.centroidConfusion = makeClassMatrix<int>(runtime.numClasses, 0);
    fusion.hemisphereWrongTargetCounts.assign(static_cast<size_t>(runtime.numClasses), 0);
    result.separabilityStats.push_back(std::move(fusion));
}

void recordSeparabilityDiagnostics(EvaluationResult& result,
                                   SeparabilityDiagnosticsRuntime& runtime,
                                   const std::vector<HemisphereRuntime>& hemispheres,
                                   const BilateralDecisionTrace& decision,
                                   int truth) {
    if (!runtime.enabled) {
        return;
    }
    if (runtime.sampleLimit > 0 && runtime.recordedSamples >= runtime.sampleLimit) {
        return;
    }

    size_t statIndex = 0;
    for (size_t hemisphereIndex = 0;
         hemisphereIndex < decision.hemisphereTraces.size() &&
         hemisphereIndex < runtime.hemisphereSlices.size();
         ++hemisphereIndex) {
        const auto& pattern = decision.hemisphereTraces[hemisphereIndex].pattern;
        const auto& slices = runtime.hemisphereSlices[hemisphereIndex];
        const int hemispherePredicted = decision.hemisphereTraces[hemisphereIndex].predicted;
        for (size_t sliceIndex = 0; sliceIndex < slices.size(); ++sliceIndex) {
            auto rawPart = slicePatternRaw(pattern, slices[sliceIndex]);
            auto part = rawPart;
            normalizeL2(part);
            updateSeparabilityStats(result.separabilityStats[statIndex],
                                    part,
                                    rawPart,
                                    truth,
                                    runtime.branchCentroids[hemisphereIndex][sliceIndex]);
            if (hemispherePredicted >= 0 && hemispherePredicted != truth) {
                auto& stats = result.separabilityStats[statIndex];
                stats.hemisphereWrongSamples++;
                const int branchPredicted = classifyByCentroid(
                    part, runtime.branchCentroids[hemisphereIndex][sliceIndex]);
                if (branchPredicted >= 0 &&
                    static_cast<size_t>(branchPredicted) < stats.hemisphereWrongTargetCounts.size()) {
                    stats.hemisphereWrongTargetCounts[static_cast<size_t>(branchPredicted)]++;
                    if (branchPredicted == hemispherePredicted) {
                        stats.branchSupportsHemisphereWrong++;
                    }
                    if (branchPredicted == truth) {
                        stats.branchSupportsTruth++;
                    }
                }
            }
            statIndex++;
        }
        updateSeparabilityStats(result.separabilityStats[statIndex],
                                pattern,
                                pattern,
                                truth,
                                runtime.hemisphereCentroids[hemisphereIndex]);
        statIndex++;
    }

    if (statIndex < result.separabilityStats.size()) {
        updateSeparabilityStats(result.separabilityStats[statIndex],
                                decision.fusionPattern,
                                decision.fusionPattern,
                                truth,
                                runtime.fusionCentroids);
    }
    runtime.recordedSamples++;
}

IndexBuckets collectLabelIndices(const VisualDomainAdapter& loader, int numClasses) {
    IndexBuckets indicesByLabel(static_cast<size_t>(std::max(0, numClasses)));
    for (size_t i = 0; i < loader.size(); ++i) {
        const int label = loader.getStimulus(i).label;
        if (label >= 0 && static_cast<size_t>(label) < indicesByLabel.size()) {
            indicesByLabel[static_cast<size_t>(label)].push_back(i);
        }
    }
    return indicesByLabel;
}

std::vector<size_t> selectStratifiedIndices(const IndexBuckets& indicesByLabel,
                                            int maxPerClass,
                                            unsigned int seed) {
    std::mt19937 rng(seed);
    std::vector<size_t> selected;

    for (size_t label = 0; label < indicesByLabel.size(); ++label) {
        auto shuffled = indicesByLabel[label];
        std::shuffle(shuffled.begin(), shuffled.end(), rng);
        const int keep = (maxPerClass > 0)
            ? std::min<int>(maxPerClass, static_cast<int>(shuffled.size()))
            : static_cast<int>(shuffled.size());
        selected.insert(selected.end(), shuffled.begin(), shuffled.begin() + keep);
    }

    std::shuffle(selected.begin(), selected.end(), rng);
    return selected;
}

std::vector<size_t> selectBalancedTestIndices(const IndexBuckets& indicesByLabel,
                                              int testLimit,
                                              unsigned int seed) {
    if (testLimit <= 0) {
        return selectStratifiedIndices(indicesByLabel, 0, seed);
    }

    std::mt19937 rng(seed);
    IndexBuckets shuffledByLabel = indicesByLabel;
    std::vector<size_t> offsets(indicesByLabel.size(), 0);
    for (auto& indices : shuffledByLabel) {
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    std::vector<size_t> selected;
    selected.reserve(static_cast<size_t>(testLimit));

    // Round-robin keeps the sample balanced even when the limit is not divisible by class count.
    while (static_cast<int>(selected.size()) < testLimit) {
        bool addedAny = false;
        for (size_t label = 0; label < shuffledByLabel.size(); ++label) {
            if (static_cast<int>(selected.size()) >= testLimit) {
                break;
            }
            auto& indices = shuffledByLabel[label];
            if (offsets[label] < indices.size()) {
                selected.push_back(indices[offsets[label]++]);
                addedAny = true;
            }
        }
        if (!addedAny) {
            break;
        }
    }

    std::shuffle(selected.begin(), selected.end(), rng);
    return selected;
}

std::vector<size_t> selectFocusedTestIndices(const IndexBuckets& indicesByLabel,
                                             const std::vector<int>& focusLabels,
                                             int limitPerLabel,
                                             unsigned int seed) {
    if (focusLabels.empty()) {
        return {};
    }

    std::mt19937 rng(seed);
    IndexBuckets shuffledByLabel = indicesByLabel;
    std::vector<size_t> offsets(indicesByLabel.size(), 0);

    for (int label : focusLabels) {
        auto& indices = shuffledByLabel[static_cast<size_t>(label)];
        std::shuffle(indices.begin(), indices.end(), rng);
        if (limitPerLabel > 0 && static_cast<int>(indices.size()) > limitPerLabel) {
            indices.resize(static_cast<size_t>(limitPerLabel));
        }
    }

    std::vector<size_t> selected;
    while (true) {
        bool addedAny = false;
        for (int label : focusLabels) {
            auto& indices = shuffledByLabel[static_cast<size_t>(label)];
            auto& offset = offsets[static_cast<size_t>(label)];
            if (offset < indices.size()) {
                selected.push_back(indices[offset++]);
                addedAny = true;
            }
        }
        if (!addedAny) {
            break;
        }
    }

    return selected;
}

EvaluationResult evaluatePatterns(std::vector<std::unique_ptr<RetinaAdapter>>& retinas,
                                  const VisualDomainAdapter& loader,
                                  const std::vector<size_t>& indices,
                                  const Config& config,
                                  const ClassificationStrategy& classifier,
                                  const std::vector<ClassificationStrategy::LabeledPattern>& trainingPatterns,
                                  const std::string& label) {
    EvaluationResult result = makeEvaluationResult(config);
    const auto start = std::chrono::high_resolution_clock::now();
    const int maxTests = static_cast<int>(indices.size());
    const int progressStep = maxTests >= 1000 ? 200 : (maxTests >= 200 ? 50 : 25);

    for (size_t index : indices) {
        const auto& image = loader.getStimulus(index);
        const int truth = image.label;
        if (truth < 0 || truth >= config.numClasses) {
            continue;
        }

        const auto pattern = extractPattern(retinas, image, config, false, false);
        const int predicted = classifier.classify(pattern, trainingPatterns, cosineSimilarity);
        result.confusion[static_cast<size_t>(truth)][static_cast<size_t>(predicted)]++;
        result.correct += (predicted == truth) ? 1 : 0;
        result.tested++;

        if (result.tested % progressStep == 0) {
            const double acc =
                100.0 * static_cast<double>(result.correct) /
                static_cast<double>(std::max(1, result.tested));
            std::cout << "  " << label << ": " << result.tested << "/" << maxTests
                      << " (" << std::fixed << std::setprecision(2) << acc << "%)" << std::endl;
        }
    }

    const auto end = std::chrono::high_resolution_clock::now();
    result.seconds = std::chrono::duration<double>(end - start).count();
    return result;
}

EvaluationResult evaluateBilateralPatterns(std::vector<HemisphereRuntime>& hemispheres,
                                           const VisualDomainAdapter& loader,
                                           const std::vector<size_t>& indices,
                                           const Config& config,
                                           const ClassificationStrategy& fusionClassifier,
                                           FusionRuntime& fusionRuntime,
                                           SeparabilityDiagnosticsRuntime* separabilityRuntime,
                                           FlowAuditRuntime* flowAuditRuntime,
                                           const std::string& label) {
    EvaluationResult result = makeEvaluationResult(config);
    if (separabilityRuntime != nullptr) {
        initializeSeparabilityStats(result, *separabilityRuntime, hemispheres);
    }
    const auto start = std::chrono::high_resolution_clock::now();
    const int maxTests = static_cast<int>(indices.size());
    const int progressStep = maxTests >= 1000 ? 200 : (maxTests >= 200 ? 50 : 25);
    std::vector<OnlineSampleRecord> records;
    records.reserve(indices.size());
    std::vector<ReplayItem> replayQueue;
    replayQueue.reserve(static_cast<size_t>(std::max(16, config.onlineReplayQueueCapacity)));
    std::vector<ReplayItem> consolidationQueue;
    consolidationQueue.reserve(static_cast<size_t>(
        std::max(8, config.onlineReplaySuccessConsolidationCapacity)));
    size_t replaySequence = 0;
    ConfusionClusterMemory confusionMemory = makeConfusionClusterMemory(config);

    auto enqueueSuccessfulConsolidation = [&](size_t recordIndex,
                                              const DecisionContext& context,
                                              size_t currentStep) {
        if (!useOnlineReplaySuccessConsolidation(config)) {
            return;
        }
        enqueueReplayItemWithCapacity(
            consolidationQueue,
            makeReplayItem(recordIndex,
                           config.onlineReplaySuccessConsolidationRepeats,
                           context.replayPriority + 0.25 * context.plasticity,
                           currentStep,
                           computeNeuromodulatorEligibilityScale(context, true, config),
                           config,
                           replaySequence++),
            config.onlineReplaySuccessConsolidationCapacity);
    };

    auto processReplayItem = [&](ReplayItem item, size_t currentStep) {
        if (item.recordIndex >= records.size()) {
            return;
        }
        auto& record = records[item.recordIndex];
        const auto& replayImage = loader.getStimulus(record.imageIndex);
        auto replayDecision = inferFusionDecision(
            hemispheres, replayImage, config, fusionClassifier, fusionRuntime);
        if (replayDecision.predicted < 0 || replayDecision.predicted >= config.numClasses) {
            return;
        }

        record.finalPredicted = replayDecision.predicted;
        result.correctionReplays++;
        recordReplayTiming(result, item, currentStep);
        const auto replayContext = buildDecisionContext(replayDecision, config, &confusionMemory);
        result.confusionClusterScoreSum += replayContext.confusionCluster;
        result.confusionClusterScoreSamples++;
        const double effectiveReward =
            std::max(0.05, replayContext.plasticity * item.eligibilityScale);

        if (record.finalPredicted == record.truth) {
            if (flowAuditRuntime != nullptr) {
                recordFlowAuditReplayMechanisms(*flowAuditRuntime, replayDecision, true);
            }
            if (!record.correctionSucceeded) {
                result.correctionSuccesses++;
                record.correctionSucceeded = true;
            }
            for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
                applyRewardToHemisphere(hemispheres[hemisphereIndex],
                                        replayDecision.hemisphereTraces[hemisphereIndex],
                                        replayDecision.predicted,
                                        effectiveReward,
                                        config);
                applyRewardStdpToHemisphere(hemispheres[hemisphereIndex],
                                            replayDecision.hemisphereTraces[hemisphereIndex],
                                            replayDecision.predicted,
                                            effectiveReward,
                                            config);
                applyTripletStdpToHemisphere(hemispheres[hemisphereIndex],
                                             replayDecision.hemisphereTraces[hemisphereIndex],
                                             replayDecision.predicted,
                                             effectiveReward,
                                             currentStep,
                                             config);
                applyVoltagePlasticityToHemisphere(
                    hemispheres[hemisphereIndex],
                    replayDecision.hemisphereTraces[hemisphereIndex],
                    replayDecision.predicted,
                    effectiveReward,
                    currentStep,
                    config);
            }
            applyRewardToFusion(fusionRuntime,
                                replayDecision.fusionPattern,
                                replayDecision.topHypotheses,
                                replayDecision.predicted,
                                effectiveReward,
                                config);
            applyRewardStdpToFusion(fusionRuntime,
                                    replayDecision.fusionPattern,
                                    replayDecision.topHypotheses,
                                    replayDecision.predicted,
                                    effectiveReward,
                                    config);
            applyTripletStdpToFusion(fusionRuntime,
                                     replayDecision.fusionPattern,
                                     replayDecision.topHypotheses,
                                     replayDecision.predicted,
                                     effectiveReward,
                                     currentStep,
                                     config);
            applyVoltagePlasticityToFusion(fusionRuntime,
                                           replayDecision.fusionPattern,
                                           replayDecision.topHypotheses,
                                           replayDecision.predicted,
                                           effectiveReward,
                                           currentStep,
                                           config);
            updateConfusionClusterMemory(confusionMemory, replayDecision, false, config);
            enqueueSuccessfulConsolidation(item.recordIndex, replayContext, currentStep);
            return;
        }

        if (flowAuditRuntime != nullptr) {
            recordFlowAuditReplayMechanisms(*flowAuditRuntime, replayDecision, false);
        }

        for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
            applyRewardToHemisphere(hemispheres[hemisphereIndex],
                                    replayDecision.hemisphereTraces[hemisphereIndex],
                                    replayDecision.predicted,
                                    -effectiveReward,
                                    config);
            applyRewardStdpToHemisphere(hemispheres[hemisphereIndex],
                                        replayDecision.hemisphereTraces[hemisphereIndex],
                                        replayDecision.predicted,
                                        -effectiveReward,
                                        config);
            applyTripletStdpToHemisphere(hemispheres[hemisphereIndex],
                                         replayDecision.hemisphereTraces[hemisphereIndex],
                                         replayDecision.predicted,
                                         -effectiveReward,
                                         currentStep,
                                         config);
            applyVoltagePlasticityToHemisphere(
                hemispheres[hemisphereIndex],
                replayDecision.hemisphereTraces[hemisphereIndex],
                replayDecision.predicted,
                -effectiveReward,
                currentStep,
                config);
        }
        applyRewardStdpToFusion(fusionRuntime,
                                replayDecision.fusionPattern,
                                replayDecision.topHypotheses,
                                replayDecision.predicted,
                                -effectiveReward,
                                config);
        applyTripletStdpToFusion(fusionRuntime,
                                 replayDecision.fusionPattern,
                                 replayDecision.topHypotheses,
                                 replayDecision.predicted,
                                 -effectiveReward,
                                 currentStep,
                                 config);
        applyVoltagePlasticityToFusion(fusionRuntime,
                                       replayDecision.fusionPattern,
                                       replayDecision.topHypotheses,
                                       replayDecision.predicted,
                                       -effectiveReward,
                                       currentStep,
                                       config);
        updateConfusionClusterMemory(confusionMemory, replayDecision, true, config);
        if (item.remainingReplays > 1) {
            const double replayEligibilityScale = std::max(
                item.eligibilityScale,
                computeNeuromodulatorEligibilityScale(replayContext, false, config));
            enqueueReplayItem(replayQueue,
                              makeReplayItem(item.recordIndex,
                                             item.remainingReplays - 1,
                                             replayContext.replayPriority,
                                             currentStep,
                                             replayEligibilityScale,
                                             config,
                                             replaySequence++),
                              config);
        }
    };

    auto processConsolidationItem = [&](ReplayItem item, size_t currentStep) {
        if (item.recordIndex >= records.size()) {
            return;
        }
        auto& record = records[item.recordIndex];
        const auto& replayImage = loader.getStimulus(record.imageIndex);
        auto replayDecision = inferFusionDecision(
            hemispheres, replayImage, config, fusionClassifier, fusionRuntime);
        if (replayDecision.predicted < 0 || replayDecision.predicted >= config.numClasses ||
            replayDecision.predicted != record.truth) {
            return;
        }

        result.correctionReplays++;
        recordReplayTiming(result, item, currentStep);
        const auto replayContext = buildDecisionContext(replayDecision, config, &confusionMemory);
        result.confusionClusterScoreSum += replayContext.confusionCluster;
        result.confusionClusterScoreSamples++;
        const double effectiveReward =
            std::max(0.05, replayContext.plasticity * item.eligibilityScale);

        if (flowAuditRuntime != nullptr) {
            recordFlowAuditReplayMechanisms(*flowAuditRuntime, replayDecision, true);
        }
        for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
            applyRewardToHemisphere(hemispheres[hemisphereIndex],
                                    replayDecision.hemisphereTraces[hemisphereIndex],
                                    replayDecision.predicted,
                                    effectiveReward,
                                    config);
            applyRewardStdpToHemisphere(hemispheres[hemisphereIndex],
                                        replayDecision.hemisphereTraces[hemisphereIndex],
                                        replayDecision.predicted,
                                        effectiveReward,
                                        config);
            applyTripletStdpToHemisphere(hemispheres[hemisphereIndex],
                                         replayDecision.hemisphereTraces[hemisphereIndex],
                                         replayDecision.predicted,
                                         effectiveReward,
                                         currentStep,
                                         config);
            applyVoltagePlasticityToHemisphere(
                hemispheres[hemisphereIndex],
                replayDecision.hemisphereTraces[hemisphereIndex],
                replayDecision.predicted,
                effectiveReward,
                currentStep,
                config);
        }
        applyRewardToFusion(fusionRuntime,
                            replayDecision.fusionPattern,
                            replayDecision.topHypotheses,
                            replayDecision.predicted,
                            effectiveReward,
                            config);
        applyRewardStdpToFusion(fusionRuntime,
                                replayDecision.fusionPattern,
                                replayDecision.topHypotheses,
                                replayDecision.predicted,
                                effectiveReward,
                                config);
        applyTripletStdpToFusion(fusionRuntime,
                                 replayDecision.fusionPattern,
                                 replayDecision.topHypotheses,
                                 replayDecision.predicted,
                                 effectiveReward,
                                 currentStep,
                                 config);
        applyVoltagePlasticityToFusion(fusionRuntime,
                                       replayDecision.fusionPattern,
                                       replayDecision.topHypotheses,
                                       replayDecision.predicted,
                                       effectiveReward,
                                       currentStep,
                                       config);
        updateConfusionClusterMemory(confusionMemory, replayDecision, false, config);
        if (item.remainingReplays > 1) {
            enqueueReplayItemWithCapacity(
                consolidationQueue,
                makeReplayItem(item.recordIndex,
                               item.remainingReplays - 1,
                               replayContext.replayPriority + 0.25 * replayContext.plasticity,
                               currentStep,
                               item.eligibilityScale,
                               config,
                               replaySequence++),
                config.onlineReplaySuccessConsolidationCapacity);
        }
    };

    auto processReplayBudget = [&](size_t currentStep, int budget, bool flushAll = false) {
        if (!flushAll &&
            (currentStep == 0 || (currentStep % static_cast<size_t>(onlineReplayPauseInterval(config))) != 0)) {
            return;
        }
        const int batchBudget = flushAll ? budget : std::min(budget, onlineReplayBatchSize(config));
        ReplayItem item;
        int remaining = batchBudget;
        while (remaining-- > 0 && popReplayItem(replayQueue, item, currentStep, flushAll)) {
            processReplayItem(item, currentStep);
        }
    };

    auto processConsolidationBudget = [&](size_t currentStep, int budget, bool flushAll = false) {
        if (!useOnlineReplaySuccessConsolidation(config)) {
            return;
        }
        if (!flushAll &&
            (currentStep == 0 || (currentStep % static_cast<size_t>(onlineReplayPauseInterval(config))) != 0)) {
            return;
        }
        const int batchBudget = flushAll
            ? budget
            : std::min(budget, onlineReplaySuccessConsolidationBatchSize(config));
        ReplayItem item;
        int remaining = batchBudget;
        while (remaining-- > 0 && popReplayItem(consolidationQueue, item, currentStep, flushAll)) {
            processConsolidationItem(item, currentStep);
        }
    };

    auto currentAccuracy = [&]() {
        const int currentCorrect = static_cast<int>(std::count_if(
            records.begin(), records.end(), [](const OnlineSampleRecord& record) {
                return record.truth >= 0 && record.truth == record.finalPredicted;
            }));
        return 100.0 * static_cast<double>(currentCorrect) /
               static_cast<double>(std::max<size_t>(1, records.size()));
    };

    for (size_t index : indices) {
        const auto& image = loader.getStimulus(index);
        const int truth = image.label;
        if (truth < 0 || truth >= config.numClasses) {
            continue;
        }

        std::vector<FlowAuditSampleCapture> flowAuditCaptures;
        const bool captureBranchAudit =
            flowAuditRuntime != nullptr &&
            (!config.activeInferenceEnabled || config.activeInferenceFixations <= 1) &&
            !config.focusAdjustmentEnabled;
        auto decision = captureBranchAudit
            ? inferSingleViewFusionDecisionWithAudit(
                  hemispheres, image, config, fusionClassifier, fusionRuntime, flowAuditCaptures)
            : inferFusionDecision(hemispheres, image, config, fusionClassifier, fusionRuntime);
        const int initialPredicted = decision.predicted;
        if (initialPredicted < 0 || initialPredicted >= config.numClasses) {
            continue;
        }
        const int leftInitialPredicted =
            decision.hemisphereTraces.size() > 0 ? decision.hemisphereTraces[0].predicted : -1;
        const int rightInitialPredicted =
            decision.hemisphereTraces.size() > 1 ? decision.hemisphereTraces[1].predicted : -1;
        recordHemisphereAgreement(result, decision, truth);
        if (separabilityRuntime != nullptr) {
            recordSeparabilityDiagnostics(
                result, *separabilityRuntime, hemispheres, decision, truth);
        }
        decision = maybeApplyFocusAdjustment(
            image,
            decision,
            truth,
            config,
            result,
            [&](const VisualStimulus& focusedImage) {
                return inferFusionDecision(
                    hemispheres, focusedImage, config, fusionClassifier, fusionRuntime);
            });
        recordActiveInferenceUsage(result, decision, config);
        const size_t sampleOrdinal = records.size() + 1;
        const bool storeFlowAuditRows =
            flowAuditRuntime != nullptr ? flowAuditShouldStoreRows(*flowAuditRuntime) : false;
        if (flowAuditRuntime != nullptr) {
            recordFlowAuditFusionAlternatives(*flowAuditRuntime, decision, truth);
            if (captureBranchAudit && flowAuditCaptures.size() == hemispheres.size()) {
                for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
                    recordFlowAuditBranchCapture(*flowAuditRuntime,
                                                 sampleOrdinal,
                                                 index,
                                                 truth,
                                                 initialPredicted,
                                                 decision.predicted,
                                                 hemispheres[hemisphereIndex].name,
                                                 decision.hemisphereTraces[hemisphereIndex],
                                                 flowAuditCaptures[hemisphereIndex],
                                                 hemisphereIndex,
                                                 storeFlowAuditRows);
                    recordFlowAuditHemisphereCapture(*flowAuditRuntime,
                                                     sampleOrdinal,
                                                     index,
                                                     truth,
                                                     initialPredicted,
                                                     decision.predicted,
                                                     hemispheres[hemisphereIndex].name,
                                                     decision.hemisphereTraces[hemisphereIndex],
                                                     &flowAuditCaptures[hemisphereIndex],
                                                     hemispheres[hemisphereIndex],
                                                     hemisphereIndex,
                                                     config,
                                                     storeFlowAuditRows);
                }
            }
            recordFlowAuditFusionRow(*flowAuditRuntime,
                                     sampleOrdinal,
                                     index,
                                     truth,
                                     initialPredicted,
                                     decision.predicted,
                                     decision,
                                     storeFlowAuditRows);
        }
        records.push_back(
            {index, truth, leftInitialPredicted, rightInitialPredicted, initialPredicted,
             initialPredicted, false});
        records.back().finalPredicted = decision.predicted;
        const auto context = buildDecisionContext(decision, config, &confusionMemory);
        result.confusionClusterScoreSum += context.confusionCluster;
        result.confusionClusterScoreSamples++;

        if (useOnlineCorrection(config)) {
            if (initialPredicted == truth) {
                for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
                    applyRewardToHemisphere(hemispheres[hemisphereIndex],
                                            decision.hemisphereTraces[hemisphereIndex],
                                            decision.predicted,
                                            std::max(0.25, 0.5 * context.plasticity),
                                            config);
                    applyRewardStdpToHemisphere(hemispheres[hemisphereIndex],
                                                decision.hemisphereTraces[hemisphereIndex],
                                                decision.predicted,
                                                std::max(0.25, 0.5 * context.plasticity),
                                                config);
                    applyTripletStdpToHemisphere(hemispheres[hemisphereIndex],
                                                 decision.hemisphereTraces[hemisphereIndex],
                                                 decision.predicted,
                                                 std::max(0.25, 0.5 * context.plasticity),
                                                 records.size(),
                                                 config);
                    applyVoltagePlasticityToHemisphere(
                        hemispheres[hemisphereIndex],
                        decision.hemisphereTraces[hemisphereIndex],
                        decision.predicted,
                        std::max(0.25, 0.5 * context.plasticity),
                        records.size(),
                        config);
                }
                applyRewardToFusion(fusionRuntime,
                                    decision.fusionPattern,
                                    decision.topHypotheses,
                                    decision.predicted,
                                    std::max(0.25, 0.5 * context.plasticity),
                                    config);
                applyRewardStdpToFusion(fusionRuntime,
                                        decision.fusionPattern,
                                        decision.topHypotheses,
                                        decision.predicted,
                                        std::max(0.25, 0.5 * context.plasticity),
                                        config);
                applyTripletStdpToFusion(fusionRuntime,
                                         decision.fusionPattern,
                                         decision.topHypotheses,
                                         decision.predicted,
                                         std::max(0.25, 0.5 * context.plasticity),
                                         records.size(),
                                         config);
                applyVoltagePlasticityToFusion(fusionRuntime,
                                               decision.fusionPattern,
                                               decision.topHypotheses,
                                               decision.predicted,
                                               std::max(0.25, 0.5 * context.plasticity),
                                               records.size(),
                                               config);
                enqueueSuccessfulConsolidation(records.size() - 1, context, records.size());
            } else {
                result.correctionEvents++;
                for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
                    applyRewardToHemisphere(hemispheres[hemisphereIndex],
                                            decision.hemisphereTraces[hemisphereIndex],
                                            decision.predicted,
                                            -context.plasticity,
                                            config);
                    applyRewardStdpToHemisphere(hemispheres[hemisphereIndex],
                                                decision.hemisphereTraces[hemisphereIndex],
                                                decision.predicted,
                                                -context.plasticity,
                                                config);
                    applyTripletStdpToHemisphere(hemispheres[hemisphereIndex],
                                                 decision.hemisphereTraces[hemisphereIndex],
                                                 decision.predicted,
                                                 -context.plasticity,
                                                 records.size(),
                                                 config);
                    applyVoltagePlasticityToHemisphere(
                        hemispheres[hemisphereIndex],
                        decision.hemisphereTraces[hemisphereIndex],
                        decision.predicted,
                        -context.plasticity,
                        records.size(),
                        config);
                }
                applyRewardStdpToFusion(fusionRuntime,
                                        decision.fusionPattern,
                                        decision.topHypotheses,
                                        decision.predicted,
                                        -context.plasticity,
                                        config);
                applyTripletStdpToFusion(fusionRuntime,
                                         decision.fusionPattern,
                                         decision.topHypotheses,
                                         decision.predicted,
                                         -context.plasticity,
                                         records.size(),
                                         config);
                applyVoltagePlasticityToFusion(fusionRuntime,
                                               decision.fusionPattern,
                                               decision.topHypotheses,
                                               decision.predicted,
                                               -context.plasticity,
                                               records.size(),
                                               config);
                enqueueReplayItem(replayQueue,
                                  makeReplayItem(records.size() - 1,
                                                 config.onlineCorrectionRepeats,
                                                 context.replayPriority +
                                                     (context.uncertainty >=
                                                              config.onlineReplayUncertaintyThreshold
                                                          ? 0.5
                                                          : 0.0),
                                                 records.size(),
                                                 computeNeuromodulatorEligibilityScale(
                                                     context, false, config),
                                                 config,
                                                 replaySequence++),
                                  config);
            }
        }
        updateConfusionClusterMemory(confusionMemory, decision, decision.predicted != truth, config);

        processReplayBudget(records.size(), std::numeric_limits<int>::max());
        processConsolidationBudget(records.size(), std::numeric_limits<int>::max());

        if (static_cast<int>(records.size()) % progressStep == 0) {
            const double acc = currentAccuracy();
            std::cout << "  " << label << ": " << records.size() << "/" << maxTests
                      << " (" << std::fixed << std::setprecision(2) << acc << "%)" << std::endl;
        }
    }

    processReplayBudget(records.size() + static_cast<size_t>(onlineReplayDelaySteps(config)),
                        std::numeric_limits<int>::max(),
                        true);
    processConsolidationBudget(records.size() +
                                   static_cast<size_t>(onlineReplayDelaySteps(config)),
                               std::numeric_limits<int>::max(),
                               true);

    const auto end = std::chrono::high_resolution_clock::now();
    finalizeEvaluationFromRecords(
        result, records, std::chrono::duration<double>(end - start).count());
    result.confusionClusterStrengths = confusionMemory.strengths;
    return result;
}

int trainingProgressStep(size_t totalItems) {
    if (totalItems >= 10000) {
        return 1000;
    }
    if (totalItems >= 5000) {
        return 500;
    }
    if (totalItems >= 1000) {
        return 250;
    }
    return 100;
}

void maybePrintTrainingProgress(const std::string& label,
                                size_t completed,
                                size_t total,
                                int progressStep,
                                const std::chrono::high_resolution_clock::time_point& start) {
    if (completed == 0 || progressStep <= 0) {
        return;
    }
    if (completed != total && (completed % static_cast<size_t>(progressStep)) != 0) {
        return;
    }

    const double elapsedSeconds =
        std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
    std::cout << "  " << label << ": " << completed << "/" << total << " ("
              << std::fixed << std::setprecision(2) << elapsedSeconds << "s)" << std::endl;
}

void printTopConfusions(const IntMatrix& confusion, const Config& config) {
    struct PairConfusion {
        int a;
        int b;
        int total;
        int aToB;
        int bToA;
    };

    std::vector<PairConfusion> pairs;
    for (size_t i = 0; i < confusion.size(); ++i) {
        for (size_t j = i + 1; j < confusion.size(); ++j) {
            const int total = confusion[i][j] + confusion[j][i];
            if (total > 0) {
                pairs.push_back({static_cast<int>(i), static_cast<int>(j), total,
                                 confusion[i][j], confusion[j][i]});
            }
        }
    }

    std::sort(pairs.begin(), pairs.end(), [](const PairConfusion& lhs, const PairConfusion& rhs) {
        if (lhs.total != rhs.total) return lhs.total > rhs.total;
        if (lhs.aToB != rhs.aToB) return lhs.aToB > rhs.aToB;
        return lhs.bToA > rhs.bToA;
    });

    std::cout << "\nTop confusion pairs:" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(10, pairs.size()); ++i) {
        const auto& p = pairs[i];
        std::cout << "  " << classLabel(config, p.a) << "<->" << classLabel(config, p.b)
                  << ": " << p.total
                  << " (" << classLabel(config, p.a) << "->" << classLabel(config, p.b)
                  << "=" << p.aToB
                  << ", " << classLabel(config, p.b) << "->" << classLabel(config, p.a)
                  << "=" << p.bToA
                  << ")" << std::endl;
    }
}

void printPerClassAccuracy(const IntMatrix& confusion, const Config& config) {
    std::cout << "\nPer-class accuracy:" << std::endl;
    for (size_t label = 0; label < confusion.size(); ++label) {
        int total = 0;
        for (size_t pred = 0; pred < confusion.size(); ++pred) {
            total += confusion[label][pred];
        }
        const double acc = total > 0
            ? (100.0 * static_cast<double>(confusion[label][label]) / static_cast<double>(total))
            : 0.0;
        std::cout << "  " << classLabel(config, static_cast<int>(label)) << ": "
                  << std::fixed << std::setprecision(2) << acc << "% ("
                  << confusion[label][label] << "/" << total << ")" << std::endl;
    }
}

void printFocusedFamilyReport(const IntMatrix& confusion,
                              const std::vector<std::vector<int>>& focusGroups,
                              const Config& config) {
    if (focusGroups.empty()) {
        return;
    }

    std::cout << "\nFocused family report:" << std::endl;
    for (const auto& group : focusGroups) {
        if (group.empty()) {
            continue;
        }

        int total = 0;
        int correct = 0;
        int inFamily = 0;
        for (int truth : group) {
            if (truth < 0 || static_cast<size_t>(truth) >= confusion.size()) {
                continue;
            }
            for (size_t pred = 0; pred < confusion.size(); ++pred) {
                const int count = confusion[static_cast<size_t>(truth)][pred];
                total += count;
                if (truth == static_cast<int>(pred)) {
                    correct += count;
                }
                if (std::find(group.begin(), group.end(), static_cast<int>(pred)) != group.end()) {
                    inFamily += count;
                }
            }
        }

        const double familyAcc = total > 0
            ? 100.0 * static_cast<double>(correct) / static_cast<double>(total)
            : 0.0;
        const double familyContainment = total > 0
            ? 100.0 * static_cast<double>(inFamily) / static_cast<double>(total)
            : 0.0;

        std::cout << "  [" << labelGroupToString(group, config) << "] accuracy="
                  << std::fixed << std::setprecision(2) << familyAcc << "% (" << correct
                  << "/" << total << "), in-family=" << familyContainment << "%" << std::endl;

        for (size_t i = 0; i < group.size(); ++i) {
            for (size_t j = i + 1; j < group.size(); ++j) {
                const int a = group[i];
                const int b = group[j];
                const int totalPair =
                    confusion[static_cast<size_t>(a)][static_cast<size_t>(b)] +
                    confusion[static_cast<size_t>(b)][static_cast<size_t>(a)];
                if (totalPair > 0) {
                    std::cout << "    " << classLabel(config, a) << "<->" << classLabel(config, b)
                              << ": " << totalPair
                              << " (" << classLabel(config, a) << "->" << classLabel(config, b) << "="
                              << confusion[static_cast<size_t>(a)][static_cast<size_t>(b)]
                              << ", " << classLabel(config, b) << "->" << classLabel(config, a) << "="
                              << confusion[static_cast<size_t>(b)][static_cast<size_t>(a)] << ")"
                              << std::endl;
                }
            }
        }

        struct EscapeConfusion {
            int truth;
            int pred;
            int count;
        };
        std::vector<EscapeConfusion> escapes;
        for (int truth : group) {
            if (truth < 0 || static_cast<size_t>(truth) >= confusion.size()) {
                continue;
            }
            for (size_t pred = 0; pred < confusion.size(); ++pred) {
                if (std::find(group.begin(), group.end(), static_cast<int>(pred)) != group.end()) {
                    continue;
                }
                const int count = confusion[static_cast<size_t>(truth)][pred];
                if (count > 0) {
                    escapes.push_back({truth, static_cast<int>(pred), count});
                }
            }
        }
        std::sort(escapes.begin(), escapes.end(),
                  [](const EscapeConfusion& lhs, const EscapeConfusion& rhs) {
                      return lhs.count > rhs.count;
                  });
        for (size_t i = 0; i < std::min<size_t>(3, escapes.size()); ++i) {
            std::cout << "    escape " << classLabel(config, escapes[i].truth) << "->"
                      << classLabel(config, escapes[i].pred) << ": " << escapes[i].count << std::endl;
        }
    }
}

void printOnlineCorrectionSummary(const EvaluationResult& result) {
    if (result.tested <= 0 || result.correctionEvents <= 0) {
        return;
    }

    const double initialAccuracy =
        100.0 * static_cast<double>(result.initialCorrect) /
        static_cast<double>(std::max(1, result.tested));
    const double finalAccuracy =
        100.0 * static_cast<double>(result.correct) /
        static_cast<double>(std::max(1, result.tested));
    const double correctionHitRate =
        100.0 * static_cast<double>(result.correctionSuccesses) /
        static_cast<double>(std::max(1, result.correctionEvents));

    std::cout << "  Initial accuracy: " << std::fixed << std::setprecision(2)
              << initialAccuracy << "%" << std::endl;
    std::cout << "  Post-correction accuracy: " << std::fixed << std::setprecision(2)
              << finalAccuracy << "%" << std::endl;
    std::cout << "  Correction events: " << result.correctionEvents
              << ", corrected: " << result.correctionSuccesses
              << " (" << std::fixed << std::setprecision(2) << correctionHitRate << "%)"
              << ", replays: " << result.correctionReplays << std::endl;
    if (result.replayDelaySamples > 0) {
        std::cout << "  Replay timing: avg_delay_steps=" << std::fixed << std::setprecision(2)
                  << (result.replayDelaySteps /
                      static_cast<double>(std::max(1, result.replayDelaySamples)))
                  << ", avg_eligibility="
                  << (result.replayEligibilityScaleSum /
                      static_cast<double>(std::max(1, result.replayDelaySamples)))
                  << std::endl;
    }
}

void printHemisphereAgreementSummary(const EvaluationResult& result) {
    if (result.hemisphereComparable <= 0) {
        return;
    }

    const double agreementRate =
        100.0 * static_cast<double>(result.hemisphereAgreements) /
        static_cast<double>(std::max(1, result.hemisphereComparable));
    const double agreementAccuracy =
        100.0 * static_cast<double>(result.hemisphereAgreementCorrect) /
        static_cast<double>(std::max(1, result.hemisphereAgreements));

    std::cout << "  Hemisphere agreement: " << result.hemisphereAgreements
              << "/" << result.hemisphereComparable
              << " (" << std::fixed << std::setprecision(2) << agreementRate << "%)"
              << std::endl;
    std::cout << "  Agreement accuracy: " << std::fixed << std::setprecision(2)
              << agreementAccuracy << "%" << std::endl;

    if (result.hemisphereDisagreements <= 0) {
        return;
    }

    const double disagreementRate =
        100.0 * static_cast<double>(result.hemisphereDisagreements) /
        static_cast<double>(std::max(1, result.hemisphereComparable));
    std::cout << "  Hemisphere disagreement: " << result.hemisphereDisagreements
              << "/" << result.hemisphereComparable
              << " (" << std::fixed << std::setprecision(2) << disagreementRate << "%)"
              << std::endl;
    std::cout << "  On disagreement: left_only_correct="
              << result.leftOnlyCorrectOnDisagreement
              << ", right_only_correct=" << result.rightOnlyCorrectOnDisagreement
              << ", both_wrong=" << result.bothWrongOnDisagreement << std::endl;
}

void printHemisphereStage1Summary(const EvaluationResult& result, const Config& config) {
    if (result.hemisphereComparable <= 0) {
        return;
    }

    int leftCorrect = 0;
    int rightCorrect = 0;
    for (size_t label = 0; label < result.leftHemisphereConfusion.size(); ++label) {
        leftCorrect += result.leftHemisphereConfusion[label][label];
        rightCorrect += result.rightHemisphereConfusion[label][label];
    }

    const double leftAccuracy =
        100.0 * static_cast<double>(leftCorrect) /
        static_cast<double>(std::max(1, result.hemisphereComparable));
    const double rightAccuracy =
        100.0 * static_cast<double>(rightCorrect) /
        static_cast<double>(std::max(1, result.hemisphereComparable));
    std::cout << "  Stage-1 accuracy by view: left=" << std::fixed << std::setprecision(2)
              << leftAccuracy << "%, right=" << rightAccuracy << "%" << std::endl;

    struct LabelDelta {
        int label = -1;
        int total = 0;
        int leftCorrect = 0;
        int rightCorrect = 0;
        double delta = 0.0;
    };
    std::vector<LabelDelta> labelDeltas;
    for (size_t label = 0; label < result.leftHemisphereConfusion.size(); ++label) {
        int total = 0;
        for (size_t predicted = 0; predicted < result.leftHemisphereConfusion.size(); ++predicted) {
            total += result.leftHemisphereConfusion[label][predicted];
        }
        if (total <= 0) {
            continue;
        }
        const int leftLabelCorrect = result.leftHemisphereConfusion[label][label];
        const int rightLabelCorrect = result.rightHemisphereConfusion[label][label];
        const double leftLabelAccuracy =
            static_cast<double>(leftLabelCorrect) / static_cast<double>(total);
        const double rightLabelAccuracy =
            static_cast<double>(rightLabelCorrect) / static_cast<double>(total);
        labelDeltas.push_back(
            {static_cast<int>(label), total, leftLabelCorrect, rightLabelCorrect,
             leftLabelAccuracy - rightLabelAccuracy});
    }
    std::sort(labelDeltas.begin(),
              labelDeltas.end(),
              [](const LabelDelta& lhs, const LabelDelta& rhs) {
                  return std::abs(lhs.delta) > std::abs(rhs.delta);
              });

    if (!labelDeltas.empty()) {
        std::cout << "  Largest view bias by label:" << std::endl;
        const size_t limit = std::min<size_t>(5, labelDeltas.size());
        for (size_t i = 0; i < limit; ++i) {
            const auto& delta = labelDeltas[i];
            const char favored = delta.delta >= 0.0 ? 'L' : 'R';
            std::cout << "    " << classLabel(config, delta.label)
                      << ": favor=" << favored
                      << ", left=" << delta.leftCorrect << "/" << delta.total
                      << ", right=" << delta.rightCorrect << "/" << delta.total
                      << ", delta=" << std::fixed << std::setprecision(2)
                      << (100.0 * std::abs(delta.delta)) << " pts" << std::endl;
        }
    }

    struct PairDelta {
        int a = -1;
        int b = -1;
        int leftTotal = 0;
        int rightTotal = 0;
    };
    std::vector<PairDelta> pairDeltas;
    for (size_t a = 0; a < result.leftHemisphereConfusion.size(); ++a) {
        for (size_t b = a + 1; b < result.leftHemisphereConfusion.size(); ++b) {
            const int leftPair = result.leftHemisphereConfusion[a][b] +
                                 result.leftHemisphereConfusion[b][a];
            const int rightPair = result.rightHemisphereConfusion[a][b] +
                                  result.rightHemisphereConfusion[b][a];
            if (leftPair == 0 && rightPair == 0) {
                continue;
            }
            pairDeltas.push_back(
                {static_cast<int>(a), static_cast<int>(b), leftPair, rightPair});
        }
    }
    std::sort(pairDeltas.begin(),
              pairDeltas.end(),
              [](const PairDelta& lhs, const PairDelta& rhs) {
                  return std::abs(lhs.leftTotal - lhs.rightTotal) >
                         std::abs(rhs.leftTotal - rhs.rightTotal);
              });

    if (!pairDeltas.empty()) {
        std::cout << "  Largest view bias by confusion pair:" << std::endl;
        const size_t limit = std::min<size_t>(5, pairDeltas.size());
        for (size_t i = 0; i < limit; ++i) {
            const auto& delta = pairDeltas[i];
            const char favored = delta.leftTotal <= delta.rightTotal ? 'L' : 'R';
            std::cout << "    " << classLabel(config, delta.a) << "<->" << classLabel(config, delta.b)
                      << ": fewer errors on " << favored
                      << " view (left=" << delta.leftTotal
                      << ", right=" << delta.rightTotal << ")" << std::endl;
        }
    }
}

void printSeparabilitySummary(const EvaluationResult& result, const Config& config) {
    if (result.separabilityStats.empty()) {
        return;
    }

    std::cout << "  Representation separability:" << std::endl;
    for (const auto& stats : result.separabilityStats) {
        if (stats.samples <= 0) {
            continue;
        }
        const double centroidAccuracy =
            100.0 * static_cast<double>(stats.centroidCorrect) /
            static_cast<double>(std::max(1, stats.samples));
        const double nonzeroCentroidAccuracy =
            100.0 * static_cast<double>(stats.centroidCorrectNonzero) /
            static_cast<double>(std::max(1, stats.nonzeroSamples));
        const double ownScore =
            stats.ownScoreSum / static_cast<double>(std::max(1, stats.samples));
        const double otherScore =
            stats.otherScoreSum / static_cast<double>(std::max(1, stats.samples));
        const double margin =
            stats.marginSum / static_cast<double>(std::max(1, stats.samples));
        const double rawL1 =
            stats.rawL1Sum / static_cast<double>(std::max(1, stats.samples));
        const double rawL2 =
            stats.rawL2Sum / static_cast<double>(std::max(1, stats.samples));
        const double rawActiveFraction =
            stats.rawActiveFractionSum / static_cast<double>(std::max(1, stats.samples));
        std::cout << "    " << stats.name
                  << ": centroid_acc=" << std::fixed << std::setprecision(2)
                  << centroidAccuracy << "%, own=" << ownScore
                  << ", other=" << otherScore
                  << ", margin=" << margin
                  << ", raw_l1=" << rawL1
                  << ", raw_l2=" << rawL2
                  << ", raw_active=" << (100.0 * rawActiveFraction) << "%"
                  << ", nonzero_centroid_acc=" << nonzeroCentroidAccuracy << "%"
                  << ", zero_vectors=" << stats.zeroVectorSamples;

        int dominantTarget = -1;
        int dominantTargetCount = 0;
        int topTruth = -1;
        int topPredicted = -1;
        int topPairCount = 0;
        for (size_t truth = 0; truth < stats.centroidConfusion.size(); ++truth) {
            for (size_t predicted = 0; predicted < stats.centroidConfusion[truth].size(); ++predicted) {
                const int count = stats.centroidConfusion[truth][predicted];
                if (count <= 0 || truth == predicted) {
                    continue;
                }
                if (count > topPairCount) {
                    topPairCount = count;
                    topTruth = static_cast<int>(truth);
                    topPredicted = static_cast<int>(predicted);
                }
                int incoming = 0;
                for (size_t sourceTruth = 0; sourceTruth < stats.centroidConfusion.size(); ++sourceTruth) {
                    if (sourceTruth == predicted) {
                        continue;
                    }
                    incoming += stats.centroidConfusion[sourceTruth][predicted];
                }
                if (incoming > dominantTargetCount) {
                    dominantTargetCount = incoming;
                    dominantTarget = static_cast<int>(predicted);
                }
            }
        }
        if (dominantTarget >= 0 && dominantTargetCount > 0) {
            std::cout << ", collapse_target=" << classLabel(config, dominantTarget)
                      << " (" << dominantTargetCount << ")";
        }
        if (topTruth >= 0 && topPredicted >= 0 && topPairCount > 0) {
            std::cout << ", top_pair=" << classLabel(config, topTruth) << "->"
                      << classLabel(config, topPredicted) << " (" << topPairCount << ")";
        }
        if (stats.hemisphereWrongSamples > 0) {
            int dominantWrongTarget = -1;
            int dominantWrongCount = 0;
            for (size_t label = 0; label < stats.hemisphereWrongTargetCounts.size(); ++label) {
                if (stats.hemisphereWrongTargetCounts[label] > dominantWrongCount) {
                    dominantWrongCount = stats.hemisphereWrongTargetCounts[label];
                    dominantWrongTarget = static_cast<int>(label);
                }
            }
            std::cout << ", hemi_wrong_agree=" << stats.branchSupportsHemisphereWrong
                      << "/" << stats.hemisphereWrongSamples
                      << ", hemi_wrong_truth=" << stats.branchSupportsTruth
                      << "/" << stats.hemisphereWrongSamples;
            if (dominantWrongTarget >= 0 && dominantWrongCount > 0) {
                std::cout << ", hemi_wrong_target=" << classLabel(config, dominantWrongTarget)
                          << " (" << dominantWrongCount << ")";
            }
        }
        std::cout << std::endl;
    }
}

std::string escapeJsonString(const std::string& value) {
    std::ostringstream oss;
    for (char c : value) {
        switch (c) {
            case '\\':
                oss << "\\\\";
                break;
            case '"':
                oss << "\\\"";
                break;
            case '\n':
                oss << "\\n";
                break;
            case '\r':
                oss << "\\r";
                break;
            case '\t':
                oss << "\\t";
                break;
            default:
                oss << c;
                break;
        }
    }
    return oss.str();
}

void writeFlowAuditArtifacts(const FlowAuditRuntime& runtime, const std::string& label) {
    if (!runtime.enabled) {
        return;
    }

    std::filesystem::path prefix(runtime.outputPrefix);
    const std::string labelSuffix = sanitizeFlowAuditStem(toLower(label));
    if (!labelSuffix.empty()) {
        prefix += "_" + labelSuffix;
    }
    if (!prefix.parent_path().empty()) {
        std::filesystem::create_directories(prefix.parent_path());
    }

    const std::filesystem::path csvPath = prefix.string() + "_samples.csv";
    const std::filesystem::path jsonPath = prefix.string() + "_summary.json";

    {
        std::ofstream csv(csvPath);
        csv << "sample_ordinal,image_index,truth,stage,hemisphere,component,predicted,initial_predicted,"
               "final_predicted,top1_label,top1_score,top2_label,top2_score,confidence_margin,"
               "best_neighbor_label,best_neighbor_similarity,topk_purity,centroid_predicted,"
               "pre_norm_mean_abs,pre_norm_l2,pre_norm_active_fraction,pre_orientation_l2,"
               "pre_auxiliary_l2,pre_own_score,pre_other_score,pre_margin,post_norm_mean_abs,"
               "post_norm_l2,post_norm_active_fraction,post_orientation_l2,post_auxiliary_l2,"
               "post_own_score,post_other_score,post_margin,fixation_count,fixation_variance,"
               "fixation_consistency,"
               "fg_owned_border_mass,fg_surface_mass,fg_junction_mass,fg_border_surface_ratio,"
               "fg_left_right_balance,fg_component_count,fg_fixation_consistency\n";
        for (const auto& row : runtime.rows) {
            csv << row.sampleOrdinal << ','
                << row.imageIndex << ','
                << row.truth << ','
                << row.stage << ','
                << row.hemisphere << ','
                << row.component << ','
                << row.predicted << ','
                << row.initialPredicted << ','
                << row.finalPredicted << ','
                << row.top1Label << ','
                << row.top1Score << ','
                << row.top2Label << ','
                << row.top2Score << ','
                << row.confidenceMargin << ','
                << row.bestNeighborLabel << ','
                << row.bestNeighborSimilarity << ','
                << row.topkPurity << ','
                << row.centroidPredicted << ','
                << row.preNormMeanAbs << ','
                << row.preNormL2 << ','
                << row.preNormActiveFraction << ','
                << row.preOrientationL2 << ','
                << row.preAuxiliaryL2 << ','
                << row.preOwnScore << ','
                << row.preOtherScore << ','
                << row.preMargin << ','
                << row.postNormMeanAbs << ','
                << row.postNormL2 << ','
                << row.postNormActiveFraction << ','
                << row.postOrientationL2 << ','
                << row.postAuxiliaryL2 << ','
                << row.postOwnScore << ','
                << row.postOtherScore << ','
                << row.postMargin << ','
                << row.fixationCount << ','
                << row.fixationVariance << ','
                << row.fixationConsistency << ','
                << row.fgOwnedBorderMass << ','
                << row.fgSurfaceMass << ','
                << row.fgJunctionMass << ','
                << row.fgBorderSurfaceRatio << ','
                << row.fgLeftRightBalance << ','
                << row.fgComponentCount << ','
                << row.fgFixationConsistency << '\n';
        }
    }

    std::vector<std::pair<std::string, FlowAuditBranchAggregate>> sortedBranchAggregates(
        runtime.branchAggregates.begin(), runtime.branchAggregates.end());
    std::sort(sortedBranchAggregates.begin(), sortedBranchAggregates.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    std::vector<std::pair<std::string, FlowAuditHemisphereAggregate>> sortedHemisphereAggregates(
        runtime.hemisphereAggregates.begin(), runtime.hemisphereAggregates.end());
    std::sort(sortedHemisphereAggregates.begin(), sortedHemisphereAggregates.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    std::vector<std::pair<std::string, FlowAuditFigureGroundAggregate>> sortedFigureGroundAggregates(
        runtime.figureGroundAggregates.begin(), runtime.figureGroundAggregates.end());
    std::sort(sortedFigureGroundAggregates.begin(), sortedFigureGroundAggregates.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    std::ofstream json(jsonPath);
    json << "{\n";
    json << "  \"output_prefix\": \"" << escapeJsonString(prefix.string()) << "\",\n";
    json << "  \"samples_recorded\": " << runtime.recordedSamples << ",\n";
    json << "  \"rows_written\": " << runtime.rows.size() << ",\n";
    json << "  \"dropped_rows\": " << runtime.droppedRows << ",\n";
    json << "  \"fusion_alternatives\": {\n";
    json << "    \"samples\": " << runtime.fusionAggregate.samples << ",\n";
    json << "    \"interaction_centroid_accuracy\": "
         << (runtime.fusionAggregate.samples > 0
                 ? (100.0 * static_cast<double>(runtime.fusionAggregate.interactionCentroidCorrect) /
                    static_cast<double>(runtime.fusionAggregate.samples))
                 : 0.0)
         << ",\n";
    json << "    \"confidence_concat_centroid_accuracy\": "
         << (runtime.fusionAggregate.samples > 0
                 ? (100.0 * static_cast<double>(runtime.fusionAggregate.confidenceConcatCentroidCorrect) /
                    static_cast<double>(runtime.fusionAggregate.samples))
                 : 0.0)
         << ",\n";
    json << "    \"hemisphere_concat_centroid_accuracy\": "
         << (runtime.fusionAggregate.samples > 0
                 ? (100.0 * static_cast<double>(runtime.fusionAggregate.hemisphereConcatCentroidCorrect) /
                    static_cast<double>(runtime.fusionAggregate.samples))
                 : 0.0)
         << "\n";
    json << "  },\n";
    json << "  \"replay_mechanisms\": {\n";
    json << "    \"replay_success_samples\": " << runtime.replayAggregate.replaySuccessSamples << ",\n";
    json << "    \"replay_failure_samples\": " << runtime.replayAggregate.replayFailureSamples << ",\n";
    json << "    \"positive_class_weight_updates\": " << runtime.replayAggregate.positiveClassWeightUpdates << ",\n";
    json << "    \"positive_prediction_weight_updates\": " << runtime.replayAggregate.positivePredictionWeightUpdates << ",\n";
    json << "    \"positive_centroid_updates\": " << runtime.replayAggregate.positiveCentroidUpdates << ",\n";
    json << "    \"positive_exemplar_insertions\": " << runtime.replayAggregate.positiveExemplarInsertions << ",\n";
    json << "    \"positive_fusion_exemplar_insertions\": " << runtime.replayAggregate.positiveFusionExemplarInsertions << ",\n";
    json << "    \"negative_class_weight_updates\": " << runtime.replayAggregate.negativeClassWeightUpdates << ",\n";
    json << "    \"negative_prediction_weight_updates\": " << runtime.replayAggregate.negativePredictionWeightUpdates << "\n";
    json << "  },\n";
    json << "  \"branch_aggregates\": [\n";
    for (size_t i = 0; i < sortedBranchAggregates.size(); ++i) {
        const auto& [name, stats] = sortedBranchAggregates[i];
        const double samples = static_cast<double>(std::max(1, stats.samples));
        json << "    {\n";
        json << "      \"name\": \"" << escapeJsonString(name) << "\",\n";
        json << "      \"samples\": " << stats.samples << ",\n";
        json << "      \"zero_pre_samples\": " << stats.zeroPreSamples << ",\n";
        json << "      \"zero_post_samples\": " << stats.zeroPostSamples << ",\n";
        json << "      \"mean_pre_margin\": " << (stats.preMarginSum / samples) << ",\n";
        json << "      \"mean_post_margin\": " << (stats.postMarginSum / samples) << ",\n";
        json << "      \"mean_pre_active_fraction\": " << (stats.preActiveFractionSum / samples) << ",\n";
        json << "      \"mean_post_active_fraction\": " << (stats.postActiveFractionSum / samples) << ",\n";
        json << "      \"mean_pre_norm_l2\": " << (stats.preNormL2Sum / samples) << ",\n";
        json << "      \"mean_post_norm_l2\": " << (stats.postNormL2Sum / samples) << ",\n";
        json << "      \"mean_pre_orientation_l2\": " << (stats.preOrientationL2Sum / samples) << ",\n";
        json << "      \"mean_pre_auxiliary_l2\": " << (stats.preAuxiliaryL2Sum / samples) << ",\n";
        json << "      \"mean_post_orientation_l2\": " << (stats.postOrientationL2Sum / samples) << ",\n";
        json << "      \"mean_post_auxiliary_l2\": " << (stats.postAuxiliaryL2Sum / samples) << ",\n";
        json << "      \"mean_fixation_variance\": " << (stats.fixationVarianceSum / samples) << ",\n";
        json << "      \"mean_fixation_consistency\": " << (stats.fixationConsistencySum / samples) << ",\n";
        json << "      \"mean_training_fixation_variance\": " << stats.meanTrainingFixationVariance << "\n";
        json << "    }" << (i + 1 < sortedBranchAggregates.size() ? "," : "") << "\n";
    }
    json << "  ],\n";
    json << "  \"hemisphere_aggregates\": [\n";
    for (size_t i = 0; i < sortedHemisphereAggregates.size(); ++i) {
        const auto& [name, stats] = sortedHemisphereAggregates[i];
        const double samples = static_cast<double>(std::max(1, stats.samples));
        json << "    {\n";
        json << "      \"name\": \"" << escapeJsonString(name) << "\",\n";
        json << "      \"samples\": " << stats.samples << ",\n";
        json << "      \"mean_confidence_margin\": " << (stats.confidenceMarginSum / samples) << ",\n";
        json << "      \"mean_centroid_margin\": " << (stats.centroidMarginSum / samples) << ",\n";
        json << "      \"mean_topk_purity\": " << (stats.topkPuritySum / samples) << ",\n";
        json << "      \"mean_best_neighbor_similarity\": " << (stats.bestNeighborSimilaritySum / samples) << ",\n";
        json << "      \"mean_fixation_consistency\": " << (stats.fixationConsistencySum / samples) << "\n";
        json << "    }" << (i + 1 < sortedHemisphereAggregates.size() ? "," : "") << "\n";
    }
    json << "  ],\n";
    json << "  \"figure_ground_aggregates\": [\n";
    for (size_t i = 0; i < sortedFigureGroundAggregates.size(); ++i) {
        const auto& [name, stats] = sortedFigureGroundAggregates[i];
        const double samples = static_cast<double>(std::max(1, stats.samples));
        json << "    {\n";
        json << "      \"name\": \"" << escapeJsonString(name) << "\",\n";
        json << "      \"samples\": " << stats.samples << ",\n";
        json << "      \"mean_owned_border_mass\": " << (stats.ownedBorderMassSum / samples) << ",\n";
        json << "      \"mean_surface_mass\": " << (stats.surfaceMassSum / samples) << ",\n";
        json << "      \"mean_junction_mass\": " << (stats.junctionMassSum / samples) << ",\n";
        json << "      \"mean_border_surface_ratio\": " << (stats.borderSurfaceRatioSum / samples) << ",\n";
        json << "      \"mean_left_right_balance\": " << (stats.leftRightBalanceSum / samples) << ",\n";
        json << "      \"mean_component_count\": " << (stats.componentCountSum / samples) << ",\n";
        json << "      \"mean_fixation_consistency\": " << (stats.fixationConsistencySum / samples) << ",\n";
        json << "      \"mean_fg_fixation_consistency\": " << (stats.fgFixationConsistencySum / samples) << "\n";
        json << "    }" << (i + 1 < sortedFigureGroundAggregates.size() ? "," : "") << "\n";
    }
    json << "  ]\n";
    json << "}\n";

    std::cout << "  Flow audit CSV: " << csvPath.string() << std::endl;
    std::cout << "  Flow audit summary: " << jsonPath.string() << std::endl;
}

void resetOrientationFlowDiagnostics(std::vector<HemisphereRuntime>& hemispheres) {
    for (auto& hemisphere : hemispheres) {
        for (auto& retina : hemisphere.retinas) {
            retina->resetOrientationFlowDiagnostics();
            retina->resetPatchInputDiagnostics();
        }
    }
}

void printOrientationFlowSummary(const std::vector<HemisphereRuntime>& hemispheres) {
    bool printedHeader = false;
    for (const auto& hemisphere : hemispheres) {
        for (const auto& retina : hemisphere.retinas) {
            const auto stats = retina->getOrientationFlowDiagnostics();
            if (stats.samples == 0) {
                continue;
            }
            if (!printedHeader) {
                std::cout << "  Orientation flow diagnostics:" << std::endl;
                printedHeader = true;
            }
            const double samples = static_cast<double>(stats.samples);
            std::cout << "    " << retina->getName()
                      << ": manual_l2=" << (stats.manualL2Sum / samples)
                      << ", manual_active="
                      << (100.0 * stats.manualActiveFractionSum / samples) << "%"
                      << ", manual_max=" << (stats.manualMaxSum / samples)
                      << ", pre_l2=" << (stats.preThresholdL2Sum / samples)
                      << ", pre_active="
                      << (100.0 * stats.preThresholdActiveFractionSum / samples) << "%"
                      << ", pre_max=" << (stats.preThresholdMaxSum / samples)
                      << ", thresh_l2=" << (stats.thresholdedL2Sum / samples)
                      << ", thresh_active="
                      << (100.0 * stats.thresholdedActiveFractionSum / samples) << "%"
                      << ", thresh_max=" << (stats.thresholdedMaxSum / samples)
                      << ", post_l2=" << (stats.postCompetitionL2Sum / samples)
                      << ", post_active="
                      << (100.0 * stats.postCompetitionActiveFractionSum / samples) << "%"
                      << ", post_max=" << (stats.postCompetitionMaxSum / samples)
                      << ", zero_manual=" << stats.allZeroManual << "/" << stats.samples
                      << ", zero_pre=" << stats.allZeroBeforeThreshold << "/" << stats.samples
                      << ", zero_thresh=" << stats.allZeroAfterThreshold << "/" << stats.samples
                      << ", zero_post=" << stats.allZeroAfterCompetition << "/" << stats.samples
                      << std::endl;
        }
    }
}

void printPatchInputSummary(const std::vector<HemisphereRuntime>& hemispheres) {
    bool printedHeader = false;
    for (const auto& hemisphere : hemispheres) {
        for (const auto& retina : hemisphere.retinas) {
            const auto stats = retina->getPatchInputDiagnostics();
            if (stats.samples == 0) {
                continue;
            }
            if (!printedHeader) {
                std::cout << "  Patch input diagnostics:" << std::endl;
                printedHeader = true;
            }
            const double samples = static_cast<double>(stats.samples);
            std::cout << "    " << retina->getName()
                      << ": mean_min=" << (stats.minValueSum / samples)
                      << ", mean_max=" << (stats.maxValueSum / samples)
                      << ", mean_range=" << (stats.rangeSum / samples)
                      << ", mean_std=" << (stats.stddevSum / samples)
                      << ", mean_grad=" << (stats.gradientEnergySum / samples)
                      << ", flat_range=" << stats.nearFlatRangeCount << "/" << stats.samples
                      << ", flat_grad=" << stats.nearFlatGradientCount << "/" << stats.samples
                      << std::endl;
        }
    }
}

void printTopDirectedPairs(const IntMatrix& matrix,
                           const std::string& heading,
                           size_t limit,
                           const Config& config) {
    struct PairCount {
        int truth = -1;
        int predicted = -1;
        int count = 0;
    };

    std::vector<PairCount> pairs;
    for (size_t truth = 0; truth < matrix.size(); ++truth) {
        for (size_t predicted = 0; predicted < matrix.size(); ++predicted) {
            if (truth == predicted || matrix[truth][predicted] <= 0) {
                continue;
            }
            pairs.push_back(
                {static_cast<int>(truth), static_cast<int>(predicted), matrix[truth][predicted]});
        }
    }
    if (pairs.empty()) {
        return;
    }

    std::sort(pairs.begin(), pairs.end(), [](const PairCount& lhs, const PairCount& rhs) {
        if (lhs.count != rhs.count) {
            return lhs.count > rhs.count;
        }
        if (lhs.truth != rhs.truth) {
            return lhs.truth < rhs.truth;
        }
        return lhs.predicted < rhs.predicted;
    });

    std::cout << "  " << heading << ":" << std::endl;
    for (size_t i = 0; i < std::min(limit, pairs.size()); ++i) {
        const auto& pair = pairs[i];
        std::cout << "    " << classLabel(config, pair.truth) << "->"
                  << classLabel(config, pair.predicted)
                  << ": " << pair.count << std::endl;
    }
}

void printBilateralAttributionSummary(const EvaluationResult& result, const Config& config) {
    if (result.tested <= 0) {
        return;
    }

    std::cout << "  Initial error attribution: fusion_rescues_one_hemi_right="
              << result.fusionRescuesOneHemisphereRight
              << ", fusion_rescues_both_hemi_wrong="
              << result.fusionRescuesBothHemispheresWrong
              << ", fusion_misses_left_was_right="
              << result.fusionMissesWhenLeftWasRight
              << ", fusion_misses_right_was_right="
              << result.fusionMissesWhenRightWasRight
              << ", initial_wrong_both_hemi_wrong="
              << result.initialWrongBothHemispheresWrong << std::endl;

    if (result.correctionEvents > 0) {
        std::cout << "  Replay correction attribution: corrected_from_one_hemi_right="
                  << result.correctedFromOneHemisphereRight
                  << ", corrected_from_both_hemi_wrong="
                  << result.correctedFromBothHemispheresWrong << std::endl;
    }

    printTopDirectedPairs(result.correctedInitialPairs, "Top corrected initial pairs", 5, config);
    printTopDirectedPairs(result.unresolvedInitialPairs, "Top unresolved initial pairs", 5, config);

    if (result.confusionClusterScoreSamples > 0) {
        const double averageClusterScore =
            result.confusionClusterScoreSum /
            static_cast<double>(std::max(1, result.confusionClusterScoreSamples));
        std::cout << "  Average confusion-cluster score: " << std::fixed
                  << std::setprecision(2) << averageClusterScore << std::endl;
    }

    struct ClusterPair {
        int a = -1;
        int b = -1;
        double strength = 0.0;
    };
    std::vector<ClusterPair> pairs;
    for (size_t a = 0; a < result.confusionClusterStrengths.size(); ++a) {
        for (size_t b = a + 1; b < result.confusionClusterStrengths.size(); ++b) {
            const double strength = result.confusionClusterStrengths[a][b];
            if (strength > 0.0) {
                pairs.push_back({static_cast<int>(a), static_cast<int>(b), strength});
            }
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const ClusterPair& lhs, const ClusterPair& rhs) {
        return lhs.strength > rhs.strength;
    });
    if (!pairs.empty()) {
        std::cout << "  Top learned confusion clusters:" << std::endl;
        for (size_t i = 0; i < std::min<size_t>(5, pairs.size()); ++i) {
            const auto& pair = pairs[i];
            std::cout << "    " << classLabel(config, pair.a) << "<->" << classLabel(config, pair.b)
                      << ": " << std::fixed << std::setprecision(2) << pair.strength
                      << std::endl;
        }
    }
}

void printFocusAdjustmentSummary(const EvaluationResult& result) {
    if (result.focusAdjustmentTriggers <= 0) {
        return;
    }

    const double overrideRate =
        100.0 * static_cast<double>(result.focusAdjustmentOverrides) /
        static_cast<double>(std::max(1, result.focusAdjustmentTriggers));
    std::cout << "  Focus adjustment: triggers=" << result.focusAdjustmentTriggers
              << ", overrides=" << result.focusAdjustmentOverrides
              << " (" << std::fixed << std::setprecision(2) << overrideRate << "%)"
              << ", corrected=" << result.focusAdjustmentCorrections
              << ", regressed=" << result.focusAdjustmentRegressions << std::endl;
    std::cout << "  Focus preset selections: original=" << result.focusOriginalSelections
              << ", left=" << result.focusLeftSelections
              << ", center=" << result.focusCenterSelections
              << ", right=" << result.focusRightSelections << std::endl;
}

void printActiveInferenceSummary(const EvaluationResult& result, const Config& config) {
    if (!config.activeInferenceEnabled || result.activeInferenceSamples <= 0) {
        return;
    }
    const double meanFixations =
        result.activeInferenceFixationSum /
        static_cast<double>(std::max(1, result.activeInferenceSamples));
    const double extraRate =
        100.0 * static_cast<double>(result.activeInferenceExtraFixationSamples) /
        static_cast<double>(std::max(1, result.activeInferenceSamples));
    const double earlyStopRate =
        100.0 * static_cast<double>(result.activeInferenceEarlyStops) /
        static_cast<double>(std::max(1, result.activeInferenceSamples));
    std::cout << "  Active inference: samples=" << result.activeInferenceSamples
              << ", mean_fixations=" << std::fixed << std::setprecision(2) << meanFixations
              << ", extra_fixations=" << result.activeInferenceExtraFixationSamples
              << " (" << extraRate << "%)"
              << ", early_stops=" << result.activeInferenceEarlyStops
              << " (" << earlyStopRate << "%)" << std::endl;
}

Config parseArgs(int argc, char* argv[]) {
    Config config;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config.configPath = argv[++i];
        }
    }

    if (!config.configPath.empty()) {
        loadDeclarativeConfig(config, config.configPath);
    }

    auto applyRetinaIntOverride = [&](const std::string& key, int value) {
        if (config.retinaConfigs.empty()) {
            return;
        }
        for (auto& retinaConfig : config.retinaConfigs) {
            retinaConfig.intParams[key] = value;
        }
    };
    auto applyRetinaDoubleOverride = [&](const std::string& key, double value) {
        if (config.retinaConfigs.empty()) {
            return;
        }
        for (auto& retinaConfig : config.retinaConfigs) {
            retinaConfig.doubleParams[key] = value;
        }
    };
    auto applyNamedRetinaIntOverride =
        [&](const std::string& nameNeedle, const std::string& key, int value) {
            if (config.retinaConfigs.empty()) {
                return;
            }
            const std::string normalizedNeedle = toLower(nameNeedle);
            for (auto& retinaConfig : config.retinaConfigs) {
                if (toLower(retinaConfig.name).find(normalizedNeedle) != std::string::npos) {
                    retinaConfig.intParams[key] = value;
                }
            }
        };
    auto applyNamedRetinaDoubleOverride =
        [&](const std::string& nameNeedle, const std::string& key, double value) {
            if (config.retinaConfigs.empty()) {
                return;
            }
            const std::string normalizedNeedle = toLower(nameNeedle);
            for (auto& retinaConfig : config.retinaConfigs) {
                if (toLower(retinaConfig.name).find(normalizedNeedle) != std::string::npos) {
                    retinaConfig.doubleParams[key] = value;
                }
            }
        };
    auto applyRetinaStringOverride = [&](const std::string& key, const std::string& value) {
        if (config.retinaConfigs.empty()) {
            return;
        }
        for (auto& retinaConfig : config.retinaConfigs) {
            retinaConfig.stringParams[key] = value;
        }
    };
    auto applyRetinaTemporalOverride = [&](double value) {
        if (config.retinaConfigs.empty()) {
            return;
        }
        for (auto& retinaConfig : config.retinaConfigs) {
            retinaConfig.temporalWindow = value;
            retinaConfig.doubleParams["neuron_window_size"] = value;
        }
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            ++i;
        } else if (arg == "--train-images" && i + 1 < argc) {
            config.trainImagesPath = argv[++i];
        } else if (arg == "--train-labels" && i + 1 < argc) {
            config.trainLabelsPath = argv[++i];
        } else if (arg == "--test-images" && i + 1 < argc) {
            config.testImagesPath = argv[++i];
        } else if (arg == "--test-labels" && i + 1 < argc) {
            config.testLabelsPath = argv[++i];
        } else if (arg == "--input-domain" && i + 1 < argc) {
            config.inputDomain = argv[++i];
        } else if (arg == "--input-variant" && i + 1 < argc) {
            config.inputVariant = argv[++i];
        } else if (arg == "--input-apply-transform" && i + 1 < argc) {
            config.inputApplyTransform = std::atoi(argv[++i]) != 0;
        } else if (arg == "--examples-per-class" && i + 1 < argc) {
            config.examplesPerClass = std::atoi(argv[++i]);
        } else if (arg == "--test-limit" && i + 1 < argc) {
            config.testLimit = std::atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            config.seed = static_cast<unsigned int>(std::atoi(argv[++i]));
        } else if (arg == "--grid-size" && i + 1 < argc) {
            config.gridSize = std::atoi(argv[++i]);
            config.gridSizes = {config.gridSize};
            config.retinaConfigs.clear();
        } else if (arg == "--grid-sizes" && i + 1 < argc) {
            config.gridSizes = parseCsvInts(argv[++i]);
            if (config.gridSizes.empty()) {
                throw std::runtime_error("grid-sizes must contain at least one integer");
            }
            config.gridSize = config.gridSizes.front();
            config.retinaConfigs.clear();
        } else if (arg == "--num-orientations" && i + 1 < argc) {
            config.numOrientations = std::atoi(argv[++i]);
            applyRetinaIntOverride("num_orientations", config.numOrientations);
        } else if (arg == "--edge-threshold" && i + 1 < argc) {
            config.edgeThreshold = std::atof(argv[++i]);
            applyRetinaDoubleOverride("edge_threshold", config.edgeThreshold);
        } else if (arg == "--temporal-window-ms" && i + 1 < argc) {
            config.temporalWindowMs = std::atof(argv[++i]);
            applyRetinaTemporalOverride(config.temporalWindowMs);
        } else if (arg == "--edge-operator" && i + 1 < argc) {
            config.edgeOperator = argv[++i];
            applyRetinaStringOverride("edge_operator", config.edgeOperator);
        } else if (arg == "--g10-lgn-burst-tonic-enabled") {
            applyNamedRetinaIntOverride("g10", "lgn_burst_tonic_enabled", 1);
        } else if (arg == "--g10-lgn-burst-threshold" && i + 1 < argc) {
            applyNamedRetinaDoubleOverride(
                "g10", "lgn_burst_threshold", std::clamp(std::atof(argv[++i]), 0.0, 1.0));
        } else if (arg == "--g10-lgn-burst-extra-strength" && i + 1 < argc) {
            applyNamedRetinaDoubleOverride(
                "g10", "lgn_burst_extra_strength", std::max(0.0, std::atof(argv[++i])));
        } else if (arg == "--g10-lgn-burst-slope" && i + 1 < argc) {
            applyNamedRetinaDoubleOverride(
                "g10", "lgn_burst_slope", std::max(0.0, std::atof(argv[++i])));
        } else if (arg == "--g10-lgn-burst-neuromodulator" && i + 1 < argc) {
            applyNamedRetinaDoubleOverride(
                "g10",
                "lgn_burst_neuromodulator",
                std::clamp(std::atof(argv[++i]), 0.0, 1.0));
        } else if (arg == "--g10-sensory-triplet-bcm-enabled") {
            applyNamedRetinaIntOverride("g10", "sensory_triplet_bcm_enabled", 1);
        } else if (arg == "--g10-sensory-triplet-bcm-learning-rate" && i + 1 < argc) {
            applyNamedRetinaDoubleOverride(
                "g10", "sensory_triplet_bcm_learning_rate", std::max(0.0, std::atof(argv[++i])));
        } else if (arg == "--g10-sensory-triplet-bcm-ltp" && i + 1 < argc) {
            applyNamedRetinaDoubleOverride(
                "g10", "sensory_triplet_bcm_ltp", std::max(0.0, std::atof(argv[++i])));
        } else if (arg == "--g10-sensory-triplet-bcm-ltd" && i + 1 < argc) {
            applyNamedRetinaDoubleOverride(
                "g10", "sensory_triplet_bcm_ltd", std::max(0.0, std::atof(argv[++i])));
        } else if (arg == "--g10-sensory-triplet-fast-decay" && i + 1 < argc) {
            applyNamedRetinaDoubleOverride(
                "g10", "sensory_triplet_fast_decay", std::clamp(std::atof(argv[++i]), 0.0, 0.9999));
        } else if (arg == "--g10-sensory-triplet-slow-decay" && i + 1 < argc) {
            applyNamedRetinaDoubleOverride(
                "g10", "sensory_triplet_slow_decay", std::clamp(std::atof(argv[++i]), 0.0, 0.9999));
        } else if (arg == "--g10-sensory-bcm-threshold-decay" && i + 1 < argc) {
            applyNamedRetinaDoubleOverride(
                "g10", "sensory_bcm_threshold_decay", std::clamp(std::atof(argv[++i]), 0.0, 0.9999));
        } else if (arg == "--g10-sensory-bcm-target-activation" && i + 1 < argc) {
            applyNamedRetinaDoubleOverride(
                "g10", "sensory_bcm_target_activation", std::clamp(std::atof(argv[++i]), 0.0, 1.0));
        } else if (arg == "--encoding-strategy" && i + 1 < argc) {
            config.encodingStrategy = argv[++i];
            applyRetinaStringOverride("encoding_strategy", config.encodingStrategy);
        } else if (arg == "--classifier" && i + 1 < argc) {
            config.classifier = argv[++i];
        } else if (arg == "--activation-mode" && i + 1 < argc) {
            config.activationMode = argv[++i];
            applyRetinaStringOverride("activation_mode", config.activationMode);
        } else if (arg == "--hierarchical-groups" && i + 1 < argc) {
            config.hierarchicalGroups = argv[++i];
        } else if (arg == "--hierarchical-coarse-strategy" && i + 1 < argc) {
            config.hierarchicalCoarseStrategy = argv[++i];
        } else if (arg == "--hierarchical-fine-strategy" && i + 1 < argc) {
            config.hierarchicalFineStrategy = argv[++i];
        } else if (arg == "--hierarchical-coarse-k" && i + 1 < argc) {
            config.hierarchicalCoarseK = std::atoi(argv[++i]);
        } else if (arg == "--hierarchical-fine-k" && i + 1 < argc) {
            config.hierarchicalFineK = std::atoi(argv[++i]);
        } else if (arg == "--knn-k" && i + 1 < argc) {
            config.knnK = std::atoi(argv[++i]);
        } else if (arg == "--classifier-exponent" && i + 1 < argc) {
            config.classifierExponent = std::atof(argv[++i]);
        } else if (arg == "--online-correction-repeats" && i + 1 < argc) {
            config.onlineCorrectionRepeats = std::atoi(argv[++i]);
        } else if (arg == "--online-exemplar-budget-per-class" && i + 1 < argc) {
            config.onlineExemplarBudgetPerClass = std::atoi(argv[++i]);
        } else if (arg == "--online-centroid-lr" && i + 1 < argc) {
            config.onlineCentroidLr = std::atof(argv[++i]);
        } else if (arg == "--online-positive-reward-gain" && i + 1 < argc) {
            config.onlinePositiveRewardGain = std::atof(argv[++i]);
        } else if (arg == "--online-negative-reward-gain" && i + 1 < argc) {
            config.onlineNegativeRewardGain = std::atof(argv[++i]);
        } else if (arg == "--online-reward-stdp-enabled") {
            config.onlineRewardStdpEnabled = true;
        } else if (arg == "--online-reward-stdp-gain" && i + 1 < argc) {
            config.onlineRewardStdpGain = std::atof(argv[++i]);
        } else if (arg == "--online-reward-stdp-ltp" && i + 1 < argc) {
            config.onlineRewardStdpLtp = std::atof(argv[++i]);
        } else if (arg == "--online-reward-stdp-ltd" && i + 1 < argc) {
            config.onlineRewardStdpLtd = std::atof(argv[++i]);
        } else if (arg == "--online-triplet-stdp-enabled") {
            config.onlineTripletStdpEnabled = true;
        } else if (arg == "--online-triplet-stdp-gain" && i + 1 < argc) {
            config.onlineTripletStdpGain = std::atof(argv[++i]);
        } else if (arg == "--online-triplet-stdp-ltp" && i + 1 < argc) {
            config.onlineTripletStdpLtp = std::atof(argv[++i]);
        } else if (arg == "--online-triplet-stdp-ltd" && i + 1 < argc) {
            config.onlineTripletStdpLtd = std::atof(argv[++i]);
        } else if (arg == "--online-triplet-stdp-fast-decay" && i + 1 < argc) {
            config.onlineTripletStdpFastDecay = std::atof(argv[++i]);
        } else if (arg == "--online-triplet-stdp-slow-decay" && i + 1 < argc) {
            config.onlineTripletStdpSlowDecay = std::atof(argv[++i]);
        } else if (arg == "--online-voltage-plasticity-enabled") {
            config.onlineVoltagePlasticityEnabled = true;
        } else if (arg == "--online-voltage-plasticity-gain" && i + 1 < argc) {
            config.onlineVoltagePlasticityGain = std::atof(argv[++i]);
        } else if (arg == "--online-voltage-plasticity-ltp" && i + 1 < argc) {
            config.onlineVoltagePlasticityLtp = std::atof(argv[++i]);
        } else if (arg == "--online-voltage-plasticity-ltd" && i + 1 < argc) {
            config.onlineVoltagePlasticityLtd = std::atof(argv[++i]);
        } else if (arg == "--online-voltage-plasticity-decay" && i + 1 < argc) {
            config.onlineVoltagePlasticityDecay = std::atof(argv[++i]);
        } else if (arg == "--online-voltage-plasticity-threshold" && i + 1 < argc) {
            config.onlineVoltagePlasticityThreshold = std::atof(argv[++i]);
        } else if (arg == "--online-replay-queue-capacity" && i + 1 < argc) {
            config.onlineReplayQueueCapacity = std::atoi(argv[++i]);
        } else if (arg == "--online-replay-delay-steps" && i + 1 < argc) {
            config.onlineReplayDelaySteps = std::atoi(argv[++i]);
        } else if (arg == "--online-replay-pause-interval" && i + 1 < argc) {
            config.onlineReplayPauseInterval = std::atoi(argv[++i]);
        } else if (arg == "--online-replay-batch-size" && i + 1 < argc) {
            config.onlineReplayBatchSize = std::atoi(argv[++i]);
        } else if (arg == "--online-replay-uncertainty-threshold" && i + 1 < argc) {
            config.onlineReplayUncertaintyThreshold = std::atof(argv[++i]);
        } else if (arg == "--online-eligibility-trace-decay" && i + 1 < argc) {
            config.onlineEligibilityTraceDecay = std::atof(argv[++i]);
        } else if (arg == "--online-neuromodulator-eligibility-enabled") {
            config.onlineNeuromodulatorEligibilityEnabled = true;
        } else if (arg == "--online-neuromodulator-uncertainty-gain" && i + 1 < argc) {
            config.onlineNeuromodulatorUncertaintyGain = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--online-neuromodulator-positive-reward-gain" && i + 1 < argc) {
            config.onlineNeuromodulatorPositiveRewardGain =
                std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--online-neuromodulator-eligibility-max" && i + 1 < argc) {
            config.onlineNeuromodulatorEligibilityMax =
                std::max(1.0, std::atof(argv[++i]));
        } else if (arg == "--online-replay-variable-delay-enabled") {
            config.onlineReplayVariableDelayEnabled = true;
        } else if (arg == "--online-replay-success-consolidation-enabled") {
            config.onlineReplaySuccessConsolidationEnabled = true;
        } else if (arg == "--online-replay-success-consolidation-capacity" && i + 1 < argc) {
            config.onlineReplaySuccessConsolidationCapacity = std::max(0, std::atoi(argv[++i]));
        } else if (arg == "--online-replay-success-consolidation-batch-size" && i + 1 < argc) {
            config.onlineReplaySuccessConsolidationBatchSize =
                std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--online-replay-success-consolidation-repeats" && i + 1 < argc) {
            config.onlineReplaySuccessConsolidationRepeats =
                std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--online-context-uncertainty-gain" && i + 1 < argc) {
            config.onlineContextUncertaintyGain = std::atof(argv[++i]);
        } else if (arg == "--online-context-disagreement-gain" && i + 1 < argc) {
            config.onlineContextDisagreementGain = std::atof(argv[++i]);
        } else if (arg == "--corpus-disagreement-gain" && i + 1 < argc) {
            config.corpusDisagreementGain = std::atof(argv[++i]);
        } else if (arg == "--prediction-error-feedback-enabled") {
            config.predictionErrorFeedbackEnabled = true;
        } else if (arg == "--prediction-error-feedback-gain" && i + 1 < argc) {
            config.predictionErrorFeedbackGain = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--prediction-error-feedback-threshold" && i + 1 < argc) {
            config.predictionErrorFeedbackThreshold =
                std::clamp(std::atof(argv[++i]), 0.0, 1.0);
        } else if (arg == "--prediction-error-feedback-max-penalty" && i + 1 < argc) {
            config.predictionErrorFeedbackMaxPenalty =
                std::clamp(std::atof(argv[++i]), 0.0, 1.0);
        } else if (arg == "--prediction-error-feedback-min-confidence" && i + 1 < argc) {
            config.predictionErrorFeedbackMinConfidence =
                std::clamp(std::atof(argv[++i]), 0.0, 1.0);
        } else if (arg == "--hemisphere-convergent-code-enabled") {
            config.hemisphereConvergentCodeEnabled = true;
        } else if (arg == "--hemisphere-convergent-summary-bins" && i + 1 < argc) {
            config.hemisphereConvergentSummaryBins = std::max(2, std::atoi(argv[++i]));
        } else if (arg == "--hemisphere-convergent-residual-gain" && i + 1 < argc) {
            config.hemisphereConvergentResidualGain = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--hemisphere-convergent-interaction-gain" && i + 1 < argc) {
            config.hemisphereConvergentInteractionGain = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--hemisphere-topographic-stage-enabled") {
            config.hemisphereTopographicStageEnabled = true;
        } else if (arg == "--hemisphere-topographic-residual-gain" && i + 1 < argc) {
            config.hemisphereTopographicResidualGain = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--hemisphere-topographic-continuity-gain" && i + 1 < argc) {
            config.hemisphereTopographicContinuityGain = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--hemisphere-topographic-junction-gain" && i + 1 < argc) {
            config.hemisphereTopographicJunctionGain = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--hemisphere-topographic-auxiliary-gain" && i + 1 < argc) {
            config.hemisphereTopographicAuxiliaryGain = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--figure-ground-stage-enabled") {
            config.figureGroundStageEnabled = true;
        } else if (arg == "--figure-ground-mask-enabled") {
            config.figureGroundStageEnabled = true;
            config.figureGroundMaskEnabled = true;
        } else if (arg == "--figure-ground-classifier-enabled") {
            config.figureGroundStageEnabled = true;
            config.figureGroundClassifierEnabled = true;
        } else if (arg == "--figure-ground-object-memory-enabled") {
            config.figureGroundStageEnabled = true;
            config.figureGroundObjectMemoryEnabled = true;
        } else if (arg == "--recurrent-sensory-state-enabled") {
            config.recurrentSensoryStateEnabled = true;
        } else if (arg == "--recurrent-population-readout-enabled") {
            config.recurrentSensoryStateEnabled = true;
            config.recurrentPopulationReadoutEnabled = true;
        } else if (arg == "--recurrent-object-state-readout-enabled") {
            config.recurrentSensoryStateEnabled = true;
            config.recurrentObjectStateReadoutEnabled = true;
        } else if (arg == "--recurrent-sensory-cycles" && i + 1 < argc) {
            config.recurrentSensoryStateEnabled = true;
            config.recurrentSensoryCycles = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--recurrent-sensory-feedforward-gain" && i + 1 < argc) {
            config.recurrentSensoryStateEnabled = true;
            config.recurrentSensoryFeedforwardGain = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--recurrent-sensory-state-gain" && i + 1 < argc) {
            config.recurrentSensoryStateEnabled = true;
            config.recurrentSensoryStateGain = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--recurrent-sensory-figure-ground-gain" && i + 1 < argc) {
            config.recurrentSensoryStateEnabled = true;
            config.recurrentSensoryFigureGroundGain = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--recurrent-sensory-continuity-gain" && i + 1 < argc) {
            config.recurrentSensoryStateEnabled = true;
            config.recurrentSensoryContinuityGain = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--recurrent-sensory-callosal-gain" && i + 1 < argc) {
            config.recurrentSensoryStateEnabled = true;
            config.recurrentSensoryCallosalGain = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--recurrent-object-state-units-per-class" && i + 1 < argc) {
            config.recurrentSensoryStateEnabled = true;
            config.recurrentObjectStateReadoutEnabled = true;
            config.recurrentObjectStateUnitsPerClass = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--recurrent-object-state-cycles" && i + 1 < argc) {
            config.recurrentSensoryStateEnabled = true;
            config.recurrentObjectStateReadoutEnabled = true;
            config.recurrentObjectStateCycles = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--recurrent-object-state-input-gain" && i + 1 < argc) {
            config.recurrentSensoryStateEnabled = true;
            config.recurrentObjectStateReadoutEnabled = true;
            config.recurrentObjectStateInputGain = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--recurrent-object-state-self-gain" && i + 1 < argc) {
            config.recurrentSensoryStateEnabled = true;
            config.recurrentObjectStateReadoutEnabled = true;
            config.recurrentObjectStateSelfGain = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--recurrent-object-state-class-support-gain" && i + 1 < argc) {
            config.recurrentSensoryStateEnabled = true;
            config.recurrentObjectStateReadoutEnabled = true;
            config.recurrentObjectStateClassSupportGain = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--recurrent-object-state-competition-gain" && i + 1 < argc) {
            config.recurrentSensoryStateEnabled = true;
            config.recurrentObjectStateReadoutEnabled = true;
            config.recurrentObjectStateCompetitionGain = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--figure-ground-cycles" && i + 1 < argc) {
            config.figureGroundCycles = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--figure-ground-border-gain" && i + 1 < argc) {
            config.figureGroundBorderGain = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--figure-ground-surface-gain" && i + 1 < argc) {
            config.figureGroundSurfaceGain = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--figure-ground-competition-gain" && i + 1 < argc) {
            config.figureGroundCompetitionGain = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--figure-ground-feedback-gain" && i + 1 < argc) {
            config.figureGroundFeedbackGain = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--figure-ground-residual-gain" && i + 1 < argc) {
            config.figureGroundResidualGain = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--figure-ground-mask-gain" && i + 1 < argc) {
            config.figureGroundStageEnabled = true;
            config.figureGroundMaskEnabled = true;
            config.figureGroundMaskGain = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--figure-ground-classifier-gain" && i + 1 < argc) {
            config.figureGroundStageEnabled = true;
            config.figureGroundClassifierEnabled = true;
            config.figureGroundClassifierGain = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--figure-ground-object-memory-gain" && i + 1 < argc) {
            config.figureGroundStageEnabled = true;
            config.figureGroundObjectMemoryEnabled = true;
            config.figureGroundObjectMemoryGain = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--figure-ground-object-memory-keep-fraction" && i + 1 < argc) {
            config.figureGroundStageEnabled = true;
            config.figureGroundObjectMemoryEnabled = true;
            config.figureGroundObjectMemoryKeepFraction =
                std::clamp(std::atof(argv[++i]), 0.05, 1.0);
        } else if (arg == "--online-confusion-cluster-gain" && i + 1 < argc) {
            config.onlineConfusionClusterGain = std::atof(argv[++i]);
        } else if (arg == "--online-confusion-cluster-decay" && i + 1 < argc) {
            config.onlineConfusionClusterDecay = std::atof(argv[++i]);
        } else if (arg == "--focus-adjustment-enabled") {
            config.focusAdjustmentEnabled = true;
        } else if (arg == "--focus-adjustment-margin-threshold" && i + 1 < argc) {
            config.focusAdjustmentMarginThreshold = std::atof(argv[++i]);
        } else if (arg == "--focus-adjustment-zoom" && i + 1 < argc) {
            config.focusAdjustmentZoom = std::atof(argv[++i]);
        } else if (arg == "--focus-adjustment-shift-px" && i + 1 < argc) {
            config.focusAdjustmentShiftPx = std::atof(argv[++i]);
        } else if (arg == "--training-curriculum-groups" && i + 1 < argc) {
            config.trainingCurriculumSpec = argv[++i];
        } else if (arg == "--training-review-fraction" && i + 1 < argc) {
            config.trainingReviewFraction = std::clamp(std::atof(argv[++i]), 0.0, 1.0);
        } else if (arg == "--training-augmentation-variants" && i + 1 < argc) {
            config.trainingAugmentationVariants = std::max(0, std::atoi(argv[++i]));
        } else if (arg == "--training-augmentation-shift-px" && i + 1 < argc) {
            config.trainingAugmentationShiftPx = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--training-augmentation-rotation-deg" && i + 1 < argc) {
            config.trainingAugmentationRotationDeg = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--training-augmentation-noise-std" && i + 1 < argc) {
            config.trainingAugmentationNoiseStd = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--guided-saccades-enabled") {
            config.guidedSaccadesEnabled = true;
        } else if (arg == "--guided-saccade-fixations" && i + 1 < argc) {
            config.guidedSaccadeFixations = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--guided-saccade-candidate-peaks" && i + 1 < argc) {
            config.guidedSaccadeCandidatePeaks = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--guided-saccade-jitter-px" && i + 1 < argc) {
            config.guidedSaccadeJitterPx = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--guided-saccade-zoom" && i + 1 < argc) {
            config.guidedSaccadeZoom = std::max(1e-3, std::atof(argv[++i]));
        } else if (arg == "--guided-saccade-ior-strength" && i + 1 < argc) {
            config.guidedSaccadeIorStrength = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--active-inference-enabled") {
            config.activeInferenceEnabled = true;
        } else if (arg == "--active-inference-fixations" && i + 1 < argc) {
            config.activeInferenceFixations = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--active-inference-min-fixations" && i + 1 < argc) {
            config.activeInferenceMinFixations = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--active-inference-remap-enabled" && i + 1 < argc) {
            config.activeInferenceRemapEnabled = std::atoi(argv[++i]) != 0;
        } else if (arg == "--active-inference-shift-px" && i + 1 < argc) {
            config.activeInferenceShiftPx = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--active-inference-zoom" && i + 1 < argc) {
            config.activeInferenceZoom = std::max(1e-3, std::atof(argv[++i]));
        } else if (arg == "--active-inference-uncertainty-threshold" && i + 1 < argc) {
            config.activeInferenceUncertaintyThreshold =
                std::clamp(std::atof(argv[++i]), 0.0, 1.0);
        } else if (arg == "--active-inference-ior-strength" && i + 1 < argc) {
            config.activeInferenceIorStrength = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--flow-audit-enabled") {
            config.flowAuditEnabled = true;
        } else if (arg == "--flow-audit-sample-limit" && i + 1 < argc) {
            config.flowAuditSampleLimit = std::max(0, std::atoi(argv[++i]));
        } else if (arg == "--flow-audit-output-prefix" && i + 1 < argc) {
            config.flowAuditOutputPrefix = argv[++i];
        } else if (arg == "--use-features") {
            config.useFeatures = true;
        } else if (arg == "--use-activations") {
            config.useFeatures = false;
        } else if (arg == "--focus-groups" && i + 1 < argc) {
            config.focusGroupSpec = argv[++i];
        } else if (arg == "--focus-limit-per-label" && i + 1 < argc) {
            config.focusLimitPerLabel = std::atoi(argv[++i]);
        } else if (arg == "--focus-only") {
            config.focusOnly = true;
        } else if (arg == "--help") {
            std::cout
                << "Usage: retina_classification [options]\n"
                << "  --config <path>\n"
                << "  --train-images <path>\n"
                << "  --train-labels <path>\n"
                << "  --test-images <path>\n"
                << "  --test-labels <path>\n"
                << "  --input-domain emnist|mnist\n"
                << "  --input-variant <letters|digits|balanced|byclass|bymerge>\n"
                << "  --input-apply-transform <0|1>\n"
                << "  --examples-per-class <n>\n"
                << "  --test-limit <n>\n"
                << "  --grid-size <n>\n"
                << "  --grid-sizes <n1,n2,...>\n"
                << "  --num-orientations <n>\n"
                << "  --edge-threshold <v>\n"
                << "  --edge-operator sobel|gabor|quadrature_gabor|orientation_energy|dog\n"
                << "  --encoding-strategy rate|temporal|population\n"
                << "  --classifier majority|weighted_similarity|weighted_distance|hierarchical\n"
                << "  --activation-mode binary|similarity|hybrid\n"
                << "  --hierarchical-groups <l1,l2;...>\n"
                << "  --hierarchical-coarse-strategy majority|weighted_similarity|weighted_distance\n"
                << "  --hierarchical-fine-strategy majority|weighted_similarity|weighted_distance\n"
                << "  --hierarchical-coarse-k <n>\n"
                << "  --hierarchical-fine-k <n>\n"
                << "  --knn-k <n>\n"
                << "  --classifier-exponent <v>\n"
                << "  --online-correction-repeats <n>\n"
                << "  --online-exemplar-budget-per-class <n>\n"
                << "  --online-centroid-lr <v>\n"
                << "  --online-positive-reward-gain <v>\n"
                << "  --online-negative-reward-gain <v>\n"
                << "  --online-reward-stdp-enabled\n"
                << "  --online-reward-stdp-gain <v>\n"
                << "  --online-reward-stdp-ltp <v>\n"
                << "  --online-reward-stdp-ltd <v>\n"
                << "  --online-triplet-stdp-enabled\n"
                << "  --online-triplet-stdp-gain <v>\n"
                << "  --online-triplet-stdp-ltp <v>\n"
                << "  --online-triplet-stdp-ltd <v>\n"
                << "  --online-triplet-stdp-fast-decay <v>\n"
                << "  --online-triplet-stdp-slow-decay <v>\n"
                << "  --online-voltage-plasticity-enabled\n"
                << "  --online-voltage-plasticity-gain <v>\n"
                << "  --online-voltage-plasticity-ltp <v>\n"
                << "  --online-voltage-plasticity-ltd <v>\n"
                << "  --online-voltage-plasticity-decay <v>\n"
                << "  --online-voltage-plasticity-threshold <v>\n"
                << "  --online-replay-queue-capacity <n>\n"
                << "  --online-replay-delay-steps <n>\n"
                << "  --online-replay-pause-interval <n>\n"
                << "  --online-replay-batch-size <n>\n"
                << "  --online-replay-uncertainty-threshold <v>\n"
                << "  --online-eligibility-trace-decay <v>\n"
                << "  --online-neuromodulator-eligibility-enabled\n"
                << "  --online-neuromodulator-uncertainty-gain <v>\n"
                << "  --online-neuromodulator-positive-reward-gain <v>\n"
                << "  --online-neuromodulator-eligibility-max <v>\n"
                << "  --online-replay-variable-delay-enabled\n"
                << "  --online-replay-success-consolidation-enabled\n"
                << "  --online-replay-success-consolidation-capacity <n>\n"
                << "  --online-replay-success-consolidation-batch-size <n>\n"
                << "  --online-replay-success-consolidation-repeats <n>\n"
                << "  --online-context-uncertainty-gain <v>\n"
                << "  --online-context-disagreement-gain <v>\n"
                << "  --corpus-disagreement-gain <v>\n"
                << "  --prediction-error-feedback-enabled\n"
                << "  --prediction-error-feedback-gain <v>\n"
                << "  --prediction-error-feedback-threshold <v>\n"
                << "  --prediction-error-feedback-max-penalty <v>\n"
                << "  --prediction-error-feedback-min-confidence <v>\n"
                << "  --hemisphere-convergent-code-enabled\n"
                << "  --hemisphere-convergent-summary-bins <n>\n"
                << "  --hemisphere-convergent-residual-gain <v>\n"
                << "  --hemisphere-convergent-interaction-gain <v>\n"
                << "  --hemisphere-topographic-stage-enabled\n"
                << "  --hemisphere-topographic-residual-gain <v>\n"
                << "  --hemisphere-topographic-continuity-gain <v>\n"
                << "  --hemisphere-topographic-junction-gain <v>\n"
                << "  --hemisphere-topographic-auxiliary-gain <v>\n"
                << "  --figure-ground-stage-enabled\n"
                << "  --figure-ground-mask-enabled\n"
                << "  --figure-ground-classifier-enabled\n"
                << "  --figure-ground-object-memory-enabled\n"
                << "  --recurrent-sensory-state-enabled\n"
                << "  --recurrent-population-readout-enabled\n"
                << "  --recurrent-object-state-readout-enabled\n"
                << "  --recurrent-sensory-cycles <n>\n"
                << "  --recurrent-sensory-feedforward-gain <v>\n"
                << "  --recurrent-sensory-state-gain <v>\n"
                << "  --recurrent-sensory-figure-ground-gain <v>\n"
                << "  --recurrent-sensory-continuity-gain <v>\n"
                << "  --recurrent-sensory-callosal-gain <v>\n"
                << "  --recurrent-object-state-units-per-class <n>\n"
                << "  --recurrent-object-state-cycles <n>\n"
                << "  --recurrent-object-state-input-gain <v>\n"
                << "  --recurrent-object-state-self-gain <v>\n"
                << "  --recurrent-object-state-class-support-gain <v>\n"
                << "  --recurrent-object-state-competition-gain <v>\n"
                << "  --figure-ground-cycles <n>\n"
                << "  --figure-ground-border-gain <v>\n"
                << "  --figure-ground-surface-gain <v>\n"
                << "  --figure-ground-competition-gain <v>\n"
                << "  --figure-ground-feedback-gain <v>\n"
                << "  --figure-ground-residual-gain <v>\n"
                << "  --figure-ground-mask-gain <v>\n"
                << "  --figure-ground-classifier-gain <v>\n"
                << "  --figure-ground-object-memory-gain <v>\n"
                << "  --figure-ground-object-memory-keep-fraction <v>\n"
                << "  --online-confusion-cluster-gain <v>\n"
                << "  --online-confusion-cluster-decay <v>\n"
                << "  --focus-adjustment-enabled\n"
                << "  --focus-adjustment-margin-threshold <v>\n"
                << "  --focus-adjustment-zoom <v>\n"
                << "  --focus-adjustment-shift-px <v>\n"
                << "  --training-curriculum-groups <A,B;C,D;...>\n"
                << "  --training-review-fraction <v>\n"
                << "  --training-augmentation-variants <n>\n"
                << "  --training-augmentation-shift-px <v>\n"
                << "  --training-augmentation-rotation-deg <v>\n"
                << "  --training-augmentation-noise-std <v>\n"
                << "  --active-inference-enabled\n"
                << "  --active-inference-fixations <n>\n"
                << "  --active-inference-min-fixations <n>\n"
                << "  --active-inference-remap-enabled <0|1>\n"
                << "  --active-inference-shift-px <v>\n"
                << "  --active-inference-zoom <v>\n"
                << "  --active-inference-uncertainty-threshold <v>\n"
                << "  --active-inference-ior-strength <v>\n"
                << "  --flow-audit-enabled\n"
                << "  --flow-audit-sample-limit <n>\n"
                << "  --flow-audit-output-prefix <path-prefix>\n"
                << "  --focus-groups <A,B;C,D;...>\n"
                << "  --focus-limit-per-label <n>\n"
                << "  --focus-only\n"
                << "  --use-features | --use-activations\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (config.trainImagesPath.empty() || config.trainLabelsPath.empty() ||
        config.testImagesPath.empty() || config.testLabelsPath.empty()) {
        throw std::runtime_error("Input image/label paths are required");
    }
    if (config.focusOnly && config.focusGroups.empty() && config.focusGroupSpec.empty()) {
        throw std::runtime_error("--focus-only requires --focus-groups");
    }

    return config;
}

} // namespace

int main(int argc, char* argv[]) {
    snnfw::Logger::getInstance().setLevel(spdlog::level::warn);

    try {
        Config config = parseArgs(argc, argv);
        const auto retinaConfigs = buildRetinaConfigs(config);

        std::cout << "=== Retina Classification ===" << std::endl;
        if (!config.configPath.empty()) {
            std::cout << "  Config: " << config.configPath << std::endl;
        }
        std::cout << "  Retina adapters: " << retinaConfigs.size() << std::endl;
        if (config.jepa.enabled) {
            const std::string outputPath = config.jepa.dumpPath.empty()
                ? "build/jepa_stage1_latents.json"
                : config.jepa.dumpPath;
            const std::string trainerPath = config.jepa.trainerDumpPath.empty()
                ? "build/jepa_minimal_trainer.json"
                : config.jepa.trainerDumpPath;
            std::cout << "  JEPA taps: enabled"
                      << " (surface=" << snnfw::jepa::tapSurfaceName(config.jepa.tapSurface)
                      << ", mask_mode=" << snnfw::jepa::maskModeName(config.jepa.maskMode)
                      << ", target_mode="
                      << snnfw::jepa::targetModeName(config.jepa.targetMode)
                      << ", visible_branches=" << config.jepa.visibleBranchCount
                      << ", hidden_branches=" << config.jepa.hiddenBranchCount
                      << ", max_samples=" << config.jepa.maxSamples
                      << ", branch_tokens="
                      << (config.jepa.includeBranchTokens ? "on" : "off")
                      << ", hemisphere_token="
                      << (config.jepa.includeHemisphereToken ? "on" : "off")
                      << ", enforce_no_leakage="
                      << (config.jepa.enforceNoLeakage ? "on" : "off")
                      << ", path=" << outputPath << ")"
                      << std::endl;
            std::cout << "  JEPA trainer: "
                      << (config.jepa.trainerEnabled ? "enabled" : "disabled")
                      << " (epochs=" << config.jepa.trainerEpochs
                      << ", projection_dim=" << config.jepa.projectionDim
                      << ", lr=" << config.jepa.trainerLearningRate
                      << ", weight_decay=" << config.jepa.trainerWeightDecay
                      << ", target_ema_decay=" << config.jepa.targetEmaDecay
                      << ", variance_floor=" << config.jepa.varianceFloor
                      << ", variance_penalty=" << config.jepa.variancePenalty
                      << ", path=" << trainerPath << ")"
                      << std::endl;
            std::cout << "  JEPA probe: "
                      << (config.jepa.probeEnabled ? "enabled" : "disabled")
                      << std::endl;
        }
        for (const auto& retinaConfig : retinaConfigs) {
            std::cout << "    - " << retinaConfig.name
                      << ": grid="
                      << retinaConfig.getIntParam("grid_size", config.gridSize)
                      << "x" << retinaConfig.getIntParam("grid_size", config.gridSize)
                      << ", orientations="
                      << retinaConfig.getIntParam("num_orientations", config.numOrientations)
                      << ", edge="
                      << retinaConfig.getStringParam("edge_operator", config.edgeOperator)
                      << ", threshold="
                      << retinaConfig.getDoubleParam("edge_threshold", config.edgeThreshold)
                      << ", lgn_burst_tonic="
                      << (retinaConfig.getIntParam("lgn_burst_tonic_enabled", 0) != 0 ? "on" : "off")
                      << ", sensory_triplet_bcm="
                      << (retinaConfig.getIntParam("sensory_triplet_bcm_enabled", 0) != 0 ? "on" : "off")
                      << ", fusion_weight="
                      << retinaConfig.getDoubleParam("fusion_weight", 1.0)
                      << ", hemisphere="
                      << retinaConfig.getStringParam("hemisphere", "default")
                      << ", rotation="
                      << retinaConfig.getDoubleParam("rotation_deg", 0.0)
                      << ", shift=("
                      << retinaConfig.getDoubleParam("shift_x_px", 0.0) << ","
                      << retinaConfig.getDoubleParam("shift_y_px", 0.0) << ")"
                      << ", encoding="
                      << retinaConfig.getStringParam("encoding_strategy", config.encodingStrategy)
                      << ", activation_mode="
                      << retinaConfig.getStringParam("activation_mode", config.activationMode)
                      << ", attach_path="
                      << retinaConfig.getStringParam("attach_path", "")
                      << std::endl;
        }
        std::cout << "  Classifier=" << config.classifier
                  << ", representation=" << (config.useFeatures ? "features" : "activations")
                  << std::endl;
        if (config.bilateralFusion) {
            std::cout << "  Fusion mode=bilateral"
                      << ", stage1=" << config.stage1Classifier
                      << " (k=" << (config.stage1K > 0 ? config.stage1K : config.knnK) << ")"
                      << ", fusion=" << config.fusionClassifier
                      << " (k=" << (config.fusionK > 0 ? config.fusionK : config.knnK) << ")"
                      << ", holdout/class="
                      << (config.fusionHoldoutPerClass > 0 ? config.fusionHoldoutPerClass : 0)
                      << ", fusion_features=" << config.fusionFeatureMode
                      << ", corpus_vote_gain=" << config.corpusVoteGain
                      << ", corpus_margin_gain=" << config.corpusMarginGain
                      << ", corpus_centroid_gain=" << config.corpusCentroidGain
                      << ", corpus_neighbor_gain=" << config.corpusNeighborGain
                      << ", corpus_disagreement_gain=" << config.corpusDisagreementGain
                      << ", prediction_error_feedback="
                      << (config.predictionErrorFeedbackEnabled ? "on" : "off")
                      << ", prediction_error_feedback_gain="
                      << config.predictionErrorFeedbackGain
                      << ", prediction_error_feedback_threshold="
                      << config.predictionErrorFeedbackThreshold
                      << ", prediction_error_feedback_max_penalty="
                      << config.predictionErrorFeedbackMaxPenalty
                      << ", prediction_error_feedback_min_confidence="
                      << config.predictionErrorFeedbackMinConfidence
                      << ", hemisphere_convergent_code="
                      << (config.hemisphereConvergentCodeEnabled ? "on" : "off")
                      << ", hemisphere_convergent_summary_bins="
                      << config.hemisphereConvergentSummaryBins
                      << ", hemisphere_convergent_residual_gain="
                      << config.hemisphereConvergentResidualGain
                      << ", hemisphere_convergent_interaction_gain="
                      << config.hemisphereConvergentInteractionGain
                      << ", hemisphere_topographic_stage="
                      << (config.hemisphereTopographicStageEnabled ? "on" : "off")
                      << ", hemisphere_topographic_residual_gain="
                      << config.hemisphereTopographicResidualGain
                      << ", hemisphere_topographic_continuity_gain="
                      << config.hemisphereTopographicContinuityGain
                      << ", hemisphere_topographic_junction_gain="
                      << config.hemisphereTopographicJunctionGain
                      << ", hemisphere_topographic_auxiliary_gain="
                      << config.hemisphereTopographicAuxiliaryGain
                      << ", figure_ground_stage="
                      << (config.figureGroundStageEnabled ? "on" : "off")
                      << ", recurrent_sensory_state="
                      << (config.recurrentSensoryStateEnabled ? "on" : "off")
                      << ", recurrent_population_readout="
                      << (config.recurrentPopulationReadoutEnabled ? "on" : "off")
                      << ", recurrent_object_state_readout="
                      << (config.recurrentObjectStateReadoutEnabled ? "on" : "off")
                      << ", recurrent_sensory_cycles="
                      << config.recurrentSensoryCycles
                      << ", recurrent_sensory_feedforward_gain="
                      << config.recurrentSensoryFeedforwardGain
                      << ", recurrent_sensory_state_gain="
                      << config.recurrentSensoryStateGain
                      << ", recurrent_sensory_figure_ground_gain="
                      << config.recurrentSensoryFigureGroundGain
                      << ", recurrent_sensory_continuity_gain="
                      << config.recurrentSensoryContinuityGain
                      << ", recurrent_sensory_callosal_gain="
                      << config.recurrentSensoryCallosalGain
                      << ", recurrent_object_state_units_per_class="
                      << config.recurrentObjectStateUnitsPerClass
                      << ", recurrent_object_state_cycles="
                      << config.recurrentObjectStateCycles
                      << ", recurrent_object_state_input_gain="
                      << config.recurrentObjectStateInputGain
                      << ", recurrent_object_state_self_gain="
                      << config.recurrentObjectStateSelfGain
                      << ", recurrent_object_state_class_support_gain="
                      << config.recurrentObjectStateClassSupportGain
                      << ", recurrent_object_state_competition_gain="
                      << config.recurrentObjectStateCompetitionGain
                      << ", figure_ground_mask="
                      << (config.figureGroundMaskEnabled ? "on" : "off")
                      << ", figure_ground_classifier="
                      << (config.figureGroundClassifierEnabled ? "on" : "off")
                      << ", figure_ground_object_memory="
                      << (config.figureGroundObjectMemoryEnabled ? "on" : "off")
                      << ", figure_ground_cycles=" << config.figureGroundCycles
                      << ", figure_ground_border_gain=" << config.figureGroundBorderGain
                      << ", figure_ground_surface_gain=" << config.figureGroundSurfaceGain
                      << ", figure_ground_competition_gain="
                      << config.figureGroundCompetitionGain
                      << ", figure_ground_feedback_gain="
                      << config.figureGroundFeedbackGain
                      << ", figure_ground_residual_gain="
                      << config.figureGroundResidualGain
                      << ", figure_ground_mask_gain="
                      << config.figureGroundMaskGain
                      << ", figure_ground_classifier_gain="
                      << config.figureGroundClassifierGain
                      << ", figure_ground_object_memory_gain="
                      << config.figureGroundObjectMemoryGain
                      << ", figure_ground_object_memory_keep_fraction="
                      << config.figureGroundObjectMemoryKeepFraction
                      << ", online_repeats=" << config.onlineCorrectionRepeats
                      << ", online_budget/class=" << config.onlineExemplarBudgetPerClass
                      << ", online_centroid_lr=" << config.onlineCentroidLr
                      << ", online_pos_gain=" << config.onlinePositiveRewardGain
                      << ", online_neg_gain=" << config.onlineNegativeRewardGain
                      << ", reward_stdp=" << (config.onlineRewardStdpEnabled ? "on" : "off")
                      << ", reward_stdp_gain=" << config.onlineRewardStdpGain
                      << ", reward_stdp_ltp=" << config.onlineRewardStdpLtp
                      << ", reward_stdp_ltd=" << config.onlineRewardStdpLtd
                      << ", triplet_stdp=" << (config.onlineTripletStdpEnabled ? "on" : "off")
                      << ", triplet_stdp_gain=" << config.onlineTripletStdpGain
                      << ", triplet_stdp_ltp=" << config.onlineTripletStdpLtp
                      << ", triplet_stdp_ltd=" << config.onlineTripletStdpLtd
                      << ", triplet_stdp_fast_decay=" << config.onlineTripletStdpFastDecay
                      << ", triplet_stdp_slow_decay=" << config.onlineTripletStdpSlowDecay
                      << ", voltage_plasticity="
                      << (config.onlineVoltagePlasticityEnabled ? "on" : "off")
                      << ", voltage_plasticity_gain=" << config.onlineVoltagePlasticityGain
                      << ", voltage_plasticity_ltp=" << config.onlineVoltagePlasticityLtp
                      << ", voltage_plasticity_ltd=" << config.onlineVoltagePlasticityLtd
                      << ", voltage_plasticity_decay=" << config.onlineVoltagePlasticityDecay
                      << ", voltage_plasticity_threshold="
                      << config.onlineVoltagePlasticityThreshold
                      << ", replay_capacity=" << config.onlineReplayQueueCapacity
                      << ", replay_delay_steps=" << config.onlineReplayDelaySteps
                      << ", replay_pause_interval=" << config.onlineReplayPauseInterval
                      << ", replay_batch_size=" << config.onlineReplayBatchSize
                      << ", replay_uncertainty=" << config.onlineReplayUncertaintyThreshold
                      << ", eligibility_decay=" << config.onlineEligibilityTraceDecay
                      << ", neuromodulator_eligibility="
                      << (config.onlineNeuromodulatorEligibilityEnabled ? "on" : "off")
                      << ", neuromodulator_uncertainty_gain="
                      << config.onlineNeuromodulatorUncertaintyGain
                      << ", neuromodulator_positive_reward_gain="
                      << config.onlineNeuromodulatorPositiveRewardGain
                      << ", neuromodulator_eligibility_max="
                      << config.onlineNeuromodulatorEligibilityMax
                      << ", replay_variable_delay="
                      << (config.onlineReplayVariableDelayEnabled ? "on" : "off")
                      << ", replay_success_consolidation="
                      << (config.onlineReplaySuccessConsolidationEnabled ? "on" : "off")
                      << ", replay_success_capacity="
                      << config.onlineReplaySuccessConsolidationCapacity
                      << ", replay_success_batch_size="
                      << config.onlineReplaySuccessConsolidationBatchSize
                      << ", replay_success_repeats="
                      << config.onlineReplaySuccessConsolidationRepeats
                      << ", context_uncertainty_gain=" << config.onlineContextUncertaintyGain
                      << ", context_disagreement_gain=" << config.onlineContextDisagreementGain
                      << ", confusion_cluster_gain=" << config.onlineConfusionClusterGain
                      << ", confusion_cluster_decay=" << config.onlineConfusionClusterDecay
                      << ", focus_adjustment=" << (config.focusAdjustmentEnabled ? "on" : "off")
                      << ", focus_margin_threshold=" << config.focusAdjustmentMarginThreshold
                      << ", focus_zoom=" << config.focusAdjustmentZoom
                      << ", focus_shift_px=" << config.focusAdjustmentShiftPx
                      << ", saccade_fixations=" << config.saccadeFixations
                      << ", saccade_training_only=" << (config.saccadeTrainingOnly ? "on" : "off")
                      << ", saccade_jitter_px=" << config.saccadeJitterPx
                      << ", saccade_zoom=" << config.saccadeZoom
                      << ", guided_saccades=" << (config.guidedSaccadesEnabled ? "on" : "off")
                      << ", guided_fixations="
                      << (config.guidedSaccadeFixations > 0
                              ? config.guidedSaccadeFixations
                              : config.saccadeFixations)
                      << ", guided_candidate_peaks=" << config.guidedSaccadeCandidatePeaks
                      << ", guided_jitter_px="
                      << (config.guidedSaccadeJitterPx > 0.0
                              ? config.guidedSaccadeJitterPx
                              : config.saccadeJitterPx)
                      << ", guided_ior=" << config.guidedSaccadeIorStrength
                      << ", curriculum="
                      << (config.trainingCurriculumSpec.empty()
                              ? std::string("<none>")
                              : config.trainingCurriculumSpec)
                      << ", review_fraction=" << config.trainingReviewFraction
                      << ", train_aug_variants=" << config.trainingAugmentationVariants
                      << ", train_aug_shift_px=" << config.trainingAugmentationShiftPx
                      << ", train_aug_rotation_deg=" << config.trainingAugmentationRotationDeg
                      << ", train_aug_noise_std=" << config.trainingAugmentationNoiseStd
                      << ", active_inference=" << (config.activeInferenceEnabled ? "on" : "off")
                      << ", active_fixations=" << config.activeInferenceFixations
                      << ", active_min_fixations=" << config.activeInferenceMinFixations
                      << ", active_shift_px=" << config.activeInferenceShiftPx
                      << ", active_zoom=" << config.activeInferenceZoom
                      << ", active_uncertainty=" << config.activeInferenceUncertaintyThreshold
                      << ", active_ior=" << config.activeInferenceIorStrength
                      << ", active_remap=" << (config.activeInferenceRemapEnabled ? "on" : "off")
                      << ", flow_audit=" << (config.flowAuditEnabled ? "on" : "off")
                      << ", flow_audit_limit=" << config.flowAuditSampleLimit
                      << ", flow_audit_prefix="
                      << (defaultFlowAuditOutputPrefix(config).empty()
                              ? std::string("<none>")
                              : defaultFlowAuditOutputPrefix(config))
                      << ", fusion_path=" << (config.fusionPath.empty() ? "<none>" : config.fusionPath)
                      << std::endl;
        }
        if (config.classifier == "hierarchical") {
            std::cout << "  Hierarchy groups: " << config.hierarchicalGroups
                      << ", coarse=" << config.hierarchicalCoarseStrategy
                      << ", fine=" << config.hierarchicalFineStrategy
                      << ", coarse_k="
                      << (config.hierarchicalCoarseK > 0 ? config.hierarchicalCoarseK : config.knnK)
                      << ", fine_k="
                      << (config.hierarchicalFineK > 0 ? config.hierarchicalFineK : config.knnK)
                      << std::endl;
        }
        if (!config.focusGroups.empty()) {
            std::cout << "  Focus groups: ";
            for (size_t i = 0; i < config.focusGroups.size(); ++i) {
                if (i > 0) {
                    std::cout << ", ";
                }
                std::cout << "[" << labelGroupToString(config.focusGroups[i], config) << "]";
            }
            if (config.focusLimitPerLabel > 0) {
                std::cout << " limit/class=" << config.focusLimitPerLabel;
            }
            if (config.focusOnly) {
                std::cout << " (focus only)";
            }
            std::cout << std::endl;
        }

        VisualDomainConfig domainConfig;
        domainConfig.source = config.inputDomain;
        domainConfig.variant = config.inputVariant;
        domainConfig.applyTransform = config.inputApplyTransform;
        auto trainLoader = snnfw::domain::createVisualDomainAdapter(domainConfig);
        auto testLoader = snnfw::domain::createVisualDomainAdapter(domainConfig);
        if (!trainLoader->load(config.trainImagesPath, config.trainLabelsPath) ||
            !testLoader->load(config.testImagesPath, config.testLabelsPath)) {
            std::cerr << "Failed to load input dataset" << std::endl;
            return 1;
        }
        configureClassMetadata(config, *trainLoader);
        if (!config.focusGroupSpec.empty()) {
            config.focusGroups = parseLabelGroups(config.focusGroupSpec, config);
        }
        if (!config.trainingCurriculumSpec.empty()) {
            config.trainingCurriculumGroups =
                parseLabelGroups(config.trainingCurriculumSpec, config);
        }
        if (config.focusOnly && config.focusGroups.empty()) {
            throw std::runtime_error("--focus-only requires --focus-groups");
        }
        std::cout << "  Input domain=" << trainLoader->domainName()
                  << ", classes=" << config.numClasses << std::endl;
        if (!config.trainingCurriculumGroups.empty()) {
            std::cout << "  Training curriculum: "
                      << labelGroupsToString(config.trainingCurriculumGroups, config)
                      << std::endl;
        }
        if (config.trainingReviewFraction > 0.0) {
            std::cout << "  Training review fraction/class=" << config.trainingReviewFraction
                      << std::endl;
        }

        const auto trainIndicesByLabel = collectLabelIndices(*trainLoader, config.numClasses);
        const auto testIndicesByLabel = collectLabelIndices(*testLoader, config.numClasses);
        const auto selectedTestIndices =
            selectBalancedTestIndices(testIndicesByLabel, config.testLimit, config.seed + 1U);
        const auto focusLabels = flattenLabelGroups(config.focusGroups);
        const auto focusedTestIndices =
            selectFocusedTestIndices(testIndicesByLabel, focusLabels, config.focusLimitPerLabel,
                                     config.seed + 7U);

        if (!config.bilateralFusion) {
            auto retinas = createRetinaAdapters(retinaConfigs);
            auto classifier = makeClassifierStrategy(config.classifier, config.knnK,
                                                     config.classifierExponent, config);

            const auto selectedTrainIndices =
                selectStratifiedIndices(trainIndicesByLabel, config.examplesPerClass, config.seed);
            const auto trainingSchedule =
                buildTrainingSchedule(selectedTrainIndices, *trainLoader, config, config.seed);
            std::vector<ClassificationStrategy::LabeledPattern> trainingPatterns;
            trainingPatterns.reserve(trainingSchedule.primaryIndices.size() +
                                     trainingSchedule.reviewIndices.size());
            if (!config.trainingCurriculumGroups.empty() || !trainingSchedule.reviewIndices.empty()) {
                std::cout << "  Training schedule: primary="
                          << trainingSchedule.primaryIndices.size()
                          << ", review=" << trainingSchedule.reviewIndices.size() << std::endl;
            }

            const auto trainingStart = std::chrono::high_resolution_clock::now();
            const size_t totalTrainingItems =
                trainingSchedule.primaryIndices.size() + trainingSchedule.reviewIndices.size();
            const int trainingStep = trainingProgressStep(totalTrainingItems);
            setRetinaHomeostaticLearning(retinas, true);
            for (size_t i : trainingSchedule.primaryIndices) {
                const auto& image = trainLoader->getStimulus(i);
                const int label = image.label;
                if (label < 0 || label >= config.numClasses) {
                    continue;
                }
                trainingPatterns.emplace_back(
                    extractPattern(retinas, image, config, true,
                                   !config.useFeatures && config.activationMode != "binary",
                                   i),
                    label);
                maybePrintTrainingProgress("Training patterns",
                                           trainingPatterns.size(),
                                           totalTrainingItems,
                                           trainingStep,
                                           trainingStart);
            }
            for (size_t i : trainingSchedule.reviewIndices) {
                const auto& image = trainLoader->getStimulus(i);
                const int label = image.label;
                if (label < 0 || label >= config.numClasses) {
                    continue;
                }
                trainingPatterns.emplace_back(
                    extractPattern(retinas, image, config, true,
                                   !config.useFeatures && config.activationMode != "binary",
                                   i ^ 0x9e3779b97f4a7c15ULL),
                    label);
                maybePrintTrainingProgress("Training patterns",
                                           trainingPatterns.size(),
                                           totalTrainingItems,
                                           trainingStep,
                                           trainingStart);
            }
            setRetinaHomeostaticLearning(retinas, false);

            std::cout << "  Stored training patterns: " << trainingPatterns.size() << std::endl;
            if (config.jepa.enabled) {
                std::cout << "  JEPA tap export skipped: unilateral path is not wired yet"
                          << std::endl;
            }
            const auto end = std::chrono::high_resolution_clock::now();
            const double trainingSeconds =
                std::chrono::duration<double>(end - trainingStart).count();

            if (!config.focusOnly) {
                const auto eval = evaluatePatterns(retinas, *testLoader, selectedTestIndices, config,
                                                   *classifier, trainingPatterns, "Testing");
                const double accuracy =
                    100.0 * static_cast<double>(eval.correct) /
                    static_cast<double>(std::max(1, eval.tested));

                std::cout << "\n=== Results ===" << std::endl;
                std::cout << "  Accuracy: " << std::fixed << std::setprecision(2) << accuracy
                          << "%" << std::endl;
                std::cout << "  Correct: " << eval.correct << "/" << eval.tested << std::endl;
                std::cout << "  Elapsed: " << std::fixed << std::setprecision(2)
                          << (trainingSeconds + eval.seconds) << "s" << std::endl;

                printPerClassAccuracy(eval.confusion, config);
                printTopConfusions(eval.confusion, config);
            }

            if (!config.focusGroups.empty()) {
                const auto focusedEval = evaluatePatterns(
                    retinas, *testLoader, focusedTestIndices, config, *classifier,
                    trainingPatterns, "Focus");
                const double focusedAccuracy =
                    100.0 * static_cast<double>(focusedEval.correct) /
                    static_cast<double>(std::max(1, focusedEval.tested));

                std::cout << "\n=== Focused Results ===" << std::endl;
                std::cout << "  Accuracy: " << std::fixed << std::setprecision(2)
                          << focusedAccuracy << "%" << std::endl;
                std::cout << "  Correct: " << focusedEval.correct << "/" << focusedEval.tested
                          << std::endl;
                std::cout << "  Elapsed: " << std::fixed << std::setprecision(2)
                          << focusedEval.seconds << "s" << std::endl;

                printPerClassAccuracy(focusedEval.confusion, config);
                printTopConfusions(focusedEval.confusion, config);
                printFocusedFamilyReport(focusedEval.confusion, config.focusGroups, config);
            }
        } else {
            const auto groupedConfigs = groupRetinaConfigsByHemisphere(retinaConfigs, config);
            if (groupedConfigs.size() < 2) {
                throw std::runtime_error(
                    "bilateral fusion requires at least two hemisphere groups");
            }

            std::vector<HemisphereRuntime> hemispheres;
            hemispheres.reserve(groupedConfigs.size());
            std::cout << "  Bilateral fusion enabled: " << groupedConfigs.size()
                      << " hemisphere groups" << std::endl;
            if (config.hierarchicalRetinaLayout) {
                std::cout << "  Hierarchical Retina layout: enabled" << std::endl;
            }
            for (const auto& [name, configsForHemisphere] : groupedConfigs) {
                HemisphereRuntime hemisphere;
                hemisphere.name = name;
                hemisphere.retinas = createRetinaAdapters(configsForHemisphere);
                initializeHemisphereRuntime(hemisphere, config.numClasses);
                hemispheres.push_back(std::move(hemisphere));
                std::cout << "    * " << name << ": " << configsForHemisphere.size()
                          << " retina branches" << std::endl;
                if (config.hierarchicalRetinaLayout) {
                    for (const auto& retinaConfig : configsForHemisphere) {
                        std::cout << "      - " << retinaConfig.name
                                  << " -> " << retinaConfig.getStringParam("attach_path", "<unbound>")
                                  << std::endl;
                    }
                }
            }

            const int stage1K = config.stage1K > 0 ? config.stage1K : config.knnK;
            const int fusionK = config.fusionK > 0 ? config.fusionK : config.knnK;
            const auto split = splitTrainingIndices(trainIndicesByLabel, config.examplesPerClass,
                                                    config.fusionHoldoutPerClass, config.seed,
                                                    useCorpusCallosumFusion(config));
            const auto stage1TrainingSchedule =
                buildTrainingSchedule(split.stage1Indices, *trainLoader, config, config.seed);
            if (!config.trainingCurriculumGroups.empty() ||
                !stage1TrainingSchedule.reviewIndices.empty()) {
                std::cout << "  Stage1 training schedule: primary="
                          << stage1TrainingSchedule.primaryIndices.size()
                          << ", review=" << stage1TrainingSchedule.reviewIndices.size()
                          << std::endl;
            }

            const auto trainingStart = std::chrono::high_resolution_clock::now();
            if (hemispheres.size() > 1) {
                std::cout << "  Hemisphere stage1 extraction: parallel workers="
                          << hemisphereWorkerCount(hemispheres.size()) << std::endl;
            }
            const size_t totalStage1Items =
                stage1TrainingSchedule.primaryIndices.size() +
                stage1TrainingSchedule.reviewIndices.size();
            const int stage1ProgressStep = trainingProgressStep(totalStage1Items);
            const auto trainingArtifacts = runHemisphereTasks<HemisphereTrainingArtifacts>(
                hemispheres.size(),
                [&](size_t hemisphereIndex) {
                    HemisphereTrainingArtifacts artifacts;
                    auto& hemisphere = hemispheres[hemisphereIndex];
                    setRetinaHomeostaticLearning(hemisphere.retinas, true);
                    if (config.flowAuditEnabled) {
                        initializeFlowAuditTraining(
                            artifacts, hemisphere.retinas, config.numClasses);
                    }
                    artifacts.trainingPatterns.reserve(stage1TrainingSchedule.primaryIndices.size() +
                                                       stage1TrainingSchedule.reviewIndices.size());
                    artifacts.trainingSourceIndices.reserve(
                        stage1TrainingSchedule.primaryIndices.size() +
                        stage1TrainingSchedule.reviewIndices.size());
                    for (size_t i : stage1TrainingSchedule.primaryIndices) {
                        const auto& image = trainLoader->getStimulus(i);
                        const int label = image.label;
                        if (label < 0 || label >= config.numClasses) {
                            continue;
                        }
                        auto batch = extractStage1TrainingPatternBatch(
                            hemisphere.retinas,
                            image,
                            config,
                            !config.useFeatures && config.activationMode != "binary",
                            i,
                            config.flowAuditEnabled);
                        for (size_t patternIndex = 0; patternIndex < batch.patterns.size();
                             ++patternIndex) {
                            artifacts.trainingPatterns.emplace_back(batch.patterns[patternIndex], label);
                            if (config.flowAuditEnabled &&
                                patternIndex < batch.captures.size()) {
                                accumulateFlowAuditTrainingCapture(
                                    artifacts, batch.captures[patternIndex], label);
                            }
                            artifacts.trainingSourceIndices.push_back(i);
                            maybePrintTrainingProgress(hemisphere.name + " stage1 patterns",
                                                       artifacts.trainingPatterns.size(),
                                                       totalStage1Items,
                                                       stage1ProgressStep,
                                                       trainingStart);
                        }
                    }
                    for (size_t i : stage1TrainingSchedule.reviewIndices) {
                        const auto& image = trainLoader->getStimulus(i);
                        const int label = image.label;
                        if (label < 0 || label >= config.numClasses) {
                            continue;
                        }
                        auto batch = extractStage1TrainingPatternBatch(
                            hemisphere.retinas,
                            image,
                            config,
                            !config.useFeatures && config.activationMode != "binary",
                            i ^ 0x9e3779b97f4a7c15ULL,
                            config.flowAuditEnabled);
                        for (size_t patternIndex = 0; patternIndex < batch.patterns.size();
                             ++patternIndex) {
                            artifacts.trainingPatterns.emplace_back(batch.patterns[patternIndex], label);
                            if (config.flowAuditEnabled &&
                                patternIndex < batch.captures.size()) {
                                accumulateFlowAuditTrainingCapture(
                                    artifacts, batch.captures[patternIndex], label);
                            }
                            artifacts.trainingSourceIndices.push_back(i);
                            maybePrintTrainingProgress(hemisphere.name + " stage1 patterns",
                                                       artifacts.trainingPatterns.size(),
                                                       totalStage1Items,
                                                       stage1ProgressStep,
                                                       trainingStart);
                        }
                    }
                    setRetinaHomeostaticLearning(hemisphere.retinas, false);
                    finalizeFlowAuditTraining(artifacts);
                    return artifacts;
                });
            for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
                auto& hemisphere = hemispheres[hemisphereIndex];
                hemisphere.rawTrainingPatterns = trainingArtifacts[hemisphereIndex].trainingPatterns;
                hemisphere.trainingSourceIndices =
                    trainingArtifacts[hemisphereIndex].trainingSourceIndices;
                hemisphere.figureGroundObjectMemoryPatterns.clear();
                hemisphere.retinaPatternSizes.clear();
                hemisphere.trainingPatterns.clear();
            }

            if (useRecurrentSensoryState(config)) {
                bool alignedSources = !hemispheres.empty();
                size_t alignedCount = hemispheres.empty()
                    ? 0
                    : std::min(hemispheres.front().rawTrainingPatterns.size(),
                               hemispheres.front().trainingSourceIndices.size());
                for (size_t hemisphereIndex = 1; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
                    alignedCount = std::min(alignedCount,
                                            std::min(hemispheres[hemisphereIndex].rawTrainingPatterns.size(),
                                                     hemispheres[hemisphereIndex]
                                                         .trainingSourceIndices.size()));
                }
                for (size_t patternIndex = 0; alignedSources && patternIndex < alignedCount;
                     ++patternIndex) {
                    const size_t sourceIndex =
                        hemispheres.front().trainingSourceIndices[patternIndex];
                    for (size_t hemisphereIndex = 1; hemisphereIndex < hemispheres.size();
                         ++hemisphereIndex) {
                        if (hemispheres[hemisphereIndex].trainingSourceIndices[patternIndex] !=
                            sourceIndex) {
                            alignedSources = false;
                            break;
                        }
                    }
                }

                for (auto& hemisphere : hemispheres) {
                    hemisphere.trainingPatterns.reserve(hemisphere.rawTrainingPatterns.size());
                }

                if (alignedSources && alignedCount > 0) {
                    for (size_t patternIndex = 0; patternIndex < alignedCount; ++patternIndex) {
                        const size_t sourceIndex =
                            hemispheres.front().trainingSourceIndices[patternIndex];
                        const auto& image = trainLoader->getStimulus(sourceIndex);
                        const auto recurrentResults = buildRecurrentSensoryPatterns(
                            hemispheres,
                            image,
                            config,
                            true,
                            sourceIndex ^
                                (static_cast<size_t>(patternIndex + 1) *
                                 0x9e3779b97f4a7c15ULL));
                        for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size();
                             ++hemisphereIndex) {
                            auto& hemisphere = hemispheres[hemisphereIndex];
                            const auto& rawPattern = hemisphere.rawTrainingPatterns[patternIndex];
                            std::vector<double> settledPattern =
                                hemisphereIndex < recurrentResults.size()
                                    ? recurrentResults[hemisphereIndex].pattern
                                    : std::vector<double>{};
                            hemisphere.trainingPatterns.emplace_back(
                                settledPattern.empty() ? rawPattern.pattern
                                                       : std::move(settledPattern),
                                rawPattern.label);
                        }
                        maybePrintTrainingProgress("Recurrent sensory state patterns",
                                                   patternIndex + 1,
                                                   alignedCount,
                                                   stage1ProgressStep,
                                                   trainingStart);
                    }
                }

                for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size();
                     ++hemisphereIndex) {
                    auto& hemisphere = hemispheres[hemisphereIndex];
                    const size_t startIndex =
                        alignedSources ? alignedCount : static_cast<size_t>(0);
                    if (!alignedSources) {
                        hemisphere.trainingPatterns.clear();
                    }
                    for (size_t patternIndex = startIndex;
                         patternIndex < hemisphere.rawTrainingPatterns.size();
                         ++patternIndex) {
                        const auto& rawPattern = hemisphere.rawTrainingPatterns[patternIndex];
                        std::vector<double> settledPattern;
                        if (patternIndex < hemisphere.trainingSourceIndices.size()) {
                            const size_t sourceIndex =
                                hemisphere.trainingSourceIndices[patternIndex];
                            const auto& image = trainLoader->getStimulus(sourceIndex);
                            auto recurrentResult = buildRecurrentSensoryPattern(
                                hemisphere,
                                image,
                                config,
                                true,
                                sourceIndex ^
                                    (static_cast<size_t>(patternIndex + 1) *
                                     0x9e3779b97f4a7c15ULL));
                            settledPattern = std::move(recurrentResult.pattern);
                        }
                        hemisphere.trainingPatterns.emplace_back(
                            settledPattern.empty() ? rawPattern.pattern
                                                   : std::move(settledPattern),
                            rawPattern.label);
                    }
                    if (hemisphere.trainingPatterns.empty()) {
                        hemisphere.trainingPatterns = hemisphere.rawTrainingPatterns;
                    }
                    if (useFigureGroundObjectMemory(config)) {
                        hemisphere.figureGroundObjectMemoryPatterns.reserve(
                            hemisphere.trainingPatterns.size());
                        for (const auto& labeledPattern : hemisphere.trainingPatterns) {
                            const auto figureGroundState = buildHemisphereFigureGroundState(
                                hemisphere, labeledPattern.pattern, config);
                            auto objectPattern = buildFigureGroundObjectMemoryPattern(
                                hemisphere, labeledPattern.pattern, figureGroundState, config);
                            if (!objectPattern.empty()) {
                                hemisphere.figureGroundObjectMemoryPatterns.emplace_back(
                                    std::move(objectPattern), labeledPattern.label);
                            }
                        }
                    }
                }
            } else {
                for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size();
                     ++hemisphereIndex) {
                    auto& hemisphere = hemispheres[hemisphereIndex];
                    hemisphere.trainingPatterns = hemisphere.rawTrainingPatterns;
                    if ((useHemisphereTopographicStage(config) ||
                         useHemisphereConvergentCode(config) ||
                         useFigureGroundStage(config)) &&
                        !hemisphere.trainingSourceIndices.empty()) {
                        const auto& referenceImage =
                            trainLoader->getStimulus(hemisphere.trainingSourceIndices.front());
                        hemisphere.retinaPatternSizes = inferHemisphereBranchPatternSizes(
                            hemisphere.retinas, referenceImage, config.useFeatures);
                        if (!hemisphere.retinaPatternSizes.empty()) {
                            std::vector<ClassificationStrategy::LabeledPattern> convergentPatterns;
                            convergentPatterns.reserve(hemisphere.rawTrainingPatterns.size());
                            for (const auto& labeledPattern : hemisphere.rawTrainingPatterns) {
                                auto convergentPattern = buildHemisphereClassifierPattern(
                                    hemisphere, labeledPattern.pattern, config);
                                convergentPatterns.emplace_back(
                                    convergentPattern.empty() ? labeledPattern.pattern
                                                              : std::move(convergentPattern),
                                    labeledPattern.label);
                            }
                            hemisphere.trainingPatterns = std::move(convergentPatterns);
                        }
                        if (useFigureGroundObjectMemory(config)) {
                            hemisphere.figureGroundObjectMemoryPatterns.reserve(
                                hemisphere.rawTrainingPatterns.size());
                            for (const auto& labeledPattern : hemisphere.rawTrainingPatterns) {
                                const auto figureGroundState = buildHemisphereFigureGroundState(
                                    hemisphere, labeledPattern.pattern, config);
                                auto objectPattern = buildFigureGroundObjectMemoryPattern(
                                    hemisphere, labeledPattern.pattern, figureGroundState, config);
                                if (!objectPattern.empty()) {
                                    hemisphere.figureGroundObjectMemoryPatterns.emplace_back(
                                        std::move(objectPattern), labeledPattern.label);
                                }
                            }
                        }
                    }
                }
            }

            for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
                auto& hemisphere = hemispheres[hemisphereIndex];
                hemisphere.classifier = makeClassifierStrategy(
                    config.stage1Classifier, stage1K, config.stage1Exponent, config);
                buildHemisphereClassCentroids(hemisphere);
                rebuildHemisphereObjectStateAttractors(hemisphere, config);
                std::cout << "  Hemisphere " << hemisphere.name
                          << " stored stage1 patterns: "
                          << hemisphere.trainingPatterns.size() << std::endl;
                if (useRecurrentObjectStateReadout(config)) {
                    std::cout << "    recurrent object-state attractors: "
                              << hemisphere.recurrentObjectStateAttractors.size()
                              << std::endl;
                }
                if (useFigureGroundObjectMemory(config)) {
                    std::cout << "    figure-ground object-memory patterns: "
                              << hemisphere.figureGroundObjectMemoryPatterns.size() << std::endl;
                }
            }

            maybeExportJepaStage1Taps(hemispheres, *trainLoader, config);
            snnfw::jepa::JepaTrainingArtifacts jepaArtifacts;
            const bool hasJepaModel =
                maybeTrainJepaModel(hemispheres, *trainLoader, config, &jepaArtifacts);

            std::unique_ptr<ClassificationStrategy> fusionClassifier;
            FusionRuntime fusionRuntime;
            initializeFusionRuntime(fusionRuntime, config.numClasses);
            if (useCorpusCallosumFusion(config)) {
                calibrateCorpusCallosumWeights(hemispheres, *trainLoader, split.fusionIndices, config);
                std::cout << "  Corpus-callosum calibration samples: "
                          << split.fusionIndices.size() << std::endl;
                for (const auto& hemisphere : hemispheres) {
                    std::cout << "    - " << hemisphere.name
                              << " overall_weight=" << std::fixed << std::setprecision(3)
                              << hemisphere.overallWeight
                              << ", class_bias[0]="
                              << (!hemisphere.classWeights.empty() ? hemisphere.classWeights[0] : 0.0)
                              << ", vote_bias[0]="
                              << (!hemisphere.predictionWeights.empty() ? hemisphere.predictionWeights[0] : 0.0)
                              << std::endl;
                }
            } else {
                fusionClassifier = makeClassifierStrategy(
                    config.fusionClassifier, fusionK, config.fusionExponent, config);
                fusionRuntime.trainingPatterns.reserve(split.fusionIndices.size());
                const int fusionProgressStep =
                    trainingProgressStep(split.fusionIndices.size());
                for (size_t i : split.fusionIndices) {
                    const auto& image = trainLoader->getStimulus(i);
                    const int label = image.label;
                    if (label < 0 || label >= config.numClasses) {
                        continue;
                    }
                    fusionRuntime.trainingPatterns.emplace_back(
                        buildFusionPattern(hemispheres, image, config), label);
                    maybePrintTrainingProgress("Fusion patterns",
                                               fusionRuntime.trainingPatterns.size(),
                                               split.fusionIndices.size(),
                                               fusionProgressStep,
                                               trainingStart);
                }

                if (fusionRuntime.trainingPatterns.empty()) {
                    throw std::runtime_error(
                        "bilateral fusion produced no fusion training patterns; increase examples-per-class or reduce fusion_holdout_per_class");
                }

                std::cout << "  Stored fusion patterns: " << fusionRuntime.trainingPatterns.size()
                          << " (mode=" << config.fusionFeatureMode << ")" << std::endl;
            }
            const auto end = std::chrono::high_resolution_clock::now();
            const double trainingSeconds =
                std::chrono::duration<double>(end - trainingStart).count();
            const auto baseSeparabilityRuntime = buildSeparabilityDiagnosticsRuntime(
                hemispheres, *trainLoader, split, config, fusionRuntime);
            const auto baseFlowAuditRuntime = buildFlowAuditRuntime(
                hemispheres, trainingArtifacts, *trainLoader, split, config, fusionRuntime);

            if (!config.focusOnly) {
                std::optional<JepaProbeResult> jepaProbeEval;
                if (config.jepa.probeEnabled) {
                    if (!hasJepaModel) {
                        throw std::runtime_error(
                            "JEPA probe requires jepa_enabled=1 and jepa_trainer_enabled=1");
                    }
                    jepaProbeEval = evaluateJepaProbe(
                        hemispheres,
                        *trainLoader,
                        *testLoader,
                        selectedTestIndices,
                        config,
                        jepaArtifacts.model,
                        "JEPA Probe");
                }

                auto separabilityRuntime = baseSeparabilityRuntime;
                auto flowAuditRuntime = baseFlowAuditRuntime;
                resetOrientationFlowDiagnostics(hemispheres);
                const auto eval = useCorpusCallosumFusion(config)
                    ? evaluateCorpusCallosumPatterns(
                          hemispheres, *testLoader, selectedTestIndices, config,
                          separabilityRuntime.enabled ? &separabilityRuntime : nullptr,
                          flowAuditRuntime.enabled ? &flowAuditRuntime : nullptr,
                          "Testing")
                    : evaluateBilateralPatterns(
                          hemispheres, *testLoader, selectedTestIndices, config, *fusionClassifier,
                          fusionRuntime,
                          separabilityRuntime.enabled ? &separabilityRuntime : nullptr,
                          flowAuditRuntime.enabled ? &flowAuditRuntime : nullptr,
                          "Testing");
                const double accuracy =
                    100.0 * static_cast<double>(eval.correct) /
                    static_cast<double>(std::max(1, eval.tested));

                std::cout << "\n=== Results ===" << std::endl;
                std::cout << "  Accuracy: " << std::fixed << std::setprecision(2) << accuracy
                          << "%" << std::endl;
                std::cout << "  Correct: " << eval.correct << "/" << eval.tested << std::endl;
                std::cout << "  Elapsed: " << std::fixed << std::setprecision(2)
                          << (trainingSeconds + eval.seconds) << "s" << std::endl;
                printHemisphereAgreementSummary(eval);
                printHemisphereStage1Summary(eval, config);
                printBilateralAttributionSummary(eval, config);
                printActiveInferenceSummary(eval, config);
                printFocusAdjustmentSummary(eval);
                printOnlineCorrectionSummary(eval);
                printSeparabilitySummary(eval, config);
                printOrientationFlowSummary(hemispheres);
                printPatchInputSummary(hemispheres);
                if (flowAuditRuntime.enabled) {
                    writeFlowAuditArtifacts(flowAuditRuntime, "testing");
                }

                printPerClassAccuracy(eval.confusion, config);
                printTopConfusions(eval.confusion, config);

                if (jepaProbeEval.has_value()) {
                    const double probeAccuracy =
                        100.0 * static_cast<double>(jepaProbeEval->correct) /
                        static_cast<double>(std::max(1, jepaProbeEval->tested));
                    std::cout << "\n=== JEPA Probe Results ===" << std::endl;
                    std::cout << "  Accuracy: " << std::fixed << std::setprecision(2)
                              << probeAccuracy << "%" << std::endl;
                    std::cout << "  Correct: " << jepaProbeEval->correct << "/"
                              << jepaProbeEval->tested << std::endl;
                    std::cout << "  Delta vs baseline: " << std::showpos
                              << std::fixed << std::setprecision(2)
                              << (probeAccuracy - accuracy) << " pts" << std::noshowpos
                              << std::endl;
                    std::cout << "  Elapsed: " << std::fixed << std::setprecision(2)
                              << (trainingSeconds + jepaProbeEval->seconds) << "s"
                              << std::endl;
                    printPerClassAccuracy(jepaProbeEval->confusion, config);
                    printTopConfusions(jepaProbeEval->confusion, config);
                }
            }

            if (!config.focusGroups.empty()) {
                auto separabilityRuntime = baseSeparabilityRuntime;
                auto flowAuditRuntime = baseFlowAuditRuntime;
                resetOrientationFlowDiagnostics(hemispheres);
                const auto focusedEval = useCorpusCallosumFusion(config)
                    ? evaluateCorpusCallosumPatterns(
                          hemispheres, *testLoader, focusedTestIndices, config,
                          separabilityRuntime.enabled ? &separabilityRuntime : nullptr,
                          flowAuditRuntime.enabled ? &flowAuditRuntime : nullptr,
                          "Focus")
                    : evaluateBilateralPatterns(
                          hemispheres, *testLoader, focusedTestIndices, config, *fusionClassifier,
                          fusionRuntime,
                          separabilityRuntime.enabled ? &separabilityRuntime : nullptr,
                          flowAuditRuntime.enabled ? &flowAuditRuntime : nullptr,
                          "Focus");
                const double focusedAccuracy =
                    100.0 * static_cast<double>(focusedEval.correct) /
                    static_cast<double>(std::max(1, focusedEval.tested));

                std::cout << "\n=== Focused Results ===" << std::endl;
                std::cout << "  Accuracy: " << std::fixed << std::setprecision(2)
                          << focusedAccuracy << "%" << std::endl;
                std::cout << "  Correct: " << focusedEval.correct << "/" << focusedEval.tested
                          << std::endl;
                std::cout << "  Elapsed: " << std::fixed << std::setprecision(2)
                          << focusedEval.seconds << "s" << std::endl;
                printHemisphereAgreementSummary(focusedEval);
                printHemisphereStage1Summary(focusedEval, config);
                printBilateralAttributionSummary(focusedEval, config);
                printActiveInferenceSummary(focusedEval, config);
                printFocusAdjustmentSummary(focusedEval);
                printOnlineCorrectionSummary(focusedEval);
                printSeparabilitySummary(focusedEval, config);
                printOrientationFlowSummary(hemispheres);
                printPatchInputSummary(hemispheres);
                if (flowAuditRuntime.enabled) {
                    writeFlowAuditArtifacts(flowAuditRuntime, "focus");
                }

                printPerClassAccuracy(focusedEval.confusion, config);
                printTopConfusions(focusedEval.confusion, config);
                printFocusedFamilyReport(focusedEval.confusion, config.focusGroups, config);
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}

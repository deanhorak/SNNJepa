#ifndef SNNFW_DECLARATIVE_NETWORK_IR_H
#define SNNFW_DECLARATIVE_NETWORK_IR_H

#include <string>
#include <vector>
#include <map>
#include <optional>

namespace snnfw {
namespace declarative {

/// Reusable neuron parameter set
struct NeuronParamsIR {
    std::string name;
    double windowSizeMs = 500.0;
    double similarityThreshold = 0.93;
    int maxReferencePatterns = 500;
    std::string similarityMetric = "cosine";
};

/// A population of neurons within a layer
struct PopulationIR {
    std::string name;
    int count = 0;
    std::string neuronParams;      // reference to NeuronParamsIR by name
    std::string gridLayout = "";   // e.g. "7x7", "" for flat
};

/// Layer within a column
struct LayerIR {
    std::string name;
    std::vector<PopulationIR> populations;
};

/// Column with feature selectivity
struct ColumnIR {
    std::string name;
    std::map<std::string, double> properties;  // orientation, spatial_frequency, etc.
    std::vector<LayerIR> layers;
};

/// Template for generating multiple columns from parameter ranges
struct ColumnTemplateIR {
    std::string templateName;
    std::vector<LayerIR> layers;
    std::vector<double> orientations;
    std::vector<double> frequencies;
    std::string namingPattern = "Orient_{orientation}_Freq_{frequency}";
};

/// Nucleus containing columns
struct NucleusIR {
    std::string name;
    std::vector<ColumnIR> columns;
    std::optional<ColumnTemplateIR> columnTemplate;
};

/// Region containing nuclei
struct RegionIR {
    std::string name;
    std::vector<NucleusIR> nuclei;
};

/// Lobe containing regions
struct LobeIR {
    std::string name;
    std::vector<RegionIR> regions;
};

/// Hemisphere containing lobes
struct HemisphereIR {
    std::string name;
    std::vector<LobeIR> lobes;
};

/// Brain — top of hierarchy
struct BrainIR {
    std::string name;
    std::vector<HemisphereIR> hemispheres;
};

/// Connectivity rule between named populations
struct ProjectionIR {
    std::string name;
    std::string source;        // path: "V1/*/L4/L4_neurons" or "InputGrid"
    std::string target;        // path: "V1/*/L2_3/L2_3_neurons" or "OutputLayer"
    std::string pattern = "random_sparse";
    double probability = 1.0;
    double weight = 1.0;
    double maxWeight = 10.0;
    double delay = 1.5;
    std::string scope = "intra_column"; // "intra_column", "inter_column", "global"
    std::string synapseGroup = "";      // "InputToL4", "L4ToL5", "L5ToOutput"

    // Tiled receptive field parameters (for pattern == "tiled_receptive_field")
    int tilesPerSide = 4;       // number of tiles along each axis
    int tilesPerColumn = 3;     // tiles assigned to each column
    int inputGridSize = 28;     // size of the input grid (assumed square)
    int targetGridSize = 7;     // size of the L4 grid per column (assumed square)
};

/// Input layer description
struct InputLayerIR {
    std::string name = "InputGrid";
    int rows = 28;
    int cols = 28;
    double latencyMs = 15.0;
    double pixelThreshold = 0.4;
    NeuronParamsIR neuronParams;
};

/// Output layer description
struct OutputLayerIR {
    std::string name = "OutputLayer";
    int numClasses = 26;
    int neuronsPerClass = 3;
    NeuronParamsIR neuronParams;
};

/// Gabor filter configuration
struct GaborConfigIR {
    double freqLow = 8.0;
    double freqHigh = 3.0;
    double sigma = 2.0;
    double gamma = 0.5;
    double threshold = 0.5;
    int kernelSize = 7;
};

/// Saccade fixation region
struct FixationRegionIR {
    std::string name;
    int rowStart = 0, rowEnd = 0, colStart = 0, colEnd = 0;
};

/// Saccade attention mechanism config
struct SaccadeConfigIR {
    bool enabled = false;
    int numFixations = 4;
    std::vector<FixationRegionIR> regions;
};

/// Simulation parameters
struct SimulationConfigIR {
    int spikeProcessorTimeSlices = 10000;
    int spikeProcessorThreads = 24;
    bool realTimeSync = false;
    unsigned int connectivitySeed = 0;  // 0 = nondeterministic
    // STDP
    bool stdpEnabled = true;
    double stdpLtdScale = 0.3;
    double stdpLtdWindowMs = 70.0;
    bool traceStdp = true;
    bool freezeStdpDuringTesting = true;
    bool enableStdpEligibilityGate = true;
    int stdpEligibilityMinUpdates = 1;
    int stdpEligibilityMinLtp = 0;
    double stdpEligibilityThreshold = -0.002;
    double stdpEligibilityLtdPenalty = 0.5;
    // Competition
    int l4Keep = 8;
    int l5Keep = 8;
    bool enableSimilarityCompetition = true;
    double l4SimilarityWeight = 0.25;
    double l5SimilarityWeight = 0.60;
    bool traceSimilarityCompetition = false;
    bool enableL5Inhibition = true;
    bool enableL5InterColumnInhibition = false;
    double l5InterColumnInhibit = 0.1;
    double l5InterColumnMinOverlap = 0.2;
    double l5InterColumnWinnerScale = 0.25;
    double l5InterColumnMaxInhibit = 1.0;
    double l5InterColumnMaxOrientationDeltaDeg = 45.0;
    double l5InterColumnMaxFrequencyOctaveDelta = 0.75;
    int l5InterColumnMaxNeighbors = 8;
    int maskMinActive = 6;  // minimum active input pixels for column to participate
    // Timing
    double interImageGapMs = 550.0;
};

/// Declarative adapter endpoint configuration
struct AdapterConfigIR {
    std::string name;
    std::string type;
    std::string role;        // "sensory" or "motor" (optional, inferred from type when empty)
    std::string bindTo;      // "input" / "output" (optional hint for runtime wiring)
    double temporalWindowMs = 10.0;
    std::map<std::string, double> doubleParams;
    std::map<std::string, int> intParams;
    std::map<std::string, std::string> stringParams;
};

/// Declarative classifier/readout configuration
struct ClassificationConfigIR {
    std::string type;  // majority, weighted_similarity, weighted_distance, hierarchical
    int k = 5;
    double distanceExponent = 1.0;
    std::map<std::string, double> doubleParams;
    std::map<std::string, int> intParams;
    std::map<std::string, std::string> stringParams;
};

/// The complete intermediate representation for a network
struct NetworkIR {
    std::string formatVersion = "1.0";
    std::string sourceFormat;  // "snnframe_json", "sonata", "neuroml", "hoc"

    /// Named neuron parameter sets (define once, reference many times)
    std::map<std::string, NeuronParamsIR> neuronParamSets;

    /// Hierarchy
    BrainIR brain;

    /// Special layers
    InputLayerIR inputLayer;
    OutputLayerIR outputLayer;

    /// Connectivity rules
    std::vector<ProjectionIR> projections;

    /// Feature extraction
    GaborConfigIR gabor;
    SaccadeConfigIR saccades;

    /// Simulation parameters
    SimulationConfigIR simulation;

    /// External adapters
    std::vector<AdapterConfigIR> adapters;

    /// Declarative classifier/readout
    ClassificationConfigIR classification;

    /// Validate the IR, returning true if valid
    bool validate() const;

    /// Get validation error messages
    std::vector<std::string> getValidationErrors() const;
};

} // namespace declarative
} // namespace snnfw

#endif // SNNFW_DECLARATIVE_NETWORK_IR_H

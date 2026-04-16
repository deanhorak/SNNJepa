#include "snnfw/experiment/SpikeEncoder.h"
#include "snnfw/adapters/AdapterFactory.h"
#include "snnfw/adapters/EMNISTAdapter.h"
#include "snnfw/features/EdgeOperator.h"
#include <chrono>
#include <thread>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace snnfw {
namespace experiment {

namespace {

struct SanitizedFixationRegion {
    int rowStart = 0;
    int rowEnd = -1;
    int colStart = 0;
    int colEnd = -1;
};

struct SaliencyTileMap {
    int gridRows = 0;
    int gridCols = 0;
    std::vector<double> scores;
};

std::string lowercaseCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

SanitizedFixationRegion sanitizeFixationRegion(
    const ExperimentConfig::FixationRegion& region,
    int rows,
    int cols) {
    SanitizedFixationRegion sanitized;
    sanitized.rowStart = std::clamp(region.rowStart, 0, std::max(0, rows - 1));
    sanitized.rowEnd = std::clamp(region.rowEnd, sanitized.rowStart, std::max(0, rows - 1));
    sanitized.colStart = std::clamp(region.colStart, 0, std::max(0, cols - 1));
    sanitized.colEnd = std::clamp(region.colEnd, sanitized.colStart, std::max(0, cols - 1));
    return sanitized;
}

bool isFullFixationRegion(
    const ExperimentConfig::FixationRegion& region,
    int rows,
    int cols) {
    const std::string name = lowercaseCopy(region.name);
    if (name.find("full") != std::string::npos) {
        return true;
    }
    return region.rowStart <= 0 && region.colStart <= 0 &&
           region.rowEnd >= rows - 1 && region.colEnd >= cols - 1;
}

EMNISTLoader::Image maskImageToRegion(
    const EMNISTLoader::Image& image,
    const SanitizedFixationRegion& region) {
    EMNISTLoader::Image masked = image;
    if (static_cast<int>(masked.pixels.size()) != image.rows * image.cols) {
        return masked;
    }

    std::fill(masked.pixels.begin(), masked.pixels.end(), static_cast<uint8_t>(0));
    for (int row = region.rowStart; row <= region.rowEnd; ++row) {
        for (int col = region.colStart; col <= region.colEnd; ++col) {
            const size_t idx = static_cast<size_t>(row * image.cols + col);
            masked.pixels[idx] = image.pixels[idx];
        }
    }
    return masked;
}

void applyImageGain(EMNISTLoader::Image& image, double gain) {
    if (gain >= 0.999) {
        return;
    }
    gain = std::clamp(gain, 0.0, 1.0);
    for (auto& pixel : image.pixels) {
        pixel = static_cast<uint8_t>(std::clamp(std::lround(static_cast<double>(pixel) * gain), 0L, 255L));
    }
}

double computeRegionEnergy(
    const EMNISTLoader::Image& image,
    const SanitizedFixationRegion& region) {
    double energy = 0.0;
    for (int row = region.rowStart; row <= region.rowEnd; ++row) {
        for (int col = region.colStart; col <= region.colEnd; ++col) {
            const size_t idx = static_cast<size_t>(row * image.cols + col);
            energy += static_cast<double>(image.pixels[idx]);
        }
    }
    return energy;
}

std::vector<uint8_t> extractSquareTileRegion(
    const EMNISTLoader::Image& image,
    int rowStart,
    int rowEnd,
    int colStart,
    int colEnd) {
    const int tileHeight = std::max(1, rowEnd - rowStart + 1);
    const int tileWidth = std::max(1, colEnd - colStart + 1);
    const int regionSize = std::max(3, std::min(tileHeight, tileWidth));
    const int centerRow = (rowStart + rowEnd) / 2;
    const int centerCol = (colStart + colEnd) / 2;
    const int half = regionSize / 2;
    const int squareRowStart = std::clamp(centerRow - half, 0, std::max(0, image.rows - regionSize));
    const int squareColStart = std::clamp(centerCol - half, 0, std::max(0, image.cols - regionSize));

    std::vector<uint8_t> region(static_cast<size_t>(regionSize * regionSize), 0);
    for (int row = 0; row < regionSize; ++row) {
        for (int col = 0; col < regionSize; ++col) {
            const int srcRow = squareRowStart + row;
            const int srcCol = squareColStart + col;
            const size_t srcIdx = static_cast<size_t>(srcRow * image.cols + srcCol);
            region[static_cast<size_t>(row * regionSize + col)] = image.pixels[srcIdx];
        }
    }
    return region;
}

double computeTileFeatureSaliency(const std::vector<uint8_t>& region, int regionSize) {
    static const auto sobel = []() {
        features::EdgeOperator::Config config;
        config.name = "sobel";
        config.numOrientations = 8;
        config.edgeThreshold = 0.05;
        return features::EdgeOperatorFactory::create("sobel", config);
    }();
    static const auto dog = []() {
        features::EdgeOperator::Config config;
        config.name = "dog";
        config.numOrientations = 8;
        config.edgeThreshold = 0.03;
        config.doubleParams["sigma1"] = 0.9;
        config.doubleParams["sigma2"] = 1.6;
        config.intParams["kernel_size"] = 5;
        return features::EdgeOperatorFactory::create("dog", config);
    }();

    const auto sobelFeatures = sobel->extractEdges(region, regionSize);
    const auto dogFeatures = dog->extractEdges(region, regionSize);

    double sobelSum = 0.0;
    double dogSum = 0.0;
    int activeOrientations = 0;
    for (double value : sobelFeatures) {
        sobelSum += value;
        if (value > 0.20) {
            ++activeOrientations;
        }
    }
    for (double value : dogFeatures) {
        dogSum += value;
    }

    const double diversity =
        static_cast<double>(activeOrientations) / static_cast<double>(std::max<size_t>(1, sobelFeatures.size()));
    return sobelSum + (0.35 * dogSum) + (0.50 * diversity);
}

SaliencyTileMap buildFeatureSaliencyMap(const EMNISTLoader::Image& image, int gridSize) {
    SaliencyTileMap map;
    map.gridRows = std::max(1, gridSize);
    map.gridCols = std::max(1, gridSize);
    map.scores.resize(static_cast<size_t>(map.gridRows * map.gridCols), 0.0);

    for (int gr = 0; gr < map.gridRows; ++gr) {
        const int rowStart = (gr * image.rows) / map.gridRows;
        const int rowEnd = ((gr + 1) * image.rows) / map.gridRows - 1;
        for (int gc = 0; gc < map.gridCols; ++gc) {
            const int colStart = (gc * image.cols) / map.gridCols;
            const int colEnd = ((gc + 1) * image.cols) / map.gridCols - 1;
            const int tileHeight = std::max(1, rowEnd - rowStart + 1);
            const int tileWidth = std::max(1, colEnd - colStart + 1);
            const int regionSize = std::max(3, std::min(tileHeight, tileWidth));
            const auto tileRegion = extractSquareTileRegion(image, rowStart, rowEnd, colStart, colEnd);
            map.scores[static_cast<size_t>(gr * map.gridCols + gc)] =
                computeTileFeatureSaliency(tileRegion, regionSize);
        }
    }

    return map;
}

double computeRegionFeatureSaliency(
    const SaliencyTileMap& tileMap,
    const SanitizedFixationRegion& region,
    int imageRows,
    int imageCols,
    int topTilesToAverage) {
    std::vector<double> scores;
    scores.reserve(static_cast<size_t>(tileMap.gridRows * tileMap.gridCols));
    for (int gr = 0; gr < tileMap.gridRows; ++gr) {
        const int rowStart = (gr * imageRows) / tileMap.gridRows;
        const int rowEnd = ((gr + 1) * imageRows) / tileMap.gridRows - 1;
        const int rowCenter = (rowStart + rowEnd) / 2;
        for (int gc = 0; gc < tileMap.gridCols; ++gc) {
            const int colStart = (gc * imageCols) / tileMap.gridCols;
            const int colEnd = ((gc + 1) * imageCols) / tileMap.gridCols - 1;
            const int colCenter = (colStart + colEnd) / 2;
            if (rowCenter < region.rowStart || rowCenter > region.rowEnd ||
                colCenter < region.colStart || colCenter > region.colEnd) {
                continue;
            }
            scores.push_back(tileMap.scores[static_cast<size_t>(gr * tileMap.gridCols + gc)]);
        }
    }

    if (scores.empty()) {
        return 0.0;
    }
    std::sort(scores.begin(), scores.end(), std::greater<double>());
    const int keep = std::max(1, std::min(topTilesToAverage, static_cast<int>(scores.size())));
    double topMean = 0.0;
    for (int i = 0; i < keep; ++i) {
        topMean += scores[static_cast<size_t>(i)];
    }
    topMean /= static_cast<double>(keep);

    double mean = 0.0;
    for (double score : scores) {
        mean += score;
    }
    mean /= static_cast<double>(scores.size());
    return (0.85 * topMean) + (0.15 * mean);
}

void applyTopKAttention(
    EMNISTLoader::Image& image,
    const SanitizedFixationRegion& region,
    int attentionGrid,
    int topK,
    double attentionFloor,
    double attentionPower) {
    if (topK <= 0 || image.pixels.empty()) {
        std::fill(image.pixels.begin(), image.pixels.end(), static_cast<uint8_t>(0));
        return;
    }

    const int regionHeight = region.rowEnd - region.rowStart + 1;
    const int regionWidth = region.colEnd - region.colStart + 1;
    if (regionHeight <= 0 || regionWidth <= 0) {
        return;
    }

    const int gridRows = std::max(1, std::min(attentionGrid, regionHeight));
    const int gridCols = std::max(1, std::min(attentionGrid, regionWidth));

    struct TileScore {
        size_t index = 0;
        int rowStart = 0;
        int rowEnd = 0;
        int colStart = 0;
        int colEnd = 0;
        double score = 0.0;
    };

    std::vector<TileScore> tiles;
    tiles.reserve(static_cast<size_t>(gridRows * gridCols));
    for (int gr = 0; gr < gridRows; ++gr) {
        const int tileRowStart = region.rowStart + (gr * regionHeight) / gridRows;
        const int tileRowEnd = region.rowStart + (((gr + 1) * regionHeight) / gridRows) - 1;
        for (int gc = 0; gc < gridCols; ++gc) {
            const int tileColStart = region.colStart + (gc * regionWidth) / gridCols;
            const int tileColEnd = region.colStart + (((gc + 1) * regionWidth) / gridCols) - 1;
            double score = 0.0;
            for (int row = tileRowStart; row <= tileRowEnd; ++row) {
                for (int col = tileColStart; col <= tileColEnd; ++col) {
                    const size_t idx = static_cast<size_t>(row * image.cols + col);
                    score += static_cast<double>(image.pixels[idx]);
                }
            }
            tiles.push_back({tiles.size(), tileRowStart, tileRowEnd, tileColStart, tileColEnd, score});
        }
    }

    std::vector<TileScore> ranked = tiles;
    std::sort(ranked.begin(), ranked.end(), [](const TileScore& a, const TileScore& b) {
        if (a.score != b.score) {
            return a.score > b.score;
        }
        if (a.rowStart != b.rowStart) {
            return a.rowStart < b.rowStart;
        }
        return a.colStart < b.colStart;
    });

    const int keepCount = std::min(topK, static_cast<int>(ranked.size()));
    const double bestScore = ranked.empty() ? 0.0 : ranked.front().score;
    attentionFloor = std::clamp(attentionFloor, 0.0, 1.0);
    attentionPower = std::max(0.1, attentionPower);
    std::vector<double> tileGains(tiles.size(), 0.0);
    for (size_t rank = 0; rank < ranked.size(); ++rank) {
        const auto& tile = ranked[rank];
        if (tile.score <= 0.0 || bestScore <= 0.0) {
            tileGains[tile.index] = 0.0;
            continue;
        }
        const double normalized = std::clamp(tile.score / bestScore, 0.0, 1.0);
        double gain = attentionFloor +
                      ((1.0 - attentionFloor) * std::pow(normalized, attentionPower));
        if (static_cast<int>(rank) < keepCount) {
            gain = 1.0;
        }
        tileGains[tile.index] = std::clamp(gain, 0.0, 1.0);
    }

    for (const auto& tile : tiles) {
        const double gain = tileGains[tile.index];
        for (int row = tile.rowStart; row <= tile.rowEnd; ++row) {
            for (int col = tile.colStart; col <= tile.colEnd; ++col) {
                const size_t idx = static_cast<size_t>(row * image.cols + col);
                image.pixels[idx] = static_cast<uint8_t>(
                    std::clamp(std::lround(static_cast<double>(image.pixels[idx]) * gain), 0L, 255L));
            }
        }
    }
}

std::vector<double> collectUniqueOrientations(
    const std::vector<declarative::ConstructedNetwork::ColumnGroup>& columns) {
    std::vector<double> orientations;
    orientations.reserve(columns.size());
    for (const auto& col : columns) {
        orientations.push_back(col.orientation);
    }
    std::sort(orientations.begin(), orientations.end());
    orientations.erase(
        std::unique(orientations.begin(), orientations.end(),
                    [](double a, double b) { return std::abs(a - b) < 1e-6; }),
        orientations.end());
    return orientations;
}

std::vector<double> collectUniqueFrequencies(
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
    return frequencies;
}

size_t findNearestOrientationIndex(double orientation, const std::vector<double>& sortedOrientations) {
    size_t bestIndex = 0;
    double bestDelta = std::numeric_limits<double>::max();
    for (size_t i = 0; i < sortedOrientations.size(); ++i) {
        const double rawDelta = std::abs(orientation - sortedOrientations[i]);
        const double delta = std::min(rawDelta, 180.0 - rawDelta);
        if (delta < bestDelta) {
            bestDelta = delta;
            bestIndex = i;
        }
    }
    return bestIndex;
}

size_t findNearestFrequencyIndex(double frequency, const std::vector<double>& sortedFrequencies) {
    size_t bestIndex = 0;
    double bestDelta = std::numeric_limits<double>::max();
    for (size_t i = 0; i < sortedFrequencies.size(); ++i) {
        const double delta = std::abs(frequency - sortedFrequencies[i]);
        if (delta < bestDelta) {
            bestDelta = delta;
            bestIndex = i;
        }
    }
    return bestIndex;
}

double retinaChannelStrength(const std::vector<double>& spikeTimes, double durationMs) {
    if (spikeTimes.empty() || durationMs <= 0.0) {
        return 0.0;
    }
    const double earliest = *std::min_element(spikeTimes.begin(), spikeTimes.end());
    const double normalized = 1.0 - std::clamp(earliest / durationMs, 0.0, 1.0);
    return std::clamp(normalized, 0.0, 1.0);
}

} // namespace

SpikeEncoder::SpikeEncoder(const ExperimentConfig& config,
                           std::shared_ptr<SpikeProcessor> spikeProcessor,
                           std::shared_ptr<NetworkPropagator> propagator,
                           std::shared_ptr<adapters::SensoryAdapter> configuredAdapter)
    : config_(config)
    , spikeProcessor_(std::move(spikeProcessor))
    , propagator_(std::move(propagator))
    , configuredAdapter_(std::move(configuredAdapter))
{
    lastEndTime_ = spikeProcessor_->getCurrentTime();
    if (const char* env = std::getenv("SNNFW_USE_EMNIST_ADAPTER")) {
        useEmnistAdapter_ = (std::string(env) != "0");
    }

    if (configuredAdapter_) {
        if (!configuredAdapter_->isInitialized() && !configuredAdapter_->initialize()) {
            throw std::runtime_error("Failed to initialize configured sensory adapter");
        }
    }

    if (useEmnistAdapter_) {
        adapters::BaseAdapter::Config adapterConfig;
        adapterConfig.name = "emnist_input";
        adapterConfig.type = "emnist";
        adapterConfig.temporalWindow = config_.inputLatencyMs;
        adapterConfig.doubleParams["pixel_threshold"] = config_.pixelThreshold;
        adapterConfig.doubleParams["input_latency_ms"] = config_.inputLatencyMs;
        adapterConfig.intParams["enable_visual_frontend"] = config_.enableVisualFrontend ? 1 : 0;
        adapterConfig.doubleParams["visual_frontend_gain"] = config_.visualFrontendGain;
        adapterConfig.doubleParams["visual_oncenter_weight"] = config_.visualOnCenterWeight;
        adapterConfig.doubleParams["visual_edge_weight"] = config_.visualEdgeWeight;
        adapterConfig.intParams["enable_visual_feature_channels"] =
            config_.enableVisualFeatureChannels ? 1 : 0;
        adapterConfig.doubleParams["visual_feature_threshold"] = config_.visualFeatureThreshold;
        adapterConfig.doubleParams["visual_on_partition_threshold"] =
            config_.visualOnPartitionThreshold;
        adapterConfig.doubleParams["visual_off_partition_threshold"] =
            config_.visualOffPartitionThreshold;
        adapterConfig.doubleParams["visual_edge_partition_threshold"] =
            config_.visualEdgePartitionThreshold;
        adapterConfig.intParams["visual_orientation_bins"] = 4;
        adapterConfig.intParams["image_rows"] = 28;
        adapterConfig.intParams["image_cols"] = 28;
        adapterConfig.intParams["output_rows"] = std::max(1, config_.inputRows);
        adapterConfig.intParams["output_cols"] = std::max(1, config_.inputCols);

        auto& factory = adapters::AdapterFactory::getInstance();
        if (!factory.hasSensoryAdapter("emnist")) {
            factory.registerSensoryAdapter(
                "emnist",
                [](const adapters::BaseAdapter::Config& cfg) {
                    return std::make_shared<adapters::EMNISTAdapter>(cfg);
                });
        }
        emnistAdapter_ = factory.createSensoryAdapter(adapterConfig);
        if (!emnistAdapter_ || !emnistAdapter_->initialize()) {
            throw std::runtime_error("Failed to initialize EMNIST sensory adapter");
        }
    }
}

void SpikeEncoder::populateMaskInputActivity(const EMNISTLoader::Image& image,
                                             const std::vector<std::shared_ptr<Neuron>>& inputNeurons) {
    const size_t maxIdx = std::min(inputNeurons.size(), image.pixels.size());
    for (size_t idx = 0; idx < maxIdx; ++idx) {
        const double norm = image.pixels[idx] / 255.0;
        if (norm > config_.pixelThreshold) {
            lastInputFired_[idx] = true;
        }
    }
}

std::vector<SpikeEncoder::FixationWindow> SpikeEncoder::buildFixationWindows(
    const EMNISTLoader::Image& image) const {
    std::vector<FixationWindow> windows;
    const int requestedFixations = std::max(1, config_.saccadeNumFixations);
    const bool useSaccades =
        config_.enableSaccades &&
        requestedFixations > 1 &&
        !config_.saccadeRegions.empty();

    if (!useSaccades) {
        FixationWindow window;
        window.image = image;
        window.durationMs = config_.inputLatencyMs;
        windows.push_back(std::move(window));
        return windows;
    }

    struct CandidateRegion {
        const ExperimentConfig::FixationRegion* region = nullptr;
        SanitizedFixationRegion sanitized;
        double energy = 0.0;
        size_t originalIndex = 0;
    };

    const auto saliencyMap = buildFeatureSaliencyMap(
        image, std::max(2, config_.saccadeAttentionGrid));
    std::vector<CandidateRegion> candidateRegions;
    candidateRegions.reserve(config_.saccadeRegions.size());
    for (size_t idx = 0; idx < config_.saccadeRegions.size(); ++idx) {
        const auto& region = config_.saccadeRegions[idx];
        const bool isFull = isFullFixationRegion(region, image.rows, image.cols);
        if (isFull && config_.saccadeDropFullFixation && config_.saccadeRegions.size() > 1) {
            continue;
        }
        CandidateRegion candidate;
        candidate.region = &region;
        candidate.sanitized = sanitizeFixationRegion(region, image.rows, image.cols);
        candidate.energy = computeRegionFeatureSaliency(
            saliencyMap,
            candidate.sanitized,
            image.rows,
            image.cols,
            std::max(1, config_.saccadeTopKRegions));
        candidate.originalIndex = idx;
        candidateRegions.push_back(candidate);
    }
    if (candidateRegions.empty()) {
        for (size_t idx = 0; idx < config_.saccadeRegions.size(); ++idx) {
            CandidateRegion candidate;
            candidate.region = &config_.saccadeRegions[idx];
            candidate.sanitized = sanitizeFixationRegion(
                config_.saccadeRegions[idx], image.rows, image.cols);
            candidate.energy = computeRegionFeatureSaliency(
                saliencyMap,
                candidate.sanitized,
                image.rows,
                image.cols,
                std::max(1, config_.saccadeTopKRegions));
            candidate.originalIndex = idx;
            candidateRegions.push_back(candidate);
        }
    }

    std::sort(candidateRegions.begin(), candidateRegions.end(),
              [](const CandidateRegion& a, const CandidateRegion& b) {
                  if (a.energy != b.energy) {
                      return a.energy > b.energy;
                  }
                  return a.originalIndex < b.originalIndex;
              });

    const int fixationCount = std::max(
        1,
        std::min({requestedFixations,
                  std::max(1, config_.saccadeMaxRegionsPerImage),
                  static_cast<int>(candidateRegions.size())}));
    const double sliceMs = config_.inputLatencyMs / static_cast<double>(fixationCount);
    windows.reserve(static_cast<size_t>(fixationCount));
    for (int fixationIdx = 0; fixationIdx < fixationCount; ++fixationIdx) {
        const auto& candidate = candidateRegions[static_cast<size_t>(fixationIdx)];
        const auto& region = *candidate.region;
        const bool isFull = isFullFixationRegion(region, image.rows, image.cols);
        const double gain = isFull ? config_.saccadeFullRegionGain : 1.0;
        FixationWindow window;
        window.image = maskImageToRegion(image, candidate.sanitized);
        if (config_.saccadeUseTileAttention) {
            applyTopKAttention(
                window.image,
                candidate.sanitized,
                std::max(1, config_.saccadeAttentionGrid),
                std::max(1, config_.saccadeTopKRegions),
                config_.saccadeAttentionFloor,
                config_.saccadeAttentionPower);
        }
        applyImageGain(window.image, gain);
        window.offsetMs = static_cast<double>(fixationIdx) * sliceMs;
        window.durationMs = sliceMs;
        windows.push_back(std::move(window));
    }

    return windows;
}

int SpikeEncoder::injectInputPattern(
    const adapters::SensoryAdapter::SpikePattern& encodedPattern,
    double baseTime,
    double offsetMs,
    double durationMs,
    std::vector<std::shared_ptr<Neuron>>& inputNeurons) {
    if (durationMs <= 0.0) {
        return 0;
    }

    const double patternDuration =
        (encodedPattern.duration > 0.0) ? encodedPattern.duration : durationMs;
    const double scale = durationMs / patternDuration;
    int spikeCount = 0;
    const size_t maxIdx = std::min(inputNeurons.size(), encodedPattern.spikeTimes.size());
    for (size_t idx = 0; idx < maxIdx; ++idx) {
        if (encodedPattern.spikeTimes[idx].empty()) {
            continue;
        }
        for (double relativeT : encodedPattern.spikeTimes[idx]) {
            const double fireT = baseTime + offsetMs + (relativeT * scale);
            inputNeurons[idx]->fireSignature(fireT);
            propagator_->fireNeuron(inputNeurons[idx]->getId(), fireT);
            ++spikeCount;
        }
        lastInputFired_[idx] = true;
    }
    return spikeCount;
}

int SpikeEncoder::injectRawInputWindow(const EMNISTLoader::Image& image,
                                       double baseTime,
                                       double offsetMs,
                                       double durationMs,
                                       std::vector<std::shared_ptr<Neuron>>& inputNeurons) {
    if (durationMs <= 0.0) {
        return 0;
    }

    int spikeCount = 0;
    for (size_t idx = 0; idx < inputNeurons.size() && idx < image.pixels.size(); ++idx) {
        const double norm = image.pixels[idx] / 255.0;
        if (norm <= config_.pixelThreshold) {
            continue;
        }
        const double fireT = baseTime + offsetMs + (1.0 - norm) * durationMs;
        inputNeurons[idx]->fireSignature(fireT);
        propagator_->fireNeuron(inputNeurons[idx]->getId(), fireT);
        lastInputFired_[idx] = true;
        ++spikeCount;
    }
    return spikeCount;
}

int SpikeEncoder::injectRetinaIntoL4(
    const adapters::SensoryAdapter::SpikePattern& encodedPattern,
    double baseTime,
    double offsetMs,
    double durationMs,
    std::vector<declarative::ConstructedNetwork::ColumnGroup>& columns) {
    if (!configuredAdapter_ || configuredAdapter_->getType() != "retina") {
        throw std::runtime_error("bind_to=l4 currently requires a retina sensory adapter");
    }
    if (durationMs <= 0.0) {
        return 0;
    }

    const int gridSize = configuredAdapter_->getConfig().getIntParam("grid_size", config_.layer4Size);
    const int orientationChannels =
        configuredAdapter_->getConfig().getIntParam("num_orientations", 1);
    const auto uniqueFrequencies = collectUniqueFrequencies(columns);
    const size_t frequencyBands = std::max<size_t>(1, uniqueFrequencies.size());
    if (gridSize != config_.layer4Size) {
        throw std::runtime_error(
            "RetinaAdapter grid_size must match the SONATA L4 grid size for bind_to=l4");
    }

    const size_t expectedFeatures =
        static_cast<size_t>(gridSize) * static_cast<size_t>(gridSize) *
        static_cast<size_t>(orientationChannels) * frequencyBands;
    if (encodedPattern.spikeTimes.size() != expectedFeatures) {
        throw std::runtime_error(
            "RetinaAdapter feature count does not match expected grid_size^2*num_orientations*num_frequencies");
    }

    const auto uniqueOrientations = collectUniqueOrientations(columns);
    if (uniqueOrientations.size() != static_cast<size_t>(orientationChannels)) {
        throw std::runtime_error(
            "RetinaAdapter num_orientations must match the number of unique column orientations");
    }

    int spikeCount = 0;
    const double patternDuration =
        (encodedPattern.duration > 0.0) ? encodedPattern.duration : durationMs;
    const double scale = durationMs / patternDuration;
    const size_t neuronsPerOrientation = static_cast<size_t>(gridSize) * static_cast<size_t>(gridSize);
    for (auto& col : columns) {
        auto l4It = col.layerNeurons.find("L4");
        if (l4It == col.layerNeurons.end()) {
            continue;
        }
        auto& l4Neurons = l4It->second;
        const size_t orientationIdx = findNearestOrientationIndex(col.orientation, uniqueOrientations);
        const size_t frequencyIdx = findNearestFrequencyIndex(col.spatialFrequency, uniqueFrequencies);
        const size_t maxNeurons = std::min(l4Neurons.size(), neuronsPerOrientation);
        for (size_t localIdx = 0; localIdx < maxNeurons; ++localIdx) {
            const size_t adapterIdx =
                (localIdx * static_cast<size_t>(orientationChannels) * frequencyBands) +
                (orientationIdx * frequencyBands) + frequencyIdx;
            const auto& spikeTimes = encodedPattern.spikeTimes[adapterIdx];
            for (double relativeT : spikeTimes) {
                l4Neurons[localIdx]->insertSpike(baseTime + offsetMs + (relativeT * scale));
                ++spikeCount;
            }
        }
    }

    return spikeCount;
}

int SpikeEncoder::injectRetinaIntoInput(
    const EMNISTLoader::Image& image,
    const adapters::SensoryAdapter::SpikePattern& encodedPattern,
    double baseTime,
    double offsetMs,
    double durationMs,
    std::vector<std::shared_ptr<Neuron>>& inputNeurons) {
    if (!configuredAdapter_ || configuredAdapter_->getType() != "retina") {
        throw std::runtime_error("bind_to=input retina projection requires a retina sensory adapter");
    }
    if (durationMs <= 0.0) {
        return 0;
    }

    const int gridSize = configuredAdapter_->getConfig().getIntParam("grid_size", config_.layer4Size);
    const int orientationChannels =
        configuredAdapter_->getConfig().getIntParam("num_orientations", 1);
    const size_t regions = static_cast<size_t>(gridSize) * static_cast<size_t>(gridSize);
    const size_t denom = regions * static_cast<size_t>(orientationChannels);
    if (denom == 0 || encodedPattern.spikeTimes.size() % denom != 0) {
        throw std::runtime_error("RetinaAdapter feature count is incompatible with input projection");
    }
    const size_t frequencyBands = std::max<size_t>(1, encodedPattern.spikeTimes.size() / denom);
    const int regionRows = std::max(1, image.rows / gridSize);
    const int regionCols = std::max(1, image.cols / gridSize);
    const double secondWeight =
        configuredAdapter_->getConfig().getDoubleParam("input_second_weight", 0.25);
    const double thirdWeight =
        configuredAdapter_->getConfig().getDoubleParam("input_third_weight", 0.0);
    const double blend =
        std::clamp(configuredAdapter_->getConfig().getDoubleParam("input_blend", 1.0), 0.0, 1.0);
    const double regionFloor =
        std::clamp(configuredAdapter_->getConfig().getDoubleParam("input_region_floor", 0.0), 0.0, 1.0);
    std::vector<double> regionDrive(regions, 0.0);

    for (size_t regionIdx = 0; regionIdx < regions; ++regionIdx) {
        double best = 0.0;
        double second = 0.0;
        double third = 0.0;
        const size_t baseIdx = regionIdx * static_cast<size_t>(orientationChannels) * frequencyBands;
        for (size_t channel = 0; channel < static_cast<size_t>(orientationChannels) * frequencyBands; ++channel) {
            const double strength =
                retinaChannelStrength(encodedPattern.spikeTimes[baseIdx + channel], encodedPattern.duration);
            if (strength > best) {
                third = second;
                second = best;
                best = strength;
            } else if (strength > second) {
                third = second;
                second = strength;
            } else if (strength > third) {
                third = strength;
            }
        }
        const double pooled = std::clamp(
            best + (secondWeight * second) + (thirdWeight * third), 0.0, 1.0);
        const double modulator = ((1.0 - blend) * 1.0) +
                                 (blend * std::max(regionFloor, pooled));
        regionDrive[regionIdx] = std::clamp(modulator, 0.0, 1.0);
    }

    int spikeCount = 0;
    for (int row = 0; row < image.rows; ++row) {
        const int regionRow = std::min(gridSize - 1, row / regionRows);
        for (int col = 0; col < image.cols; ++col) {
            const int regionCol = std::min(gridSize - 1, col / regionCols);
            const size_t regionIdx =
                static_cast<size_t>(regionRow) * static_cast<size_t>(gridSize) +
                static_cast<size_t>(regionCol);
            const size_t inputIdx = static_cast<size_t>(row * image.cols + col);
            if (inputIdx >= inputNeurons.size() || inputIdx >= image.pixels.size()) {
                continue;
            }

            const double norm = image.pixels[inputIdx] / 255.0;
            const double projected = std::clamp(norm * regionDrive[regionIdx], 0.0, 1.0);
            if (projected <= config_.pixelThreshold) {
                continue;
            }

            const double fireT = baseTime + offsetMs + (1.0 - projected) * durationMs;
            inputNeurons[inputIdx]->fireSignature(fireT);
            propagator_->fireNeuron(inputNeurons[inputIdx]->getId(), fireT);
            lastInputFired_[inputIdx] = true;
            ++spikeCount;
        }
    }

    return spikeCount;
}

double SpikeEncoder::encodeAndInject(
    const EMNISTLoader::Image& image,
    std::vector<std::shared_ptr<Neuron>>& inputNeurons,
    std::vector<declarative::ConstructedNetwork::ColumnGroup>& columns,
    std::vector<std::vector<std::shared_ptr<Neuron>>>& outputPopulations)
{
    // Wait for schedule horizon
    double nowTime = spikeProcessor_->getCurrentTime();
    const double scheduleHorizon = spikeProcessor_->getMaxScheduleAheadMs() - 2.0;
    while (lastEndTime_ + config_.interImageGapMs > nowTime + scheduleHorizon) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        nowTime = spikeProcessor_->getCurrentTime();
    }
    double baseTime = std::max(nowTime, lastEndTime_) + config_.interImageGapMs;
    lastEndTime_ = baseTime + config_.neuronWindow;
    lastBaseTime_ = baseTime;

    // Memory cleanup for all neurons
    for (auto& col : columns) {
        for (auto& [layerName, neurons] : col.layerNeurons) {
            for (auto& n : neurons) {
                n->periodicMemoryCleanup(baseTime);
            }
        }
    }
    for (auto& pop : outputPopulations) {
        for (auto& n : pop) {
            n->periodicMemoryCleanup(baseTime);
        }
    }
    for (auto& n : inputNeurons) {
        n->periodicMemoryCleanup(baseTime);
    }
    // Ensure output spikes from prior images do not leak
    for (auto& pop : outputPopulations) {
        for (auto& n : pop) {
            n->removeSpikesBefore(baseTime);
        }
    }

    lastInputFired_.assign(inputNeurons.size(), false);
    int spikeCount = 0;
    const auto fixationWindows = buildFixationWindows(image);
    const std::string configuredBindTo = configuredAdapter_
        ? configuredAdapter_->getConfig().getStringParam("bind_to", "input")
        : "";
    bool handledConfiguredAdapter = false;

    if (configuredAdapter_ && configuredBindTo == "l4") {
        for (const auto& fixation : fixationWindows) {
            populateMaskInputActivity(fixation.image, inputNeurons);
            adapters::SensoryAdapter::DataSample sample;
            sample.rawData = fixation.image.pixels;
            sample.timestamp = baseTime + fixation.offsetMs;
            auto encodedPattern = configuredAdapter_->processData(sample);
            spikeCount += injectRetinaIntoL4(
                encodedPattern, baseTime, fixation.offsetMs, fixation.durationMs, columns);
        }
        handledConfiguredAdapter = true;
    } else if (configuredAdapter_ &&
               configuredAdapter_->getType() == "retina" &&
               configuredBindTo == "input") {
        for (const auto& fixation : fixationWindows) {
            adapters::SensoryAdapter::DataSample sample;
            sample.rawData = fixation.image.pixels;
            sample.timestamp = baseTime + fixation.offsetMs;
            auto encodedPattern = configuredAdapter_->processData(sample);
            spikeCount += injectRetinaIntoInput(
                fixation.image, encodedPattern, baseTime, fixation.offsetMs,
                fixation.durationMs, inputNeurons);
        }
        handledConfiguredAdapter = true;
    } else if (emnistAdapter_) {
        const bool parityCheckEnabled =
            (std::getenv("SNNFW_EMNIST_PARITY_CHECK") != nullptr) &&
            fixationWindows.size() == 1 &&
            !config_.enableSaccades;
        for (const auto& fixation : fixationWindows) {
            adapters::SensoryAdapter::DataSample sample;
            sample.rawData = fixation.image.pixels;
            sample.timestamp = baseTime + fixation.offsetMs;
            auto encodedPattern = emnistAdapter_->processData(sample);
            if (parityCheckEnabled) {
                const size_t maxIdx = std::min(inputNeurons.size(), encodedPattern.spikeTimes.size());
                for (size_t idx = 0; idx < maxIdx; ++idx) {
                    const double norm = image.pixels[idx] / 255.0;
                    const bool legacyFires = norm > config_.pixelThreshold;
                    const bool adapterFires = !encodedPattern.spikeTimes[idx].empty();
                    if (legacyFires != adapterFires) {
                        std::cout << "[PARITY] Fire mismatch idx=" << idx
                                  << " pixel=" << static_cast<int>(image.pixels[idx])
                                  << " norm=" << norm
                                  << " threshold=" << config_.pixelThreshold
                                  << " legacy=" << legacyFires
                                  << " adapter=" << adapterFires << std::endl;
                    } else if (legacyFires) {
                        const double legacyT = (1.0 - norm) * config_.inputLatencyMs;
                        const double adapterT = encodedPattern.spikeTimes[idx].front();
                        if (std::abs(legacyT - adapterT) > 1e-12) {
                            std::cout << "[PARITY] Time mismatch idx=" << idx
                                      << " legacyT=" << legacyT
                                      << " adapterT=" << adapterT << std::endl;
                        }
                    }
                }
            }
            spikeCount += injectInputPattern(
                encodedPattern, baseTime, fixation.offsetMs, fixation.durationMs, inputNeurons);
        }
    } else if (!configuredAdapter_) {
        for (const auto& fixation : fixationWindows) {
            spikeCount += injectRawInputWindow(
                fixation.image, baseTime, fixation.offsetMs, fixation.durationMs, inputNeurons);
        }
    }

    if (configuredAdapter_ && !handledConfiguredAdapter) {
        for (const auto& fixation : fixationWindows) {
            adapters::SensoryAdapter::DataSample sample;
            sample.rawData = fixation.image.pixels;
            sample.timestamp = baseTime + fixation.offsetMs;
            auto encodedPattern = configuredAdapter_->processData(sample);
            spikeCount += injectInputPattern(
                encodedPattern, baseTime, fixation.offsetMs, fixation.durationMs, inputNeurons);
        }
    }

    static int g_encodeCount = 0;
    static int g_windowCount = 0;
    static int g_windowSpikeSum = 0;
    static int g_windowMin = std::numeric_limits<int>::max();
    static int g_windowMax = 0;
    ++g_encodeCount;
    ++g_windowCount;
    g_windowSpikeSum += spikeCount;
    g_windowMin = std::min(g_windowMin, spikeCount);
    g_windowMax = std::max(g_windowMax, spikeCount);
    if (g_encodeCount % 100 == 0) {
        const double avgSpikes =
            static_cast<double>(g_windowSpikeSum) / static_cast<double>(std::max(1, g_windowCount));
        std::cout << "[DEBUG] Encoded " << spikeCount << " input spikes for image " << g_encodeCount
                  << " (avg=" << avgSpikes << ", min=" << g_windowMin
                  << ", max=" << g_windowMax << " over last " << g_windowCount << ")"
                  << std::endl;
        g_windowCount = 0;
        g_windowSpikeSum = 0;
        g_windowMin = std::numeric_limits<int>::max();
        g_windowMax = 0;
    }

    return baseTime;
}

void SpikeEncoder::waitForSimTime(double targetTimeMs, double timeoutMs) {
    auto start = std::chrono::steady_clock::now();
    double startSimTime = spikeProcessor_->getCurrentTime();
    int iterations = 0;
    while (spikeProcessor_->getCurrentTime() < targetTimeMs) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsed > timeoutMs) {
            std::cout << "[WARN] waitForSimTime timeout after " << elapsed << "ms (target="
                      << targetTimeMs << ", current=" << spikeProcessor_->getCurrentTime() << ")" << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        iterations++;
    }

    static int g_waitCount = 0;
    static double g_totalWaitMs = 0;
    static int g_totalIterations = 0;
    auto wallTime = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    g_totalWaitMs += wallTime;
    g_totalIterations += iterations;
    if (++g_waitCount % 200 == 0) {
        double simTimeDelta = spikeProcessor_->getCurrentTime() - startSimTime;
        std::cout << "[DEBUG] waitForSimTime: avg " << (g_totalWaitMs / 200.0) << "ms wall time, "
                  << (g_totalIterations / 200.0) << " iterations, last simDelta=" << simTimeDelta << "ms" << std::endl;
        g_totalWaitMs = 0;
        g_totalIterations = 0;
    }
}

} // namespace experiment
} // namespace snnfw

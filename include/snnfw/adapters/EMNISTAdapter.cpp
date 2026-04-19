#include "snnfw/adapters/EMNISTAdapter.h"
#include <algorithm>
#include <cmath>

namespace snnfw {
namespace adapters {

namespace {

constexpr double kPi = 3.14159265358979323846;

enum class VisualChannel {
    None,
    OnCenter,
    OffCenter,
    Edge
};

struct VisualFeatureMaps {
    std::vector<double> raw;
    std::vector<double> onCenter;
    std::vector<double> offCenter;
    std::vector<double> edge;
    std::vector<int> orientationBins;
};

double getNormalizedPixel(const std::vector<uint8_t>& pixels, int rows, int cols, int row, int col) {
    if (row < 0 || row >= rows || col < 0 || col >= cols) {
        return 0.0;
    }
    return static_cast<double>(pixels[static_cast<size_t>(row * cols + col)]) / 255.0;
}

VisualFeatureMaps computeVisualFeatureMaps(
    const std::vector<uint8_t>& pixels,
    int rows,
    int cols,
    int orientationBins) {
    VisualFeatureMaps maps;
    maps.raw.assign(pixels.size(), 0.0);
    maps.onCenter.assign(pixels.size(), 0.0);
    maps.offCenter.assign(pixels.size(), 0.0);
    maps.edge.assign(pixels.size(), 0.0);
    maps.orientationBins.assign(pixels.size(), 0);

    if (pixels.empty()) {
        return maps;
    }

    std::vector<double> edgeEnergy(pixels.size(), 0.0);
    double maxEdge = 0.0;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const size_t idx = static_cast<size_t>(row * cols + col);
            const double center = getNormalizedPixel(pixels, rows, cols, row, col);
            maps.raw[idx] = center;

            double surround = 0.0;
            int surroundCount = 0;
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    if (dr == 0 && dc == 0) {
                        continue;
                    }
                    surround += getNormalizedPixel(pixels, rows, cols, row + dr, col + dc);
                    ++surroundCount;
                }
            }
            surround /= static_cast<double>(std::max(1, surroundCount));
            maps.onCenter[idx] = std::max(0.0, center - surround);
            maps.offCenter[idx] = std::max(0.0, surround - center);

            const double gx =
                (-1.0 * getNormalizedPixel(pixels, rows, cols, row - 1, col - 1)) +
                ( 1.0 * getNormalizedPixel(pixels, rows, cols, row - 1, col + 1)) +
                (-2.0 * getNormalizedPixel(pixels, rows, cols, row,     col - 1)) +
                ( 2.0 * getNormalizedPixel(pixels, rows, cols, row,     col + 1)) +
                (-1.0 * getNormalizedPixel(pixels, rows, cols, row + 1, col - 1)) +
                ( 1.0 * getNormalizedPixel(pixels, rows, cols, row + 1, col + 1));
            const double gy =
                (-1.0 * getNormalizedPixel(pixels, rows, cols, row - 1, col - 1)) +
                (-2.0 * getNormalizedPixel(pixels, rows, cols, row - 1, col    )) +
                (-1.0 * getNormalizedPixel(pixels, rows, cols, row - 1, col + 1)) +
                ( 1.0 * getNormalizedPixel(pixels, rows, cols, row + 1, col - 1)) +
                ( 2.0 * getNormalizedPixel(pixels, rows, cols, row + 1, col    )) +
                ( 1.0 * getNormalizedPixel(pixels, rows, cols, row + 1, col + 1));
            edgeEnergy[idx] = std::sqrt((gx * gx) + (gy * gy));
            maxEdge = std::max(maxEdge, edgeEnergy[idx]);

            double angle = std::atan2(gy, gx) * 180.0 / kPi;
            if (angle < 0.0) {
                angle += 180.0;
            }
            angle = std::fmod(angle + 90.0, 180.0);
            const double normalizedAngle = std::clamp(angle / 180.0, 0.0, 0.999999);
            maps.orientationBins[idx] = std::clamp(
                static_cast<int>(std::floor(normalizedAngle * std::max(1, orientationBins))),
                0,
                std::max(0, orientationBins - 1));
        }
    }

    const double safeMaxEdge = std::max(1e-6, maxEdge);
    for (size_t idx = 0; idx < pixels.size(); ++idx) {
        maps.edge[idx] = std::clamp(edgeEnergy[idx] / safeMaxEdge, 0.0, 1.0);
    }

    return maps;
}

}

EMNISTAdapter::EMNISTAdapter(const Config& config)
    : SensoryAdapter(config) {
    pixelThreshold_ = getDoubleParam("pixel_threshold", 0.4);
    inputLatencyMs_ = getDoubleParam("input_latency_ms", 15.0);
    imageRows_ = std::max(1, getIntParam("image_rows", 28));
    imageCols_ = std::max(1, getIntParam("image_cols", 28));
    outputRows_ = std::max(1, getIntParam("output_rows", imageRows_));
    outputCols_ = std::max(1, getIntParam("output_cols", imageCols_));
    enableVisualFrontend_ = (getIntParam("enable_visual_frontend", 0) != 0);
    visualFrontendGain_ = std::max(0.0, getDoubleParam("visual_frontend_gain", 0.45));
    visualOnCenterWeight_ = std::max(0.0, getDoubleParam("visual_oncenter_weight", 0.60));
    visualEdgeWeight_ = std::max(0.0, getDoubleParam("visual_edge_weight", 0.40));
    const double weightSum = visualOnCenterWeight_ + visualEdgeWeight_;
    if (weightSum > 0.0) {
        visualOnCenterWeight_ /= weightSum;
        visualEdgeWeight_ /= weightSum;
    }
    enableVisualFeatureChannels_ = (getIntParam("enable_visual_feature_channels", 0) != 0);
    visualFeatureThreshold_ = std::clamp(getDoubleParam("visual_feature_threshold", 0.58), 0.0, 1.0);
    visualOnPartitionThreshold_ = std::clamp(
        getDoubleParam("visual_on_partition_threshold", 0.18), 0.0, 1.0);
    visualOffPartitionThreshold_ = std::clamp(
        getDoubleParam("visual_off_partition_threshold", 0.22), 0.0, 1.0);
    visualEdgePartitionThreshold_ = std::clamp(
        getDoubleParam("visual_edge_partition_threshold", 0.55), 0.0, 1.0);
    visualOrientationBins_ = std::max(1, getIntParam("visual_orientation_bins", 4));

    if (outputRows_ == imageRows_ && outputCols_ >= imageCols_ && (outputCols_ % imageCols_) == 0) {
        outputPartitions_ = std::max(1, outputCols_ / imageCols_);
        partitionsHorizontal_ = true;
    } else if (outputCols_ == imageCols_ && outputRows_ >= imageRows_ &&
               (outputRows_ % imageRows_) == 0) {
        outputPartitions_ = std::max(1, outputRows_ / imageRows_);
        partitionsHorizontal_ = false;
    } else {
        outputRows_ = imageRows_;
        outputCols_ = imageCols_;
        outputPartitions_ = 1;
        partitionsHorizontal_ = true;
    }

    featureDimension_ = static_cast<size_t>(outputRows_) * static_cast<size_t>(outputCols_);
}

std::vector<double> EMNISTAdapter::preprocessActivations(const std::vector<uint8_t>& pixels) const {
    std::vector<double> activations(pixels.size(), 0.0);
    if (pixels.empty()) {
        return activations;
    }

    const auto maps = computeVisualFeatureMaps(pixels, imageRows_, imageCols_, visualOrientationBins_);
    for (size_t idx = 0; idx < pixels.size(); ++idx) {
        const double raw = maps.raw[idx];
        const double edge = maps.edge[idx];
        const double boost =
            (visualOnCenterWeight_ * maps.onCenter[idx]) + (visualEdgeWeight_ * edge);
        activations[idx] = std::clamp(raw * (1.0 + (visualFrontendGain_ * boost)), 0.0, 1.0);
    }

    return activations;
}

double EMNISTAdapter::computeFeatureSpikeTime(double strength, int orientationBin) const {
    if (inputLatencyMs_ <= 0.0) {
        return -1.0;
    }

    const double clampedStrength = std::clamp(strength, 0.0, 1.0);
    const int safeOrientationBins = std::max(1, visualOrientationBins_);
    const int safeBin = std::clamp(orientationBin, 0, safeOrientationBins - 1);

    constexpr double kOnStart = 0.58;
    constexpr double kOnWidth = 0.12;
    constexpr double kOffStart = 0.72;
    constexpr double kOffWidth = 0.08;
    constexpr double kEdgeStart = 0.80;
    constexpr double kEdgeWidth = 0.18;

    // Negative orientationBin is reserved for non-edge channels.
    if (orientationBin == -1) {
        return (kOnStart + ((1.0 - clampedStrength) * kOnWidth)) * inputLatencyMs_;
    }
    if (orientationBin == -2) {
        return (kOffStart + ((1.0 - clampedStrength) * kOffWidth)) * inputLatencyMs_;
    }

    const double bandWidth = kEdgeWidth / static_cast<double>(safeOrientationBins);
    const double withinBand = std::max(1e-6, bandWidth * 0.85);
    const double t =
        kEdgeStart +
        (static_cast<double>(safeBin) * bandWidth) +
        ((1.0 - clampedStrength) * withinBand);
    return std::min(inputLatencyMs_, t * inputLatencyMs_);
}

size_t EMNISTAdapter::mapPixelToOutputIndex(int partition, size_t pixelIndex) const {
    const int safePartition = std::clamp(partition, 0, std::max(0, outputPartitions_ - 1));
    const int row = static_cast<int>(pixelIndex / static_cast<size_t>(imageCols_));
    const int col = static_cast<int>(pixelIndex % static_cast<size_t>(imageCols_));
    if (partitionsHorizontal_) {
        return static_cast<size_t>(row * outputCols_ + (safePartition * imageCols_) + col);
    }
    return static_cast<size_t>(((safePartition * imageRows_) + row) * outputCols_ + col);
}

bool EMNISTAdapter::usePartitionedFeatureInputs() const {
    return enableVisualFeatureChannels_ && outputPartitions_ >= 3;
}

bool EMNISTAdapter::isLocalMaximum(const std::vector<double>& map, int row, int col) const {
    const size_t idx = static_cast<size_t>(row * imageCols_ + col);
    if (idx >= map.size()) {
        return false;
    }

    const double center = map[idx];
    if (center <= 0.0) {
        return false;
    }

    for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            if (dr == 0 && dc == 0) {
                continue;
            }
            const int rr = row + dr;
            const int cc = col + dc;
            if (rr < 0 || rr >= imageRows_ || cc < 0 || cc >= imageCols_) {
                continue;
            }
            const size_t neighborIdx = static_cast<size_t>(rr * imageCols_ + cc);
            if (neighborIdx < map.size() && map[neighborIdx] > center) {
                return false;
            }
        }
    }
    return true;
}

bool EMNISTAdapter::isHorizontalEdgeBin(int orientationBin) const {
    if (visualOrientationBins_ <= 1) {
        return true;
    }
    return orientationBin == 0 || orientationBin == (visualOrientationBins_ - 1);
}

bool EMNISTAdapter::initialize() {
    if (!SensoryAdapter::initialize()) {
        return false;
    }

    neurons_.clear();
    neurons_.reserve(featureDimension_);
    for (size_t i = 0; i < featureDimension_; ++i) {
        neurons_.push_back(std::make_shared<Neuron>(200.0, 0.7, 1, i));
    }
    lastActivationPattern_.assign(featureDimension_, 0.0);
    return true;
}

SensoryAdapter::SpikePattern EMNISTAdapter::processData(const DataSample& data) {
    SpikePattern pattern;
    pattern.timestamp = data.timestamp;
    pattern.duration = inputLatencyMs_;
    pattern.spikeTimes.resize(featureDimension_);

    if (lastActivationPattern_.size() != featureDimension_) {
        lastActivationPattern_.assign(featureDimension_, 0.0);
    }
    const bool needFeatureMaps = enableVisualFrontend_ || enableVisualFeatureChannels_;
    const auto maps = needFeatureMaps
        ? computeVisualFeatureMaps(data.rawData, imageRows_, imageCols_, visualOrientationBins_)
        : VisualFeatureMaps{};
    const auto activations = enableVisualFrontend_
        ? preprocessActivations(data.rawData)
        : maps.raw;

    if (usePartitionedFeatureInputs()) {
        const int onPartition = (outputPartitions_ >= 4) ? 1 : 0;
        const int offPartition = (outputPartitions_ >= 4) ? 2 : 1;
        const int edgePartition = (outputPartitions_ >= 4) ? 3 : 2;
        const bool keepRawPartition = outputPartitions_ >= 4;

        for (size_t i = 0; i < data.rawData.size(); ++i) {
            const int row = static_cast<int>(i / static_cast<size_t>(imageCols_));
            const int col = static_cast<int>(i % static_cast<size_t>(imageCols_));
            const double raw = maps.raw[i];
            const double onStrength = maps.onCenter[i];
            const double offStrength = maps.offCenter[i];
            const double edgeStrength = maps.edge[i];
            const int orientationBin = maps.orientationBins[i];
            const bool horizontalEdge = isHorizontalEdgeBin(orientationBin);
            const bool lowerBand = row >= ((imageRows_ * 2) / 3);
            const bool offPeak = isLocalMaximum(maps.offCenter, row, col);
            const bool edgePeak = isLocalMaximum(maps.edge, row, col);
            double offThreshold = visualOffPartitionThreshold_;
            if (!horizontalEdge) {
                offThreshold += 0.06;
            }
            if (!lowerBand) {
                offThreshold += 0.04;
            }
            offThreshold = std::clamp(offThreshold, 0.0, 0.95);

            double edgeThreshold = visualEdgePartitionThreshold_;
            if (!horizontalEdge) {
                edgeThreshold += 0.08;
            } else if (lowerBand) {
                edgeThreshold -= 0.07;
            } else {
                edgeThreshold += 0.02;
            }
            edgeThreshold = std::clamp(edgeThreshold, 0.0, 0.98);
            const bool allowOffSpike =
                offPeak &&
                offStrength > offThreshold &&
                (horizontalEdge || lowerBand || offStrength > (offThreshold + 0.10));

            if (keepRawPartition) {
                const size_t rawIdx = mapPixelToOutputIndex(0, i);
                lastActivationPattern_[rawIdx] = raw;
                if (raw > pixelThreshold_) {
                    pattern.spikeTimes[rawIdx].push_back((1.0 - raw) * inputLatencyMs_);
                }
            }

            const size_t onIdx = mapPixelToOutputIndex(onPartition, i);
            const size_t offIdx = mapPixelToOutputIndex(offPartition, i);
            const size_t edgeIdx = mapPixelToOutputIndex(edgePartition, i);
            lastActivationPattern_[onIdx] = onStrength;
            lastActivationPattern_[offIdx] = offStrength;
            lastActivationPattern_[edgeIdx] = edgeStrength;

            if (onStrength > visualOnPartitionThreshold_) {
                pattern.spikeTimes[onIdx].push_back(computeFeatureSpikeTime(onStrength, -1));
            }
            if (allowOffSpike) {
                pattern.spikeTimes[offIdx].push_back(computeFeatureSpikeTime(offStrength, -2));
            }
            if (edgePeak && edgeStrength > edgeThreshold) {
                pattern.spikeTimes[edgeIdx].push_back(
                    computeFeatureSpikeTime(edgeStrength, orientationBin));
            }
        }
        return pattern;
    }

    for (size_t i = 0; i < data.rawData.size(); ++i) {
        const double raw = needFeatureMaps
            ? maps.raw[i]
            : static_cast<double>(data.rawData[i]) / 255.0;
        const double norm = enableVisualFrontend_ ? activations[i] : raw;
        auto& spikes = pattern.spikeTimes[i];
        lastActivationPattern_[i] = norm;
        if (norm > pixelThreshold_) {
            spikes.push_back((1.0 - norm) * inputLatencyMs_);
        }

        if (enableVisualFeatureChannels_) {
            const double onScore = maps.onCenter[i];
            const double offScore = maps.offCenter[i] * 0.90;
            const double edgeScore = maps.edge[i] * 0.95;

            VisualChannel bestChannel = VisualChannel::None;
            double bestScore = 0.0;
            int orientationBin = 0;
            if (onScore > bestScore) {
                bestScore = onScore;
                bestChannel = VisualChannel::OnCenter;
                orientationBin = -1;
            }
            if (offScore > bestScore) {
                bestScore = offScore;
                bestChannel = VisualChannel::OffCenter;
                orientationBin = -2;
            }
            if (edgeScore > bestScore) {
                bestScore = edgeScore;
                bestChannel = VisualChannel::Edge;
                orientationBin = maps.orientationBins[i];
            }

            const bool allowFeatureSpike =
                bestScore > visualFeatureThreshold_ &&
                (raw > (pixelThreshold_ * 0.45) || bestChannel != VisualChannel::Edge);
            if (allowFeatureSpike && bestChannel != VisualChannel::None) {
                const double featureTime = computeFeatureSpikeTime(bestScore, orientationBin);
                if (featureTime >= 0.0) {
                    spikes.push_back(featureTime);
                    lastActivationPattern_[i] = std::max(lastActivationPattern_[i], bestScore);
                }
            }
        }

        if (spikes.size() > 1) {
            std::sort(spikes.begin(), spikes.end());
            spikes.erase(std::unique(spikes.begin(), spikes.end(),
                                     [](double a, double b) { return std::abs(a - b) < 1e-6; }),
                         spikes.end());
        }
    }
    return pattern;
}

SensoryAdapter::FeatureVector EMNISTAdapter::extractFeatures(const DataSample& data) {
    FeatureVector result;
    result.timestamp = data.timestamp;
    result.features.reserve(featureDimension_);
    const bool needFeatureMaps = enableVisualFrontend_ || enableVisualFeatureChannels_;
    const auto maps = needFeatureMaps
        ? computeVisualFeatureMaps(data.rawData, imageRows_, imageCols_, visualOrientationBins_)
        : VisualFeatureMaps{};
    const auto activations = enableVisualFrontend_
        ? preprocessActivations(data.rawData)
        : maps.raw;
    if (usePartitionedFeatureInputs()) {
        result.features.assign(featureDimension_, 0.0);
        const int onPartition = (outputPartitions_ >= 4) ? 1 : 0;
        const int offPartition = (outputPartitions_ >= 4) ? 2 : 1;
        const int edgePartition = (outputPartitions_ >= 4) ? 3 : 2;
        const bool keepRawPartition = outputPartitions_ >= 4;
        for (size_t i = 0; i < data.rawData.size(); ++i) {
            const double raw = maps.raw[i];
            const double onStrength = maps.onCenter[i];
            const double offStrength = maps.offCenter[i];
            const double edgeStrength = maps.edge[i];
            if (keepRawPartition) {
                result.features[mapPixelToOutputIndex(0, i)] = raw;
            }
            result.features[mapPixelToOutputIndex(onPartition, i)] = onStrength;
            result.features[mapPixelToOutputIndex(offPartition, i)] = offStrength;
            result.features[mapPixelToOutputIndex(edgePartition, i)] = edgeStrength;
        }
    } else {
    for (size_t i = 0; i < data.rawData.size(); ++i) {
        double value = enableVisualFrontend_
            ? activations[i]
            : (needFeatureMaps ? maps.raw[i] : static_cast<double>(data.rawData[i]) / 255.0);
        if (enableVisualFeatureChannels_) {
            value = std::max(value, std::max(maps.onCenter[i], std::max(maps.offCenter[i] * 0.90,
                                                                         maps.edge[i] * 0.95)));
        }
        result.features.push_back(value);
    }
    }

    if (!result.features.empty()) {
        featureDimension_ = result.features.size();
        lastActivationPattern_ = result.features;
        if (neurons_.size() != featureDimension_) {
            neurons_.clear();
            neurons_.reserve(featureDimension_);
            for (size_t i = 0; i < featureDimension_; ++i) {
                neurons_.push_back(std::make_shared<Neuron>(200.0, 0.7, 1, i));
            }
        }
    }

    return result;
}

SensoryAdapter::SpikePattern EMNISTAdapter::encodeFeatures(const FeatureVector& features) {
    SpikePattern pattern;
    pattern.timestamp = features.timestamp;
    pattern.duration = inputLatencyMs_;
    pattern.spikeTimes.resize(features.features.size());

    for (size_t i = 0; i < features.features.size(); ++i) {
        const double v = features.features[i];
        if (v > pixelThreshold_) {
            const double t = std::max(0.0, (1.0 - v) * inputLatencyMs_);
            pattern.spikeTimes[i].push_back(t);
        }
    }

    return pattern;
}

void EMNISTAdapter::clearNeuronStates() {
    for (auto& neuron : neurons_) {
        neuron->clearSpikes();
    }
    std::fill(lastActivationPattern_.begin(), lastActivationPattern_.end(), 0.0);
}

} // namespace adapters
} // namespace snnfw

#include "snnfw/Neuron.h"
#include "snnfw/Logger.h"
#include "snnfw/learning/PatternUpdateStrategy.h"
#include "snnfw/NetworkPropagator.h"
#include "snnfw/SpikeAcknowledgment.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <random>

using json = nlohmann::json;

namespace snnfw {

namespace {
size_t deterministicReplacementIndex(const BinaryPattern& pattern,
                                     size_t patternCount,
                                     uint64_t neuronId) {
    if (patternCount == 0) return 0;
    // FNV-1a over pattern bins + neuron id yields deterministic replacement.
    uint64_t hash = 1469598103934665603ULL;
    for (uint8_t bin : pattern.getData()) {
        hash ^= static_cast<uint64_t>(bin);
        hash *= 1099511628211ULL;
    }
    hash ^= neuronId + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    return static_cast<size_t>(hash % static_cast<uint64_t>(patternCount));
}

constexpr double kPrototypeReinforcementAlpha = 0.15;
constexpr double kExemplarReinforcementAlpha = 0.05;
constexpr double kPrototypeMatchMargin = 0.10;
constexpr double kExemplarAllocationMargin = 0.05;
constexpr double kPrototypePromotionMargin = 0.03;
constexpr uint16_t kPrototypeInitialSupport = 3;
constexpr uint16_t kExemplarPromotionSupport = 3;
constexpr uint16_t kProtectedPrototypeSupport = 6;

void clampSupportVector(std::vector<uint16_t>& supports,
                        size_t expectedSize,
                        uint16_t defaultValue) {
    if (supports.size() < expectedSize) {
        supports.resize(expectedSize, defaultValue);
    } else if (supports.size() > expectedSize) {
        supports.resize(expectedSize);
    }
}

size_t pickWeakestPatternIndex(const BinaryPattern& pattern,
                               const std::vector<BinaryPattern>& patterns,
                               const std::vector<uint16_t>& supports,
                               uint64_t neuronId) {
    if (patterns.empty()) {
        return 0;
    }

    size_t bestIndex = 0;
    uint16_t bestSupport = std::numeric_limits<uint16_t>::max();
    double bestSimilarity = std::numeric_limits<double>::max();
    for (size_t i = 0; i < patterns.size(); ++i) {
        const uint16_t support = (i < supports.size()) ? supports[i] : 1;
        const double similarity = BinaryPattern::cosineSimilarity(pattern, patterns[i]);
        if (support < bestSupport ||
            (support == bestSupport && similarity < bestSimilarity) ||
            (support == bestSupport && std::abs(similarity - bestSimilarity) < 1e-9 &&
             deterministicReplacementIndex(pattern, patterns.size(), neuronId) == i)) {
            bestSupport = support;
            bestSimilarity = similarity;
            bestIndex = i;
        }
    }
    return bestIndex;
}
} // namespace

Neuron::Neuron(double windowSizeMs, double similarityThreshold, size_t maxReferencePatterns, uint64_t neuronId)
    : NeuralObject(neuronId),
      windowSize(windowSizeMs),
      maxSpikeTime_(-std::numeric_limits<double>::infinity()),  // Initialize to -infinity
      threshold(similarityThreshold),
      maxPatterns(maxReferencePatterns),
      axonId(0),
      similarityMetric_(SimilarityMetric::COSINE),  // Default to cosine similarity
      inhibition_(0.0),
      incomingSpikeCount_(0),
      firingRate_(0.0),
      targetFiringRate_(5.0),  // Default target: 5 Hz
      intrinsicExcitability_(1.0),
      lastFiringTime_(-1000.0),
      firingCount_(0),
      firingWindowStart_(0.0) {
    // Generate unique temporal signature for this neuron
    generateTemporalSignature();
}

void Neuron::insertSpike(double spikeTime) {
    std::lock_guard<std::mutex> lock(spikesMutex_);
    spikes.push_back(spikeTime);

    // OPTIMIZATION: Track max spike time instead of recomputing with std::max_element
    // Spikes may arrive out-of-order under async delivery.
    // Maintain the rolling window relative to the newest spike we have seen.
    if (spikes.size() == 1 || spikeTime > maxSpikeTime_) {
        maxSpikeTime_ = spikeTime;
    }
    removeOldSpikesUnsafe(maxSpikeTime_);

    // NOTE: Removed shouldFire() check here because it causes issues during training
    // When learning patterns, we don't want the neuron to fire based on previously
    // learned patterns. Use getBestSimilarity() or checkShouldFire() explicitly
    // during testing/inference instead.

    // if (shouldFire()) {
    //     SNNFW_INFO("Neuron {} fires a new spike at time: {}", getId(), spikeTime);
    // }
}

void Neuron::learnCurrentPattern() {
    // Copy spikes under lock to avoid holding lock during pattern learning
    std::vector<double> spikesCopy;
    {
        std::lock_guard<std::mutex> lock(spikesMutex_);
        if (spikes.empty()) {
            SNNFW_DEBUG("Neuron {}: Cannot learn pattern - no spikes in window", getId());
            return;
        }
        spikesCopy = spikes;
    }

    // Convert current spike window to BinaryPattern (200 bytes, fixed size)
    BinaryPattern newPattern(spikesCopy, windowSize);

    SNNFW_DEBUG("Neuron {}: Converting {} spike times to BinaryPattern ({} total spikes)",
                getId(), spikesCopy.size(), newPattern.getTotalSpikes());

    // If a pattern update strategy is set, use it directly with BinaryPattern
    if (patternStrategy_) {
        // Create similarity metric lambda using the selected metric
        auto similarityMetric = [this](const BinaryPattern& a, const BinaryPattern& b) {
            return this->computeSimilarity(a, b);
        };

        // Call the BinaryPattern version of updatePatterns (efficient, no conversion!)
        auto existingPatterns = getLearnedPatterns();
        std::vector<BinaryPattern> updatedPatterns = existingPatterns;
        patternStrategy_->updatePatterns(updatedPatterns, newPattern, similarityMetric);
        prototypePatterns_.clear();
        prototypeSupports_.clear();
        exemplarPatterns_.clear();
        exemplarSupports_.clear();
        exemplarPatterns_ = std::move(updatedPatterns);
        exemplarSupports_.assign(exemplarPatterns_.size(), 1);

        SNNFW_DEBUG("Neuron {}: Updated patterns using {} strategy ({} patterns stored)",
                    getId(), patternStrategy_->getName(), getLearnedPatternCount());
        return;
    }

    const double effectiveThreshold = std::clamp(threshold, 0.0, 1.0);
    clampSupportVector(prototypeSupports_, prototypePatterns_.size(), kPrototypeInitialSupport);
    clampSupportVector(exemplarSupports_, exemplarPatterns_.size(), 1);
    int bestPrototypeIndex = -1;
    int bestExemplarIndex = -1;
    const double bestPrototypeSim =
        findBestSimilarity(newPattern, prototypePatterns_, &bestPrototypeIndex);
    const double bestExemplarSim =
        findBestSimilarity(newPattern, exemplarPatterns_, &bestExemplarIndex);

    if (bestPrototypeIndex != -1 &&
        bestPrototypeSim >= std::min(1.0, effectiveThreshold + kPrototypeMatchMargin)) {
        BinaryPattern::blend(
            prototypePatterns_[static_cast<size_t>(bestPrototypeIndex)],
            newPattern,
            kPrototypeReinforcementAlpha);
        prototypeSupports_[static_cast<size_t>(bestPrototypeIndex)] = static_cast<uint16_t>(
            std::min<int>(std::numeric_limits<uint16_t>::max(),
                          prototypeSupports_[static_cast<size_t>(bestPrototypeIndex)] + 1));
        SNNFW_DEBUG("Neuron {}: Reinforced prototype #{} (similarity={:.3f}, total memories={})",
                    getId(), bestPrototypeIndex, bestPrototypeSim, getLearnedPatternCount());
    } else if (bestExemplarIndex != -1 && bestExemplarSim >= effectiveThreshold) {
        const size_t exemplarIndex = static_cast<size_t>(bestExemplarIndex);
        BinaryPattern::blend(
            exemplarPatterns_[exemplarIndex],
            newPattern,
            kExemplarReinforcementAlpha);
        exemplarSupports_[exemplarIndex] = static_cast<uint16_t>(
            std::min<int>(std::numeric_limits<uint16_t>::max(), exemplarSupports_[exemplarIndex] + 1));
        const size_t prototypeCapacity = std::max<size_t>(1, maxPatterns / 3);
        if (exemplarSupports_[exemplarIndex] >= kExemplarPromotionSupport &&
            bestExemplarSim >= std::min(1.0, effectiveThreshold + kPrototypePromotionMargin) &&
            prototypePatterns_.size() < prototypeCapacity) {
            prototypePatterns_.push_back(exemplarPatterns_[exemplarIndex]);
            prototypeSupports_.push_back(
                std::max<uint16_t>(kPrototypeInitialSupport, exemplarSupports_[exemplarIndex]));
            exemplarPatterns_.erase(exemplarPatterns_.begin() + static_cast<std::ptrdiff_t>(exemplarIndex));
            exemplarSupports_.erase(exemplarSupports_.begin() + static_cast<std::ptrdiff_t>(exemplarIndex));
            SNNFW_DEBUG("Neuron {}: Promoted exemplar #{} to prototype (prototypes={}, exemplars={})",
                        getId(), exemplarIndex, prototypePatterns_.size(), exemplarPatterns_.size());
        }
        SNNFW_DEBUG("Neuron {}: Reinforced exemplar #{} (similarity={:.3f}, total memories={})",
                    getId(), bestExemplarIndex, bestExemplarSim, getLearnedPatternCount());
    } else {
        const size_t totalPatterns = getLearnedPatternCount();
        const size_t prototypeCapacity = std::max<size_t>(1, maxPatterns / 3);

        if (bestExemplarIndex != -1 &&
            bestExemplarSim >= std::max(0.0, effectiveThreshold - kExemplarAllocationMargin) &&
            prototypePatterns_.size() < prototypeCapacity &&
            exemplarSupports_[static_cast<size_t>(bestExemplarIndex)] >= (kExemplarPromotionSupport - 1)) {
            const size_t exemplarIndex = static_cast<size_t>(bestExemplarIndex);
            prototypePatterns_.push_back(exemplarPatterns_[exemplarIndex]);
            prototypeSupports_.push_back(
                std::max<uint16_t>(kPrototypeInitialSupport, exemplarSupports_[exemplarIndex]));
            exemplarPatterns_.erase(exemplarPatterns_.begin() + static_cast<std::ptrdiff_t>(exemplarIndex));
            exemplarSupports_.erase(exemplarSupports_.begin() + static_cast<std::ptrdiff_t>(exemplarIndex));
            exemplarPatterns_.push_back(newPattern);
            exemplarSupports_.push_back(1);
            SNNFW_DEBUG("Neuron {}: Consolidated exemplar into prototype and stored new exemplar (similarity={:.3f})",
                        getId(), bestExemplarSim);
        } else if (totalPatterns < maxPatterns) {
            exemplarPatterns_.push_back(newPattern);
            exemplarSupports_.push_back(1);
            SNNFW_DEBUG("Neuron {}: Stored new exemplar ({} total spikes, prototypes={}, exemplars={})",
                        getId(), newPattern.getTotalSpikes(),
                        prototypePatterns_.size(), exemplarPatterns_.size());
        } else if (!exemplarPatterns_.empty()) {
            const size_t replaceIndex =
                pickWeakestPatternIndex(newPattern, exemplarPatterns_, exemplarSupports_, getId());
            exemplarPatterns_[replaceIndex] = newPattern;
            exemplarSupports_[replaceIndex] = 1;
            SNNFW_DEBUG("Neuron {}: Replaced exemplar #{} with new pattern", getId(), replaceIndex);
        } else if (!prototypePatterns_.empty()) {
            size_t replaceIndex =
                pickWeakestPatternIndex(newPattern, prototypePatterns_, prototypeSupports_, getId());
            if (prototypeSupports_[replaceIndex] <= kProtectedPrototypeSupport) {
                prototypePatterns_[replaceIndex] = newPattern;
                prototypeSupports_[replaceIndex] = kPrototypeInitialSupport;
                SNNFW_DEBUG("Neuron {}: Replaced prototype #{} with new pattern", getId(), replaceIndex);
            } else if (totalPatterns < maxPatterns) {
                exemplarPatterns_.push_back(newPattern);
                exemplarSupports_.push_back(1);
                if (exemplarPatterns_.size() > std::max<size_t>(1, maxPatterns / 2)) {
                    const size_t trimIndex =
                        pickWeakestPatternIndex(newPattern, exemplarPatterns_, exemplarSupports_, getId());
                    exemplarPatterns_.erase(exemplarPatterns_.begin() + static_cast<std::ptrdiff_t>(trimIndex));
                    exemplarSupports_.erase(exemplarSupports_.begin() + static_cast<std::ptrdiff_t>(trimIndex));
                }
                SNNFW_DEBUG("Neuron {}: Preserved mature prototype set and recycled exemplar slot", getId());
            } else {
                SNNFW_DEBUG("Neuron {}: Skipped overwrite because all prototype memories are mature", getId());
            }
        }
    }
}

void Neuron::setPatternUpdateStrategy(std::shared_ptr<learning::PatternUpdateStrategy> strategy) {
    patternStrategy_ = strategy;
    SNNFW_INFO("Neuron {}: Set pattern update strategy to {}", getId(), strategy ? strategy->getName() : "default");
}

void Neuron::printSpikes() const {
    std::lock_guard<std::mutex> lock(spikesMutex_);
    std::string spikesStr;
    for (double spikeTime : spikes) {
        spikesStr += std::to_string(spikeTime) + " ";
    }
    SNNFW_INFO("Neuron {}: Current spikes in window: {}", getId(), spikesStr);
}

void Neuron::printReferencePatterns() const {
    const auto& patterns = getLearnedPatterns();
    SNNFW_INFO("Neuron {}: Stored reference patterns ({})", getId(), patterns.size());
    for (size_t i = 0; i < patterns.size(); ++i) {
        SNNFW_INFO("  Pattern #{}: {}", i, patterns[i].toString());
    }
}

void Neuron::removeOldSpikes(double currentTime) {
    std::lock_guard<std::mutex> lock(spikesMutex_);
    removeOldSpikesUnsafe(currentTime);
}

void Neuron::removeOldSpikesUnsafe(double currentTime) {
    // Note: Caller must hold spikesMutex_ lock
    // Do not assume insertion order (async delivery can insert out-of-order).
    size_t oldSize = spikes.size();
    spikes.erase(
        std::remove_if(spikes.begin(), spikes.end(),
                       [this, currentTime](double t) { return (currentTime - t) > windowSize; }),
        spikes.end());

    // If we removed spikes and the max was removed, recompute it
    if (spikes.size() < oldSize && !spikes.empty()) {
        // Only recompute if we actually removed spikes
        maxSpikeTime_ = *std::max_element(spikes.begin(), spikes.end());
    } else if (spikes.empty()) {
        maxSpikeTime_ = -std::numeric_limits<double>::infinity();
    }
}

// Convert spike pattern to temporal histogram (fuzzy representation)
// Uses cumulative distribution function (CDF) instead of histogram
// to preserve temporal ordering information
std::vector<double> Neuron::spikeToHistogram(const std::vector<double>& pattern) const {
    const int bins = 50;  // More bins for better temporal resolution
    std::vector<double> cdf(bins, 0.0);

    if (pattern.empty()) return cdf;

    // Sort spikes for CDF computation
    std::vector<double> sortedSpikes = pattern;
    std::sort(sortedSpikes.begin(), sortedSpikes.end());

    double binSize = windowSize / bins;

    // Build cumulative distribution function
    // CDF[i] = fraction of spikes that occurred before time i*binSize
    size_t spikeIdx = 0;
    for (int bin = 0; bin < bins; ++bin) {
        double binEnd = (bin + 1) * binSize;

        // Count spikes up to this bin
        while (spikeIdx < sortedSpikes.size() && sortedSpikes[spikeIdx] < binEnd) {
            spikeIdx++;
        }

        // Normalize by total spike count
        cdf[bin] = static_cast<double>(spikeIdx) / sortedSpikes.size();
    }

    return cdf;
}

// Compute similarity between two CDFs using hybrid approach:
// 1. Direct CDF similarity (captures overall temporal distribution)
// 2. First derivative similarity (captures spike rate changes)
// 3. L1 distance penalty (penalizes different distributions)
double Neuron::histogramSimilarity(const std::vector<double>& cdf1, const std::vector<double>& cdf2) const {
    if (cdf1.size() < 2 || cdf2.size() < 2) return 0.0;

    // 1. Cosine similarity on CDFs directly
    double dot_cdf = 0.0, norm1_cdf = 0.0, norm2_cdf = 0.0;
    for (size_t i = 0; i < cdf1.size(); ++i) {
        dot_cdf += cdf1[i] * cdf2[i];
        norm1_cdf += cdf1[i] * cdf1[i];
        norm2_cdf += cdf2[i] * cdf2[i];
    }
    double cdf_sim = 0.0;
    if (norm1_cdf > 1e-10 && norm2_cdf > 1e-10) {
        cdf_sim = dot_cdf / (std::sqrt(norm1_cdf) * std::sqrt(norm2_cdf));
    }

    // 2. Compute first derivatives (spike rate) and their similarity
    std::vector<double> deriv1, deriv2;
    for (size_t i = 0; i < cdf1.size() - 1; ++i) {
        deriv1.push_back(cdf1[i+1] - cdf1[i]);
        deriv2.push_back(cdf2[i+1] - cdf2[i]);
    }

    double dot_deriv = 0.0, norm1_deriv = 0.0, norm2_deriv = 0.0;
    for (size_t i = 0; i < deriv1.size(); ++i) {
        dot_deriv += deriv1[i] * deriv2[i];
        norm1_deriv += deriv1[i] * deriv1[i];
        norm2_deriv += deriv2[i] * deriv2[i];
    }
    double deriv_sim = 0.0;
    if (norm1_deriv > 1e-10 && norm2_deriv > 1e-10) {
        deriv_sim = dot_deriv / (std::sqrt(norm1_deriv) * std::sqrt(norm2_deriv));
    }

    // 3. L1 distance (lower is better, so invert it)
    double l1_dist = 0.0;
    for (size_t i = 0; i < cdf1.size(); ++i) {
        l1_dist += std::abs(cdf1[i] - cdf2[i]);
    }
    double l1_sim = 1.0 / (1.0 + l1_dist);  // Convert distance to similarity

    // Weighted combination: emphasize derivative (temporal structure)
    return 0.3 * cdf_sim + 0.5 * deriv_sim + 0.2 * l1_sim;
}

double Neuron::computeSimilarity(const BinaryPattern& a, const BinaryPattern& b) const {
    switch (similarityMetric_) {
        case SimilarityMetric::COSINE:
            return BinaryPattern::cosineSimilarity(a, b);
        case SimilarityMetric::HISTOGRAM:
            return BinaryPattern::histogramIntersection(a, b);
        case SimilarityMetric::EUCLIDEAN:
            return BinaryPattern::euclideanSimilarity(a, b);
        case SimilarityMetric::CORRELATION:
            return BinaryPattern::correlationSimilarity(a, b);
        case SimilarityMetric::WAVEFORM:
            return BinaryPattern::waveformSimilarity(a, b);
        default:
            return BinaryPattern::cosineSimilarity(a, b);
    }
}

double Neuron::findBestSimilarity(const BinaryPattern& currentPattern,
                                  const std::vector<BinaryPattern>& patterns,
                                  int* bestIndex) const {
    double bestSim = -1.0;
    int idx = -1;
    for (size_t i = 0; i < patterns.size(); ++i) {
        if (patterns[i].isEmpty()) continue;
        double similarity = computeSimilarity(currentPattern, patterns[i]);
        if (similarity > bestSim) {
            bestSim = similarity;
            idx = static_cast<int>(i);
        }
    }
    if (bestIndex) {
        *bestIndex = idx;
    }
    return bestSim;
}

const std::vector<BinaryPattern>& Neuron::getLearnedPatterns() const {
    combinedPatternsCache_.clear();
    combinedPatternsCache_.reserve(prototypePatterns_.size() + exemplarPatterns_.size());
    combinedPatternsCache_.insert(
        combinedPatternsCache_.end(), prototypePatterns_.begin(), prototypePatterns_.end());
    combinedPatternsCache_.insert(
        combinedPatternsCache_.end(), exemplarPatterns_.begin(), exemplarPatterns_.end());
    return combinedPatternsCache_;
}

bool Neuron::shouldFire() const {
    // Copy spikes under lock to avoid data races with concurrent delivery.
    std::vector<double> spikesCopy;
    {
        std::lock_guard<std::mutex> lock(spikesMutex_);
        if (spikes.empty()) {
            return false;
        }
        spikesCopy = spikes;
    }

    // Convert current spikes to BinaryPattern for comparison
    BinaryPattern currentPattern(spikesCopy, windowSize);

    const double effectiveThreshold = std::clamp(threshold, 0.0, 1.0);
    return std::max(findBestSimilarity(currentPattern, prototypePatterns_),
                    findBestSimilarity(currentPattern, exemplarPatterns_)) >= effectiveThreshold;
}

double Neuron::getBestSimilarity() const {
    // Copy spikes under lock
    std::vector<double> spikesCopy;
    {
        std::lock_guard<std::mutex> lock(spikesMutex_);
        if (spikes.empty() || getLearnedPatternCount() == 0) {
            return 0.0;
        }
        spikesCopy = spikes;
    }

    // Convert current spikes to BinaryPattern
    BinaryPattern currentPattern(spikesCopy, windowSize);

    double bestSim = std::max(findBestSimilarity(currentPattern, prototypePatterns_),
                              findBestSimilarity(currentPattern, exemplarPatterns_));
    return (bestSim < 0.0) ? 0.0 : bestSim;
}

int Neuron::findMostSimilarPattern(const std::vector<double>& newPattern) const {
    // Convert newPattern to BinaryPattern for comparison
    BinaryPattern newBinaryPattern(newPattern, windowSize);

    int bestIndex = -1;
    double bestSim = -1.0;

    const auto& patterns = getLearnedPatterns();
    for (size_t i = 0; i < patterns.size(); ++i) {
        double sim = computeSimilarity(patterns[i], newBinaryPattern);
        if (sim > bestSim) {
            bestSim = sim;
            bestIndex = static_cast<int>(i);
        }
    }
    return bestIndex;
}

void Neuron::blendPattern(std::vector<double>& target, const std::vector<double>& newPattern, double alpha) {
    for (size_t i = 0; i < target.size(); ++i) {
        target[i] = (1.0 - alpha) * target[i] + alpha * newPattern[i];
    }
}

double Neuron::cosineSimilarity(const std::vector<double>& a, const std::vector<double>& b) {
    double dot = 0.0, normA = 0.0, normB = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    if (normA == 0.0 || normB == 0.0) return 0.0;
    return dot / (std::sqrt(normA) * std::sqrt(normB));
}

// Compute spike distance using a simplified Victor-Purpura-like metric
// Returns a distance (lower is better), which we convert to similarity
double Neuron::spikeDistance(const std::vector<double>& spikes1, const std::vector<double>& spikes2) const {
    // Cost parameter: how much it costs to shift a spike by 1ms
    const double q = 0.5;

    // Handle empty cases
    if (spikes1.empty() && spikes2.empty()) return 0.0;
    if (spikes1.empty()) return static_cast<double>(spikes2.size());
    if (spikes2.empty()) return static_cast<double>(spikes1.size());

    // Simplified greedy matching: for each spike in spikes1, find closest in spikes2
    double totalCost = 0.0;
    std::vector<bool> matched2(spikes2.size(), false);

    for (double s1 : spikes1) {
        double minDist = std::numeric_limits<double>::max();
        int bestMatch = -1;

        // Find closest unmatched spike in spikes2
        for (size_t j = 0; j < spikes2.size(); ++j) {
            if (!matched2[j]) {
                double dist = std::abs(s1 - spikes2[j]);
                if (dist < minDist) {
                    minDist = dist;
                    bestMatch = static_cast<int>(j);
                }
            }
        }

        // Cost is either: shift cost (q * time_diff) or delete+insert cost (2.0)
        if (bestMatch >= 0) {
            double shiftCost = q * minDist;
            double deleteInsertCost = 2.0;

            if (shiftCost < deleteInsertCost) {
                totalCost += shiftCost;
                matched2[bestMatch] = true;
            } else {
                totalCost += deleteInsertCost;
            }
        } else {
            totalCost += 1.0;  // Delete cost
        }
    }

    // Add cost for unmatched spikes in spikes2 (insert cost)
    for (bool matched : matched2) {
        if (!matched) totalCost += 1.0;
    }

    return totalCost;
}

void Neuron::addDendrite(uint64_t dendriteId) {
    // Check if dendrite is already connected
    auto it = std::find(dendriteIds.begin(), dendriteIds.end(), dendriteId);
    if (it == dendriteIds.end()) {
        dendriteIds.push_back(dendriteId);
        SNNFW_DEBUG("Neuron {}: Added dendrite {} (total: {})",
                    getId(), dendriteId, dendriteIds.size());
    } else {
        SNNFW_WARN("Neuron {}: Dendrite {} already connected", getId(), dendriteId);
    }
}

bool Neuron::removeDendrite(uint64_t dendriteId) {
    auto it = std::find(dendriteIds.begin(), dendriteIds.end(), dendriteId);
    if (it != dendriteIds.end()) {
        dendriteIds.erase(it);
        SNNFW_DEBUG("Neuron {}: Removed dendrite {} (remaining: {})",
                    getId(), dendriteId, dendriteIds.size());
        return true;
    }
    SNNFW_WARN("Neuron {}: Dendrite {} not found for removal", getId(), dendriteId);
    return false;
}

std::string Neuron::toJson() const {
    json j;
    j["type"] = "Neuron";
    j["id"] = getId();
    j["windowSize"] = windowSize;
    j["threshold"] = threshold;
    j["maxPatterns"] = maxPatterns;
    j["axonId"] = axonId;
    j["dendriteIds"] = dendriteIds;
    j["spikes"] = spikes;

    // Serialize position if set
    if (hasPosition()) {
        const Position3D& pos = getPosition();
        j["position"] = {
            {"x", pos.x},
            {"y", pos.y},
            {"z", pos.z}
        };
    }

    // Serialize BinaryPatterns as arrays of spike counts
    json patternsJson = json::array();
    for (const auto& pattern : prototypePatterns_) {
        json patternJson = json::array();
        for (size_t i = 0; i < BinaryPattern::PATTERN_SIZE; ++i) {
            patternJson.push_back(pattern[i]);
        }
        patternsJson.push_back(patternJson);
    }
    j["prototypePatterns"] = patternsJson;
    j["prototypeSupports"] = prototypeSupports_;

    json exemplarJson = json::array();
    for (const auto& pattern : exemplarPatterns_) {
        json patternJson = json::array();
        for (size_t i = 0; i < BinaryPattern::PATTERN_SIZE; ++i) {
            patternJson.push_back(pattern[i]);
        }
        exemplarJson.push_back(patternJson);
    }
    j["referencePatterns"] = exemplarJson;
    j["exemplarSupports"] = exemplarSupports_;

    return j.dump();
}

bool Neuron::fromJson(const std::string& jsonStr) {
    try {
        json j = json::parse(jsonStr);

        // Validate type
        if (j["type"] != "Neuron") {
            SNNFW_ERROR("Invalid type in JSON: expected 'Neuron', got '{}'", j["type"].get<std::string>());
            return false;
        }

        // Deserialize ID from base class
        setId(j["id"]);

        // Deserialize position if present
        if (j.contains("position")) {
            float x = j["position"]["x"].get<float>();
            float y = j["position"]["y"].get<float>();
            float z = j["position"]["z"].get<float>();
            setPosition(x, y, z);
        } else {
            clearPosition();
        }

        // Deserialize fields
        windowSize = j["windowSize"];
        threshold = j["threshold"];
        maxPatterns = j["maxPatterns"];
        axonId = j["axonId"];
        dendriteIds = j["dendriteIds"].get<std::vector<uint64_t>>();
        spikes = j["spikes"].get<std::vector<double>>();

        // Deserialize BinaryPatterns from arrays of spike counts
        prototypePatterns_.clear();
        prototypeSupports_.clear();
        exemplarPatterns_.clear();
        exemplarSupports_.clear();
        if (j.contains("prototypePatterns")) {
            for (const auto& patternJson : j["prototypePatterns"]) {
                BinaryPattern pattern;
                for (size_t i = 0; i < BinaryPattern::PATTERN_SIZE && i < patternJson.size(); ++i) {
                    pattern[i] = patternJson[i].get<uint8_t>();
                }
                prototypePatterns_.push_back(pattern);
            }
        }
        if (j.contains("prototypeSupports")) {
            prototypeSupports_ = j["prototypeSupports"].get<std::vector<uint16_t>>();
        }
        clampSupportVector(prototypeSupports_, prototypePatterns_.size(), kPrototypeInitialSupport);
        if (j.contains("referencePatterns")) {
            for (const auto& patternJson : j["referencePatterns"]) {
                BinaryPattern pattern;
                for (size_t i = 0; i < BinaryPattern::PATTERN_SIZE && i < patternJson.size(); ++i) {
                    pattern[i] = patternJson[i].get<uint8_t>();
                }
                exemplarPatterns_.push_back(pattern);
            }
        }
        if (j.contains("exemplarSupports")) {
            exemplarSupports_ = j["exemplarSupports"].get<std::vector<uint16_t>>();
        }
        clampSupportVector(exemplarSupports_, exemplarPatterns_.size(), 1);

        return true;
    } catch (const std::exception& e) {
        SNNFW_ERROR("Failed to deserialize Neuron from JSON: {}", e.what());
        return false;
    }
}

void Neuron::recordIncomingSpike(uint64_t synapseId, double spikeTime, double dispatchTime) {
    std::lock_guard<std::mutex> lock(incomingSpikesMutex_);

    // Add the incoming spike to our tracking deque
    incomingSpikes_.emplace_back(synapseId, spikeTime, dispatchTime);
    incomingSpikeCount_++;

    // Clean up old spikes outside the temporal window.
    // Do not assume arrival ordering under async delivery.
    double referenceTime = spikeTime;
    for (const auto& s : incomingSpikes_) {
        referenceTime = std::max(referenceTime, s.arrivalTime);
    }
    clearOldIncomingSpikes(referenceTime);

    SNNFW_TRACE("Neuron {}: Recorded incoming spike from synapse {} at time {:.3f}ms (dispatch: {:.3f}ms, total tracked: {})",
                getId(), synapseId, spikeTime, dispatchTime, incomingSpikes_.size());
}

size_t Neuron::getIncomingSpikeCount() const {
    std::lock_guard<std::mutex> lock(incomingSpikesMutex_);
    return incomingSpikeCount_;
}

void Neuron::resetIncomingSpikeCount() {
    std::lock_guard<std::mutex> lock(incomingSpikesMutex_);
    incomingSpikeCount_ = 0;
}

int Neuron::fireAndAcknowledge(double firingTime) {
    auto propagator = networkPropagator_.lock();
    if (!propagator) {
        SNNFW_WARN("Neuron {}: Cannot send acknowledgments - NetworkPropagator not set", getId());
        return 0;
    }

    int acknowledgmentCount = 0;

    // Send acknowledgments to all presynaptic neurons that contributed spikes
    // within the temporal window
    // Note: We make a copy of the spikes under lock to avoid holding the lock during network operations
    std::vector<IncomingSpike> spikesToAcknowledge;
    {
        std::lock_guard<std::mutex> lock(incomingSpikesMutex_);
        spikesToAcknowledge.assign(incomingSpikes_.begin(), incomingSpikes_.end());
    }

    for (const auto& incomingSpike : spikesToAcknowledge) {
        // Create acknowledgment with timing information
        auto ack = std::make_shared<SpikeAcknowledgment>(
            incomingSpike.synapseId,
            getId(),  // postsynaptic neuron ID
            firingTime,
            incomingSpike.arrivalTime
        );

        // Send acknowledgment back through the NetworkPropagator
        propagator->sendAcknowledgment(ack);
        acknowledgmentCount++;

        SNNFW_TRACE("Neuron {}: Sent acknowledgment for synapse {} (Δt = {:.3f}ms)",
                    getId(), incomingSpike.synapseId, ack->getTimeDifference());
    }

    // Update firing rate statistics for homeostatic plasticity
    updateFiringRate(firingTime);

    SNNFW_DEBUG("Neuron {}: Fired at {:.3f}ms and sent {} acknowledgments",
                getId(), firingTime, acknowledgmentCount);

    return acknowledgmentCount;
}

void Neuron::clearOldIncomingSpikes(double currentTime) {
    // Note: Caller must hold incomingSpikesMutex_ lock
    // Remove spikes that are older than the temporal window
    incomingSpikes_.erase(
        std::remove_if(incomingSpikes_.begin(), incomingSpikes_.end(),
                       [this, currentTime](const IncomingSpike& s) {
                           return (currentTime - s.arrivalTime) > windowSize;
                       }),
        incomingSpikes_.end());
}

void Neuron::periodicMemoryCleanup(double currentTime) {
    // Clear old spikes from the rolling window and shrink container
    size_t spikesCount = 0;
    {
        std::lock_guard<std::mutex> lock(spikesMutex_);
        removeOldSpikesUnsafe(currentTime);
        spikes.shrink_to_fit();
        spikesCount = spikes.size();
    }

    // Clear old incoming spikes for STDP and shrink container
    size_t incomingSpikesCount = 0;
    {
        std::lock_guard<std::mutex> lock(incomingSpikesMutex_);
        clearOldIncomingSpikes(currentTime);
        incomingSpikes_.shrink_to_fit();
        incomingSpikesCount = incomingSpikes_.size();
    }

    // Shrink pattern storage to fit (BinaryPattern is fixed size, so just shrink the vector)
    prototypePatterns_.shrink_to_fit();
    prototypeSupports_.shrink_to_fit();
    exemplarPatterns_.shrink_to_fit();
    exemplarSupports_.shrink_to_fit();
    // Note: Each BinaryPattern is already fixed at 200 bytes, no need to shrink individual patterns

    SNNFW_TRACE("Neuron {}: Memory cleanup - {} spikes, {} incoming spikes, {} patterns ({}KB pattern memory)",
                getId(), spikesCount, incomingSpikesCount, getLearnedPatternCount(),
                (getLearnedPatternCount() * 200) / 1024);
}

void Neuron::generateTemporalSignature() {
    // Use neuron ID as seed for reproducibility
    std::mt19937 rng(getId());

    // Generate 1-10 spikes
    std::uniform_int_distribution<int> spikeCountDist(1, 10);
    int numSpikes = spikeCountDist(rng);

    // Spread spikes over 0-100ms
    std::uniform_real_distribution<double> timingDist(0.0, 100.0);

    temporalSignature_.clear();
    temporalSignature_.reserve(numSpikes);

    for (int i = 0; i < numSpikes; ++i) {
        temporalSignature_.push_back(timingDist(rng));
    }

    // Sort the offsets for easier processing
    std::sort(temporalSignature_.begin(), temporalSignature_.end());

    SNNFW_TRACE("Neuron {}: Generated temporal signature with {} spikes over {:.1f}ms",
                getId(), numSpikes,
                temporalSignature_.empty() ? 0.0 : (temporalSignature_.back() - temporalSignature_.front()));
}

void Neuron::fireSignature(double baseTime) {
    // Insert spikes according to this neuron's unique temporal signature
    for (double offset : temporalSignature_) {
        insertSpike(baseTime + offset);
    }

    SNNFW_TRACE("Neuron {}: Fired signature pattern with {} spikes starting at {:.3f}ms",
                getId(), temporalSignature_.size(), baseTime);
}

void Neuron::applyInhibition(double amount) {
    inhibition_ += amount;
    SNNFW_TRACE("Neuron {}: Applied inhibition {:.4f}, total inhibition: {:.4f}",
                getId(), amount, inhibition_);
}

double Neuron::getSimilarityActivation() const {
    double similarity = getBestSimilarity();
    if (similarity < 0.0) {
        return 0.0;  // No patterns learned yet
    }
    double activation = similarity * intrinsicExcitability_;
    return std::max(0.0, std::min(1.0, activation));
}

double Neuron::getActivation() const {
    // Activation = similarity-driven activation minus inhibition (clamped to [0, 1])
    double activation = getSimilarityActivation() - inhibition_;
    return std::max(0.0, std::min(1.0, activation));
}

void Neuron::updateFiringRate(double currentTime) {
    const double MEASUREMENT_WINDOW = 1000.0;  // 1 second window in ms

    // Initialize window if needed
    if (firingWindowStart_ == 0.0) {
        firingWindowStart_ = currentTime;
    }

    // Record this firing
    firingCount_++;
    lastFiringTime_ = currentTime;

    // Calculate firing rate if we have enough time elapsed
    double elapsed = currentTime - firingWindowStart_;
    if (elapsed >= MEASUREMENT_WINDOW) {
        // Calculate rate in Hz (firings per second)
        firingRate_ = (firingCount_ / elapsed) * 1000.0;

        // Reset window
        firingCount_ = 0;
        firingWindowStart_ = currentTime;

        SNNFW_TRACE("Neuron {}: Firing rate updated to {:.2f} Hz (target: {:.2f} Hz)",
                    getId(), firingRate_, targetFiringRate_);
    }
}

void Neuron::applyHomeostaticPlasticity() {
    const double LEARNING_RATE = 0.01;  // How quickly to adjust excitability
    const double MIN_EXCITABILITY = 0.5;
    const double MAX_EXCITABILITY = 2.0;

    if (firingRate_ == 0.0) {
        return;  // Not enough data yet
    }

    // Calculate error: how far are we from target?
    double error = targetFiringRate_ - firingRate_;

    // Adjust intrinsic excitability based on error
    // If firing too little (error > 0), increase excitability
    // If firing too much (error < 0), decrease excitability
    double adjustment = LEARNING_RATE * error;
    intrinsicExcitability_ += adjustment;

    // Clamp to reasonable bounds
    intrinsicExcitability_ = std::max(MIN_EXCITABILITY, std::min(MAX_EXCITABILITY, intrinsicExcitability_));

    SNNFW_DEBUG("Neuron {}: Homeostatic adjustment - firing rate: {:.2f} Hz, target: {:.2f} Hz, excitability: {:.3f}",
                getId(), firingRate_, targetFiringRate_, intrinsicExcitability_);
}

} // namespace snnfw

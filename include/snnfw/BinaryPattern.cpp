#include "snnfw/BinaryPattern.h"
#include <sstream>
#include <iomanip>
#include <cstring>

namespace snnfw {

BinaryPattern::BinaryPattern(const std::vector<double>& spikeTimes, double windowSize) : data_() {
    if (spikeTimes.empty()) {
        return;  // Empty pattern
    }

    if (windowSize <= 0.0) {
        return;  // Invalid window, treat as empty
    }

    // Anchor bins to the trailing temporal window rather than the first spike.
    // This preserves latency/phase information inside the window, which is
    // important for temporal pattern discrimination.
    double maxTime = *std::max_element(spikeTimes.begin(), spikeTimes.end());
    double windowStart = maxTime - windowSize;

    // Convert spike times to binned representation (relative to first spike)
    for (double spikeTime : spikeTimes) {
        // Normalize to relative time within [windowStart, windowStart + windowSize)
        double relativeTime = spikeTime - windowStart;

        // Only consider spikes within [0, windowSize)
        if (relativeTime < 0.0 || relativeTime >= windowSize) {
            continue;
        }

        int binIndex = 0;
        // Backward-compatible path: 200ms window maps 1ms -> 1 bin.
        if (std::abs(windowSize - static_cast<double>(PATTERN_SIZE)) < 1e-9) {
            // Round to nearest millisecond
            binIndex = static_cast<int>(std::round(relativeTime));
        } else {
            // Scale the configured window into the fixed-size 200-bin representation.
            // Example: a 500ms window is compressed into 200 bins.
            const double scaled = (relativeTime / windowSize) * static_cast<double>(PATTERN_SIZE);
            binIndex = static_cast<int>(std::floor(scaled));
        }

        // Clamp (handles round() == PATTERN_SIZE on edge cases)
        if (binIndex < 0) binIndex = 0;
        if (binIndex >= static_cast<int>(PATTERN_SIZE)) binIndex = static_cast<int>(PATTERN_SIZE) - 1;

        if (binIndex >= 0 && binIndex < static_cast<int>(PATTERN_SIZE)) {
            // Increment spike count in this bin (saturate at 255)
            if (data_[binIndex] < MAX_COUNT) {
                data_[binIndex]++;
            }
        }
    }
}

size_t BinaryPattern::getTotalSpikes() const {
    size_t total = 0;
    for (uint8_t count : data_) {
        total += count;
    }
    return total;
}

bool BinaryPattern::isEmpty() const {
    for (uint8_t count : data_) {
        if (count > 0) return false;
    }
    return true;
}

void BinaryPattern::clear() {
    data_.fill(0);
}

std::vector<double> BinaryPattern::toSpikeTimes() const {
    std::vector<double> spikeTimes;
    spikeTimes.reserve(getTotalSpikes());
    
    for (size_t i = 0; i < PATTERN_SIZE; ++i) {
        // For each spike in this bin, add a spike time at the bin center
        for (uint8_t count = 0; count < data_[i]; ++count) {
            spikeTimes.push_back(static_cast<double>(i) + 0.5);
        }
    }
    
    return spikeTimes;
}

std::string BinaryPattern::toString() const {
    std::ostringstream oss;
    oss << "BinaryPattern[" << getTotalSpikes() << " spikes]: ";
    
    // Show only non-zero bins to keep output compact
    bool first = true;
    for (size_t i = 0; i < PATTERN_SIZE; ++i) {
        if (data_[i] > 0) {
            if (!first) oss << ", ";
            oss << i << "ms:" << static_cast<int>(data_[i]);
            first = false;
        }
    }
    
    if (first) {
        oss << "(empty)";
    }
    
    return oss.str();
}

// ============================================================================
// Similarity Metrics
// ============================================================================

double BinaryPattern::cosineSimilarity(const BinaryPattern& a, const BinaryPattern& b) {
    int64_t dot = 0;
    int64_t normA = 0;
    int64_t normB = 0;
    
    for (size_t i = 0; i < PATTERN_SIZE; ++i) {
        int32_t valA = a.data_[i];
        int32_t valB = b.data_[i];
        
        dot += valA * valB;
        normA += valA * valA;
        normB += valB * valB;
    }
    
    if (normA == 0 || normB == 0) return 0.0;
    
    return static_cast<double>(dot) / std::sqrt(static_cast<double>(normA * normB));
}

double BinaryPattern::histogramIntersection(const BinaryPattern& a, const BinaryPattern& b) {
    int64_t intersection = 0;
    int64_t unionCount = 0;
    
    for (size_t i = 0; i < PATTERN_SIZE; ++i) {
        uint8_t valA = a.data_[i];
        uint8_t valB = b.data_[i];
        
        intersection += std::min(valA, valB);
        unionCount += std::max(valA, valB);
    }
    
    if (unionCount == 0) return 0.0;
    
    return static_cast<double>(intersection) / static_cast<double>(unionCount);
}

double BinaryPattern::euclideanSimilarity(const BinaryPattern& a, const BinaryPattern& b) {
    int64_t sumSquaredDiff = 0;
    
    for (size_t i = 0; i < PATTERN_SIZE; ++i) {
        int32_t diff = static_cast<int32_t>(a.data_[i]) - static_cast<int32_t>(b.data_[i]);
        sumSquaredDiff += diff * diff;
    }
    
    double distance = std::sqrt(static_cast<double>(sumSquaredDiff));
    
    // Convert distance to similarity: 1 / (1 + distance)
    return 1.0 / (1.0 + distance);
}

double BinaryPattern::correlationSimilarity(const BinaryPattern& a, const BinaryPattern& b) {
    // Compute means
    double meanA = 0.0, meanB = 0.0;
    for (size_t i = 0; i < PATTERN_SIZE; ++i) {
        meanA += a.data_[i];
        meanB += b.data_[i];
    }
    meanA /= PATTERN_SIZE;
    meanB /= PATTERN_SIZE;
    
    // Compute correlation
    double numerator = 0.0;
    double varA = 0.0;
    double varB = 0.0;
    
    for (size_t i = 0; i < PATTERN_SIZE; ++i) {
        double diffA = a.data_[i] - meanA;
        double diffB = b.data_[i] - meanB;
        
        numerator += diffA * diffB;
        varA += diffA * diffA;
        varB += diffB * diffB;
    }
    
    if (varA == 0.0 || varB == 0.0) return 0.0;
    
    double correlation = numerator / std::sqrt(varA * varB);
    
    // Shift from [-1, 1] to [0, 1]
    return (correlation + 1.0) / 2.0;
}

double BinaryPattern::waveformSimilarity(const BinaryPattern& a, const BinaryPattern& b,
                                         double sigma, int maxLag) {
    // Simplified waveform similarity: use simple moving average for smoothing
    // This is much faster than Gaussian convolution while still capturing temporal shape

    const int windowSize = 3;  // 3ms moving average window

    // Compute moving average for pattern A
    std::vector<double> waveA(PATTERN_SIZE, 0.0);
    for (size_t i = 0; i < PATTERN_SIZE; ++i) {
        double sum = 0.0;
        int count = 0;
        for (int k = -windowSize; k <= windowSize; ++k) {
            int idx = static_cast<int>(i) + k;
            if (idx >= 0 && idx < static_cast<int>(PATTERN_SIZE)) {
                sum += a.data_[idx];
                count++;
            }
        }
        waveA[i] = (count > 0) ? (sum / count) : 0.0;
    }

    // Compute moving average for pattern B
    std::vector<double> waveB(PATTERN_SIZE, 0.0);
    for (size_t i = 0; i < PATTERN_SIZE; ++i) {
        double sum = 0.0;
        int count = 0;
        for (int k = -windowSize; k <= windowSize; ++k) {
            int idx = static_cast<int>(i) + k;
            if (idx >= 0 && idx < static_cast<int>(PATTERN_SIZE)) {
                sum += b.data_[idx];
                count++;
            }
        }
        waveB[i] = (count > 0) ? (sum / count) : 0.0;
    }

    // Compute cosine similarity between smoothed waveforms
    double dotProduct = 0.0;
    double normA = 0.0;
    double normB = 0.0;

    for (size_t i = 0; i < PATTERN_SIZE; ++i) {
        dotProduct += waveA[i] * waveB[i];
        normA += waveA[i] * waveA[i];
        normB += waveB[i] * waveB[i];
    }

    if (normA == 0.0 || normB == 0.0) return 0.0;

    double similarity = dotProduct / (std::sqrt(normA) * std::sqrt(normB));

    // Clamp to [0, 1]
    return std::max(0.0, std::min(1.0, similarity));
}

// ============================================================================
// Pattern Blending Operations
// ============================================================================

void BinaryPattern::blend(BinaryPattern& target, const BinaryPattern& source, double alpha) {
    // Clamp alpha to [0, 1]
    alpha = std::max(0.0, std::min(1.0, alpha));
    
    for (size_t i = 0; i < PATTERN_SIZE; ++i) {
        // Weighted average: target = (1-α) × target + α × source
        double blended = (1.0 - alpha) * target.data_[i] + alpha * source.data_[i];
        
        // Round to nearest integer and clamp to [0, 255]
        target.data_[i] = static_cast<uint8_t>(std::round(std::max(0.0, std::min(255.0, blended))));
    }
}

void BinaryPattern::merge(BinaryPattern& target, const BinaryPattern& source, double weight) {
    // Merge is the same as blend, just different naming convention
    blend(target, source, weight);
}

} // namespace snnfw

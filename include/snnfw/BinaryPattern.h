#pragma once

#include <array>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <string>

namespace snnfw {

/**
 * @brief Fixed-size binned spike pattern representation (200 bins with spike counters)
 *
 * This class represents temporal spike patterns using a fixed-size array of
 * byte counters (200 bytes = 200 milliseconds with 1ms resolution).
 *
 * Each byte is a bin counter representing the number of spikes in that millisecond
 * (0-255 spikes per bin). This provides:
 * - Fixed memory footprint (200 bytes per pattern)
 * - Preservation of spike count information (up to 255 spikes/ms)
 * - 1ms temporal resolution (sufficient for neural processing)
 * - Fast similarity computation using integer operations
 * 
 * Memory comparison:
 * - Old: std::vector<double> with ~100 spike times = ~800 bytes (variable)
 * - New: std::array<uint8_t, 200> = 200 bytes (fixed)
 * - Reduction: ~4x memory savings + no unbounded growth
 * 
 * Biological justification:
 * - 1ms resolution matches typical neural temporal precision
 * - 255 spikes/ms far exceeds biological neuron firing rates (~1000 Hz max)
 * - 200ms window captures typical sensory processing timescales
 */
class BinaryPattern {
public:
    static constexpr size_t PATTERN_SIZE = 200;  ///< 200 milliseconds at 1ms resolution
    static constexpr uint8_t MAX_COUNT = 255;     ///< Maximum spikes per bin
    
    /**
     * @brief Default constructor - creates empty pattern (all zeros)
     */
    BinaryPattern() : data_() {}
    
    /**
     * @brief Construct from spike times
     * @param spikeTimes Vector of spike times in milliseconds
     * @param windowSize Size of the temporal window in milliseconds (default: 200ms)
     * 
     * Converts continuous spike times to binned representation:
     * - Each spike time is rounded to nearest millisecond
     * - Spike count in each bin is incremented (capped at 255)
     * - Spikes outside [0, windowSize) are ignored
     */
    explicit BinaryPattern(const std::vector<double>& spikeTimes, double windowSize = 200.0);
    
    /**
     * @brief Get the underlying data array
     * @return Const reference to the 200-byte array
     */
    const std::array<uint8_t, PATTERN_SIZE>& getData() const { return data_; }
    
    /**
     * @brief Get mutable access to the underlying data array
     * @return Reference to the 200-byte array
     */
    std::array<uint8_t, PATTERN_SIZE>& getData() { return data_; }
    
    /**
     * @brief Get spike count at specific time bin
     * @param binIndex Time bin index (0-199)
     * @return Spike count in that bin (0-255)
     */
    uint8_t operator[](size_t binIndex) const { return data_[binIndex]; }
    
    /**
     * @brief Get mutable spike count at specific time bin
     * @param binIndex Time bin index (0-199)
     * @return Reference to spike count in that bin
     */
    uint8_t& operator[](size_t binIndex) { return data_[binIndex]; }
    
    /**
     * @brief Get total number of spikes in the pattern
     * @return Sum of all spike counts across all bins
     */
    size_t getTotalSpikes() const;
    
    /**
     * @brief Check if pattern is empty (no spikes)
     * @return true if all bins are zero, false otherwise
     */
    bool isEmpty() const;
    
    /**
     * @brief Clear the pattern (set all bins to zero)
     */
    void clear();
    
    /**
     * @brief Convert back to spike times (for debugging/visualization)
     * @return Vector of spike times reconstructed from binned representation
     * 
     * Note: This loses sub-millisecond precision. Each spike is placed at
     * the center of its time bin (e.g., bin 5 → spike at 5.5ms).
     */
    std::vector<double> toSpikeTimes() const;
    
    /**
     * @brief Serialize to string (for debugging)
     * @return String representation showing non-zero bins
     */
    std::string toString() const;
    
    // ============================================================================
    // Similarity Metrics
    // ============================================================================
    
    /**
     * @brief Compute cosine similarity between two patterns
     * @param a First pattern
     * @param b Second pattern
     * @return Similarity in [0, 1] where 1 = identical, 0 = orthogonal
     * 
     * Formula: cos(θ) = (a·b) / (||a|| × ||b||)
     * - Dot product: sum of element-wise products
     * - Norms: sqrt of sum of squares
     * 
     * This is compatible with the existing cosine similarity used for
     * vector<double> patterns, making it a drop-in replacement.
     */
    static double cosineSimilarity(const BinaryPattern& a, const BinaryPattern& b);
    
    /**
     * @brief Compute histogram intersection similarity
     * @param a First pattern
     * @param b Second pattern
     * @return Similarity in [0, 1] where 1 = identical, 0 = no overlap
     * 
     * Formula: intersection / union
     * - Intersection: sum of min(a[i], b[i])
     * - Union: sum of max(a[i], b[i])
     * 
     * This metric is particularly good for spike count patterns because
     * it directly measures temporal overlap while accounting for spike density.
     */
    static double histogramIntersection(const BinaryPattern& a, const BinaryPattern& b);
    
    /**
     * @brief Compute Euclidean similarity (L2 distance converted to similarity)
     * @param a First pattern
     * @param b Second pattern
     * @return Similarity in [0, 1] where 1 = identical, 0 = very different
     * 
     * Formula: 1 / (1 + sqrt(sum((a[i] - b[i])^2)))
     * 
     * Euclidean distance measures the straight-line distance between patterns
     * in 200-dimensional space. We convert to similarity for consistency.
     */
    static double euclideanSimilarity(const BinaryPattern& a, const BinaryPattern& b);
    
    /**
     * @brief Compute correlation similarity (Pearson correlation)
     * @param a First pattern
     * @param b Second pattern
     * @return Similarity in [0, 1] where 1 = perfect correlation, 0 = no correlation
     *
     * Formula: correlation = cov(a,b) / (std(a) × std(b))
     * Shifted from [-1, 1] to [0, 1] for consistency
     *
     * Correlation measures linear relationship between patterns, which can
     * capture temporal structure even when spike counts differ.
     */
    static double correlationSimilarity(const BinaryPattern& a, const BinaryPattern& b);

    /**
     * @brief Compute waveform similarity using cross-correlation of Gaussian-smoothed spike trains
     * @param a First pattern
     * @param b Second pattern
     * @param sigma Gaussian kernel width in milliseconds (default: 3.0ms)
     * @param maxLag Maximum temporal lag to search in milliseconds (default: 5ms)
     * @return Similarity in [0, 1] where 1 = identical waveforms, 0 = uncorrelated
     *
     * Algorithm:
     * 1. Convolve each binned spike pattern with Gaussian kernel → smooth waveform
     * 2. Compute normalized cross-correlation at different temporal lags
     * 3. Return maximum correlation (best temporal alignment)
     *
     * This captures the temporal SHAPE of the spike pattern, not just spike counts.
     * It's robust to small temporal shifts and provides biologically plausible
     * temporal integration (neurons integrate inputs over time).
     *
     * Parameters:
     * - sigma: Controls temporal smoothing (larger = smoother, less temporal precision)
     * - maxLag: Allows patterns to be shifted in time (handles temporal jitter)
     */
    static double waveformSimilarity(const BinaryPattern& a, const BinaryPattern& b,
                                     double sigma = 3.0, int maxLag = 5);

    // ============================================================================
    // Pattern Blending Operations (for learning strategies)
    // ============================================================================
    
    /**
     * @brief Blend two patterns with weighted average
     * @param target Target pattern to modify (in-place)
     * @param source Source pattern to blend in
     * @param alpha Blending weight (0 = keep target, 1 = replace with source)
     * 
     * Formula: target[i] = (1-α) × target[i] + α × source[i]
     * 
     * Used by HybridStrategy for Hebbian strengthening of existing patterns.
     */
    static void blend(BinaryPattern& target, const BinaryPattern& source, double alpha);
    
    /**
     * @brief Merge two patterns into a prototype
     * @param target Target pattern to modify (in-place)
     * @param source Source pattern to merge in
     * @param weight Merge weight (typically 0.3 for 70% old, 30% new)
     * 
     * Formula: target[i] = (1-w) × target[i] + w × source[i]
     * 
     * Used by HybridStrategy for consolidating similar patterns into prototypes.
     */
    static void merge(BinaryPattern& target, const BinaryPattern& source, double weight);
    
private:
    std::array<uint8_t, PATTERN_SIZE> data_;  ///< 200 bytes: spike counts per millisecond
};

} // namespace snnfw


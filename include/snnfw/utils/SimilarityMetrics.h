#ifndef SNNFW_SIMILARITY_METRICS_H
#define SNNFW_SIMILARITY_METRICS_H

#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <functional>

namespace snnfw {
namespace utils {

/**
 * @brief Utility functions for computing similarity/distance between patterns
 *
 * Provides various metrics for comparing activation patterns:
 * - Cosine Similarity: Measures angle between vectors (0 to 1)
 * - Euclidean Distance: L2 distance (converted to similarity)
 * - Manhattan Distance: L1 distance (converted to similarity)
 * - Pearson Correlation: Linear correlation (-1 to 1, shifted to 0 to 1)
 *
 * All metrics are normalized to return values in [0, 1] where:
 * - 1.0 = identical patterns
 * - 0.0 = completely different patterns
 *
 * References:
 * - Cha (2007) - Comprehensive survey on distance/similarity measures
 * - Strehl et al. (2000) - Impact of similarity measures on clustering
 */
class SimilarityMetrics {
public:
    /**
     * @brief Cosine similarity between two vectors
     * @param a First vector
     * @param b Second vector
     * @return Similarity in [0, 1] (1 = identical direction)
     *
     * Cosine similarity measures the angle between vectors, ignoring magnitude.
     * It's particularly useful for high-dimensional sparse data.
     *
     * Formula: cos(θ) = (a · b) / (||a|| ||b||)
     */
    static double cosineSimilarity(const std::vector<double>& a, const std::vector<double>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vectors must have the same size");
        }

        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (size_t i = 0; i < a.size(); ++i) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        if (normA == 0.0 || normB == 0.0) {
            return 0.0; // Avoid division by zero
        }

        double similarity = dotProduct / (std::sqrt(normA) * std::sqrt(normB));
        
        // Clamp to [0, 1] (should already be in [-1, 1], but we want [0, 1])
        return std::max(0.0, std::min(1.0, similarity));
    }

    /**
     * @brief Euclidean distance converted to similarity
     * @param a First vector
     * @param b Second vector
     * @return Similarity in [0, 1] (1 = identical, 0 = very different)
     *
     * Euclidean distance is the L2 norm of the difference vector.
     * We convert it to similarity using: similarity = 1 / (1 + distance)
     *
     * Formula: distance = sqrt(Σ(a_i - b_i)²)
     */
    static double euclideanSimilarity(const std::vector<double>& a, const std::vector<double>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vectors must have the same size");
        }

        double sumSquaredDiff = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            double diff = a[i] - b[i];
            sumSquaredDiff += diff * diff;
        }

        double distance = std::sqrt(sumSquaredDiff);
        
        // Convert distance to similarity
        return 1.0 / (1.0 + distance);
    }

    /**
     * @brief Manhattan distance converted to similarity
     * @param a First vector
     * @param b Second vector
     * @return Similarity in [0, 1] (1 = identical, 0 = very different)
     *
     * Manhattan distance is the L1 norm of the difference vector.
     * We convert it to similarity using: similarity = 1 / (1 + distance)
     *
     * Formula: distance = Σ|a_i - b_i|
     */
    static double manhattanSimilarity(const std::vector<double>& a, const std::vector<double>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vectors must have the same size");
        }

        double sumAbsDiff = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            sumAbsDiff += std::abs(a[i] - b[i]);
        }

        // Convert distance to similarity
        return 1.0 / (1.0 + sumAbsDiff);
    }

    /**
     * @brief Pearson correlation coefficient converted to similarity
     * @param a First vector
     * @param b Second vector
     * @return Similarity in [0, 1] (1 = perfect positive correlation, 0 = no/negative correlation)
     *
     * Pearson correlation measures linear relationship between vectors.
     * Original range is [-1, 1], we shift to [0, 1] for consistency.
     *
     * Formula: r = Σ((a_i - mean_a)(b_i - mean_b)) / (std_a * std_b * n)
     */
    static double correlationSimilarity(const std::vector<double>& a, const std::vector<double>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vectors must have the same size");
        }

        if (a.empty()) {
            return 0.0;
        }

        // Compute means
        double meanA = 0.0, meanB = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            meanA += a[i];
            meanB += b[i];
        }
        meanA /= a.size();
        meanB /= b.size();

        // Compute correlation
        double numerator = 0.0;
        double varA = 0.0;
        double varB = 0.0;

        for (size_t i = 0; i < a.size(); ++i) {
            double diffA = a[i] - meanA;
            double diffB = b[i] - meanB;
            numerator += diffA * diffB;
            varA += diffA * diffA;
            varB += diffB * diffB;
        }

        if (varA == 0.0 || varB == 0.0) {
            return 0.0; // Avoid division by zero
        }

        double correlation = numerator / std::sqrt(varA * varB);
        
        // Shift from [-1, 1] to [0, 1]
        return (correlation + 1.0) / 2.0;
    }

    /**
     * @brief Get similarity function by name
     * @param metricName Name of the metric ("cosine", "euclidean", "manhattan", "correlation")
     * @return Function pointer to the similarity metric
     */
    static std::function<double(const std::vector<double>&, const std::vector<double>&)> 
    getMetric(const std::string& metricName) {
        if (metricName == "cosine") {
            return cosineSimilarity;
        } else if (metricName == "euclidean") {
            return euclideanSimilarity;
        } else if (metricName == "manhattan") {
            return manhattanSimilarity;
        } else if (metricName == "correlation") {
            return correlationSimilarity;
        } else {
            throw std::invalid_argument("Unknown similarity metric: " + metricName);
        }
    }

    /**
     * @brief Get list of available metrics
     */
    static std::vector<std::string> getAvailableMetrics() {
        return {"cosine", "euclidean", "manhattan", "correlation"};
    }

    /**
     * @brief Compute pairwise similarity matrix
     * @param patterns Vector of patterns
     * @param metric Similarity metric to use
     * @return NxN similarity matrix where N = patterns.size()
     */
    static std::vector<std::vector<double>> computeSimilarityMatrix(
        const std::vector<std::vector<double>>& patterns,
        std::function<double(const std::vector<double>&, const std::vector<double>&)> metric) {
        
        size_t n = patterns.size();
        std::vector<std::vector<double>> matrix(n, std::vector<double>(n, 0.0));

        for (size_t i = 0; i < n; ++i) {
            matrix[i][i] = 1.0; // Self-similarity is always 1
            for (size_t j = i + 1; j < n; ++j) {
                double sim = metric(patterns[i], patterns[j]);
                matrix[i][j] = sim;
                matrix[j][i] = sim; // Symmetric
            }
        }

        return matrix;
    }
};

} // namespace utils
} // namespace snnfw

#endif // SNNFW_SIMILARITY_METRICS_H


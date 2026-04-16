#ifndef SNNFW_EDGE_OPERATOR_H
#define SNNFW_EDGE_OPERATOR_H

#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <map>

namespace snnfw {
namespace features {

/**
 * @brief Base class for edge detection operators
 *
 * Edge operators extract oriented edge features from image regions.
 * Different operators have different characteristics:
 * - Sobel: Simple, fast, good for general edges
 * - Gabor: Biologically realistic (V1 cortex), tunable frequency/orientation
 * - Quadrature Gabor: phase-invariant simple-cell energy model
 * - Orientation energy: gradient-histogram simple-cell population with tensor sharpening
 * - Difference of Gaussians (DoG): Center-surround, good for blob detection
 *
 * Biological Motivation:
 * The primary visual cortex (V1) contains simple cells that respond to
 * oriented edges at specific spatial frequencies. Gabor filters closely
 * model these receptive fields.
 *
 * References:
 * - Hubel & Wiesel (1962) - Receptive fields in cat visual cortex
 * - Daugman (1985) - Uncertainty relation for resolution in space/frequency
 * - Marcelja (1980) - Mathematical description of simple cells
 */
class EdgeOperator {
public:
    /**
     * @brief Edge operator configuration
     */
    struct Config {
        std::string name;                           ///< Operator name
        int numOrientations;                        ///< Number of orientations to detect
        double edgeThreshold;                       ///< Minimum edge strength (0.0 to 1.0)
        std::map<std::string, double> doubleParams; ///< Additional double parameters
        std::map<std::string, int> intParams;       ///< Additional integer parameters
        
        // Helper methods
        double getDoubleParam(const std::string& key, double defaultValue = 0.0) const {
            auto it = doubleParams.find(key);
            return (it != doubleParams.end()) ? it->second : defaultValue;
        }
        
        int getIntParam(const std::string& key, int defaultValue = 0) const {
            auto it = intParams.find(key);
            return (it != intParams.end()) ? it->second : defaultValue;
        }
    };

    /**
     * @brief Constructor
     * @param config Operator configuration
     */
    explicit EdgeOperator(const Config& config) : config_(config) {}

    /**
     * @brief Virtual destructor
     */
    virtual ~EdgeOperator() = default;

    /**
     * @brief Extract edge features from an image region
     * @param region Flattened image region (row-major order)
     * @param regionSize Size of the square region (e.g., 4 for 4x4)
     * @return Vector of edge strengths, one per orientation (normalized 0.0 to 1.0)
     *
     * The returned vector has size equal to numOrientations.
     * Each element represents the strength of edges at that orientation.
     * Orientations are evenly distributed from 0° to 180°.
     */
    virtual std::vector<double> extractEdges(
        const std::vector<uint8_t>& region,
        int regionSize) const = 0;

    /**
     * @brief Get the number of orientations this operator detects
     */
    int getNumOrientations() const { return config_.numOrientations; }

    /**
     * @brief Get the edge threshold
     */
    double getEdgeThreshold() const { return config_.edgeThreshold; }

    /**
     * @brief Get the operator name
     */
    virtual std::string getName() const = 0;

    /**
     * @brief Get the configuration
     */
    const Config& getConfig() const { return config_; }

protected:
    Config config_; ///< Operator configuration

    /**
     * @brief Helper: Get pixel value from region
     * @param region Flattened image region
     * @param row Row index
     * @param col Column index
     * @param regionSize Size of the square region
     * @return Pixel value (0-255)
     */
    static uint8_t getPixel(const std::vector<uint8_t>& region, 
                           int row, int col, int regionSize) {
        if (row < 0 || row >= regionSize || col < 0 || col >= regionSize) {
            return 0; // Out of bounds
        }
        return region[row * regionSize + col];
    }

    /**
     * @brief Helper: Get pixel value as double (normalized 0.0 to 1.0)
     */
    static double getPixelNormalized(const std::vector<uint8_t>& region,
                                    int row, int col, int regionSize) {
        return static_cast<double>(getPixel(region, row, col, regionSize)) / 255.0;
    }

    /**
     * @brief Helper: Normalize a vector of features to [0, 1]
     * @param features Input features (any range)
     * @return Normalized features (0.0 to 1.0)
     */
    static std::vector<double> normalizeFeatures(const std::vector<double>& features) {
        if (features.empty()) return features;
        
        // Find max value
        double maxVal = 0.0;
        for (double f : features) {
            if (f > maxVal) maxVal = f;
        }
        
        // Normalize
        if (maxVal > 0.0) {
            std::vector<double> normalized(features.size());
            for (size_t i = 0; i < features.size(); ++i) {
                normalized[i] = features[i] / maxVal;
            }
            return normalized;
        }
        
        return features;
    }

    /**
     * @brief Helper: Apply threshold to features
     * @param features Input features
     * @param threshold Minimum value (values below this become 0)
     * @return Thresholded features
     */
    std::vector<double> applyThreshold(const std::vector<double>& features) const {
        std::vector<double> thresholded(features.size());
        for (size_t i = 0; i < features.size(); ++i) {
            thresholded[i] = (features[i] >= config_.edgeThreshold) ? features[i] : 0.0;
        }
        return thresholded;
    }
};

/**
 * @brief Factory for creating edge operators
 */
class EdgeOperatorFactory {
public:
    /**
     * @brief Create an edge operator from configuration
     * @param type Operator type ("sobel", "gabor", "quadrature_gabor", "orientation_energy", "dog")
     * @param config Operator configuration
     * @return Unique pointer to edge operator
     */
    static std::unique_ptr<EdgeOperator> create(
        const std::string& type,
        const EdgeOperator::Config& config);

    /**
     * @brief Get list of available edge operators
     */
    static std::vector<std::string> getAvailableOperators();
};

} // namespace features
} // namespace snnfw

#endif // SNNFW_EDGE_OPERATOR_H

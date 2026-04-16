#ifndef SNNFW_SOBEL_OPERATOR_H
#define SNNFW_SOBEL_OPERATOR_H

#include "snnfw/features/EdgeOperator.h"

namespace snnfw {
namespace features {

/**
 * @brief Sobel edge detection operator (baseline)
 *
 * The Sobel operator is a simple and fast edge detection method that uses
 * 3x3 convolution kernels to compute gradients in different orientations.
 *
 * For each orientation, we compute the gradient using directional differences
 * between neighboring pixels. This is the current baseline implementation.
 *
 * Characteristics:
 * - Fast computation (simple differences)
 * - Good for general edge detection
 * - Less biologically realistic than Gabor
 * - Works well for high-contrast edges
 *
 * References:
 * - Sobel & Feldman (1968) - A 3x3 isotropic gradient operator
 * - Duda & Hart (1973) - Pattern Classification and Scene Analysis
 */
class SobelOperator : public EdgeOperator {
public:
    /**
     * @brief Constructor
     * @param config Operator configuration
     */
    explicit SobelOperator(const Config& config);

    /**
     * @brief Extract edge features using Sobel-like gradients
     * @param region Flattened image region (row-major order)
     * @param regionSize Size of the square region
     * @return Vector of edge strengths, one per orientation
     */
    std::vector<double> extractEdges(
        const std::vector<uint8_t>& region,
        int regionSize) const override;

    /**
     * @brief Get operator name
     */
    std::string getName() const override;

private:
    /**
     * @brief Compute gradient in a specific orientation
     * @param region Image region
     * @param regionSize Size of region
     * @param orientation Orientation index (0 to numOrientations-1)
     * @return Gradient magnitude
     */
    double computeOrientedGradient(
        const std::vector<uint8_t>& region,
        int regionSize,
        int orientation) const;
};

} // namespace features
} // namespace snnfw

#endif // SNNFW_SOBEL_OPERATOR_H


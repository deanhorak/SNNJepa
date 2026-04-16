#ifndef SNNFW_DOG_OPERATOR_H
#define SNNFW_DOG_OPERATOR_H

#include "snnfw/features/EdgeOperator.h"

namespace snnfw {
namespace features {

/**
 * @brief Difference of Gaussians (DoG) edge detection operator
 *
 * The DoG operator approximates the Laplacian of Gaussian (LoG) and models
 * center-surround receptive fields found in retinal ganglion cells and
 * LGN neurons. It's computed as the difference between two Gaussian blurs
 * with different standard deviations.
 *
 * Formula:
 * DoG(x,y) = G(x,y,σ₁) - G(x,y,σ₂)
 *
 * where:
 * - G(x,y,σ) = (1/(2πσ²)) * exp(-(x² + y²)/(2σ²))
 * - σ₁ < σ₂ (typically σ₂ = 1.6 * σ₁)
 *
 * Characteristics:
 * - Models center-surround receptive fields
 * - Good for blob detection and edge enhancement
 * - Approximates Laplacian of Gaussian
 * - Scale-invariant edge detection
 *
 * Biological Motivation:
 * Retinal ganglion cells and LGN neurons have center-surround receptive
 * fields that can be modeled as DoG filters. This provides edge enhancement
 * and contrast normalization early in visual processing.
 *
 * References:
 * - Marr & Hildreth (1980) - Theory of edge detection
 * - Rodieck (1965) - Quantitative analysis of cat retinal ganglion cells
 * - Lowe (2004) - SIFT features use DoG for scale-space
 */
class DoGOperator : public EdgeOperator {
public:
    /**
     * @brief Constructor
     * @param config Operator configuration
     *
     * Additional config parameters:
     * - "sigma1": Standard deviation of first Gaussian (default: 1.0)
     * - "sigma2": Standard deviation of second Gaussian (default: 1.6)
     * - "kernel_size": Size of Gaussian kernel (default: 5)
     */
    explicit DoGOperator(const Config& config);

    /**
     * @brief Extract edge features using DoG filters
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
    double sigma1_;    ///< Standard deviation of first Gaussian
    double sigma2_;    ///< Standard deviation of second Gaussian
    int kernelSize_;   ///< Size of Gaussian kernel (must be odd)

    /**
     * @brief Apply Gaussian blur to region
     * @param region Image region
     * @param regionSize Size of region
     * @param sigma Standard deviation of Gaussian
     * @return Blurred region
     */
    std::vector<double> applyGaussianBlur(
        const std::vector<uint8_t>& region,
        int regionSize,
        double sigma) const;

    /**
     * @brief Compute Gaussian kernel value
     * @param x X coordinate (relative to center)
     * @param y Y coordinate (relative to center)
     * @param sigma Standard deviation
     * @return Kernel value
     */
    double gaussianKernel(double x, double y, double sigma) const;

    /**
     * @brief Compute DoG response for a specific orientation
     * @param dog DoG-filtered region
     * @param regionSize Size of region
     * @param orientation Orientation index
     * @return Oriented edge strength
     */
    double computeOrientedResponse(
        const std::vector<double>& dog,
        int regionSize,
        int orientation) const;
};

} // namespace features
} // namespace snnfw

#endif // SNNFW_DOG_OPERATOR_H


#ifndef SNNFW_GABOR_OPERATOR_H
#define SNNFW_GABOR_OPERATOR_H

#include "snnfw/features/EdgeOperator.h"

namespace snnfw {
namespace features {

/**
 * @brief Gabor filter edge detection operator
 *
 * Gabor filters are biologically realistic edge detectors that closely model
 * the receptive fields of simple cells in the primary visual cortex (V1).
 * They combine a Gaussian envelope with a sinusoidal carrier wave.
 *
 * Formula:
 * G(x,y,θ,λ,σ,γ) = exp(-(x'² + γ²y'²)/(2σ²)) * cos(2πx'/λ + ψ)
 *
 * where:
 * - x' = x*cos(θ) + y*sin(θ)  (rotation)
 * - y' = -x*sin(θ) + y*cos(θ) (rotation)
 * - θ = orientation angle
 * - λ = wavelength (spatial frequency)
 * - σ = Gaussian envelope width
 * - γ = spatial aspect ratio
 * - ψ = phase offset
 *
 * Characteristics:
 * - Biologically realistic (models V1 simple cells)
 * - Tuned to specific orientations and spatial frequencies
 * - Better edge detection than Sobel for complex patterns
 * - More computationally expensive
 *
 * Biological Motivation:
 * Hubel & Wiesel (1962) discovered that V1 simple cells respond to oriented
 * edges at specific spatial frequencies. Gabor filters closely match these
 * receptive field properties.
 *
 * References:
 * - Hubel & Wiesel (1962) - Receptive fields in cat visual cortex
 * - Daugman (1985) - Uncertainty relation for resolution
 * - Marcelja (1980) - Mathematical description of simple cells
 * - Jones & Palmer (1987) - Evaluation of 2D Gabor filter models
 */
class GaborOperator : public EdgeOperator {
public:
    /**
     * @brief Constructor
     * @param config Operator configuration
     *
     * Additional config parameters:
     * - "wavelength": Wavelength of sinusoidal carrier (default: 4.0 pixels)
     * - "sigma": Gaussian envelope width (default: 2.0 pixels)
     * - "gamma": Spatial aspect ratio (default: 0.5)
     * - "phase_offset": Phase offset in radians (default: 0.0)
     * - "kernel_size": Size of Gabor kernel (default: 5)
     */
    explicit GaborOperator(const Config& config);

    /**
     * @brief Extract edge features using Gabor filters
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
    double wavelength_;    ///< Wavelength of sinusoidal carrier (pixels)
    double sigma_;         ///< Gaussian envelope width (pixels)
    double gamma_;         ///< Spatial aspect ratio
    double phaseOffset_;   ///< Phase offset (radians)
    int kernelSize_;       ///< Size of Gabor kernel (must be odd)
    double gamma_sq_;      ///< Cached gamma^2
    double two_sigma_sq_;  ///< Cached 2*sigma^2

    /**
     * @brief Compute Gabor filter response for a specific orientation
     * @param region Image region
     * @param regionSize Size of region
     * @param orientation Orientation index
     * @return Filter response magnitude
     */
    double computeGaborResponse(
        const std::vector<uint8_t>& region,
        int regionSize,
        int orientation) const;

    /**
     * @brief Compute Gabor kernel value at a specific position
     * @param x X coordinate (relative to kernel center)
     * @param y Y coordinate (relative to kernel center)
     * @param theta Orientation angle in radians
     * @return Kernel value
     */
    double gaborKernel(double x, double y, double theta) const;

    /**
     * @brief Apply Gabor filter at a specific position
     * @param region Image region
     * @param regionSize Size of region
     * @param centerR Center row
     * @param centerC Center column
     * @param theta Orientation angle in radians
     * @return Filter response
     */
    double applyGaborFilter(
        const std::vector<uint8_t>& region,
        int regionSize,
        int centerR,
        int centerC,
        double theta) const;
};

} // namespace features
} // namespace snnfw

#endif // SNNFW_GABOR_OPERATOR_H

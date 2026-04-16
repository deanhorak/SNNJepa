#ifndef SNNFW_ORIENTATION_ENERGY_OPERATOR_H
#define SNNFW_ORIENTATION_ENERGY_OPERATOR_H

#include "snnfw/features/EdgeOperator.h"

namespace snnfw {
namespace features {

/**
 * @brief Phase-insensitive oriented-energy operator with tensor-based sharpening.
 *
 * This operator builds a local orientation histogram from Sobel-like gradients,
 * then optionally sharpens that histogram using a structure-tensor estimate of
 * the dominant patch orientation. It keeps the same output interface as the
 * existing edge operators while providing distinct orientation bins for natural
 * image patches.
 */
class OrientationEnergyOperator : public EdgeOperator {
public:
    explicit OrientationEnergyOperator(const Config& config);

    std::vector<double> extractEdges(
        const std::vector<uint8_t>& region,
        int regionSize) const override;

    std::string getName() const override;

private:
    double magnitudeGamma_;
    double angularSharpness_;
    double tensorMix_;
    double tensorSharpness_;
    double tensorFloor_;

    static double wrapOrientation(double angle);
    static double orientationDistance(double a, double b);
    static double cosineTuning(double delta, double sharpness);
    void computeGradient(const std::vector<uint8_t>& region,
                         int regionSize,
                         int row,
                         int col,
                         double& gx,
                         double& gy) const;
};

} // namespace features
} // namespace snnfw

#endif // SNNFW_ORIENTATION_ENERGY_OPERATOR_H

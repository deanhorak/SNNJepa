#include "snnfw/features/OrientationEnergyOperator.h"
#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace snnfw {
namespace features {

OrientationEnergyOperator::OrientationEnergyOperator(const Config& config)
    : EdgeOperator(config)
    , magnitudeGamma_(std::max(0.1, config.getDoubleParam("orientation_energy_magnitude_gamma", 1.0)))
    , angularSharpness_(std::max(1.0, config.getDoubleParam("orientation_energy_sharpness", 6.0)))
    , tensorMix_(std::clamp(config.getDoubleParam("orientation_energy_tensor_mix", 0.35), 0.0, 1.0))
    , tensorSharpness_(std::max(1.0, config.getDoubleParam("orientation_energy_tensor_sharpness", 8.0)))
    , tensorFloor_(std::clamp(config.getDoubleParam("orientation_energy_tensor_floor", 0.2), 0.0, 1.0)) {}

double OrientationEnergyOperator::wrapOrientation(double angle) {
    double wrapped = std::fmod(angle, M_PI);
    if (wrapped < 0.0) {
        wrapped += M_PI;
    }
    return wrapped;
}

double OrientationEnergyOperator::orientationDistance(double a, double b) {
    double delta = std::abs(wrapOrientation(a) - wrapOrientation(b));
    if (delta > (M_PI / 2.0)) {
        delta = M_PI - delta;
    }
    return delta;
}

double OrientationEnergyOperator::cosineTuning(double delta, double sharpness) {
    const double clampedDelta = std::min(delta, M_PI / 2.0);
    const double base = std::max(0.0, std::cos(clampedDelta));
    return std::pow(base, sharpness);
}

void OrientationEnergyOperator::computeGradient(const std::vector<uint8_t>& region,
                                                int regionSize,
                                                int row,
                                                int col,
                                                double& gx,
                                                double& gy) const {
    const double topLeft = getPixelNormalized(region, row - 1, col - 1, regionSize);
    const double top = getPixelNormalized(region, row - 1, col, regionSize);
    const double topRight = getPixelNormalized(region, row - 1, col + 1, regionSize);
    const double left = getPixelNormalized(region, row, col - 1, regionSize);
    const double right = getPixelNormalized(region, row, col + 1, regionSize);
    const double bottomLeft = getPixelNormalized(region, row + 1, col - 1, regionSize);
    const double bottom = getPixelNormalized(region, row + 1, col, regionSize);
    const double bottomRight = getPixelNormalized(region, row + 1, col + 1, regionSize);

    gx = (-topLeft + topRight) + (-2.0 * left + 2.0 * right) + (-bottomLeft + bottomRight);
    gy = (-topLeft - (2.0 * top) - topRight) + (bottomLeft + (2.0 * bottom) + bottomRight);
}

std::vector<double> OrientationEnergyOperator::extractEdges(
    const std::vector<uint8_t>& region,
    int regionSize) const {
    std::vector<double> responses(config_.numOrientations, 0.0);
    if (regionSize < 3 || region.empty() || config_.numOrientations <= 0) {
        return responses;
    }

    double jxx = 0.0;
    double jyy = 0.0;
    double jxy = 0.0;

    for (int r = 1; r < regionSize - 1; ++r) {
        for (int c = 1; c < regionSize - 1; ++c) {
            double gx = 0.0;
            double gy = 0.0;
            computeGradient(region, regionSize, r, c, gx, gy);

            const double magnitude = std::sqrt((gx * gx) + (gy * gy));
            if (magnitude <= 1e-9) {
                continue;
            }

            const double weight = std::pow(magnitude, magnitudeGamma_);
            const double edgeOrientation = wrapOrientation(std::atan2(gy, gx) + (M_PI / 2.0));

            for (int orient = 0; orient < config_.numOrientations; ++orient) {
                const double binOrientation =
                    (static_cast<double>(orient) * M_PI) / static_cast<double>(config_.numOrientations);
                responses[static_cast<size_t>(orient)] +=
                    weight * cosineTuning(orientationDistance(edgeOrientation, binOrientation),
                                          angularSharpness_);
            }

            jxx += gx * gx;
            jyy += gy * gy;
            jxy += gx * gy;
        }
    }

    if (tensorMix_ > 0.0) {
        const double tensorEnergy = jxx + jyy;
        if (tensorEnergy > 1e-12) {
            const double anisotropy =
                std::sqrt(((jxx - jyy) * (jxx - jyy)) + (4.0 * jxy * jxy)) / tensorEnergy;
            const double dominantOrientation =
                wrapOrientation(0.5 * std::atan2(2.0 * jxy, jxx - jyy) + (M_PI / 2.0));
            for (int orient = 0; orient < config_.numOrientations; ++orient) {
                const double binOrientation =
                    (static_cast<double>(orient) * M_PI) / static_cast<double>(config_.numOrientations);
                const double prior =
                    tensorFloor_ +
                    ((1.0 - tensorFloor_) * anisotropy *
                     cosineTuning(orientationDistance(dominantOrientation, binOrientation),
                                  tensorSharpness_));
                responses[static_cast<size_t>(orient)] *=
                    ((1.0 - tensorMix_) + (tensorMix_ * prior));
            }
        }
    }

    responses = normalizeFeatures(responses);
    responses = applyThreshold(responses);
    return responses;
}

std::string OrientationEnergyOperator::getName() const {
    return "OrientationEnergyOperator";
}

} // namespace features
} // namespace snnfw

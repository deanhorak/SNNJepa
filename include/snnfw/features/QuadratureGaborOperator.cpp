#include "snnfw/features/QuadratureGaborOperator.h"
#include <algorithm>
#include <cassert>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace snnfw {
namespace features {

QuadratureGaborOperator::QuadratureGaborOperator(const Config& config)
    : EdgeOperator(config)
    , wavelength_(config.getDoubleParam("wavelength", 3.2))
    , sigma_(config.getDoubleParam("sigma", 1.2))
    , gamma_(config.getDoubleParam("gamma", 0.7))
    , kernelSize_(config.getIntParam("kernel_size", 5))
    , energyGamma_(std::max(0.1, config.getDoubleParam("quadrature_energy_gamma", 1.0)))
    , gammaSq_(gamma_ * gamma_)
    , twoSigmaSq_(2.0 * sigma_ * sigma_) {
    assert(wavelength_ > 0.0 && "Wavelength must be positive.");
    assert(sigma_ > 0.0 && "Sigma must be positive.");

    if (kernelSize_ % 2 == 0) {
        kernelSize_++;
    }
    if (kernelSize_ < 3) {
        kernelSize_ = 3;
    }
}

double QuadratureGaborOperator::gaborKernel(double x,
                                            double y,
                                            double theta,
                                            double phase) const {
    const double xTheta = x * std::cos(theta) + y * std::sin(theta);
    const double yTheta = -x * std::sin(theta) + y * std::cos(theta);
    const double gaussianExponent =
        -(xTheta * xTheta + gammaSq_ * yTheta * yTheta) / twoSigmaSq_;
    const double gaussian = std::exp(gaussianExponent);
    const double carrier = std::cos((2.0 * M_PI * xTheta / wavelength_) + phase);
    return gaussian * carrier;
}

double QuadratureGaborOperator::applyGaborFilter(const std::vector<uint8_t>& region,
                                                 int regionSize,
                                                 int centerR,
                                                 int centerC,
                                                 double theta,
                                                 double phase,
                                                 double regionMean) const {
    double response = 0.0;
    const int halfKernel = kernelSize_ / 2;

    for (int kr = -halfKernel; kr <= halfKernel; ++kr) {
        for (int kc = -halfKernel; kc <= halfKernel; ++kc) {
            const int r = centerR + kr;
            const int c = centerC + kc;
            if (r < 0 || r >= regionSize || c < 0 || c >= regionSize) {
                continue;
            }
            const double pixelValue =
                getPixelNormalized(region, r, c, regionSize) - regionMean;
            response += pixelValue *
                        gaborKernel(static_cast<double>(kc),
                                    static_cast<double>(kr),
                                    theta,
                                    phase);
        }
    }

    return response;
}

double QuadratureGaborOperator::computeQuadratureEnergy(
    const std::vector<uint8_t>& region,
    int regionSize,
    int orientation) const {
    const double theta = (static_cast<double>(orientation) * M_PI) /
                         static_cast<double>(config_.numOrientations);
    const int halfKernel = kernelSize_ / 2;
    if (regionSize <= kernelSize_) {
        return 0.0;
    }

    double sum = 0.0;
    for (uint8_t pixel : region) {
        sum += static_cast<double>(pixel) / 255.0;
    }
    const double regionMean = sum / static_cast<double>(region.size());

    double totalEnergy = 0.0;
    for (int r = halfKernel; r < regionSize - halfKernel; ++r) {
        for (int c = halfKernel; c < regionSize - halfKernel; ++c) {
            const double even =
                applyGaborFilter(region, regionSize, r, c, theta, 0.0, regionMean);
            const double odd =
                applyGaborFilter(region, regionSize, r, c, theta, M_PI / 2.0, regionMean);
            const double localEnergy = std::sqrt((even * even) + (odd * odd));
            totalEnergy += std::pow(localEnergy, energyGamma_);
        }
    }

    return totalEnergy;
}

std::vector<double> QuadratureGaborOperator::extractEdges(
    const std::vector<uint8_t>& region,
    int regionSize) const {
    std::vector<double> features(config_.numOrientations, 0.0);
    if (region.empty() || regionSize < 3 || config_.numOrientations <= 0) {
        return features;
    }

    for (int orient = 0; orient < config_.numOrientations; ++orient) {
        features[static_cast<size_t>(orient)] =
            computeQuadratureEnergy(region, regionSize, orient);
    }

    features = normalizeFeatures(features);
    features = applyThreshold(features);
    return features;
}

std::string QuadratureGaborOperator::getName() const {
    return "QuadratureGaborOperator";
}

} // namespace features
} // namespace snnfw

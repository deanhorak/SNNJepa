#include "snnfw/features/GaborOperator.h"
#include <cmath>
#include <algorithm>
#include <cassert>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace snnfw {
namespace features {

GaborOperator::GaborOperator(const Config& config)
    : EdgeOperator(config) {
    
    // Parse additional configuration
    wavelength_ = config.getDoubleParam("wavelength", 4.0);
    sigma_ = config.getDoubleParam("sigma", 2.0);
    gamma_ = config.getDoubleParam("gamma", 0.5);
    phaseOffset_ = config.getDoubleParam("phase_offset", 0.0);
    kernelSize_ = config.getIntParam("kernel_size", 5);

    // Pre-calculate expensive terms for the kernel
    gamma_sq_ = gamma_ * gamma_;
    two_sigma_sq_ = 2.0 * sigma_ * sigma_;

    assert(wavelength_ > 0.0 && "Wavelength must be positive.");
    assert(sigma_ > 0.0 && "Sigma must be positive.");
    
    // Ensure kernel size is odd
    if (kernelSize_ % 2 == 0) {
        kernelSize_++;
    }
    
    // Ensure kernel size is at least 3
    if (kernelSize_ < 3) {
        kernelSize_ = 3;
    }
}

double GaborOperator::gaborKernel(double x, double y, double theta) const {
    // Rotate coordinates
    double xTheta = x * std::cos(theta) + y * std::sin(theta);
    double yTheta = -x * std::sin(theta) + y * std::cos(theta);
    
    // Gaussian envelope
    double gaussianExponent = -(xTheta * xTheta + gamma_sq_ * yTheta * yTheta) / two_sigma_sq_;
    double gaussian = std::exp(gaussianExponent);
    
    // Sinusoidal carrier
    double sinusoid = std::cos(2.0 * M_PI * xTheta / wavelength_ + phaseOffset_);
    
    // Gabor function
    return gaussian * sinusoid;
}

double GaborOperator::applyGaborFilter(
    const std::vector<uint8_t>& region,
    int regionSize,
    int centerR,
    int centerC,
    double theta) const {
    
    double response = 0.0;
    int halfKernel = kernelSize_ / 2;
    
    // Apply Gabor kernel
    for (int kr = -halfKernel; kr <= halfKernel; ++kr) {
        for (int kc = -halfKernel; kc <= halfKernel; ++kc) {
            int r = centerR + kr;
            int c = centerC + kc;
            
            // Check bounds
            if (r >= 0 && r < regionSize && c >= 0 && c < regionSize) {
                double pixelValue = getPixelNormalized(region, r, c, regionSize);
                double kernelValue = gaborKernel(static_cast<double>(kc), 
                                                static_cast<double>(kr), 
                                                theta);
                response += pixelValue * kernelValue;
            }
        }
    }
    
    return response;
}

double GaborOperator::computeGaborResponse(
    const std::vector<uint8_t>& region,
    int regionSize,
    int orientation) const {
    
    // Compute orientation angle
    // Distribute orientations evenly across 180 degrees (π radians)
    double theta = (orientation * M_PI) / config_.numOrientations;
    
    double totalResponse = 0.0;
    int halfKernel = kernelSize_ / 2;
    
    // Apply Gabor filter at each position (excluding borders)
    for (int r = halfKernel; r < regionSize - halfKernel; ++r) {
        for (int c = halfKernel; c < regionSize - halfKernel; ++c) {
            double response = applyGaborFilter(region, regionSize, r, c, theta);
            
            // Use absolute value of response (energy)
            totalResponse += std::abs(response);
        }
    }
    
    return totalResponse;
}

std::vector<double> GaborOperator::extractEdges(
    const std::vector<uint8_t>& region,
    int regionSize) const {
    
    std::vector<double> features(config_.numOrientations, 0.0);
    
    // Compute Gabor response for each orientation
    for (int orient = 0; orient < config_.numOrientations; ++orient) {
        features[orient] = computeGaborResponse(region, regionSize, orient);
    }
    
    // Normalize features to [0, 1]
    features = normalizeFeatures(features);
    
    // Apply threshold
    features = applyThreshold(features);
    
    return features;
}

std::string GaborOperator::getName() const {
    return "GaborOperator";
}

} // namespace features
} // namespace snnfw

#include "snnfw/features/DoGOperator.h"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace snnfw {
namespace features {

DoGOperator::DoGOperator(const Config& config)
    : EdgeOperator(config) {
    
    // Parse additional configuration
    sigma1_ = config.getDoubleParam("sigma1", 1.0);
    sigma2_ = config.getDoubleParam("sigma2", 1.6);
    kernelSize_ = config.getIntParam("kernel_size", 5);
    
    // Ensure kernel size is odd
    if (kernelSize_ % 2 == 0) {
        kernelSize_++;
    }
    
    // Ensure kernel size is at least 3
    if (kernelSize_ < 3) {
        kernelSize_ = 3;
    }
    
    // Ensure sigma2 > sigma1
    if (sigma2_ <= sigma1_) {
        sigma2_ = sigma1_ * 1.6;
    }
}

double DoGOperator::gaussianKernel(double x, double y, double sigma) const {
    double exponent = -(x * x + y * y) / (2.0 * sigma * sigma);
    double coefficient = 1.0 / (2.0 * M_PI * sigma * sigma);
    return coefficient * std::exp(exponent);
}

std::vector<double> DoGOperator::applyGaussianBlur(
    const std::vector<uint8_t>& region,
    int regionSize,
    double sigma) const {
    
    std::vector<double> blurred(regionSize * regionSize, 0.0);
    int halfKernel = kernelSize_ / 2;
    
    // Apply Gaussian blur
    for (int r = 0; r < regionSize; ++r) {
        for (int c = 0; c < regionSize; ++c) {
            double sum = 0.0;
            double weightSum = 0.0;
            
            // Convolve with Gaussian kernel
            for (int kr = -halfKernel; kr <= halfKernel; ++kr) {
                for (int kc = -halfKernel; kc <= halfKernel; ++kc) {
                    int nr = r + kr;
                    int nc = c + kc;
                    
                    // Check bounds
                    if (nr >= 0 && nr < regionSize && nc >= 0 && nc < regionSize) {
                        double pixelValue = getPixelNormalized(region, nr, nc, regionSize);
                        double weight = gaussianKernel(static_cast<double>(kc), 
                                                      static_cast<double>(kr), 
                                                      sigma);
                        sum += pixelValue * weight;
                        weightSum += weight;
                    }
                }
            }
            
            // Normalize by weight sum
            if (weightSum > 0.0) {
                blurred[r * regionSize + c] = sum / weightSum;
            }
        }
    }
    
    return blurred;
}

double DoGOperator::computeOrientedResponse(
    const std::vector<double>& dog,
    int regionSize,
    int orientation) const {
    
    // Compute orientation angle
    double theta = (orientation * M_PI) / config_.numOrientations;
    double dx = std::cos(theta);
    double dy = std::sin(theta);
    
    double response = 0.0;
    
    // Compute oriented gradient from DoG
    for (int r = 1; r < regionSize - 1; ++r) {
        for (int c = 1; c < regionSize - 1; ++c) {
            // Compute gradient at this position
            double gradX = dog[r * regionSize + (c + 1)] - dog[r * regionSize + (c - 1)];
            double gradY = dog[(r + 1) * regionSize + c] - dog[(r - 1) * regionSize + c];
            
            // Project gradient onto orientation direction
            double orientedGrad = std::abs(dx * gradX + dy * gradY);
            response += orientedGrad;
        }
    }
    
    return response;
}

std::vector<double> DoGOperator::extractEdges(
    const std::vector<uint8_t>& region,
    int regionSize) const {
    
    // Apply two Gaussian blurs with different sigmas
    std::vector<double> blur1 = applyGaussianBlur(region, regionSize, sigma1_);
    std::vector<double> blur2 = applyGaussianBlur(region, regionSize, sigma2_);
    
    // Compute Difference of Gaussians
    std::vector<double> dog(regionSize * regionSize);
    for (size_t i = 0; i < dog.size(); ++i) {
        dog[i] = blur1[i] - blur2[i];
    }
    
    // Extract oriented features from DoG
    std::vector<double> features(config_.numOrientations, 0.0);
    for (int orient = 0; orient < config_.numOrientations; ++orient) {
        features[orient] = computeOrientedResponse(dog, regionSize, orient);
    }
    
    // Normalize features to [0, 1]
    features = normalizeFeatures(features);
    
    // Apply threshold
    features = applyThreshold(features);
    
    return features;
}

std::string DoGOperator::getName() const {
    return "DoGOperator";
}

} // namespace features
} // namespace snnfw


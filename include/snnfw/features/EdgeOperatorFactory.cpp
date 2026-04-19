#include "snnfw/features/EdgeOperator.h"
#include "snnfw/features/SobelOperator.h"
#include "snnfw/features/GaborOperator.h"
#include "snnfw/features/QuadratureGaborOperator.h"
#include "snnfw/features/OrientationEnergyOperator.h"
#include "snnfw/features/DoGOperator.h"
#include <stdexcept>
#include <algorithm>

namespace snnfw {
namespace features {

std::unique_ptr<EdgeOperator> EdgeOperatorFactory::create(
    const std::string& type,
    const EdgeOperator::Config& config) {
    
    // Convert type to lowercase for case-insensitive comparison
    std::string lowerType = type;
    std::transform(lowerType.begin(), lowerType.end(), lowerType.begin(), ::tolower);
    
    if (lowerType == "sobel") {
        return std::make_unique<SobelOperator>(config);
    } else if (lowerType == "gabor") {
        return std::make_unique<GaborOperator>(config);
    } else if (lowerType == "quadrature_gabor" || lowerType == "gabor_energy") {
        return std::make_unique<QuadratureGaborOperator>(config);
    } else if (lowerType == "orientation_energy" || lowerType == "oriented_energy") {
        return std::make_unique<OrientationEnergyOperator>(config);
    } else if (lowerType == "dog" || lowerType == "difference_of_gaussians") {
        return std::make_unique<DoGOperator>(config);
    } else {
        throw std::invalid_argument("Unknown edge operator type: " + type);
    }
}

std::vector<std::string> EdgeOperatorFactory::getAvailableOperators() {
    return {
        "sobel",
        "gabor",
        "quadrature_gabor",
        "gabor_energy",
        "orientation_energy",
        "oriented_energy",
        "dog",
        "difference_of_gaussians"
    };
}

} // namespace features
} // namespace snnfw

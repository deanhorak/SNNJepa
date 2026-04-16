#ifndef SNNFW_QUADRATURE_GABOR_OPERATOR_H
#define SNNFW_QUADRATURE_GABOR_OPERATOR_H

#include "snnfw/features/EdgeOperator.h"

namespace snnfw {
namespace features {

/**
 * @brief Phase-invariant quadrature Gabor operator.
 *
 * Uses even/odd Gabor pairs and pools local energy as sqrt(even^2 + odd^2),
 * which is closer to a simple-cell energy model than the existing single-phase
 * Gabor response. The operator mean-centers the patch locally so the response
 * is driven by contrast structure rather than raw luminance bias.
 */
class QuadratureGaborOperator : public EdgeOperator {
public:
    explicit QuadratureGaborOperator(const Config& config);

    std::vector<double> extractEdges(
        const std::vector<uint8_t>& region,
        int regionSize) const override;

    std::string getName() const override;

private:
    double wavelength_;
    double sigma_;
    double gamma_;
    int kernelSize_;
    double energyGamma_;
    double gammaSq_;
    double twoSigmaSq_;

    double gaborKernel(double x, double y, double theta, double phase) const;
    double applyGaborFilter(const std::vector<uint8_t>& region,
                            int regionSize,
                            int centerR,
                            int centerC,
                            double theta,
                            double phase,
                            double regionMean) const;
    double computeQuadratureEnergy(const std::vector<uint8_t>& region,
                                   int regionSize,
                                   int orientation) const;
};

} // namespace features
} // namespace snnfw

#endif // SNNFW_QUADRATURE_GABOR_OPERATOR_H

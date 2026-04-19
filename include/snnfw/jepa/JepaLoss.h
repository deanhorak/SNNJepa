#ifndef SNNFW_JEPA_LOSS_H
#define SNNFW_JEPA_LOSS_H

#include <algorithm>
#include <cmath>
#include <vector>

namespace snnfw::jepa {

inline double cosineDistance(const std::vector<double>& lhs,
                             const std::vector<double>& rhs) {
    const size_t count = std::min(lhs.size(), rhs.size());
    if (count == 0) {
        return 1.0;
    }

    double dot = 0.0;
    double lhsNormSq = 0.0;
    double rhsNormSq = 0.0;
    for (size_t i = 0; i < count; ++i) {
        dot += lhs[i] * rhs[i];
        lhsNormSq += lhs[i] * lhs[i];
        rhsNormSq += rhs[i] * rhs[i];
    }

    const double lhsNorm = std::sqrt(lhsNormSq);
    const double rhsNorm = std::sqrt(rhsNormSq);
    if (lhsNorm <= 1e-12 || rhsNorm <= 1e-12) {
        return 1.0;
    }
    return 1.0 - (dot / (lhsNorm * rhsNorm));
}

inline double normalizedMse(const std::vector<double>& lhs,
                            const std::vector<double>& rhs) {
    const size_t count = std::min(lhs.size(), rhs.size());
    if (count == 0) {
        return 0.0;
    }

    double mse = 0.0;
    double scale = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double delta = lhs[i] - rhs[i];
        mse += delta * delta;
        scale += (lhs[i] * lhs[i]) + (rhs[i] * rhs[i]);
    }

    mse /= static_cast<double>(count);
    scale = std::max(scale / static_cast<double>(count), 1e-12);
    return mse / scale;
}

}  // namespace snnfw::jepa

#endif  // SNNFW_JEPA_LOSS_H

#ifndef SNNFW_JEPA_MASK_SAMPLER_H
#define SNNFW_JEPA_MASK_SAMPLER_H

#include "snnfw/jepa/JepaConfig.h"
#include "snnfw/jepa/JepaSample.h"
#include <vector>

namespace snnfw::jepa {

JepaMaskedView buildMaskedView(const JepaHemisphereView& view,
                               const JepaConfig& config,
                               size_t sampleSourceIndex,
                               size_t viewIndex);

bool validateMaskedView(const JepaMaskedView& maskedView,
                        std::vector<std::string>* errors = nullptr);

void attachMaskedViews(std::vector<JepaLatentSample>& samples,
                       const JepaConfig& config);

}  // namespace snnfw::jepa

#endif  // SNNFW_JEPA_MASK_SAMPLER_H

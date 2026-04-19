#ifndef SNNFW_JEPA_STATE_EXTRACTOR_H
#define SNNFW_JEPA_STATE_EXTRACTOR_H

#include "snnfw/jepa/JepaConfig.h"
#include "snnfw/jepa/JepaSample.h"
#include <string>
#include <vector>

namespace snnfw::jepa {

std::vector<JepaLatentSample> buildStage1LatentSamples(
    const std::vector<Stage1TapInput>& taps,
    const JepaConfig& config);

JepaExportSummary writeStage1LatentSamples(
    const std::vector<Stage1TapInput>& taps,
    const JepaConfig& config,
    const std::string& outputPath);

}  // namespace snnfw::jepa

#endif  // SNNFW_JEPA_STATE_EXTRACTOR_H

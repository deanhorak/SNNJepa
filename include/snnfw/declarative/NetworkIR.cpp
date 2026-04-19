#include "snnfw/declarative/NetworkIR.h"
#include <algorithm>
#include <cctype>

namespace snnfw {
namespace declarative {

bool NetworkIR::validate() const {
    return getValidationErrors().empty();
}

std::vector<std::string> NetworkIR::getValidationErrors() const {
    std::vector<std::string> errors;

    // Must have a brain with at least one hemisphere
    if (brain.hemispheres.empty()) {
        errors.push_back("Brain must have at least one hemisphere");
    }

    // Validate hierarchy depth
    for (const auto& hemi : brain.hemispheres) {
        if (hemi.lobes.empty()) {
            errors.push_back("Hemisphere '" + hemi.name + "' has no lobes");
        }
        for (const auto& lobe : hemi.lobes) {
            if (lobe.regions.empty()) {
                errors.push_back("Lobe '" + lobe.name + "' has no regions");
            }
            for (const auto& region : lobe.regions) {
                if (region.nuclei.empty()) {
                    errors.push_back("Region '" + region.name + "' has no nuclei");
                }
                for (const auto& nucleus : region.nuclei) {
                    // Must have either explicit columns or a column template
                    if (nucleus.columns.empty() && !nucleus.columnTemplate.has_value()) {
                        errors.push_back("Nucleus '" + nucleus.name +
                            "' has no columns and no column template");
                    }
                    // Validate column template
                    if (nucleus.columnTemplate.has_value()) {
                        const auto& tmpl = nucleus.columnTemplate.value();
                        if (tmpl.orientations.empty()) {
                            errors.push_back("Column template in nucleus '" +
                                nucleus.name + "' has no orientations");
                        }
                        if (tmpl.frequencies.empty()) {
                            errors.push_back("Column template in nucleus '" +
                                nucleus.name + "' has no frequencies");
                        }
                        if (tmpl.layers.empty()) {
                            errors.push_back("Column template in nucleus '" +
                                nucleus.name + "' has no layer definitions");
                        }
                    }
                    // Validate explicit columns
                    for (const auto& col : nucleus.columns) {
                        if (col.layers.empty()) {
                            errors.push_back("Column '" + col.name + "' has no layers");
                        }
                    }
                }
            }
        }
    }

    // Validate neuron param references in populations
    auto validatePopulations = [&](const std::vector<LayerIR>& layers) {
        for (const auto& layer : layers) {
            for (const auto& pop : layer.populations) {
                if (!pop.neuronParams.empty() &&
                    neuronParamSets.find(pop.neuronParams) == neuronParamSets.end()) {
                    errors.push_back("Population '" + pop.name +
                        "' references unknown neuron params '" + pop.neuronParams + "'");
                }
                if (pop.count <= 0) {
                    errors.push_back("Population '" + pop.name +
                        "' has invalid count: " + std::to_string(pop.count));
                }
            }
        }
    };

    // Check all column templates' layer populations
    for (const auto& hemi : brain.hemispheres) {
        for (const auto& lobe : hemi.lobes) {
            for (const auto& region : lobe.regions) {
                for (const auto& nucleus : region.nuclei) {
                    if (nucleus.columnTemplate.has_value()) {
                        validatePopulations(nucleus.columnTemplate.value().layers);
                    }
                    for (const auto& col : nucleus.columns) {
                        validatePopulations(col.layers);
                    }
                }
            }
        }
    }

    // Validate projections reference valid patterns
    for (const auto& proj : projections) {
        if (proj.source.empty()) {
            errors.push_back("Projection '" + proj.name + "' has empty source");
        }
        if (proj.target.empty()) {
            errors.push_back("Projection '" + proj.name + "' has empty target");
        }
        // Validate pattern type
        if (proj.pattern != "random_sparse" && proj.pattern != "all_to_all" &&
            proj.pattern != "one_to_one" && proj.pattern != "many_to_one" &&
            proj.pattern != "topographic" && proj.pattern != "distance_dependent" &&
            proj.pattern != "small_world" && proj.pattern != "explicit" &&
            proj.pattern != "tiled_receptive_field") {
            errors.push_back("Projection '" + proj.name +
                "' has unknown pattern type: " + proj.pattern);
        }
    }

    // Validate input layer
    if (inputLayer.rows <= 0 || inputLayer.cols <= 0) {
        errors.push_back("Input layer must have positive rows and cols");
    }

    // Validate output layer
    if (outputLayer.numClasses <= 0) {
        errors.push_back("Output layer must have positive numClasses");
    }
    if (outputLayer.neuronsPerClass <= 0) {
        errors.push_back("Output layer must have positive neuronsPerClass");
    }

    // Validate adapter declarations
    for (const auto& adapter : adapters) {
        if (adapter.name.empty()) {
            errors.push_back("Adapter has empty name");
        }
        if (adapter.type.empty()) {
            errors.push_back("Adapter '" + adapter.name + "' has empty type");
            continue;
        }
        if (adapter.type != "interneuron_rx" &&
            adapter.type != "interneuron_tx" &&
            adapter.type != "retina") {
            errors.push_back("Adapter '" + adapter.name + "' has unsupported type '" +
                             adapter.type + "'");
        }
        if (!adapter.role.empty() && adapter.role != "sensory" && adapter.role != "motor") {
            errors.push_back("Adapter '" + adapter.name + "' has invalid role '" +
                             adapter.role + "'");
        }
        if (!adapter.bindTo.empty() &&
            adapter.bindTo != "input" &&
            adapter.bindTo != "l4" &&
            adapter.bindTo != "output") {
            errors.push_back("Adapter '" + adapter.name + "' has invalid bind_to '" +
                             adapter.bindTo + "'");
        }
    }

    if (!classification.type.empty()) {
        auto normalizedType = classification.type;
        std::transform(normalizedType.begin(), normalizedType.end(), normalizedType.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        if (normalizedType != "majority" &&
            normalizedType != "majority_voting" &&
            normalizedType != "weighted_distance" &&
            normalizedType != "weighted_similarity" &&
            normalizedType != "hierarchical" &&
            normalizedType != "hierarchical_knn") {
            errors.push_back("Classification has unsupported type '" + classification.type + "'");
        }
        if (classification.k <= 0) {
            errors.push_back("Classification k must be positive");
        }
        if (classification.distanceExponent <= 0.0) {
            errors.push_back("Classification distance_exponent must be positive");
        }

        const auto coarseK = classification.intParams.find("coarse_k");
        if (coarseK != classification.intParams.end() && coarseK->second <= 0) {
            errors.push_back("Classification coarse_k must be positive");
        }
        const auto fineK = classification.intParams.find("fine_k");
        if (fineK != classification.intParams.end() && fineK->second <= 0) {
            errors.push_back("Classification fine_k must be positive");
        }
    }

    return errors;
}

} // namespace declarative
} // namespace snnfw

#include "snnfw/experiment/SupervisedTeacher.h"
#include "snnfw/declarative/NetworkConstructor.h"
#include <algorithm>

namespace snnfw {
namespace experiment {

namespace {
bool meetsStdpEligibility(const ExperimentConfig& config,
                          const std::shared_ptr<NetworkPropagator>& propagator,
                          uint64_t neuronId) {
    if (!config.enableStdpEligibilityGate || !propagator) {
        return true;
    }

    const auto stats = propagator->getNeuronStdpEligibility(neuronId);
    const uint64_t minUpdates = static_cast<uint64_t>(
        std::max(0, config.stdpEligibilityMinUpdates));
    const uint64_t minLtp = static_cast<uint64_t>(
        std::max(0, config.stdpEligibilityMinLtp));

    if (stats.totalUpdates < minUpdates || stats.ltpUpdates < minLtp) {
        return false;
    }
    return stats.score(config.stdpEligibilityLtdPenalty) >= config.stdpEligibilityThreshold;
}

double computeHybridCompetitionScore(size_t spikeCount,
                                     size_t maxSpikeCount,
                                     double bestSimilarity,
                                     double similarityWeight,
                                     bool enabled) {
    if (!enabled) {
        return static_cast<double>(spikeCount);
    }
    const double weight = std::clamp(similarityWeight, 0.0, 1.0);
    const double spikeDrive = static_cast<double>(spikeCount);
    const double similarity = std::clamp(bestSimilarity, 0.0, 1.0);
    const double similarityScale = std::max(1.0, static_cast<double>(maxSpikeCount));
    const double similarityDrive = similarity * similarityScale;
    return ((1.0 - weight) * spikeDrive) + (weight * similarityDrive);
}
} // namespace

SupervisedTeacher::SupervisedTeacher(const ExperimentConfig& config)
    : config_(config)
{
}

int SupervisedTeacher::teach(
    int classLabel,
    std::vector<declarative::ConstructedNetwork::ColumnGroup>& columns,
    const std::vector<bool>& l5WinnerGlobal,
    const std::vector<bool>& colHasL4,
    std::vector<std::vector<std::shared_ptr<Neuron>>>& outputPopulations,
    double baseTime,
    std::shared_ptr<NetworkPropagator> propagator,
    TeachStats* statsOut)
{
    int patternsLearned = 0;
    TeachStats localStats;

    if (config_.enableFullPropagation) {
        // Fire L5 winners in full propagation mode so downstream STDP can occur
        bool hasEligibleL5Winner = false;
        int eligibleL5WinnerCount = 0;
        int l5WinnerCandidateCount = 0;
        int colIdxSeq = 0;
        size_t l5Offset = 0;
        for (auto& col : columns) {
            auto l5It = col.layerNeurons.find("L5");
            if (l5It == col.layerNeurons.end()) {
                colIdxSeq++;
                continue;
            }
            auto& l5Neurons = l5It->second;

            if (colIdxSeq < static_cast<int>(colHasL4.size()) && !colHasL4[colIdxSeq]) {
                l5Offset += l5Neurons.size();
                colIdxSeq++;
                continue;
            }

            struct RankedNeuron {
                size_t idx = 0;
                size_t spikes = 0;
                double similarity = 0.0;
                double activation = 0.0;
                size_t patternCount = 0;
                double score = 0.0;
            };
            std::vector<RankedNeuron> ranked;
            ranked.reserve(l5Neurons.size());
            size_t maxSpikes = 0;
            for (size_t i = 0; i < l5Neurons.size(); ++i) {
                RankedNeuron candidate;
                candidate.idx = i;
                candidate.spikes = l5Neurons[i]->getSpikes().size();
                candidate.similarity = std::max(0.0, l5Neurons[i]->getBestSimilarity());
                candidate.activation =
                    std::max(0.0, l5Neurons[i]->getSimilarityActivation());
                candidate.patternCount = l5Neurons[i]->getLearnedPatternCount();
                ranked.push_back(candidate);
                maxSpikes = std::max(maxSpikes, candidate.spikes);
            }
            for (auto& candidate : ranked) {
                candidate.score = computeHybridCompetitionScore(
                    candidate.spikes,
                    maxSpikes,
                    candidate.similarity,
                    config_.l5SimilarityWeight,
                    config_.enableSimilarityCompetition);
            }
            std::sort(ranked.begin(), ranked.end(),
                      [](const RankedNeuron& a, const RankedNeuron& b) {
                          if (a.score != b.score) return a.score > b.score;
                          if (a.activation != b.activation) return a.activation > b.activation;
                          if (a.spikes != b.spikes) return a.spikes > b.spikes;
                          if (a.similarity != b.similarity) return a.similarity > b.similarity;
                          return a.idx < b.idx;
                      });

            const bool enforceStrictGate = false;
            const double minSimilarity = 0.0;
            const double minScoreMargin = 0.0;
            const auto hasQualifiedTopTwo = [&]() {
                int qualified = 0;
                double first = 0.0;
                double second = 0.0;
                for (const auto& candidate : ranked) {
                    if (candidate.spikes < static_cast<size_t>(config_.l5MinSpikes)) continue;
                    const bool hasPatterns = candidate.patternCount > 0;
                    if (enforceStrictGate && hasPatterns && candidate.similarity < minSimilarity) {
                        continue;
                    }
                    if (qualified == 0) {
                        first = candidate.score;
                        qualified = 1;
                    } else {
                        second = candidate.score;
                        qualified = 2;
                        break;
                    }
                }
                if (qualified == 0) return false;
                if (minScoreMargin <= 0.0 || qualified == 1) return true;
                return (first - second) >= minScoreMargin;
            };
            const bool columnPassesMarginGate = hasQualifiedTopTwo();
            std::vector<double> similarityByIdx(l5Neurons.size(), 0.0);
            std::vector<size_t> patternCountByIdx(l5Neurons.size(), 0);
            for (const auto& candidate : ranked) {
                if (candidate.idx < similarityByIdx.size()) {
                    similarityByIdx[candidate.idx] = candidate.similarity;
                    patternCountByIdx[candidate.idx] = candidate.patternCount;
                }
            }

            int localIdx = 0;
            for (auto& l5Neuron : l5Neurons) {
                size_t globalIdx = l5Offset + localIdx;
                if (globalIdx < l5WinnerGlobal.size() && l5WinnerGlobal[globalIdx] &&
                    l5Neuron->getInhibition() <= config_.l5InhibitThreshold) {
                    localStats.l5WinnerCandidates++;
                    l5WinnerCandidateCount++;
                    double l5FireTime = baseTime + 15.0 + (colIdxSeq * 0.1) + (localIdx * 0.02);
                    l5Neuron->fireSignature(l5FireTime);
                    propagator->fireNeuron(l5Neuron->getId(), l5FireTime);
                    l5Neuron->fireAndAcknowledge(l5FireTime);
                    const bool eligible = meetsStdpEligibility(config_, propagator, l5Neuron->getId());
                    hasEligibleL5Winner = hasEligibleL5Winner || eligible;
                    if (eligible) {
                        localStats.l5WinnerEligible++;
                        eligibleL5WinnerCount++;
                        bool allowL5PatternUpdate = true;
                        if (enforceStrictGate) {
                            allowL5PatternUpdate = columnPassesMarginGate;
                            if (allowL5PatternUpdate) {
                                const size_t idx = static_cast<size_t>(localIdx);
                                const bool hasPatterns =
                                    idx < patternCountByIdx.size() && patternCountByIdx[idx] > 0;
                                const double sim =
                                    (idx < similarityByIdx.size()) ? similarityByIdx[idx] : 0.0;
                                if (hasPatterns && sim < minSimilarity) {
                                    allowL5PatternUpdate = false;
                                }
                            }
                        }
                        if (allowL5PatternUpdate) {
                            l5Neuron->learnCurrentPattern();
                            localStats.l5PatternsLearned++;
                        }
                    }
                }
                localIdx++;
            }
            l5Offset += l5Neurons.size();
            colIdxSeq++;
        }

        // Supervised teaching signal: force the correct output population
        if (!config_.disableOutputTeach &&
            classLabel >= 0 && classLabel < static_cast<int>(outputPopulations.size())) {
            double teachTime = baseTime + 30.0;
            for (auto& outputNeuron : outputPopulations[classLabel]) {
                localStats.outputCandidates++;
                outputNeuron->fireSignature(teachTime);
                propagator->fireNeuron(outputNeuron->getId(), teachTime);
                outputNeuron->fireAndAcknowledge(teachTime);
                bool outputEligible = meetsStdpEligibility(config_, propagator, outputNeuron->getId());
                if (config_.enableStdpEligibilityGate && !outputEligible && hasEligibleL5Winner) {
                    // When output STDP traces lag, allow supervised write if upstream winners
                    // for this image already satisfied STDP eligibility.
                    const size_t learnedPatterns = outputNeuron->getLearnedPatternCount();
                    const size_t bootstrapThreshold = static_cast<size_t>(
                        std::max(0, config_.outputFallbackBootstrapPatterns));
                    const bool bootstrapBypass =
                        (bootstrapThreshold > 0 && learnedPatterns < bootstrapThreshold);
                    const int requiredEligible = std::max(1, config_.outputFallbackMinEligibleL5);
                    const double requiredFrac = std::clamp(
                        config_.outputFallbackMinEligibleL5Fraction, 0.0, 1.0);
                    const double frac = (l5WinnerCandidateCount > 0)
                        ? (static_cast<double>(eligibleL5WinnerCount) /
                           static_cast<double>(l5WinnerCandidateCount))
                        : 0.0;
                    if (bootstrapBypass ||
                        (eligibleL5WinnerCount >= requiredEligible && frac >= requiredFrac)) {
                        outputEligible = true;
                        localStats.outputEligibilityFallbacks++;
                    }
                }
                if (outputEligible) {
                    localStats.outputEligible++;
                    outputNeuron->learnCurrentPattern();
                    patternsLearned++;
                    localStats.outputPatternsLearned++;
                }
            }
        }
    }

    if (statsOut) {
        *statsOut = localStats;
    }
    return patternsLearned;
}

} // namespace experiment
} // namespace snnfw

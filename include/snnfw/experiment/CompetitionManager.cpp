#include "snnfw/experiment/CompetitionManager.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_set>

namespace snnfw {
namespace experiment {

namespace {
constexpr double kOrientationSigmaDeg = 30.0;
constexpr double kFrequencySigmaOctaves = 0.75;
}

CompetitionManager::CompetitionManager(const ExperimentConfig& config)
    : config_(config)
    , gen_(config.seed)
    , l4PostJitterDist_(0.0, config.l4PostJitterMs)
{
}

void CompetitionManager::setSeed(unsigned int seed) {
    gen_.seed(seed);
}

double CompetitionManager::computeOverlapRatio(
    const std::vector<int>& a,
    const std::vector<int>& b)
{
    if (a.empty() || b.empty()) return 0.0;
    const std::vector<int>* smaller = &a;
    const std::vector<int>* larger = &b;
    if (smaller->size() > larger->size()) std::swap(smaller, larger);
    std::unordered_set<int> largerSet;
    largerSet.reserve(larger->size());
    for (int idx : *larger) largerSet.insert(idx);
    size_t intersection = 0;
    for (int idx : *smaller) {
        if (largerSet.find(idx) != largerSet.end()) intersection++;
    }
    if (intersection == 0) return 0.0;
    return static_cast<double>(intersection) /
           static_cast<double>(std::min(a.size(), b.size()));
}

double CompetitionManager::computeFeatureGate(
    const declarative::ConstructedNetwork::ColumnGroup& target,
    const declarative::ConstructedNetwork::ColumnGroup& source,
    double maxOrientationDeltaDeg,
    double maxFrequencyOctaveDelta)
{
    const double oriDeltaRaw = std::fabs(target.orientation - source.orientation);
    const double oriDelta = std::min(oriDeltaRaw, 180.0 - oriDeltaRaw);
    if (oriDelta > maxOrientationDeltaDeg) {
        return 0.0;
    }
    const double oriWeight =
        std::exp(-(oriDelta * oriDelta) / (2.0 * kOrientationSigmaDeg * kOrientationSigmaDeg));

    double freqWeight = 1.0;
    if (target.spatialFrequency > 0.0 && source.spatialFrequency > 0.0) {
        const double octaveDelta =
            std::fabs(std::log2(target.spatialFrequency / source.spatialFrequency));
        if (octaveDelta > maxFrequencyOctaveDelta) {
            return 0.0;
        }
        freqWeight = std::exp(
            -(octaveDelta * octaveDelta) / (2.0 * kFrequencySigmaOctaves * kFrequencySigmaOctaves));
    }

    return oriWeight * freqWeight;
}

double CompetitionManager::computeHybridCompetitionScore(
    size_t spikeCount,
    size_t maxSpikeCount,
    double bestSimilarity,
    double similarityWeight)
{
    const double weight = std::clamp(similarityWeight, 0.0, 1.0);
    const double spikeDrive = static_cast<double>(spikeCount);
    const double similarity = std::clamp(bestSimilarity, 0.0, 1.0);
    const double similarityScale = std::max(1.0, static_cast<double>(maxSpikeCount));
    const double similarityDrive = similarity * similarityScale;
    return ((1.0 - weight) * spikeDrive) + (weight * similarityDrive);
}

void CompetitionManager::rebuildInterColumnCache(
    const std::vector<declarative::ConstructedNetwork::ColumnGroup>& columns)
{
    interColumnIncoming_.assign(columns.size(), {});
    const int maxNeighbors = std::max(1, config_.l5InterColumnMaxNeighbors);
    for (size_t target = 0; target < columns.size(); ++target) {
        for (size_t source = 0; source < columns.size(); ++source) {
            if (target == source) continue;
            const double overlap = computeOverlapRatio(
                columns[target].inputMaskActiveIdx, columns[source].inputMaskActiveIdx);
            if (overlap < config_.l5InterColumnMinOverlap) continue;
            const double featureGate = computeFeatureGate(
                columns[target],
                columns[source],
                config_.l5InterColumnMaxOrientationDeltaDeg,
                config_.l5InterColumnMaxFrequencyOctaveDelta);
            const double weight = overlap * featureGate;
            if (weight <= 0.0) continue;
            interColumnIncoming_[target].push_back({source, weight});
        }
        auto& incoming = interColumnIncoming_[target];
        std::sort(incoming.begin(), incoming.end(),
                  [](const InterColumnEdge& a, const InterColumnEdge& b) {
                      return a.weight > b.weight;
                  });
        if (static_cast<int>(incoming.size()) > maxNeighbors) {
            incoming.resize(static_cast<size_t>(maxNeighbors));
        }
    }
    interColumnCacheColumns_ = columns.size();
    interColumnCacheValid_ = true;
}

std::vector<bool> CompetitionManager::runL4Competition(
    std::vector<declarative::ConstructedNetwork::ColumnGroup>& columns,
    const std::vector<bool>& inputFired,
    double baseTime,
    std::shared_ptr<NetworkPropagator> propagator)
{
    std::vector<bool> colHasL4(columns.size(), false);
    int colIdxSeq = 0;
    double callWinnerSpikeSum = 0.0;
    double callWinnerSimSum = 0.0;
    double callWinnerScoreSum = 0.0;
    double callPoolSpikeSum = 0.0;
    double callPoolSimSum = 0.0;
    double callPoolScoreSum = 0.0;
    int callWinnerCount = 0;
    int callPoolCount = 0;

    for (auto& col : columns) {
        auto l4It = col.layerNeurons.find("L4");
        if (l4It == col.layerNeurons.end()) {
            colIdxSeq++;
            continue;
        }
        auto& l4Neurons = l4It->second;

        // maskMinActive gating: skip column if insufficient input activity in its receptive field
        if (config_.maskMinActive > 0 && !col.inputMaskActiveIdx.empty()) {
            int maskedCount = 0;
            for (int idx : col.inputMaskActiveIdx) {
                if (idx >= 0 && idx < static_cast<int>(inputFired.size()) && inputFired[idx]) {
                    maskedCount++;
                    if (maskedCount >= config_.maskMinActive) break;
                }
            }
            if (maskedCount < config_.maskMinActive) {
                colIdxSeq++;
                continue;  // Skip this column entirely
            }
        }

        const int L4_KEEP = std::max(1, config_.l4Keep);
        struct RankedNeuron {
            double score = 0.0;
            size_t spikes = 0;
            double incoming = 0.0;
            double activation = 0.0;
            double similarity = 0.0;
            size_t idx = 0;
        };

        std::vector<RankedNeuron> ranked;
        ranked.reserve(l4Neurons.size());
        size_t maxSpikes = 0;
        for (size_t i = 0; i < l4Neurons.size(); ++i) {
            RankedNeuron candidate;
            candidate.idx = i;
            candidate.spikes = l4Neurons[i]->getSpikes().size();
            candidate.incoming = static_cast<double>(l4Neurons[i]->getIncomingSpikeCount());
            candidate.activation = std::max(0.0, l4Neurons[i]->getActivation());
            candidate.similarity = std::max(0.0, l4Neurons[i]->getBestSimilarity());
            ranked.push_back(candidate);
            maxSpikes = std::max(maxSpikes, candidate.spikes);
        }
        for (auto& candidate : ranked) {
            const double feedforwardDrive =
                static_cast<double>(candidate.spikes) +
                (config_.l4IncomingWeight * candidate.incoming) +
                (config_.l4ActivationWeight * candidate.activation);
            candidate.score = config_.enableSimilarityCompetition
                ? (((1.0 - std::clamp(config_.l4SimilarityWeight, 0.0, 1.0)) * feedforwardDrive) +
                   (std::clamp(config_.l4SimilarityWeight, 0.0, 1.0) *
                    (candidate.similarity * std::max(1.0, static_cast<double>(maxSpikes)))))
                : feedforwardDrive;
            if (feedforwardDrive > 0.0) {
                callPoolSpikeSum += feedforwardDrive;
                callPoolSimSum += candidate.similarity;
                callPoolScoreSum += candidate.score;
                callPoolCount++;
            }
        }
        std::sort(ranked.begin(), ranked.end(),
                  [](const RankedNeuron& a, const RankedNeuron& b) {
                      if (a.score != b.score) return a.score > b.score;
                      if (a.incoming != b.incoming) return a.incoming > b.incoming;
                      if (a.activation != b.activation) return a.activation > b.activation;
                      if (a.spikes != b.spikes) return a.spikes > b.spikes;
                      if (a.similarity != b.similarity) return a.similarity > b.similarity;
                      return a.idx < b.idx;
                  });

        int winners = 0;
        for (size_t i = 0; i < ranked.size() && winners < L4_KEEP; ++i) {
            if (ranked[i].score <= 0.0) break;
            int localIdx = static_cast<int>(ranked[i].idx);
            int l4Row = localIdx / config_.layer4Size;
            int l4Col = localIdx % config_.layer4Size;
            double spatialDelay = (l4Row * config_.l4RowDelay) + (l4Col * config_.l4ColDelay);
            double l4Jitter = (config_.l4PostJitterMs > 0.0) ? l4PostJitterDist_(gen_) : 0.0;
            double l4Fire = baseTime + 10.0 + config_.l4PostShiftMs + (colIdxSeq * 0.5)
                            + spatialDelay + l4Jitter;
            if (l4Fire < baseTime) l4Fire = baseTime;
            auto& l4Neuron = l4Neurons[localIdx];
            l4Neuron->fireSignature(l4Fire);
            propagator->fireNeuron(l4Neuron->getId(), l4Fire);
            l4Neuron->fireAndAcknowledge(l4Fire);
            colHasL4[colIdxSeq] = true;
            callWinnerSpikeSum += static_cast<double>(ranked[i].spikes);
            callWinnerSimSum += ranked[i].similarity;
            callWinnerScoreSum += ranked[i].score;
            callWinnerCount++;
            winners++;
        }

        colIdxSeq++;
    }

    l4CompetitionCalls_++;
    if (config_.traceSimilarityCompetition && (l4CompetitionCalls_ % 200 == 0) && callPoolCount > 0) {
        const double winnerCountSafe = static_cast<double>(std::max(1, callWinnerCount));
        const double poolCountSafe = static_cast<double>(callPoolCount);
        std::cout << "[SIM][L4] call=" << l4CompetitionCalls_
                  << " winners=" << callWinnerCount
                  << " avgWinnerSpikes=" << (callWinnerSpikeSum / winnerCountSafe)
                  << " avgWinnerSim=" << (callWinnerSimSum / winnerCountSafe)
                  << " avgWinnerScore=" << (callWinnerScoreSum / winnerCountSafe)
                  << " avgPoolSpikes=" << (callPoolSpikeSum / poolCountSafe)
                  << " avgPoolSim=" << (callPoolSimSum / poolCountSafe)
                  << " avgPoolScore=" << (callPoolScoreSum / poolCountSafe)
                  << " simWeight=" << config_.l4SimilarityWeight
                  << " enabled=" << (config_.enableSimilarityCompetition ? "1" : "0")
                  << std::endl;
    }
    return colHasL4;
}

std::vector<bool> CompetitionManager::runL5Competition(
    std::vector<declarative::ConstructedNetwork::ColumnGroup>& columns,
    const std::vector<bool>& colHasL4,
    double baseTime,
    std::shared_ptr<NetworkPropagator> propagator,
    bool trainingPhase)
{
    (void)baseTime;
    (void)propagator;
    (void)trainingPhase;

    struct ColumnState {
        std::vector<bool> localWinners;
        double winnerDrive = 0.0;
        bool active = false;
    };

    // Count total L5 neurons
    size_t totalL5 = 0;
    for (auto& col : columns) {
        auto it = col.layerNeurons.find("L5");
        if (it != col.layerNeurons.end()) totalL5 += it->second.size();
    }

    const int l5Keep = std::max(1, config_.l5Keep);
    std::vector<bool> l5WinnerGlobal(totalL5, false);
    std::vector<ColumnState> perColumn(columns.size());
    int colIdxSeq = 0;
    size_t l5Offset = 0;
    double callWinnerSpikeSum = 0.0;
    double callWinnerSimSum = 0.0;
    double callWinnerScoreSum = 0.0;
    double callPoolSpikeSum = 0.0;
    double callPoolSimSum = 0.0;
    double callPoolScoreSum = 0.0;
    int callWinnerCount = 0;
    int callPoolCount = 0;

    for (auto& col : columns) {
        auto it = col.layerNeurons.find("L5");
        if (it == col.layerNeurons.end()) { colIdxSeq++; continue; }
        auto& l5Neurons = it->second;

        for (auto& n : l5Neurons) n->resetInhibition();

        if (colIdxSeq < static_cast<int>(colHasL4.size()) && !colHasL4[colIdxSeq]) {
            l5Offset += l5Neurons.size();
            colIdxSeq++;
            continue;
        }
        perColumn[colIdxSeq].active = true;

        struct RankedNeuron {
            double score = 0.0;
            size_t spikes = 0;
            double incoming = 0.0;
            double similarity = 0.0;
            double activation = 0.0;
            size_t patternCount = 0;
            bool passesInferenceGate = true;
            size_t idx = 0;
        };

        std::vector<RankedNeuron> ranked;
        ranked.reserve(l5Neurons.size());
        size_t maxSpikes = 0;
        size_t eligibleForStrictGate = 0;
        size_t maxPatternCount = 0;
        for (size_t i = 0; i < l5Neurons.size(); ++i) {
            RankedNeuron candidate;
            candidate.idx = i;
            candidate.spikes = l5Neurons[i]->getSpikes().size();
            candidate.incoming = static_cast<double>(l5Neurons[i]->getIncomingSpikeCount());
            candidate.similarity = std::max(0.0, l5Neurons[i]->getBestSimilarity());
            candidate.activation = std::max(0.0, l5Neurons[i]->getActivation());
            candidate.patternCount = l5Neurons[i]->getLearnedPatternCount();
            ranked.push_back(candidate);
            maxSpikes = std::max(maxSpikes, candidate.spikes);
            maxPatternCount = std::max(maxPatternCount, candidate.patternCount);
        }
        const size_t bootstrapPatternThreshold = static_cast<size_t>(
            std::max(0, config_.l5WinnerGateBootstrapPatterns));
        const bool gateReady =
            !trainingPhase &&
            config_.enableL5InferenceSimilarityGate &&
            ((bootstrapPatternThreshold == 0) || (maxPatternCount >= bootstrapPatternThreshold));
        const double minSimilarity = gateReady
            ? std::max(0.0, config_.l5WinnerMinSimilarity)
            : 0.0;
        const double minScoreMargin = gateReady
            ? std::max(0.0, config_.l5WinnerMinScoreMargin)
            : 0.0;
        for (auto& candidate : ranked) {
            const double feedforwardDrive =
                static_cast<double>(candidate.spikes) +
                (config_.l5IncomingWeight * candidate.incoming) +
                (config_.l5ActivationWeight * candidate.activation);
            candidate.passesInferenceGate =
                !gateReady ||
                candidate.patternCount == 0 ||
                candidate.similarity >= minSimilarity;
            if (feedforwardDrive >= static_cast<double>(config_.l5MinSpikes) &&
                candidate.passesInferenceGate) {
                ++eligibleForStrictGate;
            }
        }
        const bool enforceStrictGate = gateReady &&
            eligibleForStrictGate >= static_cast<size_t>(std::max(1, l5Keep));
        for (auto& candidate : ranked) {
            const double feedforwardDrive =
                static_cast<double>(candidate.spikes) +
                (config_.l5IncomingWeight * candidate.incoming) +
                (config_.l5ActivationWeight * candidate.activation);
            const double baseScore = config_.enableSimilarityCompetition
                ? (((1.0 - std::clamp(config_.l5SimilarityWeight, 0.0, 1.0)) * feedforwardDrive) +
                   (std::clamp(config_.l5SimilarityWeight, 0.0, 1.0) *
                    (candidate.similarity * std::max(1.0, static_cast<double>(maxSpikes)))))
                : feedforwardDrive;
            if (candidate.patternCount > 0) {
                candidate.score = baseScore;
                if (!trainingPhase && config_.l5InferenceSimilarityBias > 0.0) {
                    candidate.score += config_.l5InferenceSimilarityBias * candidate.activation;
                }
                if (gateReady && !candidate.passesInferenceGate) {
                    const double similarityScale = (minSimilarity > 0.0)
                        ? std::clamp(candidate.similarity / minSimilarity, 0.15, 1.0)
                        : 1.0;
                    candidate.score *= similarityScale;
                }
            } else {
                candidate.score = baseScore;
            }
            if (feedforwardDrive > 0.0) {
                callPoolSpikeSum += feedforwardDrive;
                callPoolSimSum += candidate.similarity;
                callPoolScoreSum += candidate.score;
                callPoolCount++;
            }
        }
        std::sort(ranked.begin(), ranked.end(),
                      [](const RankedNeuron& a, const RankedNeuron& b) {
                      if (a.score != b.score) return a.score > b.score;
                      if (a.incoming != b.incoming) return a.incoming > b.incoming;
                      if (a.activation != b.activation) return a.activation > b.activation;
                      if (a.passesInferenceGate != b.passesInferenceGate) {
                          return a.passesInferenceGate > b.passesInferenceGate;
                      }
                      if (a.spikes != b.spikes) return a.spikes > b.spikes;
                      if (a.similarity != b.similarity) return a.similarity > b.similarity;
                      return a.idx < b.idx;
                  });

        const auto hasQualifiedTopTwo = [&]() {
            int qualified = 0;
            double first = 0.0;
            double second = 0.0;
            for (const auto& candidate : ranked) {
                const double feedforwardDrive =
                    static_cast<double>(candidate.spikes) +
                    (config_.l5IncomingWeight * candidate.incoming) +
                    (config_.l5ActivationWeight * candidate.activation);
                if (feedforwardDrive < static_cast<double>(config_.l5MinSpikes)) continue;
                const bool hasPatterns = candidate.patternCount > 0;
                if (enforceStrictGate && hasPatterns && candidate.similarity < minSimilarity) continue;
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

        int winners = 0;
        std::vector<bool> l5WinnerLocal(l5Neurons.size(), false);
        double winnerDrive = 0.0;
        for (size_t i = 0; i < ranked.size() && winners < l5Keep && columnPassesMarginGate; ++i) {
            if (enforceStrictGate && !ranked[i].passesInferenceGate) continue;
            const double feedforwardDrive =
                static_cast<double>(ranked[i].spikes) +
                (config_.l5IncomingWeight * ranked[i].incoming) +
                (config_.l5ActivationWeight * ranked[i].activation);
            if (feedforwardDrive < static_cast<double>(config_.l5MinSpikes)) break;
            const bool hasPatterns = ranked[i].patternCount > 0;
            if (enforceStrictGate && hasPatterns && ranked[i].similarity < minSimilarity) continue;
            l5WinnerGlobal[l5Offset + ranked[i].idx] = true;
            l5WinnerLocal[ranked[i].idx] = true;
            winnerDrive += ranked[i].score;
            callWinnerSpikeSum += feedforwardDrive;
            callWinnerSimSum += ranked[i].similarity;
            callWinnerScoreSum += ranked[i].score;
            callWinnerCount++;
            winners++;
        }
        perColumn[colIdxSeq].localWinners = std::move(l5WinnerLocal);
        perColumn[colIdxSeq].winnerDrive = winnerDrive;

        if (config_.enableL5Inhibition) {
            for (size_t i = 0; i < l5Neurons.size(); ++i) {
                if (!perColumn[colIdxSeq].localWinners[i]) {
                    l5Neurons[i]->applyInhibition(config_.l5InhibitLoser);
                }
            }
        }
        l5Offset += l5Neurons.size();
        colIdxSeq++;
    }

    if (config_.enableL5InterColumnInhibition && config_.l5InterColumnInhibit > 0.0) {
        if (!interColumnCacheValid_ || interColumnCacheColumns_ != columns.size()) {
            rebuildInterColumnCache(columns);
        }
        for (size_t i = 0; i < columns.size(); ++i) {
            auto l5It = columns[i].layerNeurons.find("L5");
            if (l5It == columns[i].layerNeurons.end()) continue;
            auto& l5Neurons = l5It->second;
            if (l5Neurons.empty()) continue;
            if (i >= perColumn.size() || !perColumn[i].active) continue;
            if (perColumn[i].winnerDrive <= 0.0) continue;

            double crossDrive = 0.0;
            for (const auto& edge : interColumnIncoming_[i]) {
                if (edge.sourceColumn >= perColumn.size()) continue;
                if (!perColumn[edge.sourceColumn].active) continue;
                if (perColumn[edge.sourceColumn].winnerDrive <= 0.0) continue;
                crossDrive += edge.weight * perColumn[edge.sourceColumn].winnerDrive;
            }
            if (crossDrive <= 0.0) continue;

            double interColumnInhibition =
                std::min(config_.l5InterColumnMaxInhibit,
                         crossDrive * config_.l5InterColumnInhibit);
            if (interColumnInhibition <= 0.0) continue;

            for (size_t localIdx = 0; localIdx < l5Neurons.size(); ++localIdx) {
                double scale = 1.0;
                if (localIdx < perColumn[i].localWinners.size() &&
                    perColumn[i].localWinners[localIdx]) {
                    scale = config_.l5InterColumnWinnerScale;
                }
                if (scale > 0.0) {
                    l5Neurons[localIdx]->applyInhibition(interColumnInhibition * scale);
                }
            }
        }
    }

    l5CompetitionCalls_++;
    if (config_.traceSimilarityCompetition && (l5CompetitionCalls_ % 200 == 0) && callPoolCount > 0) {
        const double winnerCountSafe = static_cast<double>(std::max(1, callWinnerCount));
        const double poolCountSafe = static_cast<double>(callPoolCount);
        std::cout << "[SIM][L5] call=" << l5CompetitionCalls_
                  << " winners=" << callWinnerCount
                  << " avgWinnerSpikes=" << (callWinnerSpikeSum / winnerCountSafe)
                  << " avgWinnerSim=" << (callWinnerSimSum / winnerCountSafe)
                  << " avgWinnerScore=" << (callWinnerScoreSum / winnerCountSafe)
                  << " avgPoolSpikes=" << (callPoolSpikeSum / poolCountSafe)
                  << " avgPoolSim=" << (callPoolSimSum / poolCountSafe)
                  << " avgPoolScore=" << (callPoolScoreSum / poolCountSafe)
                  << " simWeight=" << config_.l5SimilarityWeight
                  << " enabled=" << (config_.enableSimilarityCompetition ? "1" : "0")
                  << std::endl;
    }

    return l5WinnerGlobal;
}

void CompetitionManager::applyOutputCompetition(std::vector<int>& counts) {
    if (!config_.enableOutputCompetition) return;
    if (config_.outputCompetitionKeep <= 0) return;
    int numClasses = static_cast<int>(counts.size());
    std::vector<std::pair<int, int>> ranked;
    ranked.reserve(numClasses);
    for (int i = 0; i < numClasses; ++i) {
        ranked.emplace_back(counts[i], i);
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    if (ranked.empty() || ranked.front().first < config_.outputCompetitionMinSpikes) return;
    int keep = std::min(config_.outputCompetitionKeep, numClasses);
    for (int i = keep; i < numClasses; ++i) {
        counts[ranked[i].second] = 0;
    }
}

} // namespace experiment
} // namespace snnfw

#include "snnfw/ConnectivityPattern.h"
#include "snnfw/Logger.h"
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <algorithm>

namespace snnfw {

// ============================================================================
// SpatialPosition
// ============================================================================

double SpatialPosition::distanceTo(const SpatialPosition& other) const {
    double dx = x - other.x;
    double dy = y - other.y;
    double dz = z - other.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// ============================================================================
// RandomSparsePattern
// ============================================================================

RandomSparsePattern::RandomSparsePattern(double probability, double weight, double delay)
    : probability_(probability), weight_(weight), delay_(delay) {
    if (probability_ < 0.0 || probability_ > 1.0) {
        throw std::invalid_argument("Probability must be between 0.0 and 1.0");
    }
}

std::vector<Connection> RandomSparsePattern::generateConnections(
    const std::vector<uint64_t>& sourceNeurons,
    const std::vector<uint64_t>& targetNeurons) {
    
    std::vector<Connection> connections;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (uint64_t sourceId : sourceNeurons) {
        for (uint64_t targetId : targetNeurons) {
            if (dist(rng_) < probability_) {
                connections.emplace_back(sourceId, targetId, weight_, delay_);
            }
        }
    }
    
    SNNFW_DEBUG("RandomSparsePattern: Generated {} connections from {} sources to {} targets (p={})",
                connections.size(), sourceNeurons.size(), targetNeurons.size(), probability_);
    
    return connections;
}

// ============================================================================
// AllToAllPattern
// ============================================================================

AllToAllPattern::AllToAllPattern(double weight, double delay)
    : weight_(weight), delay_(delay) {
}

std::vector<Connection> AllToAllPattern::generateConnections(
    const std::vector<uint64_t>& sourceNeurons,
    const std::vector<uint64_t>& targetNeurons) {
    
    std::vector<Connection> connections;
    connections.reserve(sourceNeurons.size() * targetNeurons.size());
    
    for (uint64_t sourceId : sourceNeurons) {
        for (uint64_t targetId : targetNeurons) {
            connections.emplace_back(sourceId, targetId, weight_, delay_);
        }
    }
    
    SNNFW_DEBUG("AllToAllPattern: Generated {} connections from {} sources to {} targets",
                connections.size(), sourceNeurons.size(), targetNeurons.size());
    
    return connections;
}

// ============================================================================
// OneToOnePattern
// ============================================================================

OneToOnePattern::OneToOnePattern(double weight, double delay)
    : weight_(weight), delay_(delay) {
}

std::vector<Connection> OneToOnePattern::generateConnections(
    const std::vector<uint64_t>& sourceNeurons,
    const std::vector<uint64_t>& targetNeurons) {
    
    if (sourceNeurons.size() != targetNeurons.size()) {
        throw std::invalid_argument(
            "OneToOnePattern requires equal number of source and target neurons");
    }
    
    std::vector<Connection> connections;
    connections.reserve(sourceNeurons.size());
    
    for (size_t i = 0; i < sourceNeurons.size(); ++i) {
        connections.emplace_back(sourceNeurons[i], targetNeurons[i], weight_, delay_);
    }
    
    SNNFW_DEBUG("OneToOnePattern: Generated {} connections", connections.size());
    
    return connections;
}

// ============================================================================
// ManyToOnePattern
// ============================================================================

ManyToOnePattern::ManyToOnePattern(double weight, double delay)
    : weight_(weight), delay_(delay) {
}

std::vector<Connection> ManyToOnePattern::generateConnections(
    const std::vector<uint64_t>& sourceNeurons,
    const std::vector<uint64_t>& targetNeurons) {
    
    std::vector<Connection> connections;
    connections.reserve(sourceNeurons.size() * targetNeurons.size());
    
    // Connect all sources to each target
    for (uint64_t targetId : targetNeurons) {
        for (uint64_t sourceId : sourceNeurons) {
            connections.emplace_back(sourceId, targetId, weight_, delay_);
        }
    }
    
    SNNFW_DEBUG("ManyToOnePattern: Generated {} connections from {} sources to {} targets",
                connections.size(), sourceNeurons.size(), targetNeurons.size());
    
    return connections;
}

// ============================================================================
// DistanceDependentPattern
// ============================================================================

DistanceDependentPattern::DistanceDependentPattern(double sigma, double weight, double delay)
    : sigma_(sigma), weight_(weight), delay_(delay) {
    if (sigma_ <= 0.0) {
        throw std::invalid_argument("Sigma must be positive");
    }
}

void DistanceDependentPattern::setPositions(
    const std::map<uint64_t, SpatialPosition>& positions) {
    positions_ = positions;
}

std::vector<Connection> DistanceDependentPattern::generateConnections(
    const std::vector<uint64_t>& sourceNeurons,
    const std::vector<uint64_t>& targetNeurons) {
    
    if (positions_.empty()) {
        throw std::runtime_error(
            "DistanceDependentPattern requires positions to be set via setPositions()");
    }
    
    std::vector<Connection> connections;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (uint64_t sourceId : sourceNeurons) {
        auto sourceIt = positions_.find(sourceId);
        if (sourceIt == positions_.end()) {
            SNNFW_WARN("Source neuron {} has no position, skipping", sourceId);
            continue;
        }
        
        for (uint64_t targetId : targetNeurons) {
            auto targetIt = positions_.find(targetId);
            if (targetIt == positions_.end()) {
                SNNFW_WARN("Target neuron {} has no position, skipping", targetId);
                continue;
            }
            
            // Calculate distance and connection probability
            double distance = sourceIt->second.distanceTo(targetIt->second);
            double probability = std::exp(-distance * distance / (2.0 * sigma_ * sigma_));
            
            if (dist(rng_) < probability) {
                connections.emplace_back(sourceId, targetId, weight_, delay_);
            }
        }
    }
    
    SNNFW_DEBUG("DistanceDependentPattern: Generated {} connections from {} sources to {} targets (sigma={})",
                connections.size(), sourceNeurons.size(), targetNeurons.size(), sigma_);
    
    return connections;
}

// ============================================================================
// TopographicPattern
// ============================================================================

TopographicPattern::TopographicPattern(double sigma, double weight, double delay)
    : sigma_(sigma), weight_(weight), delay_(delay) {
    if (sigma_ <= 0.0) {
        throw std::invalid_argument("Sigma must be positive");
    }
}

std::vector<Connection> TopographicPattern::generateConnections(
    const std::vector<uint64_t>& sourceNeurons,
    const std::vector<uint64_t>& targetNeurons) {
    
    std::vector<Connection> connections;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    size_t nSource = sourceNeurons.size();
    size_t nTarget = targetNeurons.size();
    
    if (nSource == 0 || nTarget == 0) {
        return connections;
    }
    
    // Map neurons to normalized positions [0, 1] based on their index
    for (size_t i = 0; i < nSource; ++i) {
        double sourcePos = static_cast<double>(i) / static_cast<double>(nSource - 1);
        
        for (size_t j = 0; j < nTarget; ++j) {
            double targetPos = static_cast<double>(j) / static_cast<double>(nTarget - 1);
            
            // Connection probability based on position similarity
            double posDiff = std::abs(sourcePos - targetPos);
            double probability = std::exp(-posDiff * posDiff / (2.0 * sigma_ * sigma_));
            
            if (dist(rng_) < probability) {
                connections.emplace_back(sourceNeurons[i], targetNeurons[j], weight_, delay_);
            }
        }
    }
    
    SNNFW_DEBUG("TopographicPattern: Generated {} connections from {} sources to {} targets (sigma={})",
                connections.size(), sourceNeurons.size(), targetNeurons.size(), sigma_);
    
    return connections;
}

// ============================================================================
// SmallWorldPattern
// ============================================================================

SmallWorldPattern::SmallWorldPattern(double localProbability,
                                   double longRangeProbability,
                                   double localRadius,
                                   double weight,
                                   double delay)
    : localProbability_(localProbability),
      longRangeProbability_(longRangeProbability),
      localRadius_(localRadius),
      weight_(weight),
      delay_(delay) {
    
    if (localProbability_ < 0.0 || localProbability_ > 1.0) {
        throw std::invalid_argument("Local probability must be between 0.0 and 1.0");
    }
    if (longRangeProbability_ < 0.0 || longRangeProbability_ > 1.0) {
        throw std::invalid_argument("Long-range probability must be between 0.0 and 1.0");
    }
    if (localRadius_ <= 0.0) {
        throw std::invalid_argument("Local radius must be positive");
    }
}

void SmallWorldPattern::setPositions(
    const std::map<uint64_t, SpatialPosition>& positions) {
    positions_ = positions;
}

std::vector<Connection> SmallWorldPattern::generateConnections(
    const std::vector<uint64_t>& sourceNeurons,
    const std::vector<uint64_t>& targetNeurons) {
    
    if (positions_.empty()) {
        throw std::runtime_error(
            "SmallWorldPattern requires positions to be set via setPositions()");
    }
    
    std::vector<Connection> connections;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (uint64_t sourceId : sourceNeurons) {
        auto sourceIt = positions_.find(sourceId);
        if (sourceIt == positions_.end()) {
            SNNFW_WARN("Source neuron {} has no position, skipping", sourceId);
            continue;
        }
        
        for (uint64_t targetId : targetNeurons) {
            auto targetIt = positions_.find(targetId);
            if (targetIt == positions_.end()) {
                SNNFW_WARN("Target neuron {} has no position, skipping", targetId);
                continue;
            }
            
            // Calculate distance
            double distance = sourceIt->second.distanceTo(targetIt->second);
            
            // Determine if this is a local or long-range connection
            bool isLocal = (distance <= localRadius_);
            double probability = isLocal ? localProbability_ : longRangeProbability_;
            
            if (dist(rng_) < probability) {
                connections.emplace_back(sourceId, targetId, weight_, delay_);
            }
        }
    }
    
    SNNFW_DEBUG("SmallWorldPattern: Generated {} connections from {} sources to {} targets "
                "(local_p={}, long_p={}, radius={})",
                connections.size(), sourceNeurons.size(), targetNeurons.size(),
                localProbability_, longRangeProbability_, localRadius_);
    
    return connections;
}

// ============================================================================
// TiledReceptiveFieldPattern
// ============================================================================

TiledReceptiveFieldPattern::TiledReceptiveFieldPattern(
    int partitionInputSize, int inputRows, int inputCols,
    int tilesPerSide, int tilesPerColumn,
    int l4GridSize, int columnIndex, double weight, double delay)
    : partitionInputSize_(partitionInputSize), inputRows_(inputRows), inputCols_(inputCols),
      partitionCount_(1), partitionsHorizontal_(true), tilesPerSide_(tilesPerSide),
      tilesPerColumn_(tilesPerColumn), l4GridSize_(l4GridSize),
      columnIndex_(columnIndex), weight_(weight), delay_(delay) {
    if (tilesPerSide_ <= 0 || l4GridSize_ <= 0 || partitionInputSize_ <= 0 ||
        inputRows_ <= 0 || inputCols_ <= 0) {
        throw std::invalid_argument("Grid dimensions must be positive");
    }

    if (inputRows_ == partitionInputSize_ && (inputCols_ % partitionInputSize_) == 0) {
        partitionCount_ = std::max(1, inputCols_ / partitionInputSize_);
        partitionsHorizontal_ = true;
    } else if (inputCols_ == partitionInputSize_ && (inputRows_ % partitionInputSize_) == 0) {
        partitionCount_ = std::max(1, inputRows_ / partitionInputSize_);
        partitionsHorizontal_ = false;
    } else if (inputRows_ == partitionInputSize_ && inputCols_ == partitionInputSize_) {
        partitionCount_ = 1;
        partitionsHorizontal_ = true;
    } else {
        throw std::invalid_argument(
            "TiledReceptiveFieldPattern expects square input partitions tiled horizontally or vertically");
    }
    computeTileSelection();
}

void TiledReceptiveFieldPattern::computeTileSelection() {
    int totalTiles = tilesPerSide_ * tilesPerSide_;
    int numTiles = std::min(tilesPerColumn_, totalTiles);
    int tileSize = partitionInputSize_ / tilesPerSide_;

    // Deterministic per-column tile selection (matches original experiment)
    std::mt19937 tileGen(static_cast<uint32_t>(columnIndex_ * 9973 + 17));
    std::vector<int> allTileIndices(totalTiles);
    std::iota(allTileIndices.begin(), allTileIndices.end(), 0);
    std::shuffle(allTileIndices.begin(), allTileIndices.end(), tileGen);
    tileIndices_.assign(allTileIndices.begin(), allTileIndices.begin() + numTiles);

    // Build input mask for column gating using the first partition only.
    // In the 4-partition visual-front-end path this is the raw luminance partition,
    // which preserves the original maskMinActive semantics.
    std::vector<double> mask(static_cast<size_t>(inputRows_ * inputCols_), 0.0);
    inputMaskActiveIdx_.clear();
    const int maskPartition = 0;
    for (int tileIndex : tileIndices_) {
        int tileRow = tileIndex / tilesPerSide_;
        int tileCol = tileIndex % tilesPerSide_;
        int startR = tileRow * tileSize;
        int startC = tileCol * tileSize;
        for (int r = 0; r < tileSize; ++r) {
            for (int c = 0; c < tileSize; ++c) {
                const int idx = mapPartitionIndex(maskPartition, startR + r, startC + c);
                if (idx >= 0 &&
                    idx < static_cast<int>(mask.size()) &&
                    mask[static_cast<size_t>(idx)] == 0.0) {
                    mask[static_cast<size_t>(idx)] = 1.0;
                    inputMaskActiveIdx_.push_back(idx);
                }
            }
        }
    }
}

int TiledReceptiveFieldPattern::mapPartitionIndex(int partition, int localRow, int localCol) const {
    if (localRow < 0 || localRow >= partitionInputSize_ ||
        localCol < 0 || localCol >= partitionInputSize_) {
        return -1;
    }

    if (partitionsHorizontal_) {
        const int globalCol = (partition * partitionInputSize_) + localCol;
        return (localRow * inputCols_) + globalCol;
    }

    const int globalRow = (partition * partitionInputSize_) + localRow;
    return (globalRow * inputCols_) + localCol;
}

std::vector<Connection> TiledReceptiveFieldPattern::generateConnections(
    const std::vector<uint64_t>& sourceNeurons,
    const std::vector<uint64_t>& targetNeurons) {

    std::vector<Connection> connections;
    int tileSize = partitionInputSize_ / tilesPerSide_;
    int numTiles = static_cast<int>(tileIndices_.size());

    if (numTiles == 0 || sourceNeurons.empty() || targetNeurons.empty()) {
        return connections;
    }

    const double effectiveWeight = weight_ / static_cast<double>(std::max(1, partitionCount_));

    // Each target neuron (L4) connects to specific input pixels based on
    // its grid position and the column's tile selection, replicated across
    // every feature partition of the input grid.
    for (size_t j = 0; j < targetNeurons.size(); ++j) {
        int tileChoice = tileIndices_[static_cast<int>(j) % numTiles];
        int tileRow = tileChoice / tilesPerSide_;
        int tileCol = tileChoice % tilesPerSide_;
        int tileStartR = tileRow * tileSize;
        int tileStartC = tileCol * tileSize;
        int l4Row = static_cast<int>(j) / l4GridSize_;
        int l4Col = static_cast<int>(j) % l4GridSize_;
        int patchSize = std::max(1, tileSize / l4GridSize_);
        int startR = tileStartR + l4Row * patchSize;
        int startC = tileStartC + l4Col * patchSize;

        for (int pr = 0; pr < patchSize; ++pr) {
            for (int pc = 0; pc < patchSize; ++pc) {
                for (int partition = 0; partition < partitionCount_; ++partition) {
                    const int idx = mapPartitionIndex(partition, startR + pr, startC + pc);
                    if (idx >= 0 && idx < static_cast<int>(sourceNeurons.size())) {
                        connections.emplace_back(sourceNeurons[static_cast<size_t>(idx)],
                                                 targetNeurons[j], effectiveWeight, delay_);
                    }
                }
            }
        }
    }

    SNNFW_DEBUG("TiledReceptiveFieldPattern: Generated {} connections for column {} "
                "({} tiles, {}x{} L4 grid)",
                connections.size(), columnIndex_, numTiles, l4GridSize_, l4GridSize_);

    return connections;
}

} // namespace snnfw

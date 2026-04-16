/**
 * @file RetinaAdapter.cpp
 * @brief Implementation of RetinaAdapter for visual processing with pluggable strategies
 *
 * This file implements the RetinaAdapter class, which provides biologically-inspired
 * visual processing for spiking neural networks. The adapter uses:
 * - Spatial grid decomposition to divide images into receptive fields
 * - Pluggable edge detection operators (Sobel, Gabor, DoG)
 * - Pluggable spike encoding strategies (Rate, Temporal, Population)
 * - Orientation-selective neurons mimicking V1 simple cells
 *
 * Performance on MNIST:
 * - 8×8 grid + Sobel + Rate: 94.63% accuracy (current best)
 * - 7×7 grid + Sobel + Rate: 92.71% accuracy
 * - Higher spatial resolution (more regions) improves accuracy
 * - Sobel operator outperforms Gabor for sharp-edged digits
 *
 * Key Design Decisions:
 * - Region size calculated as imageSize / gridSize (integer division)
 * - For 28×28 MNIST: 8×8 grid → 3×3 pixel regions (optimal)
 * - Each region has numOrientations neurons (typically 8)
 * - Total neurons = gridSize² × numOrientations (e.g., 8×8×8 = 512)
 */

#include "snnfw/adapters/RetinaAdapter.h"
#include "snnfw/features/EdgeOperator.h"
#include "snnfw/features/SobelOperator.h"
#include "snnfw/features/GaborOperator.h"
#include "snnfw/features/DoGOperator.h"
#include "snnfw/encoding/EncodingStrategy.h"
#include "snnfw/encoding/RateEncoder.h"
#include "snnfw/encoding/TemporalEncoder.h"
#include "snnfw/encoding/PopulationEncoder.h"
#include "snnfw/Logger.h"
#include <array>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <numeric>
#include <sstream>

namespace snnfw {
namespace adapters {

namespace {

std::vector<double> makeGaussianKernel(double sigma) {
    if (sigma <= 0.0) {
        return {1.0};
    }

    const int radius = std::max(1, static_cast<int>(std::ceil(2.5 * sigma)));
    std::vector<double> kernel(static_cast<size_t>(2 * radius + 1), 0.0);
    double sum = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        const double x = static_cast<double>(i);
        const double value = std::exp(-(x * x) / (2.0 * sigma * sigma));
        kernel[static_cast<size_t>(i + radius)] = value;
        sum += value;
    }
    if (sum > 0.0) {
        for (double& value : kernel) {
            value /= sum;
        }
    }
    return kernel;
}

int clampIndex(int value, int minValue, int maxValue) {
    return std::max(minValue, std::min(value, maxValue));
}

struct TopologyMaps {
    int rows = 0;
    int cols = 0;
    std::vector<double> contour;
    std::vector<double> endpoints;
    std::vector<double> junctions;
    std::vector<double> holes;
    std::vector<double> gapTop;
    std::vector<double> gapRight;
    std::vector<double> gapBottom;
    std::vector<double> gapLeft;
};

struct ContourGraphMaps {
    int rows = 0;
    int cols = 0;
    std::vector<std::vector<double>> channels;
};

struct ContourSequenceMaps {
    int rows = 0;
    int cols = 0;
    std::vector<std::vector<double>> bins;
};

size_t linearIndex(int row, int col, int cols) {
    return static_cast<size_t>(row * cols + col);
}

bool hasOccupiedNeighbor(const std::vector<uint8_t>& occupied, int rows, int cols, int row, int col) {
    const int dr[4] = {-1, 1, 0, 0};
    const int dc[4] = {0, 0, -1, 1};
    for (int k = 0; k < 4; ++k) {
        const int nr = row + dr[k];
        const int nc = col + dc[k];
        if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) {
            continue;
        }
        if (occupied[linearIndex(nr, nc, cols)] != 0) {
            return true;
        }
    }
    return false;
}

template <typename StartFn>
std::vector<double> buildDirectionalGapMap(const std::vector<uint8_t>& occupied,
                                           int rows,
                                           int cols,
                                           StartFn&& enqueueStarts) {
    std::vector<double> gapMap(static_cast<size_t>(rows * cols), 0.0);
    std::vector<uint8_t> visited(static_cast<size_t>(rows * cols), 0);
    std::vector<std::pair<int, int>> queue;
    queue.reserve(static_cast<size_t>(rows * cols));

    auto pushOpen = [&](int row, int col) {
        const size_t idx = linearIndex(row, col, cols);
        if (occupied[idx] == 0 && visited[idx] == 0) {
            visited[idx] = 1;
            queue.push_back({row, col});
        }
    };

    enqueueStarts(pushOpen);

    const int dr[4] = {-1, 1, 0, 0};
    const int dc[4] = {0, 0, -1, 1};
    for (size_t qi = 0; qi < queue.size(); ++qi) {
        const auto [row, col] = queue[qi];
        if (hasOccupiedNeighbor(occupied, rows, cols, row, col)) {
            gapMap[linearIndex(row, col, cols)] = 1.0;
        }
        for (int k = 0; k < 4; ++k) {
            const int nr = row + dr[k];
            const int nc = col + dc[k];
            if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) {
                continue;
            }
            const size_t idx = linearIndex(nr, nc, cols);
            if (occupied[idx] == 0 && visited[idx] == 0) {
                visited[idx] = 1;
                queue.push_back({nr, nc});
            }
        }
    }

    return gapMap;
}

TopologyMaps computeTopologyMaps(const RetinaAdapter::Image& image, double pixelThreshold) {
    TopologyMaps maps;
    maps.rows = image.rows;
    maps.cols = image.cols;
    const size_t pixelCount = static_cast<size_t>(image.rows * image.cols);
    maps.contour.assign(pixelCount, 0.0);
    maps.endpoints.assign(pixelCount, 0.0);
    maps.junctions.assign(pixelCount, 0.0);
    maps.holes.assign(pixelCount, 0.0);

    std::vector<uint8_t> occupied(pixelCount, 0);
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            const size_t idx = linearIndex(row, col, image.cols);
            occupied[idx] = image.getNormalizedPixel(row, col) >= pixelThreshold ? 1 : 0;
        }
    }

    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            const size_t idx = linearIndex(row, col, image.cols);
            if (occupied[idx] == 0) {
                continue;
            }

            bool isContour = false;
            const int dr[4] = {-1, 1, 0, 0};
            const int dc[4] = {0, 0, -1, 1};
            for (int k = 0; k < 4; ++k) {
                const int nr = row + dr[k];
                const int nc = col + dc[k];
                if (nr < 0 || nr >= image.rows || nc < 0 || nc >= image.cols ||
                    occupied[linearIndex(nr, nc, image.cols)] == 0) {
                    isContour = true;
                    break;
                }
            }
            if (isContour) {
                maps.contour[idx] = 1.0;
            }
        }
    }

    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            const size_t idx = linearIndex(row, col, image.cols);
            if (maps.contour[idx] <= 0.0) {
                continue;
            }

            int degree = 0;
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    if (dr == 0 && dc == 0) {
                        continue;
                    }
                    const int nr = row + dr;
                    const int nc = col + dc;
                    if (nr < 0 || nr >= image.rows || nc < 0 || nc >= image.cols) {
                        continue;
                    }
                    if (maps.contour[linearIndex(nr, nc, image.cols)] > 0.0) {
                        degree++;
                    }
                }
            }

            if (degree <= 1) {
                maps.endpoints[idx] = 1.0;
            } else if (degree >= 3) {
                maps.junctions[idx] = 1.0;
            }
        }
    }

    std::vector<uint8_t> openVisited(pixelCount, 0);
    std::vector<std::pair<int, int>> queue;
    queue.reserve(pixelCount);
    auto pushOpen = [&](int row, int col) {
        const size_t idx = linearIndex(row, col, image.cols);
        if (occupied[idx] == 0 && openVisited[idx] == 0) {
            openVisited[idx] = 1;
            queue.push_back({row, col});
        }
    };

    for (int col = 0; col < image.cols; ++col) {
        pushOpen(0, col);
        pushOpen(image.rows - 1, col);
    }
    for (int row = 1; row < image.rows - 1; ++row) {
        pushOpen(row, 0);
        pushOpen(row, image.cols - 1);
    }

    const int dr4[4] = {-1, 1, 0, 0};
    const int dc4[4] = {0, 0, -1, 1};
    for (size_t qi = 0; qi < queue.size(); ++qi) {
        const auto [row, col] = queue[qi];
        for (int k = 0; k < 4; ++k) {
            const int nr = row + dr4[k];
            const int nc = col + dc4[k];
            if (nr < 0 || nr >= image.rows || nc < 0 || nc >= image.cols) {
                continue;
            }
            const size_t idx = linearIndex(nr, nc, image.cols);
            if (occupied[idx] == 0 && openVisited[idx] == 0) {
                openVisited[idx] = 1;
                queue.push_back({nr, nc});
            }
        }
    }

    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            const size_t idx = linearIndex(row, col, image.cols);
            if (occupied[idx] == 0 && openVisited[idx] == 0) {
                maps.holes[idx] = 1.0;
            }
        }
    }

    maps.gapTop = buildDirectionalGapMap(
        occupied, image.rows, image.cols,
        [&](auto&& pushFn) {
            for (int col = 0; col < image.cols; ++col) {
                pushFn(0, col);
            }
        });
    maps.gapRight = buildDirectionalGapMap(
        occupied, image.rows, image.cols,
        [&](auto&& pushFn) {
            for (int row = 0; row < image.rows; ++row) {
                pushFn(row, image.cols - 1);
            }
        });
    maps.gapBottom = buildDirectionalGapMap(
        occupied, image.rows, image.cols,
        [&](auto&& pushFn) {
            for (int col = 0; col < image.cols; ++col) {
                pushFn(image.rows - 1, col);
            }
        });
    maps.gapLeft = buildDirectionalGapMap(
        occupied, image.rows, image.cols,
        [&](auto&& pushFn) {
            for (int row = 0; row < image.rows; ++row) {
                pushFn(row, 0);
            }
        });

    return maps;
}

double poolMapRegionMean(const std::vector<double>& map,
                         int rows,
                         int cols,
                         int gridSize,
                         int regionRow,
                         int regionCol) {
    const int startRow = (regionRow * rows) / gridSize;
    const int endRow = ((regionRow + 1) * rows) / gridSize;
    const int startCol = (regionCol * cols) / gridSize;
    const int endCol = ((regionCol + 1) * cols) / gridSize;
    const int height = std::max(1, endRow - startRow);
    const int width = std::max(1, endCol - startCol);
    const int count = height * width;

    double sum = 0.0;
    for (int row = startRow; row < endRow; ++row) {
        for (int col = startCol; col < endCol; ++col) {
            sum += map[linearIndex(row, col, cols)];
        }
    }
    return count > 0 ? sum / static_cast<double>(count) : 0.0;
}

std::vector<double> poolTopologyFeatures(const TopologyMaps& maps,
                                         int gridSize,
                                         int regionRow,
                                         int regionCol,
                                         double gain) {
    std::vector<double> features(8, 0.0);
    features[0] = poolMapRegionMean(maps.contour, maps.rows, maps.cols, gridSize, regionRow, regionCol);
    features[1] = poolMapRegionMean(maps.endpoints, maps.rows, maps.cols, gridSize, regionRow, regionCol);
    features[2] = poolMapRegionMean(maps.junctions, maps.rows, maps.cols, gridSize, regionRow, regionCol);
    features[3] = poolMapRegionMean(maps.holes, maps.rows, maps.cols, gridSize, regionRow, regionCol);
    features[4] = poolMapRegionMean(maps.gapTop, maps.rows, maps.cols, gridSize, regionRow, regionCol);
    features[5] = poolMapRegionMean(maps.gapRight, maps.rows, maps.cols, gridSize, regionRow, regionCol);
    features[6] = poolMapRegionMean(maps.gapBottom, maps.rows, maps.cols, gridSize, regionRow, regionCol);
    features[7] = poolMapRegionMean(maps.gapLeft, maps.rows, maps.cols, gridSize, regionRow, regionCol);

    for (double& feature : features) {
        feature = std::clamp(feature * gain, 0.0, 1.0);
    }
    return features;
}

ContourGraphMaps computeContourGraphMaps(const RetinaAdapter::Image& image, double pixelThreshold) {
    ContourGraphMaps maps;
    maps.rows = image.rows;
    maps.cols = image.cols;
    const size_t pixelCount = static_cast<size_t>(image.rows * image.cols);
    maps.channels.assign(8, std::vector<double>(pixelCount, 0.0));

    std::vector<uint8_t> occupied(pixelCount, 0);
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            occupied[linearIndex(row, col, image.cols)] =
                image.getNormalizedPixel(row, col) >= pixelThreshold ? 1 : 0;
        }
    }

    std::vector<uint8_t> contour(pixelCount, 0);
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            const size_t idx = linearIndex(row, col, image.cols);
            if (occupied[idx] == 0) {
                continue;
            }
            bool boundary = false;
            const int dr4[4] = {-1, 1, 0, 0};
            const int dc4[4] = {0, 0, -1, 1};
            for (int k = 0; k < 4; ++k) {
                const int nr = row + dr4[k];
                const int nc = col + dc4[k];
                if (nr < 0 || nr >= image.rows || nc < 0 || nc >= image.cols ||
                    occupied[linearIndex(nr, nc, image.cols)] == 0) {
                    boundary = true;
                    break;
                }
            }
            contour[idx] = boundary ? 1 : 0;
        }
    }

    std::vector<uint8_t> visited(pixelCount, 0);
    const int dr8[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const int dc8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            const size_t seedIdx = linearIndex(row, col, image.cols);
            if (contour[seedIdx] == 0 || visited[seedIdx] != 0) {
                continue;
            }

            std::vector<std::pair<int, int>> component;
            component.reserve(32);
            std::vector<std::pair<int, int>> queue;
            queue.push_back({row, col});
            visited[seedIdx] = 1;
            for (size_t qi = 0; qi < queue.size(); ++qi) {
                const auto [cr, cc] = queue[qi];
                component.push_back({cr, cc});
                for (int k = 0; k < 8; ++k) {
                    const int nr = cr + dr8[k];
                    const int nc = cc + dc8[k];
                    if (nr < 0 || nr >= image.rows || nc < 0 || nc >= image.cols) {
                        continue;
                    }
                    const size_t nidx = linearIndex(nr, nc, image.cols);
                    if (contour[nidx] != 0 && visited[nidx] == 0) {
                        visited[nidx] = 1;
                        queue.push_back({nr, nc});
                    }
                }
            }

            int endpointCount = 0;
            int junctionCount = 0;
            int minRow = image.rows;
            int maxRow = 0;
            int minCol = image.cols;
            int maxCol = 0;
            double horizontalEdges = 0.0;
            double verticalEdges = 0.0;
            double diagonalEdges = 0.0;
            double endpointTop = 0.0;
            double endpointBottom = 0.0;
            double endpointLeft = 0.0;
            double endpointRight = 0.0;

            for (const auto& [cr, cc] : component) {
                minRow = std::min(minRow, cr);
                maxRow = std::max(maxRow, cr);
                minCol = std::min(minCol, cc);
                maxCol = std::max(maxCol, cc);

                int degree = 0;
                for (int k = 0; k < 8; ++k) {
                    const int nr = cr + dr8[k];
                    const int nc = cc + dc8[k];
                    if (nr < 0 || nr >= image.rows || nc < 0 || nc >= image.cols) {
                        continue;
                    }
                    const size_t nidx = linearIndex(nr, nc, image.cols);
                    if (contour[nidx] == 0) {
                        continue;
                    }
                    degree++;
                    if (nr > cr || (nr == cr && nc > cc)) {
                        const int adr = std::abs(nr - cr);
                        const int adc = std::abs(nc - cc);
                        if (adr == 1 && adc == 1) {
                            diagonalEdges += 1.0;
                        } else if (adr == 1) {
                            verticalEdges += 1.0;
                        } else if (adc == 1) {
                            horizontalEdges += 1.0;
                        }
                    }
                }

                if (degree <= 1) {
                    endpointCount++;
                    endpointTop += 1.0 - static_cast<double>(cr) /
                                             static_cast<double>(std::max(1, image.rows - 1));
                    endpointBottom += static_cast<double>(cr) /
                                      static_cast<double>(std::max(1, image.rows - 1));
                    endpointLeft += 1.0 - static_cast<double>(cc) /
                                              static_cast<double>(std::max(1, image.cols - 1));
                    endpointRight += static_cast<double>(cc) /
                                       static_cast<double>(std::max(1, image.cols - 1));
                } else if (degree >= 3) {
                    junctionCount++;
                }
            }

            const double componentSize = static_cast<double>(component.size());
            const double edgeTotal =
                std::max(1.0, horizontalEdges + verticalEdges + diagonalEdges);
            const double lengthNorm = std::clamp(componentSize / 24.0, 0.0, 1.0);
            const double closedScore =
                endpointCount == 0 && componentSize >= 6.0 ? 1.0 : 0.0;
            const double endpointNorm = std::clamp(static_cast<double>(endpointCount) / 4.0, 0.0, 1.0);
            const double junctionNorm = std::clamp(static_cast<double>(junctionCount) / 3.0, 0.0, 1.0);
            const double horizontalBias = horizontalEdges / edgeTotal;
            const double verticalBias = verticalEdges / edgeTotal;
            const double diagonalBias = diagonalEdges / edgeTotal;
            const double spanHeight =
                static_cast<double>(std::max(1, maxRow - minRow + 1));
            const double spanWidth =
                static_cast<double>(std::max(1, maxCol - minCol + 1));
            const double slenderness =
                std::clamp(std::abs(spanHeight - spanWidth) /
                               std::max(spanHeight, spanWidth),
                           0.0, 1.0);
            const double endpointDenom = std::max(1.0, static_cast<double>(endpointCount));
            const std::array<double, 8> descriptor = {
                lengthNorm,
                closedScore,
                endpointNorm,
                junctionNorm,
                horizontalBias,
                verticalBias,
                diagonalBias,
                closedScore > 0.0
                    ? 0.5 * (1.0 - slenderness)
                    : std::clamp((endpointTop + endpointBottom + endpointLeft + endpointRight) /
                                     (4.0 * endpointDenom),
                                 0.0, 1.0),
            };

            for (const auto& [cr, cc] : component) {
                const size_t idx = linearIndex(cr, cc, image.cols);
                for (size_t ch = 0; ch < descriptor.size(); ++ch) {
                    maps.channels[ch][idx] = std::max(maps.channels[ch][idx], descriptor[ch]);
                }
            }
        }
    }

    return maps;
}

std::vector<double> poolContourGraphFeatures(const ContourGraphMaps& maps,
                                             int gridSize,
                                             int regionRow,
                                             int regionCol,
                                             double gain) {
    std::vector<double> features(maps.channels.size(), 0.0);
    for (size_t ch = 0; ch < maps.channels.size(); ++ch) {
        features[ch] = poolMapRegionMean(
            maps.channels[ch], maps.rows, maps.cols, gridSize, regionRow, regionCol);
        features[ch] = std::clamp(features[ch] * gain, 0.0, 1.0);
    }
    return features;
}

ContourSequenceMaps computeContourSequenceMaps(const RetinaAdapter::Image& image,
                                               double pixelThreshold,
                                               int sequenceBins) {
    ContourSequenceMaps maps;
    maps.rows = image.rows;
    maps.cols = image.cols;
    const size_t pixelCount = static_cast<size_t>(image.rows * image.cols);
    const int binCount = std::max(1, sequenceBins);
    maps.bins.assign(static_cast<size_t>(binCount), std::vector<double>(pixelCount, 0.0));

    std::vector<uint8_t> occupied(pixelCount, 0);
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            occupied[linearIndex(row, col, image.cols)] =
                image.getNormalizedPixel(row, col) >= pixelThreshold ? 1 : 0;
        }
    }

    std::vector<uint8_t> contour(pixelCount, 0);
    const int dr8[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const int dc8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            const size_t idx = linearIndex(row, col, image.cols);
            if (occupied[idx] == 0) {
                continue;
            }
            bool boundary = false;
            for (int k = 0; k < 8; ++k) {
                const int nr = row + dr8[k];
                const int nc = col + dc8[k];
                if (nr < 0 || nr >= image.rows || nc < 0 || nc >= image.cols ||
                    occupied[linearIndex(nr, nc, image.cols)] == 0) {
                    boundary = true;
                    break;
                }
            }
            contour[idx] = boundary ? 1 : 0;
        }
    }

    std::vector<uint8_t> seen(pixelCount, 0);
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            const size_t seedIdx = linearIndex(row, col, image.cols);
            if (contour[seedIdx] == 0 || seen[seedIdx] != 0) {
                continue;
            }

            std::vector<std::pair<int, int>> component;
            std::vector<std::pair<int, int>> queue;
            queue.push_back({row, col});
            seen[seedIdx] = 1;
            for (size_t qi = 0; qi < queue.size(); ++qi) {
                const auto [cr, cc] = queue[qi];
                component.push_back({cr, cc});
                for (int k = 0; k < 8; ++k) {
                    const int nr = cr + dr8[k];
                    const int nc = cc + dc8[k];
                    if (nr < 0 || nr >= image.rows || nc < 0 || nc >= image.cols) {
                        continue;
                    }
                    const size_t nidx = linearIndex(nr, nc, image.cols);
                    if (contour[nidx] != 0 && seen[nidx] == 0) {
                        seen[nidx] = 1;
                        queue.push_back({nr, nc});
                    }
                }
            }

            if (component.size() < 4) {
                continue;
            }

            std::vector<std::vector<size_t>> adjacency(component.size());
            std::vector<int> degree(component.size(), 0);
            std::unordered_map<size_t, size_t> componentIndex;
            componentIndex.reserve(component.size());
            for (size_t i = 0; i < component.size(); ++i) {
                componentIndex.emplace(linearIndex(component[i].first, component[i].second, image.cols), i);
            }
            for (size_t i = 0; i < component.size(); ++i) {
                const auto [cr, cc] = component[i];
                for (int k = 0; k < 8; ++k) {
                    const int nr = cr + dr8[k];
                    const int nc = cc + dc8[k];
                    if (nr < 0 || nr >= image.rows || nc < 0 || nc >= image.cols) {
                        continue;
                    }
                    const auto it =
                        componentIndex.find(linearIndex(nr, nc, image.cols));
                    if (it != componentIndex.end()) {
                        adjacency[i].push_back(it->second);
                    }
                }
                std::sort(adjacency[i].begin(), adjacency[i].end());
                adjacency[i].erase(std::unique(adjacency[i].begin(), adjacency[i].end()),
                                   adjacency[i].end());
                degree[i] = static_cast<int>(adjacency[i].size());
            }

            std::vector<size_t> ordered;
            ordered.reserve(component.size());
            std::vector<uint8_t> pathVisited(component.size(), 0);
            auto endpointIt = std::find_if(
                component.begin(), component.end(), [&](const auto& point) {
                    const size_t idx = &point - component.data();
                    return degree[idx] <= 1;
                });

            if (endpointIt != component.end()) {
                size_t current = static_cast<size_t>(endpointIt - component.begin());
                size_t previous = current;
                while (true) {
                    if (!pathVisited[current]) {
                        pathVisited[current] = 1;
                        ordered.push_back(current);
                    }
                    size_t next = current;
                    bool foundNext = false;
                    for (size_t neighbor : adjacency[current]) {
                        if (neighbor != previous && !pathVisited[neighbor]) {
                            next = neighbor;
                            foundNext = true;
                            break;
                        }
                    }
                    if (!foundNext) {
                        for (size_t neighbor : adjacency[current]) {
                            if (!pathVisited[neighbor]) {
                                next = neighbor;
                                foundNext = true;
                                break;
                            }
                        }
                    }
                    if (!foundNext) {
                        break;
                    }
                    previous = current;
                    current = next;
                }
            }

            if (ordered.size() != component.size()) {
                double rowSum = 0.0;
                double colSum = 0.0;
                for (const auto& [cr, cc] : component) {
                    rowSum += cr;
                    colSum += cc;
                }
                const double centerRow = rowSum / static_cast<double>(component.size());
                const double centerCol = colSum / static_cast<double>(component.size());
                std::vector<std::pair<double, size_t>> byAngle;
                byAngle.reserve(component.size());
                for (size_t i = 0; i < component.size(); ++i) {
                    const double angle = std::atan2(component[i].first - centerRow,
                                                    component[i].second - centerCol);
                    byAngle.push_back({angle, i});
                }
                std::sort(byAngle.begin(), byAngle.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });
                ordered.clear();
                for (const auto& entry : byAngle) {
                    ordered.push_back(entry.second);
                }
            }

            const double sizeScale =
                std::clamp(static_cast<double>(component.size()) / 24.0, 0.0, 1.0);
            for (size_t orderIdx = 0; orderIdx < ordered.size(); ++orderIdx) {
                const auto& point = component[ordered[orderIdx]];
                const size_t pixelIdx = linearIndex(point.first, point.second, image.cols);
                const int bin =
                    std::min(binCount - 1,
                             static_cast<int>((orderIdx * static_cast<size_t>(binCount)) /
                                              std::max<size_t>(1, ordered.size())));
                maps.bins[static_cast<size_t>(bin)][pixelIdx] = std::max(
                    maps.bins[static_cast<size_t>(bin)][pixelIdx], sizeScale);
            }
        }
    }

    return maps;
}

std::vector<double> poolContourSequenceFeatures(const ContourSequenceMaps& maps,
                                                int gridSize,
                                                int regionRow,
                                                int regionCol,
                                                double gain) {
    std::vector<double> features(maps.bins.size(), 0.0);
    for (size_t bin = 0; bin < maps.bins.size(); ++bin) {
        features[bin] = poolMapRegionMean(
            maps.bins[bin], maps.rows, maps.cols, gridSize, regionRow, regionCol);
        features[bin] = std::clamp(features[bin] * gain, 0.0, 1.0);
    }
    return features;
}

} // namespace

/**
 * @brief Construct a RetinaAdapter with configuration
 *
 * Initializes the adapter with pluggable edge detection and encoding strategies.
 * The constructor:
 * 1. Loads configuration parameters (grid size, orientations, thresholds)
 * 2. Creates the edge operator (Sobel, Gabor, or DoG) based on config
 * 3. Creates the encoding strategy (Rate, Temporal, or Population) based on config
 * 4. Prepares for neuron creation (done in initialize())
 *
 * @param config Configuration containing all adapter parameters
 */
RetinaAdapter::RetinaAdapter(const Config& config)
    : SensoryAdapter(config)
    , gridSize_(0)
    , regionSize_(0)
    , numOrientations_(0)
    , edgeThreshold_(0.15)
    , temporalWindow_(100.0)
    , edgeOperatorType_("sobel")
    , activationMode_("binary")
    , auxiliaryFeatureMode_("none")
    , colorEdgeMode_("none")
    , subfieldGridSize_(1)
    , subfieldIncludePooled_(true)
    , orientationFeatureGain_(1.0)
    , neuronWindowSize_(200.0)
    , neuronThreshold_(0.7)
    , neuronMaxPatterns_(100)
    , minimumRegionSize_(1)
    , edgeAnalysisRegionSize_(0)
    , maxFrequencyBandsPerFeature_(1)
    , frequencyBlurBaseSigma_(0.6)
    , orientationLateralInhibition_(0.0)
    , orientationResponseGamma_(1.0)
    , auxiliaryFeatureGain_(1.0)
    , auxiliaryAnalysisRegionSize_(0)
    , localContrastRadius_(0)
    , localContrastStrength_(1.0)
    , lgnRelayEnabled_(false)
    , lgnCenterSigma_(0.6)
    , lgnSurroundSigma_(1.4)
    , lgnCenterSurroundStrength_(0.0)
    , lgnBurstTonicEnabled_(false)
    , lgnBurstThreshold_(0.12)
    , lgnBurstExtraStrength_(0.18)
    , lgnBurstSlope_(6.0)
    , lgnBurstNeuromodulator_(1.0)
    , lgnParallelRelayEnabled_(false)
    , lgnMagnoCenterSigma_(0.45)
    , lgnMagnoSurroundSigma_(1.8)
    , lgnMagnoCenterSurroundStrength_(0.35)
    , lgnMagnoAchromaticMix_(1.0)
    , lgnMagnoExtraBlur_(0.35)
    , lgnParvoCenterSigma_(0.3)
    , lgnParvoSurroundSigma_(0.95)
    , lgnParvoCenterSurroundStrength_(0.12)
    , lgnParvoOriginalMix_(0.4)
    , lgnBandMagnoFloor_(0.1)
    , lgnAuxiliaryMagnoMix_(0.15)
    , eccentricitySamplingEnabled_(false)
    , eccentricitySamplingStrength_(0.0)
    , eccentricitySamplingGamma_(1.0)
    , retinalMaskMode_("full")
    , retinalMaskRadiusFraction_(0.30)
    , retinalMaskSoftnessFraction_(0.08)
    , retinalMaskCenterX_(0.5)
    , retinalMaskCenterY_(0.5)
    , temporalStreamBranchMode_("full")
    , temporalStreamFloor_(0.20)
    , temporalStreamDriveGain_(0.85)
    , temporalStreamOpponentSuppression_(0.20)
    , temporalStreamAuxiliaryFloor_(0.25)
    , luminanceBranchMode_("full")
    , luminanceBranchFloor_(0.20)
    , luminanceBranchDriveGain_(0.90)
    , luminanceBranchOpponentSuppression_(0.15)
    , luminanceBranchAuxiliaryFloor_(0.18)
    , temporalCoarseToFineEnabled_(false)
    , temporalCoarseBias_(0.0)
    , temporalTransientGain_(0.0)
    , temporalSustainedGain_(0.0)
    , temporalCrossBandGain_(0.0)
    , complexCellEnabled_(false)
    , complexCellPoolBlend_(0.0)
    , complexCellMaxMix_(0.5)
    , complexCellDivisiveGain_(0.0)
    , complexCellDivisiveFloor_(0.0)
    , complexCellNeighborMix_(0.5)
    , edgePatchNormalizationEnabled_(false)
    , edgePatchContrastStrength_(1.0)
    , edgePatchMinStd_(1.0)
    , cornerMinDeltaDeg_(45.0)
    , cornerMaxDeltaDeg_(110.0)
    , curveMinDeltaDeg_(10.0)
    , curveMaxDeltaDeg_(45.0)
    , endstopPixelThreshold_(0.30)
    , endstopAxisFraction_(0.35)
    , rotationDeg_(0.0)
    , scaleX_(1.0)
    , scaleY_(1.0)
    , shiftXPx_(0.0)
    , shiftYPx_(0.0)
    , mirrorX_(false)
    , mirrorY_(false)
    , orientationFlowDiagnosticsEnabled_(false)
    , orientationFlowDiagnosticsSampleLimit_(0)
    , homeostaticScalingEnabled_(false)
    , homeostaticLearningEnabled_(false)
    , homeostaticTargetActivation_(0.08)
    , homeostaticLearningRate_(0.01)
    , homeostaticActivityDecay_(0.97)
    , homeostaticGainMin_(0.75)
    , homeostaticGainMax_(1.25)
    , sensoryTripletBcmEnabled_(false)
    , sensoryTripletBcmLearningRate_(0.01)
    , sensoryTripletBcmLtp_(0.10)
    , sensoryTripletBcmLtd_(0.04)
    , sensoryTripletFastDecay_(0.80)
    , sensoryTripletSlowDecay_(0.97)
    , sensoryBcmThresholdDecay_(0.98)
    , sensoryBcmTargetActivation_(0.08)
    , sensoryTripletBcmGainMin_(0.85)
    , sensoryTripletBcmGainMax_(1.15)
    , imageRows_(0)
    , imageCols_(0)
    , imageChannels_(1)
{
    // Load configuration parameters
    gridSize_ = getIntParam("grid_size", 7);
    numOrientations_ = getIntParam("num_orientations", 8);
    edgeThreshold_ = getDoubleParam("edge_threshold", 0.15);
    temporalWindow_ = config.temporalWindow > 0 ? config.temporalWindow : 100.0;
    activationMode_ = getStringParam("activation_mode", "binary");
    edgeOperatorType_ = getStringParam("edge_operator", "sobel");
    auxiliaryFeatureMode_ = getStringParam("auxiliary_feature_mode", "none");
    colorEdgeMode_ = getStringParam("color_edge_mode", "none");
    subfieldGridSize_ = std::max(1, getIntParam("subfield_grid_size", 1));
    subfieldIncludePooled_ = getIntParam("subfield_include_pooled", 1) != 0;
    orientationFeatureGain_ = std::max(0.0, getDoubleParam("orientation_feature_gain", 1.0));

    neuronWindowSize_ = getDoubleParam("neuron_window_size", 200.0);
    neuronThreshold_ = getDoubleParam("neuron_threshold", 0.7);
    neuronMaxPatterns_ = getIntParam("neuron_max_patterns", 100);
    minimumRegionSize_ = getIntParam(
        "minimum_region_size",
        edgeOperatorType_ == "sobel" ? 3 : 1);
    edgeAnalysisRegionSize_ = std::max(0, getIntParam("edge_analysis_region_size", 0));
    maxFrequencyBandsPerFeature_ = std::max(1, getIntParam("max_frequency_bands_per_feature", 1));
    frequencyBlurBaseSigma_ = std::max(0.0, getDoubleParam("frequency_blur_base_sigma", 0.6));
    orientationLateralInhibition_ =
        std::clamp(getDoubleParam("orientation_lateral_inhibition", 0.0), 0.0, 1.0);
    orientationResponseGamma_ = std::max(0.1, getDoubleParam("orientation_response_gamma", 1.0));
    auxiliaryFeatureGain_ = std::max(0.0, getDoubleParam("auxiliary_feature_gain", 1.0));
    auxiliaryAnalysisRegionSize_ = std::max(0, getIntParam("auxiliary_analysis_region_size", 0));
    localContrastRadius_ = std::max(0, getIntParam("local_contrast_radius", 0));
    localContrastStrength_ = std::max(0.0, getDoubleParam("local_contrast_strength", 1.0));
    lgnRelayEnabled_ = getIntParam("lgn_relay_enabled", 0) != 0;
    lgnCenterSigma_ = std::max(0.0, getDoubleParam("lgn_center_sigma", 0.6));
    lgnSurroundSigma_ =
        std::max(lgnCenterSigma_ + 1e-3, getDoubleParam("lgn_surround_sigma", 1.4));
    lgnCenterSurroundStrength_ =
        std::max(0.0, getDoubleParam("lgn_center_surround_strength", 0.0));
    lgnBurstTonicEnabled_ = getIntParam("lgn_burst_tonic_enabled", 0) != 0;
    lgnBurstThreshold_ =
        std::clamp(getDoubleParam("lgn_burst_threshold", 0.12), 0.0, 1.0);
    lgnBurstExtraStrength_ =
        std::max(0.0, getDoubleParam("lgn_burst_extra_strength", 0.18));
    lgnBurstSlope_ = std::max(0.0, getDoubleParam("lgn_burst_slope", 6.0));
    lgnBurstNeuromodulator_ =
        std::clamp(getDoubleParam("lgn_burst_neuromodulator", 1.0), 0.0, 1.0);
    lgnParallelRelayEnabled_ = getIntParam("lgn_parallel_relay_enabled", 0) != 0;
    lgnMagnoCenterSigma_ =
        std::max(0.0, getDoubleParam("lgn_magno_center_sigma", 0.45));
    lgnMagnoSurroundSigma_ = std::max(
        lgnMagnoCenterSigma_ + 1e-3, getDoubleParam("lgn_magno_surround_sigma", 1.8));
    lgnMagnoCenterSurroundStrength_ =
        std::max(0.0, getDoubleParam("lgn_magno_center_surround_strength", 0.35));
    lgnMagnoAchromaticMix_ =
        std::clamp(getDoubleParam("lgn_magno_achromatic_mix", 1.0), 0.0, 1.0);
    lgnMagnoExtraBlur_ =
        std::max(0.0, getDoubleParam("lgn_magno_extra_blur", 0.35));
    lgnParvoCenterSigma_ =
        std::max(0.0, getDoubleParam("lgn_parvo_center_sigma", 0.30));
    lgnParvoSurroundSigma_ = std::max(
        lgnParvoCenterSigma_ + 1e-3, getDoubleParam("lgn_parvo_surround_sigma", 0.95));
    lgnParvoCenterSurroundStrength_ =
        std::max(0.0, getDoubleParam("lgn_parvo_center_surround_strength", 0.12));
    lgnParvoOriginalMix_ =
        std::clamp(getDoubleParam("lgn_parvo_original_mix", 0.40), 0.0, 1.0);
    lgnBandMagnoFloor_ =
        std::clamp(getDoubleParam("lgn_band_magno_floor", 0.10), 0.0, 1.0);
    lgnAuxiliaryMagnoMix_ =
        std::clamp(getDoubleParam("lgn_auxiliary_magno_mix", 0.15), 0.0, 1.0);
    eccentricitySamplingEnabled_ = getIntParam("eccentricity_sampling_enabled", 0) != 0;
    eccentricitySamplingStrength_ =
        std::clamp(getDoubleParam("eccentricity_sampling_strength", 0.0), 0.0, 1.0);
    eccentricitySamplingGamma_ =
        std::max(1.0, getDoubleParam("eccentricity_sampling_gamma", 1.0));
    retinalMaskMode_ = getStringParam("retinal_mask_mode", "full");
    retinalMaskRadiusFraction_ =
        std::clamp(getDoubleParam("retinal_mask_radius_fraction", 0.30), 0.0, 1.0);
    retinalMaskSoftnessFraction_ =
        std::clamp(getDoubleParam("retinal_mask_softness_fraction", 0.08), 0.0, 1.0);
    retinalMaskCenterX_ = std::clamp(getDoubleParam("retinal_mask_center_x", 0.5), 0.0, 1.0);
    retinalMaskCenterY_ = std::clamp(getDoubleParam("retinal_mask_center_y", 0.5), 0.0, 1.0);
    temporalStreamBranchMode_ = getStringParam("temporal_stream_branch_mode", "full");
    temporalStreamFloor_ = std::clamp(getDoubleParam("temporal_stream_floor", 0.20), 0.0, 1.0);
    temporalStreamDriveGain_ =
        std::max(0.0, getDoubleParam("temporal_stream_drive_gain", 0.85));
    temporalStreamOpponentSuppression_ =
        std::clamp(getDoubleParam("temporal_stream_opponent_suppression", 0.20), 0.0, 1.0);
    temporalStreamAuxiliaryFloor_ =
        std::clamp(getDoubleParam("temporal_stream_auxiliary_floor", 0.25), 0.0, 1.0);
    luminanceBranchMode_ = getStringParam("luminance_branch_mode", "full");
    luminanceBranchFloor_ = std::clamp(getDoubleParam("luminance_branch_floor", 0.20), 0.0, 1.0);
    luminanceBranchDriveGain_ =
        std::max(0.0, getDoubleParam("luminance_branch_drive_gain", 0.90));
    luminanceBranchOpponentSuppression_ =
        std::clamp(getDoubleParam("luminance_branch_opponent_suppression", 0.15), 0.0, 1.0);
    luminanceBranchAuxiliaryFloor_ =
        std::clamp(getDoubleParam("luminance_branch_auxiliary_floor", 0.18), 0.0, 1.0);
    temporalCoarseToFineEnabled_ = getIntParam("temporal_coarse_to_fine_enabled", 0) != 0;
    temporalCoarseBias_ = std::max(0.0, getDoubleParam("temporal_coarse_bias", 0.0));
    temporalTransientGain_ =
        std::max(0.0, getDoubleParam("temporal_transient_gain", 0.0));
    temporalSustainedGain_ =
        std::max(0.0, getDoubleParam("temporal_sustained_gain", 0.0));
    temporalCrossBandGain_ =
        std::max(0.0, getDoubleParam("temporal_cross_band_gain", 0.0));
    complexCellEnabled_ = getIntParam("complex_cell_enabled", 0) != 0;
    complexCellPoolBlend_ =
        std::clamp(getDoubleParam("complex_cell_pool_blend", 0.55), 0.0, 1.0);
    complexCellMaxMix_ =
        std::clamp(getDoubleParam("complex_cell_max_mix", 0.45), 0.0, 1.0);
    complexCellDivisiveGain_ =
        std::max(0.0, getDoubleParam("complex_cell_divisive_gain", 0.55));
    complexCellDivisiveFloor_ =
        std::clamp(getDoubleParam("complex_cell_divisive_floor", 0.08), 0.0, 1.0);
    complexCellNeighborMix_ =
        std::clamp(getDoubleParam("complex_cell_neighbor_mix", 0.65), 0.0, 1.0);
    edgePatchNormalizationEnabled_ = getIntParam("edge_patch_normalization", 0) != 0;
    edgePatchContrastStrength_ =
        std::max(0.0, getDoubleParam("edge_patch_contrast_strength", 1.0));
    edgePatchMinStd_ = std::max(1e-6, getDoubleParam("edge_patch_min_std", 1.0));
    cornerMinDeltaDeg_ = std::clamp(getDoubleParam("corner_min_delta_deg", 45.0), 0.0, 180.0);
    cornerMaxDeltaDeg_ = std::clamp(getDoubleParam("corner_max_delta_deg", 110.0), 0.0, 180.0);
    curveMinDeltaDeg_ = std::clamp(getDoubleParam("curve_min_delta_deg", 10.0), 0.0, 180.0);
    curveMaxDeltaDeg_ = std::clamp(getDoubleParam("curve_max_delta_deg", 45.0), 0.0, 180.0);
    endstopPixelThreshold_ = std::clamp(getDoubleParam("endstop_pixel_threshold", 0.30), 0.0, 1.0);
    endstopAxisFraction_ = std::clamp(getDoubleParam("endstop_axis_fraction", 0.35), 0.05, 0.49);
    rotationDeg_ = getDoubleParam("rotation_deg", 0.0);
    const double uniformScale = getDoubleParam("scale", 1.0);
    scaleX_ = std::max(1e-3, getDoubleParam("scale_x", uniformScale));
    scaleY_ = std::max(1e-3, getDoubleParam("scale_y", uniformScale));
    shiftXPx_ = getDoubleParam("shift_x_px", 0.0);
    shiftYPx_ = getDoubleParam("shift_y_px", 0.0);
    mirrorX_ = getIntParam("mirror_x", 0) != 0;
    mirrorY_ = getIntParam("mirror_y", 0) != 0;
    orientationFlowDiagnosticsEnabled_ = getIntParam("orientation_flow_diagnostics", 0) != 0;
    orientationFlowDiagnosticsSampleLimit_ =
        std::max(0, getIntParam("orientation_flow_sample_limit", 0));
    homeostaticScalingEnabled_ = getIntParam("homeostatic_scaling_enabled", 0) != 0;
    homeostaticLearningEnabled_ = homeostaticScalingEnabled_;
    homeostaticTargetActivation_ =
        std::clamp(getDoubleParam("homeostatic_target_activation", 0.08), 0.0, 1.0);
    homeostaticLearningRate_ = std::max(0.0, getDoubleParam("homeostatic_learning_rate", 0.01));
    homeostaticActivityDecay_ =
        std::clamp(getDoubleParam("homeostatic_activity_decay", 0.97), 0.0, 0.9999);
    homeostaticGainMin_ = std::max(0.05, getDoubleParam("homeostatic_gain_min", 0.75));
    homeostaticGainMax_ =
        std::max(homeostaticGainMin_, getDoubleParam("homeostatic_gain_max", 1.25));
    sensoryTripletBcmEnabled_ = getIntParam("sensory_triplet_bcm_enabled", 0) != 0;
    sensoryTripletBcmLearningRate_ =
        std::max(0.0, getDoubleParam("sensory_triplet_bcm_learning_rate", 0.01));
    sensoryTripletBcmLtp_ = std::max(0.0, getDoubleParam("sensory_triplet_bcm_ltp", 0.10));
    sensoryTripletBcmLtd_ = std::max(0.0, getDoubleParam("sensory_triplet_bcm_ltd", 0.04));
    sensoryTripletFastDecay_ =
        std::clamp(getDoubleParam("sensory_triplet_fast_decay", 0.80), 0.0, 0.9999);
    sensoryTripletSlowDecay_ =
        std::clamp(getDoubleParam("sensory_triplet_slow_decay", 0.97), 0.0, 0.9999);
    sensoryBcmThresholdDecay_ =
        std::clamp(getDoubleParam("sensory_bcm_threshold_decay", 0.98), 0.0, 0.9999);
    sensoryBcmTargetActivation_ =
        std::clamp(getDoubleParam("sensory_bcm_target_activation", 0.08), 0.0, 1.0);
    sensoryTripletBcmGainMin_ =
        std::max(0.05, getDoubleParam("sensory_triplet_bcm_gain_min", 0.85));
    sensoryTripletBcmGainMax_ = std::max(
        sensoryTripletBcmGainMin_, getDoubleParam("sensory_triplet_bcm_gain_max", 1.15));
    frequencyBands_ = parseFrequencyBands(getStringParam("frequency_values", ""));
    configureFrequencyBands();

    // Create edge operator
    features::EdgeOperator::Config edgeConfig;
    edgeConfig.name = edgeOperatorType_;
    edgeConfig.numOrientations = numOrientations_;
    edgeConfig.edgeThreshold = edgeThreshold_;

    // Copy edge operator parameters from config
    if (config.stringParams.count("edge_operator_params") > 0) {
        // Parse nested params if needed - for now use direct params
    }
    edgeConfig.doubleParams["wavelength"] = getDoubleParam("wavelength", 4.0);
    edgeConfig.doubleParams["sigma"] = getDoubleParam("sigma", 2.0);
    edgeConfig.doubleParams["gamma"] = getDoubleParam("gamma", 0.5);
    edgeConfig.doubleParams["phase_offset"] = getDoubleParam("phase_offset", 0.0);
    edgeConfig.doubleParams["sigma1"] = getDoubleParam("sigma1", 1.0);
    edgeConfig.doubleParams["sigma2"] = getDoubleParam("sigma2", 1.6);
    edgeConfig.doubleParams["quadrature_energy_gamma"] =
        getDoubleParam("quadrature_energy_gamma", 1.0);
    edgeConfig.doubleParams["orientation_energy_magnitude_gamma"] =
        getDoubleParam("orientation_energy_magnitude_gamma", 1.0);
    edgeConfig.doubleParams["orientation_energy_sharpness"] =
        getDoubleParam("orientation_energy_sharpness", 6.0);
    edgeConfig.doubleParams["orientation_energy_tensor_mix"] =
        getDoubleParam("orientation_energy_tensor_mix", 0.35);
    edgeConfig.doubleParams["orientation_energy_tensor_sharpness"] =
        getDoubleParam("orientation_energy_tensor_sharpness", 8.0);
    edgeConfig.doubleParams["orientation_energy_tensor_floor"] =
        getDoubleParam("orientation_energy_tensor_floor", 0.2);
    edgeConfig.intParams["kernel_size"] = getIntParam("kernel_size", 5);

    edgeOperator_ = features::EdgeOperatorFactory::create(edgeOperatorType_, edgeConfig);
    if (orientationFlowDiagnosticsEnabled_) {
        auto diagnosticConfig = edgeConfig;
        diagnosticConfig.edgeThreshold = 0.0;
        diagnosticEdgeOperator_ =
            features::EdgeOperatorFactory::create(edgeOperatorType_, diagnosticConfig);
    }

    // Create encoding strategy
    std::string encodingType = getStringParam("encoding_strategy", "rate");
    encoding::EncodingStrategy::Config encodingConfig;
    encodingConfig.name = encodingType;
    encodingConfig.temporalWindow = temporalWindow_;
    encodingConfig.baselineTime = 0.0;
    encodingConfig.intensityScale = temporalWindow_;

    // Copy encoding parameters
    encodingConfig.doubleParams["timing_jitter"] = getDoubleParam("timing_jitter", 0.0);
    encodingConfig.doubleParams["min_spike_interval"] = getDoubleParam("min_spike_interval", 5.0);
    encodingConfig.doubleParams["tuning_width"] = getDoubleParam("tuning_width", 0.3);
    encodingConfig.doubleParams["min_response"] = getDoubleParam("min_response", 0.1);
    encodingConfig.intParams["dual_spike_mode"] = getIntParam("dual_spike_mode", 0);
    encodingConfig.intParams["population_size"] = getIntParam("population_size", 5);

    encodingStrategy_ = encoding::EncodingStrategyFactory::create(encodingType, encodingConfig);

    SNNFW_INFO("RetinaAdapter '{}': grid={}x{}, orientations={}, threshold={}, edge={}, encoding={}",
               getName(), gridSize_, gridSize_, numOrientations_, edgeThreshold_,
               edgeOperatorType_, encodingType);
}

bool RetinaAdapter::initialize() {
    if (!SensoryAdapter::initialize()) {
        return false;
    }
    
    createNeurons();
    
    SNNFW_INFO("RetinaAdapter '{}': initialized with {} neurons", 
               getName(), neurons_.size());
    return true;
}

/**
 * @brief Create the neuron population for the retina
 *
 * Creates a 2D grid of orientation-selective neurons:
 * - Each spatial region has numOrientations_ neurons (one per orientation)
 * - Total neurons = gridSize² × numOrientations_
 * - Example: 8×8 grid with 8 orientations = 512 neurons
 *
 * Neuron organization:
 * - neuronGrid_[region][orientation] = neuron for that (region, orientation) pair
 * - neurons_ = flat list of all neurons for easy iteration
 *
 * Each neuron stores temporal spike patterns for pattern-based learning.
 */
void RetinaAdapter::createNeurons() {
    // Clear existing neurons completely before creating new ones
    neurons_.clear();
    neuronGrid_.clear();

    const int channelsPerRegion =
        static_cast<int>(getChannelsPerBand() * std::max<size_t>(1, getFrequencyBandCount()));
    int numRegions = gridSize_ * gridSize_;
    neuronGrid_.resize(numRegions);

    int neuronId = 0;
    for (int region = 0; region < numRegions; ++region) {
        neuronGrid_[region].resize(static_cast<size_t>(channelsPerRegion));
        for (int channel = 0; channel < channelsPerRegion; ++channel) {
            auto neuron = std::make_shared<Neuron>(
                neuronWindowSize_,      // Temporal window for pattern learning (ms)
                neuronThreshold_,       // Similarity threshold for pattern matching
                neuronMaxPatterns_,     // Maximum patterns to store per neuron
                neuronId++              // Unique neuron ID
            );
            neuronGrid_[region][channel] = neuron;
            neurons_.push_back(neuron);
        }
    }
}

std::vector<double> RetinaAdapter::parseFrequencyBands(const std::string& csv) const {
    std::vector<double> bands;
    if (csv.empty()) {
        return bands;
    }

    std::stringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            continue;
        }
        try {
            bands.push_back(std::stod(token));
        } catch (...) {
            // Ignore malformed frequency tokens and fall back to the remaining values.
        }
    }

    std::sort(bands.begin(), bands.end());
    bands.erase(std::unique(bands.begin(), bands.end(),
                            [](double a, double b) { return std::abs(a - b) < 1e-6; }),
                bands.end());
    return bands;
}

void RetinaAdapter::configureFrequencyBands() {
    blurSigmas_.clear();
    if (frequencyBands_.empty()) {
        blurSigmas_.push_back(0.0);
        return;
    }

    const double maxFrequency = *std::max_element(frequencyBands_.begin(), frequencyBands_.end());
    for (double frequency : frequencyBands_) {
        const double safeFrequency = std::max(1e-6, frequency);
        const double ratio = maxFrequency / safeFrequency;
        const double sigma = std::max(0.0, (ratio - 1.0) * frequencyBlurBaseSigma_);
        blurSigmas_.push_back(sigma);
    }
}

std::vector<int> RetinaAdapter::computeAxisSamplingBoundaries(int axisSize) const {
    std::vector<int> boundaries(static_cast<size_t>(std::max(0, gridSize_)) + 1, 0);
    if (boundaries.empty()) {
        return boundaries;
    }
    boundaries.front() = 0;
    boundaries.back() = std::max(0, axisSize);
    if (gridSize_ <= 0 || axisSize <= 0) {
        return boundaries;
    }

    if (!eccentricitySamplingEnabled_ ||
        eccentricitySamplingStrength_ <= 1e-6 ||
        std::abs(eccentricitySamplingGamma_ - 1.0) <= 1e-6) {
        for (int i = 1; i < gridSize_; ++i) {
            boundaries[static_cast<size_t>(i)] =
                (i * axisSize) / std::max(1, gridSize_);
        }
        return boundaries;
    }

    auto warpCoordinate = [&](double normalized) {
        normalized = std::clamp(normalized, 0.0, 1.0);
        const double centered = (2.0 * normalized) - 1.0;
        const double warpedCentered = std::copysign(
            std::pow(std::abs(centered), eccentricitySamplingGamma_), centered);
        const double warped = 0.5 * (1.0 + warpedCentered);
        return std::clamp((1.0 - eccentricitySamplingStrength_) * normalized +
                              eccentricitySamplingStrength_ * warped,
                          0.0,
                          1.0);
    };

    std::vector<double> desiredWidths(static_cast<size_t>(gridSize_), 1.0);
    double desiredWidthSum = 0.0;
    double previous = 0.0;
    for (int i = 1; i <= gridSize_; ++i) {
        const double warped = warpCoordinate(
            static_cast<double>(i) / static_cast<double>(gridSize_));
        const double width = std::max(1e-6, warped - previous);
        desiredWidths[static_cast<size_t>(i - 1)] = width;
        desiredWidthSum += width;
        previous = warped;
    }

    std::vector<int> widths(static_cast<size_t>(gridSize_), 0);
    int allocated = 0;
    if (axisSize >= gridSize_) {
        std::fill(widths.begin(), widths.end(), 1);
        allocated = gridSize_;
    }

    const int remaining = std::max(0, axisSize - allocated);
    std::vector<double> remainders(static_cast<size_t>(gridSize_), 0.0);
    int assignedExtra = 0;
    for (int i = 0; i < gridSize_; ++i) {
        const double exact =
            desiredWidthSum > 0.0
                ? (static_cast<double>(remaining) *
                   desiredWidths[static_cast<size_t>(i)] / desiredWidthSum)
                : 0.0;
        const int extra = static_cast<int>(std::floor(exact));
        widths[static_cast<size_t>(i)] += extra;
        assignedExtra += extra;
        remainders[static_cast<size_t>(i)] = exact - static_cast<double>(extra);
    }

    int leftover = remaining - assignedExtra;
    std::vector<int> order(static_cast<size_t>(gridSize_));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        return remainders[static_cast<size_t>(lhs)] > remainders[static_cast<size_t>(rhs)];
    });
    for (int i = 0; i < leftover && i < static_cast<int>(order.size()); ++i) {
        widths[static_cast<size_t>(order[static_cast<size_t>(i)])]++;
    }

    int cursor = 0;
    for (int i = 0; i < gridSize_; ++i) {
        boundaries[static_cast<size_t>(i)] = cursor;
        cursor += widths[static_cast<size_t>(i)];
    }
    boundaries.back() = axisSize;
    return boundaries;
}

double RetinaAdapter::computeRegionMaskWeight(int row, int col) const {
    if (retinalMaskMode_ != "foveal" && retinalMaskMode_ != "peripheral") {
        return 1.0;
    }

    const double x =
        (static_cast<double>(clampIndex(col, 0, std::max(0, gridSize_ - 1))) + 0.5) /
        static_cast<double>(std::max(1, gridSize_));
    const double y =
        (static_cast<double>(clampIndex(row, 0, std::max(0, gridSize_ - 1))) + 0.5) /
        static_cast<double>(std::max(1, gridSize_));
    const double dx = x - retinalMaskCenterX_;
    const double dy = y - retinalMaskCenterY_;
    const double distance = std::sqrt((dx * dx) + (dy * dy));

    const double radius = std::clamp(retinalMaskRadiusFraction_, 0.0, 1.0);
    const double softness = std::max(1e-6, retinalMaskSoftnessFraction_);
    const double inner = std::max(0.0, radius - softness);
    const double outer = std::min(1.5, radius + softness);

    double fovealWeight = 0.0;
    if (distance <= inner) {
        fovealWeight = 1.0;
    } else if (distance >= outer) {
        fovealWeight = 0.0;
    } else {
        const double t = std::clamp((distance - inner) / std::max(1e-6, outer - inner), 0.0, 1.0);
        const double smooth = t * t * (3.0 - (2.0 * t));
        fovealWeight = 1.0 - smooth;
    }

    if (retinalMaskMode_ == "peripheral") {
        return std::clamp(1.0 - fovealWeight, 0.0, 1.0);
    }
    return std::clamp(fovealWeight, 0.0, 1.0);
}

void RetinaAdapter::applyLuminanceOnOffBranchSplit(
    std::vector<std::vector<double>>& bandFeatures,
    std::vector<std::vector<double>>& auxiliaryFeatures) const {
    const bool onBranch = luminanceBranchMode_ == "on";
    const bool offBranch = luminanceBranchMode_ == "off";
    const bool luminanceBranch = luminanceBranchMode_ == "luminance";
    if ((!onBranch && !offBranch && !luminanceBranch) || bandFeatures.empty()) {
        return;
    }

    for (size_t bandIdx = 0; bandIdx < bandFeatures.size(); ++bandIdx) {
        const auto& auxiliary =
            bandIdx < auxiliaryFeatures.size() ? auxiliaryFeatures[bandIdx] : std::vector<double>{};
        if (auxiliary.size() < 7) {
            continue;
        }

        const double luminanceMean = auxiliary[0];
        const double luminanceContrast = auxiliary[1];
        const double luminanceGradient = auxiliary[2];
        const double sustainedOn = auxiliary[3];
        const double sustainedOff = auxiliary[4];
        const double transientOn = auxiliary[5];
        const double transientOff = auxiliary[6];

        const double onDrive = std::clamp(
            (0.50 * transientOn) + (0.30 * sustainedOn) + (0.20 * luminanceGradient), 0.0, 1.0);
        const double offDrive = std::clamp(
            (0.50 * transientOff) + (0.30 * sustainedOff) + (0.20 * luminanceGradient), 0.0, 1.0);
        const double achromaticDrive = std::clamp(
            (0.40 * luminanceGradient) + (0.35 * luminanceContrast) +
                (0.25 * std::abs((2.0 * luminanceMean) - 1.0)),
            0.0, 1.0);

        const double preferred =
            onBranch ? onDrive : (offBranch ? offDrive : achromaticDrive);
        const double opposing =
            onBranch ? offDrive
                     : (offBranch ? onDrive : (0.5 * (onDrive + offDrive)));
        const double scale = std::clamp(
            luminanceBranchFloor_ + (luminanceBranchDriveGain_ * preferred) -
                (luminanceBranchOpponentSuppression_ * opposing),
            0.0, 1.25);
        for (double& value : bandFeatures[bandIdx]) {
            value = std::clamp(value * scale, 0.0, 1.0);
        }

        if (bandIdx >= auxiliaryFeatures.size()) {
            continue;
        }

        auto& branchAuxiliary = auxiliaryFeatures[bandIdx];
        if (branchAuxiliary.size() < 7) {
            continue;
        }
        std::array<double, 7> weights{};
        weights.fill(luminanceBranchAuxiliaryFloor_);
        if (onBranch) {
            weights[0] = 0.55;
            weights[1] = 0.70;
            weights[2] = 0.95;
            weights[3] = 0.95;
            weights[4] = luminanceBranchAuxiliaryFloor_;
            weights[5] = 1.00;
            weights[6] = luminanceBranchAuxiliaryFloor_;
            const double preferredAuxWeight =
                std::clamp(luminanceBranchAuxiliaryFloor_ + (0.75 * preferred), 0.0, 1.0);
            const double opposingAuxWeight =
                std::clamp(luminanceBranchAuxiliaryFloor_ * (1.0 - (0.50 * opposing)), 0.0, 1.0);
            weights[3] = std::max(weights[3], preferredAuxWeight);
            weights[5] = std::max(weights[5], preferredAuxWeight);
            weights[4] = std::min(weights[4], opposingAuxWeight);
            weights[6] = std::min(weights[6], opposingAuxWeight);
        } else if (offBranch) {
            weights[0] = 0.55;
            weights[1] = 0.70;
            weights[2] = 0.95;
            weights[3] = luminanceBranchAuxiliaryFloor_;
            weights[4] = 0.95;
            weights[5] = luminanceBranchAuxiliaryFloor_;
            weights[6] = 1.00;
            const double preferredAuxWeight =
                std::clamp(luminanceBranchAuxiliaryFloor_ + (0.75 * preferred), 0.0, 1.0);
            const double opposingAuxWeight =
                std::clamp(luminanceBranchAuxiliaryFloor_ * (1.0 - (0.50 * opposing)), 0.0, 1.0);
            weights[4] = std::max(weights[4], preferredAuxWeight);
            weights[6] = std::max(weights[6], preferredAuxWeight);
            weights[3] = std::min(weights[3], opposingAuxWeight);
            weights[5] = std::min(weights[5], opposingAuxWeight);
        } else {
            weights[0] = 1.00;
            weights[1] = 1.00;
            weights[2] = 1.00;
            weights[3] = 0.35;
            weights[4] = 0.35;
            weights[5] = luminanceBranchAuxiliaryFloor_;
            weights[6] = luminanceBranchAuxiliaryFloor_;
            const double preferredAuxWeight =
                std::clamp(luminanceBranchAuxiliaryFloor_ + (0.80 * preferred), 0.0, 1.0);
            weights[0] = std::max(weights[0], preferredAuxWeight);
            weights[1] = std::max(weights[1], preferredAuxWeight);
            weights[2] = std::max(weights[2], preferredAuxWeight);
        }
        for (size_t idx = 0; idx < branchAuxiliary.size(); ++idx) {
            branchAuxiliary[idx] =
                std::clamp(branchAuxiliary[idx] * weights[idx], 0.0, 1.0);
        }
    }
}

void RetinaAdapter::applyTemporalStreamBranchSplit(
    std::vector<std::vector<double>>& bandFeatures,
    std::vector<std::vector<double>>& auxiliaryFeatures) const {
    const bool transientBranch = temporalStreamBranchMode_ == "transient";
    const bool sustainedBranch = temporalStreamBranchMode_ == "sustained";
    if ((!transientBranch && !sustainedBranch) || bandFeatures.empty()) {
        return;
    }

    for (size_t bandIdx = 0; bandIdx < bandFeatures.size(); ++bandIdx) {
        const auto& auxiliary =
            bandIdx < auxiliaryFeatures.size() ? auxiliaryFeatures[bandIdx] : std::vector<double>{};
        const double transient = estimateTransientDrive(auxiliary);
        const double sustained = estimateSustainedDrive(auxiliary);
        const double detail = estimateDetailDrive(auxiliary);
        const double preferred =
            transientBranch ? std::clamp((0.80 * transient) + (0.20 * detail), 0.0, 1.0)
                            : std::clamp((0.80 * sustained) + (0.20 * detail), 0.0, 1.0);
        const double opposing = transientBranch ? sustained : transient;
        const double scale = std::clamp(
            temporalStreamFloor_ + (temporalStreamDriveGain_ * preferred) -
                (temporalStreamOpponentSuppression_ * opposing),
            0.0, 1.25);
        for (double& value : bandFeatures[bandIdx]) {
            value = std::clamp(value * scale, 0.0, 1.0);
        }

        if (bandIdx >= auxiliaryFeatures.size()) {
            continue;
        }

        auto& branchAuxiliary = auxiliaryFeatures[bandIdx];
        if (auxiliaryFeatureMode_ == "appearance_stream_bank" && branchAuxiliary.size() >= 10) {
            std::array<double, 10> weights{};
            weights.fill(temporalStreamAuxiliaryFloor_);
            weights[0] = 0.55;
            weights[1] = 0.55;
            weights[2] = 0.55;
            weights[3] = sustainedBranch ? 0.85 : 0.60;
            weights[4] = 0.85;
            weights[5] = transientBranch ? 1.00 : 0.85;
            weights[6] = sustainedBranch ? 1.00 : temporalStreamAuxiliaryFloor_;
            weights[7] = sustainedBranch ? 1.00 : temporalStreamAuxiliaryFloor_;
            weights[8] = transientBranch ? 1.00 : temporalStreamAuxiliaryFloor_;
            weights[9] = transientBranch ? 1.00 : temporalStreamAuxiliaryFloor_;
            const double preferredAuxWeight =
                std::clamp(temporalStreamAuxiliaryFloor_ + (0.80 * preferred), 0.0, 1.0);
            const double opposingAuxWeight =
                std::clamp(temporalStreamAuxiliaryFloor_ * (1.0 - (0.50 * opposing)), 0.0, 1.0);
            if (transientBranch) {
                weights[8] = std::max(weights[8], preferredAuxWeight);
                weights[9] = std::max(weights[9], preferredAuxWeight);
                weights[6] = std::min(weights[6], opposingAuxWeight);
                weights[7] = std::min(weights[7], opposingAuxWeight);
            } else {
                weights[6] = std::max(weights[6], preferredAuxWeight);
                weights[7] = std::max(weights[7], preferredAuxWeight);
                weights[8] = std::min(weights[8], opposingAuxWeight);
                weights[9] = std::min(weights[9], opposingAuxWeight);
            }
            for (size_t idx = 0; idx < branchAuxiliary.size(); ++idx) {
                branchAuxiliary[idx] =
                    std::clamp(branchAuxiliary[idx] * weights[idx], 0.0, 1.0);
            }
        } else if (auxiliaryFeatureMode_ == "luminance_stream_bank" && branchAuxiliary.size() >= 7) {
            std::array<double, 7> weights{};
            weights.fill(temporalStreamAuxiliaryFloor_);
            weights[0] = sustainedBranch ? 0.85 : 0.60;
            weights[1] = 0.75;
            weights[2] = transientBranch ? 1.00 : 0.85;
            weights[3] = sustainedBranch ? 1.00 : temporalStreamAuxiliaryFloor_;
            weights[4] = sustainedBranch ? 1.00 : temporalStreamAuxiliaryFloor_;
            weights[5] = transientBranch ? 1.00 : temporalStreamAuxiliaryFloor_;
            weights[6] = transientBranch ? 1.00 : temporalStreamAuxiliaryFloor_;
            for (size_t idx = 0; idx < branchAuxiliary.size(); ++idx) {
                branchAuxiliary[idx] =
                    std::clamp(branchAuxiliary[idx] * weights[idx], 0.0, 1.0);
            }
        }
    }
}

double RetinaAdapter::estimateTransientDrive(const std::vector<double>& auxiliary) const {
    if (auxiliaryFeatureMode_ == "appearance_stream_bank" && auxiliary.size() >= 10) {
        return 0.5 * (auxiliary[8] + auxiliary[9]);
    }
    if (auxiliaryFeatureMode_ == "luminance_stream_bank" && auxiliary.size() >= 7) {
        return 0.5 * (auxiliary[5] + auxiliary[6]);
    }
    return 0.0;
}

double RetinaAdapter::estimateSustainedDrive(const std::vector<double>& auxiliary) const {
    if (auxiliaryFeatureMode_ == "appearance_stream_bank" && auxiliary.size() >= 10) {
        return 0.5 * (auxiliary[6] + auxiliary[7]);
    }
    if (auxiliaryFeatureMode_ == "luminance_stream_bank" && auxiliary.size() >= 7) {
        return 0.5 * (auxiliary[3] + auxiliary[4]);
    }
    return 0.0;
}

double RetinaAdapter::estimateDetailDrive(const std::vector<double>& auxiliary) const {
    if (auxiliaryFeatureMode_ == "appearance_stream_bank" && auxiliary.size() >= 10) {
        return std::clamp(0.65 * auxiliary[5] + 0.35 * auxiliary[4], 0.0, 1.0);
    }
    if (auxiliaryFeatureMode_ == "luminance_stream_bank" && auxiliary.size() >= 7) {
        return std::clamp(0.65 * auxiliary[2] + 0.35 * auxiliary[1], 0.0, 1.0);
    }
    if (auxiliaryFeatureMode_ == "appearance_bank" && auxiliary.size() >= 6) {
        return std::clamp(0.65 * auxiliary[5] + 0.35 * auxiliary[4], 0.0, 1.0);
    }
    return 0.0;
}

void RetinaAdapter::applyTemporalCoarseToFineDualPass(
    std::vector<std::vector<double>>& bandFeatures,
    const std::vector<std::vector<double>>& auxiliaryFeatures) const {
    if (!temporalCoarseToFineEnabled_ || bandFeatures.size() <= 1 || bandFeatures.empty()) {
        return;
    }

    const size_t lowBand = 0;
    const size_t highBand = bandFeatures.size() - 1;
    const double lowTransient =
        lowBand < auxiliaryFeatures.size() ? estimateTransientDrive(auxiliaryFeatures[lowBand]) : 0.0;
    const double lowDetail =
        lowBand < auxiliaryFeatures.size() ? estimateDetailDrive(auxiliaryFeatures[lowBand]) : 0.0;
    const double highSustained =
        highBand < auxiliaryFeatures.size() ? estimateSustainedDrive(auxiliaryFeatures[highBand]) : 0.0;
    const double highDetail =
        highBand < auxiliaryFeatures.size() ? estimateDetailDrive(auxiliaryFeatures[highBand]) : 0.0;

    const double coarseScale = std::max(
        0.0, 1.0 + temporalCoarseBias_ + temporalTransientGain_ * (0.70 * lowTransient + 0.30 * lowDetail));
    const double fineDrive = std::clamp(0.65 * highSustained + 0.35 * highDetail, 0.0, 1.0);

    for (size_t channel = 0; channel < bandFeatures[lowBand].size(); ++channel) {
        const double coarseValue = bandFeatures[lowBand][channel];
        const double supportedCoarse = std::clamp(coarseValue * coarseScale, 0.0, 1.0);
        bandFeatures[lowBand][channel] = supportedCoarse;

        double fineSupport = 0.0;
        if (highBand < bandFeatures.size() && channel < bandFeatures[highBand].size()) {
            fineSupport = bandFeatures[highBand][channel];
            const double fineScale = std::max(
                0.0, 1.0 + temporalSustainedGain_ * fineDrive +
                         temporalCrossBandGain_ * supportedCoarse);
            bandFeatures[highBand][channel] = std::clamp(fineSupport * fineScale, 0.0, 1.0);
        }

        for (size_t bandIdx = 1; bandIdx + 1 < bandFeatures.size(); ++bandIdx) {
            if (channel >= bandFeatures[bandIdx].size()) {
                continue;
            }
            const double mix =
                static_cast<double>(bandIdx) / static_cast<double>(highBand);
            const double mixedDrive =
                ((1.0 - mix) * (0.50 * lowTransient + 0.20 * lowDetail)) +
                (mix * (0.50 * highSustained + 0.20 * highDetail));
            const double support =
                ((1.0 - mix) * supportedCoarse) + (mix * fineSupport);
            const double scale =
                std::max(0.0, 1.0 + 0.5 * temporalTransientGain_ * mixedDrive +
                                 0.5 * temporalCrossBandGain_ * support);
            bandFeatures[bandIdx][channel] =
                std::clamp(bandFeatures[bandIdx][channel] * scale, 0.0, 1.0);
        }
    }
}

void RetinaAdapter::applyComplexCellStage(
    std::vector<std::vector<double>>& bandFeatures) const {
    if (!complexCellEnabled_ || bandFeatures.empty() || numOrientations_ <= 0) {
        return;
    }

    const size_t orientationCount = static_cast<size_t>(numOrientations_);
    const size_t blockCount = getOrientationChannelBlocks();
    const size_t colorChannelCount = getColorEdgeChannelCount();
    if (blockCount == 0 || colorChannelCount == 0) {
        return;
    }
    const size_t expectedChannels = orientationCount * blockCount * colorChannelCount;

    auto channelIndex = [&](size_t blockIdx, size_t colorIdx, size_t orientationIdx) {
        return (((blockIdx * colorChannelCount) + colorIdx) * orientationCount) + orientationIdx;
    };

    for (auto& band : bandFeatures) {
        if (band.size() != expectedChannels) {
            continue;
        }

        const std::vector<double> original = band;
        for (size_t colorIdx = 0; colorIdx < colorChannelCount; ++colorIdx) {
            std::vector<double> pooled(orientationCount, 0.0);
            double pooledMax = 0.0;

            for (size_t orientationIdx = 0; orientationIdx < orientationCount; ++orientationIdx) {
                double energySumSq = 0.0;
                double maxValue = 0.0;
                for (size_t blockIdx = 0; blockIdx < blockCount; ++blockIdx) {
                    const double value = original[channelIndex(blockIdx, colorIdx, orientationIdx)];
                    energySumSq += value * value;
                    maxValue = std::max(maxValue, value);
                }
                const double rmsValue = std::sqrt(energySumSq / static_cast<double>(blockCount));
                pooled[orientationIdx] =
                    (complexCellMaxMix_ * maxValue) + ((1.0 - complexCellMaxMix_) * rmsValue);
                pooledMax = std::max(pooledMax, pooled[orientationIdx]);
            }

            if (complexCellDivisiveGain_ > 1e-9 && pooledMax > 1e-9) {
                const double pooledMean =
                    std::accumulate(pooled.begin(), pooled.end(), 0.0) /
                    static_cast<double>(orientationCount);
                std::vector<double> normalized = pooled;
                double normalizedMax = 0.0;
                for (size_t orientationIdx = 0; orientationIdx < orientationCount; ++orientationIdx) {
                    const size_t prevIdx =
                        (orientationIdx + orientationCount - 1) % orientationCount;
                    const size_t nextIdx = (orientationIdx + 1) % orientationCount;
                    const double neighborContext =
                        0.5 * (pooled[prevIdx] + pooled[nextIdx]);
                    const double suppressivePool =
                        ((1.0 - complexCellNeighborMix_) * pooledMean) +
                        (complexCellNeighborMix_ * neighborContext);
                    const double denom =
                        1.0 + complexCellDivisiveGain_ *
                                  std::max(complexCellDivisiveFloor_, suppressivePool);
                    normalized[orientationIdx] = pooled[orientationIdx] / std::max(1e-6, denom);
                    normalizedMax = std::max(normalizedMax, normalized[orientationIdx]);
                }
                if (normalizedMax > 1e-9) {
                    const double restoreScale = pooledMax / normalizedMax;
                    for (double& value : normalized) {
                        value = std::clamp(value * restoreScale, 0.0, 1.0);
                    }
                }
                pooled.swap(normalized);
            }

            for (size_t blockIdx = 0; blockIdx < blockCount; ++blockIdx) {
                for (size_t orientationIdx = 0; orientationIdx < orientationCount; ++orientationIdx) {
                    const size_t idx = channelIndex(blockIdx, colorIdx, orientationIdx);
                    band[idx] = std::clamp(
                        ((1.0 - complexCellPoolBlend_) * original[idx]) +
                            (complexCellPoolBlend_ * pooled[orientationIdx]),
                        0.0,
                        1.0);
                }
            }
        }
    }
}

void RetinaAdapter::applyContourSupportBank(
    const std::vector<std::vector<double>>& bandFeatures,
    std::vector<std::vector<double>>& auxiliaryFeatures) const {
    if (auxiliaryFeatureMode_ != "contour_support_bank" ||
        bandFeatures.empty() ||
        auxiliaryFeatures.size() != bandFeatures.size() ||
        numOrientations_ <= 0) {
        return;
    }

    const size_t orientationCount = static_cast<size_t>(numOrientations_);
    const size_t blockCount = getOrientationChannelBlocks();
    const size_t colorChannelCount = getColorEdgeChannelCount();
    const size_t subfieldCount = getSubfieldCount();
    if (blockCount == 0 || colorChannelCount == 0) {
        return;
    }
    const size_t expectedChannels = orientationCount * blockCount * colorChannelCount;
    const bool hasPooledBlock = subfieldIncludePooled_ && subfieldCount > 0;
    const size_t firstSubfieldBlock = hasPooledBlock ? 1u : 0u;

    auto wrapOrientation = [&](int idx) {
        int wrapped = idx % static_cast<int>(orientationCount);
        if (wrapped < 0) {
            wrapped += static_cast<int>(orientationCount);
        }
        return static_cast<size_t>(wrapped);
    };

    auto channelIndex = [&](size_t blockIdx, size_t colorIdx, size_t orientationIdx) {
        return (((blockIdx * colorChannelCount) + colorIdx) * orientationCount) + orientationIdx;
    };

    auto smoothSupport = [&](const std::vector<double>& values, size_t centerIdx) {
        return std::clamp(
            values[centerIdx] +
                0.5 * (values[wrapOrientation(static_cast<int>(centerIdx) - 1)] +
                       values[wrapOrientation(static_cast<int>(centerIdx) + 1)]),
            0.0,
            1.5);
    };

    for (size_t bandIdx = 0; bandIdx < bandFeatures.size(); ++bandIdx) {
        const auto& band = bandFeatures[bandIdx];
        auto& auxiliary = auxiliaryFeatures[bandIdx];
        if (band.size() != expectedChannels || auxiliary.size() < 6) {
            continue;
        }

        std::vector<std::vector<double>> blockOrientations;
        blockOrientations.reserve(blockCount);
        for (size_t blockIdx = 0; blockIdx < blockCount; ++blockIdx) {
            std::vector<double> block(orientationCount, 0.0);
            for (size_t orientationIdx = 0; orientationIdx < orientationCount; ++orientationIdx) {
                block[orientationIdx] = band[channelIndex(blockIdx, 0, orientationIdx)];
            }
            blockOrientations.push_back(std::move(block));
        }

        std::vector<double> pooledOrientation(orientationCount, 0.0);
        if (hasPooledBlock) {
            pooledOrientation = blockOrientations[0];
        } else {
            for (const auto& block : blockOrientations) {
                for (size_t orient = 0; orient < orientationCount; ++orient) {
                    pooledOrientation[orient] += block[orient] / static_cast<double>(blockCount);
                }
            }
        }

        const auto dominantIt = std::max_element(pooledOrientation.begin(), pooledOrientation.end());
        const size_t dominantIdx =
            static_cast<size_t>(std::distance(pooledOrientation.begin(), dominantIt));
        const double dominantSupport = smoothSupport(pooledOrientation, dominantIdx);
        const size_t orthIdx = wrapOrientation(static_cast<int>(dominantIdx) +
                                              static_cast<int>(orientationCount / 4));
        const double orthSupport = smoothSupport(pooledOrientation, orthIdx);
        const double prevSupport =
            pooledOrientation[wrapOrientation(static_cast<int>(dominantIdx) - 1)];
        const double nextSupport =
            pooledOrientation[wrapOrientation(static_cast<int>(dominantIdx) + 1)];
        const double junctionness =
            dominantSupport * std::max(orthSupport, 0.5 * (prevSupport + nextSupport));

        if (subfieldCount == 0 || subfieldGridSize_ <= 1 || blockCount <= firstSubfieldBlock) {
            auxiliary[0] = std::clamp(dominantSupport * auxiliaryFeatureGain_, 0.0, 1.0);
            auxiliary[1] = std::clamp(0.5 * (prevSupport + nextSupport) * auxiliaryFeatureGain_,
                                      0.0,
                                      1.0);
            auxiliary[2] = 0.0;
            auxiliary[3] = std::clamp(junctionness * auxiliaryFeatureGain_, 0.0, 1.0);
            auxiliary[4] = 0.0;
            auxiliary[5] = 0.0;
            continue;
        }

        const double theta =
            (static_cast<double>(dominantIdx) * M_PI) / static_cast<double>(orientationCount);
        const double axisX = std::cos(theta);
        const double axisY = std::sin(theta);
        const double normalX = -axisY;
        const double normalY = axisX;

        double positiveAlong = 0.0;
        double negativeAlong = 0.0;
        double positiveAcross = 0.0;
        double negativeAcross = 0.0;
        double curvePositive = 0.0;
        double curveNegative = 0.0;

        for (size_t blockIdx = firstSubfieldBlock; blockIdx < blockOrientations.size(); ++blockIdx) {
            const size_t localIdx = blockIdx - firstSubfieldBlock;
            const int subRow = static_cast<int>(localIdx / static_cast<size_t>(subfieldGridSize_));
            const int subCol = static_cast<int>(localIdx % static_cast<size_t>(subfieldGridSize_));
            const double x =
                ((static_cast<double>(subCol) + 0.5) / static_cast<double>(subfieldGridSize_)) - 0.5;
            const double y =
                ((static_cast<double>(subRow) + 0.5) / static_cast<double>(subfieldGridSize_)) - 0.5;
            const double along = x * axisX + y * axisY;
            const double across = x * normalX + y * normalY;
            const auto& block = blockOrientations[blockIdx];
            const double blockDominant = smoothSupport(block, dominantIdx);
            const double blockCurve = std::max(
                smoothSupport(block, wrapOrientation(static_cast<int>(dominantIdx) - 1)),
                smoothSupport(block, wrapOrientation(static_cast<int>(dominantIdx) + 1)));

            if (along >= 0.0) {
                positiveAlong += blockDominant * std::abs(along);
                curvePositive += blockCurve * std::abs(along);
            } else {
                negativeAlong += blockDominant * std::abs(along);
                curveNegative += blockCurve * std::abs(along);
            }
            if (across >= 0.0) {
                positiveAcross += blockDominant * std::abs(across);
            } else {
                negativeAcross += blockDominant * std::abs(across);
            }
        }

        const double alongTotal = positiveAlong + negativeAlong;
        const double acrossTotal = positiveAcross + negativeAcross;
        const double continuation =
            alongTotal > 1e-9 ? (2.0 * std::min(positiveAlong, negativeAlong) / alongTotal) : 0.0;
        const double endstop =
            alongTotal > 1e-9 ? (std::abs(positiveAlong - negativeAlong) / alongTotal) : 0.0;
        const double curveTotal = curvePositive + curveNegative;
        const double cocircular =
            curveTotal > 1e-9 ? (2.0 * std::min(curvePositive, curveNegative) / curveTotal) : 0.0;
        const double borderLeft =
            acrossTotal > 1e-9 ? (negativeAcross / acrossTotal) : 0.0;
        const double borderRight =
            acrossTotal > 1e-9 ? (positiveAcross / acrossTotal) : 0.0;

        auxiliary[0] = std::clamp(continuation * dominantSupport * auxiliaryFeatureGain_, 0.0, 1.0);
        auxiliary[1] = std::clamp(cocircular * 0.5 * (prevSupport + nextSupport) *
                                      auxiliaryFeatureGain_,
                                  0.0,
                                  1.0);
        auxiliary[2] = std::clamp(endstop * dominantSupport * auxiliaryFeatureGain_, 0.0, 1.0);
        auxiliary[3] = std::clamp(junctionness * auxiliaryFeatureGain_, 0.0, 1.0);
        auxiliary[4] = std::clamp(borderLeft * dominantSupport * auxiliaryFeatureGain_, 0.0, 1.0);
        auxiliary[5] = std::clamp(borderRight * dominantSupport * auxiliaryFeatureGain_, 0.0, 1.0);
    }
}

size_t RetinaAdapter::getAuxiliaryChannelCount() const {
    if (auxiliaryFeatureMode_ == "none") {
        return 0u;
    }
    if (auxiliaryFeatureMode_ == "closure_bank") {
        return 3u;
    }
    if (auxiliaryFeatureMode_ == "gap_bank") {
        return 4u;
    }
    if (auxiliaryFeatureMode_ == "topology_maps") {
        return 8u;
    }
    if (auxiliaryFeatureMode_ == "contour_graph") {
        return 8u;
    }
    if (auxiliaryFeatureMode_ == "contour_sequence") {
        return 8u;
    }
    if (auxiliaryFeatureMode_ == "color_opponent") {
        return 3u;
    }
    if (auxiliaryFeatureMode_ == "appearance_bank") {
        return 6u;
    }
    if (auxiliaryFeatureMode_ == "appearance_stream_bank") {
        return 10u;
    }
    if (auxiliaryFeatureMode_ == "luminance_stream_bank") {
        return 7u;
    }
    if (auxiliaryFeatureMode_ == "contour_support_bank") {
        return 6u;
    }
    return 1u;
}

size_t RetinaAdapter::getSubfieldCount() const {
    if (subfieldGridSize_ <= 1) {
        return 0u;
    }
    return static_cast<size_t>(subfieldGridSize_ * subfieldGridSize_);
}

RetinaAdapter::Image RetinaAdapter::blurImage(const Image& image, double sigma) const {
    if (sigma <= 0.0) {
        return image;
    }

    const auto kernel = makeGaussianKernel(sigma);
    const int radius = static_cast<int>(kernel.size() / 2);
    const size_t pixelCount = static_cast<size_t>(image.rows) * static_cast<size_t>(image.cols);
    const int channels = std::max(1, image.channels);
    std::vector<double> horizontal(pixelCount * static_cast<size_t>(channels), 0.0);

    for (int channel = 0; channel < channels; ++channel) {
        for (int row = 0; row < image.rows; ++row) {
            for (int col = 0; col < image.cols; ++col) {
                double accum = 0.0;
                for (int k = -radius; k <= radius; ++k) {
                    const int srcCol = std::clamp(col + k, 0, image.cols - 1);
                    accum += kernel[static_cast<size_t>(k + radius)] *
                             static_cast<double>(image.getPixel(row, srcCol, channel));
                }
                horizontal[(static_cast<size_t>(row * image.cols + col) *
                            static_cast<size_t>(channels)) +
                           static_cast<size_t>(channel)] = accum;
            }
        }
    }

    Image blurred;
    blurred.rows = image.rows;
    blurred.cols = image.cols;
    blurred.channels = channels;
    blurred.pixels.resize(pixelCount * static_cast<size_t>(channels));
    for (int channel = 0; channel < channels; ++channel) {
        for (int row = 0; row < image.rows; ++row) {
            for (int col = 0; col < image.cols; ++col) {
                double accum = 0.0;
                for (int k = -radius; k <= radius; ++k) {
                    const int srcRow = std::clamp(row + k, 0, image.rows - 1);
                    accum += kernel[static_cast<size_t>(k + radius)] *
                             horizontal[(static_cast<size_t>(srcRow * image.cols + col) *
                                         static_cast<size_t>(channels)) +
                                        static_cast<size_t>(channel)];
                }
                const int value = static_cast<int>(std::lround(std::clamp(accum, 0.0, 255.0)));
                blurred.pixels[(static_cast<size_t>(row * image.cols + col) *
                                static_cast<size_t>(channels)) +
                               static_cast<size_t>(channel)] = static_cast<uint8_t>(value);
            }
        }
    }
    return blurred;
}

RetinaAdapter::Image RetinaAdapter::applyLgnRelayWithParams(const Image& image,
                                                            double centerSigma,
                                                            double surroundSigma,
                                                            double centerSurroundStrength) const {
    if (centerSurroundStrength <= 0.0) {
        return image;
    }

    const auto centerImage = blurImage(image, centerSigma);
    const auto surroundImage = blurImage(image, surroundSigma);
    Image relayed = centerImage;
    const int channels = std::max(1, image.channels);
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            for (int channel = 0; channel < channels; ++channel) {
                const double centerValue =
                    static_cast<double>(centerImage.getPixel(row, col, channel));
                const double surroundValue =
                    static_cast<double>(surroundImage.getPixel(row, col, channel));
                const double sharpened =
                    centerValue + centerSurroundStrength * (centerValue - surroundValue);
                relayed.pixels[(static_cast<size_t>(row * image.cols + col) *
                                static_cast<size_t>(channels)) +
                               static_cast<size_t>(channel)] =
                    static_cast<uint8_t>(std::clamp(std::lround(sharpened), 0L, 255L));
            }
        }
    }

    return relayed;
}

RetinaAdapter::Image RetinaAdapter::applyLgnBurstTonicRelay(const Image& image) const {
    if (lgnCenterSurroundStrength_ <= 0.0) {
        return image;
    }

    const auto centerImage = blurImage(image, lgnCenterSigma_);
    const auto surroundImage = blurImage(image, lgnSurroundSigma_);
    Image relayed = centerImage;
    const int channels = std::max(1, image.channels);
    const double transitionSlope = std::max(1e-6, lgnBurstSlope_);
    const double burstExtra = lgnBurstExtraStrength_ * lgnBurstNeuromodulator_;

    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            for (int channel = 0; channel < channels; ++channel) {
                const double centerValue =
                    static_cast<double>(centerImage.getPixel(row, col, channel));
                const double surroundValue =
                    static_cast<double>(surroundImage.getPixel(row, col, channel));
                const double centerSurround = centerValue - surroundValue;
                const double localSalience = std::abs(centerSurround) / 255.0;
                const double burstGate =
                    std::clamp((localSalience - lgnBurstThreshold_) * transitionSlope, 0.0, 1.0);
                const double effectiveStrength =
                    lgnCenterSurroundStrength_ + (burstGate * burstExtra);
                const double relayedValue = centerValue + effectiveStrength * centerSurround;
                relayed.pixels[(static_cast<size_t>(row * image.cols + col) *
                                static_cast<size_t>(channels)) +
                               static_cast<size_t>(channel)] =
                    static_cast<uint8_t>(std::clamp(std::lround(relayedValue), 0L, 255L));
            }
        }
    }

    return relayed;
}

RetinaAdapter::Image RetinaAdapter::applyLgnRelay(const Image& image) const {
    if (!lgnRelayEnabled_ || lgnCenterSurroundStrength_ <= 0.0) {
        return image;
    }

    if (lgnBurstTonicEnabled_) {
        return applyLgnBurstTonicRelay(image);
    }

    return applyLgnRelayWithParams(
        image, lgnCenterSigma_, lgnSurroundSigma_, lgnCenterSurroundStrength_);
}

RetinaAdapter::Image RetinaAdapter::makeAchromaticImage(const Image& image,
                                                        double achromaticMix) const {
    if (achromaticMix <= 1e-6 || image.channels <= 1) {
        return image;
    }

    Image achromatic = image;
    const int channels = std::max(1, image.channels);
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            const double luminance = static_cast<double>(image.getPixel(row, col));
            for (int channel = 0; channel < channels; ++channel) {
                const double source = static_cast<double>(image.getPixel(row, col, channel));
                const double blended =
                    ((1.0 - achromaticMix) * source) + (achromaticMix * luminance);
                achromatic.pixels[(static_cast<size_t>(row * image.cols + col) *
                                   static_cast<size_t>(channels)) +
                                  static_cast<size_t>(channel)] =
                    static_cast<uint8_t>(std::clamp(std::lround(blended), 0L, 255L));
            }
        }
    }
    return achromatic;
}

RetinaAdapter::Image RetinaAdapter::blendImages(const Image& base,
                                                const Image& overlay,
                                                double overlayWeight) const {
    if (overlayWeight <= 1e-6) {
        return base;
    }
    if (overlayWeight >= 1.0 - 1e-6) {
        return overlay;
    }

    Image blended = base;
    const int rows = std::min(base.rows, overlay.rows);
    const int cols = std::min(base.cols, overlay.cols);
    const int channels = std::min(std::max(1, base.channels), std::max(1, overlay.channels));
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            for (int channel = 0; channel < channels; ++channel) {
                const double baseValue = static_cast<double>(base.getPixel(row, col, channel));
                const double overlayValue =
                    static_cast<double>(overlay.getPixel(row, col, channel));
                const double value =
                    ((1.0 - overlayWeight) * baseValue) + (overlayWeight * overlayValue);
                blended.pixels[(static_cast<size_t>(row * base.cols + col) *
                                static_cast<size_t>(std::max(1, base.channels))) +
                               static_cast<size_t>(channel)] =
                    static_cast<uint8_t>(std::clamp(std::lround(value), 0L, 255L));
            }
        }
    }
    return blended;
}

RetinaAdapter::RelayBandImageSet RetinaAdapter::buildRelayBandImages(const Image& image) const {
    RelayBandImageSet bands;
    const size_t bandCount = blurSigmas_.empty() ? 1u : blurSigmas_.size();
    bands.orientationBands.reserve(bandCount);
    bands.auxiliaryBands.reserve(bandCount);

    if (!lgnParallelRelayEnabled_) {
        Image processed = applyLgnRelay(image);
        processed = applyLocalContrastNormalization(processed);
        if (blurSigmas_.empty()) {
            bands.orientationBands.push_back(processed);
            bands.auxiliaryBands.push_back(processed);
        } else {
            for (double sigma : blurSigmas_) {
                auto bandImage = blurImage(processed, sigma);
                bands.orientationBands.push_back(bandImage);
                bands.auxiliaryBands.push_back(std::move(bandImage));
            }
        }
        return bands;
    }

    auto magnoInput = makeAchromaticImage(image, lgnMagnoAchromaticMix_);
    auto magnoRelay = applyLgnRelayWithParams(
        magnoInput,
        lgnMagnoCenterSigma_,
        lgnMagnoSurroundSigma_,
        lgnMagnoCenterSurroundStrength_);
    magnoRelay = applyLocalContrastNormalization(magnoRelay);

    auto parvoRelay = applyLgnRelayWithParams(
        image,
        lgnParvoCenterSigma_,
        lgnParvoSurroundSigma_,
        lgnParvoCenterSurroundStrength_);
    parvoRelay = blendImages(parvoRelay, image, lgnParvoOriginalMix_);
    parvoRelay = applyLocalContrastNormalization(parvoRelay);

    if (blurSigmas_.empty()) {
        const double magnoWeight = std::clamp(0.5 + (0.5 * lgnBandMagnoFloor_), 0.0, 1.0);
        const double auxiliaryMagnoWeight =
            std::clamp(lgnAuxiliaryMagnoMix_ * magnoWeight, 0.0, 1.0);
        bands.orientationBands.push_back(blendImages(parvoRelay, magnoRelay, magnoWeight));
        bands.auxiliaryBands.push_back(
            blendImages(parvoRelay, magnoRelay, auxiliaryMagnoWeight));
        return bands;
    }

    for (size_t bandIdx = 0; bandIdx < blurSigmas_.size(); ++bandIdx) {
        const double sigma = blurSigmas_[bandIdx];
        const auto magnoBand = blurImage(magnoRelay, sigma + lgnMagnoExtraBlur_);
        const auto parvoBand = blurImage(parvoRelay, sigma);
        const double detailPosition = blurSigmas_.size() <= 1
                                          ? 0.5
                                          : static_cast<double>(bandIdx) /
                                                static_cast<double>(blurSigmas_.size() - 1);
        const double magnoWeight = std::clamp(
            lgnBandMagnoFloor_ + ((1.0 - lgnBandMagnoFloor_) * (1.0 - detailPosition)),
            0.0,
            1.0);
        const double auxiliaryMagnoWeight =
            std::clamp(lgnAuxiliaryMagnoMix_ * magnoWeight, 0.0, 1.0);
        bands.orientationBands.push_back(blendImages(parvoBand, magnoBand, magnoWeight));
        bands.auxiliaryBands.push_back(
            blendImages(parvoBand, magnoBand, auxiliaryMagnoWeight));
    }

    return bands;
}

RetinaAdapter::Image RetinaAdapter::applyViewTransform(const Image& image) const {
    const bool hasTransform = std::abs(rotationDeg_) > 1e-6 ||
                              std::abs(scaleX_ - 1.0) > 1e-6 ||
                              std::abs(scaleY_ - 1.0) > 1e-6 ||
                              std::abs(shiftXPx_) > 1e-6 ||
                              std::abs(shiftYPx_) > 1e-6 ||
                              mirrorX_ || mirrorY_;
    if (!hasTransform) {
        return image;
    }

    Image transformed;
    transformed.rows = image.rows;
    transformed.cols = image.cols;
    transformed.channels = std::max(1, image.channels);
    transformed.pixels.assign(
        static_cast<size_t>(image.rows * image.cols * transformed.channels), 0);

    const double centerX = 0.5 * static_cast<double>(image.cols - 1);
    const double centerY = 0.5 * static_cast<double>(image.rows - 1);
    const double theta = rotationDeg_ * M_PI / 180.0;
    const double cosTheta = std::cos(theta);
    const double sinTheta = std::sin(theta);

    auto sampleBilinear = [&](double row, double col, int channel) -> uint8_t {
        if (row < 0.0 || col < 0.0 ||
            row > static_cast<double>(image.rows - 1) ||
            col > static_cast<double>(image.cols - 1)) {
            return 0;
        }

        const int r0 = static_cast<int>(std::floor(row));
        const int c0 = static_cast<int>(std::floor(col));
        const int r1 = std::min(r0 + 1, image.rows - 1);
        const int c1 = std::min(c0 + 1, image.cols - 1);
        const double fr = row - static_cast<double>(r0);
        const double fc = col - static_cast<double>(c0);

        const double p00 = static_cast<double>(image.getPixel(r0, c0, channel));
        const double p01 = static_cast<double>(image.getPixel(r0, c1, channel));
        const double p10 = static_cast<double>(image.getPixel(r1, c0, channel));
        const double p11 = static_cast<double>(image.getPixel(r1, c1, channel));

        const double top = p00 * (1.0 - fc) + p01 * fc;
        const double bottom = p10 * (1.0 - fc) + p11 * fc;
        const double value = top * (1.0 - fr) + bottom * fr;
        return static_cast<uint8_t>(std::clamp(std::lround(value), 0L, 255L));
    };

    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            double x = static_cast<double>(col) - centerX;
            double y = static_cast<double>(row) - centerY;
            if (mirrorX_) {
                x = -x;
            }
            if (mirrorY_) {
                y = -y;
            }
            x -= shiftXPx_;
            y -= shiftYPx_;
            x /= scaleX_;
            y /= scaleY_;

            const double srcX = cosTheta * x + sinTheta * y + centerX;
            const double srcY = -sinTheta * x + cosTheta * y + centerY;
            for (int channel = 0; channel < transformed.channels; ++channel) {
                transformed.pixels[(static_cast<size_t>(row * image.cols + col) *
                                    static_cast<size_t>(transformed.channels)) +
                                   static_cast<size_t>(channel)] =
                    sampleBilinear(srcY, srcX, channel);
            }
        }
    }

    return transformed;
}

RetinaAdapter::Image RetinaAdapter::applyLocalContrastNormalization(const Image& image) const {
    if (localContrastRadius_ <= 0 || localContrastStrength_ <= 0.0) {
        return image;
    }

    Image normalized = image;
    const int channels = std::max(1, image.channels);
    const int radius = std::max(1, localContrastRadius_);
    const double strength = localContrastStrength_;

    for (int channel = 0; channel < channels; ++channel) {
        for (int row = 0; row < image.rows; ++row) {
            for (int col = 0; col < image.cols; ++col) {
                double sum = 0.0;
                double sumSq = 0.0;
                int count = 0;
                for (int dr = -radius; dr <= radius; ++dr) {
                    const int srcRow = clampIndex(row + dr, 0, image.rows - 1);
                    for (int dc = -radius; dc <= radius; ++dc) {
                        const int srcCol = clampIndex(col + dc, 0, image.cols - 1);
                        const double value =
                            static_cast<double>(image.getPixel(srcRow, srcCol, channel));
                        sum += value;
                        sumSq += value * value;
                        count++;
                    }
                }

                if (count <= 0) {
                    continue;
                }

                const double mean = sum / static_cast<double>(count);
                const double variance = std::max(
                    0.0, (sumSq / static_cast<double>(count)) - (mean * mean));
                const double stddev = std::sqrt(variance);
                const double pixel = static_cast<double>(image.getPixel(row, col, channel));
                const double z = (pixel - mean) / std::max(1.0, stddev);
                const double scaled = 128.0 + strength * 48.0 * z;
                normalized.pixels[(static_cast<size_t>(row * image.cols + col) *
                                   static_cast<size_t>(channels)) +
                                  static_cast<size_t>(channel)] =
                    static_cast<uint8_t>(std::clamp(std::lround(scaled), 0L, 255L));
            }
        }
    }

    return normalized;
}

std::vector<uint8_t> RetinaAdapter::normalizeEdgeRegion(const std::vector<uint8_t>& region) const {
    if (!edgePatchNormalizationEnabled_ || edgePatchContrastStrength_ <= 0.0 || region.empty()) {
        return region;
    }

    double sum = 0.0;
    double sumSq = 0.0;
    for (uint8_t pixel : region) {
        const double value = static_cast<double>(pixel);
        sum += value;
        sumSq += value * value;
    }

    const double mean = sum / static_cast<double>(region.size());
    const double variance =
        std::max(0.0, (sumSq / static_cast<double>(region.size())) - (mean * mean));
    const double stddev = std::sqrt(variance);
    const double denom = std::max(edgePatchMinStd_, stddev);

    std::vector<uint8_t> normalized(region.size(), 128);
    for (size_t i = 0; i < region.size(); ++i) {
        const double z = (static_cast<double>(region[i]) - mean) / denom;
        const double scaled = 128.0 + edgePatchContrastStrength_ * 48.0 * z;
        normalized[i] = static_cast<uint8_t>(std::clamp(std::lround(scaled), 0L, 255L));
    }

    return normalized;
}

/**
 * @brief Extract a rectangular region from the image
 *
 * Extracts a regionSize × regionSize block of pixels from the image.
 * The region is specified by its row and column indices in the grid.
 *
 * Region calculation:
 * - startRow = regionRow × regionSize
 * - startCol = regionCol × regionSize
 * - For 28×28 image with 8×8 grid: regionSize = 28/8 = 3 pixels
 *
 * @param image Input image
 * @param regionRow Row index of region in grid (0 to gridSize-1)
 * @param regionCol Column index of region in grid (0 to gridSize-1)
 * @return Flattened vector of pixel values (regionSize² elements)
 */
std::vector<uint8_t> RetinaAdapter::extractRegion(const Image& image,
                                                  int regionRow,
                                                  int regionCol) const {
    return extractRegion(image, regionRow, regionCol, regionSize_);
}

std::vector<uint8_t> RetinaAdapter::extractRegion(const Image& image,
                                                  int regionRow,
                                                  int regionCol,
                                                  int targetSize) const {
    const int effectiveSize = std::max(1, targetSize);
    std::vector<uint8_t> region(static_cast<size_t>(effectiveSize * effectiveSize));

    const auto rowBoundaries = computeAxisSamplingBoundaries(image.rows);
    const auto colBoundaries = computeAxisSamplingBoundaries(image.cols);
    const int safeRow = clampIndex(regionRow, 0, std::max(0, gridSize_ - 1));
    const int safeCol = clampIndex(regionCol, 0, std::max(0, gridSize_ - 1));
    const int startRow = rowBoundaries[static_cast<size_t>(safeRow)];
    const int endRow = rowBoundaries[static_cast<size_t>(safeRow + 1)];
    const int startCol = colBoundaries[static_cast<size_t>(safeCol)];
    const int endCol = colBoundaries[static_cast<size_t>(safeCol + 1)];
    const int sourceHeight = std::max(1, endRow - startRow);
    const int sourceWidth = std::max(1, endCol - startCol);

    for (int r = 0; r < effectiveSize; ++r) {
        for (int c = 0; c < effectiveSize; ++c) {
            const int localRow = clampIndex(
                static_cast<int>(((static_cast<double>(r) + 0.5) * sourceHeight) /
                                 static_cast<double>(effectiveSize)),
                0, sourceHeight - 1);
            const int localCol = clampIndex(
                static_cast<int>(((static_cast<double>(c) + 0.5) * sourceWidth) /
                                 static_cast<double>(effectiveSize)),
                0, sourceWidth - 1);
            const int imgRow = startRow + localRow;
            const int imgCol = startCol + localCol;
            region[static_cast<size_t>(r * effectiveSize + c)] = image.getPixel(imgRow, imgCol);
        }
    }

    return region;
}

std::vector<double> RetinaAdapter::computeColorOpponentFeatures(const Image& image,
                                                                int regionRow,
                                                                int regionCol,
                                                                int targetSize) const {
    std::vector<double> auxiliary(getAuxiliaryChannelCount(), 0.0);
    const bool colorOpponentMode =
        auxiliaryFeatureMode_ == "color_opponent" ||
        auxiliaryFeatureMode_ == "appearance_bank" ||
        auxiliaryFeatureMode_ == "appearance_stream_bank" ||
        auxiliaryFeatureMode_ == "luminance_stream_bank";
    if (!colorOpponentMode || auxiliary.empty()) {
        return auxiliary;
    }

    const int effectiveSize = std::max(1, targetSize);
    const auto rowBoundaries = computeAxisSamplingBoundaries(image.rows);
    const auto colBoundaries = computeAxisSamplingBoundaries(image.cols);
    const int safeRow = clampIndex(regionRow, 0, std::max(0, gridSize_ - 1));
    const int safeCol = clampIndex(regionCol, 0, std::max(0, gridSize_ - 1));
    const int startRow = rowBoundaries[static_cast<size_t>(safeRow)];
    const int endRow = rowBoundaries[static_cast<size_t>(safeRow + 1)];
    const int startCol = colBoundaries[static_cast<size_t>(safeCol)];
    const int endCol = colBoundaries[static_cast<size_t>(safeCol + 1)];
    const int sourceHeight = std::max(1, endRow - startRow);
    const int sourceWidth = std::max(1, endCol - startCol);

    double redSum = 0.0;
    double greenSum = 0.0;
    double blueSum = 0.0;
    double luminanceSum = 0.0;
    double luminanceSqSum = 0.0;
    double gradientSum = 0.0;
    double transientOnSum = 0.0;
    double transientOffSum = 0.0;
    const double sampleCount = static_cast<double>(effectiveSize * effectiveSize);
    std::vector<double> luminanceSamples;
    luminanceSamples.reserve(static_cast<size_t>(effectiveSize * effectiveSize));

    for (int r = 0; r < effectiveSize; ++r) {
        for (int c = 0; c < effectiveSize; ++c) {
            const int localRow = clampIndex(
                static_cast<int>(((static_cast<double>(r) + 0.5) * sourceHeight) /
                                 static_cast<double>(effectiveSize)),
                0, sourceHeight - 1);
            const int localCol = clampIndex(
                static_cast<int>(((static_cast<double>(c) + 0.5) * sourceWidth) /
                                 static_cast<double>(effectiveSize)),
                0, sourceWidth - 1);
            const int imgRow = startRow + localRow;
            const int imgCol = startCol + localCol;
            const double red = image.channels >= 3 ? image.getNormalizedPixel(imgRow, imgCol, 0)
                                                   : image.getNormalizedPixel(imgRow, imgCol);
            const double green = image.channels >= 3 ? image.getNormalizedPixel(imgRow, imgCol, 1)
                                                     : red;
            const double blue = image.channels >= 3 ? image.getNormalizedPixel(imgRow, imgCol, 2)
                                                    : red;
            const double luminance = 0.299 * red + 0.587 * green + 0.114 * blue;

            redSum += red;
            greenSum += green;
            blueSum += blue;
            luminanceSum += luminance;
            luminanceSqSum += luminance * luminance;
            luminanceSamples.push_back(luminance);

            const int leftCol = std::max(0, imgCol - 1);
            const int upRow = std::max(0, imgRow - 1);
            const int rightCol = std::min(image.cols - 1, imgCol + 1);
            const int downRow = std::min(image.rows - 1, imgRow + 1);
            const double leftRed = image.channels >= 3 ? image.getNormalizedPixel(imgRow, leftCol, 0)
                                                       : image.getNormalizedPixel(imgRow, leftCol);
            const double leftGreen = image.channels >= 3 ? image.getNormalizedPixel(imgRow, leftCol, 1)
                                                         : leftRed;
            const double leftBlue = image.channels >= 3 ? image.getNormalizedPixel(imgRow, leftCol, 2)
                                                        : leftRed;
            const double rightRed = image.channels >= 3 ? image.getNormalizedPixel(imgRow, rightCol, 0)
                                                        : image.getNormalizedPixel(imgRow, rightCol);
            const double rightGreen = image.channels >= 3 ? image.getNormalizedPixel(imgRow, rightCol, 1)
                                                          : rightRed;
            const double rightBlue = image.channels >= 3 ? image.getNormalizedPixel(imgRow, rightCol, 2)
                                                         : rightRed;
            const double upRed = image.channels >= 3 ? image.getNormalizedPixel(upRow, imgCol, 0)
                                                     : image.getNormalizedPixel(upRow, imgCol);
            const double upGreen = image.channels >= 3 ? image.getNormalizedPixel(upRow, imgCol, 1)
                                                       : upRed;
            const double upBlue = image.channels >= 3 ? image.getNormalizedPixel(upRow, imgCol, 2)
                                                      : upRed;
            const double downRed = image.channels >= 3 ? image.getNormalizedPixel(downRow, imgCol, 0)
                                                       : image.getNormalizedPixel(downRow, imgCol);
            const double downGreen = image.channels >= 3 ? image.getNormalizedPixel(downRow, imgCol, 1)
                                                         : downRed;
            const double downBlue = image.channels >= 3 ? image.getNormalizedPixel(downRow, imgCol, 2)
                                                        : downRed;
            const double leftLum = 0.299 * leftRed + 0.587 * leftGreen + 0.114 * leftBlue;
            const double rightLum = 0.299 * rightRed + 0.587 * rightGreen + 0.114 * rightBlue;
            const double upLum = 0.299 * upRed + 0.587 * upGreen + 0.114 * upBlue;
            const double downLum = 0.299 * downRed + 0.587 * downGreen + 0.114 * downBlue;
            gradientSum += 0.5 * (std::abs(rightLum - luminance) + std::abs(downLum - luminance));
            const double surroundLum = 0.25 * (leftLum + rightLum + upLum + downLum);
            transientOnSum += std::max(0.0, luminance - surroundLum);
            transientOffSum += std::max(0.0, surroundLum - luminance);
        }
    }

    const double redMean = redSum / sampleCount;
    const double greenMean = greenSum / sampleCount;
    const double blueMean = blueSum / sampleCount;
    const double luminanceMean = luminanceSum / sampleCount;
    const double luminanceVar = std::max(0.0, (luminanceSqSum / sampleCount) -
                                                  (luminanceMean * luminanceMean));
    const double luminanceStd = std::sqrt(luminanceVar);
    const double gradientMean = gradientSum / sampleCount;
    double sustainedOnSum = 0.0;
    double sustainedOffSum = 0.0;
    for (double sample : luminanceSamples) {
        sustainedOnSum += std::max(0.0, sample - luminanceMean);
        sustainedOffSum += std::max(0.0, luminanceMean - sample);
    }
    const double sustainedOnMean = sustainedOnSum / sampleCount;
    const double sustainedOffMean = sustainedOffSum / sampleCount;
    const double transientOnMean = transientOnSum / sampleCount;
    const double transientOffMean = transientOffSum / sampleCount;
    if (auxiliaryFeatureMode_ == "luminance_stream_bank" && auxiliary.size() >= 7) {
        auxiliary[0] = std::clamp(luminanceMean * auxiliaryFeatureGain_, 0.0, 1.0);
        auxiliary[1] = std::clamp(2.0 * luminanceStd * auxiliaryFeatureGain_, 0.0, 1.0);
        auxiliary[2] = std::clamp(4.0 * gradientMean * auxiliaryFeatureGain_, 0.0, 1.0);
        auxiliary[3] = std::clamp(2.0 * sustainedOnMean * auxiliaryFeatureGain_, 0.0, 1.0);
        auxiliary[4] = std::clamp(2.0 * sustainedOffMean * auxiliaryFeatureGain_, 0.0, 1.0);
        auxiliary[5] = std::clamp(4.0 * transientOnMean * auxiliaryFeatureGain_, 0.0, 1.0);
        auxiliary[6] = std::clamp(4.0 * transientOffMean * auxiliaryFeatureGain_, 0.0, 1.0);
        return auxiliary;
    }
    const double yellowMean = 0.5 * (redMean + greenMean);
    const double rgOpponent = std::clamp(0.5 + 0.5 * (redMean - greenMean), 0.0, 1.0);
    const double byOpponent = std::clamp(0.5 + 0.5 * (blueMean - yellowMean), 0.0, 1.0);
    const double chroma = std::clamp(
        0.5 * (std::abs(redMean - greenMean) + std::abs(blueMean - yellowMean)), 0.0, 1.0);

    auxiliary[0] = std::clamp(rgOpponent * auxiliaryFeatureGain_, 0.0, 1.0);
    if (auxiliary.size() > 1) {
        auxiliary[1] = std::clamp(byOpponent * auxiliaryFeatureGain_, 0.0, 1.0);
    }
    if (auxiliary.size() > 2) {
        auxiliary[2] = std::clamp(chroma * auxiliaryFeatureGain_, 0.0, 1.0);
    }
    if ((auxiliaryFeatureMode_ == "appearance_bank" ||
         auxiliaryFeatureMode_ == "appearance_stream_bank") &&
        auxiliary.size() >= 6) {
        auxiliary[3] = std::clamp(luminanceMean * auxiliaryFeatureGain_, 0.0, 1.0);
        auxiliary[4] = std::clamp(2.0 * luminanceStd * auxiliaryFeatureGain_, 0.0, 1.0);
        auxiliary[5] = std::clamp(4.0 * gradientMean * auxiliaryFeatureGain_, 0.0, 1.0);
    }
    if (auxiliaryFeatureMode_ == "appearance_stream_bank" && auxiliary.size() >= 10) {
        auxiliary[6] = std::clamp(2.0 * sustainedOnMean * auxiliaryFeatureGain_, 0.0, 1.0);
        auxiliary[7] = std::clamp(2.0 * sustainedOffMean * auxiliaryFeatureGain_, 0.0, 1.0);
        auxiliary[8] = std::clamp(4.0 * transientOnMean * auxiliaryFeatureGain_, 0.0, 1.0);
        auxiliary[9] = std::clamp(4.0 * transientOffMean * auxiliaryFeatureGain_, 0.0, 1.0);
    }
    return auxiliary;
}

/**
 * @brief Extract edge features from a region using pluggable edge operator
 *
 * Delegates edge detection to the configured EdgeOperator (Sobel, Gabor, or DoG).
 * The operator computes edge strength at multiple orientations.
 *
 * Performance comparison (MNIST):
 * - Sobel: 94.63% accuracy (best for sharp edges)
 * - Gabor: 87.20% accuracy (designed for natural images)
 * - DoG: Not yet tested
 *
 * @param region Pixel values of the region (regionSize² elements)
 * @param regionSize Size of the region (width = height)
 * @return Edge strengths for each orientation (numOrientations_ elements, range [0,1])
 */
std::vector<double> RetinaAdapter::extractEdgeFeatures(const std::vector<uint8_t>& region,
                                                        int regionSize) const {
    // Use pluggable edge operator (Sobel, Gabor, or DoG)
    return edgeOperator_->extractEdges(region, regionSize);
}

std::vector<double> RetinaAdapter::featuresToSpikes(const std::vector<double>& features) const {
    return featuresToSpikeTimes(features, temporalWindow_);
}

void RetinaAdapter::applyOrientationCompetition(std::vector<double>& responses) const {
    if (responses.empty()) {
        return;
    }

    if (orientationLateralInhibition_ > 0.0) {
        const double totalEnergy =
            std::accumulate(responses.begin(), responses.end(), 0.0);
        const double denom =
            std::max(1.0, static_cast<double>(responses.size() - 1));
        for (double& response : responses) {
            const double otherMean = (totalEnergy - response) / denom;
            response = std::max(0.0, response - orientationLateralInhibition_ * otherMean);
        }
    }

    if (std::abs(orientationResponseGamma_ - 1.0) > 1e-6) {
        for (double& response : responses) {
            response = std::pow(std::max(0.0, response), orientationResponseGamma_);
        }
    }
}

void RetinaAdapter::recordOrientationFlowDiagnostics(const std::vector<uint8_t>& operatorInputRegion,
                                                     const std::vector<double>& preThreshold,
                                                     const std::vector<double>& thresholded,
                                                     const std::vector<double>& postCompetition) {
    if (!orientationFlowDiagnosticsEnabled_) {
        return;
    }
    if (orientationFlowDiagnosticsSampleLimit_ > 0 &&
        static_cast<int>(orientationFlowDiagnostics_.samples) >=
            orientationFlowDiagnosticsSampleLimit_) {
        return;
    }

    auto summarize = [](const std::vector<double>& values,
                        double& l2Out,
                        double& activeFractionOut,
                        double& maxOut,
                        bool& allZeroOut) {
        double l2 = 0.0;
        int active = 0;
        double maxValue = 0.0;
        for (double value : values) {
            l2 += value * value;
            if (std::abs(value) > 1e-6) {
                active++;
            }
            maxValue = std::max(maxValue, std::abs(value));
        }
        l2Out = std::sqrt(l2);
        activeFractionOut =
            values.empty() ? 0.0 : static_cast<double>(active) / static_cast<double>(values.size());
        maxOut = maxValue;
        allZeroOut = active == 0;
    };

    std::vector<double> manualResponses;
    const size_t regionSide =
        static_cast<size_t>(std::llround(std::sqrt(static_cast<double>(operatorInputRegion.size()))));
    if (regionSide >= 2 && regionSide * regionSide == operatorInputRegion.size()) {
        double horizontal = 0.0;
        double vertical = 0.0;
        double diagDown = 0.0;
        double diagUp = 0.0;
        int hCount = 0;
        int vCount = 0;
        int dCount = 0;
        for (size_t r = 0; r < regionSide; ++r) {
            for (size_t c = 0; c < regionSide; ++c) {
                const double center =
                    static_cast<double>(operatorInputRegion[r * regionSide + c]) / 255.0;
                if (c + 1 < regionSide) {
                    const double right =
                        static_cast<double>(operatorInputRegion[r * regionSide + (c + 1)]) / 255.0;
                    horizontal += std::abs(right - center);
                    hCount++;
                }
                if (r + 1 < regionSide) {
                    const double down =
                        static_cast<double>(operatorInputRegion[(r + 1) * regionSide + c]) / 255.0;
                    vertical += std::abs(down - center);
                    vCount++;
                }
                if (r + 1 < regionSide && c + 1 < regionSide) {
                    const double downRight = static_cast<double>(
                                                 operatorInputRegion[(r + 1) * regionSide + (c + 1)]) /
                                             255.0;
                    diagDown += std::abs(downRight - center);
                    dCount++;
                }
                if (r > 0 && c + 1 < regionSide) {
                    const double upRight =
                        static_cast<double>(operatorInputRegion[(r - 1) * regionSide + (c + 1)]) /
                        255.0;
                    diagUp += std::abs(upRight - center);
                }
            }
        }
        manualResponses = {
            hCount > 0 ? horizontal / static_cast<double>(hCount) : 0.0,
            vCount > 0 ? vertical / static_cast<double>(vCount) : 0.0,
            dCount > 0 ? diagDown / static_cast<double>(dCount) : 0.0,
            dCount > 0 ? diagUp / static_cast<double>(dCount) : 0.0};
    }

    double manualL2 = 0.0;
    double manualActive = 0.0;
    double manualMax = 0.0;
    bool manualZero = true;
    summarize(manualResponses, manualL2, manualActive, manualMax, manualZero);

    double preL2 = 0.0;
    double preActive = 0.0;
    double preMax = 0.0;
    bool preZero = true;
    summarize(preThreshold, preL2, preActive, preMax, preZero);

    double thresholdedL2 = 0.0;
    double thresholdedActive = 0.0;
    double thresholdedMax = 0.0;
    bool thresholdedZero = true;
    summarize(thresholded, thresholdedL2, thresholdedActive, thresholdedMax, thresholdedZero);

    double postL2 = 0.0;
    double postActive = 0.0;
    double postMax = 0.0;
    bool postZero = true;
    summarize(postCompetition, postL2, postActive, postMax, postZero);

    auto& stats = orientationFlowDiagnostics_;
    stats.samples++;
    stats.manualL2Sum += manualL2;
    stats.manualActiveFractionSum += manualActive;
    stats.manualMaxSum += manualMax;
    stats.preThresholdL2Sum += preL2;
    stats.preThresholdActiveFractionSum += preActive;
    stats.preThresholdMaxSum += preMax;
    stats.thresholdedL2Sum += thresholdedL2;
    stats.thresholdedActiveFractionSum += thresholdedActive;
    stats.thresholdedMaxSum += thresholdedMax;
    stats.postCompetitionL2Sum += postL2;
    stats.postCompetitionActiveFractionSum += postActive;
    stats.postCompetitionMaxSum += postMax;
    if (preZero) {
        stats.allZeroBeforeThreshold++;
    }
    if (thresholdedZero) {
        stats.allZeroAfterThreshold++;
    }
    if (postZero) {
        stats.allZeroAfterCompetition++;
    }
    if (manualZero) {
        stats.allZeroManual++;
    }
}

void RetinaAdapter::recordPatchInputDiagnostics(const std::vector<uint8_t>& region) {
    if (!orientationFlowDiagnosticsEnabled_) {
        return;
    }
    if (orientationFlowDiagnosticsSampleLimit_ > 0 &&
        static_cast<int>(patchInputDiagnostics_.samples) >=
            orientationFlowDiagnosticsSampleLimit_) {
        return;
    }
    if (region.empty()) {
        return;
    }

    double minValue = 1.0;
    double maxValue = 0.0;
    double sum = 0.0;
    double sumSq = 0.0;
    double gradientSum = 0.0;
    int gradientCount = 0;
    const size_t size = static_cast<size_t>(std::sqrt(static_cast<double>(region.size())));

    for (size_t i = 0; i < region.size(); ++i) {
        const double value = static_cast<double>(region[i]) / 255.0;
        minValue = std::min(minValue, value);
        maxValue = std::max(maxValue, value);
        sum += value;
        sumSq += value * value;
    }

    if (size > 1 && size * size == region.size()) {
        for (size_t r = 0; r < size; ++r) {
            for (size_t c = 0; c < size; ++c) {
                const double center =
                    static_cast<double>(region[r * size + c]) / 255.0;
                if (c + 1 < size) {
                    const double right =
                        static_cast<double>(region[r * size + (c + 1)]) / 255.0;
                    gradientSum += std::abs(right - center);
                    gradientCount++;
                }
                if (r + 1 < size) {
                    const double down =
                        static_cast<double>(region[(r + 1) * size + c]) / 255.0;
                    gradientSum += std::abs(down - center);
                    gradientCount++;
                }
            }
        }
    }

    const double mean = sum / static_cast<double>(region.size());
    const double variance =
        std::max(0.0, (sumSq / static_cast<double>(region.size())) - (mean * mean));
    const double stddev = std::sqrt(variance);
    const double range = maxValue - minValue;
    const double gradientEnergy =
        gradientCount > 0 ? gradientSum / static_cast<double>(gradientCount) : 0.0;

    auto& stats = patchInputDiagnostics_;
    stats.samples++;
    stats.minValueSum += minValue;
    stats.maxValueSum += maxValue;
    stats.rangeSum += range;
    stats.stddevSum += stddev;
    stats.gradientEnergySum += gradientEnergy;
    if (range <= 0.02) {
        stats.nearFlatRangeCount++;
    }
    if (gradientEnergy <= 0.01) {
        stats.nearFlatGradientCount++;
    }
}

std::vector<double> RetinaAdapter::computeAuxiliaryFeatures(
    const std::vector<double>& orientationResponses,
    const std::vector<uint8_t>& region,
    int regionSize) const {
    std::vector<double> auxiliary(getAuxiliaryChannelCount(), 0.0);
    if (auxiliary.empty() || orientationResponses.size() < 2) {
        return auxiliary;
    }

    if (auxiliaryFeatureMode_ == "corner") {
        double bestCorner = 0.0;
        for (size_t a = 0; a < orientationResponses.size(); ++a) {
            for (size_t b = a + 1; b < orientationResponses.size(); ++b) {
                const size_t deltaBins = std::min(
                    b - a, orientationResponses.size() - (b - a));
                const double deltaDeg =
                    180.0 * static_cast<double>(deltaBins) /
                    static_cast<double>(orientationResponses.size());
                if (deltaDeg < cornerMinDeltaDeg_ || deltaDeg > cornerMaxDeltaDeg_) {
                    continue;
                }
                bestCorner = std::max(
                    bestCorner, orientationResponses[a] * orientationResponses[b]);
            }
        }
        auxiliary[0] = std::clamp(bestCorner * auxiliaryFeatureGain_, 0.0, 1.0);
    } else if (auxiliaryFeatureMode_ == "curve") {
        double bestCurve = 0.0;
        for (size_t a = 0; a < orientationResponses.size(); ++a) {
            for (size_t b = a + 1; b < orientationResponses.size(); ++b) {
                const size_t deltaBins = std::min(
                    b - a, orientationResponses.size() - (b - a));
                const double deltaDeg =
                    180.0 * static_cast<double>(deltaBins) /
                    static_cast<double>(orientationResponses.size());
                if (deltaDeg < curveMinDeltaDeg_ || deltaDeg > curveMaxDeltaDeg_) {
                    continue;
                }
                bestCurve = std::max(
                    bestCurve, orientationResponses[a] * orientationResponses[b]);
            }
        }
        auxiliary[0] = std::clamp(bestCurve * auxiliaryFeatureGain_, 0.0, 1.0);
    } else if (auxiliaryFeatureMode_ == "endstop") {
        const auto bestIt = std::max_element(orientationResponses.begin(), orientationResponses.end());
        const double dominantResponse = *bestIt;
        if (dominantResponse <= 0.0 || region.empty() || regionSize <= 0) {
            return auxiliary;
        }

        const int dominantOrientation = static_cast<int>(std::distance(orientationResponses.begin(), bestIt));
        const double theta = (static_cast<double>(dominantOrientation) * M_PI) /
                             static_cast<double>(orientationResponses.size());
        const double axisX = -std::sin(theta);
        const double axisY = std::cos(theta);
        const double center = 0.5 * static_cast<double>(regionSize - 1);
        const double endThreshold = endstopAxisFraction_ * std::max(1.0, center);

        double posMass = 0.0;
        double negMass = 0.0;
        double centerMass = 0.0;
        double totalMass = 0.0;

        for (int r = 0; r < regionSize; ++r) {
            for (int c = 0; c < regionSize; ++c) {
                const double value = static_cast<double>(region[static_cast<size_t>(r * regionSize + c)]) / 255.0;
                if (value < endstopPixelThreshold_) {
                    continue;
                }

                const double relX = static_cast<double>(c) - center;
                const double relY = static_cast<double>(r) - center;
                const double projection = relX * axisX + relY * axisY;
                totalMass += value;
                if (projection >= endThreshold) {
                    posMass += value;
                } else if (projection <= -endThreshold) {
                    negMass += value;
                } else {
                    centerMass += value;
                }
            }
        }

        if (totalMass > 0.0) {
            const double asymmetry = std::abs(posMass - negMass) / totalMass;
            const double centerSupport = centerMass / totalMass;
            const double endpointScore =
                dominantResponse * asymmetry * (0.5 + 0.5 * centerSupport);
            auxiliary[0] = std::clamp(endpointScore * auxiliaryFeatureGain_, 0.0, 1.0);
        }
    } else if (auxiliaryFeatureMode_ == "closure" ||
               auxiliaryFeatureMode_ == "closure_bank" ||
               auxiliaryFeatureMode_ == "gap_bank") {
        if (regionSize < 3 || region.empty()) {
            return auxiliary;
        }

        std::vector<uint8_t> occupied(static_cast<size_t>(regionSize * regionSize), 0);
        for (int r = 0; r < regionSize; ++r) {
            for (int c = 0; c < regionSize; ++c) {
                const double value =
                    static_cast<double>(region[static_cast<size_t>(r * regionSize + c)]) / 255.0;
                occupied[static_cast<size_t>(r * regionSize + c)] =
                    value >= endstopPixelThreshold_ ? 1 : 0;
            }
        }

        std::vector<uint8_t> visited(static_cast<size_t>(regionSize * regionSize), 0);
        std::vector<std::pair<int, int>> queue;
        queue.reserve(static_cast<size_t>(regionSize * regionSize));
        auto pushIfOpen = [&](int rr, int cc) {
            const size_t idx = static_cast<size_t>(rr * regionSize + cc);
            if (!occupied[idx] && !visited[idx]) {
                visited[idx] = 1;
                queue.push_back({rr, cc});
            }
        };

        for (int c = 0; c < regionSize; ++c) {
            pushIfOpen(0, c);
            pushIfOpen(regionSize - 1, c);
        }
        for (int r = 1; r < regionSize - 1; ++r) {
            pushIfOpen(r, 0);
            pushIfOpen(r, regionSize - 1);
        }

        for (size_t qi = 0; qi < queue.size(); ++qi) {
            const auto [rr, cc] = queue[qi];
            const int dr[4] = {-1, 1, 0, 0};
            const int dc[4] = {0, 0, -1, 1};
            for (int k = 0; k < 4; ++k) {
                const int nr = rr + dr[k];
                const int nc = cc + dc[k];
                if (nr < 0 || nr >= regionSize || nc < 0 || nc >= regionSize) {
                    continue;
                }
                const size_t idx = static_cast<size_t>(nr * regionSize + nc);
                if (!occupied[idx] && !visited[idx]) {
                    visited[idx] = 1;
                    queue.push_back({nr, nc});
                }
            }
        }

        int enclosedCount = 0;
        int openCount = 0;
        int largestHoleArea = 0;
        double largestHoleCenterBias = 0.0;
        std::vector<uint8_t> holeSeen(static_cast<size_t>(regionSize * regionSize), 0);
        for (int r = 0; r < regionSize; ++r) {
            for (int c = 0; c < regionSize; ++c) {
                const size_t idx = static_cast<size_t>(r * regionSize + c);
                if (!occupied[idx]) {
                    if (visited[idx]) {
                        openCount++;
                    } else {
                        enclosedCount++;
                        if (!holeSeen[idx]) {
                            std::vector<std::pair<int, int>> holeQueue;
                            holeQueue.push_back({r, c});
                            holeSeen[idx] = 1;
                            int holeArea = 0;
                            double holeRowSum = 0.0;
                            double holeColSum = 0.0;
                            for (size_t qi = 0; qi < holeQueue.size(); ++qi) {
                                const auto [hr, hc] = holeQueue[qi];
                                holeArea++;
                                holeRowSum += static_cast<double>(hr);
                                holeColSum += static_cast<double>(hc);
                                const int dr[4] = {-1, 1, 0, 0};
                                const int dc[4] = {0, 0, -1, 1};
                                for (int k = 0; k < 4; ++k) {
                                    const int nr = hr + dr[k];
                                    const int nc = hc + dc[k];
                                    if (nr < 0 || nr >= regionSize || nc < 0 || nc >= regionSize) {
                                        continue;
                                    }
                                    const size_t nidx = static_cast<size_t>(nr * regionSize + nc);
                                    if (!occupied[nidx] && !visited[nidx] && !holeSeen[nidx]) {
                                        holeSeen[nidx] = 1;
                                        holeQueue.push_back({nr, nc});
                                    }
                                }
                            }
                            if (holeArea > largestHoleArea) {
                                largestHoleArea = holeArea;
                                const double center = 0.5 * static_cast<double>(regionSize - 1);
                                const double holeRow = holeRowSum / static_cast<double>(holeArea);
                                const double holeCol = holeColSum / static_cast<double>(holeArea);
                                const double dist =
                                    std::sqrt((holeRow - center) * (holeRow - center) +
                                              (holeCol - center) * (holeCol - center));
                                const double maxDist = std::sqrt(2.0) * center;
                                largestHoleCenterBias =
                                    maxDist > 0.0 ? std::clamp(1.0 - dist / maxDist, 0.0, 1.0) : 1.0;
                            }
                        }
                    }
                }
            }
        }

        const int backgroundCount = enclosedCount + openCount;
        if (backgroundCount > 0) {
            const double enclosedRatio =
                static_cast<double>(enclosedCount) / static_cast<double>(backgroundCount);
            const double support =
                *std::max_element(orientationResponses.begin(), orientationResponses.end());
            if (auxiliaryFeatureMode_ == "closure") {
                auxiliary[0] = std::clamp(enclosedRatio * support * auxiliaryFeatureGain_, 0.0, 1.0);
            } else if (auxiliaryFeatureMode_ == "closure_bank") {
                const double largestHoleRatio =
                    static_cast<double>(largestHoleArea) /
                    static_cast<double>(std::max(1, backgroundCount));
                auxiliary[0] = std::clamp((enclosedCount > 0 ? support : 0.0) * auxiliaryFeatureGain_, 0.0, 1.0);
                auxiliary[1] = std::clamp(largestHoleRatio * support * auxiliaryFeatureGain_, 0.0, 1.0);
                auxiliary[2] = std::clamp(largestHoleRatio * largestHoleCenterBias *
                                              support * auxiliaryFeatureGain_,
                                          0.0, 1.0);
            } else if (auxiliaryFeatureMode_ == "gap_bank") {
                double topOpen = 0.0;
                double rightOpen = 0.0;
                double bottomOpen = 0.0;
                double leftOpen = 0.0;
                for (int c = 0; c < regionSize; ++c) {
                    topOpen += visited[static_cast<size_t>(c)] ? 1.0 : 0.0;
                    bottomOpen += visited[static_cast<size_t>((regionSize - 1) * regionSize + c)] ? 1.0 : 0.0;
                }
                for (int r = 0; r < regionSize; ++r) {
                    leftOpen += visited[static_cast<size_t>(r * regionSize)] ? 1.0 : 0.0;
                    rightOpen += visited[static_cast<size_t>(r * regionSize + (regionSize - 1))] ? 1.0 : 0.0;
                }
                const double sideNorm = static_cast<double>(regionSize);
                auxiliary[0] = std::clamp((topOpen / sideNorm) * support * auxiliaryFeatureGain_, 0.0, 1.0);
                auxiliary[1] = std::clamp((rightOpen / sideNorm) * support * auxiliaryFeatureGain_, 0.0, 1.0);
                auxiliary[2] = std::clamp((bottomOpen / sideNorm) * support * auxiliaryFeatureGain_, 0.0, 1.0);
                auxiliary[3] = std::clamp((leftOpen / sideNorm) * support * auxiliaryFeatureGain_, 0.0, 1.0);
            }
        }
    }

    return auxiliary;
}

void RetinaAdapter::applyHomeostaticScaling(std::vector<double>& features) {
    if (!homeostaticScalingEnabled_ || features.empty()) {
        return;
    }

    if (featureHomeostaticGains_.size() != features.size()) {
        featureHomeostaticGains_.assign(features.size(), 1.0);
        featureActivityAverages_.assign(features.size(), homeostaticTargetActivation_);
    }

    for (size_t i = 0; i < features.size(); ++i) {
        features[i] = std::clamp(features[i] * featureHomeostaticGains_[i], 0.0, 1.0);
    }

    if (!homeostaticLearningEnabled_) {
        return;
    }

    const double decay = homeostaticActivityDecay_;
    const double mix = 1.0 - decay;
    for (size_t i = 0; i < features.size(); ++i) {
        featureActivityAverages_[i] =
            decay * featureActivityAverages_[i] + mix * features[i];
        const double error = homeostaticTargetActivation_ - featureActivityAverages_[i];
        const double nextGain =
            featureHomeostaticGains_[i] * (1.0 + homeostaticLearningRate_ * error);
        featureHomeostaticGains_[i] =
            std::clamp(nextGain, homeostaticGainMin_, homeostaticGainMax_);
    }
}

void RetinaAdapter::applySensoryTripletBcmPlasticity(std::vector<double>& features) {
    if (!sensoryTripletBcmEnabled_ || features.empty()) {
        return;
    }

    const double target = std::max(1e-6, sensoryBcmTargetActivation_);
    if (sensoryTripletBcmGains_.size() != features.size()) {
        sensoryTripletBcmGains_.assign(features.size(), 1.0);
        sensoryTripletFastTrace_.assign(features.size(), target);
        sensoryTripletSlowTrace_.assign(features.size(), target);
        sensoryBcmThresholds_.assign(features.size(), target);
    }

    for (size_t i = 0; i < features.size(); ++i) {
        features[i] = std::clamp(features[i] * sensoryTripletBcmGains_[i], 0.0, 1.0);
    }

    if (!homeostaticLearningEnabled_) {
        return;
    }

    const double fastMix = 1.0 - sensoryTripletFastDecay_;
    const double slowMix = 1.0 - sensoryTripletSlowDecay_;
    const double thresholdMix = 1.0 - sensoryBcmThresholdDecay_;
    for (size_t i = 0; i < features.size(); ++i) {
        const double activity = features[i];
        sensoryTripletFastTrace_[i] =
            (sensoryTripletFastDecay_ * sensoryTripletFastTrace_[i]) + (fastMix * activity);
        sensoryTripletSlowTrace_[i] =
            (sensoryTripletSlowDecay_ * sensoryTripletSlowTrace_[i]) + (slowMix * activity);
        const double bcmEstimate =
            std::clamp((sensoryTripletSlowTrace_[i] * sensoryTripletSlowTrace_[i]) / target,
                       0.0,
                       1.0);
        sensoryBcmThresholds_[i] =
            (sensoryBcmThresholdDecay_ * sensoryBcmThresholds_[i]) +
            (thresholdMix * bcmEstimate);

        const double threshold = sensoryBcmThresholds_[i];
        const double depolarization = std::max(0.0, activity - threshold);
        const double traceSupport = std::max(0.0, sensoryTripletFastTrace_[i] - threshold);
        const double ltp =
            sensoryTripletBcmLtp_ * depolarization * traceSupport *
            (0.5 + sensoryTripletSlowTrace_[i]);
        const double ltd =
            sensoryTripletBcmLtd_ * activity *
            std::max(0.0, threshold - sensoryTripletFastTrace_[i]);
        const double homeostaticPull = 0.5 * (target - sensoryTripletSlowTrace_[i]);
        const double delta =
            sensoryTripletBcmLearningRate_ * (ltp - ltd + homeostaticPull);
        const double nextGain = sensoryTripletBcmGains_[i] * (1.0 + delta);
        sensoryTripletBcmGains_[i] =
            std::clamp(nextGain, sensoryTripletBcmGainMin_, sensoryTripletBcmGainMax_);
    }
}

SensoryAdapter::SpikePattern RetinaAdapter::processData(const DataSample& data) {
    // Convert raw data to image
    Image image;
    image.pixels = data.rawData;
    image.channels = std::max(1, data.channels);
    
    // Infer image dimensions if not set
    if (data.rows > 0 && data.cols > 0) {
        imageRows_ = data.rows;
        imageCols_ = data.cols;
        imageChannels_ = std::max(1, data.channels);
        regionSize_ = std::max(minimumRegionSize_, std::max(1, imageRows_ / gridSize_));
    } else if (imageRows_ == 0 || imageCols_ == 0) {
        int totalPixels = static_cast<int>(data.rawData.size());
        int inferredChannels = std::max(1, data.channels);
        int pixelsPerChannel = totalPixels / inferredChannels;
        imageRows_ = imageCols_ = static_cast<int>(std::sqrt(std::max(1, pixelsPerChannel)));
        imageChannels_ = inferredChannels;
        regionSize_ = std::max(minimumRegionSize_, std::max(1, imageRows_ / gridSize_));
        
        SNNFW_INFO("RetinaAdapter '{}': inferred image size {}x{}x{}, region size {}",
                   getName(), imageRows_, imageCols_, imageChannels_, regionSize_);
    }
    
    image.rows = imageRows_;
    image.cols = imageCols_;
    image.channels = imageChannels_;
    
    // Extract features and encode as spikes
    auto features = extractFeatures(data);
    return encodeFeatures(features);
}

SensoryAdapter::FeatureVector RetinaAdapter::extractFeatures(const DataSample& data) {
    FeatureVector result;
    result.timestamp = data.timestamp;
    
    // Convert to image
    Image image;
    image.pixels = data.rawData;
    image.rows = (data.rows > 0 ? data.rows : imageRows_);
    image.cols = (data.cols > 0 ? data.cols : imageCols_);
    image.channels = std::max(1, data.channels > 0 ? data.channels : imageChannels_);
    image = applyViewTransform(image);
    auto bandImageSet = buildRelayBandImages(image);
    auto& bandImages = bandImageSet.orientationBands;
    auto& auxiliaryBandImages = bandImageSet.auxiliaryBands;
    const size_t frequencyBandCount = bandImages.size();
    const size_t auxiliaryChannels = getAuxiliaryChannelCount();
    const size_t regionFeatureCount = getChannelsPerBand() * frequencyBandCount;
    const size_t subfieldCount = getSubfieldCount();
    const size_t colorEdgeChannels = getColorEdgeChannelCount();
    const size_t orientationBlocks = getOrientationChannelBlocks();
    const size_t orientationFeatureCount =
        static_cast<size_t>(numOrientations_) * orientationBlocks * colorEdgeChannels;
    std::vector<TopologyMaps> topologyMaps;
    if (auxiliaryFeatureMode_ == "topology_maps") {
        topologyMaps.reserve(frequencyBandCount);
        for (const auto& bandImage : bandImages) {
            topologyMaps.push_back(computeTopologyMaps(bandImage, endstopPixelThreshold_));
        }
    }
    std::vector<ContourGraphMaps> contourGraphMaps;
    if (auxiliaryFeatureMode_ == "contour_graph") {
        contourGraphMaps.reserve(frequencyBandCount);
        for (const auto& bandImage : bandImages) {
            contourGraphMaps.push_back(computeContourGraphMaps(bandImage, endstopPixelThreshold_));
        }
    }
    std::vector<ContourSequenceMaps> contourSequenceMaps;
    if (auxiliaryFeatureMode_ == "contour_sequence") {
        contourSequenceMaps.reserve(frequencyBandCount);
        for (const auto& bandImage : bandImages) {
            contourSequenceMaps.push_back(
                computeContourSequenceMaps(bandImage, endstopPixelThreshold_, 8));
        }
    }
    const int pooledEdgeRegionSize =
        std::max(regionSize_, edgeAnalysisRegionSize_ > 0 ? edgeAnalysisRegionSize_ : regionSize_);
    int analysisRegionSize = pooledEdgeRegionSize;
    if (subfieldCount > 0) {
        analysisRegionSize =
            std::max(analysisRegionSize, pooledEdgeRegionSize * subfieldGridSize_);
        analysisRegionSize = std::max(analysisRegionSize, minimumRegionSize_ * subfieldGridSize_);
        if (analysisRegionSize % subfieldGridSize_ != 0) {
            analysisRegionSize += subfieldGridSize_ - (analysisRegionSize % subfieldGridSize_);
        }
    }
    const auto rowBoundaries = computeAxisSamplingBoundaries(image.rows);
    const auto colBoundaries = computeAxisSamplingBoundaries(image.cols);

    auto buildColorEdgeRegions = [&](const Image& sourceImage,
                                     int regionRow,
                                     int regionCol,
                                     int targetSize) {
        const int effectiveSize = std::max(1, targetSize);
        std::vector<std::vector<uint8_t>> regions(
            colorEdgeChannels,
            std::vector<uint8_t>(static_cast<size_t>(effectiveSize * effectiveSize), 128));
        const int safeRow = clampIndex(regionRow, 0, std::max(0, gridSize_ - 1));
        const int safeCol = clampIndex(regionCol, 0, std::max(0, gridSize_ - 1));
        const int startRow = rowBoundaries[static_cast<size_t>(safeRow)];
        const int endRow = rowBoundaries[static_cast<size_t>(safeRow + 1)];
        const int startCol = colBoundaries[static_cast<size_t>(safeCol)];
        const int endCol = colBoundaries[static_cast<size_t>(safeCol + 1)];
        const int sourceHeight = std::max(1, endRow - startRow);
        const int sourceWidth = std::max(1, endCol - startCol);

        for (int r = 0; r < effectiveSize; ++r) {
            for (int c = 0; c < effectiveSize; ++c) {
                const int localRow = clampIndex(
                    static_cast<int>(((static_cast<double>(r) + 0.5) * sourceHeight) /
                                     static_cast<double>(effectiveSize)),
                    0, sourceHeight - 1);
                const int localCol = clampIndex(
                    static_cast<int>(((static_cast<double>(c) + 0.5) * sourceWidth) /
                                     static_cast<double>(effectiveSize)),
                    0, sourceWidth - 1);
                const int imgRow = startRow + localRow;
                const int imgCol = startCol + localCol;
                const size_t idx = static_cast<size_t>(r * effectiveSize + c);

                regions[0][idx] = sourceImage.getPixel(imgRow, imgCol);
                if (colorEdgeChannels >= 3 && sourceImage.channels >= 3) {
                    const int red = static_cast<int>(sourceImage.getPixel(imgRow, imgCol, 0));
                    const int green = static_cast<int>(sourceImage.getPixel(imgRow, imgCol, 1));
                    const int blue = static_cast<int>(sourceImage.getPixel(imgRow, imgCol, 2));
                    regions[1][idx] = static_cast<uint8_t>(
                        std::clamp(128 + (red - green) / 2, 0, 255));
                    regions[2][idx] = static_cast<uint8_t>(
                        std::clamp(128 + ((2 * blue) - red - green) / 4, 0, 255));
                }
            }
        }
        return regions;
    };
    
    // Extract features for each region
    for (int row = 0; row < gridSize_; ++row) {
        for (int col = 0; col < gridSize_; ++col) {
            const double regionMaskWeight = computeRegionMaskWeight(row, col);
            if (regionMaskWeight <= 1e-6) {
                result.features.insert(result.features.end(), regionFeatureCount, 0.0);
                continue;
            }
            std::vector<std::vector<double>> bandFeatures(
                frequencyBandCount, std::vector<double>(orientationFeatureCount, 0.0));
            std::vector<std::vector<double>> auxiliaryFeatures(
                frequencyBandCount, std::vector<double>(auxiliaryChannels, 0.0));
            for (size_t bandIdx = 0; bandIdx < frequencyBandCount; ++bandIdx) {
                const auto pooledRegions =
                    buildColorEdgeRegions(bandImages[bandIdx], row, col, pooledEdgeRegionSize);
                const auto& pooledRegionRaw = pooledRegions[0];
                recordPatchInputDiagnostics(pooledRegionRaw);
                const auto pooledRegion = normalizeEdgeRegion(pooledRegionRaw);
                auto pooledResponses = extractEdgeFeatures(pooledRegion, pooledEdgeRegionSize);
                auto thresholdedResponses = pooledResponses;
                std::vector<double> preThresholdResponses;
                if (diagnosticEdgeOperator_) {
                    preThresholdResponses =
                        diagnosticEdgeOperator_->extractEdges(pooledRegion, pooledEdgeRegionSize);
                }
                applyOrientationCompetition(pooledResponses);
                if (!preThresholdResponses.empty()) {
                    recordOrientationFlowDiagnostics(
                        pooledRegion, preThresholdResponses, thresholdedResponses, pooledResponses);
                }
                size_t writeOffset = 0;

                if (subfieldCount == 0 || subfieldIncludePooled_) {
                    for (size_t colorIdx = 0; colorIdx < colorEdgeChannels; ++colorIdx) {
                        std::vector<uint8_t> normalizedColorRegion;
                        if (colorIdx != 0) {
                            recordPatchInputDiagnostics(pooledRegions[colorIdx]);
                            normalizedColorRegion = normalizeEdgeRegion(pooledRegions[colorIdx]);
                        }
                        std::vector<double> responses =
                            (colorIdx == 0) ? pooledResponses
                                            : extractEdgeFeatures(normalizedColorRegion, pooledEdgeRegionSize);
                        if (colorIdx != 0) {
                            auto colorThresholdedResponses = responses;
                            std::vector<double> colorPreThresholdResponses;
                            if (diagnosticEdgeOperator_) {
                                colorPreThresholdResponses =
                                    diagnosticEdgeOperator_->extractEdges(
                                        normalizedColorRegion, pooledEdgeRegionSize);
                            }
                            applyOrientationCompetition(responses);
                            if (!colorPreThresholdResponses.empty()) {
                                recordOrientationFlowDiagnostics(
                                    normalizedColorRegion,
                                    colorPreThresholdResponses,
                                    colorThresholdedResponses,
                                    responses);
                            }
                        }
                        std::copy(responses.begin(), responses.end(),
                                  bandFeatures[bandIdx].begin() +
                                      static_cast<std::ptrdiff_t>(writeOffset));
                        writeOffset += static_cast<size_t>(numOrientations_);
                    }
                }

                if (subfieldCount > 0) {
                    const auto analysisRegions =
                        buildColorEdgeRegions(bandImages[bandIdx], row, col, analysisRegionSize);
                    const int subfieldSize = analysisRegionSize / subfieldGridSize_;
                    for (int subRow = 0; subRow < subfieldGridSize_; ++subRow) {
                        for (int subCol = 0; subCol < subfieldGridSize_; ++subCol) {
                            for (size_t colorIdx = 0; colorIdx < colorEdgeChannels; ++colorIdx) {
                                std::vector<uint8_t> subRegion(
                                    static_cast<size_t>(subfieldSize * subfieldSize));
                                for (int r = 0; r < subfieldSize; ++r) {
                                    for (int c = 0; c < subfieldSize; ++c) {
                                        const int srcRow = subRow * subfieldSize + r;
                                        const int srcCol = subCol * subfieldSize + c;
                                        subRegion[static_cast<size_t>(r * subfieldSize + c)] =
                                            analysisRegions[colorIdx][static_cast<size_t>(
                                                srcRow * analysisRegionSize + srcCol)];
                                    }
                                }
                                recordPatchInputDiagnostics(subRegion);
                                auto normalizedSubRegion = normalizeEdgeRegion(subRegion);
                                auto subResponses = extractEdgeFeatures(normalizedSubRegion, subfieldSize);
                                auto subThresholdedResponses = subResponses;
                                std::vector<double> subPreThresholdResponses;
                                if (diagnosticEdgeOperator_) {
                                    subPreThresholdResponses =
                                        diagnosticEdgeOperator_->extractEdges(
                                            normalizedSubRegion, subfieldSize);
                                }
                                applyOrientationCompetition(subResponses);
                                if (!subPreThresholdResponses.empty()) {
                                    recordOrientationFlowDiagnostics(
                                        normalizedSubRegion,
                                        subPreThresholdResponses,
                                        subThresholdedResponses,
                                        subResponses);
                                }
                                std::copy(subResponses.begin(), subResponses.end(),
                                          bandFeatures[bandIdx].begin() +
                                              static_cast<std::ptrdiff_t>(writeOffset));
                                writeOffset += static_cast<size_t>(numOrientations_);
                            }
                        }
                    }
                }
                auto auxiliaryRegion = pooledRegionRaw;
                int auxiliaryRegionSize = regionSize_;
                if (auxiliaryFeatureMode_ == "topology_maps") {
                    auxiliaryFeatures[bandIdx] =
                        poolTopologyFeatures(topologyMaps[bandIdx], gridSize_, row, col,
                                             auxiliaryFeatureGain_);
                } else if (auxiliaryFeatureMode_ == "contour_graph") {
                    auxiliaryFeatures[bandIdx] =
                        poolContourGraphFeatures(contourGraphMaps[bandIdx], gridSize_, row, col,
                                                 auxiliaryFeatureGain_);
                } else if (auxiliaryFeatureMode_ == "contour_sequence") {
                    auxiliaryFeatures[bandIdx] =
                        poolContourSequenceFeatures(contourSequenceMaps[bandIdx], gridSize_,
                                                    row, col, auxiliaryFeatureGain_);
                } else if (auxiliaryFeatureMode_ == "contour_support_bank") {
                    auxiliaryFeatures[bandIdx].assign(auxiliaryChannels, 0.0);
                } else if (auxiliaryFeatureMode_ == "color_opponent" ||
                           auxiliaryFeatureMode_ == "appearance_bank" ||
                           auxiliaryFeatureMode_ == "appearance_stream_bank" ||
                           auxiliaryFeatureMode_ == "luminance_stream_bank") {
                    auxiliaryFeatures[bandIdx] =
                        computeColorOpponentFeatures(auxiliaryBandImages[bandIdx], row, col,
                                                     auxiliaryAnalysisRegionSize_ > regionSize_
                                                         ? auxiliaryAnalysisRegionSize_
                                                         : regionSize_);
                } else {
                    if (auxiliaryAnalysisRegionSize_ > regionSize_) {
                        auxiliaryRegion = extractRegion(
                            auxiliaryBandImages[bandIdx], row, col, auxiliaryAnalysisRegionSize_);
                        auxiliaryRegionSize = auxiliaryAnalysisRegionSize_;
                    }
                    auxiliaryFeatures[bandIdx] = computeAuxiliaryFeatures(
                        pooledResponses, auxiliaryRegion, auxiliaryRegionSize);
                }
            }

            applyLuminanceOnOffBranchSplit(bandFeatures, auxiliaryFeatures);
            applyTemporalStreamBranchSplit(bandFeatures, auxiliaryFeatures);
            applyTemporalCoarseToFineDualPass(bandFeatures, auxiliaryFeatures);
            applyComplexCellStage(bandFeatures);
            applyContourSupportBank(bandFeatures, auxiliaryFeatures);

            for (size_t orientChannel = 0; orientChannel < orientationFeatureCount; ++orientChannel) {
                std::vector<std::pair<double, size_t>> ranked;
                ranked.reserve(frequencyBandCount);
                for (size_t bandIdx = 0; bandIdx < frequencyBandCount; ++bandIdx) {
                    ranked.push_back({bandFeatures[bandIdx][orientChannel], bandIdx});
                }
                std::sort(ranked.begin(), ranked.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });

                std::vector<double> selected(frequencyBandCount, 0.0);
                int kept = 0;
                for (const auto& entry : ranked) {
                    if (entry.first <= 0.0) {
                        break;
                    }
                    selected[entry.second] = entry.first;
                    kept++;
                    if (kept >= maxFrequencyBandsPerFeature_) {
                        break;
                    }
                }
                for (double feature : selected) {
                    result.features.push_back(
                        std::clamp(feature * orientationFeatureGain_ * regionMaskWeight, 0.0, 1.0));
                }
            }

            for (size_t auxIdx = 0; auxIdx < auxiliaryChannels; ++auxIdx) {
                for (size_t bandIdx = 0; bandIdx < frequencyBandCount; ++bandIdx) {
                    result.features.push_back(
                        std::clamp(auxiliaryFeatures[bandIdx][auxIdx] * regionMaskWeight, 0.0, 1.0));
                }
            }
        }
    }

    applyHomeostaticScaling(result.features);
    applySensoryTripletBcmPlasticity(result.features);

    return result;
}

SensoryAdapter::SpikePattern RetinaAdapter::encodeFeatures(const FeatureVector& features) {
    SpikePattern pattern;
    pattern.timestamp = features.timestamp;
    pattern.duration = temporalWindow_;
    pattern.spikeTimes.resize(neurons_.size());

    // Clear all neurons first
    clearNeuronStates();

    // Get neurons per feature from encoding strategy
    int neuronsPerFeature = encodingStrategy_->getNeuronsPerFeature();
    const size_t frequencyBandCount = std::max<size_t>(1, getFrequencyBandCount());
    const int channelsPerBand = static_cast<int>(getChannelsPerBand());

    // Encode features as spikes and insert into neurons
    size_t featureIdx = 0;
    for (int row = 0; row < gridSize_; ++row) {
        for (int col = 0; col < gridSize_; ++col) {
            for (int channel = 0; channel < channelsPerBand; ++channel) {
                for (size_t bandIdx = 0; bandIdx < frequencyBandCount; ++bandIdx) {
                    const double featureValue = features.features[featureIdx++];

                    // Use encoding strategy to generate spike times
                    std::vector<double> spikeTimes = encodingStrategy_->encode(
                        featureValue,
                        static_cast<int>((channel * static_cast<int>(frequencyBandCount)) +
                                         static_cast<int>(bandIdx)));

                    const size_t channelIdx =
                        static_cast<size_t>(channel) * frequencyBandCount + bandIdx;

                    // For rate/temporal encoding (1 neuron per feature), insert into single neuron
                    if (neuronsPerFeature == 1) {
                        auto& neuron = neuronGrid_[row * gridSize_ + col][channelIdx];
                        const int neuronIdx =
                            ((row * gridSize_ + col) * channelsPerBand *
                             static_cast<int>(frequencyBandCount)) +
                            (channel * static_cast<int>(frequencyBandCount)) +
                            static_cast<int>(bandIdx);

                        for (double spikeTime : spikeTimes) {
                            neuron->insertSpike(spikeTime);
                            pattern.spikeTimes[static_cast<size_t>(neuronIdx)].push_back(spikeTime);
                        }
                    } else {
                        // Population encoding is still mapped to a single logical channel here.
                        auto& neuron = neuronGrid_[row * gridSize_ + col][channelIdx];
                        const int neuronIdx =
                            ((row * gridSize_ + col) * channelsPerBand *
                             static_cast<int>(frequencyBandCount)) +
                            (channel * static_cast<int>(frequencyBandCount)) +
                            static_cast<int>(bandIdx);

                        for (double spikeTime : spikeTimes) {
                            neuron->insertSpike(spikeTime);
                            pattern.spikeTimes[static_cast<size_t>(neuronIdx)].push_back(spikeTime);
                        }
                    }
                }
            }
        }
    }

    return pattern;
}

std::vector<double> RetinaAdapter::getActivationPattern() const {
    std::vector<double> activations(neurons_.size(), 0.0);

    for (size_t i = 0; i < neurons_.size(); ++i) {
        const bool hasSpikes = !neurons_[i]->getSpikes().empty();
        const bool hasMemory = neurons_[i]->getLearnedPatternCount() > 0;
        const double similarity = hasMemory ? neurons_[i]->getBestSimilarity() : 0.0;

        if (activationMode_ == "similarity") {
            activations[i] = similarity;
        } else if (activationMode_ == "hybrid") {
            activations[i] = hasMemory ? similarity : (hasSpikes ? 1.0 : 0.0);
        } else {
            activations[i] = hasSpikes ? 1.0 : 0.0;
        }
    }

    return activations;
}

void RetinaAdapter::clearNeuronStates() {
    for (auto& neuron : neurons_) {
        neuron->clearSpikes();
    }
}

std::shared_ptr<Neuron> RetinaAdapter::getNeuronAt(int row, int col, int orientation) const {
    if (row < 0 || row >= gridSize_ || col < 0 || col >= gridSize_ ||
        orientation < 0 || orientation >= numOrientations_) {
        return nullptr;
    }

    return neuronGrid_[row * gridSize_ + col][static_cast<size_t>(orientation) * getFrequencyBandCount()];
}

std::vector<double> RetinaAdapter::processImage(const Image& image) {
    static int callCount = 0;
    callCount++;

    DataSample sample;
    sample.rawData = image.pixels;
    sample.timestamp = 0.0;
    sample.rows = image.rows;
    sample.cols = image.cols;
    sample.channels = image.channels;

    // Set image dimensions
    imageRows_ = image.rows;
    imageCols_ = image.cols;
    imageChannels_ = std::max(1, image.channels);
    regionSize_ = std::max(minimumRegionSize_, std::max(1, imageRows_ / gridSize_));

    if (callCount <= 6) SNNFW_INFO("processImage call #{}: About to call processData()", callCount);
    // Process and get activation pattern
    processData(sample);
    if (callCount <= 6) SNNFW_INFO("processImage call #{}: processData() done, calling getActivationPattern()", callCount);
    auto result = getActivationPattern();
    if (callCount <= 6) SNNFW_INFO("processImage call #{}: getActivationPattern() done", callCount);
    return result;
}

} // namespace adapters
} // namespace snnfw

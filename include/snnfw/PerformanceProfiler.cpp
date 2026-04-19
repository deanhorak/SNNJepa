#include "snnfw/PerformanceProfiler.h"
#include "snnfw/Logger.h"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <nlohmann/json.hpp>

namespace snnfw {

// ============================================================================
// ScopedTimer Implementation
// ============================================================================

ScopedTimer::ScopedTimer(const std::string& name, PerformanceProfiler& profiler)
    : operationName_(name), profiler_(&profiler), 
      startTime_(std::chrono::steady_clock::now()), stopped_(false) {
}

ScopedTimer::~ScopedTimer() {
    if (!stopped_ && profiler_) {
        stop();
    }
}

ScopedTimer::ScopedTimer(ScopedTimer&& other) noexcept
    : operationName_(std::move(other.operationName_)),
      profiler_(other.profiler_),
      startTime_(other.startTime_),
      stopped_(other.stopped_) {
    other.profiler_ = nullptr;
    other.stopped_ = true;
}

ScopedTimer& ScopedTimer::operator=(ScopedTimer&& other) noexcept {
    if (this != &other) {
        if (!stopped_ && profiler_) {
            stop();
        }
        operationName_ = std::move(other.operationName_);
        profiler_ = other.profiler_;
        startTime_ = other.startTime_;
        stopped_ = other.stopped_;
        other.profiler_ = nullptr;
        other.stopped_ = true;
    }
    return *this;
}

void ScopedTimer::stop() {
    if (!stopped_ && profiler_) {
        profiler_->endOperation(operationName_);
        stopped_ = true;
    }
}

// ============================================================================
// PerformanceProfiler Implementation
// ============================================================================

PerformanceProfiler::PerformanceProfiler()
    : profiling_(false), totalSpikesProcessed_(0), nextCallbackId_(1) {
    SNNFW_INFO("PerformanceProfiler created");
}

PerformanceProfiler::~PerformanceProfiler() {
    if (profiling_) {
        stopProfiling();
    }
    SNNFW_INFO("PerformanceProfiler destroyed");
}

void PerformanceProfiler::startProfiling() {
    if (profiling_) {
        SNNFW_WARN("PerformanceProfiler already profiling");
        return;
    }
    
    profiling_ = true;
    profilingStartTime_ = std::chrono::steady_clock::now();
    metrics_.clear();
    activeOperations_.clear();
    totalSpikesProcessed_ = 0;
    
    SNNFW_INFO("PerformanceProfiler started");
}

void PerformanceProfiler::stopProfiling() {
    if (!profiling_) {
        return;
    }
    
    profiling_ = false;
    profilingEndTime_ = std::chrono::steady_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        profilingEndTime_ - profilingStartTime_).count();
    
    SNNFW_INFO("PerformanceProfiler stopped - profiled for {}ms, {} operations tracked",
               duration, metrics_.size());
}

void PerformanceProfiler::startOperation(const std::string& operationName,
                                        uint64_t componentId,
                                        const std::string& componentType) {
    if (!profiling_) {
        return;
    }
    
    activeOperations_[operationName] = std::chrono::steady_clock::now();
    
    // Initialize metrics if this is the first time
    if (metrics_.find(operationName) == metrics_.end()) {
        metrics_[operationName].name = operationName;
        metrics_[operationName].componentId = componentId;
        metrics_[operationName].componentType = componentType;
    }
}

void PerformanceProfiler::endOperation(const std::string& operationName) {
    if (!profiling_) {
        return;
    }
    
    auto it = activeOperations_.find(operationName);
    if (it == activeOperations_.end()) {
        SNNFW_WARN("PerformanceProfiler: endOperation called for '{}' without matching startOperation",
                   operationName);
        return;
    }
    
    auto endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        endTime - it->second).count() / 1000.0;  // Convert to milliseconds
    
    // Get component info from existing metrics
    uint64_t componentId = 0;
    std::string componentType;
    if (metrics_.find(operationName) != metrics_.end()) {
        componentId = metrics_[operationName].componentId;
        componentType = metrics_[operationName].componentType;
    }
    
    recordOperation(operationName, duration, componentId, componentType);
    activeOperations_.erase(it);
    
    // Check for performance alerts
    checkAlerts(operationName, duration);
}

ScopedTimer PerformanceProfiler::startTimer(const std::string& operationName,
                                           uint64_t componentId,
                                           const std::string& componentType) {
    startOperation(operationName, componentId, componentType);
    return ScopedTimer(operationName, *this);
}

void PerformanceProfiler::recordSpikesProcessed(size_t count) {
    if (!profiling_) {
        return;
    }
    
    totalSpikesProcessed_ += count;
}

void PerformanceProfiler::recordMemoryUsage(size_t bytes, const std::string& operationName) {
    if (!profiling_) {
        return;
    }
    
    if (operationName.empty()) {
        // Record system-wide memory
        for (auto& pair : metrics_) {
            pair.second.currentMemoryBytes = bytes;
            if (bytes > pair.second.peakMemoryBytes) {
                pair.second.peakMemoryBytes = bytes;
            }
        }
    } else {
        // Record for specific operation
        if (metrics_.find(operationName) != metrics_.end()) {
            metrics_[operationName].currentMemoryBytes = bytes;
            if (bytes > metrics_[operationName].peakMemoryBytes) {
                metrics_[operationName].peakMemoryBytes = bytes;
            }
        }
    }
}

PerformanceMetrics PerformanceProfiler::getMetrics(const std::string& operationName) const {
    auto it = metrics_.find(operationName);
    if (it != metrics_.end()) {
        return it->second;
    }
    return PerformanceMetrics();
}

std::map<std::string, PerformanceMetrics> PerformanceProfiler::getAllMetrics() const {
    return metrics_;
}

PerformanceSnapshot PerformanceProfiler::getSnapshot(double startTimeMs, double endTimeMs) const {
    PerformanceSnapshot snapshot;
    snapshot.startTime = startTimeMs;
    snapshot.endTime = endTimeMs;
    snapshot.wallClockTimeMs = endTimeMs - startTimeMs;
    snapshot.totalCpuTimeMs = 0.0;
    snapshot.spikesProcessed = totalSpikesProcessed_;
    snapshot.peakMemoryBytes = 0;
    snapshot.avgMemoryBytes = 0;
    
    // Filter metrics by time window
    for (const auto& pair : metrics_) {
        const auto& metrics = pair.second;
        
        // Check if this operation was active during the window
        auto lastCallTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            metrics.lastCallTime - profilingStartTime_).count();
        
        if (lastCallTimeMs >= startTimeMs && lastCallTimeMs <= endTimeMs) {
            snapshot.metrics[pair.first] = metrics;
            snapshot.totalCpuTimeMs += metrics.totalTimeMs;
            
            if (metrics.peakMemoryBytes > snapshot.peakMemoryBytes) {
                snapshot.peakMemoryBytes = metrics.peakMemoryBytes;
            }
        }
    }
    
    // Calculate CPU utilization
    if (snapshot.wallClockTimeMs > 0) {
        snapshot.cpuUtilization = snapshot.totalCpuTimeMs / snapshot.wallClockTimeMs;
    } else {
        snapshot.cpuUtilization = 0.0;
    }
    
    // Calculate spike processing rate
    if (snapshot.wallClockTimeMs > 0) {
        snapshot.spikeProcessingRate = (snapshot.spikesProcessed * 1000.0) / snapshot.wallClockTimeMs;
    } else {
        snapshot.spikeProcessingRate = 0.0;
    }
    
    // Calculate average memory
    if (!snapshot.metrics.empty()) {
        size_t totalMemory = 0;
        for (const auto& pair : snapshot.metrics) {
            totalMemory += pair.second.currentMemoryBytes;
        }
        snapshot.avgMemoryBytes = totalMemory / snapshot.metrics.size();
    }
    
    return snapshot;
}

PerformanceSnapshot PerformanceProfiler::getLatestSnapshot(double windowMs) const {
    double currentTime = getElapsedTimeMs();
    double startTime = std::max(0.0, currentTime - windowMs);
    return getSnapshot(startTime, currentTime);
}

BottleneckAnalysis PerformanceProfiler::analyzeBottlenecks(size_t topN) const {
    BottleneckAnalysis analysis;
    analysis.totalProfiledTimeMs = 0.0;
    analysis.analysisTime = std::chrono::steady_clock::now();
    
    // Calculate total time
    for (const auto& pair : metrics_) {
        analysis.totalProfiledTimeMs += pair.second.totalTimeMs;
    }
    
    // Create bottleneck entries
    std::vector<BottleneckAnalysis::Bottleneck> bottlenecks;
    for (const auto& pair : metrics_) {
        const auto& metrics = pair.second;
        
        BottleneckAnalysis::Bottleneck bottleneck;
        bottleneck.operationName = metrics.name;
        bottleneck.avgTimeMs = metrics.avgTimeMs;
        bottleneck.callCount = metrics.callCount;
        bottleneck.componentId = metrics.componentId;
        bottleneck.componentType = metrics.componentType;
        
        if (analysis.totalProfiledTimeMs > 0) {
            bottleneck.percentageOfTotal = (metrics.totalTimeMs / analysis.totalProfiledTimeMs) * 100.0;
        } else {
            bottleneck.percentageOfTotal = 0.0;
        }
        
        // Generate recommendation
        if (bottleneck.percentageOfTotal > 20.0) {
            bottleneck.recommendation = "Critical bottleneck - consider optimization";
        } else if (bottleneck.percentageOfTotal > 10.0) {
            bottleneck.recommendation = "Significant time consumer - review for optimization";
        } else if (bottleneck.avgTimeMs > 10.0) {
            bottleneck.recommendation = "High average execution time - consider caching or parallelization";
        } else {
            bottleneck.recommendation = "Performance acceptable";
        }
        
        bottlenecks.push_back(bottleneck);
    }
    
    // Sort by percentage of total time (descending)
    std::sort(bottlenecks.begin(), bottlenecks.end(),
              [](const BottleneckAnalysis::Bottleneck& a, const BottleneckAnalysis::Bottleneck& b) {
                  return a.percentageOfTotal > b.percentageOfTotal;
              });
    
    // Take top N
    size_t count = std::min(topN, bottlenecks.size());
    analysis.topBottlenecks.assign(bottlenecks.begin(), bottlenecks.begin() + count);
    
    return analysis;
}

uint64_t PerformanceProfiler::registerAlertCallback(PerformanceAlertCallback callback, double thresholdMs) {
    uint64_t id = nextCallbackId_++;
    alertCallbacks_[id] = std::make_pair(callback, thresholdMs);
    SNNFW_INFO("PerformanceProfiler: Registered alert callback with ID {} (threshold: {}ms)", id, thresholdMs);
    return id;
}

void PerformanceProfiler::unregisterAlertCallback(uint64_t callbackId) {
    alertCallbacks_.erase(callbackId);
    SNNFW_INFO("PerformanceProfiler: Unregistered alert callback {}", callbackId);
}

void PerformanceProfiler::reset() {
    metrics_.clear();
    activeOperations_.clear();
    totalSpikesProcessed_ = 0;
    SNNFW_INFO("PerformanceProfiler: All data reset");
}

void PerformanceProfiler::recordOperation(const std::string& operationName, double durationMs,
                                         uint64_t componentId, const std::string& componentType) {
    auto& metrics = metrics_[operationName];
    
    metrics.name = operationName;
    metrics.callCount++;
    metrics.totalTimeMs += durationMs;
    metrics.lastTimeMs = durationMs;
    metrics.lastCallTime = std::chrono::steady_clock::now();
    metrics.componentId = componentId;
    metrics.componentType = componentType;
    
    if (durationMs < metrics.minTimeMs) {
        metrics.minTimeMs = durationMs;
    }
    if (durationMs > metrics.maxTimeMs) {
        metrics.maxTimeMs = durationMs;
    }
    
    metrics.avgTimeMs = metrics.totalTimeMs / metrics.callCount;
}

void PerformanceProfiler::checkAlerts(const std::string& operationName, double executionTimeMs) {
    for (const auto& pair : alertCallbacks_) {
        const auto& callback = pair.second.first;
        double threshold = pair.second.second;
        
        if (executionTimeMs > threshold) {
            try {
                callback(operationName, executionTimeMs, "THRESHOLD_EXCEEDED");
            } catch (const std::exception& e) {
                SNNFW_ERROR("PerformanceProfiler: Alert callback threw exception: {}", e.what());
            }
        }
    }
}

double PerformanceProfiler::getElapsedTimeMs() const {
    if (!profiling_) {
        return 0.0;
    }
    
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        now - profilingStartTime_).count();
}

std::string PerformanceProfiler::generateReport() const {
    std::ostringstream oss;
    
    oss << "=== Performance Profiling Report ===\n\n";
    
    if (metrics_.empty()) {
        oss << "No profiling data available.\n";
        return oss.str();
    }
    
    // Summary
    double totalTime = 0.0;
    size_t totalCalls = 0;
    for (const auto& pair : metrics_) {
        totalTime += pair.second.totalTimeMs;
        totalCalls += pair.second.callCount;
    }
    
    oss << "Total Operations: " << metrics_.size() << "\n";
    oss << "Total Calls: " << totalCalls << "\n";
    oss << "Total Time: " << std::fixed << std::setprecision(2) << totalTime << " ms\n";
    oss << "Spikes Processed: " << totalSpikesProcessed_ << "\n\n";
    
    // Per-operation metrics
    oss << std::left << std::setw(30) << "Operation"
        << std::right << std::setw(10) << "Calls"
        << std::setw(12) << "Total (ms)"
        << std::setw(12) << "Avg (ms)"
        << std::setw(12) << "Min (ms)"
        << std::setw(12) << "Max (ms)"
        << "\n";
    oss << std::string(88, '-') << "\n";
    
    for (const auto& pair : metrics_) {
        const auto& m = pair.second;
        oss << std::left << std::setw(30) << m.name
            << std::right << std::setw(10) << m.callCount
            << std::setw(12) << std::fixed << std::setprecision(2) << m.totalTimeMs
            << std::setw(12) << m.avgTimeMs
            << std::setw(12) << m.minTimeMs
            << std::setw(12) << m.maxTimeMs
            << "\n";
    }
    
    return oss.str();
}

std::string PerformanceProfiler::exportToJson() const {
    nlohmann::json j;
    
    j["profiling"] = profiling_;
    j["totalSpikesProcessed"] = totalSpikesProcessed_;
    j["elapsedTimeMs"] = getElapsedTimeMs();
    
    nlohmann::json metricsArray = nlohmann::json::array();
    for (const auto& pair : metrics_) {
        const auto& m = pair.second;
        nlohmann::json metricJson;
        metricJson["name"] = m.name;
        metricJson["callCount"] = m.callCount;
        metricJson["totalTimeMs"] = m.totalTimeMs;
        metricJson["avgTimeMs"] = m.avgTimeMs;
        metricJson["minTimeMs"] = m.minTimeMs;
        metricJson["maxTimeMs"] = m.maxTimeMs;
        metricJson["lastTimeMs"] = m.lastTimeMs;
        metricJson["componentId"] = m.componentId;
        metricJson["componentType"] = m.componentType;
        metricJson["peakMemoryBytes"] = m.peakMemoryBytes;
        metricJson["currentMemoryBytes"] = m.currentMemoryBytes;
        metricsArray.push_back(metricJson);
    }
    j["metrics"] = metricsArray;
    
    return j.dump(2);
}

} // namespace snnfw


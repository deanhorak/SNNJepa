#ifndef SNNFW_PERFORMANCE_PROFILER_H
#define SNNFW_PERFORMANCE_PROFILER_H

#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <memory>
#include <functional>

namespace snnfw {

// Forward declarations
class Datastore;

/**
 * @brief Performance metrics for a specific operation or component
 * 
 * Designed to support visualization of performance bottlenecks and
 * temporal performance patterns.
 */
struct PerformanceMetrics {
    std::string name;                    ///< Name of the operation/component
    size_t callCount;                    ///< Number of times called
    double totalTimeMs;                  ///< Total time spent (milliseconds)
    double minTimeMs;                    ///< Minimum execution time
    double maxTimeMs;                    ///< Maximum execution time
    double avgTimeMs;                    ///< Average execution time
    double lastTimeMs;                   ///< Most recent execution time
    std::chrono::steady_clock::time_point lastCallTime;  ///< Timestamp of last call
    
    // Memory metrics
    size_t peakMemoryBytes;              ///< Peak memory usage
    size_t currentMemoryBytes;           ///< Current memory usage
    
    // Hierarchical context (for visualization grouping)
    uint64_t componentId;                ///< ID of the component being profiled
    std::string componentType;           ///< Type of component (Neuron, Cluster, etc.)
    
    PerformanceMetrics() 
        : callCount(0), totalTimeMs(0.0), minTimeMs(std::numeric_limits<double>::max()),
          maxTimeMs(0.0), avgTimeMs(0.0), lastTimeMs(0.0),
          peakMemoryBytes(0), currentMemoryBytes(0), componentId(0) {}
};

/**
 * @brief Aggregated performance statistics over a time window
 * 
 * Provides data for temporal performance visualization (graphs, heatmaps)
 */
struct PerformanceSnapshot {
    double startTime;                    ///< Start of snapshot window (ms since profiling started)
    double endTime;                      ///< End of snapshot window
    
    std::map<std::string, PerformanceMetrics> metrics;  ///< Metrics by operation name
    
    // System-wide metrics
    double totalCpuTimeMs;               ///< Total CPU time in this window
    double wallClockTimeMs;              ///< Wall clock time for this window
    double cpuUtilization;               ///< CPU utilization (0.0 - 1.0)
    
    // Throughput metrics
    size_t spikesProcessed;              ///< Number of spikes processed
    double spikeProcessingRate;          ///< Spikes per second
    
    // Memory metrics
    size_t peakMemoryBytes;              ///< Peak memory during window
    size_t avgMemoryBytes;               ///< Average memory during window
};

/**
 * @brief Bottleneck analysis results
 * 
 * Identifies performance bottlenecks for visualization highlighting
 */
struct BottleneckAnalysis {
    struct Bottleneck {
        std::string operationName;       ///< Name of bottleneck operation
        double percentageOfTotal;        ///< Percentage of total time
        double avgTimeMs;                ///< Average execution time
        size_t callCount;                ///< Number of calls
        std::string recommendation;      ///< Optimization recommendation
        
        // For visualization
        uint64_t componentId;            ///< Component ID if applicable
        std::string componentType;       ///< Component type
    };
    
    std::vector<Bottleneck> topBottlenecks;  ///< Top N bottlenecks by time
    double totalProfiledTimeMs;              ///< Total time profiled
    std::chrono::steady_clock::time_point analysisTime;  ///< When analysis was performed
};

/**
 * @brief Callback for real-time performance alerts
 * 
 * Allows visualization systems to highlight performance issues as they occur
 */
using PerformanceAlertCallback = std::function<void(const std::string& operationName, 
                                                     double executionTimeMs,
                                                     const std::string& alertType)>;

/**
 * @brief RAII-style timer for automatic performance measurement
 * 
 * Usage:
 * @code
 * {
 *     auto timer = profiler.startTimer("MyOperation");
 *     // ... code to profile ...
 * } // Timer automatically stops and records when it goes out of scope
 * @endcode
 */
class ScopedTimer {
public:
    ScopedTimer(const std::string& name, class PerformanceProfiler& profiler);
    ~ScopedTimer();
    
    // Prevent copying
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    
    // Allow moving
    ScopedTimer(ScopedTimer&& other) noexcept;
    ScopedTimer& operator=(ScopedTimer&& other) noexcept;
    
    /// Manually stop the timer (optional, destructor will stop if not called)
    void stop();
    
private:
    std::string operationName_;
    class PerformanceProfiler* profiler_;
    std::chrono::steady_clock::time_point startTime_;
    bool stopped_;
};

/**
 * @brief Performance profiler for analyzing and optimizing neural network performance
 * 
 * The PerformanceProfiler provides comprehensive performance monitoring with
 * visualization support. It tracks execution times, memory usage, and identifies
 * bottlenecks across the neural network hierarchy.
 * 
 * Key features for visualization:
 * - Real-time performance alerts via callbacks
 * - Temporal performance snapshots for graphing
 * - Bottleneck analysis with component identification
 * - Hierarchical performance metrics (by component type/ID)
 * - RAII-style timers for easy instrumentation
 * 
 * Example usage:
 * @code
 * PerformanceProfiler profiler;
 * profiler.startProfiling();
 * 
 * // Manual timing
 * profiler.startOperation("SpikeProcessing");
 * // ... process spikes ...
 * profiler.endOperation("SpikeProcessing");
 * 
 * // RAII timing
 * {
 *     auto timer = profiler.startTimer("PatternMatching");
 *     // ... match patterns ...
 * }
 * 
 * // Get results
 * auto snapshot = profiler.getSnapshot(0.0, 1000.0);
 * auto bottlenecks = profiler.analyzeBottlenecks(10);
 * 
 * profiler.stopProfiling();
 * @endcode
 */
class PerformanceProfiler {
public:
    /**
     * @brief Construct a new PerformanceProfiler
     */
    PerformanceProfiler();
    
    /**
     * @brief Destructor
     */
    ~PerformanceProfiler();
    
    /**
     * @brief Start profiling
     */
    void startProfiling();
    
    /**
     * @brief Stop profiling
     */
    void stopProfiling();
    
    /**
     * @brief Check if currently profiling
     * @return true if profiling is active
     */
    bool isProfiling() const { return profiling_; }
    
    /**
     * @brief Start timing an operation (manual timing)
     * @param operationName Name of the operation
     * @param componentId Optional component ID for hierarchical tracking
     * @param componentType Optional component type
     */
    void startOperation(const std::string& operationName, 
                       uint64_t componentId = 0,
                       const std::string& componentType = "");
    
    /**
     * @brief End timing an operation (manual timing)
     * @param operationName Name of the operation
     */
    void endOperation(const std::string& operationName);
    
    /**
     * @brief Create a scoped timer for RAII-style timing
     * @param operationName Name of the operation
     * @param componentId Optional component ID
     * @param componentType Optional component type
     * @return ScopedTimer object that automatically stops on destruction
     */
    ScopedTimer startTimer(const std::string& operationName,
                          uint64_t componentId = 0,
                          const std::string& componentType = "");
    
    /**
     * @brief Record a spike processing event (for throughput tracking)
     * @param count Number of spikes processed
     */
    void recordSpikesProcessed(size_t count = 1);
    
    /**
     * @brief Record memory usage
     * @param bytes Current memory usage in bytes
     * @param operationName Optional operation name to associate with
     */
    void recordMemoryUsage(size_t bytes, const std::string& operationName = "");
    
    /**
     * @brief Get performance metrics for a specific operation
     * @param operationName Name of the operation
     * @return PerformanceMetrics structure, or empty metrics if not found
     */
    PerformanceMetrics getMetrics(const std::string& operationName) const;
    
    /**
     * @brief Get all current performance metrics
     * @return Map of operation name to metrics
     */
    std::map<std::string, PerformanceMetrics> getAllMetrics() const;
    
    /**
     * @brief Get performance snapshot for a time window
     * @param startTimeMs Start time (ms since profiling started)
     * @param endTimeMs End time (ms since profiling started)
     * @return PerformanceSnapshot with aggregated metrics
     */
    PerformanceSnapshot getSnapshot(double startTimeMs, double endTimeMs) const;
    
    /**
     * @brief Get the most recent snapshot
     * @param windowMs Size of the window in milliseconds (default: 100ms)
     * @return PerformanceSnapshot for the most recent window
     */
    PerformanceSnapshot getLatestSnapshot(double windowMs = 100.0) const;
    
    /**
     * @brief Analyze performance bottlenecks
     * @param topN Number of top bottlenecks to return
     * @return BottleneckAnalysis with identified bottlenecks
     */
    BottleneckAnalysis analyzeBottlenecks(size_t topN = 10) const;
    
    /**
     * @brief Register a callback for performance alerts
     * @param callback Function to call when performance issues detected
     * @param thresholdMs Threshold in milliseconds for triggering alerts
     * @return Callback ID for later unregistration
     */
    uint64_t registerAlertCallback(PerformanceAlertCallback callback, double thresholdMs = 100.0);
    
    /**
     * @brief Unregister a performance alert callback
     * @param callbackId ID returned from registerAlertCallback
     */
    void unregisterAlertCallback(uint64_t callbackId);
    
    /**
     * @brief Reset all profiling data
     */
    void reset();
    
    /**
     * @brief Generate a performance report
     * @return String containing formatted performance report
     */
    std::string generateReport() const;
    
    /**
     * @brief Export performance data to JSON
     * @return JSON string with all performance data
     */
    std::string exportToJson() const;

private:
    bool profiling_;
    std::chrono::steady_clock::time_point profilingStartTime_;
    std::chrono::steady_clock::time_point profilingEndTime_;
    
    // Performance metrics storage
    std::map<std::string, PerformanceMetrics> metrics_;
    std::map<std::string, std::chrono::steady_clock::time_point> activeOperations_;
    
    // Throughput tracking
    size_t totalSpikesProcessed_;
    
    // Alert callbacks
    std::map<uint64_t, std::pair<PerformanceAlertCallback, double>> alertCallbacks_;
    uint64_t nextCallbackId_;
    
    // Helper methods
    void recordOperation(const std::string& operationName, double durationMs,
                        uint64_t componentId, const std::string& componentType);
    void checkAlerts(const std::string& operationName, double executionTimeMs);
    double getElapsedTimeMs() const;
};

} // namespace snnfw

#endif // SNNFW_PERFORMANCE_PROFILER_H


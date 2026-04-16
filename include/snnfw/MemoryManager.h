#ifndef SNNFW_MEMORY_MANAGER_H
#define SNNFW_MEMORY_MANAGER_H

#include <cstdint>
#include <string>
#include <chrono>

namespace snnfw {

/**
 * @brief Monitors and manages memory usage during simulation
 * 
 * Provides utilities to:
 * - Track current memory usage
 * - Enforce memory limits
 * - Report memory statistics
 * - Detect memory leaks
 */
class MemoryManager {
public:
    /**
     * @brief Memory statistics snapshot
     */
    struct MemoryStats {
        uint64_t rssBytes = 0;              ///< Resident Set Size (actual physical memory)
        uint64_t vmsBytes = 0;              ///< Virtual Memory Size
        uint64_t peakRssBytes = 0;          ///< Peak RSS during execution
        double percentOfLimit = 0.0;        ///< Percentage of memory limit used
        std::string timestamp;              ///< When this snapshot was taken
    };

    /**
     * @brief Initialize memory manager
     * @param maxMemoryMB Maximum memory to allow (0 = unlimited)
     */
    static void initialize(size_t maxMemoryMB = 0);

    /**
     * @brief Get current memory usage
     * @return Memory statistics
     */
    static MemoryStats getMemoryStats();

    /**
     * @brief Check if memory usage exceeds limit
     * @return true if memory usage is above limit
     */
    static bool isMemoryLimitExceeded();

    /**
     * @brief Get percentage of memory limit used
     * @return Percentage (0-100+)
     */
    static double getMemoryPercentage();

    /**
     * @brief Get remaining memory available
     * @return Bytes remaining before limit
     */
    static uint64_t getRemainingMemory();

    /**
     * @brief Log memory statistics
     */
    static void logMemoryStats();

    /**
     * @brief Set memory limit
     * @param maxMemoryMB New memory limit in MB
     */
    static void setMemoryLimit(size_t maxMemoryMB);

    /**
     * @brief Get memory limit
     * @return Memory limit in bytes
     */
    static uint64_t getMemoryLimit();

    /**
     * @brief Check for memory leaks by comparing snapshots
     * @param initialStats Initial memory snapshot
     * @param currentStats Current memory snapshot
     * @return true if potential leak detected
     */
    static bool detectMemoryLeak(const MemoryStats& initialStats, const MemoryStats& currentStats);

private:
    static size_t maxMemoryBytes_;
    static MemoryStats lastStats_;
    static std::chrono::steady_clock::time_point lastCheckTime_;

    /**
     * @brief Read memory usage from /proc/self/status
     * @return Memory statistics
     */
    static MemoryStats readMemoryUsage();
};

} // namespace snnfw

#endif // SNNFW_MEMORY_MANAGER_H


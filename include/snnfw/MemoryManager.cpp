#include "snnfw/MemoryManager.h"
#include "snnfw/Logger.h"
#include <fstream>
#include <sstream>
#include <iomanip>

namespace snnfw {

// Static member initialization
size_t MemoryManager::maxMemoryBytes_ = 0;
MemoryManager::MemoryStats MemoryManager::lastStats_;
std::chrono::steady_clock::time_point MemoryManager::lastCheckTime_ = std::chrono::steady_clock::now();

void MemoryManager::initialize(size_t maxMemoryMB) {
    maxMemoryBytes_ = maxMemoryMB * 1024ULL * 1024ULL;
    lastStats_ = readMemoryUsage();
    lastCheckTime_ = std::chrono::steady_clock::now();
    
    if (maxMemoryMB > 0) {
        SNNFW_INFO("MemoryManager initialized with limit: {}MB", maxMemoryMB);
    } else {
        SNNFW_INFO("MemoryManager initialized (no memory limit)");
    }
}

MemoryManager::MemoryStats MemoryManager::readMemoryUsage() {
    MemoryStats stats;
    std::ifstream statusFile("/proc/self/status");
    
    if (!statusFile.is_open()) {
        SNNFW_WARN("MemoryManager: Cannot read /proc/self/status");
        return stats;
    }
    
    std::string line;
    while (std::getline(statusFile, line)) {
        if (line.find("VmRSS:") == 0) {
            std::istringstream iss(line);
            std::string label;
            uint64_t value;
            iss >> label >> value;
            stats.rssBytes = value * 1024ULL;  // Convert KB to bytes
        } else if (line.find("VmPeak:") == 0) {
            std::istringstream iss(line);
            std::string label;
            uint64_t value;
            iss >> label >> value;
            stats.peakRssBytes = value * 1024ULL;  // Convert KB to bytes
        } else if (line.find("VmSize:") == 0) {
            std::istringstream iss(line);
            std::string label;
            uint64_t value;
            iss >> label >> value;
            stats.vmsBytes = value * 1024ULL;  // Convert KB to bytes
        }
    }
    
    if (maxMemoryBytes_ > 0) {
        stats.percentOfLimit = (100.0 * stats.rssBytes) / maxMemoryBytes_;
    }
    
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%H:%M:%S");
    stats.timestamp = ss.str();
    
    return stats;
}

MemoryManager::MemoryStats MemoryManager::getMemoryStats() {
    lastStats_ = readMemoryUsage();
    return lastStats_;
}

bool MemoryManager::isMemoryLimitExceeded() {
    if (maxMemoryBytes_ == 0) return false;
    
    auto stats = readMemoryUsage();
    return stats.rssBytes > maxMemoryBytes_;
}

double MemoryManager::getMemoryPercentage() {
    if (maxMemoryBytes_ == 0) return 0.0;
    
    auto stats = readMemoryUsage();
    return (100.0 * stats.rssBytes) / maxMemoryBytes_;
}

uint64_t MemoryManager::getRemainingMemory() {
    if (maxMemoryBytes_ == 0) return UINT64_MAX;
    
    auto stats = readMemoryUsage();
    if (stats.rssBytes >= maxMemoryBytes_) {
        return 0;
    }
    return maxMemoryBytes_ - stats.rssBytes;
}

void MemoryManager::logMemoryStats() {
    auto stats = getMemoryStats();
    SNNFW_INFO("Memory Stats [{}]: RSS={:.1f}MB, VMS={:.1f}MB, Peak={:.1f}MB, Usage={:.1f}%",
               stats.timestamp,
               stats.rssBytes / (1024.0 * 1024.0),
               stats.vmsBytes / (1024.0 * 1024.0),
               stats.peakRssBytes / (1024.0 * 1024.0),
               stats.percentOfLimit);
}

void MemoryManager::setMemoryLimit(size_t maxMemoryMB) {
    maxMemoryBytes_ = maxMemoryMB * 1024ULL * 1024ULL;
    SNNFW_INFO("MemoryManager: Memory limit set to {}MB", maxMemoryMB);
}

uint64_t MemoryManager::getMemoryLimit() {
    return maxMemoryBytes_;
}

bool MemoryManager::detectMemoryLeak(const MemoryStats& initialStats, const MemoryStats& currentStats) {
    // If memory grew by more than 50% and we're using more than 500MB, flag as potential leak
    uint64_t growth = currentStats.rssBytes - initialStats.rssBytes;
    double growthPercent = (100.0 * growth) / (initialStats.rssBytes + 1);
    
    return growthPercent > 50.0 && currentStats.rssBytes > (500ULL * 1024ULL * 1024ULL);
}

} // namespace snnfw


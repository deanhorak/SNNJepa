#ifndef SNNFW_LOGGER_H
#define SNNFW_LOGGER_H

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <memory>
#include <string>

namespace snnfw {

/**
 * @brief Singleton logger class for the SNNFW framework
 * 
 * Provides a centralized logging facility with multiple severity levels:
 * - TRACE: Very detailed information, typically only enabled during development
 * - DEBUG: Detailed information useful for debugging
 * - INFO: General informational messages
 * - WARN: Warning messages for potentially harmful situations
 * - ERROR: Error messages for error events
 * - CRITICAL: Critical messages for very severe error events
 * 
 * Usage:
 *   SNNFW_TRACE("Detailed trace message: {}", value);
 *   SNNFW_DEBUG("Debug message: {}", value);
 *   SNNFW_INFO("Info message: {}", value);
 *   SNNFW_WARN("Warning message: {}", value);
 *   SNNFW_ERROR("Error message: {}", value);
 *   SNNFW_CRITICAL("Critical message: {}", value);
 */
class Logger {
public:
    /**
     * @brief Get the singleton instance of the logger
     * @return Reference to the logger instance
     */
    static Logger& getInstance();
    
    /**
     * @brief Initialize the logger with console and file output
     * @param logFileName Name of the log file (default: "snnfw.log")
     * @param level Initial log level (default: INFO)
     */
    void initialize(const std::string& logFileName = "snnfw.log", 
                   spdlog::level::level_enum level = spdlog::level::info);
    
    /**
     * @brief Set the logging level
     * @param level The new logging level
     */
    void setLevel(spdlog::level::level_enum level);
    
    /**
     * @brief Get the underlying spdlog logger
     * @return Shared pointer to the spdlog logger
     */
    std::shared_ptr<spdlog::logger> getLogger();
    
    // Delete copy constructor and assignment operator (singleton)
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
private:
    Logger();
    ~Logger() = default;
    
    std::shared_ptr<spdlog::logger> logger_;
    bool initialized_;
};

} // namespace snnfw

// Convenience macros for logging
#define SNNFW_TRACE(...)    snnfw::Logger::getInstance().getLogger()->trace(__VA_ARGS__)
#define SNNFW_DEBUG(...)    snnfw::Logger::getInstance().getLogger()->debug(__VA_ARGS__)
#define SNNFW_INFO(...)     snnfw::Logger::getInstance().getLogger()->info(__VA_ARGS__)
#define SNNFW_WARN(...)     snnfw::Logger::getInstance().getLogger()->warn(__VA_ARGS__)
#define SNNFW_ERROR(...)    snnfw::Logger::getInstance().getLogger()->error(__VA_ARGS__)
#define SNNFW_CRITICAL(...) snnfw::Logger::getInstance().getLogger()->critical(__VA_ARGS__)

#endif // SNNFW_LOGGER_H


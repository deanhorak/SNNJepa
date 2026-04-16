#include "snnfw/Logger.h"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <vector>

namespace snnfw {

Logger::Logger() : initialized_(false) {
    // Check if logger already exists
    logger_ = spdlog::get("snnfw");
    if (!logger_) {
        // Create a default console logger
        logger_ = spdlog::stdout_color_mt("snnfw");
        logger_->set_level(spdlog::level::info);
        logger_->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    }
}

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

void Logger::initialize(const std::string& logFileName, spdlog::level::level_enum level) {
    if (initialized_) {
        logger_->warn("Logger already initialized. Skipping re-initialization.");
        return;
    }

    try {
        // Drop existing logger if it exists
        spdlog::drop("snnfw");

        // Create sinks for both console and file
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::trace);
        console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(logFileName, true);
        file_sink->set_level(spdlog::level::trace);
        file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");

        // Combine sinks
        std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
        logger_ = std::make_shared<spdlog::logger>("snnfw", sinks.begin(), sinks.end());

        // Set the overall log level
        logger_->set_level(level);
        logger_->flush_on(spdlog::level::warn);

        // Register the logger
        spdlog::register_logger(logger_);

        initialized_ = true;
        logger_->info("Logger initialized with file: {}", logFileName);
    } catch (const spdlog::spdlog_ex& ex) {
        // Fall back to console-only logger
        spdlog::drop("snnfw_fallback");
        logger_ = spdlog::stdout_color_mt("snnfw_fallback");
        logger_->error("Logger initialization failed: {}", ex.what());
        logger_->warn("Falling back to console-only logging");
    }
}

void Logger::setLevel(spdlog::level::level_enum level) {
    logger_->set_level(level);
    logger_->info("Log level changed to: {}", spdlog::level::to_string_view(level));
}

std::shared_ptr<spdlog::logger> Logger::getLogger() {
    return logger_;
}

} // namespace snnfw


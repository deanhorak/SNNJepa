#ifndef SNNFW_ASSERTIONS_H
#define SNNFW_ASSERTIONS_H

#include "snnfw/Logger.h"
#include "snnfw/Datastore.h"
#include <stdexcept>
#include <string>
#include <sstream>

/**
 * @file Assertions.h
 * @brief Runtime invariant assertions for the SNNFW framework
 * 
 * This file provides a comprehensive assertion system for validating
 * critical invariants at runtime. The system supports:
 * - Graceful degradation vs hard failures (controlled by SNNFW_STRICT_MODE)
 * - Detailed error messages with file/line context
 * - Specialized assertions for common validation patterns
 * - Integration with the logging system
 * 
 * Usage:
 *   SNNFW_ASSERT(condition, "Error message");
 *   SNNFW_REQUIRE_ID_EXISTS(id, datastore);
 *   SNNFW_REQUIRE_NOT_NULL(ptr, "pointer name");
 *   SNNFW_REQUIRE_RANGE(value, min, max, "value name");
 */

namespace snnfw {

/**
 * @brief Assertion mode configuration
 * 
 * In STRICT mode, assertion failures throw exceptions.
 * In non-STRICT mode, assertion failures log errors but continue execution.
 * 
 * Set via compile-time flag: -DSNNFW_STRICT_MODE=1
 * Or at runtime via: setStrictMode(true/false)
 */
#ifndef SNNFW_STRICT_MODE
#define SNNFW_STRICT_MODE 0
#endif

/**
 * @brief Runtime control of strict mode
 */
class AssertionConfig {
public:
    static AssertionConfig& getInstance() {
        static AssertionConfig instance;
        return instance;
    }
    
    void setStrictMode(bool strict) { strictMode_ = strict; }
    bool isStrictMode() const { return strictMode_; }
    
    void setThrowOnError(bool throwOnError) { throwOnError_ = throwOnError; }
    bool shouldThrowOnError() const { return throwOnError_; }
    
private:
    AssertionConfig() : strictMode_(SNNFW_STRICT_MODE), throwOnError_(SNNFW_STRICT_MODE) {}
    bool strictMode_;
    bool throwOnError_;
};

/**
 * @brief Exception thrown by assertion failures in strict mode
 */
class AssertionError : public std::runtime_error {
public:
    AssertionError(const std::string& message, 
                   const std::string& file, 
                   int line,
                   const std::string& condition = "")
        : std::runtime_error(formatMessage(message, file, line, condition)),
          message_(message),
          file_(file),
          line_(line),
          condition_(condition) {}
    
    const std::string& getMessage() const { return message_; }
    const std::string& getFile() const { return file_; }
    int getLine() const { return line_; }
    const std::string& getCondition() const { return condition_; }
    
private:
    static std::string formatMessage(const std::string& message,
                                     const std::string& file,
                                     int line,
                                     const std::string& condition) {
        std::ostringstream oss;
        oss << "Assertion failed: " << message;
        if (!condition.empty()) {
            oss << " [" << condition << "]";
        }
        oss << " at " << file << ":" << line;
        return oss.str();
    }
    
    std::string message_;
    std::string file_;
    int line_;
    std::string condition_;
};

} // namespace snnfw

/**
 * @brief Core assertion macro
 * 
 * Checks a condition and logs an error if it fails.
 * In strict mode, throws an AssertionError exception.
 * 
 * @param condition The condition to check
 * @param message Error message (can use fmt-style formatting)
 * @param ... Optional format arguments
 */
#define SNNFW_ASSERT(condition, ...) \
    do { \
        if (!(condition)) { \
            SNNFW_ERROR("Assertion failed: {} at {}:{}", \
                       fmt::format(__VA_ARGS__), __FILE__, __LINE__); \
            if (snnfw::AssertionConfig::getInstance().shouldThrowOnError()) { \
                throw snnfw::AssertionError( \
                    fmt::format(__VA_ARGS__), __FILE__, __LINE__, #condition); \
            } \
        } \
    } while (0)

/**
 * @brief Assert that an ID exists in the datastore
 *
 * @param id The ID to check
 * @param datastore The datastore to check in
 */
#define SNNFW_REQUIRE_ID_EXISTS(id, datastore) \
    do { \
        if ((datastore).get(id) == nullptr) { \
            SNNFW_ERROR("ID {} does not exist in datastore at {}:{}", \
                       id, __FILE__, __LINE__); \
            if (snnfw::AssertionConfig::getInstance().shouldThrowOnError()) { \
                throw snnfw::AssertionError( \
                    "ID " + std::to_string(id) + " does not exist in datastore", \
                    __FILE__, __LINE__, "datastore.get(" #id ") != nullptr"); \
            } \
        } \
    } while (0)

/**
 * @brief Assert that a pointer is not null
 * 
 * @param ptr The pointer to check
 * @param name Descriptive name for the pointer
 */
#define SNNFW_REQUIRE_NOT_NULL(ptr, name) \
    do { \
        if ((ptr) == nullptr) { \
            SNNFW_ERROR("Null pointer: {} at {}:{}", name, __FILE__, __LINE__); \
            if (snnfw::AssertionConfig::getInstance().shouldThrowOnError()) { \
                throw snnfw::AssertionError( \
                    std::string("Null pointer: ") + name, \
                    __FILE__, __LINE__, #ptr " != nullptr"); \
            } \
        } \
    } while (0)

/**
 * @brief Assert that a value is within a valid range
 * 
 * @param value The value to check
 * @param min Minimum valid value (inclusive)
 * @param max Maximum valid value (inclusive)
 * @param name Descriptive name for the value
 */
#define SNNFW_REQUIRE_RANGE(value, min, max, name) \
    do { \
        if ((value) < (min) || (value) > (max)) { \
            SNNFW_ERROR("Value {} out of range [{}, {}]: {} = {} at {}:{}", \
                       name, min, max, name, value, __FILE__, __LINE__); \
            if (snnfw::AssertionConfig::getInstance().shouldThrowOnError()) { \
                throw snnfw::AssertionError( \
                    std::string(name) + " = " + std::to_string(value) + \
                    " out of range [" + std::to_string(min) + ", " + \
                    std::to_string(max) + "]", \
                    __FILE__, __LINE__, \
                    std::to_string(min) + " <= " #value " <= " + std::to_string(max)); \
            } \
        } \
    } while (0)

/**
 * @brief Assert that a value is positive
 * 
 * @param value The value to check
 * @param name Descriptive name for the value
 */
#define SNNFW_REQUIRE_POSITIVE(value, name) \
    do { \
        if ((value) <= 0) { \
            SNNFW_ERROR("Value {} must be positive: {} = {} at {}:{}", \
                       name, name, value, __FILE__, __LINE__); \
            if (snnfw::AssertionConfig::getInstance().shouldThrowOnError()) { \
                throw snnfw::AssertionError( \
                    std::string(name) + " = " + std::to_string(value) + " must be positive", \
                    __FILE__, __LINE__, #value " > 0"); \
            } \
        } \
    } while (0)

/**
 * @brief Assert that a value is non-negative
 * 
 * @param value The value to check
 * @param name Descriptive name for the value
 */
#define SNNFW_REQUIRE_NON_NEGATIVE(value, name) \
    do { \
        if ((value) < 0) { \
            SNNFW_ERROR("Value {} must be non-negative: {} = {} at {}:{}", \
                       name, name, value, __FILE__, __LINE__); \
            if (snnfw::AssertionConfig::getInstance().shouldThrowOnError()) { \
                throw snnfw::AssertionError( \
                    std::string(name) + " = " + std::to_string(value) + " must be non-negative", \
                    __FILE__, __LINE__, #value " >= 0"); \
            } \
        } \
    } while (0)

/**
 * @brief Assert that a container is not empty
 * 
 * @param container The container to check
 * @param name Descriptive name for the container
 */
#define SNNFW_REQUIRE_NOT_EMPTY(container, name) \
    do { \
        if ((container).empty()) { \
            SNNFW_ERROR("Container {} must not be empty at {}:{}", \
                       name, __FILE__, __LINE__); \
            if (snnfw::AssertionConfig::getInstance().shouldThrowOnError()) { \
                throw snnfw::AssertionError( \
                    std::string("Container ") + name + " must not be empty", \
                    __FILE__, __LINE__, "!" #container ".empty()"); \
            } \
        } \
    } while (0)

/**
 * @brief Assert that a container size is within bounds
 * 
 * @param container The container to check
 * @param maxSize Maximum allowed size
 * @param name Descriptive name for the container
 */
#define SNNFW_REQUIRE_SIZE_LIMIT(container, maxSize, name) \
    do { \
        if ((container).size() > (maxSize)) { \
            SNNFW_ERROR("Container {} exceeds size limit: {} > {} at {}:{}", \
                       name, (container).size(), maxSize, __FILE__, __LINE__); \
            if (snnfw::AssertionConfig::getInstance().shouldThrowOnError()) { \
                throw snnfw::AssertionError( \
                    std::string("Container ") + name + " size " + \
                    std::to_string((container).size()) + " exceeds limit " + \
                    std::to_string(maxSize), \
                    __FILE__, __LINE__, #container ".size() <= " #maxSize); \
            } \
        } \
    } while (0)

/**
 * @brief Assert that an ID is in the valid range for its type
 * 
 * @param id The ID to check
 * @param minId Minimum valid ID for this type
 * @param maxId Maximum valid ID for this type
 * @param typeName Name of the type (for error messages)
 */
#define SNNFW_REQUIRE_ID_RANGE(id, minId, maxId, typeName) \
    SNNFW_REQUIRE_RANGE(id, minId, maxId, typeName " ID")

/**
 * @brief Unconditional failure assertion
 * 
 * Always fails with the given message. Useful for unreachable code paths.
 * 
 * @param message Error message
 */
#define SNNFW_FAIL(...) \
    do { \
        SNNFW_ERROR("Assertion failed: {} at {}:{}", \
                   fmt::format(__VA_ARGS__), __FILE__, __LINE__); \
        if (snnfw::AssertionConfig::getInstance().shouldThrowOnError()) { \
            throw snnfw::AssertionError( \
                fmt::format(__VA_ARGS__), __FILE__, __LINE__, "FAIL"); \
        } \
    } while (0)

#endif // SNNFW_ASSERTIONS_H


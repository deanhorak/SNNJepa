#ifndef SNNFW_THREADSAFE_H
#define SNNFW_THREADSAFE_H

#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <memory>

namespace snnfw {

/**
 * @brief Thread-safe wrapper for any type using mutex protection
 * 
 * This template class wraps any type and provides thread-safe access
 * through a mutex. All operations are protected by a lock.
 * 
 * Example usage:
 * @code
 * ThreadSafe<std::vector<int>> safeVector;
 * 
 * // Thread-safe modification
 * safeVector.modify([](std::vector<int>& vec) {
 *     vec.push_back(42);
 * });
 * 
 * // Thread-safe read
 * int size = safeVector.read([](const std::vector<int>& vec) {
 *     return vec.size();
 * });
 * @endcode
 */
template<typename T>
class ThreadSafe {
public:
    /**
     * @brief Default constructor
     */
    ThreadSafe() = default;
    
    /**
     * @brief Construct with initial value
     * @param value Initial value
     */
    explicit ThreadSafe(const T& value) : data(value) {}
    
    /**
     * @brief Construct with initial value (move)
     * @param value Initial value
     */
    explicit ThreadSafe(T&& value) : data(std::move(value)) {}
    
    /**
     * @brief Execute a function with exclusive access to the data
     * @tparam F Function type
     * @param f Function to execute (receives T& as parameter)
     * @return Result of the function
     */
    template<typename F>
    auto modify(F&& f) -> decltype(f(std::declval<T&>())) {
        std::lock_guard<std::mutex> lock(mutex);
        return f(data);
    }
    
    /**
     * @brief Execute a function with read-only access to the data
     * @tparam F Function type
     * @param f Function to execute (receives const T& as parameter)
     * @return Result of the function
     */
    template<typename F>
    auto read(F&& f) const -> decltype(f(std::declval<const T&>())) {
        std::lock_guard<std::mutex> lock(mutex);
        return f(data);
    }
    
    /**
     * @brief Get a copy of the data
     * @return Copy of the protected data
     */
    T getCopy() const {
        std::lock_guard<std::mutex> lock(mutex);
        return data;
    }
    
    /**
     * @brief Set the data
     * @param value New value
     */
    void set(const T& value) {
        std::lock_guard<std::mutex> lock(mutex);
        data = value;
    }
    
    /**
     * @brief Set the data (move)
     * @param value New value
     */
    void set(T&& value) {
        std::lock_guard<std::mutex> lock(mutex);
        data = std::move(value);
    }
    
private:
    mutable std::mutex mutex;
    T data;
};

/**
 * @brief Thread-safe wrapper with read-write lock for better read performance
 * 
 * This class uses a shared_mutex to allow multiple concurrent readers
 * or a single writer. This is more efficient when reads are more common
 * than writes.
 * 
 * Example usage:
 * @code
 * ThreadSafeRW<std::map<int, std::string>> safeMap;
 * 
 * // Multiple threads can read simultaneously
 * auto value = safeMap.read([](const auto& map) {
 *     return map.at(42);
 * });
 * 
 * // Only one thread can write at a time
 * safeMap.write([](auto& map) {
 *     map[42] = "answer";
 * });
 * @endcode
 */
template<typename T>
class ThreadSafeRW {
public:
    /**
     * @brief Default constructor
     */
    ThreadSafeRW() = default;
    
    /**
     * @brief Construct with initial value
     * @param value Initial value
     */
    explicit ThreadSafeRW(const T& value) : data(value) {}
    
    /**
     * @brief Construct with initial value (move)
     * @param value Initial value
     */
    explicit ThreadSafeRW(T&& value) : data(std::move(value)) {}
    
    /**
     * @brief Execute a function with exclusive write access
     * @tparam F Function type
     * @param f Function to execute (receives T& as parameter)
     * @return Result of the function
     */
    template<typename F>
    auto write(F&& f) -> decltype(f(std::declval<T&>())) {
        std::unique_lock<std::shared_mutex> lock(mutex);
        return f(data);
    }
    
    /**
     * @brief Execute a function with shared read access
     * @tparam F Function type
     * @param f Function to execute (receives const T& as parameter)
     * @return Result of the function
     */
    template<typename F>
    auto read(F&& f) const -> decltype(f(std::declval<const T&>())) {
        std::shared_lock<std::shared_mutex> lock(mutex);
        return f(data);
    }
    
    /**
     * @brief Get a copy of the data
     * @return Copy of the protected data
     */
    T getCopy() const {
        std::shared_lock<std::shared_mutex> lock(mutex);
        return data;
    }
    
    /**
     * @brief Set the data
     * @param value New value
     */
    void set(const T& value) {
        std::unique_lock<std::shared_mutex> lock(mutex);
        data = value;
    }
    
    /**
     * @brief Set the data (move)
     * @param value New value
     */
    void set(T&& value) {
        std::unique_lock<std::shared_mutex> lock(mutex);
        data = std::move(value);
    }
    
private:
    mutable std::shared_mutex mutex;
    T data;
};

/**
 * @brief Atomic counter for thread-safe counting operations
 * 
 * Provides a simple interface for atomic increment/decrement operations.
 */
class AtomicCounter {
public:
    /**
     * @brief Construct with initial value
     * @param initial Initial counter value (default: 0)
     */
    explicit AtomicCounter(uint64_t initial = 0) : value(initial) {}
    
    /**
     * @brief Increment and return new value
     * @return New value after increment
     */
    uint64_t increment() {
        return ++value;
    }
    
    /**
     * @brief Decrement and return new value
     * @return New value after decrement
     */
    uint64_t decrement() {
        return --value;
    }
    
    /**
     * @brief Get current value
     * @return Current counter value
     */
    uint64_t get() const {
        return value.load();
    }
    
    /**
     * @brief Set value
     * @param newValue New counter value
     */
    void set(uint64_t newValue) {
        value.store(newValue);
    }
    
    /**
     * @brief Add to counter and return new value
     * @param delta Amount to add
     * @return New value after addition
     */
    uint64_t add(uint64_t delta) {
        return value.fetch_add(delta) + delta;
    }
    
    /**
     * @brief Subtract from counter and return new value
     * @param delta Amount to subtract
     * @return New value after subtraction
     */
    uint64_t subtract(uint64_t delta) {
        return value.fetch_sub(delta) - delta;
    }
    
private:
    std::atomic<uint64_t> value;
};

} // namespace snnfw

#endif // SNNFW_THREADSAFE_H


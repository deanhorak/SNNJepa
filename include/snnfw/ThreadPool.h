#ifndef SNNFW_THREADPOOL_H
#define SNNFW_THREADPOOL_H

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>

namespace snnfw {

/**
 * @brief Thread pool for efficient parallel task execution
 * 
 * This class provides a pool of worker threads that can execute tasks
 * asynchronously. Tasks are queued and executed by available threads.
 * 
 * Features:
 * - Configurable number of worker threads
 * - Task queue with automatic load balancing
 * - Future-based result retrieval
 * - Graceful shutdown
 * - Thread-safe task submission
 * 
 * Example usage:
 * @code
 * ThreadPool pool(4); // Create pool with 4 threads
 * 
 * // Submit a task and get a future
 * auto result = pool.enqueue([](int x) { return x * 2; }, 21);
 * 
 * // Get the result (blocks until task completes)
 * int value = result.get(); // value = 42
 * @endcode
 */
class ThreadPool {
public:
    /**
     * @brief Construct a thread pool with specified number of threads
     * @param numThreads Number of worker threads (default: hardware concurrency)
     */
    explicit ThreadPool(size_t numThreads = std::thread::hardware_concurrency());
    
    /**
     * @brief Destructor - waits for all tasks to complete and joins threads
     */
    ~ThreadPool();
    
    /**
     * @brief Enqueue a task for execution
     * @tparam F Function type
     * @tparam Args Argument types
     * @param f Function to execute
     * @param args Arguments to pass to the function
     * @return std::future with the result of the function
     * @throws std::runtime_error if pool is stopped
     */
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;
    
    /**
     * @brief Get the number of worker threads
     * @return Number of threads in the pool
     */
    size_t size() const { return workers.size(); }
    
    /**
     * @brief Get the number of pending tasks
     * @return Number of tasks waiting to be executed
     */
    size_t pendingTasks() const;
    
    /**
     * @brief Check if the pool is stopped
     * @return true if pool is stopped, false otherwise
     */
    bool isStopped() const { return stop; }
    
    // Prevent copying
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    
private:
    // Worker threads
    std::vector<std::thread> workers;
    
    // Task queue
    std::queue<std::function<void()>> tasks;
    
    // Synchronization
    mutable std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;
};

// Template implementation must be in header
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;
    
    // Create a packaged task
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> res = task->get_future();
    
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        
        // Don't allow enqueueing after stopping the pool
        if (stop) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }
        
        tasks.emplace([task]() { (*task)(); });
    }
    
    condition.notify_one();
    return res;
}

} // namespace snnfw

#endif // SNNFW_THREADPOOL_H


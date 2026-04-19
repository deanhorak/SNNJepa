#include "snnfw/ThreadPool.h"
#include "snnfw/Logger.h"

namespace snnfw {

ThreadPool::ThreadPool(size_t numThreads) : stop(false) {
    if (numThreads == 0) {
        numThreads = 1; // At least one thread
    }
    
    SNNFW_INFO("Creating ThreadPool with {} worker threads", numThreads);
    
    // Create worker threads
    for (size_t i = 0; i < numThreads; ++i) {
        workers.emplace_back([this, i] {
            SNNFW_DEBUG("Worker thread {} started", i);
            
            while (true) {
                std::function<void()> task;
                
                {
                    std::unique_lock<std::mutex> lock(this->queueMutex);
                    
                    // Wait for a task or stop signal
                    this->condition.wait(lock, [this] {
                        return this->stop || !this->tasks.empty();
                    });
                    
                    // Exit if stopped and no tasks remain
                    if (this->stop && this->tasks.empty()) {
                        SNNFW_DEBUG("Worker thread {} stopping", i);
                        return;
                    }
                    
                    // Get next task
                    if (!this->tasks.empty()) {
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                }
                
                // Execute task outside the lock
                if (task) {
                    try {
                        task();
                    } catch (const std::exception& e) {
                        SNNFW_ERROR("Worker thread {} caught exception: {}", i, e.what());
                    } catch (...) {
                        SNNFW_ERROR("Worker thread {} caught unknown exception", i);
                    }
                }
            }
        });
    }
    
    SNNFW_INFO("ThreadPool initialized successfully");
}

ThreadPool::~ThreadPool() {
    SNNFW_DEBUG("ThreadPool shutting down...");
    
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true;
    }
    
    // Wake up all threads
    condition.notify_all();
    
    // Wait for all threads to finish
    for (std::thread& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    
    SNNFW_INFO("ThreadPool shutdown complete. {} tasks were pending.", tasks.size());
}

size_t ThreadPool::pendingTasks() const {
    std::unique_lock<std::mutex> lock(queueMutex);
    return tasks.size();
}

} // namespace snnfw


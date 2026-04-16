#ifndef SNNFW_DATASTORE_H
#define SNNFW_DATASTORE_H

#include "snnfw/NeuralObject.h"
#include "snnfw/Serializable.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <list>
#include <mutex>
#include <functional>
#include <cstdint>

// Forward declare RocksDB types to avoid including the header
namespace rocksdb {
    class DB;
    class Options;
}

namespace snnfw {

/**
 * @brief High-performance datastore with LRU caching for NeuralObjects
 *
 * This class provides:
 * - LRU cache for up to 1 million objects in memory
 * - Automatic persistence to RocksDB backing store
 * - Lazy loading from disk when objects not in cache
 * - Automatic dirty tracking and write-back on eviction
 * - Thread-safe operations
 * - Memory leak prevention through shared_ptr management
 *
 * Usage:
 * @code
 * Datastore datastore("./neural_db", 1000000);
 * 
 * // Store a neuron
 * auto neuron = std::make_shared<Neuron>(50.0, 0.95, 20, 123);
 * datastore.put(neuron);
 * 
 * // Retrieve a neuron (from cache or disk)
 * auto retrieved = datastore.getNeuron(123);
 * 
 * // Modify and mark dirty
 * retrieved->insertSpike(10.0);
 * datastore.markDirty(123);
 * @endcode
 */
class Datastore {
public:
    /**
     * @brief Constructor
     * @param dbPath Path to the RocksDB database directory
     * @param cacheSize Maximum number of objects to keep in cache (default: 1,000,000)
     */
    explicit Datastore(const std::string& dbPath, size_t cacheSize = 1000000);

    /**
     * @brief Destructor - flushes all dirty objects to disk
     */
    ~Datastore();

    // Prevent copying
    Datastore(const Datastore&) = delete;
    Datastore& operator=(const Datastore&) = delete;

    /**
     * @brief Store or update a NeuralObject in the datastore
     * @param obj Shared pointer to the object to store
     * @return true if successful, false otherwise
     */
    bool put(std::shared_ptr<NeuralObject> obj);

    /**
     * @brief Retrieve a NeuralObject by ID
     * @param id Object ID
     * @return Shared pointer to the object, or nullptr if not found
     */
    std::shared_ptr<NeuralObject> get(uint64_t id);

    /**
     * @brief Retrieve a Neuron by ID (type-safe convenience method)
     * @param id Neuron ID
     * @return Shared pointer to the Neuron, or nullptr if not found or wrong type
     */
    std::shared_ptr<class Neuron> getNeuron(uint64_t id);

    /**
     * @brief Retrieve an Axon by ID (type-safe convenience method)
     * @param id Axon ID
     * @return Shared pointer to the Axon, or nullptr if not found or wrong type
     */
    std::shared_ptr<class Axon> getAxon(uint64_t id);

    /**
     * @brief Retrieve a Dendrite by ID (type-safe convenience method)
     * @param id Dendrite ID
     * @return Shared pointer to the Dendrite, or nullptr if not found or wrong type
     */
    std::shared_ptr<class Dendrite> getDendrite(uint64_t id);

    /**
     * @brief Retrieve a Synapse by ID (type-safe convenience method)
     * @param id Synapse ID
     * @return Shared pointer to the Synapse, or nullptr if not found or wrong type
     */
    std::shared_ptr<class Synapse> getSynapse(uint64_t id);

    /**
     * @brief Retrieve a Cluster by ID (type-safe convenience method)
     * @param id Cluster ID
     * @return Shared pointer to the Cluster, or nullptr if not found or wrong type
     */
    std::shared_ptr<class Cluster> getCluster(uint64_t id);

    /**
     * @brief Retrieve a Layer by ID (type-safe convenience method)
     * @param id Layer ID
     * @return Shared pointer to the Layer, or nullptr if not found or wrong type
     */
    std::shared_ptr<class Layer> getLayer(uint64_t id);

    /**
     * @brief Retrieve a Column by ID (type-safe convenience method)
     * @param id Column ID
     * @return Shared pointer to the Column, or nullptr if not found or wrong type
     */
    std::shared_ptr<class Column> getColumn(uint64_t id);

    /**
     * @brief Retrieve a Nucleus by ID (type-safe convenience method)
     * @param id Nucleus ID
     * @return Shared pointer to the Nucleus, or nullptr if not found or wrong type
     */
    std::shared_ptr<class Nucleus> getNucleus(uint64_t id);

    /**
     * @brief Retrieve a Region by ID (type-safe convenience method)
     * @param id Region ID
     * @return Shared pointer to the Region, or nullptr if not found or wrong type
     */
    std::shared_ptr<class Region> getRegion(uint64_t id);

    /**
     * @brief Retrieve a Lobe by ID (type-safe convenience method)
     * @param id Lobe ID
     * @return Shared pointer to the Lobe, or nullptr if not found or wrong type
     */
    std::shared_ptr<class Lobe> getLobe(uint64_t id);

    /**
     * @brief Retrieve a Hemisphere by ID (type-safe convenience method)
     * @param id Hemisphere ID
     * @return Shared pointer to the Hemisphere, or nullptr if not found or wrong type
     */
    std::shared_ptr<class Hemisphere> getHemisphere(uint64_t id);

    /**
     * @brief Retrieve a Brain by ID (type-safe convenience method)
     * @param id Brain ID
     * @return Shared pointer to the Brain, or nullptr if not found or wrong type
     */
    std::shared_ptr<class Brain> getBrain(uint64_t id);

    /**
     * @brief Mark an object as dirty (modified) so it will be written back on eviction
     * @param id Object ID
     */
    void markDirty(uint64_t id);

    /**
     * @brief Remove an object from the datastore (cache and disk)
     * @param id Object ID
     * @return true if successful, false otherwise
     */
    bool remove(uint64_t id);

    /**
     * @brief Flush all dirty objects to disk
     * @return Number of objects flushed
     */
    size_t flushAll();

    /**
     * @brief Flush a specific object to disk if dirty
     * @param id Object ID
     * @return true if flushed, false if not dirty or not found
     */
    bool flush(uint64_t id);

    /**
     * @brief Get the current cache size
     * @return Number of objects currently in cache
     */
    size_t getCacheSize() const;

    /**
     * @brief Get the maximum cache size
     * @return Maximum number of objects that can be in cache
     */
    size_t getMaxCacheSize() const { return maxCacheSize_; }

    /**
     * @brief Get cache statistics
     * @param hits Output parameter for cache hits
     * @param misses Output parameter for cache misses
     */
    void getCacheStats(uint64_t& hits, uint64_t& misses) const;

    /**
     * @brief Clear all cache statistics
     */
    void clearCacheStats();

    /**
     * @brief Register a factory function for deserializing a specific type
     * @param typeName Type name (e.g., "Neuron")
     * @param factory Function that creates an object from JSON
     */
    void registerFactory(const std::string& typeName, 
                        std::function<std::shared_ptr<NeuralObject>(const std::string&)> factory);

private:
    /**
     * @brief LRU cache entry
     */
    struct CacheEntry {
        std::shared_ptr<NeuralObject> object;
        bool dirty;

        CacheEntry() : object(nullptr), dirty(false) {}
        CacheEntry(std::shared_ptr<NeuralObject> obj, bool isDirty = false)
            : object(std::move(obj)), dirty(isDirty) {}
    };

    /**
     * @brief Load an object from the backing store
     * @param id Object ID
     * @return Shared pointer to the loaded object, or nullptr if not found
     */
    std::shared_ptr<NeuralObject> loadFromDisk(uint64_t id);

    /**
     * @brief Save an object to the backing store
     * @param id Object ID
     * @param obj Object to save
     * @return true if successful, false otherwise
     */
    bool saveToDisk(uint64_t id, std::shared_ptr<NeuralObject> obj);

    /**
     * @brief Evict the least recently used object from cache
     * @return true if an object was evicted, false if cache was empty
     */
    bool evictLRU();

    /**
     * @brief Move an object to the front of the LRU list (mark as most recently used)
     * @param id Object ID
     */
    void touchObject(uint64_t id);

    // RocksDB instance
    std::unique_ptr<rocksdb::DB> db_;

    // LRU cache: map from ID to (iterator in LRU list, cache entry)
    std::unordered_map<uint64_t, std::pair<std::list<uint64_t>::iterator, CacheEntry>> cache_;
    
    // LRU list: most recently used at front, least recently used at back
    std::list<uint64_t> lruList_;

    // Maximum cache size
    size_t maxCacheSize_;

    // Cache statistics
    mutable uint64_t cacheHits_;
    mutable uint64_t cacheMisses_;

    // Factory functions for deserializing different types
    std::unordered_map<std::string, std::function<std::shared_ptr<NeuralObject>(const std::string&)>> factories_;

    // Mutex for thread safety
    mutable std::mutex mutex_;
};

} // namespace snnfw

#endif // SNNFW_DATASTORE_H


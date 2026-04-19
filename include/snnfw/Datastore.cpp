#include "snnfw/Datastore.h"
#include "snnfw/Logger.h"
#include "snnfw/Neuron.h"
#include "snnfw/Axon.h"
#include "snnfw/Dendrite.h"
#include "snnfw/Synapse.h"
#include "snnfw/Cluster.h"
#include "snnfw/Layer.h"
#include "snnfw/Column.h"
#include "snnfw/Nucleus.h"
#include "snnfw/Region.h"
#include "snnfw/Lobe.h"
#include "snnfw/Hemisphere.h"
#include "snnfw/Brain.h"
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace snnfw {

Datastore::Datastore(const std::string& dbPath, size_t cacheSize)
    : maxCacheSize_(cacheSize), cacheHits_(0), cacheMisses_(0) {
    
    SNNFW_INFO("Initializing Datastore with cache size: {}, path: {}", cacheSize, dbPath);
    
    // Configure RocksDB options for high performance
    rocksdb::Options options;
    options.create_if_missing = true;
    options.compression = rocksdb::kSnappyCompression;
    
    // Optimize for SSD
    options.allow_mmap_reads = true;
    options.allow_mmap_writes = false;
    
    // Increase write buffer for better write performance
    options.write_buffer_size = 64 * 1024 * 1024; // 64MB
    options.max_write_buffer_number = 3;
    options.target_file_size_base = 64 * 1024 * 1024; // 64MB
    
    // Optimize for point lookups
    options.optimize_filters_for_hits = true;
    
    // Open database
    rocksdb::DB* db_ptr;
    rocksdb::Status status = rocksdb::DB::Open(options, dbPath, &db_ptr);
    
    if (!status.ok()) {
        SNNFW_ERROR("Failed to open RocksDB at {}: {}", dbPath, status.ToString());
        throw std::runtime_error("Failed to open RocksDB: " + status.ToString());
    }
    
    db_.reset(db_ptr);
    SNNFW_INFO("RocksDB opened successfully at {}", dbPath);
}

Datastore::~Datastore() {
    SNNFW_INFO("Shutting down Datastore, flushing dirty objects...");
    size_t flushed = flushAll();
    SNNFW_INFO("Flushed {} dirty objects to disk", flushed);
    
    // Log cache statistics
    uint64_t hits, misses;
    getCacheStats(hits, misses);
    double hitRate = (hits + misses > 0) ? (100.0 * hits / (hits + misses)) : 0.0;
    SNNFW_INFO("Cache statistics - Hits: {}, Misses: {}, Hit Rate: {:.2f}%", 
               hits, misses, hitRate);
}

bool Datastore::put(std::shared_ptr<NeuralObject> obj) {
    if (!obj) {
        SNNFW_WARN("Attempted to put null object");
        return false;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    uint64_t id = obj->getId();
    
    // Check if object is already in cache
    auto it = cache_.find(id);
    if (it != cache_.end()) {
        // Update existing entry
        it->second.second.object = obj;
        it->second.second.dirty = true;
        touchObject(id);
        SNNFW_DEBUG("Updated object {} in cache", id);
        return true;
    }
    
    // Need to add to cache - check if we need to evict
    if (cache_.size() >= maxCacheSize_) {
        if (!evictLRU()) {
            SNNFW_ERROR("Failed to evict LRU object");
            return false;
        }
    }
    
    // Add to cache
    lruList_.push_front(id);
    cache_[id] = std::make_pair(lruList_.begin(), CacheEntry(obj, true));
    
    SNNFW_DEBUG("Added object {} to cache (cache size: {})", id, cache_.size());
    return true;
}

std::shared_ptr<NeuralObject> Datastore::get(uint64_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Check cache first
    auto it = cache_.find(id);
    if (it != cache_.end()) {
        cacheHits_++;
        touchObject(id);
        SNNFW_DEBUG("Cache hit for object {}", id);
        return it->second.second.object;
    }
    
    // Cache miss - load from disk
    cacheMisses_++;
    SNNFW_DEBUG("Cache miss for object {}, loading from disk", id);
    
    auto obj = loadFromDisk(id);
    if (!obj) {
        SNNFW_WARN("Object {} not found in datastore", id);
        return nullptr;
    }
    
    // Add to cache
    if (cache_.size() >= maxCacheSize_) {
        evictLRU();
    }
    
    lruList_.push_front(id);
    cache_[id] = std::make_pair(lruList_.begin(), CacheEntry(obj, false));
    
    SNNFW_DEBUG("Loaded object {} from disk and added to cache", id);
    return obj;
}

std::shared_ptr<Neuron> Datastore::getNeuron(uint64_t id) {
    auto obj = get(id);
    return std::dynamic_pointer_cast<Neuron>(obj);
}

std::shared_ptr<Axon> Datastore::getAxon(uint64_t id) {
    auto obj = get(id);
    return std::dynamic_pointer_cast<Axon>(obj);
}

std::shared_ptr<Dendrite> Datastore::getDendrite(uint64_t id) {
    auto obj = get(id);
    return std::dynamic_pointer_cast<Dendrite>(obj);
}

std::shared_ptr<Synapse> Datastore::getSynapse(uint64_t id) {
    auto obj = get(id);
    return std::dynamic_pointer_cast<Synapse>(obj);
}

std::shared_ptr<Cluster> Datastore::getCluster(uint64_t id) {
    auto obj = get(id);
    return std::dynamic_pointer_cast<Cluster>(obj);
}

std::shared_ptr<Layer> Datastore::getLayer(uint64_t id) {
    auto obj = get(id);
    return std::dynamic_pointer_cast<Layer>(obj);
}

std::shared_ptr<Column> Datastore::getColumn(uint64_t id) {
    auto obj = get(id);
    return std::dynamic_pointer_cast<Column>(obj);
}

std::shared_ptr<Nucleus> Datastore::getNucleus(uint64_t id) {
    auto obj = get(id);
    return std::dynamic_pointer_cast<Nucleus>(obj);
}

std::shared_ptr<Region> Datastore::getRegion(uint64_t id) {
    auto obj = get(id);
    return std::dynamic_pointer_cast<Region>(obj);
}

std::shared_ptr<Lobe> Datastore::getLobe(uint64_t id) {
    auto obj = get(id);
    return std::dynamic_pointer_cast<Lobe>(obj);
}

std::shared_ptr<Hemisphere> Datastore::getHemisphere(uint64_t id) {
    auto obj = get(id);
    return std::dynamic_pointer_cast<Hemisphere>(obj);
}

std::shared_ptr<Brain> Datastore::getBrain(uint64_t id) {
    auto obj = get(id);
    return std::dynamic_pointer_cast<Brain>(obj);
}

void Datastore::markDirty(uint64_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = cache_.find(id);
    if (it != cache_.end()) {
        it->second.second.dirty = true;
        SNNFW_DEBUG("Marked object {} as dirty", id);
    } else {
        SNNFW_WARN("Attempted to mark non-cached object {} as dirty", id);
    }
}

bool Datastore::remove(uint64_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Remove from cache
    auto it = cache_.find(id);
    if (it != cache_.end()) {
        lruList_.erase(it->second.first);
        cache_.erase(it);
    }
    
    // Remove from disk
    std::string key = std::to_string(id);
    rocksdb::Status status = db_->Delete(rocksdb::WriteOptions(), key);
    
    if (!status.ok()) {
        SNNFW_ERROR("Failed to delete object {} from disk: {}", id, status.ToString());
        return false;
    }
    
    SNNFW_DEBUG("Removed object {} from datastore", id);
    return true;
}

size_t Datastore::flushAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t flushed = 0;
    for (auto& entry : cache_) {
        if (entry.second.second.dirty) {
            uint64_t id = entry.first;
            if (saveToDisk(id, entry.second.second.object)) {
                entry.second.second.dirty = false;
                flushed++;
            }
        }
    }
    
    return flushed;
}

bool Datastore::flush(uint64_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = cache_.find(id);
    if (it == cache_.end() || !it->second.second.dirty) {
        return false;
    }
    
    if (saveToDisk(id, it->second.second.object)) {
        it->second.second.dirty = false;
        SNNFW_DEBUG("Flushed object {} to disk", id);
        return true;
    }
    
    return false;
}

size_t Datastore::getCacheSize() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.size();
}

void Datastore::getCacheStats(uint64_t& hits, uint64_t& misses) const {
    std::lock_guard<std::mutex> lock(mutex_);
    hits = cacheHits_;
    misses = cacheMisses_;
}

void Datastore::clearCacheStats() {
    std::lock_guard<std::mutex> lock(mutex_);
    cacheHits_ = 0;
    cacheMisses_ = 0;
}

void Datastore::registerFactory(const std::string& typeName, 
                                std::function<std::shared_ptr<NeuralObject>(const std::string&)> factory) {
    std::lock_guard<std::mutex> lock(mutex_);
    factories_[typeName] = factory;
    SNNFW_DEBUG("Registered factory for type: {}", typeName);
}

std::shared_ptr<NeuralObject> Datastore::loadFromDisk(uint64_t id) {
    std::string key = std::to_string(id);
    std::string value;
    
    rocksdb::Status status = db_->Get(rocksdb::ReadOptions(), key, &value);
    
    if (!status.ok()) {
        if (!status.IsNotFound()) {
            SNNFW_ERROR("Failed to read object {} from disk: {}", id, status.ToString());
        }
        return nullptr;
    }
    
    // Parse JSON
    try {
        json j = json::parse(value);
        std::string typeName = j["type"];
        
        // Find factory for this type
        auto factoryIt = factories_.find(typeName);
        if (factoryIt == factories_.end()) {
            SNNFW_ERROR("No factory registered for type: {}", typeName);
            return nullptr;
        }
        
        // Use factory to create object
        return factoryIt->second(value);
        
    } catch (const std::exception& e) {
        SNNFW_ERROR("Failed to deserialize object {}: {}", id, e.what());
        return nullptr;
    }
}

bool Datastore::saveToDisk(uint64_t id, std::shared_ptr<NeuralObject> obj) {
    if (!obj) {
        return false;
    }
    
    // Serialize object to JSON
    Serializable* serializable = dynamic_cast<Serializable*>(obj.get());
    if (!serializable) {
        SNNFW_ERROR("Object {} does not implement Serializable interface", id);
        return false;
    }
    
    std::string json = serializable->toJson();
    std::string key = std::to_string(id);
    
    rocksdb::Status status = db_->Put(rocksdb::WriteOptions(), key, json);
    
    if (!status.ok()) {
        SNNFW_ERROR("Failed to write object {} to disk: {}", id, status.ToString());
        return false;
    }
    
    return true;
}

bool Datastore::evictLRU() {
    if (lruList_.empty()) {
        return false;
    }
    
    // Get LRU object (at back of list)
    uint64_t lruId = lruList_.back();
    auto it = cache_.find(lruId);
    
    if (it == cache_.end()) {
        SNNFW_ERROR("LRU list inconsistency: object {} not in cache", lruId);
        lruList_.pop_back();
        return false;
    }
    
    // Flush if dirty
    if (it->second.second.dirty) {
        if (!saveToDisk(lruId, it->second.second.object)) {
            SNNFW_ERROR("Failed to flush dirty object {} during eviction", lruId);
            // Continue with eviction anyway
        }
    }
    
    // Remove from cache
    lruList_.pop_back();
    cache_.erase(it);
    
    SNNFW_DEBUG("Evicted LRU object {} from cache", lruId);
    return true;
}

void Datastore::touchObject(uint64_t id) {
    auto it = cache_.find(id);
    if (it == cache_.end()) {
        return;
    }
    
    // Move to front of LRU list
    lruList_.erase(it->second.first);
    lruList_.push_front(id);
    it->second.first = lruList_.begin();
}

} // namespace snnfw


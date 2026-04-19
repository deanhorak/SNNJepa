#include "snnfw/InteractionManager.h"
#include "snnfw/Camera.h"
#include "snnfw/NetworkDataAdapter.h"
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <limits>
#include <cmath>

namespace snnfw {

InteractionManager::InteractionManager(NetworkDataAdapter& adapter)
    : adapter_(adapter)
    , hoveredNeuronId_(0)
    , highlightColor_(1.0f, 0.8f, 0.2f, 1.0f)  // Gold color
    , highlightIntensity_(2.0f)
    , pickRadius_(1.0f) {
}

InteractionManager::~InteractionManager() {
}

PickResult InteractionManager::pickNeuron(float mouseX, float mouseY, 
                                         int screenWidth, int screenHeight,
                                         const Camera& camera) {
    PickResult result;
    
    // Convert mouse coordinates to normalized device coordinates
    float ndcX = (2.0f * mouseX) / screenWidth - 1.0f;
    float ndcY = 1.0f - (2.0f * mouseY) / screenHeight;
    
    // Create ray in clip space
    glm::vec4 rayClip(ndcX, ndcY, -1.0f, 1.0f);
    
    // Transform to view space
    glm::mat4 projection = camera.getProjectionMatrix(
        static_cast<float>(screenWidth) / static_cast<float>(screenHeight));
    glm::vec4 rayView = glm::inverse(projection) * rayClip;
    rayView = glm::vec4(rayView.x, rayView.y, -1.0f, 0.0f);
    
    // Transform to world space
    glm::vec4 rayWorld = glm::inverse(camera.getViewMatrix()) * rayView;
    glm::vec3 rayDir = glm::normalize(glm::vec3(rayWorld));
    
    // Ray origin is camera position
    glm::vec3 rayOrigin = camera.getPosition();
    
    // Test intersection with all neurons
    float closestDistance = std::numeric_limits<float>::max();
    uint64_t closestNeuronId = 0;
    glm::vec3 closestPosition(0.0f);
    
    const auto& neurons = adapter_.getNeurons();
    for (const auto& neuron : neurons) {
        // Skip if filtered out
        if (!isNeuronVisible(neuron.id)) continue;

        glm::vec3 neuronPos(neuron.position.x, neuron.position.y, neuron.position.z);
        float distance;
        if (raySphereIntersection(rayOrigin, rayDir, neuronPos,
                                 pickRadius_, distance)) {
            if (distance < closestDistance) {
                closestDistance = distance;
                closestNeuronId = neuron.id;
                closestPosition = neuronPos;
            }
        }
    }
    
    if (closestNeuronId != 0) {
        result.hit = true;
        result.neuronId = closestNeuronId;
        result.distance = closestDistance;
        result.worldPosition = closestPosition;
    }
    
    return result;
}

bool InteractionManager::raySphereIntersection(const glm::vec3& rayOrigin, 
                                              const glm::vec3& rayDir,
                                              const glm::vec3& sphereCenter, 
                                              float sphereRadius,
                                              float& distance) const {
    glm::vec3 oc = rayOrigin - sphereCenter;
    float a = glm::dot(rayDir, rayDir);
    float b = 2.0f * glm::dot(oc, rayDir);
    float c = glm::dot(oc, oc) - sphereRadius * sphereRadius;
    float discriminant = b * b - 4 * a * c;
    
    if (discriminant < 0) {
        return false;
    }
    
    float t = (-b - std::sqrt(discriminant)) / (2.0f * a);
    if (t < 0) {
        t = (-b + std::sqrt(discriminant)) / (2.0f * a);
    }
    
    if (t < 0) {
        return false;
    }
    
    distance = t;
    return true;
}

void InteractionManager::selectNeuron(uint64_t neuronId, SelectionMode mode) {
    if (neuronId == 0) return;
    
    switch (mode) {
        case SelectionMode::SINGLE:
            selectedNeurons_.clear();
            selectedNeurons_.insert(neuronId);
            break;
            
        case SelectionMode::ADDITIVE:
            selectedNeurons_.insert(neuronId);
            break;
            
        case SelectionMode::SUBTRACTIVE:
            selectedNeurons_.erase(neuronId);
            break;
            
        case SelectionMode::TOGGLE:
            if (selectedNeurons_.count(neuronId)) {
                selectedNeurons_.erase(neuronId);
            } else {
                selectedNeurons_.insert(neuronId);
            }
            break;
    }
    
    notifySelectionChanged();
}

void InteractionManager::deselectNeuron(uint64_t neuronId) {
    selectedNeurons_.erase(neuronId);
    notifySelectionChanged();
}

void InteractionManager::clearSelection() {
    selectedNeurons_.clear();
    notifySelectionChanged();
}

void InteractionManager::selectAll() {
    selectedNeurons_.clear();
    const auto& neurons = adapter_.getNeurons();
    for (const auto& neuron : neurons) {
        if (isNeuronVisible(neuron.id)) {
            selectedNeurons_.insert(neuron.id);
        }
    }
    notifySelectionChanged();
}

void InteractionManager::invertSelection() {
    std::unordered_set<uint64_t> newSelection;
    const auto& neurons = adapter_.getNeurons();
    for (const auto& neuron : neurons) {
        if (isNeuronVisible(neuron.id) && !selectedNeurons_.count(neuron.id)) {
            newSelection.insert(neuron.id);
        }
    }
    selectedNeurons_ = std::move(newSelection);
    notifySelectionChanged();
}

bool InteractionManager::isSelected(uint64_t neuronId) const {
    return selectedNeurons_.count(neuronId) > 0;
}

void InteractionManager::selectByActivity(float minActivity, float maxActivity) {
    selectedNeurons_.clear();
    const auto& neurons = adapter_.getNeurons();
    for (const auto& neuron : neurons) {
        if (!isNeuronVisible(neuron.id)) continue;
        
        float activity = getNeuronActivity(neuron.id);
        if (activity >= minActivity && activity <= maxActivity) {
            selectedNeurons_.insert(neuron.id);
        }
    }
    notifySelectionChanged();
}

void InteractionManager::selectByConnectivity(int minConnections, int maxConnections) {
    selectedNeurons_.clear();
    const auto& neurons = adapter_.getNeurons();
    for (const auto& neuron : neurons) {
        if (!isNeuronVisible(neuron.id)) continue;
        
        int connections = getNeuronConnectionCount(neuron.id);
        if (connections >= minConnections && connections <= maxConnections) {
            selectedNeurons_.insert(neuron.id);
        }
    }
    notifySelectionChanged();
}

void InteractionManager::selectNeighbors(uint64_t neuronId, int hops) {
    if (neuronId == 0 || hops <= 0) return;
    
    std::unordered_set<uint64_t> visited;
    std::vector<uint64_t> currentLevel = {neuronId};
    
    for (int hop = 0; hop < hops; ++hop) {
        std::vector<uint64_t> nextLevel;
        
        for (uint64_t currentId : currentLevel) {
            if (visited.count(currentId)) continue;
            visited.insert(currentId);
            
            // Find all synapses connected to this neuron
            const auto& synapses = adapter_.getSynapses();
            for (const auto& synapse : synapses) {
                if (synapse.sourceNeuronId == currentId) {
                    nextLevel.push_back(synapse.targetNeuronId);
                } else if (synapse.targetNeuronId == currentId) {
                    nextLevel.push_back(synapse.sourceNeuronId);
                }
            }
        }
        
        currentLevel = std::move(nextLevel);
    }
    
    // Add all visited neurons to selection
    for (uint64_t id : visited) {
        if (isNeuronVisible(id)) {
            selectedNeurons_.insert(id);
        }
    }
    
    notifySelectionChanged();
}

void InteractionManager::addBookmark(const std::string& name, const Camera& camera) {
    // Remove existing bookmark with same name
    removeBookmark(name);

    CameraBookmark bookmark;
    bookmark.name = name;
    bookmark.position = camera.getPosition();
    bookmark.target = camera.getTarget();
    bookmark.fov = camera.getFOV();

    bookmarks_.push_back(bookmark);
}

void InteractionManager::removeBookmark(const std::string& name) {
    bookmarks_.erase(
        std::remove_if(bookmarks_.begin(), bookmarks_.end(),
                      [&name](const CameraBookmark& b) { return b.name == name; }),
        bookmarks_.end());
}

void InteractionManager::applyBookmark(const std::string& name, Camera& camera) {
    for (const auto& bookmark : bookmarks_) {
        if (bookmark.name == name) {
            camera.setPosition(bookmark.position);
            camera.lookAt(bookmark.target);
            camera.setFOV(bookmark.fov);
            return;
        }
    }
}

bool InteractionManager::hasBookmark(const std::string& name) const {
    return std::any_of(bookmarks_.begin(), bookmarks_.end(),
                      [&name](const CameraBookmark& b) { return b.name == name; });
}

void InteractionManager::clearBookmarks() {
    bookmarks_.clear();
}

void InteractionManager::setFilter(const NeuronFilter& filter) {
    filter_ = filter;
}

bool InteractionManager::isNeuronVisible(uint64_t neuronId) const {
    // Check type filter
    if (filter_.filterByType) {
        bool excitatory = isExcitatory(neuronId);
        if (excitatory && !filter_.showExcitatory) return false;
        if (!excitatory && !filter_.showInhibitory) return false;
    }
    
    // Check activity filter
    if (filter_.filterByActivity) {
        float activity = getNeuronActivity(neuronId);
        if (activity < filter_.minActivity || activity > filter_.maxActivity) {
            return false;
        }
    }
    
    // Check connectivity filter
    if (filter_.filterByConnectivity) {
        int connections = getNeuronConnectionCount(neuronId);
        if (connections < filter_.minConnections || connections > filter_.maxConnections) {
            return false;
        }
    }
    
    return true;
}

std::vector<uint64_t> InteractionManager::getVisibleNeurons() const {
    std::vector<uint64_t> visible;
    const auto& neurons = adapter_.getNeurons();
    for (const auto& neuron : neurons) {
        if (isNeuronVisible(neuron.id)) {
            visible.push_back(neuron.id);
        }
    }
    return visible;
}

void InteractionManager::focusOnNeuron(uint64_t neuronId, Camera& camera, float distance) {
    glm::vec3 position = getNeuronPosition(neuronId);
    if (position == glm::vec3(0.0f)) return;  // Neuron not found
    
    // Calculate camera position (offset from neuron)
    glm::vec3 offset = glm::normalize(camera.getPosition() - camera.getTarget()) * distance;
    camera.setPosition(position + offset);
    camera.lookAt(position);
}

void InteractionManager::focusOnSelection(Camera& camera, float padding) {
    if (selectedNeurons_.empty()) return;
    
    glm::vec3 center = getSelectionCenter();
    
    // Calculate bounding sphere radius
    float maxRadius = 0.0f;
    for (uint64_t neuronId : selectedNeurons_) {
        glm::vec3 pos = getNeuronPosition(neuronId);
        float dist = glm::length(pos - center);
        maxRadius = std::max(maxRadius, dist);
    }
    
    // Calculate camera distance to fit all neurons
    float distance = maxRadius * padding / std::tan(glm::radians(camera.getFOV() / 2.0f));
    
    // Position camera
    glm::vec3 offset = glm::normalize(camera.getPosition() - camera.getTarget()) * distance;
    camera.setPosition(center + offset);
    camera.lookAt(center);
}

glm::vec3 InteractionManager::getSelectionCenter() const {
    if (selectedNeurons_.empty()) return glm::vec3(0.0f);
    
    glm::vec3 sum(0.0f);
    for (uint64_t neuronId : selectedNeurons_) {
        sum += getNeuronPosition(neuronId);
    }
    return sum / static_cast<float>(selectedNeurons_.size());
}

InteractionManager::SelectionStats InteractionManager::getSelectionStats() const {
    SelectionStats stats;
    stats.totalSelected = selectedNeurons_.size();
    stats.excitatoryCount = 0;
    stats.inhibitoryCount = 0;
    stats.avgActivity = 0.0f;
    stats.avgConnections = 0;
    
    if (selectedNeurons_.empty()) return stats;
    
    float totalActivity = 0.0f;
    int totalConnections = 0;
    
    for (uint64_t neuronId : selectedNeurons_) {
        if (isExcitatory(neuronId)) {
            stats.excitatoryCount++;
        } else {
            stats.inhibitoryCount++;
        }
        
        totalActivity += getNeuronActivity(neuronId);
        totalConnections += getNeuronConnectionCount(neuronId);
    }
    
    stats.avgActivity = totalActivity / static_cast<float>(selectedNeurons_.size());
    stats.avgConnections = totalConnections / static_cast<int>(selectedNeurons_.size());
    
    return stats;
}

glm::vec3 InteractionManager::getNeuronPosition(uint64_t neuronId) const {
    const auto& neurons = adapter_.getNeurons();
    for (const auto& neuron : neurons) {
        if (neuron.id == neuronId) {
            return glm::vec3(neuron.position.x, neuron.position.y, neuron.position.z);
        }
    }
    return glm::vec3(0.0f);
}

float InteractionManager::getNeuronActivity(uint64_t neuronId) const {
    const auto& neurons = adapter_.getNeurons();
    for (const auto& neuron : neurons) {
        if (neuron.id == neuronId) {
            return neuron.activity;
        }
    }
    return 0.0f;
}

int InteractionManager::getNeuronConnectionCount(uint64_t neuronId) const {
    int count = 0;
    const auto& synapses = adapter_.getSynapses();
    for (const auto& synapse : synapses) {
        if (synapse.sourceNeuronId == neuronId || synapse.targetNeuronId == neuronId) {
            count++;
        }
    }
    return count;
}

bool InteractionManager::isExcitatory(uint64_t neuronId) const {
    const auto& neurons = adapter_.getNeurons();
    for (const auto& neuron : neurons) {
        if (neuron.id == neuronId) {
            return neuron.isExcitatory;
        }
    }
    return true;  // Default to excitatory
}

void InteractionManager::notifySelectionChanged() {
    if (selectionCallback_) {
        selectionCallback_(selectedNeurons_);
    }
}

} // namespace snnfw


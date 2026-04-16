#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <unordered_set>
#include <functional>
#include <string>
#include <memory>

namespace snnfw {

// Forward declarations
class Camera;
class NetworkDataAdapter;

/**
 * @brief Represents a camera bookmark for quick navigation
 */
struct CameraBookmark {
    std::string name;
    glm::vec3 position;
    glm::vec3 target;
    float fov;
    
    CameraBookmark() : position(0.0f), target(0.0f), fov(45.0f) {}
    CameraBookmark(const std::string& n, const glm::vec3& pos, const glm::vec3& tgt, float f)
        : name(n), position(pos), target(tgt), fov(f) {}
};

/**
 * @brief Represents a picked object in the scene
 */
struct PickResult {
    bool hit;
    uint64_t neuronId;
    float distance;
    glm::vec3 worldPosition;
    
    PickResult() : hit(false), neuronId(0), distance(0.0f), worldPosition(0.0f) {}
};

/**
 * @brief Selection mode for neurons
 */
enum class SelectionMode {
    SINGLE,      // Single selection (replaces previous)
    ADDITIVE,    // Add to selection (Ctrl+Click)
    SUBTRACTIVE, // Remove from selection (Alt+Click)
    TOGGLE       // Toggle selection (Shift+Click)
};

/**
 * @brief Filter criteria for neurons
 */
struct NeuronFilter {
    bool filterByType;
    bool showExcitatory;
    bool showInhibitory;
    
    bool filterByActivity;
    float minActivity;
    float maxActivity;
    
    bool filterByConnectivity;
    int minConnections;
    int maxConnections;
    
    NeuronFilter()
        : filterByType(false)
        , showExcitatory(true)
        , showInhibitory(true)
        , filterByActivity(false)
        , minActivity(0.0f)
        , maxActivity(1.0f)
        , filterByConnectivity(false)
        , minConnections(0)
        , maxConnections(1000) {}
};

/**
 * @brief Manages user interaction with the visualization
 * 
 * Features:
 * - Mouse picking for neuron selection
 * - Selection highlighting
 * - Camera bookmarks
 * - Filtering by type/activity/connectivity
 * - Selection callbacks
 */
class InteractionManager {
public:
    explicit InteractionManager(NetworkDataAdapter& adapter);
    ~InteractionManager();
    
    // Mouse picking
    PickResult pickNeuron(float mouseX, float mouseY, int screenWidth, int screenHeight,
                         const Camera& camera);
    
    // Selection management
    void selectNeuron(uint64_t neuronId, SelectionMode mode = SelectionMode::SINGLE);
    void deselectNeuron(uint64_t neuronId);
    void clearSelection();
    void selectAll();
    void invertSelection();
    bool isSelected(uint64_t neuronId) const;
    const std::unordered_set<uint64_t>& getSelectedNeurons() const { return selectedNeurons_; }
    size_t getSelectionCount() const { return selectedNeurons_.size(); }
    
    // Selection by criteria
    void selectByActivity(float minActivity, float maxActivity);
    void selectByConnectivity(int minConnections, int maxConnections);
    void selectNeighbors(uint64_t neuronId, int hops = 1);
    
    // Highlighting
    void setHighlightColor(const glm::vec4& color) { highlightColor_ = color; }
    glm::vec4 getHighlightColor() const { return highlightColor_; }
    void setHighlightIntensity(float intensity) { highlightIntensity_ = intensity; }
    float getHighlightIntensity() const { return highlightIntensity_; }
    
    // Camera bookmarks
    void addBookmark(const std::string& name, const Camera& camera);
    void removeBookmark(const std::string& name);
    void applyBookmark(const std::string& name, Camera& camera);
    bool hasBookmark(const std::string& name) const;
    const std::vector<CameraBookmark>& getBookmarks() const { return bookmarks_; }
    void clearBookmarks();
    
    // Filtering
    void setFilter(const NeuronFilter& filter);
    const NeuronFilter& getFilter() const { return filter_; }
    bool isNeuronVisible(uint64_t neuronId) const;
    std::vector<uint64_t> getVisibleNeurons() const;
    
    // Focus/navigation
    void focusOnNeuron(uint64_t neuronId, Camera& camera, float distance = 20.0f);
    void focusOnSelection(Camera& camera, float padding = 1.5f);
    glm::vec3 getSelectionCenter() const;
    
    // Callbacks
    using SelectionCallback = std::function<void(const std::unordered_set<uint64_t>&)>;
    void setSelectionCallback(SelectionCallback callback) { selectionCallback_ = callback; }
    
    // Hover support
    void setHoveredNeuron(uint64_t neuronId) { hoveredNeuronId_ = neuronId; }
    void clearHover() { hoveredNeuronId_ = 0; }
    uint64_t getHoveredNeuron() const { return hoveredNeuronId_; }
    bool isHovered(uint64_t neuronId) const { return hoveredNeuronId_ == neuronId; }
    
    // Statistics
    struct SelectionStats {
        size_t totalSelected;
        size_t excitatoryCount;
        size_t inhibitoryCount;
        float avgActivity;
        int avgConnections;
    };
    SelectionStats getSelectionStats() const;
    
private:
    // Ray-sphere intersection for picking
    bool raySphereIntersection(const glm::vec3& rayOrigin, const glm::vec3& rayDir,
                              const glm::vec3& sphereCenter, float sphereRadius,
                              float& distance) const;
    
    // Get neuron position from adapter
    glm::vec3 getNeuronPosition(uint64_t neuronId) const;
    
    // Get neuron activity level
    float getNeuronActivity(uint64_t neuronId) const;
    
    // Get neuron connection count
    int getNeuronConnectionCount(uint64_t neuronId) const;
    
    // Check if neuron is excitatory
    bool isExcitatory(uint64_t neuronId) const;
    
    // Notify selection changed
    void notifySelectionChanged();

private:
    NetworkDataAdapter& adapter_;
    
    // Selection state
    std::unordered_set<uint64_t> selectedNeurons_;
    uint64_t hoveredNeuronId_;
    
    // Highlighting
    glm::vec4 highlightColor_;
    float highlightIntensity_;
    
    // Camera bookmarks
    std::vector<CameraBookmark> bookmarks_;
    
    // Filtering
    NeuronFilter filter_;
    
    // Callbacks
    SelectionCallback selectionCallback_;
    
    // Picking parameters
    float pickRadius_;  // Radius for neuron picking (in world units)
};

} // namespace snnfw


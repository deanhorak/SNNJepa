#ifndef SNNFW_LAYOUT_ENGINE_H
#define SNNFW_LAYOUT_ENGINE_H

#include "snnfw/NetworkDataAdapter.h"
#include <vector>
#include <map>
#include <memory>
#include <functional>

namespace snnfw {

/**
 * @brief Layout algorithm type
 */
enum class LayoutAlgorithm {
    HIERARCHICAL_TREE,      ///< Tree-based hierarchical layout
    HIERARCHICAL_GROUPED,   ///< Hierarchical layout with spatial grouping by level
    FORCE_DIRECTED,         ///< Physics-based force-directed layout
    GRID,                   ///< Regular grid layout
    CIRCULAR,               ///< Circular/radial layout
    LAYERED,                ///< Layered (Sugiyama-style) layout
    ANATOMICAL              ///< Anatomically-inspired spatial layout
};

/**
 * @brief Layout configuration parameters
 */
struct LayoutConfig {
    LayoutAlgorithm algorithm;      ///< Layout algorithm to use
    
    // Spacing parameters
    float neuronSpacing;            ///< Minimum distance between neurons
    float clusterSpacing;           ///< Minimum distance between clusters
    float layerSpacing;             ///< Minimum distance between layers
    float columnSpacing;            ///< Minimum distance between columns
    float nucleusSpacing;           ///< Minimum distance between nuclei
    float regionSpacing;            ///< Minimum distance between regions
    float lobeSpacing;              ///< Minimum distance between lobes
    float hemisphereSpacing;        ///< Minimum distance between hemispheres
    
    // Hierarchical tree parameters
    float treeVerticalSpacing;      ///< Vertical spacing between hierarchy levels
    float treeHorizontalSpread;     ///< Horizontal spread factor
    bool treeBalanced;              ///< Balance tree layout
    
    // Force-directed parameters
    float springConstant;           ///< Spring constant for edges
    float repulsionConstant;        ///< Repulsion constant for nodes
    float dampingFactor;            ///< Damping factor (0-1)
    int maxIterations;              ///< Maximum iterations for convergence
    float convergenceThreshold;     ///< Threshold for convergence
    
    // Grid parameters
    int gridColumns;                ///< Number of columns in grid
    float gridCellSize;             ///< Size of each grid cell
    
    // Circular parameters
    float circularRadius;           ///< Radius for circular layout
    bool circularLayered;           ///< Use concentric circles for layers
    
    // Anatomical parameters
    bool anatomicalPreserveTopology; ///< Preserve topological relationships
    float anatomicalScaleFactor;    ///< Scale factor for anatomical positions
    
    // General parameters
    bool centerLayout;              ///< Center the layout at origin
    bool normalizePositions;        ///< Normalize positions to unit cube
    float boundingBoxSize;          ///< Size of bounding box (if normalizing)
    bool overrideStoredPositions;   ///< Override positions that are already set (default: false)

    LayoutConfig() :
        algorithm(LayoutAlgorithm::HIERARCHICAL_TREE),
        neuronSpacing(1.0f),
        clusterSpacing(5.0f),
        layerSpacing(10.0f),
        columnSpacing(15.0f),
        nucleusSpacing(25.0f),
        regionSpacing(40.0f),
        lobeSpacing(60.0f),
        hemisphereSpacing(100.0f),
        treeVerticalSpacing(20.0f),
        treeHorizontalSpread(1.5f),
        treeBalanced(true),
        springConstant(0.1f),
        repulsionConstant(100.0f),
        dampingFactor(0.9f),
        maxIterations(1000),
        convergenceThreshold(0.01f),
        gridColumns(10),
        gridCellSize(5.0f),
        circularRadius(50.0f),
        circularLayered(true),
        anatomicalPreserveTopology(true),
        anatomicalScaleFactor(1.0f),
        centerLayout(true),
        normalizePositions(false),
        boundingBoxSize(100.0f),
        overrideStoredPositions(false) {}
};

/**
 * @class LayoutEngine
 * @brief Computes spatial layouts for neural network visualization
 * 
 * The LayoutEngine takes network structure data from NetworkDataAdapter
 * and computes 3D positions for neurons that preserve the hierarchical
 * organization while providing clear visualization.
 * 
 * Supported Layout Algorithms:
 * 
 * 1. **Hierarchical Tree Layout** (Default)
 *    - Organizes network as a tree with hierarchy levels
 *    - Brain at top, neurons at bottom
 *    - Preserves parent-child relationships
 *    - Good for understanding structure
 * 
 * 2. **Force-Directed Layout**
 *    - Physics-based simulation
 *    - Synapses act as springs
 *    - Neurons repel each other
 *    - Good for revealing connectivity patterns
 * 
 * 3. **Grid Layout**
 *    - Regular grid arrangement
 *    - Simple and predictable
 *    - Good for small networks
 * 
 * 4. **Circular Layout**
 *    - Radial arrangement
 *    - Concentric circles for layers
 *    - Good for symmetric networks
 * 
 * 5. **Layered Layout**
 *    - Sugiyama-style layered graph
 *    - Minimizes edge crossings
 *    - Good for feedforward networks
 * 
 * 6. **Anatomical Layout**
 *    - Inspired by biological brain structure
 *    - Preserves topological relationships
 *    - Good for biologically-realistic networks
 * 
 * Usage:
 * ```cpp
 * NetworkDataAdapter adapter(datastore, inspector);
 * adapter.extractNetwork(brainId);
 * 
 * LayoutEngine layout;
 * LayoutConfig config;
 * config.algorithm = LayoutAlgorithm::HIERARCHICAL_TREE;
 * config.neuronSpacing = 2.0f;
 * 
 * layout.computeLayout(adapter, config);
 * 
 * // Positions are now updated in adapter's neuron data
 * auto neurons = adapter.getNeurons();
 * ```
 */
class LayoutEngine {
public:
    /**
     * @brief Constructor
     */
    LayoutEngine();
    
    /**
     * @brief Compute layout for network data
     * @param adapter Network data adapter (positions will be updated)
     * @param config Layout configuration
     * @return True if successful, false otherwise
     */
    bool computeLayout(NetworkDataAdapter& adapter, const LayoutConfig& config);
    
    /**
     * @brief Compute hierarchical tree layout
     * @param adapter Network data adapter
     * @param config Layout configuration
     * @return True if successful
     */
    bool computeHierarchicalTreeLayout(NetworkDataAdapter& adapter, const LayoutConfig& config);

    /**
     * @brief Compute hierarchical grouped layout
     * Spatially groups clusters within layers, layers within columns, etc.
     * @param adapter Network data adapter
     * @param config Layout configuration
     * @return True if successful
     */
    bool computeHierarchicalGroupedLayout(NetworkDataAdapter& adapter, const LayoutConfig& config);
    
    /**
     * @brief Compute force-directed layout
     * @param adapter Network data adapter
     * @param config Layout configuration
     * @return True if successful
     */
    bool computeForceDirectedLayout(NetworkDataAdapter& adapter, const LayoutConfig& config);
    
    /**
     * @brief Compute grid layout
     * @param adapter Network data adapter
     * @param config Layout configuration
     * @return True if successful
     */
    bool computeGridLayout(NetworkDataAdapter& adapter, const LayoutConfig& config);
    
    /**
     * @brief Compute circular layout
     * @param adapter Network data adapter
     * @param config Layout configuration
     * @return True if successful
     */
    bool computeCircularLayout(NetworkDataAdapter& adapter, const LayoutConfig& config);
    
    /**
     * @brief Compute layered layout
     * @param adapter Network data adapter
     * @param config Layout configuration
     * @return True if successful
     */
    bool computeLayeredLayout(NetworkDataAdapter& adapter, const LayoutConfig& config);
    
    /**
     * @brief Compute anatomical layout
     * @param adapter Network data adapter
     * @param config Layout configuration
     * @return True if successful
     */
    bool computeAnatomicalLayout(NetworkDataAdapter& adapter, const LayoutConfig& config);
    
    /**
     * @brief Get bounding box of current layout
     * @param min Output: minimum corner
     * @param max Output: maximum corner
     */
    void getBoundingBox(Position3D& min, Position3D& max) const;
    
    /**
     * @brief Center layout at origin
     * @param adapter Network data adapter
     */
    void centerLayout(NetworkDataAdapter& adapter);
    
    /**
     * @brief Normalize positions to fit in bounding box
     * @param adapter Network data adapter
     * @param boxSize Size of bounding box
     */
    void normalizePositions(NetworkDataAdapter& adapter, float boxSize);
    
    /**
     * @brief Set progress callback for long-running layouts
     * @param callback Function called with progress (0.0 to 1.0)
     */
    void setProgressCallback(std::function<void(float)> callback) {
        progressCallback_ = callback;
    }

private:
    // Hierarchical tree layout helpers
    struct TreeNode {
        uint64_t id;
        std::string type;
        std::vector<TreeNode*> children;
        Position3D position;
        float width;  // Horizontal space needed
        TreeNode* parent;
        
        TreeNode() : id(0), width(0.0f), parent(nullptr) {}
    };
    
    TreeNode* buildHierarchyTree(NetworkDataAdapter& adapter);
    void computeTreePositions(TreeNode* node, const LayoutConfig& config, int depth);
    void assignTreePositions(TreeNode* node, NetworkDataAdapter& adapter);
    void assignTreePositions(TreeNode* node, NetworkDataAdapter& adapter, const LayoutConfig& config);
    void deleteTree(TreeNode* node);
    
    // Force-directed layout helpers
    struct ForceNode {
        uint64_t neuronId;
        Position3D position;
        Position3D velocity;
        Position3D force;
        float mass;
        
        ForceNode() : neuronId(0), mass(1.0f) {}
    };
    
    void initializeForceNodes(NetworkDataAdapter& adapter, std::vector<ForceNode>& nodes);
    void computeForces(std::vector<ForceNode>& nodes, 
                      const std::vector<SynapseVisualData>& synapses,
                      const LayoutConfig& config);
    void updatePositions(std::vector<ForceNode>& nodes, const LayoutConfig& config, float dt);
    bool hasConverged(const std::vector<ForceNode>& nodes, float threshold);
    
    // Grid layout helpers
    void assignGridPositions(NetworkDataAdapter& adapter, const LayoutConfig& config);
    
    // Circular layout helpers
    void assignCircularPositions(NetworkDataAdapter& adapter, const LayoutConfig& config);
    
    // Layered layout helpers
    void assignLayeredPositions(NetworkDataAdapter& adapter, const LayoutConfig& config);
    
    // Anatomical layout helpers
    void assignAnatomicalPositions(NetworkDataAdapter& adapter, const LayoutConfig& config);

    // Hierarchical grouped layout helpers
    void assignHierarchicalGroupedPositions(NetworkDataAdapter& adapter, const LayoutConfig& config);

    // Utility functions
    void reportProgress(float progress);
    
    // Progress callback
    std::function<void(float)> progressCallback_;

    // Bounding box
    Position3D boundingBoxMin_;
    Position3D boundingBoxMax_;

    // Current config (for use in helper functions)
    LayoutConfig currentConfig_;
};

} // namespace snnfw

#endif // SNNFW_LAYOUT_ENGINE_H


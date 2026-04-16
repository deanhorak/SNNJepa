#ifndef SNNFW_VISUALIZATION_MANAGER_H
#define SNNFW_VISUALIZATION_MANAGER_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <string>
#include <memory>
#include <functional>

namespace snnfw {

// Forward declarations
class ActivityMonitor;
class NetworkInspector;
class Datastore;
class RecordingManager;
struct SimulationConfig;

/**
 * @brief Main visualization manager class
 * 
 * Manages GLFW window, OpenGL context, and rendering loop.
 * Provides integration with ImGui for UI controls.
 */
class VisualizationManager {
public:
    /**
     * @brief Constructor
     *
     * @param width Window width in pixels
     * @param height Window height in pixels
     * @param title Window title
     */
    VisualizationManager(int width, int height, const std::string& title);

    /**
     * @brief Constructor with configuration
     *
     * @param config Simulation configuration
     */
    explicit VisualizationManager(const SimulationConfig& config);

    /**
     * @brief Destructor - cleans up resources
     */
    ~VisualizationManager();
    
    /**
     * @brief Run main rendering loop
     * 
     * Blocks until window is closed
     */
    void run();
    
    /**
     * @brief Close the window
     */
    void close();
    
    /**
     * @brief Check if window should close
     * 
     * @return true if window should close
     */
    bool shouldClose() const;
    
    /**
     * @brief Set target FPS
     * 
     * @param fps Target frames per second (0 = unlimited)
     */
    void setTargetFPS(int fps);
    
    /**
     * @brief Enable/disable VSync
     * 
     * @param enable true to enable VSync
     */
    void enableVSync(bool enable);
    
    /**
     * @brief Set background color
     * 
     * @param r Red component (0-1)
     * @param g Green component (0-1)
     * @param b Blue component (0-1)
     */
    void setBackgroundColor(float r, float g, float b);
    
    /**
     * @brief Set render callback
     * 
     * Called every frame for custom rendering
     * 
     * @param callback Render callback function
     */
    void setRenderCallback(std::function<void(double)> callback);
    
    /**
     * @brief Set update callback
     *
     * Called every frame for logic updates
     *
     * @param callback Update callback function (receives delta time)
     */
    void setUpdateCallback(std::function<void(double)> callback);

    /**
     * @brief Set mouse click callback
     *
     * Called when mouse button is clicked
     *
     * @param callback Mouse click callback (button, action, mods, x, y)
     */
    void setMouseClickCallback(std::function<void(int, int, int, double, double)> callback);

    /**
     * @brief Set data sources for visualization
     *
     * @param monitor Activity monitor instance
     */
    void setActivityMonitor(ActivityMonitor* monitor);
    
    /**
     * @brief Set network inspector
     * 
     * @param inspector Network inspector instance
     */
    void setNetworkInspector(NetworkInspector* inspector);
    
    /**
     * @brief Set datastore
     *
     * @param datastore Datastore instance
     */
    void setDatastore(Datastore* datastore);

    /**
     * @brief Set recording manager for playback mode
     *
     * @param recordingManager Recording manager instance
     */
    void setRecordingManager(RecordingManager* recordingManager);

    /**
     * @brief Enable playback mode
     *
     * @param enable true to enable playback mode
     */
    void enablePlaybackMode(bool enable);

    /**
     * @brief Check if in playback mode
     *
     * @return true if in playback mode
     */
    bool isPlaybackMode() const { return playbackMode_; }

    /**
     * @brief Check if visualization is enabled
     *
     * @return true if visualization is enabled (window and OpenGL initialized)
     */
    bool isVisualizationEnabled() const { return visualizationEnabled_; }

    /**
     * @brief Get window handle
     *
     * @return GLFW window pointer
     */
    GLFWwindow* getWindow() const { return window_; }
    
    /**
     * @brief Get window width
     * 
     * @return Window width in pixels
     */
    int getWidth() const { return width_; }
    
    /**
     * @brief Get window height
     * 
     * @return Window height in pixels
     */
    int getHeight() const { return height_; }
    
    /**
     * @brief Get aspect ratio
     * 
     * @return Width / height
     */
    float getAspectRatio() const { return static_cast<float>(width_) / static_cast<float>(height_); }
    
    /**
     * @brief Get delta time
     * 
     * @return Time since last frame in seconds
     */
    double getDeltaTime() const { return deltaTime_; }
    
    /**
     * @brief Get current FPS
     * 
     * @return Frames per second
     */
    double getFPS() const { return fps_; }

private:
    /**
     * @brief Initialize GLFW
     */
    void initGLFW();
    
    /**
     * @brief Initialize GLAD (OpenGL loader)
     */
    void initGLAD();
    
    /**
     * @brief Initialize ImGui
     */
    void initImGui();
    
    /**
     * @brief Process input events
     */
    void processInput();
    
    /**
     * @brief Render frame
     */
    void render();
    
    /**
     * @brief Update logic
     * 
     * @param deltaTime Time since last frame
     */
    void update(double deltaTime);
    
    /**
     * @brief Cleanup resources
     */
    void cleanup();
    
    /**
     * @brief GLFW error callback
     */
    static void glfwErrorCallback(int error, const char* description);
    
    /**
     * @brief GLFW framebuffer size callback
     */
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    
    /**
     * @brief GLFW key callback
     */
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    
    /**
     * @brief GLFW mouse button callback
     */
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    
    /**
     * @brief GLFW cursor position callback
     */
    static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    
    /**
     * @brief GLFW scroll callback
     */
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    
    GLFWwindow* window_;                          ///< GLFW window handle
    int width_;                                   ///< Window width
    int height_;                                  ///< Window height
    std::string title_;                           ///< Window title
    
    bool vsync_;                                  ///< VSync enabled
    int targetFPS_;                               ///< Target FPS (0 = unlimited)
    glm::vec3 backgroundColor_;                   ///< Background color
    
    double lastFrameTime_;                        ///< Time of last frame
    double deltaTime_;                            ///< Time since last frame
    double fps_;                                  ///< Current FPS
    int frameCount_;                              ///< Frame counter for FPS calculation
    double fpsUpdateTime_;                        ///< Time of last FPS update
    
    std::function<void(double)> renderCallback_;  ///< Custom render callback
    std::function<void(double)> updateCallback_;  ///< Custom update callback
    std::function<void(int, int, int, double, double)> mouseClickCallback_;  ///< Mouse click callback

    ActivityMonitor* activityMonitor_;            ///< Activity monitor
    NetworkInspector* networkInspector_;          ///< Network inspector
    Datastore* datastore_;                        ///< Datastore
    RecordingManager* recordingManager_;          ///< Recording manager for playback

    bool initialized_;                            ///< Initialization flag
    bool playbackMode_;                           ///< Playback mode enabled
    bool visualizationEnabled_;                   ///< Visualization enabled flag
};

} // namespace snnfw

#endif // SNNFW_VISUALIZATION_MANAGER_H


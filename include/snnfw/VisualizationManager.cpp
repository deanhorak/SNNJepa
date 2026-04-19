#include "snnfw/VisualizationManager.h"
#include "snnfw/SimulationConfig.h"
#include "snnfw/RecordingManager.h"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <thread>

namespace snnfw {

VisualizationManager::VisualizationManager(int width, int height, const std::string& title)
    : window_(nullptr)
    , width_(width)
    , height_(height)
    , title_(title)
    , vsync_(true)
    , targetFPS_(60)
    , backgroundColor_(0.1f, 0.1f, 0.15f)
    , lastFrameTime_(0.0)
    , deltaTime_(0.0)
    , fps_(0.0)
    , frameCount_(0)
    , fpsUpdateTime_(0.0)
    , renderCallback_(nullptr)
    , updateCallback_(nullptr)
    , activityMonitor_(nullptr)
    , networkInspector_(nullptr)
    , datastore_(nullptr)
    , recordingManager_(nullptr)
    , initialized_(false)
    , playbackMode_(false)
    , visualizationEnabled_(true)  // Default constructor always enables visualization
{
    initGLFW();
    initGLAD();
    initImGui();
    initialized_ = true;
}

VisualizationManager::VisualizationManager(const SimulationConfig& config)
    : window_(nullptr)
    , width_(config.visualizationWidth)
    , height_(config.visualizationHeight)
    , title_(config.visualizationTitle)
    , vsync_(true)
    , targetFPS_(60)
    , backgroundColor_(0.1f, 0.1f, 0.15f)
    , lastFrameTime_(0.0)
    , deltaTime_(0.0)
    , fps_(0.0)
    , frameCount_(0)
    , fpsUpdateTime_(0.0)
    , renderCallback_(nullptr)
    , updateCallback_(nullptr)
    , activityMonitor_(nullptr)
    , networkInspector_(nullptr)
    , datastore_(nullptr)
    , recordingManager_(nullptr)
    , initialized_(false)
    , playbackMode_(config.playbackMode)
    , visualizationEnabled_(config.enableVisualization || config.playbackMode)
{
    // Only initialize OpenGL/GLFW if visualization is enabled
    if (visualizationEnabled_) {
        initGLFW();
        initGLAD();
        initImGui();
        initialized_ = true;
    }
}

VisualizationManager::~VisualizationManager() {
    cleanup();
}

void VisualizationManager::run() {
    if (!visualizationEnabled_) {
        // When visualization is disabled, still run update callback if set
        // This allows training/testing logic to run without visualization
        if (updateCallback_) {
            std::cout << "Running in headless mode (no visualization)..." << std::endl;
            // Simple loop - just call update callback repeatedly
            // The callback should handle its own termination logic
            double lastTime = 0.0;
            while (true) {
                double currentTime = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count()) / 1000.0;
                double deltaTime = currentTime - lastTime;
                lastTime = currentTime;

                updateCallback_(deltaTime);

                // Small sleep to prevent busy-waiting
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        return;
    }

    if (!initialized_) {
        std::cerr << "ERROR: VisualizationManager not initialized" << std::endl;
        return;
    }

    lastFrameTime_ = glfwGetTime();
    fpsUpdateTime_ = lastFrameTime_;

    while (!shouldClose()) {
        // Calculate delta time
        double currentTime = glfwGetTime();
        deltaTime_ = currentTime - lastFrameTime_;
        lastFrameTime_ = currentTime;

        // Update FPS counter
        frameCount_++;
        if (currentTime - fpsUpdateTime_ >= 1.0) {
            fps_ = frameCount_ / (currentTime - fpsUpdateTime_);
            frameCount_ = 0;
            fpsUpdateTime_ = currentTime;
        }

        // Process input
        processInput();

        // Update logic
        update(deltaTime_);

        // Render
        render();

        // Swap buffers and poll events
        glfwSwapBuffers(window_);
        glfwPollEvents();

        // Frame rate limiting (if not using VSync)
        if (!vsync_ && targetFPS_ > 0) {
            double targetFrameTime = 1.0 / targetFPS_;
            double frameTime = glfwGetTime() - currentTime;
            if (frameTime < targetFrameTime) {
                double sleepTime = targetFrameTime - frameTime;
                // Simple busy wait (could use sleep for better CPU usage)
                while (glfwGetTime() - currentTime < targetFrameTime) {
                    // Busy wait
                }
            }
        }
    }
}

void VisualizationManager::close() {
    if (!visualizationEnabled_ || !window_) return;
    glfwSetWindowShouldClose(window_, GLFW_TRUE);
}

bool VisualizationManager::shouldClose() const {
    if (!visualizationEnabled_ || !window_) return true;  // Always "closed" if no visualization
    return glfwWindowShouldClose(window_);
}

void VisualizationManager::setTargetFPS(int fps) {
    targetFPS_ = fps;
}

void VisualizationManager::enableVSync(bool enable) {
    vsync_ = enable;
    if (visualizationEnabled_ && window_) {
        glfwSwapInterval(enable ? 1 : 0);
    }
}

void VisualizationManager::setBackgroundColor(float r, float g, float b) {
    backgroundColor_ = glm::vec3(r, g, b);
}

void VisualizationManager::setRenderCallback(std::function<void(double)> callback) {
    renderCallback_ = callback;
}

void VisualizationManager::setUpdateCallback(std::function<void(double)> callback) {
    updateCallback_ = callback;
}

void VisualizationManager::setMouseClickCallback(std::function<void(int, int, int, double, double)> callback) {
    mouseClickCallback_ = callback;
}

void VisualizationManager::setActivityMonitor(ActivityMonitor* monitor) {
    activityMonitor_ = monitor;
}

void VisualizationManager::setNetworkInspector(NetworkInspector* inspector) {
    networkInspector_ = inspector;
}

void VisualizationManager::setDatastore(Datastore* datastore) {
    datastore_ = datastore;
}

void VisualizationManager::setRecordingManager(RecordingManager* recordingManager) {
    recordingManager_ = recordingManager;
}

void VisualizationManager::enablePlaybackMode(bool enable) {
    playbackMode_ = enable;
}

void VisualizationManager::initGLFW() {
    // Set error callback
    glfwSetErrorCallback(glfwErrorCallback);
    
    // Initialize GLFW
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }
    
    // Set OpenGL version (4.5 Core)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    
    // Create window
    window_ = glfwCreateWindow(width_, height_, title_.c_str(), nullptr, nullptr);
    if (!window_) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
    
    // Make context current
    glfwMakeContextCurrent(window_);
    
    // Set callbacks
    glfwSetWindowUserPointer(window_, this);
    glfwSetFramebufferSizeCallback(window_, framebufferSizeCallback);
    glfwSetKeyCallback(window_, keyCallback);
    glfwSetMouseButtonCallback(window_, mouseButtonCallback);
    glfwSetCursorPosCallback(window_, cursorPosCallback);
    glfwSetScrollCallback(window_, scrollCallback);
    
    // Enable VSync by default
    glfwSwapInterval(vsync_ ? 1 : 0);
}

void VisualizationManager::initGLAD() {
    // Load OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        glfwTerminate();
        throw std::runtime_error("Failed to initialize GLAD");
    }
    
    // Print OpenGL version
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
    
    // Enable depth testing
    glEnable(GL_DEPTH_TEST);
    
    // Enable blending for transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void VisualizationManager::initImGui() {
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    
    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init("#version 450");
}

void VisualizationManager::processInput() {
    // ESC to close
    if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        close();
    }
}

void VisualizationManager::render() {
    // Clear buffers
    glClearColor(backgroundColor_.r, backgroundColor_.g, backgroundColor_.b, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Start ImGui frame BEFORE calling render callback
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Call custom render callback (can now use ImGui)
    if (renderCallback_) {
        renderCallback_(deltaTime_);
    }
    
    // Default ImGui window with stats
    ImGui::Begin("SNNFW Visualization");
    ImGui::Text("FPS: %.1f", fps_);
    ImGui::Text("Frame Time: %.3f ms", deltaTime_ * 1000.0);
    ImGui::Separator();
    
    float bgColor[3] = { backgroundColor_.r, backgroundColor_.g, backgroundColor_.b };
    if (ImGui::ColorEdit3("Background", bgColor)) {
        setBackgroundColor(bgColor[0], bgColor[1], bgColor[2]);
    }
    
    ImGui::End();
    
    // Render ImGui
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void VisualizationManager::update(double deltaTime) {
    // Call custom update callback
    if (updateCallback_) {
        updateCallback_(deltaTime);
    }
}

void VisualizationManager::cleanup() {
    if (initialized_) {
        // Cleanup ImGui
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        
        // Cleanup GLFW
        if (window_) {
            glfwDestroyWindow(window_);
        }
        glfwTerminate();
        
        initialized_ = false;
    }
}

void VisualizationManager::glfwErrorCallback(int error, const char* description) {
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

void VisualizationManager::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    
    VisualizationManager* manager = static_cast<VisualizationManager*>(glfwGetWindowUserPointer(window));
    if (manager) {
        manager->width_ = width;
        manager->height_ = height;
    }
}

void VisualizationManager::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    // ImGui handles this
}

void VisualizationManager::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    // Get the VisualizationManager instance
    VisualizationManager* manager = static_cast<VisualizationManager*>(glfwGetWindowUserPointer(window));
    if (manager && manager->mouseClickCallback_) {
        // Get cursor position
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);

        // Call custom callback
        manager->mouseClickCallback_(button, action, mods, xpos, ypos);
    }
}

void VisualizationManager::cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    // Get the VisualizationManager instance
    VisualizationManager* manager = static_cast<VisualizationManager*>(glfwGetWindowUserPointer(window));
    if (manager) {
        // Store mouse position for picking
        // This could be used for hover effects
    }
}

void VisualizationManager::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    // ImGui handles this
}

} // namespace snnfw


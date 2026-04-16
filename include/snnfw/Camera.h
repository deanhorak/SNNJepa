#ifndef SNNFW_CAMERA_H
#define SNNFW_CAMERA_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace snnfw {

/**
 * @brief Camera class for 3D visualization
 * 
 * Provides view and projection matrix generation, and camera transformations
 * including orbit, pan, and zoom operations.
 */
class Camera {
public:
    /**
     * @brief Default constructor
     * 
     * Creates a camera at (0, 0, 5) looking at origin
     */
    Camera();
    
    /**
     * @brief Constructor with position and target
     * 
     * @param position Camera position in world space
     * @param target Point the camera is looking at
     * @param up Up vector (default: Y-axis)
     */
    Camera(const glm::vec3& position, const glm::vec3& target, const glm::vec3& up = glm::vec3(0, 1, 0));
    
    /**
     * @brief Get view matrix
     * 
     * @return View matrix for rendering
     */
    glm::mat4 getViewMatrix() const;
    
    /**
     * @brief Get projection matrix
     * 
     * @param aspect Aspect ratio (width / height)
     * @return Projection matrix for rendering
     */
    glm::mat4 getProjectionMatrix(float aspect) const;
    
    /**
     * @brief Orbit camera around target
     * 
     * @param deltaYaw Horizontal rotation in radians
     * @param deltaPitch Vertical rotation in radians
     */
    void orbit(float deltaYaw, float deltaPitch);
    
    /**
     * @brief Pan camera (move target and position together)
     * 
     * @param deltaX Horizontal pan amount
     * @param deltaY Vertical pan amount
     */
    void pan(float deltaX, float deltaY);
    
    /**
     * @brief Zoom camera (move closer/farther from target)
     * 
     * @param delta Zoom amount (positive = zoom in, negative = zoom out)
     */
    void zoom(float delta);
    
    /**
     * @brief Point camera at target
     * 
     * @param target New target position
     */
    void lookAt(const glm::vec3& target);
    
    /**
     * @brief Set camera position
     * 
     * @param position New camera position
     */
    void setPosition(const glm::vec3& position);
    
    /**
     * @brief Set target position
     * 
     * @param target New target position
     */
    void setTarget(const glm::vec3& target);
    
    /**
     * @brief Set field of view
     * 
     * @param fov Field of view in degrees
     */
    void setFOV(float fov);
    
    /**
     * @brief Set near and far clipping planes
     * 
     * @param near Near clipping plane distance
     * @param far Far clipping plane distance
     */
    void setNearFar(float near, float far);
    
    /**
     * @brief Get camera position
     * 
     * @return Camera position in world space
     */
    glm::vec3 getPosition() const { return position_; }
    
    /**
     * @brief Get target position
     * 
     * @return Target position in world space
     */
    glm::vec3 getTarget() const { return target_; }
    
    /**
     * @brief Get forward vector
     * 
     * @return Normalized forward vector
     */
    glm::vec3 getForward() const;
    
    /**
     * @brief Get right vector
     * 
     * @return Normalized right vector
     */
    glm::vec3 getRight() const;
    
    /**
     * @brief Get up vector
     * 
     * @return Normalized up vector
     */
    glm::vec3 getUp() const { return up_; }
    
    /**
     * @brief Get field of view
     * 
     * @return Field of view in degrees
     */
    float getFOV() const { return fov_; }
    
    /**
     * @brief Get distance from target
     * 
     * @return Distance to target
     */
    float getDistance() const { return distance_; }

private:
    /**
     * @brief Update camera position based on spherical coordinates
     */
    void updatePosition();
    
    /**
     * @brief Update spherical coordinates from position
     */
    void updateSpherical();
    
    glm::vec3 position_;      ///< Camera position in world space
    glm::vec3 target_;        ///< Point camera is looking at
    glm::vec3 up_;            ///< Up vector
    
    float fov_;               ///< Field of view in degrees
    float nearPlane_;         ///< Near clipping plane
    float farPlane_;          ///< Far clipping plane
    
    float distance_;          ///< Distance from target
    float yaw_;               ///< Horizontal rotation (radians)
    float pitch_;             ///< Vertical rotation (radians)
};

} // namespace snnfw

#endif // SNNFW_CAMERA_H


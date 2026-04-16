#include "snnfw/Camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>

namespace snnfw {

Camera::Camera()
    : position_(0.0f, 0.0f, 5.0f)
    , target_(0.0f, 0.0f, 0.0f)
    , up_(0.0f, 1.0f, 0.0f)
    , fov_(45.0f)
    , nearPlane_(0.1f)
    , farPlane_(1000.0f)
    , distance_(5.0f)
    , yaw_(0.0f)
    , pitch_(0.0f)
{
    updateSpherical();
}

Camera::Camera(const glm::vec3& position, const glm::vec3& target, const glm::vec3& up)
    : position_(position)
    , target_(target)
    , up_(up)
    , fov_(45.0f)
    , nearPlane_(0.1f)
    , farPlane_(1000.0f)
    , distance_(glm::length(position - target))
    , yaw_(0.0f)
    , pitch_(0.0f)
{
    updateSpherical();
}

glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(position_, target_, up_);
}

glm::mat4 Camera::getProjectionMatrix(float aspect) const {
    return glm::perspective(glm::radians(fov_), aspect, nearPlane_, farPlane_);
}

void Camera::orbit(float deltaYaw, float deltaPitch) {
    yaw_ += deltaYaw;
    pitch_ += deltaPitch;
    
    // Clamp pitch to avoid gimbal lock
    const float maxPitch = glm::radians(89.0f);
    if (pitch_ > maxPitch) pitch_ = maxPitch;
    if (pitch_ < -maxPitch) pitch_ = -maxPitch;
    
    updatePosition();
}

void Camera::pan(float deltaX, float deltaY) {
    glm::vec3 right = getRight();
    glm::vec3 up = getUp();
    
    glm::vec3 offset = right * deltaX + up * deltaY;
    
    position_ += offset;
    target_ += offset;
}

void Camera::zoom(float delta) {
    distance_ -= delta;
    
    // Clamp distance to reasonable values
    if (distance_ < 0.1f) distance_ = 0.1f;
    if (distance_ > 1000.0f) distance_ = 1000.0f;
    
    updatePosition();
}

void Camera::lookAt(const glm::vec3& target) {
    target_ = target;
    updateSpherical();
}

void Camera::setPosition(const glm::vec3& position) {
    position_ = position;
    updateSpherical();
}

void Camera::setTarget(const glm::vec3& target) {
    target_ = target;
    updateSpherical();
}

void Camera::setFOV(float fov) {
    fov_ = fov;
    
    // Clamp FOV to reasonable values
    if (fov_ < 1.0f) fov_ = 1.0f;
    if (fov_ > 120.0f) fov_ = 120.0f;
}

void Camera::setNearFar(float near, float far) {
    nearPlane_ = near;
    farPlane_ = far;
}

glm::vec3 Camera::getForward() const {
    return glm::normalize(target_ - position_);
}

glm::vec3 Camera::getRight() const {
    return glm::normalize(glm::cross(getForward(), up_));
}

void Camera::updatePosition() {
    // Convert spherical coordinates to Cartesian
    float x = distance_ * cos(pitch_) * sin(yaw_);
    float y = distance_ * sin(pitch_);
    float z = distance_ * cos(pitch_) * cos(yaw_);
    
    position_ = target_ + glm::vec3(x, y, z);
}

void Camera::updateSpherical() {
    glm::vec3 offset = position_ - target_;
    distance_ = glm::length(offset);

    if (distance_ > 0.0001f) {
        offset = glm::normalize(offset);

        // Calculate yaw (horizontal angle)
        yaw_ = atan2(offset.x, offset.z);

        // Calculate pitch (vertical angle)
        // Clamp offset.y to [-1, 1] to avoid NaN from asin
        float clampedY = glm::clamp(offset.y, -1.0f, 1.0f);
        pitch_ = asin(clampedY);
    } else {
        // Camera is at target, use default orientation
        yaw_ = 0.0f;
        pitch_ = 0.0f;
        distance_ = 0.1f;
    }
}

} // namespace snnfw


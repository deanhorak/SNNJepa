#ifndef SNNFW_GEOMETRY_RENDERER_H
#define SNNFW_GEOMETRY_RENDERER_H

#include <glad/glad.h>
#include <glm/glm.hpp>

namespace snnfw {

/**
 * @brief Basic geometry rendering class
 * 
 * Provides methods to render basic 3D shapes (cube, sphere, line).
 */
class GeometryRenderer {
public:
    /**
     * @brief Constructor
     */
    GeometryRenderer();
    
    /**
     * @brief Destructor - cleans up OpenGL resources
     */
    ~GeometryRenderer();
    
    /**
     * @brief Render a cube
     * 
     * Note: Shader must be active and uniforms set before calling
     */
    void renderCube();
    
    /**
     * @brief Render a sphere
     * 
     * Note: Shader must be active and uniforms set before calling
     */
    void renderSphere();
    
    /**
     * @brief Cleanup OpenGL resources
     */
    void cleanup();

private:
    /**
     * @brief Initialize cube geometry
     */
    void initCube();
    
    /**
     * @brief Initialize sphere geometry
     */
    void initSphere();
    
    GLuint cubeVAO_;          ///< Cube vertex array object
    GLuint cubeVBO_;          ///< Cube vertex buffer object
    
    GLuint sphereVAO_;        ///< Sphere vertex array object
    GLuint sphereVBO_;        ///< Sphere vertex buffer object
    GLuint sphereEBO_;        ///< Sphere element buffer object
    int sphereIndexCount_;    ///< Number of sphere indices
    
    bool initialized_;        ///< Initialization flag
};

} // namespace snnfw

#endif // SNNFW_GEOMETRY_RENDERER_H


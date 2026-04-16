#ifndef SNNFW_SHADER_MANAGER_H
#define SNNFW_SHADER_MANAGER_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <string>
#include <map>

namespace snnfw {

/**
 * @brief Shader management class
 * 
 * Handles shader compilation, linking, and uniform setting.
 * Provides caching and error reporting.
 */
class ShaderManager {
public:
    /**
     * @brief Constructor
     */
    ShaderManager();
    
    /**
     * @brief Destructor - cleans up all shaders
     */
    ~ShaderManager();
    
    /**
     * @brief Load and compile shader program
     * 
     * @param name Shader program name for later reference
     * @param vertexPath Path to vertex shader file
     * @param fragmentPath Path to fragment shader file
     * @return Shader program ID, or 0 on failure
     */
    GLuint loadShader(const std::string& name, 
                      const std::string& vertexPath, 
                      const std::string& fragmentPath);
    
    /**
     * @brief Load shader from source strings
     * 
     * @param name Shader program name
     * @param vertexSource Vertex shader source code
     * @param fragmentSource Fragment shader source code
     * @return Shader program ID, or 0 on failure
     */
    GLuint loadShaderFromSource(const std::string& name,
                                const std::string& vertexSource,
                                const std::string& fragmentSource);
    
    /**
     * @brief Get shader program ID by name
     * 
     * @param name Shader program name
     * @return Shader program ID, or 0 if not found
     */
    GLuint getShader(const std::string& name) const;
    
    /**
     * @brief Use shader program
     * 
     * @param name Shader program name
     */
    void useShader(const std::string& name);
    
    /**
     * @brief Delete shader program
     * 
     * @param name Shader program name
     */
    void deleteShader(const std::string& name);
    
    /**
     * @brief Delete all shader programs
     */
    void deleteAll();
    
    // Uniform setters
    void setUniform(const std::string& name, int value);
    void setUniform(const std::string& name, float value);
    void setUniform(const std::string& name, const glm::vec2& value);
    void setUniform(const std::string& name, const glm::vec3& value);
    void setUniform(const std::string& name, const glm::vec4& value);
    void setUniform(const std::string& name, const glm::mat3& value);
    void setUniform(const std::string& name, const glm::mat4& value);
    
    /**
     * @brief Get current shader program ID
     * 
     * @return Current shader program ID
     */
    GLuint getCurrentShader() const { return currentShader_; }

private:
    /**
     * @brief Compile shader from source
     * 
     * @param source Shader source code
     * @param type Shader type (GL_VERTEX_SHADER, GL_FRAGMENT_SHADER)
     * @return Shader ID, or 0 on failure
     */
    GLuint compileShader(const std::string& source, GLenum type);
    
    /**
     * @brief Link shader program
     * 
     * @param vertexShader Vertex shader ID
     * @param fragmentShader Fragment shader ID
     * @return Program ID, or 0 on failure
     */
    GLuint linkProgram(GLuint vertexShader, GLuint fragmentShader);
    
    /**
     * @brief Read file contents
     * 
     * @param path File path
     * @return File contents as string
     */
    std::string readFile(const std::string& path);
    
    /**
     * @brief Check shader compilation errors
     * 
     * @param shader Shader ID
     * @param type Shader type name for error messages
     */
    void checkCompileErrors(GLuint shader, const std::string& type);
    
    /**
     * @brief Check program linking errors
     * 
     * @param program Program ID
     */
    void checkLinkErrors(GLuint program);
    
    /**
     * @brief Get uniform location with caching
     * 
     * @param name Uniform name
     * @return Uniform location, or -1 if not found
     */
    GLint getUniformLocation(const std::string& name);
    
    std::map<std::string, GLuint> shaders_;           ///< Shader program cache
    std::map<std::string, GLint> uniformLocations_;   ///< Uniform location cache
    GLuint currentShader_;                            ///< Currently active shader
};

} // namespace snnfw

#endif // SNNFW_SHADER_MANAGER_H


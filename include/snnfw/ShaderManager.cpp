#include "snnfw/ShaderManager.h"
#include <glm/gtc/type_ptr.hpp>
#include <fstream>
#include <sstream>
#include <iostream>

namespace snnfw {

ShaderManager::ShaderManager()
    : currentShader_(0)
{
}

ShaderManager::~ShaderManager() {
    deleteAll();
}

GLuint ShaderManager::loadShader(const std::string& name, 
                                  const std::string& vertexPath, 
                                  const std::string& fragmentPath) {
    std::string vertexSource = readFile(vertexPath);
    std::string fragmentSource = readFile(fragmentPath);
    
    if (vertexSource.empty() || fragmentSource.empty()) {
        std::cerr << "ERROR::SHADER::FILE_NOT_FOUND" << std::endl;
        return 0;
    }
    
    return loadShaderFromSource(name, vertexSource, fragmentSource);
}

GLuint ShaderManager::loadShaderFromSource(const std::string& name,
                                            const std::string& vertexSource,
                                            const std::string& fragmentSource) {
    // Compile shaders
    GLuint vertexShader = compileShader(vertexSource, GL_VERTEX_SHADER);
    if (vertexShader == 0) return 0;
    
    GLuint fragmentShader = compileShader(fragmentSource, GL_FRAGMENT_SHADER);
    if (fragmentShader == 0) {
        glDeleteShader(vertexShader);
        return 0;
    }
    
    // Link program
    GLuint program = linkProgram(vertexShader, fragmentShader);
    
    // Delete shaders (they're linked into program now)
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    if (program == 0) return 0;
    
    // Store in cache
    shaders_[name] = program;
    
    return program;
}

GLuint ShaderManager::getShader(const std::string& name) const {
    auto it = shaders_.find(name);
    if (it != shaders_.end()) {
        return it->second;
    }
    return 0;
}

void ShaderManager::useShader(const std::string& name) {
    GLuint shader = getShader(name);
    if (shader != 0) {
        glUseProgram(shader);
        currentShader_ = shader;
        uniformLocations_.clear();  // Clear cache when switching shaders
    }
}

void ShaderManager::deleteShader(const std::string& name) {
    auto it = shaders_.find(name);
    if (it != shaders_.end()) {
        glDeleteProgram(it->second);
        shaders_.erase(it);
    }
}

void ShaderManager::deleteAll() {
    for (auto& pair : shaders_) {
        glDeleteProgram(pair.second);
    }
    shaders_.clear();
    uniformLocations_.clear();
    currentShader_ = 0;
}

void ShaderManager::setUniform(const std::string& name, int value) {
    GLint location = getUniformLocation(name);
    if (location != -1) {
        glUniform1i(location, value);
    }
}

void ShaderManager::setUniform(const std::string& name, float value) {
    GLint location = getUniformLocation(name);
    if (location != -1) {
        glUniform1f(location, value);
    }
}

void ShaderManager::setUniform(const std::string& name, const glm::vec2& value) {
    GLint location = getUniformLocation(name);
    if (location != -1) {
        glUniform2fv(location, 1, glm::value_ptr(value));
    }
}

void ShaderManager::setUniform(const std::string& name, const glm::vec3& value) {
    GLint location = getUniformLocation(name);
    if (location != -1) {
        glUniform3fv(location, 1, glm::value_ptr(value));
    }
}

void ShaderManager::setUniform(const std::string& name, const glm::vec4& value) {
    GLint location = getUniformLocation(name);
    if (location != -1) {
        glUniform4fv(location, 1, glm::value_ptr(value));
    }
}

void ShaderManager::setUniform(const std::string& name, const glm::mat3& value) {
    GLint location = getUniformLocation(name);
    if (location != -1) {
        glUniformMatrix3fv(location, 1, GL_FALSE, glm::value_ptr(value));
    }
}

void ShaderManager::setUniform(const std::string& name, const glm::mat4& value) {
    GLint location = getUniformLocation(name);
    if (location != -1) {
        glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(value));
    }
}

GLuint ShaderManager::compileShader(const std::string& source, GLenum type) {
    GLuint shader = glCreateShader(type);
    const char* sourceCStr = source.c_str();
    glShaderSource(shader, 1, &sourceCStr, nullptr);
    glCompileShader(shader);
    
    std::string typeName = (type == GL_VERTEX_SHADER) ? "VERTEX" : "FRAGMENT";
    checkCompileErrors(shader, typeName);
    
    // Check if compilation succeeded
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glDeleteShader(shader);
        return 0;
    }
    
    return shader;
}

GLuint ShaderManager::linkProgram(GLuint vertexShader, GLuint fragmentShader) {
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    
    checkLinkErrors(program);
    
    // Check if linking succeeded
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        glDeleteProgram(program);
        return 0;
    }
    
    return program;
}

std::string ShaderManager::readFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "ERROR::SHADER::FILE_NOT_FOUND: " << path << std::endl;
        return "";
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

void ShaderManager::checkCompileErrors(GLuint shader, const std::string& type) {
    GLint success;
    GLchar infoLog[1024];
    
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 1024, nullptr, infoLog);
        std::cerr << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" 
                  << infoLog << "\n -- --------------------------------------------------- -- " 
                  << std::endl;
    }
}

void ShaderManager::checkLinkErrors(GLuint program) {
    GLint success;
    GLchar infoLog[1024];
    
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(program, 1024, nullptr, infoLog);
        std::cerr << "ERROR::PROGRAM_LINKING_ERROR\n" 
                  << infoLog << "\n -- --------------------------------------------------- -- " 
                  << std::endl;
    }
}

GLint ShaderManager::getUniformLocation(const std::string& name) {
    // Check cache first
    auto it = uniformLocations_.find(name);
    if (it != uniformLocations_.end()) {
        return it->second;
    }
    
    // Query OpenGL
    GLint location = glGetUniformLocation(currentShader_, name.c_str());
    
    // Cache the result (even if -1, to avoid repeated queries)
    uniformLocations_[name] = location;
    
    if (location == -1) {
        std::cerr << "WARNING::SHADER::UNIFORM_NOT_FOUND: " << name << std::endl;
    }
    
    return location;
}

} // namespace snnfw


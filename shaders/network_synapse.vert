#version 450 core

// Vertex attributes
layout (location = 0) in vec3 aPos;      // Line vertex position
layout (location = 1) in vec4 aColor;    // Line vertex color

// Uniforms
uniform mat4 view;
uniform mat4 projection;

// Outputs to fragment shader
out vec4 Color;

void main() {
    Color = aColor;
    gl_Position = projection * view * vec4(aPos, 1.0);
}


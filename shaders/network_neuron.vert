#version 450 core

// Vertex attributes (sphere geometry)
layout (location = 0) in vec3 aPos;      // Sphere vertex position
layout (location = 1) in vec3 aNormal;   // Sphere vertex normal

// Instance attributes (per-neuron data)
layout (location = 2) in vec3 aInstancePos;     // Neuron position
layout (location = 3) in vec4 aInstanceColor;   // Neuron color (RGBA)
layout (location = 4) in float aInstanceRadius; // Neuron radius
layout (location = 5) in float aInstanceActivity; // Activity level (0-1)

// Uniforms
uniform mat4 view;
uniform mat4 projection;

// Outputs to fragment shader
out vec3 FragPos;
out vec3 Normal;
out vec4 Color;
out float Activity;

void main() {
    // Scale sphere vertex by neuron radius
    vec3 scaledPos = aPos * aInstanceRadius;
    
    // Translate to neuron position
    vec3 worldPos = aInstancePos + scaledPos;
    
    // Pass to fragment shader
    FragPos = worldPos;
    Normal = aNormal;  // Normal doesn't need scaling (unit sphere)
    Color = aInstanceColor;
    Activity = aInstanceActivity;
    
    // Transform to clip space
    gl_Position = projection * view * vec4(worldPos, 1.0);
}


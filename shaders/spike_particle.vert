#version 450 core

// Per-vertex attributes (billboard quad)
layout(location = 0) in vec3 aPos;

// Per-instance attributes
layout(location = 1) in vec3 instancePos;
layout(location = 2) in vec4 instanceColor;
layout(location = 3) in float instanceSize;
layout(location = 4) in vec2 instanceLifetime; // x = lifetime, y = maxLifetime

// Uniforms
uniform mat4 view;
uniform mat4 projection;

// Outputs to fragment shader
out vec4 particleColor;
out vec2 texCoord;
out float lifetimeRatio;

void main() {
    // Calculate lifetime ratio (0 = just created, 1 = about to die)
    lifetimeRatio = instanceLifetime.x / instanceLifetime.y;
    
    // Pass color to fragment shader
    particleColor = instanceColor;
    
    // Calculate texture coordinates from vertex position
    texCoord = aPos.xy + 0.5; // Map from [-0.5, 0.5] to [0, 1]
    
    // Billboard: make quad face camera
    vec3 cameraRight = vec3(view[0][0], view[1][0], view[2][0]);
    vec3 cameraUp = vec3(view[0][1], view[1][1], view[2][1]);
    
    // Scale vertex by instance size
    vec3 vertexPos = instancePos 
                   + cameraRight * aPos.x * instanceSize
                   + cameraUp * aPos.y * instanceSize;
    
    // Transform to clip space
    gl_Position = projection * view * vec4(vertexPos, 1.0);
}


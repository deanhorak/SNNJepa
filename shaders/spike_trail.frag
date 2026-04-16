#version 450 core

in vec3 worldPos;

uniform float trailLength;

out vec4 FragColor;

void main() {
    // Simple trail color (can be enhanced with textures/gradients)
    vec3 trailColor = vec3(1.0, 0.8, 0.3); // Yellow-orange
    float alpha = 0.5;
    
    FragColor = vec4(trailColor, alpha);
}


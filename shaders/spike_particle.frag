#version 450 core

// Inputs from vertex shader
in vec4 particleColor;
in vec2 texCoord;
in float lifetimeRatio;

// Uniforms
uniform float glowIntensity;
uniform float particleAlpha;

// Output
out vec4 FragColor;

void main() {
    // Calculate distance from center of particle
    vec2 center = vec2(0.5, 0.5);
    float dist = distance(texCoord, center);
    
    // Create circular particle with soft edges
    float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
    
    // Fade out near end of lifetime
    float lifetimeFade = 1.0 - lifetimeRatio;
    alpha *= lifetimeFade;
    
    // Apply base alpha
    alpha *= particleAlpha;
    
    // Glow effect: brighter in center
    float glow = 1.0 - dist * 2.0;
    glow = max(0.0, glow);
    glow = pow(glow, 2.0) * glowIntensity;
    
    // Combine color with glow
    vec3 finalColor = particleColor.rgb * (1.0 + glow);
    
    // Output with alpha
    FragColor = vec4(finalColor, alpha);
    
    // Discard fully transparent fragments
    if (FragColor.a < 0.01) {
        discard;
    }
}


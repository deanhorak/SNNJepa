#version 450 core

// Inputs from vertex shader
in vec3 FragPos;
in vec3 Normal;
in vec4 Color;
in float Activity;

// Uniforms
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;
uniform float ambientStrength;
uniform float specularStrength;
uniform bool enableLighting;

// Selection uniforms
uniform bool isSelected;
uniform vec4 selectionColor;
uniform float selectionGlowIntensity;

// Output
out vec4 FragColor;

void main() {
    vec3 result;
    
    if (enableLighting) {
        // Phong lighting model
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        vec3 viewDir = normalize(viewPos - FragPos);
        
        // Ambient
        vec3 ambient = ambientStrength * Color.rgb;
        
        // Diffuse
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * Color.rgb;
        
        // Specular
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        vec3 specular = specularStrength * spec * lightColor;
        
        result = ambient + diffuse + specular;
    } else {
        // No lighting, just use color
        result = Color.rgb;
    }
    
    // Add activity glow
    if (Activity > 0.1) {
        vec3 activityGlow = vec3(1.0, 1.0, 0.3) * Activity * 0.5;
        result += activityGlow;
    }
    
    // Selection highlighting
    if (isSelected) {
        result = mix(result, selectionColor.rgb, selectionGlowIntensity * 0.5);
        
        // Add rim lighting for selection
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 norm = normalize(Normal);
        float rim = 1.0 - max(dot(viewDir, norm), 0.0);
        rim = pow(rim, 3.0);
        result += selectionColor.rgb * rim * selectionGlowIntensity;
    }
    
    FragColor = vec4(result, Color.a);
}


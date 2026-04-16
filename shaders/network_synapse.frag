#version 450 core

// Inputs from vertex shader
in vec4 Color;

// Output
out vec4 FragColor;

void main() {
    FragColor = Color;
}


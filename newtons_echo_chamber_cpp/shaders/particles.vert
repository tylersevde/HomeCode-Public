#version 450

layout(location = 0) in vec4 inPositionLife;
layout(location = 1) in vec4 inVelocitySeed;

layout(set = 0, binding = 0, std140) uniform FrameBlock {
    mat4 viewProjection;
    vec4 cameraFogStart;
    vec4 fogEndAmbientCountTime;
    vec4 lightPosition[4];
    vec4 lightColor[4];
    vec4 flashlightPositionEnabled;
    vec4 flashlightDirectionRange;
    mat4 flashlightViewProjection;
    vec4 flashlightShadowParams;
} frame;

layout(location = 0) out float speed;
layout(location = 1) flat out uint family;
layout(location = 2) flat out uint impactDriven;

void main() {
    gl_Position = frame.viewProjection * vec4(inPositionLife.xyz, 1.0);
    gl_PointSize = 1.0;
    speed = length(inVelocitySeed.xyz);
    uint packed = floatBitsToUint(inVelocitySeed.w);
    impactDriven = packed >> 31u;
    family = (packed >> 24u) & 0x7fu;
}

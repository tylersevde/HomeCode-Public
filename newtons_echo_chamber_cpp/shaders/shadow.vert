#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 2) in vec4 inPositionScaleX;
layout(location = 3) in vec4 inQuaternion;
layout(location = 4) in vec4 inScaleYZMaterial;

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

vec3 quaternionRotate(vec4 quaternion, vec3 value) {
    return value + 2.0 * cross(quaternion.xyz,
        cross(quaternion.xyz, value) + quaternion.w * value);
}

void main() {
    vec3 scale = vec3(inPositionScaleX.w, inScaleYZMaterial.xy);
    vec3 worldPosition = inPositionScaleX.xyz
        + quaternionRotate(inQuaternion, inPosition * scale);
    gl_Position = frame.flashlightViewProjection * vec4(worldPosition, 1.0);
}

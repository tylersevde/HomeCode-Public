#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inPositionScaleX;
layout(location = 3) in vec4 inQuaternion;
layout(location = 4) in vec4 inScaleYZMaterial;
layout(location = 5) in vec4 inColor;
layout(location = 6) in vec4 inTangent;
layout(location = 7) in vec2 inTextureCoordinate;

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
    vec4 materialOptions;
} frame;

layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out vec4 color;
layout(location = 3) flat out float material;
layout(location = 4) flat out float emissive;
layout(location = 5) out vec4 flashlightClipPosition;
layout(location = 6) out vec4 worldTangent;
layout(location = 7) out vec2 textureCoordinate;
layout(location = 8) flat out vec4 materialProperties;

vec3 quaternionRotate(vec4 quaternion, vec3 value) {
    return value + 2.0 * cross(quaternion.xyz,
        cross(quaternion.xyz, value) + quaternion.w * value);
}

float textureRepeatsPerMeter(int layer) {
    if (layer == 1) return 1.15;  // rough concrete
    if (layer == 2) return 0.85;  // painted concrete
    if (layer == 3) return 5.00;  // rubber
    if (layer == 4) return 2.25;  // metal
    if (layer == 5) return 1.35;  // wood
    if (layer == 6) return 2.00;  // latex
    if (layer == 7) return 7.00;  // foam
    if (layer == 8) return 4.00;  // goo
    if (layer == 9) return 3.50;  // ceramic
    if (layer == 10) return 4.50; // fired clay
    if (layer == 11) return 2.25; // bowling urethane
    if (layer == 12) return 6.00; // plush fabric
    // Room-shell CC0 materials are authored as physically repeating surfaces,
    // not one atlas tile stretched across the 50 x 80 metre chamber.
    if (layer == 13) return 0.5556; // pavement: authored 1.8 m square
    if (layer == 14) return 0.3333; // concrete wall: authored 3.0 m span
    if (layer == 15) return 0.50;   // corrugated iron: authored 2.0 m span
    return 1.0;
}

vec4 materialParameters(int layer) {
    // x: specular strength, y: shininess, z: normal-map strength,
    // w: amount of the per-instance tint applied to sampled albedo.
    if (layer == 13) return vec4(0.11, 10.0, 0.64, 0.22); // pavement floor
    if (layer == 14) return vec4(0.08, 8.0, 0.48, 0.24);  // painted wall
    if (layer == 15) return vec4(0.66, 64.0, 0.92, 0.20); // corrugated iron
    if (layer == 1) return vec4(0.12, 12.0, 0.95, 0.32); // concrete
    if (layer == 2) return vec4(0.14, 18.0, 0.62, 0.38); // paint
    if (layer == 3) return vec4(0.30, 30.0, 0.58, 0.68); // rubber
    if (layer == 4) return vec4(0.88, 96.0, 0.30, 0.30); // metal
    if (layer == 5) return vec4(0.18, 20.0, 0.72, 0.18); // wood
    if (layer == 6) return vec4(0.52, 56.0, 0.34, 0.62); // latex
    if (layer == 7) return vec4(0.07, 7.0, 0.82, 0.62);  // foam
    if (layer == 8) return vec4(0.66, 44.0, 0.48, 0.70); // goo
    if (layer == 9) return vec4(0.82, 88.0, 0.30, 0.28); // ceramic
    if (layer == 10) return vec4(0.10, 11.0, 0.88, 0.12); // clay
    if (layer == 11) return vec4(0.82, 92.0, 0.24, 0.48); // bowling
    if (layer == 12) return vec4(0.05, 5.0, 0.92, 0.68);  // plush
    return vec4(0.20, 24.0, 0.30, 0.65);
}

void main() {
    vec3 scale = vec3(inPositionScaleX.w, inScaleYZMaterial.xy);
    vec3 safeScale = max(abs(scale), vec3(0.00001));
    worldPosition = inPositionScaleX.xyz
        + quaternionRotate(inQuaternion, inPosition * scale);
    worldNormal = normalize(quaternionRotate(inQuaternion, inNormal / safeScale));
    vec3 localTangent = normalize(inTangent.xyz);
    vec3 localBitangent = normalize(cross(inNormal, localTangent))
                        * inTangent.w;
    vec3 scaledTangent = localTangent * scale;
    worldTangent = vec4(normalize(
        quaternionRotate(inQuaternion, scaledTangent)), inTangent.w);
    int layer = clamp(int(inScaleYZMaterial.z + 0.5), 0, 15);
    materialProperties = materialParameters(layer);
    vec2 physicalScale = vec2(
        length(scaledTangent), length(localBitangent * scale));
    textureCoordinate = inTextureCoordinate * physicalScale
                      * textureRepeatsPerMeter(layer);
    color = inColor;
    material = inScaleYZMaterial.z;
    emissive = inScaleYZMaterial.w;
    flashlightClipPosition = frame.flashlightViewProjection
                           * vec4(worldPosition, 1.0);
    gl_Position = frame.viewProjection * vec4(worldPosition, 1.0);
}

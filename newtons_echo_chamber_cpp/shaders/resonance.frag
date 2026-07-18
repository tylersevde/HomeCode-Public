#version 450

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

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) flat in vec4 pulseColor;
layout(location = 3) flat in float pulseAge;
layout(location = 4) flat in float pulseStrength;
layout(location = 0) out vec4 outColor;

void main() {
    vec3 viewDirection = normalize(frame.cameraFogStart.xyz - worldPosition);
    float facing = abs(dot(normalize(worldNormal), viewDirection));
    // Keep only the grazing-angle band. The deliberately tiny far-sphere mesh
    // then reads as a crisp expanding contour instead of a translucent polygon
    // washing over most of the screen.
    float fresnel = smoothstep(0.82, 0.995,
                               clamp(1.0 - facing, 0.0, 1.0));
    if (fresnel < 0.015) {
        discard;
    }
    float shimmer = 0.88 + 0.12 * sin(
        dot(worldPosition, vec3(0.31, 0.47, 0.23)) - pulseAge * 24.0);
    float strength = clamp(pulseStrength, 0.25, 1.65);
    float cameraDistance = distance(frame.cameraFogStart.xyz, worldPosition);
    float distanceFade = 1.0 - 0.58 * clamp(
        (cameraDistance - frame.cameraFogStart.w)
        / max(0.01, frame.fogEndAmbientCountTime.x
                         - frame.cameraFogStart.w), 0.0, 1.0);
    float alpha = pulseColor.a * fresnel * fresnel * distanceFade * 0.72;
    vec3 radiance = pulseColor.rgb
                  * (0.52 + fresnel * (1.15 + strength * 0.62))
                  * shimmer;
    outColor = vec4(radiance, alpha);
}

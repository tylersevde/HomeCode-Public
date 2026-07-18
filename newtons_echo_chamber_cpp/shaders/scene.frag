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

layout(set = 0, binding = 1) uniform sampler2DShadow flashlightShadowMap;
layout(set = 0, binding = 2) uniform sampler2DArray materialAlbedoMap;
layout(set = 0, binding = 3) uniform sampler2DArray materialNormalMap;

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec4 color;
layout(location = 3) flat in float material;
layout(location = 4) flat in float emissive;
layout(location = 5) in vec4 flashlightClipPosition;
layout(location = 6) in vec4 worldTangent;
layout(location = 7) in vec2 textureCoordinate;
layout(location = 8) flat in vec4 materialProperties;
layout(location = 0) out vec4 outColor;

float sampleFlashlightShadow(vec3 coordinate) {
    // Lower quality tiers render into a smaller top-left region of the fixed
    // allocation; scaling coordinates preserves a stable projection without
    // reallocating either frame's shadow resource during AUTO transitions.
    coordinate.xy *= frame.flashlightShadowParams.x;
    // Mode 2 is one hardware-filtered comparison fetch. Mode 1 is the format
    // fallback: four nearest comparison taps retain a small soft edge.
    if (frame.flashlightShadowParams.w > 1.5) {
        return texture(flashlightShadowMap, coordinate);
    }
    float halfTexel = 0.5 / float(textureSize(flashlightShadowMap, 0).x);
    float visibility = 0.0;
    visibility += texture(flashlightShadowMap,
        vec3(coordinate.xy + vec2(-halfTexel, -halfTexel), coordinate.z));
    visibility += texture(flashlightShadowMap,
        vec3(coordinate.xy + vec2( halfTexel, -halfTexel), coordinate.z));
    visibility += texture(flashlightShadowMap,
        vec3(coordinate.xy + vec2(-halfTexel,  halfTexel), coordinate.z));
    visibility += texture(flashlightShadowMap,
        vec3(coordinate.xy + vec2( halfTexel,  halfTexel), coordinate.z));
    return visibility * 0.25;
}

void main() {
    int layer = clamp(int(material + 0.5), 0, 15);
    vec4 properties = materialProperties;
    vec3 geometricNormal = normalize(worldNormal);
    vec3 normal = geometricNormal;
    vec3 baseColor = color.rgb;
    vec3 cameraOffset = frame.cameraFogStart.xyz - worldPosition;
    float cameraDistanceSquared = dot(cameraOffset, cameraOffset);
    // Negative per-instance alpha is an internal opaque-mode bit used by the
    // kinetic lens. It never reaches the framebuffer as transparency.
    bool kineticLens = color.a < 0.0;
    if (frame.materialOptions.x > 0.5) {
        vec3 sampledAlbedo = texture(materialAlbedoMap,
            vec3(textureCoordinate, float(layer))).rgb;
        if (kineticLens) {
            // Preserve a little texture structure while making the logarithmic
            // energy color legible across every material family.
            baseColor = mix(sampledAlbedo * 0.16, color.rgb, 0.84);
        } else {
            vec3 tint = mix(vec3(1.0), color.rgb, properties.w);
            baseColor = sampledAlbedo * tint;
        }
        // z is the full-detail range and w the hard cutoff.  The distance
        // branch is intentionally outside the texture expression: distant
        // fragments do not issue a normal-map fetch at all.
        float normalFullDistanceSquared = frame.materialOptions.z
                                        * frame.materialOptions.z;
        float normalCutoffDistanceSquared = frame.materialOptions.w
                                          * frame.materialOptions.w;
        if (frame.materialOptions.y > 0.5
            && cameraDistanceSquared < normalCutoffDistanceSquared) {
            float normalDetail = 1.0 - smoothstep(
                normalFullDistanceSquared, normalCutoffDistanceSquared,
                cameraDistanceSquared);
            vec3 tangent = normalize(worldTangent.xyz
                - geometricNormal
                * dot(geometricNormal, worldTangent.xyz));
            vec3 bitangent = normalize(cross(geometricNormal, tangent))
                           * worldTangent.w;
            vec3 tangentNormal = texture(materialNormalMap,
                vec3(textureCoordinate, float(layer))).xyz * 2.0 - 1.0;
            tangentNormal.xy *= properties.z * normalDetail;
            tangentNormal.z = max(tangentNormal.z, 0.04);
            tangentNormal = normalize(tangentNormal);
            normal = normalize(mat3(tangent, bitangent, geometricNormal)
                             * tangentNormal);
        }
    }
    vec3 viewDirection = normalize(cameraOffset);
    vec3 illumination = baseColor * frame.fogEndAmbientCountTime.y;
    float specularStrength = properties.x;
    float shininess = properties.y;
    int lightCount = int(frame.fogEndAmbientCountTime.z + 0.5);
    for (int index = 0; index < 4; ++index) {
        if (index >= lightCount) {
            break;
        }
        vec3 offset = frame.lightPosition[index].xyz - worldPosition;
        float distanceToLight = max(length(offset), 0.05);
        vec3 lightDirection = offset / distanceToLight;
        float attenuation = 1.0 / (0.42 + 0.035 * distanceToLight
                                         + 0.010 * distanceToLight * distanceToLight);
        float diffuse = max(dot(normal, lightDirection), 0.0);
        vec3 halfwayDirection = normalize(lightDirection + viewDirection);
        float specular = pow(max(dot(normal, halfwayDirection), 0.0), shininess);
        illumination += (baseColor * diffuse + vec3(specular * specularStrength))
                      * frame.lightColor[index].rgb * attenuation;
    }

    // A head-mounted soft spotlight with an optional sampled shadow map. Range
    // and cone rejects happen before specular and shadow work so disabled or
    // off-cone fragments stay cheap on the VideoCore fragment path.
    if (frame.flashlightPositionEnabled.w > 0.5) {
        vec3 fromFlashlight = worldPosition
                            - frame.flashlightPositionEnabled.xyz;
        float distanceSquared = dot(fromFlashlight, fromFlashlight);
        float flashlightRange = frame.flashlightDirectionRange.w;
        if (distanceSquared > 0.0025
            && distanceSquared < flashlightRange * flashlightRange) {
            float flashlightDistance = sqrt(distanceSquared);
            vec3 beamDirection = fromFlashlight / flashlightDistance;
            float coneAlignment = dot(
                beamDirection, frame.flashlightDirectionRange.xyz);
            const float outerConeCosine = 0.8660254; // 30 degrees
            const float innerConeCosine = 0.9563048; // 17 degrees
            if (coneAlignment > outerConeCosine) {
                float cone = smoothstep(outerConeCosine, innerConeCosine,
                                        coneAlignment);
                float rangeFade = 1.0 - smoothstep(
                    flashlightRange * 0.72, flashlightRange,
                    flashlightDistance);
                vec3 lightDirection = -beamDirection;
                float diffuse = max(dot(normal, lightDirection), 0.0);
                vec3 halfwayDirection = normalize(
                    lightDirection + viewDirection);
                float specular = pow(max(dot(normal, halfwayDirection), 0.0),
                                     shininess);
                float attenuation = 2.35 / (0.72
                    + 0.045 * flashlightDistance
                    + 0.0065 * distanceSquared);
                float shadowVisibility = 1.0;
                if (frame.flashlightShadowParams.w > 0.5
                    && flashlightClipPosition.w > 0.0) {
                    vec3 shadowNdc = flashlightClipPosition.xyz
                                   / flashlightClipPosition.w;
                    vec2 shadowUv = shadowNdc.xy * 0.5 + 0.5;
                    if (all(greaterThanEqual(shadowUv, vec2(0.0)))
                        && all(lessThanEqual(shadowUv, vec2(1.0)))
                        && shadowNdc.z >= 0.0 && shadowNdc.z <= 1.0) {
                        float receiverBias = frame.flashlightShadowParams.y
                            + frame.flashlightShadowParams.z * (1.0
                                - max(dot(geometricNormal, lightDirection), 0.0));
                        shadowVisibility = sampleFlashlightShadow(
                            vec3(shadowUv, shadowNdc.z - receiverBias));
                    }
                }
                vec3 flashlightColor = vec3(1.0, 0.94, 0.82);
                illumination += (baseColor * diffuse
                               + vec3(specular * specularStrength))
                              * flashlightColor * attenuation * cone * rangeFade
                              * shadowVisibility;
            }
        }
    }
    illumination += baseColor * emissive;
    float cameraDistance = sqrt(max(cameraDistanceSquared, 0.0));
    float fog = clamp((cameraDistance - frame.cameraFogStart.w)
                    / max(0.01, frame.fogEndAmbientCountTime.x
                                      - frame.cameraFogStart.w), 0.0, 1.0);
    illumination = mix(illumination, vec3(0.045, 0.060, 0.074), fog);
    outColor = vec4(illumination, 1.0);
}

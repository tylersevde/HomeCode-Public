#version 450

layout(location = 0) in float speed;
layout(location = 1) flat in uint family;
layout(location = 2) flat in uint impactDriven;
layout(location = 0) out vec4 outColor;

void main() {
    float hot = clamp(speed * 0.09, 0.0, 1.0);
    vec3 cold = vec3(0.20, 0.56, 0.83);
    float alpha = 0.08 + hot * 0.07;
    if (impactDriven != 0u) {
        if (family == 1u) cold = vec3(1.00, 0.62, 0.18); // metal sparks
        else if (family == 2u) cold = vec3(0.72, 0.62, 0.48); // mineral dust
        else if (family == 3u) cold = vec3(0.76, 0.46, 0.19); // wood chips
        else if (family == 4u) cold = vec3(0.92, 0.25, 0.16); // rubber flecks
        else if (family == 5u) cold = vec3(0.18, 1.00, 0.35); // goo droplets
        else if (family == 6u) cold = vec3(0.93, 0.76, 0.90); // plush fibres
        else if (family == 7u) cold = vec3(1.00, 0.39, 0.77); // latex
        else if (family == 8u) cold = vec3(0.28, 0.93, 0.96); // foam
        alpha = 0.20 + hot * 0.22;
    }
    vec3 color = mix(cold, vec3(1.0, 0.88, 0.54),
                     hot * (impactDriven != 0u ? 0.62 : 0.28));
    outColor = vec4(color, alpha);
}

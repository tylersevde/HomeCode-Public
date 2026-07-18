# Texture asset licenses

The project-authored original and generated texture content in this directory
is dedicated to the public domain under the **Creative Commons CC0 1.0
Universal** dedication:

https://creativecommons.org/publicdomain/zero/1.0/

This project dedication covers `source/material_atlas_original.png`, layers
0--12 of `material_albedo.ppm`, and their seam-processed albedo and derived
tangent-space normal outputs from `newtons_texture_builder`. You may copy,
modify, redistribute, and use this content for any purpose, including
commercial use, without attribution.

Layers 13--15 and their files under `source/polyhaven/` and `tiles/` come from
Poly Haven, which separately publishes all of its assets under CC0. Poly
Haven's authoritative asset-license page is:

https://polyhaven.com/license

Those layers therefore have the same free-use, modification, redistribution,
and commercial-use permissions. Attribution is not required, but their exact
asset pages and file hashes are retained in `CC0_SOURCES.md` for provenance.

## Provenance

The original source atlas was created on 2026-07-16 with OpenAI's built-in
image generation from project-authored text prompts. No third-party image,
texture, photograph, or other visual source was supplied as input to that
generation. Layers 0--12 of the PPM remain project-local normalizations of the
generated atlas. On 2026-07-17, the three reserved cells 13--15 were replaced
with mechanical derivatives of the Poly Haven sources documented in
`CC0_SOURCES.md`. Build outputs are algorithmic derivatives produced by the
project's C++20 texture builder.

Generative systems can produce coincidental similarities to existing work, so
absolute originality cannot be guaranteed. The project makes the CC0
dedication to the extent it has rights to do so and provides this provenance
record so downstream users can make their own risk assessment.

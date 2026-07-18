# Material texture manifest

`material_albedo.ppm` is a P6 RGB atlas with exact dimensions 1024x1024. It is
divided into a 4x4 grid of 256x256 source cells. Cells are numbered in row-major
order from the top-left: `layer = row * 4 + column`.

| Layer | Cell | Intended surface |
| ---: | :---: | --- |
| 0 | 0,0 | Neutral off-white plaster fallback |
| 1 | 1,0 | Rough concrete |
| 2 | 2,0 | Painted concrete |
| 3 | 3,0 | Dark rubber |
| 4 | 0,1 | Brushed metal |
| 5 | 1,1 | Wood |
| 6 | 2,1 | Smooth latex |
| 7 | 3,1 | Porous foam |
| 8 | 0,2 | Pale viscous goo |
| 9 | 1,2 | Ceramic |
| 10 | 2,2 | Fired clay |
| 11 | 3,2 | Polished blue bowling-ball composite |
| 12 | 0,3 | Blue plush fabric |
| 13 | 1,3 | Poly Haven `concrete_pavement_02` chamber floor |
| 14 | 2,3 | Poly Haven `concrete_wall_003` worn painted-concrete wall |
| 15 | 3,3 | Poly Haven `corrugated_iron_03` industrial ceiling panel |

Layers 13--15 are 256x256 wrap-safe derivatives of official Poly Haven 1K
diffuse maps. Their preserved downloads live in `source/polyhaven/`, and the
exact install-ready cells live in `tiles/`. Layers 0--12 remain pixel-identical
to the preceding atlas. See `CC0_SOURCES.md` for source URLs, hashes, license,
and the mechanical conversion record.

`newtons_texture_builder` validates the P6 header, exact dimensions, maxval
255, payload length, and non-flat variance in every cell. It symmetrically
blends a narrow band at each cell's opposing edges, making the outer texels
match while preserving the cell interior. The builder then derives
`material_normal.ppm` with wrap-around Sobel luminance gradients independently
inside each cell. RGB encodes a normalized tangent-space vector in `[0,255]`.

The generated files remain 1024x1024 P6 RGB atlases. For GPU use, each cell may
be copied into a corresponding 256x256 2D-array layer; this avoids filtering
between unrelated atlas cells and keeps layer numbers synchronized with the
engine's `SurfaceMaterial` values. If sampled as one atlas instead, clamp local
coordinates to each cell's texel centers and do not generate cross-cell mipmaps.

See `LICENSE.md` for the CC0-1.0 dedication and provenance.

## Historical source-generation prompt

The prompt below documents the original generated atlas. Its three reserved
bottom-row cells have since been replaced by the independently sourced CC0
materials listed above; it remains here as provenance for layers 0--12 and
`source/material_atlas_original.png`.

Use case: stylized-concept. Asset type: tileable game texture atlas. Primary
request: an exact orthographic 4 by 4 grid of sixteen equal square seamless
material swatches for a Vulkan game. Cell order left-to-right, top-to-bottom:
neutral matte plaster; rough industrial concrete; painted concrete; fine-grain
rubber; brushed steel; natural wood grain; smooth latex; closed-cell
polyethylene foam; glossy PVA goo; white alumina ceramic; fired red clay;
polished urethane bowling-shell material; woven plush fabric; rough warehouse
concrete variation; painted steel variation; neutral fallback fabric.
Style/medium: realistic PBR-style diffuse albedo source atlas, neutral flat
color response. Composition: square 1024x1024, exact 4x4 cells, top-down flat
surfaces, no perspective. Constraints: every cell independently seamless at
all four edges; no borders, gutters, labels, text, logos, watermarks, objects,
lighting gradients, directional highlights, cast shadows, ambient occlusion,
or perspective; tint-friendly restrained colors; neighboring cells must remain
visually distinct.

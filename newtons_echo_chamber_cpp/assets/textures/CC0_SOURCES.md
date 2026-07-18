# Poly Haven CC0 texture provenance

Retrieved 2026-07-17 from Poly Haven's official site. Poly Haven states that
all of its assets are released under CC0 and may be used, modified,
redistributed, and used commercially without attribution:

- Authoritative asset license: https://polyhaven.com/license
- CC0 1.0 Universal deed: https://creativecommons.org/publicdomain/zero/1.0/

Attribution is not required by CC0. These links and hashes are retained so a
downstream distributor can audit the exact source of every imported pixel.

## Installed sources

| Atlas layer | Chamber use | Poly Haven asset and page | Preserved 1K diffuse source | Upstream MD5 | Local SHA-256 |
| ---: | --- | --- | --- | --- | --- |
| 13 | Floor | [`concrete_pavement_02`](https://polyhaven.com/a/concrete_pavement_02), Charlotte Baglioni | [`concrete_pavement_02_diff_1k.jpg`](https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/concrete_pavement_02/concrete_pavement_02_diff_1k.jpg), 1,052,909 bytes | `bdee8d4deebb78609414453a5fb299e2` | `e129ee6b59b6ed7f65b4c4c5ba9eb583d13d040ec26bef5d5d4e0080fe342e7a` |
| 14 | Worn painted-concrete walls | [`concrete_wall_003`](https://polyhaven.com/a/concrete_wall_003), Dimitrios Savva and Rico Cilliers | [`concrete_wall_003_diff_1k.jpg`](https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/concrete_wall_003/concrete_wall_003_diff_1k.jpg), 264,756 bytes | `1277cc6bcfc8e4f074f9852fbf19b2d3` | `7d1d4b9f5ed1aa3e6385ef3232e26e123a3dac5f364e44f9f34784e39bf8d6bb` |
| 15 | Galvanized industrial ceiling panels | [`corrugated_iron_03`](https://polyhaven.com/a/corrugated_iron_03), Charlotte Baglioni | [`corrugated_iron_03_diff_1k.jpg`](https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/corrugated_iron_03/corrugated_iron_03_diff_1k.jpg), 479,372 bytes | `e066ab142176aee5e974f86c0463c02c` | `d1378cbe96c625f79180147873c586bfc653ef5a66ea7a4a58cd5b18701f2bb4` |

The preserved downloads are the files in `source/polyhaven/`. Only diffuse
maps were imported: the current engine derives a periodic tangent-space normal
atlas from albedo luminance, so carrying unused upstream normal, roughness,
metalness, AO, or displacement maps would add distribution weight without
changing rendering.

## Mechanical conversion and installation

Each 1024x1024 sRGB JPEG was downsampled to an 8-bit 256x256 P6 PPM with
ImageMagick 6.9.11-60 using Lanczos filtering:

```sh
convert source/polyhaven/ASSET_diff_1k.jpg -filter Lanczos \
  -resize 256x256\! -colorspace sRGB -depth 8 /tmp/layerNN_raw.ppm
```

The raw tiles were composited into cells `(1,3)`, `(2,3)`, and `(3,3)` of a
temporary copy of `material_albedo.ppm`. The project builder then applied its
24-texel symmetric opposing-edge blend:

```sh
./build-vulkan/newtons_texture_builder /tmp/raw_atlas.ppm \
  /tmp/wrap_atlas.ppm /tmp/wrap_normal.ppm
```

Cells 13--15 were cropped from that validated wrap-safe output into `tiles/`
and composited back at offsets `+256+768`, `+512+768`, and `+768+768`.
ImageMagick crop comparison confirmed that atlas cells 0--12 were unchanged.

| Installed derivative | SHA-256 |
| --- | --- |
| `tiles/layer13_concrete_pavement_02.ppm` | `c525975e6c028696629c4390250ec4935dad5e9f853b80bb2b8635feaa5315e4` |
| `tiles/layer14_concrete_wall_003.ppm` | `1ff34aa1c8dd6bc339175e54fd1f694785e6b177065c5c1d2d69a3f2e2b96e0e` |
| `tiles/layer15_corrugated_iron_03.ppm` | `7329dd567b9f932a576ebd208b5ef89300dc2315a8b019eff31d584c0ed19c28` |
| `material_albedo.ppm` after installation | `8056b70c03d0783fb093087a044a67900389411fc8992b7dc42ce10cb1b2dd8a` |

All three tile inputs have identical opposing outer texels on both axes and
remain non-flat after downsampling. `newtons_texture_builder --check` is the
canonical format, variance, wrap-safety, and derived-normal validation.

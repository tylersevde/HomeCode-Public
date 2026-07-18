# Newton's Echo Chamber — C++20 / Vulkan 1.3 Pi 5 foundation

This directory is the new native foundation for Newton's Echo Chamber on a
Raspberry Pi 5. It is C++20 throughout, has no Python or PyOpenGL runtime, and
uses Vulkan for both presentation and secondary compute work. This is a
working vertical slice, not yet a feature-for-feature visual port of the older
renderer.

## What works now

- Vulkan dynamic rendering and synchronization2, with swapchain presentation,
  explicit barriers, and frames-in-flight synchronization.
- An X11 window/input backend that runs in a Raspberry Pi OS X11 session or
  through XWayland in the default Wayland desktop session.
- The existing deterministic **4,487-body** simulation as authoritative
  double-precision CPU physics at a fixed 120 Hz.
- A persistent four-core CPU layout: three worker threads plus the main thread,
  configurable down to one core for comparison and diagnosis.
- Vulkan-instanced room geometry and rigid-body drawing, with a shared,
  mipmapped material-texture array and tangent-space normal/bump mapping on
  every room surface and rigid-body proxy.
- A CC0 Material Lab room skin: physically repeated Poly Haven pavement,
  concrete-wall, and corrugated-iron layers; `F11` cycles full/albedo/uniform
  views and `F12` compares adaptive against full-distance normal detail.
- Dynamic ceiling lighting that toggles independently on `V`, plus a warm,
  soft-edged head-mounted flashlight that toggles on `C`, follows the
  interpolated view direction, and casts dynamic shadows through a 1024x1024
  depth allocation owned by each in-flight frame context. Quality tiers render
  512/768/1024-pixel regions with 18/25/32-metre ranges.
- Flashlight-only shadow rendering. The ceiling-grid point lights remain
  deliberately unshadowed so the Pi does not pay for multiple shadow maps and
  six faces per point light. Dedicated flashlight-frustum caster batches and
  three projected-size sphere LODs keep distant/off-cone stock out of expensive
  vertex and shadow work.
- Vulkan compute particles as GPU-resident secondary physics, ordered with the
  graphics work and requiring no per-frame CPU readback. A small fence-safe
  impact-command upload makes collisions emit material-specific bursts while
  the remaining particles retain the chamber's ambient dust motion.
- An in-window laboratory HUD with a crosshair, selected-object physics probe,
  experiment status, live simulation controls, and an optional help overlay.
- A player-fired authoritative Echo Pulse on `Q`: deterministic radial body
  impulses, bounded recoil, cooldown/impulse telemetry, cyan/indigo wavefronts,
  capped velocity glyphs, and a synthesized cue routed through room reverb.
- Pause, exact 120 Hz single-step, and a deterministic Galileo vacuum-drop
  experiment using the equal-diameter steel and concrete calibration balls.
  Restarting the experiment preserves its prior trajectories as visual echoes.
- A gravity-proof trick-shot experiment that fires an existing ceramic marble
  at a simultaneously released plush target. Its paired trajectory echoes,
  authoritative CCD hit marker, and hit clock expose the gravity-independent
  relative motion while the absolute arc still follows the selected gravity.
- Optional kinetic-energy and impact-resonance visualizers: a logarithmic
  energy heat lens for all rigid bodies and bounded, material-coloured shells
  emitted by the strongest authoritative impacts.
- Per-frame V3D timestamp queries that separately report particle-compute,
  flashlight-shadow, main-scene, and combined compute/shadow/scene intervals.
- A standalone measured rigid-body predictor/spatial-hash prototype with an
  independent CPU oracle, V3D timestamps, readback timing, and overflow tests.
- Native procedural audio streamed to `aplay`, with six damped wall reflections
  and a bounded four-line feedback-delay-network room reverb. The preferred
  transport requests an 80 ms buffer with 20 ms periods and falls back when an
  ALSA device rejects explicit timing. Distant/rejected impacts no longer spend
  the audible voice quota, and `--no-audio` bypasses sound-bank synthesis.
- A 144-case deterministic CPU self-check, a standalone Vulkan capability
  probe, and a validation-enabled Vulkan presentation smoke test.

The CPU remains authoritative for gameplay rigid bodies, collision response,
pickup/throw state, sleeping, and reset determinism. GPU compute currently
assists with secondary particles; it does not replace the CPU rigid-body
solver or feed nondeterministic results back into gameplay.

## Raspberry Pi 5 Vulkan compatibility

The tested target is a Raspberry Pi 5 with the BCM2712's four Cortex-A76 CPU
cores and the VideoCore VII GPU. On the tested Raspberry Pi OS image, the
device probe reports:

- `V3D 7.1.10.2` using Mesa `24.2.8`;
- physical-device API `1.2.289`;
- Vulkan loader `1.3.239`; and
- a `4 GiB` Vulkan memory heap.

Mesa 24.2.8 does not advertise core Vulkan 1.3 for this V3D device. The engine
therefore implements a Vulkan 1.3 feature profile by negotiating the promoted
`VK_KHR_dynamic_rendering` and `VK_KHR_synchronization2` features and commands
on the Vulkan 1.2 device. Shader binaries target Vulkan 1.2 for the same
reason. If a Raspberry Pi OS update supplies Mesa 24.3 or newer and V3DV
exposes core Vulkan 1.3, the renderer selects the core path automatically.

The Pi's memory is unified. The reported 4 GiB Vulkan heap is not a claim of
4 GiB dedicated VRAM, and the engine does not preallocate either that heap or
the board's full 8 GiB of RAM. It allocates the world, staging, instance,
particle, depth, and swapchain resources it actually needs, leaving the rest
available to Raspberry Pi OS and filesystem caching.

## Install and build on Raspberry Pi OS

Use a 64-bit Raspberry Pi OS installation. On a fresh Bookworm image:

```bash
sudo apt update
sudo apt install build-essential cmake glslc libvulkan-dev \
    mesa-vulkan-drivers vulkan-tools vulkan-validationlayers \
    libx11-dev zlib1g-dev alsa-utils

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
```

After both commands complete, `./build/newtons_echo_chamber_alpha` is the
canonical Vulkan executable. The other `build-*` directories are development
and verification trees. If `C` appears to do nothing after updating the source,
reconfigure and rebuild `build`: that symptom was traced once to a stale legacy
artifact, not the current Vulkan input path.

Release builds enable LTO when the compiler supports it. On AArch64 they also
enable Cortex-A76 compile tuning by default. For a portable development build,
configure with `-DNEC_NATIVE_TUNING=OFF`; LTO can be disabled with
`-DNEC_ENABLE_LTO=OFF`.

## Run

Start from this directory inside the graphical desktop session:

```bash
./build/newtons_echo_chamber_alpha
```

Useful profiles include:

```bash
./build/newtons_echo_chamber_alpha --view warehouse
./build/newtons_echo_chamber_alpha --frames 360 --view warehouse --no-audio
./build/newtons_echo_chamber_alpha --quality safe --gpu-physics off
./build/newtons_echo_chamber_alpha --maximum-throughput --frames 360
./build/newtons_echo_chamber_alpha --flashlight on --shadows on --view warehouse
./build/newtons_echo_chamber_alpha --flashlight on --shadows off --view warehouse
./build/newtons_echo_chamber_alpha --overhead-lights off --flashlight on
./build/newtons_echo_chamber_alpha --textures on --bump-mapping on
./build/newtons_echo_chamber_alpha --stress-pallet --view pallet
```

`--quality` accepts `auto`, `safe`, `balanced`, or `ultra`;
`--cpu-cores` accepts 1 through 4; `--cpu-backend` accepts `scalar`, `auto`, or
`maximum`; `--gpu-physics` accepts `off`, `auto`, or `maximum`; and
`--flashlight`, `--overhead-lights`, `--shadows`, `--textures`, and
`--bump-mapping` accept `off` or `on`.
The overhead lights and flashlight default to on and off respectively.
Textures and bump mapping both default to on. With textures disabled, bump
mapping can remain armed but performs no texture sampling.
`--shadows` controls the flashlight shadow pass; it does not add shadows to the
ceiling point lights. Use `--help` for the complete command-line list.

## Checks

The CPU self-check, standalone probe, and CTest entry do not need a window.
The Vulkan smoke test needs an X11 display (a native X11 desktop or XWayland):

```bash
./build/newtons_echo_chamber_alpha --check
./build/newtons_texture_builder --check assets/textures/material_albedo.ppm
./build/newtons_vulkan_probe
./build/newtons_gpu_physics_probe --samples 100 --warmup 10
./build/newtons_echo_chamber_alpha --vk-check
ctest --test-dir build --output-on-failure
```

`--vk-check` creates the real swapchain, enables the Khronos validation layer,
presents SAFE/BALANCED/ULTRA frames with flashlight shadows OFF, ARMED, and ON,
forces both frame-local maps through their first read-to-depth transition,
rejects a software Vulkan device, and on a Pi requires a V3D/V3DV device.
`--gl-check` is retained only as a legacy alias for `--vk-check`; it no longer
invokes OpenGL.

The probe is the quickest way to see the loader/device API versions, driver,
promoted feature support, queue capabilities, and Vulkan heaps before starting
the game. Controlled Vulkan measurements for this slice are recorded below;
they are not a claim about a future feature-complete renderer.

## Dynamic lighting and flashlight shadows

`V` toggles the ceiling-grid point lights without changing the flashlight or
its shadow policy. With the overhead grid off, its point lights are removed and
the residual room ambient is reduced rather than eliminated; an enabled
flashlight continues to illuminate and shadow the scene independently.

Dynamic shadows are intentionally limited to the head-mounted flashlight. The
two frame contexts each own a 1024x1024 depth map, so one frame cannot overwrite
a map that the GPU is still sampling from another frame. Only one map is used
by a rendered frame. SAFE/BALANCED/ULTRA render into 512/768/1024-pixel regions
of that allocation and use 18/25/32-metre flashlight ranges. When the flashlight
and `--shadows on` are both active, the renderer builds dedicated caster batches
for that tier's flashlight frustum, draws them into the active map region,
transitions it for filtered comparison sampling, and uses it while shading the
main scene. The caster pass therefore does not inherit distant geometry merely
because it is visible to the 175-metre camera.
`--shadows off` retains the same flashlight illumination and cone math without
submitting the shadow pass, which makes it the direct performance and visual
control case.

The renderer prefers a filterable D16 depth target, enables comparison
sampling and raster/receiver bias, and falls back to four explicit comparison
taps when the selected sampled-depth format cannot filter linearly.

The room's ceiling grid uses nearby point lights selected for the current
quality tier. Those lights remain unshadowed by design. Omnidirectional point
shadows would require a cube map with six views per light; multiplying that by
the selected ceiling lights is not a sensible default workload for the
VideoCore VII. This scoped design spends one shadow map on the player-directed
light where occlusion is most noticeable.

`C` toggles the flashlight itself. `--flashlight on|off` and
`--overhead-lights on|off` select the two lights' startup states, while
`--shadows on|off` independently selects whether the enabled flashlight submits
and samples its shadow map. F5 restores both configured lighting startup states
along with the deterministic world reset.

Fixed-frame benchmark output separates particle-compute, flashlight-shadow,
main-scene, and the combined compute/shadow/scene interval when V3D timestamps
are available. The combined interval ends before the final present-layout
barrier; none of these queries include XWayland compositor or presentation
latency. Performance runs should keep the camera, seed, quality, shadow
resolution, particle count, and thermal state fixed, and should compare
separate `--shadows off` and `--shadows on` processes rather than mixing states
with the interactive toggle.

Most recorded V3D tables below are retained as pre-laboratory-milestone
baselines. They predate impact-command particles, flashlight-specific caster
culling, sphere LOD, back-face culling, the HUD, resonance shells, and room
reverb. The preserved-binary culling A/B in the dynamic-shadow section is
explicitly labelled as a current exception; values from the historical tables
should not be mixed into that comparison.

## Texture and bump-material system

The room and every rigid-body proxy use two shared GPU-resident 2D texture
arrays: sRGB albedo and linear tangent-space normals. The cube and sphere
meshes carry explicit UVs and tangents, while the existing per-instance color
remains a tint so seeded object colors are preserved. The 16-layer arrays have
full CPU-built mip chains, are uploaded once to device-local memory at startup,
and add no per-frame upload or readback. The flashlight shadow pass remains
depth-only and does not sample either material array.

All 16 cells now have defined roles. Layers 0-12 retain the original neutral,
concrete, paint, rubber, metal, wood, latex, PE foam, PVA goo, alumina ceramic,
fired clay, bowling urethane, and plush surfaces byte-for-byte. The former
reserved cells add three room-shell materials from Poly Haven:

- layer 13: [Concrete Pavement 02](https://polyhaven.com/a/concrete_pavement_02)
  on the floor, repeating at its authored 1.8-metre span;
- layer 14: [Concrete Wall 003](https://polyhaven.com/a/concrete_wall_003) on
  the four walls, repeating every 3 metres; and
- layer 15: [Corrugated Iron 03](https://polyhaven.com/a/corrugated_iron_03) on
  the ceiling, repeating every 2 metres.

Poly Haven publishes these assets under CC0. The official 1K diffuse sources
are preserved locally, while the installed 256x256 tiles are edge-blended for
wrapping. Upstream and local hashes, authors, source URLs, exact conversion
commands, layer mapping, and license details are in
[`assets/textures/CC0_SOURCES.md`](assets/textures/CC0_SOURCES.md). The C++20
`newtons_texture_builder` validates every source cell, reduces wrap seams, and
derives a periodic tangent-space normal field from the diffuse luminance. No
unused upstream PBR maps are redistributed. Compute particles are single-pixel
points rather than mesh surfaces, so bump mapping is not meaningful for them.

This is normal/bump mapping, not tessellation or displacement. It changes
lighting response but not collision shapes, silhouettes, or shadow-caster
geometry. `--textures off` and `--bump-mapping off` provide uniform shader
control cases for target-hardware measurements.

`F11` cycles the live material view through albedo plus normals, albedo only,
and uniform tint. `F12` switches between the default adaptive normal detail and
a full-distance comparison. SAFE/BALANCED/ULTRA retain full normal strength to
8/13/20 metres, fade to the geometric normal by 15/24/38 metres, and issue no
normal-map fetch beyond that cutoff. Material response is selected in the
vertex stage and passed as a flat value; this removes a material-family branch
ladder from every covered fragment.

The original texture-cost table below predates the CC0 room layers and is kept
as a historical material-system baseline. On the target Pi 5/V3D hardware,
three counterbalanced 360-frame Release runs
per mode used the warehouse view, ULTRA quality, the flashlight and its dynamic
shadow, the overhead grid, and CPU-authoritative physics. These are medians:

| Surface mode | Steady FPS | 1% low | GPU total | Main scene | CPU physics |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform control | 37.60 | 22.62 FPS | 21.20 ms | 16.15 ms | 2.84 ms/frame |
| Albedo textures | 35.66 | 20.90 FPS | 22.07 ms | 16.85 ms | 3.26 ms/frame |
| Albedo + bump mapping | 33.47 | 18.84 FPS | 24.23 ms | 18.94 ms | 3.52 ms/frame |

Albedo sampling therefore cost about `0.87 ms` of median GPU time in this
stress view, and tangent-space bump mapping brought the total increase to about
`3.03 ms` versus the uniform control. All three modes reported `71 MiB` RSS and
`73 MiB` Vulkan heap usage because the arrays are allocated once at startup;
the toggles isolate shader cost. Keeping authoritative rigid-body physics on
the four CPU cores remains the efficient choice while V3D handles materials,
dynamic lighting, flashlight shadows, and secondary compute particles.

The CC0 Material Lab path received a separate preserved-binary check on the
same Pi 5/V3D device. Each row below is the median of three 720-frame Release
runs using the ULTRA warehouse, flashlight and shadows on, overhead grid on,
GPU particles off, and maximum-throughput pacing. The first 120 frames were
excluded. The preserved executable loaded its own pre-milestone shaders and
textures; the two current rows differ only by `F12`.

| Matched ULTRA warehouse | Steady FPS | 1% low | p95 frame | GPU total | Shadow | Main scene |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Preserved pre-milestone | 53.50 | 40.86 FPS | 22.26 ms | 16.25 ms | 1.33 ms | 14.87 ms |
| Current, full-distance normals | 52.90 | 38.71 FPS | 22.44 ms | 16.48 ms | 1.40 ms | 15.04 ms |
| Current, adaptive normals | 53.72 | 41.92 FPS | 22.26 ms | 16.28 ms | 1.35 ms | 14.91 ms |

Adaptive detail saves `0.20 ms` of GPU total and `0.13 ms` of main-scene work
versus forced full-distance normals in this view. The final richer room is
within `0.03 ms` of the preserved GPU total, which is measurement parity rather
than a meaningful regression or win. A rejected first shader evaluated the
16-way material ladder per fragment and measured `21.32 ms` GPU total. Putting
room layers on the fast path, using squared-distance gating, and finally moving
the invariant response to a flat vertex output recovered `5.04 ms`. An SPIR-V
`OpSwitch` experiment was also rejected after measuring substantially slower
on V3D than the final flat-output path.

## Measured GPU broadphase/body predictor

`newtons_gpu_physics_probe` is a deliberately separate feasibility harness. It
packs the real 4,487-body scene, predicts one gravity-only step on the GPU,
builds conservative swept-sphere envelopes, hashes compact bodies, exhaustively
handles large bodies, and reads the candidate stream back for deterministic
normalization. Sleeping bodies remain queryable, while sleeping/sleeping pairs
are omitted. This preserves awake/sleeping wake-up coverage without making the
GPU result authoritative.

Correctness has two independent gates:

- the normalized GPU stream must exactly equal a float all-pairs reference for
  the same conservative envelopes, with no dropped or duplicate pairs; and
- every double-precision relative-motion swept-sphere candidate using the
  production `0.006 m` broadphase skin must be contained in that stream.

The input radius adds half the production skin plus a `0.0001 m` narrowing
guard. The guard and midpoint/half-travel envelope intentionally produce a
small conservative superset. `--mixed-sleep-check` supplies a targeted case
containing awake-compact/sleeping-large and awake-large/sleeping-compact pairs;
an overlapping sleeping/sleeping pair must remain absent.

Measured on the tested Pi 5/V3D device in a Release build at a `0.25 m` cell
size, after eight deterministic preparation ticks, 100 samples and 10 warmups:

| Measurement | Median | p95 |
| --- | ---: | ---: |
| Existing four-core CPU production broadphase | 15.192 ms | 19.074 ms |
| GPU predictor + hash + pair kernels | 31.171 ms | 32.948 ms |
| Queue submit through fence | 31.405 ms | 33.636 ms |
| Prepacked input copy through verified readback/normalization | 32.992 ms | 35.300 ms |

That run classified 4,436 compact and 51 large bodies. The GPU returned 23,024
envelope pairs, containing all 22,912 double-precision true-sweep pairs with
112 conservative extras and zero overflow, duplicates, missing pairs, or
unexpected pairs. The production CPU path retained 16,888 candidates after its
existing high-speed/course-stock policies, so a GPU-assisted path would still
need an equivalent deterministic post-pass. The probe's six host-visible
buffers used 8.422 MiB and were reported host-coherent and device-local on the
unified-memory V3D device.

A settled 480-tick scene (one awake body) measured 6.848 ms median for the GPU
kernels and 7.026 ms for verified end-to-end work, versus 0.236 ms for the CPU
broadphase. Cell sizes `0.20`, `0.24`, `0.30`, and `0.50 m` were also checked;
none changed the integration decision. The targeted mixed-sleep case retained
exactly its two required candidates. A 100-pair buffer test failed safely and
reported all 22,924 dropped writes; Khronos validation, SPIR-V validation, and
an ASan/UBSan host smoke test were clean.

The result is therefore a measured **no-go** for this linked-list/atomic/readback
design, not a claim of completed GPU-assisted gameplay physics. It remains a
reproducible prototype for the next design iteration. A future candidate should
use a more V3D-friendly scan/sort or fully GPU-resident narrowphase and must beat
the CPU path including deterministic normalization before integration.

## Measured unshadowed flashlight baseline and physics placement

These measurements predate the 1024x1024 shadow pass and use `--shadows off`;
they must not be quoted as shadow performance. They were taken
on the Pi 5/V3D device described above in a Release build at 1280x720. The
application used its X11 backend through XWayland in a Wayland session. Each
table entry is the median of three alternating runs. The 720-frame tests
exclude the same first 120 frames from frame-time, CPU-physics, and
GPU-timestamp averages; the stressed 360-frame tests exclude 72. The CPU
remained at 2.4 GHz under the `ondemand` governor; observed SoC temperatures
were 57-63 C. No native-Wayland or long-duration thermal claim is implied.

First, ULTRA warehouse rendering was run uncapped with secondary particles
disabled. Both sides of the A/B used identical maximum-throughput settings, so
that bundled profile does not vary within this comparison.

| ULTRA warehouse, GPU particles off | Flashlight off | Flashlight on |
| --- | ---: | ---: |
| Steady frame rate | 59.31 FPS | 54.22 FPS |
| 1% low | 31.10 FPS | 28.42 FPS |
| p95 frame time | 27.99 ms | 31.03 ms |
| Submitted GPU work | 13.62 ms | 15.05 ms |
| Scene graphics | 13.59 ms | 15.03 ms |
| CPU rigid physics per fixed step | 1.00 ms | 1.01 ms |

The flashlight therefore costs 1.43 ms of submitted V3D work in this view,
about 10.5%, while the CPU fixed-step cost is unchanged within measurement
noise. Its enable flag, interpolated eye origin, direction, range, and cone
data are the CPU's only per-frame contribution; illumination remains fragment
work on the GPU.

The second A/B kept ULTRA's 196,608 GPU-resident particles enabled. Normal
pacing was used, so wall-frame results include FIFO/XWayland/compositor
behavior; the Vulkan query columns isolate submitted GPU work.

| ULTRA warehouse, 196,608 particles | Flashlight off | Flashlight on |
| --- | ---: | ---: |
| Steady frame rate | 37.50 FPS | 35.34 FPS |
| 1% low | 29.58 FPS | 28.26 FPS |
| p95 frame time | 30.42 ms | 32.04 ms |
| Submitted GPU work | 22.16 ms | 23.58 ms |
| Particle compute | 4.40 ms | 4.51 ms |
| Scene graphics, including particle draw | 17.78 ms | 19.03 ms |
| CPU rigid physics per fixed step | 1.16 ms | 1.16 ms |

The added 1.42 ms remains graphics cost; particle-compute time changes by only
0.11 ms. ULTRA particles plus the flashlight are already beyond a 60 FPS GPU
budget on this path, so particle count/quality should be reduced before moving
any rigid physics onto the same V3D queue.

Finally, a launched steel body stressed the pallet scene with the flashlight
on, particles off, and otherwise identical maximum-throughput settings:

| Stressed authoritative rigid physics | 1 CPU core | 4 CPU cores |
| --- | ---: | ---: |
| CPU physics per rendered frame | 23.31 ms | 17.39 ms |
| CPU physics per fixed step | 8.33 ms | 7.34 ms |
| Steady frame rate | 39.89 FPS | 50.11 FPS |
| p95 frame time | 37.18 ms | 33.73 ms |
| Frames over 33.3 ms | 57/288 | 16/288 |
| Fixed-step backlog drops | 95 | 40 |

Four cores reduce median fixed-step time by 11.9% and physics time per rendered
frame by 25.4% in this workload. A fresh 30-sample post-lighting run of the
standalone rigid predictor also passed verification and measured 15.083 ms for
the four-core production CPU broadphase, 31.313 ms for GPU kernels, and 33.428
ms verified end-to-end. The placement decision is therefore:

- keep authoritative rigid-body integration, broadphase, narrowphase, contact
  response, sleeping, and gameplay state on the four Cortex-A76 cores;
- keep the flashlight and quality-budgeted secondary particles on the GPU,
  where the particles remain resident and require no CPU readback; and
- do not integrate the measured linked-list/atomic/readback rigid-body path.
  Revisit GPU rigid physics only with scan/radix-sort and a GPU-resident
  narrowphase that wins a complete, deterministic end-to-end comparison.

## Dynamic-shadow measurement and physics repeat

The post-change matrix was measured on the Pi 5/V3D device in a Release build
at 1280x720. Every process used the same ULTRA warehouse view, flashlight-on
state, normal pacing, seed, and 720-frame duration. The first 120 frames were
excluded from the CPU, wall-frame, and GPU timing windows. Configurations were
run in the balanced order `A B D C`, `C D B A`, then `B A C D`. SoC temperature
stayed at 58-59 C; the CPU used the `ondemand` governor with a 2.4 GHz maximum.
No native-Wayland, fixed-V3D-clock, throttling-state, or long-duration thermal
claim is implied.

GPU values below are the median of three per-run means over 600 timestamped
frames; the renderer does not retain per-frame GPU samples, so they are not GPU
medians or p95 values. Wall-frame values are medians of the three reported run
statistics.

| Config | Shadows | Particles | Steady FPS | 1% low | p95 frame | Frames >33.3 ms | GPU total | Particle | Shadow | Main scene |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | Off | 0 | 39.74 | 28.79 | 29.40 ms | 7/600 | 16.61 ms | 0.00 ms | 0.00 ms | 16.57 ms |
| B | On | 0 | 40.24 | 23.55 | 34.06 ms | 78/600 | 19.40 ms | 0.00 ms | 4.98 ms | 14.38 ms |
| C | Off | 196,608 | 36.38 | 28.07 | 31.55 ms | 10/600 | 22.70 ms | 4.44 ms | 0.00 ms | 18.19 ms |
| D | On | 196,608 | 34.37 | 27.55 | 32.87 ms | 23/600 | 25.50 ms | 4.30 ms | 3.31 ms | 17.82 ms |

The shadow-enabled total rises by 2.79 ms with particles disabled and 2.80 ms
with 196,608 particles. The interaction is 0.01 ms, while particle compute
changes by 0.14 ms, so there is no measured reason to move lighting or particle
simulation onto the CPU. The shadow interval is larger than the net total
increase and the following main-scene interval can be smaller. This is likely
V3D scheduling, tile-cache, or clock-state interaction rather than negative
work; the combined interval is the correct A/B cost. Configurations C and D
already exceed a 16.67 ms GPU budget, and B crosses it once shadows are enabled.
ULTRA particles plus shadows should therefore be treated as a quality tier
below 60 FPS, not spare capacity for authoritative rigid-body compute.

### Current outward-winding and back-face-culling A/B

The Resonance Lab renderer was also compared with a preserved pre-change
executable on the same Pi 5/V3D display. Three 720-frame runs of each binary
used the ULTRA warehouse, particles off, flashlight and shadows on, normal
pacing, and `--no-audio`; each run discarded its first 120 frames. The
counterbalanced order was pre/post/post/pre/pre/post. Values are medians of the
three reported 600-frame windows:

| Metric | Preserved pre-change | Current CCW culling | Change |
| --- | ---: | ---: | ---: |
| Steady frame rate | 30.96 FPS | 39.28 FPS | +26.9% |
| 1% low | 20.11 FPS | 25.79 FPS | +28.2% |
| p95 frame time | 39.00 ms | 33.48 ms | -14.2% |
| Frames over 33.3 ms | 277/600 | 37/600 | -86.6% |
| GPU total | 22.36 ms | 17.36 ms | -22.4% |
| Flashlight shadow | 3.77 ms | 3.20 ms | -15.1% |
| Main scene | 18.47 ms | 14.18 ms | -23.2% |

The first culling attempt exposed the wrong winding convention by reducing
matched-image mean luminance from `0.4743` to `0.3317`. It was rejected before
timing. With outward sphere indices and `BACK + COUNTER_CLOCKWISE` in both
opaque and depth pipelines, current luminance is `0.4710` and the front-lit
warehouse matches the preserved image. A separate 360-frame flashlight-off
run reported zero shadow instances, zero shadow submissions, and `0.00 ms`
shadow time, confirming that inactive caster construction and upload are
actually skipped.

A separate maximum-throughput pallet stress test kept the flashlight and
shadows on, disabled particles, launched the steel body, and compared one CPU
core with all four. Each result is the median of three 360-frame runs after the
same 72-frame warm-up:

| Stressed authoritative physics with shadows | 1 CPU core | 4 CPU cores |
| --- | ---: | ---: |
| CPU physics per rendered frame | 16.72 ms | 15.06 ms |
| CPU physics per fixed step | 9.36 ms | 8.17 ms |
| Production broadphase | 3.44 ms | 2.21 ms |
| Steady frame rate | 55.57 FPS | 59.13 FPS |
| 1% low | 40.73 FPS | 42.52 FPS |
| p95 frame time | 21.99 ms | 20.66 ms |
| Fixed-step backlog drops | 130 | 76 |
| GPU shadow interval | 3.28 ms | 3.39 ms |

Four cores reduce fixed-step time by 12.7%, broadphase time by 35.8%, and
physics time per rendered frame by 9.9%, while backlog drops fall by 41.5%.
The nearly unchanged GPU shadow interval shows that this gain comes from CPU
parallelism rather than shifting the lighting workload.

The authoritative-physics placement check was then repeated with the standalone
probe. It does not share the render submission, so it isolates the proposed
rigid-body algorithm rather than hiding its cost behind graphics:

```bash
./build/newtons_gpu_physics_probe --samples 30 --warmup 10
```

| Repeated placement measurement | Median | p95 |
| --- | ---: | ---: |
| Four-core production CPU broadphase | 15.017 ms | 20.743 ms |
| GPU predictor/hash/pair kernels | 31.095 ms | 32.985 ms |
| GPU submit through fence | 31.353 ms | 34.078 ms |
| Verified GPU end-to-end, including readback/normalization | 32.937 ms | 35.623 ms |

Verification passed with 23,024 conservative envelope pairs containing all
22,912 true swept pairs, 112 expected conservative extras, and zero overflow,
duplicates, missing pairs, or unexpected pairs. Verified GPU end-to-end work is
2.19 times the production CPU broadphase median before adding any GPU-resident
narrowphase or contact solve. The placement decision therefore remains:

- keep authoritative rigid-body integration, broadphase, narrowphase, contact
  response, sleeping, and gameplay state on the four Cortex-A76 cores;
- keep flashlight lighting, its shadow map, and quality-budgeted secondary
  particles on VideoCore VII; and
- do not integrate the linked-list/atomic/readback rigid-body prototype.

A future authoritative GPU design must avoid linked-list atomic insertion and
per-tick CPU candidate readback. The intended direction is a bounded
scan/radix-sort broadphase feeding a GPU-resident narrowphase, with deterministic
ordering, a tested CPU fallback, and a complete end-to-end win before gameplay
authority moves off the Cortex-A76 cores.

## Laboratory HUD and Galileo experiment

The Vulkan scene includes a compact bitmap HUD rather than relying on the X11
title bar for interaction feedback. Its probe reports the body under the
crosshair, including material/name, mass, linear speed, and kinetic energy,
alongside gravity, friction, throw force, contact/broadphase load, experiment
state, audio availability/mute state, and the latest world message. `F1`
expands the control reference and
`F2` hides the complete overlay for unobstructed viewing.

`F6` stages the observer and releases the existing equal-diameter steel and
concrete calibration balls from the same seven-metre centre height. Only those
two bodies omit buoyancy and aerodynamic drag during the run, so the ordinary
4,487-body world remains CPU-authoritative and otherwise unchanged. The paired
trajectory is sampled at the deterministic 120 Hz fixed step. Starting another
run preserves the completed/current path as a dimmer echo for direct visual
comparison; `F7` controls both trail sets.

`F3` freezes authoritative physics and secondary particle time while leaving
the camera and naturally decaying audio tail responsive. Entering pause safely
releases a held prop, and physics-mutating movement/tuning/interaction controls
remain blocked until resume. `F4` advances one exact fixed tick, including
impact events and one matching particle step.

## Resonance Lab: gravity-proof intercept and visual lenses

`F8` stages an existing ceramic marble and one of the seeded plush bodies. The
marble receives a 30 m/s launch along their initial line of sight at the exact
fixed tick in which the target is released. Both experiment bodies use a
body-local ideal vacuum, so buoyancy, aerodynamic drag, and linear damping are
omitted for them without changing the rest of the room. Gravity accelerates
both equally: changing `R` or `F` bends their absolute paths and moves the
collision height, while their
relative firing line and intercept time remain unchanged. The reported time and
crossed hit marker come from the authoritative CCD contact rather than a visual
estimate.

The projectile trail is violet and the target trail is lime. Restarting `F8`
keeps the preceding pair as dim echoes, making gravity comparisons visible in
the same view; `F7` hides or restores both Galileo and trick-shot trails.

`F9` switches the rigid-body material tint to a logarithmic kinetic-energy
lens, progressing from blue through cyan and gold to red. `F10` controls a
bounded ring of additive, material-coloured resonance shells emitted by the
strongest authoritative impacts. Their ages follow simulation time, so pause
freezes them and `F4` advances them by exactly one tick. The deliberately slow
13.5 m/s shell expansion is a readable visual chronoscope, not a claim that
sound propagates at that speed; audible impacts still use the separate native
reflection and room-reverb model.

## Echo Pulse experiment

`Q` fires an authoritative radial Echo Pulse. A 14-metre crosshair ray selects
the origin at the first body it hits; open space uses a point seven metres
along the view direction. Bodies whose bounding surfaces lie within eight
metres enter a deterministic nearest-first cohort. The cohort is capped at 48,
ties use body index, and the crosshair target is retained even in dense stock.
Held bodies are excluded. Adhesive payload bookkeeping is resolved before the
mass-aware, smooth-falloff impulses are applied, affected bodies wake, and the
player receives a bounded 0.18-0.70 m/s recoil unless position lock is active.

The 0.65-second cooldown advances only with fixed simulation time. Pause
therefore freezes both recharge and the retained visual event; `F4` advances
each by exactly one 120 Hz tick. Reset clears the event, per-body records,
serial, and cooldown. The pulse never fabricates a contact or impact event:
subsequent solver collisions remain the only source of impact particles,
resonance shells, and impact audio.

Each retained event draws a 21.5 m/s cyan front plus a delayed 16 m/s indigo
echo, both clamped to the real eight-metre influence radius. Up to four recent
events survive, with SAFE/BALANCED/ULTRA visibility caps of 2/3/4 events and
16/32/64 strongest current-body velocity glyphs. Camera rejection bounds
additive overdraw. The HUD reports READY/COOLDOWN, affected bodies, delivered
impulse, fronts, and vectors. The synthesized 0.86-second pulse cue sends 0.58
of its level into the existing bounded room reverb.

## Controls

- `W`/`S`: move forward/back; `A`/`D`: strafe
- Mouse: look; `Space`: one-metre jump
- Right mouse: capture the pointer or pick up/drop; left mouse: throw held item
- `R`/`F`: raise/lower gravity
- `T`/`G`: raise/lower throw force
- `Y`/`H`: raise/lower room friction
- `C`: toggle the head-mounted flashlight
- `V`: toggle the overhead ceiling lights independently
- `B`: lock/unlock the player's position while retaining mouse interaction
- `Q`: fire the authoritative radial Echo Pulse
- `F1`: show/hide the full in-window control reference
- `F2`: show/hide the laboratory HUD and selected-object probe
- `F3`: pause/resume authoritative physics and secondary particle time
- `F4`: while paused, advance exactly one 120 Hz physics tick
- `F5`: deterministic full-world reset, including configured lighting states
- `F6`: start/restart the Galileo vacuum drop and stage its observer camera
- `F7`: show/hide current and previous experiment trajectory echoes
- `F8`: start/restart the gravity-proof marble/plush intercept experiment
- `F9`: toggle the logarithmic kinetic-energy lens
- `F10`: toggle material-coloured impact-resonance shells
- `F11`: cycle full materials, albedo-only, and uniform-tint views
- `F12`: toggle adaptive/full-distance normal detail
- `M`: hard-mute active voices and room tail; `Tab`: release/recapture pointer;
  `Escape`: quit

Reset restores Earth gravity (`9.80665 m/s²`), friction `0.65`, throw force
`1 N`, the player start, the seeded scene, all 4,487 bodies awake, and the
flashlight state selected by `--flashlight` (off by default).
The overhead grid likewise returns to the state selected by
`--overhead-lights` (on by default). `--shadows on|off` is a run-level rendering
policy; neither `C` nor `V` changes it.
F5 also clears Echo Pulse state, restores the configured startup material mode
with adaptive normal detail, disables the kinetic lens, and re-enables
resonance shells; HUD/help, trajectory visibility, and mute remain user
preferences.

## Scene retained by the foundation

The original 551-body room contains seven calibration objects, a wooden bat,
a wheelbarrow, a helium balloon, five foam noodles, ten goo globs, twenty
ceramic marbles, 500 clay bricks on a pallet, and five seeded stuffed animals.
The warehouse adds five pallet bases, 300 bowling balls, 300 dodgeballs, 300
goo globs, 25 stacked EPAL-style pallets, 3,000 alumina-ceramic marbles, and six
Douglas-fir logs. Together they make exactly 4,487 simulated bodies.

## Architecture and performance intent

- The four-core job system parallelizes independent integration and broadphase
  candidate generation, then merges work before the ordered contact solver so
  helper scheduling does not define gameplay results.
- Structure-of-arrays scratch storage is aligned and reused to avoid per-tick
  allocation churn and to give GCC useful Cortex-A76 AdvSIMD/NEON loops.
- A sparse spatial grid handles the large warehouse population; sleep/wake and
  authored supported loads avoid wasting every tick on settled stock.
- Rendering batches the room and bodies through instanced Vulkan draws rather
  than issuing one draw call per object.
- Camera-visible and flashlight-shadow instance sets are culled independently;
  spheres select near, medium, or far meshes from projected size and quality.
- The 14-metre HUD probe reuses that body traversal, avoiding a second complete
  scan of all 4,487 bodies merely to identify the crosshair target.
- Echo Pulse selection performs one body scan into a fixed 48-entry nearest
  cohort with stable body-index ties. The selector itself does not allocate and
  its retained-event vector reserves the same cap at construction. It changes
  solver state directly but leaves contacts and impact events authoritative.
- Opaque and flashlight-depth passes use corrected outward mesh winding and
  back-face culling. When the flashlight or its shadow policy is inactive, the
  renderer skips construction and upload of the shadow-caster batches entirely.
- Impact resonance uses a fixed 12-pulse ring, admits at most the three strongest
  events per simulation update, applies quality caps of 4/8/12 visible shells,
  and rejects camera-enclosing shells to bound additive overdraw.
- Echo Pulse graphics retain four fixed events, apply quality caps before
  building fronts/glyphs, and use simulation time so pause and single-step need
  no renderer-specific clock correction.
- Material response constants are selected per vertex and passed flat instead
  of branching per fragment. Squared camera distance gates normal-map fetches;
  only the existing fog path requires the final square root.
- Compute and graphics share ordered Vulkan submissions on V3D. Secondary
  particles stay in GPU storage, so their simulation does not add a CPU copy
  to the frame path.
- Fence-safe timestamp queries measure particle compute, the flashlight shadow
  pass, and the main scene separately; fixed-frame benchmarks reset CPU and GPU
  timing after the same warm-up window.
- Quality modes and the frame governor reduce work before a slow frame turns
  into an unbounded fixed-step backlog. `--maximum-throughput` remains an
  explicit profiling mode, not the normal interactive default.

These choices are intended to use the Pi 5's four CPU cores, VideoCore VII,
and unified memory efficiently. They are not substitutes for target-hardware
measurement, and this README intentionally records no inherited OpenGL ES
benchmark as a Vulkan result.

## Current limitations and next milestones

- The platform backend is X11-first. XWayland works, but a native Wayland
  surface/input backend is not implemented yet.
- `--capture` is parsed for command-line compatibility but is currently
  rejected; Vulkan image readback and PNG capture are not implemented yet.
- Dynamic shadows cover only the flashlight's two frame-local 1024x1024 maps
  (one used per frame). Ceiling point-light shadows, volumetric scattering, and
  specialized multilayer material response remain staged rather than silently
  consuming a disproportionate Pi GPU budget.
- GPU compute is secondary rather than authoritative rigid-body physics. The
  first measured linked-list broadphase prototype is correct but slower than
  the CPU path, so it is intentionally not wired into gameplay. Any successor
  must avoid linked-list atomics and per-tick CPU readback, preserve
  deterministic ordering, include a tested fallback, and win a complete
  scan/radix-sort plus GPU-resident-narrowphase measurement before integration.
- Resonance shells and Echo Pulse fronts visualize chronology and influence;
  they do not solve wave propagation, diffraction, or room acoustics. Audible
  reflections and reverb remain a bounded perceptual model rather than a
  geometric acoustic simulation.
- The foundation renders the room, bodies, shared texture/normal materials,
  dynamic lights, and compute particles. Object-specific geometry and
  multi-part materials (for example pallet slats or separate wheelbarrow tray,
  handles, and tyre), plus a full settings/remapping interface, remain staged
  follow-up work. The new laboratory HUD is deliberately a compact probe rather
  than an editor or general-purpose UI toolkit.
- The short controlled measurements above do not replace repeatable long-run
  tests of compositor/present pacing, thermal behavior under concurrent
  graphics/compute, memory pressure, and stability.

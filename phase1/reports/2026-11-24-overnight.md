# Overnight report, 24 Nov 2026: why TPU-RTX's path-traced images miss Cycles, and what ray sorting would buy

Two questions, both answerable without TPU time:

- **A. Image gap.** TPU-RTX's path tracer scores 9.4–10.8 dB PSNR against Blender Cycles on the Open Data scenes. Where does that go?
- **B. WP1.** How much traversal work does TPU-RTX's packet tracer spend on incoherent rays, and how much would sorting rays between bounces save?

The TPU chips were busy all night with another job, so nothing here ran on a TPU chip. Everything ran on the CPUs of a TPU host, and nothing in this report is a TPU timing.

## Short version

- **TPU-RTX's geometry and camera are exact.** First-hit depth matches Cycles within 1% on 97% of Classroom's pixels.
- **The image gap is mostly scene translation, not light transport.** On Classroom, swapping in TPU-RTX's lights costs 6.6 dB and swapping in its exported materials costs another 8.8 dB. When both render the same exported scene, Cycles and TPU-RTX's estimator agree only to 19.8 dB PSNR.
- **There are concrete exporter bugs:**
  - The only area light kept for Classroom points out of the room.
  - Instanced lamps are read at the origin.
  - The sun is 10× the scene's strength.
  - A sky is added to a scene whose world is black.
  - 61% of Classroom's triangles are exported as near-metal.
- **Sorting helps, but not enough on its own.** The best sort keys cut leaf fetches 1.5–2.8× per sample across the three scenes. If every leaf were fetched only for rays that need it, the cut would be 15–19×. The headroom is in routing rays to leaves (WP2), not in sort keys.

## How the CPU runs relate to TPU-RTX

- **The real kernel runs on CPU.** TPU-RTX's Wide-BVH-64 kernel runs unchanged in Pallas interpret mode. On a test mesh it returns bit-identical hit distances to the original, and its hits agree with Embree on every ray (`tools/smoke_interpret.py`, `tools/test_counted.py`).
- **The real path tracer runs on CPU too.** `experiments/w7_pathtracer_1b/bvh_pbr_pt.py` runs unchanged on CPU devices (`tools/render_real_cpu.py`), but too slowly for full experiments: 8 minutes for 1 sample at 1 bounce.
- **A CPU twin does the full image experiments.** `tools/twin.py` is a line-by-line NumPy port of that path tracer with Embree ray casts, about 30 s per 16-spp image.
- **The twin matches the real renderer.** At 1 spp and 1 bounce on Classroom ([`results/classroom/validation_1spp_1b.json`](../results/classroom/validation_1spp_1b.json)):
  - against a 256-spp twin reference, the real renderer scores 18.05 dB and two twin seeds score 18.03 dB;
  - real vs twin is 16.81 dB, the same as twin vs twin (16.80 dB), so the two differ only by Monte Carlo noise.
- **The twin reproduces TPU-RTX's reported numbers** (from upstream's `out/metrics/v3_w7_opendata_pathtracer.json`), against Cycles references re-rendered with TPU-RTX's own exporter at 1,024 spp:

| Scene | Resolution | spp | TPU-RTX reported | Twin on CPU |
|---|---|---|---|---|
| Classroom | 640×360 | 16 | 9.45 dB (9.44 on 1 chip) | 9.46, 9.46 (2 seeds) |
| Classroom | 640×360 | 64 | 11.07 dB | 11.13 |
| Monster | 512×512 | 16 | 9.82 dB (9.81 on 1 chip) | 9.86, 9.87 |
| Monster | 512×512 | 64 | 9.99 dB | 10.05 |
| Junkshop | 640×320 | 16 | 10.81 dB | 10.83, 10.83 |
| Junkshop | 640×320 | 64 | 10.84 dB | 10.87 |

## A. Where the 9.4 dB on Classroom goes

Each row adds one group of TPU-RTX's choices to a Cycles render, then scores the result against the Cycles reference with TPU-RTX's own `compute_psnr`/`compute_ssim`. Rows 1–3 are Cycles renders. To render TPU-RTX's lights in Cycles, I converted their units and checked each conversion: area light within 0.02%, sun, sky, emission and world exact ([`results/calibration/calib.json`](../results/calibration/calib.json)). Rows 4–6 are the twin. Data: [`results/classroom/gap_classroom.json`](../results/classroom/gap_classroom.json); images: [`results/classroom/breakdown_sheet.jpg`](../results/classroom/breakdown_sheet.jpg).

| Step | What changes | PSNR vs Cycles | SSIM | Mean brightness vs reference |
|---|---|---|---|---|
| 0 | Cycles again, new seed (reference noise floor) | 80.9 dB | 1.000 | 1.00 |
| 1 | + TPU-RTX's camera and path settings: pinhole, 2 bounces after the first hit, box filter, no sample clamp | 26.3 dB | 0.874 | 1.44 |
| 2 | + TPU-RTX's lights: 1 disk light, sun, gradient sky, in place of the scene's 10 lamps and black world | 19.6 dB | 0.807 | 0.84 |
| 3 | + TPU-RTX's exported materials: one colour, roughness and material code per triangle | 10.8 dB | 0.390 | 2.48 |
| 4 | + TPU-RTX's light transport (twin, 1,024 spp, Blender Standard view) | 10.6 dB | 0.290 | 2.65 |
| 5 | + TPU-RTX's tone curve (ACES fit, exposure 0.55): its converged output | 12.1 dB | 0.304 | |
| 6 | + 16-spp noise: its output as benchmarked | 9.47 dB | 0.082 | |

Single changes against the reference:

| Change | PSNR |
|---|---|
| Depth of field off | 32.8 dB |
| Bounces capped at 2 | 45.7 dB |
| Scene's indirect clamp (3.0) removed | 28.6 dB, brightness ×1.44 |
| TPU-RTX's lights only | 20.3 dB |

### Causes, with evidence

**1. Lights**

- **The kept area light points out of the room.** The exporter keeps one area, point or spot light, chosen by energy over squared distance to the scene centre, plus the sun. In Classroom that is `exterior_fillLight` (1,963.5 W, at the window wall), which emits toward (0.73, 0, 0.68), away from the room. Of the 644,509 rays traced in one 640×360 sample (3 bounces), 11 lead to an area-light shadow test.
- **Instanced lamps are read at the origin.** The five spot lamps come from collection instances. The exporter reads light positions from `bpy.data.objects`, not the evaluated depsgraph, so they appear at (0, 0, 0).
- **The sun is 10× too strong.** The sun's strength becomes 10 × its colour; the scene's sun is 1.0 W/m².
- **A sky is invented.** Classroom's world is black at strength 0, and TPU-RTX adds a gradient sky with radiance 0.10–0.80, plus a glow around the sun.
- **Noise follows from this.** The interior is lit only by sun patches and sky light found by random bounces through the windows. That is why 17% of the 16-spp image's pixels are exactly black, and why noise costs 2.6 dB (rows 5 → 6).

**2. Materials**

- **Mixed materials become near-metal.** The exporter checks for a Glossy BSDF node before a Diffuse one and gives Glossy the code 0.95 (metallic 0.95). Classroom mixes the two, so 61% of triangles and 43% of visible pixels are exported as near-metal ([`export_stats`](../results/classroom/gap_classroom.json)).
- **Textures collapse to one colour per triangle.** Each texture is sampled once, at the triangle's centroid UV. 14.6% of triangles end up white (0.99) where a texture slot could not be read, and 2.5% get the default 0.7 grey.
- **Bump maps and procedural textures are dropped.** Classroom has 38 bump nodes and 19 noise textures; none survive.
- **Surface colours come out far too light.** On non-metal pixels the exported colour averages (0.72, 0.70, 0.69); Cycles' Diffuse Color pass averages (0.42, 0.36, 0.28).

**3. Light transport itself**

- **The two renderers disagree on the same scene.** Rendering TPU-RTX's exported scene and lights, Cycles and the twin agree only to 19.8 dB, with TPU-RTX 7% brighter.
- **Two candidates are ruled out.** The `+0.08` added to the light's squared distance changes nothing measurable. The clamps lower brightness by about 4%.
- **The rest is not yet attributed.** Remaining candidates: the BSDF model (its weights are not Principled's), glass handling, and no multiple importance sampling (combining light sampling with BSDF sampling).

**4. Geometry and camera are not a cause**

- **Depth matches.** First-hit planar depth matches Cycles' Depth pass within 1% on 97.1% of pixels (median error 0.04%).

**5. The tone curve partly hides the problem**

- **ACES helps here.** TPU-RTX's ACES curve at exposure 0.55 scores 1.4 dB better than Blender's Standard view on this 2.6×-too-bright image. That is compensation, not a match.

### Monster and Junkshop

MONSTER_JUNKSHOP_GAP

### What this means for the proposal

- **Image parity has to come before speed.** M1 and M2 are judged with rtx-h100's gate: PSNR against a converged Cycles reference. TPU-RTX's output cannot pass that gate at any speed today, so the proposal needs an image-parity work package before WP1–WP3. It would cover:
  - all lights, including instanced ones, in physical units;
  - the world shader;
  - a real material translation: Principled, Diffuse/Glossy mixes, and per-pixel textures, which means interpolating UVs and fetching textures in the kernel;
  - normal and bump maps;
  - the scene's own view transform.
- **The renderer gap is the part a TPU path tracer must close itself.** That is the 19.8 dB on an identical scene. It should be split further (BSDF, glass, MIS) before any speed claim.

## B. WP1: what sorting rays between bounces buys

**Setup**
- **What's counted.** TPU-RTX's closest-hit and any-hit kernels were copied unchanged except for counters added to the loop state ([`tools/counted_kernel.py`](../tools/counted_kernel.py)). They were run on every ray of one twin sample (3 bounces) of each scene, in several orders.
- **What's verified.** Kernel hits agree with Embree on every ray in all 125 runs.
- **What a leaf fetch is.** Fetching a leaf means one 12 KB DMA plus a test of 128 rays against 128 triangles, whether 1 ray or 128 needed it.
- **The floor.** The "floor" assumes every leaf fetch serves 128 rays that each need that leaf: the count of ray-leaf pairs needed, divided by 128.

**Results across scenes** (thousands of leaf fetches per sample)

| Scene | As shipped (8×16 pixel packets) | Best sort per ray set and bounce | Floor |
|---|---|---|---|
| Classroom (606k triangles, sun) | 1,992 | 714 (2.8× fewer) | 130 (15× fewer) |
| Monster (1.6M triangles, area light) | 883 | 583 (1.5× fewer) | 47 (19× fewer) |
| Junkshop (3.1M triangles, area light) | 905 | 540 (1.7× fewer) | 47 (19× fewer) |

Per-bounce detail: [`results/wp1_tables.md`](../results/wp1_tables.md).

What the numbers say:

- **Incoherence is the cost.** On Classroom, bounce-1 packets fetch 273 leaves each, against 31 for camera-ray packets. Only 6% of the ray-leaf pairs tested at bounce 1 involve a ray that needed that leaf, against 73% for camera rays.
- **Bounce rays sort poorly.** The best key puts the direction class first (16–36 octahedral bins, or the octant), then the hit point's Morton code. It saves 1.15–2.6× depending on scene and bounce.
  - At bounce 1, sorting by hit point alone saves only 4.5% on Classroom, because rays leaving nearby points scatter in random directions. By bounce 2 it does better (1.7× on Classroom, and Monster's best key).
  - 256 direction bins spread the rays too thin: 1.2× on Classroom at bounce 1, against 1.8× for 36 bins.
- **Shadow rays sort well.** Rays toward one light from nearby points share a direction. Sorting by hit point alone saves 10–12× on Classroom's sun rays at bounces 1–2, and 2.6–3.4× on area-light rays.
- **Removing finished rays matters for shadow rays, not bounce rays.** For closest-hit rays it changes leaf fetches by −20% to +6%, since few paths end before bounce 2. For shadow rays, many shading points need no shadow ray, so packing only the live ones saves 19–45% at bounces 1–2.
- **Sorting alone will not reach M1's 10×.** As shipped, leaf fetches sit 15–19× above the floor; the best sort removes 1.5–2.8× of that. The rest needs rays grouped by the leaves they need, which is WP2's per-treelet ray queues. Keep WP1 as the cheap first step, especially for shadow rays, and treat WP2 as the main lever.

**Caveats**
- **Fetches are not time.** These are counts, not timings, and assume time scales with leaf fetches.
- **Sorting cost isn't included.** `jax.lax.sort` on about 200k keys per bounce is not counted.
- **One sample per scene.** All counts come from one sample at TPU-RTX's benchmark resolutions.
- **Results are tied to this BVH.** TPU-RTX's BVH has a fixed 3-level, 64-wide layout, so other BVHs would give other numbers.

## What needs TPU time

1. **Time each piece separately:** each bounce and ray type (closest, area shadow, sun shadow) with the shipped kernel. This checks that time follows leaf fetches.
2. **Measure the cost of sorting:** `jax.lax.sort` and the gather that repacks rays, on 200k–260k rays per bounce, against the measured fetch savings.
3. **Re-time with the best sort keys,** to turn 1.5–2.8× fewer fetches into a speed number.

## Housekeeping

- **Code.** Branch `phase1` of this fork: tools in `phase1/tools/`, result JSONs in `phase1/results/`. Upstream code is unchanged.
- **Machine.** Everything ran on the CPUs of one shared TPU host, at low priority, without touching the TPU chips.
- **Over-threading incident.** For about 2 minutes, my twin processes each started a full BLAS thread pool, and my counting workers' XLA thread pools spun, pushing the host's load average to about 950. Fixed with one BLAS thread per process and one core per counting worker (`tools/common.py`, `tools/count_run.py`).
- **Faithful long run.** A run of TPU-RTX's own path tracer on CPU (Classroom, 16 spp, 3 bounces) was still running at the time of writing. It is an extra check on the twin; its result goes to `results/classroom/real_renderer_16spp_3b.json` if it finishes.

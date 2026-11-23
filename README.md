# TPU-RTX

**Zero-Copy RTX-Class Ray Tracing, Multi-Bounce Monte Carlo Path Tracing, 3D/4D Gaussian Splatting, OpenUSD/Blender Studio Compilation, and Photorealistic Physical AI Rendering natively on Google Cloud TPU v5p (JAX + Pallas Mosaic).**

**Author:** Feiyang Liu · **License:** [Apache-2.0](LICENSE) · **Live Interactive Monograph:** [zaychikliu.github.io/tpu-rtx](https://zaychikliu.github.io/tpu-rtx/) ([English](https://zaychikliu.github.io/tpu-rtx/?lang=en) | [中文](https://zaychikliu.github.io/tpu-rtx/?lang=zh))

> Every benchmark, render, and parity check in this repository was executed natively on real Google Cloud **TPU v5p** silicon (`interpret=False`, `95.74 GiB` HBM/chip, `383 GiB`/host single-host to **`3.064 TB` HBM across 8 hosts / 32 chips**) with XLA HLO verified at **`host_callbacks = 0`**, **`send_recv_ops = 0`**, and **`gather_ops = 0`**.

---

## Key Verified Highlights on Live Google Cloud TPU v5p

| Domain | Verified Live TPU v5p Result | Details & Module |
|---|---|---|
| **32-Chip TPU v5p-64 Pod Capstone (8 Hosts, `3.064 TB` HBM)** | **`411,023 FPS`** E2E MJX+RTX (`B=131k`) · **`952,248 FPS`** 3DGS (`128²`) · **`1.062B` Splats** · **`63.88B` LiDAR rays/s** · **`17.18B` Unique Tris** · **`1,650.2 spm` @ 8K** | [`run_pod_32chip_capstone.py`](experiments/pod_v5p64_capstone/run_pod_32chip_capstone.py) (`395.07 s` total, `out/metrics/v3_pod_v5p64_capstone.json`) |
| **Zero-Copy MJX + Fused Pallas Shader (4c)** | **`266,209 FPS`** E2E (`64×64`, `B=8,192`, 4c) · **`921,309 FPS`** render-equiv | `w0` Fused In-VMEM GGX+AO+Shadows (`depth_mae = 3.35e-10 m`) |
| **Native 3D Gaussian Splatting + `300M` Scale** | **`142,008 FPS`** (`4c, 64×64`) · **`300,000,000` Gaussians** in 4c HBM @ 4K (`0.25 FPS`) | `w3` `tpu_gsplat` (`301.87 GB / 380 GB`, `1,171,875:1` cull, `gather_ops = 0`) |
| **Real-World Mip-NeRF 360 3DGS + `4DGS × MJX`** | **Native 8K `garden` (`5.83M` SH3) `0.88 FPS` (`29.19 MPix/s`)** · `4DGS × MJX` **`5.32 FPS` 1080p** | `w3` `32`-view `jax.grad` `23.42 steps/s` (`30.62 dB`, peak `31.31 dB`) |
| **Disney *Moana* Island OpenUSD v2.1 (`22.30B` Tris)** | **4K `5.13 s`** / **8K `17.24 s`** · **Sunset `5.60 s`** · **Underwater Reef Caustics `8.72 s`** | `w2` `openusd_loader` (`4.653 GiB` HBM) + *Moana Beach × MJX* @ 4K (`5.40 s`) |
| **Blender Cinema (`BMW27`, `Barcelona 54.9M`, `Barbershop`, `Monster SSS`, `Lone Monk`, `Fishy Cat`)** | `BMW27` **`37.65 dB`** (`0.9813` SSIM) · `Monster SSS` **`36.98 dB`** (`0.121 mm` Z-MAE) · `Barbershop` (`8.55M`, 141 tex) **`31.00 dB`** | `w5` / `w7` `blend_compiler` Cycles Shader Graph -> Wide-BVH-64 + AgX |
| **Billion-Triangle Unique Geometry @ 4K / 8K** | **`1.020B` Unique Tris** (`61.29 GB` 4c HBM) · **`102.19B` Instanced Tris** @ 8K (`5.02 s`) | `w2` (`4c` dry-run `2.15B` unique tris, `shard_map` + `jax.lax.pmin`) |
| **`TPU-DLSS 3.5` & 4K BF16 MXU Denoiser** | **`58.93 FPS` Native 4K** (`1080p -> 4K`, `3.91×` E2E speedup) · **`87.78 FPS` 4K Denoise** (`11.39 ms`) | `w4` `tpu_dlss` (`4×4` exact bilinear warp, `7.01×` flicker cut, `+15.22 dB`) |
| **Zero-Copy Time-of-Day/Material Relight & MXU Pareto** | **`78.60 FPS` Native 4K BF16** / **`307.44 FPS` 1080p** (0 raytrace ops/slider) | `w4` `FP32 20.28 dB` / `BF16 20.28 dB` (`2.11×`) / `INT8 QAT 20.23 dB` |
| **`TPU-ReSTIR DI`, Thin-Film, Dispersion, Caustics & Bokeh** | **`1,024`-Light ReSTIR** (`21.89 dB`, `7.43×` HDR VarRed) · **4K 32-spp Bokeh `108.7 ms`** (`9.20 FPS`) | `w7` / `w6` 8-bin Airy interference, aniso GGX, 12-bounce mirrors, 6-band prism, 16-step SSS |
| **`MuJoCo Menagerie` & End-to-End Diff Physics+Render** | `Aero Hand` **`4,284 FPS`**, `Panda` **`3,131 FPS`**, `G1` **`2,546 FPS`** · **Pixel->Torque `36.08 dB` (`1.63 cm` EE)** | `w0` `mjx_universal` (`27.87 grad-steps/s`, zero state supervision) |
| **Isaac `Replicator`, `128`-Beam LiDAR, `GelSight` & `Sim-to-VLA`** | `RoboTwin` **`123,858 cam-FPS`** (`+0.32%` DR) · **`1M` Factory Frames `142.6 s`** · **`270.6k` Visuo-Tactile/s** | `w1` / `w6` `isaac_replicator` (`4.59e-8 m` LiDAR parity, `-1.84%` `Sim-to-VLA` overhead) |

---

## Visual Highlights (All Audited via Visual QA Gate)

### Disney *Moana* OpenUSD v2.1 (`22.30B` Tris) — Golden Sunset & Way-of-Water Underwater Coral Reef (4K)
![Disney Moana Sunset and Underwater Reef 4K](docs/assets/web/v3_w2_moana_sunset_reef_4k.webp)

### Real-World Mip-NeRF 360 `garden` Native 8K (`7680×4320`, 1:1 Crop) & Dynamic `4DGS × MJX` RoboWild
![Real-World 8K 3DGS Crop](docs/assets/web/v3_w3_realworld_8k_crop.webp)
![Dynamic 4DGS x MJX Manipulation](docs/assets/web/v3_w3_4dgs_robowild_manipulation.webp)

### Blender Studio Parity (`BMW27`, `Barcelona 54.9M`, `Monster SSS`, `Fishy Cat`, `Barbershop`)
![BMW27 and Barcelona Pavilion](docs/assets/web/v3_w5_bmw_barcelona_cinema.webp)
![Monster SSS and Fishy Cat](docs/assets/web/v3_w5_studio_double_cinema.webp)
![Barbershop Interior](docs/assets/web/v3_w7_barbershop_cinema.webp)

### Physical FX (`Airy Thin-Film`, `12-Bounce Mirrors`, `Spectral Dispersion`, `SSS`, `TPU-ReSTIR DI`, `4K Thin-Lens Bokeh`)
![Thin-Film Anisotropic GGX and 12-Bounce Mirror Gallery](docs/assets/web/v3_w7_thinfilm_mirror_fx.webp)
![Spectral Dispersion and Random-Walk SSS](docs/assets/web/v3_w7_dispersion_sss_fx.webp)
![Native 4K Thin-Lens Bokeh](docs/assets/web/v3_w6_thinlens_bokeh_4k.webp)

### `TPU-DLSS 3.5` (58.9 FPS 4K), Interactive Relighting (`78.6 FPS` 4K BF16), `MuJoCo Menagerie` & Diff Physics-Render Servoing
![Interactive Zero-Copy Relighting and MXU Quantization Pareto](docs/assets/web/v3_w4_interactive_relight_quant.webp)
![MuJoCo Menagerie Wide-BVH-64](docs/assets/web/v3_w0_mujoco_menagerie_pbr.webp)
![End-to-End Differentiable Physics and PBR Servoing](docs/assets/web/v3_w0_diff_phys_render.webp)

---

## Quickstart SDK

```python
import tpu_rtx

scene = tpu_rtx.compile(mj_model, shader="fused_ao_ggx", res=(128, 128), sharding="auto")
obs = tpu_rtx.render(mjx_data, scene)  # -> {"rgb", "depth", "normal", "seg_id", "flow"} in HBM
```

Ecosystem adapters live under `tpu_mjx_render.adapters`:
`mjx_universal`, `isaac_replicator`, `openusd_loader`, `blend_compiler`, `tpu_gsplat`, `tpu_optix`, `tpu_dlss`, `tpu_restir`.

---

## Supported Software Retrofits & Benchmarks

### 8 Drop-In Ecosystem Adapters (`tpu_mjx_render/adapters/`)

| Target Software / Standard | TPU Adapter Module | What Was Rebuilt for Google Cloud TPU |
|---|---|---|
| **MuJoCo / MJX & Menagerie** | [`mjx_universal.py`](tpu_mjx_render/adapters/mjx_universal.py) | Compiles unmodified `MjModel`, MJCF XML & multi-mesh Menagerie assets into Wide-BVH-64 + `jax.grad` differentiable servoing |
| **NVIDIA Isaac Sim & Omniverse Replicator** | [`isaac_replicator.py`](tpu_mjx_render/adapters/isaac_replicator.py) | Drop-in `TiledCamera` (`RGB/Depth/Normal/Seg/Flow`), `rep.randomizer.*` DR, `128×2048` RTX LiDAR & `GelSight` tactile kernels in one XLA graph |
| **Blender 4.x / Cycles `.blend`** | [`blend_compiler.py`](tpu_mjx_render/adapters/blend_compiler.py) | Automated Cycles Shader Graph (`Principled/Glass/Glossy/SSS`) & Depsgraph walker -> TPU ScenePack + AgX |
| **Pixar / Disney OpenUSD** | [`openusd_loader.py`](tpu_mjx_render/adapters/openusd_loader.py) | Native `.usd/.usda/.usdc/.usdz` (`UsdGeom.Mesh`, `PointInstancer`, `BasisCurves`) -> 2-level Wide-TLAS-64 |
| **NVIDIA OptiX Python/JAX SDK** | [`tpu_optix.py`](tpu_mjx_render/adapters/tpu_optix.py) | `OptixDeviceContext`, `accelBuild(GAS/IAS)`, `ShaderBindingTable`, `optixTrace` & dynamic cloth BVH refit |
| **`gsplat` 3DGS / 4DGS** | [`tpu_gsplat.py`](tpu_mjx_render/adapters/tpu_gsplat.py) | Zero-gather chunked-VMEM EWA rasterizer + 32-view `jax.grad` training + SE(3) dynamic 4DGS |
| **NVIDIA DLSS 3.5 & Denoiser** | [`tpu_dlss.py`](tpu_mjx_render/adapters/tpu_dlss.py) | Analytical `4×4` motion vectors + BF16 MXU 2× SR, 4K MC denoiser & static G-buffer relighting |
| **SIGGRAPH ReSTIR DI** | [`tpu_restir.py`](tpu_mjx_render/adapters/tpu_restir.py) | Streaming `M=32` RIS + spatiotemporal reservoir resampling over 1,024 dynamic lights |

### Benchmark Coverage (Verified on Live TPU vs. Drop-In API Compatible)

- **Verified Live on TPU v5p**: **`RoboTwin`** (`123,858 cam-FPS` 3-cam DR), **`RoboWild` 3DGS/4DGS × MJX** (`5.32 FPS` 1080p live LEGO Dozer push in Mip-NeRF `kitchen`), **`MuJoCo Menagerie`** (`Aero Hand 4,284`, `Panda 3,131`, `G1 2,546`, `Go2 1,863 FPS`), **`Factory Sim-to-VLA + GelSight`** (`1M` frames in `142.6 s`, `270.6k` tactile/s), **`Vision PPO`** (`Cartpole 656k/s`, `Panda 211k/s`, `Humanoid 292k/s`, 32c E2E `411k FPS`), **`Autonomous 8×1080p + 2×128×2048 LiDAR`** (`66.8 cam-FPS`, 32c batch LiDAR `63.88B rays/s`), **`Blender Open Data / Studio` (8 scenes)**, **`Disney Moana OpenUSD` (`22.30B` tris)**, and **`Mip-NeRF 360` + `10M Bistro`**.
- **Drop-In API Compatible (Standard MJCF / OpenUSD / `rep.randomizer`)**: Suites exported as standard MJCF XML or OpenUSD — including **`RoboDojo`**, **`MuJoCo Playground`**, **`ALOHA`**, and **`ManiSkill`** assets — compile cleanly via `tpu_rtx.compile(...)` (dedicated per-suite telemetry logs are not yet recorded in `out/metrics/`).

---

## Repository Layout

- `tpu_rtx/` & `tpu_mjx_render/`: Core Pallas Mosaic kernels (`host_callbacks = 0`, `gather_ops = 0`), Wide-BVH-64, 3DGS/4DGS rasterizers, thin-film/dispersion/volumetric FX, and ecosystem adapters
- `experiments/`: Live TPU v5p experimental suites (`w0`–`w7`) + `pod_v5p64_capstone/` (4-chip verified dry-run + multi-host Pod scaling harness)
- `docs/`: Interactive bilingual (EN / 中文) dark & light showcase (`docs/index.html`) + WebP & full-res verified renders (`docs/assets/`)
- `out/metrics/` & `out/figures/`: Verified TPU telemetry JSONs and rendered figures

---

## Third-Party Asset Acknowledgments

Benchmark renders showcase the engine across standard public test scenes under their respective licenses:
- **Walt Disney Animation Studios *Moana Island Scene* (v2.1)**: Used under the Disney *Moana Island Scene* non-commercial license; original geometry and OpenUSD scene © Walt Disney Animation Studios.
- **Blender Open Data / Blender Studio Scenes** (*BMW27*, *Barcelona Pavilion*, *Barbershop*, *Classroom*, *Junkshop*, *Lone Monk*, *Monster Under the Bed*, *Fishy Cat*): CC0 / CC-BY via Blender Foundation & Blender Studio.
- **Mip-NeRF 360** (`garden`, `bicycle`, `kitchen`, `bonsai`): Research photogrammetry captures by Barron et al.
- **Google DeepMind `mujoco_menagerie`**: Apache-2.0 / BSD robot descriptions (`Franka Panda`, `Unitree G1`, `Unitree Go2`, `Tetheria Aero Hand`).

---

## Citation

```bibtex
@misc{liu2026tpurtx,
  author       = {Feiyang Liu},
  title        = {TPU-RTX: Zero-Copy RTX-Class Ray Tracing, Path Tracing, 3DGS, OpenUSD/Blender Compilation, and Physical AI Rendering on Google Cloud TPU v5p},
  year         = {2026},
  howpublished = {\url{https://github.com/ZaychikLiu/tpu-rtx}},
  note         = {Interactive Monograph: \url{https://zaychikliu.github.io/tpu-rtx/}}
}
```

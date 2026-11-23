# Experiments Directory

Each subdirectory under `experiments/` holds self-contained Google Cloud TPU v5p benchmark harnesses, verification scripts, and video capture workers. Directory names (`w0`–`w7`) are preserved verbatim for cross-module imports (`from experiments.w* import ...`).

## 1. 32-Chip Pod Capstone & Video Capture

| Directory | Focus | Primary Metrics |
|---|---|---|
| [`pod_v5p64_capstone/`](pod_v5p64_capstone/) | 8-host / 32-chip TPU v5p-64 Pod (`3.064 TB` HBM): 411k FPS MJX+RTX, 952k FPS 3DGS (`1.062B` splats), `63.88B` LiDAR rays/s, `17.18B` unique tris, 8K 3-bounce path tracing (`395.07 s`) | [`v3_pod_v5p64_capstone.json`](../out/metrics/v3_pod_v5p64_capstone.json) |
| [`showcase_videos/`](showcase_videos/) | 8-host parallel renderer & MP4 capture harness producing the 17 project-page showcase clips (`4K` / `1080p`, `30 fps`) | `docs/assets/videos/*.mp4` |

## 2. Embodied Robotics, Vision RL, Tactile & LiDAR

| Directory | Focus | Primary Metrics |
|---|---|---|
| [`w0_fused_shader/`](w0_fused_shader/) | Fused in-VMEM GGX+AO+shadow shader (`266,209 FPS` 4c, `921,309 FPS` render-equiv) | [`v3_w0_wave2_100k_fps.json`](../out/metrics/v3_w0_wave2_100k_fps.json) |
| [`w0_wave4_rl_gallery/`](w0_wave4_rl_gallery/) | Closed-loop Vision PPO in HBM (`Cartpole`, `PandaPickCube`, `Humanoid-21DOF`) & `tpu_rtx` SDK | [`v3_w0_wave4_rl_sdk.json`](../out/metrics/v3_w0_wave4_rl_sdk.json) |
| [`w0_wave5_menagerie/`](w0_wave5_menagerie/) | Unmodified DeepMind `mujoco_menagerie` PBR + pixel-to-torque `jax.grad` visual servoing (`36.08 dB`, `1.63 cm` EE error) | [`v3_w0_taskA.json`](../out/metrics/v3_w0_taskA.json), [`v3_w0_taskB.json`](../out/metrics/v3_w0_taskB.json) |
| [`w1_tile_cull/`](w1_tile_cull/) | Two-level frustum tile culling & `isaac_replicator` 5-modality DR on `RoboTwin` (`123,858 cam-FPS`) | [`v3_w1_tile_cull.json`](../out/metrics/v3_w1_tile_cull.json), [`v3_w1_wave3_replicator.json`](../out/metrics/v3_w1_wave3_replicator.json) |
| [`w1_wave4_vla_tactile/`](w1_wave4_vla_tactile/) | `1M`-frame VLA data factory (`142.6 s`), fused `ViT-Tiny` online training (`7,143 FPS`), 4-sensor `GelSight` (`270,625/s`) | [`v3_w1_wave4_vla_tactile_factory.json`](../out/metrics/v3_w1_wave4_vla_tactile_factory.json) |
| [`w6_highres_multicam/`](w6_highres_multicam/) | `4K` / `8K` multi-camera rendering, `RoboTwin` bimanual tabletop, & `128×2048` RTX LiDAR parity (`3.13e-8 m` XYZ MAE) | [`v3_w6_robotwin_bimanual.json`](../out/metrics/v3_w6_robotwin_bimanual.json), [`v3_w6_wave4_isaac_sensors.json`](../out/metrics/v3_w6_wave4_isaac_sensors.json) |
| [`w6_wave5_surround_bokeh/`](w6_wave5_surround_bokeh/) | `8×1080p + Dual 128-Beam LiDAR` autonomous rig (`66.8 cam-FPS`) & `4K` stochastic thin-lens `f/1.4` bokeh (`108.7 ms`) | [`v3_w6_wave5_surround_bokeh.json`](../out/metrics/v3_w6_wave5_surround_bokeh.json), [`v3_w6_thinlens_bokeh_4k.json`](../out/metrics/v3_w6_thinlens_bokeh_4k.json) |
| [`w7_ppo_4c/`](w7_ppo_4c/) | 4-chip distributed `shard_map` Vision PPO scaling (`292k env/s` Humanoid) | [`v3_w7_ppo_4c.json`](../out/metrics/v3_w7_ppo_4c.json) |

## 3. OpenUSD (*Moana*), Billion-Triangle BVH & Deformable Cloth

| Directory | Focus | Primary Metrics |
|---|---|---|
| [`w2_mesh_1m_bvh/`](w2_mesh_1m_bvh/) | 3-level 64-ary `Wide-BVH-64`, `1.02B` unique / `102.2B` instanced tris (`4c`), Disney *Moana* OpenUSD v2.1 (`22.30B` tris in `4.65 GiB`, `4K`/`8K`/Sunset/Reef/`Moana × MJX`), & `PBD` cloth bottom-up BVH refit (`973.3 FPS`) | [`v3_w2_wave5_moana_8k_mjx.json`](../out/metrics/v3_w2_wave5_moana_8k_mjx.json), [`v3_w2_wave6_moana_sunset_reef.json`](../out/metrics/v3_w2_wave6_moana_sunset_reef.json), [`v3_w2_deformable_cloth.json`](../out/metrics/v3_w2_deformable_cloth.json) |

## 4. 3DGS / 4DGS & Real-World Photogrammetry

| Directory | Focus | Primary Metrics |
|---|---|---|
| [`w3_gs3d_v3/`](w3_gs3d_v3/) | Zero-gather chunked-VMEM EWA `tpu_gsplat` (`142,008 FPS` on 4c, `30.1×` over V2 `jax_tile`) | [`v3_w3_gs3d_v3.json`](../out/metrics/v3_w3_gs3d_v3.json) |
| [`w3_realworld_3dgs/`](w3_realworld_3dgs/) | Genuine `Mip-NeRF 360` (`garden`, `bicycle`, `kitchen`, `bonsai`) `.ply` scans at `4K` (`2.59–4.20 FPS`) | [`v3_w3_wave5_realworld_3dgs_100m.json`](../out/metrics/v3_w3_wave5_realworld_3dgs_100m.json) |
| [`w3_4dgs_robowild/`](w3_4dgs_robowild/) | Single-host `300,000,000` Gaussians (`301.87 GB` HBM @ `4K`) & `4DGS × MJX` Panda pushing LEGO Dozer (`+16.43 cm`) | [`v3_w3_wave6_4dgs_300m.json`](../out/metrics/v3_w3_wave6_4dgs_300m.json) |
| [`w3_bistro_humanoid_v4/`](w3_bistro_humanoid_v4/) | `10M`-Gaussian Amazon Bistro Z-composited with live MJX Humanoid & Mobile Panda (`4K` `2.28 FPS`) | [`v3_w3_wave4_bistro_humanoid.json`](../out/metrics/v3_w3_wave4_bistro_humanoid.json) |
| [`w3_8k_and_fastgrad/`](w3_8k_and_fastgrad/) | `Mip-NeRF 360 garden` `8K` (`29.19 MPix/s`) & 32-view `jax.grad` 3DGS training (`31.31 dB` peak in `17.1 s`) | [`v3_w3_wave7_8k_and_gradtrain.json`](../out/metrics/v3_w3_wave7_8k_and_gradtrain.json) |

## 5. Neural Denoising, TPU-DLSS 3.5 & MXU Relighting

| Directory | Focus | Primary Metrics |
|---|---|---|
| [`w4_neural_pareto/`](w4_neural_pareto/) | `4K` BF16 MXU Monte Carlo denoiser (`87.78 FPS`, `+15.22 dB` from `1-spp`) & Stage-2 shader Pareto | [`v3_w4_wave3_mc_denoiser_4k.json`](../out/metrics/v3_w4_wave3_mc_denoiser_4k.json), [`v3_w4_neural_pareto.json`](../out/metrics/v3_w4_neural_pareto.json) |
| [`w4_tpu_dlss/`](w4_tpu_dlss/) | `TPU-DLSS 3.5`: `4×4` motion-vector reprojection + YCoCg clamp + `2×` MXU SR (`1080p → 4K` at `58.93 FPS`) | [`v3_w4_wave4_tpu_dlss_4k.json`](../out/metrics/v3_w4_wave4_tpu_dlss_4k.json) |
| [`w4_wave5_relight_quant/`](w4_wave5_relight_quant/) | Static G-buffer zero-raytrace relighting & `FP32` / `BF16` / `INT8 W8A8` Pareto (`78.60 FPS` 4K / `307.4 FPS` 1080p) | [`v3_w4_wave5_relight_quant.json`](../out/metrics/v3_w4_wave5_relight_quant.json) |

## 6. Blender Open Data & Studio Cycles Parity

| Directory | Focus | Primary Metrics |
|---|---|---|
| [`w5_blender_1m_showcase/`](w5_blender_1m_showcase/) | Automated `.blend` Cycles shader compiler (`Classroom`, `Junkshop`, `Lone Monk`, `rtx-h100` comparison) | [`v3_w5_lone_monk_cinema.json`](../out/metrics/v3_w5_lone_monk_cinema.json), [`v3_w5_opendata_vs_rtxh100.json`](../out/metrics/v3_w5_opendata_vs_rtxh100.json) |
| [`w5_bmw_barcelona/`](w5_bmw_barcelona/) | `BMW27` clearcoat (`37.65 dB`, `0.9813` SSIM) & `Barcelona Pavilion` (`54.91M` unique tris, `15.94 mm` Z-MAE) | [`v3_w5_wave5_bmw_barcelona.json`](../out/metrics/v3_w5_wave5_bmw_barcelona.json) |
| [`w5_studio_double/`](w5_studio_double/) | `Monster Under the Bed` SSS (`36.98 dB`, `0.121 mm` Z-MAE) & `Fishy Cat` (`7,255` fur instances, `28.80 dB`) | [`v3_w5_wave6_studio_double.json`](../out/metrics/v3_w5_wave6_studio_double.json) |

## 7. Path Tracing, ReSTIR DI & Wave Optics

| Directory | Focus | Primary Metrics |
|---|---|---|
| [`w7_pathtracer_1b/`](w7_pathtracer_1b/) | Multi-bounce Monte Carlo GI (`8K` 3-bounce), `Barbershop` (`8.55M` tris, `141` textures, `31.00 dB`), `tpu_optix` SDK, & `1,024`-light `TPU-ReSTIR DI` (`21.89 dB`, `7.43×` variance cut) | [`v3_w7_barbershop_cinema.json`](../out/metrics/v3_w7_barbershop_cinema.json), [`v3_w7_wave5_restir_cinema.json`](../out/metrics/v3_w7_wave5_restir_cinema.json) |
| [`w7_dispersion_sss/`](w7_dispersion_sss/) | `6`-band spectral Cauchy prism dispersion & `16`-step random-walk volumetric jade SSS | [`v3_w7_wave6_dispersion_sss.json`](../out/metrics/v3_w7_wave6_dispersion_sss.json) |
| [`w7_thinfilm_mirror/`](w7_thinfilm_mirror/) | `8`-bin Airy thin-film interference, radial anisotropic GGX, & `12`-bounce mirror `while_loop` recursion | [`v3_w7_wave7_thinfilm_mirror.json`](../out/metrics/v3_w7_wave7_thinfilm_mirror.json) |

# TPU-MJX Batch Visual Rendering Engine (`tpu-mjx-renderer`)

**TPU-native, zero-copy JAX/Pallas batch raytracer + BF16 MXU neural relighter consuming `mjx.Data` directly in TPU HBM (`MJX step -> Render -> Vision RL`) on live Google Cloud TPU v5p.**

---

## 1. Executive Summary

- **Zero-Copy Closed Loop Verified in XLA HLO**: Both `mjx.step + Stage 1` (`8,021` / `21,404` HLO lines) and fused `mjx.step + Stage 1 + Stage 2 M2_spatial_128` (`22,965` HLO lines on 1 chip; `23,033` lines on 4 chips; `16` BF16 `conv_ops`, `4` `while_ops`) compile with **`0` `host_callback`**, **`0` `send_recv`**, and **`0` `gather_ops`**.
- **Stage 1 Analytic Primitives Exact vs Official MuJoCo C (`mj_multiRay`)**: Branchless analytic ray-primitive intersection (plane, sphere, capsule, ellipsoid, cylinder, box) passes all locked Phase 0 parity thresholds across all 3 suites (`all_primitives` `Depth MAE = 7.53e-7 m`, `Cartpole` `7.40e-7 m`, `PandaPickCube` `1.964e-4 m`, `GeomID match = 99.952%–100.000%`).
- **Stage 2 Photorealistic Relighting vs Blender 5.2 Cycles GT Passes 8/8 Locked Quality Gates**: `M2_spatial_128` (MXU-aligned BF16 2-scale spatial U-Net with 24-ch BRDF inputs including `reflect(V, N)` & Fresnel `N·V`) reaches **`38.35 dB` PSNR (`+24.52 dB` over Stage 1)**, **`0.9889` SSIM**, **`0.9916` MS-SSIM**, **`+0.032%` luminance error**, **`30.31 dB` contact ROI PSNR**, **`0.157 px` edge alignment**, and **`0.00714` (`1/140` px) boundary occlusion mismatch** (`passed: true`).
- **Live TPU v5p End-to-End Throughput (`MJX Step + Render`)**:
  - `Cartpole` Stage 1 (`64x64`, 1 chip): **`2,166.4 FPS`** (`B=1024`), `2,127.6 FPS` (`B=2048`), `1,670.3 FPS` (`B=4096`).
  - `PandaPickCube` Stage 1 + Passing `M2_spatial_128` (`64x64`): **`714.2 FPS`** (`B=1024` on 1 chip) -> **`3,502.6 FPS`** (`B=1024` on 4 chips, `292.35 ms/step`, `4.90x` scaling) and **`2,851.7 FPS`** at `B=4096` (`4` chips).
  - `PandaPickCube` Stage 1 + Passing `M2_spatial_128` (`128x128`, `B=1024`): **`189.3 FPS`** (1 chip) -> **`881.8 FPS`** (4 chips, `4.66x` scaling).
- **Phase 3 Zero-Copy Vision PPO on Live TPU v5p (`B=512`)**: `CartpoleBalance` runs at **`1,264.9 env_steps/s`** (`276.9 s` for 40 iters); `PandaPickCube + M2_spatial_128` runs at **`377.3 env_steps/s`** (`428.9 s` for 30 iters, `mean_reward` `2.237 -> 2.489`).

---

## 2. System Architecture & Key Modules

1. **[tpu_mjx_render/primitives.py](file:///Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer/tpu_mjx_render/primitives.py)** & **[camera.py](file:///Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer/tpu_mjx_render/camera.py)**: Branchless JAX implementations of MuJoCo pinhole camera projection and analytic ray intersections for all 6 primitive geom types (`plane`, `sphere`, `capsule`, `ellipsoid`, `cylinder`, `box`), ported cleanly from `mujoco_warp/_src/ray.py` formulas.
2. **[tpu_mjx_render/renderer.py](file:///Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer/tpu_mjx_render/renderer.py)**: `RenderSceneSpec` pre-packs static scene metadata from `MjModel` once at setup while `render_batch(spec, mjx_data_batch, mode="tiled")` reads `geom_xpos`, `geom_xmat`, `cam_xpos`, `cam_xmat`, `xpos`, and `xmat` directly from live batched `mjx.Data` in HBM. Supports `naive`, `tiled` (ray-tile `jax.lax.scan`), and `pallas` modes, outputting full 20+ channel `GBuffer` (Direct RGB, Albedo, Euclidean & Z Depth, Normal, View Dir, World Pos, GeomID, Hard Shadow).
3. **[tpu_mjx_render/pallas_render.py](file:///Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer/tpu_mjx_render/pallas_render.py)**: Native TPU Pallas (`Mosaic`) kernels (`pallas_raycast_scene` and `pallas_mxu_pixel_mlp`) keeping ray/geom/activation tiles inside TPU VMEM (`128`-aligned MXU tiles) to minimize HBM spills.
4. **[tpu_mjx_render/neural_shader.py](file:///Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer/tpu_mjx_render/neural_shader.py)**: `build_gbuffer_features(gb)` packs `24` normalized physically-based channels (`rgb`, `albedo`, `normal`, `view_dir`, `pos_norm`, `z_norm`, `direct_shadow`, `hit`, `g_edge`, `z_edge`, `reflect(V, N)`, `clamp(-V·N)`). `SpatialRelighter` (`base_width=64 -> 128 -> 64`) computes multi-scale BF16 `3x3` residual convolutions for soft shadows & indirect GI, concatenated with an un-blurred `24`-ch `1x1` pixel-BRDF refinement head and calibrated via a scalar `cal_scale` measured strictly on the 32-frame `calib` split.
5. **[tpu_mjx_render/envs.py](file:///Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer/tpu_mjx_render/envs.py)** & **[scripts/run_vision_ppo.py](file:///Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer/scripts/run_vision_ppo.py)**: Pure-JAX pixel environments (`CartpoleBalanceVisionEnv`, `PandaPickCubeVisionEnv`) and single-JIT `jax.lax.scan` PPO rollout + GAE + Flax CNN actor-critic update.
6. **[gate.py](file:///Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer/gate.py)** & **[bench.py](file:///Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer/bench.py)**: Locked Phase 0 parity/quality gate evaluator and JIT-warmed multi-chip E2E benchmark harness.

---

## 3. Stage 1 Exact Track Verification (Parity, Pallas, & Tiling)

### 3.1 Parity vs Official MuJoCo C (`mujoco.mj_multiRay`, [stage1_parity.json](file:///Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer/out/metrics/stage1_parity.json))

| Scene | Poses | Depth MAE (Limit) | Normal Err (Limit) | GeomID Match (Limit) | Gate |
|---|---:|---:|---:|---:|---|
| `all_primitives` | 4 | `7.53e-7 m` (`<= 1e-4`) | `0.0014°` (`<= 0.5°`) | `100.000%` (`>= 99.8%`) | **PASS** |
| `cartpole` | 64 | `7.40e-7 m` (`<= 1e-4`) | `0.0004°` (`<= 0.5°`) | `99.9996%` (`>= 99.8%`) | **PASS** |
| `panda_pick_cube` | 64 | `1.964e-4 m` (`<= 2e-4`) | `0.0382°` (`<= 0.5°`) | `99.952%` (`>= 99.8%`) | **PASS** |

- **Pallas Kernel Verification ([pallas_kernel_parity.json](file:///Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer/out/metrics/pallas_kernel_parity.json) & Live TPU Mosaic)**:
  - CPU `interpret=True`: `pallas_vs_tiled` max depth diff `0.0 m` (`cartpole`) / `1.788e-6 m` (`panda`), `geom_match = 1.0`, `pallas_mxu_pixel_mlp_max_diff = 0.0`.
  - Live `TPU v5p` Mosaic (`interpret=False`): `pallas_mxu_pixel_mlp` max diff `0.0`; `pallas_raycast_scene` Cartpole depth diff `0.03647 m` (BF16/VPU), `geom_match = 1.0`.

![Stage 1 Exact G-Buffer](file:///Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer/out/figures/stage1_gbuffer.png)

### 3.2 Stage 1 VMEM Tiling Optimization (`PandaPickCube`, `B=1024, 64x64`, 1 TPU v5p Chip)

- `naive` (`[4096, 32]` full tensor): `2258.89 ms` E2E (**`453.3 FPS`**)
- `tiled (tile_size=128)`: `1254.67 ms` E2E (**`816.1 FPS`**, **`1.80x`** over `naive`)
- `tile_size` VMEM sweep ([live_tpu_tile_sweep_1c.json](file:///Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer/out/metrics/live_tpu_tile_sweep_1c.json)):

| `tile_size` | E2E Latency (ms) | E2E FPS | Behavior |
|---:|---:|---:|---|
| `64` | **`1198.60`** | **`854.3`** | Tightest VMEM residency (**+5.3%** over `128`) |
| `128` | `1262.62` | `811.0` | Default production tile size |
| `256` | `1281.16` | `799.3` | Slight VMEM pressure increase |
| `512` | `1660.41` | `616.7` | **VMEM spill cliff (`-24%` throughput)** |

---

## 4. Stage 2 Neural Relighting vs Blender 5.2 Cycles GT

Dataset: `640` `PandaPickCube` states (`64x64`), rendered in local **Blender 5.2.0 LTS Cycles (`48 spp`, `max_bounces=6`, `0.07 s/frame`)** after verifying Frame-0 Z-depth alignment (`z_depth_mae = 0.00000 m`, `fg_mask_match = 1.0000`):
- `544 train` (`split=0`, including `256` extra tight-grasp poses giving `2,510` gripper–cube boundary pixels)
- `32 calib` (`split=1`, held-out luminance calibration)
- `64 test_gate` (`split=2`, strictly frozen held-out test set verified bit-identical by `np.array_equal`)

### Held-Out Quality Gate Results ([stage2_summary.json](file:///Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer/out/metrics/stage2_summary.json))

| Model / Run | PSNR (dB) | Gain | SSIM | MS-SSIM | Lum Rel Err | ROI PSNR | ROI Edge | Occ Mismatch | Gate Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Locked Threshold** | `>= 30.0` | `>= +4.0` | `>= 0.94` | `>= 0.96` | `<= 0.50%` | `>= 28.0` | `<= 1.00 px` | `<= 0.010` | **All 8 Required** |
| Stage 1 Direct Exact | `13.83` | `+0.00` | — | — | — | — | — | — | Baseline |
| `M1_pixel_128` (150 ep) | `27.80` | `+13.97` | `0.9329` | `0.9366` | `-0.110%` | `23.52` | `0.374 px` | `0.0500` (`7/140`) | **FAIL** |
| `M2_spatial_32` (150 ep) | `35.58` | `+21.74` | `0.9784` | `0.9839` | `-0.049%` | `27.79` | `0.252 px` | `0.1000` (`14/140`) | **FAIL** |
| **`M2_spatial_128` (400 ep)** | **`38.35`** | **`+24.52`** | **`0.9889`** | **`0.9916`** | **`+0.032%`** | **`30.31`** | **`0.157 px`** | **`0.00714` (`1/140`)** | **PASS (8/8)** |

![Stage 2 Cycles Comparison](file:///Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer/out/figures/stage2_cycles_comparison.png)

---

## 5. Live Google TPU v5p End-to-End Benchmarks & Vision PPO

### 5.1 End-to-End Throughput (`MJX Step + Render`, JIT-Warmed, Zero `host_callback`)

| Scene | Config | Chips | Res | `B=1024` FPS (ms) | `B=2048` FPS (ms) | `B=4096` FPS (ms) | Source |
|---|---|---:|---:|---:|---:|---:|---|
| `Cartpole` | Stage 1 Tiled | 1 | `64x64` | **2,166.4** (`472.7`) | **2,127.6** (`962.6`) | **1,670.3** (`2452.3`) | [live_tpu_cartpole_1c.json](file:///Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer/out/metrics/live_tpu_cartpole_1c.json) |
| `PandaPickCube` | Stage 1 Naive | 1 | `64x64` | **453.3** (`2258.9`) | — | — | [live_tpu_panda_naive_1c.json](file:///Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer/out/metrics/live_tpu_panda_naive_1c.json) |
| `PandaPickCube` | Stage 1 Tiled | 1 | `64x64` | **816.1** (`1254.7`) | **811.8** (`2522.9`) | **621.1** (`6594.4`) | [live_tpu_panda_tiled_1c.json](file:///Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer/out/metrics/live_tpu_panda_tiled_1c.json) |
| `PandaPickCube` | Stage 1 + `M2_spatial_128` | 1 | `64x64` | **714.2** (`1433.8`) | **710.4** (`2883.0`) | **641.0** (`6390.2`) | [live_tpu_panda_m2_1c.json](file:///Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer/out/metrics/live_tpu_panda_m2_1c.json) |
| `PandaPickCube` | Stage 1 + `M2_spatial_128` | **4** | `64x64` | **3,502.6** (`292.4`) | **3,224.2** (`635.2`) | **2,851.7** (`1436.3`) | [live_tpu_panda_m2_4c.json](file:///Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer/out/metrics/live_tpu_panda_m2_4c.json) |
| `PandaPickCube` | Stage 1 + `M2_spatial_128` | 1 | `128x128` | **189.3** (`5408.2`) | — | — | [live_tpu_panda_128_1c.json](file:///Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer/out/metrics/live_tpu_panda_128_1c.json) |
| `PandaPickCube` | Stage 1 + `M2_spatial_128` | **4** | `128x128` | **881.8** (`1161.3`) | — | — | [live_tpu_panda_128_4c.json](file:///Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer/out/metrics/live_tpu_panda_128_4c.json) |

### 5.2 Closed-Loop Vision PPO on Live TPU v5p (1 Chip, `B=512`, `64x64`)

- **`CartpoleBalance` Stage 1 Vision PPO** ([ppo_cartpole.json](file:///Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer/out/metrics/ppo_cartpole.json)): `rollout_len=16`, `40` iters -> **`1,264.85 env_steps/s`**, `276.87 s` total, `hlo_zero_copy=true`, `mean_reward` `0.849–0.991`.
- **`PandaPickCube + M2_spatial_128` Vision PPO** ([ppo_panda.json](file:///Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer/out/metrics/ppo_panda.json)): `rollout_len=8`, `30` iters -> **`377.33 env_steps/s`**, `428.94 s` total, `hlo_zero_copy=true`, `mean_reward` `2.237 -> 2.489`.

![Quality-Throughput Pareto and Vision PPO](file:///Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer/out/figures/pareto_and_ppo.png)

---

## 6. Reproducibility Commands

```bash
cd /Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer

# Safety & dry-run check of the TPU benchmark harness
bash scripts/run_live_tpu_bench.sh --dry-run

# Local figure generation
python3 scripts/make_figures.py --out_dir out/figures

# On TPU v5p w0 (PYTHONPATH=/tmp/zaychik-tpu-mjx-pkgs:/tmp/zaychik-tpu-mjx):
python tests/test_primitives_parity.py
python tests/test_pallas_kernel.py
python tests/test_zero_copy_hlo.py
python tests/test_stage2_hlo.py
python scripts/train_neural_shader.py --dataset out/data/panda_cycles_dataset.npz --gt_rgb out/data/panda_cycles_gt_rgb.npz --archs M1_pixel_128 M2_spatial_32 M2_spatial_128 --epochs 400 --batch_size 32
python bench.py --scene panda_pick_cube --batches 1024 2048 4096 --res 64 --mode tiled --neural_ckpt out/metrics/M2_spatial_128_params.pkl --n_chips 4 --steps 15 --out_json out/metrics/live_tpu_panda_m2_4c.json
python scripts/run_vision_ppo.py --env panda_pick_cube --num_envs 512 --rollout_len 8 --num_iters 30 --res 64 --neural_ckpt out/metrics/M2_spatial_128_params.pkl --out_json out/metrics/ppo_panda.json
```

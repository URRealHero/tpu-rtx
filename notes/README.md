# Historical Engineering Notes (V1 & V2)

This directory archives the milestone reports and experiment ledgers from early V1 and V2 development on Google Cloud TPU v5p:

- **[`v1_report.md`](v1_report.md)** — V1 exact analytic ray-primitive parity against `mujoco.mj_multiRay`, VMEM tiling sweep (`tile_size = 64..512`), Stage-2 neural shader ablations (`M1_pixel_128`, `M2_spatial_32`, `M2_spatial_128`), and initial Vision PPO results.
- **[`v1_v2_experiments.md`](v1_v2_experiments.md)** — Chronological E0–E16 experiment log tracking Stage-1/Stage-2 parity, Pallas V2 SoA kernel speedups, negative runs, and the checkerboard gather bottleneck fix.
- **[`v2_architecture.md`](v2_architecture.md)** — V2 Pallas Mosaic kernel design (`pallas_v2.py`), static Cluster-BVH (`mesh_ray.py`), rigid-body 3DGS (`gs3d.py`), and multi-chip `shard_map` throughput matrices.

For V3 capabilities (OpenUSD *Moana*, Blender Cycles `.blend` compilation, 3DGS/4DGS, TPU-DLSS 3.5, ReSTIR DI, and the 32-chip pod capstone), see the root **[`README.md`](../README.md)**, **[`experiments/README.md`](../experiments/README.md)**, and the metric logs under **[`out/metrics/`](../out/metrics/)**.

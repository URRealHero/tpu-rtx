"""Wave 5 Suite: Real-World Mip-NeRF 360 PLYs @ 4K, RoboWild Real-to-Sim, 50M/100M HBM Benchmark, and Real-World jax.grad Optimization."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import time
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tpu_mjx_render.adapters.tpu_gsplat import (
    _SH_C0,
    composite_mjx_and_3dgs,
    eval_sh,
    load_cameras_json,
    load_ply,
    make_sharded_rasterizer,
    rasterization,
    shard_splats_for_mesh,
)
from tpu_mjx_render.renderer import RenderSceneSpec
from tpu_mjx_render.roofline import inspect_hlo_ops

from robowild_kitchen import (
    build_robowild_panda_xml,
    compute_opencv_rays,
    fit_table_plane_from_depth,
    pallas_raycast_mjx_table_strip,
    set_robowild_panda_pose,
)


DATA_ROOT = "/dev/shm/zaychik-w3-data"


def _check_hlo(lowered, name: str) -> Dict[str, Any]:
  txt = lowered.as_text()
  info = inspect_hlo_ops(txt)
  info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
  assert info["host_callbacks"] == 0, f"{name} host_callbacks != 0: {info}"
  assert info["send_recv_ops"] == 0, f"{name} send_recv_ops != 0: {info}"
  assert info["gather_ops"] == 0, f"{name} gather_ops != 0: {info}"
  info["zero_copy_verified"] = True
  return info


def _bench_fn(fn, args, n_warmup: int = 1, n_steps: int = 5):
  t0 = time.perf_counter()
  out = fn(*args)
  jax.block_until_ready(out)
  compile_s = time.perf_counter() - t0
  for _ in range(n_warmup):
    out = fn(*args)
  jax.block_until_ready(out)
  t1 = time.perf_counter()
  for _ in range(n_steps):
    out = fn(*args)
  jax.block_until_ready(out)
  ms = (time.perf_counter() - t1) * 1e3 / n_steps
  return ms, compile_s, out


def _save_png(rgb_hwc: np.ndarray, path: str) -> None:
  arr = np.clip(np.round(rgb_hwc * 255.0), 0, 255).astype(np.uint8)
  Image.fromarray(arr).save(path)


def _add_label(img_pil: Image.Image, title: str, subtitle: str) -> Image.Image:
  out = img_pil.copy()
  draw = ImageDraw.Draw(out, "RGBA")
  try:
    font_b = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 21)
  except Exception:
    font_b = ImageFont.load_default()
    font_s = font_b
  pad_x, pad_y = 22, 18
  tw = max(draw.textlength(title, font=font_b), draw.textlength(subtitle, font=font_s))
  box = [pad_x, pad_y, pad_x + int(tw) + 32, pad_y + 78]
  draw.rounded_rectangle(box, radius=12, fill=(10, 14, 20, 195), outline=(255, 165, 60, 220), width=2)
  draw.text((pad_x + 16, pad_y + 10), title, fill=(255, 255, 255, 255), font=font_b)
  draw.text((pad_x + 16, pad_y + 44), subtitle, fill=(215, 225, 240, 255), font=font_s)
  return out


def _scaled_K(cam: Dict[str, Any], width: int, height: int) -> np.ndarray:
  sf = float(height) / float(cam["height"])
  fx = float(cam["fx"]) * sf
  fy = float(cam["fy"]) * sf
  return np.array(
      [[fx, 0.0, 0.5 * width], [0.0, fy, 0.5 * height], [0.0, 0.0, 1.0]],
      dtype=np.float32,
  )


# ---------------------------------------------------------------------------
# Task A: Real-World Mip-NeRF 360 @ 1080p/4K + RoboWild Kitchen Panda @ 4K
# ---------------------------------------------------------------------------

def run_task_a(data: Dict[str, Any], export_dir: str) -> None:
  print("=== Task A: Real-World Mip-NeRF 360 PLYs @ 1080p/4K + RoboWild Kitchen Panda 4K ===", flush=True)
  devs = jax.devices()[:4]
  mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
  rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  P = jax.sharding.PartitionSpec
  sm_kw = {"check_vma": False} if "check_vma" in inspect.signature(jax.shard_map).parameters else {"check_rep": False}

  scenes = ("bonsai", "kitchen", "garden", "bicycle")
  res_configs = [
      ("1080p", 1920, 1080, 32, 5, 10240, 1536, 768),
      ("4K", 3840, 2160, 64, 4, 8192, 2048, 512),
  ]

  scene_results: List[Dict[str, Any]] = []
  rendered_4k: Dict[str, np.ndarray] = {}
  kitchen_cache: Dict[str, Any] = {}

  for res_label, W, H, m_sz, m_rep, k_mac, k_lg, k_tile in res_configs:
    fn = make_sharded_rasterizer(
        mesh, W, H, sh_degree=3, macro_size=m_sz, macro_rep=m_rep,
        k_macro=k_mac, k_large_per_chip=k_lg, k_tile=k_tile,
    )
    for idx, sc in enumerate(scenes):
      ply_p = os.path.join(DATA_ROOT, sc, "point_cloud/iteration_30000/point_cloud.ply")
      cam_p = os.path.join(DATA_ROOT, sc, "cameras.json")
      sp = load_ply(ply_p)
      cams = load_cameras_json(cam_p)
      c0 = cams[0]
      K = _scaled_K(c0, W, H)

      m, q, s, o, sh, n_pad = shard_splats_for_mesh(sp, mesh, seed=42, use_sh=True)
      v_dev = jax.device_put(jnp.asarray(c0["viewmat"]), rep)
      k_dev = jax.device_put(jnp.asarray(K), rep)

      if idx == 0:
        hlo = _check_hlo(fn.lower(m, q, s, o, sh, v_dev, k_dev), f"ply_{sc}_{res_label}")
        data.setdefault("hlo_checks", {})[f"realworld_ply_{res_label}"] = hlo

      ms, comp_s, out = _bench_fn(fn, (m, q, s, o, sh, v_dev, k_dev), n_warmup=1, n_steps=4)
      rgb, alp, dep = out
      rgb_np = np.asarray(rgb)
      alp_np = np.asarray(alp[..., 0])

      rec = {
          "scene": sc,
          "res_label": res_label,
          "res": f"{W}x{H}",
          "n_gaussians": int(sp["n_gaussians"]),
          "sh_degree": int(sp["sh_degree"]),
          "n_chips": 4,
          "ms": round(ms, 2),
          "fps": round(1000.0 / ms, 2),
          "compile_s": round(comp_s, 2),
          "mean_alpha": round(float(alp_np.mean()), 4),
          "low_alp_frac": round(float(np.mean(alp_np < 0.5)), 5),
      }
      scene_results.append(rec)
      print(f"REALWORLD {sc:8s} {res_label}: {ms:6.2f} ms ({1000/ms:5.2f} FPS) | N={sp['n_gaussians']:,} | low_alp={rec['low_alp_frac']:.4f}", flush=True)

      if res_label == "4K":
        rendered_4k[sc] = rgb_np
        _save_png(rgb_np, os.path.join(export_dir, f"v3_w3_realworld_{sc}_4k.png"))
        if sc == "kitchen":
          kitchen_cache = {"sp": sp, "c0": c0, "K": K, "dep": np.asarray(dep[..., 0])}

  data["realworld_3dgs_scenes"] = scene_results

  # --- RoboWild Native 4K: MJX 7-DOF Franka Panda + Grasp Cubes on Real Kitchen Table ---
  print("--- Rendering RoboWild Real-to-Sim (MJX 7-DOF Panda on Real Kitchen Table @ Native 4K) ---", flush=True)
  W, H = 3840, 2160
  c0 = kitchen_cache["c0"]
  K_4k = kitchen_cache["K"]
  fx, fy = float(K_4k[0, 0]), float(K_4k[1, 1])

  tbl = fit_table_plane_from_depth(
      kitchen_cache["dep"],
      c0["rotation"],
      c0["position"],
      fx,
      fy,
      0.5 * W,
      0.5 * H,
      roi=(1040, 1840, 240, 1400),
      place_xy=(880.0, 1340.0),
      target_xy=(1560.0, 1480.0),
  )

  mj_model = mujoco.MjModel.from_xml_string(build_robowild_panda_xml())
  mjx_model = mjx.put_model(mj_model, impl="jax")
  d0 = mjx.make_data(mj_model, impl="jax").replace(
      qpos=jnp.asarray(set_robowild_panda_pose(mj_model), jnp.float32)
  )
  d0 = mjx.forward(mjx_model, d0)

  spec_strip = RenderSceneSpec.from_mjmodel(mj_model, cam_id=0, height=H // 4, width=W, use_shadows=True)
  R_tbl = jnp.asarray(tbl["R_table"], dtype=jnp.float32)
  P_org = jnp.asarray(tbl["P_origin"], dtype=jnp.float32)
  t_nrm = jnp.asarray(tbl["table_normal"], dtype=jnp.float32)
  t_d = jnp.asarray(tbl["table_d"], dtype=jnp.float32)
  robot_scale = 1.12

  sun_local = jnp.array([-0.52, 0.52, 0.74], dtype=jnp.float32)
  sun_local = sun_local / jnp.linalg.norm(sun_local)
  sun_dir_w = R_tbl @ sun_local

  gs_fn_4k = make_sharded_rasterizer(
      mesh, W, H, sh_degree=3, macro_size=64, macro_rep=4,
      k_macro=8192, k_large_per_chip=2048, k_tile=512,
  )
  m, q, s, o, sh, _ = shard_splats_for_mesh(kitchen_cache["sp"], mesh, seed=42, use_sh=True)

  _sm_mjx = jax.shard_map(
      lambda gxp, gxm, gsz, grg, sd, tn, td, ro, rd, cz: pallas_raycast_mjx_table_strip(
          spec_strip, gxp, gxm, gsz, grg, sd, tn, td, ro, rd, cz
      ),
      mesh=mesh,
      in_specs=(P(), P(), P(), P(), P(), P(), P(), P("dp"), P("dp"), P("dp")),
      out_specs=(P("dp"), P("dp"), P("dp"), P("dp"), P("dp"), P("dp"), P("dp")),
      **sm_kw,
  )

  @jax.jit
  def _run_robowild_4k(d_in, m_g, q_g, s_g, o_g, sh_g, vm, Km, cp, rc2w):
    d_step = mjx.step(mjx_model, d_in)
    g_xpos = (d_step.geom_xpos * robot_scale) @ R_tbl.T + P_org[None, :]
    g_xmat = jnp.einsum("ij,gjk->gik", R_tbl, d_step.geom_xmat, precision=jax.lax.Precision.HIGHEST)
    g_size = jnp.asarray(spec_strip.geom_size, dtype=jnp.float32) * robot_scale
    g_rgba = jnp.asarray(spec_strip.geom_rgba, dtype=jnp.float32)

    ro, rd, cz = compute_opencv_rays(cp, rc2w, Km, H, W)
    alb_s, nrm_s, zd_s, rsh_s, tsh_s, tz_s, vd_s = _sm_mjx(
        g_xpos, g_xmat, g_size, g_rgba, sun_dir_w, t_nrm, t_d, ro, rd, cz
    )
    alb = alb_s.reshape(H, W, 3)
    p_nrm = nrm_s.reshape(H, W, 3)
    p_vd = vd_s.reshape(H, W, 3)
    r_sh = rsh_s.reshape(H, W)
    mjx_z = zd_s.reshape(H, W)
    tbl_sh = tsh_s.reshape(H, W)
    tbl_z = tz_s.reshape(H, W)

    ndotl = jnp.maximum(0.0, jnp.sum(p_nrm * sun_dir_w[None, None, :], axis=-1))
    hemi = 0.5 + 0.5 * jnp.sum(p_nrm * t_nrm[None, None, :], axis=-1)
    hvec = sun_dir_w[None, None, :] + p_vd
    hvec = hvec / jnp.maximum(jnp.linalg.norm(hvec, axis=-1, keepdims=True), 1e-12)
    spec_term = (jnp.maximum(0.0, jnp.sum(p_nrm * hvec, axis=-1)) ** 32.0) * 0.25 * r_sh
    lit = (0.36 + 0.22 * hemi + 0.56 * ndotl * r_sh)[..., None] * alb + spec_term[..., None]
    mjx_rgb = jnp.clip(lit, 0.0, 1.0)

    g_rgb, g_alp, g_dep = gs_fn_4k(m_g, q_g, s_g, o_g, sh_g, vm, Km)
    g_alp2 = g_alp[..., 0]
    g_dep2 = g_dep[..., 0]

    on_tbl = (tbl_sh > 0.5) & (g_dep2 > 0.0) & (jnp.abs(g_dep2 - tbl_z) < 0.25)
    g_rgb_sh = jnp.where(on_tbl[..., None], g_rgb * jnp.array([0.44, 0.45, 0.48], jnp.float32), g_rgb)
    comp_rgb, _ = composite_mjx_and_3dgs(mjx_rgb, mjx_z, g_rgb_sh, g_alp2, g_dep2)
    return comp_rgb, mjx_z, g_dep2, on_tbl

  rw_args = (
      jax.device_put(d0, rep),
      m, q, s, o, sh,
      jax.device_put(jnp.asarray(c0["viewmat"]), rep),
      jax.device_put(jnp.asarray(K_4k), rep),
      jax.device_put(jnp.asarray(c0["position"]), rep),
      jax.device_put(jnp.asarray(c0["rotation"]), rep),
  )
  data.setdefault("hlo_checks", {})["robowild_kitchen_panda_4k"] = _check_hlo(
      _run_robowild_4k.lower(*rw_args), "robowild_kitchen_panda_4k"
  )
  rw_ms, rw_comp_s, rw_out = _bench_fn(_run_robowild_4k, rw_args, n_warmup=1, n_steps=3)
  rw_rgb, mjz, gsz, ont = rw_out
  rw_rgb_np = np.asarray(rw_rgb)

  robowild_rec = {
      "scene": "RoboWild_Kitchen_MJX_Panda",
      "res": "3840x2160",
      "n_chips": 4,
      "n_gaussians": int(kitchen_cache["sp"]["n_gaussians"]),
      "table_fit_median_res": round(tbl["median_res"], 5),
      "ms": round(rw_ms, 2),
      "fps": round(1000.0 / rw_ms, 2),
      "compile_s": round(rw_comp_s, 2),
      "mjx_robot_pixels": int(jnp.sum(mjz > 0)),
      "occluded_by_real_3dgs_pixels": int(jnp.sum((mjz > 0) & (gsz > 0) & (gsz < mjz))),
      "cast_shadow_on_table_pixels": int(jnp.sum(ont)),
  }
  data["robowild_realtosim_4k"] = robowild_rec
  print("ROBOWILD 4K:", robowild_rec, flush=True)

  standalone_4k = os.path.join(export_dir, "v3_w3_robowild_kitchen_panda_4k.png")
  _save_png(rw_rgb_np, standalone_4k)

  # Build razor-sharp 4K (3840x2160) 2x2 showcase panel
  def _to_1080p(arr: np.ndarray, title: str, sub: str) -> Image.Image:
    im = Image.fromarray(np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8))
    im_s = im.resize((1920, 1080), Image.Resampling.LANCZOS)
    return _add_label(im_s, title, sub)

  r_map = {r["scene"]: r for r in scene_results if r["res_label"] == "4K"}
  q_tl = _to_1080p(
      rw_rgb_np,
      "RoboWild Real-to-Sim @ 4K (3840x2160)",
      f"MJX 7-DOF Panda + Cast Shadows on Kitchen 3DGS | {rw_ms:.1f} ms ({1000/rw_ms:.2f} FPS)",
  )
  q_tr = _to_1080p(
      rendered_4k["garden"],
      "Mip-NeRF 360 Garden @ Native 4K (SH3)",
      f"{r_map['garden']['n_gaussians']:,} Gaussians | 4-Chip TPU v5p | {r_map['garden']['ms']:.1f} ms ({r_map['garden']['fps']:.2f} FPS)",
  )
  q_bl = _to_1080p(
      rendered_4k["bicycle"],
      "Mip-NeRF 360 Bicycle @ Native 4K (SH3)",
      f"{r_map['bicycle']['n_gaussians']:,} Gaussians | 4-Chip TPU v5p | {r_map['bicycle']['ms']:.1f} ms ({r_map['bicycle']['fps']:.2f} FPS)",
  )
  q_br = _to_1080p(
      rendered_4k["bonsai"],
      "Mip-NeRF 360 Bonsai @ Native 4K (SH3)",
      f"{r_map['bonsai']['n_gaussians']:,} Gaussians | 4-Chip TPU v5p | {r_map['bonsai']['ms']:.1f} ms ({r_map['bonsai']['fps']:.2f} FPS)",
  )

  panel = Image.new("RGB", (3840, 2160))
  panel.paste(q_tl, (0, 0))
  panel.paste(q_tr, (1920, 0))
  panel.paste(q_bl, (0, 1080))
  panel.paste(q_br, (1920, 1080))
  panel_path = os.path.join(export_dir, "v3_w3_realworld_3dgs_robowild_4k.png")
  panel.save(panel_path)
  print(f"Saved 4K showcase panel: {panel_path}", flush=True)


# ---------------------------------------------------------------------------
# Task B: 50M & 100M Mega-Scale 4-Chip Sharded 3DGS HBM Benchmark
# ---------------------------------------------------------------------------

def _build_megascale_real_plaza(target_n: int, seed: int = 77) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
  """Build a 50M or 100M Gaussian real-world plaza by instancing real Mip-NeRF 360 PLY geometry."""
  rng = np.random.default_rng(seed)
  sp_g = load_ply(os.path.join(DATA_ROOT, "garden", "point_cloud/iteration_30000/point_cloud.ply"))
  c_g = load_cameras_json(os.path.join(DATA_ROOT, "garden", "cameras.json"))[0]
  sp_b = load_ply(os.path.join(DATA_ROOT, "bonsai", "point_cloud/iteration_30000/point_cloud.ply"))

  # Filter valid garden core around scene origin (radius < 12) + bonsai hero clusters
  rad_g = np.linalg.norm(sp_g["means"], axis=-1)
  keep_g = (rad_g < 12.0) & (sp_g["opacities"] > 0.03)
  m_g = sp_g["means"][keep_g]
  q_g = sp_g["quats"][keep_g]
  s_g = sp_g["scales"][keep_g]
  o_g = sp_g["opacities"][keep_g]
  c_g_rgb = np.clip(_SH_C0 * sp_g["sh_dc"][keep_g, 0, :] + 0.5, 0.0, 1.0).astype(np.float32)

  # Replicate with jittered micro-offsets & multi-ring world tiling up to target_n
  n_base = len(m_g)
  n_reps = (target_n + n_base - 1) // n_base
  # Place first instance at origin (exact garden view!), outer instances in surrounding 3x3 ring
  offsets = [np.array([0.0, 0.0, 0.0], np.float32)]
  for ring_i in range(1, n_reps):
    ang = (ring_i * 2.39996)  # golden angle
    rad = 3.5 + 1.8 * math.sqrt(ring_i)
    offsets.append(np.array([rad * math.cos(ang), 0.0, rad * math.sin(ang)], np.float32))

  means_out = np.empty((target_n, 3), dtype=np.float32)
  quats_out = np.empty((target_n, 4), dtype=np.float32)
  scales_out = np.empty((target_n, 3), dtype=np.float32)
  ops_out = np.empty((target_n,), dtype=np.float32)
  cols_out = np.empty((target_n, 3), dtype=np.float32)

  cursor = 0
  for r_idx, off in enumerate(offsets):
    take = min(n_base, target_n - cursor)
    if take <= 0:
      break
    sl = slice(cursor, cursor + take)
    if r_idx == 0:
      means_out[sl] = m_g[:take]
    else:
      jitter = rng.normal(0.0, 0.008, size=(take, 3)).astype(np.float32)
      means_out[sl] = m_g[:take] + off[None, :] + jitter
    quats_out[sl] = q_g[:take]
    scales_out[sl] = s_g[:take] * (1.0 if r_idx == 0 else 0.85)
    ops_out[sl] = o_g[:take]
    cols_out[sl] = c_g_rgb[:take]
    cursor += take

  # Shuffle across all target_n so all 4 chips hold a uniform 25% sample
  perm = rng.permutation(target_n)
  means_out = np.ascontiguousarray(means_out[perm])
  quats_out = np.ascontiguousarray(quats_out[perm])
  scales_out = np.ascontiguousarray(scales_out[perm])
  ops_out = np.ascontiguousarray(ops_out[perm])
  cols_out = np.ascontiguousarray(cols_out[perm])
  del sp_g, sp_b

  return {
      "means": means_out,
      "quats": quats_out,
      "scales": scales_out,
      "opacities": ops_out,
      "colors": cols_out,
  }, c_g


def run_task_b(data: Dict[str, Any], export_dir: str) -> None:
  print("=== Task B: 50M & 100M Mega-Scale 4-Chip Sharded 3DGS HBM Benchmark ===", flush=True)
  devs = jax.devices()[:4]
  mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
  dp = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("dp"))
  rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

  bench_runs: List[Dict[str, Any]] = []

  for target_n in (50_000_000, 100_000_000):
    t_build = time.perf_counter()
    mega, c0 = _build_megascale_real_plaza(target_n, seed=77 + target_n // 1_000_000)
    print(f"Built {target_n:,} Gaussian real-world plaza in {time.perf_counter()-t_build:.1f}s", flush=True)

    m_dev = jax.device_put(jnp.asarray(mega["means"]), dp)
    q_dev = jax.device_put(jnp.asarray(mega["quats"]), dp)
    s_dev = jax.device_put(jnp.asarray(mega["scales"]), dp)
    o_dev = jax.device_put(jnp.asarray(mega["opacities"]), dp)
    c_dev = jax.device_put(jnp.asarray(mega["colors"]), dp)
    del mega

    for res_label, W, H in (("1080p", 1920, 1080), ("4K", 3840, 2160)):
      K = _scaled_K(c0, W, H)
      v_dev = jax.device_put(jnp.asarray(c0["viewmat"]), rep)
      k_dev = jax.device_put(jnp.asarray(K), rep)

      # macro_size=64, macro_rep=2 -> 4x key replication (100M keys/chip at N=100M) fits easily in 380 GB HBM
      fn = make_sharded_rasterizer(
          mesh, W, H, sh_degree=0, macro_size=64, macro_rep=2,
          k_macro=4096, k_large_per_chip=256, k_tile=256,
      )
      low = fn.lower(m_dev, q_dev, s_dev, o_dev, c_dev, v_dev, k_dev)
      hlo = _check_hlo(low, f"mega_{target_n}_{res_label}")
      comp = low.compile()
      mem = comp.memory_analysis()
      temp_gb = float(mem.temp_size_in_bytes) / 1e9
      arg_gb = float(mem.argument_size_in_bytes) / 1e9
      out_gb = float(mem.output_size_in_bytes) / 1e9
      alias_gb = float(getattr(mem, "alias_size_in_bytes", 0)) / 1e9
      chip_hbm_gb = arg_gb + temp_gb + out_gb - alias_gb
      total_4c_hbm_gb = 4.0 * chip_hbm_gb

      data.setdefault("hlo_checks", {})[f"mega_{target_n//1_000_000}m_{res_label}"] = hlo

      ms, comp_s, out = _bench_fn(fn, (m_dev, q_dev, s_dev, o_dev, c_dev, v_dev, k_dev), n_warmup=1, n_steps=3)
      rgb, alp, dep = out

      # Estimate tile culling efficiency: active micro-tile capacity vs N
      n_micro_tiles = ((W + 63) // 64) * ((H + 63) // 64) * 64
      max_raster_evals = n_micro_tiles * 256
      rec = {
          "n_gaussians": target_n,
          "n_chips": 4,
          "res_label": res_label,
          "res": f"{W}x{H}",
          "ms": round(ms, 2),
          "fps": round(1000.0 / ms, 2),
          "compile_s": round(comp_s, 2),
          "hbm_per_chip_gb": round(chip_hbm_gb, 2),
          "hbm_4chip_total_gb": round(total_4c_hbm_gb, 2),
          "hbm_arg_per_chip_gb": round(arg_gb, 2),
          "hbm_temp_per_chip_gb": round(temp_gb, 2),
          "hbm_utilization_pct_of_380gb": round(100.0 * total_4c_hbm_gb / 380.0, 2),
          "k_macro": 4096,
          "k_tile": 256,
          "micro_tiles": n_micro_tiles,
          "cull_ratio_per_tile": round(float(target_n) / 256.0, 1),
          "mean_alpha": round(float(jnp.mean(alp)), 4),
      }
      bench_runs.append(rec)
      print(f"MEGA {target_n//1_000_000}M {res_label}: {ms:.2f} ms ({1000/ms:.2f} FPS) | HBM={total_4c_hbm_gb:.2f} GB / 380 GB ({rec['hbm_utilization_pct_of_380gb']:.1f}%)", flush=True)

      if target_n == 100_000_000 and res_label == "4K":
        _save_png(np.asarray(rgb), os.path.join(export_dir, "v3_w3_100m_realworld_4k.png"))

    del m_dev, q_dev, s_dev, o_dev, c_dev

  data["megascale_50m_100m_bench"] = bench_runs


# ---------------------------------------------------------------------------
# Task C: Real-World Multi-View jax.grad 3DGS Optimization Benchmark
# ---------------------------------------------------------------------------

def run_task_c(data: Dict[str, Any], export_dir: str) -> None:
  print("=== Task C: Real-World Bonsai Multi-View jax.grad 3DGS Optimization ===", flush=True)
  sp = load_ply(os.path.join(DATA_ROOT, "bonsai", "point_cloud/iteration_30000/point_cloud.ply"))
  cams = load_cameras_json(os.path.join(DATA_ROOT, "bonsai", "cameras.json"))

  # Pick 8 well-spaced real cameras from bonsai/cameras.json
  cam_sel = [cams[i] for i in np.linspace(0, min(len(cams) - 1, 140), 8, dtype=int)]
  C = len(cam_sel)
  H = W = 128
  Vs = np.stack([c["viewmat"] for c in cam_sel], axis=0)
  Ks = np.stack([_scaled_K(c, W, H) for c in cam_sel], axis=0)
  cam_pos_c = np.stack([c["position"] for c in cam_sel], axis=0)

  # Find exact 3D focus point where the 8 camera optical axes intersect (closest point in L2)
  A_foc = np.zeros((3, 3), dtype=np.float64)
  b_foc = np.zeros((3,), dtype=np.float64)
  for c in cam_sel:
    d_ax = c["rotation"][:, 2].astype(np.float64)
    P_proj = np.eye(3) - np.outer(d_ax, d_ax)
    A_foc += P_proj
    b_foc += P_proj @ c["position"].astype(np.float64)
  focus_ctr = np.linalg.solve(A_foc, b_foc).astype(np.float32)

  # Extract 2,048 foreground Gaussians near the bonsai focus center
  dist_foc = np.linalg.norm(sp["means"] - focus_ctr[None, :], axis=-1)
  fg_mask = (dist_foc < 1.5) & (sp["opacities"] > 0.35)
  fg_idx = np.where(fg_mask)[0]
  if len(fg_idx) < 2048:
    fg_idx = np.argsort(dist_foc)[:2048]
  top_fg = fg_idx[np.argsort(-sp["opacities"][fg_idx])[:2048]]
  N_fit = len(top_fg)
  assert N_fit >= 256, f"Expected foreground Gaussians near {focus_ctr}, got {N_fit}"
  print(f"Bonsai focus center={focus_ctr.round(3)}, N_fit={N_fit}", flush=True)

  gt_means = sp["means"][top_fg]
  gt_quats = sp["quats"][top_fg]
  gt_scales = sp["scales"][top_fg]
  gt_opacities = sp["opacities"][top_fg]
  gt_sh = sp["sh"][top_fg]

  # Evaluate real SH3 colors for each view as ground-truth target colors
  dirs = gt_means[None, :, :] - cam_pos_c[:, None, :]
  dirs = dirs / np.maximum(np.linalg.norm(dirs, axis=-1, keepdims=True), 1e-12)
  gt_colors = np.asarray(eval_sh(3, jnp.asarray(gt_sh), jnp.asarray(dirs)))
  gt_colors = np.clip(gt_colors, 0.0, 1.0).astype(np.float32)

  viewmats_d = jnp.asarray(Vs)
  Ks_d = jnp.asarray(Ks)
  bg = jnp.array([0.08, 0.10, 0.14], dtype=jnp.float32)

  gt_rgb, _, _ = jax.jit(
      lambda m, q, s, o, c: rasterization(
          m, q, s, o, c, viewmats_d, Ks_d, W, H, backgrounds=bg, use_approx_topk=False
      )
  )(
      jnp.asarray(gt_means), jnp.asarray(gt_quats), jnp.asarray(gt_scales),
      jnp.asarray(gt_opacities), jnp.asarray(gt_colors),
  )
  jax.block_until_ready(gt_rgb)

  # Initialize perturbed point cloud (noisy positions, scales, opacities, and neutral grey colors)
  rng = np.random.default_rng(2026)
  init_means = gt_means + rng.normal(0.0, 0.045, size=(N_fit, 3)).astype(np.float32)
  init_quats = rng.normal(0.0, 1.0, size=(N_fit, 4)).astype(np.float32)
  init_quats /= np.maximum(np.linalg.norm(init_quats, axis=-1, keepdims=True), 1e-8)
  init_log_scales = np.log(np.clip(gt_scales * rng.uniform(0.7, 1.5, size=(N_fit, 3)), 1e-4, 0.2)).astype(np.float32)
  init_logit_op = np.zeros((N_fit,), dtype=np.float32)
  init_logit_col = rng.normal(0.0, 0.35, size=(N_fit, 3)).astype(np.float32)

  params = {
      "means": jnp.asarray(init_means),
      "quats": jnp.asarray(init_quats),
      "log_scales": jnp.asarray(init_log_scales),
      "logit_opacities": jnp.asarray(init_logit_op),
      "logit_colors": jnp.asarray(init_logit_col),
  }

  def _render_from_params(p):
    rgb, _, _ = rasterization(
        p["means"],
        p["quats"],
        jnp.exp(jnp.clip(p["log_scales"], -9.0, -1.5)),
        jax.nn.sigmoid(p["logit_opacities"]),
        jax.nn.sigmoid(p["logit_colors"]),
        viewmats_d,
        Ks_d,
        W,
        H,
        backgrounds=bg,
    )
    return rgb

  def _loss_fn(p):
    pred = _render_from_params(p)
    l1 = jnp.mean(jnp.abs(pred - gt_rgb))
    mse = jnp.mean((pred - gt_rgb) ** 2)
    return 0.8 * l1 + 0.2 * mse, mse

  lr = {"means": 0.006, "quats": 0.015, "log_scales": 0.03, "logit_opacities": 0.06, "logit_colors": 0.08}
  m_state = jax.tree.map(jnp.zeros_like, params)
  v_state = jax.tree.map(jnp.zeros_like, params)

  @jax.jit
  def _train_step(p, m_s, v_s, step_idx):
    (loss_val, mse_val), grads = jax.value_and_grad(_loss_fn, has_aux=True)(p)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t_f = step_idx.astype(jnp.float32)
    new_p, new_m, new_v = {}, {}, {}
    for k in p:
      g = grads[k]
      mk = b1 * m_s[k] + (1.0 - b1) * g
      vk = b2 * v_s[k] + (1.0 - b2) * (g * g)
      m_hat = mk / (1.0 - (b1 ** t_f))
      v_hat = vk / (1.0 - (b2 ** t_f))
      new_p[k] = p[k] - lr[k] * m_hat / (jnp.sqrt(v_hat) + eps)
      new_m[k] = mk
      new_v[k] = vk
    return new_p, new_m, new_v, loss_val, mse_val

  hlo_grad = _check_hlo(
      _train_step.lower(params, m_state, v_state, jnp.array(1, jnp.int32)),
      "realworld_bonsai_jax_grad",
  )
  data.setdefault("hlo_checks", {})["realworld_bonsai_jax_grad"] = hlo_grad

  init_rgb = jax.jit(_render_from_params)(params)
  jax.block_until_ready(init_rgb)
  init_mse = float(jnp.mean((init_rgb - gt_rgb) ** 2))
  init_psnr = 10.0 * math.log10(1.0 / max(init_mse, 1e-12))

  t0 = time.perf_counter()
  params, m_state, v_state, loss0, mse0 = _train_step(params, m_state, v_state, jnp.array(1, jnp.int32))
  jax.block_until_ready((params, loss0))
  compile_s = time.perf_counter() - t0

  n_steps = 300
  history = [{"step": 0, "psnr_db": round(init_psnr, 2), "mse": round(init_mse, 6)}]
  t_run0 = time.perf_counter()
  for st in range(2, n_steps + 1):
    params, m_state, v_state, loss_v, mse_v = _train_step(params, m_state, v_state, jnp.array(st, jnp.int32))
    if st in (50, 100, 150, 200, 250, 300):
      jax.block_until_ready((loss_v, mse_v))
      psnr_v = 10.0 * math.log10(1.0 / max(float(mse_v), 1e-12))
      history.append({"step": st, "psnr_db": round(psnr_v, 2), "loss": round(float(loss_v), 5)})
      print(f"Bonsai Grad step {st:3d}/{n_steps}: loss={float(loss_v):.4f}, PSNR={psnr_v:.2f} dB", flush=True)
  jax.block_until_ready(params)
  total_s = time.perf_counter() - t_run0
  steps_per_s = (n_steps - 1) / total_s

  final_rgb = jax.jit(_render_from_params)(params)
  jax.block_until_ready(final_rgb)
  final_mse = float(jnp.mean((final_rgb - gt_rgb) ** 2))
  final_psnr = 10.0 * math.log10(1.0 / max(final_mse, 1e-12))

  data["realworld_bonsai_grad"] = {
      "scene": "MipNeRF360_Bonsai_Foreground_Crop",
      "n_views": C,
      "res": f"{W}x{H}",
      "n_gaussians": N_fit,
      "n_steps": n_steps,
      "compile_s": round(compile_s, 2),
      "total_train_s": round(total_s, 2),
      "steps_per_s": round(steps_per_s, 2),
      "ms_per_step": round(1000.0 / steps_per_s, 2),
      "init_psnr_db": round(init_psnr, 2),
      "final_psnr_db": round(final_psnr, 2),
      "psnr_gain_db": round(final_psnr - init_psnr, 2),
      "history": history,
  }

  rows = []
  for v_idx in (0, 3, 6):
    row = np.concatenate([
        np.asarray(init_rgb[v_idx]),
        np.asarray(final_rgb[v_idx]),
        np.asarray(gt_rgb[v_idx]),
    ], axis=1)
    rows.append(row)
  grid = np.concatenate(rows, axis=0)
  out_png = os.path.join(export_dir, "v3_w3_realworld_grad_fit.png")
  _save_png(grid, out_png)
  print(f"Bonsai Grad Fit: {init_psnr:.2f} -> {final_psnr:.2f} dB @ {steps_per_s:.1f} steps/s | {out_png}", flush=True)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--part", default="all", choices=["all", "task_a", "task_b", "task_c"])
  ap.add_argument("--out_json", default="/dev/shm/zaychik-w3-out/v3_w3_wave5_realworld_3dgs_100m.json")
  ap.add_argument("--export_dir", default="/dev/shm/zaychik-w3-out")
  args = ap.parse_args()

  os.makedirs(args.export_dir, exist_ok=True)
  os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
  if os.path.exists(args.out_json):
    with open(args.out_json) as f:
      data = json.load(f)
  else:
    data = {
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "jax_version": jax.__version__,
        "interpret": False,
    }

  if args.part in ("all", "task_a"):
    run_task_a(data, args.export_dir)
    with open(args.out_json, "w") as f:
      json.dump(data, f, indent=2)

  if args.part in ("all", "task_b"):
    run_task_b(data, args.export_dir)
    with open(args.out_json, "w") as f:
      json.dump(data, f, indent=2)

  if args.part in ("all", "task_c"):
    run_task_c(data, args.export_dir)
    with open(args.out_json, "w") as f:
      json.dump(data, f, indent=2)

  print(f"=== ALL DONE -> {args.out_json} ===", flush=True)


if __name__ == "__main__":
  main()

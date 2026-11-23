"""Zero-Copy Visuo-Tactile Pallas Mosaic Kernel (Head + Wrist + Dual GelSight Fingertips).

Fuses 4 heterogeneous sensors into ONE single `pl.pallas_call` Mosaic kernel:
  - Slot 0: `head_camera` (Perspective GGX PBR + AgX + Soft Shadows)
  - Slot 1: `wrist_camera` (Perspective GGX PBR + AgX + Soft Shadows)
  - Slot 2: `left_tactile` (Orthographic GelSight elastomer penetration depth + contact normals + RGB tricolor grazing LED photometric stereo)
  - Slot 3: `right_tactile` (Orthographic GelSight elastomer penetration depth + contact normals + RGB tricolor grazing LED photometric stereo)

Invariants: `host_callbacks == 0`, `gather_ops == 0`, `send_recv_ops == 0`,
and exact elastomer penetration depth parity (`< 1e-5 m`) vs pure-JAX orthographic raycast.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
try:
  from jax import shard_map
except ImportError:
  from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from tpu_mjx_render.adapters.isaac_replicator import (
    ReplicatorBatch,
    ReplicatorCfg,
    TEX_CHECKER,
    TEX_SOLID,
    _agx_tonemap_rgb_components,
    _apply_replicator_local,
    _build_unit_tangent_rays,
    _eval_procedural_tex_vmem,
    _shard_map_kwargs,
)
from tpu_mjx_render.pallas_v2 import (
    _compute_geom_bound_sq,
    _extract_light_batch,
    _ray_one_geom_v2,
    build_local_camera_rays,
)
from tpu_mjx_render.primitives import GEOM_PLANE, MJ_MAXVAL, RAY_FN_BY_TYPE
from tpu_mjx_render.renderer import RenderSceneSpec


def _gelsight_tricolor_led_rgb(
    nx: jax.Array,
    ny: jax.Array,
    nz: jax.Array,
    in_contact: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Classic GelSight tricolor R/G/B grazing LED photometric stereo shading (120-deg ring)."""
  # 3 grazing LEDs at 0, 120, 240 deg azimuth and ~18 deg elevation
  c_e = 0.9510565
  s_e = 0.3090170
  l_rx, l_ry, l_rz = c_e, 0.0, s_e
  l_gx, l_gy, l_gz = -0.5 * c_e, 0.8660254 * c_e, s_e
  l_bx, l_by, l_bz = -0.5 * c_e, -0.8660254 * c_e, s_e

  # Grazing directional Lambertian + focused grazing highlight + subtle specular
  dr = jnp.maximum(0.0, nx * l_rx + ny * l_ry + nz * l_rz)
  dg = jnp.maximum(0.0, nx * l_gx + ny * l_gy + nz * l_gz)
  db = jnp.maximum(0.0, nx * l_bx + ny * l_by + nz * l_bz)

  # View is +Z in pad-local frame -> half vector H_k = normalize(L_k + (0,0,1))
  def _spec(lx, ly, lz):
    hz = lz + 1.0
    inv = 1.0 / jnp.sqrt(lx * lx + ly * ly + hz * hz + 1e-12)
    ndoth = jnp.maximum(0.0, (nx * lx + ny * ly + nz * hz) * inv)
    p4 = ndoth * ndoth * ndoth * ndoth
    return p4 * p4 * p4

  sr = _spec(l_rx, l_ry, l_rz)
  sg = _spec(l_gx, l_gy, l_gz)
  sb = _spec(l_bx, l_by, l_bz)

  # Undeformed silicone membrane has neutral gray-slate baseline (~0.36)
  base_amb = 0.14
  gain = 0.72
  spec_g = 0.28
  r = jnp.clip(base_amb + gain * dr + spec_g * sr, 0.0, 1.0)
  g = jnp.clip(base_amb + gain * dg + spec_g * sg, 0.0, 1.0)
  b = jnp.clip(base_amb + gain * db + spec_g * sb, 0.0, 1.0)

  # Subtle contact-region gain boost so indentations pop crisply against membrane
  r = jnp.where(in_contact, jnp.clip(r * 1.08, 0.0, 1.0), r)
  g = jnp.where(in_contact, jnp.clip(g * 1.08, 0.0, 1.0), g)
  b = jnp.where(in_contact, jnp.clip(b * 1.08, 0.0, 1.0), b)
  return r, g, b


def _apply_elastomer_membrane_skirt_rgb(
    pen_depth: jax.Array,  # (B, 2, H, W)
    tactile_n: jax.Array,  # (B, 2, H, W, 3)
) -> jax.Array:
  """Zero-gather 3x3 elastic Poisson skirt gradient + tricolor R/G/B photometric shading."""
  # Smooth penetration height field twice (zero-gather 3x3 box blur) to model silicone membrane fillet
  def _blur3x3(x: jax.Array) -> jax.Array:
    p = jnp.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
    return (
        p[:, :, 0:-2, 0:-2] + p[:, :, 0:-2, 1:-1] + p[:, :, 0:-2, 2:]
        + p[:, :, 1:-1, 0:-2] + p[:, :, 1:-1, 1:-1] + p[:, :, 1:-1, 2:]
        + p[:, :, 2:, 0:-2] + p[:, :, 2:, 1:-1] + p[:, :, 2:, 2:]
    ) * (1.0 / 9.0)

  h_s = _blur3x3(_blur3x3(pen_depth))
  hp = jnp.pad(h_s, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
  # Physical pixel pitch ~ 0.034 / 64 ~ 5.3e-4 m; scale gradient smoothly for realistic elastomer bevel
  pix_pitch = 0.034 / float(max(pen_depth.shape[-1], 1))
  dh_dx = (hp[:, :, 1:-1, 2:] - hp[:, :, 1:-1, 0:-2]) / (2.0 * pix_pitch)
  dh_dy = (hp[:, :, 0:-2, 1:-1] - hp[:, :, 2:, 1:-1]) / (2.0 * pix_pitch)

  # Combine exact geometric surface normal with elastic membrane boundary slope
  in_contact = pen_depth > 1e-6
  has_skirt = h_s > 1e-7
  gx_m = jnp.clip(-dh_dx * 0.85, -1.6, 1.6)
  gy_m = jnp.clip(-dh_dy * 0.85, -1.6, 1.6)
  # Inside contact: blend geometric normal + membrane gradient; on skirt rim: membrane gradient
  nx_b = jnp.where(in_contact, 0.72 * tactile_n[..., 0] + 0.55 * gx_m, gx_m)
  ny_b = jnp.where(in_contact, 0.72 * tactile_n[..., 1] + 0.55 * gy_m, gy_m)
  nz_b = jnp.where(in_contact, jnp.maximum(0.72 * tactile_n[..., 2], 0.35), 1.0)
  inv_n = 1.0 / jnp.sqrt(nx_b * nx_b + ny_b * ny_b + nz_b * nz_b + 1e-12)
  nx_f = jnp.where(has_skirt, nx_b * inv_n, 0.0)
  ny_f = jnp.where(has_skirt, ny_b * inv_n, 0.0)
  nz_f = jnp.where(has_skirt, nz_b * inv_n, 1.0)

  r, g, b = _gelsight_tricolor_led_rgb(nx_f, ny_f, nz_f, has_skirt)
  return jnp.stack([r, g, b], axis=-1)


@dataclasses.dataclass(frozen=True)
class VisuoTactileCfg:
  cam_names: Tuple[str, str, str, str] = ("head_camera", "wrist_camera", "left_tactile", "right_tactile")
  height: int = 64
  width: int = 64
  gel_surface_dist_m: float = 0.0150
  shader: str = "fast_dr"
  tonemap: str = "agx"
  out_dtype: Any = jnp.float32
  block_b: int = 8
  block_r: int = 256
  use_shadows: bool = False
  shadow_light_fraction: float = 0.08
  n_chips: Optional[int] = None
  data_axis: str = "dp"
  interpret: Optional[bool] = None
  apply_membrane_skirt: bool = True


class VisuoTactileSensorRig:
  """Single-Kernel Fused 4-Sensor Visuo-Tactile Rig (Head + Wrist + Dual GelSight Fingertips)."""

  def __init__(self, mjmodel: mujoco.MjModel, cfg: Optional[VisuoTactileCfg] = None):
    self.cfg = cfg or VisuoTactileCfg()
    self.mjmodel = mjmodel
    self.cam_ids = tuple(int(mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_CAMERA, nm)) for nm in self.cfg.cam_names)
    assert all(cid >= 0 for cid in self.cam_ids), f"Missing camera in {self.cfg.cam_names}: {self.cam_ids}"

    self.specs = tuple(
        RenderSceneSpec.from_mjmodel(
            mjmodel,
            cam_id=cid,
            height=int(self.cfg.height),
            width=int(self.cfg.width),
            use_shadows=bool(self.cfg.use_shadows),
            shadow_light_fraction=float(self.cfg.shadow_light_fraction),
            znear=0.001,
        )
        for cid in self.cam_ids
    )
    self.base_fovys = tuple(float(sp.cam_fovy) for sp in self.specs)

    # Identify self-finger geoms to exclude per tactile pad
    def _gids_by_prefix(prefixes: Tuple[str, ...]) -> Tuple[int, ...]:
      out = []
      for g in self.specs[0].enabled_geom_ids:
        nm = mujoco.mj_id2name(mjmodel, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or ""
        if any(nm.startswith(p) for p in prefixes):
          out.append(int(g))
      while len(out) < 4:
        out.append(-999)
      return tuple(out[:4])

    self.left_excl_gids = _gids_by_prefix(("left_finger", "hand_"))
    self.right_excl_gids = _gids_by_prefix(("right_finger", "hand_"))

    # Floor & workbench gids for procedural texture
    self.floor_gids = tuple(
        int(g) for g in self.specs[0].enabled_geom_ids
        if (mujoco.mj_id2name(mjmodel, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "") == "floor"
    )
    self.table_gids = tuple(
        int(g) for g in self.specs[0].enabled_geom_ids
        if (mujoco.mj_id2name(mjmodel, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "") == "workbench_top"
    )

    n_devs = len(jax.devices())
    self.n_chips = min(n_devs, int(self.cfg.n_chips)) if self.cfg.n_chips is not None else 1
    if self.n_chips > 1:
      self.mesh = jax.sharding.Mesh(np.array(jax.devices()[: self.n_chips]), (self.cfg.data_axis,))
    else:
      self.mesh = None

  def shard_tree(self, tree: Any) -> Any:
    if self.mesh is None or self.n_chips <= 1:
      return jax.device_put(tree, jax.devices()[0])
    P = jax.sharding.PartitionSpec
    sharding = jax.sharding.NamedSharding(self.mesh, P(self.cfg.data_axis))
    return jax.device_put(tree, sharding)

  def render_local(
      self,
      mjx_data_batch: Any,
      rep_batch: Optional[ReplicatorBatch] = None,
  ) -> Dict[str, jax.Array]:
    """Render all 4 sensors (head, wrist, left_tactile, right_tactile) in ONE Mosaic Pallas kernel."""
    B = int(mjx_data_batch.cam_xpos.shape[0])
    N_cam = 4
    H, W = int(self.cfg.height), int(self.cfg.width)
    R = H * W
    sp0 = self.specs[0]
    gids = np.asarray(sp0.enabled_geom_ids, dtype=np.int32)
    G = len(gids)
    assert G <= 128, f"Expected Panda tactile scene G <= 128, got {G}"

    block_b = int(self.cfg.block_b)
    block_r = min(int(self.cfg.block_r), 256 if G > 40 else 512)
    B_flat = B * N_cam
    B_pad = ((B_flat + block_b - 1) // block_b) * block_b
    R_pad = ((R + block_r - 1) // block_r) * block_r
    G_pad = ((max(G, 1) + 127) // 128) * 128
    n_b_blk = B_pad // block_b
    n_r_blk = R_pad // block_r

    interp = (jax.default_backend() == "cpu") if self.cfg.interpret is None else bool(self.cfg.interpret)

    # Assemble per-sensor camera poses (B, 4, ...) -> (B*4, ...)
    cpos_stack = jnp.stack([mjx_data_batch.cam_xpos[:, cid, :] for cid in self.cam_ids], axis=1)
    cmat_stack = jnp.stack([mjx_data_batch.cam_xmat[:, cid, :, :] for cid in self.cam_ids], axis=1)
    if rep_batch is not None:
      # Only apply camera DR to perspective head/wrist slots (0, 1); keep tactile ortho cameras exact on finger pads
      cpos_p = rep_batch.cam_xpos[:, :2, :]
      cmat_p = rep_batch.cam_xmat[:, :2, :, :]
      cpos_stack = jnp.concatenate([cpos_p, cpos_stack[:, 2:, :]], axis=1)
      cmat_stack = jnp.concatenate([cmat_p, cmat_stack[:, 2:, :, :]], axis=1)
      fov_persp = jnp.tan(0.5 * rep_batch.cam_fovy_deg[:, :2] * (math.pi / 180.0))
    else:
      fov_persp = jnp.broadcast_to(
          jnp.asarray([math.tan(0.5 * f * (math.pi / 180.0)) for f in self.base_fovys[:2]], jnp.float32)[None, :],
          (B, 2),
      )

    # Ortho slots (2, 3): fov_param = 0.5 * sensor_h_m (where fovy stores sensor_h in MuJoCo ortho)
    fov_ortho = jnp.broadcast_to(
        jnp.asarray([0.5 * self.base_fovys[2], 0.5 * self.base_fovys[3]], jnp.float32)[None, :],
        (B, 2),
    )
    fov_param = jnp.concatenate([fov_persp, fov_ortho], axis=1).reshape(B_flat, 1)
    is_ortho = jnp.broadcast_to(jnp.array([[0.0], [0.0], [1.0], [1.0]], jnp.float32)[None, :, :], (B, 4, 1)).reshape(B_flat, 1)
    gel_dist = jnp.full((B_flat, 1), float(self.cfg.gel_surface_dist_m), dtype=jnp.float32)

    excl_np = np.array([
        [-999.0, -999.0, -999.0, -999.0],
        [-999.0, -999.0, -999.0, -999.0],
        list(self.left_excl_gids),
        list(self.right_excl_gids),
    ], dtype=np.float32)
    excl_arr = jnp.broadcast_to(jnp.asarray(excl_np)[None, :, :], (B, 4, 4)).reshape(B_flat, 4)

    cam_pack = jnp.concatenate([
        cpos_stack.reshape(B_flat, 3),
        cmat_stack.reshape(B_flat, 9),
        fov_param,
        is_ortho,
        gel_dist,
        excl_arr,
    ], axis=-1)  # (B_flat, 19)
    cam_128 = jnp.pad(cam_pack, ((0, B_pad - B_flat), (0, 128 - cam_pack.shape[1])))

    # Light & texture pack broadcast across 4 sensors per env
    def _tile4(x: jax.Array) -> jax.Array:
      return jnp.broadcast_to(x[:, None, ...], (B, N_cam, *x.shape[1:])).reshape(B_flat, *x.shape[1:])

    if rep_batch is not None:
      lpos = _tile4(rep_batch.light_xpos[:, 0, :])
      ldir = _tile4(rep_batch.light_xdir[:, 0, :])
      ldiff = _tile4(rep_batch.light_diffuse)
      lspec = _tile4(rep_batch.light_specular)
      lamb = _tile4(rep_batch.light_ambient)
      sky1 = _tile4(rep_batch.sky_rgb[:, 0, :])
      sky2 = _tile4(rep_batch.sky_rgb[:, 1, :])
      floor_tex = _tile4(rep_batch.floor_tex)
      table_tex = _tile4(rep_batch.table_tex)
      exposure = _tile4(rep_batch.exposure[:, None])
    else:
      lp, ld = _extract_light_batch(sp0, mjx_data_batch)
      lpos, ldir = _tile4(lp[:, 0, :]), _tile4(ld[:, 0, :])
      ldiff = jnp.broadcast_to(jnp.asarray(sp0.light_diffuse[0], jnp.float32)[None, :], (B_flat, 3))
      lspec = jnp.broadcast_to(jnp.asarray(sp0.light_specular[0], jnp.float32)[None, :], (B_flat, 3))
      lamb = jnp.broadcast_to(jnp.asarray(sp0.light_ambient[0], jnp.float32)[None, :], (B_flat, 3))
      sky1 = jnp.broadcast_to(jnp.asarray(sp0.sky_rgb1, jnp.float32)[None, :], (B_flat, 3))
      sky2 = jnp.broadcast_to(jnp.asarray(sp0.sky_rgb2, jnp.float32)[None, :], (B_flat, 3))
      c1 = jnp.asarray(sp0.checker_rgb1, jnp.float32)
      c2 = jnp.asarray(sp0.checker_rgb2, jnp.float32)
      floor_tex = jnp.broadcast_to(
          jnp.concatenate([jnp.array([float(TEX_CHECKER), float(sp0.checker_repeat[0])]), c1, c2])[None, :],
          (B_flat, 8),
      )
      t_c1 = jnp.array([0.56, 0.44, 0.30], jnp.float32)
      table_tex = jnp.broadcast_to(jnp.concatenate([jnp.array([float(TEX_SOLID), 4.0]), t_c1, t_c1])[None, :], (B_flat, 8))
      exposure = jnp.ones((B_flat, 1), dtype=jnp.float32)

    ldir = ldir / jnp.maximum(jnp.linalg.norm(ldir, axis=-1, keepdims=True), 1e-12)
    light_pack = jnp.concatenate([lpos, ldir, ldiff, lspec, lamb, sky1, sky2, floor_tex, table_tex, exposure], axis=-1)
    light_128 = jnp.pad(light_pack, ((0, B_pad - B_flat), (0, 128 - light_pack.shape[1])))

    lray_dev = _build_unit_tangent_rays(H, W, R_pad)

    # Geometry & Material fields tiled across 4 sensors
    geom_size = jnp.asarray(sp0.geom_size, dtype=jnp.float32)
    gpos = _tile4(mjx_data_batch.geom_xpos[:, gids, :])
    gmat = _tile4(mjx_data_batch.geom_xmat[:, gids, :, :].reshape(B, G, 9))
    gsz_all = jnp.broadcast_to(geom_size[gids][None, :, :], (B_flat, G, 3))
    rb_sq = jnp.broadcast_to(jnp.asarray(_compute_geom_bound_sq(sp0, gids))[None, :, None], (B_flat, G, 1))
    g_16 = jnp.concatenate([gpos, gmat, gsz_all, rb_sq], axis=-1)
    if B_pad != B_flat or G_pad != G:
      g_16 = jnp.pad(g_16, ((0, B_pad - B_flat), (0, G_pad - G), (0, 0)))
    geom_fields = (
        g_16.reshape(n_b_blk, block_b, G_pad, 16)
        .transpose(0, 3, 1, 2)
        .reshape(n_b_blk * 16 * block_b, G_pad)
    )

    if rep_batch is not None:
      g_rgb = _tile4(rep_batch.geom_rgba[:, gids, :3])
      g_rough = _tile4(rep_batch.geom_roughness[:, gids])[..., None]
      g_spec = _tile4(rep_batch.geom_specular[:, gids])[..., None]
    else:
      g_rgb = jnp.broadcast_to(jnp.asarray(sp0.geom_rgba[gids, :3], jnp.float32)[None, :, :], (B_flat, G, 3))
      base_rough = jnp.clip(1.0 - jnp.asarray(sp0.geom_shin[gids], jnp.float32) * 1.4, 0.14, 0.90)
      g_rough = jnp.broadcast_to(base_rough[None, :, None], (B_flat, G, 1))
      g_spec = jnp.broadcast_to(jnp.asarray(sp0.geom_spec[gids], jnp.float32)[None, :, None], (B_flat, G, 1))
    g_emit = jnp.broadcast_to(jnp.asarray(sp0.geom_emit[gids], jnp.float32)[None, :, None], (B_flat, G, 1))
    m_6 = jnp.concatenate([g_rgb, g_rough, g_spec, g_emit], axis=-1)
    if B_pad != B_flat or G_pad != G:
      m_6 = jnp.pad(m_6, ((0, B_pad - B_flat), (0, G_pad - G), (0, 0)))
    mat_fields = (
        m_6.reshape(n_b_blk, block_b, G_pad, 6)
        .transpose(0, 3, 1, 2)
        .reshape(n_b_blk * 6 * block_b, G_pad)
    )

    g_types_static = tuple(int(sp0.geom_type[g]) for g in gids)
    gids_static = tuple(int(g) for g in gids)
    plane_fx = tuple(float(sp0.geom_size[g, 0]) > 0.0 for g in gids)
    plane_fy = tuple(float(sp0.geom_size[g, 1]) > 0.0 for g in gids)
    floor_k = tuple((k, int(g)) for k, g in enumerate(gids) if int(g) in self.floor_gids)
    table_k = tuple((k, int(g)) for k, g in enumerate(gids) if int(g) in self.table_gids)

    znear_f = float(sp0.znear)
    zfar_f = float(sp0.zfar) if sp0.zfar > 0.0 else 50.0
    l_type_0 = int(sp0.light_type[0]) if sp0.nlight > 0 else 0
    l_att_0 = tuple(float(x) for x in sp0.light_attenuation[0]) if sp0.nlight > 0 else (1.0, 0.0, 0.0)
    do_sh_0 = bool(sp0.use_shadows) and (bool(sp0.light_castshadow[0]) if sp0.nlight > 0 else True)
    shadow_frac = float(sp0.shadow_light_fraction)
    headlight_active = bool(sp0.headlight_active)
    hl_amb = tuple(float(x) for x in sp0.headlight_ambient)
    hl_diff = tuple(float(x) for x in sp0.headlight_diffuse)
    use_agx = (self.cfg.tonemap == "agx")
    out_dtype = self.cfg.out_dtype

    def _kernel(
        cam_ref, lray_ref, gf_ref, mf_ref, light_ref,
        out_r_ref, out_g_ref, out_b_ref, out_d_ref,
        out_nx_ref, out_ny_ref, out_nz_ref, out_gid_ref,
    ):
      cam = cam_ref[...]
      lr = lray_ref[...]
      gf = gf_ref[...]
      mf = mf_ref[...]
      linfo = light_ref[...]

      tx_r = lr[0:1, :]
      ty_r = lr[1:2, :]
      cpx, cpy, cpz = cam[:, 0:1], cam[:, 1:2], cam[:, 2:3]
      c00, c01, c02 = cam[:, 3:4], cam[:, 4:5], cam[:, 5:6]
      c10, c11, c12 = cam[:, 6:7], cam[:, 7:8], cam[:, 8:9]
      c20, c21, c22 = cam[:, 9:10], cam[:, 10:11], cam[:, 11:12]
      fov_p = cam[:, 12:13]
      is_ortho = cam[:, 13:14] > 0.5
      gel_surf_dist = cam[:, 14:15]
      excl0, excl1, excl2, excl3 = cam[:, 15:16], cam[:, 16:17], cam[:, 17:18], cam[:, 18:19]

      # Perspective raygen
      ldx_raw = tx_r * fov_p
      ldy_raw = ty_r * fov_p
      inv_ln = 1.0 / jnp.sqrt(ldx_raw * ldx_raw + ldy_raw * ldy_raw + 1.0)
      ldx_p, ldy_p, ldz_p = ldx_raw * inv_ln, ldy_raw * inv_ln, -inv_ln
      rdx_pr = c00 * ldx_p + c01 * ldy_p + c02 * ldz_p
      rdy_pr = c10 * ldx_p + c11 * ldy_p + c12 * ldz_p
      rdz_pr = c20 * ldx_p + c21 * ldy_p + c22 * ldz_p
      inv_rnp = 1.0 / jnp.maximum(jnp.sqrt(rdx_pr * rdx_pr + rdy_pr * rdy_pr + rdz_pr * rdz_pr), 1e-12)
      rdx_p, rdy_p, rdz_p = rdx_pr * inv_rnp, rdy_pr * inv_rnp, rdz_pr * inv_rnp
      inv_cosp = znear_f / inv_ln
      rox_p = cpx + inv_cosp * rdx_p
      roy_p = cpy + inv_cosp * rdy_p
      roz_p = cpz + inv_cosp * rdz_p
      max_dp = (zfar_f - znear_f) / inv_ln

      # Orthographic raygen (GelSight tactile fingertips)
      lox_o = tx_r * fov_p
      loy_o = ty_r * fov_p
      inv_rno = 1.0 / jnp.maximum(jnp.sqrt(c02 * c02 + c12 * c12 + c22 * c22), 1e-12)
      rdx_o, rdy_o, rdz_o = -c02 * inv_rno, -c12 * inv_rno, -c22 * inv_rno
      rox_o = cpx + c00 * lox_o + c01 * loy_o + znear_f * rdx_o
      roy_o = cpy + c10 * lox_o + c11 * loy_o + znear_f * rdy_o
      roz_o = cpz + c20 * lox_o + c21 * loy_o + znear_f * rdz_o
      max_do = jnp.full((block_b, block_r), zfar_f - znear_f, dtype=jnp.float32)

      rdx = jnp.where(is_ortho, rdx_o, rdx_p)
      rdy = jnp.where(is_ortho, rdy_o, rdy_p)
      rdz = jnp.where(is_ortho, rdz_o, rdz_p)
      rox = jnp.where(is_ortho, rox_o, rox_p)
      roy = jnp.where(is_ortho, roy_o, roy_p)
      roz = jnp.where(is_ortho, roz_o, roz_p)
      neg_z = jnp.where(is_ortho, 1.0, inv_ln)
      best_d = jnp.where(is_ortho, max_do, max_dp)

      best_gid = jnp.full((block_b, block_r), -1, dtype=jnp.int32)
      best_nx = jnp.zeros((block_b, block_r), dtype=jnp.float32)
      best_ny = jnp.zeros((block_b, block_r), dtype=jnp.float32)
      best_nz = jnp.zeros((block_b, block_r), dtype=jnp.float32)
      alb_r = jnp.zeros((block_b, block_r), dtype=jnp.float32)
      alb_g = jnp.zeros((block_b, block_r), dtype=jnp.float32)
      alb_b = jnp.zeros((block_b, block_r), dtype=jnp.float32)
      roughness = jnp.full((block_b, block_r), 0.45, dtype=jnp.float32)
      spec_coef = jnp.zeros((block_b, block_r), dtype=jnp.float32)

      for k in range(G):
        px = gf[0 * block_b : 1 * block_b, k : k + 1]
        py = gf[1 * block_b : 2 * block_b, k : k + 1]
        pz = gf[2 * block_b : 3 * block_b, k : k + 1]
        m00, m01, m02 = gf[3 * block_b : 4 * block_b, k : k + 1], gf[4 * block_b : 5 * block_b, k : k + 1], gf[5 * block_b : 6 * block_b, k : k + 1]
        m10, m11, m12 = gf[6 * block_b : 7 * block_b, k : k + 1], gf[7 * block_b : 8 * block_b, k : k + 1], gf[8 * block_b : 9 * block_b, k : k + 1]
        m20, m21, m22 = gf[9 * block_b : 10 * block_b, k : k + 1], gf[10 * block_b : 11 * block_b, k : k + 1], gf[11 * block_b : 12 * block_b, k : k + 1]
        sx, sy, sz = gf[12 * block_b : 13 * block_b, k : k + 1], gf[13 * block_b : 14 * block_b, k : k + 1], gf[14 * block_b : 15 * block_b, k : k + 1]
        rb_sq = gf[15 * block_b : 16 * block_b, k : k + 1]

        cd, cnx, cny, cnz = _ray_one_geom_v2(
            g_types_static[k], px, py, pz, m00, m01, m02, m10, m11, m12, m20, m21, m22,
            sx, sy, sz, rb_sq, rox, roy, roz, rdx, rdy, rdz,
            plane_finite_x=plane_fx[k], plane_finite_y=plane_fy[k], compute_normal=True,
        )
        gid_f = float(gids_static[k])
        is_self = is_ortho & (
            (jnp.abs(gid_f - excl0) < 0.5)
            | (jnp.abs(gid_f - excl1) < 0.5)
            | (jnp.abs(gid_f - excl2) < 0.5)
            | (jnp.abs(gid_f - excl3) < 0.5)
        )
        imp = (~is_self) & (cd < best_d)
        best_d = jnp.where(imp, cd, best_d)
        best_gid = jnp.where(imp, jnp.int32(gids_static[k]), best_gid)
        best_nx = jnp.where(imp, cnx, best_nx)
        best_ny = jnp.where(imp, cny, best_ny)
        best_nz = jnp.where(imp, cnz, best_nz)
        alb_r = jnp.where(imp, mf[0 * block_b : 1 * block_b, k : k + 1], alb_r)
        alb_g = jnp.where(imp, mf[1 * block_b : 2 * block_b, k : k + 1], alb_g)
        alb_b = jnp.where(imp, mf[2 * block_b : 3 * block_b, k : k + 1], alb_b)
        roughness = jnp.where(imp, mf[3 * block_b : 4 * block_b, k : k + 1], roughness)
        spec_coef = jnp.where(imp, mf[4 * block_b : 5 * block_b, k : k + 1], spec_coef)

      hit = best_gid >= 0
      d_safe = jnp.where(hit, best_d, 0.0)
      nx_out = jnp.where(hit, best_nx, 0.0)
      ny_out = jnp.where(hit, best_ny, 0.0)
      nz_out = jnp.where(hit, best_nz, 0.0)
      nz_onb = jnp.where(hit, best_nz, 1.0)
      hx = rox + d_safe * rdx
      hy = roy + d_safe * rdy
      hz = roz + d_safe * rdz

      for _, fgid in floor_k:
        tr, tg_c, tb = _eval_procedural_tex_vmem(
            hx, hy, linfo[:, 21:22], linfo[:, 22:23],
            linfo[:, 23:24], linfo[:, 24:25], linfo[:, 25:26],
            linfo[:, 26:27], linfo[:, 27:28], linfo[:, 28:29],
        )
        is_fg = best_gid == fgid
        alb_r = jnp.where(is_fg, tr, alb_r)
        alb_g = jnp.where(is_fg, tg_c, alb_g)
        alb_b = jnp.where(is_fg, tb, alb_b)

      for _, tgid in table_k:
        tr, tg_c, tb = _eval_procedural_tex_vmem(
            hx, hy, linfo[:, 29:30], linfo[:, 30:31],
            linfo[:, 31:32], linfo[:, 32:33], linfo[:, 33:34],
            linfo[:, 34:35], linfo[:, 35:36], linfo[:, 36:37],
        )
        is_tg = best_gid == tgid
        alb_r = jnp.where(is_tg, tr, alb_r)
        alb_g = jnp.where(is_tg, tg_c, alb_g)
        alb_b = jnp.where(is_tg, tb, alb_b)

      # Perspective lighting & shadow pass
      lpx_l, lpy_l, lpz_l = linfo[:, 0:1], linfo[:, 1:2], linfo[:, 2:3]
      ldx_l, ldy_l, ldz_l = linfo[:, 3:4], linfo[:, 4:5], linfo[:, 5:6]
      ldiff_r, ldiff_g, ldiff_b = linfo[:, 6:7], linfo[:, 7:8], linfo[:, 8:9]
      lspec_r, lspec_g, lspec_b = linfo[:, 9:10], linfo[:, 10:11], linfo[:, 11:12]
      lamb_r, lamb_g, lamb_b = linfo[:, 12:13], linfo[:, 13:14], linfo[:, 14:15]
      sky1_r, sky1_g, sky1_b = linfo[:, 15:16], linfo[:, 16:17], linfo[:, 17:18]
      sky2_r, sky2_g, sky2_b = linfo[:, 18:19], linfo[:, 19:20], linfo[:, 20:21]
      exposure_v = linfo[:, 37:38]

      sh_ox = hx + nx_out * 2.0e-4
      sh_oy = hy + ny_out * 2.0e-4
      sh_oz = hz + nz_onb * 2.0e-4

      if l_type_0 == 1:
        lx_v = jnp.broadcast_to(-ldx_l, (block_b, block_r))
        ly_v = jnp.broadcast_to(-ldy_l, (block_b, block_r))
        lz_v = jnp.broadcast_to(-ldz_l, (block_b, block_r))
        dist_lim = 1.0e8
        attenuation = jnp.ones((block_b, block_r), dtype=jnp.float32)
      else:
        dlx, dly, dlz = lpx_l - sh_ox, lpy_l - sh_oy, lpz_l - sh_oz
        d_sq = dlx * dlx + dly * dly + dlz * dlz
        d_raw = jnp.sqrt(d_sq)
        inv_dl = 1.0 / jnp.maximum(d_raw, 1e-12)
        lx_v, ly_v, lz_v = dlx * inv_dl, dly * inv_dl, dlz * inv_dl
        dist_lim = jnp.maximum(d_raw - 1.0e-3, 0.0)
        att_denom = l_att_0[0] + l_att_0[1] * d_raw + l_att_0[2] * d_sq
        attenuation = jnp.where(att_denom > 1e-12, 1.0 / jnp.maximum(att_denom, 1e-12), 1.0)

      ndotl = jnp.maximum(0.0, nx_out * lx_v + ny_out * ly_v + nz_onb * lz_v)
      if do_sh_0:
        sh_max = jnp.where((~is_ortho) & hit & (ndotl > 0.0), dist_lim, 0.0)
        occ_i32 = jnp.zeros((block_b, block_r), dtype=jnp.int32)
        for k in range(G):
          px = gf[0 * block_b : 1 * block_b, k : k + 1]
          py = gf[1 * block_b : 2 * block_b, k : k + 1]
          pz = gf[2 * block_b : 3 * block_b, k : k + 1]
          m00, m01, m02 = gf[3 * block_b : 4 * block_b, k : k + 1], gf[4 * block_b : 5 * block_b, k : k + 1], gf[5 * block_b : 6 * block_b, k : k + 1]
          m10, m11, m12 = gf[6 * block_b : 7 * block_b, k : k + 1], gf[7 * block_b : 8 * block_b, k : k + 1], gf[8 * block_b : 9 * block_b, k : k + 1]
          m20, m21, m22 = gf[9 * block_b : 10 * block_b, k : k + 1], gf[10 * block_b : 11 * block_b, k : k + 1], gf[11 * block_b : 12 * block_b, k : k + 1]
          sx, sy, sz = gf[12 * block_b : 13 * block_b, k : k + 1], gf[13 * block_b : 14 * block_b, k : k + 1], gf[14 * block_b : 15 * block_b, k : k + 1]
          rb_sq = gf[15 * block_b : 16 * block_b, k : k + 1]
          scd, _, _, _ = _ray_one_geom_v2(
              g_types_static[k], px, py, pz, m00, m01, m02, m10, m11, m12, m20, m21, m22,
              sx, sy, sz, rb_sq, sh_ox, sh_oy, sh_oz, lx_v, ly_v, lz_v,
              plane_finite_x=plane_fx[k], plane_finite_y=plane_fy[k], compute_normal=False,
          )
          occ_i32 = occ_i32 | (scd < sh_max).astype(jnp.int32)
        vis = jnp.where((occ_i32 > 0) & hit & (ndotl > 0.0), shadow_frac, 1.0)
      else:
        vis = 1.0

      alpha = roughness * roughness
      a2 = alpha * alpha
      k_g = ((roughness + 1.0) * (roughness + 1.0)) * 0.125
      f0 = 0.04 + 0.08 * spec_coef
      ndotv = jnp.maximum(1e-4, -(nx_out * rdx + ny_out * rdy + nz_onb * rdz))
      g1_v = ndotv / (ndotv * (1.0 - k_g) + k_g + 1e-12)
      hx_v, hy_v, hz_v = lx_v - rdx, ly_v - rdy, lz_v - rdz
      inv_hn = 1.0 / jnp.maximum(jnp.sqrt(hx_v * hx_v + hy_v * hy_v + hz_v * hz_v), 1e-12)
      hx_n, hy_n, hz_n = hx_v * inv_hn, hy_v * inv_hn, hz_v * inv_hn
      ndoth = jnp.maximum(0.0, nx_out * hx_n + ny_out * hy_n + nz_onb * hz_n)
      vdoth = jnp.maximum(0.0, -(rdx * hx_n + rdy * hy_n + rdz * hz_n))
      denom_d = ndoth * ndoth * (a2 - 1.0) + 1.0
      d_ggx = a2 / (math.pi * denom_d * denom_d + 1e-12)
      omv = 1.0 - vdoth
      omv2 = omv * omv
      f_schlick = f0 + (1.0 - f0) * (omv2 * omv2 * omv)
      g1_l = ndotl / (ndotl * (1.0 - k_g) + k_g + 1e-12)
      spec_brdf = (d_ggx * f_schlick * g1_l * g1_v) / (4.0 * ndotl * ndotv + 1e-4)

      weight = attenuation * vis * ndotl
      kd = 1.0 - f_schlick
      diff_r, diff_g, diff_b = ldiff_r * weight * kd, ldiff_g * weight * kd, ldiff_b * weight * kd
      spec_fac = jnp.where((spec_coef > 0.0) & (ndotl > 0.0), spec_brdf * weight * math.pi, 0.0)
      spec_r, spec_g, spec_b = lspec_r * spec_fac, lspec_g * spec_fac, lspec_b * spec_fac

      hemi_t = 0.5 + 0.5 * nz_onb
      amb_r = lamb_r + 0.16 * (hemi_t * sky1_r + (1.0 - hemi_t) * sky2_r)
      amb_g = lamb_g + 0.16 * (hemi_t * sky1_g + (1.0 - hemi_t) * sky2_g)
      amb_b = lamb_b + 0.16 * (hemi_t * sky1_b + (1.0 - hemi_t) * sky2_b)
      if headlight_active:
        amb_r = amb_r + 0.35 * hl_amb[0]
        amb_g = amb_g + 0.35 * hl_amb[1]
        amb_b = amb_b + 0.35 * hl_amb[2]
        diff_r = diff_r + 0.25 * hl_diff[0] * ndotv
        diff_g = diff_g + 0.25 * hl_diff[1] * ndotv
        diff_b = diff_b + 0.25 * hl_diff[2] * ndotv

      tot_r = alb_r * (amb_r + diff_r) + spec_r
      tot_g = alb_g * (amb_g + diff_g) + spec_g
      tot_b = alb_b * (amb_b + diff_b) + spec_b
      sky_t = jnp.clip(0.5 * (rdz + 1.0), 0.0, 1.0)
      lin_r = jnp.where(hit, tot_r, sky_t * sky1_r + (1.0 - sky_t) * sky2_r)
      lin_g = jnp.where(hit, tot_g, sky_t * sky1_g + (1.0 - sky_t) * sky2_g)
      lin_b = jnp.where(hit, tot_b, sky_t * sky1_b + (1.0 - sky_t) * sky2_b)
      if use_agx:
        fin_r, fin_g, fin_b = _agx_tonemap_rgb_components(lin_r, lin_g, lin_b, exposure_v)
      else:
        fin_r, fin_g, fin_b = jnp.clip(lin_r, 0.0, 1.0), jnp.clip(lin_g, 0.0, 1.0), jnp.clip(lin_b, 0.0, 1.0)

      # Orthographic GelSight tactile branch (computed in SAME VMEM kernel!)
      z_depth = jnp.where(hit, znear_f + best_d * neg_z, -1.0)
      pen_depth = jnp.where(hit & (z_depth < gel_surf_dist), gel_surf_dist - z_depth, 0.0)
      in_contact = is_ortho & (pen_depth > 0.0)

      # Pad-local contact normal: cam_mat^T @ world_normal
      lnx = c00 * nx_out + c10 * ny_out + c20 * nz_out
      lny = c01 * nx_out + c11 * ny_out + c21 * nz_out
      lnz = jnp.maximum(c02 * nx_out + c12 * ny_out + c22 * nz_out, 0.0)
      inv_ln_t = 1.0 / jnp.sqrt(lnx * lnx + lny * lny + lnz * lnz + 1e-12)
      tnx = jnp.where(in_contact, lnx * inv_ln_t, 0.0)
      tny = jnp.where(in_contact, lny * inv_ln_t, 0.0)
      tnz = jnp.where(in_contact, lnz * inv_ln_t, 1.0)

      gel_r, gel_g, gel_b = _gelsight_tricolor_led_rgb(tnx, tny, tnz, in_contact)

      out_r_ref[...] = jnp.where(is_ortho, gel_r, fin_r).astype(out_dtype)
      out_g_ref[...] = jnp.where(is_ortho, gel_g, fin_g).astype(out_dtype)
      out_b_ref[...] = jnp.where(is_ortho, gel_b, fin_b).astype(out_dtype)
      out_d_ref[...] = jnp.where(is_ortho, pen_depth, z_depth).astype(out_dtype)
      out_nx_ref[...] = jnp.where(is_ortho, tnx, nx_out).astype(out_dtype)
      out_ny_ref[...] = jnp.where(is_ortho, tny, ny_out).astype(out_dtype)
      out_nz_ref[...] = jnp.where(is_ortho, tnz, nz_out).astype(out_dtype)
      out_gid_ref[...] = best_gid

    in_specs = [
        pl.BlockSpec((block_b, 128), lambda bi, ri: (bi, 0)),
        pl.BlockSpec((16, block_r), lambda bi, ri: (0, ri)),
        pl.BlockSpec((16 * block_b, G_pad), lambda bi, ri: (bi, 0)),
        pl.BlockSpec((6 * block_b, G_pad), lambda bi, ri: (bi, 0)),
        pl.BlockSpec((block_b, 128), lambda bi, ri: (bi, 0)),
    ]
    out_shape = [jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype) for _ in range(7)] + [
        jax.ShapeDtypeStruct((B_pad, R_pad), jnp.int32)
    ]

    outs = pl.pallas_call(
        _kernel,
        out_shape=out_shape,
        grid=(n_b_blk, n_r_blk),
        in_specs=in_specs,
        out_specs=[pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)) for _ in range(8)],
        compiler_params=None if interp else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
        interpret=interp,
    )(cam_128, lray_dev, geom_fields, mat_fields, light_128)

    r_out, g_out, b_out, d_out, nx_out, ny_out, nz_out, gid_out = outs
    rgb_all = jnp.stack([r_out[:B_flat, :R], g_out[:B_flat, :R], b_out[:B_flat, :R]], axis=-1).reshape(B, 4, H, W, 3)
    d_all = d_out[:B_flat, :R].reshape(B, 4, H, W)
    n_all = jnp.stack([nx_out[:B_flat, :R], ny_out[:B_flat, :R], nz_out[:B_flat, :R]], axis=-1).reshape(B, 4, H, W, 3)
    gid_all = gid_out[:B_flat, :R].reshape(B, 4, H, W)

    vis_rgb = rgb_all[:, :2]
    vis_depth = d_all[:, :2]
    vis_normals = n_all[:, :2]

    tac_depth = d_all[:, 2:]
    tac_normals = n_all[:, 2:]
    if self.cfg.apply_membrane_skirt:
      tac_rgb = _apply_elastomer_membrane_skirt_rgb(tac_depth.astype(jnp.float32), tac_normals.astype(jnp.float32)).astype(out_dtype)
    else:
      tac_rgb = rgb_all[:, 2:]

    return {
        "vis_rgb": vis_rgb,             # (B, 2, H, W, 3) head + wrist
        "vis_depth": vis_depth,         # (B, 2, H, W)
        "vis_normals": vis_normals,     # (B, 2, H, W, 3)
        "tactile_rgb": tac_rgb,         # (B, 2, H, W, 3) left + right GelSight tricolor RGB
        "tactile_depth": tac_depth,     # (B, 2, H, W) elastomer penetration depth (m)
        "tactile_normals": tac_normals, # (B, 2, H, W, 3) pad-local contact normals
        "instance_id": gid_all,         # (B, 4, H, W)
    }

  def build_step_and_render_fn(
      self,
      mjx_model: Any,
      batch_size: int,
      rep_cfg: Optional[ReplicatorCfg] = None,
      with_dr: bool = True,
      with_physics: bool = True,
      substeps: int = 1,
  ):
    """Compile ONE zero-copy XLA graph: MJX grasp physics + DR + 4-sensor Visuo-Tactile Pallas kernel."""
    sp0 = self.specs[0]
    from tpu_mjx_render.adapters.isaac_replicator import _default_category_masks
    cat_masks = _default_category_masks(self.mjmodel)

    use_spmd = (
        self.mesh is not None
        and self.n_chips > 1
        and batch_size >= self.n_chips
        and (batch_size % self.n_chips == 0)
    )

    def _local_fn(mx_l, d_b, u_b, rng_b):
      if with_physics:
        if substeps == 1:
          dn = jax.vmap(lambda di, ui: mjx.step(mx_l, di.replace(ctrl=ui)))(d_b, u_b)
        else:
          def _sub(_, ds):
            return jax.vmap(lambda di, ui: mjx.step(mx_l, di.replace(ctrl=ui)))(ds, u_b)
          dn = jax.lax.fori_loop(0, int(substeps), _sub, d_b)
      else:
        dn = d_b

      rb = (
          _apply_replicator_local(
              rng_or_keys=rng_b,
              mjx_data_batch=dn,
              base_spec=sp0,
              cam_ids=self.cam_ids,
              base_fovys=self.base_fovys,
              category_masks=cat_masks,
              cfg=rep_cfg or ReplicatorCfg(),
          )
          if with_dr
          else None
      )
      obs = self.render_local(dn, rep_batch=rb)
      return dn, obs

    if not use_spmd:
      return jax.jit(lambda d_b, u_b, rng_b: _local_fn(mjx_model, d_b, u_b, rng_b))

    P = jax.sharding.PartitionSpec
    ax = self.cfg.data_axis
    sm_kw = _shard_map_kwargs()
    out_obs_spec = {
        "vis_rgb": P(ax),
        "vis_depth": P(ax),
        "vis_normals": P(ax),
        "tactile_rgb": P(ax),
        "tactile_depth": P(ax),
        "tactile_normals": P(ax),
        "instance_id": P(ax),
    }
    _sm = shard_map(
        _local_fn,
        mesh=self.mesh,
        in_specs=(P(), P(ax), P(ax), P(ax)),
        out_specs=(P(ax), out_obs_spec),
        **sm_kw,
    )
    return jax.jit(lambda d_b, u_b, rng_b: _sm(mjx_model, d_b, u_b, rng_b))


def reference_orthographic_tactile_depth(
    mjmodel: mujoco.MjModel,
    mjx_data_batch: Any,
    cam_name: str,
    H: int,
    W: int,
    gel_surface_dist_m: float,
    excl_gids: Sequence[int],
) -> Tuple[jax.Array, jax.Array]:
  """Exact pure-JAX v2_soa reference orthographic tactile penetration depth & pad-local normal."""
  from tpu_mjx_render.renderer import render_batch

  cid = int(mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_CAMERA, cam_name))
  spec = RenderSceneSpec.from_mjmodel(mjmodel, cam_id=cid, height=H, width=W, use_shadows=False, znear=0.001)
  excl_set = {int(g) for g in excl_gids if int(g) >= 0}
  kept_gids = tuple(int(g) for g in spec.enabled_geom_ids if int(g) not in excl_set)
  groups_map: Dict[int, list[int]] = {}
  for g in kept_gids:
    groups_map.setdefault(int(spec.geom_type[g]), []).append(g)
  spec_clean = dataclasses.replace(
      spec,
      enabled_geom_ids=kept_gids,
      type_groups=tuple((gt, tuple(ids)) for gt, ids in sorted(groups_map.items())),
  )

  ref = render_batch(spec_clean, mjx_data_batch, mode="v2_soa")
  B = int(mjx_data_batch.cam_xpos.shape[0])
  z_depth = ref.z_depth.reshape(B, H, W)
  hit = (ref.geom_id.reshape(B, H, W) >= 0) & (z_depth >= 0.0)
  pen_depth = jnp.where(hit & (z_depth < gel_surface_dist_m), gel_surface_dist_m - z_depth, 0.0)
  in_contact = pen_depth > 0.0

  cam_mat = mjx_data_batch.cam_xmat[:, cid, :, :]
  world_n = ref.normal.reshape(B, H * W, 3)
  n_local = jnp.einsum("bji,brj->bri", cam_mat, world_n, precision=jax.lax.Precision.HIGHEST)
  n_local = n_local.at[..., 2].set(jnp.maximum(n_local[..., 2], 0.0))
  n_local = n_local / jnp.maximum(jnp.linalg.norm(n_local, axis=-1, keepdims=True), 1e-12)
  undeformed = jnp.broadcast_to(jnp.array([0.0, 0.0, 1.0], jnp.float32), n_local.shape)
  n_out = jnp.where(in_contact.reshape(B, H * W, 1), n_local, undeformed)
  return pen_depth, n_out.reshape(B, H, W, 3)

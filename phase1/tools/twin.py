"""CPU twin of TPU-RTX's Open Data path tracer (experiments/w7_pathtracer_1b/bvh_pbr_pt.py).

The shading code is a line-by-line NumPy port of build_complex_pbr_pathtracer's
_bounce loop; only the ray casts are swapped from the Pallas Wide-BVH-64 kernel to
Embree (smoke_interpret.py checks that the two agree on hits). Random numbers differ,
the estimator does not. Line references point at bvh_pbr_pt.py.

Options let single pieces be switched off or replaced, for the image-gap breakdown:
  linear radiance out (before the tone curve), the +0.08 falloff softening, the clamps.
"""

import dataclasses
import math
from typing import Optional

import numpy as np

import common  # noqa: F401


def _norm(v):
  return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _build_onb(n):  # bvh_pbr_pt.py:460
  nz = n[..., 2:3]
  sign = np.where(nz >= 0.0, 1.0, -1.0)
  a = -1.0 / (sign + nz)
  b = n[..., 0:1] * n[..., 1:2] * a
  t1 = np.concatenate([1.0 + sign * (n[..., 0:1] ** 2) * a, sign * b, -sign * n[..., 0:1]], axis=-1)
  near_neg_z = nz < -0.999999
  t1 = np.where(near_neg_z, np.array([0.0, -1.0, 0.0], np.float32), t1)
  t1 = _norm(t1)
  t2 = np.cross(n, t1)
  return t1.astype(np.float32), t2.astype(np.float32)


def _sample_cosine_hemisphere(u1, u2, n):  # :474
  r = np.sqrt(np.clip(u1, 0.0, 1.0))
  phi = 2.0 * math.pi * u2
  t1, t2 = _build_onb(n)
  d = t1 * (r * np.cos(phi))[..., None] + t2 * (r * np.sin(phi))[..., None] + n * np.sqrt(np.maximum(1.0 - u1, 0.0))[..., None]
  return _norm(d)


def _sample_ggx_half(u1, u2, n, alpha):  # :482
  a2 = alpha * alpha
  u1s = np.clip(u1, 1e-5, 1.0 - 1e-5)
  cos_t2 = (1.0 - u1s) / (1.0 + (a2 - 1.0) * u1s)
  cos_t = np.sqrt(np.clip(cos_t2, 0.0, 1.0))
  sin_t = np.sqrt(np.maximum(1.0 - cos_t2, 0.0))
  phi = 2.0 * math.pi * u2
  t1, t2 = _build_onb(n)
  h = t1 * (sin_t * np.cos(phi))[..., None] + t2 * (sin_t * np.sin(phi))[..., None] + n * cos_t[..., None]
  return _norm(h)


def _smith_g1(ndotv, alpha):  # :494
  a2 = alpha * alpha
  ndv = np.maximum(ndotv, 1e-4)
  return (2.0 * ndv) / (ndv + np.sqrt(a2 + (1.0 - a2) * ndv * ndv))


def _eval_pbr_brdf(n, wo, wi, albedo, metallic, roughness):  # :500
  ndotl = np.maximum(np.sum(n * wi, -1), 0.0)
  ndotv = np.maximum(np.sum(n * wo, -1), 1e-4)
  h = _norm(wo + wi)
  ndoth = np.maximum(np.sum(n * h, -1), 0.0)
  vdoth = np.maximum(np.sum(wo * h, -1), 0.0)
  alpha = np.maximum(roughness * roughness, 0.02)
  a2 = alpha * alpha
  denom_d = ndoth * ndoth * (a2 - 1.0) + 1.0
  D = a2 / (math.pi * denom_d * denom_d + 1e-6)
  G = _smith_g1(ndotl, alpha) * _smith_g1(ndotv, alpha)
  f0 = 0.04 * (1.0 - metallic[..., None]) + albedo * metallic[..., None]
  F = f0 + (1.0 - f0) * ((1.0 - vdoth) ** 5)[..., None]
  spec = (D * G / np.maximum(4.0 * ndotl * ndotv, 1e-4))[..., None] * F
  kd = (1.0 - metallic[..., None]) * (1.0 - F)
  return kd * albedo / math.pi + spec


def agx_tonemap_srgb(rgb_lin, exposure=1.0):  # :521 (ACES-fit curve with an AgX-like matrix)
  x = np.maximum(rgb_lin * exposure, 0.0)
  m = np.array([[0.84247906, 0.07843360, 0.07922375], [0.04232824, 0.87846864, 0.07916613], [0.04237565, 0.07843360, 0.87914297]], np.float32)
  inv_m = np.array([[1.19687900, -0.09802088, -0.09902974], [-0.05289685, 1.15190313, -0.09896118], [-0.05297163, -0.09804345, 1.15107367]], np.float32)
  x_in = np.einsum("...j,ij->...i", x, m)
  a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
  curved = np.clip((x_in * (a * x_in + b)) / (x_in * (c * x_in + d) + e), 0.0, 1.0)
  out = np.clip(np.einsum("...j,ij->...i", curved, inv_m), 0.0, 1.0)
  return np.power(out, 1.0 / 2.2)


def standard_view_srgb(rgb_lin, exposure_stops=0.0):
  """Blender's 'Standard' view transform: exposure, clip, sRGB piecewise transfer."""
  x = np.clip(rgb_lin * (2.0 ** exposure_stops), 0.0, 1.0)
  return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


@dataclasses.dataclass
class TwinOptions:
  falloff_soften: float = 0.08    # :702 adds this to dist^2 of the area light
  clamp_area: Optional[float] = 14.0  # :703
  clamp_sun: Optional[float] = 18.0   # :720
  clamp_spec_weight: Optional[float] = 4.0  # :745
  dump_rays_sample: Optional[int] = None  # keep per-bounce rays of this sample index


class EmbreeTracer:
  def __init__(self, verts, faces, mat_code):
    from embreex import rtcore_scene as rtcs
    from embreex.mesh_construction import TriangleMesh
    self.tri = np.ascontiguousarray(verts[faces], dtype=np.float32)
    self.scene = rtcs.EmbreeScene()
    TriangleMesh(self.scene, self.tri)
    # Shadow rays skip glass and emissive triangles (mat_code >= 1.8), as the any-hit kernel does (:337).
    self.opq_ids = np.nonzero(mat_code < 1.8)[0]
    self.shadow = rtcs.EmbreeScene()
    TriangleMesh(self.shadow, np.ascontiguousarray(self.tri[self.opq_ids]))

  def closest(self, ro, rd, tlim):
    """Returns (dist, prim, u, v); dist = tlim where nothing is hit (as the kernel's bd)."""
    n = len(ro)
    dist = tlim.astype(np.float32).copy()
    prim = np.full(n, -1, np.int64)
    u = np.zeros(n, np.float32)
    v = np.zeros(n, np.float32)
    act = tlim > 0.0
    if act.any():
      res = self.scene.run(np.ascontiguousarray(ro[act], np.float32), np.ascontiguousarray(rd[act], np.float32),
                           dists=np.ascontiguousarray(tlim[act], np.float32), output=1)
      g = np.asarray(res["geomID"])
      hit = g >= 0
      idx = np.nonzero(act)[0][hit]
      dist[idx] = np.asarray(res["tfar"])[hit]
      prim[idx] = np.asarray(res["primID"])[hit]
      u[idx] = np.asarray(res["u"])[hit]
      v[idx] = np.asarray(res["v"])[hit]
    return dist, prim, u, v

  def occluded(self, ro, rd, tlim):
    n = len(ro)
    occ = np.zeros(n, bool)
    act = tlim > 0.0
    if act.any():
      res = self.shadow.run(np.ascontiguousarray(ro[act], np.float32), np.ascontiguousarray(rd[act], np.float32),
                            dists=np.ascontiguousarray(tlim[act], np.float32), query="OCCLUDED")
      occ[np.nonzero(act)[0]] = np.asarray(res) >= 0
    return occ


class Twin:
  def __init__(self, npz_path, rig, height, width, znear=0.05, zfar=120.0):
    ds = np.load(npz_path, allow_pickle=False)
    self.verts = ds["verts"].astype(np.float32)
    self.faces = ds["faces"].astype(np.int64)
    self.vnorm = ds["vnorm"].astype(np.float32)
    self.alb = ds["tri_albedo"].astype(np.float32)
    self.rgh = np.clip(ds["tri_roughness"].astype(np.float32), 0.02, 0.99)
    # The kernel packs mat_code to 1/100 steps (:366) and decodes floor(pk/2)*0.01 (:455).
    self.mcode = (np.floor(ds["tri_mat_code"].astype(np.float32) * 100.0 + 0.5) * 0.01).astype(np.float32)
    self.cam_pos = ds["cam_pos"].astype(np.float32)
    self.cam_xmat = ds["cam_xmat"].astype(np.float32)
    self.fovy = float(ds["cam_fovy"])
    self.tracer = EmbreeTracer(self.verts, self.faces, ds["tri_mat_code"].astype(np.float32))
    self.rig = rig
    self.H, self.W = height, width
    self.znear, self.zfar = znear, zfar

  def _attrs(self, prim, u, v):
    p = np.maximum(prim, 0)
    f = self.faces[p]
    w = 1.0 - u - v
    n = w[:, None] * self.vnorm[f[:, 0]] + u[:, None] * self.vnorm[f[:, 1]] + v[:, None] * self.vnorm[f[:, 2]]
    n = _norm(n)
    miss = prim < 0
    n[miss] = 0.0
    alb = self.alb[p].copy()
    rgh = self.rgh[p].copy()
    mc = self.mcode[p].copy()
    alb[miss] = 0.0
    rgh[miss] = 0.02
    mc[miss] = 0.0
    return n.astype(np.float32), alb, rgh, mc

  def render(self, spp, max_bounces=3, seed=0, opts: Optional[TwinOptions] = None, primary_aovs=False):
    """Returns dict with 'linear' (H,W,3) mean radiance and optional AOVs / ray dumps."""
    opts = opts or TwinOptions()
    rig = self.rig
    H, W = self.H, self.W
    R = H * W
    rng = np.random.default_rng(seed)
    area_c = np.asarray(rig.area_center, np.float32)
    area_n = _norm(np.asarray(rig.area_normal, np.float32))
    at1, at2 = _build_onb(area_n[None])
    at1, at2 = at1[0], at2[0]
    area_r = float(rig.area_radius)
    area_area = math.pi * area_r * area_r
    area_emit = np.asarray(rig.area_emit, np.float32)
    sun_d = _norm(np.asarray(rig.sun_dir, np.float32))
    st1, st2 = _build_onb(sun_d[None])
    st1, st2 = st1[0], st2[0]
    sun_emit = np.asarray(rig.sun_emit, np.float32)
    sun_sin_max = math.sqrt(max(0.0, 1.0 - rig.sun_cos_half_angle ** 2))
    sky_z, sky_h, sky_g = (np.asarray(x, np.float32) for x in (rig.sky_zenith, rig.sky_horizon, rig.sky_ground))

    def env_sky(rd):  # :602
      z = rd[..., 2]
      tu = np.clip(z, 0.0, 1.0)[..., None]
      td = np.clip(-z, 0.0, 1.0)[..., None]
      s = np.where(z[..., None] >= 0.0, (1.0 - tu) * sky_h + tu * sky_z, (1.0 - td) * sky_h + td * sky_g)
      if rig.has_sun:
        cos_s = np.maximum(np.sum(rd * sun_d, -1), 0.0)
        s = s + (cos_s ** 32.0)[..., None] * sun_emit * 0.25
      return s

    half_h = self.znear * math.tan(0.5 * self.fovy * math.pi / 180.0)
    half_w = half_h * (W / H)
    px = np.arange(W, dtype=np.float32) + 0.5
    py = np.arange(H, dtype=np.float32) + 0.5
    gx, gy = np.meshgrid(px, py, indexing="xy")
    gx, gy = gx.reshape(R), gy.reshape(R)
    acc = np.zeros((R, 3), np.float64)
    aovs = None
    dumps = []
    stats = []
    for s in range(spp):
      jit = rng.uniform(-0.5, 0.5, size=(R, 2)).astype(np.float32)
      uu = (gx + jit[:, 0]) / W
      vv = (gy + jit[:, 1]) / H
      x = -half_w + 2.0 * half_w * uu
      y = half_h - 2.0 * half_h * vv
      ld = _norm(np.stack([x, y, np.full_like(x, -self.znear)], -1))
      neg_z = np.maximum(-ld[:, 2], 1e-6)
      rd = _norm(ld @ self.cam_xmat.T)
      ro = self.cam_pos[None] + (self.znear / neg_z)[:, None] * rd
      t_lim = ((self.zfar - self.znear) / neg_z).astype(np.float32)
      thr = np.ones((R, 3), np.float32)
      alive = np.ones(R, bool)
      rad = np.zeros((R, 3), np.float32)
      for b in range(max_bounces):
        tl_eff = np.where(alive, t_lim, 0.0).astype(np.float32)
        if opts.dump_rays_sample == s:
          dumps.append({"bounce": b, "ro": ro.copy(), "rd": rd.copy(), "tlim": tl_eff.copy(), "alive": alive.copy()})
        d_hit, prim, hu, hv = self.tracer.closest(ro, rd, tl_eff)
        nrm, alb, rgh, mcode = self._attrs(prim, hu, hv)
        hit = alive & (d_hit < t_lim - 1e-4) & (d_hit >= 0.0) & (prim >= 0)
        miss = alive & ~hit
        rad = rad + np.where(miss[:, None], thr * env_sky(rd), 0.0)
        wo = -rd
        entering = np.sum(nrm * wo, -1) >= 0.0
        n_ff = np.where(entering[:, None], nrm, -nrm)
        is_emit = hit & (mcode >= 2.8)
        is_glass = hit & (mcode >= 1.8) & (mcode < 2.8)
        is_opq = hit & (mcode < 1.8)
        metallic = np.clip(mcode, 0.0, 1.0)
        emit_power = np.where(mcode >= 2.8, (mcode - 2.0) * 6.0, 0.0)
        rad = rad + np.where(is_emit[:, None], thr * alb * emit_power[:, None], 0.0)
        hit_pos = ro + np.where(hit, d_hit, 0.0)[:, None] * rd
        if primary_aovs and b == 0 and s == 0:
          aovs = {"depth": np.where(hit, d_hit, np.inf).reshape(H, W), "normal": np.where(hit[:, None], nrm, 0).reshape(H, W, 3),
                  "albedo": np.where(hit[:, None], alb, 0).reshape(H, W, 3), "mcode": np.where(hit, mcode, -1).reshape(H, W),
                  "rough": np.where(hit, rgh, -1).reshape(H, W), "prim": np.where(hit, prim, -1).reshape(H, W),
                  "hit_pos": hit_pos.reshape(H, W, 3)}
        # NEE: area light (:684)
        lu, lv = rng.random(R, dtype=np.float32), rng.random(R, dtype=np.float32)
        lr = area_r * np.sqrt(lu)
        lphi = 2.0 * math.pi * lv
        lp = area_c[None] + (lr * np.cos(lphi))[:, None] * at1[None] + (lr * np.sin(lphi))[:, None] * at2[None]
        to_l = lp - hit_pos
        dist_l = np.linalg.norm(to_l, axis=-1)
        wi_l = to_l / np.maximum(dist_l[:, None], 1e-12)
        cos_s = np.maximum(np.sum(n_ff * wi_l, -1), 0.0)
        cos_l = np.maximum(np.sum(area_n[None] * (-wi_l), -1), 0.0)
        need_sh = is_opq & (cos_s > 0.0) & (cos_l > 0.0)
        sh_o = hit_pos + n_ff * 1.5e-3
        sh_lim = np.where(need_sh, np.maximum(dist_l - 2.0e-3, 0.0), 0.0).astype(np.float32)
        occ = self.tracer.occluded(sh_o, wi_l, sh_lim)
        vis = np.where(need_sh & (~occ | (sh_lim <= 1e-4)), 1.0, 0.0)
        brdf_a = _eval_pbr_brdf(n_ff, wo, wi_l, alb, metallic, rgh)
        geom_a = (cos_s * cos_l) / (dist_l * dist_l + opts.falloff_soften) * area_area
        nee_a = thr * brdf_a * area_emit[None] * (geom_a * vis)[:, None]
        if opts.clamp_area is not None:
          nee_a = np.minimum(nee_a, opts.clamp_area)
        rad = rad + np.where(is_opq[:, None], nee_a, 0.0)
        # NEE: sun (:706)
        if rig.has_sun:
          su, sv = rng.random(R, dtype=np.float32), rng.random(R, dtype=np.float32)
          sr = sun_sin_max * np.sqrt(su)
          sphi = 2.0 * math.pi * sv
          wi_sun = _norm(sun_d[None] + (sr * np.cos(sphi))[:, None] * st1[None] + (sr * np.sin(sphi))[:, None] * st2[None])
          cos_sun = np.maximum(np.sum(n_ff * wi_sun, -1), 0.0)
          need_sun = is_opq & (cos_sun > 0.0)
          sun_lim = np.where(need_sun, np.float32(self.zfar * 0.9), 0.0).astype(np.float32)
          occ_s = self.tracer.occluded(sh_o, wi_sun, sun_lim)
          vis_sun = np.where(need_sun & ~occ_s, 1.0, 0.0)
          brdf_sun = _eval_pbr_brdf(n_ff, wo, wi_sun, alb, metallic, rgh)
          nee_sun = thr * brdf_sun * sun_emit[None] * (cos_sun * vis_sun)[:, None]
          if opts.clamp_sun is not None:
            nee_sun = np.minimum(nee_sun, opts.clamp_sun)
          rad = rad + np.where(is_opq[:, None], nee_sun, 0.0)
        # Next bounce (:723)
        u_lobe = rng.random(R, dtype=np.float32)
        u1 = rng.random(R, dtype=np.float32)
        u2 = rng.random(R, dtype=np.float32)
        p_spec = np.clip(0.15 + 0.78 * metallic, 0.15, 0.90)
        choose_spec = u_lobe < p_spec
        wi_diff = _sample_cosine_hemisphere(u1, u2, n_ff)
        alpha = np.maximum(rgh * rgh, 0.02)
        h_ggx = _sample_ggx_half(u1, u2, n_ff, alpha)
        wi_spec = _norm(2.0 * np.sum(wo * h_ggx, -1, keepdims=True) * h_ggx - wo)
        wi_opq = np.where(choose_spec[:, None], wi_spec, wi_diff)
        ndotl_s = np.maximum(np.sum(n_ff * wi_spec, -1), 0.0)
        ndotv = np.maximum(np.sum(n_ff * wo, -1), 1e-4)
        ndoth = np.maximum(np.sum(n_ff * h_ggx, -1), 1e-4)
        vdoth = np.maximum(np.sum(wo * h_ggx, -1), 1e-4)
        f0 = 0.04 * (1.0 - metallic[:, None]) + alb * metallic[:, None]
        F_s = f0 + (1.0 - f0) * ((1.0 - vdoth) ** 5)[:, None]
        G_s = _smith_g1(ndotl_s, alpha) * _smith_g1(ndotv, alpha)
        w_spec = F_s * ((G_s * vdoth) / np.maximum(ndotv * ndoth * p_spec, 1e-4))[:, None]
        if opts.clamp_spec_weight is not None:
          w_spec = np.minimum(w_spec, opts.clamp_spec_weight)
        w_spec = np.where(ndotl_s[:, None] > 0.0, w_spec, 0.0)
        w_diff = (1.0 - metallic[:, None]) * alb / np.maximum(1.0 - p_spec[:, None], 0.1)
        w_opq = np.where(choose_spec[:, None], w_spec, w_diff)
        eta = np.where(entering, 1.0 / 1.50, 1.50)
        cos_i = np.clip(np.sum(wo * n_ff, -1), 0.0, 1.0)
        sin2_t = eta * eta * (1.0 - cos_i * cos_i)
        tir = sin2_t > 1.0
        r0_g = ((1.0 - eta) / (1.0 + eta)) ** 2
        fres = r0_g + (1.0 - r0_g) * ((1.0 - cos_i) ** 5)
        refl = 2.0 * cos_i[:, None] * n_ff - wo
        cos_t = np.sqrt(np.maximum(1.0 - sin2_t, 0.0))
        refr = eta[:, None] * rd + (eta * cos_i - cos_t)[:, None] * n_ff
        do_refl = tir | (u_lobe < fres)
        wi_glass = _norm(np.where(do_refl[:, None], refl, refr))
        ro_glass = np.where(do_refl[:, None], hit_pos + n_ff * 1.5e-3, hit_pos - n_ff * 1.5e-3)
        wi_next = np.where(is_glass[:, None], wi_glass, wi_opq)
        new_ro = np.where(is_glass[:, None], ro_glass, hit_pos + n_ff * 1.5e-3)
        bounce_w = np.where(is_glass[:, None], alb, w_opq)
        stats.append({"sample": s, "bounce": b, "alive": int(alive.sum()), "hit": int(hit.sum()),
                      "glass": int(is_glass.sum()), "emit": int(is_emit.sum())})
        thr = (thr * bounce_w).astype(np.float32)
        alive = (is_opq | is_glass) & (np.max(thr, -1) > 1e-3)
        ro, rd = new_ro.astype(np.float32), wi_next.astype(np.float32)
        t_lim = np.full(R, self.zfar, np.float32)
      acc += rad
    lin = (acc / spp).reshape(H, W, 3).astype(np.float32)
    return {"linear": lin, "aovs": aovs, "dumps": dumps, "stats": stats}

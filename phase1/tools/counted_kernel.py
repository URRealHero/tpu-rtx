"""TPU-RTX's Wide-BVH-64 kernel (bvh_pbr_pt.pbr_raycast_tpu, closest-hit or any-hit) with fetch counters.

The traversal is copied unchanged from experiments/w7_pathtracer_1b/bvh_pbr_pt.py:198-457;
the only additions are a (1, 128) counter row carried through the three nested loops and
written to row 8 of each tile's output. Per 128-ray tile it reports:

  c0  L2 node fetches   (one 4 KB DMA each: 64 child boxes)
  c1  L1 node fetches   (one 4 KB DMA each: 64 leaf boxes)
  c2  leaf fetches      (one 12 KB DMA each: 128 triangles, tested against all 128 rays)
  c3  ray-leaf pairs that needed the leaf (ray's box entry < its current best hit)
  c4  rays in the tile with t_limit > 0 (live rays)

A leaf fetch costs the same whether 1 or 128 rays needed it, so c3 / (128 * c2) is the
fraction of intersection work that could change a result ("lane utilization").
"""

from typing import Tuple

import common  # noqa: F401
import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from bvh_pbr_pt import BLOCK_R, BRANCH_FACTOR, _pack_rays_tiled
from tpu_mjx_render.primitives import MJ_MAXVAL

N_COUNTERS = 5


def pbr_raycast_counted(l2_roots_dev, l2_dev, l0_dev, leaf_dev, ro, rd, t_limit, H: int, W: int,
                        interpret=True, any_hit: bool = False) -> Tuple[jax.Array, ...]:
  B, R, _ = ro.shape
  rays_2d, n_tiles, can_2d_tile = _pack_rays_tiled(ro, rd, t_limit, H, W)
  total_tiles = B * n_tiles
  leaf_rows = 24

  def _kernel(ray_ref, l2_roots_ref, l2_hbm_ref, l0_hbm_ref, leaf_hbm_ref, out_ref,
              l2_vmem, l0_vmem, leaf_vmem, sem_l2, sem_l0, sem_leaf):
    cols_iota_64 = jax.lax.broadcasted_iota(jnp.int32, (1, BRANCH_FACTOR), 1)
    cnt_iota = jax.lax.broadcasted_iota(jnp.int32, (1, 128), 1)
    rt = ray_ref[...]
    l2_roots_blk = l2_roots_ref[...]
    rox = jnp.transpose(rt[0:1, :], (1, 0))
    roy = jnp.transpose(rt[1:2, :], (1, 0))
    roz = jnp.transpose(rt[2:3, :], (1, 0))
    rdx = jnp.transpose(rt[3:4, :], (1, 0))
    rdy = jnp.transpose(rt[4:5, :], (1, 0))
    rdz = jnp.transpose(rt[5:6, :], (1, 0))
    bd = jnp.transpose(rt[6:7, :], (1, 0))
    irx = jnp.where(jnp.abs(rdx) > 1e-12, 1.0 / jnp.where(jnp.abs(rdx) > 1e-12, rdx, 1.0), 1e12)
    iry = jnp.where(jnp.abs(rdy) > 1e-12, 1.0 / jnp.where(jnp.abs(rdy) > 1e-12, rdy, 1.0), 1e12)
    irz = jnp.where(jnp.abs(rdz) > 1e-12, 1.0 / jnp.where(jnp.abs(rdz) > 1e-12, rdz, 1.0), 1e12)

    def _slab(blk, bdv):
      bx0, by0, bz0 = blk[0:1, :BRANCH_FACTOR], blk[1:2, :BRANCH_FACTOR], blk[2:3, :BRANCH_FACTOR]
      bx1, by1, bz1 = blk[3:4, :BRANCH_FACTOR], blk[4:5, :BRANCH_FACTOR], blk[5:6, :BRANCH_FACTOR]
      valid = blk[6:7, :BRANCH_FACTOR] > 0.5
      tx0, tx1 = (bx0 - rox) * irx, (bx1 - rox) * irx
      ty0, ty1 = (by0 - roy) * iry, (by1 - roy) * iry
      tz0, tz1 = (bz0 - roz) * irz, (bz1 - roz) * irz
      tmin_pos = jnp.maximum(0.0, jnp.maximum(jnp.minimum(tx0, tx1), jnp.maximum(jnp.minimum(ty0, ty1), jnp.minimum(tz0, tz1))))
      tmax = jnp.minimum(jnp.maximum(tx0, tx1), jnp.minimum(jnp.maximum(ty0, ty1), jnp.maximum(tz0, tz1)))
      tmin_eff = jnp.where((tmax >= tmin_pos) & valid, tmin_pos, 1e20)
      unvis = jnp.where(valid, 1.0, 0.0)
      tmin_red = jnp.min(jnp.where(tmin_eff < bdv, tmin_eff, 1e20), axis=0, keepdims=True)
      return tmin_eff, unvis, tmin_red

    tmin2_eff, l2_unvis_f, l2_tmin = _slab(l2_roots_blk, bd)
    z1 = jnp.zeros((BLOCK_R, 1), jnp.float32)
    cnt0 = jnp.where(cnt_iota == 4, jnp.sum(jnp.where(bd > 0.0, 1.0, 0.0)), 0.0)

    def _cond(c):
      return jnp.min(c[0]) < jnp.max(c[2])

    def _l2_body(c2):
      t2, unvis2_f, bd2, nx2, ny2, nz2, ar2, ag2, ab2, pk2, cn2 = c2
      best_r = jnp.argmin(t2, axis=1)[0]
      unvis2_next_f = jnp.where(cols_iota_64 == best_r, 0.0, unvis2_f)
      cp2 = pltpu.make_async_copy(l2_hbm_ref.at[best_r, :, :], l2_vmem, sem_l2)
      cp2.start(); cp2.wait()
      cn2 = cn2 + jnp.where(cnt_iota == 0, 1.0, 0.0)
      tmin1_eff, l1_unvis_f, l1_tmin = _slab(l2_vmem[...], bd2)

      def _l1_body(c1):
        t1, unvis1_f, bd1, nx1, ny1, nz1, ar1, ag1, ab1, pk1, cn1 = c1
        best_c1 = jnp.argmin(t1, axis=1)[0]
        unvis1_next_f = jnp.where(cols_iota_64 == best_c1, 0.0, unvis1_f)
        l1_idx = best_r * BRANCH_FACTOR + best_c1
        cp0 = pltpu.make_async_copy(l0_hbm_ref.at[l1_idx, :, :], l0_vmem, sem_l0)
        cp0.start(); cp0.wait()
        cn1 = cn1 + jnp.where(cnt_iota == 1, 1.0, 0.0)
        tmin0_eff, l0_unvis_f, l0_tmin = _slab(l0_vmem[...], bd1)

        def _l0_body(c0):
          t0, unvis0_f, bd0, nx0, ny0, nz0, ar0, ag0, ab0, pk0, cn0 = c0
          best_c0 = jnp.argmin(t0, axis=1)[0]
          unvis0_next_f = jnp.where(cols_iota_64 == best_c0, 0.0, unvis0_f)
          leaf_idx = l1_idx * BRANCH_FACTOR + best_c0
          cp1 = pltpu.make_async_copy(leaf_hbm_ref.at[leaf_idx, :leaf_rows, :], leaf_vmem, sem_leaf)
          cp1.start(); cp1.wait()
          pk = leaf_vmem[...]
          col_t = jnp.sum(jnp.where(cols_iota_64 == best_c0, tmin0_eff, 0.0), axis=1, keepdims=True)
          needed = jnp.sum(jnp.where(col_t < bd0, 1.0, 0.0))
          cn0 = cn0 + jnp.where(cnt_iota == 2, 1.0, 0.0) + jnp.where(cnt_iota == 3, needed, 0.0)

          v0x, v0y, v0z = pk[0:1, :], pk[1:2, :], pk[2:3, :]
          e1x, e1y, e1z = pk[3:4, :], pk[4:5, :], pk[5:6, :]
          e2x, e2y, e2z = pk[6:7, :], pk[7:8, :], pk[8:9, :]
          tv = pk[9:10, :] > 0.5
          hx = rdy * e2z - rdz * e2y
          hy = rdz * e2x - rdx * e2z
          hz = rdx * e2y - rdy * e2x
          a = e1x * hx + e1y * hy + e1z * hz
          ok_a = tv & (jnp.abs(a) > 1e-8)
          f = jnp.where(ok_a, 1.0 / jnp.where(ok_a, a, 1.0), 0.0)
          sx, sy, sz = rox - v0x, roy - v0y, roz - v0z
          u = f * (sx * hx + sy * hy + sz * hz)
          qx, qy, qz = sy * e1z - sz * e1y, sz * e1x - sx * e1z, sx * e1y - sy * e1x
          v = f * (rdx * qx + rdy * qy + rdz * qz)
          t = f * (e2x * qx + e2y * qy + e2z * qz)
          hit_t = ok_a & (u >= 0.0) & (v >= 0.0) & ((u + v) <= 1.0) & (t >= 0.0) & (t < bd0)
          if any_hit:  # bvh_pbr_pt.py:335-341, unchanged
            hit_opq = hit_t & (pk[23:24, :] < 1.8)
            any_ray_hit = jnp.any(hit_opq, axis=1, keepdims=True)
            new_bd = jnp.where(any_ray_hit, 0.0, bd0)
            t0_next = jnp.min(jnp.where((tmin0_eff < new_bd) & (unvis0_next_f > 0.5), tmin0_eff, 1e20), axis=0, keepdims=True)
            return (t0_next, unvis0_next_f, new_bd, nx0, ny0, nz0, ar0, ag0, ab0, pk0, cn0)
          t_cand = jnp.where(hit_t, t, MJ_MAXVAL)
          min_t = jnp.min(t_cand, axis=1, keepdims=True)
          imp = min_t < bd0
          any_imp = jnp.any(imp)

          def _on_hit(_):
            n0x, n0y, n0z = pk[10:11, :], pk[11:12, :], pk[12:13, :]
            n1x, n1y, n1z = pk[13:14, :], pk[14:15, :], pk[15:16, :]
            n2x, n2y, n2z = pk[16:17, :], pk[17:18, :], pk[18:19, :]
            win = hit_t & (t_cand == min_t)
            bw = 1.0 - u - v
            lnx = jnp.sum(jnp.where(win, bw * n0x + u * n1x + v * n2x, 0.0), axis=1, keepdims=True)
            lny = jnp.sum(jnp.where(win, bw * n0y + u * n1y + v * n2y, 0.0), axis=1, keepdims=True)
            lnz = jnp.sum(jnp.where(win, bw * n0z + u * n1z + v * n2z, 0.0), axis=1, keepdims=True)
            inv_l = 1.0 / jnp.maximum(jnp.sqrt(lnx * lnx + lny * lny + lnz * lnz), 1e-12)
            lnx, lny, lnz = lnx * inv_l, lny * inv_l, lnz * inv_l
            lar = jnp.sum(jnp.where(win, pk[19:20, :], 0.0), axis=1, keepdims=True)
            lag = jnp.sum(jnp.where(win, pk[20:21, :], 0.0), axis=1, keepdims=True)
            lab = jnp.sum(jnp.where(win, pk[21:22, :], 0.0), axis=1, keepdims=True)
            lrgh = jnp.sum(jnp.where(win, pk[22:23, :], 0.0), axis=1, keepdims=True)
            lmat = jnp.sum(jnp.where(win, pk[23:24, :], 0.0), axis=1, keepdims=True)
            lpk = jnp.clip(lrgh, 0.02, 0.99) + 2.0 * jnp.floor(lmat * 100.0 + 0.5)
            new_bd = jnp.where(imp, min_t, bd0)
            t0_n = jnp.min(jnp.where((tmin0_eff < new_bd) & (unvis0_next_f > 0.5), tmin0_eff, 1e20), axis=0, keepdims=True)
            return (t0_n, new_bd, jnp.where(imp, lnx, nx0), jnp.where(imp, lny, ny0), jnp.where(imp, lnz, nz0),
                    jnp.where(imp, lar, ar0), jnp.where(imp, lag, ag0), jnp.where(imp, lab, ab0), jnp.where(imp, lpk, pk0))

          def _on_miss(_):
            return (jnp.where(unvis0_next_f > 0.5, t0, 1e20), bd0, nx0, ny0, nz0, ar0, ag0, ab0, pk0)

          t0_n, nbd, nnx, nny, nnz, nar, nag, nab, npk = jax.lax.cond(any_imp, _on_hit, _on_miss, None)
          return (t0_n, unvis0_next_f, nbd, nnx, nny, nnz, nar, nag, nab, npk, cn0)

        o = jax.lax.while_loop(_cond, _l0_body, (l0_tmin, l0_unvis_f, bd1, nx1, ny1, nz1, ar1, ag1, ab1, pk1, cn1))
        n_bd1 = o[2]
        t1_next = jnp.min(jnp.where((tmin1_eff < n_bd1) & (unvis1_next_f > 0.5), tmin1_eff, 1e20), axis=0, keepdims=True)
        return (t1_next, unvis1_next_f) + tuple(o[2:])

      o = jax.lax.while_loop(_cond, _l1_body, (l1_tmin, l1_unvis_f, bd2, nx2, ny2, nz2, ar2, ag2, ab2, pk2, cn2))
      n_bd2 = o[2]
      t2_next = jnp.min(jnp.where((tmin2_eff < n_bd2) & (unvis2_next_f > 0.5), tmin2_eff, 1e20), axis=0, keepdims=True)
      return (t2_next, unvis2_next_f) + tuple(o[2:])

    o = jax.lax.while_loop(_cond, _l2_body, (l2_tmin, l2_unvis_f, bd, z1, z1, z1, z1, z1, z1, z1, cnt0))
    _, _, out_bd, out_nx, out_ny, out_nz, out_ar, out_ag, out_ab, out_pk, out_cnt = o
    rows = [jnp.transpose(x, (1, 0)) for x in (out_bd, out_nx, out_ny, out_nz, out_ar, out_ag, out_ab, out_pk)]
    out_ref[...] = jnp.concatenate(rows + [out_cnt, jnp.zeros((7, 128), jnp.float32)], axis=0)

  out_2d = pl.pallas_call(
      _kernel,
      out_shape=jax.ShapeDtypeStruct((total_tiles * 16, BLOCK_R), jnp.float32),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(total_tiles,),
          in_specs=[
              pl.BlockSpec((8, BLOCK_R), lambda i: (i, 0)),
              pl.BlockSpec((8, 128), lambda i: (0, 0)),
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec(memory_space=pltpu.HBM),
          ],
          out_specs=pl.BlockSpec((16, BLOCK_R), lambda i: (i, 0)),
          scratch_shapes=[
              pltpu.VMEM((8, 128), jnp.float32),
              pltpu.VMEM((8, 128), jnp.float32),
              pltpu.VMEM((leaf_rows, 128), jnp.float32),
              pltpu.SemaphoreType.DMA,
              pltpu.SemaphoreType.DMA,
              pltpu.SemaphoreType.DMA,
          ],
      ),
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
      interpret=interpret,
  )(rays_2d, l2_roots_dev, l2_dev, l0_dev, leaf_dev)

  out3 = out_2d.reshape(total_tiles, 16, BLOCK_R)
  counts = out3[:, 8, :N_COUNTERS]
  out_t = out3[:, :8, :].reshape(B, n_tiles, 8, BLOCK_R).transpose(0, 1, 3, 2)
  if can_2d_tile:
    ty, tx = H // 8, W // 16
    out_hw = out_t.reshape(B, ty, tx, 8, 16, 8).transpose(0, 1, 3, 2, 4, 5).reshape(B, R, 8)
  else:
    out_hw = out_t.reshape(B, n_tiles * BLOCK_R, 8)[:, :R, :]
  return out_hw[..., 0], counts

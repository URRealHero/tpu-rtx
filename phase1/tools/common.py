"""Shared setup for phase1 tools: CPU-only JAX and import paths into the TPU-RTX tree.

Everything in phase1/ runs on CPU (JAX_PLATFORMS=cpu). TPU kernels run through
Pallas interpret mode, so results are numerically the TPU-RTX algorithm but timings
are not TPU timings.
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
# The TPU hosts are shared: one BLAS/OpenMP thread per process unless asked otherwise.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
  os.environ.setdefault(_v, "1")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (
    ROOT,
    os.path.join(ROOT, "experiments", "w7_pathtracer_1b"),
    os.path.join(ROOT, "experiments", "w2_mesh_1m_bvh"),
):
  if p not in sys.path:
    sys.path.insert(0, p)


def embree_closest_hit(verts, faces, ro, rd):
  """Closest-hit distances (inf on miss) and triangle ids (-1 on miss) from Embree."""
  import numpy as np
  from embreex import rtcore_scene as rtcs
  from embreex.mesh_construction import TriangleMesh

  scene = rtcs.EmbreeScene()
  TriangleMesh(scene, np.ascontiguousarray(verts[faces], dtype=np.float32))
  res = scene.run(
      np.ascontiguousarray(ro, dtype=np.float32),
      np.ascontiguousarray(rd, dtype=np.float32),
      output=1,
  )
  prim = np.asarray(res["primID"], dtype=np.int64)
  geom = np.asarray(res["geomID"], dtype=np.int64)
  t = np.asarray(res["tfar"], dtype=np.float32).copy()
  miss = geom < 0
  t[miss] = np.inf
  prim[miss] = -1
  return t, prim

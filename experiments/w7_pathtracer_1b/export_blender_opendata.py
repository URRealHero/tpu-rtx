"""Blender 5.3.0 Headless Depsgraph Exporter + Cycles CPU Reference GT for OpenData Scenes."""

import json
import math
import os
import sys
import time

import bpy
import numpy as np


def _extract_mat_props(mat):
  """Inspect Blender material node tree for PBR base_color/tex, roughness, metallic/glass/emissive."""
  alb = np.array([0.7, 0.7, 0.7], dtype=np.float32)
  rgh = 0.5
  mcode = 0.0
  tex_img = None

  if mat is None:
    return alb, rgh, mcode, tex_img

  if not mat.use_nodes or mat.node_tree is None:
    dc = getattr(mat, "diffuse_color", (0.7, 0.7, 0.7, 1.0))
    return np.array(dc[:3], dtype=np.float32), float(getattr(mat, "roughness", 0.5)), float(getattr(mat, "metallic", 0.0)), None

  nodes = mat.node_tree.nodes
  # Find output-connected shader or Principled BSDF
  princ = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
  emis = next((n for n in nodes if n.type == "EMISSION"), None)
  glass = next((n for n in nodes if n.type == "BSDF_GLASS"), None)
  glossy = next((n for n in nodes if n.type == "BSDF_GLOSSY"), None)
  diff = next((n for n in nodes if n.type == "BSDF_DIFFUSE"), None)

  def _trace_color_and_tex(sock):
    nonlocal tex_img
    c = np.array(sock.default_value[:3], dtype=np.float32)
    if sock.is_linked:
      fn = sock.links[0].from_node
      if fn.type == "TEX_IMAGE" and fn.image is not None:
        tex_img = fn.image
        c = np.array([1.0, 1.0, 1.0], dtype=np.float32)
      elif fn.type in ("MIX", "MIX_RGB"):
        for inp in fn.inputs:
          if inp.type == "RGBA":
            if inp.is_linked and inp.links[0].from_node.type == "TEX_IMAGE" and inp.links[0].from_node.image is not None:
              tex_img = inp.links[0].from_node.image
              c = np.array([1.0, 1.0, 1.0], dtype=np.float32)
              break
            c = np.array(inp.default_value[:3], dtype=np.float32)
      elif fn.type == "RGB":
        c = np.array(fn.outputs[0].default_value[:3], dtype=np.float32)
      elif fn.type == "VALTORGB":
        elems = fn.color_ramp.elements
        if len(elems) > 0:
          c = np.array(elems[len(elems) // 2].color[:3], dtype=np.float32)
    return np.clip(c, 0.02, 0.99)

  if princ is not None:
    if "Base Color" in princ.inputs:
      alb = _trace_color_and_tex(princ.inputs["Base Color"])
    if "Roughness" in princ.inputs:
      rgh = float(np.clip(princ.inputs["Roughness"].default_value, 0.02, 0.98))
    if "Metallic" in princ.inputs:
      mcode = float(np.clip(princ.inputs["Metallic"].default_value, 0.0, 1.0))
    trans_sock = princ.inputs.get("Transmission Weight", princ.inputs.get("Transmission", None))
    if trans_sock is not None and float(trans_sock.default_value) > 0.4:
      mcode = 2.0
      rgh = min(rgh, 0.06)
    em_str_sock = princ.inputs.get("Emission Strength", None)
    em_col_sock = princ.inputs.get("Emission Color", princ.inputs.get("Emission", None))
    em_s = float(em_str_sock.default_value) if em_str_sock is not None else 0.0
    if em_s > 0.5 and em_col_sock is not None:
      ec = np.array(em_col_sock.default_value[:3], dtype=np.float32)
      if np.max(ec) > 0.05:
        alb = np.clip(ec, 0.05, 1.0)
        mcode = 3.0 + min(em_s, 10.0) * 0.1
  elif emis is not None:
    alb = np.clip(np.array(emis.inputs["Color"].default_value[:3], dtype=np.float32), 0.05, 1.0)
    st = float(emis.inputs["Strength"].default_value)
    mcode = 3.0 + min(max(st, 1.0), 10.0) * 0.1
  elif glass is not None:
    alb = _trace_color_and_tex(glass.inputs["Color"])
    rgh = float(np.clip(glass.inputs["Roughness"].default_value, 0.02, 0.15))
    mcode = 2.0
  elif glossy is not None:
    alb = _trace_color_and_tex(glossy.inputs["Color"])
    rgh = float(np.clip(glossy.inputs["Roughness"].default_value, 0.03, 0.6))
    mcode = 0.95
  elif diff is not None:
    alb = _trace_color_and_tex(diff.inputs["Color"])
    rgh = 0.72
    mcode = 0.0

  return alb, rgh, mcode, tex_img


def main():
  argv = sys.argv
  idx = argv.index("--") + 1
  out_npz = argv[idx]
  gt_png = argv[idx + 1]
  gt_w = int(argv[idx + 2])
  gt_h = int(argv[idx + 3])
  gt_spp = int(argv[idx + 4])

  scene = bpy.context.scene
  depsgraph = bpy.context.evaluated_depsgraph_get()

  img_cache = {}

  def _get_img_arr(im):
    key = im.name
    if key not in img_cache:
      try:
        w, h = im.size[0], im.size[1]
        if w > 0 and h > 0 and len(im.pixels) == w * h * 4:
          arr = np.array(im.pixels[:], dtype=np.float32).reshape(h, w, 4)[..., :3]
          # Blender images are bottom-to-top
          img_cache[key] = arr[::-1].copy()
        else:
          img_cache[key] = None
      except Exception:
        img_cache[key] = None
    return img_cache[key]

  all_v, all_f, all_n, all_alb, all_rgh, all_mc = [], [], [], [], [], []
  v_off = 0

  for inst in depsgraph.object_instances:
    obj = inst.object
    if obj.type != "MESH":
      continue
    # Skip hidden render objects
    if obj.hide_render:
      continue
    me = obj.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    if me is None or len(me.polygons) == 0:
      obj.to_mesh_clear()
      continue
    me.calc_loop_triangles()
    n_verts = len(me.vertices)
    n_tris = len(me.loop_triangles)
    if n_verts == 0 or n_tris == 0:
      obj.to_mesh_clear()
      continue

    verts_loc = np.empty((n_verts, 3), dtype=np.float32)
    me.vertices.foreach_get("co", verts_loc.ravel())

    mat_w = np.array(inst.matrix_world, dtype=np.float32)
    R = mat_w[:3, :3]
    t = mat_w[:3, 3]
    verts_w = verts_loc @ R.T + t[None, :]
    R_n = np.linalg.inv(R).T if abs(np.linalg.det(R)) > 1e-8 else R

    tri_v = np.empty((n_tris * 3,), dtype=np.int32)
    me.loop_triangles.foreach_get("vertices", tri_v)
    faces = tri_v.reshape(n_tris, 3)
    if np.linalg.det(R) < 0:
      faces = faces[:, [0, 2, 1]]

    # Smooth vertex normals from loops/verts
    vnorm_loc = np.empty((n_verts, 3), dtype=np.float32)
    me.vertices.foreach_get("normal", vnorm_loc.ravel())
    vnorm_w = vnorm_loc @ R_n.T
    vnorm_w = vnorm_w / np.maximum(np.linalg.norm(vnorm_w, axis=-1, keepdims=True), 1e-12)

    tri_mat_idx = np.empty((n_tris,), dtype=np.int32)
    me.loop_triangles.foreach_get("material_index", tri_mat_idx)

    uv_layer = me.uv_layers.active.data if me.uv_layers.active is not None else None
    tri_loops = None
    uvs_flat = None
    if uv_layer is not None:
      n_loops = len(me.loops)
      uvs_flat = np.empty((n_loops * 2,), dtype=np.float32)
      uv_layer.foreach_get("uv", uvs_flat)
      uvs_flat = uvs_flat.reshape(n_loops, 2)
      tri_loops = np.empty((n_tris * 3,), dtype=np.int32)
      me.loop_triangles.foreach_get("loops", tri_loops)
      tri_loops = tri_loops.reshape(n_tris, 3)

    alb_f = np.full((n_tris, 3), 0.70, dtype=np.float32)
    rgh_f = np.full((n_tris,), 0.50, dtype=np.float32)
    mc_f = np.full((n_tris,), 0.0, dtype=np.float32)

    mats = list(me.materials) if len(me.materials) > 0 else [None]
    for m_i, mat in enumerate(mats):
      mask = (tri_mat_idx == m_i) if len(mats) > 1 else np.ones((n_tris,), dtype=bool)
      if not np.any(mask):
        continue
      m_alb, m_rgh, m_mc, m_tex = _extract_mat_props(mat)
      rgh_f[mask] = m_rgh
      mc_f[mask] = m_mc
      tex_arr = _get_img_arr(m_tex) if (m_tex is not None and tri_loops is not None) else None
      if tex_arr is not None:
        th, tw, _ = tex_arr.shape
        lp_sub = tri_loops[mask]
        uv_c = (uvs_flat[lp_sub[:, 0]] + uvs_flat[lp_sub[:, 1]] + uvs_flat[lp_sub[:, 2]]) * (1.0 / 3.0)
        u = np.mod(uv_c[:, 0], 1.0)
        v = np.mod(1.0 - uv_c[:, 1], 1.0)
        px = np.clip((u * (tw - 1)).astype(np.int32), 0, tw - 1)
        py = np.clip((v * (th - 1)).astype(np.int32), 0, th - 1)
        alb_f[mask] = np.clip(tex_arr[py, px], 0.02, 0.99)
      else:
        alb_f[mask] = m_alb[None, :]

    all_v.append(verts_w)
    all_f.append(faces + v_off)
    all_n.append(vnorm_w.astype(np.float32))
    all_alb.append(alb_f)
    all_rgh.append(rgh_f)
    all_mc.append(mc_f)
    v_off += n_verts
    obj.to_mesh_clear()

  v_cat = np.concatenate(all_v, axis=0)
  f_cat = np.concatenate(all_f, axis=0)
  n_cat = np.concatenate(all_n, axis=0)
  a_cat = np.concatenate(all_alb, axis=0)
  r_cat = np.concatenate(all_rgh, axis=0)
  m_cat = np.concatenate(all_mc, axis=0)

  # Extract Camera
  cam_obj = scene.camera
  cam_mat = np.array(cam_obj.matrix_world, dtype=np.float32)
  cam_pos = cam_mat[:3, 3]
  cam_xmat = cam_mat[:3, :3]
  cam_data = cam_obj.data
  # Compute vertical FOV in degrees
  res_x = int(scene.render.resolution_x)
  res_y = int(scene.render.resolution_y)
  res_pct = int(scene.render.resolution_percentage)
  if cam_data.sensor_fit == "VERTICAL":
    fovy_rad = 2.0 * math.atan(0.5 * cam_data.sensor_height / cam_data.lens)
  else:
    aspect = float(res_x) / max(float(res_y), 1.0)
    fovx_rad = 2.0 * math.atan(0.5 * cam_data.sensor_width / cam_data.lens)
    fovy_rad = 2.0 * math.atan(math.tan(0.5 * fovx_rad) / aspect)
  cam_fovy = float(math.degrees(fovy_rad))

  # Extract Lights
  lights = []
  for obj in bpy.data.objects:
    if obj.type != "LIGHT" or obj.hide_render:
      continue
    ld = obj.data
    mw = np.array(obj.matrix_world, dtype=np.float32)
    l_dir = -mw[:3, 2]
    l_dir = l_dir / max(float(np.linalg.norm(l_dir)), 1e-12)
    lights.append({
        "name": obj.name,
        "type": ld.type,
        "pos": mw[:3, 3].tolist(),
        "dir": l_dir.tolist(),
        "color": list(ld.color),
        "energy": float(ld.energy),
        "size": float(getattr(ld, "size", 0.5)),
    })

  np.savez(
      out_npz,
      verts=v_cat,
      faces=f_cat,
      vnorm=n_cat,
      tri_albedo=a_cat,
      tri_roughness=r_cat,
      tri_mat_code=m_cat,
      cam_pos=cam_pos,
      cam_xmat=cam_xmat,
      cam_fovy=np.float32(cam_fovy),
      native_res=np.array([res_x, res_y, res_pct], dtype=np.int32),
      lights_json=np.bytes_(json.dumps(lights)),
  )
  print(f"EXPORTED {out_npz}: verts={len(v_cat)} tris={len(f_cat)} fovy={cam_fovy:.2f} native={res_x}x{res_y}@{res_pct}% lights={len(lights)}", flush=True)

  # Render Cycles 5.3.0 CPU GT reference
  scene.render.engine = "CYCLES"
  scene.cycles.device = "CPU"
  scene.cycles.samples = gt_spp
  scene.cycles.use_denoising = False
  scene.render.resolution_x = gt_w
  scene.render.resolution_y = gt_h
  scene.render.resolution_percentage = 100
  scene.render.image_settings.file_format = "PNG"
  scene.render.filepath = gt_png
  t0 = time.perf_counter()
  bpy.ops.render.render(write_still=True)
  print(f"CYCLES GT {gt_png} ({gt_w}x{gt_h} @ {gt_spp} spp) rendered in {time.perf_counter()-t0:.2f}s", flush=True)


if __name__ == "__main__":
  main()

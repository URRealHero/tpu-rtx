"""Wave 3 Benchmark, Multi-Scene Parity, HLO Audit & 300k-400k+ E2E / 1M+ Render Push on w0 TPU v5p."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from typing import Any, Dict, List, Tuple

import jax
try:
  from jax import shard_map
except ImportError:
  from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from gate import evaluate_stage1
from tests.test_primitives_parity import mujoco_c_multiray_reference
import tpu_rtx
from tpu_mjx_render.renderer import RenderSceneSpec
from tpu_mjx_render.scenes import get_scene_xml


HUMANOID_21DOF_XML = """<mujoco model="humanoid_21dof">
  <option timestep="0.005" iterations="3" ls_iterations="4" integrator="implicitfast">
    <flag eulerdamp="disable"/>
  </option>
  <visual>
    <headlight ambient="0.25 0.25 0.25" diffuse="0.45 0.45 0.45" specular="0.15 0.15 0.15"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.28 0.46 0.68" rgb2="0.05 0.08 0.12" width="128" height="128"/>
    <texture name="grid" type="2d" builtin="checker" mark="edge" rgb1="0.22 0.30 0.38" rgb2="0.14 0.19 0.26" markrgb="0.75 0.78 0.82" width="128" height="128"/>
    <material name="grid" texture="grid" texuniform="true" texrepeat="4 4" reflectance="0.2"/>
    <material name="body_orange" rgba="0.88 0.45 0.18 1" specular="0.4" shininess="0.35"/>
    <material name="joint_dark" rgba="0.20 0.22 0.26 1" specular="0.5" shininess="0.4"/>
    <material name="limb_silver" rgba="0.78 0.82 0.86 1" specular="0.45" shininess="0.35"/>
  </asset>
  <default>
    <joint armature="0.02" damping="2.0" limited="true"/>
    <geom condim="3" friction="0.9 0.02 0.002" solref="0.02 1"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  <worldbody>
    <light name="sun" pos="1.5 -1.5 3.5" dir="-0.3 0.3 -1" directional="false" castshadow="true" diffuse="0.9 0.88 0.84" specular="0.3 0.3 0.3"/>
    <camera name="front" pos="2.2 -1.5 1.30" xyaxes="0.57 0.82 0 -0.22 0.15 0.96" fovy="45"/>
    <camera name="side" pos="0.0 -2.6 1.20" xyaxes="1 0 0 0 0.2 0.98" fovy="45"/>
    <geom name="floor" type="plane" pos="0 0 0" size="5 5 0.1" material="grid"/>
    <geom name="pedestal_block" type="box" pos="0.85 0.45 0.15" size="0.18 0.18 0.15" rgba="0.22 0.62 0.85 1" contype="0" conaffinity="0"/>
    <geom name="arena_sphere" type="sphere" pos="-0.75 0.65 0.22" size="0.22" rgba="0.86 0.24 0.22 1" contype="0" conaffinity="0"/>
    <geom name="arena_cyl" type="cylinder" pos="0.55 -0.75 0.20" size="0.14 0.20" rgba="0.24 0.78 0.44 1" contype="0" conaffinity="0"/>
    <body name="torso" pos="0 0 1.28">
      <freejoint name="root"/>
      <geom name="torso_geom" type="capsule" fromto="0 -0.07 0 0 0.07 0" size="0.095" material="body_orange" mass="5.0"/>
      <geom name="head" type="sphere" pos="0 0 0.21" size="0.105" material="limb_silver" mass="1.5"/>
      <geom name="uwaist" type="capsule" fromto="-0.01 -0.06 -0.12 -0.01 0.06 -0.12" size="0.075" material="joint_dark" mass="2.0"/>
      <body name="lwaist" pos="-0.01 0 -0.26">
        <geom name="lwaist_geom" type="capsule" fromto="0 -0.06 0 0 0.06 0" size="0.070" material="body_orange" mass="2.0"/>
        <joint name="abdomen_z" type="hinge" axis="0 0 1" range="-45 45"/>
        <joint name="abdomen_y" type="hinge" axis="0 1 0" range="-75 30"/>
        <body name="pelvis" pos="0 0 -0.165">
          <joint name="abdomen_x" type="hinge" axis="1 0 0" range="-35 35"/>
          <geom name="butt" type="capsule" fromto="-0.02 -0.08 0 -0.02 0.08 0" size="0.08" material="joint_dark" mass="3.0"/>
          <body name="right_thigh" pos="0 -0.10 -0.04">
            <joint name="right_hip_x" type="hinge" axis="1 0 0" range="-25 5"/>
            <joint name="right_hip_z" type="hinge" axis="0 0 1" range="-60 35"/>
            <joint name="right_hip_y" type="hinge" axis="0 1 0" range="-110 20"/>
            <geom name="right_thigh_g" type="capsule" fromto="0 0 0 0 0.01 -0.34" size="0.06" material="limb_silver" mass="3.0"/>
            <body name="right_shin" pos="0 0.01 -0.40">
              <joint name="right_knee" type="hinge" axis="0 -1 0" range="-160 -2"/>
              <geom name="right_shin_g" type="capsule" fromto="0 0 0 0 0 -0.30" size="0.049" material="body_orange" mass="2.0"/>
              <body name="right_foot" pos="0 0 -0.36">
                <joint name="right_ankle_x" type="hinge" axis="1 0 0" range="-45 45"/>
                <joint name="right_ankle_y" type="hinge" axis="0 1 0" range="-50 50"/>
                <geom name="right_foot_g" type="box" pos="0.04 0 -0.02" size="0.09 0.045 0.025" material="joint_dark" mass="1.0"/>
              </body>
            </body>
          </body>
          <body name="left_thigh" pos="0 0.10 -0.04">
            <joint name="left_hip_x" type="hinge" axis="-1 0 0" range="-25 5"/>
            <joint name="left_hip_z" type="hinge" axis="0 0 -1" range="-60 35"/>
            <joint name="left_hip_y" type="hinge" axis="0 1 0" range="-110 20"/>
            <geom name="left_thigh_g" type="capsule" fromto="0 0 0 0 -0.01 -0.34" size="0.06" material="limb_silver" mass="3.0"/>
            <body name="left_shin" pos="0 -0.01 -0.40">
              <joint name="left_knee" type="hinge" axis="0 -1 0" range="-160 -2"/>
              <geom name="left_shin_g" type="capsule" fromto="0 0 0 0 0 -0.30" size="0.049" material="body_orange" mass="2.0"/>
              <body name="left_foot" pos="0 0 -0.36">
                <joint name="left_ankle_x" type="hinge" axis="1 0 0" range="-45 45"/>
                <joint name="left_ankle_y" type="hinge" axis="0 1 0" range="-50 50"/>
                <geom name="left_foot_g" type="box" pos="0.04 0 -0.02" size="0.09 0.045 0.025" material="joint_dark" mass="1.0"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="right_upper_arm" pos="0 -0.17 0.06">
        <joint name="right_shoulder1" type="hinge" axis="2 1 1" range="-85 60"/>
        <joint name="right_shoulder2" type="hinge" axis="0 -1 1" range="-85 60"/>
        <geom name="right_uarm_g" type="capsule" fromto="0 0 0 0.16 -0.16 -0.16" size="0.042" material="limb_silver" mass="1.2"/>
        <body name="right_lower_arm" pos="0.18 -0.18 -0.18">
          <joint name="right_elbow" type="hinge" axis="0 -1 1" range="-90 50"/>
          <geom name="right_larm_g" type="capsule" fromto="0.01 0.01 0.01 0.17 0.17 0.17" size="0.034" material="body_orange" mass="0.8"/>
          <geom name="right_hand" type="sphere" pos="0.18 0.18 0.18" size="0.042" material="joint_dark" mass="0.3"/>
        </body>
      </body>
      <body name="left_upper_arm" pos="0 0.17 0.06">
        <joint name="left_shoulder1" type="hinge" axis="2 -1 1" range="-60 85"/>
        <joint name="left_shoulder2" type="hinge" axis="0 1 1" range="-60 85"/>
        <geom name="left_uarm_g" type="capsule" fromto="0 0 0 0.16 0.16 -0.16" size="0.042" material="limb_silver" mass="1.2"/>
        <body name="left_lower_arm" pos="0.18 0.18 -0.18">
          <joint name="left_elbow" type="hinge" axis="0 -1 -1" range="-90 50"/>
          <geom name="left_larm_g" type="capsule" fromto="0.01 -0.01 0.01 0.17 -0.17 0.17" size="0.034" material="body_orange" mass="0.8"/>
          <geom name="left_hand" type="sphere" pos="0.18 -0.18 0.18" size="0.042" material="joint_dark" mass="0.3"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="abdomen_y" gear="100" joint="abdomen_y"/>
    <motor name="abdomen_z" gear="100" joint="abdomen_z"/>
    <motor name="abdomen_x" gear="100" joint="abdomen_x"/>
    <motor name="right_hip_x" gear="100" joint="right_hip_x"/>
    <motor name="right_hip_z" gear="100" joint="right_hip_z"/>
    <motor name="right_hip_y" gear="200" joint="right_hip_y"/>
    <motor name="right_knee" gear="150" joint="right_knee"/>
    <motor name="right_ankle_x" gear="50" joint="right_ankle_x"/>
    <motor name="right_ankle_y" gear="50" joint="right_ankle_y"/>
    <motor name="left_hip_x" gear="100" joint="left_hip_x"/>
    <motor name="left_hip_z" gear="100" joint="left_hip_z"/>
    <motor name="left_hip_y" gear="200" joint="left_hip_y"/>
    <motor name="left_knee" gear="150" joint="left_knee"/>
    <motor name="left_ankle_x" gear="50" joint="left_ankle_x"/>
    <motor name="left_ankle_y" gear="50" joint="left_ankle_y"/>
    <motor name="right_shoulder1" gear="35" joint="right_shoulder1"/>
    <motor name="right_shoulder2" gear="35" joint="right_shoulder2"/>
    <motor name="right_elbow" gear="35" joint="right_elbow"/>
    <motor name="left_shoulder1" gear="35" joint="left_shoulder1"/>
    <motor name="left_shoulder2" gear="35" joint="left_shoulder2"/>
    <motor name="left_elbow" gear="35" joint="left_elbow"/>
  </actuator>
</mujoco>
"""


STATIC_MESH_TEST_XML = """<mujoco model="static_mesh_octa">
  <asset>
    <mesh name="octa" vertex="0 0 0.35  0.32 0 0  0 0.32 0  -0.32 0 0  0 -0.32 0  0 0 -0.35"
          face="0 1 2  0 2 3  0 3 4  0 4 1  5 2 1  5 3 2  5 4 3  5 1 4"/>
  </asset>
  <worldbody>
    <light name="sun" pos="1.0 -1.0 3.0" directional="false" castshadow="true" diffuse="0.9 0.9 0.9"/>
    <camera name="main" pos="0 -2.4 1.0" xyaxes="1 0 0 0 0.38 0.925" fovy="45"/>
    <geom name="floor" type="plane" pos="0 0 0" size="3 3 0.1" rgba="0.32 0.38 0.45 1"/>
    <body name="octa_body" pos="0 0 0.55">
      <freejoint/>
      <geom name="octa_geom" type="mesh" mesh="octa" rgba="0.90 0.42 0.16 1" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""


def _save_payload(path: str, payload: Dict[str, Any]) -> None:
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  tmp = path + ".tmp"
  with open(tmp, "w") as f:
    json.dump(payload, f, indent=2)
  os.replace(tmp, path)


def _load_model(scene_name: str, iterations: int | None = None, ls_iterations: int | None = None):
  if scene_name == "humanoid_21dof":
    xml = HUMANOID_21DOF_XML
  elif scene_name == "static_mesh_octa":
    xml = STATIC_MESH_TEST_XML
  else:
    xml = get_scene_xml(scene_name)
  m = mujoco.MjModel.from_xml_string(xml)
  if iterations is not None:
    m.opt.iterations = int(iterations)
  if ls_iterations is not None:
    m.opt.ls_iterations = int(ls_iterations)
  mx = mjx.put_model(m, impl="jax")
  return m, mx


def _make_batch_data(m, mx, batch_size: int, scene_name: str, n_step_perturb: int = 0):
  dx0 = mjx.make_data(m, impl="jax")
  d_host = mujoco.MjData(m)
  if m.nkey > 0:
    mujoco.mj_resetDataKeyframe(m, d_host, 0)
  else:
    mujoco.mj_resetData(m, d_host)
  dx0 = dx0.replace(qpos=jnp.asarray(d_host.qpos, jnp.float32))
  dx0 = mjx.forward(mx, dx0)

  dx_batch = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[None, ...], (batch_size, *x.shape)).copy(), dx0)
  if m.nu > 0 and n_step_perturb > 0:
    phases = jnp.linspace(-0.4, 0.4, batch_size, dtype=jnp.float32)[:, None]
    u = phases * jnp.sin(jnp.arange(1, m.nu + 1, dtype=jnp.float32))[None, :]
    step_fn = jax.jit(lambda d, ctrl: jax.vmap(lambda di, ui: mjx.step(mx, di.replace(ctrl=ui)))(d, ctrl))
    for _ in range(n_step_perturb):
      dx_batch = step_fn(dx_batch, u)
  return dx0, dx_batch


def run_sdk_and_parity_suite(out_json: str, payload: Dict[str, Any]) -> None:
  print("=== Running Wave 3 Universal SDK + Multi-Scene Parity + HLO Audit ===", flush=True)
  scenes_to_check = ["panda_pick_cube", "cartpole", "humanoid_21dof", "all_primitives"]
  parity_results: Dict[str, Any] = {}
  visual_samples: Dict[str, np.ndarray] = {}

  rng = np.random.default_rng(42)
  B = 16
  H, W = 128, 128

  for sname in scenes_to_check:
    m, mx = _load_model(sname)
    spec = RenderSceneSpec.from_mjmodel(m, cam_id=0, height=H, width=W, use_shadows=True)

    # Build B=16 diverse poses and compute C mj_multiRay ground truth
    depth_refs, nrm_refs, gid_refs = [], [], []
    qpos_list = []
    d = mujoco.MjData(m)
    dx_list = []
    for i in range(B):
      if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, i % m.nkey)
      else:
        mujoco.mj_resetData(m, d)
      if sname == "cartpole":
        d.qpos[0] = rng.uniform(-1.1, 1.1)
        d.qpos[1] = rng.uniform(-0.7, 0.7)
      elif sname == "panda_pick_cube":
        d.qpos[:7] += rng.uniform(-0.25, 0.25, size=7)
        d.qpos[7:9] = rng.uniform(0.0, 0.04)
        d.qpos[9] += rng.uniform(-0.12, 0.12)
        d.qpos[10] += rng.uniform(-0.15, 0.15)
      elif sname == "humanoid_21dof":
        d.qpos[7:] += rng.uniform(-0.28, 0.28, size=m.nq - 7)
        d.qpos[0] += rng.uniform(-0.15, 0.15)
        d.qpos[1] += rng.uniform(-0.15, 0.15)
      mujoco.mj_forward(m, d)
      dr, nr, gr = mujoco_c_multiray_reference(m, d, 0, H, W, spec)
      depth_refs.append(dr)
      nrm_refs.append(nr)
      gid_refs.append(gr)
      if m.nq > 0:
        qpos_list.append(d.qpos.copy())
      else:
        dx_list.append((
            d.geom_xpos.copy(),
            d.geom_xmat.copy().reshape(m.ngeom, 3, 3),
            d.cam_xpos.copy(),
            d.cam_xmat.copy().reshape(m.ncam, 3, 3),
            d.xpos.copy(),
            d.xmat.copy().reshape(m.nbody, 3, 3),
        ))

    dx0 = mjx.make_data(m, impl="jax")
    dx_b = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[None, ...], (B, *x.shape)).copy(), dx0)
    if m.nq > 0:
      dx_b = dx_b.replace(qpos=jnp.asarray(np.stack(qpos_list), jnp.float32))
      with jax.default_matmul_precision("highest"):
        dx_b = jax.jit(jax.vmap(lambda x: mjx.forward(mx, x)))(dx_b)
    else:
      dx_b = dx_b.replace(
          geom_xpos=jnp.asarray(np.stack([x[0] for x in dx_list]), jnp.float32),
          geom_xmat=jnp.asarray(np.stack([x[1] for x in dx_list]), jnp.float32),
          cam_xpos=jnp.asarray(np.stack([x[2] for x in dx_list]), jnp.float32),
          cam_xmat=jnp.asarray(np.stack([x[3] for x in dx_list]), jnp.float32),
          xpos=jnp.asarray(np.stack([x[4] for x in dx_list]), jnp.float32),
          xmat=jnp.asarray(np.stack([x[5] for x in dx_list]), jnp.float32),
      )

    # Step multiple times to generate real motion for prev_data optical flow
    prev_b = dx_b
    if m.nu > 0 and m.nv > 0:
      u_b = jnp.linspace(-0.45, 0.45, B, dtype=jnp.float32)[:, None] * jnp.ones((1, m.nu), dtype=jnp.float32)
      step_fn = jax.jit(lambda d, u: jax.vmap(lambda di, ui: mjx.step(mx, di.replace(ctrl=ui)))(d, u))
      curr_b = dx_b
      for _ in range(10):
        prev_b = curr_b
        curr_b = step_fn(curr_b, u_b)
    else:
      curr_b = dx_b

    dx_b = jax.device_put(dx_b, jax.devices()[0])
    prev_b = jax.device_put(prev_b, jax.devices()[0])
    curr_b = jax.device_put(curr_b, jax.devices()[0])

    depth_r = np.stack(depth_refs)
    nrm_r = np.stack(nrm_refs)
    gid_r = np.stack(gid_refs)

    scene_recs: Dict[str, Any] = {
        "nq": int(m.nq),
        "nv": int(m.nv),
        "nu": int(m.nu),
        "ngeom": int(m.ngeom),
    }

    for shader_name in ("fast_fused", "fused_ao_ggx"):
      # 1. Compile from MjModel (auto 4-chip sharding, fp32 parity check)
      sc = tpu_rtx.compile(m, shader=shader_name, res=(H, W), sharding="auto", dtype="float32")
      dx_s = tpu_rtx.shard_data(dx_b, sc)
      prev_s = tpu_rtx.shard_data(prev_b, sc)
      curr_s = tpu_rtx.shard_data(curr_b, sc)
      obs = tpu_rtx.render(dx_s, sc)
      jax.block_until_ready((obs["rgb"], obs["depth"], obs["normal"], obs["seg_id"], obs["flow"]))

      hlo_info = tpu_rtx.audit_hlo(tpu_rtx.lower_render(dx_s, sc))
      assert hlo_info["host_callbacks"] == 0, hlo_info
      assert hlo_info["send_recv_ops"] == 0, hlo_info
      assert hlo_info["gather_ops"] == 0, hlo_info

      depth_p = np.asarray(obs["depth"], dtype=np.float32)
      nrm_p = np.asarray(obs["normal"], dtype=np.float32)
      gid_p = np.asarray(obs["seg_id"], dtype=np.int32)

      valid = (gid_p >= 0) & (gid_r >= 0)
      same_g = valid & (gid_p == gid_r)
      depth_mae = float(np.mean(np.abs(depth_p[valid] - depth_r[valid])))
      depth_mae_same_geom = float(np.mean(np.abs(depth_p[same_g] - depth_r[same_g])))
      ndot = np.clip(np.sum(nrm_p[valid] * nrm_r[valid], axis=-1), -1.0, 1.0)
      nrm_deg = float(np.mean(np.degrees(np.arccos(ndot))))
      geom_match = float(np.mean(gid_p == gid_r))

      rec = {
          "depth_mae_m": depth_mae,
          "depth_mae_same_geom_m": depth_mae_same_geom,
          "normal_deg": nrm_deg,
          "geom_match": geom_match,
          "hlo": hlo_info,
      }
      if sname in ("cartpole", "panda_pick_cube"):
        g_res = evaluate_stage1(sname, depth_p, depth_r, nrm_p, nrm_r, gid_p, gid_r)
        rec["stage1_gate_passed"] = bool(g_res["passed"])

      # Test optical flow (prev_data & cvel)
      obs_flow = tpu_rtx.render(curr_s, sc, prev_data=prev_s)
      jax.block_until_ready(obs_flow["flow"])
      hlo_flow = tpu_rtx.audit_hlo(tpu_rtx.lower_render(curr_s, sc, prev_data=prev_s))
      assert hlo_flow["host_callbacks"] == 0 and hlo_flow["send_recv_ops"] == 0 and hlo_flow["gather_ops"] == 0
      rec["flow_mean_abs_px"] = float(np.mean(np.abs(np.asarray(obs_flow["flow"]))))
      rec["flow_hlo_zero_copy"] = bool(hlo_flow["zero_copy_verified"])

      scene_recs[shader_name] = rec
      print(f"PARITY [{sname} / {shader_name}]:", json.dumps(rec), flush=True)

      if shader_name == "fused_ao_ggx" and sname in ("panda_pick_cube", "cartpole", "humanoid_21dof"):
        visual_samples[f"{sname}_rgb"] = np.asarray(obs_flow["rgb"][0], dtype=np.float32)
        visual_samples[f"{sname}_depth"] = np.asarray(obs_flow["depth"][0], dtype=np.float32)
        visual_samples[f"{sname}_normal"] = np.asarray(obs_flow["normal"][0], dtype=np.float32)
        visual_samples[f"{sname}_seg_id"] = np.asarray(obs_flow["seg_id"][0], dtype=np.int32)
        visual_samples[f"{sname}_flow"] = np.asarray(obs_flow["flow"][0], dtype=np.float32)

    # Also verify compiling directly from mjx.Model, multi-cam, and unbatched single-env render
    sc_mx = tpu_rtx.compile(mx, shader="fused_ao_ggx", res=(64, 64), sharding="auto", dtype="bfloat16")
    obs_single = tpu_rtx.render(jax.tree_util.tree_map(lambda x: x[0], dx_b), sc_mx)
    jax.block_until_ready(obs_single["rgb"])
    scene_recs["mjx_model_compile_shape"] = list(obs_single["rgb"].shape)

    if m.ncam > 1:
      sc_mc = tpu_rtx.compile(m, shader="fast_fused", res=(64, 64), cam_id="all", sharding="auto", dtype="bfloat16")
      dx_mc = tpu_rtx.shard_data(dx_b, sc_mc)
      obs_mc = tpu_rtx.render(dx_mc, sc_mc)
      jax.block_until_ready(obs_mc["rgb"])
      hlo_mc = tpu_rtx.audit_hlo(tpu_rtx.lower_render(dx_mc, sc_mc))
      scene_recs["multi_cam_shape"] = list(obs_mc["rgb"].shape)
      scene_recs["multi_cam_hlo_zero_copy"] = bool(hlo_mc["zero_copy_verified"])

    parity_results[sname] = scene_recs

  # Static mesh scene check (octahedron mesh + analytic plane)
  m_mesh, mx_mesh = _load_model("static_mesh_octa")
  _, dx_mesh = _make_batch_data(m_mesh, mx_mesh, 8, "static_mesh_octa", n_step_perturb=3)
  sc_mesh = tpu_rtx.compile(m_mesh, shader="fast_fused", res=(64, 64), sharding="auto", dtype="bfloat16")
  dx_mesh_s = tpu_rtx.shard_data(dx_mesh, sc_mesh)
  obs_mesh = tpu_rtx.render(dx_mesh_s, sc_mesh)
  jax.block_until_ready(obs_mesh["rgb"])
  hlo_mesh = tpu_rtx.audit_hlo(tpu_rtx.lower_render(dx_mesh_s, sc_mesh))
  parity_results["static_mesh_octa"] = {
      "n_mesh_geoms": len(sc_mesh.mesh_geom_ids),
      "rgb_shape": list(obs_mesh["rgb"].shape),
      "hlo": hlo_mesh,
  }
  visual_samples["static_mesh_rgb"] = np.asarray(obs_mesh["rgb"][0], dtype=np.float32)

  os.makedirs("/tmp/zaychik-tpu-mjx-w0", exist_ok=True)
  np.savez_compressed("/tmp/zaychik-tpu-mjx-w0/v3_w0_wave3_samples.npz", **visual_samples)
  payload["parity_and_sdk"] = parity_results
  _save_payload(out_json, payload)


def _bench_render_only_config(
    scene_name: str,
    shader: str,
    res: int,
    n_chips: int,
    batch_size: int,
    block_b: int,
    block_r: int,
    dtype_str: str,
    outputs: Tuple[str, ...],
    n_scan_steps: int = 20,
) -> Dict[str, Any]:
  m, mx = _load_model(scene_name)
  _, dx_batch = _make_batch_data(m, mx, batch_size, scene_name, n_step_perturb=2)
  sc = tpu_rtx.compile(
      m,
      shader=shader,
      res=(res, res),
      sharding=n_chips,
      dtype=dtype_str,
      outputs=outputs,
      block_b=block_b,
      block_r=block_r,
  )
  dx_s = tpu_rtx.shard_data(dx_batch, sc)

  # Lower & audit HLO
  hlo_info = tpu_rtx.audit_hlo(tpu_rtx.lower_render(dx_s, sc))
  assert hlo_info["host_callbacks"] == 0 and hlo_info["send_recv_ops"] == 0 and hlo_info["gather_ops"] == 0, hlo_info

  # 1. Single-call loop (matches prior Wave 1/2 measurement style)
  for _ in range(3):
    obs = tpu_rtx.render(dx_s, sc)
  jax.block_until_ready(obs["rgb"])

  t0 = time.perf_counter()
  for _ in range(n_scan_steps):
    obs = tpu_rtx.render(dx_s, sc)
  jax.block_until_ready(obs["rgb"])
  dt_loop_ms = ((time.perf_counter() - t0) / n_scan_steps) * 1e3

  # 2. Pure on-device fori_loop render-only (zero Python host dispatch overhead, anti-LICM & anti-DCE)
  from tpu_mjx_render.adapters.mjx_universal import _render_local, _shard_map_kwargs
  if n_chips > 1:
    P = jax.sharding.PartitionSpec
    ax = sc.data_axis
    sm_kw = _shard_map_kwargs()
    def _scan_local(d_b):
      init_o = jax.tree_util.tree_map(lambda s: jnp.zeros(s.shape, s.dtype), jax.eval_shape(lambda d: _render_local(sc, d, None, None), d_b))
      def _step(i, acc):
        d_i = d_b.replace(cam_xpos=d_b.cam_xpos + i.astype(jnp.float32) * 1e-6)
        o_i = _render_local(sc, d_i, None, None)
        return jax.tree_util.tree_map(lambda a, b: a + b, acc, o_i)
      return jax.lax.fori_loop(0, n_scan_steps, _step, init_o)
    _sm = shard_map(_scan_local, mesh=sc.mesh, in_specs=(P(ax),), out_specs={k: P(ax) for k in sc.outputs}, **sm_kw)
    scan_fn = jax.jit(_sm)
  else:
    @jax.jit
    def scan_fn(d_b):
      init_o = jax.tree_util.tree_map(lambda s: jnp.zeros(s.shape, s.dtype), jax.eval_shape(lambda d: _render_local(sc, d, None, None), d_b))
      def _step(i, acc):
        d_i = d_b.replace(cam_xpos=d_b.cam_xpos + i.astype(jnp.float32) * 1e-6)
        o_i = _render_local(sc, d_i, None, None)
        return jax.tree_util.tree_map(lambda a, b: a + b, acc, o_i)
      return jax.lax.fori_loop(0, n_scan_steps, _step, init_o)

  for _ in range(2):
    obs2 = scan_fn(dx_s)
  jax.block_until_ready(obs2["rgb"])
  t1 = time.perf_counter()
  obs2 = scan_fn(dx_s)
  jax.block_until_ready(obs2["rgb"])
  dt_scan_ms = ((time.perf_counter() - t1) / n_scan_steps) * 1e3

  roof = tpu_rtx.roofline(sc, batch_size=batch_size, dt_render_ms=dt_scan_ms)
  rec = {
      "label": f"{n_chips}c_{res}x{res}_{shader}_b{block_b}_r{block_r}_{dtype_str}_B{batch_size}_{len(outputs)}out",
      "scene": scene_name,
      "shader": shader,
      "n_chips": n_chips,
      "batch_size": batch_size,
      "res": f"{res}x{res}",
      "block_b": block_b,
      "block_r": block_r,
      "dtype": dtype_str,
      "outputs": list(outputs),
      "dt_render_loop_ms": dt_loop_ms,
      "render_loop_fps": float(batch_size / (dt_loop_ms * 1e-3)),
      "dt_render_scan_ms": dt_scan_ms,
      "render_scan_fps": float(batch_size / (dt_scan_ms * 1e-3)),
      "roofline": roof,
      "hlo": hlo_info,
  }
  print("RENDER BENCH:", json.dumps({k: v for k, v in rec.items() if k != "hlo"}), flush=True)
  return rec


def _bench_e2e_rollout(
    label: str,
    scene_name: str,
    shader: str,
    res: int,
    n_chips: int,
    batch_size: int,
    block_b: int,
    block_r: int,
    dtype_str: str,
    outputs: Tuple[str, ...],
    iterations: int | None = None,
    ls_iterations: int | None = None,
    substeps: int = 1,
    n_rollout_steps: int = 20,
    bf16_matmul: bool = False,
) -> Dict[str, Any]:
  prev_prec = jax.config.jax_default_matmul_precision
  if bf16_matmul:
    jax.config.update("jax_default_matmul_precision", "bfloat16")
  try:
    m, mx = _load_model(scene_name, iterations=iterations, ls_iterations=ls_iterations)
    _, dx_batch = _make_batch_data(m, mx, batch_size, scene_name, n_step_perturb=1)
    sc = tpu_rtx.compile(
        m,
        shader=shader,
        res=(res, res),
        sharding=n_chips,
        dtype=dtype_str,
        outputs=outputs,
        block_b=block_b,
        block_r=block_r,
    )
    dx_s = tpu_rtx.shard_data(dx_batch, sc)
    ctrl_seq = jnp.zeros((batch_size, n_rollout_steps, m.nu), dtype=jnp.float32)
    ctrl_s = tpu_rtx.shard_data(ctrl_seq, sc)
    if sc.mesh is not None and n_chips > 1:
      mx_s = jax.device_put(mx, jax.sharding.NamedSharding(sc.mesh, jax.sharding.PartitionSpec()))
    else:
      mx_s = mx

    # Physics-only scan benchmark
    from tpu_mjx_render.adapters.mjx_universal import _shard_map_kwargs
    if sc.mesh is not None and n_chips > 1:
      P = jax.sharding.PartitionSpec
      ax = sc.data_axis
      sm_kw = _shard_map_kwargs()
      def _phys_scan(mx_l, d_b, c_btu):
        def _step_i(i, d_c):
          u_b = c_btu[:, i, :]
          if substeps == 1:
            return jax.vmap(lambda di, ui: mjx.step(mx_l, di.replace(ctrl=ui)))(d_c, u_b)
          return jax.lax.fori_loop(0, substeps, lambda _, ds: jax.vmap(lambda di, ui: mjx.step(mx_l, di.replace(ctrl=ui)))(ds, u_b), d_c)
        return jax.lax.fori_loop(0, n_rollout_steps, _step_i, d_b)
      phys_fn = jax.jit(shard_map(_phys_scan, mesh=sc.mesh, in_specs=(P(), P(ax), P(ax)), out_specs=P(ax), **sm_kw))
    else:
      @jax.jit
      def phys_fn(mx_l, d_b, c_btu):
        def _step_i(i, d_c):
          u_b = c_btu[:, i, :]
          if substeps == 1:
            return jax.vmap(lambda di, ui: mjx.step(mx_l, di.replace(ctrl=ui)))(d_c, u_b)
          return jax.lax.fori_loop(0, substeps, lambda _, ds: jax.vmap(lambda di, ui: mjx.step(mx_l, di.replace(ctrl=ui)))(ds, u_b), d_c)
        return jax.lax.fori_loop(0, n_rollout_steps, _step_i, d_b)

    e2e_fn = jax.jit(lambda mx_l, d_b, c_btu: tpu_rtx.rollout(mx_l, d_b, c_btu, sc, substeps=substeps, return_frames=False))
    hlo_e2e = tpu_rtx.audit_hlo(e2e_fn.lower(mx_s, dx_s, ctrl_s))
    assert hlo_e2e["host_callbacks"] == 0 and hlo_e2e["send_recv_ops"] == 0 and hlo_e2e["gather_ops"] == 0, hlo_e2e

    # Warmup
    for _ in range(2):
      dp = phys_fn(mx_s, dx_s, ctrl_s)
      de, obs_e = e2e_fn(mx_s, dx_s, ctrl_s)
    jax.block_until_ready((dp.qpos, de.qpos, obs_e["rgb"]))

    # Measure physics-only scan
    t0 = time.perf_counter()
    dp = phys_fn(mx_s, dx_s, ctrl_s)
    jax.block_until_ready(dp.qpos)
    dt_phys_ms = ((time.perf_counter() - t0) / n_rollout_steps) * 1e3

    # Measure fused E2E rollout scan
    t1 = time.perf_counter()
    de, obs_e = e2e_fn(mx_s, dx_s, ctrl_s)
    jax.block_until_ready((de.qpos, obs_e["rgb"]))
    dt_e2e_ms = ((time.perf_counter() - t1) / n_rollout_steps) * 1e3

    dt_render_ms = max(dt_e2e_ms - dt_phys_ms, 1e-6)
    e2e_fps = float(batch_size / (dt_e2e_ms * 1e-3))
    phys_substep_fps = float(batch_size * substeps / (dt_e2e_ms * 1e-3))
    render_equiv_fps = float(batch_size / (dt_render_ms * 1e-3))
    roof = tpu_rtx.roofline(sc, batch_size=batch_size, dt_render_ms=dt_render_ms)

    rec = {
        "label": label,
        "scene": scene_name,
        "shader": shader,
        "n_chips": n_chips,
        "batch_size": batch_size,
        "res": f"{res}x{res}",
        "iterations": int(m.opt.iterations),
        "ls_iterations": int(m.opt.ls_iterations),
        "substeps": substeps,
        "bf16_matmul": bf16_matmul,
        "dt_physics_ms": dt_phys_ms,
        "dt_render_ms": dt_render_ms,
        "dt_e2e_ms": dt_e2e_ms,
        "e2e_fps": e2e_fps,
        "e2e_phys_steps_per_sec": phys_substep_fps,
        "render_only_equiv_fps": render_equiv_fps,
        "physics_share_pct": 100.0 * dt_phys_ms / dt_e2e_ms,
        "roofline": roof,
        "hlo": hlo_e2e,
    }
    print(f"E2E [{label}]: e2e={e2e_fps:,.0f} FPS (phys={dt_phys_ms:.2f}ms, rend={dt_render_ms:.2f}ms, tot={dt_e2e_ms:.2f}ms, rend_equiv={render_equiv_fps:,.0f} FPS)", flush=True)
    return rec
  finally:
    jax.config.update("jax_default_matmul_precision", prev_prec)


def run_solver_parity_check() -> Dict[str, Any]:
  """Compare 20-step PandaPickCube rollout qpos parity between (5,8), (2,4), and bf16_matmul."""
  m_ref, mx_ref = _load_model("panda_pick_cube", iterations=5, ls_iterations=8)
  m_fast, mx_fast = _load_model("panda_pick_cube", iterations=2, ls_iterations=4)
  B = 128
  _, dx0 = _make_batch_data(m_ref, mx_ref, B, "panda_pick_cube", n_step_perturb=0)
  u = jnp.linspace(-0.25, 0.25, B, dtype=jnp.float32)[:, None] * jnp.ones((1, m_ref.nu), dtype=jnp.float32)

  def _roll(mx_l, d_b):
    step = jax.jit(lambda d, ctrl: jax.vmap(lambda di, ui: mjx.step(mx_l, di.replace(ctrl=ui)))(d, ctrl))
    dc = d_b
    for _ in range(20):
      dc = step(dc, u)
    return np.asarray(dc.qpos)

  q_ref = _roll(mx_ref, dx0)
  q_fast = _roll(mx_fast, dx0)
  jax.config.update("jax_default_matmul_precision", "bfloat16")
  q_bf16 = _roll(mx_fast, dx0)
  jax.config.update("jax_default_matmul_precision", "default")
  res = {
      "it2_ls4_qpos_mae": float(np.mean(np.abs(q_fast - q_ref))),
      "it2_ls4_qpos_max": float(np.max(np.abs(q_fast - q_ref))),
      "it2_ls4_bf16_qpos_mae": float(np.mean(np.abs(q_bf16 - q_ref))),
  }
  print("SOLVER PARITY:", json.dumps(res), flush=True)
  return res


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out_json", required=True)
  ap.add_argument("--step", default="all", choices=["all", "parity", "render_push", "e2e_push"])
  ap.add_argument("--e2e_start", type=int, default=0)
  ap.add_argument("--e2e_end", type=int, default=999)
  args = ap.parse_args()

  if os.path.exists(args.out_json):
    with open(args.out_json, "r") as f:
      payload = json.load(f)
  else:
    payload = {}

  if args.step in ("all", "parity"):
    run_sdk_and_parity_suite(args.out_json, payload)

  if args.step in ("all", "render_push"):
    print("=== Running Wave 3 Render-Only 1,000,000+ FPS Sweep ===", flush=True)
    render_runs = list(payload.get("render_push", []))
    done_labels = {r["label"] for r in render_runs}
    configs = [
        ("panda_pick_cube", "fast_fused", 64, 1, 2048, 16, 512, "bfloat16", ("rgb", "depth")),
        ("panda_pick_cube", "fast_fused", 64, 4, 8192, 16, 512, "bfloat16", ("rgb", "depth")),
        ("panda_pick_cube", "fast_fused", 64, 4, 8192, 8, 1024, "bfloat16", ("rgb", "depth")),
        ("panda_pick_cube", "fast_fused", 64, 4, 8192, 8, 2048, "bfloat16", ("rgb", "depth")),
        ("panda_pick_cube", "fast_fused", 64, 4, 16384, 16, 512, "bfloat16", ("rgb", "depth")),
        ("panda_pick_cube", "fast_fused", 64, 4, 16384, 8, 1024, "bfloat16", ("rgb", "depth")),
        ("panda_pick_cube", "fast_fused", 64, 4, 8192, 16, 512, "bfloat16", ("rgb", "depth", "normal", "seg_id", "flow")),
        ("panda_pick_cube", "fused_ao_ggx", 128, 4, 4096, 8, 512, "bfloat16", ("rgb", "depth", "normal", "seg_id", "flow")),
    ]
    for cfg in configs:
      sc_name, sh, rs, nc, bs, bb, br, dt_s, outs = cfg
      lbl = f"{nc}c_{rs}x{rs}_{sh}_b{bb}_r{br}_{dt_s}_B{bs}_{len(outs)}out"
      if lbl in done_labels:
        print(f"SKIP DONE render: {lbl}", flush=True)
        continue
      try:
        r = _bench_render_only_config(*cfg, n_scan_steps=20)
        render_runs.append(r)
        done_labels.add(lbl)
        payload["render_push"] = render_runs
        _save_payload(args.out_json, payload)
      except Exception as exc:
        print(f"SKIP render {cfg}: {exc}", flush=True)

  if args.step in ("all", "e2e_push"):
    print("=== Running Wave 3 E2E > 300k-400k+ FPS Push ===", flush=True)
    if "solver_parity" not in payload:
      payload["solver_parity"] = run_solver_parity_check()
      _save_payload(args.out_json, payload)
    e2e_runs = list(payload.get("e2e_push", []))
    done_e2e = {r["label"] for r in e2e_runs}

    e2e_specs = [
        (("panda_1c_64x64_std_B2048", "panda_pick_cube", "fast_fused", 64, 1, 2048, 16, 512, "bfloat16", ("rgb", "depth")), {}),
        (("panda_4c_64x64_std_scan_B8192", "panda_pick_cube", "fast_fused", 64, 4, 8192, 16, 512, "bfloat16", ("rgb", "depth")), {}),
        (("panda_4c_64x64_std_scan_B16384", "panda_pick_cube", "fast_fused", 64, 4, 16384, 16, 512, "bfloat16", ("rgb", "depth")), {}),
        (("panda_4c_64x64_it2ls4_B8192", "panda_pick_cube", "fast_fused", 64, 4, 8192, 16, 512, "bfloat16", ("rgb", "depth")), {"iterations": 2, "ls_iterations": 4}),
        (("panda_4c_64x64_it2ls4_B16384", "panda_pick_cube", "fast_fused", 64, 4, 16384, 16, 512, "bfloat16", ("rgb", "depth")), {"iterations": 2, "ls_iterations": 4}),
        (("panda_4c_64x64_it2ls4_bf16mm_B16384", "panda_pick_cube", "fast_fused", 64, 4, 16384, 16, 512, "bfloat16", ("rgb", "depth")), {"iterations": 2, "ls_iterations": 4, "bf16_matmul": True}),
        (("panda_4c_64x64_it1ls2_B16384", "panda_pick_cube", "fast_fused", 64, 4, 16384, 16, 512, "bfloat16", ("rgb", "depth")), {"iterations": 1, "ls_iterations": 2}),
        (("panda_4c_64x64_std_ar2_B16384", "panda_pick_cube", "fast_fused", 64, 4, 16384, 16, 512, "bfloat16", ("rgb", "depth")), {"substeps": 2}),
        (("cartpole_4c_64x64_B16384", "cartpole", "fast_fused", 64, 4, 16384, 16, 512, "bfloat16", ("rgb", "depth")), {}),
        (("humanoid_4c_64x64_B8192", "humanoid_21dof", "fast_fused", 64, 4, 8192, 16, 512, "bfloat16", ("rgb", "depth")), {}),
        (("panda_4c_128x128_ggx_ao_B4096", "panda_pick_cube", "fused_ao_ggx", 128, 4, 4096, 8, 512, "bfloat16", ("rgb", "depth", "normal", "seg_id", "flow")), {}),
    ]

    for pos_args, kw_args in e2e_specs[args.e2e_start : args.e2e_end]:
      lbl = pos_args[0]
      if lbl in done_e2e:
        print(f"SKIP DONE e2e: {lbl}", flush=True)
        continue
      rec = _bench_e2e_rollout(*pos_args, n_rollout_steps=10, **kw_args)
      e2e_runs.append(rec)
      done_e2e.add(lbl)
      payload["e2e_push"] = e2e_runs
      _save_payload(args.out_json, payload)

    best_1c_rfps = max((r["render_scan_fps"] for r in payload.get("render_push", []) if r["n_chips"] == 1), default=240000.0)
    payload["pod_scaling"] = tpu_rtx.verify_pod_topology(
        n_chips_list=(1, 4, 32, 64, 1000),
        per_chip_batch=4096,
        per_chip_render_fps=best_1c_rfps,
    )
    _save_payload(args.out_json, payload)
    print("Saved Wave 3 JSON:", args.out_json, flush=True)


if __name__ == "__main__":
  main()

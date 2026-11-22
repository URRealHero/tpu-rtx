"""TPU Hardware Roofline & XLA HLO Static Analyzer for TPU-MJX Batch Renderer."""

from __future__ import annotations

import dataclasses
import re
from typing import Dict


@dataclasses.dataclass(frozen=True)
class TPUHardwareSpec:
  name: str
  bf16_tflops: float      # Per physical chip
  vpu_fp32_tflops: float  # Per physical chip
  hbm_tb_s: float         # Per physical chip
  hbm_gib: float          # Usable HBM per chip
  vmem_mib: float         # On-chip SRAM per chip
  mxu_tile: int           # 128 for v5p, 256 for v6e

  @property
  def ridge_bf16(self) -> float:
    return (self.bf16_tflops * 1e12) / (self.hbm_tb_s * 1e12)

  @property
  def ridge_vpu(self) -> float:
    return (self.vpu_fp32_tflops * 1e12) / (self.hbm_tb_s * 1e12)


# Verified hardware specs from UMP tpu-infra-accel-skill/reference/01-tpu-hardware-and-roofline.md
TPU_SPECS: Dict[str, TPUHardwareSpec] = {
    "v5p": TPUHardwareSpec(
        name="TPU v5p (1 chip / 2 TC)",
        bf16_tflops=459.0,
        vpu_fp32_tflops=28.0,
        hbm_tb_s=2.765,
        hbm_gib=95.74,
        vmem_mib=64.0,
        mxu_tile=128,
    ),
    "v6e": TPUHardwareSpec(
        name="TPU v6e Trillium (1 chip / 1 TC)",
        bf16_tflops=918.0,
        vpu_fp32_tflops=57.0,
        hbm_tb_s=1.64,
        hbm_gib=31.25,
        vmem_mib=128.0,
        mxu_tile=256,
    ),
}


def inspect_hlo_ops(hlo_text: str) -> Dict[str, int]:
  """Parse compiled/lowered HLO text for zero-copy and op-mix diagnostics."""
  return {
      "host_callbacks": len(re.findall(r"host_callback|HostCallback|python_callback", hlo_text, re.I)),
      "dot_ops": len(re.findall(r"\bdot\(", hlo_text)),
      "conv_ops": len(re.findall(r"\bconvolution\(", hlo_text)),
      "while_ops": len(re.findall(r"\bwhile\(", hlo_text)),
      "gather_ops": len(re.findall(r"\bgather\(", hlo_text)),
      "custom_calls": len(re.findall(r"\bcustom-call\(", hlo_text)),
      "hlo_lines": hlo_text.count("\n"),
  }


def estimate_stage1_roofline(
    hw: TPUHardwareSpec,
    batch_size: int,
    height: int,
    width: int,
    n_geoms: int = 32,
    use_shadows: bool = True,
    pallas_vmem: bool = True,
) -> Dict[str, float]:
  """Analytical Roofline model for Stage 1 Analytic Primitives (VPU + HBM)."""
  n_rays = batch_size * height * width * (2 if use_shadows else 1)
  # Per ray-geom analytic test ~65 FLOPs (mat^T transform + quad/slab + normal)
  vpu_flops = float(n_rays) * float(n_geoms) * 65.0

  if pallas_vmem:
    # Ray origins/dirs loaded once per tile + GBuffer written once: ~128 B/pixel
    hbm_bytes = float(batch_size * height * width) * 128.0 + float(batch_size * n_geoms) * 64.0
  else:
    # S1-Naive materializes intermediate [R, G] distances/normals/masks in HBM (~48 B per ray-geom)
    hbm_bytes = float(n_rays) * float(n_geoms) * 48.0 + float(batch_size * height * width) * 128.0

  t_vpu = vpu_flops / (hw.vpu_fp32_tflops * 1e12)
  t_hbm = hbm_bytes / (hw.hbm_tb_s * 1e12)
  t_stage1 = max(t_vpu, t_hbm)
  return {
      "vpu_gflops": vpu_flops / 1e9,
      "hbm_gb": hbm_bytes / 1e9,
      "ai_flop_per_byte": vpu_flops / max(hbm_bytes, 1.0),
      "t_vpu_ms": t_vpu * 1e3,
      "t_hbm_ms": t_hbm * 1e3,
      "t_stage1_ms": t_stage1 * 1e3,
      "bound": "VPU" if t_vpu >= t_hbm else "HBM",
  }


def estimate_stage2_roofline(
    hw: TPUHardwareSpec,
    batch_size: int,
    height: int,
    width: int,
    arch: str = "M2_spatial_128",
) -> Dict[str, float]:
  """Analytical Roofline model for Stage 2 MXU Neural Shader (BF16)."""
  n_pix = float(batch_size * height * width)
  if arch == "M1_pixel_mlp_128":
    # 4 layers of width 128: (32->128, 128->128, 128->128, 128->3) -> ~2*128^2*2.5 FLOPs/pix
    flops = n_pix * (2.0 * (32 * 128 + 128 * 128 + 128 * 128 + 128 * 3))
    # VMEM-resident weights: only read input G-buffer (64 B/pix) + write RGB (12 B/pix)
    hbm_bytes = n_pix * 76.0
  else:  # M2_spatial_128: 2-scale ConvNet (ch=64 -> 128 -> 64)
    # Scale 1 (H x W): 3x3(16->64) + 2x Res(64->64, 2 convs) + Upsample 1x1(192->64) + 1x1(64->3)
    f_s1 = n_pix * 2.0 * (9 * 16 * 64 + 4 * 9 * 64 * 64 + 192 * 64 + 64 * 3)
    # Scale 2 (H/2 x W/2): 3x3 s2(64->128) + 2x Res(128->128, 2 convs)
    f_s2 = (n_pix * 0.25) * 2.0 * (9 * 64 * 128 + 4 * 9 * 128 * 128)
    flops = f_s1 + f_s2
    # Intermediate activations in HBM for fused stages (BF16 2B/el)
    hbm_bytes = n_pix * 76.0 + n_pix * 2.0 * 64.0 * 4.0 + (n_pix * 0.25) * 2.0 * 128.0 * 4.0

  t_mxu = flops / (hw.bf16_tflops * 1e12)
  t_hbm = hbm_bytes / (hw.hbm_tb_s * 1e12)
  t_stage2 = max(t_mxu, t_hbm)
  return {
      "mxu_gflops": flops / 1e9,
      "hbm_gb": hbm_bytes / 1e9,
      "ai_flop_per_byte": flops / max(hbm_bytes, 1.0),
      "t_mxu_ms": t_mxu * 1e3,
      "t_hbm_ms": t_hbm * 1e3,
      "t_stage2_ms": t_stage2 * 1e3,
      "bound": "MXU" if t_mxu >= t_hbm else "HBM",
  }

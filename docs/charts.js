/* ==========================================================================
   TPU-RTX Vector SVG Charts & Architecture Diagrams
   All numbers verified verbatim from out/metrics/*.json
   Author: Feiyang Liu
   ========================================================================== */

window.TPU_RTX_CHARTS = (function () {
  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function renderSystemStack(el, lang) {
    const zh = lang === "zh";
    el.innerHTML = `
      <svg class="svg-chart" viewBox="0 0 960 358" role="img" aria-label="TPU-RTX 4-Layer System Stack Overview">
        <!-- Layer 4: Benchmarks & Applications -->
        <rect x="8" y="8" width="944" height="74" rx="8" fill="var(--bg-elev)" stroke="var(--hairline-strong)" />
        <text x="24" y="30" class="val-lbl" style="fill:var(--ok);font-size:12px;">
          ${zh ? "第 4 层 · 机器人、具身智能与图形基准" : "Layer 4 · Robotics, Embodied AI &amp; Graphics Benchmarks"}
        </text>
        <text x="24" y="51" class="sub-lbl" style="font-size:10.8px;fill:var(--text-secondary);">
          ${zh ? "具身仿真：RoboTwin (123.9k FPS) · RoboWild (4DGS × MJX) · MuJoCo Menagerie · 百万帧 VLA + GelSight" : "Embodied AI: RoboTwin (123.9k FPS) · RoboWild (4DGS × MJX) · MuJoCo Menagerie · VLA + GelSight"}
        </text>
        <text x="24" y="69" class="sub-lbl" style="font-size:10.8px;fill:var(--muted);">
          ${zh ? "感知与渲染：Vision RL · 双 128 线 LiDAR (63.88B/s) · Blender 8 场景 · Moana (223 亿面) · Mip-NeRF 360" : "Sensors &amp; Rendering: Vision RL · Dual 128-Beam LiDAR (63.88B/s) · Blender 8 Scenes · Moana (22.30B) · Mip-NeRF 360"}
        </text>

        <!-- Layer 3: 8 Adapters -->
        <rect x="8" y="94" width="944" height="84" rx="8" fill="var(--bg-elev)" stroke="var(--cyan)" />
        <text x="24" y="116" class="val-lbl" style="fill:var(--cyan);font-size:12px;">
          ${zh ? "第 3 层 · 8 个软件接口适配层 (tpu_mjx_render/adapters/)" : "Layer 3 · 8 Software Adapters (tpu_mjx_render/adapters/)"}
        </text>
        <rect x="24" y="126" width="220" height="40" rx="5" fill="var(--bg-card)" stroke="var(--hairline)" />
        <text x="34" y="143" class="val-lbl" style="font-size:10.5px;">mjx_universal + isaac_rep</text>
        <text x="34" y="158" class="sub-lbl" style="font-size:9.8px;">${zh ? "MuJoCo / 域随机化 / LiDAR / 触觉" : "MuJoCo / DR / LiDAR / Tactile"}</text>

        <rect x="254" y="126" width="220" height="40" rx="5" fill="var(--bg-card)" stroke="var(--hairline)" />
        <text x="264" y="143" class="val-lbl" style="font-size:10.5px;">blend_compiler + openusd</text>
        <text x="264" y="158" class="sub-lbl" style="font-size:9.8px;">${zh ? ".blend Cycles / OpenUSD 编译" : ".blend Cycles / OpenUSD Loader"}</text>

        <rect x="484" y="126" width="220" height="40" rx="5" fill="var(--bg-card)" stroke="var(--hairline)" />
        <text x="494" y="143" class="val-lbl" style="font-size:10.5px;">tpu_optix + tpu_gsplat</text>
        <text x="494" y="158" class="sub-lbl" style="font-size:9.8px;">${zh ? "OptiX 接口 / 布料 / 3DGS + 4DGS" : "OptiX API / Cloth / 3DGS + 4DGS"}</text>

        <rect x="714" y="126" width="222" height="40" rx="5" fill="var(--bg-card)" stroke="var(--hairline)" />
        <text x="724" y="143" class="val-lbl" style="font-size:10.5px;">tpu_dlss + tpu_restir</text>
        <text x="724" y="158" class="sub-lbl" style="font-size:9.8px;">${zh ? "DLSS 3.5 / 重打光 / ReSTIR" : "DLSS 3.5 / Relight / ReSTIR"}</text>

        <!-- Layer 2: 3 Zero-Copy Core Kernels -->
        <rect x="8" y="190" width="944" height="74" rx="8" fill="var(--bg-elev)" stroke="var(--accent)" />
        <text x="24" y="211" class="val-lbl" style="fill:var(--accent);font-size:12px;">
          ${zh ? "第 2 层 · 3 个零拷贝核心内核 (host_callbacks = 0, gather_ops = 0)" : "Layer 2 · 3 Zero-Copy Core Kernels (host_callbacks = 0, gather_ops = 0)"}
        </text>
        <rect x="24" y="221" width="292" height="34" rx="5" fill="var(--bg-card)" stroke="var(--hairline-strong)" />
        <text x="36" y="236" class="val-lbl" style="font-size:11px;">${zh ? "1. Wide-BVH-64 光线追踪与路径追踪" : "1. Wide-BVH-64 Ray &amp; Path Tracing"}</text>
        <text x="36" y="250" class="sub-lbl" style="font-size:10px;">${zh ? "3 级 64 叉 + VMEM 融合着色 + AgX" : "3-Level 64-Ary + In-VMEM Fusion + AgX"}</text>

        <rect x="334" y="221" width="292" height="34" rx="5" fill="var(--bg-card)" stroke="var(--hairline-strong)" />
        <text x="346" y="236" class="val-lbl" style="font-size:11px;">${zh ? "2. 分块 VMEM 3DGS / 4DGS 光栅器" : "2. Chunked-VMEM 3DGS / 4DGS Engine"}</text>
        <text x="346" y="250" class="sub-lbl" style="font-size:10px;">${zh ? "无动态 Gather · 32 芯 10.62 亿高斯" : "Zero Gather · Up to 1.062B Splats (32c)"}</text>

        <rect x="644" y="221" width="292" height="34" rx="5" fill="var(--bg-card)" stroke="var(--hairline-strong)" />
        <text x="656" y="236" class="val-lbl" style="font-size:11px;">${zh ? "3. MXU 神经重建 + jax.grad 可微闭环" : "3. MXU Neural + jax.grad Diff Physics"}</text>
        <text x="656" y="250" class="sub-lbl" style="font-size:10px;">${zh ? "4K 降噪/DLSS/重打光 + 像素反传力矩" : "4K Denoise/DLSS/Relight + Pixel-to-Torque"}</text>

        <!-- Layer 1: Google Cloud TPU v5p Hardware -->
        <rect x="8" y="276" width="944" height="74" rx="8" fill="var(--accent-soft)" stroke="var(--accent)" />
        <text x="24" y="299" class="val-lbl" style="fill:var(--accent);font-size:12px;">
          ${zh ? "第 1 层 · Google Cloud TPU v5p 硬件（单芯 → 4 芯 383 GB → 32 芯 3.064 TB HBM）" : "Layer 1 · Google Cloud TPU v5p Hardware (1-Chip → 4-Chip 383 GB → 32-Chip 3.064 TB HBM)"}
        </text>
        <text x="24" y="321" class="sub-lbl" style="font-size:11px;fill:var(--text);">
          ${zh ? "459 TFLOP/s BF16/INT8 脉动阵列 MXU  ·  64 MiB 片上 VMEM  ·  95.74 GiB HBM/芯  ·  3D Torus ICI" : "459 TFLOP/s BF16/INT8 Systolic MXU  ·  64 MiB SW-Managed VMEM  ·  95.74 GiB HBM/chip  ·  3D Torus ICI"}
        </text>
        <text x="24" y="339" class="sub-lbl" style="font-size:10.5px;fill:var(--text-secondary);">
          ${zh ? "物理仿真、光线追踪、传感器建模与策略训练同驻单张 XLA 计算图，不回传 CPU。" : "Physics, ray tracing, tactile/LiDAR simulation &amp; RL/VLA gradients share one XLA HLO graph with zero CPU copies."}
        </text>
      </svg>
    `;
  }

  function renderGpuVsTpu(el, lang) {
    const zh = lang === "zh";
    el.innerHTML = `
      <svg class="svg-chart" viewBox="0 0 960 390" role="img" aria-label="GPU vs TPU Architecture Comparison">
        <defs>
          <marker id="gvt-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="var(--accent)" />
          </marker>
          <marker id="gvt-c" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="var(--cyan)" />
          </marker>
        </defs>

        <!-- LEFT: GPU -->
        <rect x="8" y="8" width="458" height="374" rx="8" fill="var(--bg-elev)" stroke="var(--hairline-strong)" />
        <rect x="8" y="8" width="458" height="34" rx="8" fill="var(--cyan-soft)" />
        <text x="24" y="30" class="val-lbl" style="fill:var(--cyan);font-size:12px;">
          ${zh ? "NVIDIA GPU (SIMT Warp + 硬件 RT Core)" : "NVIDIA GPU (SIMT Warps + Hardware RT Core)"}
        </text>

        <rect x="24" y="54" width="426" height="62" rx="6" fill="var(--bg-card)" stroke="var(--hairline-strong)" />
        <text x="38" y="74" class="val-lbl">${zh ? "硬件特征：适配动态分支与随机访问" : "Silicon: Built for Branching & Random Access"}</text>
        <text x="38" y="92" class="sub-lbl">32-Thread SIMT Warps · Dedicated RT Cores · HW L1/L2 Cache</text>
        <text x="38" y="106" class="sub-lbl">${zh ? "硬件 BVH 遍历与原子写，单卡显存通常 24–80 GB" : "Hardware BVH traversal & atomic tile scatter; 24–80 GB VRAM per GPU"}</text>

        <rect x="24" y="132" width="426" height="44" rx="5" fill="var(--bg-card)" stroke="var(--hairline)" />
        <text x="38" y="151" class="val-lbl">1. ${zh ? "单光线 SIMT Warp（允许不规则分歧与递归）" : "Per-Ray SIMT Warps (Dynamic Branching)"}</text>
        <text x="38" y="167" class="sub-lbl">${zh ? "每条光线可独立提前终止，无固定向量长度约束" : "Threads diverge freely across arbitrary ray bounces"}</text>
        <line x1="237" y1="176" x2="237" y2="192" stroke="var(--cyan)" stroke-width="1.5" marker-end="url(#gvt-c)" />

        <rect x="24" y="194" width="426" height="44" rx="5" fill="var(--bg-card)" stroke="var(--hairline)" />
        <text x="38" y="213" class="val-lbl">2. ${zh ? "指针跳转式二叉 BVH（依赖硬件 L1/L2 缓存）" : "Pointer-Chasing Binary BVH (HW L1/L2 Cache)"}</text>
        <text x="38" y="229" class="sub-lbl">${zh ? "硬件缓存平滑吸收不规则读取与原子累加" : "Hardware cache absorbs irregular pointer chasing & atomics"}</text>
        <line x1="237" y1="238" x2="237" y2="254" stroke="var(--cyan)" stroke-width="1.5" marker-end="url(#gvt-c)" />

        <rect x="24" y="256" width="426" height="52" rx="5" fill="var(--bg-card)" stroke="var(--hairline)" />
        <text x="38" y="275" class="val-lbl">3. ${zh ? "跨工具链桥接 (PhysX / OptiX / CUDA PyTorch)" : "Cross-Stack Bridges (OptiX / Isaac / CUDA PyTorch)"}</text>
        <text x="38" y="291" class="sub-lbl">${zh ? "交互式视口工具链成熟；多引擎 RL 常需缓冲交换" : "Mature viewport & Omniverse stack; multi-engine RL bridges needed"}</text>

        <rect x="24" y="320" width="426" height="48" rx="5" fill="var(--cyan-soft)" stroke="var(--cyan)" />
        <text x="38" y="340" class="val-lbl" style="fill:var(--cyan);">${zh ? "适用场景：复杂动态分支、交互式 DCC 视口、单卡 Cycles" : "Strong at: Dynamic Branching, Viewports & Single-Card Cycles"}</text>
        <text x="38" y="357" class="sub-lbl">${zh ? "约束：高批量具身 RL 或数亿级高斯会受限于单卡显存" : "Constraint: High-batch RL + 300M splats hit per-device VRAM limits"}</text>

        <!-- RIGHT: GOOGLE TPU v5p -->
        <rect x="494" y="8" width="458" height="374" rx="8" fill="var(--bg-elev)" stroke="var(--accent)" />
        <rect x="494" y="8" width="458" height="34" rx="8" fill="var(--accent-soft)" />
        <text x="510" y="30" class="val-lbl" style="fill:var(--accent);font-size:12px;">
          ${zh ? "TPU-RTX on TPU v5p (脉动阵列 MXU + 64 MiB VMEM)" : "TPU-RTX on TPU v5p (Systolic MXU + 64 MiB VMEM)"}
        </text>

        <rect x="510" y="54" width="426" height="62" rx="6" fill="var(--bg-card)" stroke="var(--accent)" />
        <text x="524" y="74" class="val-lbl">${zh ? "硬件特征：459 TFLOP/s BF16 MXU + 95.74 GiB HBM/芯" : "Silicon: 459 TFLOP/s BF16 MXU + 95.74 GiB HBM/chip"}</text>
        <text x="524" y="92" class="sub-lbl">128×128 MXU · 8×128 VPU · 64 MiB SW VMEM · 2.77 TB/s HBM</text>
        <text x="524" y="106" class="sub-lbl">${zh ? "脊点 166 FLOP/B，无硬件 RT 缓存 → Pallas 零 Gather 调度" : "166 FLOP/B ridge point · No HW RT cache → Zero-Gather Pallas"}</text>

        <rect x="510" y="132" width="426" height="44" rx="5" fill="var(--bg-card)" stroke="var(--hairline)" />
        <text x="524" y="151" class="val-lbl">1. ${zh ? "批量 (B, R) SoA 光线块对齐 (8×128)" : "Batched (B, R) SoA Ray Tiles Aligned to (8×128)"}</text>
        <text x="524" y="167" class="sub-lbl">${zh ? "并行环境与相机射线规整排布并送入 VPU 向量通道" : "Parallel simulation worlds pack cleanly into vector lanes"}</text>
        <line x1="723" y1="176" x2="723" y2="192" stroke="var(--accent)" stroke-width="1.5" marker-end="url(#gvt-a)" />

        <rect x="510" y="194" width="426" height="44" rx="5" fill="var(--bg-card)" stroke="var(--hairline)" />
        <text x="524" y="213" class="val-lbl">2. ${zh ? "64 叉 Wide-BVH + pltpu.async_copy (gather_ops = 0)" : "3-Level 64-Ary Wide-BVH + async_copy (0 Gather)"}</text>
        <text x="524" y="229" class="sub-lbl">${zh ? "单次向量指令测试 64 个 AABB，叶节点 DMA 流式进出 VMEM" : "Tests 64 boxes per VPU op; leaf streams stay inside 64 MiB VMEM"}</text>
        <line x1="723" y1="238" x2="723" y2="254" stroke="var(--accent)" stroke-width="1.5" marker-end="url(#gvt-a)" />

        <rect x="510" y="256" width="426" height="52" rx="5" fill="var(--bg-card)" stroke="var(--hairline)" />
        <text x="524" y="275" class="val-lbl">3. ${zh ? "单图融合：物理 + 渲染 + DLSS + VLA + 梯度" : "One XLA Graph: Physics + Render + DLSS + VLA + Grad"}</text>
        <text x="524" y="291" class="sub-lbl">${zh ? "host_callbacks=0：G-Buffer 留在 HBM 内直接喂给 PPO 或 jax.grad" : "host_callbacks=0: G-buffers stay in HBM for MXU & differentiable control"}</text>

        <rect x="510" y="320" width="426" height="48" rx="5" fill="var(--accent-soft)" stroke="var(--accent)" />
        <text x="524" y="340" class="val-lbl" style="fill:var(--accent);">${zh ? "适用场景：大显存场景、266k–411k FPS RL、88 FPS 4K 去噪、jax.grad" : "Strong at: High-HBM Scenes, 266k–411k FPS RL, 4K MXU, jax.grad"}</text>
        <text x="524" y="357" class="sub-lbl">${zh ? "约束：XLA AOT 冷编译 4–130 s；不规则动态 Gather 性能骤降" : "Constraint: 4–130 s AOT compile time; uncoalesced HBM gathers stall"}</text>
      </svg>
    `;
  }

  function renderEvolution(el, lang) {
    const zh = lang === "zh";
    const W = 960, H = 315, padL = 62, padR = 56, padT = 30, padB = 48;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const minL = 2.4, maxL = 6.18;

    const runs = [
      { id: "Naive",  fps: 453.3,   best: 453.3,   lbl: "453",          note: zh ? "初版 XLA" : "Naive XLA",    neg: false, dy: -11 },
      { id: "Tile64", fps: 854.3,   best: 854.3,   lbl: "854",          note: "Tile=64",                      neg: false, dy: -11 },
      { id: "Spill",  fps: 616.7,   best: 854.3,   lbl: "617 (−24%)",   note: zh ? "512 溢出" : "512 Spill",   neg: true,  dy: 20 },
      { id: "Pal1",   fps: 4883.3,  best: 4883.3,  lbl: "4,883",        note: zh ? "初版 Pallas" : "Pallas v1",neg: false, dy: -11 },
      { id: "Flat",   fps: 4845.0,  best: 4883.3,  lbl: "4,845 (flat)", note: zh ? "持平试验" : "Flat Pass", neg: true,  dy: 20 },
      { id: "NoGat",  fps: 12843.5, best: 12843.5, lbl: "12.8k",        note: zh ? "去 Gather" : "0-Gather",   neg: false, dy: -11 },
      { id: "4c",     fps: 50131.0, best: 50131.0, lbl: "50.1k",        note: zh ? "4 芯并行" : "4-Chip",      neg: false, dy: -11 },
      { id: "Fused",  fps: 266209,  best: 266209,  lbl: "266,209",      note: zh ? "VMEM 融合" : "VMEM Fused", neg: false, dy: -11 },
      { id: "Equiv",  fps: 921309,  best: 921309,  lbl: "921,309",      note: zh ? "纯渲染等效" : "Pure Render",neg: false, dy: -11 }
    ];

    const sx = (i) => padL + (i / (runs.length - 1)) * plotW;
    const sy = (v) => padT + ((maxL - Math.log10(v)) / (maxL - minL)) * plotH;

    const grid = [300, 1000, 10000, 100000, 1000000].map((v) => {
      const y = sy(v);
      const t = v >= 1e6 ? "1M" : (v >= 1e3 ? (v / 1e3) + "k" : String(v));
      return `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${W - padR}" y2="${y.toFixed(1)}" class="grid-line" />
              <text x="${padL - 8}" y="${(y + 4).toFixed(1)}" text-anchor="end" class="ax-lbl">${t}</text>`;
    }).join("");

    let envPts = [];
    runs.forEach((r, i) => {
      const x = sx(i), y = sy(r.best);
      if (i > 0) envPts.push(`${x.toFixed(1)},${sy(runs[i - 1].best).toFixed(1)}`);
      envPts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
    });

    const ptsSvg = runs.map((r, i) => {
      const x = sx(i), y = sy(r.fps);
      const col = r.neg ? "var(--warn)" : (i >= 7 ? "var(--cyan)" : "var(--accent)");
      const anchor = i === runs.length - 1 ? "middle" : (i === 0 ? "start" : "middle");
      return `
        <circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.neg ? 4.8 : 5.4}" fill="${col}" stroke="var(--bg-card)" stroke-width="1.6" />
        <text x="${x.toFixed(1)}" y="${(y + r.dy).toFixed(1)}" text-anchor="${anchor}" class="val-lbl" style="fill:${col};font-size:11px">${r.lbl}</text>
        <text x="${x.toFixed(1)}" y="${H - 14}" text-anchor="${anchor}" class="sub-lbl" style="font-size:10.5px">${r.note}</text>
      `;
    }).join("");

    el.innerHTML = `
      <svg class="svg-chart" viewBox="0 0 ${W} ${H}">
        ${grid}
        <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${H - padB}" class="ax-line" />
        <line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" class="ax-line" />
        <polyline fill="none" stroke="var(--accent)" stroke-width="2.5" points="${envPts.join(" ")}" />
        ${ptsSvg}
        <rect x="${padL + 10}" y="8" width="14" height="3.5" fill="var(--accent)" />
        <text x="${padL + 30}" y="13" class="legend-lbl" style="font-size:11.5px">${zh ? "最佳吞吐包络 (Log10 FPS)" : "Frontier Envelope (Log10 FPS)"}</text>
        <circle cx="${padL + 245}" cy="10" r="4.5" fill="var(--warn)" />
        <text x="${padL + 256}" y="13" class="legend-lbl" style="font-size:11.5px">${zh ? "负结果 / 持平试验" : "Negative / Flat Run"}</text>
      </svg>
    `;
  }

  function renderQualityOrbit(el, lang) {
    const zh = lang === "zh";
    const W = 960, H = 300, padL = 62, padR = 26, padT = 34, padB = 50;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const scenes = [
      { name: "BMW27 (0.7M)",        hero: 32.14, orbit: 37.65, zmae: "0.9813 SSIM" },
      { name: "Monster SSS (1.6M)",  hero: 35.34, orbit: 36.98, zmae: "0.12 mm Z-MAE" },
      { name: "Junkshop (OpenData)", hero: 30.36, orbit: 30.36, zmae: "Orbit 30.36 dB" },
      { name: "Barbershop (8.55M)",  hero: 26.98, orbit: 31.00, zmae: "141 Textures" },
      { name: "Lone Monk (5.61M)",   hero: 26.31, orbit: 28.83, zmae: "20.1M Inst" },
      { name: "Fishy Cat (0.91M)",   hero: 26.27, orbit: 28.80, zmae: "7.2k Fur" },
      { name: "Barcelona (54.91M)",  hero: 25.65, orbit: 27.45, zmae: "15.9 mm Z-MAE" }
    ];
    const minY = 20, maxY = 40;
    const sy = (db) => padT + ((maxY - db) / (maxY - minY)) * plotH;

    const grid = [22, 26, 30, 34, 38].map((v) => {
      const y = sy(v);
      const isGate = v === 30;
      return `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${W - padR}" y2="${y.toFixed(1)}" class="grid-line" style="${isGate ? "stroke:var(--ok);stroke-dasharray:6 4;" : ""}" />
              <text x="${padL - 8}" y="${(y + 4).toFixed(1)}" text-anchor="end" class="ax-lbl" style="${isGate ? "fill:var(--ok);font-weight:700;" : ""}">${v} dB</text>`;
    }).join("");

    const gw = plotW / scenes.length;
    const bars = scenes.map((s, i) => {
      const cx = padL + i * gw + gw / 2;
      const bw = 32;
      const hHero = (s.hero - minY) / (maxY - minY) * plotH;
      const hOrb = (s.orbit - minY) / (maxY - minY) * plotH;
      return `
        <rect x="${(cx - bw - 4).toFixed(1)}" y="${sy(s.hero).toFixed(1)}" width="${bw}" height="${hHero.toFixed(1)}" rx="3" fill="var(--cyan)" opacity="0.88" />
        <text x="${(cx - bw / 2 - 5).toFixed(1)}" y="${(sy(s.hero) - 6).toFixed(1)}" text-anchor="middle" class="val-lbl" style="font-size:10.5px">${s.hero}</text>
        <rect x="${(cx + 4).toFixed(1)}" y="${sy(s.orbit).toFixed(1)}" width="${bw}" height="${hOrb.toFixed(1)}" rx="3" fill="var(--accent)" opacity="0.92" />
        <text x="${(cx + bw / 2 + 5).toFixed(1)}" y="${(sy(s.orbit) - 6).toFixed(1)}" text-anchor="middle" class="val-lbl" style="fill:var(--accent);font-size:10.5px">${s.orbit}</text>
        <text x="${cx.toFixed(1)}" y="${H - 24}" text-anchor="middle" class="val-lbl" style="font-size:10.5px">${esc(s.name)}</text>
        <text x="${cx.toFixed(1)}" y="${H - 8}" text-anchor="middle" class="sub-lbl" style="font-size:9.5px">${esc(s.zmae)}</text>
      `;
    }).join("");

    el.innerHTML = `
      <svg class="svg-chart" viewBox="0 0 ${W} ${H}">
        ${grid}
        <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${H - padB}" class="ax-line" />
        <line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" class="ax-line" />
        <rect x="${padL + 8}" y="8" width="11" height="11" rx="2" fill="var(--cyan)" />
        <text x="${padL + 24}" y="17" class="legend-lbl" style="font-size:11px">${zh ? "主视角 Hero PSNR (dB)" : "Hero Camera PSNR (dB)"}</text>
        <rect x="${padL + 205}" y="8" width="11" height="11" rx="2" fill="var(--accent)" />
        <text x="${padL + 221}" y="17" class="legend-lbl" style="font-size:11px">${zh ? "留出环绕视角 Orbit PSNR" : "Held-Out Orbit PSNR"}</text>
        <text x="${W - padR}" y="17" text-anchor="end" class="legend-lbl" style="fill:var(--ok);font-size:11px">${zh ? "--- 30 dB 参考线" : "--- 30 dB Reference"}</text>
        ${bars}
      </svg>
    `;
  }

  function renderLogBarChart(el, cfg) {
    const W = 960, H = 300, padL = 240, padR = 120, padT = 22, padB = 38;
    const plotW = W - padL - padR;
    const minLog = cfg.minLog, maxLog = cfg.maxLog;
    const rowH = (H - padT - padB) / cfg.bars.length;

    const ticks = cfg.ticks.map((v) => {
      const x = padL + ((Math.log10(v) - minLog) / (maxLog - minLog)) * plotW;
      const lbl = v >= 1e6 ? (v / 1e6) + "M" : (v >= 1e3 ? (v / 1e3) + "k" : String(v));
      return `<line x1="${x.toFixed(1)}" y1="${padT}" x2="${x.toFixed(1)}" y2="${H - padB}" class="grid-line" />
              <text x="${x.toFixed(1)}" y="${H - 12}" text-anchor="middle" class="ax-lbl" style="font-size:10.5px">${lbl}</text>`;
    }).join("");

    const bars = cfg.bars.map((b, i) => {
      const y = padT + i * rowH + rowH * 0.18, bh = rowH * 0.62;
      const w = Math.max(6, ((Math.log10(b.val) - minLog) / (maxLog - minLog)) * plotW);
      const fill = b.warn ? "var(--warn)" : (b.cyan ? "var(--cyan)" : "var(--accent)");
      const op = b.base ? "0.48" : "0.92";
      return `<text x="${padL - 12}" y="${(y + bh * 0.66).toFixed(1)}" text-anchor="end" class="ax-lbl" style="fill:var(--text-secondary);font-size:11.5px">${esc(b.label)}</text>
              <rect x="${padL}" y="${y.toFixed(1)}" width="${w.toFixed(1)}" height="${bh.toFixed(1)}" rx="3" fill="${fill}" opacity="${op}" />
              <text x="${(padL + w + 8).toFixed(1)}" y="${(y + bh * 0.66).toFixed(1)}" class="val-lbl" style="${b.warn ? "fill:var(--warn);" : ""}font-size:11.5px">${esc(b.valText)}</text>`;
    }).join("");

    el.innerHTML = `
      <svg class="svg-chart" viewBox="0 0 ${W} ${H}">
        ${ticks}
        <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${H - padB}" class="ax-line" />
        <line x1="${padL}" y1="${H - padB}" x2="${W - padR + 12}" y2="${H - padB}" class="ax-line" />
        ${bars}
      </svg>
    `;
  }

  function renderGs3dEvo(el, lang) {
    const zh = lang === "zh";
    renderLogBarChart(el, {
      minLog: 2.5,
      maxLog: 6.2,
      ticks: [500, 2000, 10000, 50000, 200000, 1000000],
      bars: [
        { label: zh ? "V2 Pallas MXU (负结果)" : "V2 Pallas (Slower)", val: 658.6, valText: "659 FPS (−1.81×)", warn: true },
        { label: zh ? "V2 jax_tile 基线 (1 芯)" : "V2 jax_tile (1c)", val: 1193.2, valText: "1,193 FPS", base: true },
        { label: zh ? "V3 BF16-MXU (1 芯)" : "V3 BF16-MXU (1c)", val: 35916, valText: "35,916 FPS" },
        { label: zh ? "V3 4 芯 shard_map 64²" : "V3 4c 64×64", val: 142008, valText: "142,008 FPS" },
        { label: zh ? "V3 32 芯 Pod 128²" : "V3 32c Pod 128²", val: 952248, valText: "952,248 FPS (32c)", cyan: true }
      ]
    });
  }

  function renderVmemCliff(el, lang) {
    const zh = lang === "zh";
    const W = 960, H = 300, padL = 70, padR = 72, padT = 32, padB = 44;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const pts = [
      { tile: 64, fps: 854.3, tag: "854.3 FPS" },
      { tile: 128, fps: 811.0, tag: "811.0 FPS" },
      { tile: 256, fps: 799.3, tag: "799.3 FPS" },
      { tile: 512, fps: 616.7, tag: "616.7 (−24%)" }
    ];
    const minY = 550, maxY = 900;
    const sx = (i) => padL + (i / (pts.length - 1)) * plotW;
    const sy = (fps) => padT + ((maxY - fps) / (maxY - minY)) * plotH;
    const poly = pts.map((p, i) => `${sx(i).toFixed(1)},${sy(p.fps).toFixed(1)}`).join(" ");

    const grid = [600, 700, 800, 900].map((v) => {
      const y = sy(v);
      return `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${W - padR}" y2="${y.toFixed(1)}" class="grid-line" />
              <text x="${padL - 10}" y="${(y + 4).toFixed(1)}" text-anchor="end" class="ax-lbl" style="font-size:11px">${v}</text>`;
    }).join("");

    const dots = pts.map((p, i) => {
      const x = sx(i), y = sy(p.fps), isCliff = p.tile === 512, color = isCliff ? "var(--warn)" : "var(--accent)";
      return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="6" fill="${color}" stroke="var(--bg-card)" stroke-width="2.2" />
              <text x="${x.toFixed(1)}" y="${(y - 13).toFixed(1)}" text-anchor="${i === 0 ? "start" : (i === 3 ? "end" : "middle")}" class="val-lbl" style="fill:${color};font-size:12px">${p.tag}</text>
              <text x="${x.toFixed(1)}" y="${H - 15}" text-anchor="middle" class="ax-lbl" style="font-size:11.5px">tile_size = ${p.tile}</text>`;
    }).join("");

    el.innerHTML = `
      <svg class="svg-chart" viewBox="0 0 ${W} ${H}">
        <rect x="${sx(2).toFixed(1)}" y="${padT}" width="${(sx(3) - sx(2)).toFixed(1)}" height="${plotH}" fill="var(--warn-soft)" />
        <text x="${((sx(2) + sx(3)) / 2).toFixed(1)}" y="${padT + 20}" text-anchor="middle" class="sub-lbl" style="fill:var(--warn);font-size:11.5px;font-weight:700">${zh ? "64 MiB VMEM 溢出" : "64 MiB VMEM Spill"}</text>
        ${grid}
        <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${H - padB}" class="ax-line" />
        <line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" class="ax-line" />
        <polyline fill="none" stroke="var(--accent)" stroke-width="2.8" points="${poly}" />
        ${dots}
      </svg>
    `;
  }

  function renderMxuPareto(el, lang) {
    const zh = lang === "zh";
    const W = 960, H = 300, padL = 64, padR = 32, padT = 34, padB = 46;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const items = [
      { name: "FP32", fps4k: 37.2, fps1080: 143.3, psnr: "20.276 dB" },
      { name: "BF16 (MXU)", fps4k: 78.6, fps1080: 307.4, psnr: "20.276 dB (2.11×)" },
      { name: "INT8 QAT", fps4k: 78.1, fps1080: 303.5, psnr: "20.229 dB (−0.047)" }
    ];
    const maxY = 335, sy = (v) => padT + ((maxY - v) / maxY) * plotH;
    const grid = [0, 100, 200, 300].map((v) => {
      const y = sy(v);
      return `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${W - padR}" y2="${y.toFixed(1)}" class="grid-line" />
              <text x="${padL - 10}" y="${(y + 4).toFixed(1)}" text-anchor="end" class="ax-lbl" style="font-size:11px">${v}</text>`;
    }).join("");

    const gw = plotW / items.length;
    const groups = items.map((it, i) => {
      const cx = padL + i * gw + gw / 2, bw = 56;
      const h1 = plotH - (sy(it.fps1080) - padT), h4 = plotH - (sy(it.fps4k) - padT);
      return `<rect x="${(cx - bw - 6).toFixed(1)}" y="${sy(it.fps1080).toFixed(1)}" width="${bw}" height="${h1.toFixed(1)}" rx="3" fill="var(--cyan)" opacity="0.88" />
              <text x="${(cx - bw / 2 - 6).toFixed(1)}" y="${(sy(it.fps1080) - 7).toFixed(1)}" text-anchor="middle" class="val-lbl" style="font-size:11.5px">${it.fps1080}</text>
              <rect x="${(cx + 6).toFixed(1)}" y="${sy(it.fps4k).toFixed(1)}" width="${bw}" height="${h4.toFixed(1)}" rx="3" fill="var(--accent)" opacity="0.92" />
              <text x="${(cx + bw / 2 + 6).toFixed(1)}" y="${(sy(it.fps4k) - 7).toFixed(1)}" text-anchor="middle" class="val-lbl" style="font-size:11.5px">${it.fps4k}</text>
              <text x="${cx.toFixed(1)}" y="${H - 22}" text-anchor="middle" class="val-lbl" style="font-size:12px">${it.name}</text>
              <text x="${cx.toFixed(1)}" y="${H - 7}" text-anchor="middle" class="sub-lbl" style="font-size:10.5px">${it.psnr}</text>`;
    }).join("");

    el.innerHTML = `
      <svg class="svg-chart" viewBox="0 0 ${W} ${H}">
        ${grid}
        <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${H - padB}" class="ax-line" />
        <line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" class="ax-line" />
        <rect x="${padL + 10}" y="8" width="11" height="11" rx="2" fill="var(--cyan)" />
        <text x="${padL + 26}" y="18" class="legend-lbl" style="font-size:11.5px">1080p FPS (4c)</text>
        <rect x="${padL + 150}" y="8" width="11" height="11" rx="2" fill="var(--accent)" />
        <text x="${padL + 166}" y="18" class="legend-lbl" style="font-size:11.5px">${zh ? "4K FPS (4 芯)" : "4K FPS (4c)"}</text>
        ${groups}
      </svg>
    `;
  }

  function renderTrainDiffPhys(el, lang) {
    const zh = lang === "zh";
    const W = 960, H = 310, padL = 64, padR = 64, padT = 34, padB = 42;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const pts = [
      [1, 17.55, 16.03], [10, 17.87, 8.65], [20, 21.52, 2.63], [30, 21.38, 4.15],
      [40, 21.38, 5.13], [50, 22.93, 3.76], [60, 24.92, 3.2], [70, 28.64, 2.5],
      [80, 31.1, 2.47], [81, 31.3, 2.47], [90, 31.11, 2.72], [100, 30.59, 2.88],
      [110, 30.87, 2.7], [120, 31.15, 2.71], [130, 31.66, 2.73], [140, 31.84, 2.56],
      [150, 32.34, 2.51], [160, 32.67, 2.46], [170, 33.28, 2.35], [180, 33.73, 2.19],
      [190, 34.13, 2.0], [200, 34.52, 1.88], [210, 34.98, 1.79], [220, 35.37, 1.71],
      [230, 35.67, 1.66], [240, 36.08, 1.63]
    ];
    const sx = (it) => padL + ((it - 1) / 239) * plotW;
    const syP = (db) => padT + ((38 - db) / (38 - 15)) * plotH;
    const syE = (cm) => padT + ((18 - cm) / 18) * plotH;

    const polyP = pts.map((p) => `${sx(p[0]).toFixed(1)},${syP(p[1]).toFixed(1)}`).join(" ");
    const polyE = pts.map((p) => `${sx(p[0]).toFixed(1)},${syE(p[2]).toFixed(1)}`).join(" ");

    const ygrid = [18, 24, 30, 36].map((v) => {
      const y = syP(v);
      return `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${W - padR}" y2="${y.toFixed(1)}" class="grid-line" />
              <text x="${padL - 8}" y="${(y + 4).toFixed(1)}" text-anchor="end" class="ax-lbl" style="fill:var(--accent);font-size:11px">${v} dB</text>`;
    }).join("");
    const rgrid = [0, 6, 12, 18].map((cm) =>
      `<text x="${W - padR + 8}" y="${(syE(cm) + 4).toFixed(1)}" class="ax-lbl" style="fill:var(--cyan);font-size:11px">${cm} cm</text>`
    ).join("");
    const xticks = [1, 60, 120, 180, 240].map((it) =>
      `<text x="${sx(it).toFixed(1)}" y="${H - 14}" text-anchor="middle" class="ax-lbl" style="font-size:11px">iter ${it}</text>`
    ).join("");

    el.innerHTML = `
      <svg class="svg-chart" viewBox="0 0 ${W} ${H}">
        ${ygrid}${rgrid}${xticks}
        <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${H - padB}" class="ax-line" />
        <line x1="${W - padR}" y1="${padT}" x2="${W - padR}" y2="${H - padB}" class="ax-line" />
        <line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" class="ax-line" />
        <polyline fill="none" stroke="var(--accent)" stroke-width="2.6" points="${polyP}" />
        <polyline fill="none" stroke="var(--cyan)" stroke-width="2.4" stroke-dasharray="6 4" points="${polyE}" />
        <circle cx="${sx(240).toFixed(1)}" cy="${syP(36.08).toFixed(1)}" r="5" fill="var(--accent)" />
        <text x="${(sx(240) - 8).toFixed(1)}" y="${(syP(36.08) - 10).toFixed(1)}" text-anchor="end" class="val-lbl" style="fill:var(--accent);font-size:12px">36.08 dB</text>
        <circle cx="${sx(240).toFixed(1)}" cy="${syE(1.63).toFixed(1)}" r="5" fill="var(--cyan)" />
        <text x="${(sx(240) - 8).toFixed(1)}" y="${(syE(1.63) - 10).toFixed(1)}" text-anchor="end" class="val-lbl" style="fill:var(--cyan);font-size:12px">1.63 cm</text>
        <rect x="${padL + 10}" y="9" width="14" height="3.5" fill="var(--accent)" />
        <text x="${padL + 30}" y="14" class="legend-lbl" style="font-size:11.5px">${zh ? "渲染 PSNR (dB)" : "Render PSNR (dB)"}</text>
        <rect x="${padL + 195}" y="9" width="14" height="3.5" fill="var(--cyan)" />
        <text x="${padL + 215}" y="14" class="legend-lbl" style="font-size:11.5px">${zh ? "末端 3D 误差 (cm)" : "End-Effector Error (cm)"}</text>
      </svg>
    `;
  }

  function renderTrainPpo(el, lang) {
    const zh = lang === "zh";
    const W = 960, H = 315;
    const tracks = [
      {
        title: zh ? "Cartpole 成功率 (656.4k env-steps/s, 评测 94.2%)" : "Cartpole Success Rate (656.4k env-steps/s, eval 94.2%)",
        pts: [[1,0.434],[8,0.281],[16,0.337],[24,0.438],[32,0.49],[40,0.568],[48,0.644],[56,0.718],[64,0.778],[72,0.793],[80,0.802],[88,0.799],[96,0.833],[104,0.851],[112,0.863],[120,0.880]],
        minY: 0.25, maxY: 0.92, endLbl: "0.434 → 0.880", color: "var(--ok)", yTop: 14
      },
      {
        title: zh ? "PandaPickCube 单步奖励 (173.5k env-steps/s, 回报 146.9 → 165.5)" : "PandaPickCube Mean Reward (173.5k env-steps/s, return 146.9 → 165.5)",
        pts: [[1,2.242],[10,2.196],[19,2.336],[28,2.408],[38,2.399],[47,2.44],[56,2.427],[65,2.489],[75,2.443],[84,2.509],[93,2.567],[102,2.59],[112,2.619],[121,2.564],[130,2.629],[140,2.641]],
        minY: 2.15, maxY: 2.68, endLbl: "2.242 → 2.641", color: "var(--accent)", yTop: 114
      },
      {
        title: zh ? "Humanoid-21DOF 单步奖励 (221.7k env-steps/s, 回报 52.9 → 111.7)" : "Humanoid-21DOF Mean Reward (221.7k env-steps/s, return 52.9 → 111.7)",
        pts: [[1,2.358],[10,2.378],[19,2.435],[28,2.445],[38,2.456],[47,2.485],[56,2.491],[65,2.516],[75,2.506],[84,2.514],[93,2.513],[102,2.524],[112,2.526],[121,2.547],[130,2.569],[140,2.576]],
        minY: 2.34, maxY: 2.60, endLbl: "2.358 → 2.576", color: "var(--cyan)", yTop: 214
      }
    ];

    const body = tracks.map((tr) => {
      const x0 = 20, x1 = 795, y0 = tr.yTop + 24, y1 = tr.yTop + 80;
      const maxIt = tr.pts[tr.pts.length - 1][0];
      const poly = tr.pts.map(([it, val]) => {
        const px = x0 + ((it - 1) / (maxIt - 1)) * (x1 - x0);
        const py = y1 - ((val - tr.minY) / (tr.maxY - tr.minY)) * (y1 - y0);
        return `${px.toFixed(1)},${py.toFixed(1)}`;
      }).join(" ");
      return `<text x="${x0}" y="${tr.yTop + 14}" class="sub-lbl" style="fill:var(--text-secondary);font-size:11.5px">${esc(tr.title)}</text>
              <rect x="${x0}" y="${y0}" width="${x1 - x0}" height="${y1 - y0}" rx="4" fill="var(--bg-elev)" stroke="var(--hairline)" />
              <polyline fill="none" stroke="${tr.color}" stroke-width="2.6" points="${poly}" />
              <text x="${x1 + 14}" y="${(y0 + y1) / 2 + 5}" class="val-lbl" style="fill:${tr.color};font-size:12px">${tr.endLbl}</text>`;
    }).join("");

    el.innerHTML = `<svg class="svg-chart" viewBox="0 0 ${W} ${H}">${body}</svg>`;
  }

  function renderTrain3dgs(el, lang) {
    const zh = lang === "zh";
    const W = 960, H = 300, padL = 64, padR = 32, padT = 30, padB = 42;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const pts = [
      [0,20.28],[1.07,26.86],[2.15,27.64],[3.22,28.56],[4.29,29.26],[5.36,29.67],[6.43,29.97],
      [7.5,30.3],[8.57,30.62],[9.63,30.75],[10.7,30.93],[11.77,31.02],[12.83,31.17],[13.9,31.0],
      [14.97,31.14],[16.04,31.17],[17.11,31.31],[18.18,30.93],[19.24,30.8],[20.31,30.84],
      [21.38,30.94],[22.44,30.84],[23.51,30.76],[24.58,30.52],[25.52,30.63]
    ];
    const sx = (s) => padL + (s / 26) * plotW;
    const sy = (db) => padT + ((32.8 - db) / (32.8 - 20)) * plotH;
    const poly = pts.map(([s, db]) => `${sx(s).toFixed(1)},${sy(db).toFixed(1)}`).join(" ");

    const grid = [20, 24, 28, 32].map((v) =>
      `<line x1="${padL}" y1="${sy(v).toFixed(1)}" x2="${W - padR}" y2="${sy(v).toFixed(1)}" class="grid-line" />
       <text x="${padL - 8}" y="${(sy(v) + 4).toFixed(1)}" text-anchor="end" class="ax-lbl" style="font-size:11px">${v} dB</text>`
    ).join("");
    const xticks = [0, 5, 10, 15, 20, 25].map((s) =>
      `<text x="${sx(s).toFixed(1)}" y="${H - 14}" text-anchor="middle" class="ax-lbl" style="font-size:11px">${s} s</text>`
    ).join("");

    el.innerHTML = `
      <svg class="svg-chart" viewBox="0 0 ${W} ${H}">
        ${grid}${xticks}
        <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${H - padB}" class="ax-line" />
        <line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" class="ax-line" />
        <polyline fill="none" stroke="var(--cyan)" stroke-width="2.8" points="${poly}" />
        <circle cx="${sx(3.22).toFixed(1)}" cy="${sy(28.56).toFixed(1)}" r="5" fill="var(--accent)" />
        <text x="${(sx(3.22) + 10).toFixed(1)}" y="${(sy(28.56) + 18).toFixed(1)}" class="val-lbl" style="fill:var(--accent);font-size:11.5px">3.22s → 28.56 dB</text>
        <circle cx="${sx(7.5).toFixed(1)}" cy="${sy(30.3).toFixed(1)}" r="5" fill="var(--accent)" />
        <text x="${(sx(7.5) + 10).toFixed(1)}" y="${(sy(30.3) + 18).toFixed(1)}" class="val-lbl" style="fill:var(--accent);font-size:11.5px">7.50s → 30.30 dB</text>
        <circle cx="${sx(17.11).toFixed(1)}" cy="${sy(31.31).toFixed(1)}" r="5.6" fill="var(--ok)" />
        <text x="${sx(17.11).toFixed(1)}" y="${(sy(31.31) - 12).toFixed(1)}" text-anchor="middle" class="val-lbl" style="fill:var(--ok);font-size:12px">${zh ? "峰值 31.31 dB (17.1s)" : "Peak 31.31 dB (17.1s)"}</text>
      </svg>
    `;
  }

  function renderVlaSurround(el, lang) {
    const zh = lang === "zh";
    el.innerHTML = `
      <svg class="svg-chart" viewBox="0 0 960 290">
        <rect x="10" y="10" width="462" height="270" rx="8" fill="var(--bg-elev)" stroke="var(--hairline-strong)" />
        <text x="30" y="38" class="val-lbl" style="fill:var(--accent);font-size:13.5px">${zh ? "Sim-to-VLA 与 GelSight 视触觉 (4 芯)" : "Sim-to-VLA &amp; GelSight Factory (4 Chips)"}</text>
        <text x="30" y="68" class="sub-lbl" style="font-size:11px">${zh ? "百万帧工厂数据集 (G=2048)" : "1M Factory Dataset (G=2048)"}</text>
        <text x="30" y="88" class="val-lbl" style="font-size:12.5px">1,001,472 frames in 142.58 s</text>
        <text x="30" y="106" class="sub-lbl" style="font-size:11px">7,024.2 cam-FPS · 4 TPU v5p chips · 0 host_callbacks</text>

        <line x1="30" y1="120" x2="452" y2="120" class="grid-line" />
        <text x="30" y="142" class="sub-lbl" style="font-size:11px">${zh ? "融合 ViT-Tiny 在线训练 (1.02M 参数)" : "Fused ViT-Tiny VLA (1.02M params)"}</text>
        <text x="30" y="162" class="val-lbl" style="fill:var(--ok);font-size:12.5px">7,143.1 FPS (−1.84% overhead)</text>
        <text x="30" y="180" class="sub-lbl" style="font-size:11px">Loss: 0.23781 → 0.03321 in HBM</text>

        <line x1="30" y1="194" x2="452" y2="194" class="grid-line" />
        <text x="30" y="216" class="sub-lbl" style="font-size:11px">${zh ? "4 路视触觉 (头/腕相机 + 双 GelSight)" : "4-Sensor Visuo-Tactile + Dual GelSight"}</text>
        <text x="30" y="236" class="val-lbl" style="fill:var(--cyan);font-size:12.5px">270,624.8 sensor-FPS (67,656 env/s)</text>
        <text x="30" y="254" class="sub-lbl" style="font-size:11px">DR overhead: +0.15% · zero gather_ops</text>

        <rect x="488" y="10" width="462" height="270" rx="8" fill="var(--bg-elev)" stroke="var(--hairline-strong)" />
        <text x="508" y="38" class="val-lbl" style="fill:var(--cyan);font-size:13.5px">${zh ? "自动驾驶 8×1080p + 双 128 线 LiDAR" : "Autonomous 8×1080p + Dual 128-Beam LiDAR"}</text>
        <text x="508" y="68" class="sub-lbl" style="font-size:11px">${zh ? "单图融合传感器组 (4 芯 &amp; 32 芯扩展)" : "Single-Graph Sensor Rig (4c &amp; 32c Scaling)"}</text>
        <text x="508" y="88" class="val-lbl" style="font-size:12.5px">8×1080p RGB+D+Seg + 2×128×2048 LiDAR</text>
        <text x="508" y="106" class="sub-lbl" style="font-size:11px">17.11M rays/env · gather_ops = 0 · host_callbacks = 0</text>

        <line x1="508" y1="120" x2="930" y2="120" class="grid-line" />
        <text x="508" y="142" class="sub-lbl" style="font-size:11px">${zh ? "端到端实测吞吐" : "Measured E2E Throughput"}</text>
        <text x="508" y="162" class="val-lbl" style="fill:var(--accent);font-size:12.5px">4c Rig: 66.81 cam-FPS (8.35 env-FPS)</text>
        <text x="508" y="180" class="sub-lbl" style="font-size:11px">32c Batch 128×2048 LiDAR: 243,676 env-FPS (63.88B rays/s)</text>

        <line x1="508" y1="194" x2="930" y2="194" class="grid-line" />
        <text x="508" y="216" class="sub-lbl" style="font-size:11px">${zh ? "点云几何对齐 (对比 MuJoCo C)" : "LiDAR XYZ Parity vs MuJoCo C"}</text>
        <text x="508" y="236" class="val-lbl" style="fill:var(--ok);font-size:12.5px">3.13e-8 m Dual-XYZ MAE</text>
        <text x="508" y="254" class="sub-lbl" style="font-size:11px">GeomID match = 1.0000 (bitwise boundary parity)</text>
      </svg>
    `;
  }

  function renderAll(lang) {
    const map = [
      ["chart-system-stack", renderSystemStack],
      ["chart-gpu-vs-tpu", renderGpuVsTpu],
      ["chart-evolution", renderEvolution],
      ["chart-gs3d-evo", renderGs3dEvo],
      ["chart-vmem-cliff", renderVmemCliff],
      ["chart-mxu-pareto", renderMxuPareto],
      ["chart-quality-orbit", renderQualityOrbit],
      ["chart-train-diffphys", renderTrainDiffPhys],
      ["chart-train-ppo", renderTrainPpo],
      ["chart-train-3dgs", renderTrain3dgs],
      ["chart-vla-surround", renderVlaSurround]
    ];
    for (const [id, fn] of map) {
      const el = document.getElementById(id);
      if (el) fn(el, lang);
    }
  }

  return { renderAll };
})();

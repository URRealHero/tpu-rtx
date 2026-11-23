/* ==========================================================================
   TPU-RTX Bilingual Content Dictionary (English / 中文)
   All telemetry verified on live Google Cloud TPU v5p (interpret=False)
   Author: Feiyang Liu
   ========================================================================== */

window.TPU_RTX_DATA = {
  plates: [
    {
      id: "moana_sunset_reef",
      cat: "film",
      layout: "wide",
      chapter: "ch-02",
      web: "./assets/web/v3_w2_moana_sunset_reef_4k.webp",
      png: "./assets/v3_w2_moana_sunset_reef_4k.png",
      badges: {
        en: [["hot", "22.30B Instanced Tris"], ["", "Native 4K"], ["ok", "Sunset + Coral Reef"]],
        zh: [["hot", "223 亿实例化三角面"], ["", "原生 4K"], ["ok", "日落 + 水下珊瑚礁"]]
      },
      title: {
        en: "Disney Moana OpenUSD v2.1: Golden Sunset & Underwater Coral Reef Caustics",
        zh: "迪士尼 Moana OpenUSD v2.1：黄金日落与水下珊瑚礁焦散（4K）"
      },
      desc: {
        en: "45.5M unique + 14.0M instanced BLAS triangles packed into 4.653 GiB HBM on TPU v5p: Rayleigh sunset glare over ocean waves (5.60 s @ 4K) and Way-of-Water submerged coral reef with Beer-Lambert cyan-emerald absorption, Snell's Window, caustics & 16-step HG God Rays (8.72 s @ 4K).",
        zh: "4,550 万唯一三角面 + 1,400 万实例化 BLAS（总等效 223 亿面），仅占 4.653 GiB HBM：海面黄金日落辉光（4K 5.60 s）与《水之道》风格水下珊瑚礁——含 Beer-Lambert 吸收、Snell 全反射天窗、光学焦散与 16 步 Henyey-Greenstein 体积丁达尔光（4K 8.72 s）。"
      }
    },
    {
      id: "moana_mjx_beach",
      cat: "film",
      chapter: "ch-02",
      web: "./assets/web/v3_w2_moana_mjx_beach_4k.webp",
      png: "./assets/v3_w2_moana_mjx_beach_4k.png",
      badges: {
        en: [["hot", "Physical AI × Film"], ["", "Moana × MJX"], ["ok", "5.40 s @ 4K"]],
        zh: [["hot", "具身智能 × 电影场景"], ["", "Moana × MJX"], ["ok", "4K 5.40 s"]]
      },
      title: {
        en: "Live MJX Humanoid & Quadruped on Motunui Beach (Native 4K)",
        zh: "MJX 人形与四足机器人直接奔跑在 Motunui 海滩（原生 4K）"
      },
      desc: {
        en: "Single unified TLAS traces both Disney's 22.30B-tri Moana USD island and 45 articulated MuJoCo MJX robot geoms with mutual BVH sand shadows inside one XLA graph (host_callbacks = 0).",
        zh: "单个统一 TLAS 在同一张 XLA 计算图内同时追踪 223 亿面 Moana OpenUSD 世界与 45 个实时驱动的 MuJoCo MJX 关节刚体，并在沙滩上投下互影（host_callbacks = 0）。"
      }
    },
    {
      id: "moana_8k_multi",
      cat: "film",
      chapter: "ch-02",
      web: "./assets/web/v3_w2_moana_8k_multiangle.webp",
      png: "./assets/v3_w2_moana_8k_multiangle.png",
      badges: {
        en: [["hot", "Native 8K (33.18M px)"], ["", "22.29B Tris"], ["ok", "17.24 s / 4 Chips"]],
        zh: [["hot", "原生 8K (3,318 万像素)"], ["", "222.9 亿面"], ["ok", "17.24 s / 4 芯"]]
      },
      title: {
        en: "Disney Moana OpenUSD @ Native 8K UHD Multi-Angle Vista",
        zh: "迪士尼 Moana OpenUSD 原生 8K UHD（7680×4320）多视角俯瞰"
      },
      desc: {
        en: "Full 7680×4320 aerial bay overview and shoreline view traced directly over OpenUSD point instancers across 4 TPU v5p chips at 3.85 MRays/s with zero gather_ops.",
        zh: "完整 7680×4320 分辨率直出，在 4 颗 TPU v5p 芯片上直接遍历 OpenUSD PointInstancer（17.24 s，3.85 MRays/s，gather_ops = 0）。"
      }
    },
    {
      id: "bmw_barcelona",
      cat: "film",
      chapter: "ch-02",
      web: "./assets/web/v3_w5_bmw_barcelona_cinema.webp",
      png: "./assets/v3_w5_bmw_barcelona_cinema.png",
      badges: {
        en: [["hot", "BMW27 37.65 dB"], ["", "Barcelona 54.91M"], ["ok", "0.9813 SSIM"]],
        zh: [["hot", "BMW27 37.65 dB"], ["", "巴塞罗那馆 5,491 万面"], ["ok", "SSIM 0.9813"]]
      },
      title: {
        en: "Blender Open Data: BMW27 Clearcoat & 54.9M-Tri Barcelona Pavilion",
        zh: "Blender 官方基准：BMW27 车漆与 5,491 万面巴塞罗那展馆"
      },
      desc: {
        en: "Automated Cycles Shader Graph compilation to Wide-BVH-64 + AgX tonemapping. BMW27 reaches 37.65 dB orbit PSNR (-0.012% luminance bias); Barcelona Pavilion traces 54.91M unique tris at 15.94 mm Z-MAE (4K 1.11 img/s).",
        zh: "自动编译 .blend Cycles 节点图至 Wide-BVH-64 + AgX 色调映射。BMW27 达到 37.65 dB 环绕 PSNR（亮度偏差仅 -0.012%）；巴塞罗那展馆在 5,491 万面下保持 15.94 mm 深度 MAE（4K 1.11 img/s）。"
      }
    },
    {
      id: "studio_double",
      cat: "film",
      chapter: "ch-02",
      web: "./assets/web/v3_w5_studio_double_cinema.webp",
      png: "./assets/v3_w5_studio_double_cinema.png",
      badges: {
        en: [["hot", "Monster 36.98 dB"], ["", "Fishy Cat 28.80 dB"], ["ok", "0.121 mm Z-MAE"]],
        zh: [["hot", "Monster 36.98 dB"], ["", "Fishy Cat 28.80 dB"], ["ok", "0.121 mm 深度误差"]]
      },
      title: {
        en: "Blender Studio Double: Monster Under the Bed SSS (1024²) & Fishy Cat",
        zh: "Blender Studio 双作：床下怪兽 SSS（1024²）与 Fishy Cat 毛发水面"
      },
      desc: {
        en: "Monster SSS (1.60M tris) achieves 36.98 dB orbit / 35.34 dB hero PSNR and 0.121 mm Z-MAE at 10.07 img/s (4c 1024²); Fishy Cat (907k tris, 7,255 fur instances) hits 28.80 dB (SSIM 0.9592).",
        zh: "160 万面床下怪兽与次表面散射在 1024² 下达到 36.98 dB（特写 35.34 dB）、0.121 mm 深度 MAE 与 10.07 img/s（4 芯）；Fishy Cat（90.7 万面 / 7,255 毛发实例）达 28.80 dB（SSIM 0.9592）。"
      }
    },
    {
      id: "barbershop",
      cat: "film",
      chapter: "ch-02",
      web: "./assets/web/v3_w7_barbershop_cinema.webp",
      png: "./assets/v3_w7_barbershop_cinema.png",
      badges: {
        en: [["hot", "8.55M Tris · 141 Tex"], ["", "31.00 dB Orbit"], ["ok", "13.8 mm MedAE"]],
        zh: [["hot", "855 万面 · 141 纹理"], ["", "31.00 dB PSNR"], ["ok", "13.8 mm 中位深度误差"]]
      },
      title: {
        en: "Blender Barbershop Interior (8.55M Tris, 249 Materials, 141 UV Textures)",
        zh: "Blender 理发店室内场景（855 万面、249 种材质、141 张 UV 贴图）"
      },
      desc: {
        en: "Complex indoor architectural testbed with 1,111 instances, transparent-glass blocker handling, and 141 packed UV textures hitting 31.00 dB orbit (SSIM 0.9285) and 1.16 img/s @ 4K.",
        zh: "极具挑战的室内复杂光线与多纹理场景：包含 1,111 个实例与玻璃遮挡处理，在 141 张打包贴图下达成 31.00 dB PSNR（SSIM 0.9285）与 4K 1.16 img/s。"
      }
    },
    {
      id: "lone_monk",
      cat: "film",
      chapter: "ch-02",
      web: "./assets/web/v3_w5_lone_monk_cinema.webp",
      png: "./assets/v3_w5_lone_monk_cinema.png",
      badges: {
        en: [["hot", "5.61M Tris · 22 Tex"], ["", "28.83 dB Orbit"], ["ok", "1.05 img/s @ 4K"]],
        zh: [["hot", "561 万面 · 22 纹理"], ["", "28.83 dB PSNR"], ["ok", "4K 1.05 img/s"]]
      },
      title: {
        en: "Blender Open Data: Lone Monk Courtyard, Foliage & Lanterns",
        zh: "Blender 官方基准：独行僧侣庭院（植被与灯笼暖光）"
      },
      desc: {
        en: "Dense courtyard foliage (20.14M instanced tris) and warm lantern bounce lighting compiled directly to TPU Wide-BVH-64: 28.83 dB orbit, 18.91 mm Z-MAE, 3.58 img/s @ 1080p.",
        zh: "2,014 万等效实例化三角面与灯笼多材质庭院光照：环绕 28.83 dB、深度误差 18.91 mm、1080p 3.58 img/s。"
      }
    },
    {
      id: "opendata_trio",
      cat: "film",
      chapter: "ch-02",
      web: "./assets/web/v3_w5_opendata_vs_rtxh100.webp",
      png: "./assets/v3_w5_opendata_vs_rtxh100.png",
      badges: {
        en: [["hot", "Cycles GT Parity"], ["", "Monster / Junkshop / Classroom"], ["ok", "35.23 dB / 30.36 dB"]],
        zh: [["hot", "Cycles 真值对齐"], ["", "Monster / Junkshop / Classroom"], ["ok", "35.23 dB / 30.36 dB"]]
      },
      title: {
        en: "Head-to-Head Blender Open Data Trio vs Cycles Ground Truth",
        zh: "Blender Open Data 三联标准对照：TPU-RTX vs Cycles EXR 真值"
      },
      desc: {
        en: "Side-by-side Cycles EXR verification across Monster (35.23 dB, 0.13 mm Z-MAE), Junkshop (30.36 dB), and Classroom interior relighting (20.50 dB, SSIM 0.7136).",
        zh: "同机位逐像素对齐官方 Cycles EXR：Monster 35.23 dB（0.13 mm Z-MAE）、Junkshop 30.36 dB、Classroom 室内重照明 20.50 dB。"
      }
    },
    {
      id: "thinlens_bokeh_4k",
      cat: "optics",
      layout: "wide",
      chapter: "ch-03",
      web: "./assets/web/v3_w6_thinlens_bokeh_4k.webp",
      png: "./assets/v3_w6_thinlens_bokeh_4k.png",
      badges: {
        en: [["hot", "Native 4K · 32 spp"], ["", "108.7 ms / 9.20 FPS"], ["ok", "2,441 MRays/s"]],
        zh: [["hot", "原生 4K · 32 spp"], ["", "108.7 ms / 9.20 FPS"], ["ok", "2,441 MRays/s"]]
      },
      title: {
        en: "Native 4K 55mm f/1.4 Stochastic Thin-Lens Bokeh (9.20 FPS)",
        zh: "原生 4K 55mm f/1.4 随机薄透镜景深散景（9.20 FPS，零 Gather）"
      },
      desc: {
        en: "Shirley-Chiu concentric aperture sampling + Murmur3 R2 sequence in HBM: 16-spp f/16 in 54.7 ms (18.28 FPS) and full 32-spp f/1.4 creamy bokeh in 108.7 ms (9.20 FPS) on 4 chips, shown with 1:1 near/far focal crops.",
        zh: "全片内 Shirley-Chiu 同心圆孔径采样 + Murmur3 R2 低差异序列：4 芯下 16-spp f/16 仅 54.7 ms（18.28 FPS），32-spp f/1.4 大光圈散景 108.7 ms（9.20 FPS），附近景与远景 1:1 焦点锐度对比裁切。"
      }
    },
    {
      id: "thinfilm_mirror",
      cat: "optics",
      chapter: "ch-03",
      web: "./assets/web/v3_w7_thinfilm_mirror_fx.webp",
      png: "./assets/v3_w7_thinfilm_mirror_fx.png",
      badges: {
        en: [["hot", "Airy Thin-Film (8-Bin)"], ["", "Anisotropic GGX"], ["ok", "12-Bounce Mirrors"]],
        zh: [["hot", "8 波段 Airy 薄膜干涉"], ["", "径向各向异性 GGX"], ["ok", "12 次递归镜廊"]]
      },
      title: {
        en: "Airy Spectral Thin-Film, Radial Aniso GGX & 12-Bounce Mirror Gallery",
        zh: "Airy 光谱薄膜干涉、径向拉丝金属 GGX 与 12 次无限镜廊"
      },
      desc: {
        en: "8-bin CIE 1931 interference (420–680 nm) on soap bubbles & TiO₂ + Heitz VNDF brushed metals (23.95 s @ 128 spp), alongside 2-bounce vs 12-bounce infinite mirror recursion in one jax.lax.while_loop.",
        zh: "8 波段 CIE 1931 波动干涉虹彩（肥皂泡与钛氧化膜）+ Heitz VNDF 各向异性拉丝金属，以及单条 jax.lax.while_loop 内完成的 2 次 vs 12 次反弹无限镜面隧道对比。"
      }
    },
    {
      id: "dispersion_sss",
      cat: "optics",
      chapter: "ch-03",
      web: "./assets/web/v3_w7_dispersion_sss_fx.webp",
      png: "./assets/v3_w7_dispersion_sss_fx.png",
      badges: {
        en: [["hot", "6-Band Cauchy Prism"], ["", "16-Step Random-Walk SSS"], ["ok", "128 spp Pure PT"]],
        zh: [["hot", "6 波段柯西棱镜色散"], ["", "16 步随机游走玉石 SSS"], ["ok", "128 spp 纯路径追踪"]]
      },
      title: {
        en: "6-Band Cauchy Spectral Prism Dispersion & Random-Walk Jade SSS",
        zh: "6 波段柯西棱镜光谱色散与 16 步随机游走次表面散射"
      },
      desc: {
        en: "Wavelength-dependent Fresnel refraction splits white light through crown glass into crisp spectral caustics alongside 16-step volumetric random-walk subsurface scattering in jade (128 spp, host_callbacks=0).",
        zh: "依赖波长的 Fresnel 折射将白光束劈开为干净的彩虹光谱焦散带，并与 16 步次表面随机游走半透明玉雕同场景呈现（128 spp，host_callbacks=0）。"
      }
    },
    {
      id: "restir_1024",
      cat: "optics",
      chapter: "ch-03",
      web: "./assets/web/v3_w7_restir_1024lights.webp",
      png: "./assets/v3_w7_restir_1024lights.png",
      badges: {
        en: [["hot", "TPU-ReSTIR DI"], ["", "1,024 Dynamic Lights"], ["ok", "+11.61 dB / 7.43× VarRed"]],
        zh: [["hot", "TPU-ReSTIR DI"], ["", "1,024 盏动态光源"], ["ok", "+11.61 dB / 方差降 7.43 倍"]]
      },
      title: {
        en: "TPU-ReSTIR DI: 1,024-Light Night Courtyard (720p, 4 Chips)",
        zh: "TPU-ReSTIR DI：1,024 盏动态光源夜景庭院（720p，4 芯）"
      },
      desc: {
        en: "Equal-visibility-ray temporal + spatial reservoir resampling (M=32, 2 spatial passes) slashes HDR variance 7.43× over naive MIS (10.29 dB -> 15.40 dB -> 21.89 dB denoised).",
        zh: "在等可见性射线预算下运行时空蓄水池重采样（M=32，2 轮空间复用），HDR 方差较朴素 MIS 降低 7.43 倍，PSNR 从 10.29 dB 跃升至 15.40 dB / 去噪 21.89 dB。"
      }
    },
    {
      id: "caustics_volumetrics",
      cat: "optics",
      chapter: "ch-03",
      web: "./assets/web/v3_w7_physical_fx_caustics_volumetrics.webp",
      png: "./assets/v3_w7_physical_fx_caustics_volumetrics.png",
      badges: {
        en: [["hot", "Beer-Lambert Caustics"], ["", "Cathedral God Rays"], ["ok", "192 spp · 0 Host CB"]],
        zh: [["hot", "Beer-Lambert 玻璃焦散"], ["", "教堂体积丁达尔光"], ["ok", "192 spp · 零回调"]]
      },
      title: {
        en: "Dielectric Beer-Lambert Glass Caustics & Cathedral Volumetric Shafts",
        zh: "电介质折射焦散与教堂彩窗体积圣光（192 spp）"
      },
      desc: {
        en: "5-bounce dielectric glass refraction with Beer-Lambert absorption paired with BVH-shadowed volumetric crepuscular rays ray-marched natively in TPU HBM.",
        zh: "5 次反弹玻璃折射焦点能量聚集与 BVH 阴影投射的大气丁达尔光束在单张 XLA 图内联算出图。"
      }
    },
    {
      id: "garden_8k_crop",
      cat: "gs3d",
      layout: "wide",
      chapter: "ch-04",
      web: "./assets/web/v3_w3_realworld_8k_crop.webp",
      png: "./assets/v3_w3_realworld_8k_crop.png",
      badges: {
        en: [["hot", "Native 8K 3DGS"], ["", "5.83M Real SH3"], ["ok", "29.19 MPix/s · 0 Gather"]],
        zh: [["hot", "原生 8K 3DGS"], ["", "583 万真实 SH3 高斯"], ["ok", "29.19 MPix/s · 零 Gather"]]
      },
      title: {
        en: "Real-World Mip-NeRF 360 garden @ Native 8K (7680×4320, 1:1 Crop)",
        zh: "真实扫描 Mip-NeRF 360 garden 原生 8K（7680×4320）1:1 无缩放裁切"
      },
      desc: {
        en: "Genuine .ply photogrammetry capture (5,834,784 SH3 Gaussians) rendered directly at 33.18M native pixels on 4 TPU v5p chips in 1,136.6 ms (0.88 FPS, 22.83 GB/chip), shown with zero-upscaling 1:1 native crops.",
        zh: "5,834,784 个全 SH3 真实航拍重建高斯体在 4 芯 TPU 上直出 3,318 万像素原生 8K（1,136.6 ms / 0.88 FPS，每芯 22.83 GB HBM），右图展示无超分 1:1 原生像素锐度。"
      }
    },
    {
      id: "robowild_4dgs",
      cat: "gs3d",
      chapter: "ch-04",
      web: "./assets/web/v3_w3_4dgs_robowild_manipulation.webp",
      png: "./assets/v3_w3_4dgs_robowild_manipulation.png",
      badges: {
        en: [["hot", "4DGS × MJX Contact"], ["", "125k Moving Splats"], ["ok", "5.32 FPS 1080p"]],
        zh: [["hot", "4DGS × MJX 物理推挤"], ["", "12.5 万动态刚体高斯"], ["ok", "1080p 5.32 FPS"]]
      },
      title: {
        en: "Dynamic 4DGS × MJX RoboWild Contact Manipulation",
        zh: "动态 4DGS × MJX RoboWild 物理接触推挤（无 CPU 往返）"
      },
      desc: {
        en: "Live mjx.step Franka Panda arm physically collides with and pushes 125,605 SE(3)-tracked LEGO Dozer Gaussians (+16.43 cm / -17.71°) inside a real Mip-NeRF 360 kitchen at 5.32 FPS 1080p / 1.30 FPS 4K.",
        zh: "真实 Mip-NeRF 360 厨房高斯扫描场景中，Live mjx.step Franka Panda 机械臂实时撞击并推走 125,605 个乐高推土机高斯（平移 +16.43 cm，旋转 -17.71°），1080p 5.32 FPS。"
      }
    },
    {
      id: "mipnerf_quartet",
      cat: "gs3d",
      chapter: "ch-04",
      web: "./assets/web/v3_w3_realworld_3dgs_robowild_4k.webp",
      png: "./assets/v3_w3_realworld_3dgs_robowild_4k.png",
      badges: {
        en: [["hot", "Mip-NeRF 360 × 4"], ["", "SH3 · 1.24M–6.13M"], ["ok", "2.59–4.20 FPS @ 4K"]],
        zh: [["hot", "Mip-NeRF 360 四联"], ["", "SH3 · 124 万–613 万"], ["ok", "4K 2.59–4.20 FPS"]]
      },
      title: {
        en: "Photorealistic Mip-NeRF 360 Quartet @ Native 4K (Zero Gather)",
        zh: "真实摄影测量 Mip-NeRF 360 四景原生 4K 与深度合成（零 Gather）"
      },
      desc: {
        en: "Native 3840×2160 EWA rasterization on 4 chips: bonsai 4.20 FPS, kitchen 3.98 FPS, garden 2.61 FPS, bicycle 2.59 FPS — paired with analytical robot depth compositing.",
        zh: "4 芯原生 3840×2160 实测：bonsai 4.20 FPS、kitchen 3.98 FPS、garden 2.61 FPS、bicycle 2.59 FPS，并与解析机器人刚体无缝 Z-buffer 混合。"
      }
    },
    {
      id: "bistro_humanoid",
      cat: "gs3d",
      chapter: "ch-04",
      web: "./assets/web/v3_w3_bistro_humanoid_4k.webp",
      png: "./assets/v3_w3_bistro_humanoid_4k.png",
      badges: {
        en: [["hot", "10M Bistro + MJX"], ["", "Native 4K 2.28 FPS"], ["ok", "5.24× Speedup"]],
        zh: [["hot", "1,000 万 Bistro + MJX"], ["", "原生 4K 2.28 FPS"], ["ok", "提速 5.24 倍"]]
      },
      title: {
        en: "10M-Gaussian Amazon Bistro × MJX Humanoid & Mobile Panda @ 4K",
        zh: "1,000 万高斯 Amazon Bistro 街区 × MJX 人形/移动熊猫机器人（4K）"
      },
      desc: {
        en: "Real-to-sim pixel-accurate Z-buffer compositing inside a 10M-Gaussian European street scene in 439.3 ms (2.28 FPS) @ 4K across 4 TPU v5p chips.",
        zh: "千万级街区 3DGS 与实时 MJX 人形/熊猫双机器人 4K 原生深度混合渲染，4 芯单帧 439.3 ms（2.28 FPS）。"
      }
    },
    {
      id: "relight_quant",
      cat: "dlss",
      layout: "wide",
      chapter: "ch-05",
      web: "./assets/web/v3_w4_interactive_relight_quant.webp",
      png: "./assets/v3_w4_interactive_relight_quant.png",
      badges: {
        en: [["hot", "78.6 FPS @ 4K BF16"], ["", "307.4 FPS @ 1080p"], ["ok", "0 Raytrace Ops/Slider"]],
        zh: [["hot", "4K BF16 78.6 FPS"], ["", "1080p 307.4 FPS"], ["ok", "滑块调参 0 次光追"]]
      },
      title: {
        en: "Interactive Zero-Raytrace Relighting & FP32 / BF16 / INT8 MXU Pareto",
        zh: "零光追交互式昼夜/材质重打光 & MXU 精度/吞吐帕累托曲线"
      },
      desc: {
        en: "Sweep Morning→Noon→Twilight (Classroom) & Clay→Satin→Gold→Chrome (Cornell) off ONE static G-buffer: BF16 matches FP32 PSNR (20.276 dB) at 2.11× speedup (78.60 vs 37.20 FPS @ 4K), while INT8 QAT costs only -0.047 dB.",
        zh: "仅从一张静态 G-Buffer 出发，滑块零光追实时切换昼夜色温与金属度；BF16 与 FP32 画质完全持平（20.276 dB）却提速 2.11 倍（4K 78.60 vs 37.20 FPS），INT8 QAT 掉点仅 -0.047 dB。"
      }
    },
    {
      id: "tpu_dlss_4k",
      cat: "dlss",
      chapter: "ch-05",
      web: "./assets/web/v3_w4_tpu_dlss_4k.webp",
      png: "./assets/v3_w4_tpu_dlss_4k.png",
      badges: {
        en: [["hot", "TPU-DLSS 3.5"], ["", "58.93 FPS @ Native 4K"], ["ok", "3.91× Speedup · 7.01× Anti-Flicker"]],
        zh: [["hot", "TPU-DLSS 3.5"], ["", "原生 4K 58.93 FPS"], ["ok", "提速 3.91 倍 · 抗闪烁 7.01 倍"]]
      },
      title: {
        en: "TPU-DLSS 3.5 Temporal Reconstruction: 1080p 1-spp → Native 4K",
        zh: "TPU-DLSS 3.5 时序超分重建：1080p 1-spp → 原生 4K（58.9 FPS）"
      },
      desc: {
        en: "Exact 4×4 bilinear motion-vector reprojection + BF16 MXU YCoCg reconstruction cuts temporal shimmer 7.01× in 16.97 ms @ 4K (58.93 FPS) on 4 chips.",
        zh: "精确 4×4 双线性运动矢量重投影 + YCoCg 历史钳位与 BF16 MXU 重建网络，将帧间闪烁压低 7.01 倍，4 芯 4K 单帧仅 16.97 ms（58.93 FPS）。"
      }
    },
    {
      id: "mc_denoiser_4k",
      cat: "dlss",
      chapter: "ch-05",
      web: "./assets/web/v3_w4_mc_denoiser.webp",
      png: "./assets/v3_w4_mc_denoiser.png",
      badges: {
        en: [["hot", "87.78 FPS @ Native 4K"], ["", "+15.22 dB (1-spp)"], ["ok", "11.39 ms · 0.24% Lum Bias"]],
        zh: [["hot", "原生 4K 87.78 FPS"], ["", "+15.22 dB (1-spp)"], ["ok", "11.39 ms · 亮度偏差 0.24%"]]
      },
      title: {
        en: "11.39 ms Native 4K BF16 MXU Monte Carlo GI Denoiser",
        zh: "11.39 ms 原生 4K BF16 MXU 蒙特卡洛路径追踪降噪器（近 90 Hz）"
      },
      desc: {
        en: "Albedo-demodulated BF16 MXU residual U-Net elevates noisy 1-spp Monte Carlo GI (9.70 dB) to clean 24.92 dB output at 87.78 FPS in Native 4K (0.24% luminance bias).",
        zh: "基于反照率解调的 MXU 残差网络，将满屏噪点的 1-spp 路径追踪画面（9.70 dB）提升至干净的 24.92 dB，原生 4K 下跑出 87.78 FPS。"
      }
    },
    {
      id: "menagerie_pbr",
      cat: "robotics",
      chapter: "ch-06",
      web: "./assets/web/v3_w0_mujoco_menagerie_pbr.webp",
      png: "./assets/v3_w0_mujoco_menagerie_pbr.png",
      badges: {
        en: [["hot", "MuJoCo Menagerie"], ["", "Up to 4,284 FPS"], ["ok", "0 Gather · 0 Host CB"]],
        zh: [["hot", "MuJoCo Menagerie 通用加载"], ["", "最高 4,284 FPS"], ["ok", "零 Gather · 零回调"]]
      },
      title: {
        en: "Official MuJoCo Menagerie Assets in Single-Kernel Wide-BVH-64",
        zh: "DeepMind 官方 MuJoCo Menagerie 资产直编与多机器人 PBR 吞吐"
      },
      desc: {
        en: "Unmodified MJCF assets on 4 TPU v5p chips (64²): Tetheria Aero Hand 4,284 FPS, Franka Panda 3,131 FPS, Unitree G1 (393k tris) 2,546 FPS, Unitree Go2 (398k tris) 1,863 FPS — with 1024² GGX+8-AO+8-SoftShadow hero renders.",
        zh: "未改模型的官方 MJCF 直接走 MultiMeshBVHBank 编译上 TPU（4 芯 64²）：灵巧手 Aero Hand 4,284 FPS、Franka Panda 3,131 FPS、宇树 G1（39.3 万面）2,546 FPS、Go2（39.8 万面）1,863 FPS。"
      }
    },
    {
      id: "diff_phys_render",
      cat: "robotics",
      chapter: "ch-06",
      web: "./assets/web/v3_w0_diff_phys_render.webp",
      png: "./assets/v3_w0_diff_phys_render.png",
      badges: {
        en: [["hot", "End-to-End Diff Physics"], ["", "36.08 dB / 1.63 cm EE"], ["ok", "Zero State Supervision"]],
        zh: [["hot", "端到端可微物理+光追"], ["", "36.08 dB / 末端 1.63 cm"], ["ok", "零关节状态监督"]]
      },
      title: {
        en: "Pixel-to-Torque Visual Servoing via jax.grad(mjx.step + PBR)",
        zh: "纯视觉像素到力矩反传伺服：jax.grad 穿透 12 步 MJX 物理 + 光追"
      },
      desc: {
        en: "Pure RGB L2 pixel loss backpropagates cleanly through 12 mjx.step physics steps and smooth-visibility GGX raytracing (zero ground-truth state labels): 17.50 -> 36.08 dB, EE 3D error 16.81 cm -> 1.63 cm at 27.87 grad-steps/s.",
        zh: "仅靠目标照片像素 L2 误差，梯度一路穿透物理刚体动力学与平滑可见性光追着色器（完全不用真值关节角监督）：PSNR 从 17.50 升至 36.08 dB，末端 3D 误差由 16.81 cm 收敛至 1.63 cm（27.9 grad-steps/s）。"
      }
    },
    {
      id: "isaac_replicator",
      cat: "robotics",
      chapter: "ch-06",
      web: "./assets/web/v3_w1_isaac_replicator_dr.webp",
      png: "./assets/v3_w1_isaac_replicator_dr.png",
      badges: {
        en: [["hot", "Isaac Replicator DR"], ["", "123,858 cam-FPS"], ["ok", "+0.32% DR · 1M VLA / 143s"]],
        zh: [["hot", "Isaac Replicator 域随机"], ["", "123,858 cam-FPS"], ["ok", "DR 开销仅 +0.32%"]]
      },
      title: {
        en: "Isaac Sim Replicator Parity, 1M-Frame Factory & GelSight Visuo-Tactile",
        zh: "Isaac Replicator 域随机化、142.6 秒百万帧 VLA 与 GelSight 视触觉"
      },
      desc: {
        en: "rep.randomizer compiles inside VMEM at 123,858 cam-FPS (+0.32% DR overhead), followed by 1,001,472 G=2048 factory frames in 142.58 s, ViT-Tiny Sim-to-VLA at 7,143 FPS (-1.84% fused overhead), and 270,625 sensor-FPS GelSight.",
        zh: "rep.randomizer 在片内零拷贝编译（123,858 cam-FPS，DR 开销 +0.32%）；随后在 142.58 s 内生成 100 万帧 G=2048 工厂数据，ViT-Tiny 在线 Sim-to-VLA 7,143 FPS（-1.84% 融合开销），4 路 GelSight 视触觉达 270,625 sensor-FPS。"
      }
    },
    {
      id: "pod_dryrun",
      cat: "pod",
      chapter: "ch-08",
      web: "./assets/web/v3_w6_pod_dryrun_scaling.webp",
      png: "./assets/v3_w6_pod_dryrun_scaling.png",
      badges: {
        en: [["hot", "4-Chip HBM Ceiling"], ["", "2.15B Unique Tris"], ["ok", "8.25B LiDAR Rays/s"]],
        zh: [["hot", "单机 4 芯 HBM 极限"], ["", "21.5 亿唯一三角面"], ["ok", "82.5 亿 LiDAR 射线/秒"]]
      },
      title: {
        en: "4-Chip Single-Host Capstone Ceiling (246.4 s, 383 GB HBM, 0 Callbacks)",
        zh: "单机 4 芯极限天花板测试（246.4 s 全套收口，383 GB HBM，零回调）"
      },
      desc: {
        en: "Verified end-to-end across 5 benchmarks on 4 live TPU v5p chips: 51,548 FPS MJX+RTX, 155,858 FPS 3DGS + 132.7M ICI splats, 31,460 env-FPS 128-beam LiDAR (8.25B rays/s), 2.15B unique tris (147 GB HBM, ICI pmin), and Native 8K 3-bounce PT (218.9 spm).",
        zh: "单机 4 芯完整跑通 5 大极限基准：51,548 FPS MJX+RTX、155,858 FPS 3DGS + ICI 1.327 亿分布式高斯、31,460 env-FPS 128 线激光雷达（82.5 亿射线/秒）、21.5 亿唯一三角面（147 GB HBM）与原生 8K 3-bounce 路径追踪（218.9 spm）。"
      }
    }
  ],

  heroStage: [
    {
      plateId: "moana_sunset_reef",
      hudL: {
        en: "<strong>Disney Moana OpenUSD v2.1</strong> · 22.30B Tris · 4.65 GiB HBM",
        zh: "<strong>迪士尼 Moana OpenUSD v2.1</strong> · 223 亿面 · 4.65 GiB HBM"
      },
      hudR: {
        en: "Sunset <strong>5.60 s</strong> · Underwater Reef <strong>8.72 s</strong> @ 4K",
        zh: "黄金日落 <strong>5.60 s</strong> · 珊瑚礁焦散 <strong>8.72 s</strong> @ 4K"
      }
    },
    {
      plateId: "garden_8k_crop",
      hudL: {
        en: "<strong>Real-World 3DGS</strong> · 5.83M SH3 <code>garden</code> · gather_ops=0",
        zh: "<strong>真实航拍 3DGS</strong> · 583 万 SH3 <code>garden</code> · 零 Gather"
      },
      hudR: {
        en: "Native <strong>7680×4320 (8K)</strong> in <strong>1.14 s</strong> (29.19 MPix/s)",
        zh: "原生 <strong>7680×4320 (8K)</strong> 单帧 <strong>1.14 s</strong> (29.19 MPix/s)"
      }
    },
    {
      plateId: "thinlens_bokeh_4k",
      hudL: {
        en: "<strong>55mm f/1.4 Thin-Lens Bokeh</strong> · Native 3840×2160 (32 spp)",
        zh: "<strong>55mm f/1.4 薄透镜景深</strong> · 原生 3840×2160 (32 spp)"
      },
      hudR: {
        en: "<strong>108.7 ms / frame</strong> (9.20 FPS) · <strong>2,441 MRays/s</strong>",
        zh: "<strong>108.7 ms / 帧</strong> (9.20 FPS) · <strong>2,441 MRays/s</strong>"
      }
    }
  ],

  benchmarks: {
    mjx: {
      title: {
        en: "Embodied Robotics, Isaac Replicator DR, Menagerie, Tactile & Sim-to-VLA (host_callbacks=0)",
        zh: "具身机器人、Isaac 域随机、Menagerie、GelSight 视触觉与 Sim-to-VLA（零回调）"
      },
      note: {
        en: "Single-XLA-program physics + rendering + domain randomization + VLA training directly in TPU HBM.",
        zh: "物理仿真、光线追踪、域随机化与策略网络前/反向传播全部驻留 TPU HBM 单图执行。"
      },
      rows: [
        { label: { en: "V3 Fused 4c MJX+Render (64×64, B=8192)", zh: "V3 融合 4 芯 MJX+渲染 (64×64, B=8192)" }, val: 266209, unit: "FPS E2E", pct: 98, extra: { en: "921,309 render-equiv", zh: "921,309 纯渲染等效" } },
        { label: { en: "4-Sensor Visuo-Tactile GelSight+Cams (4c)", zh: "4 路视触觉 GelSight+双目相机 (4 芯)" }, val: 270625, unit: "sensor-FPS", pct: 100, extra: { en: "67,656 env-steps/s (0.15% DR)", zh: "67,656 env/s (DR 仅 +0.15%)" } },
        { label: { en: "Isaac Replicator RoboTwin 3-Cam+DR (4c)", zh: "Isaac Replicator RoboTwin 三摄+DR (4 芯)" }, val: 123858, unit: "cam-FPS", pct: 46, extra: { en: "41,286 env/s (+0.32% DR)", zh: "41,286 env/s (DR +0.32%)" } },
        { label: { en: "V3 Closed-Loop Vision PPO Cartpole (4c)", zh: "V3 零拷贝闭环 Vision PPO Cartpole (4 芯)" }, val: 60145, unit: "env-steps/s", pct: 23, extra: { en: "Panda 29,267/s · host_cb=0", zh: "Panda 29,267/s · 零回调" } },
        { label: { en: "Factory G=2048 Sim-to-VLA ViT-Tiny (4c)", zh: "千帧工厂 G=2048 在线 Sim-to-VLA (4 芯)" }, val: 7143, unit: "frames/s", pct: 12, extra: { en: "1M frames/142.6s · -1.84% VLA", zh: "百万帧 142.6s · -1.84% 融合开销" } },
        { label: { en: "MuJoCo Menagerie Wide-BVH-64 (Aero/Panda/G1)", zh: "MuJoCo Menagerie Wide-BVH-64 (4 芯)" }, val: 4284, unit: "FPS (64²)", pct: 8, extra: { en: "3,131 Panda / 2,546 G1", zh: "Panda 3,131 / G1 2,546" } },
        { label: { en: "End-to-End Diff MJX+PBR Servoing (w0)", zh: "端到端可微 MJX+光追 像素→力矩伺服" }, val: 27.9, unit: "grad-steps/s", pct: 4, extra: { en: "36.08 dB · 1.63 cm EE err", zh: "36.08 dB · 末端误差 1.63 cm" } }
      ]
    },
    gs3d: {
      title: {
        en: "Native 3DGS / 4DGS × MJX & 300M-Splat 4-Chip HBM Scaling (gather_ops=0)",
        zh: "原生 3DGS / 4DGS × MJX 与单机 4 芯 3 亿高斯容量极限（gather_ops=0）"
      },
      note: {
        en: "Zero dynamic HBM gathers; scales from 142k FPS (N=2048) to 300M Gaussians (79.44% of 4-chip 380 GB HBM) and 8K photogrammetry.",
        zh: "无任何动态 HBM Gather；吞吐高达 14.2 万 FPS，HBM 容量推至 3 亿个高斯（占 4 芯 380 GB 显存 79.44%）及原生 8K 实景渲染。"
      },
      rows: [
        { label: { en: "V2 jax_tile Baseline (1c 64×64, N=2048)", zh: "V2 jax_tile 基线 (单芯 64×64, N=2048)" }, val: 1194, unit: "FPS", pct: 2, extra: { en: "baseline", zh: "基线" }, base: true },
        { label: { en: "V3 gs3d_v3 BF16-MXU (1c 64×64, N=2048)", zh: "V3 gs3d_v3 BF16-MXU (单芯 64×64)" }, val: 35916, unit: "FPS", pct: 25, extra: { en: "30.1× V2", zh: "30.1× 提速" } },
        { label: { en: "V3 gs3d_v3 BF16-MXU (4c shard_map, 64×64)", zh: "V3 gs3d_v3 4 芯扩展 (64×64)" }, val: 142008, unit: "FPS", pct: 100, extra: { en: "155,858 FPS @ 128² dry-run", zh: "128² 验证 155,858 FPS" } },
        { label: { en: "Fast 4c 32-View jax.grad Training (garden)", zh: "4 芯 32 视角 jax.grad 快速训练 (garden)" }, val: 749, unit: "view-steps/s", pct: 18, extra: { en: "23.4 steps/s -> 31.3 dB peak", zh: "23.4 steps/s · 峰值 31.3 dB" } },
        { label: { en: "Mip-NeRF 360 4K Quartet (bonsai/kitchen/garden)", zh: "Mip-NeRF 360 原生 4K 实景 (SH3)" }, val: 4.20, unit: "FPS @ 4K", pct: 12, extra: { en: "8K garden 29.19 MPix/s", zh: "8K garden 29.19 MPix/s" } },
        { label: { en: "4DGS × MJX Dynamic Push (125.6k LEGO splats)", zh: "4DGS × MJX 动态推挤 (12.5 万高斯)" }, val: 5.32, unit: "FPS @ 1080p", pct: 14, extra: { en: "1.30 FPS @ 4K (+16.4 cm push)", zh: "4K 1.30 FPS (推挤 +16.4 cm)" } },
        { label: { en: "Single-Host 300,000,000 Gaussians @ 4K (4c)", zh: "单机 4 芯 300,000,000 高斯 @ 4K" }, val: 301.87, unit: "GB HBM", pct: 80, extra: { en: "0.25 FPS · 1,171,875:1 cull", zh: "0.25 FPS · 剔除比 117 万:1" } }
      ]
    },
    cinema: {
      title: {
        en: "Disney Moana OpenUSD v2.1 (22.30B Tris) & Blender Studio Scenes on TPU v5p",
        zh: "迪士尼 Moana OpenUSD (223 亿面) 与 Blender Studio 电影标准场景对齐"
      },
      note: {
        en: "Compiled directly from official OpenUSD and .blend files into Wide-BVH-64 packs + AgX tonemapping.",
        zh: "直接由官方 .usd 与 .blend 原生编译为 Wide-BVH-64 与 Cycles 材质图 + AgX 色调映射。"
      },
      rows: [
        { label: { en: "BMW27 Clearcoat (Orbit PSNR)", zh: "BMW27 双层清漆跑车 (环绕 PSNR)" }, val: 37.65, unit: "dB PSNR", pct: 100, extra: { en: "0.9813 SSIM · -0.012% lum", zh: "SSIM 0.9813 · 亮度差 -0.012%" } },
        { label: { en: "Monster Under the Bed SSS 1024² (Orbit)", zh: "床下怪兽 SSS 1024² (环绕 PSNR)" }, val: 36.98, unit: "dB PSNR", pct: 98, extra: { en: "35.34 dB hero · 0.121 mm Z-MAE", zh: "特写 35.34 dB · 深度差 0.121 mm" } },
        { label: { en: "Barbershop Interior 8.55M Tris (141 Tex)", zh: "理发店室内 855 万面 (141 张纹理)" }, val: 31.00, unit: "dB PSNR", pct: 82, extra: { en: "13.8 mm MedAE · 4K 1.16 img/s", zh: "中位深度 13.8 mm · 4K 1.16 img/s" } },
        { label: { en: "Junkshop / Lone Monk / Fishy Cat (Orbit)", zh: "Junkshop / Lone Monk / Fishy Cat" }, val: 30.36, unit: "dB PSNR", pct: 80, extra: { en: "28.83 Monk / 28.80 Fishy Cat", zh: "Monk 28.83 / Fishy Cat 28.80 dB" } },
        { label: { en: "Barcelona Pavilion (54.91M Unique Tris)", zh: "巴塞罗那展馆 (5,491 万唯一三角面)" }, val: 27.45, unit: "dB PSNR", pct: 73, extra: { en: "15.94 mm Z-MAE · 4K 1.11 img/s", zh: "深度差 15.94 mm · 4K 1.11 img/s" } },
        { label: { en: "Disney Moana OpenUSD (22.30B Tris, 4K)", zh: "迪士尼 Moana OpenUSD (223 亿面, 4K)" }, val: 5130, unit: "ms / frame", pct: 60, extra: { en: "8K 17.2s · Underwater 8.7s", zh: "8K 17.2s · 水下珊瑚礁 8.7s" } }
      ]
    },
    dlss: {
      title: {
        en: "TPU-DLSS 3.5, 4K MC Denoiser & Zero-Copy Static G-Buffer Relighting (4-Chip MXU)",
        zh: "TPU-DLSS 3.5、4K 路径追踪降噪与零光追交互式重打光（4 芯 MXU）"
      },
      note: {
        en: "Exact 4×4 bilinear reprojection + optimization-barrier BF16/INT8 MXU inference at 1080p and Native 4K.",
        zh: "精确 4×4 双线性重投影 + optimization_barrier BF16/INT8 MXU 推理。"
      },
      rows: [
        { label: { en: "Interactive Static G-Buffer Relight (4c 1080p BF16)", zh: "零光追静态 G-Buffer 交互重打光 (1080p BF16)" }, val: 307.4, unit: "FPS", pct: 100, extra: { en: "20.28 dB · 0 raytrace ops", zh: "20.28 dB · 0 次光追开销" } },
        { label: { en: "4K BF16 MXU MC AI Denoiser (4c 3840×2160)", zh: "原生 4K BF16 MXU 蒙特卡洛降噪器 (4 芯)" }, val: 87.8, unit: "FPS @ 4K", pct: 35, extra: { en: "11.39 ms · +15.22 dB (1-spp)", zh: "11.39 ms · +15.22 dB (1-spp)" } },
        { label: { en: "Interactive Relight Native 4K BF16 (vs 37.2 FP32)", zh: "原生 4K BF16 交互重打光 (对比 FP32 37.2)" }, val: 78.6, unit: "FPS @ 4K", pct: 31, extra: { en: "2.11× FP32 · INT8 78.1 FPS", zh: "提速 2.11× · INT8 78.1 FPS" } },
        { label: { en: "TPU-DLSS 3.5 Native 4K (from 1080p 1-spp, 4c)", zh: "TPU-DLSS 3.5 原生 4K 重建 (1080p→4K, 4 芯)" }, val: 58.9, unit: "FPS @ 4K", pct: 25, extra: { en: "16.97 ms · 3.91× E2E speedup", zh: "16.97 ms · 端到端提速 3.91×" } }
      ]
    },
    fx: {
      title: {
        en: "TPU-Native Monte Carlo GI, ReSTIR DI (1,024 Lights), Thin-Lens Bokeh & Physical FX",
        zh: "原生蒙特卡洛全局光、ReSTIR DI（1,024 盏灯）、薄透镜散景与波动光学"
      },
      note: {
        en: "100% pure ray/path transport & wave optics compiled with zero host callbacks.",
        zh: "100% 纯射线/路径传输与波动光学，零主机回调直接跑在 TPU 上。"
      },
      rows: [
        { label: { en: "Native 4K 16-spp f/16 Thin-Lens Bokeh (4c)", zh: "原生 4K 16-spp f/16 薄透镜散景 (4 芯)" }, val: 18.28, unit: "FPS @ 4K", pct: 100, extra: { en: "54.7 ms · 32-spp f/1.4: 9.20 FPS", zh: "54.7 ms · 32-spp f/1.4 9.20 FPS" } },
        { label: { en: "3-Bounce Native 8K Path Tracer (4c Dry-Run)", zh: "3 次反弹原生 8K 路径追踪 (4 芯实测)" }, val: 218.9, unit: "spm @ 8K", pct: 68, extra: { en: "726.3 MRays/s · host_cb=0", zh: "726.3 MRays/s · 零回调" } },
        { label: { en: "1-Bounce / 3-Bounce Physical GI (512², 1c)", zh: "单芯 512² 1-bounce / 3-bounce 全局光" }, val: 3121, unit: "spm @ 512²", pct: 85, extra: { en: "835 spm 3-bounce (64 spp)", zh: "3-bounce 835 spm (64 spp)" } },
        { label: { en: "TPU-ReSTIR DI 1,024-Light Courtyard (720p, 4c)", zh: "TPU-ReSTIR DI 1,024 动态光源庭院 (720p)" }, val: 21.89, unit: "dB PSNR", pct: 60, extra: { en: "+11.61 dB · 7.43× HDR VarRed", zh: "+11.61 dB · HDR 方差降 7.43×" } },
        { label: { en: "Airy Thin-Film (8-bin) + 12-Bounce Mirrors (4c)", zh: "8 波段 Airy 薄膜 + 12 次递归镜廊 (4 芯)" }, val: 12, unit: "bounces", pct: 50, extra: { en: "128 spp · single XLA while_loop", zh: "128 spp · 单条 XLA while_loop" } }
      ]
    }
  },

  copy: {
    en: {
      pageTitle: "TPU-RTX — A Systolic Array Can Ray-Trace, Path-Trace & Splat Like a GPU | Feiyang Liu",
      themeNight: "Obsidian",
      themeDay: "Paper",
      nav: {
        c1: "01 Silicon", c2: "02 Film USD", c3: "03 Optics", c4: "04 3DGS",
        c5: "05 Neural", c6: "06 Robotics", c7: "07 Telemetry", c8: "08 Limits"
      },
      hero: {
        kicker: "LIVE GOOGLE CLOUD TPU v5p SILICON · ZERO HOST_CALLBACK · ZERO GATHER_OPS",
        by: "By",
        license: "Apache-2.0",
        title: "A Systolic Array Can <em>Ray-Trace</em>, Path-Trace &amp; Splat Like a GPU.",
        lead: "<strong>TPU-RTX</strong> compiles OpenUSD feature films (Disney <em>Moana</em>, <strong>22.30B</strong> tris), Cycles studio shaders, real-world <strong>8K / 300M</strong> Gaussian Splatting, wave optics, and <strong>266,209 FPS</strong> differentiable MuJoCo MJX + Isaac physics straight into <strong>Google Cloud TPU v5p</strong> HBM and 64 MiB VMEM — with zero CPU round-trips and zero HBM gathers.",
        ctaGallery: "Inspect 24 Audited Plates",
        ctaBench: "Telemetry Console",
        stamp: "QA-Audited Monograph · Every frame run & inspected on live TPU v5p (interpret=False)",
        t0i: "PLATE 01 · OPENUSD v2.1",
        t0t: "Disney Moana (22.30B Tris): Sunset & Underwater Reef",
        t1i: "PLATE 02 · NATIVE 8K 3DGS",
        t1t: "Mip-NeRF 360 garden (5.83M SH3) @ 7680×4320 1:1 Crop",
        t2i: "PLATE 03 · 4K PHYSICAL OPTICS",
        t2t: "Native 4K f/1.4 32-spp Thin-Lens Bokeh (108.7 ms)"
      },
      kpis: [
        ["4-Chip Fused MJX+RTX", "266,209 FPS", "<strong>921,309 FPS</strong> render-equiv (<code>64², B=8192</code>)"],
        ["Disney Moana OpenUSD", "22.30B Tris", "<strong>4K 5.13 s · 8K 17.24 s</strong> · Sunset & Reef"],
        ["3DGS Speed & Ceiling", "142k FPS / 300M", "<code>142,008 FPS</code> (<code>64²</code>) · <strong>300M Splats</strong> @ 4K HBM"],
        ["Native 8K Real 3DGS", "7680×4320", "<code>5.83M</code> SH3 <code>garden</code> @ <strong>29.19 MPix/s</strong>"],
        ["DLSS 3.5 & Relight (4K)", "58.9 / 78.6 FPS", "DLSS <strong>3.91×</strong> · Static G-Buf Relight <strong>78.6 FPS</strong> BF16"],
        ["Blender Cycles Parity", "37.65 / 36.98 dB", "<code>BMW27</code> <strong>0.9813</strong> SSIM · <code>Monster</code> <strong>0.121 mm</strong>"],
        ["Unique Tris @ 4K / 8K", "2.15B / 102B", "<code>2.15B</code> unique (ICI <code>pmin</code>, 4c) · <strong>102.2B</strong> inst. 8K"],
        ["Diff Phys + Sim-to-VLA", "1.63 cm / 1M 143s", "<code>jax.grad</code> <strong>36.08 dB</strong> · GelSight <strong>270.6k/s</strong>"]
      ],
      c1: {
        num: "01 / 08 · SILICON THESIS",
        rail: "Why naive SIMT raytracing stalls on systolic arrays — and how 64-ary VMEM alignment turns a TPU into an RTX engine.",
        title: "Treating a TPU like CUDA stalls on gather latency. Aligning with <em>64 MiB VMEM</em> unlocks everything.",
        lead: "A Google Cloud TPU v5p gives each chip <strong>459 TFLOP/s</strong> of BF16 compute, <strong>2.765 TB/s</strong> HBM bandwidth, <strong>95.74 GiB</strong> HBM capacity, and <strong>64 MiB</strong> ultra-fast VMEM — yet its arithmetic ridge point is <strong>166 FLOP/B</strong>, and random HBM gathers crater throughput. Every breakthrough in TPU-RTX comes from rewriting traversal, splatting, and shading as sequential 128-lane vector blocks with <code>gather_ops = 0</code>.",
        pillars: [
          ["VMEM Traversal", "1. 3-Level 64-Ary Wide-BVH", "Packing 64 child AABBs into (8, 128) vector tiles lets Pallas Mosaic test 64 children in single VPU ops via pltpu.async_copy leaf streaming — scaling seamlessly to 22.30B-tri Disney Moana and 2.15B unique triangles."],
          ["Zero-Gather Raster", "2. In-VMEM Kernel Fusion & EWA", "Fusing raygen, traversal, GGX/AO/shadows, and material deduplication avoids HBM G-buffer round-trips (921k FPS render-equiv), while chunked VMEM take_along_axis accelerates 3DGS from 1.2k to 142k FPS and packs 300M splats into 4c HBM."],
          ["Native MXU + Grad", "3. BF16/INT8 MXUs & jax.grad Physics", "Keeping every pixel inside TPU HBM lets BF16 MXUs run 4K denoising at 87.8 FPS, TPU-DLSS 3.5 at 58.9 FPS 4K, static G-buffer relighting at 78.6 FPS 4K, Sim-to-VLA at -1.84% overhead, and pixel-to-torque backprop through mjx.step."]
        ],
        diagCap: "Architectural invariant verified in XLA HLO across all modules: host_callbacks = 0, send_recv_ops = 0, gather_ops = 0.",
        sdkBar: "Quickstart SDK · 8 Ecosystem Adapters under tpu_mjx_render.adapters"
      },
      c2: {
        num: "02 / 08 · FEATURE FILM & STUDIO",
        rail: "Disney Moana OpenUSD v2.1 (22.30B tris) & canonical Blender Studio scenes compiled straight into Wide-BVH-64 + AgX.",
        title: "22.30 billion triangles of Disney <em>Moana</em> and Blender Studio Cycles parity — on tensor silicon.",
        lead: "Instead of abstract synthetic benchmarks, we compile unmodified production scenes directly from official <strong>OpenUSD</strong> and <code>.blend</code> Cycles node graphs. Disney's <em>Moana Island Scene</em> v2.1 (22.30B instanced triangles) packs into just <strong>4.653 GiB</strong> HBM and mixes live with MJX robots on Motunui Beach; canonical Blender scenes match Cycles EXR down to sub-millimeter Z-depth."
      },
      c3: {
        num: "03 / 08 · PHYSICAL OPTICS",
        rail: "Spectral interference, anisotropic brushed metals, 12-bounce mirrors, Cauchy dispersion, volumetric SSS, 1,024-light ReSTIR, and 4K thin-lens bokeh.",
        title: "Wave optics, 12-bounce infinite mirrors &amp; <em>TPU-ReSTIR DI</em> at 1,024 lights.",
        lead: "Real path transport requires more than primary rays. TPU-RTX compiles 8-bin CIE 1931 Airy thin-film interference, Heitz VNDF radial anisotropic GGX, 12-bounce mirror recursion inside a single <code>jax.lax.while_loop</code>, 6-band Cauchy prism dispersion, 16-step random-walk SSS, 1,024-light temporal+spatial ReSTIR DI, and <strong>108.7 ms Native 4K f/1.4 thin-lens bokeh</strong>."
      },
      c4: {
        num: "04 / 08 · 3DGS & 4DGS",
        rail: "Eliminating scatter/gather unlocks 142k FPS splatting, 300M Gaussians in 4-chip HBM, Native 8K photogrammetry, and live MJX collision push.",
        title: "<em>300 million</em> Gaussians in HBM, Native 8K photogrammetry &amp; live 4DGS × MJX physics.",
        lead: "Conventional GPU 3DGS kernels rely heavily on atomic tile scatters and irregular gathers. Replacing them with chunked VMEM EWA sorting yields <strong>142,008 FPS</strong> (<code>64²</code>), scales to <strong>300,000,000 Gaussians</strong> inside 4-chip HBM (<code>301.87 / 380 GB</code>), renders 5.83M-splat real-world Mip-NeRF 360 <em>garden</em> at <strong>Native 8K</strong> (<code>29.19 MPix/s</code>), trains 32-view <code>jax.grad</code> at <code>23.42 steps/s</code>, and lets a live MJX Panda collide with 125.6k LEGO Gaussians."
      },
      c5: {
        num: "05 / 08 · NEURAL SILICON",
        rail: "Putting 459 TFLOP/s BF16 MXUs to work on temporal reconstruction (TPU-DLSS 3.5), 4K denoising, and zero-raytrace relighting.",
        title: "What systolic MXUs were built for: <em>58.9 FPS</em> 4K DLSS, <em>87.8 FPS</em> 4K Denoising &amp; Zero-Raytrace Relight.",
        lead: "Because raycast G-buffers never leave TPU HBM, feeding them into the BF16/INT8 MXU incurs near-zero data movement. Our exact 4×4 bilinear reprojection + MXU <strong>TPU-DLSS 3.5</strong> reconstructs 1080p 1-spp into Native 4K at <strong>58.93 FPS</strong> (<code>3.91×</code> E2E speedup, <code>7.01×</code> flicker reduction); our 4K MC denoiser hits <strong>87.78 FPS</strong> (<code>11.39 ms</code>); and our static G-buffer relighter sweeps time-of-day and Chrome/Gold BRDFs at <strong>78.60 FPS @ 4K BF16</strong>."
      },
      c6: {
        num: "06 / 08 · PHYSICAL AI",
        rail: "From 266,209 FPS MJX vision RL and MuJoCo Menagerie to 1M factory frames in 142.6 s, GelSight visuo-tactile, and pixel-to-torque gradients.",
        title: "From <em>266,209 FPS</em> Embodied RL to 1M VLA frames in 143 s and end-to-end <em>jax.grad</em> control.",
        lead: "In robotics, rendering only matters when physics, sensors, domain randomization, and policy gradients live together. We hit <strong>266,209 FPS</strong> fused MJX+Render, load unmodified <strong>MuJoCo Menagerie</strong> robots up to <strong>4,284 FPS</strong>, match <strong>Isaac Replicator</strong> at <strong>123,858 cam-FPS</strong> (<code>+0.32%</code> DR overhead), generate <strong>1,001,472</strong> factory frames in <code>142.58 s</code>, fuse <code>Sim-to-VLA</code> with <code>-1.84%</code> overhead, simulate 4-sensor <strong>GelSight</strong> at <strong>270,625 sensor-FPS</strong>, and backprop pure RGB pixel loss through 12 <code>mjx.step</code> physics steps."
      },
      c7: {
        num: "07 / 08 · TELEMETRY & INDEX",
        rail: "Switch subsystem telemetry tracks or filter all 24 QA-audited PNG plates. Click any image for the Darkroom Lightbox & 1:1 Pixel Loupe.",
        title: "Subsystem Telemetry Console &amp; Full <em>24-Plate</em> Verified Archive.",
        lead: "Inspect exact live TPU v5p measurements across all five domains below, or filter every visually audited render. Every plate opens in the full-resolution Darkroom Lightbox with a native <strong>1:1 Pixel Loupe</strong> (<code>L</code>) to inspect 4K/8K detail without compression artifacts."
      },
      c8: {
        num: "08 / 08 · LIMITS & COLOPHON",
        rail: "Single-host 4-chip ceiling, honest notes on what failed QA or stayed unrun before handback, hardware tradeoff table, and BibTeX.",
        title: "4-Chip HBM Extreme Ceiling, <em>Honest Boundaries</em> &amp; Tradeoffs.",
        lead: "Good engineering states boundaries clearly alongside breakthroughs. Here are the single-host 4-chip Pod Capstone measurements (<code>383 GB</code> HBM), what our strict Visual QA Gate rejected, and how TPU-RTX compares honestly against NVIDIA H100 Cycles and Isaac Sim / OptiX.",
        honest: "1) Visual QA Gate Rejections: Any render exhibiting tile seams, clay shading, or clipped framing (such as our early 16×16-seamed <code>garden_32v_grad</code> output) was explicitly excluded from publication. 2) 32-Chip Multi-Host Boundary: All single-host 1-chip & 4-chip benchmarks (including the 246.4 s Pod Capstone 4-chip dry-run) ran and verified end-to-end on live TPU v5p silicon. During our 8-host (32-chip) cross-node run, an older SSH cleanup script split multiline PIDs and left <code>/dev/vfio/3</code> busy right before our strict 09:00 BJT cluster return deadline. Rather than risk overtime or extrapolate unverified Pod numbers, we patched the harness cleanly, scrubbed all 8 hosts to 8/8 IDLE on schedule, and report exclusively 4-chip measured numbers here.",
        authorBio: "Systems & graphics engineer exploring compiler-native rendering, differentiable physics, and high-throughput Physical AI on custom ML accelerators.",
        ack: "Third-Party Asset Acknowledgments: Walt Disney Animation Studios Moana Island Scene v2.1 (Non-Commercial License); Blender Open Data / Blender Studio (BMW27, Barcelona Pavilion, Barbershop, Classroom, Junkshop, Lone Monk, Monster Under the Bed, Fishy Cat — CC0 / CC-BY); Mip-NeRF 360 photogrammetry captures (Barron et al.); Google DeepMind mujoco_menagerie (Apache-2.0 / BSD)."
      },
      podStats: [
        ["Bench 1 & 2 · Embodied Physics + 3DGS", "51,548 FPS MJX+RTX  |  155,858 FPS 3DGS", "Plus distributed 132.7M splats @ 1080p composited via ICI all_gather (3.30 FPS, 438.1 MSplats/s)."],
        ["Bench 3 · 128×2048 Batch 360° LiDAR", "31,460 env-FPS  |  8.25 Billion Rays/s", "B=4,096 parallel 128×2048 LiDAR scans in 130.2 ms (host_callbacks=0, gather_ops=0)."],
        ["Bench 4 · Geometry-Sharded Unique Triangles", "2.15B Unique Non-Instanced Tris (147 GB HBM)", "Constructed on-device in 1.89 s and traversed via ICI jax.lax.pmin @ 4K (4.47 s) & 8K (8.54 s)."],
        ["Bench 5 · Spatial-Sharded Native 8K 3-Bounce PT", "218.9 samples/min @ 7680×4320 (726.3 MRays/s)", "Unbiased 3-bounce Monte Carlo GI @ 33.18M pixels (host_callbacks=0, send_recv_ops=0)."]
      ],
      compRows: [
        ["Embodied Vision RL, VLA & Tactile", "<strong>Zero-Copy Single XLA Graph</strong>: <code>266k FPS</code> E2E, <code>1M</code> factory frames in <code>142.6s</code>, <code>270.6k/s</code> GelSight, <code>-1.84%</code> Sim-to-VLA overhead", "Static offline rendering (no interactive RL loop)", "Mature CUDA Warp/Isaac ecosystem on RTX/L40S/H100"],
        ["Real-to-Sim 3DGS / 4DGS + Physics", "<strong>Zero-Gather Native TPU</strong>: <code>142k FPS</code> (<code>64²</code>), <code>300M</code> splats @ 4K (4c HBM), <code>8K garden 29.2 MPix/s</code>, <code>23.4 steps/s</code> 32-view <code>jax.grad</code>, live <code>4DGS × MJX</code> push", "Cycles mesh + NRC only", "Requires separate CUDA gsplat + USD/PhysX bridge"],
        ["End-to-End Diff Physics + Render", "<strong>36.08 dB / 1.63 cm EE Error</strong> via <code>jax.value_and_grad(mjx.step + PBR)</code> at <code>27.9 grad-steps/s</code>", "Non-differentiable offline path tracer", "Separate Warp diff-sim / Nvdiffrast integration"],
        ["Film OpenUSD & Blender Studio", "<strong>Compiles Direct from Source</strong>: Disney <em>Moana</em> (<code>22.30B</code> tris, 4K/8K/Sunset/Reef), <code>BMW27 37.65 dB</code>, <code>Monster 36.98 dB</code>, <code>Barcelona 54.9M</code>, <code>Barbershop 31.0 dB</code>", "<strong>Full Native Blender 5.3 Cycles</strong> (mature BSDF library)", "<strong>Mature Omniverse Kit + MaterialX</strong> + hardware RT Cores"],
        ["AI Relighting, Denoising & DLSS", "<strong>Native MXU BF16/INT8</strong>: 4K Denoise <code>87.8 FPS</code>, 4K DLSS <code>58.9 FPS</code>, Static Relight <code>78.6 FPS</code> 4K / <code>307 FPS</code> 1080p", "External denoiser / Optix OIDN", "Hardware Tensor Core DLSS 3.5 + Ray Reconstruction"]
      ],
      ui: {
        copy: "Copy", copied: "Copied", zoom4k: "Inspect PNG", loupe: "1:1 Pixel Loupe (L)", raw: "Open Raw PNG ↗",
        footerL: "TPU-RTX · Google Cloud TPU v5p (JAX + Pallas Mosaic · Zero host_callbacks · Zero HBM Gathers)",
        b_mjx: "MJX & Robotics (w0/w1/w6)", b_gs3d: "3DGS / 4DGS / 300M (w3)", b_cinema: "Moana & Blender Studio (w2/w5/w7)", b_dlss: "TPU-DLSS 3.5 & Relight (w4)", b_fx: "Path Tracing, ReSTIR & Optics (w7/w6)",
        g_all: "All Audited Plates (24)", g_film: "Film & Blender Studio (8)", g_optics: "Optical FX & ReSTIR (5)", g_gs3d: "Real-World 3DGS / 4DGS (4)", g_dlss: "TPU-DLSS 3.5 & Relight (3)", g_robotics: "Menagerie, Isaac & Diff Phys (3)", g_pod: "4-Chip HBM Ceiling (1)",
        th1: "Workload", th2: "TPU-RTX (Google TPU v5p)", th3: "GPU Cycles (rtx-h100)", th4: "NVIDIA Isaac Sim / OptiX"
      }
    },

    zh: {
      pageTitle: "TPU-RTX — 脉动阵列也能像 GPU 一样光追、路径追踪与 3DGS | Feiyang Liu",
      themeNight: "暗房",
      themeDay: "纸本",
      nav: {
        c1: "01 架构", c2: "02 电影USD", c3: "03 光学", c4: "04 3DGS",
        c5: "05 神经渲染", c6: "06 具身智能", c7: "07 数据与图库", c8: "08 极限与边界"
      },
      hero: {
        kicker: "实测于 GOOGLE CLOUD TPU v5p 真机硅片 · ZERO HOST_CALLBACK · ZERO GATHER_OPS",
        by: "作者",
        license: "Apache-2.0 开源",
        title: "让脉动阵列像 GPU 一样<em>光线追踪</em>、路径追踪与高斯溅射。",
        lead: "<strong>TPU-RTX</strong> 直接用原生 <strong>JAX + Pallas Mosaic</strong> 把迪士尼《海洋奇缘》<em>Moana</em> OpenUSD（<strong>223 亿</strong>三角面）、Blender Studio 电影场景、原生 <strong>8K / 3 亿</strong> 3DGS、高阶波动光学，以及 <strong>266,209 FPS</strong> 可微 MuJoCo MJX + Isaac 具身仿真全部编译进 TPU v5p 的 HBM 与 64 MiB VMEM —— 全程零主机回调、零动态 HBM Gather。",
        ctaGallery: "检阅 24 张过审实景图",
        ctaBench: "交互式性能控制台",
        stamp: "视觉 QA 严审专刊 · 全部帧均在真机 TPU v5p 跑通并逐张人工目视核验 (interpret=False)",
        t0i: "展板 01 · OPENUSD v2.1",
        t0t: "迪士尼 Moana (223 亿面)：黄金日落与《水之道》珊瑚礁",
        t1i: "展板 02 · 原生 8K 3DGS",
        t1t: "Mip-NeRF 360 garden (583 万 SH3) @ 7680×4320 1:1 裁切",
        t2i: "展板 03 · 4K 物理光学",
        t2t: "原生 4K f/1.4 32-spp 薄透镜景深散景 (108.7 ms/帧)"
      },
      kpis: [
        ["4 芯融合 MJX+光追", "266,209 FPS", "<strong>921,309 FPS</strong> 纯渲染等效 (<code>64², B=8192</code>)"],
        ["迪士尼 Moana OpenUSD", "223 亿三角面", "<strong>4K 5.13 s · 8K 17.24 s</strong> · 日落与水下珊瑚礁"],
        ["3DGS 速度与显存上限", "14.2 万 FPS / 3 亿", "<code>142,008 FPS</code> (<code>64²</code>) · 4 芯 HBM 塞入 <strong>3 亿高斯</strong>"],
        ["原生 8K 实景 3DGS", "7680×4320", "<code>583 万</code> SH3 <code>garden</code> <strong>29.19 MPix/s</strong>"],
        ["DLSS 3.5 与 4K 重打光", "58.9 / 78.6 FPS", "DLSS <strong>3.91×</strong> · 零光追重打光 <strong>78.6 FPS</strong> BF16"],
        ["Blender Cycles 对齐度", "37.65 / 36.98 dB", "<code>BMW27</code> <strong>0.9813</strong> SSIM · <code>Monster</code> <strong>0.121 mm</strong>"],
        ["亿级唯一三角面 @ 4K/8K", "21.5 亿 / 1,022 亿", "<code>21.5 亿</code>唯一面 (ICI <code>pmin</code>, 4c) · <strong>1,022 亿</strong>实例化 8K"],
        ["可微物理 + 百万帧 VLA", "1.63 cm / 142.6s", "<code>jax.grad</code> <strong>36.08 dB</strong> · GelSight <strong>270.6k/s</strong>"]
      ],
      c1: {
        num: "01 / 08 · 硅片洞察",
        rail: "为什么把 SIMT GPU 光追代码硬搬到脉动阵列会瞬间卡死，以及 64 叉 Wide-BVH + VMEM 对齐如何逆转局面。",
        title: "用 CUDA 思路写 TPU 光追会撞死在 HBM 延迟墙；围绕 <em>64 MiB VMEM</em> 重构即刻起飞。",
        lead: "Google Cloud TPU v5p 单芯拥有 <strong>459 TFLOP/s</strong> BF16 算力、<strong>2.765 TB/s</strong> HBM 带宽、<strong>95.74 GiB</strong> 巨大显存以及 <strong>64 MiB</strong> 超高速片上 VMEM —— 但算力脊点（Ridge Point）高达 <strong>166 FLOP/B</strong>，任何动态 HBM Gather 都会让性能雪崩。TPU-RTX 的每次数量级跃迁，都来自把求交、溅射与着色彻底改写为 128 通道对齐的片上向量块（<code>gather_ops = 0</code>）。",
        pillars: [
          ["VMEM 宽树遍历", "1. 三级 64 叉 Wide-BVH", "把 64 个子节点包围盒紧致打包成 (8, 128) 向量块，使 Pallas Mosaic 在单条 VPU 指令内完成 64 路包围盒测试，并配合 pltpu.async_copy 异步流水流式读取叶节点，平稳扩展至 223 亿面 Moana 与 21.5 亿唯一三角面。"],
          ["片上核融合与零 Gather", "2. In-VMEM 核融合与分块 3DGS", "把光线生成、求交、GGX/AO/软阴影与材质去重直接融进 VMEM 寄存器，彻底消除 G-Buffer 往返（92.1 万 FPS 等效）；分块 VMEM take_along_axis 将 3DGS 从 1.2k 推至 14.2 万 FPS，并把 3 亿个高斯塞进 4 芯 HBM。"],
          ["MXU 与端到端可微", "3. 原生 MXU 神经加速与 jax.grad 物理", "像素全程不离 TPU HBM，让 BF16 MXU 可以几乎零搬运地运行 87.8 FPS 4K 去噪、58.9 FPS 4K TPU-DLSS 3.5、78.6 FPS 零光追重打光、-1.84% 开销 Sim-to-VLA，以及直接穿透 mjx.step 的像素到力矩反向传播。"]
        ],
        diagCap: "全模块在 XLA HLO 层通过硬验证：host_callbacks = 0，send_recv_ops = 0，gather_ops = 0。",
        sdkBar: "极速接入 SDK · tpu_mjx_render.adapters 下内置 8 大生态适配器"
      },
      c2: {
        num: "02 / 08 · 电影 OpenUSD 与 Studio",
        rail: "直接编译迪士尼 Moana OpenUSD v2.1 (223 亿面) 与 Blender Studio 官方场景至 Wide-BVH-64 + AgX。",
        title: "223 亿面迪士尼《海洋奇缘》<em>Moana</em> 与 Blender 电影级基准，全部直跑 TPU 硅片。",
        lead: "拒绝只在简单几何块上自嗨——我们直接从官方 <strong>OpenUSD</strong> 与 <code>.blend</code> Cycles 材质节点图编译真实影视资产。迪士尼 <em>Moana Island</em> v2.1（等效 223 亿三角面）在 TPU 上仅占 <strong>4.653 GiB</strong> HBM，支持黄金日落、水下珊瑚礁焦散，甚至让 MJX 人形机器人直接踩在 Motunui 沙滩上跑；Blender 官方场景与 Cycles EXR 深度误差低至 <strong>0.121 mm</strong>。"
      },
      c3: {
        num: "03 / 08 · 高阶物理光学",
        rail: "8 波段 Airy 薄膜干涉、各向异性拉丝金属、12 次递归镜廊、柯西棱镜色散、玉石 SSS、1,024 盏灯 ReSTIR 与 4K 景深。",
        title: "波动光学、12 次无限镜廊与 1,024 光源 <em>TPU-ReSTIR DI</em>。",
        lead: "真正的电影级光传输远不止一次弹射光线。TPU-RTX 在单条 <code>jax.lax.while_loop</code> 与 HBM 内实现了 8 波段 CIE 1931 薄膜干涉、Heitz VNDF 各向异性拉丝金属、12 次递归镜面反射、6 波段柯西棱镜色散、16 步随机游走玉石 SSS、1,024 动态光源 ReSTIR DI，以及 <strong>108.7 ms 原生 4K f/1.4 薄透镜虚化</strong>。"
      },
      c4: {
        num: "04 / 08 · 3DGS 与 4DGS",
        rail: "消除 Scatter/Gather 后实现 14.2 万 FPS 溅射、4 芯 3 亿高斯 HBM 极限、原生 8K 航拍重建与 MJX 实时物理推挤。",
        title: "单机 HBM 塞入 <em>3 亿</em> 高斯、原生 8K 实景与 4DGS × MJX 动态推挤。",
        lead: "传统 GPU 3DGS 高度依赖原子写与不规则 Gather。改为 VMEM 分块 EWA 排序与矩阵混合后，批量速度飙到 <strong>142,008 FPS</strong>（<code>64²</code>），单机 4 芯硬塞进 <strong>300,000,000</strong> 个高斯（占 380 GB HBM 的 79.44%），原生 <strong>8K</strong> 渲染 583 万 SH3 的真实 <code>garden</code> 场景（29.19 MPix/s），32 视角 <code>jax.grad</code> 训练 23.42 steps/s，并能在真实厨房高斯里实时推挤乐高推土机。"
      },
      c5: {
        num: "05 / 08 · MXU 神经渲染加速",
        rail: "把 459 TFLOP/s BF16 MXU 用在刀刃上：TPU-DLSS 3.5 时序超分、4K 路径追踪降噪与零光追交互重打光。",
        title: "让 MXU 做它最擅长的事：<em>58.9 FPS</em> 4K DLSS、<em>87.8 FPS</em> 4K 降噪与零光追重打光。",
        lead: "既然 G-Buffer 本就在 HBM 里，直接接进 BF16/INT8 MXU 几乎零额外通信。基于精确 4×4 双线性重投影的 <strong>TPU-DLSS 3.5</strong> 将 1080p 1-spp 以 <strong>58.93 FPS</strong> 重建至原生 4K（端到端提速 <strong>3.91×</strong>、闪烁下降 <strong>7.01×</strong>）；4K 蒙特卡洛 AI 降噪器达到 <strong>87.78 FPS</strong>（11.39 ms）；静态 G-Buffer 昼夜与材质重打光在原生 4K BF16 下跑出 <strong>78.60 FPS</strong>（零光线追踪开销）。"
      },
      c6: {
        num: "06 / 08 · 具身智能与可微物理",
        rail: "从 266,209 FPS MJX 视觉 RL 与 MuJoCo Menagerie，到 142.6 秒生成百万 VLA 帧、GelSight 视触觉与像素反传控制。",
        title: "从 <em>266,209 FPS</em> 视觉 RL、142.6 秒百万 VLA 帧，到像素穿透 <em>MJX</em> 反传力矩。",
        lead: "在具身智能中，渲染必须与物理、传感器和策略网络融为一体才有意义。我们实现 <strong>266,209 FPS</strong> 物理+渲染端到端，通用加载官方 <strong>MuJoCo Menagerie</strong> 最高 <strong>4,284 FPS</strong>，<strong>Isaac Replicator</strong> 域随机化仅增 <code>+0.32%</code> 开销（123,858 cam-FPS），<strong>142.58 秒生成 100 万帧</strong> 工厂数据，<strong>Sim-to-VLA</strong> 融合开销低至 <code>-1.84%</code>，4 路 <strong>GelSight</strong> 视触觉 <strong>270,625 sensor-FPS</strong>，并让纯 RGB 像素误差直接反传 12 步 <code>mjx.step</code> 求解 7 自由度关节力矩。"
      },
      c7: {
        num: "07 / 08 · 实测控制台与全集索引",
        rail: "切换五大子系统实测跑道，或按门类筛选全部 24 张过审高清图；点击任意图开启暗房灯箱与 1:1 像素放大镜。",
        title: "子系统实时数据控制台与全 <em>24 张</em> 过审出图总览。",
        lead: "下方可切换五大子系统的真机实测条形图，或筛选全部 24 张过审高清出图。点击任意展板都会拉起原图 <strong>4K/8K 暗房灯箱</strong>，支持按 <code>L</code> 开启 <strong>1:1 原生像素放大镜</strong>，逐像素检视焦散、毛发与 8K 锐度。"
      },
      c8: {
        num: "08 / 08 · 极限上限、边界与署名",
        rail: "单机 4 芯 HBM 极限压测结果、严谨公开的视觉 QA 淘汰与 32 芯截停边界说明、架构对比表及 BibTeX 引用。",
        title: "单机 4 芯极限天花板、<em>坦诚边界</em> 与硬件对比。",
        lead: "好的系统工程不仅展示高光数字，更要诚实标明边界。下方汇总单机 4 芯 Pod Capstone（383 GB HBM）全套实测数据、视觉 QA 门禁淘汰原则，以及 TPU-RTX 与 NVIDIA H100 Cycles / Isaac Sim / OptiX 的客观对比。",
        honest: "1）视觉 QA 严审淘汰机制：任何出现 16×16 拼缝、白模感或黑底裁切的渲染图（例如早期出现拼缝的 garden_32v_grad 输出）均被直接拦截，不予纳入展示与仓库提交。2）32 芯多机截停边界说明：全部单机 1 芯与 4 芯实验（含 246.4 s 跑完的 4 芯 Capstone 全套压测）均已在真机完成校验；最后一次 8 机（32 芯）联合压测时，因旧清理脚本 PID 换行残留占住了 /dev/vfio/3 且已临近 09:00 BJT 归还集群死线，为了守约提前把 8 台主机干干净净交还给大盘团队，我们没有强行超时重跑，也绝不把 4 芯数据外推冒充 32 芯真测。多机脚本中的设备映射与清理缺陷均已在仓库中修复备妥。",
        authorBio: "聚焦编译器原生图形渲染、可微物理仿真与 TPU/AI 加速器高吞吐具身智能系统的研究者与工程师。",
        ack: "第三方资产鸣谢：Walt Disney Animation Studios Moana Island Scene v2.1（非商业研究许可）；Blender Open Data / Blender Studio 场景（BMW27、Barcelona Pavilion、Barbershop、Classroom、Junkshop、Lone Monk、Monster Under the Bed、Fishy Cat — CC0 / CC-BY）；Mip-NeRF 360 真实摄影测量数据集；Google DeepMind mujoco_menagerie（Apache-2.0 / BSD）。"
      },
      podStats: [
        ["基准 1 & 2 · 具身物理 + 3DGS", "51,548 FPS MJX+RTX  |  155,858 FPS 3DGS", "外加通过 ICI all_gather 合成的 1.327 亿分布式高斯 @ 1080p（3.30 FPS，438.1 MSplats/s）。"],
        ["基准 3 · 128×2048 批量 360° 激光雷达", "31,460 env-FPS  |  每秒 82.5 亿条射线", "B=4,096 并发 128×2048 线 LiDAR 扫描仅需 130.2 ms（host_callbacks=0，gather_ops=0）。"],
        ["基准 4 · 几何分片唯一三角面上限", "21.5 亿唯一非实例化三角面 (147 GB HBM)", "1.89 s 片上构建完毕，并通过 ICI jax.lax.pmin 完成 4K（4.47 s）与 8K（8.54 s）遍历。"],
        ["基准 5 · 空间分片原生 8K 3-Bounce 路径追踪", "7680×4320 达 218.9 spm (726.3 MRays/s)", "3,318 万像素无偏 3 次反弹蒙特卡洛全局光（host_callbacks=0，send_recv_ops=0）。"]
      ],
      compRows: [
        ["具身视觉 RL、VLA 与触觉", "<strong>单张 XLA 零拷贝图</strong>：端到端 <code>266k FPS</code>、142.6s 生成百万帧、GelSight <code>270.6k/s</code>、Sim-to-VLA 仅 <code>-1.84%</code> 开销", "静态离线渲染器（无交互式 RL 闭环）", "成熟的 CUDA Warp / Isaac 生态与 RTX/H100 工具链"],
        ["Real-to-Sim 3DGS / 4DGS + 物理", "<strong>零 Gather 原生 TPU</strong>：<code>142k FPS</code>（<code>64²</code>）、4 芯 <code>3 亿</code>高斯 4K、<code>8K garden 29.2 MPix/s</code>、32 视角训练 <code>23.4 steps/s</code>、实时 <code>4DGS × MJX</code> 推挤", "仅支持 Cycles Mesh + NRC", "需要独立 CUDA gsplat 与 USD/PhysX 跨端桥接"],
        ["端到端可微物理 + 光追渲染", "<strong>36.08 dB / 末端误差 1.63 cm</strong>：<code>jax.value_and_grad(mjx.step + PBR)</code> 实测 <code>27.9 grad-steps/s</code>", "不可微离线路径追踪器", "需单独组合 Warp 可微物理与 Nvdiffrast"],
        ["电影 OpenUSD 与 Blender Studio", "<strong>源资产直编</strong>：迪士尼 <em>Moana</em>（<code>223 亿</code>面 4K/8K/日落/珊瑚礁）、<code>BMW27 37.65 dB</code>、<code>Monster 36.98 dB</code>、<code>Barcelona 5,491 万面</code>、<code>Barbershop 31.0 dB</code>", "<strong>完整原生 Blender 5.3 Cycles</strong>（全特性 BSDF 节点库）", "<strong>成熟 Omniverse Kit + MaterialX</strong> 与硬件 RT Core"],
        ["AI 重打光、降噪与超分 DLSS", "<strong>原生 MXU BF16/INT8</strong>：4K 降噪 <code>87.8 FPS</code>、4K DLSS <code>58.9 FPS</code>、零光追重打光 4K <code>78.6 FPS</code> / 1080p <code>307 FPS</code>", "依赖外置降噪器 / OptiX OIDN", "专用 Tensor Core 硬件 DLSS 3.5 与光线重建"]
      ],
      ui: {
        copy: "复制", copied: "已复制", zoom4k: "检视原图 PNG", loupe: "1:1 像素放大镜 (L)", raw: "打开原始 PNG ↗",
        footerL: "TPU-RTX · Google Cloud TPU v5p (JAX + Pallas Mosaic · 零 host_callbacks · 零 HBM Gather)",
        b_mjx: "MJX 与具身仿真 (w0/w1/w6)", b_gs3d: "3DGS / 4DGS / 3 亿高斯 (w3)", b_cinema: "Moana 与 Blender 电影 (w2/w5/w7)", b_dlss: "TPU-DLSS 3.5 与重打光 (w4)", b_fx: "路径追踪 / ReSTIR / 光学 (w7/w6)",
        g_all: "全部过审图集 (24)", g_film: "电影 USD 与 Blender (8)", g_optics: "物理光学与 ReSTIR (5)", g_gs3d: "实景 3DGS / 4DGS (4)", g_dlss: "DLSS 3.5 与重打光 (3)", g_robotics: "Menagerie / Isaac / 可微物理 (3)", g_pod: "4 芯 HBM 极限 (1)",
        th1: "任务类型", th2: "TPU-RTX (Google TPU v5p)", th3: "GPU Cycles (rtx-h100)", th4: "NVIDIA Isaac Sim / OptiX"
      }
    }
  }
};

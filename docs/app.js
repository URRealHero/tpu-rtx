/* ==========================================================================
   TPU-RTX Interactive Monograph Engine
   Bilingual (EN / 中文) · Dual-Theme (Obsidian / Paper) ·
   3 Split Sliders (19 Presets) · 2 Single-Scene Stages · 10 Vector SVG Charts ·
   Zero-Overflow Searchable JSON Ledger · 25-Plate Quick-Jump Directory · 1:1 Loupe
   Author: Feiyang Liu
   ========================================================================== */

(function () {
  const DATA = window.TPU_RTX_DATA;
  const platesById = Object.fromEntries(DATA.plates.map((p) => [p.id, p]));
  const GITHUB_METRICS_BASE = "https://github.com/ZaychikLiu/tpu-rtx/blob/main/out/metrics/";

  let state = {
    lang: document.documentElement.getAttribute("data-lang") === "zh" ? "zh" : "en",
    theme: document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark",
    heroIdx: 0,
    stage3dgsIdx: 0,
    stageMenagerieIdx: 0,
    videoCh3Cat: "all",
    videoCh3Idx: 0,
    videoCh5Idx: 0,
    benchKey: "mjx",
    dirFilter: "all",
    ledgerCat: "all",
    ledgerQuery: "",
    cmpIdx: { blender: 0, optics: 0, neural: 0 },
    cmpPos: { blender: 50, optics: 50, neural: 50 },
    lbIdx: 0,
    loupeOn: false
  };

  function getT() {
    return DATA.copy[state.lang];
  }

  function updateThemeButton() {
    const t = getT();
    const icon = document.getElementById("theme-icon");
    const label = document.getElementById("theme-label");
    if (state.theme === "dark") {
      icon.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`;
      label.textContent = t.themeNight;
    } else {
      icon.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.93 4.93l1.41 1.41m11.32 11.32 1.41 1.41M2 12h2m16 0h2M4.93 19.07l1.41-1.41m11.32-11.32 1.41-1.41"/></svg>`;
      label.textContent = t.themeDay;
    }
  }

  function updateLangButton() {
    const btn = document.getElementById("lang-toggle");
    btn.textContent = state.lang === "en" ? "中文 / EN" : "EN / 中文";
  }

  function renderPlateCard(plate) {
    const lang = state.lang;
    const t = getT();
    const badges = plate.badges[lang]
      .map(([cls, txt]) => `<span class="badge ${cls}">${txt}</span>`)
      .join("");
    return `
      <article class="plate" data-cat="${plate.cat}" data-plate-id="${plate.id}">
        <div class="plate-media" data-open-id="${plate.id}">
          <img src="${plate.web}" alt="${plate.title[lang]}" loading="lazy" />
        </div>
        <div class="plate-body">
          <div class="plate-meta-bar">
            <div class="plate-badges">${badges}</div>
            <button class="plate-inspect" type="button" data-open-id="${plate.id}">${t.ui.zoom4k}</button>
          </div>
          <div class="plate-title">${plate.title[lang]}</div>
          <div class="plate-desc">${plate.desc[lang]}</div>
        </div>
      </article>
    `;
  }

  function renderCh3Spreads() {
    const el = document.getElementById("spread-ch3");
    if (!el) return;
    el.innerHTML = DATA.spreadCh3
      .map((id) => platesById[id])
      .filter(Boolean)
      .map((plate) => renderPlateCard(plate))
      .join("");
  }

  function renderHeroStage(idx) {
    state.heroIdx = idx;
    const item = DATA.heroStage[idx];
    const lang = state.lang;
    const view = document.getElementById("hero-stage-view");
    if (item.video) {
      view.classList.add("is-video");
      delete view.dataset.openId;
      view.innerHTML = `<video id="hero-stage-video" src="${item.video}" poster="${item.poster}" autoplay loop muted playsinline controls preload="metadata"></video>`;
      const v = document.getElementById("hero-stage-video");
      if (v) v.play().catch(() => {});
    } else {
      const plate = platesById[item.plateId];
      view.classList.remove("is-video");
      view.dataset.openId = plate.id;
      view.innerHTML = `<img id="hero-stage-img" src="${plate.web}" alt="${plate.title[lang]}" />`;
    }
    document.getElementById("hero-hud-left").innerHTML = item.hudL[lang];
    document.getElementById("hero-hud-right").innerHTML = item.hudR[lang];

    document.querySelectorAll("#hero-tabs .cinema-tab").forEach((b) => {
      b.classList.toggle("active", Number(b.dataset.hero) === idx);
    });
  }

  function renderVideoStageCh3() {
    const items = DATA.videoStageCh3.filter((v) => state.videoCh3Cat === "all" || v.cat === state.videoCh3Cat);
    if (!items.length) return;
    if (state.videoCh3Idx >= items.length) state.videoCh3Idx = 0;
    const item = items[state.videoCh3Idx];
    const lang = state.lang;
    const vid = document.getElementById("ch3-video-el");
    const tabs = document.getElementById("ch3-video-tabs");
    if (!vid || !tabs) return;
    if (vid.getAttribute("src") !== item.mp4) {
      vid.setAttribute("src", item.mp4);
      vid.setAttribute("poster", item.poster);
      vid.load();
      vid.play().catch(() => {});
    }
    document.getElementById("ch3-video-hudL").innerHTML = item.hudL[lang];
    document.getElementById("ch3-video-hudR").innerHTML = item.hudR[lang];
    document.querySelectorAll("#ch3-video-cats .tab-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.vcat === state.videoCh3Cat);
    });
    tabs.innerHTML = items
      .map(
        (it, i) => `
        <button class="cinema-tab ${i === state.videoCh3Idx ? "active" : ""}" data-v3="${i}" type="button">
          <span class="cinema-tab-idx">${it.tabIdx[lang]}</span>
          <span class="cinema-tab-title">${it.tabTitle[lang]}</span>
        </button>`
      )
      .join("");
  }

  function renderVideoStageCh5() {
    const items = DATA.videoStageCh5;
    if (!items.length) return;
    if (state.videoCh5Idx >= items.length) state.videoCh5Idx = 0;
    const item = items[state.videoCh5Idx];
    const lang = state.lang;
    const vid = document.getElementById("ch5-video-el");
    const tabs = document.getElementById("ch5-video-tabs");
    if (!vid || !tabs) return;
    if (vid.getAttribute("src") !== item.mp4) {
      vid.setAttribute("src", item.mp4);
      vid.setAttribute("poster", item.poster);
      vid.load();
      vid.play().catch(() => {});
    }
    document.getElementById("ch5-video-hudL").innerHTML = item.hudL[lang];
    document.getElementById("ch5-video-hudR").innerHTML = item.hudR[lang];
    tabs.innerHTML = items
      .map(
        (it, i) => `
        <button class="cinema-tab ${i === state.videoCh5Idx ? "active" : ""}" data-v5="${i}" type="button">
          <span class="cinema-tab-idx">${it.tabIdx[lang]}</span>
          <span class="cinema-tab-title">${it.tabTitle[lang]}</span>
        </button>`
      )
      .join("");
  }

  function renderSingleStage(prefix, items, idx, dataAttr) {
    const item = items[idx];
    const plate = platesById[item.plateId];
    const lang = state.lang;
    const img = document.getElementById(`${prefix}-img`);
    const view = document.getElementById(`${prefix}-view`);
    const hudL = document.getElementById(`${prefix}-hudL`);
    const hudR = document.getElementById(`${prefix}-hudR`);
    const tabsEl = document.getElementById(`${prefix}-tabs`);
    if (!img || !view || !tabsEl || !plate) return;

    img.src = plate.web;
    img.alt = plate.title[lang];
    view.dataset.openId = plate.id;
    hudL.innerHTML = item.hudL[lang];
    hudR.innerHTML = item.hudR[lang];

    tabsEl.innerHTML = items
      .map(
        (it, i) => `
        <button class="cinema-tab ${i === idx ? "active" : ""}" data-${dataAttr}="${i}" type="button">
          <span class="cinema-tab-idx">${it.tabIdx[lang]}</span>
          <span class="cinema-tab-title">${it.tabTitle[lang]}</span>
        </button>`
      )
      .join("");
  }

  function renderCompareLab(containerId, labKey) {
    const el = document.getElementById(containerId);
    if (!el) return;
    const lab = DATA.compareLabs[labKey];
    const lang = state.lang;
    const idx = state.cmpIdx[labKey];
    const pos = state.cmpPos[labKey];
    const preset = lab.presets[idx];

    el.innerHTML = `
      <div class="compare-head">
        <span class="compare-head-title">${lab.title[lang]}</span>
        <div class="compare-presets">
          ${lab.presets
            .map(
              (pr, i) => `<button type="button" class="tab-btn ${i === idx ? "active" : ""}" data-cmp-lab="${labKey}" data-cmp-idx="${i}">${pr.tab[lang]}</button>`
            )
            .join("")}
        </div>
      </div>
      <div class="compare-stage" style="--pos:${pos}%">
        <div class="compare-inner">
          <img class="compare-img-base" src="${preset.bImg}" alt="${preset.bLbl[lang]}" />
          <img class="compare-img-top" src="${preset.aImg}" alt="${preset.aLbl[lang]}" />
          <div class="compare-divider"><div class="compare-handle">&#8249;&#8250;</div></div>
          <input class="compare-range" type="range" min="2" max="98" value="${pos}" aria-label="${preset.tab[lang]}" data-cmp-range="${labKey}" />
        </div>
      </div>
      <div class="compare-bar">
        <span class="compare-side-a">${preset.aLbl[lang]}</span>
        <span class="compare-pct" data-cmp-pct="${labKey}">${pos}% / ${100 - pos}%</span>
        <span class="compare-side-b">${preset.bLbl[lang]}</span>
      </div>
    `;
  }

  function renderPostMortems() {
    const t = getT();
    document.getElementById("postmortem-grid").innerHTML = t.c6.pmCards
      .map(
        ([tag, id, title, cause, fix]) => `
        <div class="pm-card">
          <div class="pm-head">
            <span class="pm-tag ${tag.includes("NEG") || tag.includes("负") || tag.includes("FAIL") ? "fail" : "fix"}">${tag}</span>
            <span class="pm-id">${id}</span>
          </div>
          <h3>${title}</h3>
          <p class="pm-row"><strong>${state.lang === "zh" ? "现象/根因:" : "Root Cause:"}</strong>${cause}</p>
          <p class="pm-row"><strong>${state.lang === "zh" ? "修复与结论:" : "Fix & Outcome:"}</strong>${fix}</p>
        </div>`
      )
      .join("");

    document.getElementById("ablation-tbody").innerHTML = t.c6.ablationRows
      .map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`)
      .join("");

    document.getElementById("rejected-pair").innerHTML = t.c6.rejected
      .map(
        (rj) => `
        <article class="plate rejected">
          <div class="rejected-banner">${rj.banner}</div>
          <div class="plate-media" data-open-id="${rj.id}">
            <img src="${rj.img}" alt="${rj.title}" loading="lazy" />
          </div>
          <div class="plate-body">
            <div class="plate-meta-bar">
              <div class="plate-badges"><span class="badge fail">${state.lang === "zh" ? "QA 淘汰存档" : "QA Rejected Archive"}</span></div>
              <button class="plate-inspect" type="button" data-open-id="${rj.id}">${t.ui.zoom4k}</button>
            </div>
            <div class="plate-title">${rj.title}</div>
            <div class="plate-desc">${rj.desc}</div>
          </div>
        </article>`
      )
      .join("");
  }

  function renderLedger() {
    const q = state.ledgerQuery.trim().toLowerCase();
    const cat = state.ledgerCat;
    const rows = DATA.ledger.filter((r) => {
      if (cat !== "all" && r.cat !== cat) return false;
      if (!q) return true;
      return `${r.name} ${r.scale} ${r.speed} ${r.acc} ${r.file}`.toLowerCase().includes(q);
    });

    const tbody = document.getElementById("ledger-tbody");
    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="5" style="color:var(--muted);text-align:center;">${state.lang === "zh" ? "无匹配实验记录" : "No matching experiments."}</td></tr>`;
      return;
    }

    tbody.innerHTML = rows
      .map(
        (r) => `
        <tr>
          <td><strong>${r.name}</strong></td>
          <td><code>${r.scale}</code></td>
          <td class="tbl-num">${r.speed}</td>
          <td class="tbl-num">${r.acc}</td>
          <td><a class="json-link" href="${GITHUB_METRICS_BASE}${r.file}" target="_blank" rel="noopener" title="${r.file}">JSON ↗</a></td>
        </tr>`
      )
      .join("");
  }

  function renderPlateDirectory() {
    const dir = document.getElementById("plate-directory");
    if (!dir) return;
    const lang = state.lang;
    const t = getT();
    const catLabels = {
      film: { en: "Moana & Studio", zh: "电影/Studio" },
      optics: { en: "Optics", zh: "物理光学" },
      gs3d: { en: "Real 3DGS", zh: "实景 3DGS" },
      robotics: { en: "Menagerie & RoboTwin", zh: "具身机器人" },
      rejected: { en: "QA-Rejected", zh: "QA 淘汰" }
    };

    dir.innerHTML = DATA.plates
      .map((p, i) => {
        const badgeCls = p.cat === "rejected" ? "fail" : "cyan";
        return `
        <div class="dir-row" data-cat="${p.cat}">
          <div class="dir-main">
            <div class="dir-meta">
              <span>#${String(i + 1).padStart(2, "0")}</span>
              <span class="badge ${badgeCls}">${catLabels[p.cat][lang]}</span>
            </div>
            <div class="dir-title">${p.title[lang]}</div>
          </div>
          <div class="dir-actions">
            <button class="plate-inspect" type="button" data-open-id="${p.id}">${t.ui.zoom4k}</button>
          </div>
        </div>`;
      })
      .join("");

    applyDirectoryFilter(state.dirFilter);
  }

  function applyDirectoryFilter(filter) {
    state.dirFilter = filter;
    document.querySelectorAll("#plate-filters .tab-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.filter === filter);
    });
    document.querySelectorAll("#plate-directory .dir-row").forEach((row) => {
      row.classList.toggle("hidden", filter !== "all" && row.dataset.cat !== filter);
    });
  }

  function renderBench(key) {
    state.benchKey = key;
    const lang = state.lang;
    const d = DATA.benchmarks[key];
    document.querySelectorAll("#bench-tabs .tab-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.bench === key);
    });
    document.getElementById("bench-content").innerHTML = `
      <div class="bench-head-title">${d.title[lang]}</div>
      <div class="bench-head-note">${d.note[lang]}</div>
      <div class="bar-list">
        ${d.rows
          .map(
            (r) => `
          <div class="bar-row">
            <div class="bar-label">${r.label[lang]}</div>
            <div class="bar-track"><div class="bar-fill ${r.base ? "base" : ""}" style="width:${Math.max(r.pct, 3)}%"></div></div>
            <div class="bar-val">${typeof r.val === "number" ? r.val.toLocaleString() : r.val} ${r.unit} <span>(${r.extra[lang]})</span></div>
          </div>`
          )
          .join("")}
      </div>
    `;
  }

  function renderAllText() {
    const t = getT();
    document.title = t.pageTitle;
    document.documentElement.setAttribute("data-lang", state.lang);
    document.documentElement.setAttribute("lang", state.lang === "zh" ? "zh-CN" : "en");

    const mapI18n = {
      "nav.c1": t.nav.c1, "nav.c2": t.nav.c2, "nav.c3": t.nav.c3, "nav.c4": t.nav.c4,
      "nav.c5": t.nav.c5, "nav.c6": t.nav.c6,
      "hero.kicker": t.hero.kicker, "hero.by": t.hero.by, "hero.license": t.hero.license,
      "hero.ctaStart": t.hero.ctaStart, "hero.ctaGallery": t.hero.ctaGallery, "hero.stamp": t.hero.stamp,
      "hero.t0i": t.hero.t0i, "hero.t0t": t.hero.t0t,
      "hero.t1i": t.hero.t1i, "hero.t1t": t.hero.t1t,
      "hero.t2i": t.hero.t2i, "hero.t2t": t.hero.t2t,
      "hero.t3i": t.hero.t3i, "hero.t3t": t.hero.t3t,
      "vid.all": t.ui.vid_all, "vid.gs3d": t.ui.vid_gs3d, "vid.optics": t.ui.vid_optics, "vid.relight": t.ui.vid_relight,
      "hero.stackTag": t.hero.stackTag, "hero.stackTitle": t.hero.stackTitle, "hero.stackSub": t.hero.stackSub,
      "c1.num": t.c1.num, "c1.rail": t.c1.rail, "c1.gvtTag": t.c1.gvtTag, "c1.gvtTitle": t.c1.gvtTitle, "c1.gvtSub": t.c1.gvtSub,
      "c2.num": t.c2.num, "c2.rail": t.c2.rail, "c2.sdkBar": t.c2.sdkBar,
      "c2.ecoTag": t.c2.ecoTag, "c2.ecoTitle": t.c2.ecoTitle, "c2.ecoSub": t.c2.ecoSub,
      "c2.bmkTag": t.c2.bmkTag, "c2.bmkTitle": t.c2.bmkTitle, "c2.bmkSub": t.c2.bmkSub,
      "bs.h1": t.ui.bs1, "bs.h2": t.ui.bs2, "bs.h3": t.ui.bs3, "bs.h4": t.ui.bs4,
      "c3.num": t.c3.num, "c3.rail": t.c3.rail,
      "c4.num": t.c4.num, "c4.rail": t.c4.rail,
      "c5.num": t.c5.num, "c5.rail": t.c5.rail,
      "c6.num": t.c6.num, "c6.rail": t.c6.rail,
      "c6.honestHead": state.lang === "zh" ? "透明度声明 · 1 芯 / 4 芯 / 32 芯实测范围与 GPU 领先边界" : "TRANSPARENCY NOTE · VERIFIED 1c / 4c / 32c MEASUREMENTS & GPU BOUNDARIES",
      "c6.authorRole": state.lang === "zh" ? "作者与系统架构" : "AUTHOR & ARCHITECT",
      "c6.bibBar": state.lang === "zh" ? "BibTeX 引用 · Apache-2.0 协议" : "BibTeX Citation · Apache-2.0",
      "bench.t_mjx": t.ui.b_mjx, "bench.t_gs3d": t.ui.b_gs3d, "bench.t_cinema": t.ui.b_cinema,
      "bench.t_dlss": t.ui.b_dlss, "bench.t_fx": t.ui.b_fx,
      "gal.all": t.ui.g_all, "gal.film": t.ui.g_film, "gal.optics": t.ui.g_optics,
      "gal.gs3d": t.ui.g_gs3d, "gal.robotics": t.ui.g_robotics, "gal.rej": t.ui.g_rej,
      "tbl.h1": t.ui.th1, "tbl.h2": t.ui.th2, "tbl.h3": t.ui.th3, "tbl.h4": t.ui.th4,
      "gvt.h1": t.ui.gvt1, "gvt.h2": t.ui.gvt2, "gvt.h3": t.ui.gvt3,
      "ab.h1": t.ui.ab1, "ab.h2": t.ui.ab2, "ab.h3": t.ui.ab3, "ab.h4": t.ui.ab4, "ab.h5": t.ui.ab5,
      "ld.all": t.ui.ld_all, "ld.rob": t.ui.ld_rob, "ld.film": t.ui.ld_film, "ld.gs": t.ui.ld_gs, "ld.nn": t.ui.ld_nn,
      "lt.h1": t.ui.lt1, "lt.h2": t.ui.lt2, "lt.h3": t.ui.lt3, "lt.h4": t.ui.lt4, "lt.h5": t.ui.lt5,
      "ch.evoTag": t.ui.ev_t, "ch.evoTitle": t.ui.ev_h, "ch.evoSub": t.ui.ev_s,
      "ch.gsTag": t.ui.gs_t, "ch.gsTitle": t.ui.gs_h, "ch.gsSub": t.ui.gs_s,
      "ch.vmemTag": t.ui.vm_t, "ch.vmemTitle": t.ui.vm_h, "ch.vmemSub": t.ui.vm_s,
      "ch.mxuTag": t.ui.mx_t, "ch.mxuTitle": t.ui.mx_h, "ch.mxuSub": t.ui.mx_s,
      "ch.orbTag": t.ui.orb_t, "ch.orbTitle": t.ui.orb_h, "ch.orbSub": t.ui.orb_s,
      "ch.dpTag": t.ui.dp_t, "ch.dpTitle": t.ui.dp_h, "ch.dpSub": t.ui.dp_s,
      "ch.ppoTag": t.ui.pp_t, "ch.ppoTitle": t.ui.pp_h, "ch.ppoSub": t.ui.pp_s,
      "ch.g3dTag": t.ui.g3_t, "ch.g3dTitle": t.ui.g3_h, "ch.g3dSub": t.ui.g3_s,
      "ch.vlaTag": t.ui.vl_t, "ch.vlaTitle": t.ui.vl_h, "ch.vlaSub": t.ui.vl_s,
      "common.copy": t.ui.copy, "common.zoom4k": t.ui.zoom4k,
      "lb.loupe": t.ui.loupe, "lb.raw": t.ui.raw
    };

    document.querySelectorAll("[data-i18n]").forEach((el) => {
      const k = el.getAttribute("data-i18n");
      if (mapI18n[k] !== undefined) el.textContent = mapI18n[k];
    });

    document.getElementById("ledger-search").placeholder = t.ui.searchPlaceholder;
    document.getElementById("hero-title").innerHTML = t.hero.title;
    document.getElementById("hero-lead").innerHTML = t.hero.lead;

    document.getElementById("kpi-ledger").innerHTML = t.kpis
      .map(
        ([lbl, val, sub]) => `
        <div class="kpi-cell">
          <div class="kpi-label">${lbl}</div>
          <div class="kpi-val">${val}</div>
          <div class="kpi-sub">${sub}</div>
        </div>`
      )
      .join("");

    ["c1", "c2", "c3", "c4", "c5", "c6"].forEach((id) => {
      document.getElementById(`${id}-title`).innerHTML = t[id].title;
      document.getElementById(`${id}-lead`).innerHTML = t[id].lead;
    });

    document.getElementById("gvt-tbody").innerHTML = t.c1.gvtRows
      .map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`)
      .join("");

    const GH_ADAPTER_BASE = "https://github.com/ZaychikLiu/tpu-rtx/blob/main/tpu_mjx_render/adapters/";
    document.getElementById("adapter-grid").innerHTML = t.c2.adapters
      .map(
        ([name, file, desc, stat]) => `
        <div class="adapter-card">
          <div class="adapter-top">
            <div class="adapter-name">${name}</div>
            <a class="adapter-mod" href="${GH_ADAPTER_BASE}${file}" target="_blank" rel="noopener">${file} ↗</a>
          </div>
          <p class="adapter-desc">${desc}</p>
          <div class="adapter-stat">${stat}</div>
        </div>`
      )
      .join("");

    document.getElementById("bench-support-tbody").innerHTML = t.c2.benchSupportRows
      .map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`)
      .join("");

    document.getElementById("pod-list").innerHTML = t.podStats
      .map(
        ([k, v, d]) => `
        <div class="pod-item">
          <div class="pod-k">${k}</div>
          <div class="pod-v">${v}</div>
          <div class="pod-d">${d}</div>
        </div>`
      )
      .join("");

    document.getElementById("c6-honest-body").innerHTML = t.c6.honest;
    document.getElementById("comp-tbody").innerHTML = t.compRows
      .map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`)
      .join("");

    document.getElementById("c6-author-bio").textContent = t.c6.authorBio;
    document.getElementById("c6-ack").textContent = t.c6.ack;
    document.getElementById("footer-left").textContent = t.ui.footerL;

    if (window.TPU_RTX_CHARTS) {
      window.TPU_RTX_CHARTS.renderAll(state.lang);
    }

    document.getElementById("ch3-video-head").innerHTML = t.ui.v3_head;
    document.getElementById("ch5-video-head").innerHTML = t.ui.v5_head;

    renderHeroStage(state.heroIdx);
    renderVideoStageCh3();
    renderVideoStageCh5();
    renderCompareLab("compare-blender", "blender");
    renderCompareLab("compare-optics", "optics");
    renderCompareLab("compare-neural", "neural");
    renderCh3Spreads();
    renderSingleStage("stage-3dgs", DATA.stage3dgs, state.stage3dgsIdx, "s3d");
    renderSingleStage("stage-menagerie", DATA.stageMenagerie, state.stageMenagerieIdx, "mng");
    renderPostMortems();
    renderBench(state.benchKey);
    renderLedger();
    renderPlateDirectory();
    updateThemeButton();
    updateLangButton();
  }

  // Darkroom Lightbox + 1:1 Pixel Loupe
  const lb = document.getElementById("lightbox");
  const lbImg = document.getElementById("lb-img");
  const lbStage = document.getElementById("lb-stage");
  const lbCap = document.getElementById("lb-cap");
  const lbCounter = document.getElementById("lb-counter");
  const lbRaw = document.getElementById("lb-raw");
  const lbLoupe = document.getElementById("lb-loupe");
  const lbLoupeBtn = document.getElementById("lb-loupe-btn");

  function openLightboxById(id) {
    const idx = DATA.plates.findIndex((p) => p.id === id);
    if (idx < 0) return;
    showLightboxAt(idx);
    lb.classList.add("open");
    document.body.style.overflow = "hidden";
  }

  function showLightboxAt(idx) {
    state.lbIdx = (idx + DATA.plates.length) % DATA.plates.length;
    const plate = DATA.plates[state.lbIdx];
    const lang = state.lang;
    lbImg.src = plate.png;
    lbImg.alt = plate.title[lang];
    lbRaw.href = plate.png;
    lbCap.textContent = `${plate.title[lang]} — ${plate.desc[lang]}`;
    lbCounter.textContent = `${String(state.lbIdx + 1).padStart(2, "0")} / ${DATA.plates.length}`;
    if (state.loupeOn) {
      lbLoupe.style.backgroundImage = `url("${plate.png}")`;
    }
  }

  function closeLightbox() {
    lb.classList.remove("open");
    document.body.style.overflow = "";
    setLoupe(false);
  }

  function setLoupe(on) {
    state.loupeOn = on;
    lbLoupeBtn.classList.toggle("active", on);
    lbStage.classList.toggle("loupe-on", on);
    if (!on) {
      lbLoupe.style.display = "none";
    } else if (lbImg.src) {
      lbLoupe.style.backgroundImage = `url("${lbImg.src}")`;
    }
  }

  lbStage.addEventListener("mousemove", (e) => {
    if (!state.loupeOn || !lbImg.naturalWidth) return;
    const rect = lbImg.getBoundingClientRect();
    if (
      e.clientX < rect.left ||
      e.clientX > rect.right ||
      e.clientY < rect.top ||
      e.clientY > rect.bottom
    ) {
      lbLoupe.style.display = "none";
      return;
    }
    const rx = (e.clientX - rect.left) / rect.width;
    const ry = (e.clientY - rect.top) / rect.height;
    const lw = lbLoupe.offsetWidth || 220;
    const lh = lbLoupe.offsetHeight || 220;
    const nw = lbImg.naturalWidth;
    const nh = lbImg.naturalHeight;

    lbLoupe.style.display = "block";
    lbLoupe.style.left = `${e.clientX - lw / 2}px`;
    lbLoupe.style.top = `${e.clientY - lh / 2}px`;
    lbLoupe.style.backgroundSize = `${nw}px ${nh}px`;
    lbLoupe.style.backgroundPosition = `${-(rx * nw - lw / 2)}px ${-(ry * nh - lh / 2)}px`;
  });

  lbStage.addEventListener("mouseleave", () => {
    lbLoupe.style.display = "none";
  });

  // Event delegation & interactive controls
  document.getElementById("lang-toggle").addEventListener("click", () => {
    state.lang = state.lang === "en" ? "zh" : "en";
    try {
      localStorage.setItem("tpu_rtx_lang", state.lang);
      const u = new URL(window.location.href);
      u.searchParams.set("lang", state.lang);
      window.history.replaceState({}, "", u.toString());
    } catch (e) {}
    renderAllText();
  });

  document.getElementById("theme-toggle").addEventListener("click", () => {
    state.theme = state.theme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", state.theme);
    try {
      localStorage.setItem("tpu_rtx_theme", state.theme);
    } catch (e) {}
    updateThemeButton();
  });

  document.getElementById("hero-tabs").addEventListener("click", (e) => {
    const btn = e.target.closest(".cinema-tab");
    if (!btn) return;
    renderHeroStage(Number(btn.dataset.hero));
  });

  document.getElementById("ch3-video-cats").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-vcat]");
    if (!btn) return;
    state.videoCh3Cat = btn.dataset.vcat;
    state.videoCh3Idx = 0;
    renderVideoStageCh3();
  });

  document.getElementById("ch3-video-tabs").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-v3]");
    if (!btn) return;
    state.videoCh3Idx = Number(btn.dataset.v3);
    renderVideoStageCh3();
  });

  document.getElementById("ch5-video-tabs").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-v5]");
    if (!btn) return;
    state.videoCh5Idx = Number(btn.dataset.v5);
    renderVideoStageCh5();
  });

  document.getElementById("stage-3dgs-tabs").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-s3d]");
    if (!btn) return;
    state.stage3dgsIdx = Number(btn.dataset.s3d);
    renderSingleStage("stage-3dgs", DATA.stage3dgs, state.stage3dgsIdx, "s3d");
  });

  document.getElementById("stage-menagerie-tabs").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-mng]");
    if (!btn) return;
    state.stageMenagerieIdx = Number(btn.dataset.mng);
    renderSingleStage("stage-menagerie", DATA.stageMenagerie, state.stageMenagerieIdx, "mng");
  });

  document.getElementById("bench-tabs").addEventListener("click", (e) => {
    const btn = e.target.closest(".tab-btn");
    if (!btn) return;
    renderBench(btn.dataset.bench);
  });

  document.getElementById("ledger-tabs").addEventListener("click", (e) => {
    const btn = e.target.closest(".tab-btn");
    if (!btn) return;
    state.ledgerCat = btn.dataset.ld;
    document.querySelectorAll("#ledger-tabs .tab-btn").forEach((b) => {
      b.classList.toggle("active", b === btn);
    });
    renderLedger();
  });

  document.getElementById("ledger-search").addEventListener("input", (e) => {
    state.ledgerQuery = e.target.value;
    renderLedger();
  });

  document.getElementById("plate-filters").addEventListener("click", (e) => {
    const btn = e.target.closest(".tab-btn");
    if (!btn) return;
    applyDirectoryFilter(btn.dataset.filter);
  });

  document.addEventListener("click", (e) => {
    const cmpBtn = e.target.closest("[data-cmp-lab]");
    if (cmpBtn) {
      const labKey = cmpBtn.dataset.cmpLab;
      state.cmpIdx[labKey] = Number(cmpBtn.dataset.cmpIdx);
      renderCompareLab(`compare-${labKey}`, labKey);
      return;
    }

    const media = e.target.closest("[data-open-id]");
    if (media && media.dataset.openId) {
      openLightboxById(media.dataset.openId);
    }
  });

  document.addEventListener("input", (e) => {
    const rng = e.target.closest("[data-cmp-range]");
    if (!rng) return;
    const labKey = rng.dataset.cmpRange;
    const val = Number(rng.value);
    state.cmpPos[labKey] = val;
    const stage = rng.closest(".compare-stage");
    if (stage) stage.style.setProperty("--pos", `${val}%`);
    const pct = document.querySelector(`[data-cmp-pct="${labKey}"]`);
    if (pct) pct.textContent = `${val}% / ${100 - val}%`;
  });

  document.getElementById("lb-close").addEventListener("click", closeLightbox);
  document.getElementById("lb-prev").addEventListener("click", () => showLightboxAt(state.lbIdx - 1));
  document.getElementById("lb-next").addEventListener("click", () => showLightboxAt(state.lbIdx + 1));
  document.getElementById("lb-loupe-btn").addEventListener("click", () => setLoupe(!state.loupeOn));
  lb.addEventListener("click", (e) => {
    if (e.target === lb) closeLightbox();
  });

  document.addEventListener("keydown", (e) => {
    if (!lb.classList.contains("open")) return;
    if (e.key === "Escape") closeLightbox();
    else if (e.key === "ArrowLeft") showLightboxAt(state.lbIdx - 1);
    else if (e.key === "ArrowRight") showLightboxAt(state.lbIdx + 1);
    else if (e.key === "l" || e.key === "L") setLoupe(!state.loupeOn);
  });

  function wireCopy(btnId, codeId) {
    const btn = document.getElementById(btnId);
    btn.addEventListener("click", async () => {
      const text = document.getElementById(codeId).textContent;
      try {
        await navigator.clipboard.writeText(text);
        btn.textContent = getT().ui.copied;
        setTimeout(() => {
          btn.textContent = getT().ui.copy;
        }, 1600);
      } catch (err) {}
    });
  }
  wireCopy("copy-sdk-btn", "sdk-code");
  wireCopy("copy-bib-btn", "bib-code");

  const prog = document.getElementById("progress-fill");
  window.addEventListener(
    "scroll",
    () => {
      const h = document.documentElement;
      const denom = Math.max(1, h.scrollHeight - h.clientHeight);
      prog.style.width = `${Math.min(100, (h.scrollTop / denom) * 100)}%`;
    },
    { passive: true }
  );

  const navObs = new IntersectionObserver(
    (entries) => {
      entries.forEach((en) => {
        if (!en.isIntersecting) return;
        const id = en.target.id;
        document.querySelectorAll(".nav-links a").forEach((a) => {
          a.classList.toggle("active", a.dataset.nav === id);
        });
      });
    },
    { rootMargin: "-35% 0px -55% 0px" }
  );
  document.querySelectorAll("section.chapter").forEach((el) => navObs.observe(el));

  renderAllText();
})();

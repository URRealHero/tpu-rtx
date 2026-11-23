/* ==========================================================================
   TPU-RTX Interactive Engine
   Bilingual (EN / 中文) · Dual-Theme (Obsidian / Paper) · 1:1 Pixel Loupe
   Author: Feiyang Liu
   ========================================================================== */

(function () {
  const DATA = window.TPU_RTX_DATA;
  const platesById = Object.fromEntries(DATA.plates.map((p) => [p.id, p]));

  let state = {
    lang: document.documentElement.getAttribute("data-lang") === "zh" ? "zh" : "en",
    theme: document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark",
    heroIdx: 0,
    benchKey: "mjx",
    galleryFilter: "all",
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

  function renderPlateCard(plate, extraClass) {
    const lang = state.lang;
    const t = getT();
    const badges = plate.badges[lang]
      .map(([cls, txt]) => `<span class="badge ${cls}">${txt}</span>`)
      .join("");
    return `
      <article class="plate ${extraClass || ""}" data-cat="${plate.cat}" data-plate-id="${plate.id}">
        <div class="plate-media ${plate.layout === "wide" ? "ultra" : ""}" data-open-id="${plate.id}">
          <img src="${plate.web}" alt="${plate.title[lang]}" loading="lazy" />
          <span class="plate-zoom">${t.ui.zoom4k}</span>
        </div>
        <div class="plate-body">
          <div class="plate-badges">${badges}</div>
          <div class="plate-title">${plate.title[lang]}</div>
          <div class="plate-desc">${plate.desc[lang]}</div>
        </div>
      </article>
    `;
  }

  function renderChapterSpreads() {
    const renderSpread = (targetId, chapters) => {
      const el = document.getElementById(targetId);
      if (!el) return;
      const list = DATA.plates.filter((p) => p.chapter === chapters);
      const wide = list.filter((p) => p.layout === "wide");
      const rest = list.filter((p) => p.layout !== "wide");

      let html = "";
      if (wide.length) {
        html += `<div class="spread-wide">${wide.map((p) => renderPlateCard(p)).join("")}</div>`;
      }
      if (rest.length === 3) {
        html += `<div class="spread-three">${rest.map((p) => renderPlateCard(p)).join("")}</div>`;
      } else if (rest.length > 0) {
        html += `<div class="spread-two">${rest.map((p) => renderPlateCard(p)).join("")}</div>`;
      }
      el.innerHTML = html;
    };

    renderSpread("c2-plates", "ch-02");
    renderSpread("c3-plates", "ch-03");
    renderSpread("c4-plates", "ch-04");
    renderSpread("c5-plates", "ch-05");
    renderSpread("c6-plates", "ch-06");
  }

  function renderContactGrid() {
    const grid = document.getElementById("gallery-grid");
    grid.innerHTML = DATA.plates.map((p) => renderPlateCard(p)).join("");
    applyGalleryFilter(state.galleryFilter);
  }

  function applyGalleryFilter(filter) {
    state.galleryFilter = filter;
    document.querySelectorAll("#gallery-tabs .tab-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.filter === filter);
    });
    document.querySelectorAll("#gallery-grid .plate").forEach((c) => {
      c.classList.toggle("hidden", filter !== "all" && c.dataset.cat !== filter);
    });
  }

  function renderHeroStage(idx) {
    state.heroIdx = idx;
    const item = DATA.heroStage[idx];
    const plate = platesById[item.plateId];
    const lang = state.lang;
    const img = document.getElementById("hero-stage-img");
    img.src = plate.web;
    img.alt = plate.title[lang];
    document.getElementById("hero-stage-view").dataset.openId = plate.id;
    document.getElementById("hero-hud-left").innerHTML = item.hudL[lang];
    document.getElementById("hero-hud-right").innerHTML = item.hudR[lang];

    document.querySelectorAll("#hero-tabs .cinema-tab").forEach((b) => {
      b.classList.toggle("active", Number(b.dataset.hero) === idx);
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
          </div>
        `
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

    // Nav & Hero simple i18n keys
    const mapI18n = {
      "nav.c1": t.nav.c1, "nav.c2": t.nav.c2, "nav.c3": t.nav.c3, "nav.c4": t.nav.c4,
      "nav.c5": t.nav.c5, "nav.c6": t.nav.c6, "nav.c7": t.nav.c7, "nav.c8": t.nav.c8,
      "hero.kicker": t.hero.kicker, "hero.by": t.hero.by, "hero.license": t.hero.license,
      "hero.ctaGallery": t.hero.ctaGallery, "hero.ctaBench": t.hero.ctaBench, "hero.stamp": t.hero.stamp,
      "hero.t0i": t.hero.t0i, "hero.t0t": t.hero.t0t,
      "hero.t1i": t.hero.t1i, "hero.t1t": t.hero.t1t,
      "hero.t2i": t.hero.t2i, "hero.t2t": t.hero.t2t,
      "c1.num": t.c1.num, "c1.rail": t.c1.rail, "c1.sdkBar": t.c1.sdkBar,
      "c2.num": t.c2.num, "c2.rail": t.c2.rail,
      "c3.num": t.c3.num, "c3.rail": t.c3.rail,
      "c4.num": t.c4.num, "c4.rail": t.c4.rail,
      "c5.num": t.c5.num, "c5.rail": t.c5.rail,
      "c6.num": t.c6.num, "c6.rail": t.c6.rail,
      "c7.num": t.c7.num, "c7.rail": t.c7.rail,
      "c8.num": t.c8.num, "c8.rail": t.c8.rail,
      "c8.podBadge1": state.lang === "zh" ? "单机 4 芯已实测" : "4-Chip Single-Host Verified",
      "c8.podFigTitle": state.lang === "zh" ? "4 芯 Capstone 极限压测 (383 GB HBM, host_callbacks = 0)" : "4-Chip Capstone Suite (383 GB HBM, host_callbacks = 0)",
      "c8.podFigDesc": state.lang === "zh" ? "单机 4 芯全套 5 项极限基准真机实测结果；多机 32 芯脚本已修补就绪。" : "Measured end-to-end across all 5 ceiling benchmarks on 4 live chips. Multi-host 32-chip harness is patched cleanly for a future reservation window.",
      "c8.honestHead": state.lang === "zh" ? "透明度声明 · 视觉 QA 门禁淘汰与多机集群边界" : "TRANSPARENCY NOTE · VISUAL QA GATE & CLUSTER BOUNDARIES",
      "c8.authorRole": state.lang === "zh" ? "作者与系统架构" : "AUTHOR & ARCHITECT",
      "c8.bibBar": state.lang === "zh" ? "BibTeX 引用 · Apache-2.0 协议" : "BibTeX Citation · Apache-2.0",
      "bench.t_mjx": t.ui.b_mjx, "bench.t_gs3d": t.ui.b_gs3d, "bench.t_cinema": t.ui.b_cinema,
      "bench.t_dlss": t.ui.b_dlss, "bench.t_fx": t.ui.b_fx,
      "gal.all": t.ui.g_all, "gal.film": t.ui.g_film, "gal.optics": t.ui.g_optics,
      "gal.gs3d": t.ui.g_gs3d, "gal.dlss": t.ui.g_dlss, "gal.robotics": t.ui.g_robotics, "gal.pod": t.ui.g_pod,
      "tbl.h1": t.ui.th1, "tbl.h2": t.ui.th2, "tbl.h3": t.ui.th3, "tbl.h4": t.ui.th4,
      "common.copy": t.ui.copy, "common.zoom4k": t.ui.zoom4k,
      "lb.loupe": t.ui.loupe, "lb.raw": t.ui.raw
    };

    document.querySelectorAll("[data-i18n]").forEach((el) => {
      const k = el.getAttribute("data-i18n");
      if (mapI18n[k] !== undefined) el.textContent = mapI18n[k];
    });

    // Rich HTML blocks
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

    document.getElementById("c1-title").innerHTML = t.c1.title;
    document.getElementById("c1-lead").innerHTML = t.c1.lead;
    document.getElementById("c1-pillars").innerHTML = t.c1.pillars
      .map(
        ([tag, h3, p]) => `
        <div class="pillar-card">
          <div class="pillar-tag">${tag}</div>
          <h3>${h3}</h3>
          <p>${p}</p>
        </div>`
      )
      .join("");
    document.getElementById("c1-diag-cap").textContent = t.c1.diagCap;

    ["c2", "c3", "c4", "c5", "c6", "c7", "c8"].forEach((id) => {
      document.getElementById(`${id}-title`).innerHTML = t[id].title;
      document.getElementById(`${id}-lead`).innerHTML = t[id].lead;
    });

    document.getElementById("pod-stats").innerHTML = t.podStats
      .map(
        ([k, v, d]) => `
        <div class="pod-item">
          <div class="pod-k">${k}</div>
          <div class="pod-v">${v}</div>
          <div class="pod-d">${d}</div>
        </div>`
      )
      .join("");

    document.getElementById("c8-honest-body").innerHTML = t.c8.honest;
    document.getElementById("comp-tbody").innerHTML = t.compRows
      .map(
        (row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`
      )
      .join("");

    document.getElementById("c8-author-bio").textContent = t.c8.authorBio;
    document.getElementById("c8-ack").textContent = t.c8.ack;
    document.getElementById("footer-left").textContent = t.ui.footerL;

    renderHeroStage(state.heroIdx);
    renderChapterSpreads();
    renderBench(state.benchKey);
    renderContactGrid();
    updateThemeButton();
    updateLangButton();
  }

  // Lightbox + 1:1 Pixel Loupe
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

  // Event wiring
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

  document.getElementById("hero-stage-view").addEventListener("click", (e) => {
    const id = e.currentTarget.dataset.openId;
    if (id) openLightboxById(id);
  });

  document.getElementById("bench-tabs").addEventListener("click", (e) => {
    const btn = e.target.closest(".tab-btn");
    if (!btn) return;
    renderBench(btn.dataset.bench);
  });

  document.getElementById("gallery-tabs").addEventListener("click", (e) => {
    const btn = e.target.closest(".tab-btn");
    if (!btn) return;
    applyGalleryFilter(btn.dataset.filter);
  });

  document.getElementById("pod-fig-trigger").addEventListener("click", () => {
    openLightboxById("pod_dryrun");
  });

  document.addEventListener("click", (e) => {
    const media = e.target.closest("[data-open-id]");
    if (media && media.id !== "hero-stage-view") {
      openLightboxById(media.dataset.openId);
    }
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

  // Scroll progress + section observer
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

  const revealObs = new IntersectionObserver(
    (entries) => {
      entries.forEach((en) => {
        if (en.isIntersecting) en.target.classList.add("in");
      });
    },
    { threshold: 0.06 }
  );
  document.querySelectorAll(".reveal").forEach((el) => revealObs.observe(el));

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

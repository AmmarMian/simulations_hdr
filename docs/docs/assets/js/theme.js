(function () {
  "use strict";

  /* ── Apply saved theme before paint (also called inline in <head>) ── */
  var LIGHT = "warm";
  var DARK  = "dark";
  var KEY   = "hdr-paper";

  function getPaper() { return localStorage.getItem(KEY) || LIGHT; }
  function setPaper(p) {
    document.documentElement.setAttribute("data-paper", p);
    localStorage.setItem(KEY, p);
    syncThemeIcon(p);
  }
  function syncThemeIcon(p) {
    var moon = document.getElementById("ico-moon");
    var sun  = document.getElementById("ico-sun");
    if (moon) moon.style.display = p === DARK ? "none"  : "block";
    if (sun)  sun.style.display  = p === DARK ? "block" : "none";
  }

  setPaper(getPaper());

  document.addEventListener("DOMContentLoaded", function () {

    /* ── Search ────────────────────────────────────────────────── */
    (function () {
      var wrap    = document.getElementById("search-wrap");
      var btnS    = document.getElementById("btn-search");
      var inputW  = document.getElementById("search-input-wrap");
      var input   = document.getElementById("search-input");
      var results = document.getElementById("search-results");
      if (!btnS || !input) return;

      var index = null;

      function loadIndex(cb) {
        if (index) { cb(); return; }
        var base = (typeof HDR_BASE !== "undefined" ? HDR_BASE : "").replace(/\/$/, "");
        fetch(base + "/search/search_index.json")
          .then(function (r) { return r.json(); })
          .then(function (data) { index = data.docs || []; cb(); })
          .catch(function () { index = []; });
      }

      function openSearch() {
        wrap.classList.add("open");
        inputW.hidden = false;
        results.hidden = true;
        setTimeout(function () { input.focus(); }, 50);
        loadIndex(function () {});
      }
      function closeSearch() {
        wrap.classList.remove("open");
        inputW.hidden = true;
        results.hidden = true;
        input.value = "";
      }

      btnS.addEventListener("click", function (e) {
        e.stopPropagation();
        wrap.classList.contains("open") ? closeSearch() : openSearch();
      });
      document.addEventListener("click", function (e) {
        if (!wrap.contains(e.target)) closeSearch();
      });
      document.addEventListener("keydown", function (e) {
        if (e.key === "Escape") { closeSearch(); return; }
        if (e.key === "k" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); openSearch(); return; }
        if (!wrap.classList.contains("open")) return;
        if (e.key !== "ArrowDown" && e.key !== "ArrowUp" && e.key !== "Enter") return;

        var items = results.querySelectorAll(".search-result-item");
        if (!items.length) return;
        e.preventDefault();

        var active = results.querySelector(".search-result-item.active");
        var idx = active ? Array.prototype.indexOf.call(items, active) : -1;

        if (e.key === "Enter" && active) { active.click(); return; }

        if (active) active.classList.remove("active");
        if (e.key === "ArrowDown") idx = (idx + 1) % items.length;
        else if (e.key === "ArrowUp") idx = (idx - 1 + items.length) % items.length;
        items[idx].classList.add("active");
        items[idx].scrollIntoView({ block: "nearest" });
      });

      function query(q) {
        if (!q || !index) { results.hidden = true; return; }
        var terms = q.toLowerCase().split(/\s+/).filter(Boolean);
        var hits = index.filter(function (doc) {
          var hay = (doc.title + " " + doc.text).toLowerCase();
          return terms.every(function (t) { return hay.includes(t); });
        }).slice(0, 8);

        function esc(s) {
          return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
        }
        function highlight(text, terms) {
          var out = esc(text);
          terms.forEach(function (t) {
            var re = new RegExp("(" + t.replace(/[.*+?^${}()|[\]\\]/g,"\\$&") + ")", "gi");
            out = out.replace(re, "<strong>$1</strong>");
          });
          return out;
        }

        results.innerHTML = "";
        if (!hits.length) {
          var empty = document.createElement("div");
          empty.className = "search-empty";
          empty.textContent = "No results";
          results.appendChild(empty);
        } else {
          hits.forEach(function (doc) {
            var a = document.createElement("a");
            a.className = "search-result-item";
            var base = (typeof HDR_BASE !== "undefined" ? HDR_BASE : "").replace(/\/$/, "");
            a.href = base + "/" + doc.location;

            var title = document.createElement("div");
            title.className = "search-result-title";
            title.innerHTML = highlight(doc.title || doc.location, terms);
            a.appendChild(title);

            var loc = document.createElement("div");
            loc.className = "search-result-loc";
            loc.textContent = doc.location;
            a.appendChild(loc);

            if (doc.text) {
              var idx = doc.text.toLowerCase().indexOf(terms[0]);
              var snip = idx >= 0
                ? doc.text.slice(Math.max(0, idx - 40), idx + 100)
                : doc.text.slice(0, 120);
              var ex = document.createElement("div");
              ex.className = "search-result-excerpt";
              ex.innerHTML = "…" + highlight(snip.trim(), terms) + "…";
              a.appendChild(ex);
            }
            results.appendChild(a);
          });
        }
        results.hidden = false;
      }

      input.addEventListener("input", function () { loadIndex(function () { query(input.value.trim()); }); });
    })();

    /* ── Theme toggle ──────────────────────────────────────────── */
    var btn = document.getElementById("btn-theme");
    if (btn) btn.addEventListener("click", function () {
      setPaper(getPaper() === DARK ? LIGHT : DARK);
    });
    syncThemeIcon(getPaper());

    /* ── Reading-scale A / A ───────────────────────────────────── */
    var SCALES   = [.88, 1, 1.14];
    var SCALE_KEY = "hdr-scale";
    var scaleIdx  = 1;
    var saved = parseFloat(localStorage.getItem(SCALE_KEY));
    if (!isNaN(saved)) {
      var idx = SCALES.indexOf(saved);
      if (idx !== -1) { scaleIdx = idx; applyScale(SCALES[scaleIdx]); }
    }
    function applyScale(s) {
      document.documentElement.style.setProperty("--reading-scale", s);
    }
    function cycleScale(dir) {
      scaleIdx = Math.max(0, Math.min(SCALES.length - 1, scaleIdx + dir));
      var s = SCALES[scaleIdx];
      applyScale(s);
      localStorage.setItem(SCALE_KEY, s);
    }
    var btnSm = document.getElementById("btn-az-sm");
    var btnLg = document.getElementById("btn-az-lg");
    if (btnSm) btnSm.addEventListener("click", function () { cycleScale(-1); });
    if (btnLg) btnLg.addEventListener("click", function () { cycleScale(+1); });

    /* ── Code block window injection ───────────────────────────── */
    /*   Two output shapes depending on linenums:
         - No linenums:  <div class="language-python highlight">…</div>
         - With linenums: <table class="highlighttable">…</table>
                           language class is on the inner .highlight div
         We inject a .cbhead in both.                                  */

    function readLangFile(container) {
      var lang = "";
      container.classList.forEach(function (c) {
        if (c.startsWith("language-")) lang = c.slice(9);
      });
      /* For highlighttable, pymdownx puts language-* on the outer wrapper div (parent),
         and the .filename span is also a child of that wrapper, not of the table. */
      var fileRoot = container;
      if (!lang && container.parentElement) {
        container.parentElement.classList.forEach(function (c) {
          if (c.startsWith("language-")) lang = c.slice(9);
        });
        fileRoot = container.parentElement;
      }
      var file = "";
      var fileEl = fileRoot.querySelector(".filename");
      if (fileEl) { file = fileEl.textContent.trim(); fileEl.style.display = "none"; }
      return { lang: lang, file: file };
    }

    function makeCbhead(lang, file) {
      var head = document.createElement("div");
      head.className = "cbhead";
      if (lang) {
        var ls = document.createElement("span");
        ls.className = "cblang";
        ls.textContent = lang.toUpperCase();
        head.appendChild(ls);
      }
      if (file) {
        var fs = document.createElement("span");
        fs.className = "cbfile";
        fs.textContent = file;
        head.appendChild(fs);
      }
      return head;
    }

    /* plain (no linenums) blocks — skip API doc elements and outer wrappers around highlighttable */
    document.querySelectorAll(".highlight:not(.highlighttable .highlight):not(.doc .highlight)").forEach(function (block) {
      if (block.querySelector(".cbhead")) return;
      if (block.querySelector("table.highlighttable")) return;  /* handled by thead loop below */
      var lf = readLangFile(block);
      if (!lf.lang && !lf.file) return;
      block.insertBefore(makeCbhead(lf.lang, lf.file), block.firstChild);
    });

    /* line-numbered blocks — inject a <thead> with two cells matching column structure */
    document.querySelectorAll("table.highlighttable:not(.doc table.highlighttable)").forEach(function (table) {
      if (table.querySelector(".cbhead")) return;
      var lf = readLangFile(table);
      if (!lf.lang && !lf.file) return;
      var thead  = document.createElement("thead");
      var tr     = document.createElement("tr");
      /* linenos header cell — same panel background as the column below */
      var thLn   = document.createElement("th");
      thLn.className = "cbhead-linenos";
      /* code header cell — holds the language / filename label */
      var thCode = document.createElement("th");
      thCode.className = "cbhead";
      if (lf.lang) {
        var ls = document.createElement("span");
        ls.className = "cblang";
        ls.textContent = lf.lang.toUpperCase();
        thCode.appendChild(ls);
      }
      if (lf.file) {
        var fs = document.createElement("span");
        fs.className = "cbfile";
        fs.textContent = lf.file;
        thCode.appendChild(fs);
      }
      tr.appendChild(thLn);
      tr.appendChild(thCode);
      thead.appendChild(tr);
      table.insertBefore(thead, table.firstChild);
    });

    /* ── TOC drawer ─────────────────────────────────────────────── */
    var toc   = document.getElementById("toc");
    var scrim = document.getElementById("toc-scrim");
    var btnToc = document.getElementById("btn-toc");

    function openToc()  {
      if (!toc) return;
      toc.classList.add("open");
      if (scrim) scrim.classList.add("open");
      document.body.style.overflow = "hidden";
    }
    function closeToc() {
      if (!toc) return;
      toc.classList.remove("open");
      if (scrim) scrim.classList.remove("open");
      document.body.style.overflow = "";
    }
    if (btnToc) btnToc.addEventListener("click", openToc);
    if (scrim)  scrim.addEventListener("click", closeToc);
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") closeToc();
    });

    /* ── Progress bar ───────────────────────────────────────────── */
    var bar = document.getElementById("progress");
    if (bar) {
      function updateProgress() {
        var el  = document.documentElement;
        var top = el.scrollTop || document.body.scrollTop;
        var h   = el.scrollHeight - el.clientHeight;
        bar.style.width = (h > 0 ? (top / h) * 100 : 0) + "%";
      }
      window.addEventListener("scroll", updateProgress, { passive: true });
      updateProgress();
    }

    /* ── Rebuild aside TOC from DOM on API pages ────────────────── */
    /* page.toc is extracted from markdown before mkdocstrings injects its headings,
       so API pages only show the h1. If .doc elements exist, rebuild from the DOM. */
    (function () {
      var aside = document.querySelector(".aside-toc");
      if (!aside) return;
      if (!document.querySelector(".page-content .doc")) return;  /* not an API page */

      var domHeadings = Array.from(
        document.querySelectorAll(".page-content h2[id], .page-content h3[id], .page-content h4[id]")
      ).filter(function (h) {
        /* skip mkdocstrings parameter/raises headings (h5) and the markdown h1 */
        return h.id && !h.id.includes("--");
      });
      if (domHeadings.length < 2) return;  /* nothing interesting to add */

      var ul = document.createElement("ul");
      domHeadings.forEach(function (h) {
        var li = document.createElement("li");
        var tag = h.tagName;
        li.className = tag === "H4" ? "h3" : (tag === "H3" ? "h2" : "h1");
        var a = document.createElement("a");
        a.href = "#" + h.id;
        /* strip badges/labels text — keep only the code element text */
        var codeEl = h.querySelector("code");
        a.textContent = codeEl ? codeEl.textContent.trim()
                                : h.textContent.replace(/[¶#]/g, "").trim();
        li.appendChild(a);
        ul.appendChild(li);
      });

      /* replace existing ul */
      var existingUl = aside.querySelector("ul");
      if (existingUl) existingUl.replaceWith(ul);
      else aside.appendChild(ul);
    })();

    /* ── Active aside-TOC link on scroll ────────────────────────── */
    var asideLinks = Array.from(document.querySelectorAll(".aside-toc a[href^='#']"));
    if (asideLinks.length) {
      var headings = asideLinks.map(function (a) {
        return document.querySelector(a.getAttribute("href"));
      }).filter(Boolean);

      function markActive() {
        var top = window.scrollY + 80;
        var cur = headings[0];
        headings.forEach(function (h) { if (h.offsetTop <= top) cur = h; });
        asideLinks.forEach(function (a) {
          a.classList.toggle("active", cur && cur.id === a.getAttribute("href").slice(1));
        });
      }
      window.addEventListener("scroll", markActive, { passive: true });
      markActive();
    }

    /* ── Mark active TOC-drawer link ────────────────────────────── */
    var cur = location.pathname;
    document.querySelectorAll(".toc nav a").forEach(function (a) {
      if (a.pathname === cur) a.classList.add("active");
    });

    /* ── Floating page-TOC FAB ──────────────────────────────────── */
    var headings = Array.from(
      document.querySelectorAll(".page-content h2, .page-content h3")
    ).filter(function (h) { return h.id; });

    if (headings.length) {
      /* build FAB button */
      var fab = document.createElement("button");
      fab.className = "toc-fab";
      fab.setAttribute("aria-label", "Page contents");
      var fabSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      fabSvg.setAttribute("viewBox", "0 0 24 24");
      fabSvg.setAttribute("fill", "none");
      fabSvg.setAttribute("stroke", "currentColor");
      fabSvg.setAttribute("stroke-width", "1.8");
      fabSvg.setAttribute("stroke-linecap", "round");
      ["M3 6h18", "M3 12h12", "M3 18h8"].forEach(function (d) {
        var p = document.createElementNS("http://www.w3.org/2000/svg", "path");
        p.setAttribute("d", d);
        fabSvg.appendChild(p);
      });
      fab.appendChild(fabSvg);

      /* build panel */
      var panel = document.createElement("div");
      panel.className = "toc-fab-panel";

      var label = document.createElement("div");
      label.className = "toc-fab-panel-label";
      label.textContent = "On this page";
      panel.appendChild(label);

      var ul = document.createElement("ul");
      headings.forEach(function (h) {
        var li = document.createElement("li");
        li.className = h.tagName === "H3" ? "h3" : "h2";
        var a = document.createElement("a");
        a.href = "#" + h.id;
        a.textContent = h.textContent.replace(/[¶#]/g, "").trim();
        a.addEventListener("click", function () { closePanel(); });
        li.appendChild(a);
        ul.appendChild(li);
      });
      panel.appendChild(ul);

      document.body.appendChild(panel);
      document.body.appendChild(fab);

      var panelOpen = false;
      function openPanel()  { panelOpen = true;  panel.classList.add("open"); }
      function closePanel() { panelOpen = false; panel.classList.remove("open"); }

      fab.addEventListener("click", function (e) {
        e.stopPropagation();
        panelOpen ? closePanel() : openPanel();
      });
      document.addEventListener("click", function (e) {
        if (panelOpen && !panel.contains(e.target)) closePanel();
      });
      document.addEventListener("keydown", function (e) {
        if (e.key === "Escape" && panelOpen) closePanel();
      });

      /* active link tracking */
      var fabLinks = Array.from(ul.querySelectorAll("a"));
      function markFabActive() {
        var top = window.scrollY + 90;
        var active = headings[0];
        headings.forEach(function (h) { if (h.offsetTop <= top) active = h; });
        fabLinks.forEach(function (a) {
          a.classList.toggle("active", active && a.getAttribute("href") === "#" + active.id);
        });
      }
      window.addEventListener("scroll", markFabActive, { passive: true });
      markFabActive();
    }

  });
})();

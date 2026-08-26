/* ============================================================
   HDR Simulations — theme behaviour
   Light/dark toggle, search, navigation drawer, code-block
   chrome, table wrapping and TOC scroll-spy.
   ============================================================ */
(function () {
  "use strict";

  var THEME_KEY = "hdr-theme";
  var root = document.documentElement;

  function currentTheme() { return root.dataset.theme === "dark" ? "dark" : "light"; }
  function setTheme(t) {
    root.dataset.theme = t;
    try { localStorage.setItem(THEME_KEY, t); } catch (e) { /* private mode */ }
  }

  function base() {
    return (typeof HDR_BASE !== "undefined" ? HDR_BASE : "").replace(/\/$/, "");
  }
  function esc(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  document.addEventListener("DOMContentLoaded", function () {

    /* ── Theme toggle ────────────────────────────────────── */
    var btnTheme = document.getElementById("btn-theme");
    if (btnTheme) {
      btnTheme.addEventListener("click", function () {
        setTheme(currentTheme() === "dark" ? "light" : "dark");
      });
    }
    /* Follow the OS only while the visitor has not chosen. */
    try {
      if (!localStorage.getItem(THEME_KEY)) {
        matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function (e) {
          root.dataset.theme = e.matches ? "dark" : "light";
        });
      }
    } catch (e) { /* ignore */ }

    /* ── Navigation drawer ───────────────────────────────── */
    (function () {
      var sidebar = document.getElementById("sidebar");
      var scrim   = document.getElementById("nav-scrim");
      var btnNav  = document.getElementById("btn-nav");
      if (!sidebar || !btnNav) return;

      function open() {
        sidebar.classList.add("open");
        if (scrim) scrim.classList.add("open");
        btnNav.setAttribute("aria-expanded", "true");
        document.body.style.overflow = "hidden";
      }
      function close() {
        sidebar.classList.remove("open");
        if (scrim) scrim.classList.remove("open");
        btnNav.setAttribute("aria-expanded", "false");
        document.body.style.overflow = "";
      }
      btnNav.addEventListener("click", function () {
        sidebar.classList.contains("open") ? close() : open();
      });
      if (scrim) scrim.addEventListener("click", close);
      document.addEventListener("keydown", function (e) {
        if (e.key === "Escape") close();
      });
      /* Following a link inside the drawer should close it. */
      sidebar.addEventListener("click", function (e) {
        if (e.target.closest("a")) close();
      });
      /* Leaving the drawer breakpoint must not strand the scroll lock. */
      matchMedia("(min-width: 1060px)").addEventListener("change", function (e) {
        if (e.matches) close();
      });

      /* Keep the active entry in view in the docked sidebar. */
      var active = sidebar.querySelector("a.active");
      if (active) {
        var r = active.getBoundingClientRect();
        if (r.top < 0 || r.bottom > window.innerHeight) {
          active.scrollIntoView({ block: "center" });
        }
      }
    })();

    /* ── Search ──────────────────────────────────────────── */
    (function () {
      var btn     = document.getElementById("btn-search");
      var panel   = document.getElementById("search-panel");
      var input   = document.getElementById("search-input");
      var results = document.getElementById("search-results");
      var close   = document.getElementById("search-close");
      if (!btn || !panel || !input) return;

      var index = null;

      function loadIndex(cb) {
        if (index) { cb(); return; }
        fetch(base() + "/search/search_index.json")
          .then(function (r) { return r.json(); })
          .then(function (d) { index = d.docs || []; cb(); })
          .catch(function () { index = []; cb(); });
      }

      function openSearch() {
        panel.hidden = false;
        results.innerHTML = "";
        loadIndex(function () {});
        setTimeout(function () { input.focus(); }, 40);
      }
      function closeSearch() {
        panel.hidden = true;
        input.value = "";
        results.innerHTML = "";
      }

      btn.addEventListener("click", function (e) {
        e.stopPropagation();
        panel.hidden ? openSearch() : closeSearch();
      });
      if (close) close.addEventListener("click", closeSearch);
      document.addEventListener("click", function (e) {
        if (panel.hidden) return;
        if (!panel.contains(e.target) && !btn.contains(e.target)) closeSearch();
      });

      document.addEventListener("keydown", function (e) {
        if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
          e.preventDefault();
          panel.hidden ? openSearch() : input.focus();
          return;
        }
        if (panel.hidden) return;
        if (e.key === "Escape") { closeSearch(); btn.focus(); return; }
        if (["ArrowDown", "ArrowUp", "Enter"].indexOf(e.key) === -1) return;

        var items = results.querySelectorAll(".search-result-item");
        if (!items.length) return;
        e.preventDefault();

        var active = results.querySelector(".search-result-item.active");
        var idx = active ? Array.prototype.indexOf.call(items, active) : -1;
        if (e.key === "Enter") {
          (active || items[0]).click();
          return;
        }
        if (active) active.classList.remove("active");
        idx = e.key === "ArrowDown"
          ? (idx + 1) % items.length
          : (idx - 1 + items.length) % items.length;
        items[idx].classList.add("active");
        items[idx].scrollIntoView({ block: "nearest" });
      });

      function mark(text, terms) {
        var out = esc(text);
        terms.forEach(function (t) {
          var re = new RegExp("(" + t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "gi");
          out = out.replace(re, "<strong>$1</strong>");
        });
        return out;
      }

      function query(q) {
        results.innerHTML = "";
        if (!q || !index) return;
        var terms = q.toLowerCase().split(/\s+/).filter(Boolean);
        var hits = index.filter(function (doc) {
          var hay = ((doc.title || "") + " " + (doc.text || "")).toLowerCase();
          return terms.every(function (t) { return hay.indexOf(t) !== -1; });
        }).slice(0, 10);

        if (!hits.length) {
          var empty = document.createElement("div");
          empty.className = "search-empty";
          empty.textContent = "No results for “" + q + "”";
          results.appendChild(empty);
          return;
        }

        hits.forEach(function (doc) {
          var a = document.createElement("a");
          a.className = "search-result-item";
          a.href = base() + "/" + doc.location;

          var title = document.createElement("div");
          title.className = "search-result-title";
          title.innerHTML = mark(doc.title || doc.location, terms);
          a.appendChild(title);

          var loc = document.createElement("div");
          loc.className = "search-result-loc";
          loc.textContent = doc.location;
          a.appendChild(loc);

          if (doc.text) {
            var i = doc.text.toLowerCase().indexOf(terms[0]);
            var snip = i >= 0
              ? doc.text.slice(Math.max(0, i - 40), i + 110)
              : doc.text.slice(0, 130);
            var ex = document.createElement("div");
            ex.className = "search-result-excerpt";
            ex.innerHTML = "…" + mark(snip.trim(), terms) + "…";
            a.appendChild(ex);
          }
          results.appendChild(a);
        });
      }

      var debounce;
      input.addEventListener("input", function () {
        clearTimeout(debounce);
        debounce = setTimeout(function () {
          loadIndex(function () { query(input.value.trim()); });
        }, 90);
      });
    })();

    /* ── Code-block chrome ───────────────────────────────── */
    (function () {
      function readLangFile(container) {
        var lang = "";
        container.classList.forEach(function (c) {
          if (c.indexOf("language-") === 0) lang = c.slice(9);
        });
        var fileRoot = container;
        if (!lang && container.parentElement) {
          container.parentElement.classList.forEach(function (c) {
            if (c.indexOf("language-") === 0) lang = c.slice(9);
          });
          fileRoot = container.parentElement;
        }
        var file = "";
        var fileEl = fileRoot.querySelector(".filename");
        if (fileEl) { file = fileEl.textContent.trim(); fileEl.style.display = "none"; }
        return { lang: lang, file: file };
      }

      function copyBtn(getCode) {
        var b = document.createElement("button");
        b.className = "cb-copy";
        b.setAttribute("aria-label", "Copy code");
        b.innerHTML =
          '<svg class="cb-copy-icon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
            '<rect x="9" y="2" width="6" height="4" rx="1"></rect>' +
            '<path d="M17 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path>' +
          "</svg>" +
          '<svg class="cb-check-icon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">' +
            '<polyline points="20 6 9 17 4 12"></polyline>' +
          "</svg>";
        b.addEventListener("click", function () {
          navigator.clipboard.writeText(getCode()).then(function () {
            b.classList.add("copied");
            setTimeout(function () { b.classList.remove("copied"); }, 1600);
          });
        });
        return b;
      }

      function labelInto(el, lang, file) {
        if (lang) {
          var ls = document.createElement("span");
          ls.className = "cblang";
          ls.textContent = lang.toUpperCase();
          el.appendChild(ls);
        }
        if (file) {
          var fs = document.createElement("span");
          fs.className = "cbfile";
          fs.textContent = file;
          el.appendChild(fs);
        }
      }

      document.querySelectorAll(".highlight:not(.highlighttable .highlight):not(.doc .highlight)").forEach(function (block) {
        if (block.querySelector(".cbhead")) return;
        if (block.querySelector("table.highlighttable")) return;  /* handled below */
        var lf = readLangFile(block);
        if (!lf.lang && !lf.file) return;
        var head = document.createElement("div");
        head.className = "cbhead";
        labelInto(head, lf.lang, lf.file);
        head.appendChild(copyBtn(function () {
          return (block.querySelector("pre") || block).innerText;
        }));
        block.insertBefore(head, block.firstChild);
      });

      /* Line-numbered blocks: the header goes on the wrapping .highlight
         div, not inside the table, so it does not scroll with the code. */
      document.querySelectorAll("table.highlighttable:not(.doc table.highlighttable)").forEach(function (table) {
        var wrap = table.parentElement;
        if (!wrap || wrap.querySelector(".cbhead")) return;
        var lf = readLangFile(table);
        if (!lf.lang && !lf.file) return;
        var head = document.createElement("div");
        head.className = "cbhead";
        labelInto(head, lf.lang, lf.file);
        head.appendChild(copyBtn(function () {
          return (table.querySelector("td.code pre") || table).innerText;
        }));
        wrap.insertBefore(head, table);
      });
    })();

    /* ── Wide tables scroll inside their own box ─────────── */
    document.querySelectorAll(
      ".page-content table:not(.highlighttable):not(.table-scroll table):not(.doc table)"
    ).forEach(function (table) {
      var wrap = document.createElement("div");
      wrap.className = "table-scroll";
      table.parentNode.insertBefore(wrap, table);
      wrap.appendChild(table);
    });

    /* ── Reading progress ────────────────────────────────── */
    var bar = document.getElementById("progress");
    if (bar) {
      var tick = function () {
        var el = document.documentElement;
        var h = el.scrollHeight - el.clientHeight;
        bar.style.width = (h > 0 ? (el.scrollTop / h) * 100 : 0) + "%";
      };
      window.addEventListener("scroll", tick, { passive: true });
      tick();
    }

    /* ── Rebuild the rail TOC on API pages ───────────────── */
    /* page.toc is extracted before mkdocstrings injects its headings,
       so API pages otherwise show only the h1. */
    (function () {
      var aside = document.querySelector(".aside-toc");
      if (!aside || !document.querySelector(".page-content .doc")) return;

      var heads = Array.prototype.filter.call(
        document.querySelectorAll(".page-content h2[id], .page-content h3[id], .page-content h4[id]"),
        function (h) { return h.id && h.id.indexOf("--") === -1; }
      );
      if (heads.length < 2) return;

      var ul = document.createElement("ul");
      heads.forEach(function (h) {
        var li = document.createElement("li");
        li.className = h.tagName === "H4" ? "h3" : (h.tagName === "H3" ? "h2" : "h1");
        var a = document.createElement("a");
        a.href = "#" + h.id;
        var codeEl = h.querySelector("code");
        a.textContent = codeEl ? codeEl.textContent.trim()
                               : h.textContent.replace(/[¶#]/g, "").trim();
        li.appendChild(a);
        ul.appendChild(li);
      });
      var existing = aside.querySelector("ul");
      existing ? existing.replaceWith(ul) : aside.appendChild(ul);
    })();

    /* ── Heading anchors ─────────────────────────────────── */
    document.querySelectorAll(
      ".page-content h2[id], .page-content h3[id], .page-content h4[id]"
    ).forEach(function (h) {
      if (h.querySelector(".heading-anchor")) return;
      var a = document.createElement("a");
      a.className = "heading-anchor";
      a.href = "#" + h.id;
      a.setAttribute("aria-label", "Link to this section");
      a.innerHTML =
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">' +
        '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>' +
        '<path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>' +
        "</svg>";
      h.appendChild(a);
    });

    /* ── Floating page-TOC (tablet widths) ───────────────── */
    var pageHeads = Array.prototype.filter.call(
      document.querySelectorAll(".page-content h2, .page-content h3"),
      function (h) { return h.id; }
    );

    if (pageHeads.length > 1) {
      var fab = document.createElement("button");
      fab.className = "toc-fab";
      fab.setAttribute("aria-label", "Page contents");
      fab.innerHTML =
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">' +
        '<path d="M3 6h18"/><path d="M3 12h12"/><path d="M3 18h8"/></svg>';

      var panel = document.createElement("div");
      panel.className = "toc-fab-panel";
      var label = document.createElement("p");
      label.className = "toc-fab-panel-label";
      label.textContent = "On this page";
      panel.appendChild(label);

      var list = document.createElement("ul");
      pageHeads.forEach(function (h) {
        var li = document.createElement("li");
        li.className = h.tagName === "H3" ? "h3" : "h2";
        var a = document.createElement("a");
        a.href = "#" + h.id;
        a.textContent = h.textContent.replace(/[¶#]/g, "").trim();
        a.addEventListener("click", function () { panel.classList.remove("open"); });
        li.appendChild(a);
        list.appendChild(li);
      });
      panel.appendChild(list);
      document.body.appendChild(panel);
      document.body.appendChild(fab);

      fab.addEventListener("click", function (e) {
        e.stopPropagation();
        panel.classList.toggle("open");
      });
      document.addEventListener("click", function (e) {
        if (!panel.contains(e.target)) panel.classList.remove("open");
      });
      document.addEventListener("keydown", function (e) {
        if (e.key === "Escape") panel.classList.remove("open");
      });
    }

    /* ── Scroll-spy for both TOCs ────────────────────────── */
    (function () {
      var links = Array.prototype.slice.call(
        document.querySelectorAll(".aside-toc a[href^='#'], .toc-fab-panel a[href^='#']")
      );
      if (!links.length) return;

      var targets = links.map(function (a) {
        try { return document.querySelector(a.getAttribute("href")); }
        catch (e) { return null; }
      });

      function spy() {
        var top = window.scrollY + 100;
        var currentId = null;
        targets.forEach(function (t) {
          if (t && t.offsetTop <= top) currentId = t.id;
        });
        links.forEach(function (a, i) {
          var t = targets[i];
          a.classList.toggle("active", !!t && t.id === currentId);
        });
      }
      window.addEventListener("scroll", spy, { passive: true });
      spy();
    })();

    /* ── Experiments overview: hover/tap preview panels ──── */
    (function () {
      var toc = document.getElementById("xp-toc");
      if (!toc) return;
      var items  = Array.prototype.slice.call(toc.querySelectorAll(".xp-toc-item"));
      var panels = Array.prototype.slice.call(toc.querySelectorAll(".xp-toc-panel"));
      if (!items.length) return;

      function show(id) {
        items.forEach(function (a) { a.classList.toggle("is-active", a.dataset.panel === id); });
        panels.forEach(function (p) { p.hidden = p.id !== id; });
      }
      var touch = matchMedia("(hover: none)").matches;
      items.forEach(function (a) {
        a.addEventListener("mouseenter", function () { show(a.dataset.panel); });
        a.addEventListener("focus", function () { show(a.dataset.panel); });
        /* On touch, the first tap previews the chapter, the second opens it. */
        if (touch) {
          a.addEventListener("click", function (e) {
            if (!a.classList.contains("is-active")) {
              e.preventDefault();
              show(a.dataset.panel);
              var panel = document.getElementById(a.dataset.panel);
              if (panel) panel.scrollIntoView({ block: "nearest", behavior: "smooth" });
            }
          });
        }
      });
    })();

  });
})();

/* ============================================================
   Plotly figure loader
   ------------------------------------------------------------
   Exported specs carry a hard `layout.width` (760–950 px), which
   defeats Plotly's own `responsive` option and makes figures
   overflow to the right on narrow screens. We strip the fixed
   width, keep the authored aspect ratio, and recompute the
   height from the container on load, on resize, and whenever the
   colour theme changes.
   ============================================================ */
(function () {
  "use strict";

  var MIN_H = 200;
  var MAX_H = 720;

  function themeColors() {
    var s = getComputedStyle(document.documentElement);
    function v(name, fallback) {
      return (s.getPropertyValue(name) || "").trim() || fallback;
    }
    return {
      grid:  v("--line", "#e4ded0"),
      line:  v("--line2", "#d8d1c1"),
      ink:   v("--ink", "#26241f"),
      body:  v("--body", "#3d382f"),
      muted: v("--muted", "#6f6a5e"),
      paper: v("--card", "#fbf9f4")
    };
  }

  /* Recolour a spec in place to match the current theme. */
  function applyTheme(spec, c) {
    var L = spec.layout;
    L.paper_bgcolor = "rgba(0,0,0,0)";
    L.plot_bgcolor  = "rgba(0,0,0,0)";
    L.font = L.font || {};
    L.font.color = c.body;
    L.font.family = L.font.family ||
      "'IBM Plex Sans', system-ui, -apple-system, sans-serif";

    Object.keys(L).forEach(function (key) {
      if (!/^[xy]axis/.test(key)) return;
      var ax = L[key];
      if (!ax || typeof ax !== "object") return;
      if (ax.showgrid !== false) ax.gridcolor = c.grid;
      ax.zerolinecolor = c.line;
      ax.linecolor = c.line;
      ax.tickcolor = c.line;
      ax.tickfont = ax.tickfont || {};
      ax.tickfont.color = c.muted;
      if (ax.title) {
        if (typeof ax.title === "string") ax.title = { text: ax.title };
        ax.title.font = ax.title.font || {};
        ax.title.font.color = c.body;
      }
    });

    if (L.legend) {
      L.legend.font = L.legend.font || {};
      L.legend.font.color = c.body;
      L.legend.bgcolor = "rgba(0,0,0,0)";
      L.legend.bordercolor = c.line;
    }
    if (L.title) {
      if (typeof L.title === "string") L.title = { text: L.title };
      L.title.font = L.title.font || {};
      L.title.font.color = c.ink;
    }
    if (L.coloraxis && L.coloraxis.colorbar) {
      L.coloraxis.colorbar.tickfont = { color: c.muted };
      L.coloraxis.colorbar.outlinecolor = c.line;
    }
    return spec;
  }

  /* Scale every font in the spec. Figures authored at 800–950 px keep
     their absolute font sizes when squeezed onto a phone, which is what
     makes subplot titles collide with tick labels. */
  function scaleFonts(spec, k) {
    var L = spec.layout;
    function fs(obj, base) {
      if (!obj) return;
      obj.size = Math.max(7, Math.round((obj.size || base) * k));
    }
    L.font = L.font || {};
    fs(L.font, 12);
    if (L.legend) { L.legend.font = L.legend.font || {}; fs(L.legend.font, 12); }
    if (L.title && typeof L.title === "object") {
      L.title.font = L.title.font || {};
      fs(L.title.font, 16);
    }
    (L.annotations || []).forEach(function (a) {
      a.font = a.font || {};
      fs(a.font, 13);
    });
    Object.keys(L).forEach(function (key) {
      if (!/^[xy]axis/.test(key)) return;
      var ax = L[key];
      if (!ax || typeof ax !== "object") return;
      ax.tickfont = ax.tickfont || {};
      fs(ax.tickfont, 11);
      if (ax.title && typeof ax.title === "object") {
        ax.title.font = ax.title.font || {};
        fs(ax.title.font, 12);
      }
    });
  }

  /* Fit the figure to its container while preserving the aspect ratio. */
  function fit(entry) {
    var w = entry.inner.clientWidth;
    if (!w) return;
    var h = Math.round(w / entry.ratio);

    if (w < 560) {
      /* A multi-panel figure squeezed to phone width becomes unreadable
         if it also keeps its authored aspect ratio — give it height. */
      h = Math.max(h, Math.min(entry.authoredH, Math.round(window.innerHeight * 0.6)));
      var m = entry.spec.layout.margin;
      if (m) {
        m.l = Math.min(m.l || 60, 46);
        m.r = Math.min(m.r || 20, 12);
        m.t = Math.min(m.t || 60, 40);
        m.b = Math.min(m.b || 60, 42);
      }
    }
    h = Math.max(MIN_H, Math.min(MAX_H, h));
    entry.spec.layout.height = h;
    Plotly.relayout(entry.inner, { height: h, width: null, autosize: true });
  }

  var entries = [];

  function render(entry) {
    var w = entry.inner.clientWidth || window.innerWidth;
    /* Always start from the pristine layout so repeated renders
       (resize, theme flip) do not compound the font scaling. */
    entry.spec.layout = JSON.parse(JSON.stringify(entry.baseLayout));
    var spec = applyTheme(entry.spec, themeColors());
    if (w < 560) scaleFonts(spec, 0.78);
    else if (w < 900) scaleFonts(spec, 0.9);
    Plotly.react(entry.inner, spec.data, spec.layout, {
      displayModeBar: false,
      displaylogo: false,
      responsive: true,
      scrollZoom: false,
      /* Touch devices: let the page scroll rather than trapping the gesture. */
      dragmode: entry.touch ? false : "zoom"
    }).then(function () { fit(entry); });
  }

  function setup() {
    var wraps = document.querySelectorAll(".plotly-wrap[data-src]");
    if (!wraps.length) return;
    var touch = matchMedia("(hover: none)").matches;

    wraps.forEach(function (el) {
      if (el.dataset.title) {
        var top = document.createElement("div");
        top.className = "figtop";
        var tag = document.createElement("span");
        tag.className = "figtag";
        tag.textContent = el.dataset.title;
        top.appendChild(tag);
        el.appendChild(top);
      }
      var inner = document.createElement("div");
      inner.className = "plotly-inner";
      el.appendChild(inner);

      var skeleton = document.createElement("div");
      skeleton.className = "plotly-skeleton";
      skeleton.textContent = "Loading figure…";
      el.appendChild(skeleton);

      fetch(el.dataset.src)
        .then(function (r) { return r.json(); })
        .then(function (spec) {
          skeleton.remove();
          spec.layout = spec.layout || {};
          var w = spec.layout.width  || 800;
          var h = spec.layout.height || 500;
          /* The fixed width is what breaks mobile — drop it. */
          delete spec.layout.width;
          spec.layout.autosize = true;

          var entry = {
            inner: inner, spec: spec,
            baseLayout: JSON.parse(JSON.stringify(spec.layout)),
            authoredH: h,
            ratio: w / h, touch: touch
          };
          entries.push(entry);
          render(entry);
        })
        .catch(function () {
          skeleton.textContent = "Figure could not be loaded.";
          skeleton.classList.add("is-error");
        });
    });

    /* Re-render (not just refit) on resize: crossing a width threshold
       changes the font scale as well as the height. */
    var resizeTimer;
    window.addEventListener("resize", function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () { entries.forEach(render); }, 150);
    }, { passive: true });

    /* Re-render on theme flip so figure colours follow the page. */
    new MutationObserver(function () {
      entries.forEach(render);
    }).observe(document.documentElement, {
      attributes: true, attributeFilter: ["data-theme"]
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setup);
  } else {
    setup();
  }
})();

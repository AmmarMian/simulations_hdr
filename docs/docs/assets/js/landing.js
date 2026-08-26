/* ============================================================
   HDR Simulations — landing page behaviour

   The page is static: it ships a poster of the flow field and no
   WebGL at all. One button swaps in the live field, pulling
   three.js only at that point, and swaps back out again.
   ============================================================ */
(function () {
  "use strict";

  var STATE_KEY = "hdr-hero-live";

  document.addEventListener("DOMContentLoaded", function () {
    var hero = document.getElementById("hero");
    var button = document.getElementById("hero-spark");
    var label = document.getElementById("hero-spark-label");
    var stage = document.getElementById("hero-live");
    var hint = document.getElementById("hero-hint");
    if (!hero || !button || !stage) return;

    var field = null;
    var loading = false;

    function remember(live) {
      try { localStorage.setItem(STATE_KEY, live ? "1" : "0"); } catch (e) { /* private mode */ }
    }

    function setLabel(text) { if (label) label.textContent = text; }

    function stop() {
      if (field) { field.dispose(); field = null; }
      hero.classList.remove("is-live");
      button.classList.remove("is-live");
      button.setAttribute("aria-pressed", "false");
      stage.setAttribute("aria-hidden", "true");
      if (hint) hint.hidden = true;
      setLabel("Set the field in motion");
      remember(false);
    }

    function start() {
      if (loading || field) return;
      loading = true;
      button.classList.add("is-loading");
      button.disabled = true;
      setLabel("Waking the field…");

      /* The template hands us a document-relative path; a dynamic import in a
         classic script would otherwise resolve it against this file's own URL. */
      import(new URL(window.HDR_HERO_MODULE, document.baseURI).href)
        .then(function (mod) { return mod.start(stage); })
        .then(function (instance) {
          field = instance;
          loading = false;
          button.disabled = false;
          button.classList.remove("is-loading");
          button.classList.add("is-live");
          button.setAttribute("aria-pressed", "true");
          hero.classList.add("is-live");
          stage.setAttribute("aria-hidden", "false");
          if (hint) hint.hidden = false;
          setLabel("Let the field rest");
          remember(true);
        })
        .catch(function (err) {
          /* No WebGL, no network, blocked CDN — the poster is already on
             screen, so the page stays whole. Say why, but stay clickable:
             the cause may well be transient. */
          console.error("hero field failed to start", err);
          loading = false;
          button.disabled = false;
          button.classList.remove("is-loading");
          setLabel("Live field unavailable — retry");
          remember(false);
        });
    }

    button.addEventListener("click", function () {
      if (field) stop(); else start();
    });

    /* Someone who turned the field on last visit gets it back, but only if
       they have not asked the OS to keep motion down. */
    var wantsLive = false;
    try { wantsLive = localStorage.getItem(STATE_KEY) === "1"; } catch (e) { /* private mode */ }
    if (wantsLive && !matchMedia("(prefers-reduced-motion: reduce)").matches) start();
  });
})();

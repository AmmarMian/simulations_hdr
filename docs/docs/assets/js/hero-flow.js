/* ============================================================
   HDR Simulations — live hero field

   Tracer particles advected along the curl of a 3-D simplex-noise
   field on the unit sphere, drawn as additive ribbon trails inside
   a glass shell. This is the animated counterpart of the static
   poster in assets/img/hero-field.svg; both integrate the same
   field, so the still is a long exposure of what runs here.

   Loaded on demand — the landing page ships static until the
   visitor asks for the live view.
   ============================================================ */

/* The "three" and "three/addons/" specifiers are resolved by the import map
   the landing template declares — three's example modules import the core by
   bare name, so they need it. */

/* Plasma palette, matching gen_hero_field.py. */
const PAL = {
  bg: "#0b0306",
  head: [1.0, 0.74, 0.5],
  trailA: [1.0, 0.18, 0.52],
  trailB: [1.0, 0.78, 0.34],
  fill: 0x16060d,
  rim: [1.0, 0.46, 0.72],
};

const CFG = {
  flowSpeed: 0.13,
  lineWidth: 0.15,
  glow: 0.3,
  glassBrightness: 0.6,
  autoRotate: true,
};

/* A full-fat field is 8000 tracers; narrow or low-memory devices get
   a lighter one rather than a slideshow. */
function fieldSize() {
  const slim = window.innerWidth < 900 || (navigator.deviceMemory || 8) < 4;
  return slim
    ? { count: 2200, trail: 90 }
    : { count: 8000, trail: 170 };
}

/* ── 3-D simplex noise ─────────────────────────────────────── */
const GRAD3 = [
  [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
  [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
  [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
];

function buildNoise() {
  const p = [];
  for (let i = 0; i < 256; i++) p[i] = i;
  let seed = 1337;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  for (let i = 255; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    const t = p[i]; p[i] = p[j]; p[j] = t;
  }
  const perm = new Uint8Array(512), pmod = new Uint8Array(512);
  for (let i = 0; i < 512; i++) { perm[i] = p[i & 255]; pmod[i] = perm[i] % 12; }

  const F3 = 1 / 3, G3 = 1 / 6;
  return function noise3(xin, yin, zin) {
    let n0, n1, n2, n3;
    const s = (xin + yin + zin) * F3;
    const i = Math.floor(xin + s), j = Math.floor(yin + s), k = Math.floor(zin + s);
    const t = (i + j + k) * G3;
    const x0 = xin - (i - t), y0 = yin - (j - t), z0 = zin - (k - t);
    let i1, j1, k1, i2, j2, k2;
    if (x0 >= y0) {
      if (y0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
      else if (x0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1; }
      else { i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1; }
    } else {
      if (y0 < z0) { i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1; }
      else if (x0 < z0) { i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1; }
      else { i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
    }
    const x1 = x0 - i1 + G3, y1 = y0 - j1 + G3, z1 = z0 - k1 + G3;
    const x2 = x0 - i2 + 2 * G3, y2 = y0 - j2 + 2 * G3, z2 = z0 - k2 + 2 * G3;
    const x3 = x0 - 1 + 3 * G3, y3 = y0 - 1 + 3 * G3, z3 = z0 - 1 + 3 * G3;
    const ii = i & 255, jj = j & 255, kk = k & 255;
    let t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
    if (t0 < 0) n0 = 0; else { t0 *= t0; const g = GRAD3[pmod[ii + perm[jj + perm[kk]]]]; n0 = t0 * t0 * (g[0] * x0 + g[1] * y0 + g[2] * z0); }
    let t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
    if (t1 < 0) n1 = 0; else { t1 *= t1; const g = GRAD3[pmod[ii + i1 + perm[jj + j1 + perm[kk + k1]]]]; n1 = t1 * t1 * (g[0] * x1 + g[1] * y1 + g[2] * z1); }
    let t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
    if (t2 < 0) n2 = 0; else { t2 *= t2; const g = GRAD3[pmod[ii + i2 + perm[jj + j2 + perm[kk + k2]]]]; n2 = t2 * t2 * (g[0] * x2 + g[1] * y2 + g[2] * z2); }
    let t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
    if (t3 < 0) n3 = 0; else { t3 *= t3; const g = GRAD3[pmod[ii + 1 + perm[jj + 1 + perm[kk + 1]]]]; n3 = t3 * t3 * (g[0] * x3 + g[1] * y3 + g[2] * z3); }
    return 32 * (n0 + n1 + n2 + n3);
  };
}

/* ── the field ─────────────────────────────────────────────── */
class HeroFlow {
  constructor(container, THREE, parts) {
    this.el = container;
    this.THREE = THREE;
    this.noise3 = buildNoise();
    this.elapsed = 0;
    this.frame = 0;
    this.tframe = 0;
    this.burstPtr = 0;
    Object.assign(this, fieldSize());
    this.setup(parts);
  }

  setup({ OrbitControls, EffectComposer, RenderPass, UnrealBloomPass, OutputPass }) {
    const THREE = this.THREE, el = this.el;
    const w = el.clientWidth || window.innerWidth;
    const h = el.clientHeight || window.innerHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(PAL.bg);
    const camera = new THREE.PerspectiveCamera(42, w / h, 0.1, 100);
    camera.position.set(0, 0.45, 3.35);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    Object.assign(renderer.domElement.style, {
      position: "absolute", inset: "0", width: "100%", height: "100%",
    });
    el.appendChild(renderer.domElement);
    this.canvas = renderer.domElement;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.rotateSpeed = 0.55;
    controls.minDistance = 1.7;
    controls.maxDistance = 8;
    controls.enablePan = false;
    controls.autoRotate = CFG.autoRotate;
    controls.autoRotateSpeed = 0.45;

    /* Invisible twin of the shell, raycast against for click bursts. */
    const pick = new THREE.Mesh(
      new THREE.SphereGeometry(1, 32, 32),
      new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: false }),
    );
    scene.add(pick);

    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    composer.addPass(new UnrealBloomPass(new THREE.Vector2(w, h), CFG.glow, 0.5, 0.2));
    composer.addPass(new OutputPass());

    Object.assign(this, { scene, camera, renderer, controls, composer, pick });
    this.raycaster = new THREE.Raycaster();
    this.clock = new THREE.Clock();

    this.buildParticles();
    this.buildShell();
    this.bindEvents();
    this.tick = this.tick.bind(this);
    this.raf = requestAnimationFrame(this.tick);
  }

  buildParticles() {
    const THREE = this.THREE, N = this.count, L = this.trail;

    this.pos = new Float32Array(N * 3);
    this.age = new Float32Array(N);
    this.life = new Float32Array(N);
    this.boost = new Float32Array(N);
    this.trailBuf = new Float32Array(N * L * 3);

    const VP = L * 2;  /* two ribbon verts per trail sample */
    this.trailPos = new Float32Array(N * VP * 3);
    this.trailDir = new Float32Array(N * VP * 3);
    this.trailAlpha = new Float32Array(N * VP);
    const trailCol = new Float32Array(N * VP * 3);
    const trailSide = new Float32Array(N * VP);
    const idx = new Uint32Array(N * (L - 1) * 6);

    let ti = 0;
    for (let i = 0; i < N; i++) {
      const vo = i * VP;
      for (let j = 0; j < L; j++) { trailSide[vo + j * 2] = 1; trailSide[vo + j * 2 + 1] = -1; }
      for (let j = 0; j < L - 1; j++) {
        const a = vo + j * 2, b = a + 1, c = a + 2, d = a + 3;
        idx[ti++] = a; idx[ti++] = b; idx[ti++] = c;
        idx[ti++] = b; idx[ti++] = d; idx[ti++] = c;
      }
      /* Each tracer keeps a fixed spot on the trailA → trailB ramp. */
      const m = Math.random();
      const r = PAL.trailA[0] + (PAL.trailB[0] - PAL.trailA[0]) * m;
      const g = PAL.trailA[1] + (PAL.trailB[1] - PAL.trailA[1]) * m;
      const b = PAL.trailA[2] + (PAL.trailB[2] - PAL.trailA[2]) * m;
      for (let v = 0; v < VP; v++) {
        const o = (vo + v) * 3;
        trailCol[o] = r; trailCol[o + 1] = g; trailCol[o + 2] = b;
      }
    }

    this.headPos = new Float32Array(N * 3);
    this.headSize = new Float32Array(N);
    this.headAlpha = new Float32Array(N);
    const headCol = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      headCol[i * 3] = PAL.head[0];
      headCol[i * 3 + 1] = PAL.head[1];
      headCol[i * 3 + 2] = PAL.head[2];
    }

    const dyn = (arr, size) =>
      new THREE.BufferAttribute(arr, size).setUsage(THREE.DynamicDrawUsage);

    const tg = new THREE.BufferGeometry();
    tg.setIndex(new THREE.BufferAttribute(idx, 1));
    tg.setAttribute("position", dyn(this.trailPos, 3));
    tg.setAttribute("aDir", dyn(this.trailDir, 3));
    tg.setAttribute("aSide", new THREE.BufferAttribute(trailSide, 1));
    tg.setAttribute("aAlpha", dyn(this.trailAlpha, 1));
    tg.setAttribute("aColor", new THREE.BufferAttribute(trailCol, 3));
    const tm = new THREE.ShaderMaterial({
      uniforms: { uIntensity: { value: 0.62 }, uWidth: { value: CFG.lineWidth * 0.006 } },
      vertexShader:
        "attribute vec3 aDir; attribute float aSide; attribute float aAlpha; attribute vec3 aColor;" +
        "uniform float uWidth; varying float vA; varying vec3 vC;" +
        "void main(){ vA=aAlpha; vC=aColor; vec4 mv=modelViewMatrix*vec4(position,1.0);" +
        "vec3 t=(modelViewMatrix*vec4(aDir,0.0)).xyz; float tl=length(t);" +
        "if(tl<1e-5){t=vec3(1.0,0.0,0.0);}else{t/=tl;} vec3 vd=normalize(-mv.xyz);" +
        "vec3 off=cross(t,vd); float ol=length(off); if(ol<1e-5){off=vec3(0.0);}else{off/=ol;}" +
        "mv.xyz+=off*aSide*uWidth; gl_Position=projectionMatrix*mv; }",
      fragmentShader:
        "precision mediump float; varying float vA; varying vec3 vC; uniform float uIntensity;" +
        "void main(){ if(vA<=0.002) discard; gl_FragColor=vec4(vC*uIntensity, vA); }",
      transparent: true, blending: THREE.AdditiveBlending,
      depthWrite: false, side: THREE.DoubleSide,
    });
    this.trailGeo = tg;
    const trailMesh = new THREE.Mesh(tg, tm);
    trailMesh.frustumCulled = false;
    this.scene.add(trailMesh);

    const hg = new THREE.BufferGeometry();
    hg.setAttribute("position", dyn(this.headPos, 3));
    hg.setAttribute("aSize", dyn(this.headSize, 1));
    hg.setAttribute("aAlpha", dyn(this.headAlpha, 1));
    hg.setAttribute("aColor", new THREE.BufferAttribute(headCol, 3));
    const hm = new THREE.ShaderMaterial({
      uniforms: { uIntensity: { value: 1.05 }, uCore: { value: 0.6 } },
      vertexShader:
        "attribute float aSize; attribute float aAlpha; attribute vec3 aColor;" +
        "varying float vA; varying vec3 vC;" +
        "void main(){ vA=aAlpha; vC=aColor; vec4 mv=modelViewMatrix*vec4(position,1.0);" +
        "gl_PointSize=aSize*(300.0/ -mv.z); gl_Position=projectionMatrix*mv; }",
      fragmentShader:
        "precision mediump float; varying float vA; varying vec3 vC;" +
        "uniform float uIntensity; uniform float uCore;" +
        "void main(){ if(vA<=0.002) discard; vec2 c=gl_PointCoord-0.5; float d=length(c);" +
        "if(d>0.5) discard; float a=smoothstep(0.5,0.0,d); float core=smoothstep(0.2,0.0,d);" +
        "vec3 col=mix(vC,vec3(1.0),core*uCore); gl_FragColor=vec4(col*uIntensity, a*vA); }",
      transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
    });
    this.headGeo = hg;
    const headPoints = new THREE.Points(hg, hm);
    headPoints.frustumCulled = false;
    this.scene.add(headPoints);

    for (let i = 0; i < N; i++) this.respawn(i, true);
  }

  buildShell() {
    const THREE = this.THREE;
    const mat = new THREE.ShaderMaterial({
      uniforms: {
        uRim: { value: new THREE.Color(PAL.rim[0], PAL.rim[1], PAL.rim[2]) },
        uFill: { value: new THREE.Color(PAL.fill) },
        uBright: { value: CFG.glassBrightness },
      },
      vertexShader:
        "varying vec3 vN; varying vec3 vV;" +
        "void main(){ vec4 mv=modelViewMatrix*vec4(position,1.0);" +
        "vN=normalize(normalMatrix*normal); vV=normalize(-mv.xyz);" +
        "gl_Position=projectionMatrix*mv; }",
      fragmentShader:
        "precision mediump float; varying vec3 vN; varying vec3 vV;" +
        "uniform vec3 uRim; uniform vec3 uFill; uniform float uBright;" +
        "void main(){ float f=pow(1.0-max(dot(vN,vV),0.0),2.6);" +
        "vec3 col=mix(uFill,uRim,f)*uBright; float a=mix(0.10,0.85,f)*mix(0.45,1.0,uBright);" +
        "gl_FragColor=vec4(col,a); }",
      transparent: true, blending: THREE.AdditiveBlending,
      depthWrite: false, side: THREE.FrontSide,
    });
    this.scene.add(new THREE.Mesh(new THREE.SphereGeometry(0.998, 96, 96), mat));
  }

  respawn(i, fresh) {
    const L = this.trail, i3 = i * 3;
    const u = Math.random() * 2 - 1;
    const th = Math.random() * Math.PI * 2;
    const r = Math.sqrt(1 - u * u);
    const x = r * Math.cos(th), y = u, z = r * Math.sin(th);
    this.pos[i3] = x; this.pos[i3 + 1] = y; this.pos[i3 + 2] = z;
    this.age[i] = fresh ? Math.random() * 0.6 : 0;
    this.life[i] = 16 + Math.random() * 20;
    this.boost[i] = 0;
    const b = i * L * 3;
    for (let k = 0; k < L; k++) {
      const t = b + k * 3;
      this.trailBuf[t] = x; this.trailBuf[t + 1] = y; this.trailBuf[t + 2] = z;
    }
  }

  step(dt, now) {
    const N = this.count, L = this.trail, pos = this.pos, tb = this.trailBuf;
    const stp = 0.36 * CFG.flowSpeed * dt;
    const f = 1.75, tt = now * 0.06, e = 0.12;
    this.tframe++;
    const head = this.tframe % L;

    for (let i = 0; i < N; i++) {
      const i3 = i * 3;
      let px = pos[i3], py = pos[i3 + 1], pz = pos[i3 + 2];
      /* Velocity is p × ∇n: tangent to the sphere and divergence-free. */
      const n0 = this.noise3(px * f + tt, py * f, pz * f - tt);
      const nx = this.noise3((px + e) * f + tt, py * f, pz * f - tt) - n0;
      const ny = this.noise3(px * f + tt, (py + e) * f, pz * f - tt) - n0;
      const nz = this.noise3(px * f + tt, py * f, (pz + e) * f - tt) - n0;
      let vx = py * nz - pz * ny, vy = pz * nx - px * nz, vz = px * ny - py * nx;
      let vl = Math.hypot(vx, vy, vz);
      if (vl < 1e-6) { vx = -pz; vy = 0; vz = px; vl = Math.hypot(vx, vy, vz) || 1; }
      px += vx / vl * stp; py += vy / vl * stp; pz += vz / vl * stp;
      const pl = Math.hypot(px, py, pz) || 1;
      px /= pl; py /= pl; pz /= pl;
      pos[i3] = px; pos[i3 + 1] = py; pos[i3 + 2] = pz;

      const ti = (i * L + head) * 3;
      tb[ti] = px; tb[ti + 1] = py; tb[ti + 2] = pz;

      this.age[i] += dt;
      if (this.boost[i] > 0) this.boost[i] = Math.max(0, this.boost[i] - dt * 1.4);
      if (this.age[i] > this.life[i]) this.respawn(i, false);
    }
  }

  writeGeometry() {
    const N = this.count, L = this.trail, tb = this.trailBuf, pos = this.pos;
    const head = this.tframe % L, VP = L * 2, Lm = L - 1;
    const tp = this.trailPos, td = this.trailDir, ta = this.trailAlpha;
    const hp = this.headPos, hs = this.headSize, ha = this.headAlpha;

    for (let i = 0; i < N; i++) {
      const a = this.age[i], lf = this.life[i];
      let fade = Math.min(1, a / 0.45) * Math.min(1, (lf - a) / 1.0);
      if (fade < 0) fade = 0;
      fade *= 1 + this.boost[i] * 1.6;

      const baseSlot = i * L, vo = i * VP;
      for (let j = 0; j < L; j++) {
        const sj = (head - j + L * 1024) % L;
        const pj = (baseSlot + sj) * 3;
        const x = tb[pj], y = tb[pj + 1], z = tb[pj + 2];
        /* Ribbon orientation from the local tangent, central-differenced. */
        const jp = j > 0 ? j - 1 : 0, jn = j < Lm ? j + 1 : Lm;
        const pP = (baseSlot + ((head - jp + L * 1024) % L)) * 3;
        const pN = (baseSlot + ((head - jn + L * 1024) % L)) * 3;
        const dx = tb[pP] - tb[pN], dy = tb[pP + 1] - tb[pN + 1], dz = tb[pP + 2] - tb[pN + 2];
        const v0 = (vo + j * 2) * 3, v1 = v0 + 3;
        tp[v0] = x; tp[v0 + 1] = y; tp[v0 + 2] = z;
        tp[v1] = x; tp[v1 + 1] = y; tp[v1 + 2] = z;
        td[v0] = dx; td[v0 + 1] = dy; td[v0 + 2] = dz;
        td[v1] = dx; td[v1 + 1] = dy; td[v1 + 2] = dz;
        const al = fade * Math.pow(1 - j / Lm, 0.8) * 0.14;
        ta[vo + j * 2] = al; ta[vo + j * 2 + 1] = al;
      }

      const i3 = i * 3;
      hp[i3] = pos[i3]; hp[i3 + 1] = pos[i3 + 1]; hp[i3 + 2] = pos[i3 + 2];
      hs[i] = 0.032 * Math.min(1.4, fade) * (1 + this.boost[i] * 0.9);
      ha[i] = Math.min(1, fade) * 0.85;
    }

    for (const k of ["position", "aDir", "aAlpha"]) this.trailGeo.attributes[k].needsUpdate = true;
    for (const k of ["position", "aSize", "aAlpha"]) this.headGeo.attributes[k].needsUpdate = true;
  }

  burst(pt) {
    const N = this.count, px = pt.x, py = pt.y, pz = pt.z;
    for (let i = 0; i < N; i++) {
      const i3 = i * 3;
      const dp = this.pos[i3] * px + this.pos[i3 + 1] * py + this.pos[i3 + 2] * pz;
      if (dp > 0.55) this.boost[i] = Math.min(2.2, this.boost[i] + (dp - 0.55) * 3.2);
    }
    const n = Math.min(160, Math.floor(N * 0.09));
    for (let q = 0; q < n; q++) {
      const i = (this.burstPtr++) % N, i3 = i * 3;
      let x = px + (Math.random() - 0.5) * 0.05;
      let y = py + (Math.random() - 0.5) * 0.05;
      let z = pz + (Math.random() - 0.5) * 0.05;
      const l = Math.hypot(x, y, z) || 1;
      x /= l; y /= l; z /= l;
      this.pos[i3] = x; this.pos[i3 + 1] = y; this.pos[i3 + 2] = z;
      this.age[i] = 0;
      this.life[i] = 3 + Math.random() * 3.5;
      this.boost[i] = 1.8;
      const b = i * this.trail * 3;
      for (let k = 0; k < this.trail; k++) {
        const t = b + k * 3;
        this.trailBuf[t] = x; this.trailBuf[t + 1] = y; this.trailBuf[t + 2] = z;
      }
    }
  }

  bindEvents() {
    const THREE = this.THREE, el = this.el, cv = this.canvas;

    this.onResize = () => {
      const w = el.clientWidth, h = el.clientHeight;
      if (!w || !h) return;
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
      this.composer.setSize(w, h);
    };
    window.addEventListener("resize", this.onResize);

    const hit = (ev) => {
      const r = cv.getBoundingClientRect();
      const ndc = new THREE.Vector2(
        ((ev.clientX - r.left) / r.width) * 2 - 1,
        -((ev.clientY - r.top) / r.height) * 2 + 1,
      );
      this.raycaster.setFromCamera(ndc, this.camera);
      const it = this.raycaster.intersectObject(this.pick);
      return it.length ? it[0].point.clone().normalize() : null;
    };

    let down = null;
    cv.addEventListener("pointerdown", (ev) => { down = [ev.clientX, ev.clientY]; });
    cv.addEventListener("pointerup", (ev) => {
      if (!down) return;
      /* A press that did not travel is a click, not the end of an orbit drag. */
      if (Math.hypot(ev.clientX - down[0], ev.clientY - down[1]) < 5) {
        const p = hit(ev);
        if (p) this.burst(p);
      }
      down = null;
    });

    /* Idle in the background rather than burning a core on a hidden tab. */
    this.onVisibility = () => {
      if (document.hidden) {
        cancelAnimationFrame(this.raf);
        this.raf = 0;
      } else if (!this.raf && !this.disposed) {
        this.clock.getDelta();
        this.raf = requestAnimationFrame(this.tick);
      }
    };
    document.addEventListener("visibilitychange", this.onVisibility);
  }

  tick() {
    this.raf = requestAnimationFrame(this.tick);
    const dt = Math.min(this.clock.getDelta(), 0.05);
    this.elapsed += dt;
    this.controls.update();
    this.step(dt, this.elapsed);
    this.writeGeometry();
    this.composer.render();
  }

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    window.removeEventListener("resize", this.onResize);
    document.removeEventListener("visibilitychange", this.onVisibility);
    this.controls.dispose();
    this.scene.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose();
    });
    this.composer.dispose();
    this.renderer.dispose();
    if (this.canvas && this.canvas.parentNode) this.canvas.parentNode.removeChild(this.canvas);
  }
}

/* ── entry point ───────────────────────────────────────────── */
export async function start(container) {
  const [THREE, oc, ec, rp, ub, op] = await Promise.all([
    import("three"),
    import("three/addons/controls/OrbitControls.js"),
    import("three/addons/postprocessing/EffectComposer.js"),
    import("three/addons/postprocessing/RenderPass.js"),
    import("three/addons/postprocessing/UnrealBloomPass.js"),
    import("three/addons/postprocessing/OutputPass.js"),
  ]);
  return new HeroFlow(container, THREE, {
    OrbitControls: oc.OrbitControls,
    EffectComposer: ec.EffectComposer,
    RenderPass: rp.RenderPass,
    UnrealBloomPass: ub.UnrealBloomPass,
    OutputPass: op.OutputPass,
  });
}

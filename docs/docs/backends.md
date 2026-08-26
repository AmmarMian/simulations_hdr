# Backend-agnostic code

Every estimator and detector in `hdrlib` is written once and runs on NumPy, PyTorch, CuPy
or JAX, on CPU, CUDA, MPS or Metal, without a line of the algorithm changing. This page
explains how that is arranged, the handful of patterns it rests on, and what it buys —
because the arrangement is not free, and it is worth being explicit about what it costs.

## The problem

The [scalability](getting-started.md) concern is easy to state and awkward to satisfy: the
same covariance estimator should run on a laptop with nothing but NumPy, on a workstation
with an NVIDIA card, and on an Apple Silicon machine — and it should be the *same*
estimator in each case, or the results are not comparable.

There are two obvious answers and both are bad. Hard-coding NumPy gives up the GPU
entirely. Writing the estimator once per library gives you four implementations that drift
apart, four sets of bugs, and benchmark numbers that compare implementations rather than
hardware. What is needed is a single implementation whose *compute library is a runtime
value*.

## The arrangement

The rule the whole of `hdrlib.core.backend` exists to enforce: **an algorithm never imports
a compute library.** It receives a backend and asks for the module it should call. The
import happens in exactly one place, and it happens late.

<figure class="infra-figure">
<div class="infra-scroll">
<svg viewBox="0 0 880 620" width="880" role="img"
     aria-label="Dispatch diagram: a backend command-line flag is parsed into a Backend dataclass of library and device, which resolves to one of four compute modules; the algorithm code is written once against whichever module it is handed, with adapter functions covering the places the libraries diverge, and to_numpy converting results back at the boundary.">
  <defs>
    <marker id="be-arrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--faint)"/>
    </marker>
  </defs>

  <g font-family="var(--font-ui)" font-size="13">

    <!-- 1. the flag -->
    <rect x="315" y="8" width="250" height="48" rx="8" fill="var(--card)" stroke="var(--line2)"/>
    <text x="440" y="30" text-anchor="middle" fill="var(--ink)"
          font-family="var(--font-mono)" font-size="12.5">--backend torch-cuda</text>
    <text x="440" y="47" text-anchor="middle" fill="var(--muted)" font-size="11.5">a runtime value, not an import</text>

    <path d="M 440 56 L 440 92" stroke="var(--line2)" fill="none" marker-end="url(#be-arrow)"/>
    <text x="452" y="79" fill="var(--muted)" font-family="var(--font-mono)" font-size="11">Backend.from_str</text>

    <!-- 2. the descriptor -->
    <rect x="295" y="92" width="290" height="52" rx="8"
          fill="var(--accent-bg)" stroke="var(--accent-line)"/>
    <text x="440" y="114" text-anchor="middle" fill="var(--ink)"
          font-family="var(--font-mono)" font-size="12.5">Backend(lib, device)</text>
    <text x="440" y="132" text-anchor="middle" fill="var(--body)" font-size="11.5">library and device, decoupled</text>

    <path d="M 440 144 L 440 180" stroke="var(--line2)" fill="none" marker-end="url(#be-arrow)"/>
    <text x="452" y="167" fill="var(--muted)" font-family="var(--font-mono)" font-size="11">get_backend_module</text>

    <!-- 3. the four modules -->
    <rect x="20" y="180" width="180" height="46" rx="8" fill="var(--card)" stroke="var(--line2)"/>
    <text x="110" y="202" text-anchor="middle" fill="var(--ink)"
          font-family="var(--font-mono)" font-size="12">numpy</text>
    <text x="110" y="218" text-anchor="middle" fill="var(--muted)" font-size="11">cpu</text>

    <rect x="240" y="180" width="180" height="46" rx="8" fill="var(--card)" stroke="var(--line2)"/>
    <text x="330" y="202" text-anchor="middle" fill="var(--ink)"
          font-family="var(--font-mono)" font-size="12">torch</text>
    <text x="330" y="218" text-anchor="middle" fill="var(--muted)" font-size="11">cpu · cuda · mps</text>

    <rect x="460" y="180" width="180" height="46" rx="8" fill="var(--card)" stroke="var(--line2)"/>
    <text x="550" y="202" text-anchor="middle" fill="var(--ink)"
          font-family="var(--font-mono)" font-size="12">cupy</text>
    <text x="550" y="218" text-anchor="middle" fill="var(--muted)" font-size="11">cuda</text>

    <rect x="680" y="180" width="180" height="46" rx="8" fill="var(--card)" stroke="var(--line2)"/>
    <text x="770" y="202" text-anchor="middle" fill="var(--ink)"
          font-family="var(--font-mono)" font-size="12">jax.numpy</text>
    <text x="770" y="218" text-anchor="middle" fill="var(--muted)" font-size="11">cpu · cuda · metal</text>

    <!-- converge -->
    <path d="M 110 226 L 110 258 L 440 258 L 440 292" stroke="var(--line2)" fill="none" marker-end="url(#be-arrow)"/>
    <path d="M 330 226 L 330 258" stroke="var(--line2)" fill="none"/>
    <path d="M 550 226 L 550 258" stroke="var(--line2)" fill="none"/>
    <path d="M 770 226 L 770 258 L 440 258" stroke="var(--line2)" fill="none"/>

    <!-- 4. the algorithm -->
    <rect x="130" y="292" width="620" height="104" rx="8"
          fill="var(--accent-bg)" stroke="var(--accent-line)"/>
    <text x="440" y="318" text-anchor="middle" fill="var(--ink)" font-size="13.5">the algorithm, written once</text>
    <text x="440" y="340" text-anchor="middle" fill="var(--body)" font-size="11.5">estimators · detectors · simulators — calling only the module it was handed</text>
    <line x1="180" y1="354" x2="700" y2="354" stroke="var(--accent-line)"/>
    <text x="440" y="373" text-anchor="middle" fill="var(--muted)" font-size="11.5">where the libraries genuinely diverge, an adapter absorbs it:</text>
    <text x="440" y="389" text-anchor="middle" fill="var(--muted)"
          font-family="var(--font-mono)" font-size="11">expand_dims · batched_eigh · Unfold2D · sample_standard_normal</text>

    <path d="M 440 396 L 440 440" stroke="var(--line2)" fill="none" marker-end="url(#be-arrow)"/>
    <text x="452" y="423" fill="var(--muted)" font-family="var(--font-mono)" font-size="11">to_numpy</text>

    <!-- 5. the boundary -->
    <rect x="255" y="440" width="370" height="52" rx="8" fill="var(--card-2)" stroke="var(--line2)"/>
    <text x="440" y="462" text-anchor="middle" fill="var(--ink)" font-size="12.5">host arrays, at the boundary only</text>
    <text x="440" y="480" text-anchor="middle" fill="var(--muted)" font-size="11.5">plots · .npz exports · assertions</text>

    <!-- aside: the failure mode -->
    <rect x="130" y="530" width="620" height="66" rx="8"
          fill="none" stroke="var(--line2)" stroke-dasharray="4 4"/>
    <text x="440" y="553" text-anchor="middle" fill="var(--muted)" font-size="11.5">A missing optional library is not an error until it is asked for:</text>
    <text x="440" y="572" text-anchor="middle" fill="var(--muted)"
          font-family="var(--font-mono)" font-size="11">get_backend_module("cupy") → ImportError: … install it with: uv sync --extra cupy</text>
    <text x="440" y="588" text-anchor="middle" fill="var(--faint)" font-size="11">so a laptop with only NumPy installs and runs cleanly</text>

  </g>
</svg>
</div>
<figcaption>The backend is resolved once, at the top; everything below it is written a single time.</figcaption>
</figure>

## The patterns

### A backend is a value, not an import

`Backend` is a frozen dataclass holding two independent things — which library, and which
device — because they genuinely are independent: torch runs on three devices, JAX on three,
CuPy only on CUDA. Validation happens once, in `__post_init__`, so an impossible
combination such as `cupy` on `cpu` fails at construction with a message naming the valid
choices, rather than deep inside a linear-algebra call.

```python
from hdrlib.core.backend import Backend, get_backend_module

b = Backend.from_str("torch-cuda")   # Backend(lib="torch", device="cuda")
bm = get_backend_module(b)           # the torch module itself
b.is_gpu, str(b)                     # True, "torch-cuda"
```

The string round-trip matters more than it looks: it is what lets a backend travel from a
`--backend` flag, through a qanat run record, into a results sidecar, and back out again
when a figure is redrawn.

### The algorithm calls a module it was handed

Given `bm`, the estimator is written the way it would have been written against NumPy:

```python
mean = self.backend_module.sum(X, axis=-2) / n_samples
X = X - expand_dims(self.backend_name, mean, axis=-2)
return (1 / X.shape[-2]) * self.backend_module.swapaxes(X, -1, -2).conj() @ X
```

That is the whole sample-covariance estimator, and it is also the complete torch, CuPy and
JAX implementation. Note what is *not* there: no `if torch`, no device juggling, no
`.cpu()`.

### Adapters absorb the divergences

The line above uses `expand_dims` rather than a `keepdims=` argument, because that keyword
is spelled differently across these libraries. This is the general shape of the fix: where
the APIs disagree, a small function in `backend.py` takes the backend and papers over the
difference, so that the disagreement is stated once instead of at every call site. The same
applies to `permute`, `concatenate`, `masked_set`, `batched_trace` and friends.

### Capability dispatch, where the hardware disagrees

Some divergences are not cosmetic. `batched_eigh` computes one mathematical thing —
eigendecomposition of a stack of SPD matrices — across four different realities:

* **JAX** maps `jax.vmap` over the batch dimensions, which works on Metal natively;
* **torch-mps** has no `eigh` on MPS at all, so it transparently round-trips through the CPU;
* **CUDA**, torch or CuPy, chunks batches beyond 16 000 matrices to stay inside cuSOLVER's limits;
* **everything else** calls `linalg.eigh` directly.

A caller writes `batched_eigh(backend, X)` and receives eigenvalues and eigenvectors. The
fact that one of those four paths silently moved data to the host and back is the adapter's
problem, not the estimator's.

### Strategy objects, where the divergence is structural

When the difference is large enough that a single function would be unreadable, it becomes
an object with one method per backend family. `Unfold2D` extracts sliding windows from an
image time series, and the three libraries do not even agree on the concept:

```python
def __call__(self, data, backend):
    b = _normalize_backend(backend)
    if b.is_torch:
        return self._call_torch(data)          # torch.nn.Unfold, GPU-native
    if b.is_jax:
        return self._call_jax(data, b)         # host fallback, then device_put
    return self._call_numpy_like(data, get_backend_module(backend))
```

NumPy and CuPy share a branch because CuPy deliberately mirrors the NumPy API — which is
worth exploiting wherever it holds.

### Convert at the boundary, not in the middle

`to_numpy` exists so that exactly one function knows how to get a host array out of any of
the four libraries. Everything that leaves the compute path — a plot, an `.npz` export, a
test assertion — goes through it, and nothing inside the compute path does.

## Why bother

**One implementation is one thing to trust.** There is a single sample-covariance
estimator, so it is verified once. More usefully, the test suite parametrises over backends
— `ALL_BACKEND_PARAMS` in `tests/conftest.py` — so the same test runs on each, and where it
matters it asserts they agree: `test_torch_backend_agrees_with_numpy`, in
`tests/test_riemannian_minimisers.py`, is exactly that check for the manifold minimisers.
Cross-backend equality is a test rather than a hope, and a backend-specific bug surfaces as
a disagreement instead of as a wrong figure nobody notices.

**Benchmarks compare hardware, not implementations.** When a GPU timing is set against a
CPU timing, the code being timed is identical. That is the only way the comparison means
anything.

**The parallelisation strategy becomes a choice.** The Monte-Carlo harness reads the
backend and picks its approach accordingly: NumPy fans trials out across a
`multiprocessing.Pool`, one worker per trial, while every array backend stacks the trials
into a leading batch dimension and runs them as one call. Same experiment, same estimator,
two entirely different ways of going fast — selected by a string on the command line.

**A laptop stays a first-class target.** CuPy and JAX are optional extras, imported only
when requested, and a missing one raises an `ImportError` naming the exact `uv sync
--extra` that fixes it. The base install has no GPU dependency and every experiment runs.

## What it costs

It would be dishonest to present this as free.

The adapters are real code with real maintenance, and each is a small admission that the
abstraction leaks: MPS has no `eigh`, JAX-on-Metal rejects complex dtypes — which matters a
great deal for SAR data — and JAX has no native sliding-window primitive, so `Unfold2D`
computes on the host and copies back. None of these are hidden; they are documented in the
docstring of the adapter that handles them, which is the right place for them, but they do
not disappear.

The discipline also has to be maintained. One `np.` slipped into an estimator works
perfectly on the numpy backend and fails, or silently falls back to the CPU, on the others.
That is the standing cost of the arrangement, and running the test suite across backends is
what keeps it visible.

The full API is documented under [`hdrlib.core.backend`](api/core/backend.md).

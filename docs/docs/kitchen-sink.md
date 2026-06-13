<div class="page-header">
  <div class="eyebrow">Design Reference</div>
  <h1>Kitchen <em>Sink</em></h1>
  <p class="standfirst">Every component in one place — use this page to verify the theme
  under warm, cool, and dark paper modes. Resize the window to check the responsive collapse.</p>
</div>

## Body text and sidenotes

The estimator <label for="sn-1" class="sidenote-number"></label><input type="checkbox" id="sn-1" class="margin-toggle"/><span class="sidenote">The Sample Covariance Matrix (SCM) is the maximum likelihood estimator under the Gaussian model. It is consistent but biased, and breaks down when the number of samples is comparable to the dimension.</span> takes a sequence of observation vectors and returns a Hermitian positive-definite matrix. Under the Gaussian model the natural choice is the SCM, but for heavy-tailed distributions — SAR amplitude, sonar returns — robust alternatives such as Tyler's M-estimator converge to the true scatter matrix regardless of the tail weight.

The iterative fixed-point equation <label for="sn-2" class="sidenote-number"></label><input type="checkbox" id="sn-2" class="margin-toggle"/><span class="sidenote">Convergence is guaranteed by a contraction argument on the Riemannian manifold of HPD matrices. Each iterate strictly decreases the negative log-likelihood.</span> is solved by alternating between a quadratic form and a normalisation step until the Frobenius distance between successive iterates falls below a tolerance.

The Kronecker model factors the scatter matrix as a tensor product $\mathbf{A} \otimes \mathbf{B}$, reducing the number of free parameters from $p^2$ to $a^2 + b^2$ when the data has separable spatial and spectral structure.

---

## Code blocks

A basic Python block — the language label and window frame are injected automatically:

```python
def fixed_point_m_estimation(X, psi, max_iter=100, tol=1e-6):
    """Fixed-point M-estimator (centred)."""
    n, p = X.shape
    Sigma = np.eye(p, dtype=X.dtype)
    for _ in range(max_iter):
        Q = np.real(np.einsum("...i,ij,...j->...", X.conj(), np.linalg.inv(Sigma), X))
        weights = psi(Q, p) / p
        Sigma_new = (weights[:, None, None] * np.einsum("...i,...j->...ij", X, X.conj())).mean(0)
        if np.linalg.norm(Sigma_new - Sigma, "fro") < tol:
            return Sigma_new
        Sigma = Sigma_new
    return Sigma
```

A block with a filename annotation:

```python title="src/estimation_kronecker.py"
def kronecker_mm_h0(X, a, b, max_iter=50, tol=1e-5):
    """MM algorithm for Kronecker structured scatter under H0."""
    A = np.eye(a, dtype=complex)
    B = np.eye(b, dtype=complex)
    for _ in range(max_iter):
        A_new = _kronecker_mm_step(X, B, a, b)
        B_new = _kronecker_mm_step(X, A_new, b, a, swap=True)
        if np.linalg.norm(A_new - A) + np.linalg.norm(B_new - B) < tol:
            return A_new, B_new
        A, B = A_new, B_new
    return A, B
```

Followed by a terminal output block:

<div class="terminal">
  <div class="term-head">
    <span class="tdot tdot-r"></span>
    <span class="tdot tdot-y"></span>
    <span class="tdot tdot-g"></span>
    <span class="term-label">Output</span>
  </div>
  <pre><span class="tp">$</span> <span class="tv">uv run python 2-detection/sar_experiments/mc_simulations/mc_kronecker_h1.py --T-max 500 --n-trials 1000</span>
<span class="ta">INFO  backend: torch-cpu  trials: 1000  T_max: 500</span>
<span class="tv">Running H0 pass …  [████████████████████]  done  (4.2 s)</span>
<span class="tv">Running H1 pass …  [████████████████████]  done  (18.7 s)</span>
<span class="ta">Exported → results/sar_mc_kron_h1/run_9/mc_kronecker_h1_a2_b3_T500_n1000.npy</span></pre>
</div>

A YAML config block:

```yaml title="2-detection/experiments/sar_mc_kron_h1.yaml"
name: sar_mc_kron_h1
description: MC power curve — OnlineKroneckerGLRT vs KroneckerGLRT under H1
path: 2-detection/sar_experiments/mc_simulations
executable: 2-detection/sar_experiments/mc_simulations/mc_kronecker_h1.py
tags: [detection, kronecker, H1, monte-carlo, power-curve]
```

---

## Figure placeholder

<div class="figframe">
  <div class="figtop">
    <span class="figtag">Figure 2.4 — Power curves · Kronecker model · H1</span>
  </div>
  <div class="fig-placeholder">Figure goes here — run the experiment and drop the PNG output</div>
  <div class="figcap">
    <span class="fignum">Fig.&nbsp;2.4</span>
    <span>Detection probability as a function of the number of time steps <em>T</em>
    for the online Kronecker GLRT (solid) and the offline Kronecker GLRT (dashed),
    at a fixed false-alarm rate PFA = 10<sup>−2</sup>.
    The online detector approaches the offline curve as <em>T</em> grows.</span>
  </div>
</div>

---

## Callout blocks

<div class="block note">
  <div class="block-head">Note</div>
  <div class="block-body">The MM algorithm requires the data dimension to satisfy
  <em>n &gt; a·b</em> to guarantee a positive-definite estimate at each step.
  For under-sampled regimes, use the regularised variant with a diagonal loading term.</div>
</div>

<div class="block warning">
  <div class="block-head">Warning</div>
  <div class="block-body">The Kronecker estimator assumes the covariance factors
  <strong>A</strong> and <strong>B</strong> share no common scale. Always normalise
  <code>trace(A) = a</code> after each MM step to avoid the scale ambiguity.</div>
</div>

---

## Tables

| Estimator | Model | Convergence | Backend |
|-----------|-------|-------------|---------|
| `SCMEstimator` | Gaussian | Closed form | all |
| `TylerEstimator` | Elliptical | Fixed point | all |
| `HuberEstimator` | Huber | Fixed point | all |
| `ScaledGaussianNaturalGradientEstimator` | Scaled Gaussian | Riemannian GD | all |
| `OnlineScaledGaussianEstimator` | Scaled Gaussian | Online | all |
| `OnlineKroneckerEstimator` | Kronecker SG | Online | torch, cupy |

---

## Margin note with figure

<span class="marginnote">
  <span class="mn-label">Geometry</span>
  The Hermitian positive-definite manifold is a Riemannian symmetric space.
  The geodesic between two points <strong>A</strong> and <strong>B</strong> is
  <em>A</em><sup>1/2</sup>(<em>A</em><sup>−1/2</sup><em>B A</em><sup>−1/2</sup>)<sup><em>t</em></sup><em>A</em><sup>1/2</sup>.
</span>

The natural gradient on the scaled-Gaussian Fisher information metric reduces to a
Riemannian gradient step on the product manifold
**HPD**(*p*) × ℝ<sub>+</sub>.
Each step jointly updates the shape matrix Σ and the scale τ along the geodesic direction,
giving quadratic convergence near the maximum likelihood estimate.

---

## Interactive figure

<span class="marginnote">
  <span class="mn-label">Parameters</span>
  <code>--a 2</code> · Kronecker factor A size<br>
  <code>--b 3</code> · Kronecker factor B size<br>
  <code>--T-max 500</code> · max time steps<br>
  <code>--n-trials 10 000</code> · Monte Carlo trials<br>
  <code>--pfa 1e-2</code> · false-alarm rate<br>
  <code>--seed 42</code>
</span>

<div class="plotly-wrap" data-src="../../assets/data/example_power_curve.json" data-title="Fig. 2.4 — Power curves · Kronecker model · H1"></div>

---

## Blockquote

> The geometry of a covariance matrix is not in its eigenvalues alone,
> but in the interplay between magnitude and direction as the data distribution shifts.

# What ReEig does to the spectrum

Serves the two remarks on the covariance-pooling regime and on ReEig as
spectral shrinkage. It makes no claim about classification performance: it
measures what ReEig does to the spectrum and what that buys backpropagation.

## The statement

Backpropagating through a spectral layer multiplies the incoming error by the
Loewner matrix

$$\boldsymbol{G}_{ij} = \frac{h(\lambda_i)-h(\lambda_j)}{\lambda_i-\lambda_j}.$$

For $h=\log$, the mean value theorem gives
$|\boldsymbol{G}_{ij}| \le 1/\min(\lambda_i,\lambda_j)$, so

$$\max_{ij}|\boldsymbol{G}_{ij}| = 1/\lambda_{\min}.$$

The instability of the eigenvalue-decomposition gradient is therefore governed
by the **smallest** eigenvalue and by nothing else. A ReEig layer with
threshold $\varepsilon$ imposes $\lambda_{\min}\ge\varepsilon$, and so **bounds
the factor by $1/\varepsilon$**. That is the precise sense in which ReEig is a
spectral shrinkage: it pays a bias on the small eigenvalues to buy a bound on
the gradient.

## What is measured

| | script | data |
|---|---|---|
| mechanism, sweep over $(M/d, \text{decay})$ | `main.py` | simulated |
| what it gives on real datasets, sweep over $\varepsilon$ | `real_data.py` | HDM05 / HyperLeaf / Rices90 |

`main.py` models covariance pooling for what it is — a sample covariance over
$M$ pixels for $d$ filters — and sweeps two axes, because the two count
independently: the sampling ratio says how far the empirical
spectrum falls below the true one, the decay says where the true one already
was.

## Result (simulated, `--n_trials 100`, $d=256$, $\varepsilon=10^{-4}$)

| decay | ratio | $\lambda_{\min}$ | cond. | % clipped | Loewner | + ReEig |
|---|---|---|---|---|---|---|
| $10^2$ | 3 | 4.7e-03 | 2.7e+02 | 0.0 % | 2.1e+02 | 2.1e+02 |
| $10^6$ | 0.75 | −3.1e-18 | **singular** | 43.0 % | **undefined** | 1.0e+04 |
| $10^6$ | 3 | 5.9e-07 | 1.8e+06 | 35.2 % | 1.7e+06 | **1.0e+04** |

Three things to take away:

1. **Below ratio 1 the matrix is singular** — centring caps its rank at
   $M-1$ — and LogEig is not defined at all, whatever the decay. This is
   not poor conditioning, it is an absent value.
2. **At the paper's operating point** (ratio $\approx 3$) and at the
   conditioning of the datasets involved ($\sim 10^6$; HDM05 is quoted at
   $9.1\times10^5$), ReEig clips **a third of the spectrum** and divides the
   Loewner factor by **170**.
3. **At low decay, ReEig does nothing** (0 % clipped). The claim that ReEig is
   a necessity rather than a refinement is therefore true *because of the
   spectral decay of the data*, not because of the dimensional regime alone.
   It should be stated in those terms.

## Running it

```sh
uv sync                     # from the root of simulations_hdr
python main.py --n_trials 100                     # simulated figure
python main.py --device cuda --n_trials 1000      # on GPU
```

**MPS is refused explicitly** by `common.py`: no float64, and `linalg.eigh` is
not implemented there, so every spectral layer would fall back to the CPU one
operation at a time. Measured: 0.388 s against 0.364 s in pure CPU. Use CPU or
CUDA.

### Real data

`real_data.py` has **not been run** here (the datasets are not on this
machine). Check first that it runs, with no data at all:

```sh
python real_data.py --self-test
```

then, on a machine that has them — `spdnet-datasets` is also needed, and the
repository does not install it:

```sh
uv pip install git+https://github.com/Yet-Another-Research-Organisation/spdnet-datasets.git
python real_data.py --dataset hdm05 --data-root "$DATA_ROOT/HDM05" \
                    --scaling-factor 190.0 --device cuda
```

The script cross-checks the conditioning it measures against the published one
and reports a discrepancy, on the dimension as well as on the conditioning.

**Mind the scaling factor.** $\varepsilon$ is an *absolute* threshold, hence
not scale invariant: the published configurations apply a `scaling_factor`
(190.0 for HDM05), and multiplying the matrices by a constant multiplies the
spectrum without moving $\varepsilon$. A percentage of clipped eigenvalues can
only be read against the scale of the spectrum, which the script prints
alongside.

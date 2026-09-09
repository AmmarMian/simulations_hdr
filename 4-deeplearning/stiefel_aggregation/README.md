# To what order do projavg and rlavg agree?

Serves the federated-aggregation section. No data, no training: 2 seconds.

## Why

The claim is that the two aggregations agree to within $O(\varepsilon^2)$ when
the local weights stay $O(\varepsilon)$ from the global iterate. The intended
support was the EEG validation curves, "which show superimposed trajectories".
Superimposed trajectories establish that the two schemes agree; they do not
measure **to what order**, and it is the order that decides whether the
recommendation (`projavg`: fewer constants, no need to keep the previous
iterate) is a trade-off or a free choice.

## Result

Order fitted on $\log\lVert\mathrm{projavg}-\mathrm{rlavg}\rVert_F$ against
$\log\varepsilon$, above the rounding floor:

| geometry | $K=2$ | $K=8$ | $K=32$ |
|---|---|---|---|
| $\mathrm{St}(40,20)$ | 2.98 | 2.99 | 2.99 |
| $\mathrm{St}(128,32)$ | 2.99 | 2.99 | 3.00 |
| $\mathrm{St}(64,60)$ | 2.99 | 2.99 | 3.00 |

**The order is three, not two**, on all nine configurations. The claim is true
and conservative.

The number to keep: at a dispersion of $10^{-2}$ between clients, the gap
between the two aggregations is $\sim 10^{-6}$ times the displacement of the
aggregate itself. At the scale federated learning works at, the two schemes are
not "close", they are indistinguishable.

## Consequence

Two options:

1. state the result as $O(\varepsilon^3)$ — which then requires redoing the two
   lines of algebra, the second-order term having to cancel;
2. keep $O(\varepsilon^2)$, which is correct, and say that the bound is met
   with margin, measurement in support.

The second is the safer one as long as the cancellation of the second-order
term has not been established on paper. This script measures, it does not
prove.

## Running it

```sh
python main.py --n_repeats 20
```

Options: `--dimensions 40 20 --dimensions 128 32` (repeat the flag),
`--n_clients 2 8 32`, `--dispersions`, `--device cuda`.

The dispersions stop at $10^{-4}$: below that the gap reaches the float64
rounding floor (~$10^{-14}$) and stops carrying an exponent. `--floor` is what
keeps those points out of the fit.

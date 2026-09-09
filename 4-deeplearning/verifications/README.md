# Session checks

Throwaway scripts backing `../NOTE-reproduction.md`. These are **not** qanat
experiments: no export, no `make_mc_parser`. To be rewritten in the
repository's format if the corresponding figures are kept.

| Script | What it shows | Note |
|---|---|---|
| `smoke.py` | both gradient paths (`use_autograd`) run for the 5 means; time per pass | §6 |
| `iso.py` | the manual/autograd disagreement is localised in `Whitening` / `CongruenceSPD`, on the **matrix** gradient only | §8 |
| `equiv.py` | `\|\|projavg - rlavg\|\|_F` against ε on `St(40,20)`: slope 3 | §4 |

The Wishart / inverse Wishart grid (§3) has no script here: it re-runs as it
stands from `eusipco_2026`.

```sh
python -m eusipco_2026.simulation --wishart-inverse --cpu --n-jobs 8 --output-dir results/wi
```

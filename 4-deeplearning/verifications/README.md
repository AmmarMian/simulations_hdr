# Session checks

Throwaway scripts that check three specific claims. These are **not** qanat
experiments: no export, no `make_mc_parser`. To be rewritten in the
repository's format if the corresponding figures are kept.

| Script | What it shows |
|---|---|
| `smoke.py` | both gradient paths (`use_autograd`) run for the 5 means; time per pass |
| `iso.py` | the manual/autograd disagreement is localised in `Whitening` / `CongruenceSPD`, on the **matrix** gradient only, and only for a non-symmetric incoming gradient — with a symmetric one, the case a network of SPD layers actually produces, the two paths agree to machine precision |
| `equiv.py` | `\|\|projavg - rlavg\|\|_F` against ε on `St(40,20)`: slope 3 |

The Wishart / inverse Wishart grid has no script here: it re-runs as it stands
from `eusipco_2026`.

```sh
python -m eusipco_2026.simulation --wishart-inverse --cpu --n-jobs 8 --output-dir results/wi
```

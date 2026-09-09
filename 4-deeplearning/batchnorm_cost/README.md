# What manual backpropagation through the normalisation layer costs

Part 1 of the batch-normalisation results. No data.

**This is a rewrite, not a reproduction**: none of the five SPDnet
repositories contains any timing or memory measurement code. The three
cost/memory figures of the paper were produced by code that was never
committed.

## Measuring memory

The paper measures `torch.cuda.max_memory_allocated`, which needs a GPU and
mixes the quantity of interest with the allocator's behaviour. What the theory
predicts is narrower and exactly measurable: automatic differentiation has to
**retain the $n$ iterates** of the fixed point, where the manual formula
recomputes.

`torch.autograd.graph.saved_tensors_hooks` intercepts every tensor the graph
keeps alive; summing their sizes measures precisely that, on any hardware, and
it is directly attributable. The allocator peak is recorded as well under CUDA,
for comparison with the paper.

## The mechanism, isolated (`--sweep iterations`, not in the paper)

The sweep that explains the other three. 64×64 matrices, batch 64, geometric
mean:

| iterations $n$ | manual | autograd | ratio |
|---|---|---|---|
| 1 | 5.47 MiB | 7.49 MiB | 1.37× |
| 5 | 7.05 MiB | 18.64 MiB | 2.64× |
| 20 | 20.72 MiB | 60.38 MiB | **2.91×** |

The factor of 2 usually quoted is the few-iterations regime; it grows with $n$,
which **is** the statement being made — two Sylvester equations *per iteration*
for the geometric mean, against two in total for GAH.

## The paper's three figures

`--sweep size` (batch 64), `--sweep batch` (64×64 matrices), `--sweep depth`
(64×64 matrices, batch 64). Extracts, CPU float64, condition number $10^5$:

| sweep | point | mean | mem. manual | mem. autograd | ratio |
|---|---|---|---|---|---|
| size | 512 | geometric | 461.5 MiB | 1115.7 MiB | 2.42× |
| size | 512 | GAH | 312.0 MiB | 404.7 MiB | 1.30× |
| batch | 512 | geometric | 54.4 MiB | 134.9 MiB | 2.48× |
| depth | 32 | geometric | 280.5 MiB | 873.7 MiB | **3.12×** |
| depth | 32 | GAH | 70.8 MiB | 144.1 MiB | 2.04× |

The ratio **grows with all three parameters**, as expected, and it is
systematically higher for the geometric mean than for GAH — the unrolled fixed
point is what carries it.

## Time: not to be oversold

Measured:

| sweep | point | mean | t manual | t autograd |
|---|---|---|---|---|
| size | 512 | geometric | 6001 ms | 6153 ms |
| size | 512 | GAH | 753 ms | 909 ms |
| depth | 32 | geometric | 2684 ms | 2767 ms |

The timings are **comparable**, to within a few percent, and the gap goes one
way as often as the other. **The argument is memory and numerical robustness,
not speed.** The speed-up the paper reports on real data (2× to 5×) comes from
the choice of mean — GAH is an order of magnitude faster than the geometric
one, here as there — and not from the manual derivation.

## Running it

```sh
python main.py --sweep size   --n_repeats 5
python main.py --sweep batch  --n_repeats 5
python main.py --sweep depth  --n_repeats 5
python main.py --sweep iterations --n_repeats 5
```

`--device cuda` adds the allocator peak next to the retained memory. MPS is
refused (no float64, no `linalg.eigh`).

The numbers above are at `--n_repeats 3` on CPU; the memory figures are
deterministic, the timings need re-running on the measurement machine.

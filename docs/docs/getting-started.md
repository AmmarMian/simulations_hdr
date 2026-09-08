# Getting started


<p>
  This page groups documentation that accompany my dissertation for the diploma of Habilitation à Diriger des recherches (HDR) called:
  <br><br>
  <b>Matrices de covariances : des statistiques multivariées à l'apprentissage profond</b>,
  <br><br>
  for which the PDF is available <a href="#">here</a>
  <label for="sn-1" class="sidenote-number"></label>
  <input type="checkbox" id="sn-1" class="margin-toggle"/>
  <span class="sidenote">
    Only available in French language. The documentation being useful to a broader community outside of France has been done in English.
  </span>
</p>


Given that the numerous results presented depend on numerical experimenting, this sidecar allows, for any interested reader, to be able to reproduce, play and experiment each of them<label for="sn-2" class="sidenote-number"></label><input type="checkbox" id="sn-2" class="margin-toggle"/><span class="sidenote">To the exception of experiments on real Sonar and GPR data, not having been granted permission to share the datasets.</span>. The aim is to have a diversity of useful code for working on covariance matrices. This is done with several concerns in mind :

* **reproducibility:** the ability to obtain same results and conclusions on any given computer;
* **scalability:** being able to run on a small laptop or take advantage of CPU/GPU parallelisation when available<label for="sn-3" class="sidenote-number"></label><input type="checkbox" id="sn-3" class="margin-toggle"/><span class="sidenote">We omit the case of HPC parallelisation which introduce a specialized layer that obfuscate the algorithms used.</span>;

The tooling that makes both possible — uv, just and qanat — is listed below; the reasons
for those choices, and how a run turns into a traceable figure, are laid out in
[Infrastructure](infrastructure.md).

## The tools

| Tool | Role | Documentation |
| --- | --- | --- |
| [uv](https://docs.astral.sh/uv/) | Resolves and installs the Python environment | [docs.astral.sh/uv](https://docs.astral.sh/uv/) |
| [just](https://github.com/casey/just) | Runs this repository's recipes — `just --list` shows them all | [just.systems](https://just.systems/man/en/) |
| [qanat](https://ammarmian.fr/qanat/) | Runs the experiments and tracks every execution | [ammarmian.fr/qanat](https://ammarmian.fr/qanat/) |

Only the first two are installed by hand; qanat arrives with the environment.

## Installation

```sh
# uv (macOS / Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# just — pick whichever suits the machine
uv tool install rust-just       # simplest: reuses the uv installed above
brew install just               # macOS
sudo apt install just           # packaged on recent Debian/Ubuntu; also dnf, pacman
cargo install just              # anywhere with a Rust toolchain

# or a prebuilt binary, on any Linux, with none of the above
curl -sSf https://just.systems/install.sh | bash -s -- --to ~/.local/bin
```

The last one needs `~/.local/bin` on your `PATH`.

Then the environment. Each chapter declares its own extras, so install only what you need:

```sh
# base (numpy, torch-cpu, scipy — and qanat)
uv sync

# chapter 2 — optional compute backends
uv sync --extra cupy        # NVIDIA CUDA
uv sync --extra jax         # JAX CPU
uv sync --extra jax-cuda    # JAX CUDA
uv sync --extra jax-metal   # Apple Silicon
```

`uv sync` installs qanat along with everything else — it comes from git rather than PyPI,
which `[tool.uv.sources]` in `pyproject.toml` takes care of. The environment lands in
`.venv/`; activate it once per shell and every command below works unprefixed:

```sh
source .venv/bin/activate            # bash / zsh
source .venv/bin/activate.fish       # fish
```

If you would rather not activate anything, prefix each command with `uv run` instead
(`uv run qanat experiment list`). The `just` recipes do that themselves, so they work
either way.

### Setting up the qanat project

qanat keeps its bookkeeping — the run database and the experiment registry — in a
`.qanat/` directory that is deliberately **not** committed. A fresh clone creates its own,
then registers the experiment definitions found in the chapter folders:

```sh
just init-qanat             # create .qanat/ in this clone — once per clone
just register-experiments   # register every experiments/*.yaml with qanat
```

Re-run `just register-experiments` after adding an experiment YAML, or after wiping
`.qanat/` to start the bookkeeping over.

## Running an experiment

Each YAML file in a chapter's `experiments/` directory defines one experiment, which
[qanat](https://ammarmian.fr/qanat/) then runs and tracks:

```sh
# list all registered experiments
qanat experiment list

# run a specific experiment
qanat experiment run sar_mc_kron_mse

# see that experiment's runs, with their ids
qanat experiment status sar_mc_kron_mse

# run an action on one of them — the run id is required
qanat experiment action sar_mc_kron_mse plot 3
```

Actions always name a specific run: `qanat experiment action <experiment> <action> <run_id>`.
There is no implicit "last run", so read the id off `experiment status` first. When a run
holds several parameter groups, `--group_no` picks between them.

Results land in `results/<experiment_name>/run_<N>/` alongside a self-contained `_plot.py` that can be run standalone.

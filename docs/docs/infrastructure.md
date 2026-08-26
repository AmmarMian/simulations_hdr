# Infrastructure

The two concerns stated on the [Getting started](getting-started.md) page — that a result
should be reproducible on any machine, and that the same code should scale from a laptop
to a cluster — are not properties of the numerical code. They are properties of what
surrounds it: how the environment is pinned, how a run is launched, and what is recorded
while it happens. This page explains the three tools that provide that, and follows a
single experiment from its declaration to the figure printed in the dissertation.

## Why these three

**[uv](https://docs.astral.sh/uv/)** pins the environment. `uv.lock` records the exact
resolved version of every dependency, so `uv sync` reconstructs the same interpreter and
the same package set on any machine — which is the precondition for a result being
reproducible at all. It also makes the optional compute backends tractable: CuPy, JAX-CUDA
and JAX-Metal are hardware-specific and mutually exclusive, so they are declared as extras
and installed only where they can work.

**[just](https://github.com/casey/just)** is the entry point for anything with more than
one step. The multi-command sequences in this repository — registering every experiment
YAML, syncing figures into the dissertation, regenerating these pages — are recipes rather
than instructions in a README, so they cannot drift from what actually works. `just --list`
is the index.

**[qanat](https://ammarmian.fr/qanat/)** runs the experiments and keeps the record. It is
a command-line experiment tracking system: it runs any experiment expressible as a script
taking command-line arguments, and takes responsibility for launching it and recording
everything around the execution — the parameters it was given, the git commit the code sat
at, the duration and exit status, the stdout and stderr, and the directory the outputs went
to. It is built for terminal work on remote machines, where there is no GUI and where jobs
are often handed to a scheduler such as HTCondor or Slurm.

That last point is what makes scalability a configuration rather than a rewrite: the same
experiment definition runs serially on a laptop, across the cores of a workstation, or as
jobs on a cluster, because the *runner* is chosen at launch time and the executable never
learns which one it got.

## The vocabulary

Four qanat terms recur throughout these pages.

* **Experiment** — a workflow to be tracked, declared once. Here each is a YAML file in a
  chapter's `experiments/` directory, naming the executable to run, the actions attached
  to it, and how it is tagged and grouped. The parameters are not listed there: they are
  the executable's own argparse flags, supplied per run.
* **Run** — one execution of an experiment, with a specific set of parameters. Runs are
  numbered, kept side by side, and tied to the commit they ran at, so superseded and even
  failed attempts stay on the record rather than being overwritten.
* **Action** — a script attached to an experiment that operates on a run's results
  directory. Three recur here: `plot` redraws a run's figure, `register` stages a chosen
  figure for the LaTeX dissertation, and `add_to_docs` exports data for these pages.
* **Group run** — a sweep, launching one run per point of a parameter grid in a single
  command.

qanat also handles datasets, containers, comments and document dependencies, none of which
this repository leans on heavily. Its full documentation is at
<https://ammarmian.fr/qanat/>.

## From declaration to trace

Everything downstream hangs off one directory: `results/<experiment>/run_<N>/`. It is
created by qanat, it holds both the outputs and the record of how they were produced, and
the three actions do nothing but read it.

<figure class="infra-figure">
<div class="infra-scroll">
<svg viewBox="0 0 880 610" width="880" role="img"
     aria-label="Flow diagram: an experiment YAML is registered with qanat, run to produce a numbered run directory holding parameters, git commit and outputs, which three actions then consume — plot redraws the figure, register stages it for the dissertation, and add_to_docs exports data to these pages.">
  <defs>
    <marker id="infra-arrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--faint)"/>
    </marker>
  </defs>

  <g font-family="var(--font-ui)" font-size="13">

    <!-- 1. declaration -->
    <rect x="315" y="8" width="250" height="52" rx="8"
          fill="var(--card)" stroke="var(--line2)"/>
    <text x="440" y="30" text-anchor="middle" fill="var(--ink)"
          font-family="var(--font-mono)" font-size="12.5">experiments/*.yaml</text>
    <text x="440" y="48" text-anchor="middle" fill="var(--muted)" font-size="11.5">one declaration per experiment</text>

    <path d="M 440 60 L 440 100" stroke="var(--line2)" fill="none" marker-end="url(#infra-arrow)"/>
    <text x="452" y="84" fill="var(--muted)" font-family="var(--font-mono)" font-size="11">just register-experiments</text>

    <!-- 2. registry -->
    <rect x="315" y="100" width="250" height="52" rx="8"
          fill="var(--card)" stroke="var(--line2)"/>
    <text x="440" y="122" text-anchor="middle" fill="var(--ink)"
          font-family="var(--font-mono)" font-size="12.5">.qanat/</text>
    <text x="440" y="140" text-anchor="middle" fill="var(--muted)" font-size="11.5">registry and run database</text>

    <path d="M 440 152 L 440 196" stroke="var(--line2)" fill="none" marker-end="url(#infra-arrow)"/>
    <text x="452" y="171" fill="var(--muted)" font-family="var(--font-mono)" font-size="11">qanat experiment run</text>
    <text x="452" y="187" fill="var(--faint)" font-size="11">runner: local · parallel · HTCondor</text>

    <!-- 3. the trace -->
    <rect x="255" y="196" width="370" height="96" rx="8"
          fill="var(--accent-bg)" stroke="var(--accent-line)"/>
    <text x="440" y="220" text-anchor="middle" fill="var(--ink)"
          font-family="var(--font-mono)" font-size="12.5">results/&lt;experiment&gt;/run_&lt;N&gt;/</text>
    <text x="440" y="245" text-anchor="middle" fill="var(--body)" font-size="11.5">parameters · git commit · duration · status</text>
    <text x="440" y="263" text-anchor="middle" fill="var(--body)" font-size="11.5">stdout · stderr · outputs</text>
    <text x="440" y="281" text-anchor="middle" fill="var(--muted)" font-size="11.5">and a standalone _plot.py</text>

    <!-- fan-out to the three actions -->
    <path d="M 440 292 L 440 316 L 140 316 L 140 350" stroke="var(--line2)" fill="none" marker-end="url(#infra-arrow)"/>
    <path d="M 440 292 L 440 350" stroke="var(--line2)" fill="none" marker-end="url(#infra-arrow)"/>
    <path d="M 440 292 L 440 316 L 740 316 L 740 350" stroke="var(--line2)" fill="none" marker-end="url(#infra-arrow)"/>

    <!-- 4. actions -->
    <rect x="25" y="350" width="230" height="58" rx="8" fill="var(--card)" stroke="var(--line2)"/>
    <text x="140" y="373" text-anchor="middle" fill="var(--ink)"
          font-family="var(--font-mono)" font-size="12">action: plot</text>
    <text x="140" y="392" text-anchor="middle" fill="var(--muted)" font-size="11.5">redraws the figure in place</text>

    <rect x="325" y="350" width="230" height="58" rx="8" fill="var(--card)" stroke="var(--line2)"/>
    <text x="440" y="373" text-anchor="middle" fill="var(--ink)"
          font-family="var(--font-mono)" font-size="12">action: register</text>
    <text x="440" y="392" text-anchor="middle" fill="var(--muted)" font-size="11.5">stages one chosen figure</text>

    <rect x="625" y="350" width="230" height="58" rx="8" fill="var(--card)" stroke="var(--line2)"/>
    <text x="740" y="373" text-anchor="middle" fill="var(--ink)"
          font-family="var(--font-mono)" font-size="12">action: add_to_docs</text>
    <text x="740" y="392" text-anchor="middle" fill="var(--muted)" font-size="11.5">exports plot data as JSON</text>

    <path d="M 440 408 L 440 446" stroke="var(--line2)" fill="none" marker-end="url(#infra-arrow)"/>
    <path d="M 740 408 L 740 446" stroke="var(--line2)" fill="none" marker-end="url(#infra-arrow)"/>

    <!-- 5. destinations -->
    <rect x="325" y="446" width="230" height="52" rx="8" fill="var(--card-2)" stroke="var(--line2)"/>
    <text x="440" y="468" text-anchor="middle" fill="var(--ink)"
          font-family="var(--font-mono)" font-size="12">hdr_exports/</text>
    <text x="440" y="486" text-anchor="middle" fill="var(--muted)" font-size="11.5">committed hand-off point</text>

    <rect x="625" y="446" width="230" height="52" rx="8" fill="var(--card-2)" stroke="var(--line2)"/>
    <text x="740" y="468" text-anchor="middle" fill="var(--ink)" font-size="12">these pages</text>
    <text x="740" y="486" text-anchor="middle" fill="var(--muted)" font-size="11.5">interactive figures</text>

    <path d="M 440 498 L 440 546" stroke="var(--line2)" fill="none" marker-end="url(#infra-arrow)"/>
    <text x="452" y="522" fill="var(--muted)" font-family="var(--font-mono)" font-size="11">rsync · just figures</text>

    <rect x="285" y="546" width="310" height="56" rx="8"
          fill="var(--card-2)" stroke="var(--line2)"/>
    <text x="440" y="569" text-anchor="middle" fill="var(--ink)" font-size="12">the dissertation PDF</text>
    <text x="440" y="588" text-anchor="middle" fill="var(--muted)" font-size="11.5">figure + provenance line linking back here</text>

  </g>
</svg>
</div>
<figcaption>How an experiment declaration becomes a traced result, and where that trace ends up.</figcaption>
</figure>

## Reading the trace backwards

The point of the arrangement is that it also runs in reverse. Every generated figure in the
dissertation carries a provenance line naming the experiment and the parameters that
produced it, with the experiment name hyperlinked to its page on this site. From a figure
in the PDF you reach its documentation, from there the experiment YAML and the executable,
and from the run record the exact commit the numbers came from — which is what
reproducibility means here in practice.

The two hand-off points deserve a note. `hdr_exports/` is committed on purpose: it lets
experiments run on one machine and figures be compiled on another, with the staged `.tex`
travelling through git rather than being regenerated. And `_plot.py` is written into each
run directory as a standalone script, so a figure can be redrawn years later without qanat,
without this repository, and without reconstructing the environment that produced it.

Both are described step by step in [Getting started](getting-started.md).

# HDR Simulations

# List available commands
default:
    @just --list

# Create the .qanat project database for this clone (.qanat/ is not committed)
init-qanat:
    uv run qanat init .
    @echo "→ then: just register-experiments"

# Register all experiment YAMLs with qanat (run after `qanat init .` or after wiping .qanat)
register-experiments:
    for f in $(find . -path ./.venv -prune -o -path ./results -prune -o -name "*.yaml" -path "*experiments*" -print | sort); do \
        echo "=== $f ==="; \
        uv run qanat experiment new -f "$f"; \
    done

# Regenerate experiment doc pages and chapter cards from YAML configs
docs:
    uv run python docs/scripts/gen_experiment_index.py

# Regenerate the static flow-field poster behind the docs landing page
docs-hero:
    uv run python docs/scripts/gen_hero_field.py

dissertation := "../Dissertation"

# Sync NEW figures only into the dissertation — never touches already-synced plot.tex
sync-figures:
    rsync -av --ignore-existing --exclude figures.toml hdr_exports/ {{dissertation}}/gfx/generated/
    @echo "→ then: cd {{dissertation}} && just figures"

# Show what a full sync WOULD overwrite, without writing anything
sync-figures-diff:
    @rsync -avn --itemize-changes --exclude figures.toml hdr_exports/ {{dissertation}}/gfx/generated/ \
        | grep -E '^>f\.' \
        || echo "Nothing would be overwritten."

# Deliberately replace ONE figure, keeping a .bak of the previous plot.tex
sync-figure exp id:
    rsync -av --backup --suffix=.bak --exclude figures.toml \
        hdr_exports/{{exp}}/{{id}}/ {{dissertation}}/gfx/generated/{{exp}}/{{id}}/
    @echo "→ previous version kept as plot.tex.bak — reapply manual tweaks, then:"
    @echo "  cd {{dissertation}} && just figures"

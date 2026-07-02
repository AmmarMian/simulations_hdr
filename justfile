# HDR Simulations

# List available commands
default:
    @just --list

# Register all experiment YAMLs with qanat (run after `qanat init .` or after wiping .qanat)
register-experiments:
    for f in $(find . -path ./.venv -prune -o -path ./results -prune -o -name "*.yaml" -path "*experiments*" -print | sort); do \
        echo "=== $f ==="; \
        uv run qanat experiment new -f "$f"; \
    done

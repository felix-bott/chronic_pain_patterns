# Locations used by the shell scripts; paths.py is the Python counterpart.
# The code location comes from this file. Everything else can be set with environment variables.
CPP_CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TURBO_ROOT="${TURBO_ROOT:-$(cd "$CPP_CODE/.." && pwd)}"                # holds data/ and the external weights
CPP_RESULTS="${CPP_RESULTS:-$TURBO_ROOT/CPP_results}"                  # everything the pipeline writes
CPP_CHECKPOINTS="${CPP_CHECKPOINTS:-$TURBO_ROOT/CPP_checkpoints}"      # checkpoints of the deep models
LABRAM_WEIGHTS="${LABRAM_WEIGHTS:-$TURBO_ROOT/OTiS_benchmarks/LaBraM-main/checkpoints/labram-base.pth}"
OTIS_WEIGHTS="${OTIS_WEIGHTS:-$TURBO_ROOT/OTiS_weights/checkpoint-197-ncc-0.8818.pth}"
export CPP_CODE TURBO_ROOT CPP_RESULTS CPP_CHECKPOINTS LABRAM_WEIGHTS OTIS_WEIGHTS

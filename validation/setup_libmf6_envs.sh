#!/usr/bin/env bash
# Create two throwaway Julia environments for driving libmf6 through ModflowInterface.jl
# (task card T0.2, docs/reference_tooling.md):
#
#   $REFERENCE_DIR/julia-env/mf6.3.0   Modflow6_jll 6.3.0 with its own libmf6 (as ModflowCoupling)
#   $REFERENCE_DIR/julia-env/mf6.8.2   same packages, but Modflow6_jll's `libmf6_path` and
#                                      `mf6_path` preferences point at the 6.8.2 release
#                                      binaries from validation/fetch_reference.sh
#
# The 6.8.2 override is a JLLWrappers preference in that environment's
# LocalPreferences.toml, so it affects only Julia sessions started with
# --project=<that env>; nothing global (no ~/.julia/artifacts/Overrides.toml) is changed.
# Idempotent: existing environments are kept.
#
# Usage:
#   validation/setup_libmf6_envs.sh
#   julia --project=$REFERENCE_DIR/julia-env/mf6.8.2 validation/libmf6_probe.jl <example> <outdir>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REFERENCE_DIR="${REFERENCE_DIR:-$REPO_ROOT/reference}"
REFERENCE_DIR="$(cd "$REFERENCE_DIR" && pwd)"
LIB682="$REFERENCE_DIR/mf6.8.2_linux/bin/libmf6.so"
BIN682="$REFERENCE_DIR/mf6.8.2_linux/bin/mf6"
[[ -f "$LIB682" ]] || {
    echo "ERROR: $LIB682 missing; run validation/fetch_reference.sh first" >&2
    exit 1
}

make_env() {
    local dir="$1"
    # The marker is written only after the packages were added and load; an env
    # left half-built by an interrupted run is rebuilt.
    if [[ -f "$dir/.setup-complete" ]]; then
        echo "[setup_libmf6_envs] keep $dir" >&2
        return
    fi
    mkdir -p "$dir"
    echo "[setup_libmf6_envs] create $dir" >&2
    julia --startup-file=no --project="$dir" -e '
        using Pkg
        Pkg.add([
            PackageSpec(name = "ModflowInterface", version = "1.0.0"),
            PackageSpec(name = "BasicModelInterface", version = "0.1.1"),
            PackageSpec(name = "Modflow6_jll", version = "6.3.0"),
            PackageSpec(name = "TOML"),
        ])
        Pkg.pin("Modflow6_jll")
    '
    julia --startup-file=no --project="$dir" -e 'using ModflowInterface, BasicModelInterface, Modflow6_jll'
    touch "$dir/.setup-complete"
}

make_env "$REFERENCE_DIR/julia-env/mf6.3.0"
make_env "$REFERENCE_DIR/julia-env/mf6.8.2"

# Point the 6.8.2 environment at the release library (rewritten every run so a moved
# REFERENCE_DIR is picked up).
cat >"$REFERENCE_DIR/julia-env/mf6.8.2/LocalPreferences.toml" <<EOF
# Written by validation/setup_libmf6_envs.sh: use the MODFLOW 6.8.2 release binaries
# instead of the Modflow6_jll 6.3.0 artifact. Scoped to this environment only.
[Modflow6_jll]
libmf6_path = "$LIB682"
mf6_path = "$BIN682"
EOF

# Precompile both, and show which library each one loads.
for v in 6.3.0 6.8.2; do
    julia --startup-file=no --project="$REFERENCE_DIR/julia-env/mf$v" -e '
        using ModflowInterface, Modflow6_jll, Libdl
        f = Libdl.dlsym_e(Modflow6_jll.libmf6_handle, :get_version)
        version = "<no get_version symbol>"
        if f != C_NULL
            buf = zeros(UInt8, 256)
            ccall(f, Cint, (Ptr{UInt8},), buf)
            version = String(buf[1:findfirst(iszero, buf)-1])
        end
        println("[setup_libmf6_envs] mf'"$v"': ", Modflow6_jll.libmf6_path, " -> version ", version)
    '
done

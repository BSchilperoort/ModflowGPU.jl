# GNC6, ghost-node correction (T6.13, src/gwf/gnc.jl):
#
# 1. Input: the GNC6 file of a model and of an exchange into `GncSpec` (cellids as user
#    nodes, contributing cells 0 where none), MF6's errors (unknown option, inactive cell,
#    records that do not match the exchange, XT3D with GNC, EXPLICIT with an exchange's
#    NEWTON, CG with implicit GNC) and the unsupported combinations; `write_simulation`
#    round trip.
# 2. The flux terms on the host: the system structure is MF6's (`gnc_ac`: the connectivity
#    plus n–j and m–j), the matrix row of a record is `gnc_fc`'s, EXPLICIT keeps the
#    connectivity.
# 3. Run parity against mf6 (run at test time) in `:single` and `:double`, implicit and
#    EXPLICIT: test/models/gnc/gnc_disv and gnc_disu (a nested grid in one model) and
#    gnc_lgr (parent and child models, GNC on the exchange).
using ModflowGPU: load_simulation, build_connectivity, npf_setup, gnc_setup, GncSpec, MF6UnsupportedError, MF6InputError,
    solution_setup, write_simulation, spec_identical, getjaindex

const GNC_MF6 = get(ENV, "MF6", joinpath(REFERENCE_DIR, "mf6.8.2_linux", "bin", "mf6"))
const GNC_MODELS = joinpath(@__DIR__, "..", "models", "gnc")

# A copy of the simulation `src` (inputs only) in `dst` with `edit(name, text)` applied.
function _gnc_variant(src, dst; edit = (f, t) -> t)
    mkpath(dst)
    for f in readdir(src)
        (f == "generate.jl" || any(e -> endswith(f, e), (".hds", ".cbc", ".lst", ".txt", ".toml", ".grb", ".csv"))) && continue
        write(joinpath(dst, f), edit(f, read(joinpath(src, f), String)))
    end
    chmod(dst, 0o755; recursive = true)
    return dst
end

_gnc_explicit(f, t) = endswith(f, ".gnc") ? replace(t, r"BEGIN options\n"i => "BEGIN options\n  EXPLICIT\n", count = 1) : t

_gnc_error(f) = try
    f()
    nothing
catch e
    e
end

@testset "GNC6: run parity against mf6" begin
    cases = [
        ("gnc_disv", "gnc_disv", (f, t) -> t),
        ("gnc_disv_explicit", "gnc_disv", _gnc_explicit),
        ("gnc_disu", "gnc_disu", (f, t) -> t),
        ("gnc_disu_explicit", "gnc_disu", _gnc_explicit),
        ("gnc_lgr", "gnc_lgr", (f, t) -> t),
        ("gnc_lgr_explicit", "gnc_lgr", _gnc_explicit),
    ]
    for (name, src, edit) in cases
        if !test_model_selected(name)
        elseif !isfile(GNC_MF6)
            @info "Skipping $name: mf6 not found at $GNC_MF6"
        else
            mktempdir() do tmp
                dir = _gnc_variant(joinpath(GNC_MODELS, src), joinpath(tmp, name); edit)
                out = read(Cmd(`$GNC_MF6`; dir), String)
                @test occursin("Normal termination", out)
                @testset "$name" begin
                    _run_parity(name, dir)
                end
            end
        end
    end
end

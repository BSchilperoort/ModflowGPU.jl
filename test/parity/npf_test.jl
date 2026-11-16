# NPF oracle (T2.2): `npf_setup` against libmf6 6.8.2's `<MODEL>/NPF/CONDSAT` (and K11,
# K22, K33, ANGLE1/2/3, ICELLTYPE, IC/STRT) after initialisation, within 1e-14 relative,
# on ex-gwf-toth, ex-gwf-hanic, ex-gwf-hanir, ex-gwf-twri01 (convertible layer 1: condsat
# is still the full-thickness conductance), the DISV examples ex-gwf-u1disv,
# ex-gwf-curve-90, ex-gwf-curvilin (T6.4) and synthetic grids (the coupling-like and
# pass-through grids of test/parity/connectivity_test.jl, as DIS, DISV and DISU, T6.4) with
# heterogeneous K, K22, K33, all four averaging methods, K22OVERK/K33OVERK and ANGLE1/2/3
# (on DISV/DISU the K ellipse uses the connection angles `anglex`). Then the conversion to the
# engine precision (`NpfModel{T}`) for every preset. libmf6 runs in a subprocess
# (test/libmf6_oracle.jl); skips with a message when reference/ or the libmf6 environment is
# missing.
using ModflowGPU: GwfSpec, NpfModel, NpfSetup, npf_setup, load_simulation

_max_rel_err(a, b) = maximum((abs(x - y) / max(abs(x), abs(y), floatmin(Float64)) for (x, y) in zip(a, b)); init = 0.0)

# Compare `npf_setup(m)` with libmf6's arrays of the model; returns the setup and the
# largest relative condsat difference.
function _check_libmf6_npf(m::GwfSpec, lib::Dict{String, Any})
    c = build_connectivity(m)
    s = npf_setup(m, c)
    p = s.properties
    @test length(s.condsat) == length(lib["NPF/CONDSAT"]) == c.njas
    err = _max_rel_err(s.condsat, lib["NPF/CONDSAT"])
    @test err <= 1.0e-14
    @test _rel_close(p.k11, lib["NPF/K11"])
    @test _rel_close(p.k22, lib["NPF/K22"])
    @test _rel_close(p.k33, lib["NPF/K33"])
    @test p.icelltype == lib["NPF/ICELLTYPE"]
    for (a, name) in ((p.angle1, "NPF/ANGLE1"), (p.angle2, "NPF/ANGLE2"), (p.angle3, "NPF/ANGLE3"))
        @test isnothing(a) == !haskey(lib, name)
        isnothing(a) || @test _rel_close(a, lib[name])
    end
    @test m.ic.strt[c.nodeuser] == lib["IC/STRT"]
    return s, err
end

# Deterministic heterogeneous arrays on a DIS grid `(ncol, nrow, nlay)`: K over four orders
# of magnitude with some neighbours within 0.5 % of each other (the arithmetic branch of
# `logmean`), K22/K and K33/K ratios, angles in degrees, and ICELLTYPE with -1, 0 and 1.
function _npf_test_arrays(dis)
    sz = size(dis.botm)
    lin = LinearIndices(sz)
    k = [10.0^(((7 * j + 3 * i + 5 * kk) % 13) / 3 - 2) for j in 1:sz[1], i in 1:sz[2], kk in 1:sz[3]]
    for I in CartesianIndices(sz)
        I[1] > 1 && lin[I] % 4 == 0 && (k[I] = k[I - CartesianIndex(1, 0, 0)] * (1 + 0.001 * (lin[I] % 3)))
    end
    k22r = [0.05 + 0.9 * ((3 * lin[I]) % 7) / 6 for I in CartesianIndices(sz)]
    k33r = [0.01 + 0.2 * ((5 * lin[I]) % 11) / 10 for I in CartesianIndices(sz)]
    angle1 = [-90.0 + 15.0 * (lin[I] % 13) for I in CartesianIndices(sz)]
    angle2 = [-30.0 + 7.5 * (lin[I] % 9) for I in CartesianIndices(sz)]
    angle3 = [-45.0 + 11.25 * (lin[I] % 9) for I in CartesianIndices(sz)]
    icelltype = Int32[(lin[I] % 3) - 1 for I in CartesianIndices(sz)]
    strt = [dis.top[I[1], I[2]] + 10.0 * sin(lin[I]) for I in CartesianIndices(sz)]
    return (; k, k22r, k33r, angle1, angle2, angle3, icelltype, strt)
end

# (label, NPF options, NPF arrays) for the synthetic oracle cases on grid `dis`.
function _npf_synthetic_cases(dis)
    a = _npf_test_arrays(dis)
    base = Dict{String, Any}("icelltype" => a.icelltype, "k" => a.k, "k33" => a.k .* a.k33r)
    cases = Any[]
    for avg in ("", "LOGARITHMIC", "AMT-LMK", "AMT-HMK")
        opts = isempty(avg) ? String[] : ["ALTERNATIVE_CELL_AVERAGING $avg"]
        push!(cases, ("K, K33, $(isempty(avg) ? "harmonic" : avg)", opts, base))
    end
    push!(
        cases, (
            "K22OVERK, K33OVERK, ANGLE1/2/3", ["K22OVERK", "K33OVERK"],
            Dict{String, Any}(
                "icelltype" => a.icelltype, "k" => a.k, "k22" => a.k22r, "k33" => a.k33r,
                "angle1" => a.angle1, "angle2" => a.angle2, "angle3" => a.angle3,
            ),
        ),
    )
    push!(
        cases, (
            "K22, ANGLE1, AMT-HMK", ["ALTERNATIVE_CELL_AVERAGING AMT-HMK"],
            Dict{String, Any}("icelltype" => 0, "k" => a.k, "k22" => a.k .* a.k22r, "angle1" => a.angle1),
        ),
    )
    return cases, a.strt
end

@testset "NPF: libmf6 CONDSAT oracle" begin
    exdir = joinpath(REFERENCE_DIR, "mf6examples")
    if !libmf6_available()
        @info "Skipping the libmf6 NPF oracle: libmf6 environment $LIBMF6_ENV not found (validation/fetch_reference.sh, validation/setup_libmf6_envs.sh)"
    else
        setups = Pair{String, NpfSetup}[]
        @testset "$name" for name in ("ex-gwf-toth", "ex-gwf-hanic", "ex-gwf-hanir", "ex-gwf-twri01", "ex-gwf-u1disv", "ex-gwf-curve-90", "ex-gwf-curvilin")
            simdir = joinpath(exdir, name)
            if !isdir(simdir)
                @info "Skipping the NPF oracle on $name: $simdir not found"
                continue
            end
            m = only(load_simulation(simdir; strict = false).models)
            s, err = _check_libmf6_npf(m, libmf6_connectivity(simdir)[uppercase(m.name)])
            @info "NPF CONDSAT vs libmf6 6.8.2: $name, $(length(s.condsat)) connections, max relative difference $err"
            push!(setups, name => s)
        end
        # T6.4: the grids also as DISV (and without pass-through cells as DISU with vertices),
        # the arrays in that grid's shape (same user node order).
        asdisv(dis) = (_dis_as_disv(dis), a -> a isa AbstractArray ? reshape(a, :, size(a, 3)) : a)
        asdisu(dis) = (_dis_as_disu(dis; vertices = true), a -> a isa AbstractArray ? vec(a) : a)
        @testset "synthetic $glabel: $label" for (glabel, dis, conv) in (
                    ("coupling-like", _synthetic_coupling_dis(), d -> (d, identity)),
                    ("pass-through", _synthetic_passthrough_dis(), d -> (d, identity)),
                    ("coupling-like DISV", _synthetic_coupling_dis(), asdisv), ("pass-through DISV", _synthetic_passthrough_dis(), asdisv),
                    ("coupling-like DISU", _synthetic_coupling_dis(), asdisu),
                ),
                (label, opts, arrays) in first(_npf_synthetic_cases(dis))

            grid, shape = conv(dis)
            mktempdir() do dir
                write_dis_simulation(
                    dir, grid; name = "synth", npf_options = opts, npf_arrays = Dict{String, Any}(k => shape(v) for (k, v) in arrays),
                    strt = shape(last(_npf_synthetic_cases(dis))),
                )
                m = only(load_simulation(dir; strict = false).models)
                s, err = _check_libmf6_npf(m, libmf6_connectivity(dir)["SYNTH"])
                @info "NPF CONDSAT vs libmf6 6.8.2: synthetic $glabel ($label), $(length(s.condsat)) connections, max relative difference $err"
                push!(setups, "$glabel: $label" => s)
            end
        end

        # Conversion to the engine precision: one rounding of the Float64 host values.
        for_each_precision() do p
            for (label, s) in setups
                npf = NpfModel{p.T}(s)
                @test eltype(npf.condsat) == eltype(npf.Δr) == p.T
                @test npf.condsat == p.T.(s.condsat)
                @test npf.Δr == p.T.(s.Δr)
                tol = eps(p.T) / 2
                @test _max_rel_err(Float64.(npf.condsat), s.condsat) <= tol
                @test all(abs(Float64(x) - y) <= tol * abs(y) for (x, y) in zip(npf.Δr, s.Δr))
            end
        end
    end
end

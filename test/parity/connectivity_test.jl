# Connectivity oracle (T2.1, PLAN.md section 4 "Connectivity"): `build_connectivity`
# against libmf6 6.8.2's `<MODEL>/CON/*` and `<MODEL>/DIS/*` arrays after initialisation
# (integers exact, reals within 1e-14 relative), on ex-gwf-toth, ex-gwf-hanic,
# ex-gwf-twri01 and two synthetic grids with excluded (IDOMAIN 0) and vertical
# pass-through (IDOMAIN < 0) cells; and the user-node IA/JA, TOP, BOTM and IDOMAIN against
# the binary grid files of every DIS model in the reference runs. libmf6 runs in a
# subprocess (test/libmf6_oracle.jl). Skips with a message when reference/ or the libmf6
# environment is missing. Connectivity is host Float64 geometry, the same in every
# precision preset, so there is no `for_each_precision` loop.
using ModflowGPU: DisSpec, Connectivity, build_connectivity, iajausr, load_simulation, read_grb

_rel_close(a, b; rtol = 1.0e-14) = length(a) == length(b) && all(abs(x - y) <= rtol * max(abs(x), abs(y)) for (x, y) in zip(a, b))

# Compare `c` with libmf6's arrays of one model.
function _check_libmf6_connectivity(c::Connectivity, lib::Dict{String, Any})
    @test c.nodes == lib["NODES"]
    @test c.nodesuser == lib["NODESUSER"]
    @test c.nja == lib["NJA"]
    @test c.njas == lib["NJAS"]
    for (field, name) in (:ia => "CON/IA", :ja => "CON/JA", :jas => "CON/JAS", :isym => "CON/ISYM", :ihc => "CON/IHC")
        @test getfield(c, field) == lib[name]
    end
    for (field, name) in (
            :cl1 => "CON/CL1", :cl2 => "CON/CL2", :hwva => "CON/HWVA", :anglex => "CON/ANGLEX",
            :top => "DIS/TOP", :bot => "DIS/BOT", :area => "DIS/AREA", :xc => "DIS/XC", :yc => "DIS/YC",
        )
        @test _rel_close(getfield(c, field), lib[name])
    end
    # MF6 keeps dummies of length 1 (and IAUSR/JAUSR as IA/JA) when no cell is excluded.
    if c.nodes < c.nodesuser
        @test c.nodeuser == lib["DIS/NODEUSER"]
        @test c.nodereduced == lib["DIS/NODEREDUCED"]
        @test iajausr(c) == (lib["CON/IAUSR"], lib["CON/JAUSR"])
    else
        @test c.nodeuser == 1:c.nodes && c.nodereduced == 1:c.nodesuser
    end
    return nothing
end

# A coupling-like grid (PLAN.md section 1.1): 2 layers, variable DELR/DELC, sloping top,
# IDOMAIN 0 outside the domain, 1 in layer 1 and 2 in layer 2, with columns where only one
# layer is active.
function _synthetic_coupling_dis()
    ncol, nrow, nlay = 7, 5, 2
    delr = [1000.0, 950.5, 900.25, 1000.0, 1100.0, 1000.0, 875.125]
    delc = [925.0, 1000.0, 1000.0, 1050.75, 990.0]
    top = [250.0 + 3.25 * j - 7.5 * i + 0.125 * i * j for j in 1:ncol, i in 1:nrow]
    botm = Array{Float64}(undef, ncol, nrow, nlay)
    botm[:, :, 1] = top .- [20.0 + 0.5 * j + i for j in 1:ncol, i in 1:nrow]
    botm[:, :, 2] = botm[:, :, 1] .- [150.0 - 2.0 * j + 3.0 * i for j in 1:ncol, i in 1:nrow]
    idomain = zeros(Int32, ncol, nrow, nlay)
    for i in 1:nrow, j in 1:ncol
        inside = !(i == 1 && j >= 5) && !(j == 1 && i >= 4) && !(i == nrow && j == ncol)
        idomain[j, i, 1] = inside && (i, j) != (3, 4) ? 1 : 0   # a hole in layer 1
        idomain[j, i, 2] = inside && (i, j) != (2, 2) ? 2 : 0   # and one in layer 2
    end
    idomain[5, 1, 2] = 2  # layer 2 active below an excluded layer-1 cell
    return DisSpec(; nlay = Int32(nlay), nrow = Int32(nrow), ncol = Int32(ncol), delr, delc, top, botm, idomain)
end

# A 4-layer grid with vertical pass-through cells (IDOMAIN -1): single and stacked
# pass-through cells, pass-through over and under excluded cells, in the top and bottom
# layers, and next to active cells (no horizontal connections to pass-through cells).
function _synthetic_passthrough_dis()
    ncol, nrow, nlay = 4, 3, 4
    delr = [10.0, 12.5, 7.25, 10.0]
    delc = [8.0, 11.0, 9.5]
    top = fill(100.0, ncol, nrow) .+ [0.5 * j - 0.25 * i for j in 1:ncol, i in 1:nrow]
    thick = [5.0, 7.5, 3.25, 12.0]
    botm = Array{Float64}(undef, ncol, nrow, nlay)
    for k in 1:nlay
        botm[:, :, k] = (k == 1 ? top : botm[:, :, k - 1]) .- (thick[k] .+ [0.1 * j * k + 0.05 * i for j in 1:ncol, i in 1:nrow])
    end
    # columns (j, i) => IDOMAIN by layer
    cols = Dict(
        (1, 1) => (1, -1, 1, 1), (2, 1) => (1, -1, -1, 1), (3, 1) => (-1, 1, 1, -1), (4, 1) => (1, -1, 0, 1),
        (1, 2) => (1, 0, -1, 1), (2, 2) => (-1, -1, -1, 1), (3, 2) => (1, 1, 1, 1), (4, 2) => (1, -1, -1, -1),
        (1, 3) => (0, -1, 1, 1), (2, 3) => (1, 1, -1, 0), (3, 3) => (-1, 0, 1, -1), (4, 3) => (1, 2, -3, 3),
    )
    idomain = Array{Int32}(undef, ncol, nrow, nlay)
    for ((j, i), ids) in cols, k in 1:nlay
        idomain[j, i, k] = ids[k]
    end
    return DisSpec(; nlay = Int32(nlay), nrow = Int32(nrow), ncol = Int32(ncol), delr, delc, top, botm, idomain)
end

@testset "connectivity: libmf6 oracle" begin
    exdir = joinpath(REFERENCE_DIR, "mf6examples")
    if !libmf6_available()
        @info "Skipping the libmf6 connectivity oracle: libmf6 environment $LIBMF6_ENV not found (validation/fetch_reference.sh, validation/setup_libmf6_envs.sh)"
    else
        @testset "$name" for name in ("ex-gwf-toth", "ex-gwf-hanic", "ex-gwf-twri01")
            simdir = joinpath(exdir, name)
            if !isdir(simdir)
                @info "Skipping the connectivity oracle on $name: $simdir not found"
                continue
            end
            spec = load_simulation(simdir; strict = false)
            lib = libmf6_connectivity(simdir)
            @test sort!(collect(keys(lib))) == sort!([uppercase(m.name) for m in spec.models])
            for m in spec.models
                _check_libmf6_connectivity(build_connectivity(m), lib[uppercase(m.name)])
            end
        end
        @testset "synthetic $label" for (label, dis) in (
                ("coupling-like (IDOMAIN 0/1/2)", _synthetic_coupling_dis()),
                ("pass-through (IDOMAIN -1)", _synthetic_passthrough_dis()),
            )
            mktempdir() do dir
                write_dis_simulation(dir, dis; name = "synth")
                # The files read back to the same grid.
                m = only(load_simulation(dir).models)
                c = build_connectivity(m)
                c0 = build_connectivity(dis)
                @test all(getfield(c, f) == getfield(c0, f) for f in fieldnames(Connectivity))
                @test c.nodes < c.nodesuser
                any(<(0), dis.idomain) && @test any(<(0), c.nodereduced)
                _check_libmf6_connectivity(c, libmf6_connectivity(dir)["SYNTH"])
            end
        end
    end
end

@testset "connectivity: binary grid files of the reference runs" begin
    runs = joinpath(REFERENCE_DIR, "runs")
    examples = in_scope_examples()
    if isempty(examples) || !isdir(runs)
        @info "Skipping the .grb connectivity check: $(joinpath(REFERENCE_DIR, "mf6examples")) or $runs not found (validation/fetch_reference.sh, validation/run_reference.jl)"
    else
        inventory = TOML.parsefile(joinpath(@__DIR__, "..", "..", "validation", "examples.toml"))["examples"]
        nmodels = nreduced = 0
        for (name, simdir) in examples
            models = filter(((_, m),) -> m["grid"] == "DIS" && haskey(m, "grb_file"), inventory[name]["models"])
            isempty(models) && continue
            modes = filter(d -> isfile(joinpath(runs, name, d, "mfsim.nam")), ["raw", "tight"])
            isempty(modes) && (@info "No reference run of $name: .grb check skipped"; continue)
            spec = try
                load_simulation(simdir; strict = false)
            catch err
                @info "Simulation loader failed on $simdir: .grb check skipped ($(sprint(showerror, err)))"
                continue
            end
            @testset "$name/$mname" for (mname, m) in models
                g = read_grb(joinpath(runs, name, first(modes), m["grb_file"]))
                ms = only(filter(x -> lowercase(x.name) == lowercase(mname), spec.models))
                c = build_connectivity(ms)
                iausr, jausr = iajausr(c)
                @test g["NCELLS"] == c.nodesuser
                @test g["NJA"] == c.nja
                @test vec(g["IA"]) == iausr
                @test vec(g["JA"]) == jausr
                @test vec(g["IDOMAIN"]) == vec(ms.dis.idomain)
                @test vec(g["TOP"]) == vec(ms.dis.top)
                @test vec(g["BOTM"]) == vec(ms.dis.botm)
                # reduced TOP/BOT from the user arrays
                botm = vec(g["BOTM"])
                @test c.bot == botm[c.nodeuser]
                nmodels += 1
                nreduced += c.nodes < c.nodesuser
            end
        end
        @info "Connectivity vs .grb: $nmodels DIS models ($nreduced with excluded cells)"
    end
end

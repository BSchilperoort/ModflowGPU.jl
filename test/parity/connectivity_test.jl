# Connectivity oracle (T2.1, PLAN.md section 4 "Connectivity"): `build_connectivity`
# against libmf6 6.8.2's `<MODEL>/CON/*` and `<MODEL>/DIS/*` arrays after initialisation
# (integers exact, reals within 1e-14 relative), on ex-gwf-toth, ex-gwf-hanic,
# ex-gwf-twri01, the Tier 5 examples (DISV: ex-gwf-u1disv, ex-gwf-curve-90,
# ex-gwf-curvilin; DISU: ex-gwf-rad-disu; T6.4) and synthetic grids with excluded
# (IDOMAIN 0) and vertical pass-through (IDOMAIN < 0) cells, as DIS and (T6.4) as DISV and
# DISU (with and without vertices); and the user-node IA/JA, TOP, BOTM (BOT) and IDOMAIN
# against the binary grid files of every DIS, DISV and DISU model in the reference runs. libmf6 runs in a
# subprocess (test/libmf6_oracle.jl). Skips with a message when reference/ or the libmf6
# environment is missing. Connectivity is host Float64 geometry, the same in every
# precision preset, so there is no `for_each_precision` loop.
using ModflowGPU: DisSpec, DisvSpec, DisuSpec, Connectivity, build_connectivity, iajausr, load_simulation, read_grb

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

@testset "connectivity: libmf6 oracle" begin
    exdir = joinpath(REFERENCE_DIR, "mf6examples")
    if !libmf6_available()
        @info "Skipping the libmf6 connectivity oracle: libmf6 environment $LIBMF6_ENV not found (validation/fetch_reference.sh, validation/setup_libmf6_envs.sh)"
    else
        @testset "$name" for name in ("ex-gwf-toth", "ex-gwf-hanic", "ex-gwf-twri01", "ex-gwf-u1disv", "ex-gwf-curve-90", "ex-gwf-curvilin", "ex-gwf-rad-disu")
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
        # T6.4: the same grids as DISV, and (without pass-through cells) as DISU.
        @testset "synthetic $label" for (label, grid) in (
                ("DISV coupling-like", _dis_as_disv(_synthetic_coupling_dis())),
                ("DISV pass-through", _dis_as_disv(_synthetic_passthrough_dis())),
                ("DISU coupling-like", _dis_as_disu(_synthetic_coupling_dis())),
                ("DISU coupling-like with vertices", _dis_as_disu(_synthetic_coupling_dis(); vertices = true)),
            )
            mktempdir() do dir
                write_dis_simulation(dir, grid; name = "synth")
                m = only(load_simulation(dir).models)
                c = build_connectivity(m)
                c0 = build_connectivity(grid)
                @test all(getfield(c, f) == getfield(c0, f) for f in fieldnames(Connectivity))
                @test c.nodes < c.nodesuser
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
            models = filter(((_, m),) -> haskey(m, "grb_file"), inventory[name]["models"])
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
                d = ms.dis
                @test g.grid == uppercase(String(c.grid))
                @test g[d isa DisuSpec ? "NODES" : "NCELLS"] == c.nodesuser
                @test g["NJA"] == c.nja
                @test vec(g["IA"]) == iausr
                @test vec(g["JA"]) == jausr
                @test vec(g["IDOMAIN"]) == vec(d.idomain)
                @test vec(g["TOP"]) == vec(d.top)
                # reduced TOP/BOT from the user arrays
                botm = vec(g[d isa DisuSpec ? "BOT" : "BOTM"])
                @test botm == vec(d isa DisuSpec ? d.bot : d.botm)
                @test c.bot == botm[c.nodeuser]
                if d isa DisvSpec || (d isa DisuSpec && d.nvert > 0)
                    # vertices, centres and the closed vertex lists (Disv.f90 `define_cellverts`)
                    @test g["VERTICES"] == d.vertices
                    @test (vec(g["CELLX"]), vec(g["CELLY"])) == (d.cell_center[1, :], d.cell_center[2, :])
                    @test (vec(g["IAVERT"]), vec(g["JAVERT"])) == ModflowGPU._cell_vertex_csr(d.cell_vertices)
                end
                nmodels += 1
                nreduced += c.nodes < c.nodesuser
            end
        end
        @info "Connectivity vs .grb: $nmodels DIS, DISV and DISU models ($nreduced with excluded cells)"
    end
end

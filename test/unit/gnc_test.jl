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
    solution_setup, write_simulation, spec_identical, getjaindex, GwfModel, apply_period!, start_timestep!, formulate!,
    time_steps, time_step_window, backend_name

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

# The model without its GNC6 package (name file entry or exchange option removed).
_gnc_remove(f, t) = endswith(f, ".gnc") ? t : replace(t, r"\n *GNC6 +FILEIN[^\n]*"i => "", r"\n *GNC6 [^\n]*"i => "")

_gnc_write(dir, name, f) = write(joinpath(dir, name), f(read(joinpath(dir, name), String)))

@testset "GNC6: input" begin
    m = only(load_simulation(joinpath(GNC_MODELS, "gnc_disv")).models)
    g = m.gnc
    @test g isa GncSpec && !g.explicit && g.numalphaj == 1 && length(g.noden) == 72
    # First record: (1, 10) (1, 15) (1, 9) 1/3; the second has no contributing cell.
    @test (g.noden[1], g.nodem[1], g.nodesj[1, 1]) == (10, 15, 9) && g.alphasj[1, 1] == 1 / 3
    @test g.nodesj[1, 2] == 0 && g.alphasj[1, 2] == 0.0
    @test g.noden[37] == 121 + 10  # layer 2
    u = only(load_simulation(joinpath(GNC_MODELS, "gnc_disu")).models).gnc
    @test u.noden == g.noden && u.nodem == g.nodem && u.nodesj == g.nodesj && u.alphasj == g.alphasj
    ex = only(load_simulation(joinpath(GNC_MODELS, "gnc_lgr")).exchanges)
    @test ex.gnc isa GncSpec && ex.gnc.noden == ex.data.nodem1 && ex.gnc.nodem == ex.data.nodem2
    mktempdir() do tmp
        # Round trip through write_simulation (also EXPLICIT).
        for src in ("gnc_disu", "gnc_lgr")
            dir = _gnc_variant(joinpath(GNC_MODELS, src), joinpath(tmp, src); edit = _gnc_explicit)
            a = load_simulation(dir)
            @test all(x -> isnothing(x.gnc) || x.gnc.explicit, [a.models; a.exchanges])
            out = joinpath(tmp, src * "_written")
            write_simulation(out, a)
            @test spec_identical(a, load_simulation(out); ignore = (:simdir, :given))
        end
        # MF6's input errors (GhostNode.f90 read_options, read_data).
        dir = _gnc_variant(joinpath(GNC_MODELS, "gnc_disv"), joinpath(tmp, "e"))
        gnc = read(joinpath(dir, "gnc_disv.gnc"), String)
        _gnc_write(dir, "gnc_disv.gnc", t -> replace(t, "BEGIN options\n" => "BEGIN options\n  FOO\n"))
        e = input_error_of(() -> load_simulation(dir))
        @test e isa MF6InputError && occursin("Unknown GNC option: FOO", e.msg)
        write(joinpath(dir, "gnc_disv.gnc"), replace(gnc, "NUMGNC  72" => "NUMGNC  71"))
        @test input_error_of(() -> load_simulation(dir)) isa MF6InputError
        write(joinpath(dir, "gnc_disv.gnc"), replace(gnc, "  1  10  1  15  1  9" => "  1  10  1  15  1  999"))
        e = input_error_of(() -> load_simulation(dir))
        @test e isa MF6InputError && occursin("outside the model grid", e.msg)
        # A record between cells that are not connected (gnc_mc).
        write(joinpath(dir, "gnc_disv.gnc"), replace(gnc, "  1  10  1  15  1  9" => "  1  1  1  15  1  9"))
        e = _gnc_error(() -> GwfModel(CPU(), Float64, only(load_simulation(dir).models), 2))
        @test e isa MF6InputError && occursin("GHOST NODE ERROR.  Cell 1 in model GNC_DISV is not connected to cell 15 in model GNC_DISV", e.msg)
        write(joinpath(dir, "gnc_disv.gnc"), gnc)
        # XT3D with GNC (npf_df); NEWTON, convertible cells: unsupported.
        npf = read(joinpath(dir, "gnc_disv.npf"), String)
        write(joinpath(dir, "gnc_disv.npf"), replace(npf, "SAVE_FLOWS" => "SAVE_FLOWS\n  XT3D"))
        e = _gnc_error(() -> GwfModel(CPU(), Float64, only(load_simulation(dir).models), 2))
        @test e isa MF6InputError && occursin("The XT3D option cannot be used with the GNC Package", e.msg)
        write(joinpath(dir, "gnc_disv.npf"), replace(npf, "icelltype\n    CONSTANT  0" => "icelltype\n    CONSTANT  1"))
        e = _gnc_error(() -> GwfModel(CPU(), Float64, only(load_simulation(dir).models), 2))
        @test e isa MF6UnsupportedError && occursin("GNC6 with convertible cells", e.msg)
        write(joinpath(dir, "gnc_disv.npf"), npf)
        _gnc_write(dir, "gnc_disv.nam", t -> replace(t, "SAVE_FLOWS" => "SAVE_FLOWS\n  NEWTON"))
        e = _gnc_error(() -> GwfModel(CPU(), Float64, only(load_simulation(dir).models), 2))
        @test e isa MF6UnsupportedError && occursin("GNC6 with the NEWTON formulation", e.msg)
        _gnc_write(dir, "gnc_disv.nam", t -> replace(t, "\n  NEWTON" => ""))
        # CG with implicit GNC: an asymmetric matrix (sln_ar); EXPLICIT keeps it symmetric.
        _gnc_write(dir, "gnc_disv.ims", t -> replace(t, "BICGSTAB" => "CG"))
        e = _gnc_error(() -> ModflowGPU.initialize(load_simulation(dir); write_output = false))
        @test e isa MF6InputError && occursin("MODEL **GNC_DISV** PRODUCES AN ASYMMETRIC COEFFICIENT MATRIX", e.msg)
        write(joinpath(dir, "gnc_disv.gnc"), replace(gnc, "BEGIN options\n" => "BEGIN options\n  EXPLICIT\n"))
        @test isnothing(_gnc_error(() -> ModflowGPU.initialize(load_simulation(dir); write_output = false)))
        # Exchange: records that do not match the exchange (read_gnc), an inactive cell,
        # EXPLICIT with the exchange's NEWTON, XT3D with GNC (GwfGwfConnection.f90), CG.
        ldir = _gnc_variant(joinpath(GNC_MODELS, "gnc_lgr"), joinpath(tmp, "l"))
        lg = read(joinpath(ldir, "gnclgr.gnc"), String)
        write(joinpath(ldir, "gnclgr.gnc"), replace(lg, "  1 2 3  1 1 1  1 2 2" => "  1 2 4  1 1 1  1 2 2"))
        e = _gnc_error(() -> solution_setup(load_simulation(ldir)))
        @test e isa MF6InputError && occursin("EXCHANGE NODES 10 AND 1 NOT CONSISTENT WITH GNC NODES 11 AND 1", e.msg)
        write(joinpath(ldir, "gnclgr.gnc"), replace(lg, "  1 2 3  1 1 1  1 2 2" => "  1 3 3  1 1 1  1 2 2"))
        e = input_error_of(() -> load_simulation(ldir))
        @test e isa MF6InputError && occursin("PARENT Cell is outside active grid domain: (1,3,3)", e.msg)
        write(joinpath(ldir, "gnclgr.gnc"), replace(lg, "BEGIN options\n" => "BEGIN options\n  EXPLICIT\n"))
        exg = read(joinpath(ldir, "gnclgr.gwfgwf"), String)
        write(joinpath(ldir, "gnclgr.gwfgwf"), replace(exg, "SAVE_FLOWS" => "SAVE_FLOWS\n  NEWTON"))
        e = _gnc_error(() -> solution_setup(load_simulation(ldir)))
        @test e isa MF6InputError && occursin("GNC is explicit, but GWF exchange has active newton", e.msg)
        write(joinpath(ldir, "gnclgr.gnc"), lg)
        write(joinpath(ldir, "gnclgr.gwfgwf"), replace(exg, "SAVE_FLOWS" => "SAVE_FLOWS\n  XT3D"))
        e = _gnc_error(() -> solution_setup(load_simulation(ldir)))
        @test e isa MF6InputError && occursin("Ghost node correction not supported combined with XT3D for exchange GWF-GWF_1", e.msg)
        write(joinpath(ldir, "gnclgr.gwfgwf"), exg)
        _gnc_write(ldir, "gnclgr.ims", t -> replace(t, "BICGSTAB" => "CG"))
        e = _gnc_error(() -> ModflowGPU.initialize(load_simulation(ldir); write_output = false))
        @test e isa MF6InputError && occursin("EXCHANGE **GWF-GWF_1** PRODUCES AN ASYMMETRIC COEFFICIENT MATRIX", e.msg)
    end
end

# The flux terms on the host against a direct port of gnc_fc on MF6's matrix (the
# conductance terms, then per record diag(n) += c, (n, j) −= c, (m, n) −= c, (m, j) += c with
# c = α·cond).
@testset "GNC6: structure and matrix (gnc_ac, gnc_fc)" begin
    for case in ("gnc_disv", "gnc_disu")
        spec = only(load_simulation(joinpath(GNC_MODELS, case)).models)
        conn = build_connectivity(spec)
        setup = npf_setup(spec, conn)
        g = spec.gnc
        ref = Dict{Tuple{Int, Int}, Float64}()
        add(i, j, v) = (ref[(i, j)] = get(ref, (i, j), 0.0) + v)
        for n in 1:conn.nodes, ii in (conn.ia[n] + 1):(conn.ia[n + 1] - 1)
            c = setup.condsat[conn.jas[ii]]
            add(n, conn.ja[ii], c)
            add(n, n, -c)
        end
        for k in eachindex(g.noden)
            g.nodesj[1, k] == 0 && continue
            n, m, j = conn.nodereduced[g.noden[k]], conn.nodereduced[g.nodem[k]], conn.nodereduced[g.nodesj[1, k]]
            c = g.alphasj[1, k] * setup.condsat[conn.jas[getjaindex(conn, n, m)]]
            add(n, n, c), add(n, j, -c), add(m, n, -c), add(m, j, c)
        end
        xs = gnc_setup(spec, conn, setup)
        ours = Dict((n, Int(xs.ja[ii])) => xs.amat[ii] for n in 1:conn.nodes for ii in xs.ia[n]:(xs.ia[n + 1] - 1))
        # gnc_ac: the connectivity plus n–j and m–j, both ways.
        @test Set(keys(ours)) == union(Set(keys(ref)), Set((b, a) for (a, b) in keys(ref)))
        @test maximum(k -> abs(get(ours, k, 0.0) - get(ref, k, 0.0)), keys(ours)) <= 1.0e-14 * maximum(abs, values(ref))
        @test length(xs.ja) > conn.nja
        # EXPLICIT: the connectivity and the conductance matrix (`rhs`).
        ge = GncSpec(; (f => getfield(g, f) for f in fieldnames(GncSpec) if f != :explicit)..., explicit = true)
        se = ModflowGPU.GwfSpec(; (f => getfield(spec, f) for f in fieldnames(ModflowGPU.GwfSpec) if f != :gnc)..., gnc = ge)
        xe = gnc_setup(se, conn, setup)
        @test xe.rhs && xe.ia == conn.ia && xe.ja == conn.ja
    end
end

# libmf6's first-solve system against `formulate!` (a single model or the merged model).
function _gnc_oracle(name, simdir)
    sim = load_simulation(simdir)
    multi = length(sim.models) > 1
    o = _libmf6_first_solve(simdir)
    amax = maximum(abs, o.amat)
    return for_each_precision() do p
        T = p.T
        np = ModflowGPU.nper(sim.tdis)
        m = multi ? GwfModel(TEST_BACKEND, T, solution_setup(sim), np) : GwfModel(TEST_BACKEND, T, only(sim.models), np)
        r = m.setup.r
        mf = Dict{Tuple{Int, Int}, Float64}()
        bmf = copy(o.rhs)
        bscale = abs.(o.rhs)
        for n in 1:m.conn.nodes, ii in o.ia[n]:(o.ia[n + 1] - 1)
            mf[(n, Int(o.ja[ii]))] = o.amat[ii]
            t = o.amat[ii] * r[o.ja[ii]]
            bmf[n] -= t
            bscale[n] += abs(t)
        end
        apply_period!(m, 1; window = time_step_window(time_steps(sim.tdis), 1))
        start_timestep!(m, o.delt)
        formulate!(m)
        KernelAbstractions.synchronize(TEST_BACKEND)
        ia, ja, a = Array(m.scsr.ia), Array(m.scsr.ja), Float64.(Array(m.sys.amat))
        @test ia == o.ia && ja == o.ja
        ours = Dict((n, Int(ja[ii])) => a[ii] for n in 1:m.conn.nodes for ii in ia[n]:(ia[n + 1] - 1))
        ea = maximum(k -> abs(get(ours, k, 0.0) - get(mf, k, 0.0)), union(keys(ours), keys(mf))) / amax
        eb = maximum(abs.(Float64.(Array(m.sys.rhs)) .- bmf)) / maximum(bscale)
        tol = T === Float64 ? 1.0e-12 : 1.0e-6
        @test ea < tol
        @test eb < tol
        @info "GNC oracle vs libmf6 (first solve): $name, :$(p.preset), $(backend_name(TEST_BACKEND)), $(length(ja)) entries (connectivity $(m.conn.nja)): amat max |Δ| / max |a| = $(round(ea; sigdigits = 3)), b' max |Δ| / max(|RHS| + Σ|a·r|) = $(round(eb; sigdigits = 3))"
    end
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
                inputs = _gnc_variant(dir, joinpath(tmp, name * "_inputs"))
                out = read(Cmd(`$GNC_MF6`; dir), String)
                @test occursin("Normal termination", out)
                @testset "$name" begin
                    libmf6_available() ? _gnc_oracle(name, inputs) :
                        @info "Skipping the GNC oracle vs libmf6: libmf6 environment not found at $LIBMF6_ENV"
                    _run_parity(name, dir)
                    # The correction matters: without GNC6 the heads differ by far more than
                    # the parity tolerance.
                    if src == name
                        nognc = _gnc_variant(inputs, joinpath(tmp, name * "_nognc"); edit = _gnc_remove)
                        a = ModflowGPU.run(inputs; backend = TEST_BACKEND, precision = :double, write_output = false)
                        b = ModflowGPU.run(nognc; backend = TEST_BACKEND, precision = :double, write_output = false)
                        d = maximum(abs.(a.heads .- b.heads))
                        @test d > 1.0e-3
                        @info "GNC6 effect: $name, max |Δh| with and without GNC6 = $(round(d; sigdigits = 3)) m"
                    end
                end
            end
        end
    end
end

# MF6Model (T8.10): libmf6's BMI as ModflowInterface.jl wraps it. The script of
# validation/bmi_script.jl (written for ModflowInterface.jl) runs on MF6Model in :single
# and :double and, where the libmf6 environments of validation/setup_libmf6_envs.sh
# exist, through ModflowInterface.jl on libmf6 6.3.0 (Modflow6_jll, as ModflowInterface
# uses) and 6.8.2; every result is compared, except the documented differences below.
# Uses the models of bmi_test.jl (`_bmi_model`).
import Serialization

const BMI = ModflowGPU.BMI   # bmi_script.jl calls `BMI.` (also bound by bmi_test.jl)
include(joinpath(@__DIR__, "..", "..", "validation", "bmi_script.jl"))

# The libmf6 environments: version => Julia environment.
const BMI_LIBMF6_ENVS = ["6.3.0" => joinpath(REFERENCE_DIR, "julia-env", "mf6.3.0"), "6.8.2" => LIBMF6_ENV]

# Run validation/libmf6_bmi.jl with the libmf6 environment `env`: the script's results.
function _libmf6_bmi(env, simdir, tmp)
    outdir = joinpath(tmp, "out")
    path = joinpath(@__DIR__, "..", "..", "validation", "libmf6_bmi.jl")
    e = copy(ENV)
    foreach(k -> delete!(e, k), ("JULIA_LOAD_PATH", "JULIA_PROJECT", "JULIA_NUM_THREADS"))
    julia = first(Base.julia_cmd().exec)
    out = IOBuffer()
    cmd = setenv(`$julia --startup-file=no --project=$env $path $simdir $outdir`, e)
    success(pipeline(cmd; stdout = out, stderr = out)) || error("libmf6 subprocess failed: $cmd\n" * String(take!(out)))
    return Serialization.deserialize(joinpath(outdir, "result.jls"))
end

# The calls for which libmf6 reports a failure status (ModflowInterface.jl then returns 0
# or an empty array) and MF6Model throws, per grid type (`face_nodes` is not called then).
const BMI_LIBMF6_FAILURES = Dict(
    "DIS" => ["/grid/node_count", "/grid/face_count", "/grid/nodes_per_face", "/grid/face_nodes"],
    "DISV" => ["/grid/rank", "/grid/size", "/grid/node_count", "/grid/face_count", "/grid/nodes_per_face", "/grid/face_nodes"],
    "DISU" => ["/grid/rank"],
)

# Results that differ by design (docs/src/coupling/bmi.md):
# - ModflowInterface.jl's get_time_step calls libmf6's get_end_time (a bug);
# - before the first time step, libmf6's XOLD and SIMVALS are not initialised, TDIS/DELT
#   is 0 (ours: the first step length), and the NODEUSER/NODEREDUCED dummies of length 1
#   hold arbitrary values (ours: 0);
# - DIS y: libmf6 writes ncol:-1:0 (a bug; ours nrow:-1:0, checked separately);
# - an unknown address: libmf6 returns "" with a failure status, MF6Model throws;
# - libmf6 6.8 keeps the WEL rates in "WEL/Q", not in BOUND (which stays 0), so a write
#   into BOUND has no effect there: the steps after the write are compared with 6.3.0 only.
function _bmi_mf6_skip(path, version, gridtype)
    path in ("/times0", "/unknown_var_type") && return true
    occursin(r"/steps\[\d\]/delt$", path) && return true
    occursin(r"^/vars/BMI/(XOLD|CHD/SIMVALS|WEL/SIMVALS)/(ptr|value|value_dest)$", path) && return true
    occursin(r"^/vars/TDIS/DELT/(ptr|value|value_dest)$", path) && return true
    occursin(r"^/vars/BMI/DIS/NODE(USER|REDUCED)/(ptr|value|value_dest)$", path) && return true
    gridtype == "DIS" && path == "/grid/y" && return true
    version != "6.3.0" && occursin(r"^/steps\[[234]\]/", path) && return true
    version != "6.3.0" && occursin(r"^/steps\[1\]/wel_bound", path) && return true
    return false
end

# Compare the script results `a` (MF6Model) and `b` (libmf6): the paths that differ.
function _bmi_mf6_diff!(bad, a, b, path, version, gridtype, tol)
    _bmi_mf6_skip(path, version, gridtype) && return bad
    if a isa BmiScriptError
        path in BMI_LIBMF6_FAILURES[gridtype] || push!(bad, "$path: $(a.msg)")
    elseif isnothing(a) && path in BMI_LIBMF6_FAILURES[gridtype]
        # not called (face_nodes after a failed face count)
    elseif a isa AbstractDict && b isa AbstractDict
        keys(a) == keys(b) || push!(bad, "$path: keys")
        for k in keys(a)
            haskey(b, k) && _bmi_mf6_diff!(bad, a[k], b[k], "$path/$k", version, gridtype, tol)
        end
    elseif a isa AbstractVector && b isa AbstractVector && !isempty(a) && first(a) isa AbstractDict
        length(a) == length(b) || push!(bad, "$path: length")
        for i in eachindex(a, b)
            _bmi_mf6_diff!(bad, a[i], b[i], "$path[$i]", version, gridtype, tol)
        end
    elseif a isa AbstractArray{<:AbstractFloat} && b isa AbstractArray{<:AbstractFloat}
        (typeof(a) == typeof(b) && size(a) == size(b) && isapprox(a, b; atol = tol, rtol = tol)) || push!(bad, path)
    elseif a isa Tuple{Vararg{Float64}} && b isa Tuple{Vararg{Float64}}
        (length(a) == length(b) && all(isapprox.(a, b; atol = tol))) || push!(bad, path)
    elseif a isa AbstractFloat && b isa AbstractFloat
        isapprox(a, b; atol = tol, rtol = tol) || push!(bad, path)
    else
        isequal(a, b) || push!(bad, path)
    end
    return bad
end

@testset "MF6Model: libmf6's BMI (T8.10)" begin
    base = _synthetic_coupling_dis()   # 7 × 5 × 2, IDOMAIN 0 cells: nodes < nodesuser
    act = findall(>(0), vec(base.idomain))
    chd, wel = act[[1, 2]], act[20]
    opts = "  XORIGIN  1000.0\n  YORIGIN  2000.0\n  ANGROT  30.0"
    mktempdir() do tmp
        one = _dis(; delr = [100.0, 150.0, 100.0, 120.0], delc = [50.0, 80.0, 60.0], top = fill(200.0, 4, 3), botm = fill(0.0, 4, 3, 1))
        models = [
            "DIS" => _bmi_model(joinpath(tmp, "dis"), base; chd, wel, options = opts),
            "DIS" => _bmi_model(joinpath(tmp, "dis1"), one; chd = [1, 12], wel = 6, options = "  XORIGIN  1000.0\n  YORIGIN  2000.0"),
            "DISV" => _bmi_model(joinpath(tmp, "disv"), _dis_as_disv(base); chd, wel, options = opts),
            "DISU" => _bmi_model(joinpath(tmp, "disu"), _dis_as_disu(base; vertices = true); chd, wel, options = opts),
        ]
        for (gridtype, dir) in models
            @testset "$(basename(dir))" begin
                lib = Dict{String, Any}()
                for (version, env) in BMI_LIBMF6_ENVS
                    if isfile(joinpath(env, ".setup-complete"))
                        lib[version] = mktempdir(t -> _libmf6_bmi(env, dir, t))
                    else
                        @info "Skipping the comparison with libmf6 $version: no environment $env (validation/setup_libmf6_envs.sh)"
                    end
                end
                d = only(load_simulation(dir).models).dis
                for_each_precision() do p
                    r = bmi_script(ModflowGPU, MF6Model, dir; backend = TEST_BACKEND, precision = p.preset, write_output = false)
                    # ModflowGPU's results, with or without libmf6.
                    @test r["initialize_returns_model"] && r["reinitialize_returns_model"]
                    @test r["update_returns_model"] && r["finalize_returns_model"]
                    @test r["component_name"] == "MODFLOW 6"
                    @test r["times0"] == (0.0, 0.0, 5.0, 1.0) && r["times_reinit"] == (0.0, 5.0) && r["time_update"] == 1.0
                    @test r["output_names_subset"] && r["input_equals_output"] && r["counts_match"]
                    @test r["unknown_var_type"] isa BmiScriptError && occursin("unknown address", r["unknown_var_type"].msg)
                    @test r["grid_id"] == 1 && r["subcomponent_count"] == 1
                    @test r["var_address"] == ("BMI/X", "BMI/WEL/BOUND", "SLN_1/MXITER")
                    @test r["last_error_type"] == "String"
                    steps = r["steps"]
                    @test [s["time"] for s in steps] == [1.0, 2.0, 3.0, 5.0] == [s["totim_ptr"] for s in steps]
                    @test [s["delt"] for s in steps] == [1.0, 1.0, 1.0, 2.0]
                    @test [(s["kper"], s["kstp"]) for s in steps] == [(1, 1), (2, 1), (2, 2), (3, 1)]
                    @test all(s -> s["returns_model"] && s["converged"], steps)
                    @test [only(s["wel_simvals"]) for s in steps] ≈ [-50.0, -80.0, -80.0, -80.0] rtol = 1.0e-4
                    @test [only(s["wel_bound"]) for s in steps] == [-50.0, -80.0, -80.0, -80.0]
                    @test all(==(95.0), r["x_reinit"]) && r["x_update"][1] == 100.0
                    v = r["vars"]
                    nodes = length(v["BMI/X"]["ptr"])
                    ok = Dict{String, Bool}()
                    for a in BMI_SCRIPT_ADDRESSES
                        x = v[a]
                        ptr = x["ptr"]
                        t = eltype(ptr) == Float64 ? "DOUBLE" : "INTEGER"
                        ok[a] = x["grid"] == (startswith(a, "BMI/") ? 1 : -1) && x["rank"] == ndims(ptr) && x["shape"] == size(ptr) &&
                            x["nbytes"] == sizeof(ptr) && x["itemsize"] == sizeof(eltype(ptr)) &&
                            x["type"] == (ndims(ptr) == 0 ? t : "$t ($(join(size(ptr), ',')))") &&
                            isequal(x["value"], ptr) && isequal(x["value_dest"], ptr)
                    end
                    @test all(values(ok))
                    @test ndims(v["TDIS/TOTIM"]["ptr"]) == 0 && ndims(v["BMI/DIS/NODES"]["ptr"]) == 0
                    @test size(v["BMI/CHD/BOUND"]["ptr"]) == (1, 2) && v["BMI/CHD/BOUND"]["type"] == "DOUBLE (1,2)"
                    @test v["BMI/X"]["type"] == "DOUBLE ($nodes)" && v["SLN_1/MXITER"]["type"] == "INTEGER"
                    nu = v["BMI/DIS/NODESUSER"]["ptr"][]
                    @test length(v["BMI/DIS/NODEUSER"]["ptr"]) == (nodes == nu ? 1 : nodes)
                    g = r["grid"]
                    if gridtype == "DIS"
                        @test g["type"] == "rectilinear" && g["rank"] == (d.nlay == 1 ? 2 : 3) && g["size"] == d.nlay * d.nrow * d.ncol
                        @test g["shape"] == (d.nlay == 1 ? (d.ncol, d.nrow) : (d.ncol, d.nrow, d.nlay))
                        @test g["x"] == 0:d.ncol && g["y"] == d.nrow:-1:0
                    else
                        @test g["type"] == "unstructured" && g["x"] == d.vertices[1, :] && g["y"] == d.vertices[2, :]
                    end
                    if gridtype == "DISU"
                        @test g["size"] == g["node_count"] == d.nvert && g["face_count"] == nodes
                        @test g["nodes_per_face"] == length.(d.cell_vertices)
                        @test isnothing(g["face_nodes"])   # not called: face count (nodes) < user nodes
                    end
                    for (k, err) in g
                        err isa BmiScriptError || continue
                        @test "/grid/$k" in BMI_LIBMF6_FAILURES[gridtype] && occursin("libmf6 returns a failure status", err.msg)
                    end
                    # libmf6's results.
                    tol = p.T == Float64 ? 1.0e-6 : 1.0e-3
                    for (version, rl) in lib
                        bad = _bmi_mf6_diff!(String[], r, rl, "", version, gridtype, tol)
                        @test isempty(bad)
                        isempty(bad) || @info "MF6Model results that differ from libmf6 $version ($(basename(dir))): $bad"
                        gridtype == "DIS" && @test rl["grid"]["y"] == d.ncol:-1:0   # libmf6's bug, see above
                        # The comparison sees a difference deep in the results.
                        rx = deepcopy(rl)
                        rx["steps"][1]["x"][end] += 1.0
                        @test _bmi_mf6_diff!(String[], r, rx, "", version, gridtype, tol) == ["/steps[1]/x"]
                    end
                end
            end
        end
    end
end

# An initialised MF6Model that is not referenced after the call: its simulation.
@noinline function _mf6model_dropped(dir)
    m = MF6Model(dir; backend = TEST_BACKEND, write_output = false)
    BMI.initialize(m)
    BMI.update(m)
    return m.simulation
end

@testset "MF6Model: handle, XMI calls and errors (T8.10)" begin
    mktempdir() do tmp
        one = _dis(; delr = [100.0, 150.0, 100.0, 120.0], delc = [50.0, 80.0, 60.0], top = fill(200.0, 4, 3), botm = fill(0.0, 4, 3, 1))
        dir = _bmi_model(joinpath(tmp, "dis1"), one; chd = [1, 12], wel = 6)
        @test_throws r"not a simulation directory" MF6Model(joinpath(tmp, "nowhere"))
        @test_throws r"unknown keyword `precison`" MF6Model(dir; precison = :single)
        for_each_precision() do p
            m = MF6Model(joinpath(dir, "mfsim.nam"); backend = TEST_BACKEND, precision = p.preset, write_output = false)
            @test sprint(show, m) == "MF6Model(\"dis1\", uninitialized)"
            @test_throws r"not initialized" BMI.get_current_time(m)
            @test occursin("not initialized", ModflowGPU.get_last_bmi_error(m))
            @test BMI.finalize(m) === m   # nothing to do
            @test BMI.initialize(m) === m && m.simulation isa Simulation && m.simulation.T == p.T
            @test sprint(show, m) == "MF6Model(\"dis1\", initialized)"
            # The full XMI step, with do_time_step, and set_value.
            @test prepare_time_step(m, 0.0) === m
            @test BMI.set_value(m, "BMI/WEL/BOUND", [-70.0]) === nothing
            @test do_time_step(m) === m && finalize_time_step(m) === m
            @test only(BMI.get_value(m, "BMI/WEL/SIMVALS")) ≈ -70.0 rtol = 1.0e-4
            @test ModflowGPU.get_var_rank(m, "TDIS/KPER") == 0 && ModflowGPU.get_var_shape(m, "BMI/CHD/BOUND") == (1, 2)
            @test BMI.get_grid_shape(m, 1, zeros(Int, 2)) == [3, 4]   # libmf6's C order
            @test BMI.get_grid_x(m, 1, zeros(5)) == 0:4
            # ncol (4) > nrow (3): get_grid_y gives nrow + 1 values (libmf6 6.8.2 writes
            # ncol + 1), and a short destination is an error, never written past its end.
            @test BMI.get_grid_y(m, 1, fill(-1.0, 4)) == 3:-1:0
            ybuf = fill(-1.0, 6)
            @test BMI.get_grid_y(m, 1, ybuf) === ybuf && ybuf == [3, 2, 1, 0, -1, -1]
            xbuf = zeros(2)
            @test_throws r"length 2, expected at least 5" BMI.get_grid_x(m, 1, xbuf)
            @test xbuf == [0, 0]
            @test_throws ArgumentError BMI.get_grid_y(m, 1, zeros(3))
            @test_throws ArgumentError BMI.get_grid_shape(m, 1, zeros(Int, 1))
            @test_throws r"no model for grid id 2" BMI.get_grid_type(m, 2)
            @test_throws r"solution 2" ModflowGPU.solve(m, 2)
            # Functions libmf6 does not implement.
            for f in (
                    () -> BMI.update_until(m, 5.0), () -> BMI.get_var_units(m, "BMI/X"), () -> BMI.get_var_location(m, "BMI/X"),
                    () -> BMI.get_time_units(m), () -> BMI.get_grid_z(m, 1, zeros(1)), () -> BMI.get_grid_edge_count(m, 1),
                    () -> BMI.get_value_at_indices(m, "BMI/X", zeros(1), [1]),
                )
                @test_throws r"not implemented by libmf6.*Simulation" f()
            end
            @test_throws r"not supported" BMI.initialize(MF6Model, dir)
            # Re-initialise: back to the start.
            @test BMI.initialize(m) === m && BMI.get_current_time(m) == 0.0
            @test BMI.update(m) === m && BMI.get_current_time(m) == 1.0
            @test BMI.finalize(m) === m && isnothing(m.simulation)
            # BMI.initialize(MF6Model): the simulation in the current directory (CPU).
            m2 = cd(() -> BMI.initialize(MF6Model), dir)
            @test m2 isa MF6Model && BMI.get_end_time(m2) == 5.0
            BMI.finalize(m2)
        end
        # A SimulationSpec, initialised twice from the same spec.
        m = MF6Model(load_simulation(dir); backend = TEST_BACKEND, write_output = false)
        BMI.initialize(m)
        BMI.update(m)
        BMI.initialize(m)
        @test BMI.get_current_time(m) == 0.0
        BMI.finalize(m)
        # The finalizer (as ModflowInterface.jl's ModflowModel): an initialised model that
        # goes out of scope is finalized in a task after garbage collection.
        sim = _mf6model_dropped(dir)
        @test sim.stage !== :freed
        @test timedwait(() -> (GC.gc(true); sim.stage === :freed), 60.0; pollint = 0.5) === :ok
        m = MF6Model(dir; backend = TEST_BACKEND, write_output = false)
        BMI.initialize(m)
        BMI.finalize(m)
        @test BMI.finalize(m) === m   # idempotent
        finalize(m)                   # Base.finalize runs the finalizer: nothing left to do
        @test isnothing(m.simulation)
    end
end

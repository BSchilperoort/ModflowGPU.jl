# BMI conformance (T8.9): every one of the 41 functions of BasicModelInterface.jl, called
# with the argument types the specification gives, on DIS, DISV and DISU models (and a
# DISU model without vertices): return types, in-place getters return the destination,
# sizes (nbytes = itemsize · length), grid/variable consistency (the length of a "node"
# variable is the size of its grid, of a "face" variable the face count), coordinates
# (XORIGIN, YORIGIN, ANGROT), unstructured connectivity, time and `update_until`, the
# configuration file, and the documented error of unsupported functions.
const BMI = ModflowGPU.BMI  # BasicModelInterface, through the package (not a test dependency)
using ModflowGPU: Simulation

# The 41 functions of the specification.
const BMI_FUNCTIONS = Set(
    filter(n -> !startswith(string(n), "#") && getproperty(BMI, n) isa Function && !(n in (:eval, :include)), names(BMI; all = true)),
)

const BMI_GRID_TYPES = ("scalar", "points", "vector", "unstructured", "structured_quadrilateral", "rectilinear", "uniform_rectilinear")

# The MF6 cellid of user node `n` of grid `d`.
function _bmi_cellid(d, n)
    d isa ModflowGPU.DisuSpec && return string(n)
    ncpl = d isa ModflowGPU.DisSpec ? Int(d.nrow * d.ncol) : Int(d.ncpl)
    k, c = divrem(n - 1, ncpl) .+ (1, 1)
    d isa ModflowGPU.DisvSpec && return "$k $c"
    i, j = divrem(c - 1, Int(d.ncol)) .+ (1, 1)
    return "$k $i $j"
end

# A model on grid `d` (`write_dis_simulation`) with CHD at user nodes `chd` (heads 100 and
# 90) and WEL at `wel` (−50), three periods (1, 2, 2 days; 1, 2, 1 steps: step ends 1, 2,
# 3, 5), steady state, and the grid options `options` (XORIGIN, ...).
function _bmi_model(dir, d; chd, wel, options = "")
    write_dis_simulation(dir, d; name = "bmi", npf_arrays = Dict{String, Any}("icelltype" => 0, "k" => 10.0), strt = 95.0)
    dis = joinpath(dir, "bmi.dis")
    write(dis, replace(read(dis, String), "LENGTH_UNITS  meters" => "LENGTH_UNITS  meters\n$options"; count = 1))
    write(
        joinpath(dir, "sim.tdis"), """
        BEGIN options
          TIME_UNITS  days
        END options
        BEGIN dimensions
          NPER  3
        END dimensions
        BEGIN perioddata
          1.0  1  1.0
          2.0  2  1.0
          2.0  1  1.0
        END perioddata
        """,
    )
    nam = joinpath(dir, "bmi.nam")
    write(nam, replace(read(nam, String), "END packages" => "  CHD6  bmi.chd  chd\n  WEL6  bmi.wel  wel\nEND packages"))
    rec(nodes, vals) = join(("  $(_bmi_cellid(d, n))  $v" for (n, v) in zip(nodes, vals)), "\n")
    write(joinpath(dir, "bmi.chd"), "BEGIN dimensions\n  MAXBOUND  $(length(chd))\nEND dimensions\nBEGIN period  1\n$(rec(chd, [100.0, 90.0]))\nEND period  1\n")
    write(joinpath(dir, "bmi.wel"), "BEGIN dimensions\n  MAXBOUND  1\nEND dimensions\nBEGIN period  1\n$(rec([wel], [-50.0]))\nEND period  1\n")
    return dir
end

# A BMI configuration file `cfg` for the simulation in `simdir` (relative path).
function _bmi_config(cfg, simdir, preset; extra = "")
    mkpath(dirname(cfg))
    write(
        cfg, """
        simulation = "$(relpath(simdir, dirname(cfg)))"
        backend = "$TEST_BACKEND_NAME"
        precision = "$preset"
        write_output = false
        $extra
        """,
    )
    return cfg
end

# `f()` throws the documented error of an unsupported BMI function `name`.
function _bmi_unsupported(f, name)
    try
        f()
    catch e
        return e isa ArgumentError && occursin("BMI.$name is not supported", e.msg)
    end
    return false
end

const BMI_ELTYPES = Dict("Float64" => Float64, "Int32" => Int32)

# Call every grid function on every grid; records the calls in `called`.
function _bmi_check_grids!(called, s, ngrid)
    for g in 0:(ngrid - 1)
        type = BMI.get_grid_type(s, g)
        rank = BMI.get_grid_rank(s, g)
        n = BMI.get_grid_size(s, g)
        @test type isa String && type in BMI_GRID_TYPES
        @test rank isa Int && n isa Int
        @test BMI.get_grid_node_count(s, g) === n
        push!(called, :get_grid_type, :get_grid_rank, :get_grid_size, :get_grid_node_count)
        if type in ("rectilinear", "structured_quadrilateral", "vector", "scalar")
            shape = zeros(Int, rank)
            @test BMI.get_grid_shape(s, g, shape) === shape
            @test prod(shape) == n
        else
            @test _bmi_unsupported(() -> BMI.get_grid_shape(s, g, zeros(Int, 3)), :get_grid_shape)
        end
        push!(called, :get_grid_shape)
        @test _bmi_unsupported(() -> BMI.get_grid_spacing(s, g, zeros(rank)), :get_grid_spacing)
        @test _bmi_unsupported(() -> BMI.get_grid_origin(s, g, zeros(rank)), :get_grid_origin)
        push!(called, :get_grid_spacing, :get_grid_origin)
        # Coordinates: per axis the length the spec gives, or the documented error.
        for (k, f) in enumerate((BMI.get_grid_x, BMI.get_grid_y, BMI.get_grid_z))
            push!(called, nameof(f))
            len = type == "rectilinear" ? (k <= rank ? shape[end - k + 1] : 0) : n
            has = type in ("rectilinear", "structured_quadrilateral", "unstructured", "points") && k <= rank
            if has
                c = zeros(len)
                @test f(s, g, c) === c
                @test all(isfinite, c)
            else
                @test _bmi_unsupported(() -> f(s, g, zeros(max(len, 1))), nameof(f))
            end
        end
        fnames = (:get_grid_face_count, :get_grid_edge_count, :get_grid_edge_nodes, :get_grid_face_edges, :get_grid_face_nodes, :get_grid_nodes_per_face)
        push!(called, fnames...)
        if type == "unstructured"
            nf = BMI.get_grid_face_count(s, g)
            ne = BMI.get_grid_edge_count(s, g)
            @test nf isa Int && ne isa Int
            npf = zeros(Int, nf)
            @test BMI.get_grid_nodes_per_face(s, g, npf) === npf
            fn = zeros(Int, sum(npf))
            fe = zeros(Int, sum(npf))
            en = zeros(Int, 2ne)
            @test BMI.get_grid_face_nodes(s, g, fn) === fn
            @test BMI.get_grid_face_edges(s, g, fe) === fe
            @test BMI.get_grid_edge_nodes(s, g, en) === en
            @test all(in(1:n), fn) && all(in(1:n), en) && all(in(1:ne), fe)
            # Faces counter-clockwise (positive area), their edges join consecutive nodes.
            x, y = BMI.get_grid_x(s, g, zeros(n)), BMI.get_grid_y(s, g, zeros(n))
            ok_area = ok_edge = true
            off = 0
            for k in 1:nf
                v = fn[(off + 1):(off + npf[k])]
                e = fe[(off + 1):(off + npf[k])]
                a = sum(x[v[i]] * y[v[mod1(i + 1, end)]] - x[v[mod1(i + 1, end)]] * y[v[i]] for i in eachindex(v))
                ok_area &= a > 0
                for i in eachindex(v)
                    ok_edge &= minmax(en[2e[i] - 1], en[2e[i]]) == minmax(v[i], v[mod1(i + 1, end)])
                end
                off += npf[k]
            end
            @test ok_area
            @test ok_edge
        else
            @test _bmi_unsupported(() -> BMI.get_grid_face_count(s, g), :get_grid_face_count)
            @test _bmi_unsupported(() -> BMI.get_grid_edge_count(s, g), :get_grid_edge_count)
            @test _bmi_unsupported(() -> BMI.get_grid_edge_nodes(s, g, zeros(Int, 2)), :get_grid_edge_nodes)
            @test _bmi_unsupported(() -> BMI.get_grid_face_edges(s, g, zeros(Int, 2)), :get_grid_face_edges)
            @test _bmi_unsupported(() -> BMI.get_grid_face_nodes(s, g, zeros(Int, 2)), :get_grid_face_nodes)
            @test _bmi_unsupported(() -> BMI.get_grid_nodes_per_face(s, g, zeros(Int, 2)), :get_grid_nodes_per_face)
        end
    end
    @test_throws ArgumentError BMI.get_grid_type(s, ngrid)
    return nothing
end

# Call every variable function on every variable; records the calls in `called`.
function _bmi_check_variables!(called, s, mname)
    inp = BMI.get_input_var_names(s)
    out = BMI.get_output_var_names(s)
    @test inp isa Vector{String} && out isa Vector{String} && issubset(inp, out)
    @test BMI.get_input_item_count(s) === length(inp) && BMI.get_output_item_count(s) === length(out)
    push!(called, :get_input_var_names, :get_output_var_names, :get_input_item_count, :get_output_item_count)
    ok = Dict{String, Bool}()
    for name in out
        g = BMI.get_var_grid(s, name)
        type = BMI.get_var_type(s, name)
        units = BMI.get_var_units(s, name)
        isz = BMI.get_var_itemsize(s, name)
        nb = BMI.get_var_nbytes(s, name)
        loc = BMI.get_var_location(s, name)
        T = BMI_ELTYPES[type]
        ptr = BMI.get_value_ptr(s, name)
        n = length(ptr)
        dest = zeros(T, n)
        val = BMI.get_value(s, name, dest)
        di = zeros(T, 2)
        vi = BMI.get_value_at_indices(s, name, di, [n, 1])
        expected_len = loc == "face" ? BMI.get_grid_face_count(s, g) : BMI.get_grid_size(s, g)
        ok[name] = g isa Int && units isa String && !isempty(units) && isz === sizeof(T) && nb === isz * n &&
            loc in ("node", "face") && ptr isa DenseVector{T} && val === dest && val == ptr && vi === di &&
            di == ptr[[n, 1]] && n == expected_len
    end
    push!(called, :get_var_grid, :get_var_type, :get_var_units, :get_var_itemsize, :get_var_nbytes, :get_var_location)
    push!(called, :get_value_ptr, :get_value, :get_value_at_indices)
    bad = [k for (k, v) in ok if !v]
    @test isempty(bad)
    isempty(bad) || @info "BMI variables that fail the checks: $bad"
    @test BMI.get_var_units(s, "$mname/X") == "m"
    @test BMI.get_var_units(s, "$mname/DIS/AREA") == "m2"
    @test BMI.get_var_units(s, "$mname/WEL/SIMVALS") == "m3 d-1"
    @test BMI.get_var_units(s, "$mname/WEL/BOUND") == "m3 d-1"
    @test BMI.get_var_units(s, "$mname/CHD/BOUND") == "m"
    @test BMI.get_var_units(s, "TDIS/DELT") == "d"
    @test BMI.get_var_units(s, "TDIS/KPER") == "1"
    @test BMI.get_var_grid(s, "TDIS/TOTIM") == 0 && BMI.get_var_grid(s, "$mname/WEL/NBOUND") == 0
    @test_throws DimensionMismatch BMI.get_value(s, "$mname/X", zeros(1))
    return nothing
end

# The whole conformance check of one simulation `s` (`mname`: model name), with `ngrid`
# grids. Leaves the simulation finalized.
function _bmi_conformance(s, mname, ngrid)
    called = Set{Symbol}([:initialize])
    @test s isa Simulation
    @test BMI.get_component_name(s) isa String
    push!(called, :get_component_name)
    # Time: before the first step.
    @test BMI.get_start_time(s) === 0.0 && BMI.get_current_time(s) === 0.0
    @test BMI.get_end_time(s) === 5.0 && BMI.get_time_step(s) === 1.0
    @test BMI.get_time_units(s) == "days"
    push!(called, :get_start_time, :get_current_time, :get_end_time, :get_time_step, :get_time_units)
    _bmi_check_variables!(called, s, mname)
    _bmi_check_grids!(called, s, ngrid)
    # A time step, then setters (period 1 data replace values set before it; periods 2
    # and 3 have none, so set values are kept), then the next step.
    heads = BMI.get_value_ptr(s, "$mname/X")
    @test BMI.update(s) === nothing
    push!(called, :update)
    @test BMI.get_current_time(s) == 1.0
    @test only(BMI.get_value(s, "$mname/WEL/SIMVALS", zeros(1))) ≈ -50.0 rtol = 1.0e-5
    @test heads == BMI.get_value(s, "$mname/X", zeros(length(heads)))   # the pointer follows the state
    @test BMI.set_value(s, "$mname/WEL/BOUND", [-80.0]) === nothing
    @test BMI.set_value_at_indices(s, "$mname/CHD/BOUND", [2], [92.0]) === nothing
    push!(called, :set_value, :set_value_at_indices)
    @test BMI.get_value_at_indices(s, "$mname/CHD/BOUND", zeros(2), [1, 2]) == [100.0, 92.0]
    @test_throws DimensionMismatch BMI.set_value_at_indices(s, "$mname/CHD/BOUND", [1, 2], [92.0])
    @test BMI.update_until(s, 2.0) === nothing
    push!(called, :update_until)
    @test BMI.get_current_time(s) == 2.0 && BMI.get_time_step(s) == 1.0
    @test only(BMI.get_value(s, "$mname/WEL/SIMVALS", zeros(1))) ≈ -80.0 rtol = 1.0e-5
    @test heads == BMI.get_value(s, "$mname/X", zeros(length(heads)))
    @test BMI.update_until(s, 2.0) === nothing && BMI.get_current_time(s) == 2.0   # no-op
    @test_throws r"not the end of a time step" BMI.update_until(s, 2.5)
    @test_throws r"before the current time" BMI.update_until(s, 1.0)
    @test_throws r"after the end time" BMI.update_until(s, 6.0)
    @test BMI.get_current_time(s) == 2.0
    BMI.update_until(s, 5.0 * (1 + 1.0e-13))   # within the tolerance of the last step end
    @test BMI.get_current_time(s) == 5.0 && BMI.get_time_step(s) == 2.0
    @test BMI.finalize(s) === nothing
    push!(called, :finalize)
    @test_throws ArgumentError BMI.get_value(s, "$mname/X", zeros(length(heads)))
    @test called == BMI_FUNCTIONS
    return nothing
end

@testset "BMI conformance (T8.9)" begin
    @test length(BMI_FUNCTIONS) == 41
    base = _synthetic_coupling_dis()   # 7 × 5 × 2, IDOMAIN 0 cells: nodes < nodesuser
    act = findall(>(0), vec(base.idomain))
    chd, wel = act[[1, 2]], act[20]
    opts = "  XORIGIN  1000.0\n  YORIGIN  2000.0\n  ANGROT  30.0"
    mktempdir() do tmp
        models = [
            "dis" => _bmi_model(joinpath(tmp, "dis"), base; chd, wel, options = opts),
            "disv" => _bmi_model(joinpath(tmp, "disv"), _dis_as_disv(base); chd, wel, options = opts),
            "disu" => _bmi_model(joinpath(tmp, "disu"), _dis_as_disu(base; vertices = true); chd, wel, options = opts),
        ]
        one = _dis(; delr = [100.0, 150.0, 100.0, 120.0], delc = [50.0, 80.0, 60.0], top = fill(200.0, 4, 3), botm = fill(0.0, 4, 3, 1))
        push!(models, "dis1" => _bmi_model(joinpath(tmp, "dis1"), one; chd = [1, 12], wel = 6, options = "  XORIGIN  1000.0\n  YORIGIN  2000.0"))
        for (label, dir) in models
            @testset "$label" begin
                for_each_precision() do p
                    cfg = _bmi_config(joinpath(tmp, "cfg_$label", "run.toml"), dir, p.preset)
                    s = BMI.initialize(Simulation, cfg)
                    @test s.T == p.T
                    nodes, nodesuser = Int(s.model.conn.nodes), Int(s.model.conn.nodesuser)
                    xgrid = BMI.get_var_grid(s, "BMI/X")
                    @test xgrid == (nodes == nodesuser ? 1 : 2)
                    nodeuser = Int.(BMI.get_value_ptr(s, "BMI/DIS/NODEUSER"))
                    d = s.model.spec.dis
                    if label == "dis"
                        # Rotated 3-layer DIS: structured_quadrilateral, nodes at the cell centres.
                        @test BMI.get_grid_type(s, 1) == "structured_quadrilateral" && BMI.get_grid_rank(s, 1) == 3
                        @test BMI.get_grid_shape(s, 1, zeros(Int, 3)) == [2, 5, 7]
                        x = BMI.get_grid_x(s, 1, zeros(nodesuser))
                        y = BMI.get_grid_y(s, 1, zeros(nodesuser))
                        z = BMI.get_grid_z(s, 1, zeros(nodesuser))
                        xm, ym = base.delr[1] / 2, sum(base.delc) - base.delc[1] / 2
                        @test x[1] ≈ 1000.0 + xm * cosd(30) - ym * sind(30)
                        @test y[1] ≈ 2000.0 + xm * sind(30) + ym * cosd(30)
                        @test z[1] ≈ (base.top[1, 1] + base.botm[1, 1, 1]) / 2
                        @test z[36] ≈ (base.botm[1, 1, 1] + base.botm[1, 1, 2]) / 2
                        @test BMI.get_grid_type(s, 2) == "points"
                        @test BMI.get_grid_x(s, 2, zeros(nodes)) == x[nodeuser]
                        @test BMI.get_var_location(s, "BMI/X") == "node"
                    elseif label == "dis1"
                        # One layer, every cell active, no rotation: rectilinear, X on grid 1.
                        @test BMI.get_grid_type(s, 1) == "rectilinear" && BMI.get_grid_rank(s, 1) == 2
                        @test BMI.get_grid_shape(s, 1, zeros(Int, 2)) == [3, 4]
                        @test BMI.get_grid_x(s, 1, zeros(4)) == 1000.0 .+ [50.0, 175.0, 300.0, 410.0]
                        @test BMI.get_grid_y(s, 1, zeros(3)) == 2000.0 .+ [165.0, 100.0, 30.0]
                        @test _bmi_unsupported(() -> BMI.get_grid_z(s, 1, zeros(1)), :get_grid_z)
                        @test BMI.get_var_location(s, "BMI/X") == "node"
                    else
                        # DISV, DISU with vertices: unstructured, the faces are the cells.
                        @test BMI.get_grid_type(s, 1) == "unstructured" && BMI.get_grid_rank(s, 1) == 2
                        @test BMI.get_grid_size(s, 1) == d.nvert == size(d.vertices, 2)
                        @test BMI.get_grid_face_count(s, 1) == nodesuser
                        @test BMI.get_grid_edge_count(s, 1) == 5 * 8 + 7 * 6   # the edges of the 7 × 5 cells
                        x = BMI.get_grid_x(s, 1, zeros(d.nvert))
                        @test x[1] ≈ 1000.0 + d.vertices[1, 1] * cosd(30) - d.vertices[2, 1] * sind(30)
                        @test BMI.get_var_location(s, "BMI/DIS/NODEREDUCED") == "face"
                        @test BMI.get_var_location(s, "BMI/X") == "node"   # on the points grid 2
                        @test BMI.get_grid_type(s, 2) == "points"
                    end
                    _bmi_conformance(s, "BMI", 3 + 2 * 2)
                end
            end
        end

        @testset "DISU without vertices (test/models/disu_staggered)" begin
            stg = joinpath(@__DIR__, "..", "models", "disu_staggered")
            for_each_precision() do p
                s = BMI.initialize(Simulation, _bmi_config(joinpath(tmp, "cfg_stg", "run.toml"), stg, p.preset))
                @test BMI.get_grid_type(s, 1) == "vector" && BMI.get_grid_type(s, 2) == "vector"
                @test BMI.get_grid_size(s, 1) == 5 && BMI.get_var_grid(s, "STG/X") == 1
                @test _bmi_unsupported(() -> BMI.get_grid_x(s, 1, zeros(5)), :get_grid_x)
                @test _bmi_unsupported(() -> BMI.get_grid_face_count(s, 1), :get_grid_face_count)
                called = Set{Symbol}()
                _bmi_check_variables!(called, s, "STG")
                BMI.update_until(s, BMI.get_end_time(s))
                @test BMI.get_current_time(s) == 1.0
                BMI.finalize(s)
            end
        end

        @testset "configuration file" begin
            dir = last(models[end])
            # A directory or an mfsim.nam: the defaults (CPU backend).
            for cfg in (dir, joinpath(dir, "mfsim.nam"))
                s = BMI.initialize(Simulation, cfg)
                @test s.model.spec.name == "BMI"
                BMI.finalize(s)
            end
            # Every key; output into a directory relative to the configuration file.
            cfg = _bmi_config(
                joinpath(tmp, "cfg_all", "all.toml"), joinpath(dir, "mfsim.nam"), "double";
                extra = """
                preconditioner = "mc_sgs"
                outdir = "out"
                [closures]
                outer_dvclose = 1.0e-5
                outer_maximum = 30
                [precond_options]
                relax = 0.9
                """,
            )
            write(cfg, replace(read(cfg, String), "write_output = false" => "write_output = true"))
            if ModflowGPU.supports_float64(TEST_BACKEND)
                s = BMI.initialize(Simulation, cfg)
                @test s.T == Float64 && s.solver.settings.krylov.preconditioner === :mc_sgs
                @test s.solver.settings.maxiter == 30
                BMI.update_until(s, BMI.get_end_time(s))
                BMI.finalize(s)
                @test isdir(joinpath(tmp, "cfg_all", "out")) && !isempty(readdir(joinpath(tmp, "cfg_all", "out")))
            end
            # Errors: keywords, no configuration, unknown keys, bad values, wrong files.
            @test_throws r"BMI.initialize takes no keywords" BMI.initialize(Simulation, dir; precision = :double)
            @test _bmi_unsupported(() -> BMI.initialize(Simulation), :initialize)
            bad(text) = (f = joinpath(tmp, "bad.toml"); write(f, text); f)
            @test_throws r"unknown key `precison`" BMI.initialize(Simulation, bad("simulation = \"$(escape_string(dir))\"\nprecison = \"single\""))
            @test_throws r"`simulation`.*is missing" BMI.initialize(Simulation, bad("backend = \"cpu\""))
            @test_throws r"`write_output` must be a bool" BMI.initialize(Simulation, bad("simulation = \"$(escape_string(dir))\"\nwrite_output = 1"))
            @test_throws r"unknown closure `dvclose`" BMI.initialize(Simulation, bad("simulation = \"$(escape_string(dir))\"\n[closures]\ndvclose = 1.0"))
            @test_throws r"not a directory or an mfsim.nam" BMI.initialize(Simulation, bad("simulation = \"nowhere\""))
            @test_throws r"must be a TOML configuration file" BMI.initialize(Simulation, joinpath(dir, "bmi.nam"))
            @test_throws r"does not exist" BMI.initialize(Simulation, joinpath(tmp, "missing.toml"))
        end
    end
end

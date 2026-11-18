# One script written for libmf6 through ModflowInterface.jl (T8.10), run unchanged on
# ModflowGPU: `bmi_script(ModflowInterface, ModflowModel, simdir)` or
# `bmi_script(ModflowGPU, ModflowGPUModel, simdir; backend, precision, ...)`. Only the
# module and the model type differ. It records the result of every BMI and XMI call that
# ModflowInterface.jl wraps, for test/unit/bmi_test.jl to compare (libmf6 side:
# validation/libmf6_bmi.jl in a subprocess).
#
# The simulation must have a model `BMI` with packages `CHD` and `WEL` (the models of
# test/unit/bmi_test.jl). A call that throws is recorded as `BmiScriptError(message)`.
# Destination buffers are larger than needed and filled with a sentinel, so that a
# function that writes more or fewer values than expected (libmf6's get_grid_y of DIS,
# get_grid_nodes_per_face of DISU) shows it without writing past the buffer.

# `BMI` is BasicModelInterface, bound by the includer (`import BasicModelInterface as BMI`).

struct BmiScriptError
    msg::String
end

# The call `f()`, or the error it throws.
function _bmis_try(f)
    try
        return f()
    catch e
        return BmiScriptError(sprint(showerror, e))
    end
end

# The values written into a buffer filled with `sentinel`: the prefix before the first
# sentinel value.
function _bmis_written(buf, sentinel)
    k = findfirst(x -> isequal(x, sentinel), buf)
    return buf[1:(isnothing(k) ? end : k - 1)]
end

const BMI_SCRIPT_MODEL = "BMI"

"The libmf6 addresses that the script reads (model `BMI`, packages `CHD` and `WEL`)."
const BMI_SCRIPT_ADDRESSES = [
    "BMI/X", "BMI/XOLD", "BMI/IBOUND",
    ("BMI/DIS/" .* ["NODEUSER", "NODEREDUCED", "TOP", "BOT", "AREA", "NODES", "NODESUSER"])...,
    ("BMI/CHD/" .* ["BOUND", "NODELIST", "SIMVALS", "NBOUND", "MAXBOUND"])...,
    ("BMI/WEL/" .* ["BOUND", "NODELIST", "SIMVALS", "NBOUND", "MAXBOUND"])...,
    "SLN_1/MXITER",
    ("TDIS/" .* ["DELT", "TOTIM", "PERTIM", "KPER", "KSTP", "NPER"])...,
]

# Every variable function of one address.
function _bmi_script_variable(MF, m, a)
    ptr = _bmis_try(() -> BMI.get_value_ptr(m, a))
    return Dict{String, Any}(
        "grid" => _bmis_try(() -> BMI.get_var_grid(m, a)),
        "type" => _bmis_try(() -> BMI.get_var_type(m, a)),
        "itemsize" => _bmis_try(() -> BMI.get_var_itemsize(m, a)),
        "nbytes" => _bmis_try(() -> BMI.get_var_nbytes(m, a)),
        "rank" => _bmis_try(() -> MF.get_var_rank(m, a)),
        "shape" => _bmis_try(() -> MF.get_var_shape(m, a)),
        "ptr_type" => ptr isa BmiScriptError ? ptr : string(typeof(ptr)),
        "ptr" => ptr isa BmiScriptError ? ptr : copy(ptr),
        "value" => _bmis_try(() -> BMI.get_value(m, a)),
        "value_dest" => ptr isa BmiScriptError ? ptr : _bmis_try(() -> (d = zeros(eltype(ptr), size(ptr)); BMI.get_value(m, a, d) === d && d)),
    )
end

# Every grid function of grid `g`.
function _bmi_script_grid(m, g)
    type = BMI.get_grid_type(m, g)
    r = Dict{String, Any}(
        "type" => type,
        "rank" => _bmis_try(() -> BMI.get_grid_rank(m, g)),
        "size" => _bmis_try(() -> BMI.get_grid_size(m, g)),
        # Only for rectilinear grids: on an unstructured grid libmf6 writes up to three
        # values while ModflowInterface.jl passes a buffer of length get_grid_rank = 0.
        "shape" => type == "rectilinear" ? _bmis_try(() -> BMI.get_grid_shape(m, g)) : nothing,
        "node_count" => _bmis_try(() -> BMI.get_grid_node_count(m, g)),
        "face_count" => _bmis_try(() -> BMI.get_grid_face_count(m, g)),
    )
    n = 10_000
    for (k, f) in (("x", BMI.get_grid_x), ("y", BMI.get_grid_y))
        r[k] = _bmis_try(() -> (buf = fill(NaN, n); f(m, g, buf) === buf && _bmis_written(buf, NaN)))
    end
    written(f) = _bmis_try(() -> (buf = fill(typemin(Cint), n); f(m, g, buf) === buf && _bmis_written(buf, typemin(Cint))))
    r["nodes_per_face"] = written(BMI.get_grid_nodes_per_face)
    # libmf6's get_grid_face_nodes writes the nodes per face of every user node into a
    # buffer of get_grid_face_count (reduced nodes) values: heap corruption when cells
    # are excluded (IDOMAIN), so the script calls it only when the two counts agree.
    nf, npf = r["face_count"], r["nodes_per_face"]
    r["face_nodes"] = nf isa Integer && npf isa Vector && nf == length(npf) ? written(BMI.get_grid_face_nodes) : nothing
    return r
end

"""
    bmi_script(MF, Model, simdir; kwargs...) -> Dict{String, Any}

Run the simulation in `simdir` with `Model(simdir; kwargs...)` and the calls of a
ModflowInterface.jl script (`MF`: the module with the XMI functions), and return what
every call returned.
"""
function bmi_script(MF::Module, Model::Type, simdir::AbstractString; kwargs...)
    r = Dict{String, Any}()
    m = Model(simdir; kwargs...)
    r["initialize_returns_model"] = BMI.initialize(m) === m
    r["component_name"] = BMI.get_component_name(m)
    r["times0"] = (BMI.get_start_time(m), BMI.get_current_time(m), BMI.get_end_time(m), BMI.get_time_step(m))
    names = BMI.get_output_var_names(m)
    r["output_names_subset"] = issubset(BMI_SCRIPT_ADDRESSES, names)
    r["input_equals_output"] = BMI.get_input_var_names(m) == names
    r["counts_match"] = BMI.get_output_item_count(m) == length(names) == BMI.get_input_item_count(m)
    r["vars"] = Dict(a => _bmi_script_variable(MF, m, a) for a in BMI_SCRIPT_ADDRESSES)
    r["unknown_var_type"] = _bmis_try(() -> BMI.get_var_type(m, "BMI/NOSUCHVAR"))
    g = BMI.get_var_grid(m, "BMI/X")
    r["grid_id"] = g
    r["grid"] = _bmi_script_grid(m, g)
    r["var_address"] = (
        MF.get_var_address(m, "x", "bmi"),
        MF.get_var_address(m, "bound", "bmi"; subcomponent_name = "wel"),
        MF.get_var_address(m, "MXITER", "SLN_1"),
    )
    r["subcomponent_count"] = MF.get_subcomponent_count(m)

    # The time loop through the XMI calls, with the well rate written through the pointer.
    x = BMI.get_value_ptr(m, "BMI/X")
    maxiter = BMI.get_value_ptr(m, "SLN_1/MXITER")[]
    wel = BMI.get_value_ptr(m, "BMI/WEL/BOUND")
    totim = BMI.get_value_ptr(m, "TDIS/TOTIM")
    steps = []
    while BMI.get_current_time(m) < BMI.get_end_time(m)
        ret = MF.prepare_time_step(m, 0.0) === m
        istep = length(steps) + 1
        istep == 2 && (wel[1, 1] = -80.0)
        ret &= MF.prepare_solve(m, 1) === m
        iters = 0
        converged = false
        for _ in 1:maxiter
            iters += 1
            converged = MF.solve(m, 1)
            converged isa Bool || error("solve returned $(typeof(converged))")
            converged && break
        end
        ret &= MF.finalize_solve(m, 1) === m
        ret &= MF.finalize_time_step(m) === m
        push!(
            steps, Dict{String, Any}(
                "returns_model" => ret, "converged" => converged, "time" => BMI.get_current_time(m),
                "totim_ptr" => totim[], "delt" => BMI.get_time_step(m), "x" => copy(x),
                "kper" => BMI.get_value_ptr(m, "TDIS/KPER")[], "kstp" => BMI.get_value_ptr(m, "TDIS/KSTP")[],
                "wel_simvals" => copy(BMI.get_value_ptr(m, "BMI/WEL/SIMVALS")),
                "chd_simvals" => copy(BMI.get_value_ptr(m, "BMI/CHD/SIMVALS")),
                "wel_bound" => copy(wel), "xold" => BMI.get_value(m, "BMI/XOLD"),
            ),
        )
    end
    r["steps"] = steps
    r["last_error_type"] = string(typeof(MF.get_last_bmi_error(m)))

    # Initialise again (BMI.initialize on an initialised model), then BMI.update.
    r["reinitialize_returns_model"] = BMI.initialize(m) === m
    r["times_reinit"] = (BMI.get_current_time(m), BMI.get_end_time(m))
    r["x_reinit"] = copy(BMI.get_value_ptr(m, "BMI/X"))
    r["update_returns_model"] = BMI.update(m) === m
    r["time_update"] = BMI.get_current_time(m)
    r["x_update"] = copy(BMI.get_value_ptr(m, "BMI/X"))
    r["finalize_returns_model"] = BMI.finalize(m) === m
    return r
end

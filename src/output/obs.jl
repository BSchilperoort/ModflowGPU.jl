# Observations (OBS6, T6.7): values per time step and their output files.
#
# Ports, for the GWF model and its stress packages:
# - gwf-obs.f90 `gwf_obs_bd` (HEAD = x(n), DRAWDOWN = strt(n) − x(n), FLOW-JA-FACE =
#   flowja(jaindex)), `gwf_process_intercell_obs_id` (the connection must exist,
#   `getjaindex`);
# - BoundaryPackage.f90 `bnd_rp_obs` (at the start of every stress period: the records of
#   each observation, by boundary name or by cell, in record order), `bnd_bd_obs` (the sum
#   of `simvals` over those records, DNODATA when there is none; TO-MVR gives DNODATA
#   without a mover), gwf-wel.f90 `wel_bd_obs` (WEL-REDUCTION = q·mult + rhs, i.e. the
#   specified rate minus the rate applied, with AUTO_FLOW_REDUCE);
# - Obs.f90 `SaveOneSimval` (cumulative types skip DNODATA), `build_headers`,
#   `write_obs_simvals`, ObsUtility.f90 `write_fmtd_obs` / `write_unfmtd_obs` (one row per
#   time step: the time as G20.13, the values as G0 or with DIGITS as
#   G(DIGITS+8).DIGITS E3; BINARY files: a 100-byte header, NOBS, the names, then Float64
#   rows), gwf.f90 `gwf_ot_obs` (every time step, after the flows).
#
# The values are gathered on the device into one small buffer per time step
# (`obs_gather_kernel!`, one launch per source array: heads, FLOWJA, the SIMVALS and the
# BOUND of a package) and copied to the host once; heads are made absolute there in
# Float64 (`r + h'`, HDRY for dry cells), sums are taken in Float64. A model without
# observations does none of this.

@kernel function obs_gather_kernel!(vals, @Const(src), @Const(idx), off)
    i = @index(Global)
    @inbounds vals[off + i] = src[idx[i]]
end

"""
    obs_gather!(vals, src, idx, off)

Launcher of `obs_gather_kernel!`: `vals[off + i] = src[idx[i]]` for every `i` of `idx`
(linear indices into `src`, which may be a matrix).
"""
function obs_gather!(vals, src, idx, off::Integer)
    n = length(idx)
    n == 0 && return vals
    obs_gather_kernel!(get_backend(vals), _WG)(vals, src, idx, Int32(off); ndrange = n)
    return vals
end

# One observation as the engine evaluates it.
struct _Obs
    name::String
    type::String
    owner::Int        # 0: model; else the package index in `m.packages`
    node::Int32       # reduced node (HEAD, DRAWDOWN, package cellid), else 0
    jaindex::Int32    # FLOW-JA-FACE
    boundname::String
end

# One output file.
struct _ObsFile
    path::String
    io::IOStream
    binary::Bool
    fmt::Int          # DIGITS (0 = G0)
    cols::Vector{Int} # observations, in column order
end

"""
    ObservationOutput

The observations of a model run (the model's OBS6 file and its packages'): the output
files (`paths`, in the order MF6 opens them: model, then packages in MF6's order) and the
device buffers used to collect the values of a time step. Internal; built by
`ModelOutput` when the model has observations.
"""
mutable struct ObservationOutput{VT <: AbstractVector}
    const obs::Vector{_Obs}
    const files::Vector{_ObsFile}
    const paths::Vector{String}
    # records of each package observation (`bnd_rp_obs`), by observation
    const recs::Vector{Vector{Int32}}
    # gather plan: per source (`:h`, `:flowja`, `(:simvals, ip)`, `(:bound, ip)`) the
    # device indices and their offset in `vals`
    sources::Vector{Any}
    idx::Vector{Any}
    offs::Vector{Int}
    vals::VT
    host::Vector
    # position of each observation's first value in `host` (records follow), and of the
    # BOUND values of WEL-REDUCTION
    slot::Vector{Int}
    bslot::Vector{Int}
    const strt::Vector{Float64}
    # T6.9: the host copies of `idx` (the plan of the last `_obs_plan!`, reused when a new
    # period gives the same records)
    hostidx::Vector{Vector{Int32}}
end

_has_obs(spec::GwfSpec) = !isnothing(spec.obs) || any(b -> !isnothing(b.obs), spec.boundaries)

function _obs_open(simdir, outdir, o::ObsOutputSpec, digits, cols)
    p = _output_path(simdir, outdir, o.file)
    io = open(p, "w")
    names = [o.observations[i].name for i in eachindex(o.observations)]
    if o.binary
        # Obs.f90 build_headers: 'cont double', LENOBSNAME as i4, blanks to byte 100.
        write(io, rpad("cont double" * lpad(string(LENOBSNAME), 4), 100))
        write(io, Int32(length(names)))
        foreach(n -> write(io, rpad(n, LENOBSNAME)), names)
    else
        print(io, "time")
        foreach(n -> print(io, ",", n), names)
        println(io)
    end
    return _ObsFile(p, io, o.binary, digits, cols)
end

# `_obs_open` for each `(o, digits, cols)` of `opens`; when one fails, the files already
# opened are closed (T6.10).
function _obs_open_all(simdir, outdir, opens)
    files = _ObsFile[]
    try
        for (o, digits, cols) in opens
            push!(files, _obs_open(simdir, outdir, o, digits, cols))
        end
    catch
        foreach(f -> close(f.io), files)
        rethrow()
    end
    return files
end

# T6.6: positions in the engine arrays of a model of a multi-model solution: reduced nodes
# and package indices shifted into the merged model, FLOW-JA-FACE entries through `jamap`.
_obs_node(::GwfModel, n) = n
_obs_node(v::ModelView, n) = n > 0 ? n + _voffset(v) : n
_obs_ja(::GwfModel, ja) = ja
_obs_ja(v::ModelView, ja) = ja > 0 ? v.part.jamap[ja] : ja

"""
    ObservationOutput(m::GwfModel, simdir, outdir) -> ObservationOutput or nothing
    ObservationOutput(v::ModelView, simdir, outdir) -> ObservationOutput or nothing

Open the observation output files of model `m` (`nothing` without observations) and
resolve the observations on the model's grid: reduced nodes, and the FLOW-JA-FACE
connections (`getjaindex`; a pair of cells without a connection is an `MF6InputError`
with MF6's message). For a model of a multi-model solution (T6.6) the nodes, connections
and packages are those of the merged model, which `obs_rp!`, `obs_values` and
`write_observations` then take.
"""
function ObservationOutput(m::Union{GwfModel, ModelView}, simdir::AbstractString, outdir::AbstractString)
    spec = _vspec(m)
    _has_obs(spec) || return nothing
    mm = _vmodel(m)
    T = eltype(mm)
    conn = _vconn(m)
    obs = _Obs[]
    files = _ObsFile[]
    owners = Tuple{Int, ObsSpec}[]
    isnothing(spec.obs) || push!(owners, (0, spec.obs))
    for ip in _vpackages(m)
        pkg = mm.packages[ip]
        isnothing(pkg.spec.obs) || push!(owners, (ip, pkg.spec.obs))
    end
    opens = Tuple{ObsOutputSpec, Int, Vector{Int}}[]
    for (owner, os) in owners, o in os.outputs
        cols = Int[]
        for e in o.observations
            node = isempty(e.cells) ? Int32(0) : Int32(conn.nodereduced[e.cells[1]])
            ja = Int32(0)
            if e.type == "FLOW-JA-FACE"
                n2 = conn.nodereduced[e.cells[2]]
                ja = Int32(getjaindex(conn, node, n2))
                ja == 0 && throw(MF6InputError(e.file, e.line, "Error: No connection exists between cells identified in text: $(e.idstring)"))
            end
            push!(obs, _Obs(e.name, e.type, owner, Int32(_obs_node(m, node)), Int32(_obs_ja(m, ja)), e.boundname))
            push!(cols, length(obs))
        end
        push!(opens, (o, os.digits, cols))
    end
    # The files are opened once every observation is resolved (no file left open by an error).
    append!(files, _obs_open_all(simdir, outdir, opens))
    strt = zeros(Int(mm.conn.nodes))
    strt[_vrange(m)] .= Float64[spec.ic.strt[u] for u in conn.nodeuser]
    m = mm
    z = similar(m.h, 0)
    out = ObservationOutput{typeof(z)}(
        obs, files, [f.path for f in files], [Int32[] for _ in obs], Any[], Any[], Int[], z, T[], zeros(Int, length(obs)),
        zeros(Int, length(obs)), strt, Vector{Int32}[],
    )
    try
        _obs_plan!(out, m)
    catch
        close(out)
        rethrow()
    end
    return out
end

function Base.close(o::ObservationOutput)
    foreach(f -> close(f.io), o.files)
    return nothing
end

"""
    obs_rp!(o::ObservationOutput, m::GwfModel)

The records of each package observation for the stress period that starts
(`bnd_rp_obs`): by boundary name (BOUNDNAMES) or by cell, among the records in force, in
record order; then the gather plan is rebuilt.
"""
function obs_rp!(o::ObservationOutput, m::GwfModel)
    for (i, ob) in enumerate(o.obs)
        ob.owner == 0 && continue
        pkg = m.packages[ob.owner]
        r = o.recs[i]
        empty!(r)
        if !isempty(ob.boundname)
            names = pkg.boundname
            for j in 1:min(pkg.nbound, length(names))
                names[j] == ob.boundname && push!(r, Int32(j))
            end
        else
            for j in 1:pkg.nbound
                pkg.nodelist_host[j] == ob.node && push!(r, Int32(j))
            end
        end
    end
    _obs_plan!(o, m)
    return o
end

# The gather plan: device index vectors per source array and the slots of the
# observations in the host buffer.
function _obs_plan!(o::ObservationOutput, m::GwfModel{T}) where {T}
    lists = Dict{Any, Vector{Int32}}()
    order = Any[]
    function add!(src, i)
        haskey(lists, src) || (lists[src] = Int32[]; push!(order, src))
        push!(lists[src], Int32(i))
        return length(lists[src])
    end
    # (source, position within the source) of each observation's values
    pos = Vector{Tuple{Any, Int}}(undef, length(o.obs))
    bpos = Vector{Tuple{Any, Int}}(undef, length(o.obs))
    for (i, ob) in enumerate(o.obs)
        bpos[i] = (nothing, 0)
        if ob.type in ("HEAD", "DRAWDOWN")
            pos[i] = (:h, add!(:h, ob.node))
        elseif ob.type == "FLOW-JA-FACE"
            pos[i] = (:flowja, add!(:flowja, ob.jaindex))
        elseif ob.type == "TO-MVR" || isempty(o.recs[i])
            pos[i] = (nothing, 0)
        else
            src = (:simvals, ob.owner)
            p = 0
            for (k, r) in enumerate(o.recs[i])
                q = add!(src, r)
                k == 1 && (p = q)
            end
            pos[i] = (src, p)
            if ob.type == "WEL-REDUCTION"
                # BOUND column 1 (q, multiplier applied) of each record: linear index.
                ncol = size(m.packages[ob.owner].bound, 1)
                b = (:bound, ob.owner)
                pb = 0
                for (k, r) in enumerate(o.recs[i])
                    q = add!(b, (Int(r) - 1) * ncol + 1)
                    k == 1 && (pb = q)
                end
                bpos[i] = (b, pb)
            end
        end
    end
    backend = m.backend
    hostidx = Vector{Int32}[lists[src] for src in order]
    # T6.9: the same plan as in force (e.g. a new period with the same records): nothing
    # to rebuild.
    if !isempty(o.offs) || !isempty(o.sources)
        if isequal(order, o.sources) && hostidx == o.hostidx
            return o
        end
    end
    offs = Int[]
    idx = Any[]
    off = 0
    for (k, src) in enumerate(order)
        push!(offs, off)
        # T6.9: reuse the device index vector of the same position when it has the length.
        old = k <= length(o.idx) ? o.idx[k] : nothing
        if !isnothing(old) && length(old) == length(hostidx[k])
            copyto!(old, hostidx[k])
            push!(idx, old)
        else
            push!(idx, _device_copy(backend, hostidx[k]))
        end
        off += length(hostidx[k])
    end
    start = Dict{Any, Int}(src => offs[k] for (k, src) in enumerate(order))
    for i in eachindex(o.obs)
        s, p = pos[i]
        o.slot[i] = isnothing(s) ? 0 : start[s] + p
        s, p = bpos[i]
        o.bslot[i] = isnothing(s) ? 0 : start[s] + p
    end
    o.sources = order
    o.idx = idx
    o.offs = offs
    o.hostidx = hostidx
    length(o.vals) == off || (o.vals = fill!(similar(m.h, off), zero(T)))
    length(o.host) == off || (o.host = zeros(T, off))
    return o
end

_obs_source(m::GwfModel, s::Symbol) = s === :h ? m.h : m.flowja
_obs_source(m::GwfModel, s::Tuple) = first(s) === :simvals ? m.packages[last(s)].simvals : m.packages[last(s)].bound

"""
    obs_values(o::ObservationOutput, m::GwfModel) -> Vector{Float64}

The value of every observation at the end of the time step (`gwf_obs_bd`, `bnd_bd_obs`,
`wel_bd_obs`), in Float64: the device values are gathered into one buffer and copied to
the host once.
"""
function obs_values(o::ObservationOutput, m::GwfModel)
    for (k, src) in enumerate(o.sources)
        obs_gather!(o.vals, _obs_source(m, src), o.idx[k], o.offs[k])
    end
    isempty(o.host) || copyto!(o.host, o.vals)
    v = zeros(length(o.obs))
    r = m.setup.r
    for (i, ob) in enumerate(o.obs)
        s = o.slot[i]
        if ob.type in ("HEAD", "DRAWDOWN")
            n = ob.node
            x = r[n] + Float64(o.host[s])
            # Dry cells hold HDRY in MF6's x (T6.1, `absolute_heads`).
            isnothing(m.wd) || m.ibound_host[n] != 0 || (x = HDRY)
            v[i] = ob.type == "HEAD" ? x : o.strt[n] - x
        elseif ob.type == "FLOW-JA-FACE"
            v[i] = Float64(o.host[s])
        elseif s == 0
            v[i] = OBSNODATA
        elseif ob.type == "WEL-REDUCTION"
            if m.packages[ob.owner].spec.auto_flow_reduce > 0
                b = o.bslot[i]
                v[i] = sum(Float64(o.host[b + k]) - Float64(o.host[s + k]) for k in 0:(length(o.recs[i]) - 1))
            else
                v[i] = OBSNODATA
            end
        else
            v[i] = sum(Float64(o.host[s + k]) for k in 0:(length(o.recs[i]) - 1))
        end
    end
    return v
end

"""
    obs_format(x, d; lead0 = false, edigits = 2) -> String

`x` as Fortran G editing with `d` significant digits writes it (the observation output
of ObsUtility.f90 `write_fmtd_obs`, as MF6's Intel build prints it): fixed notation for
`0.1 ≤ |x| < 10^d` after rounding, else `0.ddd…E±ee`; the zero before the decimal point
only with `lead0` (a field width, G20.13 and DIGITS; MF6's default G0 omits it). G0 is
`d = 16`, `lead0 = false`.
"""
function obs_format(x::Float64, d::Integer; lead0::Bool = false, edigits::Integer = 2)
    isfinite(x) || return string(x)
    if x == 0
        return "0." * "0"^(d - 1)
    end
    s = @sprintf("%.*e", d - 1, abs(x))
    epos = findfirst('e', s)
    digits = s[1:1] * s[3:(epos - 1)]
    k = parse(Int, s[(epos + 1):end]) + 1   # x = 0.ddd × 10^k
    sgn = x < 0 ? "-" : ""
    if 0 <= k <= d
        k == 0 && return sgn * (lead0 ? "0." : ".") * digits
        return sgn * digits[1:k] * "." * digits[(k + 1):end]
    end
    e = abs(k)
    return sgn * (lead0 ? "0." : ".") * digits * "E" * (k < 0 ? "-" : "+") * lpad(string(e), max(edigits, ndigits(e)), '0')
end

_obs_value_text(x, fmt) = fmt == 0 ? obs_format(x, 16) : obs_format(x, fmt; lead0 = true, edigits = 3)

"""
    write_observations(o::ObservationOutput, m::GwfModel, totim)

Append the observations of the time step that ends at `totim` to their files
(`gwf_ot_obs`, `write_obs_simvals`).
"""
function write_observations(o::ObservationOutput, m::GwfModel, totim::Float64)
    v = obs_values(o, m)
    for f in o.files
        if f.binary
            write(f.io, totim)
            foreach(i -> write(f.io, v[i]), f.cols)
        else
            print(f.io, obs_format(totim, 13; lead0 = true))
            foreach(i -> print(f.io, ",", _obs_value_text(v[i], f.fmt)), f.cols)
            println(f.io)
        end
        flush(f.io)
    end
    return v
end

"""
    ObservationFile

An observation output file read by [`read_observations`](@ref): `path`, the observation
`names` (upper case, the column order), the `times` (end of each time step) and
`values`, a matrix with one row per time and one column per observation (Float64; MF6
writes `3.0e30` for a package observation that matches no boundary in a step).
"""
struct ObservationFile
    path::String
    names::Vector{String}
    times::Vector{Float64}
    values::Matrix{Float64}
end

Base.show(io::IO, f::ObservationFile) =
    print(io, "ObservationFile(\"", f.path, "\", ", length(f.names), " observation(s), ", length(f.times), " time(s))")

"""
    read_observations(path) -> ObservationFile

Read an observation output file of ModflowGPU or MF6 (OBS6 `CONTINUOUS FILEOUT`): the CSV
file (a `time` column, then one column per observation) or the BINARY file (MF6's
`cont double` or `cont single` format).

# Example
```julia
result = run_simulation("path/to/simulation"; outdir = "out")
obs = read_observations(result.obs_files[1])
obs.names, obs.times, obs.values[:, 1]
```
"""
function read_observations(path::AbstractString)
    head = open(io -> read(io, min(4, filesize(path))), path)
    String(head) == "cont" && return _read_binary_observations(path)
    lines = readlines(path)
    isempty(lines) && throw(ArgumentError("read_observations: $path is empty"))
    hdr = split(strip(lines[1]), ',')
    lowercase(hdr[1]) == "time" || throw(ArgumentError("read_observations: $path does not start with a \"time\" column"))
    names = String.(strip.(hdr[2:end]))
    rows = filter(!isempty ∘ strip, lines[2:end])
    times = zeros(length(rows))
    vals = zeros(length(rows), length(names))
    for (i, l) in enumerate(rows)
        w = split(strip(l), ',')
        length(w) == length(names) + 1 || throw(ArgumentError("read_observations: $path, line $(i + 1) has $(length(w)) values, expected $(length(names) + 1)"))
        times[i] = parse(Float64, w[1])
        for j in eachindex(names)
            vals[i, j] = parse(Float64, w[j + 1])
        end
    end
    return ObservationFile(String(path), names, times, vals)
end

function _read_binary_observations(path)
    return open(path) do io
        hdr = String(read(io, 100))
        F = startswith(hdr, "cont single") ? Float32 : Float64
        len = parse(Int, strip(hdr[12:15]))
        nobs = Int(read(io, Int32))
        names = [String(rstrip(String(read(io, len)))) for _ in 1:nobs]
        data = F[]
        while !eof(io)
            push!(data, read(io, F))
        end
        nt = length(data) ÷ (nobs + 1)
        m = reshape(Float64.(data[1:(nt * (nobs + 1))]), nobs + 1, nt)
        ObservationFile(String(path), names, m[1, :], permutedims(m[2:end, :]))
    end
end

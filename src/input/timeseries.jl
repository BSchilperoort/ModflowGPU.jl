# Time series (TS6) and time-array series (TAS6) on the host, in Float64 (T6.3).
#
# Ports src/Utilities/TimeSeries/:
# - TimeSeries.f90 `Initializetsfile`, `read_tsfile_line` (TS6 files: ATTRIBUTES NAME(S),
#   METHOD(S), SFAC(S), AUTODEALLOCATE; TIMESERIES records `time value1 value2 ...`, several
#   series per file, DNODATA (3.0e30) skips a value of one series, values times SFAC);
#   `GetValue`, `get_average_value`, `get_integrated_value`, `get_value_at_time`,
#   `get_surrounding_records`, `get_latest_preceding_node` (evaluation).
# - TimeArraySeries.f90 `tas_init`, `read_next_array` (TAS6 files: ATTRIBUTES NAME,
#   METHOD, SFAC, AUTODEALLOCATE; one `BEGIN TIME t` block per array, NCPL values),
#   `GetAverageValues`, `get_integrated_values`, `get_values_at_time`,
#   `get_surrounding_records`, `get_latest_preceding_node`.
# - TimeSeriesManager.f90 `tsmgr_ad` and TimeArraySeriesManager.f90 `tasmgr_ad`: per time
#   step every linked value is the series value over `[totimc, totimc + delt]`
#   (`ts_value`, `tas_values!`); the boundary framework (`advance_time_series!`) writes
#   them into the period data in force.
#
# Evaluation rules (MF6's, kept as they are):
# - STEPWISE and LINEAR give the time-weighted average over the step: the integral of the
#   step function or of the linear interpolant from `t0` to `t1`, divided by `t1 − t0`
#   (the value at `t0` when `t1 == t0`). LINEAREND (TS6 only) gives the linearly
#   interpolated value at the end of the step, `t1`.
# - Times are compared with `is_close` (relative 100·eps) where MF6 does.
# - A series that starts after `t0` integrates to 0 (MF6 finds no preceding record and
#   returns 0; no error), except at time 0: MF6 first evaluates a list series at time 0
#   when it reads the first PERIOD block, an error without a value there (`apply_period!`).
#   A series that ends before `t1` is an error (MF6 does not
#   extend a series to the end of the simulation unless asked, and the GWF list and array
#   loaders never ask).
# MF6 reads the records lazily and drops those before the step (AUTODEALLOCATE); here a
# file is read once at load time and kept, which changes no value.

"""
    TimeSeries

One time series of a TS6 file (TimeSeries.f90 `TimeSeriesType`): `name` (upper case),
`method` (`:stepwise`, `:linear` or `:linearend`), the record `times` and `values` (SFAC
applied; records whose value was DNODATA left out), and the `file` it was read from.
Evaluate it with `ts_value`.
"""
struct TimeSeries
    name::Symbol
    method::Symbol
    times::Vector{Float64}
    values::Vector{Float64}
    file::String
end

Base.show(io::IO, s::TimeSeries) = print(io, "TimeSeries(:", s.name, ", :", s.method, ", ", length(s.times), " records)")

"""
    TimeArraySeries

A time-array series of a TAS6 file (TimeArraySeries.f90 `TimeArraySeriesType`): `name`
(upper case), `method` (`:stepwise` or `:linear`), the `times` of the `arrays` (one array
of NCPL values per `TIME` block, SFAC applied), and the `file`. Evaluate it with
`tas_values!`.
"""
struct TimeArraySeries
    name::Symbol
    method::Symbol
    times::Vector{Float64}
    arrays::Vector{Vector{Float64}}
    file::String
end

Base.show(io::IO, s::TimeArraySeries) =
    print(io, "TimeArraySeries(:", s.name, ", :", s.method, ", ", length(s.times), " arrays of ", isempty(s.arrays) ? 0 : length(s.arrays[1]), ")")

# MathUtil.f90 is_close with the defaults (rtol = DSAME = 100·eps, atol = 0, symmetric).
_ts_is_close(a::Float64, b::Float64) = a == b || abs(a - b) <= 100 * eps(Float64) * max(abs(a), abs(b))

const _TS_METHODS = Dict("STEPWISE" => :stepwise, "LINEAR" => :linear, "LINEAREND" => :linearend)

function _ts_method(l::InputLine, word, allowed)
    m = get(_TS_METHODS, uppercase(word), nothing)
    (isnothing(m) || !(m in allowed)) && input_error(l, "Unknown interpolation method: \"$(uppercase(word))\"")
    return m
end

"""
    read_time_series_file(path; simdir = dirname(path)) -> Vector{TimeSeries}

Read a TS6 file (`Initializetsfile`, `read_tsfile_line`): the ATTRIBUTES block (`NAME` or
`NAMES` first, then `METHOD` or `METHODS`, `SFAC` or `SFACS`, `AUTODEALLOCATE`) and the
TIMESERIES block (one record per line: the time, then one value per series; DNODATA
leaves the series without a record at that time). Values are multiplied by the series'
SFAC. Errors are `MF6InputError`s.
"""
function read_time_series_file(path::AbstractString; simdir::AbstractString = dirname(path))
    f = read_input_file(path; ftype = "TS6", simdir)
    attr = find_block(f, "ATTRIBUTES")
    isnothing(attr) && throw(MF6InputError(path, 0, "ATTRIBUTES block not found in time-series input file \"$path\""))
    names = String[]
    methods = Symbol[]
    sfacs = Float64[]
    method_set = false
    for l in attr.lines
        k = keyword(l, 1)
        k == "NAMES" && (k = "NAME")
        if k == "NAME"
            names = [uppercase(String(t)) for t in Iterators.drop(eachtoken(l), 1)]
            isempty(names) && input_error(l, "Name not specified for time series.")
            for n in names
                ncodeunits(n) <= 40 || input_error(l, "time-series name \"$n\" is longer than 40 characters")
            end
            methods = fill(:undefined, length(names))
            sfacs = ones(length(names))
        elseif k in ("METHOD", "METHODS", "SFAC", "SFACS")
            isempty(names) && input_error(l, "NAME attribute not provided before $k in file: $path")
            if k == "METHOD" || k == "SFAC"
                w = token(l, 2)
                isnothing(w) && input_error(l, "missing value after $k")
                if k == "METHOD"
                    fill!(methods, _ts_method(l, w, (:stepwise, :linear, :linearend)))
                    method_set = true
                else
                    fill!(sfacs, get_double(l, 2; name = "SFAC"))
                end
            elseif k == "METHODS"
                ws = collect(Iterators.drop(eachtoken(l), 1))
                length(ws) < length(names) && input_error(l, "METHODS attribute does not list a method for all time series.")
                for j in eachindex(names)
                    methods[j] = _ts_method(l, ws[j], (:stepwise, :linear, :linearend))
                end
                method_set = true
            else  # SFACS: one GetDouble per series
                for j in eachindex(names)
                    sfacs[j] = get_double(l, j + 1; name = "SFACS")
                end
            end
        elseif k == "AUTODEALLOCATE"
            # Memory management only (records before the step are dropped): no effect here.
        else
            input_error(l, "Unknown option found in ATTRIBUTES block: \"$k\"")
        end
    end
    isempty(names) && throw(MF6InputError(path, 0, "Name not specified for time series."))
    method_set || throw(MF6InputError(path, 0, "Interpolation method was not set.  METHOD or METHODS must be specified in the ATTRIBUTES block for this time series file."))
    ts = find_block(f, "TIMESERIES")
    (isnothing(ts) || isempty(ts.lines)) && throw(MF6InputError(path, 0, "Error: No time-series data contained in file: $path"))
    n = length(names)
    times = [Float64[] for _ in 1:n]
    values = [Float64[] for _ in 1:n]
    for l in ts.lines
        t = get_double(l, 1; name = "time")
        for j in 1:n
            v = get_double(l, j + 1; name = "value of time series $(names[j])")
            v == DNODATA && continue
            push!(times[j], t)
            push!(values[j], v * sfacs[j])
        end
    end
    return [TimeSeries(Symbol(names[j]), methods[j], times[j], values[j], String(path)) for j in 1:n]
end

"""
    read_time_array_series_file(path, ncpl; simdir = dirname(path)) -> TimeArraySeries

Read a TAS6 file (`tas_init`, `read_next_array`): the ATTRIBUTES block (`NAME`, `METHOD`
STEPWISE or LINEAR, `SFAC`, `AUTODEALLOCATE`), then one `BEGIN TIME t` block per array,
each one array of `ncpl` values (a control record and its data, as READARRAY), multiplied
by SFAC. Errors are `MF6InputError`s.
"""
function read_time_array_series_file(path::AbstractString, ncpl::Integer; simdir::AbstractString = dirname(path))
    f = read_input_file(path; ftype = "TAS6", simdir)
    attr = find_block(f, "ATTRIBUTES")
    isnothing(attr) && throw(MF6InputError(path, 0, "Error: Attributes block not found in file: $path"))
    name = ""
    method = :undefined
    sfac = 1.0
    for l in attr.lines
        k = keyword(l, 1)
        w = token(l, 2)
        (isnothing(w) && k != "AUTODEALLOCATE") && input_error(l, "missing value after $k")
        if k == "NAME"
            name = uppercase(String(w))
        elseif k == "METHOD"
            method = _ts_method(l, w, (:stepwise, :linear))
        elseif k == "SFAC"
            sfac = get_double(l, 2; name = "SFAC")
        elseif k != "AUTODEALLOCATE"
            input_error(l, "Unknown option found in ATTRIBUTES block: \"$k\"")
        end
    end
    isempty(name) && throw(MF6InputError(path, 0, "Name not specified for time array series in file: $path"))
    method === :undefined && throw(MF6InputError(path, 0, "Interpolation method not specified for time array series in file: $path"))
    times = Float64[]
    arrays = Vector{Float64}[]
    for b in find_blocks(f, "TIME")
        isempty(b.suffix) && input_error(b.header, "missing time after BEGIN TIME")
        t = get_double(b.header, 3; name = "TIME")
        a = zeros(ncpl)
        isempty(b.lines) && input_error(b.header, "missing array in TIME block $t")
        k = read_array!(a, b.lines, 1; name, simdir)
        k <= length(b.lines) && input_error(b.lines[k], "unexpected line after the array of TIME block $t")
        sfac == 1.0 || (a .*= sfac)
        push!(times, t)
        push!(arrays, a)
    end
    isempty(times) && throw(MF6InputError(path, 0, "Error encountered reading time-array data from file: $path"))
    return TimeArraySeries(Symbol(name), method, times, arrays, String(path))
end

@noinline _ts_error(s, msg) = throw(MF6InputError(s.file, 0, msg))

# TimeSeries.f90 get_value_at_time (via get_surrounding_records): the value at `time`.
function _ts_value_at(s::TimeSeries, time::Float64, extend::Bool)
    tt, vv = s.times, s.values
    n = length(tt)
    n == 0 && _ts_error(s, "Error getting value at time $time for time series \"$(s.name)\"")
    # currNode: advance while the next record is before `time` (and not close to it).
    k = 1
    while k < n && tt[k + 1] < time && !_ts_is_close(tt[k + 1], time)
        k += 1
    end
    # earlier record (only the first record can be later than `time`)
    k0 = k
    # later record: advance from currNode while before `time`
    k1 = k
    while tt[k1] < time && !_ts_is_close(tt[k1], time) && k1 < n
        k1 += 1
    end
    early = tt[k0] < time || _ts_is_close(tt[k0], time)
    late = tt[k1] > time || _ts_is_close(tt[k1], time)
    if early
        if late
            s.method === :stepwise && return vv[k0]
            # LINEAR and LINEAREND: linear interpolation.
            time0, time1 = tt[k0], tt[k1]
            timediff = time1 - time0
            ratio = timediff > 0 ? (time - time0) / timediff : 0.5
            return vv[k0] + (ratio * (vv[k1] - vv[k0]))
        elseif extend || _ts_is_close(tt[k0], time) || s.method === :stepwise
            return vv[k0]
        end
    elseif late && _ts_is_close(tt[k1], time)
        return vv[k1]
    end
    return _ts_error(s, "Error getting value at time $time for time series \"$(s.name)\"")
end

# TimeSeries.f90 get_latest_preceding_node: the last record at or before `time` (is_close),
# or 0 when the series starts later.
function _ts_preceding(times::Vector{Float64}, time::Float64)
    n = length(times)
    k = 1
    while k < n && (times[k + 1] < time || _ts_is_close(times[k + 1], time))
        k += 1
    end
    return (times[k] < time || _ts_is_close(times[k], time)) ? k : 0
end

# TimeSeries.f90 get_integrated_value: the integral over [time0, time1] (LINEAREND: the
# interpolated value at the end of the last interval).
function _ts_integrated(s::TimeSeries, time0::Float64, time1::Float64, extend::Bool)
    tt, vv = s.times, s.values
    n = length(tt)
    value = 0.0
    k = _ts_preceding(tt, time0)
    k == 0 && return value
    t1 = -1.0
    while true
        currTime = tt[k]
        _ts_is_close(currTime, time1) && break
        if currTime < time1
            (k == n && !extend) && _ts_error(s, "Error encountered while performing integration for time series \"$(s.name)\" for time interval: $time0 to $time1")
            currVal = vv[k]
            if k < n
                nextTime, nextVal = tt[k + 1], vv[k + 1]
            else  # extend: the last value holds until time1
                nextTime, nextVal = time1, currVal
            end
            t0 = (currTime > time0 || _ts_is_close(currTime, time0)) ? currTime : time0
            t1 = (nextTime < time1 || _ts_is_close(nextTime, time1)) ? nextTime : time1
            t01 = t1 - t0
            if s.method === :stepwise
                area = currVal * t01
            else
                timediff = nextTime - currTime
                ratio0 = (t0 - currTime) / timediff
                ratio1 = (t1 - currTime) / timediff
                valuediff = nextVal - currVal
                value0 = currVal + ratio0 * valuediff
                value1 = currVal + ratio1 * valuediff
                if s.method === :linear
                    area = 0.5 * t01 * (value0 + value1)
                else  # LINEAREND
                    area = 0.0
                    value = value1
                end
            end
            value = value + area
        end
        (t1 > time1 || _ts_is_close(t1, time1)) && break
        k == n && _ts_error(s, "Error encountered while performing integration for time series \"$(s.name)\" for time interval: $time0 to $time1")
        k += 1
    end
    return value
end

"""
    ts_value(s::TimeSeries, t0, t1; extend = false) -> Float64

The value of the series for the time step `[t0, t1]` (TimeSeries.f90 `GetValue`):
STEPWISE and LINEAR give the time-weighted average over the step
(`get_average_value`: integral over `[t0, t1]` divided by `t1 − t0`; the value at `t0`
when `t1 == t0`), LINEAREND the interpolated value at `t1` (`get_value_at_time`).
`extend` keeps the last value after the end of the series (MF6's
`extendToEndOfSimulation`; off in the GWF stress packages). Float64, host.
"""
function ts_value(s::TimeSeries, t0::Real, t1::Real; extend::Bool = false)
    time0, time1 = Float64(t0), Float64(t1)
    # A series whose values are all DNODATA has no records (MF6: get_latest_preceding_node,
    # "probable programming error").
    isempty(s.times) && _ts_error(s, "time series \"$(s.name)\" has no records (every value is DNODATA)")
    s.method === :linearend && return _ts_value_at(s, time1, extend)
    timediff = time1 - time0
    timediff > 0 || return _ts_value_at(s, time0, extend)
    return _ts_integrated(s, time0, time1, extend) / timediff
end

# TimeArraySeries.f90 get_values_at_time (via get_surrounding_records, which compares
# times with <= and >=, not is_close).
function _tas_values_at!(out, s::TimeArraySeries, time::Float64)
    tt = s.times
    n = length(tt)
    k = 1
    while k < n && tt[k + 1] <= time
        k += 1
    end
    k0 = k
    k1 = k
    while tt[k1] < time && k1 < n
        k1 += 1
    end
    early = tt[k0] <= time
    late = tt[k1] >= time
    if early
        if late
            if s.method === :stepwise
                copyto!(out, s.arrays[k0])
            else
                time0, time1 = tt[k0], tt[k1]
                timediff = time1 - time0
                ratio = timediff > 0 ? (time - time0) / timediff : 0.5
                a0, a1 = s.arrays[k0], s.arrays[k1]
                for i in eachindex(out)
                    out[i] = a0[i] + (ratio * (a1[i] - a0[i]))
                end
            end
            return out
        elseif _ts_is_close(tt[k0], time) || s.method === :stepwise
            return copyto!(out, s.arrays[k0])
        end
    elseif late && _ts_is_close(tt[k1], time)
        return copyto!(out, s.arrays[k1])
    end
    return _ts_error(s, "Error getting array at time $time for time-array series \"$(s.name)\"")
end

# TimeArraySeries.f90 get_integrated_values.
function _tas_integrated!(out, s::TimeArraySeries, time0::Float64, time1::Float64)
    fill!(out, 0.0)
    tt = s.times
    n = length(tt)
    k = _ts_preceding(tt, time0)
    k == 0 && return out
    err() = _ts_error(s, "Error encountered while performing integration for time-array series \"$(s.name)\" for time interval: $time0 to $time1")
    t1 = -1.0
    while true
        currTime = tt[k]
        if currTime < time1
            k == n && err()
            nextTime = tt[k + 1]
            t0 = currTime >= time0 ? currTime : time0
            t1 = nextTime <= time1 ? nextTime : time1
            t01 = t1 - t0
            a0, a1 = s.arrays[k], s.arrays[k + 1]
            if s.method === :stepwise
                for i in eachindex(out)
                    out[i] = out[i] + a0[i] * t01
                end
            else
                for i in eachindex(out)
                    timediff = nextTime - currTime
                    ratio0 = (t0 - currTime) / timediff
                    ratio1 = (t1 - currTime) / timediff
                    valuediff = a1[i] - a0[i]
                    value0 = a0[i] + ratio0 * valuediff
                    value1 = a0[i] + ratio1 * valuediff
                    out[i] = out[i] + 0.5 * t01 * (value0 + value1)
                end
            end
        else
            break
        end
        t1 >= time1 && break
        k == n && err()
        k += 1
    end
    return out
end

"""
    tas_values!(out, s::TimeArraySeries, t0, t1) -> out

The array of the series for the time step `[t0, t1]` (TimeArraySeries.f90
`GetAverageValues`): the time-weighted average of STEPWISE or LINEAR interpolation
(integral over the step divided by `t1 − t0`; the values at `t0` when `t1 == t0`).
Float64, host.
"""
function tas_values!(out::AbstractVector{Float64}, s::TimeArraySeries, t0::Real, t1::Real)
    length(out) == length(s.arrays[1]) || throw(DimensionMismatch("tas_values!: $(length(out)) values for arrays of $(length(s.arrays[1]))"))
    time0, time1 = Float64(t0), Float64(t1)
    timediff = time1 - time0
    timediff > 0 || return _tas_values_at!(out, s, time0)
    _tas_integrated!(out, s, time0, time1)
    for i in eachindex(out)
        out[i] = out[i] / timediff
    end
    return out
end

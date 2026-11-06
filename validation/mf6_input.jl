# Minimal MODFLOW 6 input reader for the example inventory (T0.3).
#
# Standalone (Julia stdlib only); not the ModflowGPU parser. It understands just
# enough of the MF6 input format to describe a simulation: blocks, simple
# keyword options, list blocks and READARRAY-style grid arrays (CONSTANT,
# INTERNAL, OPEN/CLOSE text files, LAYERED). Binary arrays are reported as
# unreadable (`nothing`).

module MF6Input

export read_tokens, read_blocks, Block, find_block, find_blocks, option_lines,
    has_option, option_value, read_griddata, SimInfo, read_sim

"Tokens of one input line, original case, comments removed."
const Line = Vector{String}

const TOKEN_RE = r"'[^']*'|\"[^\"]*\"|[^\s,]+"

function tokenize(line::AbstractString)
    # MF6 treats '#' and '!' as the start of a comment (also inline).
    i = findfirst(c -> c == '#' || c == '!', line)
    i === nothing || (line = line[1:prevind(line, i)])
    return [strip(m.match, ['\'', '"']) for m in eachmatch(TOKEN_RE, line)]
end

"Non-empty token lines of a text file."
function read_tokens(path::AbstractString)
    lines = Line[]
    for l in eachline(path)
        t = tokenize(l)
        isempty(t) || push!(lines, String.(t))
    end
    return lines
end

struct Block
    name::String          # upper case, e.g. "OPTIONS", "PERIOD"
    arg::String           # e.g. the period number, "" if none
    lines::Vector{Line}
end

function read_blocks(path::AbstractString)
    blocks = Block[]
    cur = nothing
    for t in read_tokens(path)
        k = uppercase(t[1])
        if k == "BEGIN" && cur === nothing
            cur = Block(uppercase(t[2]), length(t) >= 3 ? t[3] : "", Line[])
        elseif k == "END" && cur !== nothing
            push!(blocks, cur)
            cur = nothing
        elseif cur !== nothing
            push!(cur.lines, t)
        end
    end
    return blocks
end

find_blocks(bs::Vector{Block}, name) = filter(b -> b.name == uppercase(name), bs)
function find_block(bs::Vector{Block}, name)
    f = find_blocks(bs, name)
    return isempty(f) ? nothing : f[1]
end

"Lines of the OPTIONS block (empty if none)."
function option_lines(bs::Vector{Block}, name = "OPTIONS")
    b = find_block(bs, name)
    return b === nothing ? Line[] : b.lines
end

has_option(lines::Vector{Line}, key) = any(l -> uppercase(l[1]) == uppercase(key), lines)

"Tokens after `key` on its first line, or `nothing`."
function option_value(lines::Vector{Line}, key)
    for l in lines
        uppercase(l[1]) == uppercase(key) && return l[2:end]
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Grid arrays

"""
    read_griddata(block, dir, sizefun, nlay) -> Dict{String,Union{Vector{Float64},Nothing}}

Read every array of a GRIDDATA block. `sizefun(name)` returns `(n, layerable)`:
the total length and whether a LAYERED read splits it in `nlay` equal parts.
Arrays in binary files give `nothing`.
"""
function read_griddata(b::Block, dir::AbstractString, sizefun, nlay::Int)
    toks = String[]
    for l in b.lines
        append!(toks, l)
    end
    out = Dict{String, Union{Vector{Float64}, Nothing}}()
    i = 1
    while i <= length(toks)
        name = lowercase(toks[i])
        i += 1
        layered = false
        if i <= length(toks) && uppercase(toks[i]) == "LAYERED"
            layered = true
            i += 1
        end
        n, _ = sizefun(name)
        nrec = layered ? nlay : 1
        m = n ÷ nrec
        vals = Float64[]
        readable = true
        for _ in 1:nrec
            ctrl = uppercase(toks[i])
            i += 1
            if ctrl == "CONSTANT"
                append!(vals, fill(parse(Float64, toks[i]), m))
                i += 1
            elseif ctrl == "INTERNAL"
                fac, i, _ = read_ctrl_opts(toks, i)
                for j in 0:(m - 1)
                    push!(vals, fac * parse(Float64, toks[i + j]))
                end
                i += m
            elseif ctrl == "OPEN/CLOSE"
                fname = toks[i]
                i += 1
                fac, i, binary = read_ctrl_opts(toks, i)
                if binary
                    readable = false
                else
                    ft = String[]
                    foreach(l -> append!(ft, l), read_tokens(joinpath(dir, fname)))
                    append!(vals, fac .* parse.(Float64, ft[1:m]))
                end
            else
                error("unknown array control '$ctrl' for $name in $(b.name)")
            end
        end
        out[name] = readable ? vals : nothing
    end
    return out
end

function read_ctrl_opts(toks, i)
    fac = 1.0
    binary = false
    while i <= length(toks)
        k = uppercase(toks[i])
        if k == "FACTOR"
            fac = parse(Float64, toks[i + 1])
            i += 2
        elseif k == "IPRN"
            i += 2
        elseif k == "(BINARY)"
            binary = true
            i += 1
        else
            break
        end
    end
    return fac, i, binary
end

# ---------------------------------------------------------------------------
# Simulation name file

struct ModelRef
    mtype::String   # "GWF6"
    namfile::String
    name::String
end

struct ExchangeRef
    etype::String   # "GWF6-GWF6"
    file::String
    m1::String
    m2::String
end

struct SolutionRef
    stype::String   # "IMS6"
    file::String
    models::Vector{String}
end

struct SimInfo
    dir::String
    tdis::String
    models::Vector{ModelRef}
    exchanges::Vector{ExchangeRef}
    solutions::Vector{SolutionRef}
end

function read_sim(dir::AbstractString)
    bs = read_blocks(joinpath(dir, "mfsim.nam"))
    tdis = ""
    for l in option_lines(bs, "TIMING")
        uppercase(l[1]) == "TDIS6" && (tdis = l[2])
    end
    models = ModelRef[]
    for l in option_lines(bs, "MODELS")
        push!(models, ModelRef(uppercase(l[1]), l[2], l[3]))
    end
    exgs = ExchangeRef[]
    for l in option_lines(bs, "EXCHANGES")
        push!(exgs, ExchangeRef(uppercase(l[1]), l[2], l[3], l[4]))
    end
    sols = SolutionRef[]
    for b in find_blocks(bs, "SOLUTIONGROUP"), l in b.lines
        uppercase(l[1]) == "MXITER" && continue
        push!(sols, SolutionRef(uppercase(l[1]), l[2], l[3:end]))
    end
    return SimInfo(dir, tdis, models, exgs, sols)
end

end # module

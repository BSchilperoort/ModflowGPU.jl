# Budget tables of an MF6 model listing file.
#
# `budget_ot` (Budget.f90) prints, at the end of each time step with PRINT BUDGET:
#
#   <title> BUDGET FOR <zone> AT END OF TIME STEP<I5>, STRESS PERIOD<I4>
#   (column headings)
#   IN:
#   1X,3X,A16,' =',A17,6X,A16,' =',A17[,5X,A]   per term: name = cumulative  name = rate  [package]
#   TOTAL IN = ...  TOTAL IN = ...
#   OUT:
#   (one line per term, as above)
#   TOTAL OUT = ...  TOTAL OUT = ...
#   IN - OUT = ...  IN - OUT = ...
#   PERCENT DISCREPANCY = F15.2  PERCENT DISCREPANCY = F15.2
#
# The values come from `value_to_string`: F17.4, or 1PE17.4 (ES17.4E3 for 3-digit
# exponents) below 0.1 or from 1e12 (1e11 for IN - OUT) on. GWF prints "VOLUME BUDGET FOR
# ENTIRE MODEL"; the package name column appears when the budget is labeled (GWF in MF6
# 6.x). Rate and cumulative volume of a term are matched on the term name and package
# name; a model may have several terms with the same name (e.g. three RCH packages).

"""
    ModflowGPU.ListingBudgetTerm

One term of a listing-file budget table, an element of `ListingBudget.terms`: `name`
(budget text, e.g. `"WEL"`, `"STO-SS"`),
`package` (package name column, `""` when the table is unlabeled), cumulative volumes
`cum_in`, `cum_out` and rates `rate_in`, `rate_out` (both non-negative, as printed).
"""
struct ListingBudgetTerm
    name::String
    package::String
    cum_in::Float64
    cum_out::Float64
    rate_in::Float64
    rate_out::Float64
end

"""
    ListingBudget

One budget table of a listing file: `title` (e.g. `"VOLUME"`), `zone` (e.g.
`"ENTIRE MODEL"`), `kstp`, `kper` (0 if the
printed number overflowed its field), the `terms` ([`ModflowGPU.ListingBudgetTerm`](@ref)), the
totals `total_cum_in`, `total_cum_out`, `total_rate_in`, `total_rate_out`, the
differences `cum_in_minus_out`, `rate_in_minus_out` and the printed
`cum_percent_discrepancy`, `rate_percent_discrepancy`.
"""
struct ListingBudget
    title::String
    zone::String
    kstp::Int
    kper::Int
    terms::Vector{ListingBudgetTerm}
    total_cum_in::Float64
    total_cum_out::Float64
    total_rate_in::Float64
    total_rate_out::Float64
    cum_in_minus_out::Float64
    rate_in_minus_out::Float64
    cum_percent_discrepancy::Float64
    rate_percent_discrepancy::Float64
end

const _LISTING_BUDGET_TITLE = r"^\s*(\S.*?) BUDGET FOR (.*?) AT END OF TIME STEP\s*(\S+)\s*, STRESS PERIOD\s*(\S+)"

function _listing_number(s::AbstractString, path, lineno)
    v = tryparse(Float64, strip(s))
    v === nothing && throw(MF6InputError(path, lineno, "cannot read the budget value \"$(strip(s))\""))
    return v
end

# `1X,3X,A16,' =',A17,6X,A16,' =',A17[,5X,A]`: columns 5-20 name, 23-39 value,
# 46-61 name, 64-80 value, from 86 package name.
function _listing_term_line(line::AbstractString, path, lineno)
    cu = codeunits(line)
    (length(cu) >= 80 && cu[21] == UInt8(' ') && cu[22] == UInt8('=') && cu[62] == UInt8(' ') && cu[63] == UInt8('=')) ||
        return nothing
    name1 = strip(line[5:20])
    name2 = strip(line[46:61])
    name1 == name2 || throw(MF6InputError(path, lineno, "budget term names differ: \"$name1\" and \"$name2\""))
    cum = _listing_number(line[23:39], path, lineno)
    rate = _listing_number(line[64:80], path, lineno)
    package = length(cu) >= 86 ? String(strip(line[86:end])) : ""
    return String(name1), package, cum, rate
end

function _listing_pair(line::AbstractString, key::AbstractString, path, lineno)
    parts = split(line, key * " =")
    length(parts) == 3 || throw(MF6InputError(path, lineno, "expected two '$key =' values"))
    return _listing_number(parts[2], path, lineno), _listing_number(parts[3], path, lineno)
end

"""
    read_listing_budgets(path) -> Vector{ListingBudget}

All budget tables (`... BUDGET FOR ... AT END OF TIME STEP ...`) of a model listing file
(`.lst`) of ModflowGPU or MF6, in file order, with the values as printed (rates L³/T,
volumes L³). Throws an [`ModflowGPU.MF6InputError`](@ref) for a table it cannot read.

# Example
```julia
tables = read_listing_budgets("path/to/simulation/model.lst")
last(tables).rate_percent_discrepancy
[(t.name, t.rate_in, t.rate_out) for t in last(tables).terms]
```
"""
function read_listing_budgets(path::AbstractString)
    path = String(path)
    lines = readlines(path)
    out = ListingBudget[]
    i = 1
    while i <= length(lines)
        m = match(_LISTING_BUDGET_TITLE, lines[i])
        if m === nothing
            i += 1
            continue
        end
        title = String(m[1])
        zone = String(m[2])
        kstp = something(tryparse(Int, m[3]), 0)
        kper = something(tryparse(Int, m[4]), 0)
        start = i
        names_in = Tuple{String, String, Float64, Float64}[]
        names_out = Tuple{String, String, Float64, Float64}[]
        section = :none
        tot = Dict{String, Tuple{Float64, Float64}}()
        i += 1
        while i <= length(lines)
            l = lines[i]
            s = strip(l)
            if s == "IN:" || startswith(s, "IN:")
                section = :in
            elseif startswith(s, "OUT:")
                section = :out
            elseif startswith(s, "TOTAL IN =")
                tot["TOTAL IN"] = _listing_pair(l, "TOTAL IN", path, i)
                section = :none
            elseif startswith(s, "TOTAL OUT =")
                tot["TOTAL OUT"] = _listing_pair(l, "TOTAL OUT", path, i)
                section = :none
            elseif startswith(s, "IN - OUT =")
                tot["IN - OUT"] = _listing_pair(l, "IN - OUT", path, i)
            elseif startswith(s, "PERCENT DISCREPANCY =")
                tot["PERCENT DISCREPANCY"] = _listing_pair(l, "PERCENT DISCREPANCY", path, i)
                break
            elseif section !== :none
                t = _listing_term_line(l, path, i)
                t === nothing || push!(section === :in ? names_in : names_out, t)
            end
            i += 1
        end
        length(tot) == 4 || throw(MF6InputError(path, start, "incomplete budget table"))
        length(names_in) == length(names_out) ||
            throw(MF6InputError(path, start, "budget table has $(length(names_in)) IN and $(length(names_out)) OUT terms"))
        terms = ListingBudgetTerm[]
        for (a, b) in zip(names_in, names_out)
            (a[1] == b[1] && a[2] == b[2]) ||
                throw(MF6InputError(path, start, "IN term $(a[1]) ($(a[2])) and OUT term $(b[1]) ($(b[2])) differ"))
            push!(terms, ListingBudgetTerm(a[1], a[2], a[3], b[3], a[4], b[4]))
        end
        ti, to, d, p = tot["TOTAL IN"], tot["TOTAL OUT"], tot["IN - OUT"], tot["PERCENT DISCREPANCY"]
        push!(out, ListingBudget(title, zone, kstp, kper, terms, ti[1], to[1], ti[2], to[2], d[1], d[2], p[1], p[2]))
        i += 1
    end
    return out
end

"""
    listing_rates(path, term; title = "VOLUME", zone = "ENTIRE MODEL")
        -> Vector{Tuple{Float64, Float64}}

`(rate in, rate out)` of the budget term `term` in every budget table of the listing
file with the given `title` and `zone` (by default the model's volume budget; e.g. one
table per stress period with PRINT BUDGET LAST), summed over the rows whose name or
package name equals `term` (case-insensitive). Ported from the `listing_rates` helper
of ModflowCoupling's tests, which took the first matching row; summing gives the same
for a single package and the model total for several packages of one type. Rates are in
L³/T in the model's units.

# Example
```julia
listing_rates("path/to/simulation/model.lst", "RIV")   # [(in, out) per table]
```
"""
function listing_rates(
        path::AbstractString, term::AbstractString; title::AbstractString = "VOLUME",
        zone::AbstractString = "ENTIRE MODEL",
    )
    tables = filter(b -> _text_eq(b.title, title) && _text_eq(b.zone, zone), read_listing_budgets(path))
    return map(tables) do b
        rows = filter(t -> _text_eq(t.name, term) || _text_eq(t.package, term), b.terms)
        isempty(rows) && throw(ArgumentError("no budget term $term in the table of KSTP $(b.kstp), KPER $(b.kper) of $path"))
        (sum(t -> t.rate_in, rows), sum(t -> t.rate_out, rows))
    end
end

"""
    percent_discrepancy(total_in, total_out) -> Float64
    percent_discrepancy(b::ModelBudget) -> Float64

MF6's percent discrepancy `100 (in - out) / ((in + out) / 2)`, zero when both are zero
(`budget_ot` in Budget.f90). For a [`ModelBudget`](@ref): of its total rates in and out
(see [`cum_percent_discrepancy`](@ref) for the cumulative volumes).

# Example
```jldoctest
julia> percent_discrepancy(101.0, 99.0)
2.0

julia> percent_discrepancy(0.0, 0.0)
0.0
```
"""
function percent_discrepancy(total_in::Real, total_out::Real)
    avg = (total_in + total_out) / 2
    return avg == 0 ? 0.0 : 100 * (total_in - total_out) / avg
end

# Budget.f90 `value_to_string`: F17.4, or 1PE17.4 (ES17.4E3 beyond 2-digit exponents) for
# non-zero values from `big` on or below `small`.
function _budget_value(v::Float64, big::Float64)
    a = abs(v)
    if v != 0 && (a >= big || a < 0.1)
        s = @sprintf("%.4E", v)
        m = match(r"^(.*)E([+-])(\d+)$", s)
        e = parse(Int, m[3])
        s = (a >= 9.99995e99 || a < 1.0e-99) ? string(m[1], "E", m[2], lpad(e, 3, '0')) : string(m[1], "E", m[2], lpad(e, 2, '0'))
        return lpad(s, 17)
    end
    return lpad(@sprintf("%.4f", v), 17)
end

"""
    write_listing_budget(io, kstp, kper, terms; title = "VOLUME", zone = "ENTIRE MODEL")

Write one budget table in the layout of MF6's `budget_ot` (Budget.f90, labeled form, as
GWF prints it; read back by [`read_listing_budgets`](@ref)). `terms` is a collection of
`(name, package, cum_in, cum_out, rate_in, rate_out)` tuples (Float64 values).
"""
function write_listing_budget(io::IO, kstp::Integer, kper::Integer, terms; title::AbstractString = "VOLUME", zone::AbstractString = "ENTIRE MODEL")
    big1, big2 = 9.99999e11, 9.99999e10
    @printf(io, "\n\n  %s BUDGET FOR %s AT END OF TIME STEP%5d, STRESS PERIOD%4d\n  %s\n", title, zone, kstp, kper, "-"^99)
    print(io, "\n     CUMULATIVE VOLUME      L**3       RATES FOR THIS TIME STEP      L**3/T          PACKAGE NAME    \n")
    print(io, "     ------------------                 ------------------------                     ----------------\n\n")
    print(io, " "^11, "IN:", " "^38, "IN:\n", " "^11, "---", " "^38, "---\n")
    line(name, a, b, pkg) = print(io, "    ", lpad(name, 16), " =", _budget_value(a, big1), " "^6, lpad(name, 16), " =", _budget_value(b, big1), " "^5, rpad(pkg, 16), "\n")
    for t in terms
        line(t[1], t[3], t[5], t[2])
    end
    tin = (sum(t -> t[3], terms; init = 0.0), sum(t -> t[5], terms; init = 0.0))
    tout = (sum(t -> t[4], terms; init = 0.0), sum(t -> t[6], terms; init = 0.0))
    print(io, "\n", " "^12, "TOTAL IN =", _budget_value(tin[1], big1), " "^14, "TOTAL IN =", _budget_value(tin[2], big1), "\n")
    print(io, "\n", " "^10, "OUT:", " "^37, "OUT:\n", " "^10, "----", " "^37, "----\n")
    for t in terms
        line(t[1], t[4], t[6], t[2])
    end
    print(io, "\n", " "^11, "TOTAL OUT =", _budget_value(tout[1], big1), " "^13, "TOTAL OUT =", _budget_value(tout[2], big1), "\n")
    print(io, "\n", " "^12, "IN - OUT =", _budget_value(tin[1] - tout[1], big2), " "^14, "IN - OUT =", _budget_value(tin[2] - tout[2], big2), "\n")
    @printf(io, "\n PERCENT DISCREPANCY = %14.2f     PERCENT DISCREPANCY = %14.2f\n\n", percent_discrepancy(tin[1], tout[1]), percent_discrepancy(tin[2], tout[2]))
    return nothing
end

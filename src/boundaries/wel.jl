# WEL: specified flows (gwf-wel.f90, `wel_cf`). With AUTO_FLOW_REDUCE (T6.1) an extraction
# in a convertible cell is reduced as the head approaches the cell bottom (`sQSaturation`
# between the bottom and bot + AUTO_FLOW_REDUCE · thickness, or · 1 with
# FLOW_REDUCTION_LENGTH; AUTO_FLOW_REDUCE_AUXNAME gives the value per record): the record
# kind is then a `WelAfr` (`record_kind`). AUTO_FLOW_REDUCE_CSV output is not supported.

bound_columns(::BoundaryKind{:WEL}) = ("q",)

# AUTO_FLOW_REDUCE_AUXNAME: the auxiliary variable is column 2 of `bound`.
function bound_layout(::BoundaryKind{:WEL}, spec::BoundarySpec)
    (spec.auto_flow_reduce != 0 && !isempty(spec.auto_flow_reduce_auxname)) || return Any[("q", 1)]
    iaux = findfirst(==(uppercase(spec.auto_flow_reduce_auxname)), uppercase.(spec.auxnames))
    isnothing(iaux) && throw(
        MF6InputError(
            spec.file, 0,
            "AUTO_FLOW_REDUCE_AUXNAME was specified as $(spec.auto_flow_reduce_auxname) but no AUX variable found with this name.",
        ),
    )
    return Any[("q", 1), (:aux, iaux)]
end

# wel_cf: hcof = 0, rhs = −q (q > 0 is injection).
@inline formulate_record(::BoundaryKind{:WEL}, i, n, bound, h, area) = (zero(eltype(bound)), -bound[1, i])

"""
    sq_saturation(top, bot, x) -> y

`sQSaturation` (SmoothingFunctions.f90) with the default coefficients −2 and 3: 0 below
`bot`, 1 above `top`, the cubic `−2·w³/b³ + 3·w²/b²` (`w = x − bot`, `b = top − bot`) in
between. Device-safe in the precision of the arguments.
"""
@inline function sq_saturation(top::T, bot::T, x::T) where {T}
    w = x - bot
    b = top - bot
    s = w / b
    cof1 = T(-2) / b^3
    cof2 = T(3) / b^2
    s < zero(T) && return zero(T)
    s < one(T) && return cof1 * w^3 + cof2 * w^2
    return one(T)
end

"""
    WelAfr

The record kind of a WEL package with AUTO_FLOW_REDUCE in a model with convertible cells
(T6.1; `wel_cf`): `icelltype`, `botoff` (`bot − r`), `thk` (`top − bot`) by node (from the
model's `ConvertibleNpf`), `flowred` (AUTO_FLOW_REDUCE after MF6's adjustments), `len`
(FLOW_REDUCTION_LENGTH) and `aux` (AUTO_FLOW_REDUCE_AUXNAME: the value is `bound[2, i]`).
"""
struct WelAfr{T, V <: AbstractVector{T}, VI <: AbstractVector{Int32}}
    icelltype::VI
    botoff::V
    thk::V
    flowred::T
    len::Bool
    aux::Bool
end

Adapt.@adapt_structure WelAfr

@inline function formulate_record(k::WelAfr, i, n, bound, h, area)
    T = eltype(bound)
    @inbounds begin
        q = bound[1, i]
        if q < zero(T) && k.icelltype[n] != Int32(0)
            bt = k.botoff[n]
            thick = k.len ? one(T) : k.thk[n]
            f = k.aux ? bound[2, i] : k.flowred
            q = q * sq_saturation(bt + f * thick, bt, h[n])
        end
        return (zero(T), -q)
    end
end

# The AUTO_FLOW_REDUCE value as MF6 uses it (gwf-wel.f90 `source_options`): <= 0 is 0.1
# (an error with FLOW_REDUCTION_LENGTH, `check_boundary_spec`), above 1 is 1 without it.
function _wel_flowred(spec::BoundarySpec)
    f = spec.auto_flow_reduce
    f <= 0 && return 0.1
    return (f > 1 && !spec.flow_reduction_length) ? 1.0 : f
end

"""
    record_kind(pkg::BoundaryPackage, cv) -> kind

The kind passed to the record formulation of `pkg` (`formulate_record`): a `WelAfr` for a
WEL package with AUTO_FLOW_REDUCE in a model with convertible cells (`cv`, the model's
`ConvertibleNpf`), otherwise `BoundaryKind{K}()`.
"""
record_kind(pkg::BoundaryPackage{K}, cv) where {K} = BoundaryKind{K}()
function record_kind(pkg::BoundaryPackage{:WEL, T}, cv) where {T}
    (pkg.spec.auto_flow_reduce == 0 || isnothing(cv)) && return BoundaryKind{:WEL}()
    s = pkg.spec
    return WelAfr(cv.icelltype, cv.botoff, cv.thk, T(_wel_flowred(s)), s.flow_reduction_length, !isempty(s.auto_flow_reduce_auxname))
end

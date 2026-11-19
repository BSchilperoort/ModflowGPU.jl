# DRN: drains (gwf-drn.f90, `drn_cf`, `drn_ck`, `get_drain_elevations`,
# `get_drain_factor`). `bound[1, i]` is the drain elevation as an offset from the cell
# reference, `bound[2, i]` the conductance (MULTIPLIER applied, `cond_mult`), and
# `bound[3, i]` the drainage depth DDRN from the AUXDEPTHNAME auxiliary variable (0 without
# it). With DDRN ≠ 0 the conductance is scaled linearly between the drain elevation and
# elevation + DDRN (`sQuadraticSaturation` with `eps = 0`, SmoothingFunctions.f90).
# DEV_CUBIC_SCALING (`sQSaturation`) is a developer option and is rejected by the loader.
# With the Newton formulation (T6.2) a package with AUXDEPTHNAME scales with the cubic
# `sQSaturation(..., c1 = −1, c2 = 2)` (`icubic_scaling`) and adds the Jacobian of `drn_fn`
# (record kind `DrnNewton`, `record_kind`).

bound_columns(::BoundaryKind{:DRN}) = ("elev", "cond")
head_columns(::BoundaryKind{:DRN}) = (1,)
mult_column(::BoundaryKind{:DRN}) = 2

function bound_layout(::BoundaryKind{:DRN}, spec::BoundarySpec)
    isempty(spec.auxdepthname) && return Any[("elev", 1), ("cond", 1), 0.0]
    iaux = findfirst(==(uppercase(spec.auxdepthname)), spec.auxnames)
    isnothing(iaux) &&
        throw(MF6InputError(spec.file, 0, "AUXDEPTHNAME was specified as $(spec.auxdepthname) but no AUX variable found with this name."))
    return Any[("elev", 1), ("cond", 1), (:aux, iaux)]
end

# drn_ck: the drain bottom (`get_drain_elevations`) must not be below the bottom of a
# convertible cell (T6.1), the conductance multiplier and the conductance as given not
# negative.
function check_records(::BoundaryKind{:DRN}, pkg, bound, n)
    errors = String[]
    for i in 1:n
        elev, ddrn = bound[1, i], bound[3, i]
        drnbot = ddrn != 0 ? min(elev + ddrn, elev) : elev
        if _convertible_cell(pkg, i)
            bt = _cell_bottom(pkg, i)
            drnbot < bt && push!(
                errors,
                ddrn != 0 ? "SCALED-CONDUCTANCE DRN BOUNDARY ($i) BOTTOM ELEVATION ($drnbot) IS LESS THAN CELL BOTTOM ($bt)" :
                    "DRN BOUNDARY ($i) ELEVATION ($drnbot) IS LESS THAN CELL BOTTOM ($bt)",
            )
        end
        _check_cond_mult!(errors, "DRN", bound, i, 2)
    end
    return _throw_record_errors(pkg, errors)
end

"""
    linear_saturation(top, bot, x) -> y

`sQuadraticSaturation(top, bot, x, eps = 0)` (SmoothingFunctions.f90): 0 below `bot`, 1
above `top`, linear in between; a step at `bot` when `top <= bot`. Device-safe in the
precision of the arguments.
"""
@inline function linear_saturation(top, bot, x)
    b = top - bot
    if b > zero(b)
        x < bot && return zero(b)
        x > top && return one(b)
        # eps = 0: av = 1, y = av·br + ½(1 − av) = br for 0 <= br < 1, and 1 at br = 1.
        return (x - bot) / b
    else
        return x < bot ? zero(b) : one(b)
    end
end

# drn_cf: hcof = −f·C, rhs = −f·C·drnbot with the scaling factor f of get_drain_factor
# (DDRN = 0: f = 0 for h <= elev, else 1).
@inline function formulate_record(::BoundaryKind{:DRN}, i, n, bound, h, area)
    elev = bound[1, i]
    c = bound[2, i]
    ddrn = bound[3, i]
    x = h[n]
    if ddrn != zero(ddrn)
        e = elev + ddrn
        top = max(e, elev)
        bot = min(e, elev)
        f = linear_saturation(top, bot, x)
    else
        bot = elev
        f = x <= bot ? zero(c) : one(c)
    end
    fc = f * c
    return (-fc, -fc * bot)
end

"""
    DrnNewton

The record kind of a DRN package with AUXDEPTHNAME in a model with the Newton formulation
(T6.2; `drn_allocate_scalars` sets `icubic_scaling`): the conductance is scaled with
`sq_saturation(top, bot, h, −1, 2)` between the drain elevation and elevation + DDRN
(`get_drain_factor`), and `drn_fn` adds `S'·C·(drnbot − h)` to the diagonal
(`newton_record`). Records without a drainage depth are as without Newton.
"""
struct DrnNewton end

@inline function formulate_record(::DrnNewton, i, n, bound, h, area)
    elev = bound[1, i]
    c = bound[2, i]
    ddrn = bound[3, i]
    x = h[n]
    T = eltype(bound)
    if ddrn != zero(ddrn)
        e = elev + ddrn
        top = max(e, elev)
        bot = min(e, elev)
        f = sq_saturation(top, bot, x, T(-1), T(2))
    else
        bot = elev
        f = x <= bot ? zero(c) : one(c)
    end
    fc = f * c
    return (-fc, -fc * bot)
end

@inline function newton_record(::DrnNewton, i, n, bound, h, area)
    T = eltype(bound)
    @inbounds begin
        ddrn = bound[3, i]
        ddrn == zero(T) && return zero(T)
        elev = bound[1, i]
        e = elev + ddrn
        top = max(e, elev)
        bot = min(e, elev)
        x = h[n]
        return sq_saturation_derivative(top, bot, x, T(-1), T(2)) * bound[2, i] * (bot - x)
    end
end

# DRN: drains (gwf-drn.f90, `drn_cf`, `drn_ck`, `get_drain_elevations`,
# `get_drain_factor`). `bound[1, i]` is the drain elevation as an offset from the cell
# reference, `bound[2, i]` the conductance (MULTIPLIER applied, `cond_mult`), and
# `bound[3, i]` the drainage depth DDRN from the AUXDEPTHNAME auxiliary variable (0 without
# it). With DDRN ≠ 0 the conductance is scaled linearly between the drain elevation and
# elevation + DDRN (`sQuadraticSaturation` with `eps = 0`, SmoothingFunctions.f90).
# DEV_CUBIC_SCALING (`sQSaturation`) is a developer option and is rejected by the loader.
# The Newton terms (`drn_fn`) arrive with T6.2.

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

# drn_ck: the conductance must not be negative. (The check against the cell bottom applies
# to convertible cells only, T6.1.)
function check_records(::BoundaryKind{:DRN}, pkg, n)
    for i in 1:n
        cond = pkg.bound_host[2, i]
        cond < 0 &&
            throw(MF6InputError(pkg.spec.file, 0, "$(pkg.spec.name): DRN BOUNDARY ($i) CONDUCTANCE ($cond) IS LESS THAN ZERO"))
    end
    return nothing
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

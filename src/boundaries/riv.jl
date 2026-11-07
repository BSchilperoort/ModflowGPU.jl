# RIV: rivers (gwf-riv.f90, `riv_cf`, `riv_ck`). `bound[1, i]` is the river stage and
# `bound[3, i]` the river bottom, both as offsets from the cell reference; `bound[2, i]`
# is the conductance (MULTIPLIER applied to the conductance, `cond_mult`).

bound_columns(::BoundaryKind{:RIV}) = ("stage", "cond", "rbot")
head_columns(::BoundaryKind{:RIV}) = (1, 3)
mult_column(::BoundaryKind{:RIV}) = 2

# riv_ck: the stage must not be below the river bottom, and the conductance not negative.
# (The checks against the cell bottom apply to convertible cells only, T6.1.)
function check_records(::BoundaryKind{:RIV}, pkg, n)
    for i in 1:n
        stage, cond, rbot = pkg.bound_host[1, i], pkg.bound_host[2, i], pkg.bound_host[3, i]
        stage < rbot &&
            throw(MF6InputError(pkg.spec.file, 0, "$(pkg.spec.name): RIV BOUNDARY ($i) RIVER STAGE ($stage) IS LESS THAN RIVER BOTTOM ($rbot)"))
        cond < 0 &&
            throw(MF6InputError(pkg.spec.file, 0, "$(pkg.spec.name): RIV BOUNDARY ($i) CONDUCTANCE ($cond) IS LESS THAN ZERO"))
    end
    return nothing
end

# riv_cf: head above the river bottom: hcof = −C, rhs = −C·stage; at or below it the
# leakage is constant: hcof = 0, rhs = −C·(stage − rbot). In offset form the comparison
# h' <= rbot' and the difference stage' − rbot' are those of MF6.
@inline function formulate_record(::BoundaryKind{:RIV}, i, n, bound, h, area)
    c = bound[2, i]
    stage = bound[1, i]
    rbot = bound[3, i]
    if h[n] <= rbot
        return (zero(c), -c * (stage - rbot))
    else
        return (-c, -c * stage)
    end
end

# RIV: rivers (gwf-riv.f90, `riv_cf`, `riv_ck`). `bound[1, i]` is the river stage and
# `bound[3, i]` the river bottom, both as offsets from the cell reference; `bound[2, i]`
# is the conductance (MULTIPLIER applied to the conductance, `cond_mult`).

bound_columns(::BoundaryKind{:RIV}) = ("stage", "cond", "rbot")
head_columns(::BoundaryKind{:RIV}) = (1, 3)
mult_column(::BoundaryKind{:RIV}) = 2

# riv_ck: the stage must not be below the river bottom, the conductance multiplier and the
# conductance as given not negative, and (convertible cells, T6.1) the river bottom and
# the stage not below the cell bottom. All errors of a block in one message.
function check_records(::BoundaryKind{:RIV}, pkg, bound, n)
    errors = String[]
    for i in 1:n
        stage, rbot = bound[1, i], bound[3, i]
        conv = _convertible_cell(pkg, i)
        bt = conv ? _cell_bottom(pkg, i) : 0.0
        (conv && rbot < bt) && push!(errors, "RIV BOUNDARY ($i) RIVER BOTTOM ($rbot) IS LESS THAN CELL BOTTOM ($bt)")
        stage < rbot && push!(errors, "RIV BOUNDARY ($i) RIVER STAGE ($stage) IS LESS THAN RIVER BOTTOM ($rbot)")
        (conv && stage < bt) && push!(errors, "RIV BOUNDARY ($i) RIVER STAGE ($stage) IS LESS THAN CELL BOTTOM ($bt)")
        _check_cond_mult!(errors, "RIV", bound, i, 2)
    end
    return _throw_record_errors(pkg, errors)
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

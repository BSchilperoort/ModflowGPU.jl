# GHB: general-head boundaries (gwf-ghb.f90, `ghb_cf`). `bound[1, i]` is the boundary head
# as an offset from the cell reference, `bound[2, i]` the conductance (MULTIPLIER applied
# to the conductance, `cond_mult`).

bound_columns(::BoundaryKind{:GHB}) = ("bhead", "cond")
head_columns(::BoundaryKind{:GHB}) = (1,)
mult_column(::BoundaryKind{:GHB}) = 2

# ghb_cf: hcof = −cond, rhs = −cond·bhead (offset form: the flow cond·(bhead' − h') is
# unchanged).
@inline function formulate_record(::BoundaryKind{:GHB}, i, n, bound, h, area)
    c = bound[2, i]
    return (-c, -c * bound[1, i])
end

# ghb_ck (T6.1): the boundary head must not be below the bottom of a convertible cell, the
# conductance multiplier and the conductance as given not negative.
function check_records(::BoundaryKind{:GHB}, pkg, bound, n)
    errors = String[]
    for i in 1:n
        bhead = bound[1, i]
        if _convertible_cell(pkg, i)
            bt = _cell_bottom(pkg, i)
            bhead < bt && push!(errors, "GHB BOUNDARY ($i) HEAD ($bhead) IS LESS THAN CELL BOTTOM ($bt)")
        end
        _check_cond_mult!(errors, "GHB", bound, i, 2)
    end
    return _throw_record_errors(pkg, errors)
end

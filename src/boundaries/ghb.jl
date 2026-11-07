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

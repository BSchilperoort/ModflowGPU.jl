# WEL: specified flows (gwf-wel.f90, `wel_cf`). AUTO_FLOW_REDUCE needs convertible cells
# and is rejected (`check_boundary_spec`) until T6.1.

bound_columns(::BoundaryKind{:WEL}) = ("q",)

# wel_cf: hcof = 0, rhs = −q (q > 0 is injection).
@inline formulate_record(::BoundaryKind{:WEL}, i, n, bound, h, area) = (zero(eltype(bound)), -bound[1, i])

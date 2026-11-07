# RCH: recharge (gwf-rch.f90, `rch_cf`), list (RCH) and array (RCHA, READASARRAYS) input.
#
# `bound[1, i]` is the recharge rate per unit area (MULTIPLIER applied on the host); the
# formulation multiplies by the cell area, as MF6 >= 6.4 does (6.3 multiplied list rates
# by the area when reading). Without FIXED_CELL a record whose cell has IBOUND 0 moves to
# the highest active cell below (`apply_period!`, `highest_active`). Array cells that are
# not in the reduced grid (IDOMAIN <= 0) get node 0 and no recharge, as in MF6
# (`nlarray_to_nodelist` → `get_nodenumber` <= 0 → `rch_cf` skips them).

bound_columns(::BoundaryKind{:RCH}) = ("recharge",)

# rch_cf: hcof = 0, rhs = −recharge·area.
@inline formulate_record(::BoundaryKind{:RCH}, i, n, bound, h, area) = (zero(eltype(bound)), -bound[1, i] * area[n])

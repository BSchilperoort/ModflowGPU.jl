# EVT: evapotranspiration (gwf-evt.f90, `evt_cf`, `check_pxdp`, `evt_source_options`,
# `evt_source_dimensions`), list (EVT, NSEG >= 1, SURF_RATE_SPECIFIED) and array (EVTA,
# READASARRAYS) input.
#
# Columns of `bound`: 1 the ET surface as an offset from the cell reference, 2 the maximum
# rate per unit area (MULTIPLIER applied on the host; MF6 multiplies `rate·area·mult` in
# `evt_cf`), 3 the extinction depth, 4 PETM0 (1 without SURF_RATE_SPECIFIED, which gives
# MF6's `petm1 = DONE` and the constant rate `c` above the surface), then the NSEG − 1
# PXDP and the NSEG − 1 PETM values. NSEG is recovered in the kernel from the number of
# columns, `nseg = (ncol − 2) ÷ 2`. Without FIXED_CELL a record whose cell has IBOUND 0
# moves to the highest active cell below (`apply_period!`, `highest_active`).

bound_columns(::BoundaryKind{:EVT}) = ("surface", "rate", "depth")
head_columns(::BoundaryKind{:EVT}) = (1,)
mult_column(::BoundaryKind{:EVT}) = 2

function bound_layout(::BoundaryKind{:EVT}, spec::BoundarySpec)
    nseg = Int(spec.nseg)
    layout = Any[("surface", 1), ("rate", 1), ("depth", 1), spec.surf_rate_specified ? ("petm0", 1) : 1.0]
    append!(layout, [("pxdp", j) for j in 1:(nseg - 1)])
    append!(layout, [("petm", j) for j in 1:(nseg - 1)])
    return layout
end

# evt_source_options / evt_source_dimensions.
function _check_evt_spec(spec::BoundarySpec)
    spec.nseg < 1 && throw(MF6InputError(spec.file, 0, "Error: In EVT, NSEG must be > 0 but is specified as $(spec.nseg)"))
    if spec.readasarrays
        spec.surf_rate_specified &&
            throw(MF6InputError(spec.file, 0, "READASARRAYS option is not compatible with the SURF_RATE_SPECIFIED option."))
        spec.nseg > 1 &&
            throw(MF6InputError(spec.file, 0, "In the EVT package, NSEG cannot be greater than 1 when READASARRAYS is used."))
    end
    return nothing
end

# check_pxdp: with NSEG > 1 every PXDP is strictly between 0 and 1 and the PXDP values of a
# record increase monotonically.
function check_records(::BoundaryKind{:EVT}, pkg, bound, n)
    nseg = Int(pkg.spec.nseg)
    nseg > 1 || return nothing
    for i in 1:n
        pxdp1 = 0.0
        for j in 1:(nseg - 1)
            pxdp2 = bound[4 + j, i]
            (pxdp2 <= 0 || pxdp2 >= 1) &&
                throw(MF6InputError(pkg.spec.file, 0, "$(pkg.spec.name): PXDP must be between 0 and 1.  Found $pxdp2 for record $i"))
            pxdp2 < pxdp1 &&
                throw(MF6InputError(pkg.spec.file, 0, "$(pkg.spec.name): PXDP is not monotonically increasing for record $i"))
            pxdp1 = pxdp2
        end
    end
    return nothing
end

# evt_cf, with c = rate·area (the multiplier is in the rate): head at or above the surface:
# constant ET, rhs = PETM0·c; depth below the surface d >= extinction depth x: no ET;
# otherwise one segment: hcof = −c/x, rhs = c − c·s/x, and NSEG > 1: the segment
# containing d, hcof = −(petm1 − petm2)·c/((pxdp2 − pxdp1)·x),
# rhs = hcof·(s − pxdp1·x) + petm1·c. All heads and the surface are offsets; s − h and
# hcof·(s' − pxdp1·x) + petm1·c are those of MF6 (rhs' = rhs − hcof·r).
@inline function formulate_record(::BoundaryKind{:EVT}, i, n, bound, h, area)
    s = bound[1, i]
    c = bound[2, i] * area[n]
    x = bound[3, i]
    petm0 = bound[4, i]
    hn = h[n]
    z = zero(c)
    hn >= s && return (z, petm0 * c)
    d = s - hn
    d < x || return (z, z)
    nseg = (size(bound, 1) - 2) ÷ 2
    if nseg > 1
        pxdp1 = z
        petm1 = petm0
        pxdp2 = one(c)
        petm2 = z
        for iseg in 1:nseg
            if iseg < nseg
                pxdp2 = bound[4 + iseg, i]
                petm2 = bound[3 + nseg + iseg, i]
            else
                pxdp2 = one(c)
                petm2 = z
            end
            d <= pxdp2 * x && break
            pxdp1 = pxdp2
            petm1 = petm2
        end
        thcof = -(petm1 - petm2) * c / ((pxdp2 - pxdp1) * x)
        return (thcof, thcof * (s - pxdp1 * x) + petm1 * c)
    else
        return (-c / x, c - c * s / x)
    end
end

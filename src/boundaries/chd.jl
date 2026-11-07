# CHD: constant heads (gwf-chd.f90, MF6 >= 6.4; see docs/mf6_notes.md).
#
# A CHD cell keeps its row in the system but is not an unknown: `chd_rp` sets its IBOUND
# to a negative value, `chd_ad` sets its head, `chd_fc` adds nothing, and the solution
# replaces its row by `h_n = head` (`sln_ls`); active neighbours see it as a fixed head.
# Its flow comes from the connection flows after the solve (`calc_chd_rate`).

bound_columns(::BoundaryKind{:CHD}) = ("head",)
head_columns(::BoundaryKind{:CHD}) = (1,)

# chd_fc adds nothing (and CHD cells have IBOUND < 0, so this is never reached).
@inline formulate_record(::BoundaryKind{:CHD}, i, n, bound, h, area) = (zero(eltype(bound)), zero(eltype(bound)))

@kernel function chd_heads_kernel!(h, @Const(nodelist), @Const(bound))
    i = @index(Global)
    @inbounds begin
        n = nodelist[i]
        n > 0 && (h[n] = bound[1, i])
    end
end

"""
    apply_chd_heads!(h, pkg::BoundaryPackage{:CHD}) -> h

Set the offset head of every constant-head cell to its CHD head (`chd_ad`: `xnew = xold
= head`; call it for the old heads too).
"""
function apply_chd_heads!(h, pkg::BoundaryPackage{:CHD})
    pkg.nbound == 0 && return h
    chd_heads_kernel!(get_backend(h))(h, pkg.nodelist, pkg.bound; ndrange = pkg.nbound)
    return h
end

@kernel function chd_rate_kernel!(simvals, ratein, rateout, flowja, @Const(nodelist), @Const(ia), @Const(ja), @Const(ibound))
    i = @index(Global)
    T = eltype(simvals)
    @inbounds begin
        n = nodelist[i]
        rate = zero(T)
        qin = zero(T)
        qout = zero(T)
        if n > 0
            for ipos in (ia[n] + Int32(1)):(ia[n + 1] - Int32(1))
                q = flowja[ipos]
                rate -= q
                # chin/chout only over connections to active cells.
                if ibound[ja[ipos]] > 0
                    if q < zero(T)
                        qin -= q
                    else
                        qout += q
                    end
                end
            end
            flowja[ia[n]] += rate
        end
        simvals[i] = rate
        ratein[i] = qin
        rateout[i] = qout
    end
end

"""
    calc_chd_rate!(pkg::BoundaryPackage{:CHD}, flowja, ia, ja, ibound) -> pkg

The CHD flows after all connection flows are in `flowja` (`calc_chd_rate`): per record,
`simvals = −Σ flowja` over the cell's connections (the net flow from the boundary into
the model), `ratein` and `rateout` its split over the connections to active cells (the
listing's CHD in/out), and the rate is added to `flowja` at the diagonal `ia[n]`. `ia`,
`ja` are MF6's CSR arrays (diagonal first), `ibound` as from `boundary_ibound`. Each CHD
cell is in one record only, so the diagonal update needs no atomics.
"""
function calc_chd_rate!(pkg::BoundaryPackage{:CHD}, flowja, ia, ja, ibound)
    pkg.nbound == 0 && return pkg
    chd_rate_kernel!(get_backend(flowja))(pkg.simvals, pkg.ratein, pkg.rateout, flowja, pkg.nodelist, ia, ja, ibound; ndrange = pkg.nbound)
    return pkg
end

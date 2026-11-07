# Time discretization (tdis.f90 `tdis_set_timestep`, `tdis_set_delt`), on the host in
# Float64 (PLAN.md section 3.7, rule 4).

"""
    TimeStep

One time step of the simulation (`tdis_set_timestep`): `kper`, `kstp`, the step length
`delt`, the time since the start of the period `pertim` and of the simulation `totim`
(both at the end of the step), and `endofperiod`. All times are Float64 on the host.
"""
struct TimeStep
    kper::Int
    kstp::Int
    delt::Float64
    pertim::Float64
    totim::Float64
    endofperiod::Bool
end

"""
    time_steps(tdis::TdisSpec) -> Vector{TimeStep}

Every time step of the simulation as MF6 computes them (`tdis_set_delt`,
`tdis_set_timestep`; no ATS): the first step of a period is `perlen/nstp`, or
`perlen·(1 − tsmult)/(1 − tsmult^nstp)` with TSMULT ≠ 1; later steps are `tsmult·delt`,
except the last one of a period, `topertim + perlen − totim` (so the period ends exactly
at `perlen`); `pertim` is set to `perlen` at the end of each period.
"""
function time_steps(tdis::TdisSpec)
    isempty(tdis.ats_file) || throw(MF6UnsupportedError(tdis.file, "unsupported: ATS6 (adaptive time stepping) in TDIS"))
    steps = TimeStep[]
    totim = 0.0
    delt = 0.0
    for kper in 1:nper(tdis)
        perlen, nstp, mult = tdis.perlen[kper], Int(tdis.nstp[kper]), tdis.tsmult[kper]
        topertim = totim
        pertim = 0.0
        for kstp in 1:nstp
            if kstp == 1
                # MF6 divides by float(nstp) (default real); exact for any realistic nstp.
                delt = mult != 1.0 ? perlen * (1.0 - mult) / (1.0 - mult^nstp) : perlen / Float64(Float32(nstp))
            elseif kstp == nstp
                delt = topertim + perlen - totim
            else
                delt = mult * delt
            end
            totim += delt
            pertim += delt
            eop = kstp == nstp
            eop && (pertim = perlen)
            push!(steps, TimeStep(kper, kstp, delt, pertim, totim, eop))
        end
    end
    return steps
end

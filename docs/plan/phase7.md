# Phase 7: beyond MF6 parity and 30″ (design notes first; build after Milestone B)

Task cards for this phase of [PLAN.md](../../PLAN.md). Read PLAN.md sections 1–4 and 6 for the goals, conventions, architecture and validation rules these cards refer to.

Size: S ≈ < 1 day of agent work, M ≈ 1–3 days, L ≈ > 3 days.
`‖` marks cards that can run in parallel with their siblings.

**T7.1 Longitude wrap-around for global grids (M).** Connect column 1 and
column `ncol` with a periodic connection, using the T6.6 self-exchange.
Validate against mf6 with an equivalent `GWF6-GWF6` self-exchange. Also remove
ModflowCoupling's limitation "a domain that crosses the edge of the input maps
cannot be cut". Open question for the owner (PLAN.md section 8): `ex-gwf-spbc` needs
only a self-exchange of one model, so the wrap could be a periodic connection in the
DIS connectivity and would then not wait for T6.6.

**T7.2 Spherical geometry (M).** Row-dependent cell width and area (true
geometry instead of RCH/EVT area-factor scaling). This deviates from MF6 DIS;
validate against an mf6 DISU model with the same geometry.

**T7.3 Regional 30″ on one GPU (M).** Depends: T4.3, T4.4. Run the
`regional30s` synthetic model (and, if the owner provides 30″ Indus inputs, the
real Indus at 30″) as steady state and daily transient, in `:single` and
`:mixed`. Report memory, AMG
iteration counts versus the 5′ model (grid-independence check), and time per
daily step. This is the first Milestone D result and needs no extra hardware.

**T7.4 30″ design note (S).** Depends: T7.3. Before building multi-GPU, write
`docs/design_30s.md`. It covers: memory per active node measured in T7.3, scaled
to global 30″ (starting point from T3.4, `docs/scaling.md`: 189 GiB in `:single` and
300 GiB in `:double` for the model arrays alone, without the preconditioner); the index
type (3.9 G CSR entries exceed `Int32`, so one global model needs `Int64` row pointers,
while partitions may stay within `Int32`); minimum GPU count and memory per GPU; partitioning (latitude
bands vs land-balanced blocks, given that ~70 % of the grid is ocean);
communication volume per iteration; and whether the coupling with mGV can be
partitioned the same way. End with a go/no-go recommendation for T7.5 and the
hardware it needs.

**T7.5 Multi-GPU domain decomposition (L).** Depends: T7.4 (go). MPI.jl,
partitions from T7.4, halo exchange of heads, per-partition AMG inside a
block-Jacobi/additive Schwarz preconditioner, and global reductions. Develop and
test on this machine with several MPI ranks sharing the one GPU (and on the CPU
backend). The decomposed run must match the single-domain run on `regional30s`
within 1e-8 m. Real multi-GPU scaling runs need hardware that this machine does
not have.

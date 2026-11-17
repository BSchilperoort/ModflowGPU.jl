# Phase 7: beyond MF6 parity (build after Milestone B)

Task cards for this phase of [PLAN.md](../../../PLAN.md). Read PLAN.md sections 1–4 and 6 for the goals, conventions, architecture and validation rules these cards refer to.

Size: S ≈ < 1 day of agent work, M ≈ 1–3 days, L ≈ > 3 days.
`‖` marks cards that can run in parallel with their siblings.

Scope (owner, 2026-11-16, PLAN.md section 8 decision 11): spherical geometry (T7.2) and
multi-GPU domain decomposition (T7.5) stay. Longitude wrap-around (T7.1) and the 30″
goals (T7.3, T7.4) are dropped. The card numbers are kept so that older reports and
merge messages still point at the right card.

**T7.1 Longitude wrap-around for global grids: dropped (2026-11-16).** Very little
land crosses the date line, so columns 1 and `ncol` stay unconnected. ModflowCoupling's
limitation "a domain that crosses the edge of the input maps cannot be cut" stays too.
The self-exchange of `ex-gwf-spbc` is still part of T6.6 (Milestone C); it is just no
longer the base for a wrap.

**T7.2 Spherical geometry (M).** Row-dependent cell width and area (true
geometry instead of RCH/EVT area-factor scaling). This deviates from MF6 DIS;
validate against an mf6 DISU model with the same geometry (DISU is supported since
T6.4).

**Docs:** update `docs/src/compatibility/differences.md` (spherical geometry deviates from MF6 DIS) and `compatibility/packages.md`; no gallery example (no MF6 counterpart).

**T7.3 Regional 30″ on one GPU: dropped (2026-11-16).** 30″ is not relevant for now.
The `regional30s` and `global30s` synthetic sizes (T3.4) and the 30″ memory estimate
in `docs/dev/scaling.md` stay as they are, for reference.

**T7.4 30″ design note: dropped (2026-11-16).** The design questions that multi-GPU
needs at 5′ (partitioning, communication volume, coupling with mGV) move into T7.5.

**T7.5 Multi-GPU domain decomposition (L).** Depends: T4.3, T4.4 (no longer on T7.3
or T7.4). Target: the global 5′ model, not 30″.
- First, a short design section in `docs/design_multigpu.md` (part of this card, no
  separate go/no-go): partitioning (latitude bands vs land-balanced blocks, given
  that ~70 % of the grid is ocean and everything north of 60°N is masked),
  communication volume per iteration, whether the coupling with mGV can be
  partitioned the same way, and what speed-up to expect at 5′, where one GPU already
  holds the whole model.
- Deliver: MPI.jl, partitions from the design section, halo exchange of heads, the
  default preconditioner (MC-ILU0, or AMG) per partition inside a block-Jacobi /
  additive Schwarz preconditioner, and global reductions. Index types stay `Int32`.
- Develop and test on this machine with several MPI ranks sharing the one GPU (and on
  the CPU backend). The decomposed run must match the single-domain run on the
  synthetic `global5m` model within 1e-8 m in `:double`, and the iteration counts
  against the number of partitions are reported. Real multi-GPU scaling runs need
  hardware that this machine does not have.

**Docs:** update `docs/src/gpu.md` (running on several GPUs) and `docs/src/performance.md` (scaling numbers); no gallery example.

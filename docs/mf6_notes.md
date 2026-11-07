# Notes on MF6 behaviour

Findings from reading the MF6 Fortran (`reference/modflow6`, 6.8.2 source) that the port
depends on. Each section names the routines it is based on.

## Constant heads (CHD), MF6 >= 6.4

Routines: `chd_rp`, `chd_ad`, `chd_fc`, `chd_cq`, `calc_chd_rate`, `chd_bd`
(`gwf-chd.f90`); `sln_ls` (`NumericalSolution.f90`); `default_flow_cq`, `hcond`/`vcond`
(`gwf-npf.f90`).

Since 6.4 a CHD cell is no longer an extra equation with a large diagonal. It keeps its
row in the system but is not an unknown:

1. **IBOUND.** `chd_rp` first resets the cells of the previous period's records to
   `ibound = ibcnum` (active), then reads the new period data and sets
   `ibound(node) = -ibcnum` for each record. A cell that is already negative (in two CHD
   records, or in two CHD packages) is an error ("Cell is already a constant head").
2. **Head.** `chd_ad` sets `xnew(node) = xold(node) = head` (with the MULTIPLIER aux
   applied, `head_mult`) at the start of every time step.
3. **Formulate.** `chd_fc` adds nothing. NPF (`npf_fc`) still adds the connection
   conductances to the CHD row and to its active neighbours' rows; other boundary
   packages skip cells with `ibound <= 0` in their `cf` routines.
4. **Row replacement.** In `sln_ls`, for every row with `active <= 0` (CHD and no-flow
   alike): diagonal = 1, off-diagonals zeroed (`zero_row_offdiag`), `rhs = x`. So the CHD
   row reads `h_n = head`. With a symmetric matrix (`isymmetric == 1`, CG) in sequential
   mode, each active row's entries in CHD columns are moved to the right-hand side
   (`rhs -= a_nm * x_m`, `a_nm = 0`) to keep the matrix symmetric. Active neighbours
   therefore see the CHD cell as a fixed head either way.
5. **Flows.** NPF computes `flowja` for every connection from the heads (the conductance
   is zero when either cell is inactive, so CHD-to-active connections carry flow).
   `calc_chd_rate` (called from `chd_cq` and again from `chd_bd`, after exchanges have
   finished `flowja`) then, per record: `rate = -Σ flowja(ipos)` over the cell's
   off-diagonal connections (the net flow from the boundary into the model, stored in
   `simvals` and the CBC); `ratechdin`/`ratechdout` split this over connections to
   **active** neighbours only (`ibound(n2) > 0`), for the budget table; `rhs(i) = -rate`,
   `hcof(i) = 0`; and `flowja(idiag) += rate`.

**Port (T2.3).** `boundary_ibound` sets IBOUND -1 for CHD cells (error on a duplicate);
`apply_chd_heads!` writes the CHD heads (as offsets) into the head vector;
the CHD formulate is a no-op and the generic formulate kernel skips `ibound <= 0`; the
row replacement belongs to the assembly (T2.6); `calc_chd_rate!` ports `calc_chd_rate`
(`simvals`, `ratein`, `rateout`, diagonal of `flowja`). It does not write `rhs = -rate`,
which in MF6 only feeds observations and the MVR. Verified against MF6's CBC CHD records on
ex-gwf-toth and ex-gwf-hanic (`test/parity/boundaries_test.jl`).

**Port (T2.6).** `assemble!` (src/solver/csr.jl) applies `sln_ls`: rows with
`ibound <= 0` become identity rows with `rhs = h'` (the offset CHD head), active rows
with a diagonal below 1e-15 in magnitude get diagonal −1 and `rhs −= h'`, and with
`symmetric = true` (IMS `LINEAR_ACCELERATION CG`, sequential mode) the CHD columns of
active rows are moved to the right-hand side. The off-diagonals of all other rows keep
the coupling to CHD cells (`hcond`/`vcond` are zero only for `ibound == 0`). The
flux-form residual is zero on CHD rows. Verified bitwise (`:double`) against libmf6's
`SLN_1/AMAT` after the first solve on toth (CG), hanic/hanir (BiCGSTAB), fhb_const (CG)
and coupling_rde (`test/parity/assembly_test.jl`). Note: BMI `get_time_step` does not
return the step length; read `TDIS/DELT`.

## Other boundary details used in T2.3

- **MULTIPLIER aux / AUXMULTNAME** (`bnd_rp`, `BndExt`): the aux column named by
  `AUXMULTNAME` multiplies one `bound` column: CHD head, WEL rate, GHB conductance, RCH
  rate. The port applies it on the host in Float64 before head columns become offsets.
- **WEL `AUTO_FLOW_REDUCE`** (`wel_cf`) reduces the rate in partly dry convertible cells;
  rejected (`MF6UnsupportedError`) until convertible cells exist (T6.1).
- **RCH without FIXED_CELL** (`rch_cf`): when a record's cell has `ibound == 0`, the
  recharge goes to the highest active cell below in the same column; if none is active it
  stays on the bottom cell (and is then skipped). With FIXED_CELL nothing moves. RCHA
  (READASARRAYS) has one record per column (NCPL), with IRCH selecting the layer.
- **Area.** RCH rates are per unit area; `rch_cf` multiplies by the cell area
  (`dis%get_area`). GHB `rhs = -C * bhead`, `hcof = -C` (`ghb_cf`); WEL `rhs = -q`
  (`wel_cf`).

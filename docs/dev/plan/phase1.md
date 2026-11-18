# Phase 1: MF6 input and output (host only)

Task cards for this phase of [PLAN.md](../../../PLAN.md). Read PLAN.md sections 1–4 and 6 for the goals, conventions, architecture and validation rules these cards refer to.

Size: S ≈ < 1 day of agent work, M ≈ 1–3 days, L ≈ > 3 days.
`‖` marks cards that can run in parallel with their siblings.

**T1.1 Tokenizer and block parser (M)**
- Depends: T0.1, T0.2.
- Read: MF6 IO guide (`mf6io.pdf` in the release, "Input Instructions" chapter),
  `src/Utilities/BlockParser.f90`, DFN files.
- Deliver: `src/input/tokenizer.jl`, `blocks.jl`: case-insensitive keywords;
  comments `#`, `!`, `//`; quoted strings; `BEGIN name [n] … END name`; repeated
  period blocks; `OPEN/CLOSE` for whole-block external files.
- Use the DFN files (`reference/dfns/*.dfn`) to drive a schema: block names,
  variable names, types, shapes, optional flags. Generate Julia definitions
  from the DFNs with a script committed under `src/input/dfn/`, rather than
  hand-writing each package's options.
- Done when: every input file of every Tier 0–7 example tokenises into blocks
  without error (test loops over `reference/examples`, skipped when missing).

**T1.2 Array input (S)**
- Depends: T1.1.
- Deliver: `src/input/arrays.jl`: `CONSTANT`, `INTERNAL [FACTOR f] [IPRN n]`,
  `OPEN/CLOSE file [FACTOR f] [(BINARY)]`, `LAYERED`, shapes from DIS/DISV/DISU
  dimensions, integer and double arrays.
- Done when: unit tests cover each form, and the binary form round-trips through
  a writer.

**T1.3 List and period input (M)**
- Depends: T1.1, T1.2.
- Deliver: `src/input/lists.jl`: cellid forms (`lay row col`, `lay cell2d`,
  `node`), `AUXILIARY`, `BOUNDNAMES`, `OPEN/CLOSE` list files, and period
  semantics: data persists until the next `PERIOD` block, an empty block clears
  it, and `READASARRAYS` (`RCHA`, `EVTA`). Time-series names in place of
  numbers are kept as symbols for T6.3.
- Done when: the period data of advtidal, twri01 and fhb match hand-checked
  values in tests.

**T1.4 Simulation loader → `SimulationSpec` (M)**
- Depends: T1.3.
- Deliver: `src/spec/*.jl` (plain host structs: `SimulationSpec`, `TdisSpec`,
  `GwfSpec`, `DisSpec`/`DisvSpec`/`DisuSpec`, `NpfSpec`, `StoSpec`, `IcSpec`,
  `BoundarySpec{Kind}`, `ImsSpec`, `OcSpec`, `ExchangeSpec`) and
  `src/input/simulation.jl` reading `mfsim.nam`, TDIS, model name files, and
  packages. Unsupported packages raise an error naming the package. Options that
  are parsed but not implemented raise an error, not a silent ignore.
- Done when: all Tier 0–2 examples load. Higher tiers load or fail with
  "unsupported: <thing>".

**T1.5 Binary output readers/writers (S) ‖**
- Depends: T0.1.
- Read: MF6 IO guide, "Description of Binary Output Files"; flopy's
  `binaryfile` reader for reference.
- Deliver: `src/output/hds.jl`, `cbc.jl`: read and write head (DIS, DISV, DISU
  layouts) and budget files (including `FLOW-JA-FACE`, imeth 1/6 records with
  aux), and a listing-file budget table parser (port `listing_rates` from
  ModflowCoupling's tests).
- Done when: reading the reference runs of T0.3 works for all tiers, and our
  writer → reader round-trip is exact.

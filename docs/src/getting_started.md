# Getting started

```@meta
CurrentModule = ModflowGPU
DocTestSetup = :(using ModflowGPU)
```

This page runs a small MODFLOW 6 simulation with ModflowGPU, reads its results back, and
shows the options that matter most. Everything on it runs on the CPU, so you can follow it
right after [Installation](install.md), without a GPU and without downloading a model. The
same steps work for your own simulation: replace the folder by the one that holds your
`mfsim.nam`.

The code blocks are meant to be typed into the Julia REPL (started with
`julia --project=.` in the folder of your environment). Lines starting with `julia>` are
input; the lines below them are what Julia prints.

## A small model to run

ModflowGPU reads ordinary MODFLOW 6 input files. To have a simulation to run, we build
one in Julia with the [builder functions](api/builder.md) and write it as MF6 input files
with [`write_simulation`](@ref). If you already have a simulation folder, skip to the
next section.

The model is a confined aquifer of 20 columns, 10 rows and 2 layers of 100 m × 100 m
cells: an upper layer from 0 to −10 m with K = 10 m/d and a lower layer from −10 to −50 m
with K = 1 m/d. A river or lake on the west edge holds the head at 0 m (CHD in column 1
of layer 1), 1 mm/d of recharge falls on the other cells of layer 1, and a well pumps
500 m³/d from layer 2. One steady-state stress period.

```jldoctest gs
julia> using ModflowGPU

julia> botm = cat(fill(-10.0, 20, 10), fill(-50.0, 20, 10); dims = 3);  # (ncol, nrow, nlay)

julia> dis = build_dis(; delr = 100.0, delc = 100.0, top = 0.0, botm, length_units = :meters);

julia> npf = build_npf(dis; k = [10.0, 1.0]);  # one value per layer

julia> chd = build_boundary(:CHD, dis;
           periods = [1 => (; cells = [(1, i, 1) for i in 1:10], values = zeros(10))]);

julia> rch = build_boundary(:RCH, dis;
           periods = [1 => (; cells = [(1, i, j) for i in 1:10 for j in 2:20], values = fill(1.0e-3, 190))]);

julia> wel = build_boundary(:WEL, dis; periods = [1 => (; cells = [(2, 5, 15)], values = [-500.0])]);

julia> oc = build_oc(; head_file = "flow.hds", budget_file = "flow.cbc",
           periods = [1 => ["SAVE HEAD ALL", "SAVE BUDGET ALL", "PRINT BUDGET ALL"]]);

julia> gwf = build_model("flow"; dis, npf, ic = build_ic(dis; strt = 0.0), oc, boundaries = [chd, wel, rch],
           save_flows = true);

julia> sim = build_simulation(gwf; tdis = build_tdis(; perlen = 1.0, time_units = :days));

julia> simdir = write_simulation(joinpath(mktempdir(), "flow"), sim);

julia> readdir(simdir)
11-element Vector{String}:
 "flow.chd-1"
 "flow.dis"
 "flow.ic"
 "flow.ims"
 "flow.nam"
 "flow.npf"
 "flow.oc"
 "flow.rch-1"
 "flow.tdis"
 "flow.wel-1"
 "mfsim.nam"
```

Cells are given as MF6 cell ids `(layer, row, column)`. Grid arrays such as `botm` have
the shape `(ncol, nrow, nlay)`, the order in which MF6 stores them. `save_flows = true` is
the SAVE_FLOWS option of the model name file: without it, as in MODFLOW 6, no flows are
written to the budget file. `simdir` is now an
ordinary MF6 simulation folder; `mf6` runs it too.

## Run the simulation

[`run_simulation`](@ref) reads `mfsim.nam` and the files it lists, runs every time step,
and writes the output that the OC package asks for into the same folder:

```jldoctest gs
julia> result = run_simulation(simdir);
┌ Info: IMS preconditioner ILU0 is replaced by mc_ilu0 (multicolour ILU0: red–black on DIS grids, greedy colouring on DISV and DISU; the default, see the `preconditioner` option)
└   ims.file = "flow.ims"

julia> result.converged
true

julia> filter(f -> endswith(f, r"\.(hds|cbc|lst)"), readdir(simdir))
3-element Vector{String}:
 "flow.cbc"
 "flow.hds"
 "flow.lst"
```

The first run takes a while (about 20 s on a desktop CPU), because Julia compiles the code
first; a second run in the same Julia session takes less than a second. The `Info`
message says that the linear-solver preconditioner of IMS was replaced; see
[Options](@ref) below.

The output files have the names given in the input:

- `flow.hds`, the **head file**, and `flow.cbc`, the **budget file**, as asked by the
  `SAVE HEAD` and `SAVE BUDGET` records of OC. They are MF6 binary files in the same
  format as MODFLOW 6 writes.
- `flow.lst`, the **listing file** of the model, with the volumetric budget tables (as
  asked by `PRINT BUDGET`) and the iteration counts of each time step. It is shorter than
  MODFLOW 6's listing file: it does not echo the input.

To write the output somewhere else, pass `outdir = "path/to/output"`; to write nothing,
`write_output = false`.

## Read the results

### In Julia

The [`RunResult`](@ref) that `run_simulation` returns already holds the final heads and the
budget of every time step:

```jldoctest gs
julia> length(result.heads)   # one head per active cell, in MF6's node order
400

julia> for t in result.budgets[end].terms   # the budget of the last time step
           println(rpad(t.package, 7), "in: ", round(t.rate_in; digits = 1), "  out: ", round(t.rate_out; digits = 1))
       end
WEL-1  in: 0.0  out: 500.0
RCH-1  in: 1900.0  out: 0.0
CHD-1  in: 0.0  out: 1400.0

julia> abs(percent_discrepancy(result.budgets[end])) < 0.01
true
```

Rates are in the model's units, here m³/d. The output files can be read with the
[readers](api/output.md) of ModflowGPU, which read MODFLOW 6's own output files as well.
A head file holds one record per saved time step and layer:

```jldoctest gs
julia> hf = open_head_file(joinpath(simdir, "flow.hds"));

julia> h = read_head(hf, length(hf));   # the last saved heads

julia> size(h)   # (ncol, nrow, nlay)
(20, 10, 2)

julia> round(h[15, 5, 2]; digits = 2)   # the well cell: column 15, row 5, layer 2
5.9

julia> round(maximum(h); digits = 2)    # the highest head, at the east edge
8.69
```

Note the order of the indices: `h[column, row, layer]`. The budget file has one record
per budget term and saved time step:

```jldoctest gs
julia> bf = open_budget_file(joinpath(simdir, "flow.cbc"));

julia> budget_texts(bf)
4-element Vector{String}:
 "FLOW-JA-FACE"
 "WEL"
 "RCH"
 "CHD"

julia> wel = read_budget(bf; text = "WEL")[end];   # the last saved time step

julia> wel.id1, wel.q   # cell (node number) and flow, negative out of the model
(Int32[295], [-500.0])
```

The budget tables of the listing file can be read too, with
[`read_listing_budgets`](@ref):

```jldoctest gs
julia> tables = read_listing_budgets(joinpath(simdir, "flow.lst"));

julia> [(t.name, round(t.rate_in), round(t.rate_out)) for t in tables[end].terms]   # m³/d
3-element Vector{Tuple{String, Float64, Float64}}:
 ("WEL", 0.0, 500.0)
 ("RCH", 1900.0, 0.0)
 ("CHD", 0.0, 1400.0)
```

### In Python with FloPy

The head and budget files are MF6 binary files, so
[FloPy](https://github.com/modflowpy/flopy) reads them as it reads MODFLOW 6's:

```python
import flopy

hds = flopy.utils.HeadFile("flow/flow.hds")
h = hds.get_data()                    # last saved heads, shape (nlay, nrow, ncol)

cbc = flopy.utils.CellBudgetFile("flow/flow.cbc")
wel = cbc.get_data(text="WEL")[-1]    # record array with node, node2 and q
```

FloPy's arrays are `(nlay, nrow, ncol)` with indices from 0, so Julia's `h[15, 5, 2]` is
FloPy's `h[1, 4, 14]`.

## Options

The keywords of [`run_simulation`](@ref) (and of [`initialize`](@ref) for the stepping API)
that you are most likely to need:

| Keyword | Default | What it does |
| --- | --- | --- |
| `backend` | the CPU | where the model runs: `select_backend("auto")` for the first GPU that works |
| `precision` | `:auto` | `:double`, `:mixed` or `:single`; `:auto` is `:mixed` where the backend has Float64, else `:single` |
| `preconditioner` | `:auto` (= `:mc_ilu0`) | the preconditioner that replaces IMS's ILU0, MILU0 or ILUT: `:mc_ilu0`, `:amg`, `:jacobi`, `:mc_sgs`, `:chebyshev` |
| `precond_options` | none | settings of the preconditioner, e.g. `(; relax = 0.97)` for MILU with `:mc_ilu0` |
| `outdir` | the input folder | where the output files go |
| `write_output` | `true` | `false` writes no output files |
| `closures` | the IMS values | override IMS closures, e.g. `(; outer_dvclose = 1e-4, inner_maximum = 200)` |

Note that the default backend is the CPU: to run on a GPU, pass
`backend = select_backend("auto")` (after loading the GPU package, see the
[GPU guide](gpu.md)).

The same model in single precision, without writing output:

```jldoctest gs
julia> backend = select_backend("cpu");   # select_backend("auto") to use a GPU

julia> r32 = run_simulation(simdir; backend, precision = :single, write_output = false);
┌ Info: IMS preconditioner ILU0 is replaced by mc_ilu0 (multicolour ILU0: red–black on DIS grids, greedy colouring on DISV and DISU; the default, see the `preconditioner` option)
└   ims.file = "flow.ims"

julia> r32.T, r32.Ts, r32.Tp   # state, linear solver, preconditioner
(Float32, Float32, Float32)

julia> maximum(abs.(r32.heads .- result.heads)) < 1e-4   # metres
true
```

### Which options should I pick?

Start with the defaults: `precision = :auto` and `preconditioner = :auto`. On a GPU with
Float64 (AMD, NVIDIA) and on the CPU that is `:mixed` with `:mc_ilu0`, which gives
MODFLOW 6's accuracy and was the fastest combination with that accuracy on every benchmark
model. Change them only for a reason:

| If ... | then use | because |
| --- | --- | --- |
| you want the fastest transient run and can accept heads about 1e-4 m less accurate | `precision = :single` | about 2.5 times faster than `:mixed` on a 6.6 million cell model, about 30 % less memory |
| the GPU has no Float64 (Apple, some Intel) | the default (`:auto` gives `:single`) | `:double` and `:mixed` cannot run there |
| you run a long steady state (spin-up) | the default `:mixed`, not `:single` | in `:single` such steady states can stop centimetres short of the solution |
| a time step does not converge (`ConvergenceError`), especially in `:single` | `preconditioner = :amg`, or `:mixed` | AMG converges on badly conditioned steady states where `:mc_ilu0` does not |
| the model does not fit in GPU memory | `precision = :single`, or `preconditioner = :jacobi` | see the memory table in the [GPU guide](@ref "Memory") |
| you compare with MODFLOW 6 to the last digit | `precision = :double` | Float64 everywhere, as MODFLOW 6 |

The [GPU guide](gpu.md) explains the presets, their accuracy and memory use in more detail.

## Unsupported input

ModflowGPU never ignores input silently. Input that it does not support stops the run
with an [`MF6UnsupportedError`](@ref) that names the file, the package and the option,
before anything runs. For example, after adding the XT3D option (not supported yet) to the
NPF file:

```jldoctest gs; filter = r"\(/.*flow\.npf\)" => "(flow.npf)"
julia> npf_file = joinpath(simdir, "flow.npf");

julia> original = read(npf_file, String);

julia> write(npf_file, replace(original, "BEGIN OPTIONS" => "BEGIN OPTIONS\n  XT3D"));

julia> run_simulation(simdir)
ERROR: MF6UnsupportedError: unsupported: NPF6 option XT3D (model 'FLOW') (/tmp/jl_A1b2C3/flow/flow.npf)
[...]
```

Packages outside the scope of ModflowGPU (SFR, LAK, MAW, UZF, MVR, CSUB, the transport
models, ...) give the same error. [Supported packages](compatibility/packages.md) lists
what is supported.

Reading is strict by default. [`load_simulation`](@ref) with `strict = false` reads all
input that ModflowGPU can parse, without checking it against what the engine supports, so
that you can inspect or change a simulation in Julia (and write it back with
[`write_simulation`](@ref)). The engine still refuses what it cannot compute, when the
run starts:

```jldoctest gs
julia> spec = load_simulation(simdir; strict = false)
SimulationSpec(1 model(s): FLOW; 1 period(s); 0 exchange(s))

julia> run_simulation(spec; write_output = false)
ERROR: MF6UnsupportedError: unsupported: NPF6 option XT3D (flow.npf)
[...]

julia> write(npf_file, original);   # restore the NPF file
```

### IMS options that are replaced

Some IMS options are accepted but work differently in ModflowGPU. They are reported in
the log when the run starts (most of them once per Julia session), as an `Info` when
nothing you set yourself is lost, or as a `Warning` when it is. A simulation with `COMPLEXITY COMPLEX`, for example:

```jldoctest gs; filter = r"└ @ ModflowGPU .*"
julia> ims = build_ims(; complexity = :complex);

julia> sim_complex = build_simulation(gwf; tdis = build_tdis(; perlen = 1.0, time_units = :days), ims);

julia> r = run_simulation(sim_complex; write_output = false);
┌ Warning: IMS BACKTRACKING_NUMBER = 20 (COMPLEXITY COMPLEX or given): backtracking is not implemented yet; outer iterations may differ from MF6
│   ims.file = "flow.ims"
└ @ ModflowGPU ~/ModflowGPU/src/solver/picard.jl:77
┌ Info: IMS preconditioner ILUT is replaced by mc_ilu0 (multicolour ILU0: red–black on DIS grids, greedy colouring on DISV and DISU; the default, see the `preconditioner` option)
└   ims.file = "flow.ims"

julia> r.converged
true
```

(A [`SimulationSpec`](@ref) built in Julia can be run directly, without writing files.)

What these messages mean:

- **The preconditioner.** IMS's ILU0, MILU0 and ILUT (set by PRECONDITIONER_LEVELS,
  RELAXATION_FACTOR or COMPLEXITY) are replaced by the `preconditioner` option, because
  they do not run in parallel. This changes the number of inner iterations, not the
  converged heads. With the default `preconditioner = :auto` it is an `Info` for ILU0
  and for preconditioners that come from COMPLEXITY alone, and a `Warning` when you set
  RELAXATION_FACTOR or PRECONDITIONER_LEVELS yourself, because their values are not
  used (when you choose `preconditioner` yourself it is always an `Info`): for example
  `IMS preconditioner MILU0 is replaced by mc_ilu0 (...)`. Pass
  `precond_options = (; relax = 0.97)` to use MILU relaxation with `:mc_ilu0`.
- **Backtracking** (BACKTRACKING_NUMBER > 0, also set by COMPLEXITY COMPLEX) is not
  implemented: a `Warning`. The outer iterations then differ from MODFLOW 6's, but the
  converged heads agree within the closure criteria.
- **REORDERING_METHOD** has no effect with ModflowGPU's preconditioners: an `Info`.
- In `:single`, closures below what Float32 can resolve are raised: an `Info`. See the
  [GPU guide](@ref "What the presets mean for accuracy").

## Next steps

- [GPU guide](gpu.md): run on a GPU, precision, memory and troubleshooting.
- [Supported packages](compatibility/packages.md): check that your model is supported.
- [Stepping API](coupling/stepping.md): run the model one time step at a time and change
  its boundary data from Julia.
- [API reference](api/index.md): every function and its options.

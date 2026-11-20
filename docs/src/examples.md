# Example gallery

Each page of the gallery runs one model with ModflowGPU, draws the figures of the
equivalent [MODFLOW 6 example](https://modflow6-examples.readthedocs.io/en/latest/) from
ModflowGPU's output, and compares the results with MODFLOW 6 on the same input.

| Page | MODFLOW 6 example | Shows |
| :-- | :-- | :-- |
| [Tóth: flow systems in a small drainage basin](examples/ex-gwf-toth.md) | `ex-gwf-toth` | a cross-section of 20 000 cells; heads and the stream function |
| [Horizontal anisotropy](examples/ex-gwf-hani.md) | `ex-gwf-hanir`, `ex-gwf-hanic` | anisotropic K rotated with `ANGLE1`; drawdown maps |
| [Flow and head boundaries from time series](examples/ex-gwf-fhb.md) | `ex-gwf-fhb` | a transient model with CHD and WEL time series; hydrographs |
| [TWRI](examples/ex-gwf-twri01.md) | `ex-gwf-twri01` | five layers, a convertible top layer, wells, drains and recharge; heads and specific discharge |
| [BCF2SS](examples/ex-gwf-bcf2ss.md) | `ex-gwf-bcf2ss-p01a` | two aquifers, a river and wells; dry cells and rewetting |
| [Flow diversion around a high bottom](examples/ex-gwf-bump.md) | `ex-gwf-bump-p01b` | unconfined flow around dry cells on a bump of the bottom |
| [A rising water table under a recharge pond](examples/ex-gwf-nwt-p02.md) | `ex-gwf-nwt-p02b` | 14 layers, transient, rewetting of dry cells; water-table sections |
| [Highlands next to a tidal estuary](examples/ex-gwf-advtidal.md) | `ex-gwf-advtidal` | GHB, RIV, WEL, three RCH zones and EVT with time series, 361 steps; time series of heads and flows |
| [A well screened across several layers](examples/ex-gwf-maw-p03.md) | `ex-gwf-maw-p03a`, `ex-gwf-maw-p03c` | a regional section and a local model with a high-conductivity well |
| [A horizontal well](examples/ex-gwf-hwell.md) | `ex-gwf-hwell-a` | 153 164 cells, 60 steps, AMG; drawdown against an analytical solution |
| [A valley aquifer with a river](examples/ex-gwf-lgrv-gc.md) | `ex-gwf-lgrv-gc` | 9 layers, RIV and RCH, K zones; maps and a cross-section |
| [A valley aquifer: the refined model](examples/ex-gwf-lgrv-gr.md) | `ex-gwf-lgrv-gr` | the same at 672 525 cells |
| [A valley aquifer with a refined child model](examples/ex-gwf-lgrv-lgr.md) | `ex-gwf-lgrv-lgr` | two models coupled by a GWF–GWF exchange (local grid refinement); maps and a cross-section through both |
| [A nested grid with DISV](examples/ex-gwf-u1disv.md) | `ex-gwf-u1disv` | an unstructured grid with a refined block; heads and their error |
| [Curvilinear grids with DISV](examples/ex-gwf-curvilinear.md) | `ex-gwf-curve-90`, `ex-gwf-curvilin` | curvilinear cells; heads against the radial-flow solution |
| [An axisymmetric model with DISU](examples/ex-gwf-rad-disu.md) | `ex-gwf-rad-disu` | a radial model of rings with convertible storage; heads at three depths |
| [A periodic boundary](examples/ex-gwf-spbc.md) | `ex-gwf-spbc` | a model connected to itself by an exchange; heads against the analytical solution |
| [A nested model in a hole: GWF–GWF exchanges](examples/ex-gwf-u1gwfgwf.md) | `ex-gwf-u1gwfgwf-s1` to `-s4` | an inner model in the hole of an outer one, with and without XT3D on the exchange; heads and exchange flows |
| [A coupling-type model, step by step](examples/synthetic-coupling.md) | none | RIV, DRN, EVT and RCH on a synthetic island, run with the stepping API |

## All MODFLOW 6 examples in scope

The table lists every MODFLOW 6 groundwater-flow example that is in the scope of
ModflowGPU (made from `validation/examples.toml` in the repository when the manual is
built), smallest tier first. Every example that ModflowGPU supports has a page.
Examples with the packages that are out of scope (SFR, LAK, MAW, UZF, CSUB, transport, ...) are not listed. ModflowGPU writes the observation files
(OBS) of the examples that have them; the pages of `ex-gwf-fhb`, `ex-gwf-advtidal` and
`ex-gwf-rad-disu` show them against those of MODFLOW 6 (the `ex-gwf-maw-p03` page shows the
head and budget files only).

```@eval
Main.ExampleTools.gallery_index_table()
```

## How the pages are made

The pages are made from the scripts in the `examples/` folder of the repository when the
manual is built. Each script:

1. takes the input of the example from the released `mf6examples.zip` of the
   [MODFLOW 6 examples](https://github.com/MODFLOW-ORG/modflow6-examples), and MODFLOW
   6.8.2 from its release, both downloaded and checked against a pinned sha256;
2. sets the tight closures of the [parity tests](@ref "Comparing results") in the IMS
   file and runs MODFLOW 6.8.2, so that both programs solve the same input to the same
   closure;
3. runs ModflowGPU on the same folder on the CPU, in `:double` and `:single` precision,
   with the default options;
4. draws the figures of the MODFLOW 6 example from ModflowGPU's output, with
   [CairoMakie](https://docs.makie.org/stable/);
5. shows the head differences with MODFLOW 6, and a table of the largest head and budget
   differences next to the values of the [parity table](@ref "Parity table").

The wall times on the pages are those of the CPU of the machine that built the manual
(including Julia's compilation for the first run of a session) and, from the parity
table, those of an AMD Radeon RX 7800 XT. The models are small, so these times say
little about the speed on large models; see [Performance](performance.md) for that.

## Running an example yourself

The scripts also run on their own, from a clone of the repository (Linux, as the
mf6 binary they download is the Linux one):

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs examples/ex-gwf-toth.jl
```

The downloads, the runs and the figures (PNG files) go to `docs/.cache` (set
`MFGPU_GALLERY_CACHE` for another folder). The helper functions the scripts call
(`example_run`, `run_modflowgpu`, `compare_with_mf6` and the plot helpers) are in
`docs/ExampleTools.jl`; they are part of the documentation, not of the package.

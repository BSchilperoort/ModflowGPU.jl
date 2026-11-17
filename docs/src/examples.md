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
| [A valley aquifer with a river](examples/ex-gwf-lgrv-gc.md) | `ex-gwf-lgrv-gc` | 9 layers, RIV and RCH, K zones; maps and a cross-section |
| [A coupling-type model, step by step](examples/synthetic-coupling.md) | none | RIV, DRN, EVT and RCH on a synthetic island, run with the stepping API |

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

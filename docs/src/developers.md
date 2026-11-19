# For developers

This section is for people who change ModflowGPU. The developer material is not part of
the manual; it lives in the repository.

- [PLAN.md](https://github.com/BSchilperoort/ModflowGPU.jl/blob/main/PLAN.md): goals, architecture, conventions and the roadmap.
- [CLAUDE.md](https://github.com/BSchilperoort/ModflowGPU.jl/blob/main/CLAUDE.md): working rules for agents and contributors (git workflow,
  precision rules, tests).
- [docs/dev/](https://github.com/BSchilperoort/ModflowGPU.jl/tree/main/docs/dev): developer notes. Start with
  [api_map.md](https://github.com/BSchilperoort/ModflowGPU.jl/blob/main/docs/dev/api_map.md) (map of `src/`),
  [support_matrix.md](https://github.com/BSchilperoort/ModflowGPU.jl/blob/main/docs/dev/support_matrix.md) (what is supported, card by card),
  [mf6_notes.md](https://github.com/BSchilperoort/ModflowGPU.jl/blob/main/docs/dev/mf6_notes.md) (how MODFLOW 6 behaviour is mapped),
  [precision.md](https://github.com/BSchilperoort/ModflowGPU.jl/blob/main/docs/dev/precision.md) and [scaling.md](https://github.com/BSchilperoort/ModflowGPU.jl/blob/main/docs/dev/scaling.md).
- [docs/dev/plan/](https://github.com/BSchilperoort/ModflowGPU.jl/tree/main/docs/dev/plan): the task cards, one file per phase.
- [benchmark/results/](https://github.com/BSchilperoort/ModflowGPU.jl/tree/main/benchmark/results): committed benchmark results.

## Running the tests

```sh
julia --project -e 'using Pkg; Pkg.test()'                    # CPU tests (what CI runs)
MFGPU_TEST_BACKEND=rocm julia --project=test test/runtests.jl # GPU tests, on a machine with a GPU
MFGPU_TEST_GROUPS=connectivity,npf julia --project=test test/runtests.jl # selected groups
MFGPU_TEST_JOBS=4 julia --project=test test/runtests.jl       # groups in 4 parallel processes
julia --project=@runic -m Runic --check .                     # formatting
```

`MFGPU_TEST_JOBS=N` (N > 1) runs the test groups in N worker processes, each running the
serial suite on its share of the groups, and prints a combined summary with the test counts;
the run fails if any worker fails. Unset or 1 is the serial run, and so is `Pkg.test()`. The
GPU workers share one GPU (about 1 GB each), so use N=3 with `MFGPU_TEST_BACKEND=rocm`; N=8 suits
the CPU suite on a 16-core machine (17.5 min serial, 5.4 min with N=8). `MFGPU_TEST_LOGDIR` keeps
the worker logs. The long `run` group is split by model (`MFGPU_TEST_MODELS`, which also accepts
`!a,b` for all but those); the weights that order the units are in `test/parallel.jl`.

## Building this manual

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The site is written to `docs/build/`. Warnings fail the build. A change that alters what
users see (a new package or option, a changed keyword, a new address) should update the
matching page under `docs/src/` in the same change.

The build also makes the [example gallery](examples.md): `docs/make.jl` turns the
[Literate](https://fredrikekre.github.io/Literate.jl/stable/) scripts in `examples/` into
pages, and Documenter runs them. They download `mf6examples.zip` and the mf6 6.8.2 Linux
binary (pinned sha256 in `docs/ExampleTools.jl`) into `docs/.cache` (gitignored; CI keeps
the downloads between runs). To use copies you already have, set
`MFGPU_GALLERY_ARCHIVES` to a folder that holds them, for example the `reference/archives`
folder made by `validation/fetch_reference.sh`. The build takes about 11 minutes on one
CPU thread (most of it in the larger examples: ex-gwf-hwell-a, ex-gwf-lgrv-gr and
ex-gwf-nwt-p02b, with their MODFLOW 6 runs) and 4.3 GB of memory.

A new gallery page is a script in `examples/` plus an entry in `ExampleTools.GALLERY`
(script, title and the MODFLOW 6 examples it runs). The overview page lists every
in-scope example of `validation/examples.toml` with its page, and the build fails when an
example that ModflowGPU supports has no page: the card that adds a feature (Newton–Raphson,
XT3D, exchanges) adds the pages of the examples it unlocks.

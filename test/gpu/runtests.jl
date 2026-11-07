# GPU entry of the test suite (T3.1): the full suite, including every parity test, on a GPU
# backend (local machine only, never in CI):
#
#     julia --project=test test/gpu/runtests.jl                       # rocm
#     MFGPU_TEST_BACKEND=cuda julia --project=test test/gpu/runtests.jl
#
# MFGPU_TEST_GROUPS selects groups as in test/runtests.jl. The parity tests run on the
# backend and compare with MF6; the run and assembly parity tests also compare it with the
# CPU backend (`:double` < 1e-10 relative, `:single` heads < 1e-4 m; PLAN.md section 4),
# and the kernel_hygiene group checks the device IR (ROCm).
if lowercase(get(ENV, "MFGPU_TEST_BACKEND", "cpu")) == "cpu"
    ENV["MFGPU_TEST_BACKEND"] = "rocm"
end
include(joinpath(@__DIR__, "..", "runtests.jl"))

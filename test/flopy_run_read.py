# Reads the head and budget files of a ModflowGPU.run (test/parity/run_test.jl, "flopy"
# testset, opt-in with MFGPU_FLOPY=1) with flopy and prints one "key values..." line per
# item (floats with repr, so they round-trip exactly).
# Run: uv run --no-project --with flopy --with numpy python -I flopy_run_read.py <hds> <cbc>
import sys

import flopy
import numpy as np


def emit(key, vals):
    print(key, *[repr(float(v)) for v in np.ravel(vals)])


h = flopy.utils.HeadFile(sys.argv[1])
emit("hds_times", h.get_times())
emit("hds_kstpkper", h.get_kstpkper())
emit("hds_last", h.get_data(kstpkper=h.get_kstpkper()[-1]))

c = flopy.utils.CellBudgetFile(sys.argv[2], precision="double")
names = [n.decode() if isinstance(n, bytes) else n for n in c.get_unique_record_names()]
print("cbc_names", *[n.strip() for n in names])
emit("cbc_times", c.get_times())
last = c.get_kstpkper()[-1]
for n in names:
    key = n.strip()
    d = c.get_data(text=n, kstpkper=last)[0]
    if d.dtype.names is not None:
        emit("cbc_" + key + "_node", d["node"])
        emit("cbc_" + key + "_q", d["q"])
    else:
        emit("cbc_" + key, d)

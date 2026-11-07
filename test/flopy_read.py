# Reads the head and budget files written by test/unit/output_test.jl ("flopy" testset,
# opt-in with MFGPU_FLOPY=1) with flopy and prints what it finds, one "key values..."
# line per item (floats with repr, so they round-trip exactly).
# Run: uv run --no-project --with flopy --with numpy python -I flopy_read.py <dir>
import sys

import flopy
import numpy as np

d = sys.argv[1]


def emit(key, vals):
    print(key, *[repr(float(v)) for v in np.ravel(vals)])


h = flopy.utils.HeadFile(d + "/t.hds")
emit("hds_times", h.get_times())
emit("hds_kstpkper", h.get_kstpkper())
emit("hds_shape", h.get_data(totim=2.0).shape)
emit("hds_data", h.get_data(totim=2.0))

c = flopy.utils.CellBudgetFile(d + "/t.cbc", precision="double")
names = [n.decode() if isinstance(n, bytes) else n for n in c.get_unique_record_names()]
print("cbc_names", *[n.strip() for n in names])
emit("cbc_times", c.get_times())
r = c.get_data(text="RCH", totim=2.0)[0]
emit("rch_node", r["node"])
emit("rch_node2", r["node2"])
emit("rch_q", r["q"])
emit("rch_mult", r["MULT"])
emit("wel_n", [len(c.get_data(text="WEL", totim=1.0)[0])])
emit("fjf", c.get_data(text="FLOW-JA-FACE", totim=1.0)[0])
emit("sto_shape", c.get_data(text="STO-SS", totim=1.0)[0].shape)
emit("sto", c.get_data(text="STO-SS", totim=1.0)[0])

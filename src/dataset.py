# ---------------------------------------------------------------------
# data_netcdf.py  (or paste directly into train.py)
# ---------------------------------------------------------------------
import torch
from torch.utils.data import Dataset
from netCDF4 import Dataset as NetCDF   # pip install netcdf4

class NetCDFWeatherDataset(Dataset):
    """
    Lazily streams samples from a NetCDF file.

    Expected on‑disk layout
    -----------------------
    root/
      ├─ input/          # NetCDF group  ➜ input channels   (C_in, H, W)
      │     data(float32)[N, C_in, H, W]        or
      │     <var1>[N, H, W], <var2>[N, H, W] ...
      └─ output/         # NetCDF group  ➜ target channels (C_out, H, W)
            data(float32)[N, C_out, H, W]       or
            <var1>[N, H, W], <var2>[N, H, W] ...

    Notes
    -----
    * Opens the NetCDF **once** and keeps the handle around
      (memory‑mapped – slices are loaded on demand, not all at once).
    * Works with either a single 4‑D `"data"` variable *or* many 3‑D
      variables inside each group (they will be stacked into channels).
    * Returns float32 tensors on every `__getitem__`.
    """

    def __init__(
        self,
        nc_path: str,
        input_group: str = "input",
        output_group: str = "output",
        dtype=torch.float32,
    ):
        super().__init__()
        self.nc = NetCDF(nc_path, "r", diskless=False, mmap=True)

        # ---------------- inputs ----------------
        g_in  = self.nc.groups[input_group]
        if "data" in g_in.variables:                       # preferred: single 4‑D var
            self.in_var = g_in.variables["data"]
            self.stack_in = False
            self.C_in = self.in_var.shape[1]
        else:                                              # fall‑back: many 3‑D vars
            self.in_vars = list(g_in.variables.values())
            self.stack_in = True
            self.C_in = len(self.in_vars)

        # ---------------- outputs ---------------
        g_out = self.nc.groups[output_group]
        if "data" in g_out.variables:
            self.out_var = g_out.variables["data"]
            self.stack_out = False
            self.C_out = self.out_var.shape[1]
        else:
            self.out_vars = list(g_out.variables.values())
            self.stack_out = True
            self.C_out = len(self.out_vars)

        # assume first dimension is sample/time
        self.N = (
            self.in_var.shape[0]
            if not self.stack_in
            else self.in_vars[0].shape[0]
        )
        assert self.N == (
            self.out_var.shape[0] if not self.stack_out else self.out_vars[0].shape[0]
        ), "input/output sample counts differ"

        self.dtype = dtype

    def __len__(self):  # number of samples
        return self.N

    def _load_inputs(self, idx):
        if self.stack_in:
            arr = [v[idx][...] for v in self.in_vars]  # list of (H, W)
            return torch.tensor(arr, dtype=self.dtype)  # (C_in, H, W)
        else:
            return torch.tensor(self.in_var[idx][...], dtype=self.dtype)

    def _load_outputs(self, idx):
        if self.stack_out:
            arr = [v[idx][...] for v in self.out_vars]
            return torch.tensor(arr, dtype=self.dtype)
        else:
            return torch.tensor(self.out_var[idx][...], dtype=self.dtype)

    def __getitem__(self, idx):
        x = self._load_inputs(idx)
        y = self._load_outputs(idx)
        return x, y

    def __del__(self):
        # make sure the file handle closes when the dataset is deleted
        try:
            self.nc.close()
        except Exception:
            pass

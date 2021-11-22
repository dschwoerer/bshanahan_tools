import zoidberg as zb
import numpy as np
import xarray as xr
from mms_helper import lst


def gen_grid_const(nx, ny, nz, R0, r0, r1, mode=0):
    mode = [
        ("const", np.linspace),
        ("exp", np.geomspace),
    ][mode]
    one = np.ones((nx, ny, nz))
    r = mode[1](r0, r1, nx)[:, None]
    theta = np.linspace(0, 2 * np.pi, nz, False)[None, :]
    phi = np.linspace(0, 2 * np.pi / 5, ny, False)
    R = R0 + np.cos(theta) * r
    Z = np.sin(theta) * r
    pol_grid = zb.poloidal_grid.StructuredPoloidalGrid(R, Z)

    field = zb.field.CurvedSlab(Bz=0, Bzprime=0, Rmaj=R0)
    grid = zb.grid.Grid(pol_grid, phi, 5, yperiodic=True)

    fn = f"radial_{mode[0]}_{nx}_{ny}_{nz}_{R0}.fci.nc"

    maps = zb.make_maps(grid, field, quiet=True)
    zb.write_maps(
        grid,
        field,
        maps,
        fn,
        metric2d=False,
    )
    with xr.open_dataset(fn) as ds:
        dims = ds.dz.dims
        ds["r_minor"] = dims, one * r[:, None, :]
        ds["phi"] = "y", phi
        ds["theta"] = dims, one * theta[:, None, :]
        ds.to_netcdf(fn)


for nx in lst:
    for mode in range(2):
        gen_grid_const(nx, 2, 4, 1, 0.1, 0.5, mode=mode)
        gen_grid_const(nx, 2, 4, 2, 0.1, 1.1, mode=mode)

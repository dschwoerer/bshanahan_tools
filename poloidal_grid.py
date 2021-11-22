import zoidberg as zb
import numpy as np
import xarray as xr
from mms_helper import lst

nonlin = 0.8
assert nonlin < 1


def var1(*args):
    pos = np.linspace(*args)
    return pos + np.sin(pos) * nonlin


def var2(*args):
    pos = np.linspace(*args)
    return pos + np.cos(pos) * nonlin


def var3(*args):
    pos = np.linspace(*args)
    return pos + np.cos(pos * 2) / 2 * nonlin


def var4(*args):
    pos = np.linspace(*args)
    return pos + np.sin(pos) / 2 * nonlin


def gen_grid(nx, ny, nz, R0, r0, r1, mode=0):
    mode = [
        ("const", np.linspace),
        ("var1", var1),
        ("var2", var2),
        ("var3", var3),
        ("var4", var4),
    ][mode]
    one = np.ones((nx, ny, nz))
    r = np.linspace(r0, r1, nx)[:, None]
    theta = mode[1](0, 2 * np.pi, nz, False)[None, :]
    phi = np.linspace(0, 2 * np.pi / 5, ny, False)
    R = R0 + np.cos(theta) * r
    Z = np.sin(theta) * r
    pol_grid = zb.poloidal_grid.StructuredPoloidalGrid(R, Z)

    field = zb.field.CurvedSlab(Bz=0, Bzprime=0, Rmaj=R0)
    grid = zb.grid.Grid(pol_grid, phi, 5, yperiodic=True)

    fn = f"poloidal_{mode[0]}_{nx}_{ny}_{nz}_{R0}.fci.nc"

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


for nz in lst:
    for mode in range(5):
        gen_grid(4, 2, nz, 1, 0.1, 0.5, mode=mode)
        # gen_grid_const(4, 2, nz, 2, 0.1, 1.1, mode=mode)

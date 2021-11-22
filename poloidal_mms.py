import boutcore as bc
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from mms_helper import extend, clean, lst
from poloidal_grid import grids


def test(fn):
    grid = xr.open_dataset(fn)
    bc.Options().set("mesh:file", fn, force=True)
    mesh = bc.Mesh(section="")
    # print(bc.__file__)
    # print()
    # print()
    f = bc.create3D("0", mesh)
    t = extend(grid.theta)
    Z = extend(grid.Z)
    one = np.ones_like(Z)

    inp = np.sin(t)
    ana = np.cos(t)

    # Set input field
    f[:, :, :] = inp
    mesh.communicate(f)
    calc = bc.DDZ(f).get()
    l2 = np.sqrt(np.mean(clean(ana - calc)[1:-1] ** 2))

    if 0:
        fig, axs = plt.subplots(1, 3)
        calcd = clean(calc)[:, 1]
        anad = ana[:, 1]
        for dat, label, ax in zip(
            [anad, calcd, (calcd - anad)], ["ana", "calc", "err"], axs
        ):
            plot = ax.imshow(dat[2:-2, 1:-1])
            ax.set_title(fn[-20:] + " " + label)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(plot, cax=cax, orientation="vertical")
    # plt.figure()
    # plt.imshow((calcd - anad)[1:-1])
    # plt.title(fn + " err")
    # plt.colorbar()
    return l2


if __name__ == "__main__":
    bc.init("-d mms -q -q -q")

    for mode, todo in grids.items():
        l2 = [test(x[0]) for x in todo]
        lst = [x[-1] for x in todo]

        errc = np.log(l2[-2] / l2[-1])
        difc = np.log(lst[-1] / lst[-2])
        conv = errc / difc
        print(mode, conv, l2)

    plt.show()
# for x in lst:

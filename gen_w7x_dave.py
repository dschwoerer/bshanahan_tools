#!/usr/bin/python3
import sys
import re

import zoidberg_examples as zbe
from xemc3.core.utils import timeit
from boututils.datafile import DataFile

# conf          name
# 0	w7x	standard case
# 3	w7x	low iota
# 4	w7x	high iota
# 6	w7x	high mirror


try:
    import proxy_local_12345
except ImportError:
    pass


def gen(conf, shape, **kw):
    fn = f"""W7X-conf{conf}-{"x".join([str(d) for d in shape])}.fci.nc"""
    print(fn)
    with timeit():
        zbe.W7X(
            *shape,
            outer_vessel=True,
            inner_VMEC=True,
            outer_poincare=True,
            configuration=conf,
            calc_curvature=True,
            fname=fn,
            **kw,
        )
        with DataFile(fn, write=True) as f:
            for ff in __file__, zbe.__file__:
                with open(ff) as f2:
                    code = f2.read()
                ff = ff.split("/")[-1]
                print(ff, code)
                f.write_file_attribute(ff, code)


def gen2(fn):
    args = re.findall(r"W7X-conf(\d+)-(\d+)x(\d+)x(\d+).fci", fn)
    if not args:
        print(f"'{fn}' is not a valid filename")
        return

    assert len(args) == 1
    args = [int(x) for x in args[0]]
    conf = args[0]
    shape = args[1:]
    gen(conf, shape, show_maps=False, field_refine=0)


for fn in sys.argv[1:]:
    gen2(fn)

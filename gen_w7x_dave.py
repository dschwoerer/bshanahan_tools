#!/usr/bin/python3
import sys
import zoidberg_examples as zbe
from xemc3.core.utils import timeit

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
            configuration=conf,
            fname=fn,
            **kw,
        )


# gen(4, (16 + 4, 4, 128), show_maps=True)

if "0" in sys.argv:
    with timeit():
        gen(4, (32 + 4, 16, 128), show_maps=False)
if "1" in sys.argv:
    with timeit():
        gen(4, (64 + 4, 16, 256), show_maps=False, field_refine=2)
if "2" in sys.argv:
    with timeit():
        gen(4, (128 + 4, 16, 512), show_maps=False, field_refine=4)
if "3" in sys.argv:
    with timeit():
        gen(4, (256 + 4, 16, 1024), show_maps=False, field_refine=8)
if "4" in sys.argv:
    with timeit():
        gen(4, (128 + 4, 32, 512), show_maps=False, field_refine=8)
if "5" in sys.argv:
    with timeit():
        gen(4, (128 + 4, 72, 512), show_maps=False, field_refine=4)
if "6" in sys.argv:
    with timeit():
        gen(4, (128 + 4, 32, 512), show_maps=False, field_refine=4)
if "7" in sys.argv:
    with timeit():
        gen(4, (128 + 4, 72, 512), show_maps=False, field_refine=0)
if "8" in sys.argv:
    with timeit():
        gen(4, (128 + 4, 16, 512), show_maps=False, field_refine=0)
if "9" in sys.argv:
    with timeit():
        gen(4, (256 + 4, 16, 1024), show_maps=False, field_refine=0)
if "10" in sys.argv:
    with timeit():
        gen(4, (256 + 4, 16, 2048), show_maps=False, field_refine=0)
if "11" in sys.argv:
    with timeit():
        gen(4, (256 + 4, 36, 1024), show_maps=False, field_refine=0)
if "12" in sys.argv:
    with timeit():
        gen(4, (128 + 4, 36, 1024), show_maps=False, field_refine=0)
if "13" in sys.argv:
    with timeit():
        gen(4, (512 + 4, 36, 2024), show_maps=False, field_refine=0)
if "14" in sys.argv:
    with timeit():
        gen(4, (1024 + 4, 36, 8096), show_maps=False, field_refine=0)
if "15" in sys.argv:
    with timeit():
        gen(4, (256 + 4, 18, 2048), show_maps=True, field_refine=0)
if "16" in sys.argv:
    with timeit():
        gen(4, (256 + 4, 18, 1024), show_maps=False, field_refine=0)
if "17" in sys.argv:
    with timeit():
        gen(4, (256 + 4, 72, 1024), show_maps=False, field_refine=0)

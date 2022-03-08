#!/usr/bin/python3

try:
    import proxy_local_12345
except ImportError:
    pass

import zoidberg_examples as zbe

#zbe.W7X(outer_vessel=True, inner_VMEC=True, nx=140, ny=72, nz=512, show_maps=True)
zbe.W7X(outer_vessel=True, inner_VMEC=True, nx=70, ny=36, nz=256, show_maps=True)

#!/usr/bin/env python
"""
Visualize the transformation.

History
  create  -  Feng Zhou (zhfe99@gmail.com), 08-16-2015
  modify  -  Feng Zhou (zhfe99@gmail.com), 08-18-2015
"""
import py_lib as lib

# read from hdf
ha = lib.hdfRIn('tmp.h5')
grid = lib.hdfR(ha, 'grid')
input0 = lib.hdfR(ha, 'input0')
input1 = lib.hdfR(ha, 'input1')
lib.hdfROut(ha)

# dimension
nTop = 10
n, h, w, _ = grid.shape

# show
rows = 3; cols = nTop
Ax = lib.iniAx(1, rows, cols, [3 * rows, 3 * cols], flat=False)

for iTop in range(nTop):
    # original input
    lib.shImg(input0[iTop, 0], ax = Ax[0, iTop])

    idxYs = [0, 0, h - 1, h - 1, 0]
    idxXs = [0, w - 1, w - 1, 0, 0]
    xs, ys = lib.zeros(5, n=2)
    for i in range(5):
        idxY = idxYs[i]
        idxX = idxXs[i]
        ys[i] = (grid[iTop, idxY, idxX, 0] + 1) / 2 * h
        xs[i] = (grid[iTop, idxY, idxX, 1] + 1) / 2 * w
    lib.plt.plot(xs, ys, 'r-')

    # input
    lib.shImg(input1[iTop, 0], ax = Ax[1, iTop])
lib.show()

lib.shSvPath('tmp.pdf')

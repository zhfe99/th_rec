#!/usr/bin/env python
"""
Visualize the transformation.

History
  create  -  Feng Zhou (zhfe99@gmail.com), 2015-08
  modify  -  Feng Zhou (zhfe99@gmail.com), 2015-08
"""
import os
import sys
import numpy as np
import py_lib as lib

# input
dbe = sys.argv[1]
ver = sys.argv[2]
con = sys.argv[3]
epo = int(sys.argv[4])

tmpFold = os.path.join(os.environ['HOME'],
                       'save/{}/torch/tmp/{}_{}_{}'.format(dbe, dbe, ver, con))
h5Path = '{}/test_{}_{}.h5'.format(tmpFold, epo, 1)

# read from hdf
ha = lib.hdfRIn(h5Path)
grid1 = lib.hdfR(ha, 'grid1')
grid2 = lib.hdfR(ha, 'grid2')
grids = [grid1, grid2]
input0 = lib.hdfR(ha, 'input0')
input1 = lib.hdfR(ha, 'input1')
input2 = lib.hdfR(ha, 'input2')
inputs = [input1, input2]
bias = lib.hdfR(ha, 'bias')
weight = lib.hdfR(ha, 'weight')
lib.hdfROut(ha)

# dimension
n, h, w, _ = grid1.shape
nTop = min(input0.shape[0], 7)

# show
rows = 2
cols = nTop
Ax = lib.iniAx(1, rows * 2, cols, [3 * rows * 2, 3 * cols], flat=False)

for row in range(rows):
    input = inputs[row]
    grid = grids[row]

    for iTop in range(nTop):
        col = iTop

        # original input
        lib.shImg(input0[iTop, 0], ax=Ax[row * 2, col])

        idxYs = [0, 0, h - 1, h - 1, 0]
        idxXs = [0, w - 1, w - 1, 0, 0]
        xs, ys = lib.zeros(5, n=2)
        for i in range(5):
            idxY = idxYs[i]
            idxX = idxXs[i]

            ys[i] = (grid[iTop, idxY, idxX, 0] + 1) / 2 * h
            xs[i] = (grid[iTop, idxY, idxX, 1] + 1) / 2 * w
        lib.plt.plot(xs, ys, 'r-')
        lib.plt.axis('image')
        # import pdb; pdb.set_trace()

        # input
        lib.shImg(input[iTop, 0], ax=Ax[row * 2 + 1, col])

lib.show()
lib.shSvPath('tmp.pdf')

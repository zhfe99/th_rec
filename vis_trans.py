#!/usr/bin/env python
"""
Visualize the transformation.

History
  create  -  Feng Zhou (zhfe99@gmail.com), 2015-08
  modify  -  Feng Zhou (zhfe99@gmail.com), 2015-08
"""
import matplotlib as mpl
mpl.use('Agg')
import os
import sys
import numpy as np
import py_lib as lib
lib.prSet(3)

# input
dbe = sys.argv[1]
ver = sys.argv[2]
con = sys.argv[3]
m = int(sys.argv[4])
nEpo = int(sys.argv[5])

# fold
tmpFold = os.path.join(os.environ['HOME'],
                       'save/{}/torch/tmp/{}_{}_{}'.format(dbe, dbe, ver, con))

lib.prCIn('epoch', nEpo + 1, 1)
for iEpo in range(1, nEpo + 1):
    lib.prC(iEpo)

    # path
    h5Path = '{}/test_{}_{}.h5'.format(tmpFold, iEpo, 1)
    pdfPath = '{}/test_{}_{}.pdf'.format(tmpFold, iEpo, 1)

    # read from hdf
    ha = lib.hdfRIn(h5Path)
    inputs, grids = [], []
    for i in range(m):
        gridi = lib.hdfR(ha, 'grid{}'.format(i + 1))
        inputi = lib.hdfR(ha, 'input{}'.format(i + 1))
        grids.append(gridi)
        inputs.append(inputi)
    input0 = lib.hdfR(ha, 'input0')
    bias = lib.hdfR(ha, 'bias')
    weight = lib.hdfR(ha, 'weight')
    lib.hdfROut(ha)

    # dimension
    n, h, w, _ = grids[0].shape
    nTop = min(input0.shape[0], 7)

    # show
    cols = nTop
    Ax = lib.iniAx(1, m * 2, cols, [3 * m * 2, 3 * cols], flat=False)

    # each transformer
    for row in range(m):
        input = inputs[row]
        grid = grids[row]

        # each example
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

            # input
            lib.shImg(input[iTop, 0], ax=Ax[row * 2 + 1, col])
    lib.shSvPath(pdfPath)
lib.prCOut(nEpo + 1)

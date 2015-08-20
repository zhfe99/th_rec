#!/usr/bin/env python
"""
Visualize the transformation.

History
  create  -  Feng Zhou (zhfe99@gmail.com), 08-16-2015
  modify  -  Feng Zhou (zhfe99@gmail.com), 08-20-2015
"""
import os
import py_lib as lib

tmpFold = os.path.join(os.environ['HOME'], 'save/car/torch/tmp')
epos = [1, 20, 40, 60, 80]
nEpo = len(epos)
imgIds = range(20, 30)
nTop = len(imgIds)

# show
rows = nEpo; cols = nTop
Ax = lib.iniAx(1, rows, cols, [3 * rows, 3 * cols], flat=False)

# each epoch
for iEpo in range(nEpo):
    epo = epos[iEpo]

    h5Path = '{}/test_{}_{}.h5'.format(tmpFold, epo, 1)

    # read from hdf
    ha = lib.hdfRIn(h5Path)
    grid = lib.hdfR(ha, 'grid')
    input0 = lib.hdfR(ha, 'input0')
    input1 = lib.hdfR(ha, 'input1')
    lib.hdfROut(ha)

    # dimension
    n, h, w, _ = grid.shape

    for iTop in range(nTop):
        imgId = imgIds[iTop]

        # original input
        lib.shImg(input0[imgId, 0], ax = Ax[iEpo, iTop])

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
        # lib.shImg(input1[iTop, 0], ax = Ax[1, iTop])
lib.show()
# lib.shSvPath('tmp.pdf')

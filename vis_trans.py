#!/usr/bin/env python
"""
Visualize the transformation.

Example
  ./vis_trans.py bird v1 alexS1S2 --nstn 1 --epo 1:2

History
  create  -  Feng Zhou (zhfe99@gmail.com), 2015-08
  modify  -  Feng Zhou (zhfe99@gmail.com), 2015-08
"""
import matplotlib as mpl
mpl.use('Agg')
import argparse
import os
import py_lib as lib
lib.prSet(3)


# fold
def shEpoTran(dbe, ver, con, nStn, epo):
    """
    Show the transformation for each epoch.

    Input
      dbe   -  database
      ver   -  version
      con   -  configuration
      nStn  -  #stn
      epo   -  epoch id
    """
    tmpFold = os.path.join(os.environ['HOME'],
                           'save/{}/torch/tmp/{}_{}_{}'.format(dbe, dbe, ver, con))

    # path
    h5Path = '{}/test_{}_{}.h5'.format(tmpFold, epo, 1)
    pdfPath = '{}/test_{}_{}.pdf'.format(tmpFold, epo, 1)

    # read from hdf
    ha = lib.hdfRIn(h5Path)
    inputs, grids = [], []
    for iStn in range(nStn):
        gridi = lib.hdfR(ha, 'grid{}'.format(iStn + 1))
        inputi = lib.hdfR(ha, 'input{}'.format(iStn + 1))
        grids.append(gridi)
        inputs.append(inputi)
    input0 = lib.hdfR(ha, 'input0')
    # bias = lib.hdfR(ha, 'bias')
    # weight = lib.hdfR(ha, 'weight')
    lib.hdfROut(ha)

    # dimension
    n, h, w, _ = grids[0].shape
    nTop = min(input0.shape[0], 7)

    # show
    cols = nTop
    Ax = lib.iniAx(1, nStn * 2, cols, [3 * nStn * 2, 3 * cols], flat=False)

    # each transformer
    for iStn in range(nStn):
        input = inputs[iStn]
        grid = grids[iStn]

        # each example
        for iTop in range(nTop):
            col = iTop

            # original input
            input0New = input0[iTop].transpose((1, 2, 0))
            lib.shImg(input0New, ax=Ax[iStn * 2, col])

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
            inputNew = input[iTop].transpose((1, 2, 0))
            lib.shImg(inputNew, ax=Ax[iStn * 2 + 1, col])

    # save
    # lib.show()
    lib.shSvPath(pdfPath)

if __name__ == '__main__':
    # argument
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+', help='dbe ver con')
    parser.add_argument('--nstn', help='#stn', default=1, dest='nStn', type=int)
    parser.add_argument('--epo', help='epo range', default='1:2', dest='epos')
    args = parser.parse_args()

    dbe = args.inputs[0]
    ver = args.inputs[1]
    con = args.inputs[2]
    epos = lib.str2ran(args.epos)
    nEpo = len(epos)
    nStn = args.nStn

    # each epoch
    lib.prCIn('epo', nEpo, 1)
    for iEpo in range(nEpo):
        lib.prC(iEpo)
        epo = epos[iEpo]

        shEpoTran(dbe, ver, con, nStn, epo)
    lib.prCOut(nEpo)

#!/usr/bin/env python
"""
Visualize debug information.

Example
  ./vis_tran.py bird v1 alexS1S2 --nstn 1 --epo 1:2

History
  create  -  Feng Zhou (zhfe99@gmail.com), 2015-08
  modify  -  Feng Zhou (zhfe99@gmail.com), 2015-09
"""
import matplotlib as mpl
mpl.use('Agg')
import argparse
import os
import numpy as np
import py_lib as lib
lib.prSet(3)


def shEpoTrImg(dbe, ver, con, nStn, epo, iBat=1):
    """
    Show the transformation of each epoch.

    Input
      dbe   -  database
      ver   -  version
      con   -  configuration
      nStn  -  #stn
      epo   -  epoch id
      iBat  -  batch id
    """
    # fold
    nm = '{}_{}_{}'.format(dbe, ver, con)
    tmpFold = os.path.join(os.environ['HOME'],
                           'save/{}/torch/tmp/{}'.format(dbe, nm))
    outFold = os.path.join(os.environ['HOME'],
                           'save/{}/torch/deb_stn/{}'.format(dbe, nm))
    lib.mkDir(outFold)

    # path
    outPath = '{}/tr_{}_{}_img.jpg'.format(outFold, epo, iBat)

    # read from hdf
    h5Path = '{}/tr_{}_{}_grid.h5'.format(tmpFold, epo, iBat)
    ha = lib.hdfRIn(h5Path)
    gridCorns = lib.cells(nStn)
    for iStn in range(nStn):
        gridCorns[iStn] = lib.hdfR(ha, 'gridCorn{}'.format(iStn + 1))
    lib.hdfROut(ha)

    # read from hdf
    h5Path = '{}/tr_{}_{}_img_in.h5'.format(tmpFold, epo, iBat)
    ha = lib.hdfRIn(h5Path)
    imgIn0 = lib.hdfR(ha, 'imgIn0')
    imgIns = lib.cells(nStn)
    for iStn in range(nStn):
        imgIns[iStn] = lib.hdfR(ha, 'imgIn{}'.format(iStn + 1))
    lib.hdfROut(ha)

    # dimension
    n, _, h, w = imgIn0.shape
    nTop = min(n, 7)

    # show
    rows = 2
    cols = nTop
    Ax = lib.iniAx(1, nStn * rows, cols, [3 * nStn * rows, 3 * cols], flat=False)

    # each transformer
    for iStn in range(nStn):
        grid = gridCorns[iStn]

        # each example
        for iTop in range(nTop):
            col = iTop

            # original input
            lib.shImg(imgIn0[iTop].transpose((1, 2, 0)), ax=Ax[iStn * 2, col])

            idxYs = [0, 0, 1, 1, 0]
            idxXs = [0, 1, 1, 0, 0]
            xs, ys = lib.zeros(5, n=2)
            for i in range(5):
                idxY = idxYs[i]
                idxX = idxXs[i]

                ys[i] = (grid[iTop, idxY, idxX, 0] + 1) / 2 * h
                xs[i] = (grid[iTop, idxY, idxX, 1] + 1) / 2 * w
            lib.plt.plot(xs, ys, 'r-')
            # lib.plt.axis('image')

            # input
            lib.shImg(imgIns[iStn][iTop].transpose((1, 2, 0)), ax=Ax[iStn * 2 + 1, col])

        # mean
        # inMe0 = input0.mean(0)
        # inMe = input.mean(0)
        # lib.shImg(inMe0.transpose((1, 2, 0)), ax=Ax[iStn * 2, nTop])
        # lib.shImg(inMe.transpose((1, 2, 0)), ax=Ax[iStn * 2 + 1, nTop])

    # save
    # lib.show()
    lib.shSvPath(outPath, type='jpg')


def shEpoTrGrid(dbe, ver, con, nStn, epo, iBat=1):
    """
    Show the transformation of each epoch.

    Input
      dbe   -  database
      ver   -  version
      con   -  configuration
      nStn  -  #stn
      epo   -  epoch id
      iBat  -  batch id
    """
    # fold
    nm = '{}_{}_{}'.format(dbe, ver, con)
    tmpFold = os.path.join(os.environ['HOME'],
                           'save/{}/torch/tmp/{}'.format(dbe, nm))
    outFold = os.path.join(os.environ['HOME'],
                           'save/{}/torch/deb_stn/{}'.format(dbe, nm))
    lib.mkDir(outFold)

    # path
    h5Path = '{}/tr_{}_{}_grid.h5'.format(tmpFold, epo, iBat)
    outPath = '{}/tr_{}_{}_grid.jpg'.format(outFold, epo, iBat)

    # read from hdf
    ha = lib.hdfRIn(h5Path)
    gridCorns, gridGrads = lib.cells(nStn, n=2)
    for iStn in range(nStn):
        gridCorns[iStn] = lib.hdfR(ha, 'gridCorn{}'.format(iStn + 1))
        gridGrads[iStn] = lib.hdfR(ha, 'gridGrad{}'.format(iStn + 1))
    lib.hdfROut(ha)

    # dimension
    n, _, _, _ = gridCorns[0].shape
    _, h, w, _ = gridGrads[0].shape

    # show
    rows = 1
    cols = 2
    Ax = lib.iniAx(1, nStn * rows, cols, [3 * nStn * rows, 3 * cols], flat=False)

    # each transformer
    for iStn in range(nStn):
        # show grid
        lib.setAx(Ax[iStn, 0])
        co = 0
        for iExp in range(n):
            idxYs = [0, 0, 1, 1, 0]
            idxXs = [0, 1, 1, 0, 0]
            xs, ys = lib.zeros(5, n=2)
            for i in range(5):
                idxY = idxYs[i]
                idxX = idxXs[i]

                ys[i] = (gridCorns[iStn][iExp, idxY, idxX, 0] + 1) / 2 * h
                xs[i] = (gridCorns[iStn][iExp, idxY, idxX, 1] + 1) / 2 * w
            if np.any(xs < 0) or np.any(xs > w) or np.any(ys < 0) or np.any(ys > h):
                co += 1
            lib.plt.plot(xs, ys, 'r-')
        lib.plt.axis('equal')
        lib.plt.axis([0, h, 0, w])
        lib.plt.gca().invert_yaxis()
        lib.plt.title('{}/{}'.format(co, n))

        # show gradient
        lib.setAx(Ax[iStn, 1])
        GX = gridGrads[iStn][0][:, :, 0]
        GY = gridGrads[iStn][0][:, :, 1]
        Q = lib.plt.quiver(GX, GY)
        lib.plt.quiverkey(Q, 0.5, 0.92, 2, '', labelpos='W')
        lib.plt.axis([0, h, 0, w])
        lib.plt.gca().invert_yaxis()

        gX = gridGrads[iStn][0][:, :, 0].mean()
        gY = gridGrads[iStn][0][:, :, 1].mean()
        lib.plt.title('{:.2f} {:.3e}'.format(np.arctan2(gY, gX) * 180 / np.pi, np.linalg.norm([gX, gY])))

    # save
    # lib.show()
    lib.shSvPath(outPath, type='jpg')


def shEpoTran(dbe, ver, con, nStn, epo, iBat=1):
    """
    Show the transformation of each epoch.

    Input
      dbe   -  database
      ver   -  version
      con   -  configuration
      nStn  -  #stn
      epo   -  epoch id
      iBat  -  batch id
    """
    # fold
    nm = '{}_{}_{}'.format(dbe, ver, con)
    tmpFold = os.path.join(os.environ['HOME'],
                           'save/{}/torch/tmp/{}'.format(dbe, nm))
    outFold = os.path.join(os.environ['HOME'],
                           'save/{}/torch/deb_stn/{}'.format(dbe, nm))
    lib.mkDir(outFold)

    # path
    h5Path = '{}/test_{}_{}.h5'.format(tmpFold, epo, iBat)
    outPath = '{}/test_{}_{}.jpg'.format(outFold, epo, iBat)

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
    cols = nTop + 1
    Ax = lib.iniAx(1, nStn * 2, cols, [3 * nStn * 2, 3 * cols], flat=False)

    lib.setAx(Ax[0, nTop])
    lib.plt.axis('off')

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

        # mean
        inMe0 = input0.mean(0)
        inMe = input.mean(0)
        lib.shImg(inMe0.transpose((1, 2, 0)), ax=Ax[iStn * 2, nTop])
        lib.shImg(inMe.transpose((1, 2, 0)), ax=Ax[iStn * 2 + 1, nTop])

    # save
    # lib.show()
    lib.shSvPath(pdfPath)


def shEpoTranCmp(dbe, ver, con, nStn, epo, iBat=1, rows=2, cols=5):
    """
    Show the transformation of each epoch in a more compressed way.

    Input
      dbe   -  database
      ver   -  version
      con   -  configuration
      nStn  -  #stn
      epo   -  epoch id
      iBat  -  batch id
      rows  -  #row, {2}
      cols  -  #col, {5}
    """
    # fold
    nm = '{}_{}_{}'.format(dbe, ver, con)
    tmpFold = os.path.join(os.environ['HOME'],
                           'save/{}/torch/tmp/{}'.format(dbe, nm))
    pdfFold = os.path.join(os.environ['HOME'],
                           'save/{}/torch/deb_stn/{}'.format(dbe, nm))
    lib.mkDir(pdfFold)

    # path
    h5Path = '{}/test_{}_{}.h5'.format(tmpFold, epo, iBat)

    # read from hdf
    ha = lib.hdfRIn(h5Path)
    inputs, grids = [], []
    for iStn in range(nStn):
        gridi = lib.hdfR(ha, 'grid{}'.format(iStn + 1))
        inputi = lib.hdfR(ha, 'input{}'.format(iStn + 1))
        grids.append(gridi)
        inputs.append(inputi)
    input0 = lib.hdfR(ha, 'input0')
    lib.hdfROut(ha)

    # dimension
    n, h, w, _ = grids[0].shape
    nGrp = n / (rows * cols)

    # each group
    for iGrp in range(nGrp):
        # group path
        pdfPath = '{}/test_{}_{}_{}.pdf'.format(pdfFold, epo, iBat, iGrp)

        # show
        Ax = lib.iniAx(1, rows, cols, [3 * rows, 3 * cols], flat=False)

        # each example
        for iExp in range(rows * cols):
            # position
            pExp = iGrp * rows * cols + iExp
            row = iExp / cols
            col = iExp % cols

            # original input
            input0New = input0[pExp].transpose((1, 2, 0))
            lib.shImg(input0New, ax=Ax[row, col])

            # each transformer
            for iStn in range(nStn):
                grid = grids[iStn]

                idxYs = [0, 0, h - 1, h - 1, 0]
                idxXs = [0, w - 1, w - 1, 0, 0]
                xs, ys = lib.zeros(5, n=2)
                for i in range(5):
                    idxY = idxYs[i]
                    idxX = idxXs[i]

                    ys[i] = (grid[pExp, idxY, idxX, 0] + 1) / 2 * h
                    xs[i] = (grid[pExp, idxY, idxX, 1] + 1) / 2 * w

                _, cl = lib.genMkCl(iStn)
                lib.plt.plot(xs, ys, '-', color=cl)
                lib.plt.axis('image')
        # save
        # lib.show()
        lib.shSvPath(pdfPath)


def shEpoGrad(dbe, ver, con, nStn, epo):
    """
    Show the gradient of each epoch.

    Input
      dbe   -  database
      ver   -  version
      con   -  configuration
      nStn  -  #stn
      epo   -  epoch id
    """
    tmpFold = os.path.join(os.environ['HOME'],
                           'save/{}/torch/tmp/{}_{}_{}'.format(dbe, dbe, ver, con))
    pdfFold = os.path.join(os.environ['HOME'],
                           'save/{}/torch/deb_stn/{}_{}_{}'.format(dbe, dbe, ver, con))
    lib.mkDir(pdfFold)

    # path
    h5Path = '{}/train_{}_{}_grad.h5'.format(tmpFold, epo, 1)
    pdfPath = '{}/train_{}_{}_grad.pdf'.format(pdfFold, epo, 1)

    # read from hdf
    ha = lib.hdfRIn(h5Path)
    imgOuts, imgGrads, gridOuts, gridGrads = [], [], [], []
    for iStn in range(nStn):
        imgOuts.append(lib.hdfR(ha, 'imgOut{}'.format(iStn + 1)))
        imgGrads.append(lib.hdfR(ha, 'imgGrad{}'.format(iStn + 1)))
        gridOuts.append(lib.hdfR(ha, 'gridOut{}'.format(iStn + 1)))
        gridGrads.append(lib.hdfR(ha, 'gridGrad{}'.format(iStn + 1)))
    imgOut0 = lib.hdfR(ha, 'imgOut0')
    # bias = lib.hdfR(ha, 'bias')
    # weight = lib.hdfR(ha, 'weight')
    lib.hdfROut(ha)

    # dimension
    n, d, h, w = imgOut0.shape
    nTop = min(n, 7)

    # show
    cols = nTop
    rowStn = 4
    Ax = lib.iniAx(1, nStn * rowStn, cols, [3 * nStn * rowStn, 3 * cols], flat=False)

    # each transformer
    for iStn in range(nStn):
        imgOut = imgOuts[iStn]
        imgGrad = imgGrads[iStn]
        gridOut = gridOuts[iStn]
        gridGrad = gridGrads[iStn]

        # each example
        for iTop in range(nTop):
            col = iTop

            # original input
            lib.shImg(imgOut0[iTop].transpose((1, 2, 0)), ax=Ax[iStn * rowStn, col])

            idxYs = [0, 0, h - 1, h - 1, 0]
            idxXs = [0, w - 1, w - 1, 0, 0]
            xs, ys = lib.zeros(5, n=2)
            for i in range(5):
                idxY = idxYs[i]
                idxX = idxXs[i]

                ys[i] = (gridOut[iTop, idxY, idxX, 0] + 1) / 2 * h
                xs[i] = (gridOut[iTop, idxY, idxX, 1] + 1) / 2 * w
            lib.plt.plot(xs, ys, 'r-')
            # lib.plt.axis('image')

            # crop input
            lib.shImg(imgOut[iTop].transpose((1, 2, 0)), ax=Ax[iStn * rowStn + 1, col])

            # image gradient
            imgGradMa = imgGrad[iTop].max()
            imgGradMi = imgGrad[iTop].min()
            lib.shImg((imgGrad[iTop] - imgGradMi) / (imgGradMa - imgGradMi), ax=Ax[iStn * rowStn + 2, col])

            # grid gradient
            lib.setAx(Ax[iStn * rowStn + 3, col])
            GX = gridGrad[iTop][:, :, 0]
            GY = gridGrad[iTop][:, :, 1]
            Q = lib.plt.quiver(GX, GY)
            qk = lib.plt.quiverkey(Q, 0.5, 0.92, 2, '', labelpos='W')
            # lib.plt.axis('image')
            lib.plt.gca().invert_yaxis()

        # mean
        # inMe0 = input0.mean(0)
        # inMe = input.mean(0)
        # lib.shImg(inMe0.transpose((1, 2, 0)), ax=Ax[iStn * 2, nTop])
        # lib.shImg(inMe.transpose((1, 2, 0)), ax=Ax[iStn * 2 + 1, nTop])

    # save
    # lib.show()
    # import pdb; pdb.set_trace()
    lib.shSvPath(pdfPath)

if __name__ == '__main__':
    # argument
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+', help='dbe ver con')
    parser.add_argument('--nstn', help='#stn', default=1, dest='nStn', type=int)
    parser.add_argument('--epo', help='epo range', default='1:2', dest='epos')
    parser.add_argument('--bat', help='batch range', default='1:2', dest='bats')
    args = parser.parse_args()

    dbe = args.inputs[0]
    ver = args.inputs[1]
    con = args.inputs[2]

    epos = lib.str2ran(args.epos)
    bats = lib.str2ran(args.bats)

    nEpo = len(epos)
    nBat = len(bats)
    nStn = args.nStn

    # each epoch
    lib.prCIn('epo', nEpo, 1)
    for iEpo in range(nEpo):
        lib.prC(iEpo)

        # shEpoTran(dbe, ver, con, nStn, epo)
        # shEpoGrad(dbe, ver, con, nStn, epo)

        lib.prCIn('bat', nBat, 1)
        for iBat in range(nBat):
            lib.prC(iBat)
            # shEpoTrImg(dbe, ver, con, nStn, epos[iEpo], bats[iBat])
            shEpoTrGrid(dbe, ver, con, nStn, epos[iEpo], bats[iBat])
            # shEpoTran(dbe, ver, con, nStn, epos[iEpo], bats[iBat])
        lib.prCOut(nBat)
    lib.prCOut(nEpo)

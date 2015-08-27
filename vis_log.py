#!/usr/bin/env python
"""
Read a log file and parse it.

Usage
  python vis_log.py dbe1 ver1 con1 dbe2 ver2 con2 ...
  where
    dbe1  -  database name, car | food
    ver1  -  version, v2 | v3 | v23
    con1  -  configuration name, caffe | google
    ...

Example
  python vis_log.py car v1c alex

History
  create  -  Feng Zhou (zhfe99@gmail.com), 2015-08
  modify  -  Feng Zhou (zhfe99@gmail.com), 2015-08
"""
import argparse
import numpy as np
import os
import py_lib as lib
import lua_th as th
lib.init(prL=3)


def shLog(trLrs, trLosss, trAccss, trTi0ss, teLosss, teAccss, teTi0ss):
    """
    Input
      trTiss   -  time, nCon x
      accNms
      lossNms
      logFold  -  log fold to save pdf
      logNms   -  log name, nCon x
      isLr     -  flag of showing learning rate or not, True | {False}
    """
    # dimension
    nCon = len(trLrs)

    # range
    maLos = 0
    maAcc = 0
    miAcc = 1
    trTiss, trEposs, teTiss, teEposs = lib.cells(nCon, n=4)
    for i in range(nCon):
        # maximum loss
        maLos = max(maLos, max(trLosss[i]))
        maLos = max(maLos, max(teLosss[i]))

        # maximum accuracy
        maAcc = max(maAcc, max(teAccss[i]))
        miAcc = min(miAcc, min(teAccss[i]))

        # normalize the time
        ti0 = trTi0ss[i][0]
        trTis, teTis = [], []
        for j in range(trTi0ss[i].shape[0]):
            trTis.append((trTi0ss[i][j] - ti0).total_seconds())
        for j in range(teTi0ss[i].shape[0]):
            teTis.append((teTi0ss[i][j] - ti0).total_seconds())
        trTiss[i] = np.array(trTis)
        teTiss[i] = np.array(teTis)

        # epoch
        trEposs[i] = np.array(range(len(trLosss[i])))
        teEposs[i] = np.array(range(len(teLosss[i])))

    # show
    rows = 2
    cols = 3
    Ax = lib.iniAx(1, rows, cols, [rows * 4, cols * 5],
                   flat=False, hs=[2, 1], dpi=None)

    # show train
    lib.setAx(Ax[0, 0])
    has = []
    co1 = 0
    co2 = 0
    for i in range(nCon):
        co1 += 1
        _, cl = lib.genMkCl(co1)
        xs = trEposs[i]
        ha, = lib.plt.plot(xs, trLosss[i], '-',
                           color=cl, label='{} tr'.format(cons[i]))
        has.append(ha)

        # testing loss
        co2 += 1
        xs = teEposs[i]
        mk, _ = lib.genMkCl(co2)
        ha, = lib.plt.plot(xs, teLosss[i], 'r-',
                           marker=mk, label='{}: te'.format(cons[i]))
        has.append(ha)

    # lib.plt.xlabel(xaxis)
    lib.plt.ylabel('loss')

    lib.setAx(Ax[1, 0])
    lib.plt.axis('off')
    lib.plt.legend(handles=has, ncol=1)

    # test
    lib.setAx(Ax[0, 1])
    has = []
    co = 0
    for i in range(nCon):
        co += 1
        mk, _ = lib.genMkCl(co)
        ha, = lib.plt.plot(teEposs[i], teAccss[i], 'r-', marker=mk,
                           label = '{}: te'.format(cons[i]))
        has.append(ha)

    # lib.plt.xlabel(xaxis)
    lib.plt.ylabel('accuracy')

    lib.setAx(Ax[1, 1])
    lib.plt.axis('off')
    lib.plt.legend(handles=has, loc=1)

    # time
    lib.setAx(Ax[0, 2])
    has = []
    for i in range(nCon):
        mk, _ = lib.genMkCl(2 * i)

        ha, = lib.plt.plot(trEposs[i], 1.0 * trTiss[i] / 3600, 'r-', label='{}: tr time'.format(cons[i]))
        has.append(ha)

        mk, _ = lib.genMkCl(2 * i + 1)
        ha, = lib.plt.plot(teEposs[i], 1.0 * teTiss[i] / 3600, 'bs', marker=mk, label='{}: te time'.format(cons[i]), markersize=3)
        has.append(ha)
    lib.plt.xlabel('#epochs')
    lib.plt.ylabel('hours')

    lib.setAx(Ax[1, 2])
    lib.plt.axis('off')
    lib.plt.legend(handles=has, loc=1)

    lib.show()

    # save to pdf
    if False:
        pdfNm = logNms[0]
        for i in range(1, nCon):
            pdfNm += '_' + logNms[i]
            pdfPath = os.path.join(logFold, pdfNm + '.pdf')
        lib.svImgPath(pdfPath)

if __name__ == '__main__':
    # argument
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+', help='dbe ver con')
    parser.add_argument('--tr-loss', help='train loss name', default=[], dest='trLossNms', action='append')
    parser.add_argument('--te-loss', help='test loss name', default=[], dest='teLossNms', action='append')
    parser.add_argument('--acc', help='accuracy name', default=[], dest='accNms', action='append')
    parser.add_argument('--xaxis', help='x axis name', default='epoch', choices=['epoch', 'iter', 'time'], dest='xaxis')
    parser.add_argument('--sh-lr', help='show learning rate', default=0, dest='isLr', type=int)
    parser.add_argument('--old', help='using old caffe', const=True, action='store_const', dest='isOld')
    args = parser.parse_args()

    # each configuration
    nCon = len(args.inputs) / 3
    dbes, vers, cons, logPaths, logNms, \
        trLrs, trLosss, trAccss, trTiss, teLosss, teAccss, teTiss = lib.cells(nCon, n=12)
    for i in range(nCon):
        dbes[i] = args.inputs[i * 3 + 0]
        vers[i] = args.inputs[i * 3 + 1]
        cons[i] = args.inputs[i * 3 + 2]

        # log path
        logNms[i] = '{}_{}_{}.log'.format(dbes[i], vers[i], cons[i])
        logFold = os.path.join(os.environ['HOME'], 'save/{}/torch/log'.format(dbes[i]))
        logPaths[i] = os.path.join(logFold, logNms[i] + '_train')

        # parse
        trLrs[i], _, trLosss[i], trAccss[i], trTiss[i], teLosss[i], teAccss[i], teTiss[i] = th.logParse(logPaths[i])

    # show
    shLog(trLrs, trLosss, trAccss, trTiss, teLosss, teAccss, teTiss)

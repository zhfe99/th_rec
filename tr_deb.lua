#!/usr/bin/env th
-- Debug in training.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 2015-08
--   modify  -  Feng Zhou (zhfe99@gmail.com), 2015-09

local lib = require('lua_lib')
local tr_deb = {}

----------------------------------------------------------------------
-- Debug for epoch.
--
-- Input
--   model    -  data provider
--   tmpFold  -  fold to write
--   epo      -  epoch index
--   iMini    -  mini-batch index
--   train    -  train or test
--   opt      -  option
--   con      -  solver
function tr_deb.debStn(model, tmpFold, epo, iMini, train, opt, con, denormalize)
  -- create fold if necessary
  local tmpFold = opt.CONF.tmpFold
  if not paths.dirp(tmpFold) then
    lib.pr('%s not exist', tmpFold)
    os.execute('mkdir -p ' .. tmpFold)
  end

  -- parameter
  nStn = con.nStn

  -- get component
  local tmpWeight = model:findModules('nn.Linear')[2].weight
  local tmpBias = model:findModules('nn.Linear')[2].bias
  local tmpIn0 = model:findModules('nn.Identity')[1].output:clone()
  tmpIn0 = denormalize(tmpIn0)
  local tmpIns = {}
  local tmpGrids = {}
  for iStn = 1, nStn do
    tmpIns[iStn] = model:findModules('nn.Transpose')[iStn + 1].output:clone()
    tmpIns[iStn] = denormalize(tmpIns[iStn])
    tmpGrids[iStn] = model:findModules('nn.AffineGridGeneratorBHWD')[iStn].output
  end

  -- hdf handler
  local ha
  if train then
    ha = lib.hdfWIn(string.format('%s/train_%d_%d.h5', tmpFold, epo, iMini))
  else
    ha = lib.hdfWIn(string.format('%s/test_%d_%d.h5', tmpFold, epo, iMini))
  end

  -- write hdf
  lib.hdfW(ha, tmpIn0:float(), 'input0')
  for iStn = 1, nStn do
    lib.hdfW(ha, tmpIns[iStn]:float(), string.format('input%d', iStn))
    lib.hdfW(ha, tmpGrids[iStn]:float(), string.format('grid%d', iStn))
  end

  lib.hdfWOut(ha)
end

----------------------------------------------------------------------
-- Debug for the gradient.
--
-- Input
--   model    -  data provider
--   tmpFold  -  fold to write
--   epo      -  epoch index
--   iMini    -  mini-batch index
--   train    -  train or test
--   opt      -  option
--   con      -  solver
--   denormalize  -  denormalization function
function tr_deb.debStnGrad(model, tmpFold, epo, iMini, train, opt, con, denormalize)
  -- only for train
  if not train then
    return
  end

  -- create fold if necessary
  local tmpFold = opt.CONF.tmpFold
  if not paths.dirp(tmpFold) then
    lib.pr('%s not exist', tmpFold)
    os.execute('mkdir -p ' .. tmpFold)
  end

  -- parameter
  nStn = con.nStn

  -- original image
  local imgOut0 = model:findModules('nn.Identity')[1].output:clone()
  imgOut0 = denormalize(imgOut0)

  -- stn
  local imgOuts = {}
  local imgGrads = {}
  local gridOuts = {}
  local gridGrads = {}
  for iStn = 1, nStn do
    -- image input
    local imgMod = model:findModules('nn.Transpose')[iStn + 1]
    imgOuts[iStn] = imgMod.output:clone()
    imgOuts[iStn] = denormalize(imgOuts[iStn])
    imgGrads[iStn] = imgMod.gradInput

    -- grid
    local gridMod = model:findModules('nn.AffineGridGeneratorBHWD')[iStn]
    gridOuts[iStn] = gridMod.output
    gridGrads[iStn] = gridMod.gradInput
  end

  -- hdf handler
  local ha = lib.hdfWIn(string.format('%s/train_%d_%d_grad.h5', tmpFold, epo, iMini))

  -- write hdf
  lib.hdfW(ha, imgOut0:float(), 'imgOut0')
  for iStn = 1, nStn do
    lib.hdfW(ha, imgOuts[iStn]:float(), string.format('imgOut%d', iStn))
    lib.hdfW(ha, imgGrads[iStn]:float(), string.format('imgGrad%d', iStn))

    lib.hdfW(ha, gridOuts[iStn]:float(), string.format('gridOut%d', iStn))
    lib.hdfW(ha, gridGrads[iStn]:float(), string.format('gridGrad%d', iStn))
  end

  lib.hdfWOut(ha)
end

return tr_deb

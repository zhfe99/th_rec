#!/usr/bin/env th
-- Debug in training.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 2015-08
--   modify  -  Feng Zhou (zhfe99@gmail.com), 2015-08

local lib = require('lua_lib')
local tr_deb = {}

----------------------------------------------------------------------
-- Update for one epoch.
--
-- Input
--   model    -  data provider
--   towpath  -  path
--   epo      -  epoch index
--   iMini    -  mini-batch index
--   train    -  train or test
--   opt      -  option
--   solConf  -  solver
function tr_deb.debStn(model, tmpFold, epo, iMini, train, opt, solConf, denormalize)
  -- create fold if necessary
  local tmpFold = opt.CONF.tmpFold
  if not paths.dirp(tmpFold) then
    lib.pr('%s not exist', tmpFold)
    os.execute('mkdir -p ' .. tmpFold)
  end

  -- parameter
  nStn = solConf.nStn

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
  lib.hdfW(ha, tmpWeight:float(), 'weight')
  lib.hdfW(ha, tmpBias:float(), 'bias')

  lib.hdfWOut(ha)
end

return tr_deb

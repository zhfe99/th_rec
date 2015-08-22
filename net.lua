#!/usr/bin/env th
-- Provide the net.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-18-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-22-2015

local lib = require('lua_lib')
local th = require('lua_th')
local alex = require('model.alex')
local goo = require('model.goo')
local net = {}

----------------------------------------------------------------------
-- Create a new model.
--
-- Input
--   solConf  -  solver configuration
--   opt      -  options
--
-- Output
--   model    -  model
--   loss     -  loss
--   modelSv  -  model for saving
--   mod1s    -  modules (level 1)
--   mod2s    -  modules (level 2)
--   optStat  -  optimize state
function net.newMod(solConf, opt)
  -- default parameter
  local iniAlg = solConf.iniAlg or 'xavier_caffe'
  local nC = solConf.nC or #opt.DATA.cNms

  -- model & sub-modules
  local model, mod1s, mod2s
  if lib.startswith(solConf.netNm, 'alexTS') then
    model, mod1s, mod2s = alex.newStn(nC, true, iniAlg, solConf.tran, solConf.loc)

  elseif lib.startswith(solConf.netNm, 'alexT') then
    model, mod1s = alex.newT(nC, true, iniAlg)

  elseif lib.startswith(solConf.netNm, 'alex') then
    model, mod1s = alex.new(nC, true, iniAlg)

  elseif lib.startswith(solConf.netNm, 'goo') then
    model, mod1s = goo.new(nC, true, iniAlg)

  else
    assert(nil, string.format('unknown net: %s', solConf.netNm))
  end

  -- index of sub-modules
  local idx1 = th.idxMod(model, mod1s)
  local idx2 = th.idxMod(model, mod2s)

  -- loss
  local loss = nn.ClassNLLCriterion()

  -- multi-gpu
  model = th.getModGpu(model, opt.gpus)

  -- convert inner data to gpu
  if #opt.gpus == 1 then
    model:cuda()
  end
  loss:cuda()

  -- re-locate sub-module
  mod1s = th.subMod(model, idx1)
  mod2s = th.subMod(model, idx2)

  -- save model
  local modelSv = th.getModSv(model, #opt.gpus)

  -- init optimization state
  local optStat = {
    learningRate = solConf.lrs[1][3],
    weightDecay = solConf.lrs[1][4],
    momentum = 0.9,
    learningRateDecay = 0.0,
    dampening = 0.0
  }

  return model, loss, modelSv, mod1s, mod2s, optStat
end

return net

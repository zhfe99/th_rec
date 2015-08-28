#!/usr/bin/env th
-- Provide the net.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-18-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 2015-08

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
  local ini = solConf.ini or 'xavier_caffe'
  local bn = solConf.bn or 1
  local nC = solConf.nC or #opt.DATA.cNms

  -- model & sub-modules
  local model, mod1s, mod2s
  if lib.startswith(solConf.netNm, 'alexTS2') then
    model, mod1s, mod2s = alex.newTS2(nC, bn, ini, solConf.tran, solConf.loc, 2)

  elseif lib.startswith(solConf.netNm, 'alexTS') then
    model, mod1s, mod2s = alex.newTS(nC, bn, ini, solConf.tran, solConf.loc)

  elseif lib.startswith(solConf.netNm, 'alexT') then
    model, mod1s = alex.newT(nC, bn, ini)

  elseif lib.startswith(solConf.netNm, 'alexd') then
    model, mod1s = alex.newd(nC, bn, ini)

  elseif lib.startswith(solConf.netNm, 'alexc') then
    model, mod1s = alex.newc(nC, bn, ini)

  elseif lib.startswith(solConf.netNm, 'alex') then
    model, mod1s = alex.new(nC, bn, ini)

  elseif lib.startswith(solConf.netNm, 'gooTS') then
    model, mod1s, mod2s = goo.newTS(nC, bn, ini, solConf.tran, solConf.loc)

  elseif lib.startswith(solConf.netNm, 'gooT') then
    model, mod1s = goo.newT(nC, bn, ini)

  elseif lib.startswith(solConf.netNm, 'gooc') then
    model, mod1s = goo.newc(nC, bn, ini)

  elseif lib.startswith(solConf.netNm, 'goo') then
    model, mod1s = goo.new(nC, bn, ini)

  else
    assert(nil, string.format('unknown net: %s', solConf.netNm))
  end

  -- index of sub-modules
  local idx1 = th.idxMod(model, mod1s)
  local idx2 = th.idxMod(model, mod2s)

  -- loss
  local loss
  if lib.startswith(solConf.netNm, 'alex') then
    loss = nn.ClassNLLCriterion()
  elseif lib.startswith(solConf.netNm, 'gooc') then
    local NLL = nn.ClassNLLCriterion()
    loss = nn.ParallelCriterion(true):add(NLL):add(NLL,0.3):add(NLL,0.3)
  else
    loss = nn.ClassNLLCriterion()
  end

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

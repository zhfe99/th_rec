#!/usr/bin/env th
-- Provide the net.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 2015-08
--   modify  -  Feng Zhou (zhfe99@gmail.com), 2015-08

local lib = require('lua_lib')
local th = require('lua_th')
local alx = require('model.alx')
local goo = require('model.goo')
local len = require('model.len')
local stn = require('model.stnet')
local net = {}

----------------------------------------------------------------------
-- Create the STN model with fine-tuning.
--
-- In: 1 image
-- Out: nC x softmax scores
--
-- Input
--   base   -  base net name
--   nC     -  #classes
--   bn     -  type of BN
--   ini    -  init method
--   tran   -  transformation name
--   loc    -  locnet name
--   m      -  #transformation
--
-- Output
--   model  -  model
--   mods   -  module needed to be update, m x
--   modSs  -  module needed to be update, m x
function net.newStn(base, nC, bn, ini, tran, loc, m)
  assert(tran)
  assert(loc)
  assert(m)

  -- concat
  local model = nn.Sequential()

  -- localization net
  local locNet, modLs, k
  if base == 'alx' then
    locNet, modLs, k = alx.newStnLoc(bn, ini, loc)
  elseif base == 'goo' then
    locNet, modLs, k = goo.newStnLoc(bn, ini, loc)
  else
    assert(nil, string.format('unknown base: %s', base))
  end

  -- stn net: 1 image => m images
  local inSiz = 224
  local stnNet, modSs = stn.new(locNet, tran, k, inSiz, m)
  model:add(stnNet)

  -- classify net: m images => nC x softmax scores
  local clfyNet, modAs
  if base == 'alx' then
    clfyNet, modAs = alx.newStnClfy(nC, ini, m)
  elseif base == 'goo' then
    clfyNet, modAs = goo.newStnClfy(nC, ini, m)
  else
    assert(nil, string.format('unknown base: %s', base))
  end
  model:add(clfyNet)

  -- model needed to re-train
  local mods = lib.tabCon(modLs, modSs, modAs)

  return model, mods, modSs
end

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
function net.new(solConf, opt)
  lib.prIn('net.new')

  -- default parameter
  local ini = solConf.ini or 'xavier_caffe'
  local nStn = solConf.nStn or 1
  local bn = solConf.bn or 1
  local nC = solConf.nC or #opt.DATA.cNms
  local tran = solConf.tran or 'aff'
  local loc = solConf.loc or 'type1'

  -- model & sub-modules
  local model, mod1s, mod2s
  if lib.startswith(solConf.netNm, 'alxS') then
    model, mod1s, mod2s = net.newStn('alx', nC, bn, ini, tran, loc, nStn)

  elseif lib.startswith(solConf.netNm, 'gooS') then
    model, mod1s, mod2s = net.newStn('goo', nC, bn, ini, tran, loc, nStn)

  elseif lib.startswith(solConf.netNm, 'alxT') then
    model, mod1s = alx.newT(nC, bn, ini)

  elseif lib.startswith(solConf.netNm, 'alxd') then
    model, mod1s = alx.newd(nC, bn, ini)

  elseif lib.startswith(solConf.netNm, 'alxc') then
    model, mod1s = alx.newc(nC, bn, ini)

  elseif lib.startswith(solConf.netNm, 'alx') then
    model, mod1s = alx.new(nC, bn, ini)

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
  if lib.startswith(solConf.netNm, 'gooc') then
    local NLL = nn.ClassNLLCriterion()
    loss = nn.ParallelCriterion(true):add(NLL):add(NLL,0.3):add(NLL,0.3)
  else
    loss = nn.ClassNLLCriterion()
  end

  -- multi-gpu
  model = th.getModGpu(model, opt.nGpu)

  -- convert inner data to gpu
  if opt.nGpu == 1 then
    model:cuda()
  end
  loss:cuda()

  -- re-locate sub-module
  mod1s = th.subMod(model, idx1)
  mod2s = th.subMod(model, idx2)

  -- save model
  local modelSv = th.getModSv(model, opt.nGpu)

  -- init optimization state
  local optStat = {
    learningRate = solConf.lrs[1][3],
    weightDecay = solConf.lrs[1][4],
    momentum = 0.9,
    learningRateDecay = 0.0,
    dampening = 0.0
  }

  lib.prOut()
  return model, loss, modelSv, mod1s, mod2s, optStat
end

return net

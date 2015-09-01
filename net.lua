#!/usr/bin/env th
-- Create network.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 2015-08
--   modify  -  Feng Zhou (zhfe99@gmail.com), 2015-09

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
-- In: d x h x w image
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
--   modss  -  sub-modules, two-level table
function net.newStn(base, nC, bn, ini, tran, loc, m)
  -- concat
  local model = nn.Sequential()

  -- localization net
  local locNet, locMods, k, inSiz
  if base == 'alx' then
    locNet, locMods, k = alx.newStnLoc(bn, ini, loc)
    inSiz = 224
  elseif base == 'goo' then
    locNet, locMods, k = goo.newStnLoc(bn, ini, loc)
    inSiz = 224
  elseif base == 'len' then
    locNet, locMods, k = len.newStnLoc(ini)
    inSiz = 32
  else
    assert(nil, string.format('unknown base: %s', base))
  end

  -- stn net: 1 image => m images
  local stnNet, stnMods = stn.new(locNet, tran, k, inSiz, m)
  model:add(stnNet)

  -- classify net: m images => nC x softmax scores
  local clfyNet, clfyMods
  if base == 'alx' then
    clfyNet, clfyMods = alx.newStnClfy(nC, ini, m)
  elseif base == 'goo' then
    clfyNet, clfyMods = goo.newStnClfy(nC, ini, m)
  elseif base == 'len' then
    clfyNet, clfyMods = len.newStnClfy(nC, ini, m)
  else
    assert(nil, string.format('unknown base: %s', base))
  end
  model:add(clfyNet)

  return model, {clfyMods, {locNet}, locMods, stnMods}
end

----------------------------------------------------------------------
-- Create a new model.
--
-- Input
--   con      -  solver configuration
--   opt      -  options
--
-- Output
--   model    -  model
--   loss     -  loss
--   modelSv  -  model for saving
--   modss    -  modules of different levels
--   optStat  -  optimize state
function net.new(con, opt)
  lib.prIn('net.new')

  -- default parameter
  local ini = con.ini or 'xavier_caffe'
  local nStn = con.nStn or 1
  local bn = con.bn or 1
  local nC = con.nC or #opt.DATA.cNms
  local tran = con.tran or 'aff'
  local loc = con.loc or 'type1'

  -- model & sub-modules
  local model, modss
  if lib.startswith(con.netNm, 'alxS') then
    model, modss = net.newStn('alx', nC, bn, ini, tran, loc, nStn)

  elseif lib.startswith(con.netNm, 'gooS') then
    model, modss = net.newStn('goo', nC, bn, ini, tran, loc, nStn)

  elseif lib.startswith(con.netNm, 'lenS') then
    model, modss = net.newStn('len', nC, bn, ini, tran, loc, nStn)

  elseif lib.startswith(con.netNm, 'alxT') then
    model, modss = alx.newT(nC, bn, ini)

  elseif lib.startswith(con.netNm, 'alxd') then
    model, modss = alx.newd(nC, bn, ini)

  elseif lib.startswith(con.netNm, 'alxc') then
    model, modss = alx.newc(nC, bn, ini)

  elseif lib.startswith(con.netNm, 'alx') then
    model, modss = alx.new(nC, bn, ini)

  elseif lib.startswith(con.netNm, 'gooT') then
    model, modss = goo.newT(nC, bn, ini)

  elseif lib.startswith(con.netNm, 'gooc') then
    model, modss = goo.newc(nC, bn, ini)

  elseif lib.startswith(con.netNm, 'goo') then
    model, modss = goo.new(nC, bn, ini)

  else
    assert(nil, string.format('unknown net: %s', con.netNm))
  end

  -- index of sub-modules
  local idxs = th.idxMod(model, modss)

  -- loss
  local loss
  if lib.startswith(con.netNm, 'gooc') then
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
  modss = th.subMod(model, idxs)

  -- save model
  local modelSv = th.getModSv(model, opt.nGpu)

  -- init optimization state
  local optStat = {
    learningRate = con.lrs[1][3],
    weightDecay = con.lrs[1][4],
    momentum = 0.9,
    learningRateDecay = 0.0,
    dampening = 0.0
  }

  lib.prOut()
  return model, loss, modelSv, modss, optStat
end

return net

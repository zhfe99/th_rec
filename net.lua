#!/usr/bin/env th
-- Net.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-18-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-19-2015

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
--   model
--   loss
--   modelSv
--   optStat
--   modTs
function net.newMod(solConf, opt)
  -- model & sub-modules
  local model, mods
  if lib.startswith(solConf.netNm, 'alexbnT') then
    model, mods = alex.newT(solConf.nC, true, solConf.iniAlg)

  elseif lib.startswith(solConf.netNm, 'alexbnS') then
    model, mods = alex.newStn(solConf.nC, true, solConf.iniAlg)

  elseif lib.startswith(solConf.netNm, 'alexbn') then
    model, mods = alex.new(solConf.nC, true, solConf.iniAlg)

  elseif lib.startswith(solConf.netNm, 'goobn') then
    model, mods = goo.new(solConf.nC, true, solConf.iniAlg)

  else
    assert(nil, string.format('unknown net: %s', solConf.netNm))
  end

  -- index of sub-modules
  local idx = th.idxMod(model, mods)

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
  local mods = th.subMod(model, idx)

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

  return model, loss, modelSv, mods, optStat
end

return net

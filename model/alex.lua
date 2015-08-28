#!/usr/bin/env th
-- AlexNet Model.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-04-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 2015-08

require 'cudnn'
require 'cunn'
local lib = require('lua_lib')
local th = require('lua_th')
local alex = {}
local modPath0 = paths.concat(paths.home, 'save/imgnet/torch/model/imgnet_v2_alexbn_2gpu.t7')

local eps = 1e-5
local isAff = false

----------------------------------------------------------------------
-- Create the basic alexnet model.
--
-- Input
--   nC     -  #classes
--   bn     -  type of BN
--   ini    -  initialize method
--
-- Output
--   model  -  model
--   mods   -  {}
function alex.newc(nC, bn, ini)
  -- convolution
  local features = nn.Sequential()

  -- conv1
  features:add(cudnn.SpatialConvolution(3,96,11,11,4,4,2,2))       -- 224 -> 55
  features:add(cudnn.ReLU(true))
  features:add(cudnn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27

  -- conv2
  if isBn then
    features:add(nn.SpatialBatchNormalization(96, nil, nil, false))
  end
  features:add(cudnn.SpatialConvolution(96,256,5,5,1,1,2,2))       -- 27 -> 27
  features:add(cudnn.ReLU(true))
  features:add(cudnn.SpatialMaxPooling(3,3,2,2))                   -- 27 ->  13

  -- conv3
  if isBn then
    features:add(nn.SpatialBatchNormalization(256, nil, nil, false))
  end
  features:add(cudnn.SpatialConvolution(256,384,3,3,1,1,1,1))      -- 13 ->  13
  features:add(cudnn.ReLU(true))

  -- conv4
  if isBn then
    features:add(nn.SpatialBatchNormalization(384, 1e-3))
  end
  features:add(cudnn.SpatialConvolution(384,256,3,3,1,1,1,1))      -- 13 ->  13
  features:add(cudnn.ReLU(true))

  -- conv5
  if isBn then
    features:add(nn.SpatialBatchNormalization(256, nil, nil, false))
  end
  features:add(cudnn.SpatialConvolution(256,256,3,3,1,1,1,1))      -- 13 ->  13
  features:add(cudnn.ReLU(true))
  features:add(cudnn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

  -- full1
  if isBn then
    features:add(nn.SpatialBatchNormalization(256, nil, nil, false))
  end
  local classifier = nn.Sequential()
  classifier:add(nn.View(256*6*6))
  classifier:add(nn.Dropout(0.5))
  classifier:add(nn.Linear(256*6*6, 4096))
  classifier:add(nn.Threshold(0, 1e-6))

  -- full2
  if isBn then
    classifier:add(nn.BatchNormalization(4096, nil, nil, false))
  end
  classifier:add(nn.Dropout(0.5))
  local ln = nn.Linear(4096, 4096)
  classifier:add(ln)
  classifier:add(nn.Threshold(0, 1e-6))

  -- full3
  if isBn then
    classifier:add(nn.BatchNormalization(4096, nil, nil, false))
  end
  classifier:add(nn.Linear(4096, nC))

  -- prob
  classifier:add(nn.LogSoftMax())

  -- concatenate
  local model = nn.Sequential()
  model:add(features):add(classifier)

  -- init
  th.iniMod(model, ini)

  return model, {}
end

----------------------------------------------------------------------
-- Create the basic alexnet model.
--
-- Input
--   nC      -  #classes
--   bn      -  type of bn, 0 | 1 | 2
--   ini     -  initialize method
--
-- Output
--   model   -  model
--   mods    -  {}
function alex.new(nC, bn, ini)
  -- conv1
  local features = nn.Sequential()
  features:add(cudnn.SpatialConvolution(3,96,11,11,4,4,2,2))       -- 224 -> 55
  th.addSBN(features, 96, bn)
  features:add(cudnn.ReLU(true))
  features:add(cudnn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27

  -- conv2
  features:add(cudnn.SpatialConvolution(96,256,5,5,1,1,2,2))       -- 27 -> 27
  th.addSBN(features, 256, bn)
  features:add(cudnn.ReLU(true))
  features:add(cudnn.SpatialMaxPooling(3,3,2,2))                   -- 27 ->  13

  -- conv3
  features:add(cudnn.SpatialConvolution(256,384,3,3,1,1,1,1))      -- 13 ->  13
  th.addSBN(features, 384, bn)
  features:add(cudnn.ReLU(true))

  -- conv4
  features:add(cudnn.SpatialConvolution(384,256,3,3,1,1,1,1))      -- 13 ->  13
  th.addSBN(features, 256, bn)
  features:add(cudnn.ReLU(true))

  -- conv5
  features:add(cudnn.SpatialConvolution(256,256,3,3,1,1,1,1))      -- 13 ->  13
  th.addSBN(features, 256, bn)
  features:add(cudnn.ReLU(true))
  features:add(cudnn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

  -- full1
  local classifier = nn.Sequential()
  classifier:add(nn.View(256*6*6))
  classifier:add(nn.Dropout(0.5))
  classifier:add(nn.Linear(256*6*6, 4096))
  th.addBN(classifier, 4096, bn)
  classifier:add(nn.Threshold(0, 1e-6))

  -- full2
  classifier:add(nn.Dropout(0.5))
  classifier:add(nn.Linear(4096, 4096))
  th.addBN(classifier, 4096, bn)
  classifier:add(nn.Threshold(0, 1e-6))

  -- output
  classifier:add(nn.Linear(4096, nC))
  classifier:add(nn.LogSoftMax())

  -- concatenate
  local model = nn.Sequential()
  model:add(features):add(classifier)

  -- init
  th.iniMod(model, ini)

  return model, {}
end

----------------------------------------------------------------------
-- Create alexnet model for fine-tuning.
--
-- Input
--   nC     -  #classes
--   bn     -  type of BN
--   ini    -  initialize method
--
-- Output
--   model  -  pre-trained model
--   mods   -  sub-modules needed to re-train, m x
function alex.newT(nC, bn, ini)
  local model = torch.load(modPath0)

  -- remove last fully connected layer
  model.modules[2]:remove(10)

  -- insert a new one
  local mod = nn.Linear(4096, nC)
  model.modules[2]:insert(mod, 10)

  -- init
  th.iniMod(mod, ini)

  return model, {mod}
end

----------------------------------------------------------------------
-- Create alexnet model for fine-tuning.
--
-- Input
--   m       -  #model
--   nC      -  #classes
--   bn      -  type of BN
--   ini     -  initialize method
--
-- Output
--   model   -  pre-trained model
--   mods    -  sub-modules needed to re-train, m x
function alex.newT2(m, nC, isBn, ini)
  -- alex net
  local model = nn.Sequential()

  -- feature extraction
  local alNets = nn.ParallelTable()
  model:add(alNets)
  for i = 1, m do
    local alNet = torch.load(modPath0)
    alNets:add(alNet)

    -- remove last fully connected layer
    alNet.modules[2]:remove(11)
    alNet.modules[2]:remove(10)
  end

  -- concate the output
  model:add(nn.JoinTable(2))

  -- insert a new last layer
  local mod = nn.Linear(4096 * m, nC)
  model:add(mod)
  model:add(nn.LogSoftMax())

  -- init
  th.iniMod(mod, ini)

  return model, {mod}
end

----------------------------------------------------------------------
-- Create the localization net for STN.
--
-- Input
--   bn     -  type of BN
--   ini    -  init method
--   loc    -  localization network
--
-- Output
--   model  -  model
--   mods   -  sub-modules needed to re-train, m x
function alex.newStnLoc(bn, ini, loc)
  -- load old model
  local model = torch.load(modPath0)
  local mod, k

  if loc == 'type1' then
    -- remove the classifier layer
    model:remove(2)

    -- add a new classifier layer
    k = 128
    local classifier = nn.Sequential()
    classifier:add(nn.View(256 * 6 * 6))
    -- classifier:add(nn.Dropout(0.5))

    mod = nn.Linear(256 * 6 * 6, k)
    classifier:add(mod)
    th.addBN(classifier, k, bn)
    -- classifier:add(nn.Threshold(0, 1e-6))
    classifier:add(cudnn.ReLU(true))

    model:add(classifier)

    -- init
    th.iniMod(classifier, ini)

  elseif loc == 'type2' then
    -- remove the classifier layer
    model:remove(2)

    -- add a new classifier layer
    k = 64
    local classifier = nn.Sequential()
    classifier:add(nn.View(256 * 6 * 6))
    classifier:add(nn.Dropout(0.5))

    mod = nn.Linear(256 * 6 * 6, k)
    classifier:add(mod)

    th.addBN(classifier, k, bn)
    classifier:add(cudnn.ReLU(true))

    model:add(classifier)

    -- init
    th.iniMod(classifier, ini)

  else
    assert(nil, string.format('unknown loc: %s', loc))
  end

  return model, {mod}, k
end

----------------------------------------------------------------------
-- Create the alexnet stn model with fine-tuning.
--
-- Input
--   nC     -  #classes
--   bn     -  type of BN
--   ini    -  init method
--   tran   -  transformation name
--   loc    -  locnet name
--
-- Output
--   model  -  model
--   mods   -  module needed to be update, m x
--   modSs  -  module needed to be update, m x
function alex.newTS(nC, bn, ini, tran, loc)
  local stn = require('model.stnet')
  assert(tran)
  assert(loc)

  -- locnet
  local locnet, modLs, k = alex.newStnLoc(bn, ini, loc)

  -- stn net
  local inSiz = 224
  local stnet, modSs = stn.new(locnet, bn, tran, k, inSiz)

  -- alex net
  local alnet, modAs = alex.newT(nC, bn, ini)

  -- concat
  local model = nn.Sequential()
  model:add(stnet)
  model:add(alnet)

  -- model needed to re-train
  local mods = lib.tabCon(modLs, modSs, modAs)

  return model, mods, modSs
end

----------------------------------------------------------------------
-- Create the alexnet stn model with fine-tuning.
--
-- Input
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
function alex.newTS2(nC, bn, ini, tran, loc, m)
  local stn = require('model.stnet')
  assert(tran)
  assert(loc)
  assert(m)

  -- concat
  local model = nn.Sequential()

  -- locnet
  local locNet, modLs, k = alex.newStnLoc(bn, ini, loc)

  -- stn net
  local inSiz = 224
  local stnNet, modSs = stn.new2(locNet, tran, k, inSiz, m)
  model:add(stnNet)

  -- alex net
  local alNet, modAs = alex.newT2(m, nC, bn, ini)
  model:add(alNet)

  -- model needed to re-train
  local mods = lib.tabCon(modLs, modSs, modAs)

  return model, mods, modSs
end

return alex

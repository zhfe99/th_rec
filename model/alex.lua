#!/usr/bin/env th
-- AlexNet Model.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-04-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-19-2015

require 'cudnn'
require 'cunn'
local lib = require('lua_lib')
local th = require('lua_th')
local alex = {}
local modPath0 = paths.concat(paths.home, 'save/imgnet/torch/model/imgnet_v2_alexbn_2gpu.t7')

----------------------------------------------------------------------
-- Create the basic alexnet model.
--
-- Input
--   nC      -  #classes
--   isBn    -  flag of using BN, true | false
--   iniAlg  -  initialize method
--
-- Output
--   model   -  model
--   mods    -  {}
function alex.new(nC, isBn, iniAlg)
  -- convolution
  local features = nn.Sequential()
  features:add(cudnn.SpatialConvolution(3,96,11,11,4,4,2,2))       -- 224 -> 55
  if isBn then
    features:add(nn.SpatialBatchNormalization(96, 1e-3))
  end
  features:add(cudnn.ReLU(true))
  features:add(cudnn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
  features:add(cudnn.SpatialConvolution(96,256,5,5,1,1,2,2))       -- 27 -> 27
  if isBn then
    features:add(nn.SpatialBatchNormalization(256, 1e-3))
  end
  features:add(cudnn.ReLU(true))
  features:add(cudnn.SpatialMaxPooling(3,3,2,2))                   -- 27 ->  13
  features:add(cudnn.SpatialConvolution(256,384,3,3,1,1,1,1))      -- 13 ->  13
  if isBn then
    features:add(nn.SpatialBatchNormalization(384, 1e-3))
  end
  features:add(cudnn.ReLU(true))
  features:add(cudnn.SpatialConvolution(384,256,3,3,1,1,1,1))      -- 13 ->  13
  if isBn then
    features:add(nn.SpatialBatchNormalization(256, 1e-3))
  end
  features:add(cudnn.ReLU(true))
  features:add(cudnn.SpatialConvolution(256,256,3,3,1,1,1,1))      -- 13 ->  13
  if isBn then
    features:add(nn.SpatialBatchNormalization(256, 1e-3))
  end
  features:add(cudnn.ReLU(true))
  features:add(cudnn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

  -- fully-connected
  local classifier = nn.Sequential()
  classifier:add(nn.View(256*6*6))
  classifier:add(nn.Dropout(0.5))
  classifier:add(nn.Linear(256*6*6, 4096))
  if isBn then
    classifier:add(nn.BatchNormalization(4096, 1e-3))
  end
  classifier:add(nn.Threshold(0, 1e-6))
  classifier:add(nn.Dropout(0.5))
  local ln = nn.Linear(4096, 4096)
  classifier:add(ln)
  if isBn then
    classifier:add(nn.BatchNormalization(4096, 1e-3))
  end
  classifier:add(nn.Threshold(0, 1e-6))
  classifier:add(nn.Linear(4096, nC))
  classifier:add(nn.LogSoftMax())

  -- concatenate
  local model = nn.Sequential()
  model:add(features):add(classifier)

  -- init
  th.iniMod(model, iniAlg)

  return model, {}
end

----------------------------------------------------------------------
-- Create alexnet model for fine-tuning.
--
-- Input
--   nC      -  #classes
--   isBn    -  flag of using batch normalization
--   iniAlg  -  initialize method
--
-- Output
--   model   -  pre-trained model
--   mods    -  sub-modules needed to re-train, m x
function alex.newT(nC, isBn, iniAlg)
  local model = torch.load(modPath0)

  -- remove last fully connected layer
  local mod0 = model.modules[2].modules[10]
  model.modules[2]:remove(10)

  -- insert a new one
  local mod = nn.Linear(4096, nC)
  model.modules[2]:insert(mod, 10)

  -- init
  th.iniMod(mod, iniAlg)

  return model, {mod}
end

----------------------------------------------------------------------
-- Create the localization net for STN.
--
-- Input
--   isBn    -  flag of using BN, true | false
--   iniAlg  -  init method
--
-- Output
--   model   -  model
--   mods    -  sub-modules needed to re-train, m x
function alex.newStnLoc(isBn, iniAlg)
  -- load old model
  local model = torch.load(modPath0)

  -- remove the classifier layer
  model:remove(2)

  -- add a new classifier layer
  local classifier = nn.Sequential()
  classifier:add(nn.View(256 * 6 * 6))
  classifier:add(nn.Dropout(0.5))
  local mod = nn.Linear(256 * 6 * 6, 128)
  classifier:add(mod)
  classifier:add(nn.Threshold(0, 1e-6))
  model:add(classifier)

  -- remove last fully connected layer
  -- local debugger = require('fb.debugger')
  -- debugger.enter()
  -- local mod0 = model.modules[2].modules[10]
  -- model.modules[2]:remove(11)
  -- model.modules[2]:remove(10)

  -- model.modules[2]:remove(9)
  -- model.modules[2]:remove(8)
  -- model.modules[2]:remove(7)

  -- model.modules[2]:remove(6)
  -- model.modules[2]:remove(5)
  -- model.modules[2]:remove(4)
  -- model.modules[2]:remove(3)

  -- insert a new one
  -- local mod = nn.Linear(4096, nC)
  -- model.modules[2]:insert(mod, 10)
  -- model.modules[2]:add(nn.Dropout(0.5))

  -- init
  th.iniMod(classifier, iniAlg)

  return model, {mod}
end

----------------------------------------------------------------------
-- Create the alexnet model.
--
-- Input
--   nC      -  #classes
--   isBn    -  flag of using BN, true | false
--   iniAlg  -  init method
--
-- Output
--   model   -  model
--   idxT    -  index of sub-modules
function alex.newStn(nC, isBn, iniAlg)
  -- stn net
  local stn = require 'model.stnet'
  local locnet, modLs = alex.newStnLoc(isBn, iniAlg)
  local stnet, modSs = stn.new(locnet, isBn, 224)

  -- alex net
  local alnet, modAs = alex.newT(nC, isBn, iniAlg)

  -- concat
  local model = nn.Sequential()
  model:add(stnet)
  model:add(alnet)

  -- model needed to re-train
  local mods = lib.tabCon(modLs, modSs, modAs)
  -- local debugger = require('fb.debugger')
  -- debugger.enter()

  return model, mods
end

return alex

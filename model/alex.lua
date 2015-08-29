#!/usr/bin/env th
-- AlexNet Model.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 2015-08
--   modify  -  Feng Zhou (zhfe99@gmail.com), 2015-08

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
--   bn      -  type of bn, 0 | 1 | 2
--   ini     -  initialize method
--
-- Output
--   model   -  model
--   mods    -  {}
function alex.newd(nC, bn, ini)
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
  classifier:add(nn.Linear(256*6*6, 4096))
  th.addBN(classifier, 4096, bn)
  classifier:add(nn.Threshold(0, 1e-6))

  -- full2
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
-- Create the basic alexnet model.
--
-- In: an image
-- Output: a nC-dimension softmax vector
--
-- Input
--   nC     -  #classes
--   bn     -  type of bn, 0 | 1 | 2
--   ini    -  initialize method
--
-- Output
--   model  -  model
--   mods   -  {}
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
-- In: 1 image
-- Output: nC x softmax
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
-- In: m images
-- Out: nC x softmax
--
-- Input
--   nC     -  #classes
--   ini    -  initialize method
--   m      -  #input images
--
-- Output
--   model  -  model
--   mods   -  sub-modules needed to re-train, m x
function alex.newStnClfy(nC, ini, m)
  -- alex net
  local model = nn.Sequential()

  -- feature extraction
  local alexNets = nn.ParallelTable()
  model:add(alexNets)
  for i = 1, m do
    local alexNet = torch.load(modPath0)
    alexNets:add(alexNet)

    -- remove last fully connected layer
    alexNet.modules[2]:remove(11)
    alexNet.modules[2]:remove(10)
  end

  -- concate the output
  model:add(nn.JoinTable(2))

  -- insert a new fully connected layer
  local mod = nn.Linear(4096 * m, nC)
  model:add(mod)

  -- soft-max
  model:add(nn.LogSoftMax())

  -- init
  th.iniMod(mod, ini)

  return model, {mod}
end

----------------------------------------------------------------------
-- Create the localization net for STN.
--
-- In: 1 image
-- Out: k x vector
--
-- Input
--   bn     -  type of BN
--   ini    -  init method
--   loc    -  localization network, 'type1' | 'type2'
--               'type1': k = 128, new Linear layer
--               'type2': k = 128, no Linear layer
--
-- Output
--   model  -  model
--   mods   -  sub-modules needed to re-train, m x
--   k      -  k
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

return alex

#!/usr/bin/env th
-- LeNet Model.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 2015-08
--   modify  -  Feng Zhou (zhfe99@gmail.com), 2015-08

require 'cudnn'
require 'cunn'
local lib = require('lua_lib')
local th = require('lua_th')
local len = {}

----------------------------------------------------------------------
-- Create the basic alexnet model.
--
-- In: 1 x 32 x 32 image
-- Output: a nC-dimension softmax vector
--
-- Input
--   nC     -  #classes
--   ini    -  initialize method
--
-- Output
--   model  -  model
--   mods   -  {}
function len.new(nC, ini)
  -- conv1
  local model = nn.Sequential()
  model:add(nn.View(32 * 32))
  model:add(nn.Linear(32 * 32, 128))
  model:add(cudnn.ReLU(true))
  model:add(nn.Linear(128, 128))
  model:add(cudnn.ReLU(true))
  model:add(nn.Linear(128, nC))
  model:add(nn.LogSoftMax())

  -- init
  th.iniMod(model, ini)

  return model, {}
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
function len.newStnClfy(nC, ini, m)
  -- alex net
  local model = nn.Sequential()

  -- feature extraction
  local alxNets = nn.ParallelTable()
  model:add(alxNets)
  for i = 1, m do
    local alxNet = torch.load(modPath0)
    alxNets:add(alxNet)

    -- remove last fully connected layer
    alxNet.modules[2]:remove(11)
    alxNet.modules[2]:remove(10)
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
-- In: 1 x 32 x 32 image
-- Out: k x vector
--
-- Input
--   bn     -  type of BN
--   ini    -  init method
--
-- Output
--   model  -  model
--   mods   -  sub-modules needed to re-train, m x
--   k      -  k
function len.newStnLoc(bn, ini, loc)
  local model = nn.Sequential()
  local k = 20

  model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
  local mod1 = cudnn.SpatialConvolution(1, 20, 5, 5)
  model:add(mod1)
  model:add(cudnn.ReLU(true))
  model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
  local mod2 = cudnn.SpatialConvolution(20, 20, 5, 5)
  model:add(mod2)
  model:add(cudnn.ReLU(true))
  model:add(nn.View(20 * 2 * 2))
  local mod3 = nn.Linear(20 * 2 * 2, k)
  model:add(mod3)
  model:add(cudnn.ReLU(true))

  -- init
  th.iniMod(model, ini)

  return model, {mod1, mod2, mod3}, k
end

return len

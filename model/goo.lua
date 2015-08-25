#!/usr/bin/env th
-- GoogLetNet Model.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-05-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-25-2015

require 'cudnn'
require 'nn'
local lib = require('lua_lib')
local th = require('lua_th')
local goo = {}
local modPath0 = paths.concat(paths.home, 'save/imgnet/torch/model/imgnet_v2_goobn_4gpu.t7')

----------------------------------------------------------------------
-- Create the inception component.
--
-- Input
--   input_size  -  input size
--   config      -  configuration
--   isBn        -  flag of using BN, true | false
local function inception(input_size, config, isBn)
  local concat = nn.Concat(2)
  if config[1][1] ~= 0 then
    local conv1 = nn.Sequential()
    conv1:add(cudnn.SpatialConvolution(input_size, config[1][1], 1, 1, 1, 1))
    if isBn then
      conv1:add(nn.SpatialBatchNormalization(config[1][1], 1e-3))
    end
    conv1:add(cudnn.ReLU(true))
    concat:add(conv1)
  end

  local conv3 = nn.Sequential()
  conv3:add(cudnn.SpatialConvolution(input_size, config[2][1],1,1,1,1))
  if isBn then
    conv3:add(nn.SpatialBatchNormalization(config[2][1], 1e-3))
  end
  conv3:add(cudnn.ReLU(true))
  conv3:add(cudnn.SpatialConvolution(config[2][1], config[2][2],3,3,1,1,1,1))
  if isBn then
    conv3:add(nn.SpatialBatchNormalization(config[2][2], 1e-3))
  end
  conv3:add(cudnn.ReLU(true))
  concat:add(conv3)

  local conv3xx = nn.Sequential()
  conv3xx:add(cudnn.SpatialConvolution(input_size, config[3][1],1,1,1,1))
  if isBn then
    conv3xx:add(nn.SpatialBatchNormalization(config[3][1], 1e-3))
  end
  conv3xx:add(cudnn.ReLU(true))
  conv3xx:add(cudnn.SpatialConvolution(config[3][1], config[3][2],3,3,1,1,1,1))
  if isBn then
    conv3xx:add(nn.SpatialBatchNormalization(config[3][2], 1e-3))
  end
  conv3xx:add(cudnn.ReLU(true))
  conv3xx:add(cudnn.SpatialConvolution(config[3][2], config[3][2],3,3,1,1,1,1))
  if isBn then
    conv3xx:add(nn.SpatialBatchNormalization(config[3][2], 1e-3))
  end
  conv3xx:add(cudnn.ReLU(true))
  concat:add(conv3xx)

  local pool = nn.Sequential()
  pool:add(nn.SpatialZeroPadding(1,1,1,1)) -- remove after getting cudnn R2 into fbcode
  if config[4][1] == 'max' then
    pool:add(cudnn.SpatialMaxPooling(3,3,1,1):ceil())
  elseif config[4][1] == 'avg' then
    pool:add(cudnn.SpatialAveragePooling(3,3,1,1):ceil())
  else
    error('Unknown pooling')
  end
  if config[4][2] ~= 0 then
    pool:add(cudnn.SpatialConvolution(input_size,config[4][2],1,1,1,1))
    if isBn then
      pool:add(nn.SpatialBatchNormalization(config[4][2], 1e-3))
    end
    pool:add(cudnn.ReLU(true))
  end
  concat:add(pool)

  return concat
end

----------------------------------------------------------------------
-- Create the basic GoogLeNet model.
--
-- Input
--   nC     -  #classes
--   isBn   -  flag of using BN, true | false
--   iniAlg -  init method
--
-- Output
--   model  -  model
function goo.new(nC, isBn, iniAlg)
  local features = nn.Sequential()
  features:add(cudnn.SpatialConvolution(3,64,7,7,2,2,3,3))
  if isBn then
    features:add(nn.SpatialBatchNormalization(64, 1e-3))
  end
  features:add(cudnn.ReLU(true))
  features:add(cudnn.SpatialMaxPooling(3,3,2,2):ceil())
  features:add(cudnn.SpatialConvolution(64,64,1,1))
  if isBn then
    features:add(nn.SpatialBatchNormalization(64, 1e-3))
  end
  features:add(cudnn.ReLU(true))
  features:add(cudnn.SpatialConvolution(64,192,3,3,1,1,1,1))
  if isBn then
    features:add(nn.SpatialBatchNormalization(192, 1e-3))
  end
  features:add(cudnn.ReLU(true))
  features:add(cudnn.SpatialMaxPooling(3,3,2,2):ceil())
  features:add(inception(192, {{ 64},{ 64, 64},{ 64, 96},{'avg', 32}}, isBn)) -- 3(a)
  features:add(inception(256, {{ 64},{ 64, 96},{ 64, 96},{'avg', 64}}, isBn)) -- 3(b)
  features:add(inception(320, {{  0},{128,160},{ 64, 96},{'max',  0}}, isBn)) -- 3(c)
  features:add(cudnn.SpatialConvolution(576,576,2,2,2,2))
  if isBn then
    features:add(nn.SpatialBatchNormalization(576, 1e-3))
  end
  features:add(inception(576, {{224},{ 64, 96},{ 96,128},{'avg',128}}, isBn)) -- 4(a)
  features:add(inception(576, {{192},{ 96,128},{ 96,128},{'avg',128}}, isBn)) -- 4(b)
  features:add(inception(576, {{160},{128,160},{128,160},{'avg', 96}}, isBn)) -- 4(c)
  features:add(inception(576, {{ 96},{128,192},{160,192},{'avg', 96}}, isBn)) -- 4(d)

  local main_branch = nn.Sequential()
  main_branch:add(inception(576, {{  0},{128,192},{192,256},{'max', 0}}, isBn)) -- 4(e)
  main_branch:add(cudnn.SpatialConvolution(1024,1024,2,2,2,2))
  if isBn then
    main_branch:add(nn.SpatialBatchNormalization(1024, 1e-3))
  end
  main_branch:add(inception(1024, {{352},{192,320},{160,224},{'avg',128}}, isBn)) -- 5(a)
  main_branch:add(inception(1024, {{352},{192,320},{192,224},{'max',128}}, isBn)) -- 5(b)
  main_branch:add(cudnn.SpatialAveragePooling(7,7,1,1))
  main_branch:add(nn.View(1024):setNumInputDims(3))
  main_branch:add(nn.Linear(1024, nC))
  main_branch:add(nn.LogSoftMax())

  -- add auxillary classifier here (thanks to Christian Szegedy for the details)
  -- local aux_classifier = nn.Sequential()
  -- aux_classifier:add(cudnn.SpatialAveragePooling(5,5,3,3):ceil())
  -- aux_classifier:add(cudnn.SpatialConvolution(576,128,1,1,1,1))
  -- aux_classifier:add(nn.View(128*4*4):setNumInputDims(3))
  -- aux_classifier:add(nn.Linear(128*4*4,768))
  -- aux_classifier:add(nn.ReLU())
  -- aux_classifier:add(nn.Linear(768,196))
  -- aux_classifier:add(nn.LogSoftMax())

  -- local splitter = nn.Concat(2)
  -- splitter:add(main_branch):add(aux_classifier)
  -- local model = nn.Sequential():add(features):add(splitter)
  local model = nn.Sequential():add(features):add(main_branch)

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
function goo.newT(nC, isBn, iniAlg)
  local model = torch.load(modPath0)

  -- remove last fully connected layer
  local mod0 = model.modules[2].modules[8]
  model.modules[2]:remove(8)

  -- insert a new one
  local mod = nn.Linear(1024, nC)
  model.modules[2]:insert(mod, 8)

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
--   loc     -  localization network
--
-- Output
--   model   -  model
--   mods    -  sub-modules needed to re-train, m x
function goo.newStnLoc(isBn, iniAlg, loc)
  -- load old model
  local model = torch.load(modPath0)
  local mod1, mod2, k

  if loc == 'type1' then
    local main = model.modules[2]

    -- remove the classifier layer
    main:remove(9)
    main:remove(8)
    main:remove(7)
    main:remove(6)
    -- local debugger = require('fb.debugger')
    -- debugger.enter()

    -- add a new 1x1 convolutional layer
    k = 128
    mod1 = cudnn.SpatialConvolution(1024, 128, 1, 1, 1, 1)
    main:add(mod1)
    main:add(cudnn.ReLU(true))

    -- add a fully-connected layer
    main:add(nn.View(128 * 7 * 7))
    mod2 = nn.Linear(128 * 7 * 7, k)
    main:add(mod2)
    main:add(cudnn.ReLU(true))

    -- init
    th.iniMod(mod1, iniAlg)
    th.iniMod(mod2, iniAlg)

  else
    assert(nil, string.format('unknown loc: %s', loc))
  end

  return model, {mod1, mod2}, k
end

----------------------------------------------------------------------
-- Create the goonet stn model with fine-tuning.
--
-- Input
--   nC      -  #classes
--   isBn    -  flag of using BN, true | false
--   iniAlg  -  init method
--   tran    -  transformation name
--   loc     -  locnet name
--
-- Output
--   model   -  model
--   mods    -  module needed to be update, m x
--   modSs   -  module needed to be update, m x
function goo.newTS(nC, isBn, iniAlg, tran, loc)
  local stn = require('model.stnet')
  assert(tran)
  assert(loc)

  -- locnet
  local locnet, modLs, k = goo.newStnLoc(isBn, iniAlg, loc)

  -- stn net
  local inSiz = 224
  local stnet, modSs = stn.new(locnet, isBn, tran, k, inSiz)

  -- goo net
  local goonet, modAs = goo.newT(nC, isBn, iniAlg)

  -- concat
  local model = nn.Sequential()
  model:add(stnet)
  model:add(goonet)

  -- model needed to re-train
  local mods = lib.tabCon(modLs, modSs, modAs)

  return model, mods, modSs
end

return goo

#!/usr/bin/env th
-- GoogLetNet Model.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-05-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-08-2015

require 'nn'
require 'cudnn'
local w_init = require('lua_th.w_init')

local goo = {}

----------------------------------------------------------------------
-- Inception component.
--
-- Input
--   input_size
--   config
--   isBn        -  flag of using BN, true | false
--   iniAlg      -  init method
local function inception(input_size, config, isBn, iniAlg)
  local concat = nn.Concat(2)
  if config[1][1] ~= 0 then
    local conv1 = nn.Sequential()
    conv1:add(cudnn.SpatialConvolution(input_size, config[1][1], 1, 1, 1, 1))
    if isBn then
      conv1:add(nn.SpatialBatchNormalization(config[1][1], 1e-3))
    end
    conv1:add(cudnn.ReLU(true))
    conv1 = w_init(conv1, iniAlg)
    concat:add(conv1)
  end

  local conv3 = nn.Sequential()
  conv3:add(cudnn.SpatialConvolution(  input_size, config[2][1],1,1,1,1))
  if isBn then
    conv3:add(nn.SpatialBatchNormalization(config[2][1], 1e-3))
  end
  conv3:add(cudnn.ReLU(true))
  conv3:add(cudnn.SpatialConvolution(config[2][1], config[2][2],3,3,1,1,1,1))
  if isBn then
    conv3:add(nn.SpatialBatchNormalization(config[2][2], 1e-3))
  end
  conv3:add(cudnn.ReLU(true))
  conv3 = w_init(conv3, iniAlg)
  concat:add(conv3)

  local conv3xx = nn.Sequential()
  conv3xx:add(cudnn.SpatialConvolution(  input_size, config[3][1],1,1,1,1))
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
  conv3xx = w_init(conv3xx, iniAlg)
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
  pool = w_init(pool, iniAlg)
  concat:add(pool)

  return concat
end

----------------------------------------------------------------------
-- Create the GoogLeNet model.
--
-- Input
--   nC     -  #classes
--   nGpu   -  #gpus
--   isBn   -  flag of using BN, true | false
--   iniAlg -  init method
--
-- Output
--   model  -  model
function goo.new(nC, nGpu, isBn, iniAlg)
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
  features:add(inception(192, {{ 64},{ 64, 64},{ 64, 96},{'avg', 32}}, isBn, iniAlg)) -- 3(a)
  features:add(inception(256, {{ 64},{ 64, 96},{ 64, 96},{'avg', 64}}, isBn, iniAlg)) -- 3(b)
  features:add(inception(320, {{  0},{128,160},{ 64, 96},{'max',  0}}, isBn, iniAlg)) -- 3(c)
  features:add(cudnn.SpatialConvolution(576,576,2,2,2,2))
  if isBn then
    features:add(nn.SpatialBatchNormalization(576, 1e-3))
  end
  features:add(inception(576, {{224},{ 64, 96},{ 96,128},{'avg',128}}, isBn, iniAlg)) -- 4(a)
  features:add(inception(576, {{192},{ 96,128},{ 96,128},{'avg',128}}, isBn, iniAlg)) -- 4(b)
  features:add(inception(576, {{160},{128,160},{128,160},{'avg', 96}}, isBn, iniAlg)) -- 4(c)
  features:add(inception(576, {{ 96},{128,192},{160,192},{'avg', 96}}, isBn, iniAlg)) -- 4(d)
  features = w_init(features, iniAlg)

  local main_branch = nn.Sequential()
  main_branch:add(inception(576, {{  0},{128,192},{192,256},{'max', 0}}, isBn, iniAlg)) -- 4(e)
  main_branch:add(cudnn.SpatialConvolution(1024,1024,2,2,2,2))
  if isBn then
    main_branch:add(nn.SpatialBatchNormalization(1024, 1e-3))
  end
  main_branch:add(inception(1024, {{352},{192,320},{160,224},{'avg',128}}, isBn, iniAlg)) -- 5(a)
  main_branch:add(inception(1024, {{352},{192,320},{192,224},{'max',128}}, isBn, iniAlg)) -- 5(b)
  main_branch:add(cudnn.SpatialAveragePooling(7,7,1,1))
  main_branch:add(nn.View(1024):setNumInputDims(3))
  main_branch:add(nn.Linear(1024, nC))
  main_branch:add(nn.LogSoftMax())
  main_branch = w_init(main_branch, iniAlg)

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

  if nGpu > 1 then
    assert(nGpu <= cutorch.getDeviceCount(), 'number of GPUs less than nGpu specified')
    -- require 'fbcunn'
    require 'fbcunn_files.AbstractParallel'
    require 'fbcunn_files.DataParallel'

    local model_single = model
    model = nn.DataParallel(1)
    for i = 1, nGpu do
      cutorch.withDevice(i, function() model:add(model_single:clone()) end)
    end
  end

  return model
end

return goo

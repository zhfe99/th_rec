#!/usr/bin/env th
-- AlexNet Model.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-04-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-06-2015

----------------------------------------------------------------------
-- Create the alexnet model.
--
-- Input
--   nC     -  #classes
--   nGpu   -  #gpus
--   isBn   -  flag of using BN, true | false
--   iniAlg -  init method
--
-- Output
--   model  -  model
function newModel(nC, nGpu, isBn, iniAlg)
  require 'cudnn'
  require 'cunn'
  local SpatialConvolution = cudnn.SpatialConvolution
  local SpatialMaxPooling = cudnn.SpatialMaxPooling

  -- convolution
  local features = nn.Sequential()
  features:add(SpatialConvolution(3,96,11,11,4,4,2,2))       -- 224 -> 55
  if isBn then
    features:add(nn.SpatialBatchNormalization(96, 1e-3))
  end
  features:add(cudnn.ReLU(true))
  features:add(SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
  features:add(SpatialConvolution(96,256,5,5,1,1,2,2))       --  27 -> 27
  if isBn then
    features:add(nn.SpatialBatchNormalization(256, 1e-3))
  end
  features:add(cudnn.ReLU(true))
  features:add(SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
  features:add(SpatialConvolution(256,384,3,3,1,1,1,1))      --  13 ->  13
  if isBn then
    features:add(nn.SpatialBatchNormalization(384, 1e-3))
  end
  features:add(cudnn.ReLU(true))
  features:add(SpatialConvolution(384,256,3,3,1,1,1,1))      --  13 ->  13
  if isBn then
    features:add(nn.SpatialBatchNormalization(256, 1e-3))
  end
  features:add(cudnn.ReLU(true))
  features:add(SpatialConvolution(256,256,3,3,1,1,1,1))      --  13 ->  13
  if isBn then
    features:add(nn.SpatialBatchNormalization(256, 1e-3))
  end
  features:add(cudnn.ReLU(true))
  features:add(SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

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
  classifier:add(nn.Linear(4096, 4096))
  if isBn then
    classifier:add(nn.BatchNormalization(4096, 1e-3))
  end
  classifier:add(nn.Threshold(0, 1e-6))
  classifier:add(nn.Linear(4096, nC))
  classifier:add(nn.LogSoftMax())

  -- function fillBias(m)
  --   for i=1, #m.modules do
  --     if m:get(i).bias then
  --       m:get(i).bias:fill(0.1)
  --     end
  --   end
  -- end
  -- fillBias(features)
  -- fillBias(classifier)

  local model = nn.Sequential()
  model:add(features):add(classifier)

  -- init
  w_init = require('weight_init')
  model.modules[1] = w_init(model.modules[1], iniAlg)
  model.modules[2] = w_init(model.modules[2], iniAlg)

  return model
end

#!/usr/bin/env th
-- Train using Torch.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-03-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-09-2015

require 'torch'
require 'xlua'
require 'optim'
require 'pl'
require 'eladtools'
require 'trepl'
paths.dofile('fbcunn_files/Optim.lua')
local opts = paths.dofile('opts.lua')
local th = require('lua_th')

-- argument
opt = opts.parse(arg)

-- model + loss:
local model, loss, nEpo, nEpoSv, batchSiz, lrs, mom = require(opt.network)

-- config
local config = require(opt.conf)

-- data
local data = require(opt.dataPath)

-- confusion
local dat = ThDat(opt.dbe, opt.ver)
local classes = dat.DATA.cNms
local confusion = optim.ConfusionMatrix(classes)

-- cuda
local TensorType = 'torch.FloatTensor'
if opt.type == 'cuda' then
  model:cuda()
  loss = loss:cuda()
  TensorType = 'torch.CudaTensor'
end

-- savedModel - lower footprint model to save
local savedModel = model:clone('weight', 'bias', 'running_mean', 'running_std')

-- Optimization Configuration
local optimState = {
  learningRate = 0.01,
  momentum = mom,
  weightDecay = 5e-4,
  learningRateDecay = 0.0,
  dampening = 0.0
}

local optimator = nn.Optim(model, optimState)

local function ExtractSampleFunc(data0, label0)
  assert(torch.type(data0) == 'torch.ByteTensor')
  assert(torch.type(label0) == 'torch.IntTensor')
  local data = data0
  local label = label0

  -- fit the data to multi-gpu
  if data0:size(1) % opt.nGpu > 0 then
    local b0 = data0:size(1)
    local d = data0:size(2)
    local h = data0:size(3)
    local w = data0:size(4)
    local b = b0 - b0 % opt.nGpu
    data = torch.ByteTensor(b, d, h, w):copy(data0[{{1, b}, {}, {}, {}}])
    label = torch.IntTensor(b):copy(label0[{{1, b}}])
  end
  return Normalize(data), label
end

local function paramsForEpoch(epoch)
  for _, row in ipairs(lrs) do
    if epoch >= row[1] and epoch <= row[2] then
      return {learningRate=row[3], weightDecay=row[4]}, epoch == row[1]
    end
  end
end

local function Forward(DB, train, epoch)
  confusion:zero()

  -- adjust optimizer
  if train then
    local params, newRegime = paramsForEpoch(epoch)
    optimator:setParameters(params)

    if newRegime then
      -- zero the momentum vector by throwing away previous state.
      optimator = nn.Optim(model, optimState)
    end
  end

  local SizeData = DB:size()
  local dataIndices = torch.range(1, SizeData, opt.bufferSize):long()

  -- shuffle batches from LMDB
  if train and opt.shuffle then
    dataIndices = dataIndices:index(1, torch.randperm(dataIndices:size(1)):long())
  end

  local numBuffers = 2
  local currBuffer = 1
  local BufferSources = {}
  for i = 1, numBuffers do
    BufferSources[i] = DataProvider {
      Source = {torch.ByteTensor(), torch.IntTensor()}
    }
  end

  local currBatch = 1

  local BufferNext = function()
    currBuffer = currBuffer % numBuffers + 1
    if currBatch > dataIndices:size(1) then
      BufferSources[currBuffer] = nil
      return
    end

    local sizeBuffer = math.min(opt.bufferSize, SizeData - dataIndices[currBatch] + 1)

    BufferSources[currBuffer].Data:resize(sizeBuffer, unpack(config.SampleSize))
    BufferSources[currBuffer].Labels:resize(sizeBuffer)
    DB:AsyncCacheSeq(config.Key(dataIndices[currBatch]),
                     sizeBuffer,
                     BufferSources[currBuffer].Data,
                     BufferSources[currBuffer].Labels)
    currBatch = currBatch + 1
  end

  local MiniBatch = DataProvider {
    Name = 'GPU_Batch',
    MaxNumItems = batchSiz,
    Source = BufferSources[currBuffer],
    ExtractFunction = ExtractSampleFunc,
    TensorType = TensorType
  }

  local yt = MiniBatch.Labels
  local y = torch.Tensor()
  local x = MiniBatch.Data
  local NumSamples = 0
  local loss_val = 0
  local currLoss = 0

  BufferNext()

  while NumSamples < SizeData do
    DB:Synchronize()
    MiniBatch:Reset()
    MiniBatch.Source = BufferSources[currBuffer]
    if train and opt.shuffle then MiniBatch.Source:ShuffleItems() end
    BufferNext()

    while MiniBatch:GetNextBatch() do
      if train then
        -- y, currLoss = optimizer:optimize(x, yt)
        currLoss, y = optimator:optimize(optim.sgd, x, yt, loss)
      else
        y = model:forward(x)
        currLoss = loss:forward(y, yt)
      end
      loss_val = currLoss + loss_val

      -- table results - always take first prediction
      if type(y) == 'table' then
        y = y[1]
      end

      confusion:batchAdd(y, yt)
      NumSamples = NumSamples + x:size(1)
      xlua.progress(NumSamples, SizeData)
    end
    collectgarbage()
  end
  xlua.progress(NumSamples, SizeData)
  return(loss_val / math.ceil(SizeData / batchSiz))
end

data.ValDB:Threads()
data.TrainDB:Threads()

-- each epoch
print(string.format('nEpo %d', nEpo))
for epoch = 1, nEpo do
  -- train
  model:training()
  local LossTrain = Forward(data.TrainDB, true, epoch)
  confusion:updateValids()
  print(string.format('Epoch %d, Learning Rate %f', epoch, optimator.originalOptState.learningRate))
  print(string.format('Epoch %d, Weight Decay %f', epoch, optimator.originalOptState.weightDecay))
  print(string.format('Epoch %d, Training Loss %f', epoch, LossTrain))
  print(string.format('Epoch %d, Training Acc %f', epoch, confusion.totalValid))

  -- test
  model:evaluate()
  local LossVal = Forward(data.ValDB, false, epoch)
  confusion:updateValids()
  print(string.format('Epoch %d, Validation Loss %f', epoch, LossVal))
  print(string.format('Epoch %d, Validation Acc %f', epoch, confusion.totalValid))

  -- save
  if epoch % nEpoSv == 0 then
    torch.save(opt.modPath .. '_' .. epoch .. '.t7', savedModel)
  end
end

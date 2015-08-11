#!/usr/bin/env th
-- Train using Torch.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-03-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-10-2015

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

-- data
local dat = ThDat(opt.dbe, opt.ver)
PATH = dat.PATH

-- network
local model, loss, nEpo, nEpoSv, batchSiz, sampleSiz, optStat, paramsForEpoch = require(opt.network)

-- data loader
local data = require('data_load')

-- confusion
local confusion = optim.ConfusionMatrix(dat.DATA.cNms)

-- save model
local savedModel = model:clone('weight', 'bias', 'running_mean', 'running_std')

-- init optimization
local optimator = nn.Optim(model, optStat)

----------------------------------------------------------------------
-- Update for one epoch.
--
-- Input
--   DB     -  data provider
--   train  -  train or test
--   epoch  -  epoch index
local function Forward(DB, train, epoch)
  confusion:zero()

  -- adjust optimizer
  if train then
    local params, newRegime = paramsForEpoch(epoch)
    optimator:setParameters(params)

    if newRegime then
      -- zero the momentum vector by throwing away previous state.
      optimator = nn.Optim(model, optStat)
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

    BufferSources[currBuffer].Data:resize(sizeBuffer, unpack(sampleSiz))
    BufferSources[currBuffer].Labels:resize(sizeBuffer)
    local key = string.format('%07d', dataIndices[currBatch])
    -- config.Key(dataIndices[currBatch])
    DB:AsyncCacheSeq(key, sizeBuffer, BufferSources[currBuffer].Data, BufferSources[currBuffer].Labels)
    currBatch = currBatch + 1
  end

  local MiniBatch = DataProvider {
    Name = 'GPU_Batch',
    MaxNumItems = batchSiz,
    Source = BufferSources[currBuffer],
    ExtractFunction = ExtractSampleFunc,
    TensorType = 'torch.CudaTensor'
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

-- create threads for dataloader
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

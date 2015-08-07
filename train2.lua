#!/usr/bin/env th
-- Train using Torch.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-03-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-07-2015

require 'torch'
require 'xlua'
require 'optim'
require 'pl'
require 'eladtools'
require 'trepl'
paths.dofile('fbcunn_files/Optim.lua')
local th = require('lua_th')

cmd = torch.CmdLine()
cmd:addTime()
cmd:option('-dbe', 'car', 'database name')
cmd:option('-ver', 'v1c', 'version')
cmd:option('-con', 'alex', 'configuration')
cmd:option('-bufferSize', 1280, 'buffer size')
cmd:option('-testonly', false, 'Just test loaded net on validation set')
cmd:option('-threads', 8, 'number of threads')
cmd:option('-type', 'cuda', 'float or cuda')
cmd:option('-devid', 1, 'device ID (if using CUDA)')
cmd:option('-load', '', 'load existing net weights')
cmd:option('-shuffle', true, 'shuffle training samples')

opt = cmd:parse(arg or {})
local dbe = opt.dbe
local ver = opt.ver
local con = opt.con
opt.network = string.format('./Models/%s_%s_%s', dbe, ver, con)
opt.conf = string.format('./Models/%s_%s_conf', dbe, ver)
opt.saveFold = string.format('./save/%s/torch', dbe)
opt.dataPath = string.format('data_%s_%s', dbe, ver)

torch.setnumthreads(opt.threads)
cutorch.setDevice(opt.devid)
torch.setdefaulttensortype('torch.FloatTensor')

if opt.testonly then opt.epoch = 2 end

-- model + loss:
local model, loss, nEpo, nEpoSv, batchSiz, lrs, mom = require(opt.network)

-- config
local config = require(opt.conf)

-- data
local data = require(opt.dataPath)

-- confusion
dat = ThDat(dbe, ver)
local classes = dat.DATA.cNms
local confusion = optim.ConfusionMatrix(classes)

-- output files
local logFold = string.format('%s/log', opt.saveFold)
os.execute('mkdir -p ' .. logFold)
local logPath = string.format('%s/%s_%s_%s.log', logFold, dbe, ver, con)
cmd:log(logPath)
local modFold = string.format('%s/model', opt.saveFold)
os.execute('mkdir -p ' .. modFold)
local modPath = string.format('%s/%s_%s_%s', modFold, dbe, ver, con)

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

local function ExtractSampleFunc(data, label)
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

        -- y = model:forward(x)
        local function tmp()
          y = model:forward(x)
        end
        if not pcall(tmp) then
          local debugger = require('fb.debugger')
          debugger.enter()
        end
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
  local AccTrain, LossTrain

  -- train
  if not opt.testonly then
    model:training()
    LossTrain = Forward(data.TrainDB, true, epoch)
    confusion:updateValids()
    AccTrain = confusion.totalValid
    print(string.format('Epoch %d, Learning Rate %f', epoch, optimator.originalOptState.learningRate))
    print(string.format('Epoch %d, Weight Decay %f', epoch, optimator.originalOptState.weightDecay))
    print(string.format('Epoch %d, Training Loss %f', epoch, LossTrain))
    print(string.format('Epoch %d, Training Acc %f', epoch, AccTrain))

    -- save
    if epoch % nEpoSv == 0 then
      torch.save(modPath .. '_' .. epoch .. '.t7', savedModel)
    end
  end

  -- test
  model:evaluate()
  local LossVal = Forward(data.ValDB, false, epoch)
  confusion:updateValids()
  local AccVal = confusion.totalValid
  print(string.format('Epoch %d, Validation Loss %f', epoch, LossVal))
  print(string.format('Epoch %d, Validation Acc %f', epoch, AccVal))
end

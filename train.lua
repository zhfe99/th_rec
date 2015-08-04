#!/usr/bin/env th
-- Train using Torch.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-03-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-03-2015

require 'torch'
require 'xlua'
require 'optim'
require 'pl'
require 'eladtools'
require 'trepl'
local th = require 'lua_th'

cmd = torch.CmdLine()
cmd:addTime()
cmd:text('Training a network on ILSVRC ImageNet task')
cmd:option('-dbe', 'car', 'database name')
cmd:option('-ver', 'v1c', 'version')
cmd:option('-con', 'alex', 'configuration')
cmd:option('-LR', 0.01, 'learning rate')
cmd:option('-LRDecay', 0, 'learning rate decay (in #samples)')
cmd:option('-weightDecay', 5e-4, 'L2 penalty on the weights')
cmd:option('-momentum', 0.9, 'momentum')
cmd:option('-batchSize', 128, 'batch size')
cmd:option('-optimization', 'sgd', 'optimization method')
cmd:option('-epoch', -1, 'number of epochs to train, -1 for unbounded')
cmd:option('-testonly', false, 'Just test loaded net on validation set')
cmd:option('-threads', 8, 'number of threads')
cmd:option('-type', 'cuda', 'float or cuda')
cmd:option('-bufferSize', 1280, 'buffer size')
cmd:option('-devid', 1, 'device ID (if using CUDA)')
cmd:option('-nGPU', 1, 'num of gpu devices used')
cmd:option('-load', '', 'load existing net weights')
cmd:option('-optState', false, 'Save optimization state every epoch')
cmd:option('-shuffle', true, 'shuffle training samples')

opt = cmd:parse(arg or {})
local dbe = opt.dbe
local ver = opt.ver
local con = opt.con
opt.network = string.format('./Models/%s_%s_%s', dbe, ver, con)
opt.conf = string.format('./Models/%s_%s_conf', dbe, ver)
opt.save = string.format('./save/%s/torch/log/ver_con', dbe, ver, con)

torch.setnumthreads(opt.threads)
cutorch.setDevice(opt.devid)
torch.setdefaulttensortype('torch.FloatTensor')

if opt.testonly then opt.epoch = 2 end

-- model + loss:
local model = require(opt.network)
local loss = nn.ClassNLLCriterion()
if torch.type(model) == 'table' then
  if model.loss then
    loss = model.loss
  end
  model = model.model
end

-- load old model
if paths.filep(opt.load) then
  model = torch.load(opt.load)
  print('==>Loaded Net from: ' .. opt.load)
end

-- config
local config = require(opt.conf)

-- data
dataPath = string.format('data_%s_%s', dbe, ver)
dat = ThDat(dbe, ver)
local data = require(dataPath)
local classes = dat.DATA.cNms

-- confusion
local confusion = optim.ConfusionMatrix(classes)

-- output files
os.execute('mkdir -p ' .. opt.save)
cmd:log(opt.save .. '/Log.txt', opt)
local netFilename = paths.concat(opt.save, 'Net')
local logFilename = paths.concat(opt.save, 'ErrorRate.log')
local optStateFilename = paths.concat(opt.save, 'optState')
local Log = optim.Logger(logFilename)

-- cuda
local TensorType = 'torch.FloatTensor'
if opt.type == 'cuda' then
  model:cuda()
  loss = loss:cuda()
  TensorType = 'torch.CudaTensor'
end

-- optimization configuration
local Weights, Gradients = model:getParameters()

-- savedModel - lower footprint model to save
local savedModel = model:clone('weight', 'bias', 'running_mean', 'running_std')

-- Optimization Configuration
local optimState = {
  learningRate = opt.LR,
  momentum = opt.momentum,
  weightDecay = opt.weightDecay,
  learningRateDecay = opt.LRDecay
}

local optimizer = Optimizer {
  Model = model,
  Loss = loss,
  OptFunction = _G.optim[opt.optimization],
  OptState = optimState,
  Parameters = {Weights, Gradients},
}

local function ExtractSampleFunc(data, label)
  return Normalize(data), label
end

local function Forward(DB, train)
  confusion:zero()

  local SizeData = DB:size()
  local dataIndices = torch.range(1, SizeData, opt.bufferSize):long()
  if train and opt.shuffle then --shuffle batches from LMDB
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
    MaxNumItems = opt.batchSize,
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
        y, currLoss = optimizer:optimize(x, yt)

        local debugger = require('fb.debugger')
        debugger.enter()
      else
        y = model:forward(x)
        currLoss = loss:forward(y, yt)
      end
      loss_val = currLoss + loss_val
      if type(y) == 'table' then --table results - always take first prediction
        y = y[1]
      end

      confusion:batchAdd(y, yt)
      NumSamples = NumSamples + x:size(1)
      xlua.progress(NumSamples, SizeData)
    end
    collectgarbage()
  end
  xlua.progress(NumSamples, SizeData)
  return(loss_val / math.ceil(SizeData / opt.batchSize))
end

data.ValDB:Threads()
data.TrainDB:Threads()

-- each epoch
local epoch = 1
while epoch ~= opt.epoch do
  local ErrTrain, LossTrain
  if not opt.testonly then
    print('\nEpoch ' .. epoch)

    -- train
    model:training()
    LossTrain = Forward(data.TrainDB, true)
    confusion:updateValids()
    ErrTrain = (1 - confusion.totalValid)
    print('Training Loss: ' .. LossTrain)
    print('Training Classification Error: ' .. ErrTrain)

    -- save
    torch.save(netFilename .. '_' .. epoch .. '.t7', savedModel)
    if opt.optState then
      torch.save(optStateFilename .. '_epoch_' .. epoch .. '.t7', optimState)
    end
  end

  -- test
  model:evaluate()
  local LossVal = Forward(data.ValDB, false)
  confusion:updateValids()
  local ErrVal = (1 - confusion.totalValid)
  print('Validation Loss: ' .. LossVal)
  print('Validation Classification Error = ' .. ErrVal)

  -- log
  if not opt.testonly then
    Log:add{['Training Error'] = ErrTrain, ['Validation Error'] = ErrVal}
    Log:style{['Training Error'] = '-', ['Validation Error'] = '-'}
    -- Log:plot()
  end

  epoch = epoch + 1
end

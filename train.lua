#!/usr/bin/env th
-- Train using Torch.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-03-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-17-2015

require 'torch'
require 'xlua'
require 'optim'
require 'pl'
require 'eladtools'
require 'trepl'
require 'fbcunn.Optim'
local th = require('lua_th')
local lib = require('lua_lib')

-- argument
local opts = require('opts')
opt = opts.parse(arg, 'train')

-- network
local model, loss, solConf, optStat, modTs = dofile(opt.CONF.protTr)
rawset(_G, 'solConf', solConf)

-- data loader
local data_load = require('data_load')

-- confusion
local confusion = optim.ConfusionMatrix(opt.DATA.cNms)

-- save model (TODO: move to w_init)
local modelSv
if #opt.gpus > 1 then
  model:syncParameters()
  modelSv = model.modules[1]:clone('weight', 'bias', 'running_mean', 'running_std')
else
  modelSv = model:clone('weight', 'bias', 'running_mean', 'running_std')
end

-- init optimization
local optimator = nn.Optim(model, optStat)

----------------------------------------------------------------------
-- Get the parameters for each epoch.
--
-- Input
--   epoch  -  epoch id
local function parEpo(epoch, lrs)
  for _, row in ipairs(lrs) do
    if epoch >= row[1] and epoch <= row[2] then
      return {learningRate=row[3], weightDecay=row[4]}, epoch == row[1]
    end
  end
end

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
    local params, newRegime = parEpo(epoch, solConf.lrs)
    optimator:setParameters(params)

    -- zero the momentum vector by throwing away previous state.
    if newRegime then
      optimator = nn.Optim(model, optStat)
    end

    -- fine-tune model
    for _, mod in ipairs(modTs) do
      optimator.modulesToOptState[mod][1].learningRate = params.learningRate * 10
      optimator.modulesToOptState[mod][2].learningRate = params.learningRate * 10
    end

    -- local debugger = require('fb.debugger')
    -- debugger.enter()
  end

  -- dimension
  local nImg = DB:size()
  local batchSiz = solConf.batchSiz
  local bufSiz = solConf.bufSiz

  -- TODO: fix this for testing
  nImg = math.floor(nImg / batchSiz) * batchSiz

  -- buffers position
  local bufSts = torch.range(1, nImg, bufSiz):long()
  local nBuf = bufSts:size(1)
  if train and opt.shuffle then
    bufSts = bufSts:index(1, torch.randperm(nBuf):long())
  end

  -- create buffer src
  local nBufSrc = 2
  local bufSrcs = {}
  for i = 1, nBufSrc do
    bufSrcs[i] = DataProvider {Source = {torch.FloatTensor(), torch.IntTensor()}}
  end

  -- next buffer
  local iBuf = 1
  local iBufSrc = 1
  local BufferNext = function()
    iBufSrc = iBufSrc % nBufSrc + 1
    if iBuf > nBuf then
      bufSrcs[iBufSrc] = nil
      return
    end

    local bufSizi = math.min(bufSiz, nImg - bufSts[iBuf] + 1)
    bufSizi = math.floor(bufSizi / batchSiz) * batchSiz

    -- copy from LMDB provider
    bufSrcs[iBufSrc].Data:resize(bufSizi, unpack(solConf.smpSiz))
    bufSrcs[iBufSrc].Labels:resize(bufSizi)
    local key = string.format('%07d', bufSts[iBuf])
    DB:AsyncCacheSeq(key, bufSizi, bufSrcs[iBufSrc].Data, bufSrcs[iBufSrc].Labels)
    iBuf = iBuf + 1
  end

  -- mini batch
  local MiniBatch = DataProvider {
    Name = 'minibatch',
    MaxNumItems = batchSiz,
    Source = bufSrcs[iBufSrc],
    ExtractFunction = data_load.ExtractSampleFunc,
    TensorType = 'torch.CudaTensor'
  }

  -- upvalue
  local x = MiniBatch.Data
  local yt = MiniBatch.Labels
  local y = torch.Tensor()
  local nImgCurr = 0
  local loss_val = 0
  local currLoss = 0

  -- init buffer
  BufferNext()

  -- one epoch
  while nImgCurr < nImg do
    DB:Synchronize()
    MiniBatch:Reset()
    MiniBatch.Source = bufSrcs[iBufSrc]
    if train and opt.shuffle then MiniBatch.Source:ShuffleItems() end

    -- update buffer
    BufferNext()

    -- each minibatch / buffer
    while MiniBatch:GetNextBatch() do
      if train then
        if #opt.gpus > 1 then
          model:zeroGradParameters()
          model:syncParameters()
        end
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
      nImgCurr = nImgCurr + x:size(1)
      xlua.progress(nImgCurr, nImg)
    end
    collectgarbage()
  end
  xlua.progress(nImgCurr, nImg)

  return(loss_val / math.ceil(nImg / batchSiz))
end

-- create threads for dataloader
local trDB = data_load.newTrainDB()
trDB:Threads()
local teDB = data_load.newTestDB()
teDB:Threads()

-- each epoch
for epoch = 1, solConf.nEpo do
  -- train
  model:training()
  local trLoss = Forward(trDB, true, epoch)
  confusion:updateValids()
  print(string.format('epoch %d/%d, lr %f, wd %f', epoch, solConf.nEpo, optimator.originalOptState.learningRate, optimator.originalOptState.weightDecay))
  print(string.format('tr, loss %f, acc %f', trLoss, confusion.totalValid))

  -- save
  if epoch % solConf.nEpoSv == 0 then
    torch.save(opt.CONF.modPath .. '_' .. epoch .. '.t7', modelSv)
  end

  -- test
  model:evaluate()
  local teLoss = Forward(teDB, false, epoch)
  confusion:updateValids()
  print(string.format('te, loss %f, acc %f', teLoss, confusion.totalValid))
end

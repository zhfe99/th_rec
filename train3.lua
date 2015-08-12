#!/usr/bin/env th
-- Train using Torch.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-03-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-12-2015

require 'torch'
require 'xlua'
require 'optim'
require 'pl'
require 'eladtools'
require 'trepl'
paths.dofile('fbcunn_files/Optim.lua')
local th = require('lua_th')

-- argument
local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)

-- data
local dat = ThDat(opt.dbe, opt.ver)
PATH = dat.PATH

-- network
local model, loss, nEpo, nEpoSv, batchSiz, bufSiz, sampleSiz, optStat, parEpo = require(opt.network)

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
    local params, newRegime = parEpo(epoch)
    optimator:setParameters(params)

    -- zero the momentum vector by throwing away previous state.
    if newRegime then
      optimator = nn.Optim(model, optStat)
    end
  end

  -- dimension
  local nImg = DB:size()

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
    bufSrcs[iBufSrc].Data:resize(bufSizi, unpack(sampleSiz))
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
    ExtractFunction = ExtractSampleFunc,
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
data.TrainDB:Threads()
data.ValDB:Threads()

-- each epoch
for epoch = 1, nEpo do
  -- train
  model:training()
  local trLoss = Forward(data.TrainDB, true, epoch)
  confusion:updateValids()
  print(string.format('epoch %d/%d, lr %f, wd %f', epoch, nEpo, optimator.originalOptState.learningRate, optimator.originalOptState.weightDecay))
  print(string.format('tr, loss %f, acc %f', trLoss, confusion.totalValid))

  -- test
  model:evaluate()
  local teLoss = Forward(data.ValDB, false, epoch)
  confusion:updateValids()
  print(string.format('te, loss %f, acc %f', teLoss, confusion.totalValid))

  -- save
  if epoch % nEpoSv == 0 then
    torch.save(opt.modPath .. '_' .. epoch .. '.t7', savedModel)
  end
end

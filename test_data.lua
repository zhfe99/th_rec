#!/usr/bin/env th
-- Train using Torch.
--
-- Example
--   export CUDA_VISIBLE_DEVICES=0,1,2,3
--   ./train.lua -dbe imgnet -ver v2 -con alex_4gpu -gpu 0,1,2,3
--   ./train.lua -ver v1 -con alexS1 -deb
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-03-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-26-2015

require('torch')
require('xlua')
require('optim')
require('pl')
require('eladtools')
require('trepl')
require('fbcunn.Optim')
local th = require('lua_th')
local lib = require('lua_lib')
local opts = require('opts')
local net = require('net')
local data_load = require('data_load')

-- argument
opt = opts.parse(arg, 'train')

-- network
local solConf = dofile(opt.CONF.protTr)
lib.prTab(solConf, 'solConf')
local tmpFold = opt.CONF.tmpFold

-- data loader
data_load.init(opt, solConf)

----------------------------------------------------------------------
-- Update for one epoch.
--
-- Input
--   DB     -  LMDB data provider
--   train  -  train or test
--   epoch  -  epoch index
local function Forward(DB, train, epoch, confusion)
  confusion:zero()

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
    bufSrcs[i] = DataProvider({Source = {torch.FloatTensor(),
                                         torch.IntTensor()}})
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
    local smpSiz = solConf.smpSiz or {3, 224, 224}
    bufSrcs[iBufSrc].Data:resize(bufSizi, unpack(smpSiz))
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

  -- one minibatch
  local iMini = 0
  while nImgCurr < nImg do
    DB:Synchronize()
    MiniBatch:Reset()
    MiniBatch.Source = bufSrcs[iBufSrc]
    if train and opt.shuffle then MiniBatch.Source:ShuffleItems() end

    -- update buffer
    BufferNext()

    -- each minibatch / buffer
    while MiniBatch:GetNextBatch() do
      iMini = iMini + 1

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

-- confusion
local confusion = optim.ConfusionMatrix(opt.DATA.cNms)

-- each epoch
for epoch = 1, solConf.nEpo do
  -- train
  local trLoss = Forward(trDB, true, epoch, confusion)

  -- save
  if epoch % solConf.nEpoSv == 0 then
    torch.save(opt.CONF.modPath .. '_' .. epoch .. '.t7', modelSv)
  end
end

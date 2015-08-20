#!/usr/bin/env th
-- Train using Torch.
--
-- Example
--   CUDA_VISIBLE_DEVICES=0,1,2,3 th train.lua -dbe imgnet -ver v2 -con alexbn_4gpu -gpu 0,1,2,3
--   CUDA_VISIBLE_DEVICES=4,5,6,7 th train.lua -dbe imgnet -ver v2 -con alexbn_4gpu -gpu 0,1,2,3
--   CUDA_VISIBLE_DEVICES=1,2,3,4 th train.lua -dbe imgnet -ver v2 -con goobn_4gpu -gpu 0,1,2,3
--   CUDA_VISIBLE_DEVICES=7 th train.lua -ver v1 -con alexbnS -deb
--   CUDA_VISIBLE_DEVICES=7 th train.lua -ver v1 -con alexbnS2 -deb
--
-- Cudnn R3
--   export LD_LIBRARY_PATH=$apps/cudnn_v3:$LD_LIBRARY_PATH
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-03-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-20-2015

require('torch')
require('xlua')
require('optim')
require('pl')
require('eladtools')
require('trepl')
require('fbcunn.Optim')
local th = require('lua_th')
local lib = require('lua_lib')

-- argument
local opts = require('opts')
opt = opts.parse(arg, 'train')

-- network
local solConf = dofile(opt.CONF.protTr)
local net = require('net')
local model, loss, modelSv, modTs, optStat = net.newMod(solConf, opt)

local tmpFold = '/home/ma/feng/save/car/torch/tmp'

-- data loader
local data_load = require('data_load')
data_load.init(opt, solConf)

-- init optimization
local optimator = nn.Optim(model, optStat)

----------------------------------------------------------------------
-- Update for one epoch.
--
-- Input
--   DB     -  data provider
--   train  -  train or test
--   epoch  -  epoch index
local function Forward(DB, train, epoch, confusion)
  confusion:zero()

  -- adjust optimizer
  if train then
    local params, newRegime = th.parEpo(epoch, solConf.lrs)
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

    if opt.deb then
      optimator.modulesToOptState[modTs[2]][1].learningRate = params.learningRate * 0.1
      optimator.modulesToOptState[modTs[2]][2].learningRate = params.learningRate * 0.1
      print(modTs[2])
      -- optimator.modulesToOptState[modTs[1]][1].weightDecay = params.weightDecay * 100
      -- local debugger = require('fb.debugger')
      -- debugger.enter()
    end
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
    bufSrcs[i] = DataProvider({Source = {torch.FloatTensor(), torch.IntTensor()}})
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

      if train then
        if #opt.gpus > 1 then
          model:zeroGradParameters()
          model:syncParameters()
        end

        -- local debugger = require('fb.debugger')
        -- debugger.enter()
        currLoss, y = optimator:optimize(optim.sgd, x, yt, loss)

        if opt.deb and (iMini - 1) % 100 == 0 then
          local tmpWeight = model:findModules('nn.Linear')[2].weight
          local tmpBias = model:findModules('nn.Linear')[2].bias
          local tmpIn0 = model:findModules('nn.Identity')[1].output
          local tmpIn1 = model:findModules('nn.Transpose')[2].output
          local tmpGrid = model:findModules('nn.AffineGridGeneratorBHWD')[1].output

          ha = lib.hdfWIn(string.format('%s/train_%d_%d.h5', tmpFold, epoch, iMini))
          lib.hdfW(ha, tmpIn0:float(), 'input0')
          lib.hdfW(ha, tmpIn1:float(), 'input1')
          lib.hdfW(ha, tmpGrid:float(), 'grid')
          lib.hdfW(ha, tmpWeight:float(), 'weight')
          lib.hdfW(ha, tmpBias:float(), 'bias')
          lib.hdfWOut(ha)
          -- local debugger = require('fb.debugger')
          -- debugger.enter()
        end
      else
        y = model:forward(x)
        currLoss = loss:forward(y, yt)

        if opt.deb and (iMini - 1) % 100 == 0 then
          local tmpWeight = model:findModules('nn.Linear')[2].weight
          local tmpBias = model:findModules('nn.Linear')[2].bias
          local tmpIn0 = model:findModules('nn.Identity')[1].output
          local tmpIn1 = model:findModules('nn.Transpose')[2].output
          local tmpGrid = model:findModules('nn.AffineGridGeneratorBHWD')[1].output

          ha = lib.hdfWIn(string.format('%s/test_%d_%d.h5', tmpFold, epoch, iMini))
          lib.hdfW(ha, tmpIn0:float(), 'input0')
          lib.hdfW(ha, tmpIn1:float(), 'input1')
          lib.hdfW(ha, tmpGrid:float(), 'grid')
          lib.hdfW(ha, tmpWeight:float(), 'weight')
          lib.hdfW(ha, tmpBias:float(), 'bias')
          lib.hdfWOut(ha)
          -- local debugger = require('fb.debugger')
          -- debugger.enter()
        end
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

-- confusion
local confusion = optim.ConfusionMatrix(opt.DATA.cNms)

-- each epoch
for epoch = 1, solConf.nEpo do
  -- train
  model:training()
  local trLoss = Forward(trDB, true, epoch, confusion)
  confusion:updateValids()
  print(string.format('epoch %d/%d, lr %f, wd %f', epoch, solConf.nEpo, optimator.originalOptState.learningRate, optimator.originalOptState.weightDecay))
  print(string.format('tr, loss %f, acc %f', trLoss, confusion.totalValid))

  -- save
  if epoch % solConf.nEpoSv == 0 then
    torch.save(opt.CONF.modPath .. '_' .. epoch .. '.t7', modelSv)
  end

  -- test
  model:evaluate()
  local teLoss = Forward(teDB, false, epoch, confusion)
  confusion:updateValids()
  print(string.format('te, loss %f, acc %f', teLoss, confusion.totalValid))
end

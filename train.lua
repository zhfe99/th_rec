#!/usr/bin/env th
-- Train using Torch.
--
-- Example
--   export CUDA_VISIBLE_DEVICES=0,1,2,3
--   ./train.lua -dbe imgnet -ver v2 -con alex_4gpu -gpu 0,1,2,3
--   ./train.lua -ver v1 -con alexS1 -deb
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 2015-08
--   modify  -  Feng Zhou (zhfe99@gmail.com), 2015-08

require('torch')
require('xlua')
require('optim')
require('pl')
require('cunn')
require('trepl')
local th = require('lua_th')
local lib = require('lua_lib')
local opts = require('opts')
local net = require('net')
local dp = require('lmdb_provider')

-- argument
opt, solConf = opts.parse(arg, 'train')

-- network
local model, loss, modelSv, mod1s, mod2s, optStat = net.new(solConf, opt)
local optimator

-- data loader
dp.init(opt, solConf)

----------------------------------------------------------------------
-- Update for one epoch.
--
-- Input
--   DB     -  data provider
--   train  -  train or test
--   epoch  -  epoch index
local function Forward(DB, train, epoch)
  -- confusion
  local confusion = optim.ConfusionMatrix(opt.DATA.cNms)

  -- optimizer
  if train then
    optimator = th.optim(optimator, model, mod1s, mod2s, optStat, epoch, solConf.lrs)
  end

  -- init data provider
  local MiniBatch, nImg, batchSiz = dp.fordInit(DB, train, epoch, opt, solConf)

  -- upvalue
  local x = MiniBatch.Data
  local yt = MiniBatch.Labels
  local y = torch.Tensor()
  local nImgCurr = 0
  local loss_val = 0
  local currLoss = 0

  -- each buffer
  local iMini = 0
  while nImgCurr < nImg do
    dp.fordReset(DB, train, opt)

    -- each minibatch
    while MiniBatch:GetNextBatch() do
      -- do somthing
      iMini = iMini + 1
      if train then
        if #opt.gpus > 1 then
          model:zeroGradParameters()
          model:syncParameters()
        end

        local debugger = require('fb.debugger')
        debugger.enter()
        currLoss, y = optimator:optimize(optim.sgd, x, yt, loss)
        -- local debugger = require('fb.debugger')
        -- debugger.enter()
      else
        y = model:forward(x)
        currLoss = loss:forward(y, yt)
      end
      loss_val = currLoss + loss_val

      -- debug
      if opt.deb and (iMini - 1) % 100 == 0 then
        local tmpWeight = model:findModules('nn.Linear')[2].weight
        local tmpBias = model:findModules('nn.Linear')[2].bias
        local tmpIn0 = model:findModules('nn.Identity')[1].output
        local tmpIn1 = model:findModules('nn.Transpose')[2].output
        -- local tmpIn2 = model:findModules('nn.Transpose')[3].output
        local tmpGrid1 = model:findModules('nn.AffineGridGeneratorBHWD')[1].output
        -- local tmpGrid2 = model:findModules('nn.AffineGridGeneratorBHWD')[2].output

        local ha
        if train then
          ha = lib.hdfWIn(string.format('%s/train_%d_%d.h5', opt.CONF.tmpFold, epoch, iMini))
        else
          ha = lib.hdfWIn(string.format('%s/test_%d_%d.h5', opt.CONF.tmpFold, epoch, iMini))
        end
        lib.hdfW(ha, tmpIn0:float(), 'input0')
        lib.hdfW(ha, tmpIn1:float(), 'input1')
        -- lib.hdfW(ha, tmpIn2:float(), 'input2')
        lib.hdfW(ha, tmpGrid1:float(), 'grid1')
        -- lib.hdfW(ha, tmpGrid2:float(), 'grid2')
        lib.hdfW(ha, tmpWeight:float(), 'weight')
        lib.hdfW(ha, tmpBias:float(), 'bias')
        lib.hdfWOut(ha)

        -- local debugger = require('fb.debugger')
        -- debugger.enter()
      end

      -- table results - always take first prediction
      if type(y) == 'table' then
        y = y[1]
      end

      -- update
      confusion:batchAdd(y, yt)
      nImgCurr = nImgCurr + x:size(1)
      xlua.progress(nImgCurr, nImg)
    end
    collectgarbage()
  end
  xlua.progress(nImgCurr, nImg)

  -- print
  local loss = loss_val / math.ceil(nImg / batchSiz)
  confusion:updateValids()
  local acc = confusion.totalValid
  if train then
    print(string.format('epoch %d/%d, lr %f, wd %f', epoch, solConf.nEpo, optimator.originalOptState.learningRate, optimator.originalOptState.weightDecay))
    print(string.format('tr, loss %f, acc %f', loss, acc))
  else
    print(string.format('te, loss %f, acc %f', loss, acc))
  end
end

-- create threads for dataloader
local trDB = dp.newTr()
local teDB = dp.newTe()

-- each epoch
for epoch = 1, solConf.nEpo do
  -- train
  model:training()
  Forward(trDB, true, epoch)

  -- save
  if epoch % solConf.nEpoSv == 0 then
    torch.save(opt.CONF.modPath .. '_' .. epoch .. '.t7', modelSv)
  end

  -- test
  model:evaluate()
  Forward(teDB, false, epoch)
end

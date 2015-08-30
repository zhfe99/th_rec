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
local dp = require('dp_lmdb')
local tr_deb = require('tr_deb')

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
local function ford(DB, train, epo)
  -- confusion
  local confusion = optim.ConfusionMatrix(opt.DATA.cNms)

  -- optimizer
  if train then
    optimator = th.optim(optimator, model, mod1s, mod2s, optStat, epo, solConf.lrs)
  end

  -- init data provider
  local nImg, batchSiz, nMini = dp.fordInit(DB, train, epo, opt, solConf)

  -- each mini batch
  local nImgCurr = 0
  local lossVal = 0
  for iMini = 1, nMini do
    local x, yt = dp.fordNextBatch(DB, train, opt)
    local y = torch.Tensor()
    local currLoss = 0

    -- do somthing
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
    lossVal = currLoss + lossVal

    -- table results - always take first prediction
    if type(y) == 'table' then
      y = y[1]
    end

    -- debug
    if opt.deb and (iMini - 1) % 100 == 0 then
      tr_deb.debStn(model, tmpFold, epo, iMini, train, opt, solConf, dp.denormalize)
    end

    -- update
    confusion:batchAdd(y, yt)
    nImgCurr = nImgCurr + x:size(1)
    xlua.progress(nImgCurr, nImg)
    collectgarbage()
  end

  -- print
  local loss = lossVal / nMini
  confusion:updateValids()
  local acc = confusion.totalValid
  if train then
    print(string.format('epoch %d/%d, lr %f, wd %f', epo, solConf.nEpo, optimator.originalOptState.learningRate, optimator.originalOptState.weightDecay))
    print(string.format('tr, loss %f, acc %f', loss, acc))
  else
    print(string.format('te, loss %f, acc %f', loss, acc))
  end
end

-- create threads for dataloader
local trDB = dp.newTr()
local teDB = dp.newTe()

-- each epo
for epo = 1, solConf.nEpo do
  -- train
  model:training()
  ford(trDB, true, epo)

  -- save
  if epo % solConf.nEpoSv == 0 then
    torch.save(opt.CONF.modPath .. '_' .. epo .. '.t7', modelSv)
  end

  -- test
  model:evaluate()
  ford(teDB, false, epo)
end

#!/usr/bin/env th
-- Train using Torch.
--
-- Example
--   export CUDA_VISIBLE_DEVICES=0,1,2,3
--   ./train.lua -dbe imgnet -ver v2 -con alx
--   ./train.lua -ver v1 -con alexS1 -deb
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 2015-08
--   modify  -  Feng Zhou (zhfe99@gmail.com), 2015-09

require('torch')
require('optim')
-- require('cunn')
local th = require('lua_th')
local lib = require('lua_lib')
local opts = require('opts')
local net = require('net')
local dp = require('dp_load')
local tr_deb = require('tr_deb')

-- option
local opt, con = opts.parse(arg, 'train')

-- network
local model, loss, modelSv, modss, optStat = net.new(con, opt)
local optimator

-- data provider
local trDB, teDB = dp.init(opt, con)

----------------------------------------------------------------------
-- Update one epoch.
--
-- Input
--   DB     -  data provider
--   train  -  train or test
--   epo    -  epoch
local function ford(DB, train, epo)
  lib.prIn('ford', 'train %s, epo %d', train, epo)

  -- confusion
  local confusion = optim.ConfusionMatrix(opt.DATA.cNms)

  -- optimizer
  if train then
    optimator = th.optim(optimator, model, modss, optStat, epo, con.lrs)
  end

  -- init data provider
  local nImg, nBat = dp.fordInit(DB, train, epo, opt, con)

  -- each batch
  local lossVal = 0
  lib.prCIn('batch', nBat, .2)
  for iBat = 1, nBat do
    lib.prC(iBat)
    local x, yt = dp.fordNext(DB, train, opt)
    local y = torch.Tensor()
    local lossVali = 0

    -- do somthing
    if train then
      if opt.nGpu > 1 then
        model:zeroGradParameters()
        model:syncParameters()
      end

      lossVali, y = optimator:optimize(optim.sgd, x, yt, loss)
    else
      y = model:forward(x)
      lossVali = loss:forward(y, yt)
    end

    -- table results
    if type(y) == 'table' then
      y = y[1]
    end

    -- debug
    if opt.deb and (iBat - 1) % 100 == 0 then
      tr_deb.debStn(model, tmpFold, epo, iBat, train, opt, con, dp.denormalize)
      tr_deb.debStnGrad(model, tmpFold, epo, iBat, train, opt, con, dp.denormalize)
      local debugger = require('fb.debugger')
      debugger.enter()
    end

    -- update
    lossVal = lossVal + lossVali
    confusion:batchAdd(y, yt)
    collectgarbage()
  end
  lib.prCOut(nBat)

  -- print to log
  local loss = lossVal / nBat
  confusion:updateValids()
  local acc = confusion.totalValid
  if train then
    lib.pr('epoch %d/%d, lr %f, wd %f', epo, con.nEpo, optimator.originalOptState.learningRate, optimator.originalOptState.weightDecay)
    lib.pr('tr, loss %f, acc %f', loss, acc)
  else
    lib.pr('te, loss %f, acc %f', loss, acc)
  end

  lib.prOut()
end

-- each epo
lib.prCIn('epo', con.nEpo, 1)
for epo = 1, con.nEpo do
  lib.prC(epo)

  -- train
  model:training()
  ford(trDB, true, epo)

  -- save
  if epo % con.nEpoSv == 0 then
    torch.save(opt.CONF.modPath .. '_' .. epo .. '.t7', modelSv)
  end

  -- test
  model:evaluate()
  ford(teDB, false, epo)
end
lib.prCOut(con.nEpo)

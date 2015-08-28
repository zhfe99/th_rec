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
--   modify  -  Feng Zhou (zhfe99@gmail.com), 2015-08

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
local dp = require('dp_lmdb')

-- argument
opt = opts.parse(arg, 'train')

-- network
local solConf = dofile(opt.CONF.protTr)
lib.prTab(solConf, 'solConf')
local tmpFold = opt.CONF.tmpFold

-- data loader
dp.init(opt, solConf)

----------------------------------------------------------------------
-- Update for one epoch.
--
-- Input
--   DB     -  LMDB data provider
--   train  -  train or test
--   epoch  -  epoch index
local function Forward(DB, train, epoch)
  -- confusion:zero()

  -- init
  local nImg, batchSiz = dp.fordInit(DB, train, epoch, opt, solConf)

  -- check
  local nImgCurr = 0
  local loss_val = 0
  local currLoss = 0

  -- each buffer
  while nImgCurr < nImg do
    -- dp.fordReset(DB, train, opt)

    -- each minibatch
    local x, yt = dp.fordNextBatch(DB, train, opt)
    local y = torch.Tensor()

    -- do somthing
    nImgCurr = nImgCurr + x:size(1)

    -- clean
    xlua.progress(nImgCurr, nImg)
    collectgarbage()
  end
  return loss_val / math.ceil(nImg / batchSiz)
end

-- create threads for dataloader
local trDB = dp.newTr()

-- each epoch
for epoch = 1, solConf.nEpo do

  -- train
  local trLoss = Forward(trDB, true, epoch)

  -- save
  if epoch % solConf.nEpoSv == 0 then
    torch.save(opt.CONF.modPath .. '_' .. epoch .. '.t7', modelSv)
  end
end

#!/usr/bin/env th
--[[
Data provider.

History
  create  -  Feng Zhou (zhfe99@gmail.com), 2015-08
  modify  -  Feng Zhou (zhfe99@gmail.com), 2015-11
]]--

local lib = require('lua_lib')
local dp_dbe = {}

local dp = {}
local dbe

----------------------------------------------------------------------
-- Initialize data-loader.
--
-- Input
--   opt      -  option
--   solConf  -  solver configuration
--
-- Output
--   trDB     -  train DB
--   teDB     -  test DB
function dp.init(opt, solConf)
  dbe = opt.dbe
  if dbe == 'mnist' then
    dp_dbe = require('dp_mnist')
  else
    dp_dbe = require('dp_lmdb')
  end

  local trDB, teDB = dp_dbe.init(opt, solConf)
  return trDB, teDB
end

----------------------------------------------------------------------
-- De-normalize the data.
--
-- Input
--   data  -  b x d x h x w
--
-- Output
--   data  -  b x d x h x w
function dp.denormalize(data)
  return dp_dbe.denormalize(data)
end

----------------------------------------------------------------------
-- Initalize forward passing.
--
-- Input
--   DB       -  lmdb data provider
--   train    -  train or test
--   epoch    -  epoch id
--   opt      -  option
--   solConf  -  solver configuration
--
-- Output
--   nImg     -  #total image
--   nBat     -  #batch
function dp.fordInit(DB, train, epoch, opt, solConf)
  local nImg, wBat, nBat = dp_dbe.fordInit(DB, train, epo, opt, solConf)
  return nImg, nBat
end

----------------------------------------------------------------------
-- Move to the next batch.
--
-- Input
--   DB     -  lmdb provider
--   train  -  train or test
--   opt    -  option
--
-- Output
--   x      -  x value
--   yt     -  ground-truth
function dp.fordNext(DB, train, opt)
  local x, yt = dp_dbe.fordNextBatch(DB, train, opt)
  return x, yt
end

return dp

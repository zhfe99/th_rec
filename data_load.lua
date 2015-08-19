#!/usr/bin/env th
-- Data loader.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-01-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-18-2015

require 'eladtools'
require 'xlua'
require 'lmdb'
local Threads = require 'threads'
local ffi = require 'ffi'

-- upvalue used by function
local sampleSiz, InputSize, meanInfo, trLmdb, teLmdb, DataMean, DataStd

local data_load = {}

----------------------------------------------------------------------
-- Normalize the data by subtracting the mean and dividing the scale.
--
-- Input
--   data  -  b x d x h x w
function data_load.Normalize(data)
  local data = data:float()
  for j = 1, 3 do
    data[{{}, j, {}, {}}]:add(-DataMean[j])
    data[{{}, j, {}, {}}]:div(DataStd[j])
  end

  return data
end

function data_load.ExtractSampleFunc(data0, label0)
  local data = data0
  local label = label0

  return data_load.Normalize(data), label
end

----------------------------------------------------------------------
-- Extract from LMDB training data.
--
-- Input
--   key    -  key
--   data   -  data
--
-- Output
--   img    -  image
--   class  -  class
function data_load.ExtractFromLMDBTrain(key, data)
  -- class
  local class = data.c

  -- decompress
  local img = image.decompressJPG(data.img)

  -- random crop
  local nDim = img:dim()
  local start_x = math.random(img:size(nDim) - InputSize)
  local start_y = math.random(img:size(nDim - 1) - InputSize)
  img = img:narrow(nDim, start_x, InputSize):narrow(nDim - 1, start_y, InputSize)

  -- flip
  local hflip = math.random(2) == 1
  if hflip then
    img = image.hflip(img)
  end

  return img:float(), class
end

----------------------------------------------------------------------
-- Extract from LMDB testing data.
--
-- Input
--   key    -  key
--   data   -  data
--
-- Output
--   img    -  image
--   class  -  class
function data_load.ExtractFromLMDBTest(key, data)
  -- class
  local class = data.c

  -- decompress
  local img = image.decompressJPG(data.img)

  -- crop
  local nDim = img:dim()
  local start_x = math.ceil((img:size(nDim) - InputSize) / 2)
  local start_y = math.ceil((img:size(nDim - 1) - InputSize) / 2)
  img = img:narrow(nDim, start_x, InputSize):narrow(nDim - 1, start_y, InputSize)

  return img:float(), class
end

----------------------------------------------------------------------
-- Create training data-provider.
--
-- Output
--   db  -  data provider
function data_load.newTrainDB()
  local db = eladtools.LMDBProvider {
    Source = lmdb.env({Path = trLmdb, RDONLY = true}),
    SampleSize = sampleSiz,
    ExtractFunction = data_load.ExtractFromLMDBTrain,
    Name = 'train'
  }
  return db
end

----------------------------------------------------------------------
-- Create testing data-provider.
--
-- Output
--   db  -  data provider
function data_load.newTestDB()
  local db = eladtools.LMDBProvider {
    Source = lmdb.env({Path = teLmdb, RDONLY = true}),
    SampleSize = sampleSiz,
    ExtractFunction = data_load.ExtractFromLMDBTest,
    Name = 'test'
  }
  return db
end

----------------------------------------------------------------------
-- Initialize data-loader.
--
-- Input
--   opt      -  option
--   solConf  -  solver configuration
function data_load.init(opt, solConf)
  local lib = require('lua_lib')
  lib.prIn('data_load init')

  -- dimension
  sampleSiz = solConf.smpSiz
  InputSize = sampleSiz[2]

  -- mean
  meanInfo = torch.load(opt.PATH.meanPath)
  DataMean = meanInfo.me
  DataStd = meanInfo.std
  -- train
  trLmdb = opt.PATH.trLmdb
  local trLmdb2 = trLmdb:gsub(paths.home, '/workplace/feng')
  if paths.dirp(trLmdb2) then
    trLmdb = trLmdb2
    lib.pr('local tr lmdb: %s', trLmdb)
  end

  -- test
  teLmdb = opt.PATH.teLmdb
  local teLmdb2 = teLmdb:gsub(paths.home, '/workplace/feng')
  if paths.dirp(teLmdb2)  then
    teLmdb = teLmdb2
    lib.pr('local te lmdb: %s', teLmdb)
  end

  lib.prOut()
end

return data_load

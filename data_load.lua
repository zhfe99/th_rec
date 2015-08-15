#!/usr/bin/env th
-- Data loader.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-01-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-15-2015

require 'eladtools'
require 'xlua'
require 'lmdb'

local Threads = require 'threads'
local ffi = require 'ffi'
local sampleSiz = solConf.smpSiz
local InputSize = sampleSiz[2]
local meanInfo = torch.load(opt.PATH.meanPath)
local TRAINING_DIR = opt.PATH.trLmdb
local VALIDATION_DIR = opt.PATH.teLmdb
local DataMean = meanInfo.me
local DataStd = meanInfo.std

local data_load = {}

----------------------------------------------------------------------
-- Normalize the data by subtracting the mean and dividing the scale.
--
-- Input
--   data  -  b x d x h x w
function data_load.Normalize(data)


  -- local debugger = require('fb.debugger')
  -- debugger.enter()

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

  -- fit the data to multi-gpu
  if data0:size(1) % #opt.gpus > 0 then
    assert(torch.type(data0) == 'torch.ByteTensor')
    assert(torch.type(label0) == 'torch.IntTensor')
    local b0 = data0:size(1)
    local d = data0:size(2)
    local h = data0:size(3)
    local w = data0:size(4)
    local b = b0 - b0 % opt.nGpu
    data = torch.ByteTensor(b, d, h, w):copy(data0[{{1, b}, {}, {}, {}}])
    label = torch.IntTensor(b):copy(label0[{{1, b}}])
  end

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
  local class = data.c
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
  local class = data.c

  -- decompress
  local img = image.decompressJPG(data.img)

  -- local lib = require 'lua_lib'
  -- lib.imgSave('tmp1.jpg', img)
  -- local debugger = require('fb.debugger')
  -- debugger.enter()

  -- crop
  local nDim = img:dim()
  local start_x = math.ceil((img:size(nDim) - InputSize) / 2)
  local start_y = math.ceil((img:size(nDim - 1) - InputSize) / 2)
  img = img:narrow(nDim, start_x, InputSize):narrow(nDim - 1, start_y, InputSize)

  -- lib.imgSave('tmp2.jpg', img)
  -- local debugger = require('fb.debugger')
  -- debugger.enter()

  return img:float(), class
end

-- lmdb training
-- local TrainDB = eladtools.LMDBProvider {
--   Source = lmdb.env({Path = TRAINING_DIR, RDONLY = true}),
--   SampleSize = sampleSiz,
--   ExtractFunction = ExtractFromLMDBTrain,
--   Name = 'train'
-- }

-- lmdb testing
-- local ValDB = eladtools.LMDBProvider {
--   Source = lmdb.env({Path = VALIDATION_DIR, RDONLY = true}),
--   SampleSize = sampleSiz,
--   ExtractFunction = ExtractFromLMDBTest,
--   Name = 'test'
-- }

-- return {ValDB = ValDB, TrainDB = TrainDB}

-- train db

----------------------------------------------------------------------
-- Create training data-provider.
--
-- Output
--   db  -  data provider
function data_load.newTrainDB()
  local db = eladtools.LMDBProvider {
    Source = lmdb.env({Path = TRAINING_DIR, RDONLY = true}),
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
    Source = lmdb.env({Path = VALIDATION_DIR, RDONLY = true}),
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
  local sampleSiz = solConf.smpSiz
  local InputSize = sampleSiz[2]
  local meanInfo = torch.load(opt.PATH.meanPath)
  local TRAINING_DIR = opt.PATH.trLmdb
  local VALIDATION_DIR = opt.PATH.teLmdb
  local DataMean = meanInfo.me
  local DataStd = meanInfo.std
end

return data_load

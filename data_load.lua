require 'eladtools'
require 'xlua'
require 'lmdb'

local Threads = require 'threads'
local ffi = require 'ffi'
-- local conf = dat
local InputSize = 224
local sampleSiz = {3, 224, 224}
local meanInfo = torch.load(PATH.meanPath)
local TRAINING_DIR = PATH.trLmdb
local VALIDATION_DIR = PATH.teLmdb
local DataMean = meanInfo.me
local DataStd = meanInfo.std

function ExtractSampleFunc(data0, label0)
  -- local debugger = require('fb.debugger')
  -- debugger.enter()

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

  return Normalize(data), label
end

----------------------------------------------------------------------
-- Normalize the data by subtracting the mean and dividing the scale.
--
-- Input
--   data  -  b x d x h x w
function Normalize(data)
  -- local debugger = require('fb.debugger')
  -- debugger.enter()

  data = data:float()
  for j = 1, 3 do
    data[{{}, j, {}, {}}]:add(-DataMean[j])
    data[{{}, j, {}, {}}]:div(DataStd[j])
  end

  return data
end

local function ExtractFromLMDBTrain(key, data)
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

local function ExtractFromLMDBTest(key, data)
  local class = data.c
  local img = image.decompressJPG(data.img)

  -- crop
  local nDim = img:dim()
  local start_x = math.ceil((img:size(nDim) - InputSize) / 2)
  local start_y = math.ceil((img:size(nDim - 1) - InputSize) / 2)
  img = img:narrow(nDim, start_x, InputSize):narrow(nDim - 1, start_y, InputSize)

  return img:float(), class
end

local TrainDB = eladtools.LMDBProvider {
  Source = lmdb.env({Path = TRAINING_DIR, RDONLY = true}),
  SampleSize = sampleSiz,
  ExtractFunction = ExtractFromLMDBTrain,
  Name = 'train'
}

local ValDB = eladtools.LMDBProvider {
  Source = lmdb.env({Path = VALIDATION_DIR, RDONLY = true}),
  SampleSize = sampleSiz,
  ExtractFunction = ExtractFromLMDBTest,
  Name = 'test'
}

return {
  ValDB = ValDB,
  TrainDB = TrainDB,
}

require 'eladtools'
require 'xlua'
require 'lmdb'

local Threads = require 'threads'
local ffi = require 'ffi'
local config = require './Models/car_v1c_conf'

function Normalize(data)
  data = data:float()

  for j = 1, 3 do
    data[j]:add(-config.DataMean[j])
    data[j]:div(config.DataStd[j])
  end

  return data
  -- return data:float():add(-config.DataMean):div(config.DataStd)
end

function ExtractFromLMDBTrain(key, data)
  local wnid = string.split(data.Name, '/')[1]
  local class = config.info.cNm2Ids[wnid]
  local img = data.Data
  local nDim = img:dim()
  local start_x = math.random(img:size(nDim) - config.InputSize)
  local start_y = math.random(img:size(nDim - 1) - config.InputSize)
  img = img:narrow(nDim, start_x, config.InputSize):narrow(nDim - 1, start_y, config.InputSize)
  local hflip = math.random(2) == 1
  if hflip then
    img = image.hflip(img)
  end
  return img, class
end

function ExtractFromLMDBTest(key, data)
  local wnid = string.split(data.Name, '/')[1]
  local class = config.info.cNm2Ids[wnid]
  local img = data.Data
  local nDim = img:dim()
  local start_x = math.ceil((img:size(nDim) - config.InputSize) / 2)
  local start_y = math.ceil((img:size(nDim - 1) - config.InputSize) / 2)
  img = img:narrow(nDim, start_x, config.InputSize):narrow(nDim - 1, start_y, config.InputSize)

  return img, class
end

local TrainDB = eladtools.LMDBProvider {
  Source = lmdb.env({Path = config.TRAINING_DIR, RDONLY = true}),
  SampleSize = config.SampleSize,
  ExtractFunction = ExtractFromLMDBTrain
}

local ValDB = eladtools.LMDBProvider {
  Source = lmdb.env({Path = config.VALIDATION_DIR, RDONLY = true}),
  SampleSize = config.SampleSize,
  ExtractFunction = ExtractFromLMDBTest
}

return {
  ValDB = ValDB,
  TrainDB = TrainDB,
}

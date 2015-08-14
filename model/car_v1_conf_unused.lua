#!/usr/bin/env th
-- Info for car configuration.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-03-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-10-2015

local th = require('lua_th')

local dat = ThDat('car', 'v1')
local PATH = dat.PATH

local info = torch.load('data/car/v1_info.t7')

local meanInfo = torch.load(PATH.meanPath)

-- local debugger = require('fb.debugger')
-- debugger.enter()

function Key(num)
  return string.format('%07d', num)
end

return {
  TRAINING_DIR = PATH.trLmdb,
  VALIDATION_DIR = PATH.teLmdb,
  ImageSize = 256,
  InputSize = 224,
  SampleSize = {3, 224, 224},
  info = info,
  -- DataMean = 118.380948,
  -- DataMean = {107, 101, 102},
  DataMean = meanInfo.me,
  -- DataStd = 61.896913,
  -- DataStd = {68, 67, 68},
  DataStd = meanInfo.std,
  Key = Key
}

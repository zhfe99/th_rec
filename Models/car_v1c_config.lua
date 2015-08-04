#!/usr/bin/env th
-- Info for car configuration.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-03-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-03-2015

local info = torch.load('data/car/v1c_info.t7')

function Key(num)
  return string.format('%07d', num)
end

return {
  TRAINING_PATH = 'data/car/v1c/train/',
  VALIDATION_PATH = 'data/car/v1c/test/',
  TRAINING_DIR = 'save/car/torch/data/car_v1c_train/',
  VALIDATION_DIR = 'save/car/torch/data/car_v1c_test/',
  ImageSize = 256,
  SampleSize = {3, 224, 224},
  info = info,
  DataMean = 118.380948,
  DataStd = 61.896913,
  Compressed = true,
  Key = Key
}

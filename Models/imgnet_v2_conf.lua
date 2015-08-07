#!/usr/bin/env th
-- Info for car configuration.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-03-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-06-2015

local info = torch.load('data/imgnet/v2_info.t7')

function Key(num)
  return string.format('%07d', num)
end

return {
  TRAINING_PATH = 'data/imgnet/v2/train/',
  VALIDATION_PATH = 'data/imgnet/v2/test/',
  TRAINING_DIR = 'save/imgnet/torch/data/imgnet_v2_train/',
  VALIDATION_DIR = 'save/imgnet/torch/data/imgnet_v2_test/',
  ImageSize = 256,
  InputSize = 224,
  SampleSize = {3, 224, 224},
  info = info,
  -- DataMean = 118.380948,
  DataMean = {107, 101, 102},
  -- DataStd = 61.896913,
  DataStd = {68, 67, 68},
  Key = Key
}

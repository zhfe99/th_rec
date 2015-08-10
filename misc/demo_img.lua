#!/usr/bin/env th
-- Demo of lmdb.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-09-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-09-2015

local lib = require 'lua_lib'
require 'image'

local lmdbPath = paths.concat(os.getenv('HOME'), 'save/car/torch/data/car_v1_te_lmdb')

local ha = lib.lmdbRIn(lmdbPath)

print(ha.env:stat())

for i = 1, 10 do
  local val = lib.lmdbR(ha)
  local img0 = val.img
  local img1 = image.decompressJPG(img0)

  local img = (img1 * 255):type('torch.ByteTensor')

  image.save(string.format('tmp_%d.jpg', i), img:float() / 255)
end

lib.lmdbROut(ha)

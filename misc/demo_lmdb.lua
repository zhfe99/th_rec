#!/usr/bin/env th
-- Demo of lmdb.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-09-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-10-2015

local lib = require 'lua_lib'
require 'image'

local lmdbPath = paths.concat(os.getenv('HOME'), 'save/car/torch/data/car_v1_te_lmdb')

local ha = lib.lmdbRIn(lmdbPath)

print(ha.env:stat())

local val = lib.lmdbR(ha)
local img0 = val.Data
local img = image.compressJPG(img0)
local img2 = image.decompressJPG(img)

local debugger = require('fb.debugger')
debugger.enter()

local dif = torch.add(img2:float(), -1, img0:float())
print(dif:abs():max())

-- lib.lmdbROut(ha)

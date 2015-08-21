#!/usr/bin/env th
-- Demo of lmdb.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-09-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-21-2015

local lib = require 'lua_lib'
require 'image'

local lmdbPath = paths.concat(os.getenv('HOME'), 'save/car/torch/data/car_v1c_tr_lmdb_ori')

local ha = lib.lmdbRIn(lmdbPath)

print(ha.env:stat())

local key, val = lib.lmdbR(ha)
local img0 = val.img

local debugger = require('fb.debugger')
debugger.enter()
local img = img0:float() / 255

image.save('tmp.jpg', img)

local debugger = require('fb.debugger')
debugger.enter()
local img = image.compressJPG(img0)
local img2 = image.decompressJPG(img)

local debugger = require('fb.debugger')
debugger.enter()

local dif = torch.add(img2:float(), -1, img0:float())
print(dif:abs():max())

-- lib.lmdbROut(ha)

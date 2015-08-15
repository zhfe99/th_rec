#!/usr/bin/env th
-- Test using Torch.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-03-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-15-2015

require 'torch'
require 'xlua'
require 'optim'
require 'pl'
require 'eladtools'
require 'trepl'
require 'net.alex'
local th = require('lua_th')
local lib = require('lua_lib')

-- argument
local opts = require('opts')
opt = opts.parse(arg, 'test')

-- data
local dat = ThDat(opt.dbe, opt.ver)

-- network
local net = ThNet(opt.dbe, opt.ver, opt.con)
local _, _, solConf, _ = dofile(opt.CONF.protTr)
rawset(_G, 'solConf', solConf)

-- data loader
local data_load = require('data_load')

-- trained model
local model = torch.load(net.CONF.modPath)
model:evaluate()

-- each image
local ha = lib.lmdbRIn(dat.PATH.teLmdb)
local co = 0
for iImg = 1, ha.n do
  -- read one
  local key, val = lib.lmdbR(ha)

  -- extract img
  local img, c = data_load.ExtractFromLMDBTest(key, val)
  -- lib.imgSave('tmp3aa.jpg', img)
  -- local debugger = require('fb.debugger')
  -- debugger.enter()

  -- img = img:resize(1, unpack(solConf.smpSiz))
  local img2 = torch.Tensor(1, unpack(solConf.smpSiz))
  img2[1] = img
  -- lib.imgSave('tmp3b.jpg', img2[1])

  -- img = img:contiguous()
  -- img = img:resize(1, 3, 224, 224)
  -- lib.imgSave('tmp3a.jpg', img[1])

  -- local debugger = require('fb.debugger')
  -- debugger.enter()

  -- normalize
  img = data_load.Normalize(img2)
  img = img:type('torch.CudaTensor')
  lib.imgSave('tmp3.jpg', img[1])
  local debugger = require('fb.debugger')
  debugger.enter()

  -- classify
  local y = model:forward(img)

  -- accuracy
  local a, b = y:max(2)

  if b[1][1] == c then
    co = co + 1
  end

  local debugger = require('fb.debugger')
  debugger.enter()


  print(string.format('%d/%d', co, iImg))
end
lib.lmdbROut(ha)

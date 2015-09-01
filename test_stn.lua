#!/usr/bin/env th
-- Train using Torch.
--
-- Example
--   export CUDA_VISIBLE_DEVICES=0,1,2,3
--   ./train.lua -dbe imgnet -ver v2 -con alx
--   ./train.lua -ver v1 -con alexS1 -deb
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 2015-08
--   modify  -  Feng Zhou (zhfe99@gmail.com), 2015-09

require('torch')
require('optim')
require('cunn')
require('image')
local th = require('lua_th')
local lib = require('lua_lib')
local opts = require('opts')
local net = require('net')
local dp = require('dp_load')
local tr_deb = require('tr_deb')

-- option
local opt, con = opts.parse(arg, 'train')

-- network
local model, loss, modelSv, modss, optStat = net.new(con, opt)
local optimator

-- example image
local img0 = image.lena()

-- pre-process
img0 = lib.imgSizNew(img0, 256)
img0 = lib.imgCrop(img0, 224)
local x = img0:reshape(1, 3, 224, 224):cuda()

-- forward
local y = model:forward(x)

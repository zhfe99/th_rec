#!/usr/bin/env th
-- Train using Torch.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-03-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-13-2015

require 'torch'
require 'xlua'
require 'optim'
require 'pl'
require 'eladtools'
require 'trepl'
paths.dofile('fbcunn_files/Optim.lua')
local th = require('lua_th')
local lib = require('lua_lib')

-- argument
local opts = paths.dofile('opts.lua')
opt = opts.parse(arg, 'demo')

-- data
local dat = ThDat(opt.dbe, opt.ver)
PATH = dat.PATH

-- network
local model, loss, nEpo, nEpoSv, batchSiz, bufSiz, sampleSiz, optStat, parEpo = require(opt.network)

local modPath = opt.modPath .. '_' .. 80 .. '.t7'
local model1 = torch.load(modPath)

-- img
imgPath = '00028.jpg'
imgPath = '/home/ma/feng/data/car/v1c/test/Acura_Integra__Type-R__2001/00128.jpg'
img = lib.imgLoad(imgPath)

-- scale
local sizMa = 256
img = lib.imgSizNew(img, sizMa)

-- local debugger = require('fb.debugger')
-- debugger.enter()

-- crop
local InputSize = 224
local nDim = img:dim()
local start_x = math.ceil((img:size(nDim) - InputSize) / 2)
local start_y = math.ceil((img:size(nDim - 1) - InputSize) / 2)
img = img:narrow(nDim, start_x, InputSize):narrow(nDim - 1, start_y, InputSize)

-- normalize
local meanInfo = torch.load(PATH.meanPath)
local DataMean = meanInfo.me
local DataStd = meanInfo.std
img = img:float()
for j = 1, 3 do
  img[{j, {}, {}}]:add(-DataMean[j])
  img[{j, {}, {}}]:div(DataStd[j])
end

img = img:type('torch.CudaTensor')
img = img:resize(1, 3, 224, 224)
model1:evaluate()
y = model1:forward(img)
a, b = y:max(2)
print(dat.DATA.cNms[b[1][1]])

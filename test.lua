#!/usr/bin/env th
-- Test using Torch.
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
local th = require('lua_th')
local lib = require('lua_lib')

-- argument
local opts = paths.dofile('opts.lua')
opt = opts.parse(arg, 'demo')

-- data
local dat = ThDat(opt.dbe, opt.ver)
PATH = dat.PATH
local imgFold = PATH.dataFold .. '/test'
local imgNms = lib.loadLns(PATH.teListCaf)

-- network
local model, loss, nEpo, nEpoSv, batchSiz, bufSiz, sampleSiz, optStat, parEpo = require(opt.network)

local modPath = opt.modPath .. '_' .. 1 .. '.t7'
local modPath = opt.modPath .. '_' .. 80 .. '.t7'
local model1 = torch.load(modPath)

local co = 0
for i, ln in ipairs(imgNms) do
  local parts = lib.split(ln, ' ')
  local imgNm = parts[1]
  local cT = tonumber(parts[2]) + 1

  -- img
  local imgPath = imgFold .. '/' .. imgNm
  img = lib.imgLoad(imgPath)

  -- scale
  local sizMa = 256
  img = lib.imgSizNew(img, sizMa)

  -- crop
  local InputSize = 224
  local nDim = img:dim()
  local start_x = math.ceil((img:size(nDim) - InputSize) / 2)
  local start_y = math.ceil((img:size(nDim - 1) - InputSize) / 2)
  img = img:narrow(nDim, start_x, InputSize):narrow(nDim - 1, start_y, InputSize)

  -- normalize
  local meanInfo = torch.load(PATH.meanPath)
  img = img:float()
  for j = 1, 3 do
    img[{j, {}, {}}]:add(-meanInfo.me[j])
    img[{j, {}, {}}]:div(meanInfo.std[j])
  end

  img = img:type('torch.CudaTensor')
  img = img:resize(1, 3, 224, 224)
  model1:evaluate()
  y = model1:forward(img)
  a, b = y:max(2)
  c = b[1][1]

  if cT == c then
    co = co + 1
  end
  print(string.format('%d/%d', co, i))

  -- print(dat.DATA.cNms[b[1][1]])
end
